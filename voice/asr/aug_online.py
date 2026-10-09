"""On-the-fly acoustic augmentation for train_asr.py --aug_schedule, on the GPU: each epoch a fraction p(epoch) of the
utterances in every training batch is re-rendered (fresh every step, nothing stored): speed 0.9 / 1.0 / 1.1, synthetic
room reverb (30 %), noise at 5-20 dB SNR (half babble = 2-4 OTHER utterances of the same batch, half white / pink /
brown), a laptop-mic band limit (50 %: high-pass 100-300 Hz, low-pass 3.4-7.5 kHz) and gain -6..+3 dB.

Why GPU: the CPU version (in the loader workers) cost 84 ms per utterance -- np.convolve with a 7k-tap RIR alone 74 ms
-- and made run5 CPU-bound: 6,200 -> 4,450 audio-s/s. Here the whole batch is one set of FFT ops after the host->device
copy (LightningModule.on_after_batch_transfer), and p is read in the main process (the CPU version's p was inherited
by forked workers, which a mid-epoch resume forked too early: run5's resumed epoch trained un-augmented).
Seeded by (seed, global step): same seed -> same augmentation, a resume reproduces it.
"""
import torch
import torchaudio.functional as AF

SR = 16000
STATE = {"p": 0.0, "epoch": 0}


def p_for(schedule, epoch):
    return schedule[min(epoch, len(schedule) - 1)]


def _rfft_n(n):
    return 1 << (n - 1).bit_length()


def _mask(lens, t):
    return torch.arange(t, device=lens.device)[None] < lens[:, None]


def _speed(x, lens, g):
    """Per utterance speed 0.9 / 1.0 / 1.1 (resample as if recorded at 16k*speed) -> new padded batch + lengths."""
    sp = torch.randint(0, 3, (x.shape[0],), device=x.device, generator=g)
    new = lens.clone()
    for k, f in ((0, 0.9), (2, 1.1)):
        new = torch.where(sp == k, torch.ceil(lens / f).long(), new)
    out = torch.zeros(x.shape[0], int(new.max()), device=x.device, dtype=x.dtype)
    out[sp == 1, : x.shape[1]] = x[sp == 1]
    for k, f in ((0, 0.9), (2, 1.1)):
        idx = (sp == k).nonzero().squeeze(1)
        if len(idx):
            y = AF.resample(x[idx, : int(lens[idx].max())], int(SR * f), SR)
            n = min(y.shape[1], out.shape[1])
            out[idx, :n] = y[:, :n]
    return out * _mask(new, out.shape[1]), new


def _reverb(x, g):
    """Exponentially decaying noise RIR, T60 0.2-0.7 s, direct path 1 (as augment._rir); FFT convolution."""
    b, t = x.shape
    k = int(0.7 * SR)
    t60 = (0.2 + 0.5 * torch.rand(b, 1, device=x.device, generator=g)) * SR
    n = torch.arange(k, device=x.device)[None].float()
    h = torch.randn(b, k, device=x.device, generator=g) * torch.exp(-6.9 * n / t60) * (n < t60)
    h[:, 0] = 1.0
    h = h / h.abs().sum(1, keepdim=True) * 4
    m = _rfft_n(t + k - 1)
    return torch.fft.irfft(torch.fft.rfft(x, m) * torch.fft.rfft(h, m), m)[:, :t]


def _colored(b, t, g, dev):
    w = torch.randn(b, t, device=dev, generator=g)
    kind = torch.randint(0, 3, (b, 1), device=dev, generator=g)          # white / pink (1/f) / brown (1/f^2) power
    f = torch.fft.rfftfreq(t, 1 / SR, device=dev).clamp(min=SR / t)
    shape = torch.where(kind == 0, torch.ones_like(f), torch.where(kind == 1, f.rsqrt(), 1 / f))
    return torch.fft.irfft(torch.fft.rfft(w) * shape, t)


def _babble(src, rows, t, g):
    """For each target row (index into the batch): 2-4 OTHER utterances of the batch, peak-normalised, cropped /
    zero-padded to t."""
    b, n = src.shape[0], len(rows)
    s = src[:, :t] / (src.abs().amax(1, keepdim=True) + 1e-6)
    s = torch.nn.functional.pad(s, (0, t - s.shape[1]))
    noise = torch.zeros(n, t, device=src.device)
    count = torch.randint(2, 5, (n,), device=src.device, generator=g)
    for j in range(4):
        other = (rows + 1 + torch.randint(0, b - 1, (n,), device=src.device, generator=g)) % b   # never itself
        noise += s[other] * (j < count)[:, None]
    return noise


def _mic(x, g):
    """Laptop mic: 2nd-order Butterworth-magnitude high-pass 100-300 Hz and low-pass 3.4-7.5 kHz (zero phase)."""
    b, t = x.shape
    f = torch.fft.rfftfreq(t, 1 / SR, device=x.device)[None]
    hp = 100 + 200 * torch.rand(b, 1, device=x.device, generator=g)
    lp = 3400 + 4100 * torch.rand(b, 1, device=x.device, generator=g)
    gain = 1 / torch.sqrt(1 + (hp / f.clamp(min=1)) ** 4) / torch.sqrt(1 + (f / lp) ** 4)
    return torch.fft.irfft(torch.fft.rfft(x) * gain, t)


def augment(audio, lens, p, g):
    """audio (B, T) float on the GPU, lens (B,) samples -> (audio', lens'): a p-fraction of rows re-rendered."""
    b = audio.shape[0]
    sel = (torch.rand(b, device=audio.device, generator=g) < p).nonzero().squeeze(1)
    if p <= 0 or len(sel) == 0:
        return audio, lens
    x, xl = _speed(audio[sel].float(), lens[sel], g)
    n, t = x.shape
    m = _mask(xl, t)
    # each effect only on the rows that draw it (FFTs over all rows then masking cost ~3x)
    rv = (torch.rand(n, device=x.device, generator=g) < 0.3).nonzero().squeeze(1)
    if len(rv):
        x[rv] = _reverb(x[rv], g) * m[rv]
    bab = torch.rand(n, device=x.device, generator=g) < 0.5
    if b < 3:
        bab[:] = False
    noise = torch.empty_like(x)
    bi, ci = bab.nonzero().squeeze(1), (~bab).nonzero().squeeze(1)
    if len(bi):
        noise[bi] = _babble(audio.float(), sel[bi], t, g)
    if len(ci):
        noise[ci] = _colored(len(ci), t, g, x.device)
    noise *= m
    snr = 5 + 15 * torch.rand(n, 1, device=x.device, generator=g)
    cnt = xl[:, None].clamp(min=1)
    ps = (x ** 2).sum(1, keepdim=True) / cnt + 1e-10
    pn = (noise ** 2).sum(1, keepdim=True) / cnt + 1e-10
    x = x + noise * torch.sqrt(ps / (pn * 10 ** (snr / 10)))
    mic = (torch.rand(n, device=x.device, generator=g) < 0.5).nonzero().squeeze(1)
    if len(mic):
        x[mic] = _mic(x[mic], g)
    x = x * m
    x = x * 10 ** ((-6 + 9 * torch.rand(n, 1, device=x.device, generator=g)) / 20)
    peak = x.abs().amax(1, keepdim=True)
    x = torch.where(peak > 1, x / peak * 0.95, x)
    out_t = max(audio.shape[1], t)
    out = torch.nn.functional.pad(audio, (0, out_t - audio.shape[1]))
    out[sel] = torch.nn.functional.pad(x, (0, out_t - t)).to(out.dtype)
    new = lens.clone()
    new[sel] = xl
    return out, new


def install(model, seed):
    """Augment training batches on the GPU right after the host->device copy (validation untouched)."""
    orig = model.on_after_batch_transfer

    def on_after_batch_transfer(batch, dataloader_idx):
        batch = orig(batch, dataloader_idx)
        if model.training and STATE["p"] > 0:
            g = torch.Generator(device=batch[0].device).manual_seed(seed * 1_000_003 + model.trainer.global_step)
            audio, lens = augment(batch[0], batch[1], STATE["p"], g)
            batch = (audio, lens, *batch[2:])
        return batch

    model.on_after_batch_transfer = on_after_batch_transfer


if __name__ == "__main__":
    import time
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    lens = torch.randint(16000, 16000 * 15, (150,), device=dev)
    a = torch.randn(150, int(lens.max()), device=dev) * 0.1 * _mask(lens, int(lens.max()))
    run = lambda p, s: augment(a, lens, p, torch.Generator(device=dev).manual_seed(s))  # noqa: E731
    y1, l1 = run(0.5, 5)
    y2, l2 = run(0.5, 5)
    assert torch.equal(y1, y2) and torch.equal(l1, l2), "not deterministic"
    assert torch.isfinite(y1).all() and y1.abs().max() <= 1.0 + 1e-6
    assert (y1[_mask(l1, y1.shape[1]).logical_not()] == 0).all(), "audio beyond lens"
    changed = (l1 != lens).sum().item()
    assert 0 < changed < 150, changed                                    # some rows sped up / slowed down
    y0, l0 = run(0.0, 5)
    assert y0 is a and l0 is lens
    for _ in range(3):
        run(0.5, 1)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(10):
        run(0.5, s)
    if dev == "cuda":
        torch.cuda.synchronize()
    print(f"aug_online ok ({dev}): {1000 * (time.perf_counter() - t0) / 10:.1f} ms per 150-utterance batch at p=0.5, "
          f"{changed} lengths changed")
