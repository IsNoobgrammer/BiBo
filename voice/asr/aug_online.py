"""On-the-fly acoustic augmentation for train_asr.py --aug_schedule, on the GPU: each epoch a fraction p(epoch) of the
utterances in every training batch is re-rendered (fresh every step, nothing stored): speed 0.9 / 1.0 / 1.1, synthetic
room reverb (30 %), noise at 5-20 dB SNR (half babble = 2-4 OTHER utterances of the same batch, half white / pink /
brown), a laptop-mic band limit (50 %: high-pass 100-300 Hz, low-pass 3.4-7.5 kHz) and gain -6..+3 dB.

Why GPU: the CPU version (in the loader workers) cost 84 ms per utterance -- np.convolve with a 7k-tap RIR alone 74 ms
-- and made run5 CPU-bound: 6,200 -> 4,450 audio-s/s. Here the batch is augmented right after the host->device copy.
NO host syncs: the first GPU version (nonzero / .max() on GPU tensors, ~10 per batch) stalled the CPU from queuing
the next step and cost 21 % (6,334 -> 5,009 audio-s/s at p=0.5). Now every shape decision (which rows, which speed)
is made on the CPU from the lengths taken BEFORE the copy, the per-row randomness is drawn on the GPU and applied
with masks, index tensors go up pinned + non_blocking, and FFT sizes are powers of two (no per-batch cuFFT plans).
Seeded by (seed, global step): same seed -> same augmentation, a resume reproduces it.
"""
import torch
import torchaudio

SR = 16000
STATE = {"p": 0.0, "epoch": 0}


def p_for(schedule, epoch):
    return schedule[min(epoch, len(schedule) - 1)]


_RESAMPLERS = {}


def _resampler(f, dev):
    """torchaudio's functional resample rebuilds its sinc kernel per call, with a host sync inside
    (torch.tensor(1.0).to(t)); the transform builds it once."""
    key = (f, str(dev))
    if key not in _RESAMPLERS:
        _RESAMPLERS[key] = torchaudio.transforms.Resample(int(SR * f), SR).to(dev)
    return _RESAMPLERS[key]


def _pow2(n):
    return 1 << (n - 1).bit_length()


def _up(t, dev):
    return t.pin_memory().to(dev, non_blocking=True) if dev.type == "cuda" else t.to(dev)


def _reverb(x, gg):
    """Exponentially decaying noise RIR, T60 0.2-0.7 s, direct path 1 (as augment._rir); FFT convolution."""
    b, t = x.shape
    k = int(0.7 * SR)
    t60 = (0.2 + 0.5 * torch.rand(b, 1, device=x.device, generator=gg)) * SR
    n = torch.arange(k, device=x.device)[None].float()
    h = torch.randn(b, k, device=x.device, generator=gg) * torch.exp(-6.9 * n / t60) * (n < t60)
    h[:, 0] = 1.0
    h = h / h.abs().sum(1, keepdim=True) * 4
    m = _pow2(t + k - 1)
    return torch.fft.irfft(torch.fft.rfft(x, m) * torch.fft.rfft(h, m), m)[:, :t]


def _colored(b, t, gg, dev):
    m = _pow2(t)
    w = torch.randn(b, m, device=dev, generator=gg)
    kind = torch.randint(0, 3, (b, 1), device=dev, generator=gg)         # white / pink (1/f) / brown (1/f^2) power
    f = torch.fft.rfftfreq(m, 1 / SR, device=dev).clamp(min=SR / m)
    shape = torch.where(kind == 0, torch.ones_like(f), torch.where(kind == 1, f.rsqrt(), 1 / f))
    return torch.fft.irfft(torch.fft.rfft(w) * shape, m)[:, :t]


def _babble(src, rows, t, gg):
    """For each target row (device index into the batch): 2-4 OTHER utterances of the batch, peak-normalised,
    cropped / zero-padded to t."""
    b, n = src.shape[0], rows.shape[0]
    s = src[:, :t] / (src.abs().amax(1, keepdim=True) + 1e-6)
    s = torch.nn.functional.pad(s, (0, t - s.shape[1]))
    noise = torch.zeros(n, t, device=src.device)
    count = torch.randint(2, 5, (n, 1), device=src.device, generator=gg)
    for j in range(4):
        other = (rows + 1 + torch.randint(0, b - 1, (n,), device=src.device, generator=gg)) % b   # never itself
        noise += s[other] * (j < count)
    return noise


def _mic(x, gg):
    """Laptop mic: 2nd-order Butterworth-magnitude high-pass 100-300 Hz and low-pass 3.4-7.5 kHz (zero phase)."""
    b, t = x.shape
    m = _pow2(t)
    f = torch.fft.rfftfreq(m, 1 / SR, device=x.device)[None]
    hp = 100 + 200 * torch.rand(b, 1, device=x.device, generator=gg)
    lp = 3400 + 4100 * torch.rand(b, 1, device=x.device, generator=gg)
    gain = 1 / torch.sqrt(1 + (hp / f.clamp(min=1)) ** 4) / torch.sqrt(1 + (f / lp) ** 4)
    return torch.fft.irfft(torch.fft.rfft(x, m) * gain, m)[:, :t]


def augment(audio, lens_cpu, p, gc, gg):
    """audio (B, T) on the device, lens_cpu (B,) samples ON THE CPU, gc a CPU generator (shapes), gg a generator on
    audio's device (per-row randomness) -> (audio', lens' on the device): a p-fraction of rows re-rendered."""
    dev = audio.device
    b = audio.shape[0]
    sel = (torch.rand(b, generator=gc) < p).nonzero().squeeze(1)
    if p <= 0 or len(sel) == 0 or b < 3:
        STATE["cpu_lens"] = lens_cpu                                    # fused_joint sizes its lattice from these
        return audio, _up(lens_cpu, dev)
    n = len(sel)
    L = lens_cpu[sel].long()
    sp = torch.randint(0, 3, (n,), generator=gc)
    newL = torch.where(sp == 1, L, torch.ceil(L / torch.tensor([0.9, 1.0, 1.1])[sp]).long())
    t = int(newL.max())
    sel_d = _up(sel, dev)
    x = torch.zeros(n, t, device=dev)
    for k, f in enumerate((0.9, 1.0, 1.1)):                            # speed: one resample per group
        idx = (sp == k).nonzero().squeeze(1)
        if len(idx):
            seg = audio[_up(sel[idx], dev), : int(L[idx].max())].float()
            if f != 1.0:
                seg = _resampler(f, dev)(seg)
            c = min(seg.shape[1], t)
            x[_up(idx, dev), :c] = seg[:, :c]
    newL_d = _up(newL, dev)
    m = torch.arange(t, device=dev)[None] < newL_d[:, None]           # on the device (on the CPU: 20 MB + a copy)
    x = x * m
    r = torch.rand(n, 3, device=dev, generator=gg)                     # per row: reverb / babble / mic
    x = torch.where(r[:, :1] < 0.3, _reverb(x, gg), x) * m
    noise = torch.where(r[:, 1:2] < 0.5, _babble(audio.float(), sel_d, t, gg), _colored(n, t, gg, dev)) * m
    snr = 5 + 15 * torch.rand(n, 1, device=dev, generator=gg)
    cnt = newL_d[:, None].clamp(min=1).float()
    ps = (x ** 2).sum(1, keepdim=True) / cnt + 1e-10
    pn = (noise ** 2).sum(1, keepdim=True) / cnt + 1e-10
    x = x + noise * torch.sqrt(ps / (pn * 10 ** (snr / 10)))
    x = torch.where(r[:, 2:3] < 0.5, _mic(x, gg), x) * m
    x = x * 10 ** ((-6 + 9 * torch.rand(n, 1, device=dev, generator=gg)) / 20)
    peak = x.abs().amax(1, keepdim=True)
    x = torch.where(peak > 1, x / peak * 0.95, x)
    out_t = max(audio.shape[1], t)
    out = torch.nn.functional.pad(audio, (0, out_t - audio.shape[1]))
    out[sel_d] = torch.nn.functional.pad(x, (0, out_t - t)).to(out.dtype)
    new = lens_cpu.clone()
    new[sel] = newL.to(new.dtype)                                      # NeMo's lens are int32
    STATE["cpu_lens"] = new                                             # the CPU lengths after speed perturbation
    return out, _up(new, dev)


def install(model, seed):
    """Augment training batches on the GPU right after the host->device copy (validation untouched). The lengths
    are taken on the CPU before the copy, so no augmentation decision ever waits on the GPU."""
    before, after = model.on_before_batch_transfer, model.on_after_batch_transfer
    cpu = {}

    def on_before_batch_transfer(batch, dataloader_idx):
        batch = before(batch, dataloader_idx)
        cpu["lens"] = batch[1].clone() if model.training else None
        return batch

    def on_after_batch_transfer(batch, dataloader_idx):
        batch = after(batch, dataloader_idx)
        if model.training and STATE["p"] > 0 and cpu.get("lens") is not None:
            s = seed * 1_000_003 + model.trainer.global_step
            gc = torch.Generator().manual_seed(s)
            gg = torch.Generator(device=batch[0].device).manual_seed(s)
            audio, lens = augment(batch[0], cpu["lens"], STATE["p"], gc, gg)
            batch = (audio, lens, *batch[2:])
        return batch

    model.on_before_batch_transfer = on_before_batch_transfer
    model.on_after_batch_transfer = on_after_batch_transfer


if __name__ == "__main__":
    import time
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    lens = torch.randint(16000, 16000 * 15, (150,), dtype=torch.int32)
    T = int(lens.max()) + 4000                                          # padded wider than the longest row
    a = (torch.randn(150, T) * 0.1 * (torch.arange(T)[None] < lens[:, None])).to(dev)

    def run(p, s):
        return augment(a, lens, p, torch.Generator().manual_seed(s), torch.Generator(device=dev).manual_seed(s))

    y1, l1 = run(0.5, 5)
    y2, l2 = run(0.5, 5)
    assert torch.equal(y1, y2) and torch.equal(l1, l2), "not deterministic"
    assert l1.dtype == lens.dtype and l1.device.type == dev.type
    assert torch.isfinite(y1).all() and y1.abs().max() <= 1.0 + 1e-6
    assert (y1[torch.arange(y1.shape[1], device=dev)[None] >= l1[:, None]] == 0).all(), "audio beyond lens"
    changed = (l1.cpu() != lens).sum().item()
    assert 0 < changed < 150, changed                                    # some rows sped up / slowed down
    for p in (0.2, 0.05, 1.0):                                           # incl. every row / few rows (run5 crash)
        for s in range(20):
            run(p, s)
    y0, _ = run(0.0, 5)
    assert y0 is a
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(10):
        run(0.5, 100 + s)
    t_host = (time.perf_counter() - t0) / 10                             # what blocks the training loop
    if dev.type == "cuda":
        torch.cuda.synchronize()
    print(f"aug_online ok ({dev}): host {1000 * t_host:.1f} ms, host+device {1000 * (time.perf_counter() - t0) / 10:.1f} "
          f"ms per 150-utterance batch at p=0.5, {changed} lengths changed")
