"""Local first-bench verification (run: python -m ablate.common.smoke).
Builds BOTH arms, asserts EXACT param match, patches with liger_norm+liger_rope+moe, runs a few
bf16 fwd/bwd/step iters with fused CE, and checks everything is finite. Tiny batch/seq for a 4GB card.
"""
from . import _paths  # noqa: F401
import torch
import torch.nn.functional as F
from .models import build_arm, count_params
from . import patches
from kernels.sm120.cross_entropy import fused_linear_cross_entropy

DEV = "cuda"
PATCH = ["liger_norm", "liger_rope", "moe"]
B, S = 2, 128
STEPS = 3


def run_arm(arm):
    torch.manual_seed(0)
    model, cfg = build_arm(arm, device=DEV, dtype=torch.float32)
    total, trainable, active = count_params(model)
    # fused CE consumes hidden + lm_head.weight; grab the base model + lm_head uniformly
    base = model.model
    lm_head = model.lm_head
    vocab = cfg.vocab_size
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    gen = torch.Generator(device=DEV).manual_seed(0)
    losses = []
    for _ in range(STEPS):
        ids = torch.randint(0, vocab, (B, S), generator=gen, device=DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = base(input_ids=ids, use_cache=False)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            sh = h[:, :-1, :].reshape(-1, h.shape[-1])
            sl = ids[:, 1:].reshape(-1)
            loss = fused_linear_cross_entropy(sh, lm_head.weight, sl)
        loss.backward()
        gnorm = torch.sqrt(sum(p.grad.detach().float().pow(2).sum()
                               for p in model.parameters() if p.grad is not None))
        opt.step()
        losses.append((loss.item(), gnorm.item()))
    del model, opt
    torch.cuda.empty_cache()
    return total, trainable, active, losses


def main():
    print(f"[smoke] patches={PATCH}  bf16  B={B} S={S} steps={STEPS}", flush=True)
    patches.apply(PATCH)
    res = {}
    for arm in ("qwen", "bibo_min"):
        total, trainable, active, losses = run_arm(arm)
        res[arm] = (total, trainable, active)
        ls = "  ".join(f"L={l:.3f}|g={g:.2f}" for l, g in losses)
        fin = all(torch.isfinite(torch.tensor([l for l, _ in losses])))
        print(f"[{arm:9s}] total={total/1e6:.3f}M trainable={trainable/1e6:.3f}M active={active/1e6:.3f}M "
              f"finite={fin}  {ls}", flush=True)
    qtr, btr = res["qwen"][1], res["bibo_min"][1]
    inert = abs(res["qwen"][0] - res["bibo_min"][0])
    print(f"\nparam match (trainable): qwen={qtr/1e6:.4f}M  bibo_min={btr/1e6:.4f}M  Δ={abs(qtr-btr)} "
          f"({'EXACT MATCH' if qtr == btr else 'MISMATCH'});  inert-only Δtotal={inert} "
          f"(BiBo router bias, requires_grad=False)", flush=True)
    assert qtr == btr, f"trainable param mismatch: qwen {qtr} vs bibo_min {btr}"
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
