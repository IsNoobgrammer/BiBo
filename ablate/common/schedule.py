"""LR schedules. WSD (warmup-stable-decay, linear decay to `final_frac`) and cosine (warmup + cosine
anneal to `final_frac` over the whole post-warmup span). Pick via make_scheduler(kind, ...)."""
import math
import torch


def wsd_lambda(total_steps, warmup_frac=0.05, decay_frac=0.20, final_frac=0.0):
    warm = max(int(total_steps * warmup_frac), 1)
    decay_start = int(total_steps * (1.0 - decay_frac))

    def f(step):
        if step < warm:
            return step / warm
        if step < decay_start:
            return 1.0
        prog = (step - decay_start) / max(total_steps - decay_start, 1)
        return final_frac + (1.0 - final_frac) * (1.0 - prog)
    return f


def cosine_lambda(total_steps, warmup_frac=0.05, final_frac=0.0):
    """Linear warmup then cosine anneal 1.0 -> final_frac over the whole post-warmup span (decay_frac
    is not used — cosine has no stable phase)."""
    warm = max(int(total_steps * warmup_frac), 1)

    def f(step):
        if step < warm:
            return step / warm
        prog = (step - warm) / max(total_steps - warm, 1)               # 0..1 over post-warmup
        return final_frac + (1.0 - final_frac) * 0.5 * (1.0 + math.cos(math.pi * prog))
    return f


def make_wsd(optimizers, total_steps, warmup_frac=0.05, decay_frac=0.20, final_frac=0.0):
    fn = wsd_lambda(total_steps, warmup_frac, decay_frac, final_frac)
    return [torch.optim.lr_scheduler.LambdaLR(o, fn) for o in optimizers]


def make_scheduler(kind, optimizers, total_steps, warmup_frac=0.05, decay_frac=0.20, final_frac=0.0):
    """kind in {'wsd','cosine'}. warmup_frac applies to both; decay_frac only to WSD."""
    if kind == "wsd":
        fn = wsd_lambda(total_steps, warmup_frac, decay_frac, final_frac)
    elif kind == "cosine":
        fn = cosine_lambda(total_steps, warmup_frac, final_frac)
    else:
        raise ValueError(f"unknown scheduler {kind!r}; valid: wsd, cosine")
    return [torch.optim.lr_scheduler.LambdaLR(o, fn) for o in optimizers]
