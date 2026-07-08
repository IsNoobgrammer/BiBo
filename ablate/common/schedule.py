"""Warmup-Stable-Decay (WSD) LR schedule. warmup 0.05 (default), stable, linear decay to `final_frac`."""
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


def make_wsd(optimizers, total_steps, warmup_frac=0.05, decay_frac=0.20, final_frac=0.0):
    fn = wsd_lambda(total_steps, warmup_frac, decay_frac, final_frac)
    return [torch.optim.lr_scheduler.LambdaLR(o, fn) for o in optimizers]
