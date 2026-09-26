"""LR schedules. WSD (warmup-stable-decay, linear decay to `final_frac`) and cosine (warmup + cosine
anneal to `final_frac` over the whole post-warmup span). Pick via make_scheduler(kind, ...)."""
import math
import torch


def wsd_lambda(total_steps, warmup_frac=0.05, decay_frac=0.20, final_frac=0.0,
               decay_shape="linear", cycles=0, cycle_floor=0.1):
    """cycles > 0 turns the stable phase into `cycles` smooth cosine waves peak -> cycle_floor -> peak
    (no hard restarts), so the final decay still starts from the peak. decay_shape 'sqrt' is the
    1 - sqrt(progress) cooldown (drops fast first, then flattens) instead of linear."""
    warm = max(int(total_steps * warmup_frac), 1)
    decay_start = int(total_steps * (1.0 - decay_frac))

    def f(step):
        if step < warm:
            return step / warm
        if step < decay_start:
            if cycles <= 0:
                return 1.0
            ph = ((step - warm) / max(decay_start - warm, 1) * cycles) % 1.0
            return cycle_floor + (1.0 - cycle_floor) * 0.5 * (1.0 + math.cos(2 * math.pi * ph))
        prog = (step - decay_start) / max(total_steps - decay_start, 1)
        d = 1.0 - (math.sqrt(prog) if decay_shape == "sqrt" else prog)
        return final_frac + (1.0 - final_frac) * d
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


def make_wd_schedule(optimizers, total_steps, wd_start, wd_end, warmup_frac=0.0):
    """REVERSE-COSINE weight decay: wd RISES wd_start -> wd_end over training (LR still decays).

    Motivation (Jul 30 2026 wd sweep, 64x4/524M): wd 0.01 beat wd 0.1 on train loss through step
    ~1500 and then gave the whole lead back during the cosine anneal (silu 1500-2000 window 1.77590
    vs 1.76404, final bpb 0.68703 vs 0.68171). Low decay early, high decay late is the shape that
    would take both halves. Under Muon the two things wd controls move oppositely --
    ||W*|| ~ sqrt(lr/2wd) (ratio) and relative step sqrt(2*lr*wd) (product) -- so a rising wd both
    contracts the weight scale and holds the relative step up as lr decays, instead of letting it
    fall as sqrt(cosine).

    Shape is the mirror of cosine_lambda: 0.5*(1-cos(pi*p)) rises 0->1 with a flat start and end.
    Returns a callable step(t); call it once per optimizer step alongside the LR schedulers.

    Groups whose weight_decay was initialized to 0 STAY at 0 -- that is the act-scale gain group
    (see build_optimizers), where decay would re-impose the very equilibrium alpha exists to escape.
    """
    base = [[g.get("weight_decay", 0.0) for g in o.param_groups] for o in optimizers]
    warm = max(int(total_steps * warmup_frac), 0)

    def step(t):
        p = 0.0 if t < warm else (t - warm) / max(total_steps - warm - 1, 1)
        p = min(max(p, 0.0), 1.0)
        wd = wd_start + (wd_end - wd_start) * 0.5 * (1.0 - math.cos(math.pi * p))
        for o, bs in zip(optimizers, base):
            for g, b in zip(o.param_groups, bs):
                if b > 0.0:
                    g["weight_decay"] = wd
        return wd
    return step


def make_scheduler(kind, optimizers, total_steps, warmup_frac=0.05, decay_frac=0.20, final_frac=0.0,
                   decay_shape="linear", cycles=3, cycle_floor=0.1):
    """kind in {'wsd','cosine','cyclic'}. warmup_frac applies to all; decay_frac / decay_shape to wsd
    and cyclic; cyclic = wsd whose stable phase is `cycles` cosine waves down to cycle_floor."""
    if kind in ("wsd", "cyclic"):
        fn = wsd_lambda(total_steps, warmup_frac, decay_frac, final_frac, decay_shape,
                        cycles if kind == "cyclic" else 0, cycle_floor)
    elif kind == "cosine":
        fn = cosine_lambda(total_steps, warmup_frac, final_frac)
    else:
        raise ValueError(f"unknown scheduler {kind!r}; valid: wsd, cosine, cyclic")
    return [torch.optim.lr_scheduler.LambdaLR(o, fn) for o in optimizers]
