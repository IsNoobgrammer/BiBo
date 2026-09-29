"""fp8 health for --moe_fp8 runs: turns on tkf moe_fp8.HEALTH for ONE micro-batch of a log step, then
aggregates what every fp8 layer recorded into W&B keys `fp8/<tensor>/<metric>_{mean,max}` (over layers).

Tensors: W_gate_up / W_down (exact flush vs the fp32 master), x_F1_in, GU_F1_out, act_up_F3_in, eo_F3_out,
dO_B3_in, d_inter_B3_out, dGU_B6_in, dx_rows_B6_out. Metrics: zero_pct (flush proxy), subnormal_pct,
at_max_pct (block max landing on 448; true overflow is 0 by construction), scale_exp_{min,mean,max},
block_max_over_rms_p99, amax_over_rms, kurtosis, flush_pct (weights). A drifting scale_exp_mean or a
growing zero_pct / subnormal_pct over training is the signal that fp8 is losing range.
"""
import importlib

_M8 = None


def _mod():
    global _M8
    if _M8 is None:
        _M8 = importlib.import_module("kernels.sm120.moe_fp8")
    return _M8


def start():
    _mod().HEALTH = {}


def stop():
    """-> ({wandb key: value}, one-line console summary)."""
    m = _mod()
    h, m.HEALTH = m.HEALTH, None
    out = {}
    if not h:
        return out, ""
    for tag, rows in h.items():
        keys = set().union(*(r.keys() for r in rows))
        for k in sorted(keys):
            v = [r[k] for r in rows if k in r]
            if not v:
                continue
            out[f"fp8/{tag}/{k}_mean"] = sum(v) / len(v)
            out[f"fp8/{tag}/{k}_max"] = max(v)
    g = lambda t, k: out.get(f"fp8/{t}/{k}_max", float("nan"))
    line = (f"fp8: zero% x {g('x_F1_in', 'zero_pct'):.3f} act {g('act_up_F3_in', 'zero_pct'):.3f} "
            f"dGU {g('dGU_B6_in', 'zero_pct'):.3f} | subn% act {g('act_up_F3_in', 'subnormal_pct'):.2f} "
            f"dGU {g('dGU_B6_in', 'subnormal_pct'):.2f} | W flush% {g('W_gate_up', 'flush_pct'):.4f} "
            f"| dGU exp {out.get('fp8/dGU_B6_in/scale_exp_mean_mean', float('nan')):+.1f} "
            f"kurt act {g('act_up_F3_in', 'kurtosis'):.0f}")
    return out, line
