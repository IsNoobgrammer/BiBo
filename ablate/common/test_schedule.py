"""Self-check for the LR schedules — no torch optimizer needed, just the lambda shapes.

  python -m ablate.common.test_schedule
"""
from .schedule import wsd_lambda, cosine_lambda


def test_wsd():
    T, wf, df = 1000, 0.05, 0.20                 # warm=50, decay_start=800
    f = wsd_lambda(T, wf, df, final_frac=0.0)
    assert f(0) == 0.0                            # warmup starts at 0
    assert abs(f(50) - 1.0) < 1e-9               # end of warmup -> full LR
    assert f(400) == 1.0                          # stable phase pinned at 1.0
    assert f(799) == 1.0                          # still stable just before decay
    assert f(800) == 1.0                          # decay just begins at 1.0
    assert f(T - 1) < 0.05 and f(T - 1) >= 0.0   # linear decay -> ~0 at the end
    print("OK: wsd warmup->stable->linear decay to 0")


def test_cosine():
    T, wf = 1000, 0.05                            # warm=50
    f = cosine_lambda(T, wf, final_frac=0.0)
    assert f(0) == 0.0
    assert abs(f(50) - 1.0) < 1e-9               # end of warmup -> peak
    mid = f(50 + (T - 50) // 2)                   # halfway through anneal ~ cos(pi/2)=0 -> ~0.5
    assert abs(mid - 0.5) < 0.02
    assert f(T - 1) < 0.01                         # cosine anneals to ~0
    assert f(400) > f(700) > f(T - 1)             # monotonically decreasing after warmup
    print("OK: cosine warmup->cosine anneal to 0, monotone")


def test_final_frac_floor():
    f = cosine_lambda(1000, 0.05, final_frac=0.1)
    assert abs(f(999) - 0.1) < 0.01               # floors at final_frac, not 0
    print("OK: final_frac floor respected")


if __name__ == "__main__":
    test_wsd()
    test_cosine()
    test_final_frac_floor()
    print("all self-checks passed")
