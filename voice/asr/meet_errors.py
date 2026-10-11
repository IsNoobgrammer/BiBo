"""Meeting-clip error dump: align ab_meeting --dump hypotheses to the references and list what went wrong.

    python voice/asr/meet_errors.py DUMP.json [--model NAME] [--la 3] [--bp 1.00] [--ctx 4] > errors.txt

Per clip: every substitution (ref -> hyp), deletion and insertion with a few words of reference context, then the
most frequent ones pooled over the clips. One minimum-edit-distance alignment (the same cost as score.edits).
"""
import argparse
import collections
import json


def align(ref, hyp):
    """[(op, ref_index, ref_word, hyp_word)] with op in = S D I, by edit distance with backtrace."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append(("=" if ref[i - 1] == hyp[j - 1] else "S", i - 1, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append(("D", i - 1, ref[i - 1], ""))
            i -= 1
        else:
            out.append(("I", i, "", hyp[j - 1]))
            j -= 1
    return out[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--model", default=None, help="model name in the dump (default: the first)")
    ap.add_argument("--la", default="3", help="look-ahead in 80 ms frames, -1 = full")
    ap.add_argument("--bp", default="1.00")
    ap.add_argument("--ctx", type=int, default=4)
    a = ap.parse_args()
    dmp = json.load(open(a.dump, encoding="utf-8"))
    model = a.model or next(iter(dmp["hyps"]))
    hyps = dmp["hyps"][model][a.la][a.bp]
    pooled = {"S": collections.Counter(), "D": collections.Counter(), "I": collections.Counter()}
    tot = collections.Counter()
    for clip, ref, hyp in zip(dmp["clips"], dmp["refs"], hyps):
        r, h = ref.split(), hyp.split()
        al = align(r, h)
        c = collections.Counter(op for op, *_ in al)
        tot.update(c)
        print(f"\n=== {clip}: {len(r)} ref words | S {c['S']} D {c['D']} I {c['I']} | "
              f"WER {100 * (c['S'] + c['D'] + c['I']) / len(r):.1f}")
        for op, i, rw, hw in al:
            if op == "=":
                continue
            ctx = " ".join(r[max(0, i - a.ctx):i]) + " [" + (rw or "+") + "] " + " ".join(r[i + 1:i + 1 + a.ctx])
            what = {"S": f"{rw} -> {hw}", "D": f"{rw} -> (dropped)", "I": f"(extra) {hw}"}[op]
            print(f"  {op} {what:34s} | {ctx}")
            pooled[op][{"S": f"{rw} -> {hw}", "D": rw, "I": hw}[op]] += 1
    n = sum(len(r.split()) for r in dmp["refs"])
    print(f"\n=== pooled ({model}, la {a.la}, bp {a.bp}): {n} words | S {tot['S']} D {tot['D']} I {tot['I']} | "
          f"WER {100 * (tot['S'] + tot['D'] + tot['I']) / n:.2f}")
    for op, name in (("S", "substitutions"), ("D", "deletions"), ("I", "insertions")):
        print(f"  top {name}: " + ", ".join(f"{k} x{v}" for k, v in pooled[op].most_common(40)))


if __name__ == "__main__":
    main()
