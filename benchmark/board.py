"""Every saved benchmark result next to the reference models, one markdown table per shot count.

    python -m benchmark.board [--shots 0,5]

Reads benchmark/results/*.json (written by benchmark/eval.py) and benchmark/references.json. One metric
per task, the one papers quote: acc_norm for HellaSwag / PIQA / XStoryCloze, acc for ARC-Easy,
WinoGrande, LAMBADA and Belebele (letter answers, where length normalisation means nothing).
"""
import argparse
import glob
import json
import os

METRIC = {"hellaswag": "acc_norm", "arc_easy": "acc", "arc_challenge": "acc_norm", "piqa": "acc_norm",
          "winogrande": "acc", "lambada": "acc", "belebele_eng": "acc",
          "xstorycloze_hi": "acc_norm", "indiccopa_hi": "acc", "xnli_hi": "acc", "arc_challenge_hi": "acc_norm",
          "belebele_hin": "acc", "mmlu_hi": "acc"}
HERE = os.path.dirname(__file__)


def rows():
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
        r = json.load(open(p))
        m = r["meta"]
        out.append((f"{m['run_tag']} @ {m['sub']}", m["tokens"], r["scores"]))
    ref = json.load(open(os.path.join(HERE, "references.json")))
    out += [(f"{k} *", v["tokens"], v["scores"]) for k, v in ref.items() if not k.startswith("_")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="0,5")
    a = ap.parse_args()
    R = rows()
    for k in a.shots.split(","):
        print(f"\n### {k}-shot  (* = approximate published number, not rerun)\n")
        print("| model | tokens | " + " | ".join(f"{t} ({m})" for t, m in METRIC.items()) + " |")
        print("|---|---|" + "---|" * len(METRIC))
        for name, tok, sc in R:
            cells = [sc.get(f"{t}_{k}shot", {}).get(m) for t, m in METRIC.items()]
            if all(c is None for c in cells):
                continue
            ts = f"{tok / 1e9:.2f}B" if tok else "-"
            print(f"| {name} | {ts} | " + " | ".join("-" if c is None else f"{c:.1f}" for c in cells) + " |")


if __name__ == "__main__":
    main()
