"""Re-calibrate the threshold, now that refusing has two mechanisms.

`calibrate.py` swept the threshold against in-domain accuracy alone, which was
the whole picture when the threshold was the only way to refuse. It is not any
more. The 78-class model can reject by class as well, its confidence
distribution has moved, and the threshold now trades two things against each
other:

    raise it   -> catch more off-topic traffic, refuse more real customers
    lower it   -> refuse fewer real customers, miss more off-topic traffic

So the curve has to carry both, and the two models have to be compared at a
matched operating point rather than at whatever threshold each happens to use.
The point chosen here is the highest off-topic recall available while refusing
no more than a fixed share of genuine banking traffic -- because false rejection
is what the customer feels, and it is the constraint a bank would actually set.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import intents

GRID = [0.0, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80,
        0.85, 0.90, 0.93, 0.95, 0.97, 0.99]


@torch.no_grad()
def scores(path, texts, batch=64):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    names, confs = [], []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=64).to(device)
        top = model(**enc).logits.softmax(-1).max(-1)
        names += [model.config.id2label[int(j)] for j in top.indices]
        confs += top.values.tolist()
    return names, confs


def sweep(path, id_texts, id_gold, ood_texts):
    """One row per threshold: what it costs in-domain, what it catches off-topic."""
    id_pred, id_conf = scores(path, id_texts)
    ood_pred, ood_conf = scores(path, ood_texts)
    rows = []
    for t in GRID:
        id_ref = [p == intents.OOS or c < t for p, c in zip(id_pred, id_conf)]
        kept_ok = [p == g for p, g, r in zip(id_pred, id_gold, id_ref) if not r]
        ood_ref = [p == intents.OOS or c < t for p, c in zip(ood_pred, ood_conf)]
        rows.append({
            "threshold": t,
            "false_rejection": round(sum(id_ref) / len(id_ref), 4),
            "handled_locally": round(1 - sum(id_ref) / len(id_ref), 4),
            "accuracy_on_kept": round(sum(kept_ok) / len(kept_ok), 4) if kept_ok else 0.0,
            "ood_recall_unseen": round(sum(ood_ref) / len(ood_ref), 4),
        })
    return rows


def best_under(rows, max_false_rejection):
    ok = [r for r in rows if r["false_rejection"] <= max_false_rejection]
    return max(ok, key=lambda r: r["ood_recall_unseen"]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="runs/xlmr/best,runs/xlmr-neg/best")
    ap.add_argument("--budgets", default="0.03,0.05,0.07,0.10")
    ap.add_argument("--out", default="calibration_ood.json")
    ap.add_argument("--adjacent", action="store_true")
    args = ap.parse_args()

    _, test = intents.english()
    id_texts = list(test["text"])
    id_gold = [test.features["label"].names[i] for i in test["label"]]
    ood_texts = [r["text"] for r in intents.negatives(args.adjacent)["test_unseen"]]

    out = {}
    for path in args.models.split(","):
        if not Path(path).exists():
            print(f"skip {path}")
            continue
        rows = sweep(path, id_texts, id_gold, ood_texts)
        out[path] = {"curve": rows, "operating_points": {}}
        print(f"\n=== {path} ===")
        print(f"{'t':>6} {'false rej':>10} {'acc on kept':>12} {'OOD unseen':>11}")
        for r in rows:
            print(f"{r['threshold']:>6.2f} {r['false_rejection']:>10.2%} "
                  f"{r['accuracy_on_kept']:>12.2%} {r['ood_recall_unseen']:>11.2%}")
        for b in args.budgets.split(","):
            b = float(b)
            best = best_under(rows, b)
            out[path]["operating_points"][str(b)] = best
            if best:
                print(f"  best t under {b:.0%} false rejection: t={best['threshold']:.2f} "
                      f"-> OOD {best['ood_recall_unseen']:.2%}, "
                      f"acc on kept {best['accuracy_on_kept']:.2%}")

    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
