"""Pick the escalation threshold from data instead of from taste.

The tradeoff is explicit: a higher threshold sends more traffic to the LLM,
which costs money and latency on every escalated turn but catches more of the
classifier's mistakes. This prints the curve so the number in router.py can be
defended with a reason rather than chosen because it looked round.
"""
import argparse
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import intents


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/distilbert/best")
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    _, test = intents.english()
    texts, labels = list(test["text"]), list(test["label"])

    confs, preds = [], []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], return_tensors="pt", padding=True,
                  truncation=True, max_length=64).to(device)
        probs = model(**enc).logits.softmax(-1)
        confs += probs.max(-1).values.tolist()
        preds += probs.argmax(-1).tolist()

    rows = []
    for t in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]:
        kept = [(c, p, l) for c, p, l in zip(confs, preds, labels) if c >= t]
        if not kept:
            continue
        rows.append({
            "threshold": t,
            "handled_locally_pct": round(100 * len(kept) / len(labels), 1),
            # Accuracy on the traffic the small model actually keeps. This is the
            # number that matters: the escalated tail is the LLM's problem.
            "accuracy_on_kept": round(sum(p == l for _, p, l in kept) / len(kept), 4),
            "escalated_pct": round(100 * (1 - len(kept) / len(labels)), 1),
        })

    print(json.dumps(rows, indent=2))
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
