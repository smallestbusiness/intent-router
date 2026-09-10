"""Does an explicit negative class beat a confidence threshold?

Two ways to make a router refuse a non-banking message:

  A. THRESHOLD   77-class model, reject when max softmax < t.
                 Needs no negative training data at all.
  B. CLASS       78-class model with an out_of_scope class trained on 200
                 non-banking examples. Reject when that class wins.
  B+. CLASS+THRESHOLD   both, since they are not mutually exclusive.

Four numbers decide it, and three of them are about what the change costs:

  in-domain accuracy      did adding the class hurt the job it already did?
  false rejection rate    how much real banking traffic now gets refused?
  recall on SEEN OOD      does it catch the categories it was trained on?
  recall on UNSEEN OOD    does it catch the ones it was not?

The last is the whole question. "Not banking" is unbounded, so a model that only
catches the five categories it saw has not learned to refuse -- it has learned
five more intents.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import intents


@torch.no_grad()
def predict(path, texts, batch=64):
    """Return (predicted label name, confidence) for each text."""
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    names, confs = [], []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=64).to(device)
        probs = model(**enc).logits.softmax(-1)
        top = probs.max(-1)
        names += [model.config.id2label[int(j)] for j in top.indices]
        confs += top.values.tolist()
    return names, confs


def rate(flags):
    return sum(flags) / len(flags) if flags else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold-model", default="runs/xlmr/best")
    ap.add_argument("--class-model", default="runs/xlmr-neg/best")
    ap.add_argument("--t", type=float, default=0.75)
    ap.add_argument("--out", default="ood_results.json")
    args = ap.parse_args()

    _, test = intents.english()
    id_texts = list(test["text"])
    id_gold = [test.features["label"].names[i] for i in test["label"]]
    neg = intents.negatives()

    results = {"threshold": args.t}

    for tag, path in [("A_threshold_only", args.threshold_model),
                      ("B_negative_class", args.class_model)]:
        if not Path(path).exists():
            print(f"skip {tag}: {path} not built")
            continue
        block = {"model": path}

        # ---- in-domain: 3,080 real banking queries ----
        pred, conf = predict(path, id_texts)
        # Correct means the right intent AND not refused.
        by_class = [p == g for p, g in zip(pred, id_gold)]
        refused_class = [p == intents.OOS for p in pred]
        refused_thresh = [c < args.t for c in conf]
        refused_both = [a or b for a, b in zip(refused_class, refused_thresh)]

        block["in_domain_accuracy_raw"] = round(rate(by_class), 4)
        block["false_rejection_class_only"] = round(rate(refused_class), 4)
        block["false_rejection_threshold_only"] = round(rate(refused_thresh), 4)
        block["false_rejection_both"] = round(rate(refused_both), 4)
        block["in_domain_accuracy_after_refusal"] = round(
            rate([ok and not r for ok, r in zip(by_class, refused_both)]), 4)

        # ---- out of domain: seen categories, then unseen ones ----
        for split in ("test_seen", "test_unseen"):
            texts = [r["text"] for r in neg[split]]
            cats = [r["category"] for r in neg[split]]
            pred, conf = predict(path, texts)
            rc = [p == intents.OOS for p in pred]
            rt = [c < args.t for c in conf]
            rb = [a or b for a, b in zip(rc, rt)]
            block[split] = {
                "n": len(texts),
                "recall_class_only": round(rate(rc), 4),
                "recall_threshold_only": round(rate(rt), 4),
                "recall_both": round(rate(rb), 4),
            }
            per_cat = defaultdict(list)
            for c, r in zip(cats, rb):
                per_cat[c].append(r)
            block[split]["per_category_recall_both"] = {
                c: round(rate(v), 3) for c, v in sorted(per_cat.items())}

        results[tag] = block
        print(json.dumps({tag: block}, indent=2), flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
