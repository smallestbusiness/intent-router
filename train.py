"""Fine-tune an encoder classifier for intent routing.

The point of this file: routing is a closed-set
classification problem with 77 fixed outcomes and no generation in it. Handing
it to a general LLM buys nothing and costs a network round trip, per-token
billing, and a non-deterministic answer on a step whose output gets written to
an audit log. A 3-hour fine-tune replaces that with a model that answers in
single-digit milliseconds for the same money every time.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

import banking77


def metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro")}


def main():
    ap = argparse.ArgumentParser()
    # distilbert for the English baseline; xlm-roberta-base when the run has to
    # answer in Hebrew as well. Same script, so the two are comparable.
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--out", default="runs/distilbert")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=64)   # p99 query is far shorter
    args = ap.parse_args()

    train, test = banking77.english()
    names = test.features["label"].names
    print(f"{len(train)} train / {len(test)} test / {len(names)} intents")

    tok = AutoTokenizer.from_pretrained(args.model)

    def encode(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    train = train.map(encode, batched=True)
    test = test.map(encode, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(names),
        id2label=dict(enumerate(names)),
        label2id={n: i for i, n in enumerate(names)})

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=128,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,          # 48 cores on the box; this is plenty
        report_to=[],
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train,
                      eval_dataset=test, processing_class=tok,
                      compute_metrics=metrics)

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    final = trainer.evaluate()
    out = Path(args.out)
    trainer.save_model(out / "best")
    tok.save_pretrained(out / "best")
    (out / "result.json").write_text(json.dumps({
        "base_model": args.model,
        "train_seconds": round(train_seconds, 1),
        "accuracy": final["eval_accuracy"],
        "f1_macro": final["eval_f1_macro"],
        "epochs": args.epochs,
    }, indent=2))
    print(json.dumps({"accuracy": final["eval_accuracy"],
                      "f1_macro": final["eval_f1_macro"],
                      "train_seconds": round(train_seconds, 1)}, indent=2))


if __name__ == "__main__":
    main()
