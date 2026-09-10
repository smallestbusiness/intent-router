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

import intents


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
    # Freeze the pretrained encoder and train only the added head -- a "linear
    # probe". Cheaper and safer-sounding, and the ablation exists to show what
    # it actually costs in accuracy rather than to argue about it.
    ap.add_argument("--freeze-encoder", action="store_true")
    # LoRA: freeze the base weights and learn a low-rank update to the attention
    # projections instead. The reason it is here is memory, not accuracy --
    # xlm-roberta-large needs ~9 GB of AdamW optimiser state to fine-tune fully
    # and the card has 8 GB, so full fine-tuning it is not an option and this is.
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    # XLM-R / RoBERTa name their attention projections query/key/value. Q and V
    # are the conventional targets: the original paper found adapting those two
    # matches adapting all four at half the parameters.
    ap.add_argument("--lora-target", default="query,value")
    # Add a 78th "out_of_scope" class trained on non-banking messages, instead
    # of relying on a confidence threshold to spot them.
    ap.add_argument("--negatives", action="store_true")
    args = ap.parse_args()

    if args.negatives:
        train, test, names = intents.english_with_negatives()
    else:
        train, test = intents.english()
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

    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.1,
            target_modules=args.lora_target.split(",")))
        model.print_trainable_parameters()

    if args.freeze_encoder:
        trainable = ("classifier", "pre_classifier", "score")
        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(t) for t in trainable)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"frozen encoder: training {n_train:,} of {n_all:,} parameters "
              f"({100*n_train/n_all:.2f}%)")

    steps_per_epoch = -(-len(train) // args.batch)
    warmup_steps = int(0.10 * steps_per_epoch * args.epochs)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=128,
        learning_rate=args.lr,
        # transformers 5.x dropped warmup_ratio; warmup_steps is computed from
        # the same 10% so the schedule is unchanged and the number stays visible.
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        # Not fp16: the 1080 is Pascal, which has no tensor cores and runs half
        # precision at a fraction of fp32 throughput. Half precision is a win on
        # Ampere and later, and a loss here.
        fp16=False,
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

    if args.lora:
        # Merge the adapter into the base weights and save an ordinary model.
        # This is the production path -- W' = W + BA is just a weight matrix, so
        # a merged LoRA model has exactly zero inference overhead. It also means
        # bench.py loads it with plain from_pretrained and the comparison stays
        # like for like.
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(out / "best")
    else:
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
