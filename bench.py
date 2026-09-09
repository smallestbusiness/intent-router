"""Head-to-head: the fine-tuned router against an LLM doing the same job.

Three numbers decide whether a fine-tune was worth doing, and they are not all
accuracy. Accuracy says whether the small model can do the job at all. Latency
says what it costs the customer, on a step that sits in front of every single
turn. Price says what it costs the bank at volume. A router is the highest-QPS
node in the graph -- every message hits it -- so a saving there is multiplied by
everything.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import anthropic
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import banking77

LLM_MODEL = "claude-sonnet-4-6"
PRICE_IN, PRICE_OUT = 3.00 / 1e6, 15.00 / 1e6      # $/token, claude-sonnet-4-6


# ---------- the fine-tuned router ----------

def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    return tok, model


@torch.no_grad()
def encoder_eval(tok, model, texts, labels, device, batch=64):
    model.to(device)
    preds = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=64).to(device)
        preds += model(**enc).logits.argmax(-1).tolist()
    correct = sum(p == l for p, l in zip(preds, labels))

    # Latency is measured one query at a time, because that is how a router is
    # actually called. Batched throughput would flatter it.
    warm = tok(texts[0], return_tensors="pt").to(device)
    for _ in range(10):
        model(**warm)
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for text in texts[:200]:
        enc = tok(text, return_tensors="pt").to(device)
        t0 = time.perf_counter()
        model(**enc)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return {"accuracy": correct / len(labels),
            "p50_ms": round(statistics.median(times), 2),
            "p95_ms": round(statistics.quantiles(times, n=20)[18], 2),
            "device": device}


# ---------- the LLM doing the same job ----------

def llm_eval(texts, labels, names, limit):
    """Same 77 classes, forced through a strict tool schema so the model cannot
    answer with anything that is not a valid intent. This is the fair version of
    the baseline -- free-text output would lose on parsing, not on judgement."""
    client = anthropic.Anthropic()
    tool = {"name": "route", "description": "Assign the customer message to one intent.",
            "strict": True,
            "input_schema": {"type": "object", "additionalProperties": False,
                             "required": ["intent"],
                             "properties": {"intent": {"type": "string", "enum": names}}}}
    system = ("You route a bank customer's message to exactly one of 77 intents. "
              "Call the route tool with the single best-fitting intent.")

    correct, times, tin, tout = 0, [], 0, 0
    for n, (text, label) in enumerate(zip(texts[:limit], labels[:limit])):
        t0 = time.perf_counter()
        msg = client.messages.create(
            model=LLM_MODEL, max_tokens=200, system=system, tools=[tool],
            tool_choice={"type": "tool", "name": "route"},
            messages=[{"role": "user", "content": text}])
        times.append((time.perf_counter() - t0) * 1000)
        tin += msg.usage.input_tokens
        tout += msg.usage.output_tokens
        block = next((b for b in msg.content if b.type == "tool_use"), None)
        if block and block.input["intent"] == names[label]:
            correct += 1
        if (n + 1) % 25 == 0:
            print(f"  llm {n+1}/{min(limit, len(texts))} acc={correct/(n+1):.3f}", flush=True)

    n = len(times)
    return {"accuracy": correct / n, "n": n,
            "p50_ms": round(statistics.median(times), 1),
            "p95_ms": round(statistics.quantiles(times, n=20)[18], 1),
            "usd_per_1m_requests": round((tin / n * PRICE_IN + tout / n * PRICE_OUT) * 1e6, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/distilbert/best")
    ap.add_argument("--llm-limit", type=int, default=385)   # 5 per intent
    ap.add_argument("--hebrew", action="store_true")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    _, test = banking77.english()
    names = test.features["label"].names
    tok, model = load(args.model)

    results = {"base_model": model.config._name_or_path, "llm_model": LLM_MODEL}

    texts, labels = list(test["text"]), list(test["label"])
    results["encoder_en_gpu"] = encoder_eval(tok, model, texts, labels, "cuda")
    results["encoder_en_cpu"] = encoder_eval(tok, model, texts, labels, "cpu")
    print(json.dumps(results["encoder_en_gpu"], indent=2), flush=True)

    if args.hebrew:
        rows = banking77.hebrew_test()
        he_texts = [r["text_he"] for r in rows]
        he_labels = [r["label"] for r in rows]
        # The same examples in English, so the delta is language and nothing else.
        results["encoder_he_gpu"] = encoder_eval(tok, model, he_texts, he_labels, "cuda")
        results["encoder_en_samesample"] = encoder_eval(
            tok, model, [r["text"] for r in rows], he_labels, "cuda")
        results["llm_he"] = llm_eval(he_texts, he_labels, names, args.llm_limit)

    results["llm_en"] = llm_eval(texts, labels, names, args.llm_limit)

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
