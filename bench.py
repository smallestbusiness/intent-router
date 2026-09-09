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

# torch, transformers and the dataset library are imported inside the functions
# that need them. The LLM half of this benchmark runs on the laptop, which holds
# the API key and none of the training stack; keeping these out of module scope
# is what lets bench_llm.py import llm_eval there.

LLM_MODEL = "claude-sonnet-4-6"
PRICE_IN, PRICE_OUT = 3.00 / 1e6, 15.00 / 1e6      # $/token, claude-sonnet-4-6
PRICE_CACHE_WRITE, PRICE_CACHE_READ = 3.75 / 1e6, 0.30 / 1e6


# ---------- the fine-tuned router ----------

def load(path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    return tok, model


def encoder_eval(tok, model, texts, labels, device, batch=64):
    import torch
    with torch.no_grad():
        return _encoder_eval(torch, tok, model, texts, labels, device, batch)


def _encoder_eval(torch, tok, model, texts, labels, device, batch):
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

    # No prompt caching, and not for want of trying. The ~1.3k-token tool schema
    # is identical on every request and is exactly what caching exists for, so
    # the intent was to measure the baseline with it on -- beating an unoptimised
    # opponent proves nothing. What actually happened, on anthropic SDK 0.69 /
    # claude-sonnet-4-6:
    #
    #   cache_control on the tool definition   -> 0 write, 0 read (dropped)
    #   cache_control on a system content block-> 0 write, 0 read (dropped)
    #   top-level cache_control=ephemeral      -> ~1,287 write per request, 0 read
    #
    # The third one fires but places the breakpoint after the last cacheable
    # block, which is the customer message -- so the prefix changes every request
    # and the result is a cache write every time and never a hit. That made the
    # baseline *more* expensive than no caching ($5,435 vs $4,460 per 1M), which
    # would have flattered the encoder for the wrong reason.
    #
    # So the figure below is the uncached one, and it is the honest one to quote.
    # A correctly placed breakpoint would put the ~1.3k prefix at 0.1x, taking
    # this to roughly $930 per 1M requests -- still three orders of magnitude
    # above an encoder running on hardware that is already paid for.
    correct, times, tin, tout, tread, twrite = 0, [], 0, 0, 0, 0
    for n, (text, label) in enumerate(zip(texts[:limit], labels[:limit])):
        t0 = time.perf_counter()
        msg = client.messages.create(
            model=LLM_MODEL, max_tokens=200, system=system, tools=[tool],
            tool_choice={"type": "tool", "name": "route"},
            messages=[{"role": "user", "content": text}])
        times.append((time.perf_counter() - t0) * 1000)
        tin += msg.usage.input_tokens
        tout += msg.usage.output_tokens
        tread += msg.usage.cache_read_input_tokens or 0
        twrite += msg.usage.cache_creation_input_tokens or 0
        block = next((b for b in msg.content if b.type == "tool_use"), None)
        if block and block.input["intent"] == names[label]:
            correct += 1
        if (n + 1) % 25 == 0:
            print(f"  llm {n+1}/{min(limit, len(texts))} acc={correct/(n+1):.3f}", flush=True)

    n = len(times)
    per_request = (tin * PRICE_IN + tout * PRICE_OUT
                   + tread * PRICE_CACHE_READ + twrite * PRICE_CACHE_WRITE) / n
    return {"accuracy": correct / n, "n": n,
            "p50_ms": round(statistics.median(times), 1),
            "p95_ms": round(statistics.quantiles(times, n=20)[18], 1),
            "usd_per_1m_requests": round(per_request * 1e6, 2),
            "cache_hit_rate": round(tread / max(tread + tin, 1), 3),
            "tokens_per_request": {"uncached_in": round(tin / n, 1),
                                   "cache_read": round(tread / n, 1),
                                   "cache_write": round(twrite / n, 1),
                                   "out": round(tout / n, 1)}}


def main():
    import intents
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/distilbert/best")
    ap.add_argument("--llm-limit", type=int, default=385)   # 5 per intent
    ap.add_argument("--hebrew", action="store_true")
    # The LLM side of the comparison does not change when the encoder does, so
    # it is measured once and reused rather than re-billed per encoder.
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--out", default="results.json")
    # The LLM half runs on the laptop, which holds the credential; this writes
    # the exact examples it must score so both halves see identical inputs.
    ap.add_argument("--dump-llm-set", default="data/llm_eval_set.json")
    args = ap.parse_args()

    _, test = intents.english()
    names = test.features["label"].names
    tok, model = load(args.model)

    results = {"base_model": model.config._name_or_path, "llm_model": LLM_MODEL}

    texts, labels = list(test["text"]), list(test["label"])
    results["encoder_en_gpu"] = encoder_eval(tok, model, texts, labels, "cuda")
    results["encoder_en_cpu"] = encoder_eval(tok, model, texts, labels, "cpu")
    print(json.dumps(results["encoder_en_gpu"], indent=2), flush=True)

    if args.hebrew:
        rows = intents.hebrew_test()
        he_texts = [r["text_he"] for r in rows]
        he_labels = [r["label"] for r in rows]
        # The same examples in English, so the delta is language and nothing else.
        results["encoder_he_gpu"] = encoder_eval(tok, model, he_texts, he_labels, "cuda")
        results["encoder_en_samesample"] = encoder_eval(
            tok, model, [r["text"] for r in rows], he_labels, "cuda")
        if not args.skip_llm:
            results["llm_he"] = llm_eval(he_texts, he_labels, names, args.llm_limit)

    if not args.skip_llm:
        results["llm_en"] = llm_eval(texts, labels, names, args.llm_limit)

    if args.dump_llm_set:
        d = Path(args.dump_llm_set)
        d.parent.mkdir(parents=True, exist_ok=True)
        payload = {"names": names,
                   "en": {"texts": texts[:args.llm_limit],
                          "labels": labels[:args.llm_limit]}}
        if args.hebrew:
            rows = intents.hebrew_test()
            payload["he"] = {"texts": [r["text_he"] for r in rows][:args.llm_limit],
                             "labels": [r["label"] for r in rows][:args.llm_limit]}
        d.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {d} for the LLM half")

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
