"""What Hebrew costs, in tokens, on the models a bank would actually call.

The claim "Hebrew is expensive" is repeated everywhere and almost never with a
number. There are 308 sentence pairs in data/test_he.json that say the same
thing in both languages, which is exactly the corpus needed to measure it.

Two tokenizers matter for different reasons:
  - the LLM's, because it sets latency and the bill on every turn
  - the encoder's, because it sets whether a fine-tuned router can read Hebrew
"""
import json
import os
import statistics
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
import anthropic

ROWS = json.loads(Path("data/test_he.json").read_text(encoding="utf-8"))
BATCH = 40


def claude_tokens(client, model, texts):
    """count_tokens is the only honest way to do this -- no local tokenizer for
    Claude, and estimating from character counts is how people get this wrong."""
    total = []
    for t in texts:
        n = client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": t}]).input_tokens
        total.append(n)
    return total


def main():
    client = anthropic.Anthropic()
    sample = ROWS[:120]                      # enough for a stable ratio
    en = [r["text"] for r in sample]
    he = [r["text_he"] for r in sample]

    out = {}
    for model in ("claude-sonnet-4-6",):
        en_tok = claude_tokens(client, model, en)
        he_tok = claude_tokens(client, model, he)
        ratios = [h / e for h, e in zip(he_tok, en_tok) if e]
        out[model] = {
            "n": len(sample),
            "en_tokens_mean": round(statistics.mean(en_tok), 1),
            "he_tokens_mean": round(statistics.mean(he_tok), 1),
            "ratio_mean": round(statistics.mean(ratios), 3),
            "ratio_median": round(statistics.median(ratios), 3),
            "en_chars_mean": round(statistics.mean(len(t) for t in en), 1),
            "he_chars_mean": round(statistics.mean(len(t) for t in he), 1),
        }
        print(model, json.dumps(out[model], indent=2), flush=True)

    # The encoder side, offline.
    from transformers import AutoTokenizer
    for name, path in [("xlm-roberta (multilingual)", "runs/xlmr/best"),
                       ("distilbert (English-only)", "runs/distilbert/best")]:
        if not Path(path).exists():
            continue
        tok = AutoTokenizer.from_pretrained(path)
        e = [len(tok.tokenize(t)) for t in en]
        h = [len(tok.tokenize(t)) for t in he]
        out[name] = {
            "en_tokens_mean": round(statistics.mean(e), 1),
            "he_tokens_mean": round(statistics.mean(h), 1),
            "ratio_mean": round(statistics.mean(hh / ee for hh, ee in zip(h, e) if ee), 3),
        }
        print(name, json.dumps(out[name], indent=2), flush=True)

    Path("hebrew_tokens.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
