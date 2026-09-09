"""Build the Hebrew evaluation slice.

BANKING77 is English only. An Israeli bank's customers mostly are not, and a banking assistant is sold as
multilingual, so an English-only accuracy figure answers the easy half of the
question. This translates a stratified sample of the English test set -- every
intent represented -- so the same examples can be scored in both languages and
the drop attributed to language rather than to a different sample.

Machine translation with spot-checking, and reported that way. A translated test
set measures whether the classifier survives Hebrew; it does not measure whether
it survives how Israelis actually write to their bank, which is code-switched,
abbreviated and full of English product names. That needs real logs.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

MODEL = "claude-sonnet-4-6"
BATCH = 25


def stratified(test, per_intent):
    by_intent = defaultdict(list)
    for row in test:
        by_intent[row["label"]].append(row["text"])
    out = []
    for label, texts in sorted(by_intent.items()):
        for text in texts[:per_intent]:
            out.append({"text": text, "label": label})
    return out


def dump(args):
    """Write the English sample. Runs where `datasets` is -- the GPU box."""
    import intents
    _, test = intents.english()
    rows = stratified(test, args.per_intent)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} examples across "
          f"{len(set(r['label'] for r in rows))} intents to {out}")


def translate(args):
    """Translate the sample. Runs where the API key is -- the laptop.

    Split from `dump` on purpose: the GPU box needs the dataset library and the
    laptop holds the credential, and neither needs to acquire the other's.
    """
    import anthropic

    rows = json.loads(Path(args.out).read_text(encoding="utf-8"))
    todo = [r for r in rows if not r.get("text_he")]
    print(f"{len(todo)} of {len(rows)} still to translate")

    client = anthropic.Anthropic()
    for start in range(0, len(todo), BATCH):
        chunk = todo[start:start + BATCH]
        numbered = "\n".join(f"{i+1}. {r['text']}" for i, r in enumerate(chunk))
        msg = client.messages.create(
            model=MODEL, max_tokens=4000,
            system=("Translate each numbered line into natural modern Hebrew as an "
                    "Israeli bank customer would write it in a chat window. Keep it "
                    "colloquial, not formal written Hebrew. Leave product and brand "
                    "names in Latin script if that is how Israelis write them. "
                    "Return the same numbering, one translation per line, nothing else."),
            messages=[{"role": "user", "content": numbered}])
        lines = [l for l in msg.content[0].text.strip().splitlines() if l.strip()]
        assert len(lines) == len(chunk), f"got {len(lines)} for {len(chunk)}"
        for row, line in zip(chunk, lines):
            row["text_he"] = line.split(".", 1)[1].strip() if "." in line[:4] else line.strip()
        print(f"  {start + len(chunk)}/{len(todo)}", flush=True)
        # Written every batch, so an API hiccup costs one batch and not the run.
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                  encoding="utf-8")

    print(f"wrote {args.out}")
    for r in rows[:3]:
        print(f"  {r['text']}  ->  {r['text_he']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["dump", "translate"])
    ap.add_argument("--per-intent", type=int, default=4)   # 77 * 4 = 308 examples
    ap.add_argument("--out", default="data/test_he.json")
    a = ap.parse_args()
    (dump if a.step == "dump" else translate)(a)
