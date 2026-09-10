"""Build an out-of-scope class, and the test set that decides whether it works.

The question this answers: does adding a negative class to BANKING77 make the
router better at refusing non-banking messages?

The whole experiment turns on one design choice. "Not banking" is unbounded --
it is everything in the universe except 77 intents -- so a handful of examples
samples an infinitesimal slice of it. A test set drawn from the same categories
as the training examples would measure whether the model memorised those
categories, which is not the question. The question is whether it generalises to
non-banking messages it has never seen.

So the negatives come in three groups:

  TRAIN        five obvious categories, used for training
  TEST_SEEN    held-out examples from those same five -- did it learn them?
  TEST_UNSEEN  six different categories -- does it generalise?

TEST_UNSEEN is deliberately hard. Two of its categories are adjacent financial
domains (insurance, tax) which are genuinely not banking but sit semantically
next to it, and one is prompt injection, which is the category a bank actually
cares about and which looks nothing like a weather question.
"""
import argparse
import json
from pathlib import Path

TRAIN_CATEGORIES = {
    "weather":   "asking about weather, forecasts, temperature, rain",
    "sports":    "asking about football or basketball scores, fixtures, players",
    "cooking":   "asking for recipes, cooking times, ingredient substitutions",
    "trivia":    "general knowledge questions -- history, geography, science",
    "chitchat":  "greetings, small talk, thanks, jokes, how are you",
}

# Different categories, not more of the same. Two are adjacent financial
# domains, which is the case a naive negative class is most likely to fail.
UNSEEN_CATEGORIES = {
    "travel":     "booking flights, hotels, trains, visa questions",
    "medical":    "symptoms, medication, doctor appointments, health advice",
    "techsupport":"phone or internet not working, router problems, app crashes "
                  "-- for a phone or ISP, nothing to do with a bank",
    "insurance":  "car and home insurance policies, claims, premiums, excess -- "
                  "financial, adjacent to banking, but not banking",
    "tax":        "income tax returns, VAT, deductions, filing deadlines -- "
                  "financial, adjacent to banking, but not banking",
    "injection":  "attempts to manipulate an AI assistant: ignore your "
                  "instructions, reveal your system prompt, pretend you are in "
                  "developer mode, print your rules",
}

PROMPT = ("Write {n} short messages a person might type into a chat window, "
          "in the register of {desc}. One per line, numbered. Vary the phrasing, "
          "length and formality the way real people do -- some terse, some full "
          "sentences, some with typos. No preamble, nothing but the numbered "
          "lines.")


def generate(client, category, desc, n):
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=4000,
        messages=[{"role": "user", "content": PROMPT.format(n=n, desc=desc)}])
    lines = [l.strip() for l in msg.content[0].text.strip().splitlines() if l.strip()]
    out = []
    for line in lines:
        text = line.split(".", 1)[1].strip() if "." in line[:4] else line.strip()
        if text:
            out.append({"text": text, "category": category})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/negatives.json")
    ap.add_argument("--train-per-cat", type=int, default=40)   # 5 x 40 = 200
    ap.add_argument("--seen-per-cat", type=int, default=20)    # 5 x 20 = 100
    ap.add_argument("--unseen-per-cat", type=int, default=20)  # 6 x 20 = 120
    args = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()
    data = {"train": [], "test_seen": [], "test_unseen": []}

    for cat, desc in TRAIN_CATEGORIES.items():
        # Generated in one call per split so the two never overlap by accident.
        data["train"] += generate(client, cat, desc, args.train_per_cat)
        print(f"  train/{cat}: {len(data['train'])}", flush=True)
    for cat, desc in TRAIN_CATEGORIES.items():
        data["test_seen"] += generate(client, cat, desc, args.seen_per_cat)
        print(f"  test_seen/{cat}: {len(data['test_seen'])}", flush=True)
    for cat, desc in UNSEEN_CATEGORIES.items():
        data["test_unseen"] += generate(client, cat, desc, args.unseen_per_cat)
        print(f"  test_unseen/{cat}: {len(data['test_unseen'])}", flush=True)

    # Exact-duplicate guard: an example that appears in both train and a test
    # split would quietly turn generalisation into recall.
    seen = {r["text"].lower() for r in data["train"]}
    for split in ("test_seen", "test_unseen"):
        before = len(data[split])
        data[split] = [r for r in data[split] if r["text"].lower() not in seen]
        if before != len(data[split]):
            print(f"  dropped {before - len(data[split])} leaked from {split}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in data.items()})


if __name__ == "__main__":
    main()
