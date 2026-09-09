# Intent router — a fine-tuned classifier in front of the agent

A LangGraph agent routes every incoming message before it does anything else.
In the banking copilot that step was an LLM call with a structured output. This
replaces it with a fine-tuned encoder and measures what changed.

Routing is closed-set classification: 77 fixed outcomes, no generation. A
general model does it well, but pays a network round trip and per-token billing
on the highest-QPS node in the graph — every message hits the router — and
returns an answer that can differ between two identical inputs, on a step whose
output is written to an audit log.

## What's here

| File | |
|---|---|
| `train.py` | Fine-tune an encoder on BANKING77 (13,083 queries, 77 intents) |
| `bench.py` | Head-to-head against the same job done by an LLM: accuracy, latency, price |
| `calibrate.py` | Pick the escalation threshold from the confidence curve, not by taste |
| `router.py` | The drop-in replacement for the copilot's `route` node |
| `translate_he.py` | Build a Hebrew evaluation slice — the classes are English-only otherwise |

## The design

Cascade, the same shape as the copilot's grounding check: the cheap local model
settles the common case, the expensive general model runs only on what the cheap
one can't account for. There the residual was arithmetic it couldn't derive;
here it's confidence below threshold.

The escalation path is load-bearing, not a nicety. BANKING77 has 77 classes and
every one of them is a banking request — there is no negative class. The
classifier has never seen "what's the weather" or a prompt-injection attempt and
will assign one of its 77 labels to both, confidently. Max softmax is the
out-of-distribution signal and everything under the threshold goes to the LLM,
which can say "this isn't banking".

That is a real limitation, and a threshold tuned on in-distribution data tells
you nothing about traffic you've never seen. A production version mines a
negative class out of chat logs.

## Hebrew

BANKING77 is English. An Israeli bank's customers mostly are not, and the multilingual claim
is central to the product, so an English-only accuracy figure answers the easy
half of the question. `translate_he.py` builds a stratified Hebrew slice — every
intent represented, the same examples in both languages, so the delta is
attributable to language and not to a different sample.

Machine-translated and spot-checked, and reported that way. It measures whether
the classifier survives Hebrew. It does not measure whether it survives how
Israelis actually write to their bank, which is code-switched, abbreviated, and
full of English product names in Latin script. That needs real logs.

## Results

Not yet measured — this section gets the numbers from `results.json` and
`calibration.json`, not prose.

## Running it

Training runs on the GTX 1080 box, not a laptop.

```bash
python train.py --model distilbert-base-uncased --out runs/distilbert
python train.py --model xlm-roberta-base       --out runs/xlmr      # multilingual
python calibrate.py --model runs/distilbert/best
python translate_he.py
python bench.py --model runs/xlmr/best --hebrew
```
