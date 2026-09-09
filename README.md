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

Trained on the GTX 1080 box. Accuracy is the full 3,080-example BANKING77
test set; latency is single-query, one at a time, because that is how a
router is actually called — batched throughput would flatter it.

| Model | Accuracy | F1-macro | Train | p50 GPU | p50 CPU |
|---|---|---|---|---|---|
| `distilbert-base-uncased` | 91.3% | 91.2% | 183 s | 2.29 ms | 13.41 ms |
| `xlm-roberta-base` | 93.1% | 93.1% | 492 s | 4.56 ms | 26.5 ms |

### Hebrew

The same 308 examples in both languages, so the difference is language and
not a different sample.

| Model | English | Hebrew | Drop |
|---|---|---|---|
| `distilbert-base-uncased` | 90.3% | 5.8% | -84.4 pts |
| `xlm-roberta-base` | 90.9% | 65.6% | -25.3 pts |

The English-only model is not degraded in Hebrew, it is destroyed — 5.8%
against a 1.3% chance floor. It never saw the script. That is the expected
result and it is here as a control, because "fine-tune a small model for
routing" is a sentence that needs the multilingual caveat attached to it
in an Israeli bank.

The multilingual model is the actual finding: 65.6% in Hebrew against
90.9% on the same sentences in English. It works, and it is nowhere near
good enough to route Hebrew traffic on its own at a threshold that makes
it worth having. Closing that gap needs Hebrew training data, not a better
base model — the model has the script, it does not have the domain in the
script.

### Escalation threshold

Max softmax on held-out English data. This is what the 0.75 default in
`router.py` is chosen from.

| Threshold | Handled locally | Accuracy on kept | Escalated |
|---|---|---|---|
| 0.5 | 92.9% | 94.7% | 7.1% |
| 0.7 | 84.6% | 97.6% | 15.4% |
| 0.75 **←** | 80.5% | 98.2% | 19.5% |
| 0.8 | 76.4% | 98.4% | 23.6% |
| 0.9 | 57.1% | 99.4% | 42.9% |
| 0.95 | 22.3% | 99.9% | 77.7% |

### Against an LLM doing the same job

`claude-sonnet-4-6` with a strict 77-value enum tool and forced tool choice,
so it cannot answer with anything that is not a valid intent — the baseline
loses on judgement or not at all. 150 requests per language, sequential,
because latency is what a router is judged on and concurrency would hide it.

| | Accuracy EN | Accuracy HE | p50 | p95 | $ / 1M requests |
|---|---|---|---|---|---|
| `xlm-roberta-base` | **93.1%** | 65.6% | 4.56 ms | 8.03 ms | no per-request billing |
| `distilbert-base-uncased` | 91.3% | 5.8% | 2.29 ms | 4.33 ms | no per-request billing |
| `claude-sonnet-4-6` | 85.3% | **77.3%** | 2080 ms | 5218 ms | $4,470 |

Read the two bolded cells together, because they disagree and that is the
useful part.

**In English the fine-tune wins outright** — 93.1% against 85.3%, at 4.6 ms
against 2,080 ms. That is a 450x latency difference on the node every
single message passes through, and the small model is *more* accurate,
because 10,003 labelled examples of exactly this task beat general
capability at a task this narrow.

**In Hebrew it loses** — 65.6% against 77.3%. The LLM degrades gracefully
across languages; the fine-tune degrades badly, because its Hebrew came
from the base model's pretraining and none of its 77 classes did.

Which is the argument for the cascade rather than for replacement. Route
English on the encoder and let confidence send the tail to the LLM; Hebrew
traffic escalates far more often until there is Hebrew training data, and
the confidence threshold makes that happen on its own rather than needing a
language switch in the code.

**On the price column.** The encoder cell says "no per-request billing"
rather than $0 — the GPU box is a fixed cost either way, and at 437
requests/second single-stream it is nowhere near saturated by this
workload. The honest comparison is a fixed cost against $4,470 per million,
not a made-up per-request number.

**The LLM figure is uncached, and that is a real caveat rather than a
rhetorical one.** The ~1.3k-token tool schema is identical on every request
and is exactly what prompt caching is for. Three placements were tried;
`cache_control` on the tool definition and on a system content block were
both silently dropped, and top-level auto-caching put the breakpoint after
the customer message, producing a cache write on every request and never a
read — which made the baseline *more* expensive ($5,435 per 1M) rather than
less. A correctly placed breakpoint would put that prefix at 0.1x and take
the LLM to roughly $930 per million. That does not change the conclusion,
and quoting the $4,470 without saying this would be dishonest.

## Running it

Training runs on the GTX 1080 box, not a laptop.

```bash
python train.py --model distilbert-base-uncased --out runs/distilbert
python train.py --model xlm-roberta-base       --out runs/xlmr      # multilingual
python calibrate.py --model runs/distilbert/best
python translate_he.py
python bench.py --model runs/xlmr/best --hebrew
```
