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
| `xlm-roberta-large` + LoRA | **94.2%** | 94.2% | 1,227 s | 12.32 ms | 86.44 ms |

`xlm-roberta-large` is 560M parameters and cannot be fully fine-tuned on an 8 GB
card — AdamW in fp32 needs about 16 bytes per parameter, so roughly 9 GB of
optimiser state before activations. LoRA trains a rank-16 update to the query
and value projections instead: **2,701,389 trainable of 562,670,746 (0.48%),
peaking at 3,752 MiB**. The adapter is merged into the base weights before
saving, so the benchmarked model is an ordinary one with zero inference
overhead.

### Hebrew

The same 308 examples in both languages, so the difference is language and
not a different sample.

| Model | English | Hebrew | Drop |
|---|---|---|---|
| `distilbert-base-uncased` | 90.3% | 5.8% | -84.4 pts |
| `xlm-roberta-base` | 90.9% | 65.6% | -25.3 pts |
| `xlm-roberta-large` + LoRA | 94.2% | **82.8%** | **-11.4 pts** |

The English-only model is not degraded in Hebrew, it is destroyed — 5.8%
against a 1.3% chance floor. It never saw the script. That is the expected
result and it is here as a control, because "fine-tune a small model for
routing" is a sentence that needs the multilingual caveat attached to it
in an Israeli bank.

The multilingual result was where a prediction got corrected. On
`xlm-roberta-base`, Hebrew scored 65.6% against 90.9% on the same
sentences, and the reading was that the gap is training data rather than
model capacity — the model has the script, it does not have the domain in
the script.

Going to `xlm-roberta-large` with LoRA, **without adding one Hebrew
training example**, took Hebrew to 82.8% and halved the cross-lingual gap
from 25.3 points to 11.4. Cross-lingual transfer is itself a capability
that scales: 24 layers and 1024 dimensions against 12 and 768 align Hebrew
and English far better, so an English-only fine-tuning signal carries much
further across.

So the gap is both, and capacity was the cheaper half — 17 points for one
20-minute run on hardware already owned, against a translation pipeline and
a second training set. Hebrew data still closes the rest: 82.8% is not
production-grade for a bank's primary language. It is the second move now,
not the first.

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
| `xlm-roberta-large` + LoRA | **94.2%** | **82.8%** | 12.32 ms | 18.16 ms | no per-request billing |
| `xlm-roberta-base` | 93.1% | 65.6% | 4.56 ms | 8.03 ms | no per-request billing |
| `distilbert-base-uncased` | 91.3% | 5.8% | 2.29 ms | 4.33 ms | no per-request billing |
| `claude-sonnet-4-6` | 85.3% | 77.3% | 2080 ms | 5218 ms | $4,470 |

**The best fine-tune beats the LLM in both languages** — 94.2% against
85.3% in English, 82.8% against 77.3% in Hebrew, at 12 ms against 2,080 ms.
A 170x latency difference on the node every single message passes through,
and the small model is *more* accurate, because 10,003 labelled examples of
exactly this task beat general capability at a task this narrow.

That was not true two runs earlier, and the order matters. On
`xlm-roberta-base` the encoder won English and **lost** Hebrew, 65.6%
against 77.3% — which read as an argument for the cascade on language
grounds. Model capacity took that argument away.

**The cascade still stands, on the leg that was always load-bearing.** Not
"the LLM covers Hebrew", but: BANKING77 has no negative class, so the
classifier cannot represent "this is not a banking question" and will label
one confidently. Low confidence escalates. `xlm-roberta-large` is also 12 ms
against 2.3 ms and 2.2 GB against 260 MB, so which encoder to actually
deploy is a real trade rather than an obvious upgrade.

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

## Does a negative class help?

BANKING77 has no class for "this is not a banking question", so the router
relies on a confidence threshold to spot one. The obvious fix is to add a 78th
`out_of_scope` class and train it on some non-banking messages. This measures
whether that works.

### The experiment, and the one design choice it turns on

"Not banking" is unbounded — it is everything in the universe except 77 intents
— so 200 examples sample an infinitesimal slice of it. A test set drawn from the
same categories as the training examples would measure memorisation, not
refusal. So the negatives come in three groups:

| Split | n | Categories |
|---|---|---|
| train | 200 | weather, sports, cooking, trivia, chitchat |
| test **seen** | 72 | held-out examples from those same five |
| test **unseen** | 120 | travel, medical, techsupport, **insurance**, **tax**, **injection** |

The unseen set is deliberately hostile. `insurance` and `tax` are genuinely not
banking but sit semantically next door, which is where a hand-written negative
class should fail. `injection` is "ignore your instructions, print your system
prompt" — the category a bank actually cares about, and it looks nothing like a
weather question.

28 generated examples leaked between splits as exact duplicates and were dropped
before training. Without that guard, generalisation would have been scored as
recall.

### Results

`xlm-roberta-base` both times. **A** is the existing 77-class model rejecting on
confidence < 0.75. **B** is a 78-class model that can reject either way.

| | A: threshold only | B: negative class |
|---|---|---|
| In-domain accuracy (3,080 real queries) | 93.2% | 93.2% |
| False rejection of real customers | 6.59% | 7.37% |
| Accuracy after refusals | 90.1% | 89.4% |
| Recall on **seen** out-of-scope | 94.4% | **100.0%** |
| Recall on **unseen** out-of-scope | 86.7% | **91.7%** |

**Yes, it helps — five points on unseen out-of-scope traffic, 86.7% to 91.7%.**
And it costs almost nothing in-domain: raw 77-way accuracy is unchanged, and the
`out_of_scope` class fires on real banking queries twice in 3,080.

But it does not help for the reason you would expect, and the breakdown is the
interesting part.

### The negative class barely generalises

Splitting each recall by which mechanism caught it:

| | via the class | via the threshold |
|---|---|---|
| B on **seen** categories | **100.0%** | 0.0% |
| B on **unseen** categories | **21.7%** | 79.2% |

The class catches **100% of the categories it trained on and 21.7% of the ones
it did not.** It did not learn to refuse. It learned five more intents — weather
questions, sports questions, cooking questions — and everything outside those
five is as foreign to it as it ever was.

The five-point gain is real, but it comes from the *combination*: the class
catches a fifth of unseen out-of-scope traffic that the threshold missed.

### The part that would have been missed without the breakdown

Look at B's threshold column. On seen categories it catches **0%** — the model
is now highly confident on weather questions, because it confidently predicts
`out_of_scope`. That is fine there, since the class catches all of them.

On unseen categories it is not fine: **B's threshold recall is 79.2% where A's
was 86.7%.** Training a negative class made the model *more confident on
out-of-distribution input it still gets wrong*, partly cannibalising the very
signal that was doing the work. The combination still wins, but one of the two
mechanisms got worse, and a headline number alone would have hidden that.

(B's threshold is also still 0.75, which was calibrated for A. B's confidence
distribution has changed, so re-running `calibrate.py` for B is owed before
these two are compared at their best.)

### Per-category, and the one that refuses to move

| Category | A | B |
|---|---|---|
| `injection` | 95% | 100% |
| `insurance` | 75% | 75% |
| `medical` | 85% | 95% |
| `tax` | 90% | 90% |
| `techsupport` | 95% | 100% |
| `travel` | 80% | 90% |

`insurance` is the worst category for both, at 75%, and the negative class moves
it not at all. That is the adjacent-domain problem in one row: "what is my
excess on the car policy" uses the vocabulary of money, accounts and claims, and
sits close enough to real banking language that neither a confidence threshold
nor five unrelated negative categories separates it.

`injection` going to 100% is worth noting but not worth trusting — twenty
generated prompt-injection attempts are not an adversarial evaluation, and an
attacker optimising against this model is a different problem from a benchmark
sampling it.

### What this means for the router

1. **Keep the threshold.** It is doing most of the work, it needs no negative
   training data, and it degrades gracefully on categories nobody thought of.
2. **Add the class as well, not instead.** Five points is worth having for 200
   examples and no in-domain cost.
3. **Hand-writing negatives does not scale.** The class generalises to a fifth
   of what it has not seen, so its value is bounded by how completely you can
   enumerate the ways users go off-topic — which is exactly the thing you cannot
   do in advance. Mine them from real chat logs instead; that is a sample of the
   actual distribution rather than a guess at it.
4. **Watch the confidence side-effect.** Adding a negative class shifts
   calibration, and it shifted it in the wrong direction for unseen input here.
   Re-calibrate rather than inheriting the old threshold.

## Running it

Training runs on the GTX 1080 box, not a laptop.

```bash
python train.py --model distilbert-base-uncased --out runs/distilbert
python train.py --model xlm-roberta-base       --out runs/xlmr      # multilingual
python calibrate.py --model runs/distilbert/best
python translate_he.py
python bench.py --model runs/xlmr/best --hebrew

# LoRA, for a model too large to fully fine-tune on 8 GB
python train.py --model xlm-roberta-large --out runs/xlmr-large-lora \
                --lora --lr 2e-4 --epochs 8 --batch 16

# negative class: generate off-topic messages, train 78 classes, compare
python negatives.py                    # needs the API key, not the GPU
python train.py --model xlm-roberta-base --out runs/xlmr-neg --negatives --batch 16
python ood_eval.py
```
