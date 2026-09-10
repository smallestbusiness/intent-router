"""BANKING77: 13,083 real customer-service queries over 77 banking intents.

Chosen over a synthetic set for one reason: the classes are genuinely confusable
("card_not_working" vs "card_payment_not_recognised" vs "declined_card_payment").
A router that scores well here is being tested on the distinctions that actually
cost money to get wrong, not on telling a balance query from a greeting.
"""
import json
import os
from pathlib import Path

from datasets import load_dataset

CACHE = Path(os.environ.get("ROUTER_DATA", "data"))
HEBREW_TEST = CACHE / "test_he.json"


# Not PolyAI/banking77, which is the canonical home but ships a loader script;
# datasets 5.x refuses to execute those. This mirror is the same 10,003/3,080
# split as parquet, and keeps `label` as a ClassLabel so the 77 intent names
# come from the data rather than from a list copied into this file.
SOURCE = "legacy-datasets/banking77"


def english():
    """train / test splits as HuggingFace datasets, with `label` and `text`."""
    ds = load_dataset(SOURCE)
    return ds["train"], ds["test"]


OOS = "out_of_scope"
NEGATIVES = CACHE / "negatives.json"


def negatives():
    return json.loads(NEGATIVES.read_text(encoding="utf-8"))


def english_with_negatives():
    """BANKING77 plus a 78th out-of-scope class, as plain Datasets.

    The training eval set gets the *seen* negatives so that checkpoint selection
    rewards a model that can refuse. The *unseen* negatives are never touched
    during training -- they are the only honest measure of whether an explicit
    negative class generalises past the categories it was shown.
    """
    from datasets import Dataset

    train, test = english()
    names = list(test.features["label"].names) + [OOS]
    oos = len(names) - 1
    neg = negatives()

    tr = ([{"text": r["text"], "label": r["label"]} for r in train]
          + [{"text": r["text"], "label": oos} for r in neg["train"]])
    ev = ([{"text": r["text"], "label": r["label"]} for r in test]
          + [{"text": r["text"], "label": oos} for r in neg["test_seen"]])
    return Dataset.from_list(tr), Dataset.from_list(ev), names


def label_names():
    _, test = english()
    return test.features["label"].names


def hebrew_test():
    """The Hebrew evaluation slice, if it has been built.

    Not part of BANKING77 -- built by translating a stratified sample of the
    English test set (see translate_he.py). It exists because an Israeli bank's
    customers write in Hebrew, and an English-only accuracy number says nothing
    about whether the router survives contact with them. Machine-translated and
    spot-checked, which is a real limitation and is reported as one.
    """
    if not HEBREW_TEST.exists():
        raise FileNotFoundError(
            f"{HEBREW_TEST} not built yet -- run translate_he.py first")
    rows = json.loads(HEBREW_TEST.read_text(encoding="utf-8"))
    return rows
