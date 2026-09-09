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


def english():
    """train / test splits as HuggingFace datasets, with `label` and `text`."""
    ds = load_dataset("PolyAI/banking77")
    return ds["train"], ds["test"]


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
