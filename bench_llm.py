"""The LLM half of the benchmark, run where the API key is.

Reads the exact examples bench.py scored on the GPU box, so the two halves are
compared on identical inputs rather than on two samples that happen to be the
same size.
"""
import argparse
import json
from pathlib import Path

from bench import llm_eval

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="data/llm_eval_set.json")
    ap.add_argument("--out", default="llm_results.json")
    # Latency is measured one request at a time because that is how a router is
    # called, so this is deliberately sequential and therefore slow. 150 per
    # language gives roughly +/-4 points of sampling error on accuracy, which is
    # far narrower than the gap being measured.
    ap.add_argument("--limit", type=int, default=150)
    a = ap.parse_args()

    payload = json.loads(Path(a.set).read_text(encoding="utf-8"))
    names = payload["names"]
    out = {}
    for lang in ("en", "he"):
        if lang not in payload:
            continue
        print(f"--- {lang} ---", flush=True)
        out[f"llm_{lang}"] = llm_eval(payload[lang]["texts"], payload[lang]["labels"],
                                      names, a.limit)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2))
