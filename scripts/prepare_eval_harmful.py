#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import io
import random
import urllib.request
from pathlib import Path

from persona_exp.utils import write_json, write_jsonl


XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
STRONGREJECT_URL = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
    "strongreject_dataset/strongreject_dataset.csv"
)


def load_csv(url: str) -> list[dict]:
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def deterministic_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    if len(rows) < n:
        raise ValueError(f"Need {n} rows, only found {len(rows)}")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), n))
    return [rows[i] for i in indices]


def normalize_xstest(rows: list[dict], n: int, seed: int) -> list[dict]:
    unsafe = [row for row in rows if row["label"] == "unsafe"]
    selected = deterministic_sample(unsafe, n, seed)
    return [
        {
            "prompt_id": f"harmful_xstest_{idx:03d}",
            "category": "harmful",
            "subcategory": "xstest_unsafe",
            "text": row["prompt"],
            "source": "xstest",
            "source_id": row["id"],
            "metadata": {
                "type": row.get("type", ""),
                "label": row.get("label", ""),
                "focus": row.get("focus", ""),
                "note": row.get("note", ""),
            },
        }
        for idx, row in enumerate(selected)
    ]


def normalize_harmbench(rows: list[dict], n: int, seed: int) -> list[dict]:
    selected = deterministic_sample(rows, n, seed)
    return [
        {
            "prompt_id": f"harmful_harmbench_{idx:03d}",
            "category": "harmful",
            "subcategory": "harmbench",
            "text": row["Behavior"],
            "source": "harmbench",
            "source_id": row["BehaviorID"],
            "metadata": {
                "functional_category": row.get("FunctionalCategory", ""),
                "semantic_category": row.get("SemanticCategory", ""),
                "tags": row.get("Tags", ""),
                "context_string": row.get("ContextString", ""),
            },
        }
        for idx, row in enumerate(selected)
    ]


def normalize_strongreject(rows: list[dict], n: int, seed: int) -> list[dict]:
    selected = deterministic_sample(rows, n, seed)
    return [
        {
            "prompt_id": f"harmful_strongreject_{idx:03d}",
            "category": "harmful",
            "subcategory": "strongreject",
            "text": row["forbidden_prompt"],
            "source": "strongreject",
            "source_id": f"{row.get('source', '')}:{idx}",
            "metadata": {
                "raw_category": row.get("category", ""),
                "raw_source": row.get("source", ""),
            },
        }
        for idx, row in enumerate(selected)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/eval_prompts/harmful_100.jsonl")
    parser.add_argument("--staged-dir", default="data/eval_prompts/staged")
    parser.add_argument("--report", default="data/eval_prompts/reports/harmful_100.report.json")
    args = parser.parse_args()

    xstest_raw = load_csv(XSTEST_URL)
    harmbench_raw = load_csv(HARMBENCH_URL)
    strongreject_raw = load_csv(STRONGREJECT_URL)

    xstest = normalize_xstest(xstest_raw, 50, args.seed)
    harmbench = normalize_harmbench(harmbench_raw, 25, args.seed)
    strongreject = normalize_strongreject(strongreject_raw, 25, args.seed)
    combined = xstest + harmbench + strongreject
    if len(combined) != 100:
        raise ValueError(f"Expected 100 harmful prompts, got {len(combined)}")

    staged_dir = Path(args.staged_dir)
    write_jsonl(staged_dir / "harmful_xstest_50.jsonl", xstest)
    write_jsonl(staged_dir / "harmful_harmbench_25.jsonl", harmbench)
    write_jsonl(staged_dir / "harmful_strongreject_25.jsonl", strongreject)
    write_jsonl(Path(args.out), combined)
    write_json(
        Path(args.report),
        {
            "category": "harmful",
            "num_prompts": len(combined),
            "seed": args.seed,
            "composition": {
                "xstest_unsafe": {"selected": len(xstest), "available": sum(1 for r in xstest_raw if r["label"] == "unsafe"), "url": XSTEST_URL},
                "harmbench": {"selected": len(harmbench), "available": len(harmbench_raw), "url": HARMBENCH_URL},
                "strongreject": {"selected": len(strongreject), "available": len(strongreject_raw), "url": STRONGREJECT_URL},
            },
            "note": "Direct harmful/unsafe request pressure; jailbreak wrappers are handled separately.",
        },
    )
    print(f"wrote {len(combined)} harmful prompts to {args.out}")


if __name__ == "__main__":
    main()

