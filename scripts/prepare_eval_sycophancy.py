#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from persona_exp.utils import write_json, write_jsonl


BASE_URL = "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets"
SOURCES = [
    ("feedback", 60),
    ("answer", 20),
    ("are_you_sure", 20),
]


def load_jsonl(name: str) -> list[dict]:
    url = f"{BASE_URL}/{name}.jsonl"
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def extract_human_text(row: dict) -> str:
    messages = row["prompt"]
    return "\n".join(m["content"] for m in messages if m.get("type") == "human").strip()


def normalize_rows(source_name: str, rows: list[dict], n: int) -> list[dict]:
    out = []
    for idx, row in enumerate(rows[:n]):
        out.append(
            {
                "prompt_id": f"sycophancy_{source_name}_{idx:03d}",
                "category": "sycophancy",
                "subcategory": source_name,
                "text": extract_human_text(row),
                "source": "meg_tong_sycophancy_eval",
                "source_id": f"{source_name}:{idx}",
                "metadata": {
                    "source_dataset": source_name,
                    "base": row.get("base", {}),
                    "metadata": row.get("metadata", {}),
                },
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/eval_prompts/sycophancy_100.jsonl")
    parser.add_argument("--staged-dir", default="data/eval_prompts/staged")
    parser.add_argument("--report", default="data/eval_prompts/reports/sycophancy_100.report.json")
    args = parser.parse_args()

    staged_dir = Path(args.staged_dir)
    combined = []
    source_counts = {}
    for source_name, n in SOURCES:
        raw = load_jsonl(source_name)
        rows = normalize_rows(source_name, raw, n)
        if len(rows) != n:
            raise ValueError(f"Expected {n} rows for {source_name}, got {len(rows)}")
        write_jsonl(staged_dir / f"sycophancy_{source_name}_{n}.jsonl", rows)
        combined.extend(rows)
        source_counts[source_name] = {"selected": len(rows), "available": len(raw)}

    if len(combined) != 100:
        raise ValueError(f"Expected 100 sycophancy prompts, got {len(combined)}")

    write_jsonl(Path(args.out), combined)
    write_json(
        Path(args.report),
        {
            "category": "sycophancy",
            "num_prompts": len(combined),
            "source": "meg_tong_sycophancy_eval",
            "base_url": BASE_URL,
            "composition": source_counts,
        },
    )
    print(f"wrote {len(combined)} sycophancy prompts to {args.out}")


if __name__ == "__main__":
    main()

