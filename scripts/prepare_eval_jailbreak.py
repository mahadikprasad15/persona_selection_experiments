#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from persona_exp.utils import write_json, write_jsonl


TARGETS = {
    "vanilla_benign": 25,
    "vanilla_harmful": 25,
    "adversarial_benign": 25,
    "adversarial_harmful": 25,
}


def read_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def deterministic_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    if len(rows) < n:
        raise ValueError(f"Need {n} rows, only found {len(rows)}")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), n))
    return [rows[i] for i in indices]


def prompt_text_for(row: dict) -> str:
    data_type = row["data_type"]
    if data_type.startswith("adversarial_"):
        return row["adversarial"].strip()
    return row["vanilla"].strip()


def normalize(rows: list[dict], seed: int) -> tuple[list[dict], dict]:
    out = []
    report = {}
    for data_type, n in TARGETS.items():
        subset = [row for row in rows if row["data_type"] == data_type and prompt_text_for(row)]
        selected = deterministic_sample(subset, n, seed)
        report[data_type] = {"available": len(subset), "selected": len(selected)}
        for idx, row in enumerate(selected):
            out.append(
                {
                    "prompt_id": f"jailbreak_wildjailbreak_{data_type}_{idx:03d}",
                    "category": "jailbreak",
                    "subcategory": data_type,
                    "text": prompt_text_for(row),
                    "source": "wildjailbreak",
                    "source_id": f"{data_type}:{idx}",
                    "metadata": {
                        "data_type": data_type,
                        "has_vanilla": bool(row.get("vanilla", "").strip()),
                        "has_adversarial": bool(row.get("adversarial", "").strip()),
                    },
                }
            )
    return out, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        default="data/eval_prompts/raw/wildjailbreak/train.tsv",
        help="Path to WildJailbreak train.tsv. This raw file is large and should not be committed.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/eval_prompts/jailbreak_100.jsonl")
    parser.add_argument("--staged", default="data/eval_prompts/staged/jailbreak_wildjailbreak_100.jsonl")
    parser.add_argument("--report", default="data/eval_prompts/reports/jailbreak_100.report.json")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    rows = read_tsv(raw_path)
    data_type_counts = Counter(row["data_type"] for row in rows)
    normalized, composition = normalize(rows, args.seed)
    if len(normalized) != 100:
        raise ValueError(f"Expected 100 jailbreak prompts, got {len(normalized)}")
    write_jsonl(Path(args.staged), normalized)
    write_jsonl(Path(args.out), normalized)
    write_json(
        Path(args.report),
        {
            "category": "jailbreak",
            "num_prompts": len(normalized),
            "source": "wildjailbreak",
            "seed": args.seed,
            "raw_path": str(raw_path),
            "raw_rows": len(rows),
            "raw_data_type_counts": dict(sorted(data_type_counts.items())),
            "composition": composition,
            "note": "Uses vanilla prompt for vanilla_* rows and adversarial prompt for adversarial_* rows. Completion text is not passed to the model.",
        },
    )
    print(f"wrote {len(normalized)} jailbreak prompts to {args.out}")


if __name__ == "__main__":
    main()

