#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from persona_exp.utils import read_jsonl


REQUIRED = {"prompt_id", "category", "subcategory", "text", "source", "source_id"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    seen = set()
    total = 0
    for raw_path in args.paths:
        path = Path(raw_path)
        rows = read_jsonl(path)
        for idx, row in enumerate(rows, start=1):
            missing = REQUIRED - set(row)
            if missing:
                raise ValueError(f"{path}:{idx} missing required keys {sorted(missing)}")
            if row["prompt_id"] in seen:
                raise ValueError(f"duplicate prompt_id {row['prompt_id']}")
            if not str(row["text"]).strip():
                raise ValueError(f"{path}:{idx} has empty text")
            seen.add(row["prompt_id"])
        total += len(rows)
        print(f"{path}: {len(rows)} rows")
    print(f"validated {total} eval prompts across {len(args.paths)} files")


if __name__ == "__main__":
    main()

