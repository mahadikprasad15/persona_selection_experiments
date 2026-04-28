#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import random
from pathlib import Path
from typing import Any

import pandas as pd

from persona_exp.utils import write_json, write_jsonl


BASE_URL = "https://huggingface.co/datasets/AISC-Linear-Probe-Gen/deception_taxonomy_paper/resolve/main"
SOURCES = {
    "instructed_deception": "instructed-deception/test-00000-of-00001.parquet",
    "convincing_game": "convincing-game/test-00000-of-00001.parquet",
}


def parse_meta(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def messages_to_prompt_text(messages: Any) -> str:
    parts = []
    for message in list(messages):
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "assistant":
            break
        if role == "system":
            parts.append(f"System instruction:\n{content}")
        elif role == "user":
            parts.append(f"User message:\n{content}")
        else:
            parts.append(f"{role.title()} message:\n{content}")
    text = "\n\n".join(parts).strip()
    if not text:
        raise ValueError("No non-assistant prompt text found")
    return text


def deterministic_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) < n:
        raise ValueError(f"Need {n} rows, only found {len(df)}")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(df)), n))
    return df.iloc[indices].reset_index(drop=True)


def normalize_instructed(df: pd.DataFrame, n: int, seed: int) -> tuple[list[dict], dict]:
    parsed_meta = df["meta"].map(parse_meta)
    mask = parsed_meta.map(lambda m: bool(m.get("instructed_deceptiveness")))
    filtered = df[mask].copy()
    filtered["_parsed_meta"] = parsed_meta[mask]
    selected = deterministic_sample(filtered, n, seed)
    rows = []
    for idx, row in selected.iterrows():
        meta = row["_parsed_meta"]
        rows.append(
            {
                "prompt_id": f"deception_instructed_deception_{idx:03d}",
                "category": "deception",
                "subcategory": "instructed_deception",
                "text": messages_to_prompt_text(row["messages"]),
                "source": "aisc_deception_taxonomy_paper",
                "source_id": f"instructed_deception:{int(row['index'])}",
                "metadata": {
                    "raw_dataset": row.get("dataset", ""),
                    "raw_dataset_index": int(row.get("dataset_index", -1)),
                    "raw_model": row.get("model", ""),
                    "raw_deceptive_label": bool(row.get("deceptive", False)),
                    "meta": meta,
                },
            }
        )
    return rows, {"available": int(mask.sum()), "selected": len(rows)}


def normalize_convincing(df: pd.DataFrame, n: int, seed: int) -> tuple[list[dict], dict]:
    parsed_meta = df["meta"].map(parse_meta)
    mask = parsed_meta.map(
        lambda m: bool(m.get("template_system_instructs_convincing"))
        or bool(m.get("template_system_instructs_lying"))
    )
    filtered = df[mask].copy()
    filtered["_parsed_meta"] = parsed_meta[mask]
    selected = deterministic_sample(filtered, n, seed)
    rows = []
    for idx, row in selected.iterrows():
        meta = row["_parsed_meta"]
        rows.append(
            {
                "prompt_id": f"deception_convincing_game_{idx:03d}",
                "category": "deception",
                "subcategory": "convincing_game",
                "text": messages_to_prompt_text(row["messages"]),
                "source": "aisc_deception_taxonomy_paper",
                "source_id": f"convincing_game:{int(row['index'])}",
                "metadata": {
                    "raw_model": row.get("model", ""),
                    "raw_deceptive_label": bool(row.get("deceptive", False)),
                    "meta": meta,
                },
            }
        )
    return rows, {"available": int(mask.sum()), "selected": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/eval_prompts/deception_100.jsonl")
    parser.add_argument("--staged-dir", default="data/eval_prompts/staged")
    parser.add_argument("--report", default="data/eval_prompts/reports/deception_100.report.json")
    args = parser.parse_args()

    instructed_path = SOURCES["instructed_deception"]
    convincing_path = SOURCES["convincing_game"]
    instructed_df = pd.read_parquet(f"{BASE_URL}/{instructed_path}")
    convincing_df = pd.read_parquet(f"{BASE_URL}/{convincing_path}")

    instructed, instructed_report = normalize_instructed(instructed_df, 50, args.seed)
    convincing, convincing_report = normalize_convincing(convincing_df, 50, args.seed)
    combined = instructed + convincing
    if len(combined) != 100:
        raise ValueError(f"Expected 100 deception prompts, got {len(combined)}")

    staged_dir = Path(args.staged_dir)
    write_jsonl(staged_dir / "deception_instructed_deception_50.jsonl", instructed)
    write_jsonl(staged_dir / "deception_convincing_game_50.jsonl", convincing)
    write_jsonl(Path(args.out), combined)
    write_json(
        Path(args.report),
        {
            "category": "deception",
            "num_prompts": len(combined),
            "source": "aisc_deception_taxonomy_paper",
            "seed": args.seed,
            "base_url": BASE_URL,
            "composition": {
                "instructed_deception": {**instructed_report, "path": instructed_path},
                "convincing_game": {**convincing_report, "path": convincing_path},
            },
            "prompt_extraction": "Use non-assistant messages before the first saved assistant completion. Assistant completions and canary fields are not passed to the model.",
        },
    )
    print(f"wrote {len(combined)} deception prompts to {args.out}")


if __name__ == "__main__":
    main()

