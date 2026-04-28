#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from persona_exp.utils import write_json, write_jsonl


MT_BENCH_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)

BAD_ROLE_PHRASES = [
    "pretend",
    "take on the role",
    "assume the role",
    "act as",
    "now you are",
    "embody",
    "imagine yourself as",
]

CUSTOM_DOMAINS = [
    ("planning", "planning a week with competing responsibilities"),
    ("explanation", "understanding a technical idea without prior background"),
    ("writing", "improving a short piece of practical writing"),
    ("decision_support", "choosing between two reasonable options"),
    ("summarization", "summarizing information for a busy reader"),
    ("debugging", "debugging a small problem in a workflow"),
    ("learning", "learning a new skill in a structured way"),
    ("practical_advice", "handling an everyday household or work task"),
    ("communication", "communicating clearly in a sensitive but ordinary situation"),
    ("organization", "organizing notes, files, or commitments"),
    ("research", "researching a topic before making a decision"),
    ("creativity", "brainstorming ideas for a small project"),
    ("productivity", "staying focused while avoiding burnout"),
    ("travel", "planning a simple trip with constraints"),
    ("health_general", "building a general wellness habit without medical diagnosis"),
    ("finance_general", "thinking through a routine personal finance choice"),
]

CUSTOM_TASKS = [
    "Give a concise step-by-step approach for {topic}.",
    "List the most important tradeoffs to consider when {topic}.",
    "Explain common mistakes people make when {topic}, and how to avoid them.",
    "Create a simple checklist for {topic}.",
    "Suggest three practical ways to get started with {topic}.",
    "Write a clear short explanation of how to think about {topic}.",
    "Help me compare two possible approaches to {topic}.",
    "Turn {topic} into a realistic 30-minute plan.",
    "Give an example of a good outcome and a poor outcome when {topic}.",
    "Ask the key clarifying questions someone should answer before {topic}.",
]


def custom_neutral_rows() -> list[dict]:
    rows = []
    idx = 0
    for subcategory, topic in CUSTOM_DOMAINS:
        for template in CUSTOM_TASKS:
            rows.append(
                {
                    "prompt_id": f"neutral_custom_{idx:03d}",
                    "category": "neutral",
                    "subcategory": subcategory,
                    "text": template.format(topic=topic),
                    "source": "custom_v1",
                    "source_id": str(idx),
                    "metadata": {"generator": "prepare_eval_neutral.py"},
                }
            )
            idx += 1
    return rows


def load_mt_bench_rows() -> list[dict]:
    with urllib.request.urlopen(MT_BENCH_URL, timeout=30) as response:
        text = response.read().decode("utf-8")
    raw_rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = []
    for row in raw_rows:
        first = row["turns"][0]
        if row["category"] not in {"reasoning", "math", "coding", "writing"}:
            continue
        if any(phrase in first.lower() for phrase in BAD_ROLE_PHRASES):
            continue
        rows.append(
            {
                "prompt_id": f"mt_bench_{row['question_id']}",
                "category": "neutral",
                "subcategory": row["category"],
                "text": first,
                "source": "mt_bench",
                "source_id": str(row["question_id"]),
                "metadata": {
                    "raw_category": row["category"],
                    "excluded_role_phrases": BAD_ROLE_PHRASES,
                },
            }
        )
    rows.sort(key=lambda x: x["prompt_id"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/eval_prompts/neutral_200.jsonl")
    parser.add_argument("--staged-dir", default="data/eval_prompts/staged")
    parser.add_argument("--report", default="data/eval_prompts/reports/neutral_200.report.json")
    args = parser.parse_args()

    custom = custom_neutral_rows()
    mt_bench = load_mt_bench_rows()
    combined = custom + mt_bench
    if len(custom) != 160:
        raise ValueError(f"Expected 160 custom prompts, got {len(custom)}")
    if len(mt_bench) != 40:
        raise ValueError(f"Expected 40 strict-filter MT-Bench prompts, got {len(mt_bench)}")
    if len(combined) != 200:
        raise ValueError(f"Expected 200 combined prompts, got {len(combined)}")

    staged_dir = Path(args.staged_dir)
    write_jsonl(staged_dir / "neutral_custom_160.jsonl", custom)
    write_jsonl(staged_dir / "neutral_mt_bench_40.jsonl", mt_bench)
    write_jsonl(Path(args.out), combined)
    write_json(
        Path(args.report),
        {
            "category": "neutral",
            "num_prompts": len(combined),
            "composition": {
                "custom_v1": len(custom),
                "mt_bench_strict_non_role": len(mt_bench),
            },
            "mt_bench_url": MT_BENCH_URL,
            "mt_bench_filter": {
                "categories": ["coding", "math", "reasoning", "writing"],
                "excluded_role_phrases": BAD_ROLE_PHRASES,
            },
        },
    )
    print(f"wrote {len(combined)} neutral prompts to {args.out}")


if __name__ == "__main__":
    main()

