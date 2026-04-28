#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from persona_exp.utils import read_jsonl, write_json, write_jsonl


DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/safety-research/assistant-axis/master/data/roles/instructions"
)

REMOVED_MISSING_AA_ROLES = [
    "helper",
    "clerk",
    "professor",
    "explainer",
    "fact_checker",
    "ethicist",
    "arbiter",
    "safety_reviewer",
    "risk_assessor",
    "policy_guardian",
    "salesperson",
    "publicist",
    "spin_doctor",
    "debater",
    "flatterer",
    "approval_seeker",
    "people_pleaser",
    "loyal_subordinate",
    "undercover_agent",
    "secret_keeper",
    "manipulator",
    "con_artist",
]


def fetch_role_instructions(role_id: str, base_url: str) -> list[str]:
    url = f"{base_url}/{role_id}.json"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    positives = [item["pos"] for item in payload["instruction"] if "pos" in item]
    if len(positives) < 2:
        raise ValueError(f"{role_id} has fewer than two positive instructions")
    return positives[:2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", default="data/personas/aa_roles_50.jsonl")
    parser.add_argument("--out", default="data/templates/aa_role_instructions_2.jsonl")
    parser.add_argument("--report", default="data/templates/aa_role_instructions_2.report.json")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    roles_path = Path(args.roles)
    rows = []
    fetched = []
    for role in read_jsonl(roles_path):
        instructions = fetch_role_instructions(role["role_id"], args.base_url)
        fetched.append(role["role_id"])
        for idx, instruction in enumerate(instructions):
            rows.append(
                {
                    "role_id": role["role_id"],
                    "instruction_id": f"aa_pos_{idx}",
                    "source": "assistant_axis",
                    "template": instruction,
                }
            )

    write_jsonl(Path(args.out), rows)
    write_json(
        Path(args.report),
        {
            "source": "assistant_axis",
            "base_url": args.base_url,
            "retained_roles": fetched,
            "removed_missing_aa_roles": REMOVED_MISSING_AA_ROLES,
            "num_instruction_rows": len(rows),
        },
    )
    print(f"wrote {len(rows)} instruction rows to {args.out}")


if __name__ == "__main__":
    main()

