#!/usr/bin/env python
from __future__ import annotations

import argparse

from persona_exp.config import get_model, load_config, load_personas, load_questions, load_role_instructions, run_dir
from persona_exp.formatting import format_role_prompt
from persona_exp.generation import generate_responses
from persona_exp.io import write_jsonl_shards, write_status
from persona_exp.model_loader import load_model_and_tokenizer
from persona_exp.utils import add_common_args, set_seed


def build_records(cfg, model_name: str):
    dialogue_template = cfg["formatting"]["template"]
    records = []
    instructions = load_role_instructions(cfg)
    for role in load_personas(cfg):
        for question in load_questions(cfg):
            for template in instructions:
                if template.get("role_id", role["role_id"]) != role["role_id"]:
                    continue
                template_id = template.get("instruction_id", template.get("template_id"))
                example_id = f"role={role['role_id']}__q={question['question_id']}__instruction={template_id}"
                records.append(
                    {
                        "example_id": example_id,
                        "model_name": model_name,
                        "role_id": role["role_id"],
                        "role_name": role["role_name"],
                        "cluster": role["cluster"],
                        "question_id": question["question_id"],
                        "instruction_id": template_id,
                        "instruction_source": template.get("source", "unknown"),
                        "prompt_text": format_role_prompt(role, question, template, dialogue_template),
                    }
                )
    return records


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.model:
        raise ValueError("--model is required")
    set_seed(int(cfg["run"]["seed"]))
    model_cfg = get_model(cfg, args.model)
    out_dir = run_dir(cfg) / "role_rollouts" / f"model={model_cfg.name}"
    if (out_dir / "_SUCCESS").exists() and not args.overwrite:
        print(f"role rollouts already complete: {out_dir}")
        return
    write_status(run_dir(cfg), "running", "role_rollouts", {"model": model_cfg.name})
    model, tokenizer, device = load_model_and_tokenizer(model_cfg)
    records = build_records(cfg, model_cfg.name)
    generated = generate_responses(model, tokenizer, records, cfg["generation"]["role"], device)
    shard_size = int(cfg["storage"]["shard_size"]["role_rollouts"])
    write_jsonl_shards(out_dir, generated, shard_size, overwrite=args.overwrite)
    write_status(run_dir(cfg), "completed", "role_rollouts", {"model": model_cfg.name, "num_records": len(generated)})


if __name__ == "__main__":
    main()
