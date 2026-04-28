# Persona Selection Experiments

Config-driven pipeline for testing whether post-training shifts a model's default Assistant state within a role/persona vector space.

The v001 pilot builds residualized role vectors from role-conditioned generations, validates role recovery, checks cross-model alignment, then scores evaluation-prompt activations against the persona dictionary.

## Current Status

This repository is scaffolded for the full SmolLM2 pilot. Role, question, and role-instruction inputs live under `data/`; evaluation JSONL files are expected to be supplied in a later pass. Tiny sample inputs under `data/sample/` are included only for smoke tests.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For model generation/extraction, install PyTorch appropriate for your machine if it is not already installed.

## Prepare Role Instructions

```bash
PYTHONPATH=src python scripts/prepare_role_inputs.py
```

This fetches two positive Assistant Axis instructions per retained role and writes `data/templates/aa_role_instructions_2.jsonl`.

## Prepare Evaluation Banks

Neutral prompts:

```bash
PYTHONPATH=src python scripts/prepare_eval_neutral.py
PYTHONPATH=src python scripts/validate_eval_prompts.py data/eval_prompts/neutral_200.jsonl
```

Sycophancy prompts:

```bash
PYTHONPATH=src python scripts/prepare_eval_sycophancy.py
PYTHONPATH=src python scripts/validate_eval_prompts.py data/eval_prompts/sycophancy_100.jsonl
```

Vulnerable-user prompts:

```bash
PYTHONPATH=src python scripts/prepare_eval_vulnerable_user.py
PYTHONPATH=src python scripts/validate_eval_prompts.py data/eval_prompts/vulnerable_user_100.jsonl
```

## Validate

Real pilot config, after datasets are added:

```bash
python scripts/00_validate_config.py --config configs/default_smollm2_pilot.yaml
```

Smoke config:

```bash
python scripts/00_validate_config.py --config configs/smoke_test.yaml --skip-model-load
```

## Pipeline

Each stage accepts `--config`, optional `--model`, `--resume`, and `--overwrite`.

```bash
python scripts/01_generate_role_rollouts.py --config configs/default_smollm2_pilot.yaml --model base --resume
python scripts/02_extract_role_activations.py --config configs/default_smollm2_pilot.yaml --model base --resume
python scripts/03_build_role_vectors.py --config configs/default_smollm2_pilot.yaml --model base
python scripts/04_check_persona_alignment.py --config configs/default_smollm2_pilot.yaml
python scripts/05_generate_eval_rollouts.py --config configs/default_smollm2_pilot.yaml --model base --resume
python scripts/06_extract_eval_activations.py --config configs/default_smollm2_pilot.yaml --model base --resume
python scripts/07_score_eval_against_roles.py --config configs/default_smollm2_pilot.yaml --model base
python scripts/08_aggregate_scores.py --config configs/default_smollm2_pilot.yaml
python scripts/09_make_figures.py --config configs/default_smollm2_pilot.yaml
python scripts/10_export_hf_dataset.py --config configs/default_smollm2_pilot.yaml
```

## Data

Real inputs are not checked in yet. Expected schemas are documented in `data/README.md`.

Outputs live under the configured `run.output_dir`, defaulting to `runs/smollm2_pilot_v001`.
