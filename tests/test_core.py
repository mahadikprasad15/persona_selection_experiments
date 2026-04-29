from __future__ import annotations

import numpy as np

from persona_exp.config import load_config, validate_data_files
from persona_exp.formatting import format_eval_prompt, format_role_prompt
from persona_exp.generation import trim_generated_ids
from persona_exp.scoring import score_dot, softmax_scores
from persona_exp.vector_building import build_all_vector_types


def test_smoke_config_data_validates():
    cfg = load_config("configs/smoke_test.yaml")
    counts = validate_data_files(cfg)
    assert counts == {"personas": 3, "questions": 3, "templates": 2, "eval_prompts": 4}


def test_default_role_inputs_validate_without_eval_bank():
    cfg = load_config("configs/default_smollm2_pilot.yaml")
    counts = validate_data_files(cfg, require_eval=False)
    assert counts["personas"] == 42
    assert counts["questions"] == 64
    assert counts["templates"] == 84


def test_prompt_formatting():
    template = "User: {user_text}\nAssistant:\n"
    role = {"role_id": "auditor", "role_name": "Auditor", "cluster": "epistemic_integrity"}
    question = {"question_id": "q001", "text": "Explain a hard idea."}
    role_template = {
        "role_id": "auditor",
        "instruction_id": "aa_pos_0",
        "source": "assistant_axis",
        "template": "You are an auditor.",
    }
    assert format_role_prompt(role, question, role_template, template) == (
        "User: You are an auditor. Explain a hard idea.\nAssistant:"
    )
    assert format_eval_prompt({"text": "Help me."}, template) == "User: Help me.\nAssistant:"


def test_vector_building_shapes():
    import pandas as pd

    rows = []
    vectors = []
    for role_id in ["a", "b"]:
        for q in ["q1", "q2"]:
            rows.append(
                {
                    "role_id": role_id,
                    "role_name": role_id.upper(),
                    "cluster": "c",
                    "question_id": q,
                    "template_id": "t1",
                }
            )
            vectors.append([1.0, 0.0] if role_id == "a" else [0.0, 1.0])
    built = build_all_vector_types(pd.DataFrame(rows), np.array(vectors, dtype=np.float32))
    assert set(built) == {"raw", "question_residualized", "question_residualized_centered"}
    assert built["raw"][1].shape == (2, 2)


def test_scores_and_softmax():
    h = np.array([[1.0, 0.0]], dtype=np.float32)
    v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    scores = score_dot(h, v)
    probs = softmax_scores(scores)
    assert scores.tolist() == [[1.0, 0.0]]
    assert np.isclose(probs.sum(), 1.0)


def test_score_schema_metadata_passthrough_keys():
    # Keep this explicit so downstream aggregation can rely on these columns existing.
    expected = {
        "prompt_subcategory",
        "prompt_source",
        "prompt_source_id",
        "prompt_source_metadata_json",
    }
    row = {
        "prompt_subcategory": "coding",
        "prompt_source": "mt_bench",
        "prompt_source_id": "101",
        "prompt_source_metadata_json": "{}",
    }
    assert expected <= set(row)


def test_trim_generated_ids_removes_eos_and_padding_tail():
    import torch

    class Tok:
        eos_token_id = 2
        pad_token_id = 0

    assert trim_generated_ids(Tok(), torch.tensor([10, 11, 2, 0, 0])).tolist() == [10, 11]
    assert trim_generated_ids(Tok(), torch.tensor([10, 11, 12])).tolist() == [10, 11, 12]
