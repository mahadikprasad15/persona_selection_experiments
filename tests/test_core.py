from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from persona_exp.activation_extract import extract_hidden_states_teacher_forced
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
    assert trim_generated_ids(Tok(), torch.tensor([2, 0, 0])).tolist() == [2]
    assert trim_generated_ids(Tok(), torch.tensor([10, 11, 12])).tolist() == [10, 11, 12]


def test_teacher_forced_extraction_uses_saved_response_token_ids_for_empty_decoded_text():
    import torch

    class Batch(dict):
        def to(self, device):
            return self

    class Tok:
        def __call__(self, text, return_tensors=None):
            ids = [ord(ch) for ch in text]
            if return_tensors == "pt":
                return Batch(
                    {
                        "input_ids": torch.tensor([ids], dtype=torch.long),
                        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
                    }
                )
            return {"input_ids": ids}

    class Model:
        def __call__(self, input_ids, attention_mask, output_hidden_states, use_cache):
            seq_len = input_ids.shape[1]
            hidden = torch.arange(seq_len * 2, dtype=torch.float32).reshape(1, seq_len, 2)
            return SimpleNamespace(hidden_states=[hidden, hidden])

    prompt = "User: test\nAssistant:"
    response_token_ids = [999]
    tensors, meta = extract_hidden_states_teacher_forced(
        Model(),
        Tok(),
        prompt,
        "",
        [SimpleNamespace(layer_idx=0, layer_tag="L0")],
        ["role_response_mean"],
        "cpu",
        "float32",
        response_token_ids,
    )

    assert meta["response_start"] == len(prompt)
    assert meta["response_stop"] == len(prompt) + 1
    assert tensors["L0/role_response_mean"].tolist() == [
        float(len(prompt) * 2),
        float(len(prompt) * 2 + 1),
    ]


def test_teacher_forced_extraction_adds_eos_for_empty_saved_response_token_ids():
    import torch

    class Batch(dict):
        def to(self, device):
            return self

    class Tok:
        eos_token_id = 2
        pad_token_id = 0

        def __call__(self, text, return_tensors=None):
            ids = [ord(ch) for ch in text]
            if return_tensors == "pt":
                return Batch(
                    {
                        "input_ids": torch.tensor([ids], dtype=torch.long),
                        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
                    }
                )
            return {"input_ids": ids}

    class Model:
        def __call__(self, input_ids, attention_mask, output_hidden_states, use_cache):
            seq_len = input_ids.shape[1]
            hidden = torch.arange(seq_len * 2, dtype=torch.float32).reshape(1, seq_len, 2)
            return SimpleNamespace(hidden_states=[hidden, hidden])

    tensors, meta = extract_hidden_states_teacher_forced(
        Model(),
        Tok(),
        "User: test\nAssistant:",
        "",
        [SimpleNamespace(layer_idx=0, layer_tag="L0")],
        ["role_response_mean"],
        "cpu",
        "float32",
        [],
    )

    assert meta["response_stop"] == meta["response_start"] + 1
    assert meta["used_synthetic_eos_for_empty_response"] is True
    assert "L0/role_response_mean" in tensors
