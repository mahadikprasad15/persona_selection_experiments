from __future__ import annotations

from typing import Any


def format_dialogue(user_text: str, dialogue_template: str) -> str:
    return dialogue_template.format(user_text=user_text).rstrip()


def format_role_prompt(
    role: dict[str, Any],
    question: dict[str, Any],
    template: dict[str, Any],
    dialogue_template: str,
) -> str:
    instruction = template["template"].format(role_name=role["role_name"]).strip()
    user_text = f"{instruction} {question['text']}".strip()
    return format_dialogue(user_text, dialogue_template)


def format_eval_prompt(prompt: dict[str, Any], dialogue_template: str) -> str:
    return format_dialogue(prompt["text"], dialogue_template)


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def get_response_token_slice(tokenizer, prompt_text: str, response_text: str) -> slice:
    prompt_len = _token_len(tokenizer, prompt_text)
    full_len = _token_len(tokenizer, prompt_text + response_text)
    if full_len <= prompt_len:
        full_len = _token_len(tokenizer, prompt_text + " " + response_text)
    return slice(prompt_len, full_len)


def find_assistant_marker_token_index(tokenizer, prompt_text: str) -> int:
    ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("Prompt tokenized to zero tokens")
    return len(ids) - 1


def find_pre_assistant_final_index(tokenizer, prompt_text: str) -> int:
    marker = "Assistant:"
    pos = prompt_text.rfind(marker)
    if pos < 0:
        raise ValueError("Prompt does not contain Assistant: marker")
    prefix = prompt_text[:pos].rstrip()
    ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("Pre-assistant prompt tokenized to zero tokens")
    return len(ids) - 1

