from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from persona_exp.utils import read_json, read_jsonl, write_json


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hf_id: str
    stage: str
    dtype: str = "float16"
    device: str = "auto"


@dataclass(frozen=True)
class ResolvedLayer:
    layer_tag: str
    layer_idx: int
    num_layers: int
    layer_frac: float


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def repo_root_from_config(cfg: dict[str, Any]) -> Path:
    return Path(cfg["_config_path"]).resolve().parent.parent


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root_from_config(cfg) / path


def get_models(cfg: dict[str, Any]) -> list[ModelConfig]:
    return [ModelConfig(**m) for m in cfg["models"]]


def get_model(cfg: dict[str, Any], name: str) -> ModelConfig:
    for model in get_models(cfg):
        if model.name == name:
            return model
    raise KeyError(f"Unknown model {name!r}. Available: {[m.name for m in get_models(cfg)]}")


def run_dir(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg, cfg["run"]["output_dir"])


def load_personas(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return read_jsonl(resolve_path(cfg, cfg["data"]["personas_path"]))


def load_questions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return read_jsonl(resolve_path(cfg, cfg["data"]["questions_path"]))


def load_role_templates(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return read_jsonl(resolve_path(cfg, cfg["data"]["role_templates_path"]))


def load_role_instructions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return load_role_templates(cfg)


def load_role_clusters(cfg: dict[str, Any]) -> dict[str, str]:
    return read_json(resolve_path(cfg, cfg["data"]["role_clusters_path"]))


def load_eval_prompts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for category, rel_path in cfg["data"]["eval_prompts"].items():
        for record in read_jsonl(resolve_path(cfg, rel_path)):
            item = dict(record)
            item.setdefault("category", category)
            out.append(item)
    return out


def validate_unique(records: list[dict[str, Any]], key: str, label: str) -> None:
    values = [r[key] for r in records]
    duplicates = sorted({v for v in values if values.count(v) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {label} {key}s: {duplicates[:10]}")


def validate_data_files(cfg: dict[str, Any], require_eval: bool = True) -> dict[str, int]:
    required = [
        cfg["data"]["personas_path"],
        cfg["data"]["role_clusters_path"],
        cfg["data"]["questions_path"],
        cfg["data"]["role_templates_path"],
    ]
    if require_eval:
        required.extend(cfg["data"]["eval_prompts"].values())
    missing = [str(resolve_path(cfg, p)) for p in required if not resolve_path(cfg, p).exists()]
    if missing:
        raise FileNotFoundError("Missing required data files:\n" + "\n".join(missing))

    personas = load_personas(cfg)
    questions = load_questions(cfg)
    templates = load_role_templates(cfg)
    eval_prompts = load_eval_prompts(cfg) if require_eval else []
    clusters = load_role_clusters(cfg)

    validate_unique(personas, "role_id", "persona")
    validate_unique(questions, "question_id", "question")
    template_key = "instruction_id" if templates and "instruction_id" in templates[0] else "template_id"
    if template_key == "template_id":
        validate_unique(templates, template_key, "template")
    else:
        instruction_keys = [(t["role_id"], t["instruction_id"]) for t in templates]
        duplicates = sorted({v for v in instruction_keys if instruction_keys.count(v) > 1})
        if duplicates:
            raise ValueError(f"Duplicate role instruction ids: {duplicates[:10]}")
    if require_eval:
        validate_unique(eval_prompts, "prompt_id", "eval prompt")

    for role in personas:
        role_id = role["role_id"]
        if role_id not in clusters:
            raise ValueError(f"role_id {role_id!r} missing from role cluster map")
        role_templates = [t for t in templates if t.get("role_id", role_id) == role_id]
        if not role_templates:
            raise ValueError(f"role_id {role_id!r} has no role instructions/templates")
        for template in role_templates:
            template["template"].format(role_name=role["role_name"])

    return {
        "personas": len(personas),
        "questions": len(questions),
        "templates": len(templates),
        "eval_prompts": len(eval_prompts),
    }


def resolve_layer_indices(layer_fracs: list[float], num_layers: int) -> list[ResolvedLayer]:
    layers = []
    for frac in layer_fracs:
        idx = int(round(frac * (num_layers - 1)))
        pct = int(round(frac * 100))
        layers.append(ResolvedLayer(f"L{pct}", idx, num_layers, frac))
    return layers


def save_resolved_config(cfg: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
    out = dict(cfg)
    out.pop("_config_path", None)
    if extra:
        out["resolved"] = extra
    path = run_dir(cfg) / "config.resolved.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    tmp.replace(path)
    return path


def write_initial_run_metadata(cfg: dict[str, Any], status: str = "initialized") -> None:
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_name": cfg["run"]["name"],
        "output_dir": str(Path(cfg["run"]["output_dir"])),
        "config_path": cfg["_config_path"],
    }
    write_json(rd / "manifest.json", manifest)
    write_json(rd / "status.json", {"status": status})
