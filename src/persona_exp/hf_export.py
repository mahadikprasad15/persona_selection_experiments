from __future__ import annotations

from pathlib import Path
from typing import Iterable

from persona_exp.utils import write_json


DEFAULT_EXCLUDE_PATTERNS = [
    "*.tmp",
    "**/*.tmp",
    "**/__pycache__/**",
    "**/.DS_Store",
    "**/.cache/**",
    "**/raw/**",
    "**/full_hidden_state_dumps/**",
    "**/unpooled_token_activations/**",
]

CORE_RUN_DIRS = [
    "role_rollouts",
    "eval_rollouts",
    "role_vectors",
    "role_validation",
    "alignment",
    "scores",
    "aggregates",
    "figures",
    "reports",
]

POOLED_ACTIVATION_DIRS = [
    "role_activations",
    "eval_activations",
]


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file())


def should_skip(path: Path, include_pooled_activations: bool) -> bool:
    parts = set(path.parts)
    if "__pycache__" in parts:
        return True
    if path.name == ".DS_Store" or path.name.endswith(".tmp"):
        return True
    if "raw" in parts and "eval_prompts" in parts:
        return True
    if "full_hidden_state_dumps" in parts or "unpooled_token_activations" in parts:
        return True
    if not include_pooled_activations and (
        "role_activations" in parts or "eval_activations" in parts
    ):
        return True
    return False


def collect_export_files(repo_root: Path, run_dir: Path, include_pooled_activations: bool) -> list[Path]:
    files: list[Path] = []
    for path in iter_files(repo_root / "data"):
        if not should_skip(path, include_pooled_activations):
            files.append(path)
    for path in iter_files(run_dir):
        if not should_skip(path, include_pooled_activations):
            files.append(path)
    for rel in ["README.md", "pyproject.toml"]:
        path = repo_root / rel
        if path.exists():
            files.append(path)
    for path in iter_files(repo_root / "configs"):
        files.append(path)
    return sorted(set(files))


def write_export_manifest(
    repo_root: Path,
    run_dir: Path,
    output_path: Path,
    include_pooled_activations: bool = True,
) -> Path:
    files = collect_export_files(repo_root, run_dir, include_pooled_activations)
    rel_files = [str(p.relative_to(repo_root)) for p in files]
    manifest = {
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "include_pooled_activations": include_pooled_activations,
        "num_files": len(rel_files),
        "files": sorted(rel_files),
        "excluded": [
            "data/eval_prompts/raw/",
            "*.tmp",
            "__pycache__/",
            "full_hidden_state_dumps/",
            "unpooled_token_activations/",
        ],
    }
    write_json(output_path, manifest)
    return output_path


def upload_to_hf_dataset(
    repo_root: Path,
    run_dir: Path,
    repo_id: str,
    private: bool,
    include_pooled_activations: bool,
    commit_message: str,
    token: str | None = None,
) -> str:
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    run_pattern = f"{run_dir.relative_to(repo_root).as_posix()}/**"
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(repo_root),
        path_in_repo=".",
        commit_message=commit_message,
        allow_patterns=[
            "README.md",
            "pyproject.toml",
            "configs/**",
            "data/**",
            run_pattern,
        ],
        ignore_patterns=DEFAULT_EXCLUDE_PATTERNS
        + ([] if include_pooled_activations else ["runs/**/role_activations/**", "runs/**/eval_activations/**"]),
    )
    return f"https://huggingface.co/datasets/{repo_id}"
