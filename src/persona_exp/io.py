from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors.torch import save_file

from persona_exp.utils import read_json, utc_now, write_json, write_jsonl


def clean_tmp_files(root: Path) -> int:
    if not root.exists():
        return 0
    count = 0
    for path in root.rglob("*.tmp"):
        path.unlink()
        count += 1
    return count


def write_status(run_dir: Path, status: str, stage: str, extra: dict[str, Any] | None = None) -> None:
    payload = {"status": status, "stage": stage, "updated_at": utc_now()}
    if extra:
        payload.update(extra)
    write_json(run_dir / "status.json", payload)


def update_manifest(run_dir: Path, key: str, value: Any) -> None:
    path = run_dir / "manifest.json"
    manifest = read_json(path) if path.exists() else {}
    manifest[key] = value
    manifest["updated_at"] = utc_now()
    write_json(path, manifest)


def completed_marker(path: Path) -> Path:
    return path / "_SUCCESS"


def is_completed(path: Path) -> bool:
    return completed_marker(path).exists()


def mark_completed(path: Path, payload: dict[str, Any] | None = None) -> None:
    write_json(completed_marker(path), payload or {"completed_at": utc_now()})


def shard_records(records: list[dict[str, Any]], shard_size: int) -> list[list[dict[str, Any]]]:
    return [records[i : i + shard_size] for i in range(0, len(records), shard_size)]


def write_jsonl_shards(
    out_dir: Path,
    records: list[dict[str, Any]],
    shard_size: int,
    prefix: str = "part",
    overwrite: bool = False,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_completed(out_dir) and not overwrite:
        return sorted(out_dir.glob(f"{prefix}-*.jsonl"))
    clean_tmp_files(out_dir)
    paths = []
    for idx, shard in enumerate(shard_records(records, shard_size)):
        path = out_dir / f"{prefix}-{idx:05d}.jsonl"
        if path.exists() and not overwrite:
            paths.append(path)
            continue
        write_jsonl(path, shard)
        paths.append(path)
    mark_completed(out_dir, {"num_records": len(records), "num_shards": len(paths)})
    return paths


def write_tensor_shard(
    tensor_path: Path,
    metadata_path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: pd.DataFrame,
) -> None:
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_tensor = tensor_path.with_suffix(tensor_path.suffix + ".tmp")
    tmp_meta = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    save_file(tensors, str(tmp_tensor))
    tmp_tensor.replace(tensor_path)
    metadata.to_parquet(tmp_meta, index=False)
    tmp_meta.replace(metadata_path)


def append_jsonl_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

