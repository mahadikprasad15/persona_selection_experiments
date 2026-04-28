from __future__ import annotations

from pathlib import Path

from persona_exp.utils import write_json


def write_export_manifest(run_dir: Path, output_path: Path) -> Path:
    files = [str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file()]
    manifest = {
        "run_dir": str(run_dir),
        "num_files": len(files),
        "files": sorted(files),
        "note": "HF upload is intentionally explicit; use this manifest to review export scope.",
    }
    write_json(output_path, manifest)
    return output_path

