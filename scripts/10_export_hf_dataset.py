#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

from persona_exp.config import load_config, repo_root_from_config, run_dir
from persona_exp.hf_export import upload_to_hf_dataset, write_export_manifest
from persona_exp.io import write_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-id", default=None, help="HF dataset repo id, e.g. username/persona-selection-smollm2-pilot-v001")
    parser.add_argument("--private", action="store_true", help="Create/upload to a private dataset repo")
    parser.add_argument("--dry-run", action="store_true", help="Only write and print export manifest")
    parser.add_argument("--include-pooled-activations", action="store_true", default=True)
    parser.add_argument("--exclude-pooled-activations", dest="include_pooled_activations", action="store_false")
    parser.add_argument("--commit-message", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    repo_root = repo_root_from_config(cfg)
    rd = run_dir(cfg)
    write_status(rd, "running", "hf_export_manifest")
    output = rd / "hf_export_manifest.json" if args.output is None else rd / args.output
    write_export_manifest(repo_root, rd, output, include_pooled_activations=args.include_pooled_activations)
    if args.dry_run:
        write_status(rd, "completed", "hf_export_manifest", {"manifest": str(output), "dry_run": True})
        print(output)
        return
    if not args.repo_id:
        raise ValueError("--repo-id is required unless --dry-run is set")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    commit_message = args.commit_message or f"Upload {cfg['run']['name']} artifacts"
    write_status(rd, "running", "hf_upload", {"repo_id": args.repo_id, "manifest": str(output)})
    url = upload_to_hf_dataset(
        repo_root=repo_root,
        run_dir=rd,
        repo_id=args.repo_id,
        private=args.private,
        include_pooled_activations=args.include_pooled_activations,
        commit_message=commit_message,
        token=token,
    )
    write_status(rd, "completed", "hf_upload", {"repo_id": args.repo_id, "url": url, "manifest": str(output)})
    print(url)


if __name__ == "__main__":
    main()
