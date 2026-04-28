#!/usr/bin/env python
from __future__ import annotations

import argparse

from persona_exp.config import load_config, run_dir
from persona_exp.hf_export import write_export_manifest
from persona_exp.io import write_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rd = run_dir(cfg)
    write_status(rd, "running", "hf_export_manifest")
    output = rd / "hf_export_manifest.json" if args.output is None else rd / args.output
    write_export_manifest(rd, output)
    write_status(rd, "completed", "hf_export_manifest", {"manifest": str(output)})
    print(output)


if __name__ == "__main__":
    main()

