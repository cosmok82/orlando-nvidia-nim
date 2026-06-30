#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_catalog.py
-------------------
Author: Cosimo Orlando (CosOr)

Converts nim_langchain_dump.json (output of nim_probe_models.py) into the
models.json file used by the orlando-nvidia-nim extension.

Flow:
  1. run nim_probe_models.py  -> produces nim_langchain_dump.json
  2. run this script           -> produces ../models.json
  3. run /nim-refresh-catalog in pi (or /reload)
Usage:
  python generate_catalog.py [path_to_dump.json]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DUMP = ROOT / "nim_langchain_dump.json"
OUT = ROOT / "models.json"


def main() -> None:
    dump_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DUMP
    if not dump_path.exists():
        print(f"Dump not found: {dump_path}")
        sys.exit(1)
    data = json.loads(dump_path.read_text(encoding="utf-8"))
    models = data.get("models") or []
    out = {
        "sdk_version": data.get("sdk_version"),
        "generated_at": data.get("generated_at"),
        "models": models,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written {OUT} ({len(models)} models)")


if __name__ == "__main__":
    main()
