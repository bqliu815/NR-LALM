#!/usr/bin/env python3
"""Fail on common accidental leaks or malformed frozen configurations."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".sbatch", ".toml", ".txt"}
FORBIDDEN_TEXT = (
    "/Users/",
    "paper——SOCLALM",
    "formal_experiments/",
    "numerical_experiments/",
    "NV_H100",
    "TO_BE_FILLED_BEFORE_LAUNCH",
)


def main() -> None:
    issues: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if path.stat().st_size > 5 * 1024 * 1024:
            issues.append(f"large file: {relative}")
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            issues.append(f"generated Python file: {relative}")
        if "data" in path.parts and path.name != "README.md":
            issues.append(f"data payload: {relative}")
        if "results" in path.parts and path.name != "README.md":
            issues.append(f"result payload: {relative}")
        if (
            relative != Path("scripts/validate_release.py")
            and (path.suffix in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore"})
        ):
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN_TEXT:
                if token in text:
                    issues.append(f"machine/internal reference {token!r}: {relative}")
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        if path.suffix == ".toml":
            tomllib.loads(path.read_text(encoding="utf-8"))
    if issues:
        raise SystemExit("release validation failed:\n- " + "\n- ".join(issues))
    print("release validation: PASS")


if __name__ == "__main__":
    main()
