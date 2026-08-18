#!/usr/bin/env python3
"""Validate the public source package and its paper configurations."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".sbatch", ".toml", ".txt"}
GENERATED_SUFFIXES = {
    ".csv",
    ".jpeg",
    ".jpg",
    ".jsonl",
    ".log",
    ".mp4",
    ".npy",
    ".npz",
    ".out",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".svg",
    ".tsv",
}
ALLOWED_DOCUMENTATION_ARTIFACTS = {
    Path("docs/assets/lean_lemma_2_6_walkthrough.mp4"),
    Path("docs/assets/tr_lalm_theorem_map.png"),
}
SENSITIVE_TEXT_PATTERNS = (
    ("macOS user path", re.compile(r"/Users/[A-Za-z0-9._-]+/")),
    ("Unix home path", re.compile(r"/home/[A-Za-z0-9._-]+/")),
    (
        "Windows user path",
        re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
    ),
    (
        "cluster filesystem path",
        re.compile(r"/(?:public_hw|scratch|gpfs|lustre)/"),
    ),
    (
        "private key",
        re.compile(
            r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"
        ),
    ),
    (
        "GitHub token",
        re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    ),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
)


def release_paths() -> list[Path]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    return [
        ROOT / item.decode("utf-8")
        for item in completed.stdout.split(b"\0")
        if item
    ]


def main() -> None:
    issues: list[str] = []
    for path in release_paths():
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if (
            path.suffix.lower() in GENERATED_SUFFIXES
            and relative not in ALLOWED_DOCUMENTATION_ARTIFACTS
        ):
            issues.append(f"generated artifact: {relative}")
        if (
            path.stat().st_size > 5 * 1024 * 1024
            and relative not in ALLOWED_DOCUMENTATION_ARTIFACTS
        ):
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
            for label, pattern in SENSITIVE_TEXT_PATTERNS:
                if pattern.search(text):
                    issues.append(f"sensitive {label}: {relative}")
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        if path.suffix == ".toml":
            tomllib.loads(path.read_text(encoding="utf-8"))
    if issues:
        raise SystemExit("release validation failed:\n- " + "\n- ".join(issues))
    print("release validation: PASS")


if __name__ == "__main__":
    main()
