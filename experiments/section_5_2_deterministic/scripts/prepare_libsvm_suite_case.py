#!/usr/bin/env python3
"""Download and deterministically derive one frozen LIBSVM suite case."""

from __future__ import annotations

import argparse
import bz2
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import lzma
from pathlib import Path
import re
import subprocess
import tarfile
from typing import Any, BinaryIO, Iterator

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return
    partial = output.with_suffix(output.suffix + ".part")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "8",
            "--retry-delay",
            "5",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    partial.replace(output)


@contextmanager
def source_lines(
    path: Path,
    archive_type: str,
    *,
    tar_member_regex: str | None,
) -> Iterator[tuple[Iterator[bytes], dict[str, Any]]]:
    metadata: dict[str, Any] = {"tar_members": []}
    if archive_type == "bz2":
        with bz2.open(path, mode="rb") as stream:
            yield iter(stream), metadata
        return
    if archive_type == "xz":
        with lzma.open(path, mode="rb") as stream:
            yield iter(stream), metadata
        return
    if archive_type == "plain":
        with path.open("rb") as stream:
            yield iter(stream), metadata
        return
    if archive_type != "tar.xz":
        raise ValueError(f"unsupported archive type {archive_type}")
    if not tar_member_regex:
        raise ValueError("tar.xz source requires tar_member_regex")
    pattern = re.compile(tar_member_regex, flags=re.IGNORECASE)
    # The official Criteo archive concatenates multiple XZ streams.  The
    # streaming tarfile backend stops after the first stream, whereas the
    # seekable LZMAFile backend continues across all concatenated streams.
    archive = tarfile.open(path, mode="r:xz")

    def iter_tar_lines() -> Iterator[bytes]:
        matched = 0
        try:
            for member in archive:
                if not member.isfile() or not pattern.fullmatch(member.name):
                    continue
                extracted: BinaryIO | None = archive.extractfile(member)
                if extracted is None:
                    continue
                matched += 1
                metadata["tar_members"].append(member.name)
                with extracted:
                    yield from extracted
        finally:
            archive.close()
        if matched == 0:
            raise ValueError(
                f"no tar member matched {tar_member_regex!r}"
            )

    try:
        yield iter_tar_lines(), metadata
    finally:
        archive.close()


def split_raw_line(raw_line: bytes) -> tuple[float, list[bytes]]:
    fields = raw_line.split()
    if not fields:
        raise ValueError("empty line")
    return float(fields[0]), fields[1:]


def canonical_line(label: int, features: list[bytes]) -> bytes:
    prefix = b"1" if label > 0 else b"-1"
    if features:
        return prefix + b" " + b" ".join(features) + b"\n"
    return prefix + b"\n"


def validate_source_feature_tail(
    features: list[bytes],
    *,
    expected_dimension: int,
    case_id: str,
    row_number: int,
) -> tuple[int, int]:
    """Validate any raw indices beyond the declared dimension are zero."""

    if not features:
        return 0, 0
    maximum_index = int(features[-1].split(b":", maxsplit=1)[0])
    ignored_out_of_range_zeros = 0
    if maximum_index <= expected_dimension:
        return maximum_index, ignored_out_of_range_zeros
    for item in reversed(features):
        index_text, value_text = item.split(b":", maxsplit=1)
        index = int(index_text)
        if index <= expected_dimension:
            break
        if float(value_text) != 0.0:
            raise ValueError(
                f"{case_id}: nonzero feature {index} exceeds dimension "
                f"{expected_dimension} on source row {row_number}"
            )
        ignored_out_of_range_zeros += 1
    return maximum_index, ignored_out_of_range_zeros


def canonicalize_sample_features(
    features: list[bytes],
    *,
    expected_dimension: int,
    case_id: str,
    row_number: int,
) -> tuple[list[bytes], int, int]:
    """Drop explicit zeros and validate sorted, in-range nonzero entries."""

    canonical: list[bytes] = []
    previous = 0
    maximum_nonzero_index = 0
    removed_zeros = 0
    for item in features:
        index_text, value_text = item.split(b":", maxsplit=1)
        index = int(index_text)
        if index <= previous:
            raise ValueError(
                f"{case_id}: nonincreasing feature indices "
                f"on source row {row_number}"
            )
        previous = index
        value = float(value_text)
        if value == 0.0:
            removed_zeros += 1
            continue
        if index > expected_dimension:
            raise ValueError(
                f"{case_id}: nonzero feature {index} exceeds "
                f"dimension {expected_dimension}"
            )
        canonical.append(item)
        maximum_nonzero_index = max(maximum_nonzero_index, index)
    return canonical, maximum_nonzero_index, removed_zeros


def derive_case(
    source: Path,
    output: Path,
    *,
    archive_type: str,
    tar_member_regex: str | None,
    expected_rows: int,
    expected_dimension: int,
    sample_size: int,
    seed: int,
    case_id: str,
    source_url: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    if expected_rows <= 0 or expected_dimension <= 0:
        raise ValueError("invalid expected source dimensions")
    if sample_size <= 0 or sample_size > expected_rows:
        raise ValueError("invalid sample size")
    output = output.resolve()
    report_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() and report_path.exists() and not overwrite:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("sample_sha256") == sha256(output)
            and report.get("source_sha256") == sha256(source)
        ):
            return report
        raise ValueError(f"stale derived artifact {output}")
    rng = np.random.default_rng(seed)
    reservoir: list[tuple[int, float, list[bytes]]] = []
    rows = 0
    raw_label_counts: Counter[float] = Counter()
    maximum_source_raw_feature_index = 0
    source_out_of_range_explicit_zeros = 0
    with source_lines(
        source,
        archive_type,
        tar_member_regex=tar_member_regex,
    ) as (lines, stream_metadata):
        for raw_line in lines:
            if not raw_line.strip():
                continue
            rows += 1
            raw_label, features = split_raw_line(raw_line)
            raw_label_counts[raw_label] += 1
            raw_maximum, ignored_zeros = validate_source_feature_tail(
                features,
                expected_dimension=expected_dimension,
                case_id=case_id,
                row_number=rows,
            )
            maximum_source_raw_feature_index = max(
                maximum_source_raw_feature_index, raw_maximum
            )
            source_out_of_range_explicit_zeros += ignored_zeros
            if rows <= sample_size:
                reservoir.append((rows, raw_label, features))
                continue
            replacement = int(rng.integers(0, rows))
            if replacement < sample_size:
                reservoir[replacement] = (rows, raw_label, features)
    if rows != expected_rows:
        raise ValueError(
            f"{case_id}: source row count {rows} != expected {expected_rows}"
        )
    labels = sorted(raw_label_counts)
    if len(labels) != 2:
        raise ValueError(
            f"{case_id}: expected two labels, found {labels}"
        )
    label_map = {labels[0]: -1, labels[1]: 1}
    reservoir.sort(key=lambda item: item[0])
    selected_rows: list[int] = []
    mapped_label_counts: Counter[int] = Counter()
    maximum_feature = 0
    sampled_nonzeros = 0
    sample_explicit_zeros_removed = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.unlink(missing_ok=True)
    with bz2.open(temporary, mode="wb", compresslevel=9) as stream:
        for line_number, raw_label, features in reservoir:
            selected_rows.append(line_number)
            mapped = label_map[raw_label]
            mapped_label_counts[mapped] += 1
            canonical, row_maximum, removed_zeros = (
                canonicalize_sample_features(
                    features,
                    expected_dimension=expected_dimension,
                    case_id=case_id,
                    row_number=line_number,
                )
            )
            sampled_nonzeros += len(canonical)
            sample_explicit_zeros_removed += removed_zeros
            maximum_feature = max(maximum_feature, row_maximum)
            stream.write(canonical_line(mapped, canonical))
    temporary.replace(output)
    selected_bytes = np.asarray(selected_rows, dtype="<i8").tobytes()
    report: dict[str, Any] = {
        "schema": "libsvm_binary_suite_derived_case_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "algorithm": (
            "Algorithm R over all nonempty source lines; full retention "
            "when sample_size equals source_rows"
        ),
        "canonicalization": (
            "drop every explicit zero-valued feature entry; an index above "
            "the declared dimension is accepted only when its value is zero"
        ),
        "source_url": source_url,
        "source_path": str(source.resolve()),
        "source_archive_type": archive_type,
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256(source),
        "source_rows": rows,
        "expected_source_rows": expected_rows,
        "expected_dimension": expected_dimension,
        "raw_label_counts": {
            repr(label): count
            for label, count in sorted(raw_label_counts.items())
        },
        "label_mapping": {
            repr(label): mapped for label, mapped in label_map.items()
        },
        "stream_metadata": stream_metadata,
        "sample_path": str(output),
        "sample_bytes": output.stat().st_size,
        "sample_sha256": sha256(output),
        "sample_size": sample_size,
        "sample_seed": seed,
        "selected_row_indices_sha256": hashlib.sha256(
            selected_bytes
        ).hexdigest(),
        "minimum_selected_row": selected_rows[0],
        "maximum_selected_row": selected_rows[-1],
        "negative_labels": mapped_label_counts[-1],
        "positive_labels": mapped_label_counts[1],
        "sampled_nonzeros": sampled_nonzeros,
        "maximum_sampled_feature_index": maximum_feature,
        "sample_explicit_zero_entries_removed": (
            sample_explicit_zeros_removed
        ),
        "maximum_source_raw_feature_index": (
            maximum_source_raw_feature_index
        ),
        "source_out_of_range_explicit_zero_entries_ignored": (
            source_out_of_range_explicit_zeros
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--case-index", required=True, type=int)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    package_root = args.package_root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = manifest["cases"][args.case_index]
    source = (
        package_root
        / "data"
        / "libsvm_binary_suite_v1"
        / "source"
        / str(case["source_filename"])
    )
    if args.download:
        download(str(case["source_url"]), source)
    if not source.exists():
        raise FileNotFoundError(source)
    output = package_root / str(case["derived_path"])
    report = derive_case(
        source,
        output,
        archive_type=str(case["source_archive_type"]),
        tar_member_regex=case.get("tar_member_regex"),
        expected_rows=int(case["official_source_rows"]),
        expected_dimension=int(case["dimension"]),
        sample_size=int(case["sample_size"]),
        seed=int(case["sample_seed"]),
        case_id=str(case["case_id"]),
        source_url=str(case["source_url"]),
        overwrite=args.overwrite,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
