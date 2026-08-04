from __future__ import annotations

import bz2
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_libsvm_suite_case import derive_case


def test_reservoir_sample_is_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "toy.libsvm"
    source.write_text(
        "".join(
            f"{1 if index % 2 else -1} 1:{index + 1}.0 3:1.0\n"
            for index in range(30)
        ),
        encoding="utf-8",
    )
    outputs = [tmp_path / "first.bz2", tmp_path / "second.bz2"]
    reports = [
        derive_case(
            source,
            output,
            archive_type="plain",
            tar_member_regex=None,
            expected_rows=30,
            expected_dimension=3,
            sample_size=12,
            seed=20260731,
            case_id="toy",
            source_url="https://example.invalid/toy",
            overwrite=False,
        )
        for output in outputs
    ]
    assert reports[0]["selected_row_indices_sha256"] == reports[1][
        "selected_row_indices_sha256"
    ]
    assert bz2.open(outputs[0], "rb").read() == bz2.open(outputs[1], "rb").read()
