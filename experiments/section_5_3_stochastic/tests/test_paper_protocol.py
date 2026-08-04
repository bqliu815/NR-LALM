from __future__ import annotations

import json
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def test_paper_protocol_freezes_ten_streams_and_eighty_runs() -> None:
    config = json.loads(
        (EXPERIMENT_DIR / "configs" / "paper_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["protocol_status"] == "PAPER_FROZEN"
    assert config["oracle_repeats"] == 10
    assert len(config["datasets"]) == 2
    assert config["displayed_methods"] == [
        "NR-LALM",
        "NR-LALM+SOC",
        "MLALM",
        "S-SQP",
    ]
    array_tasks = len(config["datasets"]) * config["oracle_repeats"]
    assert array_tasks == 20
    assert array_tasks * len(config["displayed_methods"]) == 80
