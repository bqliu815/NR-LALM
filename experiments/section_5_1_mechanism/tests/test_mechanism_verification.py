from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PACKAGE_ROOT / "scripts" / "run_mechanism_verification.py"
CONFIG = PACKAGE_ROOT / "configs" / "paper_v1.toml"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_tree_sha256(project_root: Path) -> str:
    source_files = [
        project_root / "scripts" / "run_mechanism_verification.py",
        *sorted((project_root / "src").rglob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(
            path.relative_to(project_root).as_posix().encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class MechanismVerificationIntegrationTest(unittest.TestCase):
    def run_protocol(
        self, root: Path, config_path: Path, output_name: str
    ) -> Path:
        output_dir = root / output_name
        environment = dict(os.environ)
        environment["MPLBACKEND"] = "Agg"
        subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
            cwd=PACKAGE_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        return output_dir

    def test_protocol_outputs_and_label_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.run_protocol(root, CONFIG, "first")
            raw = json.loads((first / "raw.json").read_text())
            self.assertEqual(raw["status"], "PASS")
            self.assertEqual(raw["protocol_status"], "FROZEN")
            protocol = tomllib.loads(CONFIG.read_text())
            certificate = raw["certificate"]
            total_grid_points = (
                int(protocol["certificate"]["delta_points"])
                * int(protocol["certificate"]["lambda_points"])
            )
            base_count = int(certificate["base_feasible_grid_points"])
            corrected_count = int(
                certificate["corrected_feasible_grid_points"]
            )
            self.assertGreater(base_count, 0)
            self.assertGreater(corrected_count, base_count)
            self.assertLess(corrected_count, total_grid_points)
            self.assertFalse(
                raw["certificate"]["strict_base_membership"]
            )
            self.assertTrue(
                raw["certificate"]["strict_corrected_membership"]
            )
            self.assertTrue(
                raw["certificate"]["common_base_membership"]
            )
            self.assertTrue(
                raw["certificate"]["common_corrected_membership"]
            )
            order = protocol["order"]
            base_slope = float(
                raw["order"]["base_slope_summary"]["median"]
            )
            corrected_slope = float(
                raw["order"]["corrected_slope_summary"]["median"]
            )
            self.assertLessEqual(order["base_slope_min"], base_slope)
            self.assertLessEqual(base_slope, order["base_slope_max"])
            self.assertLessEqual(
                order["corrected_slope_min"], corrected_slope
            )
            self.assertLessEqual(
                corrected_slope, order["corrected_slope_max"]
            )

            manifest = json.loads(
                (first / "manifest.json").read_text()
            )
            self.assertEqual(
                manifest["method_colors"],
                {
                    "base": "#0072B2",
                    "corrected": "#D55E00",
                },
            )
            self.assertEqual(
                manifest["axis_labels"],
                {
                    "certificate_x": r"model-step bound $\Delta$",
                    "certificate_y": r"multiplier bound $\Lambda$",
                    "order_x": r"step norm $\Vert p\Vert$",
                    "order_y": r"error norm $\Vert d\Vert$",
                },
            )
            self.assertEqual(
                manifest["panel_titles"],
                {
                    "certificate": "(a) Sufficient parameter regions",
                    "order": "(b) Constraint linearization error",
                },
            )
            artifact_map = {
                "config_sha256": first / "config.toml",
                "runner_sha256": RUNNER,
                "raw_sha256": first / "raw.json",
                "pdf_sha256": first / "mechanism_verification.pdf",
                "png_sha256": first / "mechanism_verification.png",
                "caption_sha256": first / "figure_caption.tex",
                "environment_sha256": first / "environment.json",
                "command_sha256": first / "command.txt",
            }
            for key, path in artifact_map.items():
                self.assertEqual(manifest[key], file_sha256(path))
            self.assertEqual(
                manifest["source_tree_sha256"],
                source_tree_sha256(PACKAGE_ROOT),
            )
            source_snapshot = first / manifest["source_snapshot"]
            self.assertEqual(
                manifest["source_snapshot_sha256"],
                source_tree_sha256(source_snapshot),
            )
            self.assertEqual(
                manifest["source_snapshot_sha256"],
                manifest["source_tree_sha256"],
            )

            renamed_config = root / "renamed.toml"
            display = tomllib.loads(CONFIG.read_text())["display"]
            renamed_config.write_text(
                CONFIG.read_text()
                .replace(
                    f'base_method = "{display["base_method"]}"',
                    'base_method = "Base method"',
                )
                .replace(
                    "corrected_method = "
                    f'"{display["corrected_method"]}"',
                    'corrected_method = "Corrected method"',
                )
            )
            second = self.run_protocol(
                root, renamed_config, "second"
            )
            self.assertEqual(
                (first / "raw.json").read_bytes(),
                (second / "raw.json").read_bytes(),
            )
            renamed_manifest = json.loads(
                (second / "manifest.json").read_text()
            )
            self.assertEqual(
                renamed_manifest["display_labels"],
                {
                    "base": "Base method",
                    "corrected": "Corrected method",
                },
            )


if __name__ == "__main__":
    unittest.main()
