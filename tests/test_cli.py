"""Exercise the public CSV example without any external dataset."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class CsvExampleTests(unittest.TestCase):
    def test_example_reports_held_out_metrics(self) -> None:
        rng = np.random.default_rng(31)
        n_rows = 80
        values = rng.normal(size=n_rows)
        frame = pd.DataFrame(
            {
                "measurement": values,
                "category": np.where(values > 0, "right", "left"),
                "outcome": (values > 0).astype(int),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic.csv"
            predictions = root / "predictions.csv"
            frame.to_csv(source, index=False)
            script = Path(__file__).resolve().parents[1] / "examples" / "train_csv.py"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--csv", str(source),
                    "--target", "outcome",
                    "--positive-label", "1",
                    "--max-clusters", "2",
                    "--predictions", str(predictions),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(result.stdout)
            self.assertEqual(report["n_train"] + report["n_test"], n_rows)
            self.assertGreaterEqual(report["f1"], 0)
            self.assertLessEqual(report["f1"], 1)
            self.assertEqual(len(pd.read_csv(predictions)), report["n_test"])


if __name__ == "__main__":
    unittest.main()
