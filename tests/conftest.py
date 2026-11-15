from __future__ import annotations

from pathlib import Path

import pytest

from alert_explainer.data.etl import run_etl
from alert_explainer.data.splits import run_splits
from alert_explainer.data.synthetic import make_synthetic


@pytest.fixture(scope="session")
def pipeline_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic CSVs → ETL → splits, once per test session."""
    root = tmp_path_factory.mktemp("pipeline")
    make_synthetic(root / "raw", rows_per_file=700, seed=3)
    run_etl(root / "raw", root / "processed", root / "reports", root / "docs")
    run_splits(root / "processed", root / "reports", seed=42)
    return root
