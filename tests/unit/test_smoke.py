import alert_explainer
from alert_explainer.config import get_settings


def test_package_imports() -> None:
    assert alert_explainer.__version__


def test_settings_paths() -> None:
    s = get_settings()
    assert s.raw_dir.name == "raw"
    assert s.processed_dir.parent == s.data_dir
