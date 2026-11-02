from pathlib import Path

import numpy as np
import yaml

from exoplanet.config import DEFAULT_CONFIG, REPO_ROOT, load_config, set_seed


def test_default_config_loads_with_absolute_paths():
    config = load_config()
    assert config["seed"] == 42
    for path in config["paths"].values():
        assert path.is_absolute()


def test_relative_paths_resolve_under_repo_and_absolute_paths_are_kept(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["paths"] = {"rel": "data/raw", "abs": str(tmp_path / "cache")}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    paths = load_config(path)["paths"]
    assert paths["rel"] == REPO_ROOT / "data" / "raw"
    assert paths["abs"] == Path(tmp_path / "cache").resolve()


def test_candidates_excluded_from_labels():
    labels = load_config()["labels"]
    assert "CANDIDATE" in labels["exclude"]
    assert labels["positive"] not in labels["exclude"]
    assert labels["negative"] not in labels["exclude"]


def test_set_seed_is_reproducible():
    first = set_seed(0).random(3)
    assert np.array_equal(first, set_seed(0).random(3))
