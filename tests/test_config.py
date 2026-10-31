import numpy as np

from exoplanet.config import REPO_ROOT, load_config, set_seed


def test_default_config_loads_with_absolute_paths():
    config = load_config()
    assert config["seed"] == 42
    for path in config["paths"].values():
        assert path.is_absolute()
        assert path.is_relative_to(REPO_ROOT)


def test_candidates_excluded_from_labels():
    labels = load_config()["labels"]
    assert "CANDIDATE" in labels["exclude"]
    assert labels["positive"] not in labels["exclude"]
    assert labels["negative"] not in labels["exclude"]


def test_set_seed_is_reproducible():
    first = set_seed(0).random(3)
    assert np.array_equal(first, set_seed(0).random(3))
