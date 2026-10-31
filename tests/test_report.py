import json

import numpy as np
import pandas as pd
import pytest

from exoplanet.config import DEFAULT_CONFIG, load_config
from exoplanet.data.splits import KOI, make_splits
from exoplanet.evaluation.report import build_report


@pytest.fixture
def splits():
    rng = np.random.default_rng(0)
    n = 200
    sample = pd.DataFrame(
        {
            KOI: [f"K{i:05d}.01" for i in range(n)],
            "kepid": np.arange(n),
            "label": rng.integers(0, 2, n),
        }
    )
    return make_splits(sample, test_fraction=0.2, n_folds=5, seed=0)


@pytest.fixture
def config(tmp_path):
    config = load_config()
    config["paths"]["outputs_dir"] = tmp_path / "outputs"
    config["paths"]["raw_dir"] = tmp_path / "raw"
    config["paths"]["outputs_dir"].mkdir()
    return config


def report(config, splits, tmp_path, views_kois=None):
    return build_report(
        config,
        DEFAULT_CONFIG,
        splits,
        pd.Index(views_kois if views_kois is not None else [], name=KOI),
        None,
        tmp_path / "figures",
    )


def test_everything_pending_on_fresh_pipeline(config, splits, tmp_path):
    text = report(config, splits, tmp_path)
    assert "200 labelled KOIs" in text
    assert "_Pending: no light-curve views built" in text
    for model in ["tabular", "tabular_lc", "cnn"]:
        assert f"_Pending: `{model}` not trained" in text
    assert "_Pending: development folds not evaluated" in text
    assert "_Pending: held-out test set not evaluated" in text


def test_sections_filled_from_outputs(config, splits, tmp_path):
    outputs = config["paths"]["outputs_dir"]
    dev = splits[splits["split"] == "dev"]
    rng = np.random.default_rng(1)
    for model in ["tabular", "tabular_lc"]:
        pd.DataFrame(
            {
                KOI: dev[KOI],
                "fold": dev["fold"],
                "label": dev["label"],
                "score": rng.random(len(dev)),
            }
        ).to_csv(outputs / f"{model}_oof.csv", index=False)
    (outputs / "evaluation_dev_pr.png").write_bytes(b"\x89PNG fake")
    summary = {
        "models": ["tabular", "tabular_lc"],
        "n_kois": 160,
        "n_total": 160,
        "positive_fraction": 0.4,
        "target_precision": 0.95,
        "n_bootstrap": 10,
        "pr_auc": {
            m: {"pr_auc": 0.9, "ci_low": 0.8, "ci_high": 0.95} for m in ["tabular", "tabular_lc"]
        },
        "differences": {
            "tabular - tabular_lc": {
                "difference": 0.0,
                "ci_low": -0.1,
                "ci_high": 0.1,
                "p_value": 1.0,
            }
        },
        "operating_points": {
            m: {
                "threshold": 0.5,
                "precision": 0.95,
                "recall": 0.8,
                "tp": 50,
                "fp": 3,
                "fn": 12,
                "tn": 95,
            }
            for m in ["tabular", "tabular_lc"]
        },
        "fp_rejection": {"all false positives": {"n": 98, "tabular": 0.97, "tabular_lc": 0.96}},
        "figure": "evaluation_dev_pr.png",
    }
    (outputs / "evaluation_dev.json").write_text(json.dumps(summary), encoding="utf-8")

    text = report(config, splits, tmp_path, views_kois=list(splits[KOI][:50]))
    assert "50 of 200 KOIs (25.0%) have light-curve views" in text
    assert "| tabular | 160 |" in text and "_Pending: `cnn` not trained" in text
    assert "_Pending: `tabular` not trained" not in text
    assert "_Incomplete: evaluated ['tabular', 'tabular_lc'], not ['cnn']" in text
    assert "![Precision-recall curves](figures/evaluation_dev_pr.png)" in text
    assert (tmp_path / "figures" / "evaluation_dev_pr.png").exists()
    assert "| tabular - tabular_lc | 0.000 |" in text
    assert "_Pending: held-out test set not evaluated" in text
