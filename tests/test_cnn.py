import numpy as np
import pandas as pd
import pytest
import torch

from exoplanet.data.splits import KOI, held_out_kois, make_splits
from exoplanet.evaluation.metrics import SCORE, pooled_pr_auc
from exoplanet.models.cnn import TwoTowerCNN, cross_validate, fit_final, predict, train

GLOBAL_LEN, LOCAL_LEN = 101, 31
PARAMS = {
    "epochs": 8,
    "batch_size": 32,
    "learning_rate": 3e-3,
    "weight_decay": 0.0,
    "flip_augment": True,
    "arch": {
        "global": {"channels": [4, 8], "kernel": 5, "pool": 3, "pool_stride": 2},
        "local": {"channels": [4], "kernel": 5, "pool": 3, "pool_stride": 2},
        "fc": [16],
        "dropout": 0.1,
    },
}


def synthetic_views(n=300, seed=0):
    """Positives: flat-bottomed transit. Negatives: V-shaped eclipse plus a secondary eclipse
    at phase 0.5. Both normalised to minimum -1, with noise.
    """
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    xg, xl = np.linspace(-0.5, 0.5, GLOBAL_LEN), np.linspace(-2, 2, LOCAL_LEN)
    g_rows, l_rows = [], []
    for y in label:
        if y:
            local = -(np.abs(xl) < 0.5).astype(float)
            glob = -(np.abs(xg) < 0.01).astype(float)
        else:
            local = -np.clip(1 - np.abs(xl) / 0.5, 0, None)
            glob = -np.clip(1 - np.abs(xg) / 0.015, 0, None)
            glob -= 0.5 * (np.abs(np.abs(xg) - 0.5) < 0.012)
        g_rows.append(glob + rng.normal(0, 0.15, GLOBAL_LEN))
        l_rows.append(local + rng.normal(0, 0.15, LOCAL_LEN))
    kois = [f"K{i:05d}.01" for i in range(n)]
    views = pd.DataFrame(
        {"global_view": g_rows, "local_view": l_rows}, index=pd.Index(kois, name=KOI)
    )
    sample = pd.DataFrame({KOI: kois, "kepid": np.arange(n), "label": label})
    return views, make_splits(sample, test_fraction=0.2, n_folds=3, seed=0)


@pytest.fixture(scope="module")
def data():
    return synthetic_views()


@pytest.fixture(scope="module")
def cv(data):
    views, splits = data
    return cross_validate(views, splits, PARAMS, seed=0)


def test_forward_shape():
    model = TwoTowerCNN(GLOBAL_LEN, LOCAL_LEN, PARAMS["arch"])
    out = model(torch.zeros(5, 1, GLOBAL_LEN), torch.zeros(5, 1, LOCAL_LEN))
    assert out.shape == (5,)


def test_learns_transit_shape(cv):
    oof, metrics, curves = cv
    assert pooled_pr_auc(oof) > 0.9
    assert oof[SCORE].between(0, 1).all()
    assert curves.shape == (PARAMS["epochs"], 3) and list(curves.index) == list(range(1, 9))


def test_oof_covers_dev_kois_with_views_only(data):
    views, splits = data
    missing = views.index[:20]
    oof, _, _ = cross_validate(views.drop(index=missing), splits, PARAMS | {"epochs": 1}, 0)
    test = set(held_out_kois(splits))
    assert not set(oof.index) & test
    assert set(oof.index) == set(splits[KOI]) - test - set(missing)


def test_test_views_never_used(data):
    views, splits = data
    corrupted = views.copy()
    for koi in held_out_kois(splits):
        corrupted.at[koi, "global_view"] = np.full(GLOBAL_LEN, 1e6)
    params = PARAMS | {"epochs": 2}
    a, _, _ = cross_validate(views, splits, params, 0)
    b, _, _ = cross_validate(corrupted, splits, params, 0)
    pd.testing.assert_frame_equal(a, b)
    dev = views.drop(index=held_out_kois(splits))
    np.testing.assert_array_equal(
        predict(fit_final(views, splits, params, 0), dev),
        predict(fit_final(corrupted, splits, params, 0), dev),
    )


def test_deterministic(data):
    views, splits = data
    labels = splits.set_index(KOI)["label"].loc[views.index]
    params = PARAMS | {"epochs": 2}
    a, _ = train(views, labels, params, seed=3)
    b, _ = train(views, labels, params, seed=3)
    np.testing.assert_array_equal(predict(a, views), predict(b, views))


def test_requires_koi_index(data):
    views, splits = data
    with pytest.raises(ValueError, match="indexed by"):
        cross_validate(views.reset_index(), splits, PARAMS, 0)
