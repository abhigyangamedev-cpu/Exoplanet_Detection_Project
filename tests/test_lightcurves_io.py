import numpy as np
import pandas as pd
import pytest

from exoplanet.config import load_config
from exoplanet.data.lightcurves import (
    ephemerides,
    load_views,
    save_star_views,
    star_cache_dirs,
    star_path,
    view_params,
)
from exoplanet.features.lightcurve import Views


@pytest.fixture
def config(tmp_path):
    config = load_config()
    config["paths"]["views_dir"] = tmp_path / "views"
    config["paths"]["lightcurve_dir"] = tmp_path / "lc"
    return config


def views(seed):
    rng = np.random.default_rng(seed)
    return Views(rng.normal(size=11), rng.normal(size=5), 0.0, 0.25, 40 + seed)


def test_ephemerides_converts_hours_and_skips_incomplete():
    df = pd.DataFrame(
        {
            "kepoi_name": ["K1.01", "K1.02", "K1.03"],
            "koi_period": [3.0, np.nan, 5.0],
            "koi_time0bk": [130.0, 131.0, 132.0],
            "koi_duration": [2.4, 3.0, 0.0],
        }
    )
    eph = ephemerides(df)
    assert list(eph) == ["K1.01"]
    assert eph["K1.01"].duration == pytest.approx(0.1)
    assert eph["K1.01"].epoch == 130.0


def test_save_and_load_roundtrip(config):
    save_star_views(star_path(100, config), {"K1.02": views(2), "K1.01": views(1)})
    save_star_views(star_path(200, config), {"K2.01": views(3)})
    (config["paths"]["views_dir"] / "300.tmp.npz").write_bytes(b"partial")

    df = load_views(config)
    assert list(df.index) == ["K1.01", "K1.02", "K2.01"]
    assert list(df["kepid"]) == [100, 100, 200]
    np.testing.assert_array_equal(df.loc["K1.02", "global_view"], views(2).global_view)
    assert df.loc["K2.01", "n_in_transit"] == 43 and df.loc["K1.01", "local_empty"] == 0.25

    subset = load_views(config, pd.Series(["K2.01", "K9.01"]))
    assert list(subset.index) == ["K2.01"]


def test_load_views_empty_dir(config):
    config["paths"]["views_dir"].mkdir(parents=True)
    assert load_views(config).empty


def test_star_cache_dirs_match_zero_padded_kepid(config):
    root = config["paths"]["lightcurve_dir"] / "mastDownload" / "Kepler"
    for name in ["kplr000757450_lc_Q011", "kplr000757450_lc_Q111", "kplr007574500_lc_Q1"]:
        (root / name).mkdir(parents=True)
    assert [d.name for d in star_cache_dirs(757450, config)] == [
        "kplr000757450_lc_Q011",
        "kplr000757450_lc_Q111",
    ]


def test_view_params_merges_sections(config):
    params = view_params(config)
    for key in ("mask_width", "window_days", "clip_sigma", "global_bins", "local_bin_width"):
        assert key in params
