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
    return Views(
        global_view=rng.normal(size=11),
        local_view=rng.normal(size=5),
        odd_view=rng.normal(size=5),
        even_view=rng.normal(size=5),
        secondary_view=rng.normal(size=5),
        global_empty=0.0,
        local_empty=0.25,
        odd_empty=0.0,
        even_empty=0.0,
        secondary_empty=0.5,
        n_in_transit=40 + seed,
    )


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
    np.testing.assert_array_equal(df.loc["K1.02", "secondary_view"], views(2).secondary_view)
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


def test_select_stars_test_first_and_dev_capped():
    from exoplanet.data.lightcurves import select_stars

    splits = pd.DataFrame(
        {
            "kepoi_name": [f"K{i}.01" for i in range(20)] + ["K0.02"],
            "kepid": list(range(20)) + [0],
            "split": ["test"] * 5 + ["dev"] * 15 + ["test"],
        }
    )
    built = {1, 7, 8}
    sel = select_stars(splits, dev_stars=6, built=built, seed=0)
    assert list(sel["kepid"][:5]) == [0, 1, 2, 3, 4]  # all test stars, first
    dev = sel[sel["split"] == "dev"]
    assert len(dev) == 6 and {7, 8} <= set(dev["kepid"])
    assert list(dev["kepid"][:2]) == [7, 8]  # already built come first
    assert sel["kepid"].is_unique
    again = select_stars(splits, dev_stars=6, built=built, seed=0)
    pd.testing.assert_frame_equal(sel, again)
    full = select_stars(splits, dev_stars=None, built=built, seed=0)
    assert len(full) == 20


def test_complete_cache_files_detects_missing_quarters(config):
    from exoplanet.data.lightcurves import complete_cache_files

    root = config["paths"]["lightcurve_dir"] / "mastDownload" / "Kepler"
    full = root / "kplr000000100_lc_Q0111"
    full.mkdir(parents=True)
    for q in range(3):
        (full / f"kplr000000100-{q}_llc.fits").write_bytes(b"")
    assert len(complete_cache_files(100, config)) == 3

    partial = root / "kplr000000200_lc_Q1111"
    partial.mkdir(parents=True)
    (partial / "kplr000000200-0_llc.fits").write_bytes(b"")
    assert complete_cache_files(200, config) is None
    assert complete_cache_files(300, config) is None  # not cached
