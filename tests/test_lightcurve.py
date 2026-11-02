import numpy as np
import pytest

from exoplanet.features.lightcurve import (
    Ephemeris,
    bin_median,
    clip_upward_outliers,
    detrend,
    make_views,
    normalise_view,
    phase,
    process_star,
    scale_like,
    split_segments,
    stitch,
    transit_mask,
    transit_number,
    view_transit_snr,
)

CADENCE = 29.4 / 60 / 24  # Kepler long cadence, days
PARAMS = {
    "mask_width": 2.0,
    "window_days": 1.0,
    "window_durations": 3.0,
    "gap_days": 0.75,
    "clip_sigma": 5.0,
    "global_bins": 201,
    "local_bins": 61,
    "local_durations": 2.0,
    "local_bin_width": 0.16,
}
PLANET = Ephemeris(period=3.5, epoch=1.2, duration=0.15)
DEPTH = 2e-3


def box(time, eph, depth):
    return np.where(np.abs(phase(time, eph.period, eph.epoch)) < eph.duration / 2, -depth, 0.0)


def quarters(n=4, length=40.0, gap=3.0, noise=1e-4, seed=0, transits=((PLANET, DEPTH),)):
    """``n`` quarters with different flux levels, slow variability and box transits."""
    rng = np.random.default_rng(seed)
    out = []
    for q in range(n):
        start = q * (length + gap)
        t = np.arange(start, start + length, CADENCE)
        rel = 1 + 0.01 * np.sin(2 * np.pi * t / 7.0) + rng.normal(0, noise, len(t))
        for eph, depth in transits:
            rel += box(t, eph, depth)
        out.append((t, (1000 + 500 * q) * rel))
    return out


def test_phase_range_and_centre():
    t = np.linspace(-10, 10, 1001)
    ph = phase(t, 3.0, 0.5)
    assert ph.min() >= -1.5 and ph.max() < 1.5
    assert phase(np.array([0.5, 3.5, -2.5]), 3.0, 0.5) == pytest.approx([0, 0, 0])


def test_stitch_normalises_quarters_and_drops_nans():
    t1, f1 = np.arange(5.0), np.array([10, 10, np.nan, 10, 10.0])
    t0, f0 = np.arange(-5.0, 0), np.full(5, 300.0)
    time, flux = stitch([(t1, f1), (t0, f0)])
    assert len(time) == 9 and np.all(np.diff(time) > 0)
    assert flux == pytest.approx(np.ones(9))


def test_split_segments_at_gaps():
    t = np.array([0, 0.1, 0.2, 5.0, 5.1, 20.0])
    assert [(s.start, s.stop) for s in split_segments(t, gap=1.0)] == [(0, 3), (3, 5), (5, 6)]


def test_transit_mask_covers_each_ephemeris():
    t = np.arange(0, 20, 0.01)
    other = Ephemeris(period=7.0, epoch=2.0, duration=0.2)
    mask = transit_mask(t, [PLANET, other], width_factor=1.0)
    in_planet = np.abs(phase(t, PLANET.period, PLANET.epoch)) < PLANET.duration / 2
    in_other = np.abs(phase(t, other.period, other.epoch)) < other.duration / 2
    np.testing.assert_array_equal(mask, in_planet | in_other)


def test_detrend_removes_variability_and_preserves_depth():
    time, flux = stitch(quarters(noise=0))
    mask = transit_mask(time, [PLANET], 2.0)
    t, f = detrend(time, flux, mask, window=1.0, gap=0.75)
    assert len(t) == len(time)
    in_transit = np.abs(phase(t, PLANET.period, PLANET.epoch)) < PLANET.duration / 4
    out = ~transit_mask(t, [PLANET], 2.0)
    assert np.std(f[out]) < 2e-4  # the 1% sinusoid is gone
    assert 1 - np.median(f[in_transit]) == pytest.approx(DEPTH, rel=0.1)


def test_detrend_without_mask_erodes_long_transits():
    """Sanity check of why masking matters: a filter as short as the transit eats it."""
    eph = Ephemeris(period=10.0, epoch=5.0, duration=0.5)
    time, flux = stitch(quarters(noise=0, transits=((eph, DEPTH),)))
    t, f = detrend(time, flux, np.zeros(len(time), bool), window=0.5, gap=0.75)
    centre = np.abs(phase(t, eph.period, eph.epoch)) < 0.05
    assert 1 - np.median(f[centre]) < 0.5 * DEPTH


def test_clip_removes_flares_not_transits():
    rng = np.random.default_rng(0)
    flux = 1 + rng.normal(0, 1e-4, 1000)
    flux[100] += 0.01  # flare
    flux[500:510] -= 0.01  # transit
    mask = np.zeros(1000, bool)
    mask[500:510] = True
    keep = clip_upward_outliers(flux, mask, n_sigma=5)
    assert not keep[100] and keep[500:510].all() and keep.sum() >= 990


def test_bin_median_fills_empty_bins():
    x = np.array([0.05, 0.15, 0.85, 0.95])
    y = np.array([1.0, 3.0, 5.0, 7.0])
    view, empty = bin_median(x, y, n_bins=10, lo=0.0, hi=1.0)
    assert empty == pytest.approx(0.6)
    assert view[0] == 1 and view[1] == 3 and view[9] == 7
    assert view[2] == pytest.approx(3 + (5 - 3) / 7)


def test_bin_median_wide_bins_overlap_and_smooth():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 2000))
    y = rng.normal(0, 1, 2000)
    narrow, _ = bin_median(x, y, n_bins=100, lo=0, hi=1)
    wide, empty = bin_median(x, y, n_bins=100, lo=0, hi=1, width=0.08)
    assert empty == 0 and np.std(wide) < 0.5 * np.std(narrow)
    # Bins never reach outside [lo, hi).
    edge, _ = bin_median(np.array([-0.01, 0.01]), np.array([100.0, 1.0]), 10, 0, 1, width=0.5)
    assert edge[0] == 1


def test_bin_median_all_empty_is_nan():
    view, empty = bin_median(np.array([5.0]), np.array([1.0]), n_bins=4, lo=0, hi=1)
    assert empty == 1 and np.isnan(view).all()


def test_normalise_view():
    v = normalise_view(np.array([1.0, 1.0, 0.98, 1.0, 1.0]))
    assert np.median(v) == 0 and v.min() == pytest.approx(-1)
    flat = normalise_view(np.full(5, 3.0))
    assert (flat == 0).all()


def test_views_are_centred_on_transit():
    time, flux = stitch(quarters())
    mask = transit_mask(time, [PLANET], 2.0)
    t, f = detrend(time, flux, mask, window=1.0, gap=0.75)
    views = make_views(t, f, PLANET, global_bins=201, local_bins=61, local_durations=2.0)
    assert views.global_view.shape == (201,) and views.local_view.shape == (61,)
    assert views.local_view.min() == pytest.approx(-1)
    assert abs(np.argmin(views.local_view) - 30) <= 8  # within the transit
    assert abs(np.argmin(views.global_view) - 100) <= 3
    # Out-of-transit ends of the local view sit near zero.
    assert np.abs(views.local_view[:10]).max() < 0.3
    assert views.local_empty == 0 and views.n_in_transit > 100


def test_process_star_masks_all_kois_and_returns_targets():
    other = Ephemeris(period=11.3, epoch=4.0, duration=0.4)
    segs = quarters(seed=3, transits=((PLANET, DEPTH), (other, 4e-3)))
    flare_t, flare_f = segs[0]
    flare_f = flare_f.copy()
    flare_f[200] *= 1.05
    segs[0] = (flare_t, flare_f)
    out = process_star(segs, {"K1.01": PLANET, "K1.02": other}, [PLANET, other], PARAMS)
    assert set(out) == {"K1.01", "K1.02"}
    for v in out.values():
        assert v.local_view.min() == pytest.approx(-1)
        assert np.isfinite(v.global_view).all() and v.global_view.max() < 1


def test_process_star_with_no_usable_data():
    assert (
        process_star([(np.arange(5.0), np.full(5, np.nan))], {"K": PLANET}, [PLANET], PARAMS) == {}
    )


def eclipsing_binary(odd_depth, even_depth, secondary_depth, seed=0):
    """Detrended-like flux: primary eclipses alternating in depth, plus a secondary at 0.5."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 160, CADENCE)
    flux = 1 + rng.normal(0, 1e-4, len(t))
    in_primary = np.abs(phase(t, PLANET.period, PLANET.epoch)) < PLANET.duration / 2
    n = transit_number(t, PLANET.period, PLANET.epoch)
    flux -= np.where(in_primary & (n % 2 == 1), odd_depth, 0.0)
    flux -= np.where(in_primary & (n % 2 == 0), even_depth, 0.0)
    sec = np.abs(phase(t, PLANET.period, PLANET.epoch + PLANET.period / 2)) < PLANET.duration / 2
    flux -= np.where(sec, secondary_depth, 0.0)
    return t, flux


def make(t, f):
    return make_views(t, f, PLANET, 201, 61, 2.0, 0.16)


def test_planet_has_matching_odd_even_and_no_secondary():
    v = make(*eclipsing_binary(DEPTH, DEPTH, 0.0))
    assert v.odd_view.min() == pytest.approx(-1, abs=0.1)
    assert v.even_view.min() == pytest.approx(-1, abs=0.1)
    assert abs(v.secondary_view.min()) < 0.15
    assert v.odd_empty == v.even_empty == v.secondary_empty == 0


def test_odd_even_depth_difference_is_preserved():
    """Normalised alone both would reach -1; on the shared scale the difference survives."""
    v = make(*eclipsing_binary(2 * DEPTH, DEPTH, 0.0))
    assert v.local_view.min() == pytest.approx(-1)
    # The 2:1 depth ratio survives (each view alone would normalise to -1).
    assert v.odd_view.min() / v.even_view.min() == pytest.approx(2, rel=0.15)
    assert v.even_view.min() > -0.7


def test_secondary_eclipse_depth_relative_to_primary():
    v = make(*eclipsing_binary(DEPTH, DEPTH, DEPTH / 2))
    assert v.secondary_view.min() == pytest.approx(-0.5, abs=0.1)
    assert abs(np.argmin(v.secondary_view) - 30) <= 8


def test_single_transit_leaves_even_view_flat():
    t = np.arange(0.0, 3.0, CADENCE)  # only transit 0 (even) observed
    flux = 1 + box(t, PLANET, DEPTH)  # box() is already negative in transit
    v = make(t, flux)
    assert v.odd_empty == 1 and (v.odd_view == 0).all()
    assert v.even_view.min() == pytest.approx(-1, abs=0.1)


def test_scale_like_uses_reference_scale():
    ref = np.array([0.0, 0.0, -2.0, 0.0, 0.0])
    np.testing.assert_allclose(scale_like(np.array([0.0, -1.0, 0.0]), ref), [0, -0.5, 0])
    assert (scale_like(np.full(3, np.nan), ref) == 0).all()


def test_view_transit_snr_detects_a_wrong_period():
    """Folding on the catalogue period stacks the transits; a 2% period error smears them
    across phase over ~48 orbits, so the centre of the local view carries no dip."""
    time, flux = stitch(quarters())
    mask = transit_mask(time, [PLANET], 2.0)
    t, f = detrend(time, flux, mask, window=1.0, gap=0.75)
    right = make_views(t, f, PLANET, 201, 61, 2.0, 0.16)
    wrong_eph = Ephemeris(PLANET.period * 1.02, PLANET.epoch, PLANET.duration)
    wrong = make_views(t, f, wrong_eph, 201, 61, 2.0, 0.16)
    assert view_transit_snr(right.local_view, 2.0) > 10
    assert view_transit_snr(wrong.local_view, 2.0) < 3
