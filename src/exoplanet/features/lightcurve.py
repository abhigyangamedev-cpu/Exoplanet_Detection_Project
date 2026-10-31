"""Light-curve detrending, phase folding and binned global/local views.

Follows the view construction of Shallue & Vanderburg (2018): a global view of the whole
orbit and a local view zoomed on the transit, each median-binned and normalised so that the
out-of-transit level is 0 and the transit minimum is -1. Times are in days (BKJD for Kepler)
and all functions work on plain NumPy arrays so they can be tested without downloading data.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

# Robust standard deviation from the median absolute deviation (Gaussian noise).
MAD_TO_STD = 1.4826


@dataclass(frozen=True)
class Ephemeris:
    period: float  # days
    epoch: float  # mid-transit time, same time system as the light curve
    duration: float  # days


def stitch(segments: Iterable[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate (time, flux) segments (e.g. Kepler quarters), dropping non-finite points and
    dividing each segment by its median so quarter-to-quarter offsets vanish.
    """
    times, fluxes = [], []
    for time, flux in segments:
        time, flux = np.asarray(time, float), np.asarray(flux, float)
        ok = np.isfinite(time) & np.isfinite(flux)
        if not ok.any():
            continue
        median = np.median(flux[ok])
        if median <= 0:
            continue
        times.append(time[ok])
        fluxes.append(flux[ok] / median)
    if not times:
        return np.empty(0), np.empty(0)
    time, flux = np.concatenate(times), np.concatenate(fluxes)
    order = np.argsort(time, kind="stable")
    return time[order], flux[order]


def phase(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """Time from the nearest transit centre, in days, in ``[-period / 2, period / 2)``."""
    return np.mod(time - epoch + period / 2, period) - period / 2


def transit_mask(
    time: np.ndarray, ephemerides: Iterable[Ephemeris], width_factor: float
) -> np.ndarray:
    """True for points within ``width_factor * duration / 2`` of any listed transit centre."""
    mask = np.zeros(len(time), dtype=bool)
    for eph in ephemerides:
        mask |= np.abs(phase(time, eph.period, eph.epoch)) < width_factor * eph.duration / 2
    return mask


def split_segments(time: np.ndarray, gap: float) -> list[slice]:
    """Slices of contiguous data, breaking wherever consecutive points are > ``gap`` apart."""
    if len(time) == 0:
        return []
    breaks = np.flatnonzero(np.diff(time) > gap) + 1
    edges = [0, *breaks.tolist(), len(time)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def detrend(
    time: np.ndarray,
    flux: np.ndarray,
    mask: np.ndarray,
    window: float,
    gap: float,
    min_points: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Divide out stellar and instrumental variability with a moving median.

    Masked (in-transit) points are excluded from the trend by interpolating across them, so
    transits are not filtered away. Each contiguous segment is filtered separately, with a
    window of ``window`` days converted to cadences. Segments with fewer than ``min_points``
    unmasked points are dropped. Returns the retained ``(time, detrended_flux)``.
    """
    keep_t, keep_f = [], []
    for seg in split_segments(time, gap):
        t, f, m = time[seg], flux[seg], mask[seg]
        if (~m).sum() < min_points:
            continue
        cadence = np.median(np.diff(t))
        size = max(3, int(round(window / cadence)) | 1)  # odd window, in cadences
        filled = np.interp(t, t[~m], f[~m])
        trend = median_filter(filled, size=size, mode="nearest")
        keep_t.append(t)
        keep_f.append(f / trend)
    if not keep_t:
        return np.empty(0), np.empty(0)
    return np.concatenate(keep_t), np.concatenate(keep_f)


def clip_upward_outliers(flux: np.ndarray, mask: np.ndarray, n_sigma: float) -> np.ndarray:
    """Boolean keep-mask removing points more than ``n_sigma`` above the out-of-transit median
    (flares, cosmic rays). Downward outliers are kept: transits are downward.
    """
    ref = flux[~mask] if (~mask).any() else flux
    median = np.median(ref)
    std = MAD_TO_STD * np.median(np.abs(ref - median))
    if std == 0:
        return np.ones(len(flux), dtype=bool)
    return flux - median <= n_sigma * std


def bin_median(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
    lo: float,
    hi: float,
    width: float | None = None,
) -> tuple[np.ndarray, float]:
    """Median of ``y`` in ``n_bins`` evenly spaced bins over ``[lo, hi)``.

    Each bin is centred on its slot and spans ``width`` (default: the bin spacing, i.e.
    adjacent non-overlapping bins); wider bins overlap and smooth noise. Empty bins are
    filled by linear interpolation. Returns ``(view, empty_fraction)``; the view is all-NaN
    if every bin is empty.
    """
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    spacing = (hi - lo) / n_bins
    half = (width if width is not None else spacing) / 2
    centres = lo + (np.arange(n_bins) + 0.5) * spacing
    left = np.searchsorted(x, np.maximum(centres - half, lo), side="left")
    right = np.searchsorted(x, np.minimum(centres + half, hi), side="left")
    view = np.array(
        [np.median(y[a:b]) if b > a else np.nan for a, b in zip(left, right, strict=True)]
    )
    filled = np.isfinite(view)
    empty_fraction = 1 - filled.mean()
    if filled.any() and not filled.all():
        slots = np.arange(n_bins)
        view = np.interp(slots, slots[filled], view[filled])
    return view, float(empty_fraction)


def normalise_view(view: np.ndarray) -> np.ndarray:
    """Shift and scale so the median is 0 and the minimum is -1 (if there is any dip)."""
    centred = view - np.median(view)
    depth = -centred.min()
    return centred / depth if depth > 0 else centred


def scale_like(view: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Apply ``reference``'s normalisation (its median and dip depth) to ``view``.

    Odd/even and secondary views must share the primary's scale: normalised on their own,
    every dip would reach -1 and depth differences (the binary signature) would vanish.
    An all-NaN view (no data, e.g. a single observed transit) becomes flat zeros.
    """
    if not np.isfinite(view).any():
        return np.zeros_like(view)
    offset = np.median(reference)
    depth = offset - reference.min()
    return (view - offset) / depth if depth > 0 else view - offset


def transit_number(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """Index of the nearest transit (0 at ``epoch``)."""
    return np.round((time - epoch) / period).astype(int)


@dataclass(frozen=True)
class Views:
    global_view: np.ndarray
    local_view: np.ndarray
    odd_view: np.ndarray  # local view of odd-numbered transits, on the local view's scale
    even_view: np.ndarray  # local view of even-numbered transits, same scale
    secondary_view: np.ndarray  # local-width view at phase 0.5, same scale
    global_empty: float  # fraction of global bins with no data before interpolation
    local_empty: float
    odd_empty: float
    even_empty: float
    secondary_empty: float
    n_in_transit: int  # points within half a duration of mid-transit


def make_views(
    time: np.ndarray,
    flux: np.ndarray,
    eph: Ephemeris,
    global_bins: int,
    local_bins: int,
    local_durations: float,
    local_bin_width: float | None = None,
) -> Views:
    """Global view over the full orbit, local view over ``±local_durations`` durations, and
    odd-transit, even-transit and secondary-eclipse (phase 0.5) views on the local grid.

    ``local_bin_width`` (in transit durations) widens the local bins so they overlap; Kepler's
    30-minute cadence is coarser than the local bin spacing for short transits, so narrow
    bins mostly add noise. The odd, even and secondary views share the local view's
    normalisation, so their depths are directly comparable with the primary transit.
    """
    ph = phase(time, eph.period, eph.epoch)
    half = min(local_durations * eph.duration, eph.period / 2)
    width = local_bin_width * eph.duration if local_bin_width else None
    global_view, global_empty = bin_median(ph, flux, global_bins, -eph.period / 2, eph.period / 2)
    local_raw, local_empty = bin_median(ph, flux, local_bins, -half, half, width)

    odd = transit_number(time, eph.period, eph.epoch) % 2 == 1
    odd_raw, odd_empty = bin_median(ph[odd], flux[odd], local_bins, -half, half, width)
    even_raw, even_empty = bin_median(ph[~odd], flux[~odd], local_bins, -half, half, width)
    sec_ph = phase(time, eph.period, eph.epoch + eph.period / 2)
    sec_raw, sec_empty = bin_median(sec_ph, flux, local_bins, -half, half, width)
    return Views(
        global_view=normalise_view(global_view),
        local_view=normalise_view(local_raw),
        odd_view=scale_like(odd_raw, local_raw),
        even_view=scale_like(even_raw, local_raw),
        secondary_view=scale_like(sec_raw, local_raw),
        global_empty=global_empty,
        local_empty=local_empty,
        odd_empty=odd_empty,
        even_empty=even_empty,
        secondary_empty=sec_empty,
        n_in_transit=int((np.abs(ph) < eph.duration / 2).sum()),
    )


def process_star(
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    targets: dict[str, Ephemeris],
    mask_ephemerides: Iterable[Ephemeris],
    params: dict,
) -> dict[str, Views]:
    """Detrend one star's light curve once, then build views for each target KOI.

    ``mask_ephemerides`` should cover every known KOI on the star (including unvetted ones)
    so that no transit leaks into the trend. ``params`` holds the ``lightcurves`` and
    ``views`` settings merged: ``mask_width``, ``window_days``, ``window_durations``,
    ``gap_days``, ``clip_sigma``, ``global_bins``, ``local_bins``, ``local_durations``
    and optionally ``local_bin_width``.
    """
    mask_ephemerides = list(mask_ephemerides)
    time, flux = stitch(segments)
    mask = transit_mask(time, mask_ephemerides, params["mask_width"])
    longest = max((e.duration for e in mask_ephemerides), default=0.0)
    window = max(params["window_days"], params["window_durations"] * longest)
    time, flux = detrend(time, flux, mask, window, params["gap_days"])
    if len(time) == 0:
        return {}

    mask = transit_mask(time, mask_ephemerides, params["mask_width"])
    keep = clip_upward_outliers(flux, mask, params["clip_sigma"])
    time, flux = time[keep], flux[keep]
    return {
        koi: make_views(
            time,
            flux,
            eph,
            params["global_bins"],
            params["local_bins"],
            params["local_durations"],
            params.get("local_bin_width"),
        )
        for koi, eph in targets.items()
    }
