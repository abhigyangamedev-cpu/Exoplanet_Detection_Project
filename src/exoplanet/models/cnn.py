"""Two-tower 1D CNN on global and local light-curve views (after Shallue & Vanderburg 2018).

Each tower is a stack of conv-conv-maxpool blocks; their flattened outputs are concatenated
and passed through a small fully connected head to a single logit. Pooling keeps positional
information, so features at specific phases (e.g. a secondary eclipse at phase 0.5) remain
distinguishable. Training runs for a fixed number of epochs: the validation fold is scored
every epoch for monitoring only and never used to stop or select a model, so out-of-fold
scores are not optimistically biased. Training uses development folds only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from exoplanet.data.labels import LABEL
from exoplanet.data.splits import FOLD, KOI, SPLIT, TEST, cv_folds
from exoplanet.evaluation.metrics import SCORE, fold_metrics

GLOBAL, LOCAL, SCALARS = "global_view", "local_view", "scalars"


def conv_tower(
    length: int, channels: list[int], kernel: int, pool: int, pool_stride: int
) -> tuple[nn.Sequential, int]:
    """Conv blocks for a 1-channel input of ``length``; returns the tower and its flat size."""
    layers: list[nn.Module] = []
    in_ch = 1
    for out_ch in channels:
        layers += [
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.MaxPool1d(pool, stride=pool_stride),
        ]
        in_ch = out_ch
    tower = nn.Sequential(*layers, nn.Flatten())
    with torch.no_grad():
        flat = tower(torch.zeros(1, 1, length)).shape[1]
    return tower, flat


class TwoTowerCNN(nn.Module):
    """Two conv towers plus, optionally, ``n_scalars`` per-KOI scalar inputs fed to the head.

    Scalars are standardised with training-set statistics held in buffers (so they are saved
    with the weights); missing values become 0, the training mean.
    """

    def __init__(
        self, global_len: int, local_len: int, arch: dict[str, Any], n_scalars: int = 0
    ) -> None:
        super().__init__()
        self.global_tower, g_flat = conv_tower(global_len, **arch["global"])
        self.local_tower, l_flat = conv_tower(local_len, **arch["local"])
        self.register_buffer("scalar_mean", torch.zeros(n_scalars))
        self.register_buffer("scalar_std", torch.ones(n_scalars))
        head: list[nn.Module] = []
        width = g_flat + l_flat + n_scalars
        for units in arch["fc"]:
            head += [nn.Linear(width, units), nn.ReLU(), nn.Dropout(arch["dropout"])]
            width = units
        head.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*head)

    def forward(
        self, global_view: torch.Tensor, local_view: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        scaled = torch.nan_to_num((scalars - self.scalar_mean) / self.scalar_std, nan=0.0)
        features = torch.cat(
            [self.global_tower(global_view), self.local_tower(local_view), scaled], 1
        )
        return self.head(features).squeeze(1)


def attach_scalars(views: pd.DataFrame, sample: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add a ``scalars`` column: log10 of the given catalogue columns per KOI (non-positive or
    missing values become NaN). ``sample`` must be indexed by ``kepoi_name``.
    """
    values = sample.loc[views.index, columns].apply(pd.to_numeric, errors="coerce")
    logged = np.log10(values.where(values > 0)).to_numpy(np.float32)
    return views.assign(**{SCALARS: list(logged)})


def view_tensors(views: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(n, 1, length) view tensors and an (n, n_scalars) scalar tensor (width 0 if none)."""
    g = np.stack(views[GLOBAL].to_numpy()).astype(np.float32)[:, None, :]
    l_ = np.stack(views[LOCAL].to_numpy()).astype(np.float32)[:, None, :]
    if SCALARS in views.columns and len(views):
        s = np.stack(views[SCALARS].to_numpy()).astype(np.float32)
    else:
        s = np.zeros((len(views), 0), np.float32)
    return torch.from_numpy(g), torch.from_numpy(l_), torch.from_numpy(s)


@torch.no_grad()
def predict(model: TwoTowerCNN, views: pd.DataFrame, batch_size: int = 512) -> np.ndarray:
    """Positive-class probabilities, in the row order of ``views``."""
    model.eval()
    g, l_, s = view_tensors(views)
    out = [
        torch.sigmoid(model(g[i : i + batch_size], l_[i : i + batch_size], s[i : i + batch_size]))
        for i in range(0, len(g), batch_size)
    ]
    return torch.cat(out).numpy() if out else np.empty(0)


def train(
    views: pd.DataFrame,
    labels: pd.Series,
    params: dict[str, Any],
    seed: int,
    monitor: tuple[pd.DataFrame, pd.Series] | None = None,
) -> tuple[TwoTowerCNN, list[float]]:
    """Train for ``params["epochs"]`` epochs. Returns the model and, if ``monitor`` views and
    labels are given, their PR-AUC after each epoch (for learning curves only).
    """
    torch.manual_seed(seed)
    g, l_, s = view_tensors(views)
    y = torch.as_tensor(labels.to_numpy(), dtype=torch.float32)
    model = TwoTowerCNN(g.shape[2], l_.shape[2], params["arch"], n_scalars=s.shape[1])
    if s.shape[1]:
        # Standardise scalars with training-set statistics only.
        model.scalar_mean.copy_(torch.from_numpy(np.nanmean(s.numpy(), axis=0)))
        model.scalar_std.copy_(torch.from_numpy(np.nanstd(s.numpy(), axis=0)).clamp_min(1e-6))
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    curve = []
    for _ in range(params["epochs"]):
        model.train()
        order = torch.randperm(len(y), generator=generator)
        for start in range(0, len(y), params["batch_size"]):
            idx = order[start : start + params["batch_size"]]
            gb, lb = g[idx], l_[idx]
            if params["flip_augment"]:
                # Views are symmetric about mid-transit, so time reversal is label-preserving.
                flip = torch.rand(len(idx), generator=generator) < 0.5
                gb = torch.where(flip[:, None, None], gb.flip(2), gb)
                lb = torch.where(flip[:, None, None], lb.flip(2), lb)
            optimiser.zero_grad()
            loss_fn(model(gb, lb, s[idx]), y[idx]).backward()
            optimiser.step()
        if monitor is not None:
            curve.append(average_precision_score(monitor[1], predict(model, monitor[0])))
    return model, curve


def _dev_views(views: pd.DataFrame, splits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Development split rows restricted to KOIs that have views, and those views."""
    if views.index.name != KOI:
        raise ValueError(f"views must be indexed by {KOI}")
    dev = splits[(splits[SPLIT] != TEST) & splits[KOI].isin(views.index)]
    return dev, views.loc[dev[KOI]]


def cross_validate(
    views: pd.DataFrame, splits: pd.DataFrame, params: dict[str, Any], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One model per development fold, scored on its validation fold.

    KOIs without views are left out (they are reported as coverage, not imputed). Returns
    ``(oof, fold_metrics, curves)``: out-of-fold scores, per-fold PR-AUC/ROC-AUC, and
    validation PR-AUC per epoch (rows: epochs, columns: folds).
    """
    dev, dev_views = _dev_views(views, splits)
    labels = dev.set_index(KOI)[LABEL]
    oof, metrics, curves = [], [], {}
    for fold, train_kois, val_kois in cv_folds(dev):
        model, curves[fold] = train(
            dev_views.loc[train_kois],
            labels.loc[train_kois],
            params,
            seed + fold,
            monitor=(dev_views.loc[val_kois], labels.loc[val_kois]),
        )
        score = predict(model, dev_views.loc[val_kois])
        y_val = labels.loc[val_kois]
        oof.append(pd.DataFrame({FOLD: fold, LABEL: y_val, SCORE: score}, index=val_kois))
        metrics.append(fold_metrics(fold, y_val.to_numpy(), score))
    curves = pd.DataFrame(curves).rename_axis("epoch")
    curves.index += 1
    return pd.concat(oof).rename_axis(KOI), pd.DataFrame(metrics).set_index(FOLD), curves


def fit_final(
    views: pd.DataFrame, splits: pd.DataFrame, params: dict[str, Any], seed: int
) -> TwoTowerCNN:
    """Train on all development KOIs with views, for evaluation on the held-out test set."""
    dev, dev_views = _dev_views(views, splits)
    model, _ = train(dev_views, dev.set_index(KOI)[LABEL], params, seed)
    return model
