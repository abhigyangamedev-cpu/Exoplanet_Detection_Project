"""Scene 6, the two models, as peers: the neural network on top, the decision trees below.

Top row: the real global and local views of K00020.01 (the two pictures from scene 5) enter
two separate towers of 1D convolutions (src/exoplanet/models/cnn.py, TwoTowerCNN); the
towers merge into dense layers and one score from 0 to 1 comes out. No layer count is shown
and no score value: the score is drawn as a bare 0-1 scale.

Bottom row: gradient boosting (src/exoplanet/models/tabular.py) as a row of small trees,
each one added after the last, fed by the three catalogue feature groups the project uses
(configs/default.yaml features.groups: transit, stellar, centroid), labelled in plain words.
Each part appears as it is narrated; small dots flow through a model once it is complete,
passing behind its labelled shapes, never over text. Each model keeps its colour from scene
8: style.C_NET for the network, style.C_TREES for the trees (the views stay C_PLANET).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import style
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from scenes import cue_fraction

# layout in inches of the 16 x 9 overlay axes (caption band: y < 1.35)
X0 = 0.8
CNN_HEAD_Y, TREE_HEAD_Y = 7.45, 4.05
THUMB_X = (X0, 3.6)
GLOBAL_Y, LOCAL_Y = (5.85, 6.85), (4.45, 5.45)        # thumbnail boxes (y0, y1)
TOWER_X = (4.1, 6.4)
MERGE_BOX = (7.0, 8.4)
DENSE_BOX = (8.9, 10.8)
NODE_H = 0.8
CNN_MID = (GLOBAL_Y[0] + LOCAL_Y[1]) / 2               # where the towers merge
GAUGE_X = (11.9, 15.0)
GAUGE_H = 0.34
FEATURE_X = (X0, 5.55)
FEATURE_YS = (3.2, 2.55, 1.9)
FEATURE_H = 0.5
TREE_MID = FEATURE_YS[1]
TREE_DX = 0.95
TREE_XS = tuple(6.5 + TREE_DX * i for i in range(5))
TREE_W, TREE_H = 0.72, 1.05
N_PULSES = 4

# the three catalogue feature groups: (project term, plain words, narrated phrase)
GROUPS = (("transit", "the dip: depth, length, ...", "depth and length"),
          ("stellar", "the star itself", "the star itself"),
          ("centroid", "image shift during the dip", "image shifts"))

_GAUGE_CMAP = LinearSegmentedColormap.from_list("score", [style.C_FALSE, style.C_PLANET])


@lru_cache(maxsize=64)
def text_width(text: str, size: float, mono: bool = False, weight: str = "normal") -> float:
    """Width in inches of text drawn at size points."""
    prop = FontProperties(family="monospace" if mono else "sans-serif", size=size,
                          weight=weight)
    return RendererAgg(8, 8, 72).get_text_width_height_descent(text, prop, ismath=False)[0] / 72


def polyline_point(points: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Points at fractions s (0-1) of the way along a polyline, by arc length."""
    seg = np.hypot(*np.diff(points, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)]) / seg.sum()
    s = np.clip(np.asarray(s, float), 0.0, 1.0)
    return np.column_stack([np.interp(s, cum, points[:, 0]), np.interp(s, cum, points[:, 1])])


def _box(ax, x, y, w, h, alpha, edge=None, face=None, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor=face or style.TERM_BG, edgecolor=edge or style.MUTED,
                                linewidth=lw, alpha=alpha))


def _arrow(ax, a, b, alpha):
    if alpha > 0:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.6,
                                     color=style.MUTED, alpha=alpha, shrinkA=0, shrinkB=0))


def _tree_segments(x, y_mid, variant):
    """Edges and nodes of a small depth-2 tree centred at x; variant drops some leaves."""
    top, mid, low = y_mid + TREE_H / 2, y_mid, y_mid - TREE_H / 2
    dx1, dx2 = TREE_W * 0.3, TREE_W * 0.17
    nodes = [(x, top), (x - dx1, mid), (x + dx1, mid)]
    leaves = [(x - dx1 - dx2, low), (x - dx1 + dx2, low), (x + dx1 - dx2, low),
              (x + dx1 + dx2, low)]
    keep = [i for i in range(4) if i != variant % 5]          # variant 4 keeps all four
    edges = [((x, top), nodes[1]), ((x, top), nodes[2])]
    for i in keep:
        edges.append((nodes[1 + i // 2], leaves[i]))
    return edges, nodes, [leaves[i] for i in keep]


def build(ctx):
    star = ctx.load("star_planet.npz")
    views = {"global": np.asarray(star["global_view"], float),
             "local": np.asarray(star["local_view"], float)}

    # timing, tied to the spoken words
    f_nn = cue_fraction(ctx, "A neural network", 0.1)
    f_pics = cue_fraction(ctx, "two pictures", 0.17)
    f_score = cue_fraction(ctx, "gives", 0.25)
    f_zero = cue_fraction(ctx, "zero to one", 0.3)
    f_team = cue_fraction(ctx, "A team", 0.36)
    f_fix = cue_fraction(ctx, "each fixing", 0.45)
    f_reads = cue_fraction(ctx, "reads catalogue", 0.58)
    f_groups = [cue_fraction(ctx, phrase, d) for (_, _, phrase), d in
                zip(GROUPS, (0.66, 0.74, 0.86), strict=True)]
    f_tree_out = f_reads + 0.03                  # the trees' score, as they read numbers
    # the network is built between "two pictures" and "gives a score"
    step = max(f_score - f_pics, 0.08) / 3
    f_tow, f_merge, f_dense = f_pics + 0.3 * step, f_pics + 1.2 * step, f_pics + 2.1 * step

    # pulse paths (ambient flow once each model is complete)
    g_mid, l_mid = sum(GLOBAL_Y) / 2, sum(LOCAL_Y) / 2
    cnn_paths = [np.array([[THUMB_X[1], y], [TOWER_X[1], y], [MERGE_BOX[0], CNN_MID],
                           [GAUGE_X[0] - 0.15, CNN_MID]]) for y in (g_mid, l_mid)]
    tree_paths = [np.array([[FEATURE_X[1], y], [TREE_XS[0] - 0.55, TREE_MID],
                            [GAUGE_X[0] - 0.15, TREE_MID]]) for y in FEATURE_YS]
    # (x0, x1, y0, y1) where the dots are not drawn: the towers (their "convolutions" label)
    # and the trees' "..." label
    cnn_hide = [(TOWER_X[0] - 0.05, TOWER_X[1] + 0.05, y0, y1) for y0, y1 in (GLOBAL_Y, LOCAL_Y)]
    cnn_hide += [(x0, x1, CNN_MID - NODE_H / 2, CNN_MID + NODE_H / 2)     # behind the boxes
                 for x0, x1 in (MERGE_BOX, DENSE_BOX)]
    tree_hide = [(TREE_XS[-1] + 0.35, TREE_XS[-1] + 0.95, TREE_MID - 0.4, TREE_MID + 0.4)]

    def a_at(f, start, dur=0.03):
        return style.ease(style.progress(f, start, start + dur))

    def draw(fig, t):
        f = t / ctx.duration
        style.starfield(fig.add_axes((0, 0, 1, 1)), 6, t)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.set_axis_off()
        style.heading(fig, "Two kinds of model")
        _draw_cnn(ax, f, t)
        _draw_trees(ax, f, t)

    def _header(ax, y, title, sub, alpha, colour):
        ax.text(X0, y, title, ha="left", va="center", size=style.BODY_SIZE,
                weight="semibold", color=colour, alpha=alpha)
        x_sub = X0 + text_width(title, style.BODY_SIZE, weight="semibold") + 0.25
        ax.text(x_sub, y, sub, ha="left", va="center", size=style.LABEL_SIZE,
                color=style.MUTED, alpha=alpha)

    def _draw_cnn(ax, f, t):
        a_head = a_at(f, f_nn - 0.01)
        if a_head <= 0:
            return
        _header(ax, CNN_HEAD_Y, "Neural network", "looks at the two pictures", a_head,
                style.C_NET)
        a_pic = a_at(f, f_pics - 0.02, 0.04)
        for (y0, y1), key, plain in ((GLOBAL_Y, "global", "whole orbit"),
                                     (LOCAL_Y, "local", "close-up of the transit")):
            w = THUMB_X[1] - THUMB_X[0]
            _box(ax, THUMB_X[0], y0, w, y1 - y0, a_pic)
            v = views[key]
            x = THUMB_X[0] + 0.1 + np.linspace(0, w - 0.2, len(v))
            y = y1 - 0.38 + (v / -v.min()) * (y1 - y0 - 0.52)
            ax.plot(x, y, color=style.C_PLANET, lw=1.8, alpha=a_pic)
            ax.text(THUMB_X[0], y1 + 0.13, plain, ha="left", va="center",
                    size=style.LABEL_SIZE - 4, color=style.FG, alpha=a_pic)
            ax.text(THUMB_X[1] - 0.08, y1 - 0.15, f"{key} view", ha="right", va="center",
                    family="monospace", size=style.LABEL_SIZE - 6, color=style.MUTED,
                    alpha=a_pic)
        # two towers of 1D convolutions: tapering shapes, no layer count implied
        a_tow = a_at(f, f_tow, 0.04)
        for y0, y1 in (GLOBAL_Y, LOCAL_Y):
            mid, h = (y0 + y1) / 2, y1 - y0
            _arrow(ax, (THUMB_X[1] + 0.05, mid), (TOWER_X[0] - 0.05, mid), a_tow)
            ax.add_patch(Polygon([(TOWER_X[0], mid - h / 2), (TOWER_X[1], mid - h / 4.5),
                                  (TOWER_X[1], mid + h / 4.5), (TOWER_X[0], mid + h / 2)],
                                 closed=True, facecolor=style.C_NET, alpha=0.16 * a_tow,
                                 edgecolor="none"))
            ax.add_patch(Polygon([(TOWER_X[0], mid - h / 2), (TOWER_X[1], mid - h / 4.5),
                                  (TOWER_X[1], mid + h / 4.5), (TOWER_X[0], mid + h / 2)],
                                 closed=True, facecolor="none", edgecolor=style.C_NET,
                                 lw=1.6, alpha=a_tow))
            ax.text((TOWER_X[0] + TOWER_X[1]) / 2 - 0.1, mid, "convolutions", ha="center",
                    va="center", size=style.LABEL_SIZE - 5, color=style.FG, alpha=a_tow)
        a_merge = a_at(f, f_merge, 0.04)
        for y0, y1 in (GLOBAL_Y, LOCAL_Y):
            _arrow(ax, (TOWER_X[1] + 0.05, (y0 + y1) / 2),
                   (MERGE_BOX[0] - 0.02, CNN_MID + (0.2 if y0 > CNN_MID else -0.2)), a_merge)
        for (x0, x1), label, a in ((MERGE_BOX, "merged", a_merge),
                                   (DENSE_BOX, "dense layers", a_at(f, f_dense, 0.04))):
            _box(ax, x0, CNN_MID - NODE_H / 2, x1 - x0, NODE_H, a)
            ax.text((x0 + x1) / 2, CNN_MID, label, ha="center", va="center",
                    size=style.LABEL_SIZE - 2, color=style.FG, alpha=a)
        _arrow(ax, (MERGE_BOX[1] + 0.04, CNN_MID), (DENSE_BOX[0] - 0.04, CNN_MID),
               a_at(f, f_dense, 0.04))
        a_gauge = a_at(f, f_score, 0.04)
        _arrow(ax, (DENSE_BOX[1] + 0.04, CNN_MID), (GAUGE_X[0] - 0.12, CNN_MID), a_gauge)
        _gauge(ax, CNN_MID, a_gauge, a_at(f, f_zero - 0.01))
        if f >= f_score + 0.04:
            _pulses(ax, cnn_paths, t, a_at(f, f_score + 0.04, 0.04), cnn_hide)

    def _draw_trees(ax, f, t):
        a_head = a_at(f, f_team - 0.01)
        if a_head <= 0:
            return
        _header(ax, TREE_HEAD_Y, "Decision trees", "a team of small trees", a_head,
                style.C_TREES)
        span = max(f_fix - f_team, 0.04)
        for i, x in enumerate(TREE_XS):
            a = a_at(f, f_team + span * i / len(TREE_XS), 0.025)
            if a <= 0:
                continue
            edges, nodes, leaves = _tree_segments(x, TREE_MID, i + 2)
            for p, q in edges:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=style.C_TREES, lw=1.6,
                        alpha=0.8 * a)
            ax.scatter(*np.transpose(nodes), s=70, color=style.C_TREES, alpha=a, linewidths=0,
                       zorder=3)
            ax.scatter(*np.transpose(leaves), s=45, color=style.C_TREES, alpha=a, linewidths=0,
                       zorder=3)
            if i:
                _arrow(ax, (x - TREE_DX + TREE_W / 2 + 0.02, TREE_MID),
                       (x - TREE_W / 2 - 0.02, TREE_MID), a * a_at(f, f_fix - 0.01))
        a_more = a_at(f, f_fix - 0.02)
        ax.text(TREE_XS[-1] + 0.45, TREE_MID, "...", ha="left", va="center",
                size=style.BODY_SIZE, color=style.FG, alpha=a_more)
        ax.text((TREE_XS[0] + TREE_XS[-1]) / 2, TREE_MID - TREE_H / 2 - 0.3,
                "each fixes the last one's mistakes", ha="center", va="center",
                size=style.LABEL_SIZE - 2, color=style.FG, alpha=a_at(f, f_fix))

        a_reads = a_at(f, f_reads - 0.01)
        if a_reads > 0:
            ax.text(FEATURE_X[0], FEATURE_YS[0] + FEATURE_H / 2 + 0.2, "catalogue numbers",
                    ha="left", va="center", size=style.LABEL_SIZE - 4, color=style.MUTED,
                    alpha=a_reads)
        for (term, plain, _), y, fg in zip(GROUPS, FEATURE_YS, f_groups, strict=True):
            a = a_at(f, fg - 0.015)
            if a <= 0:
                continue
            _box(ax, FEATURE_X[0], y - FEATURE_H / 2, FEATURE_X[1] - FEATURE_X[0], FEATURE_H,
                 a)
            ax.text(FEATURE_X[0] + 0.15, y, plain, ha="left", va="center",
                    size=style.LABEL_SIZE - 3, color=style.FG, alpha=a)
            ax.text(FEATURE_X[1] - 0.15, y, term, ha="right", va="center",
                    family="monospace", size=style.LABEL_SIZE - 7, color=style.MUTED,
                    alpha=a)
            _arrow(ax, (FEATURE_X[1] + 0.05, y), (TREE_XS[0] - 0.5, TREE_MID + (y - TREE_MID)
                                                  * 0.3), a)
        a_out = a_at(f, f_tree_out - 0.02, 0.04)
        _arrow(ax, (TREE_XS[-1] + 0.95, TREE_MID), (GAUGE_X[0] - 0.12, TREE_MID), a_out)
        _gauge(ax, TREE_MID, a_out, a_out)
        if f >= f_groups[-1] + 0.02:
            _pulses(ax, tree_paths, t, a_at(f, f_groups[-1] + 0.02), tree_hide)

    def _gauge(ax, y, alpha, a_ends):
        if alpha <= 0:
            return
        x0, x1 = GAUGE_X
        ax.imshow(np.linspace(0, 1, 64)[None, :], cmap=_GAUGE_CMAP, aspect="auto",
                  extent=(x0, x1, y - GAUGE_H / 2, y + GAUGE_H / 2), alpha=alpha,
                  interpolation="bilinear", zorder=2)
        ax.text((x0 + x1) / 2, y + GAUGE_H / 2 + 0.25, "score", ha="center", va="center",
                size=style.LABEL_SIZE, weight="semibold", color=style.FG, alpha=alpha)
        for x, label in ((x0, "0"), (x1, "1")):
            ax.text(x, y - GAUGE_H / 2 - 0.22, label, ha="center", va="center",
                    size=style.LABEL_SIZE, color=style.FG, alpha=a_ends)

    def _pulses(ax, paths, t, alpha, hide):
        s = (0.35 * t + np.arange(N_PULSES) / N_PULSES) % 1.0       # ambient, seconds
        pts = np.vstack([polyline_point(p, (s + 0.13 * k) % 1.0) for k, p in enumerate(paths)])
        for x0, x1, y0, y1 in hide:          # pass behind the labelled shapes, never over text
            inside = ((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1))
            pts = pts[~inside]
        ax.scatter(pts[:, 0], pts[:, 1], s=40, color=style.C_ACCENT, alpha=0.9 * alpha,
                   linewidths=0, zorder=0.5, gid="pulses")   # under boxes: seen on arrows

    return draw
