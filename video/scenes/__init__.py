"""Scene modules: scene id ``NN_name`` lives in ``scenes/sNN_name.py``.

Each module exposes ``build(ctx: engine.Context) -> Callable[[Figure, float], None]``.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def import_scene(scene_id: str) -> ModuleType:
    """Return the module ``scenes.s<scene_id>``; ImportError if it does not exist."""
    return importlib.import_module(f"{__name__}.s{scene_id}")


def cue_fraction(ctx, phrase: str, default: float) -> float:
    """Fraction of the scene at which ``phrase`` is spoken, estimated from ``ctx.cues``.

    The first cue containing ``phrase`` is used; the time is interpolated by character
    position inside that cue. Times are divided by ``max(ctx.duration, end of last cue)``,
    so a duration override shorter than the narration scales the timing down rather than
    pushing it past the end. Returns ``default`` when no cue contains the phrase.
    """
    if not ctx.cues:
        return default
    total = max(ctx.duration, ctx.cues[-1][1])
    for start, end, text in ctx.cues:
        pos = text.find(phrase)
        if pos >= 0:
            return (start + (end - start) * pos / len(text)) / total
    return default
