"""Scene modules: scene id ``NN_name`` lives in ``scenes/sNN_name.py``.

Each module exposes ``build(ctx: engine.Context) -> Callable[[Figure, float], None]``.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def import_scene(scene_id: str) -> ModuleType:
    """Return the module ``scenes.s<scene_id>``; ImportError if it does not exist."""
    return importlib.import_module(f"{__name__}.s{scene_id}")


UNMATCHED: dict[str, list[str]] = {}   # scene id -> phrases cue_fraction did not find


def cue_fraction(ctx, phrase: str, default: float) -> float:
    """Fraction of the scene at which ``phrase`` is spoken, estimated from ``ctx.cues``.

    The first cue containing ``phrase`` is used; the time is interpolated by character
    position inside that cue. Times are divided by ``max(ctx.duration, end of last cue)``,
    so a duration override shorter than the narration scales the timing down rather than
    pushing it past the end. Returns ``default`` when no cue contains the phrase; when the
    scene has cues, the phrase is also recorded in ``UNMATCHED[ctx.scene_id]`` (a script
    edit reworded it), which the build reports.
    """
    if not ctx.cues:
        return default
    total = max(ctx.duration, ctx.cues[-1][1])
    for start, end, text in ctx.cues:
        pos = text.find(phrase)
        if pos >= 0:
            return (start + (end - start) * pos / len(text)) / total
    # a phrase split across two cues: time it at its first word, in the earlier cue
    joined, offsets = "", []
    for start, end, text in ctx.cues:
        offsets.append(len(joined))
        joined += text + " "
    pos = joined.find(phrase)
    if pos >= 0:
        i = max(k for k, off in enumerate(offsets) if off <= pos)
        start, end, text = ctx.cues[i]
        return (start + (end - start) * (pos - offsets[i]) / len(text)) / total
    missing = UNMATCHED.setdefault(ctx.scene_id, [])
    if phrase not in missing:
        missing.append(phrase)
    return default


def unmatched_phrases(ctx, build) -> list[str]:
    """The phrases a scene's build(ctx) looked for and did not find in its captions."""
    UNMATCHED.pop(ctx.scene_id, None)
    build(ctx)
    return list(UNMATCHED.get(ctx.scene_id, []))
