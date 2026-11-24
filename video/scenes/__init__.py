"""Scene modules: scene id ``NN_name`` lives in ``scenes/sNN_name.py``.

Each module exposes ``build(ctx: engine.Context) -> Callable[[Figure, float], None]``.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def import_scene(scene_id: str) -> ModuleType:
    """Return the module ``scenes.s<scene_id>``; ImportError if it does not exist."""
    return importlib.import_module(f"{__name__}.s{scene_id}")
