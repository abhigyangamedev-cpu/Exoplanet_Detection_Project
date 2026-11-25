"""Load the narration script (video/narration.md) and fill its number placeholders.

Script format: one ``## <scene_id>`` heading per scene, in SCENE_IDS order; the body below
it is the spoken text (line breaks are joined into one paragraph). HTML comments
(``<!-- ... -->``) are ignored, so notes can sit in the script without being spoken.

Placeholders ``{key}`` are filled from video/build/data/numbers.json, so spoken numbers
cannot drift from the data. Spoken-format convention (see ``format_value``):

- dotted keys reach into nested values: ``{test_pr_auc.tabular}``; an integer segment
  indexes a list: ``{test_pr_auc.cnn.1}`` is the lower 95% bound;
- integers get thousands separators: ``{n_kois}`` -> "6,677";
- floats get three decimals: ``{leak_lc_audited}`` -> "0.969";
- a list of numbers is ``[point, low, high]``: it is spoken as its point estimate,
  ``{test_pr_auc.tabular}`` -> "0.993";
- strings are spoken as written, with unit abbreviations expanded:
  ``{cnn_full_train_time_text}`` "3 min 23 s" -> "3 minutes 23 seconds".

An unknown key raises KeyError naming the full placeholder.
"""

from __future__ import annotations

import re
from pathlib import Path

SCENE_IDS = [
    "01_hook",
    "02_problem",
    "03_data",
    "04_leakage",
    "05_pipeline",
    "06_models",
    "07_integration",
    "08_results",
    "09_meaning",
    "10_end",
]

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][\w.]*)\}")
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HEADING = re.compile(r"^##\s+(\S+)\s*$", re.M)
_UNITS = [(re.compile(r"(\d)\s*min\b"), r"\1 minutes"), (re.compile(r"(\d)\s*s\b"), r"\1 seconds")]


def format_value(value) -> str:
    """Render one numbers.json value in spoken-friendly form (convention in module doc)."""
    if isinstance(value, bool):
        raise TypeError(f"cannot speak a boolean: {value!r}")
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple)) and value and all(
        isinstance(v, (int, float)) for v in value
    ):
        return format_value(value[0])
    if isinstance(value, str):
        for pattern, repl in _UNITS:
            value = pattern.sub(repl, value)
        return value
    raise TypeError(f"cannot speak value {value!r}")


def resolve(numbers: dict, key: str):
    """Look up a dotted key in numbers; KeyError names the key if any segment is missing."""
    node = numbers
    for part in key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, (list, tuple)) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            raise KeyError(f"unknown narration placeholder {{{key}}}: no such key {key!r}")
    return node


def fill(text: str, numbers: dict) -> str:
    return _PLACEHOLDER.sub(lambda m: format_value(resolve(numbers, m.group(1))), text)


def load_narration(path, numbers: dict) -> dict[str, str]:
    """Return {scene_id: spoken text} in script order, placeholders filled."""
    source = _COMMENT.sub("", Path(path).read_text(encoding="utf-8"))
    parts = _HEADING.split(source)
    scenes: dict[str, str] = {}
    for scene_id, body in zip(parts[1::2], parts[2::2], strict=True):
        if scene_id in scenes:
            raise ValueError(f"scene {scene_id} appears twice in {path}")
        scenes[scene_id] = fill(" ".join(body.split()), numbers)
    return scenes
