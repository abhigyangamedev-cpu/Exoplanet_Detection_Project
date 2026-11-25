"""Render the project's markdown reports (docs/*.md) as HTML for the website."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import mistune
from mistune.renderers.html import HTMLRenderer

REPORT_TABS = [("results", "Results"), ("evaluation", "Evaluation")]
_TAG = re.compile(r"<[^>]+>")
_NOT_SLUG = re.compile(r"[^a-z0-9]+")


class _Renderer(HTMLRenderer):
    """HTML with an id on every heading, collecting them as a table of contents."""

    def __init__(self) -> None:
        super().__init__(escape=True)
        self.toc: list[dict[str, Any]] = []
        self._used: set[str] = set()

    def heading(self, text: str, level: int, **attrs: Any) -> str:
        plain = html.unescape(_TAG.sub("", text)).strip()
        base = _NOT_SLUG.sub("-", plain.lower()).strip("-") or "section"
        slug, n = base, 2
        while slug in self._used:
            slug, n = f"{base}-{n}", n + 1
        self._used.add(slug)
        self.toc.append({"id": slug, "text": plain, "level": level})
        return f'<h{level} id="{slug}">{text}</h{level}>\n'


def doc_names(docs_dir: Path) -> set[str]:
    return {p.stem for p in docs_dir.glob("*.md")} if docs_dir.is_dir() else set()


def render_doc(docs_dir: Path, name: str) -> dict[str, Any] | None:
    """Render ``docs_dir/<name>.md``; None unless it is a markdown file directly in docs_dir."""
    if name not in doc_names(docs_dir):  # a whitelist of real files, so no path can escape
        return None
    renderer = _Renderer()
    markdown = mistune.create_markdown(renderer=renderer, plugins=["table", "strikethrough"])
    body = markdown((docs_dir / f"{name}.md").read_text(encoding="utf-8"))
    title = renderer.toc[0]["text"] if renderer.toc else name
    return {"title": title, "html": body, "toc": renderer.toc}
