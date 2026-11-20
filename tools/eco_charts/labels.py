"""Give a Match Room run of an eco game its chart replay's words for each decision.

tools/build_data.py calls annotate() on every run it builds. For a run that has a chart replay
(site/eco/data/<game>.js, or <game>__no_history.js for the run without chat history), each `move`
event gets the replay's one-line summary of that decision (`head`) and the name of what it did
(`label`), and seats named only P0, P1, ... take the replay's seat names. Runs without a replay, or
whose decisions do not line up one for one with the replay's, are left as they are.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[2] / "site" / "eco" / "data"


def _replay(game: str, run_id: str) -> dict[str, Any] | None:
    """The chart replay that draws this run: the game's own run, its run without chat history, or its DeepSeek run."""
    if run_id in (game, f"{game}_deepseek"):
        key = game
    elif run_id == f"{game}_no_history":
        key = f"{game}__no_history"
    else:
        return None
    path = DATA / f"{key}.js"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return json.loads(text[text.index('"] = ') + 5: text.rindex("}") + 1])


def annotate(game: str, run_id: str, run: dict[str, Any]) -> None:
    page = _replay(game, run_id)
    if page is None:
        return
    steps = page.get("steps", [])
    moves = [e for e in run.get("events", []) if e.get("event") == "move"]
    if not steps or len(steps) != len(moves):
        return
    kinds = page.get("kinds", {})
    for event, step in zip(moves, steps):
        event["head"] = step.get("head") or None
        event["label"] = (kinds.get(step.get("kind")) or [step.get("kind")])[0] or None
    names = page.get("seatNames") or []
    players = run.get("players", [])
    if len(names) == len(players) > 1:
        for p, name in zip(players, names):
            if p.get("name") == f"P{p.get('seat')}":
                p["name"] = re.sub(r"\s*\(P\d+\)$", "", name)  # the page shows the seat number itself
