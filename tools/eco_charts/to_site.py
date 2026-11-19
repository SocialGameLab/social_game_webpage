"""Write a builder's JSON as the eco replay page's data file, without the reasoning text.

    python3 tools/eco_charts/to_site.py OUT/eco_cournot.json [OUT/eco_beer_game__no_history.json ...]
    python3 tools/eco_charts/to_site.py site/eco/data/*.js        # re-clean files already in place

Each input becomes site/eco/data/<file stem>.js, which the page loads with a script tag (so it also opens from
file://). A file named <game>__no_history.json is the second recording of that game.

The site shows no model reasoning. Besides each step's reasoning, this drops the sentences of the page copy that
quote it verbatim or point to it (a Reasoning tab, "shown with each step"). Facts about the recording stay, such as
how many answers came after the reasoning ran out of tokens, and counts taken from the reasoning.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "site" / "eco" / "data"

# (pattern, replacement), applied in order to every text field of the page copy.
_DROP_REASONING = [
    (r"\n?From (?:her|his|its|their) reasoning, verbatim:.*\Z", ""),
    (r" (?:Her|His|Its|Their) reasoning .*?\(quoted below\)\.", ""),
    (r" Its reasoning before these orders: \"[^\"]*\"", ""),
    (r" The reasoning of decision \d+, [^\"]*\"[^\"]*\"\.?", ""),
    (r" Firm \w+'s reasoning: \"[^\"]*\"", ""),
    (r" and is shown with each step", " (not shown on this site)"),
    (r"Reasoning is shown for all", "Reasoning was saved (not shown on this site) for all"),
    (r"(?:Observations and reasoning|Reasoning and observations) come from", "Observations come from"),
    (r" ?Reasoning comes from the saved model calls\.", ""),
    (r" ?Reasoning: from the model-call log[^.]*\.", ""),
    (r", and its reasoning is shown up to the cut-off", ""),
    (r" \(marked in the reasoning\)", ""),
    (r", whose reasoning the step shows above the final attempt's", ""),
    (r", and the step shows both replies' reasoning \([^)]*\)", ""),
    (r" \(the earlier attempt is shown first, with the game's reason\)", ""),
    (r" The full reasoning of every decision was saved \(Reasoning tab\)\.", ""),
    (r" The quotes in the transcript are verbatim sentences of that decision's reasoning[^.]*\.", ""),
    (r" Where the reasoning ends with a paragraph that only restates the answer format, .*?so that the Decision tab's conclusion shows the model's decision\.", ""),
]
_RULES = [(re.compile(p, re.S), r) for p, r in _DROP_REASONING]


def _clean_text(s: str) -> str:
    for pattern, repl in _RULES:
        s = pattern.sub(repl, s)
    return s


def clean(page: dict) -> dict:
    """The page without reasoning: no step keeps its reasoning, and the copy neither quotes nor promises it."""
    for step in page.get("steps", []):
        step.pop("thinking", None)
    for key in ("lede", "foot"):
        if isinstance(page.get(key), str):
            page[key] = _clean_text(page[key])
    for item, key in [(t, "s") for t in page.get("tiles", [])] + [(f, "text") for f in page.get("feed") or []] + [(c, "sub") for c in page.get("charts", [])]:
        if isinstance(item.get(key), str):
            item[key] = _clean_text(item[key])
    return page


def _read(src: Path) -> dict:
    text = src.read_text(encoding="utf-8")
    if src.suffix == ".js":  # a data file already in site/eco/data
        return json.loads(text[text.index('"] = ') + 5: text.rindex("}") + 1])
    return json.loads(text)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    DATA.mkdir(parents=True, exist_ok=True)
    for arg in sys.argv[1:]:
        src = Path(arg)
        key = src.stem
        body = json.dumps(clean(_read(src)), ensure_ascii=False, separators=(",", ":"))
        out = DATA / f"{key}.js"
        out.write_text(f'window.__REPLAY_DATA__ = window.__REPLAY_DATA__ || {{}};\nwindow.__REPLAY_DATA__["{key}"] = {body};\n', encoding="utf-8")
        print(f"wrote {out.relative_to(DATA.parents[2])} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
