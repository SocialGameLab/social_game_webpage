"""Inline the whole site into one self-contained HTML page.

Run after build_data.py (plain Python, no dependencies)::

    python3 tools/build_single_file.py            # writes dist/match-room.html
    python3 tools/build_single_file.py --fragment # no doctype/head/body, for hosts that add their own

Every stylesheet, game view and data file is inlined, so MatchRoom.loadGame
finds each game already registered and fetches nothing. Fonts still come
from Google Fonts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
OUT = ROOT / "dist" / "match-room.html"


def _script(text: str) -> str:
    # A literal "</script" inside the data would end the inline block early.
    return "<script>\n" + text.replace("</script", "<\\/script") + "\n</script>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fragment", action="store_true", help="omit doctype, <head> and <body> wrappers")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    index = (SITE / "index.html").read_text(encoding="utf-8")
    manifest = (SITE / "data" / "manifest.js").read_text(encoding="utf-8")
    payload = re.search(r"MatchRoom\.setManifest\((.*)\);\s*$", manifest, re.S).group(1)
    game_ids = [g["id"] for g in json.loads(payload)["games"]]

    css = list(re.findall(r'<link rel="stylesheet" href="(assets/[^"]+)">', index))
    css += [f"games/{g}/view.css" for g in game_ids]
    styles = "<style>\n" + "\n".join((SITE / p).read_text(encoding="utf-8") for p in css) + "\n</style>"

    scripts = ["assets/js/core.js", "assets/js/ui.js", "assets/js/home.js", "data/manifest.js"]
    scripts += [p for g in game_ids for p in (f"games/{g}/view.js", f"data/{g}.js")]
    scripts += ["assets/js/app.js"]
    inline_js = "\n".join(_script((SITE / p).read_text(encoding="utf-8")) for p in scripts)

    head = re.search(r"<head>(.*?)</head>", index, re.S).group(1)
    head = re.sub(r'\s*<link rel="stylesheet" href="assets/[^"]+">', "", head)
    head = re.sub(r"\s*<!--.*?-->", "", head, flags=re.S)
    head = head.replace('<meta charset="utf-8">', "").replace('<meta name="viewport" content="width=device-width, initial-scale=1">', "") if args.fragment else head
    body = re.search(r"<body>(.*?)</body>", index, re.S).group(1)
    body = re.sub(r"\s*<script src=[^>]+></script>", "", body)
    body = re.sub(r"\s*<!--.*?-->", "", body, flags=re.S)

    if args.fragment:
        page = f"{head.strip()}\n{styles}\n{body.strip()}\n{inline_js}\n"
    else:
        page = f'<!doctype html>\n<html lang="en">\n<head>{head}{styles}\n</head>\n<body>{body}\n{inline_js}\n</body>\n</html>\n'

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
