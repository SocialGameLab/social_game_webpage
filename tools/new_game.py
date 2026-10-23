"""Scaffold a new game from templates/game/.

    python3 tools/new_game.py werewolf "Werewolf"
    python3 tools/new_game.py stag_hunt "Stag Hunt" --view rounds     # draw with a shared kind
    python3 tools/new_game.py mafia "Mafia" --view-from werewolf      # reuse another game's view

Creates content/games/<id>.yaml and, unless the game uses a shared view,
site/games/<id>/view.{js,css}; then adds the id to the game list in
content/site.yaml. Existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "game"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("id", help="snake_case id, matching the game's name in social-game-env")
    parser.add_argument("title", help="display title")
    shared = parser.add_mutually_exclusive_group()
    shared.add_argument("--view", metavar="KIND", help="draw with site/kinds/KIND instead of a view of its own")
    shared.add_argument("--view-from", metavar="GAME", help="reuse GAME's view")
    args = parser.parse_args()
    if args.view and not (ROOT / "site" / "kinds" / args.view / "view.js").exists():
        sys.exit(f"no kind {args.view!r} in site/kinds/")
    if args.view_from and not (ROOT / "site" / "games" / args.view_from / "view.js").exists():
        sys.exit(f"no view for {args.view_from!r} in site/games/")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.id):
        sys.exit("id must be snake_case: lowercase letters, digits and underscores")

    targets = {TEMPLATE / "content.yaml": ROOT / "content" / "games" / f"{args.id}.yaml"}
    if not (args.view or args.view_from):
        targets[TEMPLATE / "view.js"] = ROOT / "site" / "games" / args.id / "view.js"
        targets[TEMPLATE / "view.css"] = ROOT / "site" / "games" / args.id / "view.css"
    if clash := [str(t.relative_to(ROOT)) for t in targets.values() if t.exists()]:
        sys.exit(f"already exists: {', '.join(clash)}")

    for src, dst in targets.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding="utf-8").replace("__ID__", args.id).replace("__TITLE__", args.title)
        if src.name == "content.yaml" and (args.view or args.view_from):
            line = f"view: {args.view}\n" if args.view else f"view_from: {args.view_from}\n"
            text = text.replace("\nsetup:\n", f"\n{line}\nsetup:\n", 1)
        dst.write_text(text, encoding="utf-8")
        print(f"created {dst.relative_to(ROOT)}")

    site = ROOT / "content" / "site.yaml"
    text = site.read_text(encoding="utf-8")
    if not re.search(rf"^  - {re.escape(args.id)}$", text, re.M):
        site.write_text(text.rstrip("\n") + f"\n  - {args.id}\n", encoding="utf-8")
        print(f"added {args.id} to content/site.yaml")

    print("\nNext: fill in the copy, add demo/<id>/demo.yaml in social-game-env and record a run there,\n"
          "rebuild the data, then draw the board in view.js.")


if __name__ == "__main__":
    main()
