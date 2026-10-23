"""Check that site/ is complete and consistent before it is published.

    python3 tools/check_site.py

Standard library only, so CI can run it without social-game-env. Checks that
every game in the manifest has its content file, view.js and view.css (its
own, its shared kind's under site/kinds/, or the game named by view_from), and data
file; that each data file matches its game id; that every JS file parses
(when node is installed); and that the manifest lists every game in
content/site.yaml. When social-game-env is checked out next to this repo, it
also checks that every run shown still exists in its demo/index.json. It
cannot tell whether a run's data is stale; rebuild before committing.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DEMO_INDEX = ROOT.parent / "social_game_env" / "demo" / "index.json"


def _payload(path: Path, call: str) -> dict:
    m = re.search(rf"{re.escape(call)}\((.*)\);\s*$", path.read_text(encoding="utf-8"), re.S)
    if not m:
        raise ValueError(f"{path.relative_to(ROOT)} does not call {call}(...)")
    return json.loads(m.group(1))


def main() -> int:
    errors: list[str] = []
    manifest = _payload(SITE / "data" / "manifest.js", "MatchRoom.setManifest")
    ids = [g["id"] for g in manifest["games"]]

    # site.yaml is the source of the manifest's game list; a mismatch means a stale build.
    site_yaml = (ROOT / "content" / "site.yaml").read_text(encoding="utf-8")
    listed = re.findall(r"^  - ([a-z][a-z0-9_]*)\s*$", site_yaml.split("\ngames:\n", 1)[-1], re.M)
    if listed != ids:
        errors.append(f"manifest games {ids} differ from content/site.yaml {listed}: run tools/build_data.py")

    for card in manifest["games"]:
        gid = card["id"]
        # A game draws with its own view, a shared kind, or another game's view.
        if card.get("view"):
            vdir, call = f"site/kinds/{card['view']}", f'MatchRoom.defineKind("{card["view"]}"'
        elif card.get("view_from"):
            vdir, call = f"site/games/{card['view_from']}", f'MatchRoom.defineGame("{card["view_from"]}"'
        else:
            vdir, call = f"site/games/{gid}", f'MatchRoom.defineGame("{gid}"'
        for rel in (f"content/games/{gid}.yaml", f"{vdir}/view.js", f"{vdir}/view.css", f"site/data/{gid}.js"):
            if not (ROOT / rel).exists():
                errors.append(f"{gid}: missing {rel}")
        view = ROOT / vdir / "view.js"
        if view.exists() and call not in view.read_text(encoding="utf-8"):
            errors.append(f"{gid}: {vdir}/view.js does not call {call}, ...)")
        data = SITE / "data" / f"{gid}.js"
        if data.exists():
            game = _payload(data, "MatchRoom.addData")
            if game.get("id") != gid:
                errors.append(f"{gid}: data file holds id {game.get('id')!r}")

    # CI has no social-game-env checkout; locally, catch runs dropped from demo/.
    if DEMO_INDEX.exists():
        exported = {gid: {r["id"] for r in g["runs"]} for gid, g in json.loads(DEMO_INDEX.read_text(encoding="utf-8"))["games"].items()}
        for gid in ids:
            data = SITE / "data" / f"{gid}.js"
            if data.exists():
                for run in _payload(data, "MatchRoom.addData")["runs"]:
                    if run["id"] not in exported.get(gid, set()):
                        errors.append(f"{gid}: run {run['id']!r} is not in social_game_env/demo/index.json")
    else:
        print("social_game_env/demo not found next to this repo; skipped run checks")

    node = shutil.which("node")
    if node:
        for js in sorted(SITE.rglob("*.js")):
            if js.parent.name == "data":
                continue  # generated JSON payloads; parsed above
            res = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
            if res.returncode:
                lines = res.stderr.strip().splitlines()
                errors.append(f"{js.relative_to(ROOT)}: {next((l for l in lines if 'Error' in l), lines[0])}")
    else:
        print("node not found; skipped JS syntax checks")

    for e in errors:
        print(f"error: {e}")
    print(f"{len(ids)} games checked, {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
