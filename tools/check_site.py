"""Check that site/ is complete and consistent before it is published.

    python3 tools/check_site.py

Standard library only, so CI can run it without social-game-env. Checks that
every game in the manifest has its content file, view.js, view.css and data
file; that each data file matches its game id; that every JS file parses
(when node is installed); and that the manifest lists every game in
content/site.yaml. It cannot tell whether site/data/ is stale relative to
recordings/, because that needs the engine; rebuild before committing.
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

    for gid in ids:
        for rel in (f"content/games/{gid}.yaml", f"site/games/{gid}/view.js", f"site/games/{gid}/view.css", f"site/data/{gid}.js"):
            if not (ROOT / rel).exists():
                errors.append(f"{gid}: missing {rel}")
        view = SITE / "games" / gid / "view.js"
        if view.exists() and f'MatchRoom.defineGame("{gid}"' not in view.read_text(encoding="utf-8"):
            errors.append(f"{gid}: view.js does not call MatchRoom.defineGame(\"{gid}\", ...)")
        data = SITE / "data" / f"{gid}.js"
        if data.exists():
            game = _payload(data, "MatchRoom.addData")
            if game.get("id") != gid:
                errors.append(f"{gid}: data file holds id {game.get('id')!r}")

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
