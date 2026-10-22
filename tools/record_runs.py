"""Record new runs with the LLM configured in social-game-env's .env.

    uv run --project ../social_game_env python tools/record_runs.py                       # every run, in parallel
    uv run --project ../social_game_env python tools/record_runs.py avalon scripts_00023  # only these game or run ids
    uv run --project ../social_game_env python tools/record_runs.py --env /path/to/social_game_env

Each run listed in content/games/<id>.yaml is played inside the env checkout
and written to its `path` under recordings/: a new timestamped directory plus
run.log. build_data.py always uses the newest finished one, so rebuild after:

    uv run --project ../social_game_env python tools/build_data.py

Each game is played by shelling out to `uv run` inside the env checkout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "content"

GENERIC = (
    "import sys; sys.path.insert(0, 'src'); from dotenv import load_dotenv; load_dotenv(); "
    "from runners.play import play_game; play_game({config!r}, output_root={out!r})"
)


def _command(engine: str, game_cfg: str | None, run: dict[str, str], out: Path) -> list[str]:
    if engine == "jubensha_qa":
        return ["uv", "run", "python", "-m", "jubensha_qa", "play", f"configs/jubensha_qa/{run['script']}",
                "--seed", "0", "--output-root", str(out)]
    return ["uv", "run", "python", "-c", GENERIC.format(config=game_cfg, out=str(out))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--env", type=Path, default=ROOT.parent / "social_game_env", help="social-game-env checkout (default: sibling directory)")
    parser.add_argument("only", nargs="*", help="game ids or run ids to record (default: all)")
    args = parser.parse_args()
    env = args.env.resolve()

    jobs = []
    for path in sorted((CONTENT / "games").glob("*.yaml")):
        game = yaml.safe_load(path.read_text(encoding="utf-8"))
        gid, engine, cfg = game["id"], game.get("engine", "generic"), game.get("config")
        for run in game.get("runs", []):
            if args.only and gid not in args.only and run["id"] not in args.only:
                continue
            out = ROOT / run["path"]
            jobs.append((f"{gid}/{run['id']}", _command(engine, cfg, run, out), out))

    if not jobs:
        sys.exit(f"Nothing matches {args.only}.")

    def play(job: tuple[str, list[str], Path]) -> None:
        name, cmd, out = job
        out.mkdir(parents=True, exist_ok=True)
        with (out / "run.log").open("w") as log:
            code = subprocess.call(cmd, cwd=env, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        print(f"{name} exit={code}", flush=True)

    print(f"Recording {len(jobs)} run(s) in {env} …")
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        list(pool.map(play, jobs))
    print("Done. Rebuild the site data with: uv run --project ../social_game_env python tools/build_data.py")


if __name__ == "__main__":
    main()
