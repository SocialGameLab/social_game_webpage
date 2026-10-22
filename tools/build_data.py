"""Bundle recorded runs and page copy into the data files the site reads.

Needs a checkout of social-game-env, whose engine rebuilds each game's first
prompt from the recorded seed. No model is called. Run it with that repo's
environment::

    uv run --project ../social_game_env python tools/build_data.py
    uv run --project ../social_game_env python tools/build_data.py --env /path/to/social_game_env

For every game listed in ``content/site.yaml`` it reads
``content/games/<id>.yaml``, picks the newest completed run directory under
each run's ``path``, slims the event log, and writes:

* ``site/data/<id>.js``     the game's copy and runs, as ``MatchRoom.addData({...})``
* ``site/data/manifest.js`` site settings, home-page copy, and one card per game

Both are script files, not JSON, so the page also works from ``file://``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "content"
DATA = ROOT / "site" / "data"

# Events whose payload is already carried elsewhere in the bundle.
_DROP_FIELDS = {
    "setup": ("knowledge", "roles"),  # moved onto players[]
    "game_over": ("variables",),  # summary carries the final state
    "briefing": ("background", "clues"),  # moved onto run.briefing
}

# Content keys that only the build uses; they are not shipped to the page.
_BUILD_ONLY = ("engine",)

ENV: Path  # social-game-env checkout, set in main()


def _import_env(env: Path) -> None:
    """Make social-game-env's ``src`` importable and pull in the engine."""
    global ENV, build_action_prompt, is_discussion_turn, build_discussion_prompt
    global GenericEnv, load_game_spec, load_instance, resolve_game_path, resolve_prompt_path
    ENV = env
    os.chdir(env)  # the engine resolves spec and prompt paths against the repo root
    sys.path.insert(0, str(env / "src"))
    from agents.action_io import build_action_prompt, is_discussion_turn
    from agents.discussion_prompt import build_discussion_prompt
    from engine import GenericEnv, load_game_spec, load_instance, resolve_game_path
    from engine.config import resolve_prompt_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _latest_run(root: Path, marker: str) -> Path | None:
    done = sorted(p.parent for p in root.glob(f"*/{marker}"))
    return done[-1] if done else None


def _rewards_from_log(run_root: Path) -> dict[str, float] | None:
    """`runners play` prints rewards but does not save them; read them back."""
    log = run_root / "run.log"
    if not log.exists():
        return None
    for line in log.read_text().splitlines():
        if line.startswith("Rewards: "):
            return {str(k): v for k, v in ast.literal_eval(line[len("Rewards: "):]).items()}
    return None


def _slim(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slim = []
    for event in events:
        drop = _DROP_FIELDS.get(event.get("event"), ())
        slim.append({k: v for k, v in event.items() if k not in drop})
    return slim


# --------------------------------------------------------------------------- #
# First prompt, rebuilt offline from the run's seed
# --------------------------------------------------------------------------- #
def _render_prompt(env: Any, spec: Any, prompt_path: Path | None) -> dict[str, Any]:
    env.reset()
    seat = env.agent_selection
    obs = env.observe(seat)
    if prompt_path is not None and is_discussion_turn(obs, spec):
        text, kind = build_discussion_prompt(obs, prompt_path, spec), "discussion"
    else:
        text, kind = build_action_prompt(obs), "action"
    return {"seat": seat, "phase": obs.phase, "kind": kind, "text": text}


# --------------------------------------------------------------------------- #
# Engines: how a recorded run becomes page data. A game's content file names
# one with `engine:`; most games use "generic".
# --------------------------------------------------------------------------- #
def _generic_marker(game: dict[str, Any]) -> str:
    return "summary.json"


def _generic_extras(game: dict[str, Any], run_root: Path, run_dir: Path, events: list, summary: dict) -> dict[str, Any]:
    instance = load_instance(ENV / game["config"])
    spec_path = resolve_game_path(instance)
    spec = load_game_spec(spec_path)
    prompt_path = resolve_prompt_path(spec_path, spec, "discussion")
    return {
        "result": {
            "winner": summary.get("winner"),
            "winners": summary.get("winners", []),
            "rewards": _rewards_from_log(run_root),
            "variables": summary.get("variables", {}),
        },
        "prompt": _render_prompt(GenericEnv(spec, instance), spec, prompt_path),
    }


def _jubensha_marker(game: dict[str, Any]) -> str:
    # Jubensha writes summary.json before judging; scores.json comes last.
    return "scores.json"


def _jubensha_extras(game: dict[str, Any], run_root: Path, run_dir: Path, events: list, summary: dict) -> dict[str, Any]:
    from jubensha_qa.data import load_script
    from jubensha_qa.runner import SPEC_PATH, GameConfig, build_entities, build_instance

    briefing = next(e for e in events if e.get("event") == "briefing")
    scores = json.loads((run_dir / "scores.json").read_text())

    script = load_script(ENV / "configs" / "jubensha_qa" / summary["script"])
    cfg = GameConfig(
        seed=summary["seed"],
        group_rounds=summary["group_rounds"],
        private_turns=summary["private_turns"],
        reveal_protected=summary["reveal_protected"],
    )
    # Read the stored key directly: build_answer_key may write to configs/.
    entities = build_entities(script, list(script.answer_key))
    instance = build_instance(script, entities, cfg)
    spec = load_game_spec(SPEC_PATH)
    prompt_path = resolve_prompt_path(SPEC_PATH, spec, "discussion")

    return {
        "briefing": {"background": briefing["background"], "clues": briefing["clues"]},
        "questions": [e["question"] for e in events if e.get("event") == "question"],
        "verdict": {
            "answer_key": scores["answer_key"],
            "owners": scores["owners"],
            "grades": scores["grades"],
            "per_question_accuracy": scores["per_question_accuracy"],
            "per_agent_accuracy": scores["per_agent_accuracy"],
            "per_agent_defense": scores["per_agent_defense"],
            "judge_fallbacks": scores["judge_fallbacks"],
            "overall_accuracy": summary["overall_accuracy"],
        },
        "prompt": _render_prompt(GenericEnv(spec, instance), spec, prompt_path),
    }


Extras = Callable[[dict, Path, Path, list, dict], dict[str, Any]]
ENGINES: dict[str, tuple[Callable[[dict], str], Extras]] = {
    "generic": (_generic_marker, _generic_extras),
    "jubensha_qa": (_jubensha_marker, _jubensha_extras),
}


# --------------------------------------------------------------------------- #
def _build_run(game: dict[str, Any], run_meta: dict[str, Any]) -> dict[str, Any] | None:
    marker, extras = ENGINES[game.get("engine", "generic")]
    run_root = ROOT / run_meta["path"]
    run_dir = _latest_run(run_root, marker(game))
    if run_dir is None:
        print(f"  skip {run_meta['path']}: no completed run yet")
        return None

    events = _read_jsonl(run_dir / "game_log.jsonl")
    summary = json.loads((run_dir / "summary.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    setup = next(e for e in events if e.get("event") == "setup")

    spec = load_game_spec(ENV / (config.get("game") or game["spec"]))
    team_of = {
        name: role.get("team") if isinstance(role, dict) else getattr(role, "team", None)
        for name, role in spec.roles.items()
    }

    names = {str(k): v for k, v in summary.get("assignments", {}).items()}
    players = []
    for seat_key, role in sorted(setup["roles"].items(), key=lambda kv: int(kv[0])):
        players.append({
            "seat": int(seat_key),
            "name": names.get(seat_key, f"P{seat_key}"),
            "role": role,
            "team": team_of.get(role),
            "knowledge": setup.get("knowledge", {}).get(seat_key, {}),
        })

    return {
        "id": run_meta["id"],
        "label": run_meta.get("label"),
        "note": run_meta.get("note"),
        "source": str(run_dir.relative_to(ROOT)),
        "seed": summary.get("seed", config.get("seed")),
        "players": players,
        "events": _slim(events),
        **extras(game, run_root, run_dir, events, summary),
    }


def _write_js(path: Path, call: str, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    path.write_text(f"/* Generated by tools/build_data.py. Do not edit. */\n{call}({body});\n", encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--env", type=Path, default=ROOT.parent / "social_game_env", help="social-game-env checkout (default: sibling directory)")
    parser.add_argument("games", nargs="*", help="only rebuild these game ids (the manifest is always rewritten)")
    args = parser.parse_args()
    _import_env(args.env.resolve())

    site = yaml.safe_load((CONTENT / "site.yaml").read_text())
    DATA.mkdir(parents=True, exist_ok=True)
    cards = []
    for gid in site["games"]:
        game = yaml.safe_load((CONTENT / "games" / f"{gid}.yaml").read_text())
        marker = ENGINES[game.get("engine", "generic")][0](game)
        finished = sum(_latest_run(ROOT / r["path"], marker) is not None for r in game.get("runs", []))
        # Enough for the home-page card and the game menu, without loading the game.
        cards.append({k: game.get(k) for k in ("id", "title", "kind", "players", "tagline")} | {"runs": finished})
        if args.games and gid not in args.games:
            continue
        print(f"{gid}:")
        runs = [r for r in (_build_run(game, meta) for meta in game.pop("runs", [])) if r is not None]
        for r in runs:
            print(f"  {r['id']}: {len(r['events'])} events from {r['source']}")
        shipped = {k: v for k, v in game.items() if k not in _BUILD_ONLY}
        _write_js(DATA / f"{gid}.js", "MatchRoom.addData", {**shipped, "runs": runs})

    site_meta = {k: v for k, v in site.items() if k != "games"}
    _write_js(DATA / "manifest.js", "MatchRoom.setManifest", {"site": site_meta, "games": cards})


if __name__ == "__main__":
    main()
