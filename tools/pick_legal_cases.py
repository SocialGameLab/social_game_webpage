"""Draw the legal_court cases shown on the site: one per case set, at random
but under fixed rules, so the choice is reproducible and not hand-picked.

    python3 tools/pick_legal_cases.py --seed 20261023
    python3 tools/pick_legal_cases.py --seed 20261023 --env /path/to/social_game_env

Rules (in order):
1. One case from each set under configs/legal_court/ (civil, admin, criminal,
   ecthr, singapore), drawn uniformly from the eligible cases of that set.
2. Eligible = not on the EXCLUDED list (heavy subject matter: sexual offences,
   domestic violence, a killing inside a family) and with public facts of a
   readable length (at least 150 characters, at most 12,000).
3. The real outcome of at least two of the five cases must be a defence win
   (the sealed label is DEFENSE): two of the sets that contain such cases are
   chosen at random and draw only from their DEFENSE cases, so visitors can see
   the jury decide against the plaintiff as well as for it.

Standard library only; prints the picks and the content-file snippet.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = ("civil", "admin", "criminal", "ecthr", "singapore")
MIN_DEFENSE = 2
MIN_FACTS, MAX_FACTS = 150, 12_000

# Cases left out of the showcase for their subject matter, not their quality.
EXCLUDED = {
    "2001_SGHC_193": "sexual abuse of a child",
    "2003_SGHC_226": "sexual offence",
    "2022_湘09刑初16号": "killing of a parent",
    "001-58374_Khalfaoui_v._France": "sexual assault on a patient",
    "001-151051_Rohlena_v._the_Czech_Republic": "domestic violence",
    "001-71042_Reinprecht_v._Austria": "sexual coercion charge",
}


def label_of(gt: dict) -> str:
    return gt.get("label") or gt.get("heuristic_label") or "?"


def eligible(env: Path, case_set: str) -> list[tuple[str, str]]:
    """(case id, label) for every case of a set that passes rule 2, sorted by id."""
    out = []
    for cdir in sorted(p for p in (env / "configs" / "legal_court" / case_set).iterdir() if p.is_dir()):
        if cdir.name in EXCLUDED:
            continue
        facts = json.loads((cdir / "facts.json").read_text(encoding="utf-8"))["facts"]
        if not MIN_FACTS <= len(facts) <= MAX_FACTS:
            continue
        gt = json.loads((cdir / "ground_truth.json").read_text(encoding="utf-8"))
        out.append((cdir.name, label_of(gt)))
    return out


def draw(env: Path, seed: int) -> dict[str, tuple[str, str]]:
    rng = random.Random(seed)
    pools = {s: eligible(env, s) for s in SETS}
    with_defense = [s for s in SETS if any(lbl == "DEFENSE" for _, lbl in pools[s])]
    forced = set(rng.sample(sorted(with_defense), MIN_DEFENSE))
    picks = {}
    for s in SETS:
        pool = [c for c in pools[s] if c[1] == "DEFENSE"] if s in forced else pools[s]
        picks[s] = rng.choice(pool)
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--env", type=Path, default=ROOT.parent / "social_game_env")
    args = parser.parse_args()
    picks = draw(args.env.resolve(), args.seed)
    print(f"seed {args.seed}:")
    for s, (cid, lbl) in picks.items():
        print(f"  {s:10} {lbl:9} {cid}")
    print("\nruns:")
    for s, (cid, _) in picks.items():
        print(f"  - id: {s}\n    path: recordings/legal_court/{s}\n    case: configs/legal_court/{s}/{cid}")


if __name__ == "__main__":
    main()
