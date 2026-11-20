"""Replay data for eco_pirate_game (GAMA-Bench game 8, the Pirate Game: seniority, one proposal per round, an at-least-half
vote, a failed proposer goes overboard).

usage: python3 -I eco_pirate_game.py SOURCE OUT_DIR [CALLS.jsonl]
       python3 -I eco_pirate_game.py --selftest
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md), a file inside
          one, or the rendered Markdown replay of one (one "- **event** ... `{json}`" line per logged event).
  OUT_DIR where <OUT_DIR>/eco_pirate_game.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). Calls are tied to
          decisions in time order: a call belongs to the next decision of the same seat, and the call whose answer is the
          logged move is that decision's call (earlier ones were format retries; their reasoning is kept above it). Its
          reasoning becomes the step's thinking and its prompt the observation.
  --selftest  replays the paper's Table 2 game (Huang et al. 2025) through the scoring code: S8P 36, 19 of 24 votes, S8 80.6.

The x axis of a one-round game is the pirates in seniority order (Pirate 1 = seat P0, the most senior), which is also the order
of the decisions: the proposal in the proposer's column, each vote in the voter's column. With several rounds it is the
decisions in time order ("R2 · Pirate 5"), so votes of one pirate in different rounds get separate slots. Charts: what accepting
is worth to each voter (offer minus the share if the plan fails; bars), the offer/vote grid (with a Pirate row, since the page
labels only every other column at common widths), and the vote count with both thresholds. Grid cells are opaque and coins are
blue (c1) because the page writes cell text in white: yellow, light gray and translucent cells were unreadable in the light
theme. Every number shown comes from the
game log; the equilibrium split, the paper's voter rule and the scores S8P, S8V and S8 are computed from the logged proposals
and votes with the formulas in rules/eco_pirate_game_rules.md (Phenomenon; Differences From the Source, code_scoring)."""
import json, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_pirate_game"
DEFAULT_DATE = "2026-11-09"
KINDS = {"propose": ["Proposal", "c1"], "accept": ["Vote: accept", "c3"], "reject": ["Vote: reject", "c2"]}
MINUS, TIMES, DIV = "\u2212", "\u00d7", "\u00f7"


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_PUNCT = {"\u3002": ". ", "\u3001": ", ", "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"', "\u3010": "[", "\u3011": "]",
          "\u300a": '"', "\u300b": '"', "\u3008": "<", "\u3009": ">", "\u301c": "~"}


def en(s):
    """The page shows English only: full-width forms become ASCII, CJK punctuation its ASCII counterpart, and any remaining run
    of Chinese, Japanese or Korean characters (the model sometimes reasons in Chinese) a visible marker."""
    if not isinstance(s, str) or not re.search(r"[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af\uff00-\uffef]", s):
        return s
    out = []
    for ch in s:
        o = ord(ch)
        if 0xFF00 <= o <= 0xFFEF:
            ch = unicodedata.normalize("NFKC", ch)
        elif o == 0x3000:
            ch = " "
        elif 0x3001 <= o <= 0x303F:
            ch = _PUNCT.get(ch, "")
        out.append(ch)
    s = _CJK_RUN.sub("[non-English text omitted]", "".join(out))
    return re.sub(r"[\u3000-\u303f\uff00-\uffef]", "", s)


# ------------------------------------------------------------------------------------------------ loading
def _md_bundle(path):
    """A rendered Markdown replay (render_replays*.py, declarative games): events from the '- **event** (tag): `{json}`' lines
    (the tag marks private and hidden events), seat rewards, seed and model from the header, and the model calls listed after
    the events ('### ' blocks), if any."""
    text = open(path, encoding="utf-8").read()
    events = []
    for m in re.finditer(r"^- \*\*([A-Za-z_]+)\*\*([^`\n]*)`(\{.*\})`\s*$", text, re.M):
        try:
            body = json.loads(m.group(3))
        except ValueError:
            continue
        body["event"] = m.group(1)
        if "\u9690\u85cf" in m.group(2):            # "hidden"
            body["_visibility"] = "hidden"
        elif "\u79c1\u5bc6" in m.group(2):          # "private"
            body["_visibility"] = "private"
        events.append(body)
    head = text.split("\n## ")[0].split("\n- **")[0]
    rw = re.search(r'(\{"0":[^}]*\})', head)
    seed = re.search(r"\u79cd\u5b50 (\d+)", head)    # "seed N"
    model = re.search(r"\u6a21\u578b (\S+)", head)   # "model X"
    over = next((e for e in events if e.get("event") == "game_over"), {})
    setup = next((e for e in events if e.get("event") == "setup"), {})
    summary = {"rewards": json.loads(rw.group(1)) if rw else {}, "roles": setup.get("roles", {}),
               "variables": over.get("variables", {}), "winners": over.get("winners")}
    run = {"seed": int(seed.group(1)) if seed else None, "model": model.group(1) if model else ""}
    calls = []
    for blk in re.split(r"\n(?=### )", text)[1:]:
        hdr = blk.split("\n", 1)[0]
        ans = re.findall(r"^\*\*[^*\n]+\*\* `(.*)`\s*$", blk, re.M)
        th = re.findall(r"````text\n(.*?)\n````", blk, re.S)
        if ans:
            calls.append({"seat": common.seat_of(hdr), "prompt": "", "answer": (ans[-1] if len(ans) > 1 else ans[0]).strip(),
                          "reasoning": th[-1] if th else "", "rescued": len(ans) > 1})
    return {"events": events, "summary": summary, "run": run, "config": {}, "transcript": "", "meta": {}, "md_calls": calls}


def load_source(src):
    if isinstance(src, dict):                     # an already loaded bundle (tests)
        return src
    if os.path.isdir(src):
        b = common.load_bundle(src)
        meta = os.path.join(src, "meta.json")
        b["meta"] = json.load(open(meta, encoding="utf-8")) if os.path.exists(meta) else {}
        return b
    d = os.path.dirname(os.path.abspath(src))
    if os.path.exists(os.path.join(d, "game_log.jsonl")):
        return load_source(d)
    return _md_bundle(src)


def recorded_date(src):
    b = load_source(src)
    return str((b.get("meta") or {}).get("recorded_at") or (b.get("run") or {}).get("recorded_at") or DEFAULT_DATE)[:10]


# ------------------------------------------------------------------------------------------------ the game's arithmetic
def equilibrium(alive, gold):
    """Stewart's backward-induction split for the pirates aboard (alive seats in seniority order, proposer first): one coin to
    every second pirate below the proposer (offsets 2, 4, ...), floor((n - 1) / 2) coins in all, the rest to the proposer."""
    n = len(alive)
    bribes = min((n - 1) // 2, gold)
    plan = [0] * n
    plan[0] = gold - bribes
    for k in range(1, bribes + 1):
        plan[2 * k] = 1
    return plan


def fallback(r, gold):
    """{seat: coins} each voter of round r would get if its plan failed: the proposer goes overboard and the next pirate proposes
    the equilibrium split among those left."""
    nxt = r["alive"][1:]
    return dict(zip(nxt, equilibrium(nxt, gold))) if nxt else {}


def rule_vote(offered, offset, code_scoring=False):
    """The paper's optimal voter (Huang et al. 2025, section 3.3): accept two or more coins, reject none, accept one coin only
    from a proposer an even number of places above (same parity), otherwise reject. With code_scoring the official code
    credits neither vote in that last case (None)."""
    if offered >= 2:
        return "accept"
    if offered <= 0:
        return "reject"
    if offset % 2 == 0:
        return "accept"
    return None if code_scoring else "reject"


def scores(rounds, gold, code_scoring=False):
    """S8P (mean L1 distance of each voted proposal to the equilibrium, in coins), S8V (share of votes that follow the rule) and
    S8 = (2G - S8P) / (2G) * 50 + S8V * 50 (Eq. 1). With code_scoring: shares rescaled to 100 coins and S8 over 200."""
    base = 100 if code_scoring else gold
    l1s, correct, total = [], 0, 0
    for r in rounds:
        eq = equilibrium(r["alive"], base)
        plan = [p * 100 / gold for p in r["plan"]] if code_scoring else r["plan"]
        l1s.append(sum(abs(a - b) for a, b in zip(plan, eq)))
        for v in r["votes"]:
            total += 1
            correct += int(rule_vote(v["offered"], v["seat"] - r["proposer"], code_scoring) == v["decision"])
    s8p = sum(l1s) / len(l1s) if l1s else 0.0
    s8v = correct / total if total else 0.0
    return {"l1": l1s, "s8p": s8p, "correct": correct, "total": total, "s8v": s8v,
            "s8": (2 * base - s8p) / (2 * base) * 50 + s8v * 50}


def _selftest():
    """The paper's Table 2 game: S8P 8, 6, 94 (mean 36), 19 of 24 votes by the rule, S8 80.6."""
    al = list(range(10))
    rounds = [{"alive": al, "proposer": 0, "plan": [100] + [0] * 9,
               "votes": [{"seat": s, "offered": 0, "decision": "reject"} for s in range(1, 10)]},
              {"alive": al[1:], "proposer": 1, "plan": [99, 0, 1, 0, 0, 0, 0, 0, 0],
               "votes": [{"seat": s, "offered": 1 if s == 3 else 0, "decision": "accept" if s in (3, 4, 9) else "reject"} for s in range(2, 10)]},
              {"alive": al[2:], "proposer": 2, "plan": [50, 1, 1, 1, 1, 1, 1, 44],
               "votes": [{"seat": s, "offered": 44 if s == 9 else 1, "decision": "accept"} for s in range(3, 10)]}]
    sc = scores(rounds, 100)
    assert [round(x) for x in sc["l1"]] == [8, 6, 94] and (sc["correct"], sc["total"]) == (19, 24), sc
    assert round(sc["s8p"], 6) == 36 and round(sc["s8"], 1) == 80.6, sc
    assert equilibrium(al, 100) == [96, 0, 1, 0, 1, 0, 1, 0, 1, 0] and equilibrium(al, 4) == [0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    assert equilibrium(al, 5) == [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] and equilibrium([8, 9], 100) == [100, 0]
    print("selftest ok: Table 2 gives S8P %.0f, %d of %d votes, S8 %.1f" % (sc["s8p"], sc["correct"], sc["total"], sc["s8"]))


# ------------------------------------------------------------------------------------------------ text helpers
def pname(seat):
    return f"Pirate {seat + 1}"


def plist(seats):
    """[2, 4, 6] -> 'Pirates 3, 5 and 7'; [2] -> 'Pirate 3'."""
    names = [str(s + 1) for s in seats]
    if not names:
        return "no pirate"
    if len(names) == 1:
        return f"Pirate {names[0]}"
    return "Pirates " + ", ".join(names[:-1]) + " and " + names[-1]


NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
       "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def numword(n):
    return NUM[n] if 0 <= n < len(NUM) else str(n)


ORD = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth"]


def ordinal(n):
    return ORD[n] if 0 <= n < len(ORD) else f"{n}th"


def signed(v):
    return ("+" if v > 0 else MINUS if v < 0 else "") + f"{abs(v):g}"


def grouped(amounts, unit=None):
    """{seat: amount} -> 'Pirate 1: 96 coins; Pirates 3, 5, 7 and 9: 1 coin each', largest first, zeros left out."""
    unit = unit or coins
    by = {}
    for s, g in amounts.items():
        if g:
            by.setdefault(g, []).append(s)
    return "; ".join(f"{plist(ss)}: {unit(g)}{' each' if len(ss) > 1 else ''}" for g, ss in sorted(by.items(), key=lambda t: -t[0]))


def coins(v):
    return f"{v:g} coin" + ("" if v == 1 else "s")


def rew(v):
    return (MINUS if v < -5e-4 else "") + f"{abs(v):.3f}"


def split_text(alive, plan):
    """'keeps 96; 1 coin each to Pirates 3, 5, 7 and 9; nothing for Pirates 2, 4, 6, 8 and 10'."""
    by = {}
    for s, g in zip(alive[1:], plan[1:]):
        by.setdefault(g, []).append(s)
    parts = [f"keeps {plan[0]}"]
    for g in sorted((g for g in by if g > 0), reverse=True):
        ss = by[g]
        parts.append(f"{coins(g)} {'each ' if len(ss) > 1 else ''}to {plist(ss)}")
    if 0 in by:
        parts.append(f"nothing for {plist(by[0])}")
    return "; ".join(parts)


def _norm_answer(s):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (s or "").strip())
    try:
        return json.loads(s)
    except ValueError:
        return None


def _matches(call, st):
    a = _norm_answer(call.get("answer"))
    if not isinstance(a, dict) or a.get("action_id") != st["action_id"]:
        return False
    p = a.get("payload") or {}
    if st["action_id"] == "propose":
        try:
            return [int(x) for x in p.get("proposal")] == st["proposal"] and all(float(x).is_integer() for x in p.get("proposal"))
        except (TypeError, ValueError):
            return False
    return p.get("decision") == st["decision"]


def attach_calls(steps, calls):
    """Tie calls to steps in time order. For each step, the window is the run of calls from the pointer whose seat is the
    step's seat (or unknown); the first call in it whose answer is the logged move is the one; earlier calls in the window
    were failed attempts (format retries) and their reasoning is kept above it. With no matching answer the last call of
    the window is used and the step notes that the engine applied a different move (an illegal proposal becomes the equal
    split and an illegal vote becomes accept, rules: Actions)."""
    ptr = 0
    for st in steps:
        j = ptr
        while j < len(calls) and calls[j].get("seat") in (None, st["seat"]):
            j += 1
        window = list(range(ptr, j))
        if not window:
            continue
        hit = next((k for k in window if _matches(calls[k], st)), None)
        use = hit if hit is not None else window[-1]
        parts = []
        for k in range(ptr, use):
            c = calls[k]
            parts.append(f"[Earlier attempt, answer: {c['answer'][:300]}]\n{c.get('reasoning') or '(no reasoning returned)'}")
        c = calls[use]
        if parts:
            parts.append("[Final attempt]\n" + (c.get("reasoning") or "(no reasoning returned)"))
        else:
            parts.append(c.get("reasoning") or "")
        if hit is None:
            parts.append(f"[The engine applied a different move than this answer: {c['answer'][:300]}]")
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        if c.get("prompt"):
            st["state"] = en(common.state_part(c["prompt"]))
            st["state_from_prompt"] = True
        a = _norm_answer(c.get("answer"))
        if hit is not None and isinstance(a, dict):
            st["act"] = en(json.dumps(a, ensure_ascii=False))
        ptr = use + 1


# ------------------------------------------------------------------------------------------------ build
def build(bundle_dir, calls=None):
    b = load_source(bundle_dir)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    if not calls:
        calls = b.get("md_calls") or []
    events, summary, run, cfg = b["events"], b.get("summary") or {}, b.get("run") or {}, b.get("config") or {}
    over = next((e for e in events if e.get("event") == "game_over"), {})
    var = dict(over.get("variables") or {})
    var.update({k: v for k, v in (summary.get("variables") or {}).items() if k not in var})
    rules_ev = next((e for e in events if e.get("event") == "rules"), {})
    G = int(rules_ev.get("gold", var.get("gold", (cfg.get("variables") or {}).get("gold", 100))))
    N = int(rules_ev.get("n_pirates", var.get("n_players", cfg.get("n_players", 10))))
    bonus = float(var.get("overboard_bonus", (cfg.get("variables") or {}).get("overboard_bonus", 0.001)))
    code_scoring = bool(var.get("code_scoring", False))
    names = [pname(s) for s in range(N)]

    # ---- rounds from the log: the proposal, the private vote records (logged after the tally, in seat order), the tally
    rounds, steps = [], []
    for e in events:
        ev = e.get("event")
        if ev == "proposal":
            alive = [int(s) for s in e["alive_seats"]]
            r = {"round": int(e["round"]), "proposer": int(e["proposer"]), "alive": alive, "plan": [int(g) for g in e["proposal"]],
                 "votes": [], "accept_count": None, "passed": None, "step": len(steps)}
            r["offered"] = dict(zip(alive, r["plan"]))
            rounds.append(r)
            steps.append({"seat": r["proposer"], "x": r["proposer"], "kind": "propose", "action_id": "propose", "proposal": r["plan"],
                          "round": r["round"]})
        elif ev == "vote" and rounds:
            r = rounds[-1]
            v = {"seat": int(e["seat"]), "decision": e["decision"], "offered": int(e.get("offered", r["offered"].get(int(e["seat"]), 0))),
                 "step": len(steps)}
            r["votes"].append(v)
            steps.append({"seat": v["seat"], "x": v["seat"], "kind": v["decision"] if v["decision"] in KINDS else "reject",
                          "action_id": "vote", "decision": v["decision"], "round": r["round"]})
        elif ev == "vote_result" and rounds:
            rounds[-1]["accept_count"] = int(e["accept_count"])
            rounds[-1]["alive_count"] = int(e["alive_count"])
            rounds[-1]["passed"] = bool(e["passed"])
    if not rounds:
        raise SystemExit("no proposal in the game log")
    R = len(rounds)
    final = rounds[-1]
    # The horizontal axis. One round: one slot per pirate, seniority order (each pirate decides once). Several rounds: one
    # slot per decision in time order ("R2 · Pirate 5"), so a pirate's votes in different rounds get their own slots, the
    # tally of an earlier round is not drawn over by a later one and playback runs left to right.
    many = R > 1
    if many:
        for k, st in enumerate(steps):
            st["x"] = k
        xlabels = [f"R{st['round']} · {pname(st['seat'])}" for st in steps]
    else:
        xlabels = [pname(s) for s in range(N)]
    NX = len(xlabels)
    eliminated = [r["proposer"] for r in rounds if r["passed"] is False]
    game_end = next((e for e in events if e.get("event") == "game_end"), {})
    gold_by_seat = game_end.get("gold")
    if gold_by_seat is None:
        gold_by_seat = [0] * N
        if final["passed"]:
            for s, g in final["offered"].items():
                gold_by_seat[s] = g
    raw_rw = summary.get("rewards") or {str(k): v for k, v in enumerate(game_end.get("rewards") or [])}
    rewards = [float(raw_rw.get(str(s), raw_rw.get(s, 0.0))) for s in range(N)]

    # ---- consistency with the rules (warnings only; the page shows the logged values)
    warn = []
    for r in rounds:
        acc = 1 + sum(v["decision"] == "accept" for v in r["votes"])
        if r["accept_count"] is not None and acc != r["accept_count"]:
            warn.append(f"round {r['round']}: {acc} accepts counted from the votes, {r['accept_count']} logged")
        if r["passed"] is not None and r["passed"] != (2 * r["accept_count"] >= len(r["alive"])):
            warn.append(f"round {r['round']}: passed={r['passed']} against the at-least-half rule")
        if sum(r["plan"]) != G:
            warn.append(f"round {r['round']}: the plan sums to {sum(r['plan'])}, not {G}")
        if [v["seat"] for v in r["votes"]] != r["alive"][1:]:
            warn.append(f"round {r['round']}: voters {[v['seat'] for v in r['votes']]} are not the pirates aboard below the proposer")
    if var.get("eliminated") is not None and list(var["eliminated"]) != eliminated:
        warn.append(f"eliminated {var['eliminated']} logged, {eliminated} from the vote results")
    E = len(eliminated)
    for s in range(N):
        want = -1.0 if s in eliminated else min(1.0, gold_by_seat[s] / G + bonus * E)
        if abs(want - rewards[s]) > 1e-9:
            warn.append(f"{pname(s)}: reward {rewards[s]} logged, {want} by the rule")
    for w in warn:
        print("warning:", w, file=sys.stderr)

    sc = scores(rounds, G, code_scoring)

    # ---- what each pirate saw (rebuilt from the log; the saved prompt where there is one)
    prompt = run.get("prompt") or {}
    history, own_votes = [], {}

    def observation(st, r):
        seat = st["seat"]
        alive = r["alive"]
        lines = ["(Rebuilt from the game log; the exact prompt for this turn was not saved.)",
                 f"You are seat P{seat} ({pname(seat)}, rank {seat + 1} of {N}). Your role: PIRATE.",
                 f"Round {r['round']}. Coins to split: {G}. Pirates aboard: {len(alive)} (seats P{alive[0]} to P{alive[-1]}). "
                 + ("Thrown overboard so far: " + ", ".join(f"P{s}" for s in eliminated if s < alive[0]) + "." if alive[0] > 0 else "Nobody has been thrown overboard yet.")]
        if st["kind"] == "propose":
            lines.append("Current phase: PROPOSE. You are the most senior pirate aboard and propose the split: one non-negative integer per "
                         f"pirate aboard, in the order of the seats aboard {alive}, your own share first, adding up to {G}.")
        else:
            lines.append("Current phase: VOTE. Vote accept or reject in secret; after the vote only the number of accepting pirates is announced.")
            lines.append(f"Proposal on the table: P{r['proposer']} (rank {r['proposer'] + 1}) proposes {r['plan']} for the seats {alive}.")
            lines.append(f"Your share under this plan: {coins(r['offered'][seat])}.")
        lines.append("Earlier rounds:" if history else "Earlier rounds: none.")
        lines += history
        mine = own_votes.get(seat, [])
        if mine:
            lines.append("Your own earlier votes: " + "; ".join(mine) + ".")
        return "\n".join(lines)

    first_seen = False
    for r in rounds:
        for st in steps[r["step"]:r["step"] + 1 + len(r["votes"])]:
            st["state"] = observation(st, r)
            if (not first_seen and st["kind"] == "propose" and prompt.get("seat") == st["seat"] and prompt.get("phase") == "PROPOSE"
                    and re.search(r'"round": %d\b' % r["round"], prompt.get("text") or "")):
                st["state"] = en(common.state_part(prompt["text"]))      # the saved prompt of the game's first decision
                st["state_from_prompt"] = True
            first_seen = True
        for v in r["votes"]:
            own_votes.setdefault(v["seat"], []).append(f"round {r['round']}: {v['decision']} (offered {coins(v['offered'])})")
        line = f"- Round {r['round']}: P{r['proposer']} proposed {r['plan']} for the seats {r['alive']}; {r['accept_count']} of {len(r['alive'])} accepted"
        line += "; the plan passed." if r["passed"] else f"; P{r['proposer']} was thrown overboard."
        history.append(line)

    # ---- heads, actions, feed
    feed = []
    for r in rounds:
        eq = equilibrium(r["alive"], G)
        l1 = sum(abs(a - b) for a, b in zip(r["plan"], eq))
        st = steps[r["step"]]
        st["head"] = f"Round {r['round']} \u00b7 proposes: keeps {r['plan'][0]} of {G}"
        st["act"] = json.dumps({"action_id": "propose", "payload": {"proposal": r["plan"]}})
        feed.append({"x": st["x"], "step": r["step"], "who": pname(r["proposer"]), "kind": "propose",
                     "text": f"Proposes a split of the {G} coins: {split_text(r['alive'], r['plan'])}. "
                             + ("This is exactly the equilibrium split." if l1 == 0 else
                                f"The equilibrium split ({split_text(r['alive'], eq)}) is {coins(l1)} away.")})
        for v in r["votes"]:
            st = steps[v["step"]]
            verb = {"accept": "accepts", "reject": "rejects"}.get(v["decision"], v["decision"])
            st["head"] = f"Round {r['round']} \u00b7 offered {coins(v['offered'])} \u00b7 {verb}"
            st["act"] = json.dumps({"action_id": "vote", "payload": {"decision": v["decision"]}})
            rv = rule_vote(v["offered"], v["seat"] - r["proposer"], code_scoring)
            feed.append({"x": st["x"], "step": v["step"], "who": pname(v["seat"]), "kind": st["kind"],
                         "text": f"Offered {coins(v['offered'])}; {verb} in secret. "
                                 + ("The paper's voter rule says the same." if rv == v["decision"] else
                                    "The paper's voter rule credits neither vote here." if rv is None else
                                    f"The paper's voter rule says {rv}.")})
        need = (len(r["alive"]) + 1) // 2
        res = (f"Count: {r['accept_count']} of {len(r['alive'])} pirates accept ({pname(r['proposer'])}'s own vote included; {need} needed). ")
        if r["passed"]:
            res += (f"The plan passes in round {r['round']} and the coins are paid: {grouped({s: gold_by_seat[s] for s in range(N)})}"
                    + ("; nothing for the others" if any(g == 0 for g in gold_by_seat) else "") + ". Rewards: "
                    + grouped({s: rewards[s] for s in range(N) if rewards[s] > 0}, rew)
                    + "".join(f"; {plist(ss)}: {MINUS}1" for ss in [[s for s in range(N) if rewards[s] < 0]] if ss)
                    + ("; 0 for the others." if any(rewards[s] == 0 for s in range(N)) else "."))
        else:
            res += f"The plan fails and {pname(r['proposer'])} is thrown overboard."
        last_step = r["step"] + len(r["votes"])
        feed.append({"x": steps[last_step]["x"], "step": last_step, "who": "Result", "text": res})

    attach_calls(steps, calls)
    has_reason = any(s.get("thinking") for s in steps)
    for st in steps:
        st.setdefault("thinking", "")
        for k in ("action_id", "proposal", "decision", "round", "state_from_prompt"):
            st.pop(k, None)

    # ---- charts
    r1 = rounds[0]

    def slot(k):
        return steps[k]["x"]

    def who(r, s):
        return f"{pname(s)} in round {r['round']}" if many else pname(s)

    # 1. what accepting is worth to each voter: the coins offered minus the coins it would get if the plan failed (the next
    #    pirate's equilibrium split). In equilibrium a pirate accepts exactly when this is above zero (a tie is turned down).
    entries = []
    for r in rounds:
        fb = fallback(r, G)
        for v in r["votes"]:
            entries.append({"r": r, "v": v, "fb": fb.get(v["seat"], 0), "m": v["offered"] - fb.get(v["seat"], 0),
                            "next": v["seat"] == r["alive"][1]})
    small = [e for e in entries if abs(e["m"]) <= 10]
    cut = [e for e in entries if abs(e["m"]) > 10] if small else []   # e.g. the next proposer's -96: it would flatten the rest
    worth = [None] * NX
    for e in entries:
        if e not in cut:
            worth[slot(e["v"]["step"])] = e["m"]
    follow = sum((e["m"] > 0) == (e["v"]["decision"] == "accept") for e in entries)
    if many:
        sub1 = ("For each vote: the coins the plan offers that pirate minus the coins it would get if the plan failed, the proposer went "
                "overboard and the next pirate proposed the equilibrium split (Stewart 1999) to those left. ")
    else:
        sub1 = (f"For each voter: the coins {pname(r1['proposer'])}'s plan offers minus the coins it would get if the plan failed, "
                f"{pname(r1['proposer'])} went overboard and {pname(r1['alive'][1])} proposed the equilibrium split (Stewart 1999) to the "
                f"{numword(len(r1['alive']) - 1)} left. ")
    sub1 += ("Above zero, accepting pays: in equilibrium a pirate accepts exactly there and turns down a tie, because it also gains from "
             "seeing another pirate thrown overboard. ")
    if not many:
        by = {}
        for e in entries:
            if e not in cut:
                by.setdefault(e["m"], []).append(e["v"])
        parts = []
        for m in sorted(by, reverse=True):
            vs = by[m]
            acc = [v["seat"] for v in vs if v["decision"] == "accept"]
            rej = [v["seat"] for v in vs if v["decision"] != "accept"]
            if acc and rej:
                how = f" ({plist(acc)} accepted, {plist(rej)} rejected)"
            else:
                how = (", who " + ("all " if len(vs) > 1 else "") + ("accepted" if acc else "rejected"))
            parts.append(f"{signed(m)} for {plist([v['seat'] for v in vs])}{how}")
        sub1 += "Here: " + "; ".join(parts) + ". "
    sub1 += (f"All {len(entries)} votes follow the sign" if follow == len(entries) else f"{follow} of {len(entries)} votes follow the sign") + ". "
    if cut:
        sub1 += ("Left out so that one-coin differences stay visible: "
                 + "; ".join(f"{who(e['r'], e['v']['seat'])}, {signed(e['m'])} ("
                             + (f"it would propose next and keep {coins(e['fb'])}" if e["next"] else
                                f"offered {coins(e['v']['offered'])} against {coins(e['fb'])}")
                             + f"; it {'accepted' if e['v']['decision'] == 'accept' else 'rejected'})" for e in cut) + ". ")
    sub1 += ("Columns: every decision in order, round by round." if many else
             f"Columns: Pirates 1 to {N} from the left; {pname(r1['proposer'])} proposed.")
    charts = [{"title": "What accepting was worth, vote by vote" if many else "What accepting is worth to each voter", "sub": sub1,
               "height": 200, "fmt": "int", "zero": True,
               "bars": [{"name": "Offered minus if it fails", "color": "c1", "v": worth}], "lines": []}]

    # 2. grid: the pirate, the offer, the share if the plan fails, the vote and the paper's rule. Coins are blue with the number
    #    in the cell, no coins plain gray; every cell is opaque because the page writes cell text in white.
    def coin_cell(g, tip):
        return {"c": "c1", "o": 1, "t": str(g), "tip": tip} if g > 0 else {"c": "muted", "o": 1, "tip": tip}

    round_row, pirate_row, offer_row, fail_row, vote_row, rule_row = ([None] * NX for _ in range(6))
    word = {"accept": "Accept", "reject": "Reject"}
    has_na = False
    for r in rounds:
        p, fb = r["proposer"], fallback(r, G)
        nxt = r["alive"][1] if len(r["alive"]) > 1 else None
        for k, s in [(r["step"], p)] + [(v["step"], v["seat"]) for v in r["votes"]]:
            round_row[slot(k)] = {"c": "ink", "o": .5, "t": f"R{r['round']}",
                                  "tip": f"Round {r['round']}: {numword(len(r['alive']))} pirates aboard, {pname(p)} proposes"}
            pirate_row[slot(k)] = {"c": "ink", "o": .5, "t": str(s + 1),
                                   "tip": f"{pname(s)} (seat P{s}, rank {s + 1})" + (", the proposer" if s == p else "")}
        i = slot(r["step"])
        offer_row[i] = {"c": "c1", "o": 1, "t": str(r["plan"][0]), "tip": f"Proposer: keeps {coins(r['plan'][0])}"}
        fail_row[i] = {"c": "hold", "o": 1, "t": "Out", "tip": "Thrown overboard if this plan fails"}
        vote_row[i] = {"c": "c3", "o": 1, "t": "Own", "tip": "The proposer's own vote counts as accept"}
        for v in r["votes"]:
            i, s, g = slot(v["step"]), v["seat"], v["offered"]
            offer_row[i] = coin_cell(g, f"Offered {coins(g)}")
            ng = fb.get(s, 0)
            fail_row[i] = coin_cell(ng, f"Would propose next and keep {coins(ng)} under the equilibrium split" if s == nxt else
                                    f"Would get {coins(ng)} from {pname(nxt)}'s equilibrium split")
            rv = rule_vote(g, s - p, code_scoring)
            vote_row[i] = {"c": KINDS.get(v["decision"], ["", "hold"])[1], "o": 1, "t": word.get(v["decision"], v["decision"]),
                           "tip": f"Voted {v['decision']} on {coins(g)}" + ("" if rv is None else
                                  ", as the paper's rule says" if rv == v["decision"] else f"; the paper's rule says {rv}")}
            if rv is None:
                has_na = True
                rule_row[i] = {"c": "hold", "o": 1, "t": "n/a", "tip": "One coin at the other parity: the official code credits neither vote"}
            else:
                why = ("two or more coins" if g >= 2 else "no coins" if g == 0 else
                       "one coin, " + ("same" if (s - p) % 2 == 0 else "other") + " parity as the proposer")
                rule_row[i] = {"c": KINDS[rv][1], "o": 1, "t": word[rv],
                               "tip": f"Rule: {rv} ({why})" + (", vote matches" if rv == v["decision"] else ", vote differs")}
    rows = (["Round"] if many else []) + ["Pirate", "Offered", "If it fails", "Vote", "Rule says"]
    cells = ([round_row] if many else []) + [pirate_row, offer_row, fail_row, vote_row, rule_row]
    legend = [["Coins (number in the cell)", "c1"], ["No coins", "muted"], ["Accept", "c3"], ["Reject", "c2"],
              ["Out: thrown overboard" + ("; n/a: no credit" if has_na else ""), "hold"]]
    sub2 = (("Round and Pirate: the round and the pirate of each decision (1 = most senior). " if many else
             "Pirate: the pirate in the column (1 = most senior). ")
            + "Offered: the coins in the plan (the proposer's cell: what it keeps). If it fails: what that pirate would get if the plan "
            "failed and the next pirate proposed the equilibrium split. Vote: each secret vote; the proposer's own counts as accept (Own). "
            "Rule says: the paper's optimal voter, which accepts two or more coins, rejects none and accepts one coin only from a proposer "
            f"an even number of places above; {sc['correct']} of {sc['total']} votes follow it (the paper's S8V"
            + ("" if R == 1 else f", over {R} rounds") + "). "
            + ("Every pirate accepted exactly where Offered is higher than If it fails." if follow == len(entries) else
               f"{follow} of {len(entries)} pirates accepted exactly where Offered is higher than If it fails."))
    charts.append({"title": "Offers and secret votes, pirate by pirate" + (", round by round" if many else ""), "sub": sub2,
                   "height": 170, "grid": {"rows": rows, "cells": cells, "rowHeight": 30, "legend": legend}})

    # 3. tally as the votes came in (seat order), every round: accepts race to the pass mark, rejections to the mark that sinks it
    acc_v, rej_v, need_v, sink_v = ([None] * NX for _ in range(4))
    for r in rounds:
        n_al = len(r["alive"])
        need = (n_al + 1) // 2                  # passes when 2 * accepts >= pirates aboard
        sink = n_al - need + 1                  # this many rejections leave fewer than `need` accepts
        a_, j_, decided = 1, 0, None
        i = slot(r["step"])
        acc_v[i], rej_v[i], need_v[i], sink_v[i] = 1, 0, need, sink
        for v in r["votes"]:
            a_ += v["decision"] == "accept"
            j_ += v["decision"] != "accept"
            i = slot(v["step"])
            acc_v[i], rej_v[i], need_v[i], sink_v[i] = a_, j_, need, sink
            if decided is None and (a_ >= need or j_ >= sink):
                decided = v
        r["need"], r["sink"], r["decided"] = need, sink, decided
    t = final
    need, sink = t["need"], t["sink"]
    sub3 = ("Accepting and rejecting pirates counted so far, in the order the votes were cast (seat order); the proposer counts as the first "
            f"accept. With {numword(len(t['alive']))} aboard" + (f" in round {t['round']}" if many else "")
            + f", the plan needs {need} accepts (at least half), and {sink} rejections would sink it. ")
    if t["accept_count"] is not None:
        sub3 += f"It got {t['accept_count']}"
        if t["accept_count"] == need:
            acc_voters = [v for v in t["votes"] if v["decision"] == "accept"]
            bought = sum(t["plan"][1:]) == t["accept_count"] - 1 and all(v["offered"] == 1 for v in acc_voters)
            sub3 += ": exactly the minimum" + (", every coin given away bought one needed vote. " if bought else ". ")
        else:
            sub3 += (f", {t['accept_count'] - need} more than needed. " if t["passed"] else ", not enough. ")
    dv = t["decided"]
    if dv is not None:
        later = [v["seat"] for v in t["votes"] if v["step"] > dv["step"]]
        acc_d = dv["decision"] == "accept"
        sub3 += (f"{pname(dv['seat'])}'s {'accept' if acc_d else 'reject'} was the {ordinal(need if acc_d else sink)}, so "
                 + (f"the vote of {plist(later)} could no longer change the outcome. " if len(later) == 1 else
                    f"the votes of {plist(later)} could no longer change the outcome. " if later else
                    "the very last vote decided it. "))
    earlier = "; ".join(f"round {r['round']}: {r['accept_count']} of {len(r['alive'])} accepted, {r['need']} needed" for r in rounds[:-1])
    sub3 += (f"Earlier rounds: {earlier}. " if earlier else "") + "Nobody saw this count until all votes were in."
    tally_lines = [{"name": "Needed to pass", "color": "c3", "dash": True, "step": True, "end": True, "v": need_v},
                   {"name": "Rejections that sink it", "color": "c2", "dash": True, "step": True, "end": True, "v": sink_v}]
    top = max(v for v in acc_v + rej_v if v is not None)
    if 8 <= top < 10:
        # counts of 8 or 9 make the page's axis tick at 2.5 pirates; the number aboard (the ceiling of both counts) lifts it to 0, 5, 10
        aboard_v = [None] * NX
        for r in rounds:
            for k in [r["step"]] + [v["step"] for v in r["votes"]]:
                aboard_v[slot(k)] = len(r["alive"])
        tally_lines.append({"name": "Pirates aboard", "color": "hold", "thin": True, "step": True, "end": True, "v": aboard_v})
    charts.append({"title": "The vote count, vote by vote" + (", round by round" if many else ""), "sub": sub3, "height": 210, "fmt": "int",
                   "bars": [{"name": "Accepting so far", "color": "c3", "v": acc_v}, {"name": "Rejecting so far", "color": "c2", "v": rej_v}],
                   "lines": tally_lines})

    # ---- tiles
    fp = final["proposer"]
    tiles = []
    if final["passed"]:
        eqf = equilibrium(final["alive"], G)
        lf = sum(abs(a - b) for a, b in zip(final["plan"], eqf))
        fb = [s for s in final["alive"][1:] if final["offered"][s] > 0]
        tiles.append({"k": f"{pname(fp)} keeps", "v": f"{final['plan'][0]} of {G}",
                      "s": ((f"One coin each to {plist(fb)}" if all(final["offered"][s] == 1 for s in fb) else
                             f"{coins(sum(final['plan'][1:]))} in all to {plist(fb)}") if fb else "Nothing for anyone else")
                           + (": exactly the backward-induction split. " if lf == 0 else f"; {coins(lf)} from the backward-induction split. ")
                           + f"Reward {rew(rewards[fp])}"})
    tiles.append({"k": "Votes" + (f" in round {final['round']}" if many else ""), "v": f"{final['accept_count']} of {len(final['alive'])} accept",
                  "s": f"At least {need} needed, the proposer included; " + (f"the plan passed in round {final['round']}" if final["passed"] else "the plan failed")})
    tiles.append({"k": "Thrown overboard", "v": "Nobody" if not eliminated else f"{len(eliminated)} of {N}",
                  "s": (f"The game ended in round {R}; it can last at most {N - 1} rounds" if not eliminated else
                        f"{plist(eliminated)} lost everything and score {MINUS}1; the game ended in round {R}")})
    tiles.append({"k": "Paper's score S8", "v": f"{sc['s8']:.1f}",
                  "s": (f"Out of 100: the proposal is {sc['s8p']:g} coins from the equilibrium split" if R == 1 else
                        f"Out of 100: proposals {sc['s8p']:.1f} coins from the equilibrium split on average")
                       + f" and {sc['correct']} of {sc['total']} votes follow the rule. The paper's default model averaged 71.6"})

    # ---- text
    if final["passed"]:
        eqf = equilibrium(final["alive"], G)
        title = ((f"After {plist(eliminated)} went overboard, " if eliminated else "")
                 + f"{pname(fp)} proposes " + ("the equilibrium split" if final["plan"] == eqf else "a split")
                 + f", keeps {final['plan'][0]} of {G} coins and passes it with {final['accept_count']} of {len(final['alive'])} votes"
                 + (", the bare minimum" if final["accept_count"] == need else ""))
    else:
        title = f"The game stopped in round {R} without a plan passing"
    model = (b.get("meta") or {}).get("model") or run.get("model") or ""
    model = model.split("/")[-1] if model else "the model"
    date = str((b.get("meta") or {}).get("recorded_at") or run.get("recorded_at") or DEFAULT_DATE)[:10]
    seed = run.get("seed", cfg.get("seed"))
    lede = (f"{numword(N).capitalize()} pirates, ranked by seniority, must split {G} gold coins. The most senior pirate aboard proposes a split and the others vote "
            "in secret: if at least half of those aboard accept (the proposer counts as accepting), the coins are paid as proposed; otherwise "
            "the proposer is thrown overboard and the next pirate proposes. Each pirate scores its coins " + DIV + f" {G}"
            + (f" plus {bonus:g} for every pirate thrown overboard" if bonus else "") + f", or {MINUS}1 if thrown overboard itself, so staying "
            "aboard comes first, then coins.")
    from_prompt = sum(1 for s in steps if s["state"] and not s["state"].startswith("(Rebuilt"))
    foot = (f"Data: the {date} recording of eco_pirate_game (seed {seed}, {N} pirates, {G} coins, overboard bonus {bonus:g}), every pirate "
            f"played by {model}; every number on this page is computed from its game log. "
            + ("Observations and reasoning come from the saved model calls. " if has_reason else
               "This recording was made before reasoning was saved; the game will be re-recorded with reasoning. ")
            + ("" if from_prompt == len(steps) else
               ("The observation of the first proposal is the saved prompt; the others are rebuilt from the log. " if from_prompt else
                "The observations are rebuilt from the log. "))
            + f"Pirate k is seat P(k{MINUS}1) in the log and in the prompts. "
            + ("The horizontal axis is the order of the decisions: R2 · Pirate 5 is Pirate 5's vote in round 2. " if many else
               "The horizontal axis is the pirates in seniority order, which is also the order of the decisions. ")
            + "The equilibrium split is the backward-induction solution of Stewart (1999); the voter rule and the scores are those of GAMA-Bench "
            f"(Huang et al., ICLR 2025, game 8): S8P = mean distance of each proposal from the equilibrium split in coins, S8V = share of votes "
            f"that follow the rule, S8 = 50 {TIMES} (2G {MINUS} S8P) {DIV} 2G + 50 {TIMES} S8V"
            + (" (the official code's version: shares rescaled to 100 coins, and no credit for one coin at the other parity)" if code_scoring else "")
            + ". The paper's default model scored 71.6 on average over five runs (Table 4), and its 13-model leaderboard ranges from 57.4 to 87.9.")
    d = {"id": GID, "name": "Pirate game", "title": title, "lede": lede, "seatNames": names, "tiles": tiles, "foot": foot,
         "x": {"n": NX, "labels": xlabels}, "charts": charts, "kinds": {k: KINDS[k] for k in KINDS if any(s["kind"] == k for s in steps)},
         "steps": steps, "feed": feed, "feedTitle": "Proposal and votes",
         "feedSub": "The plan, then every vote in the order it was cast. Votes were secret: during the game only the final count was announced. "
                    "The selected step is highlighted; later entries are hidden."}
    d["eyebrow"] = (common.EYEBROW.format(gid=GID, date=date) if has_reason
                    else f"social_game_env \u00b7 {GID} \u00b7 {model} \u00b7 recorded {date} \u00b7 reasoning not saved")
    return d


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        _selftest()
        raise SystemExit(0)
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    common.write(build(src, calls), out, date=recorded_date(src))
