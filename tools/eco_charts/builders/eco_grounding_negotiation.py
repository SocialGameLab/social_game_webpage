"""Replay data for eco_grounding_negotiation (the "Talk is Cheap" shared-supply negotiation): two agents buy wood, stone and
gold (themed names) from one pool with a fixed supply per round; each has three private projects that turn resources into
points; a round is cheap talk (a party may lock in its order at any time) and then simultaneous, secret orders; if the two
orders together exceed the supply of any resource the round is annulled and both earn 0.

usage: python3 -I eco_grounding_negotiation.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE   a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, transcript.md), or the rendered
           Markdown replay of one (replays_*/eco_grounding_negotiation.md); for a Markdown file the raw bundle next to it
           (raw/eco_grounding_negotiation/ or the file's own directory) is used when it exists.
  OUT_DIR  where <OUT_DIR>/eco_grounding_negotiation.json is written (via common.write).
  CALLS    optional model-call log (JSON lines: prompt or messages, reasoning, answer), in time order. Each call is tied to
           the step whose seat and move it matches; its reasoning becomes the step's thinking and its prompt the observation.

Every number shown comes from the game log: the moves and what each seat saw (game_log.jsonl) and the per-round results the
engine wrote (summary.json, info.history and info.scenarios). The round results are re-derived from the moves with the
game's resolution rule (rules/eco_grounding_negotiation_rules.md, Round Procedure 4) and any disagreement is printed; the
solo and joint optima are re-derived by checking every legal order and compared with the engine's. When the summary is
missing (a Markdown source without its bundle) the re-derived values are used. Who locked in first (and at which exchange)
is read from the order of the moves; the best total over plans that leave both sides at least what the worse-off side got
comes from the same check of every legal pair of orders."""
import itertools, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_grounding_negotiation"
DEFAULT_DATE = "2026-11-09"
MINUS, TIMES = "\u2212", "\u00d7"
KINDS_ALL = {"say": ["Message", "c4"], "lock": ["Order, locked in during talk", "c3"], "order": ["Order after the talk", "c1"],
             "empty": ["Empty order (fallback)", "muted"]}
SIDE_COLOR = {"A": "c1", "B": "c2"}
# The paper's outcomes at each compatibility bucket (Yao, Zou and Hawkins 2026, Table 3, as quoted in
# rules/eco_grounding_negotiation_rules.md, Phenomenon).
PAPER = {0.5: "In the paper, its three models reached 56.8% to 68.2% efficiency in self-play at this ratio, and 24.7% of its "
              "rounds at this ratio were overdrawn.",
         0.8: "In the paper, its three models reached 61.2% to 85.6% efficiency in self-play at this ratio.",
         1.0: "In the paper, its three models reached 76.1% to 94.0% efficiency in self-play at this ratio, and 6.9% of its "
              "rounds at this ratio were overdrawn."}
TRUNC = "\uff08\u540e\u9762\u7701\u7565\uff09"     # the Markdown renderer's "(rest omitted)" tag


# ------------------------------------------------------------------------------------------------ loading
def _md_source(path):
    """A rendered Markdown replay: one block per step ('## <step n> \u00b7 Pk', the state the seat saw, the move it made and,
    when recorded, its reasoning in ````text blocks), the seat rewards in the header and the game_over event at the end."""
    text = open(path, encoding="utf-8").read().replace("\u2026" + TRUNC, "\u2026 (truncated)").replace(TRUNC, "(truncated)")
    events = [{"event": "setup"}]
    calls = []
    for m in re.finditer(r"\n## \u7b2c (\d+) \u6b65 \u00b7 P(\d+)\n(.*?)(?=\n## |\Z)", text, re.S):
        blk, seat = m.group(3), int(m.group(2))
        st = re.search(r"\*\*\u770b\u5230\uff1a\*\*\n\n```\n(.*?)\n```", blk, re.S)
        act = re.findall(r"\*\*\u505a\u4e86\uff1a\*\* `(.*?)`\s*\n", blk)
        th = re.findall(r"````text\n(.*?)\n````", blk, re.S)
        if not act:
            continue
        events.append({"event": "move", "actor": seat, "move": act[-1], "view": st.group(1) if st else ""})
        if th:
            calls.append({"seat": seat, "prompt": "", "answer": act[-1], "reasoning": th[-1], "rescued": False})
    head = text.split("\n## ")[0]
    rw = re.search(r"(\{\"0\":[^}]*\})", head)
    seed = re.search(r"\u79cd\u5b50 (\d+)", head)
    model = re.search(r"\u6a21\u578b (\S+)", head)
    over = re.search(r"```\n(\{\"event\": \"game_over\".*?\})\n```", text, re.S)
    if over:
        events.append(json.loads(over.group(1)))
    summary = {"rewards": json.loads(rw.group(1)) if rw else {}, "seed": int(seed.group(1)) if seed else None}
    return {"events": events, "summary": summary, "run": {"seed": summary["seed"], "model": model.group(1) if model else ""},
            "config": {}, "transcript": "", "meta": {}, "md_calls": calls}


def load_source(src):
    if os.path.isdir(src):
        b = common.load_bundle(src)
        meta = os.path.join(src, "meta.json")
        b["meta"] = json.load(open(meta, encoding="utf-8")) if os.path.exists(meta) else {}
        b["md_calls"] = []
        return b
    here = os.path.dirname(os.path.abspath(src))
    for d in (os.path.join(here, "raw", GID), here):
        if os.path.exists(os.path.join(d, "game_log.jsonl")):
            return load_source(d)
    return _md_source(src)


# ------------------------------------------------------------------------------------------------ parsing what a seat saw
def parse_settings(view):
    sup = re.search(r"supply per round (\{[^}]*\})", view)
    cost = re.search(r"unit costs (\{[^}]*\})", view)
    bud = re.search(r"your budget per round ([\d.]+)", view)
    mt = re.search(r"at most (\d+) different resources", view)
    talk = re.search(r"Cheap talk: up to (\d+) messages per party", view)
    if not (sup and cost and bud):
        return None
    supply = json.loads(sup.group(1))
    return {"resources": list(supply), "supply": {k: int(v) for k, v in supply.items()}, "cost": json.loads(cost.group(1)),
            "budget": float(bud.group(1)), "max_types": int(mt.group(1)) if mt else 2, "talk_turns": int(talk.group(1)) if talk else None}


def parse_projects(view, header="Your projects this round:"):
    """[(name, [(resource, units), ...], reward)] in the listed order."""
    i = view.find(header)
    if i < 0:
        return None
    out = []
    for line in view[i + len(header):].split("\n")[1:]:
        m = re.match(r"\s+- (.+?): requires \[([^\]]*)\], reward = (\d+(?:\.\d+)?) per run", line)
        if not m:
            break
        req = []
        for part in m.group(2).split(","):
            q = re.match(r"\s*(.+?) x(\d+)\s*$", part)
            if q:
                req.append((q.group(1), int(q.group(2))))
        rw = float(m.group(3))
        out.append((m.group(1), req, int(rw) if rw.is_integer() else rw))
    return out or None


def parse_role(view):
    m = re.search(r"Your role: (\w+)", view)
    return m.group(1) if m else None


def parse_now(view):
    m = re.search(r"Now: cheap talk, exchange (\d+) of (\d+)", view)
    if m:
        return "talk", int(m.group(1))
    if re.search(r"\nNow: ", view):
        return "decide", None
    return None, None


# ------------------------------------------------------------------------------------------------ game arithmetic
def runs_affordable(left, req):
    return min(left.get(r, 0) // u for r, u in req) if req else 0


def resolve_runs(purchase, requested, projects):
    """The game's run rule: requested runs first (each cut to what the purchase still affords), then the leftover resources
    go to the projects in the listed order."""
    left = dict(purchase)
    runs = {}
    for name, req, _ in projects:
        k = min(int(requested.get(name, 0) or 0), runs_affordable(left, req))
        if k:
            runs[name] = k
            for r, u in req:
                left[r] -= k * u
    for name, req, _ in projects:
        k = runs_affordable(left, req)
        if k:
            runs[name] = runs.get(name, 0) + k
            for r, u in req:
                left[r] -= k * u
    return runs, left


def points(runs, projects):
    rw = {name: r for name, _, r in projects}
    return sum(k * rw[n] for n, k in runs.items())


def best_points(purchase, projects):
    """The most a purchase can earn with the best choice of runs, and those runs."""
    bounds = [range(runs_affordable(purchase, req) + 1) for _, req, _ in projects]
    best, arg = 0, {}
    for combo in itertools.product(*bounds):
        use = {}
        for k, (_, req, _) in zip(combo, projects):
            for r, u in req:
                use[r] = use.get(r, 0) + k * u
        if all(use.get(r, 0) <= purchase.get(r, 0) for r in use):
            p = sum(k * rw for k, (_, _, rw) in zip(combo, projects))
            if p > best:
                best, arg = p, {projects[i][0]: k for i, k in enumerate(combo) if k}
    return best, arg


def legal_orders(st):
    res, out = st["resources"], []
    for q in itertools.product(*[range(st["supply"][r] + 1) for r in res]):
        if sum(1 for v in q if v) > st["max_types"]:
            continue
        if sum(v * float(st["cost"][r]) for v, r in zip(q, res)) > st["budget"] + 1e-3:
            continue
        out.append(dict(zip(res, q)))
    return out


def oracle(st, proj_a, proj_b):
    """Solo optima of both sides and the joint optimum over every legal pair of orders that fits the supply, with, for each
    side, the joint-optimal plan that favours it using the fewest units, and the set of every (A points, B points) pair some
    such pair of orders reaches (for the split statements: the best total when both sides get at least a given amount)."""
    orders = legal_orders(st)
    val_a = [(o, *best_points(o, proj_a)) for o in orders]
    val_b = [(o, *best_points(o, proj_b)) for o in orders]
    val_a = [v for v in val_a if v[1] > 0] + [({r: 0 for r in st["resources"]}, 0, {})]
    val_b = [v for v in val_b if v[1] > 0] + [({r: 0 for r in st["resources"]}, 0, {})]
    solo_a, solo_b = max(v[1] for v in val_a), max(v[1] for v in val_b)
    best, plans, pairs = -1, [], set()
    for oa, pa, ra in val_a:
        for ob, pb, rb in val_b:
            if any(oa[r] + ob[r] > st["supply"][r] for r in st["resources"]):
                continue
            pairs.add((pa, pb))
            if pa + pb > best:
                best, plans = pa + pb, []
            if pa + pb == best:
                plans.append((oa, pa, ra, ob, pb, rb))
    pick = {}
    for p in plans:
        side = "A" if p[1] > p[4] else "B" if p[4] > p[1] else "="
        units = sum(p[0].values()) + sum(p[3].values())
        if side not in pick or units < pick[side][0]:
            pick[side] = (units, p)
    return solo_a, solo_b, best, [pick[s][1] for s in ("A", "B", "=") if s in pick], pairs


# ------------------------------------------------------------------------------------------------ text helpers
def num(v):
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}"


def order_text(o, res, empty="nothing"):
    parts = [f"{o[r]} {r}" for r in res if o.get(r)]
    return " + ".join(parts) if parts else empty


def req_text(req):
    return " + ".join(f"{u} {r}" for r, u in req)


def runs_text(runs, projects, empty="no run"):
    order = [p[0] for p in projects]
    parts = [f"{n} {TIMES}{runs[n]}" for n in sorted(runs, key=lambda n: order.index(n) if n in order else 99) if runs[n]]
    return ", ".join(parts) if parts else empty


def pts_w(v):
    return f"{num(v)} point" + ("" if v == 1 else "s")


def pct(v, d=1):
    return f"{100 * v:.{d}f}%"


def short_label(s, n=11):
    s = s.replace("_", " ")
    s = s[:1].upper() + s[1:]
    return s if len(s) <= n else s[:n - 1] + "."


def _decode_answer(s):
    """The move object in a model answer: the whole answer as JSON (code fences stripped), else the first JSON object in the
    text that has an action_id (answers often wrap the JSON in a sentence such as 'Here is my move: {...}')."""
    s = s.strip()
    try:
        return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", s))
    except ValueError:
        pass
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", s):
        try:
            o, _ = dec.raw_decode(s, m.start())
        except ValueError:
            continue
        if isinstance(o, dict) and "action_id" in o:
            return o
    return None


def _canon(o):
    """A move or an answer as a comparable object: action_id, talk (stripped), purchase and runs as integers."""
    if isinstance(o, str):
        o = _decode_answer(o)
    if not isinstance(o, dict):
        return None
    p = o.get("payload") or {}
    out = {"action_id": o.get("action_id"), "talk": (p.get("talk") or "").strip()}
    for k in ("purchase", "runs"):
        if isinstance(p.get(k), dict):
            try:
                out[k] = {a: int(v) for a, v in p[k].items() if int(v)}
            except (TypeError, ValueError):
                out[k] = p[k]
    return out


# ------------------------------------------------------------------------------------------------ calls -> steps
def attach_calls(steps, calls, fallback_steps=(), fallbacks_unplaced=0):
    """Tie calls to steps in time order. A step's window is the run of calls from the pointer whose seat is the step's seat
    (or unknown); the first call in it whose answer is the step's move is the one, and earlier calls in the window were failed
    attempts whose reasoning is kept above it. With no matching answer the window's calls up to the next same-seat step's
    match are used. A note that the engine played a fallback is added only when the engine recorded one: for the steps in
    `fallback_steps` (placed by the round results), or, when `fallbacks_unplaced` fallbacks or invalid actions were counted
    but not placed, as a possibility; an answer that merely fails to match gets no note. Returns how the calls were used:
    steps with a call, steps whose call answer is the move played, earlier attempts kept, follow-up answers."""
    ptr = 0
    stats = {"calls": len(calls), "steps": 0, "matched": 0, "earlier": 0, "rescued": 0}
    for si, st in enumerate(steps):
        j = ptr
        while j < len(calls) and calls[j].get("seat") in (None, st["seat"]):
            j += 1
        window = list(range(ptr, j))
        if not window:
            continue
        want = _canon(st["act"])
        hit = next((k for k in window if _canon(calls[k].get("answer")) == want), None)
        if hit is None:
            nxt = next((s for s in steps[si + 1:] if s["seat"] == st["seat"]), None)
            stop = next((k for k in window if nxt is not None and _canon(calls[k].get("answer")) == _canon(nxt["act"])), None)
            use = (stop - 1) if stop is not None and stop > window[0] else window[0]
        else:
            use = hit
        parts = []
        for k in range(ptr, use):
            c = calls[k]
            parts.append(f"[Earlier attempt, answer: {(c.get('answer') or '')[:300]}]\n{c.get('reasoning') or '(no reasoning returned)'}")
        c = calls[use]
        parts.append(("[Final attempt]\n" if parts else "") + (c.get("reasoning") or ("(no reasoning returned)" if parts else "")))
        if hit is None and si in fallback_steps:
            parts.append(f"[The engine played a fallback move instead of this answer: {(c.get('answer') or '')[:300]}]")
        elif hit is None and fallbacks_unplaced:
            parts.append(f"[This answer does not match the move played; the engine counted {fallbacks_unplaced} fallback or "
                         f"invalid move(s) in this game, and this may be one: {(c.get('answer') or '')[:300]}]")
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        st["thinking"] = "\n\n".join(p for p in parts if p).strip()
        if c.get("prompt"):
            st["state"] = common.state_part(c["prompt"])
        stats["steps"] += 1
        stats["matched"] += hit is not None
        stats["earlier"] += use - ptr
        stats["rescued"] += bool(c.get("rescued"))
        ptr = use + 1
    return stats


# ------------------------------------------------------------------------------------------------ build
def build(bundle_dir, calls=None):
    b = load_source(bundle_dir)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    calls = list(calls or [])
    from_md = not calls and bool(b.get("md_calls"))     # reasoning read from a rendered Markdown replay, not a call log
    calls = calls or b.get("md_calls") or []
    ev, summ, run = b["events"], b["summary"] or {}, b["run"] or {}
    info = summ.get("info") or {}
    hist = info.get("history") or []
    scen = info.get("scenarios") or []
    metrics = info.get("metrics") or {}
    params = info.get("params") or {}
    warn = []

    moves = [e for e in ev if e.get("event") == "move"]
    if not moves:
        raise SystemExit("no moves in the log")
    st = None
    for e in moves:
        st = st or parse_settings(e.get("view") or "")
    if st is None:
        raise SystemExit("the game settings were not found in any observation")
    res = st["resources"]

    # ---- seats and roles
    roles = {}
    for e in moves:
        r = parse_role(e.get("view") or "")
        if r:
            roles.setdefault(int(e["actor"]), r)
    seat_a = metrics.get("seat_a", next((s for s, r in roles.items() if r == "AGENT_A"), 0))
    seats_b = metrics.get("seats_b") or sorted(s for s, r in roles.items() if r == "AGENT_B") or [1]
    n_seats = max([seat_a] + list(seats_b) + [int(e["actor"]) for e in moves]) + 1

    # ---- decisions, grouped into game rounds (a round ends when both active seats have ordered)
    decs, rnd, ordered = [], 1, set()
    for e in moves:
        seat, view = int(e["actor"]), e.get("view") or ""
        mv = json.loads(e["move"]) if isinstance(e["move"], str) else e["move"]
        p = mv.get("payload") or {}
        phase, exch = parse_now(view)
        m = re.search(r"Round (\d+) of (\d+)\.", view)
        if m and seat == seat_a and int(m.group(1)) != rnd:
            warn.append(f"seat {seat} saw round {m.group(1)} where the move count gives round {rnd}")
        decs.append({"seat": seat, "round": rnd, "move": mv, "aid": mv.get("action_id"), "talk": (p.get("talk") or "").strip(),
                     "purchase": {r: int((p.get("purchase") or {}).get(r, 0) or 0) for r in res} if mv.get("action_id") == "submit" else None,
                     "runs": {k: int(v or 0) for k, v in (p.get("runs") or {}).items()}, "phase": phase, "exchange": exch, "view": view,
                     "projects": parse_projects(view)})
        if mv.get("action_id") == "submit":
            ordered.add(seat)
            if len(ordered) >= 2:
                rnd, ordered = rnd + 1, set()
    for d in decs:
        if d["exchange"] is None and d["phase"] != "decide":
            d["exchange"] = sum(1 for e in decs if e["round"] == d["round"] and e["seat"] == d["seat"] and e is not d
                                and decs.index(e) < decs.index(d)) + 1
    R_played = max(d["round"] for d in decs if d["purchase"] is not None) if any(d["purchase"] is not None for d in decs) else 1
    R = int(params.get("num_rounds") or metrics.get("rounds") or R_played)
    xn = max(R, max(d["round"] for d in decs))

    # ---- per-round facts: who played, their projects, orders, results
    rounds = []
    for r in range(1, xn + 1):
        rd = [d for d in decs if d["round"] == r]
        h = next((x for x in hist if int(x.get("round", 0)) == r), None)
        sc = next((x for x in scen if int(x.get("round", 0)) == r), {})
        sa = int(sc.get("seat_a", seat_a))
        sb = int(sc.get("seat_b", next((d["seat"] for d in rd if d["seat"] != sa), seats_b[min(r - 1, len(seats_b) - 1)])))
        proj = {s: next((d["projects"] for d in rd if d["seat"] == s and d["projects"]), None) for s in (sa, sb)}
        sub = {s: next((d for d in rd if d["seat"] == s and d["purchase"] is not None), None) for s in (sa, sb)}
        order = {s: (sub[s]["purchase"] if sub[s] else {x: 0 for x in res}) for s in (sa, sb)}
        demand = {x: order[sa][x] + order[sb][x] for x in res}
        over = any(demand[x] > st["supply"][x] for x in res)
        mine = {}
        for s in (sa, sb):
            if proj[s] and sub[s]:
                rr, left = resolve_runs(order[s], sub[s]["runs"], proj[s])
                mine[s] = {"runs": rr, "left": left, "pts": 0 if over else points(rr, proj[s])}
        if h:
            hp = {int(k): {x: int(v.get(x, 0)) for x in res} for k, v in (h.get("purchases") or {}).items()}
            for s in (sa, sb):
                if s in hp and hp[s] != order[s]:
                    warn.append(f"round {r}: seat {s} ordered {order[s]} but the summary says {hp[s]}")
            runs = {int(k): {n: int(v) for n, v in (h.get("runs") or {}).get(k, {}).items() if int(v)} for k in (h.get("runs") or {})}
            pts = {int(k): v for k, v in (h.get("rewards") or {}).items()}
            over_h = bool(h.get("overdrawn"))
            if over_h != over:
                warn.append(f"round {r}: overdraw {over} from the orders but {over_h} in the summary")
            over = over_h
            for s in (sa, sb):
                if s in mine and (mine[s]["runs"] != runs.get(s, {}) or mine[s]["pts"] != pts.get(s)):
                    warn.append(f"round {r}: seat {s} re-derived runs {mine[s]['runs']} / {mine[s]['pts']} points, summary {runs.get(s)} / {pts.get(s)}")
            fallback = set(int(x) for x in (h.get("fallback_decisions") or []) if str(x).isdigit())
        else:
            runs = {s: mine[s]["runs"] for s in mine}
            pts = {s: mine[s]["pts"] for s in mine}
            fallback = set()
        left = {}
        for s in (sa, sb):
            if proj[s]:
                used = {x: 0 for x in res}
                for n, k in runs.get(s, {}).items():
                    for x, u in next((p[1] for p in proj[s] if p[0] == n), []):
                        used[x] += k * u
                left[s] = {x: order[s][x] - used[x] for x in res}
        played = bool(h is not None or (sub[sa] and sub[sb]))
        rounds.append({"r": r, "sa": sa, "sb": sb, "proj": proj, "order": order, "demand": demand, "over": over, "runs": runs,
                       "pts": pts, "left": left, "fallback": fallback, "sc": sc, "played": played})

    # ---- optima: re-derived per distinct scenario, compared with the engine's
    opt_cache = {}
    for rr in rounds:
        pa, pb = rr["proj"][rr["sa"]], rr["proj"][rr["sb"]]
        if not (pa and pb):
            rr["opt"] = None
            continue
        key = json.dumps([pa, pb])
        if key not in opt_cache:
            opt_cache[key] = oracle(st, pa, pb)
        sa_v, sb_v, M, plans, pairs = opt_cache[key]
        sc = rr["sc"]
        if sc:
            for k, mine in (("solo_a", sa_v), ("solo_b", sb_v), ("joint_optimum", M)):
                if sc.get(k) is not None and int(sc[k]) != mine:
                    warn.append(f"round {rr['r']}: {k} re-derived {mine}, engine {sc[k]}")
        ok = not sc or all(sc.get(k) is None or int(sc[k]) == v for k, v in (("solo_a", sa_v), ("solo_b", sb_v), ("joint_optimum", M)))
        rr["opt"] = {"solo_a": int(sc.get("solo_a", sa_v)), "solo_b": int(sc.get("solo_b", sb_v)), "M": int(sc.get("joint_optimum", M)),
                     "plans": plans if ok else [], "id": sc.get("id")}
        if ok:
            # the split facts: the most the worse-off side can get in any plan, what it gets at the optimum, and the best
            # total over plans that leave both sides at least what the worse-off side actually got this round
            lo_got = min(rr["pts"].get(rr["sa"], 0), rr["pts"].get(rr["sb"], 0))
            rr["opt"].update({"maxmin": max(min(p) for p in pairs), "low_at_M": max(min(p) for p in pairs if sum(p) == M),
                              "floor": lo_got, "floor_best": max([sum(p) for p in pairs if min(p) >= lo_got] or [M])})

    # ---- lock-in order: the round's first order, if it was placed during the talk (an order after the talk is simultaneous)
    for rr in rounds:
        rd = [d for d in decs if d["round"] == rr["r"]]
        subs = [d for d in rd if d["purchase"] is not None]
        first = subs[0] if subs and subs[0]["phase"] != "decide" else None
        rr["first_lock"] = first
        rr["first_speaker"] = int(rr["sc"].get("first_speaker", rd[0]["seat"] if rd else rr["sa"]))
        if first is not None:
            before = rd[:rd.index(first)]
            rr["msgs_before"] = sum(1 for d in before if d["aid"] == "say")
            rr["reply_to_other"] = bool(before) and before[-1]["seat"] != first["seat"] and before[-1]["aid"] == "say"
            rr["own_msgs_before"] = sum(1 for d in before if d["aid"] == "say" and d["seat"] == first["seat"])
        rr["second_sub"] = next((d for d in subs if first is None or d is not first), None)

    for w in warn:
        print("WARNING:", w, file=sys.stderr)

    # ---- names
    shifting = len(set(seats_b)) > 1
    seat_names = []
    for s in range(n_seats):
        if s == seat_a:
            seat_names.append("Agent A")
        elif shifting and s in seats_b:
            seat_names.append(f"Agent B, R{seats_b.index(s) + 1}")
        elif roles.get(s) == "AGENT_B" or s in seats_b:
            seat_names.append("Agent B")
        else:
            seat_names.append(f"Seat {s}")
    A_name, B_name = "Agent A", "Agent B"

    def proj_label(name):
        m = re.match(r"project_(\w+)$", name)
        return m.group(1) if m else name

    # ---- steps and feed
    used_kinds, steps, feed = set(), [], []
    played_rounds = [rr for rr in rounds if rr["played"]]
    first = rounds[0]
    pa0, pb0 = first["proj"][first["sa"]], first["proj"][first["sb"]]
    fixed = len({json.dumps([rr["proj"][rr["sa"]], rr["proj"][rr["sb"]]]) for rr in played_rounds}) == 1
    if pa0 and pb0:
        o = first["opt"] or {}
        setup = (f"Each agent sees only its own projects (needs per run \u2192 points). {A_name}: "
                 + "; ".join(f"{n} ({req_text(q)}) \u2192 {num(w)}" for n, q, w in pa0)
                 + f". {B_name}: " + "; ".join(f"{n} ({req_text(q)}) \u2192 {num(w)}" for n, q, w in pb0) + ".")
        if o:
            setup += (f" Alone, {A_name} could earn at most {o['solo_a']} a round and {B_name} {o['solo_b']}; together they could earn "
                      f"at most {o['M']}." + (f" The projects stay the same in all {R} rounds." if fixed else " The projects change every round."))
        feed.append({"x": 0, "step": 0, "who": "Setup", "text": setup})
    last_of_round = {}
    for k, d in enumerate(decs):
        last_of_round[d["round"]] = k
    for k, d in enumerate(decs):
        r, x, seat = d["round"], d["round"] - 1, d["seat"]
        rr = rounds[x]
        proj = d["projects"] or rr["proj"].get(seat) or []
        if d["aid"] == "say":
            kind = "say"
            head = f"Round {r}, exchange {d['exchange']} \u00b7 message" if d["exchange"] else f"Round {r} \u00b7 message"
            if not d["talk"]:
                head += " (empty)"
            text = f"\u201c{d['talk']}\u201d" if d["talk"] else "(an empty message)"
        else:
            o = d["purchase"]
            empty = not any(o.values())
            if seat in rr["fallback"] and empty:
                kind = "empty"
            elif d["phase"] == "decide":
                kind = "order"
            else:
                kind = "lock"
            req_runs = {n: v for n, v in d["runs"].items() if v}
            what = order_text(o, res)
            if kind == "lock":
                head = (f"Round {r}, exchange {d['exchange']} \u00b7 " if d["exchange"] else f"Round {r} \u00b7 ") + f"locks in {what}"
                verb = "Locks in"
            elif kind == "order":
                head = f"Round {r}, after the talk \u00b7 orders {what}"
                verb = "Orders"
            else:
                head = f"Round {r} \u00b7 no valid order: empty order played"
                verb = "Fallback: orders"
            cost = sum(o[x_] * float(st["cost"][x_]) for x_ in res)
            text = (f"{verb} {what} (cost {num(cost)} of {num(st['budget'])})"
                    + (f"; asks to run {runs_text(req_runs, proj)}" if req_runs else "; leaves the runs to the game")
                    + "." + (f" \u201c{d['talk']}\u201d" if d["talk"] and kind != "empty" else ""))
        used_kinds.add(kind)
        mv = d["move"]
        p = dict(mv.get("payload") or {})
        pay = {}
        if "purchase" in p:
            pay["purchase"] = {x_: p["purchase"].get(x_, 0) for x_ in res if x_ in p["purchase"]}
            pay["purchase"].update({x_: v for x_, v in p["purchase"].items() if x_ not in pay["purchase"]})
        if "runs" in p:
            pay["runs"] = p["runs"]
        pay.update({kk: v for kk, v in p.items() if kk not in pay})
        steps.append({"seat": seat, "x": x, "head": head, "kind": kind,
                      "act": json.dumps({"action_id": mv.get("action_id"), "payload": pay}, ensure_ascii=False),
                      "state": common.state_part(d["view"]) if d["view"] else "Not recorded.", "thinking": ""})
        feed.append({"x": x, "step": k, "who": seat_names[seat] if seat < len(seat_names) else f"Seat {seat}", "kind": kind, "text": text})
        # the round's result, right after its last order
        if d["purchase"] is not None and k == last_of_round[r] and rr["played"]:
            sa, sb = rr["sa"], rr["sb"]
            parts = []
            for x_ in res:
                if rr["demand"][x_]:
                    parts.append(f"{x_} {rr['order'][sa][x_]} + {rr['order'][sb][x_]} = {rr['demand'][x_]} of {st['supply'][x_]}")
            txt = ("Orders together: " + "; ".join(parts) + ". ") if parts else "Nobody ordered anything. "
            if rr["over"]:
                bad = [x_ for x_ in res if rr["demand"][x_] > st["supply"][x_]]
                txt += f"Over the supply of {', '.join(bad)}: the round is annulled and both earn 0."
            else:
                for s, nm in ((sa, A_name), (sb, B_name)):
                    pj = rr["proj"].get(s) or []
                    lf = {x_: v for x_, v in (rr["left"].get(s) or {}).items() if v}
                    txt += (f"{nm}: {runs_text(rr['runs'].get(s, {}), pj)}, {pts_w(rr['pts'].get(s, 0))}"
                            + (f" ({order_text(lf, res)} bought but unused)" if lf else "") + ". ")
                if rr["opt"]:
                    j = rr["pts"].get(sa, 0) + rr["pts"].get(sb, 0)
                    txt += f"Together {j} of the {rr['opt']['M']} possible ({pct(j / rr['opt']['M'])})."
            feed.append({"x": x, "step": k, "who": f"Result R{r}", "text": txt.strip()})

    # fallbacks the engine recorded: placed per step by the round results, and the game's totals (a fallback or invalid move
    # the round results do not place is only flagged as a possibility on a step whose answer does not match its move)
    fallback_steps = {k for k, d in enumerate(decs) if d["purchase"] is not None and d["seat"] in rounds[d["round"] - 1]["fallback"]}
    fb_total = max(int(summ.get("fallbacks") or 0), int((run.get("extras") or {}).get("fallbacks") or 0),
                   int(metrics.get("invalid_actions") or 0))
    call_stats = attach_calls(steps, calls, fallback_steps, max(0, fb_total - len(fallback_steps)))
    has_thinking = any(s["thinking"] for s in steps)

    # ---- series
    def per_round(f):
        return [f(rr) if rr["played"] else None for rr in rounds]
    ptsA = per_round(lambda rr: rr["pts"].get(rr["sa"], 0))
    ptsB = per_round(lambda rr: rr["pts"].get(rr["sb"], 0))
    joint = per_round(lambda rr: rr["pts"].get(rr["sa"], 0) + rr["pts"].get(rr["sb"], 0))
    Ms = [rr["opt"]["M"] if rr["opt"] else None for rr in rounds]
    fair = [None if v is None else v / 2 for v in Ms]
    labels = [f"Round {i + 1}" for i in range(xn)]
    # every played round gave both agents the same points as the round before: a per-round chart would only repeat one
    # fact, so the points chart shows running totals and a talk chart (which does change) is added
    flat = (sum(1 for v in ptsA if v is not None) > 1 and len({v for v in ptsA if v is not None}) == 1
            and len({v for v in ptsB if v is not None}) == 1 and all(v is not None for v in Ms))

    def running(v):
        out, t = [], 0
        for x_ in v:
            t = t + x_ if x_ is not None else t
            out.append(t if x_ is not None else None)
        return out
    # cheap-talk messages per round and agent ("say" moves; the note sent with an order is not counted)
    talk_n = {side: [sum(1 for d in decs if d["round"] == rr["r"] and d["aid"] == "say" and d["seat"] == rr["s" + side.lower()])
                     if rr["played"] else None for rr in rounds] for side in ("A", "B")}

    # supply grid: one row per resource
    status_used = set()
    cells_sup = []
    for x_ in res:
        row = []
        for rr in rounds:
            if not rr["played"]:
                row.append(None)
                continue
            a, b_, t, sp = rr["order"][rr["sa"]][x_], rr["order"][rr["sb"]][x_], rr["demand"][x_], st["supply"][x_]
            base = f"{A_name} {a} + {B_name} {b_} = {t} of {sp}"
            if t > sp:
                c, tag, note = "c2", "over", f"{base}: {t - sp} over the supply, the round is annulled"
            elif t == sp and t:
                c, tag, note = "c4", "full", f"{base}: the whole supply is used"
            elif t:
                c, tag, note = "c3", "spare", f"{base}: {sp - t} left in the pool"
            else:
                c, tag, note = "muted", "none", f"nobody ordered any (supply {sp})"
            status_used.add(tag)
            cell = {"c": c, "tip": note}
            if t:
                cell["t"] = f"{a} + {b_} = {t}/{sp}"
            else:
                cell["o"] = 0.35
            row.append(cell)
        cells_sup.append(row)
    sup_legend = [lab for tag, lab in (("spare", ["Some left in the pool", "c3"]), ("full", ["Whole supply used", "c4"]),
                                        ("over", ["Over the supply: round annulled", "c2"]), ("none", ["Nobody ordered", "muted"])) if tag in status_used]

    # projects grid: one row per (side, project slot)
    n_proj = max(len(rr["proj"].get(s) or []) for rr in rounds for s in (rr["sa"], rr["sb"]))
    proj_rows, proj_cells, proj_keys = [], [], []
    for side, nm in (("A", A_name), ("B", B_name)):
        for i in range(n_proj):
            pnames = {(rr["proj"].get(rr["sa" if side == "A" else "sb"]) or [(None,)] * n_proj)[i][0] for rr in rounds if rr["played"]}
            lab = proj_label(next(iter(pnames))) if len(pnames) == 1 and None not in pnames else chr(ord("a") + i)
            proj_rows.append(f"{side} \u00b7 proj {lab}" if len(lab) <= 2 else f"{side} \u00b7 {lab[:7]}")
            proj_keys.append((side, lab))
            row = []
            for rr in rounds:
                s = rr["sa"] if side == "A" else rr["sb"]
                pj = rr["proj"].get(s) or []
                if not rr["played"] or i >= len(pj):
                    row.append(None)
                    continue
                pname, req, w = pj[i]
                k = rr["runs"].get(s, {}).get(pname, 0)
                desc = f"{pname} ({req_text(req)} per run, {pts_w(w)})"
                if rr["over"]:
                    row.append({"c": "muted", "o": 0.35, "tip": f"{desc}: round annulled, 0 points"})
                elif k:
                    row.append({"c": SIDE_COLOR[side], "t": f"{TIMES}{k} = {num(k * w)}", "tip": f"{desc}: ran {k} {'time' if k == 1 else 'times'}, {pts_w(k * w)}"})
                else:
                    row.append({"c": "muted", "o": 0.35, "tip": f"{desc}: not run"})
            proj_cells.append(row)
    # rows of projects that never ran in a scored round only repeat "not run": they are left out and named in the sub
    # (which lists every project's needs and points); all rows stay when none ever ran
    ran_rows = [i for i, row in enumerate(proj_cells) if any(c_ and c_.get("t") for c_ in row)]
    never_run = [proj_keys[i] for i in range(len(proj_cells)) if i not in ran_rows] if ran_rows else []
    if ran_rows:
        proj_rows, proj_cells = [proj_rows[i] for i in ran_rows], [proj_cells[i] for i in ran_rows]
    proj_muted = any(c_ and not c_.get("t") for row in proj_cells for c_ in row)

    # ---- numbers for the text
    rw = {str(k): v for k, v in (summ.get("rewards") or next((e.get("rewards") for e in ev if e.get("event") == "game_over"), {}) or {}).items()}
    sA = rw.get(str(seat_a))
    totA = sum(v for v in ptsA if v is not None)
    totB = sum(v for v in ptsB if v is not None)
    totJ = sum(v for v in joint if v is not None)
    totM = sum(rr["opt"]["M"] for rr in played_rounds if rr["opt"])
    nplay = len(played_rounds)
    n_over = sum(1 for rr in played_rounds if rr["over"])
    n_opt = sum(1 for rr in played_rounds if rr["opt"] and not rr["over"] and (rr["pts"].get(rr["sa"], 0) + rr["pts"].get(rr["sb"], 0)) == rr["opt"]["M"])
    # the most contested resource: highest share of its supply ordered over the game
    hot = max(res, key=lambda x_: (sum(rr["demand"][x_] for rr in played_rounds) / st["supply"][x_], -res.index(x_)))
    hot_tot = [rr["demand"][hot] for rr in played_rounds]
    same_joint = len({(j, m) for j, m in zip(joint, Ms) if j is not None}) == 1
    o1 = first["opt"]
    mc = (o1["M"] / (o1["solo_a"] + o1["solo_b"])) if o1 and (o1["solo_a"] + o1["solo_b"]) else None
    bucket = params.get("mc_bucket", metrics.get("mc_bucket"))
    theme = metrics.get("theme") or info.get("theme")

    # reward normalisation (rules, Reward): joint = points / M, half_joint = points / (M / 2), solo = points / own solo optimum
    norm = params.get("reward_norm") or metrics.get("reward_norm")
    if norm not in ("joint", "half_joint", "solo") and played_rounds and all(rr["opt"] for rr in played_rounds) and sA is not None:
        cand = {"joint": lambda rr: rr["opt"]["M"], "half_joint": lambda rr: rr["opt"]["M"] / 2,
                "solo": lambda rr: rr["opt"]["solo_a"]}
        for k_, f_ in cand.items():
            if abs(sum(rr["pts"].get(rr["sa"], 0) / f_(rr) for rr in played_rounds) / len(played_rounds) - sA) < 1e-6:
                norm = k_
                break
    norm = norm or "joint"
    one_M = len({m for m in Ms if m is not None}) == 1 and Ms[0]
    if norm == "half_joint":
        norm_lede = "divided by half the most both could have earned together (half the joint optimum)"
        norm_tile = f"score = mean of points \u00f7 {num(Ms[0] / 2)}" if one_M else "score = mean of points \u00f7 half the joint optimum"
        norm_foot = "Score = mean over rounds of own points \u00f7 half the joint optimum (the authors' fair efficiency), so the two scores average to the joint efficiency"
    elif norm == "solo":
        norm_lede = "divided by the most it could have earned alone"
        norm_tile = "score = mean of points \u00f7 its own best-alone total"
        norm_foot = "Score = mean over rounds of own points \u00f7 the most that agent could have earned alone"
    else:
        norm_lede = "divided by the most both could have earned together (the joint optimum)"
        norm_tile = f"score = mean of points \u00f7 {Ms[0]}" if one_M else "score = mean of points \u00f7 that round's joint optimum"
        norm_foot = "Score = mean over rounds of own points \u00f7 the joint optimum, so the two scores add up to the joint efficiency"

    def rlist(v):
        v = [f"R{i}" for i in v]
        return v[0] if len(v) == 1 else ", ".join(v[:-1]) + " and " + v[-1]

    def side_of(rr, s):
        return "A" if s == rr["sa"] else "B"

    def nm_of(rr, s):
        return A_name if s == rr["sa"] else B_name

    def other(rr, s):
        return rr["sb"] if s == rr["sa"] else rr["sa"]

    # ---- lock-in order against the split (per round: the round's first order, if placed during the talk)
    lock_rounds = [rr for rr in played_rounds if rr["first_lock"] is not None and not rr["over"]]
    first_more = [rr for rr in lock_rounds
                  if rr["pts"].get(rr["first_lock"]["seat"], 0) > rr["pts"].get(other(rr, rr["first_lock"]["seat"]), 0)]
    a_first = [rr["r"] for rr in lock_rounds if rr["first_lock"]["seat"] == rr["sa"]]
    b_first = [rr["r"] for rr in lock_rounds if rr["first_lock"]["seat"] == rr["sb"]]
    who_first = "; ".join(p for p in (f"{A_name} locked in first in {rlist(a_first)}" if a_first else "",
                                      (f"{B_name} in {rlist(b_first)}" if a_first else f"{B_name} locked in first in {rlist(b_first)}")
                                      if b_first else "") if p).replace("; ", ", ")
    lock_all = bool(lock_rounds) and len(first_more) == len(lock_rounds) == nplay and nplay > 1
    lock_count = (f"The agent that locked in first took the larger share in {len(first_more)} of {len(lock_rounds)} rounds"
                  + (" with a lock-in" if len(lock_rounds) < nplay else "") + ".") if lock_rounds else ""

    def seized_txt(rr):
        """A round where the second speaker locked in first and took the larger share: what it did and how the other side
        answered."""
        f = rr["first_lock"]
        s = f["seat"]
        o = other(rr, s)
        k_own = rr["own_msgs_before"]
        txt = (f"In R{rr['r']}, {nm_of(rr, s)}, the second speaker, locked in {order_text(f['purchase'], res)} "
               + ("as its first move" if not k_own else f"after {k_own} message{'' if k_own == 1 else 's'} of its own")
               + ((f", right after {nm_of(rr, o)}'s " + ("opening message" if rr["msgs_before"] == 1 else "message")) if rr["reply_to_other"] else ""))
        so = side_of(rr, o)
        now = rr["order"][o][hot]
        prev = [(q["r"], q["order"][q["sa"] if so == "A" else q["sb"]][hot]) for q in played_rounds if q["r"] < rr["r"]]
        if prev and len({v for _, v in prev}) == 1 and prev[0][1] > now:
            txt += f"; {nm_of(rr, o)}, which had ordered {prev[0][1]} {hot} in {rlist([r for r, _ in prev])}, then took {now} {hot}"
        else:
            txt += f"; {nm_of(rr, o)} then ordered {order_text(rr['order'][o], res)}"
        return txt + "."
    seized = " ".join(seized_txt(rr) for rr in [q for q in lock_rounds if q["first_lock"]["seat"] != q["first_speaker"]
                                                and q in first_more][:2])

    # what the lock-in messages said: the other side never saw the order itself, only the message sent with it
    locks = [d for d in decs if d["purchase"] is not None and d["phase"] != "decide" and any(d["purchase"].values())]

    def states_order(d):
        return all(re.search(rf"\b{q}\s+{re.escape(r_)}\b", d["talk"], re.I) for r_, q in d["purchase"].items() if q)
    n_stated = sum(1 for d in locks if states_order(d))
    results_shown = any("The other party bought" in (d["view"] or "") for d in decs)
    any_after = any(s_["kind"] == "order" for s_ in steps)

    # ---- the split facts (same in every round, or not stated)
    split = {(rr["opt"]["floor"], rr["opt"]["floor_best"], rr["opt"]["maxmin"], rr["opt"]["low_at_M"], rr["opt"]["M"])
             for rr in played_rounds if rr["opt"] and "floor_best" in rr["opt"] and not rr["over"]}
    split = next(iter(split)) if len(split) == 1 and not n_over else None
    fair_v = fair[0] if len({v for v in fair if v is not None}) == 1 else None
    fair_whole = all(v is None or float(v).is_integer() for v in fair)
    split_core, split_tile, split_title = "", "", ""
    if split and same_joint and joint[0] < Ms[0] and split[1] == joint[0]:
        fl, fb, mm, low, M_ = split
        split_core = (f"the best total that gives both agents {fl}+" + (f", and no plan gives both more than {mm}" if fl == mm else "")
                      + f"; the {M_}-point optimum leaves one agent only {low}")
        split_tile = f", the best total that gives both agents {fl}+"
        split_title = f"{fb} of {M_} points every round, never overdrawn"

    # who took the larger share; whether one agent did in every round
    a_more = [i + 1 for i, (a, b_) in enumerate(zip(ptsA, ptsB)) if a is not None and a > b_]
    b_more = [i + 1 for i, (a, b_) in enumerate(zip(ptsA, ptsB)) if a is not None and b_ > a]
    one_side = (A_name if len(a_more) == nplay and not b_more else B_name if len(b_more) == nplay and not a_more else None) if nplay > 1 else None

    # the authors' fair-share yardstick (half the joint optimum) against the agent with fewer points
    fair_tot = sum(f_ for f_, rr in zip(fair, rounds) if rr["played"] and f_ is not None)
    low_side = None if shifting or totA == totB else ("B" if totB < totA else "A")
    fair_foot = ""
    if low_side and fair_tot:
        low_nm, low_tot = (B_name, totB) if low_side == "B" else (A_name, totA)
        fair_foot = (f" Half the joint optimum" + (f" ({num(fair_v)} a round)" if fair_v is not None else "") + " is the authors' "
                     f"fair-share yardstick: what each agent would average by taking turns at the optimum. {low_nm} ended with "
                     f"{pts_w(low_tot)} against {num(fair_tot)}, {num(fair_tot - low_tot)} below it.")

    # title
    words = {1: "one round", 2: "two rounds", 3: "three rounds", 4: "four rounds", 5: "five rounds", 6: "six rounds"}
    span = words.get(nplay, f"{nplay} rounds")
    head = f"Sharing {st['supply'][hot]} {hot} for {span}"
    if nplay and not n_over and lock_all:
        title = f"{head}: never overdrawn, and whoever locked in first took the larger share every round"
    elif nplay and not n_over and same_joint and o1 and joint[0] == Ms[0]:
        title = f"{head}: the joint optimum every round"
    elif nplay and not n_over and same_joint and o1:
        title = (f"Sharing {st['supply'][hot]} {hot}: {split_title}" if split_title
                 else f"{head}: never overdrawn, {joint[0]} of a possible {Ms[0]} points every round")
    elif nplay and not n_over:
        title = f"{head}: never overdrawn, {pct(totJ / totM if totM else 0)} of the possible points"
    else:
        title = f"{head}: {n_over} of {nplay} rounds overdrawn and annulled"

    # lede
    sup_txt = ", ".join(f"{st['supply'][x_]} {x_}" for x_ in res)
    n_words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    if params.get("enable_talk") is False:
        talk_txt = "Each round they order in secret and at the same time (no talk in this condition)"
    else:
        talk_txt = (f"Each round they talk in turns (up to {st['talk_turns'] or 5} messages each), and either may lock in its order at any "
                    f"point; an agent that has not locked in orders after the talk, and neither ever sees the other's order before the "
                    f"round resolves")
    lede = (f"Two agents buy {', '.join(res[:-1])} and {res[-1]} from one shared pool (per round: {sup_txt}; a budget of "
            f"{num(st['budget'])} each; at most {st['max_types']} kinds of resource per order) to run "
            f"{n_words.get(n_proj, n_proj)} private projects each that turn resources into points. {talk_txt}. If the two orders "
            f"together exceed the supply of any resource, the round is annulled and both earn nothing; each agent's score is its "
            f"points per round {norm_lede}, averaged over the rounds it played.")

    # tiles
    def score_tile(nm, seat, tot, series, first_tile):
        v = rw.get(str(seat))
        vals = [x_ for x_ in series if x_ is not None]
        how = norm_tile if first_tile else "same scale"
        per = (f"{pts_w(tot)}, {num(vals[0])} in each of {nplay} rounds" if len(vals) > 1 and len(set(vals)) == 1
               else f"{pts_w(tot)} over {nplay} rounds ({', '.join(str(x_) for x_ in vals)})")
        return {"k": f"{nm} score", "v": f"{v:.3f}" if v is not None else "—", "s": f"{per}; {how}"}
    tiles = [score_tile(A_name, seat_a, totA, ptsA, True)]
    if not shifting:
        tiles.append(score_tile(B_name, seats_b[0], totB, ptsB, False))
    eff_s = f"{totJ} of the {totM} points possible"
    if same_joint and o1 and nplay > 1:
        eff_s += f", {joint[0]} of {Ms[0]} every round"
    if split_tile:
        eff_s += split_tile
    else:
        eff_s += "; the optimum was never reached" if not n_opt else f"; the optimum was reached in {n_opt} of {nplay} rounds"
    tiles.append({"k": "Joint efficiency", "v": pct(totJ / totM) if totM else "—", "s": eff_s})
    hot_s = f"{hot.capitalize()} ordered by both: " + ", ".join(str(v) for v in hot_tot) + f" of the {st['supply'][hot]} available each round"
    if not n_over and lock_rounds and (lock_all or not one_side or not fair_foot):
        tiles.append({"k": "First to lock in got more", "v": f"{len(first_more)} of {len(lock_rounds)}",
                      "s": ("Of the rounds with a lock-in. " if len(lock_rounds) < nplay else "") + f"{who_first}"
                           + (f"; {one_side} got more every round" if one_side and not lock_all else "")
                           + f"; never overdrawn ({hot} ordered: " + ", ".join(str(v) for v in hot_tot) + f" of {st['supply'][hot]})"})
    elif not n_over and one_side and fair_foot:
        # one agent got more every round whoever locked in first: the story is the split, measured against fair share
        low_series = [v for v in (ptsB if low_side == "B" else ptsA) if v is not None]
        tiles.append({"k": f"{low_nm} vs fair share", "v": f"{num(low_tot)} of {num(fair_tot)}",
                      "s": (f"{num(low_series[0])} a round against {num(fair_v)}" if len(set(low_series)) == 1 and fair_v is not None
                            else "Its points against half the joint optimum, summed over the rounds")
                           + ": half the joint optimum, the authors' fair-share yardstick"})
    else:
        tiles.append({"k": "Rounds overdrawn", "v": f"{n_over} of {nplay}", "s": hot_s})

    # optimum plan sentence
    def plan_text(rr):
        if not rr["opt"] or not rr["opt"]["plans"]:
            return ""
        pa, pb = rr["proj"][rr["sa"]], rr["proj"][rr["sb"]]
        out = []
        for oa, va, ra, ob, vb, rb in rr["opt"]["plans"][:2]:
            sa_ = (A_name, oa, va, ra, pa)
            sb_ = (B_name, ob, vb, rb, pb)
            big, small = (sb_, sa_) if vb > va else (sa_, sb_)
            out.append(f"{big[0]} takes {order_text(big[1], res)} ({runs_text(big[3], big[4], 'no run')} = {big[2]}) while {small[0]} "
                       f"buys {order_text(small[1], res)} ({runs_text(small[3], small[4], 'no run')} = {small[2]})")
        return ("either " + out[0] + ", or " + out[1]) if len(out) == 2 else out[0]
    opt_sentence = plan_text(first) if fixed else ""

    # whether the larger share went with locking in first (the lock chart follows the points chart unless a talk chart is
    # added between them)
    share_txt = ""
    if a_more and b_more:
        share_txt = f" The larger share went to {A_name} in {rlist(a_more)} and to {B_name} in {rlist(b_more)}"
    elif a_more:
        share_txt = f" {A_name} got the larger share in every round"
    elif b_more:
        share_txt = f" {B_name} got the larger share in every round"
    if share_txt:
        share_txt += (": every time, the agent that locked in first (next chart)." if lock_all
                      else "." + (f" {lock_count[:-1]} (next chart)." if lock_count else ""))

    # what the projects need, and whether both sides' best project needs the contested resource
    def top_needs(pj):
        return pj and max(pj, key=lambda p: p[2])[1]
    both_hot = all(any(r_ == hot for r_, _ in (top_needs(first["proj"][s]) or [])) for s in (first["sa"], first["sb"]))
    proj_table = ""
    if fixed and pa0 and pb0:
        proj_table = (f" Needs per run → points. {A_name}: " + "; ".join(f"{proj_label(n)} ({req_text(q)}) → {num(w)}" for n, q, w in pa0)
                      + f". {B_name}: " + "; ".join(f"{proj_label(n)} ({req_text(q)}) → {num(w)}" for n, q, w in pb0) + ".")

    # points chart. The half-optimum reference line is drawn only when every value is a whole number (the page shows this
    # chart's values with no decimals); the fair-share figures are always in the notes (fair_foot). When every round gave
    # the same points the chart shows running totals instead of four identical rounds.
    on_played = lambda v: [x_ if rr["played"] else None for x_, rr in zip(v, rounds)]
    if flat:
        a1, b1, j1, m1 = (next(v for v in s_ if v is not None) for s_ in (ptsA, ptsB, joint, Ms))
        fair_line = running(on_played(fair))
        pts_title = "Points, running total"
        pts_sub = (f"Running totals. Every round went the same way: {A_name} {a1}, {B_name} {b1}, together {j1} of the {m1} possible "
                   f"(dashed: the joint optimum)." + (f" {split[1]} is {split_core}." if split_core else ""))
        pts_bars = []
        pts_lines = [{"name": A_name, "color": "c1", "v": running(ptsA), "dots": True, "end": True},
                     {"name": B_name, "color": "c2", "v": running(ptsB), "dots": True, "end": True},
                     {"name": "Both together", "color": "ink", "v": running(joint), "end": True},
                     {"name": "Joint optimum", "color": "hold", "v": running(on_played(Ms)), "dash": True, "end": True}]
    else:
        fair_line = fair
        pts_title = "Points per round"
        pts_sub = ("Bars: each agent's points from its projects. Line: both together, against the joint optimum (dashed)."
                   + (f" Together they made {split[1]} every round, {split_core}." if split_core else ""))
        pts_bars = [{"name": A_name, "color": "c1", "v": ptsA}, {"name": B_name, "color": "c2", "v": ptsB}]
        pts_lines = [{"name": "Both together", "color": "ink", "v": joint, "dots": True},
                     {"name": "Joint optimum", "color": "hold", "v": Ms, "dash": True}]
    if all(v is None or float(v).is_integer() for v in fair_line):
        pts_lines.append({"name": "Half the optimum", "color": "muted", "v": fair_line, "dash": True, "thin": True})
        pts_sub += " Light dashed: half the optimum, the authors' fair-share yardstick (see the notes below)."
    if not flat:
        pts_sub += share_txt
    points_chart = {"title": pts_title, "fmt": "int", "height": 230, "sub": pts_sub, "bars": pts_bars, "lines": pts_lines}
    if not pts_bars:
        del points_chart["bars"]

    # talk chart (only with flat points, when the talk does change): messages per round, and the round the orders settled
    talk_chart = None
    talk_tot = [None if a is None else a + b_ for a, b_ in zip(talk_n["A"], talk_n["B"])]
    if flat and params.get("enable_talk") is not False and len({v for v in talk_tot if v is not None}) > 1:
        vals = [str(v) for v in talk_tot if v is not None]
        tsub = (f"Cheap-talk messages each agent sent per round, not counting the note sent with an order: "
                f"{', '.join(vals[:-1])} and {vals[-1]} in all.")
        ords = [(rr["r"], json.dumps([rr["order"][rr["sa"]], rr["order"][rr["sb"]]], sort_keys=True)) for rr in played_rounds]
        r0 = next((i for i in range(len(ords) - 1) if all(o == ords[i][1] for _, o in ords[i:])), None)
        if r0 is not None:
            q = played_rounds[r0]
            pair = f"{A_name} {order_text(q['order'][q['sa']], res)}, {B_name} {order_text(q['order'][q['sb']], res)}"
            later = [str(talk_tot[rr["r"] - 1]) for rr in played_rounds[r0 + 1:]]
            if r0 == 0:
                tsub += f" The orders were the same in every round ({pair})."
            else:
                tsub += (f" The orders settled in R{q['r']} ({pair}) and did not change after it; the later rounds had "
                         + (f"{later[0]} message{'' if later[0] == '1' else 's'}" if len(later) == 1
                            else f"{', '.join(later[:-1])} and {later[-1]} messages") + ".")
        talk_chart = {"title": "Messages before the orders", "fmt": "int", "height": 170, "sub": tsub,
                      "bars": [{"name": A_name, "color": "c1", "v": talk_n["A"]}, {"name": B_name, "color": "c2", "v": talk_n["B"]}],
                      "lines": []}

    # lock-in grid: who locked in first (and when), and who earned more
    lock_cells, more_cells, muted_used = [], [], False
    for rr in rounds:
        if not rr["played"]:
            lock_cells.append(None)
            more_cells.append(None)
            continue
        f = rr["first_lock"]
        if f is None:
            muted_used = True
            lock_cells.append({"c": "muted", "o": 0.35, "tip": "Nobody locked in during the talk: both ordered after it, at the same time"})
        else:
            s, o = f["seat"], other(rr, f["seat"])
            sec = rr["second_sub"]
            when = f"at exchange {f['exchange']}" if f["exchange"] else "during the talk"
            mb = rr["msgs_before"]
            tip = (f"{nm_of(rr, s)} locked in {order_text(f['purchase'], res)} {when}, "
                   + ("before any message this round" if not mb else f"after {mb} message{'s' if mb > 1 else ''} this round"))
            if sec is not None:
                tip += (f"; {nm_of(rr, o)} then " + ("ordered after the talk" if sec["phase"] == "decide" else
                                                      f"locked in at exchange {sec['exchange']}" if sec["exchange"] else "locked in")
                        + f": {order_text(sec['purchase'], res)}")
            cell = {"c": SIDE_COLOR[side_of(rr, s)], "tip": tip}
            cell["t"] = f"{side_of(rr, s)} · ex {f['exchange']}" if f["exchange"] else side_of(rr, s)
            lock_cells.append(cell)
        pa_, pb_ = rr["pts"].get(rr["sa"], 0), rr["pts"].get(rr["sb"], 0)
        if rr["over"]:
            muted_used = True
            more_cells.append({"c": "muted", "o": 0.35, "tip": "Round annulled: both earned 0"})
        elif pa_ == pb_:
            muted_used = True
            more_cells.append({"c": "muted", "o": 0.35, "tip": f"Equal: {pts_w(pa_)} each"})
        else:
            sd = "A" if pa_ > pb_ else "B"
            more_cells.append({"c": SIDE_COLOR[sd], "t": f"{sd} · {max(pa_, pb_)} to {min(pa_, pb_)}",
                               "tip": f"{A_name} {pts_w(pa_)}, {B_name} {pts_w(pb_)}"})
    lock_sub = ("Top row: whose order was locked in first, and at which exchange (ex) of the talk; an exchange is one message slot "
                "per agent. Bottom row: who earned more that round.")
    if lock_rounds:
        if lock_all:
            lock_sub += f" {who_first}; each time, the first to lock in took the larger share."
        elif one_side:
            lock_sub += f" {who_first}, yet {one_side} got more in every round: locking in first did not decide the shares here."
        else:
            lock_sub += f" {who_first}; " + lock_count[0].lower() + lock_count[1:]
    lock_sub += (" " + seized) if seized else ""
    lock_chart = {"title": "Who locked in first, and who got more",
                  "sub": lock_sub,
                  "grid": {"rows": ["Locked first", "More points"], "cells": [lock_cells, more_cells],
                           "legend": [[A_name, "c1"], [B_name, "c2"]] + ([["Nobody / equal / annulled", "muted"]] if muted_used else []),
                           "rowHeight": 30}}

    stated_txt = ""
    if locks:
        stated_txt = (" (here every lock-in message stated the order anyway)" if n_stated == len(locks)
                      else f" (here {n_stated} of {len(locks)} lock-in messages stated the order)" if n_stated else "")
    # the rows left out are named by count only: the needs list below names every project, and the rows shown are the rest
    never_txt = (f"; the other {len(never_run)} project{'' if len(never_run) == 1 else 's'} never ran" if never_run else "")
    charts = [
        points_chart,
        lock_chart,
        {"title": "Orders against the shared supply",
         "sub": (f"Each cell: {A_name}'s units + {B_name}'s units = the total / the round's supply. The engine never shows one side's "
                 f"order to the other before the round resolves{stated_txt}; going over in any row annuls the round."
                 + (f" Both agents' best-paying projects need {hot}, so {hot} is where the orders collide." if both_hot else "")),
         "grid": {"rows": [short_label(x_) for x_ in res], "cells": cells_sup, "legend": sup_legend, "rowHeight": 30}},
        {"title": "Projects run",
         "sub": ("How many times each private project ran (× runs = points)" + never_txt + "."
                 + (proj_table or " Hover a cell for what a run needs and pays.")
                 + (f" The joint optimum needs a different mix: {opt_sentence}." if opt_sentence else "")),
         "grid": {"rows": proj_rows, "cells": proj_cells,
                  "legend": [[f"{A_name} ran it", "c1"], [f"{B_name} ran it", "c2"]] + ([["Not run", "muted"]] if proj_muted else []),
                  "rowHeight": 26}},
    ]
    if talk_chart:
        charts.insert(1, talk_chart)
    if not lock_rounds and not muted_used:
        charts.remove(lock_chart)

    # foot
    date = ((run.get("recorded_at") or (b.get("meta") or {}).get("recorded_at") or "")[:10]) or DEFAULT_DATE
    model = (run.get("model") or (b.get("meta") or {}).get("model") or "deepseek-flash").split("/")[-1]
    variant = os.path.basename((b.get("config") or {}).get("variant") or "").replace(".yaml", "")
    commit = run.get("env_commit") or (b.get("meta") or {}).get("env_commit") or ""
    if commit.endswith("-dirty"):
        commit = commit[:-len("-dirty")] + " with uncommitted changes"
    seed = run.get("seed", summ.get("seed"))
    cond = ["a new partner every round" if shifting else "a fixed partner", "fixed projects" if fixed else "new projects every round"]
    if params.get("enable_talk") is False:
        cond.append("no talk")
    if params.get("full_transparency"):
        cond.append("both project lists shown to both")
    if theme:
        cond.append(f"the {theme} theme")
    cond_txt = ", ".join(cond[:-1]) + " and " + cond[-1]
    foot = (f"Data: the {date} recording of {GID}"
            + (f" (descriptor {variant}, seed {seed}" if variant else f" (seed {seed}") + (f", env commit {commit}" if commit else "") + ")"
            + f", both agents played by {model}; every number on this page comes from its game log (the moves and what each agent saw) "
            + ("and the round results the engine wrote. " if hist else "and round results re-derived from the moves with the game's rules. "))
    if has_thinking and from_md:
        foot += "Reasoning comes from the reasoning blocks of the rendered replay. "
    elif has_thinking:
        cs, n_st = call_stats, len(steps)
        if cs["calls"] == n_st == cs["matched"] and not cs["earlier"]:
            detail = f"{cs['calls']} calls, one per move, no retries"
        else:
            detail = (f"{cs['calls']} calls for {n_st} moves"
                      + (f"; {cs['earlier']} earlier attempt{'' if cs['earlier'] == 1 else 's'} shown above the final reasoning" if cs["earlier"] else "")
                      + (f"; {n_st - cs['matched']} answer{'' if n_st - cs['matched'] == 1 else 's'} not matching the move played"
                         if cs["matched"] < cs["steps"] else ""))
        if cs["rescued"]:
            detail += f"; {cs['rescued']} answer{'' if cs['rescued'] == 1 else 's'} given to a follow-up question"
        foot += (("Observations and reasoning come" if any(c_.get("prompt") for c_ in calls) else "Reasoning comes")
                 + f" from the saved model calls ({detail}). ")
    else:
        foot += "This recording was made before reasoning was saved; the game will be re-recorded with reasoning. "
    if o1 and fixed:
        foot += ((f"Scenario {o1['id']} from the paper's M/C = {bucket} pool" if o1["id"] and bucket is not None else "One scenario")
                 + f", played with {cond_txt}: alone, {A_name} could earn at most {o1['solo_a']} a round and "
                 f"{B_name} {o1['solo_b']}; together at most {o1['M']}, so M/C = {o1['M']} \u00f7 ({o1['solo_a']} + {o1['solo_b']}) = {mc:.2f}. "
                 f"The optima were re-checked here over every legal pair of orders. ")
    elif bucket is not None:
        foot += f"Scenarios from the paper's M/C = {bucket} pool, played with {cond_txt}. "
    foot += (f"{norm_foot}" + (f" ({pct(totJ / totM)} here)" if totM and norm != "solo" else "") + "." + fair_foot
             + " Game from Yao, Zou and Hawkins (2026), \u201cTalk is Cheap, Communication is Hard\u201d (COLM 2026).")
    try:
        ref = PAPER.get(round(float(bucket), 1)) if bucket is not None else None
    except (TypeError, ValueError):
        ref = None
    if ref:
        foot += " " + ref

    # transcript and timeline captions: what each side saw of an order (rules, Round Procedure 13 and Privacy; checked against
    # the recorded views), and that a round's moves were made one at a time
    feed_sub = ("Every message and order in time order, with each round's result after its last order; within a round the moves "
                "were made one at a time, in the order listed"
                + (", except the orders placed after the talk, which were simultaneous" if any_after else "") + ".")
    if locks:
        feed_sub += (" During the talk the other side saw only the message sent with an order and a notice that it was locked in, "
                     "never the order itself" + (f"; here every lock-in message stated the order" if n_stated == len(locks)
                                                 else f"; here {n_stated} of {len(locks)} lock-in messages stated the order" if n_stated else "")
                     + ".")
    if results_shown:
        feed_sub += " After each round, both orders were shown to both sides."
    feed_sub += " The selected step is highlighted; later entries are hidden."
    timeline_sub = ("Each block is one move, colored by type; within a round, a seat's blocks stack bottom to top in the order played. "
                    "Talk is turn by turn" + (", and only the orders placed after the talk are simultaneous" if any_after else
                                              "; every order in this game was locked in during the talk, so no two moves were simultaneous")
                    + ". Click a block to open it.")

    d = {"id": GID, "name": "Talk is Cheap", "title": title, "lede": lede, "seatNames": seat_names[:n_seats], "tiles": tiles, "foot": foot,
         "x": {"n": xn, "labels": labels}, "charts": charts,
         "kinds": {k: v for k, v in KINDS_ALL.items() if k in used_kinds}, "steps": steps, "feed": feed,
         "feedTitle": "Negotiation transcript",
         "feedSub": feed_sub, "timelineSub": timeline_sub}
    d["eyebrow"] = (common.EYEBROW.format(gid=GID, date=date) if has_thinking
                    else f"social_game_env \u00b7 {GID} \u00b7 {model} \u00b7 recorded {date} \u00b7 reasoning not saved")
    return d


def recording_date(src):
    """The day the game was recorded (run.json or meta.json of the bundle), else the default."""
    b = load_source(src)
    run, meta = b.get("run") or {}, b.get("meta") or {}
    return ((run.get("recorded_at") or meta.get("recorded_at") or "")[:10]) or DEFAULT_DATE


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    src, out = argv[0], argv[1]
    calls = common.load_calls(argv[2]) if len(argv) > 2 else None
    common.write(build(src, calls), out, date=recording_date(src))


if __name__ == "__main__":
    main(sys.argv[1:])
