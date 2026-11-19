"""Replay data for eco_colored_trails (CT-Bench: Colored Trails with Pay-for-Partner promises and a self-negotiated contract).

usage: python3 -I eco_colored_trails.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, transcript.md), a file inside one, or
          the rendered Markdown replay of one (one block per decision: the seat, the state part of the prompt it saw, its move,
          and the reasoning when it was saved).
  OUT_DIR where <OUT_DIR>/eco_colored_trails.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). The prompt may be
          the full prompt (the hook's format), only its state part, or missing. Calls are tied to decisions by the cheapest
          in-order alignment (attach_calls): a call goes to a decision of its seat, of the column its prompt names and with the
          state its prompt shows (white space ignored), preferring the decision whose logged move is its answer. Its reasoning
          becomes the step's thinking; earlier calls for the same decision (format retries) are kept above it, and a decision
          whose call answered something else is marked as the engine's fallback. A warning goes to stderr when calls were given
          but fewer than 90% of the decisions received reasoning.

Every number shown comes from the game log. The x axis has one column per contract-negotiation turn ("Talk k": P-Red speaks,
then P-Blue) and one per game turn ("Turn t": proposals, then moves). Positions and chips at the end of a game turn are the
logged end-of-turn state (summary.json info.turn_log); during the negotiation nothing moves, so they are the starting state
read from the first prompts. Scores are the logged ones (info.scores), checked against the rule 20 + 5 x chips left for a
player on the goal and 0 otherwise (rules/eco_colored_trails_rules.md, Reward); reward = score / 150."""
import json, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_colored_trails"
DEFAULT_DATE = "2026-11-09"
PLAYERS = ("P-Red", "P-Blue")
SEAT_COLOR = {"P-Red": "c2", "P-Blue": "c1"}          # the palette has no red: P-Red and red tiles are drawn in orange (c2)
TILE_KIND = {"R": "move_r", "B": "move_b", "G": "move_g"}
TILE_COLOR = {"R": "c2", "B": "c1", "G": "c3"}
WORD = {"R": "red", "B": "blue", "G": "green"}
LETTER = {"r": "R", "red": "R", "b": "B", "blue": "B", "g": "G", "green": "G"}
GOAL_POINTS, CHIP_POINTS = 20, 5
KINDS = {
    "talk": ["Contract talk, no terms", "hold"],
    "propose": ["Proposal: contract or exchange", "c4"],
    "accept": ["Accept or honour", "ink"],
    "reject": ["Reject or refuse", "hold"],
    "pass": ["Skip trading or stay", "muted"],
    "move_r": ["Move onto a red tile", "c2"],
    "move_b": ["Move onto a blue tile", "c1"],
    "move_g": ["Move onto a green tile", "c3"],
}
# The paper's averages over six models and its 80 boards (Wyse et al. 2026, Tables 1-3, as read in the rules file).
PAPER = {("pay_for_partner", "prog_trading"): "with Prog-Trading contracts a normalized joint reward of 0.89, both players finishing "
                                              "in 79% of games and a contract accepted in 80%",
         ("pay_for_partner", "prog_points"): "with Prog-Points contracts a normalized joint reward of 0.78, both players finishing "
                                             "in 62% of games and a contract accepted in 92%",
         ("pay_for_partner", "none"): "without a contract a normalized joint reward of 0.77, both players finishing in 60% of games "
                                      "and promises broken at a rate of 0.32"}
MINUS, TIMES = "\u2212", "\u00d7"


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


def en_obj(o):
    if isinstance(o, str):
        return en(o)
    if isinstance(o, list):
        return [en_obj(v) for v in o]
    if isinstance(o, dict):
        return {k: en_obj(v) for k, v in o.items()}
    return o


# ------------------------------------------------------------------------------------------------ loading
def _md_source(path):
    """A rendered Markdown replay (render_replays*.py, Python-lane games): one '## step N - Pk' block per decision with the
    state the seat saw, its move and, when saved, the reasoning ('````text' blocks; the last one is the decision's own, earlier
    ones are failed attempts); the header gives the seed, the model and the seat rewards; the closing block the game_over event."""
    text = open(path, encoding="utf-8").read()
    events, md_think = [], []
    for blk in re.split(r"\n(?=## )", text):
        m = re.match(r"## \u7b2c (\d+) \u6b65 \u00b7 P(\d+)", blk)                      # "## step N . Pk"
        if m:
            st = re.search(r"\*\*\u770b\u5230\uff1a\*\*\s*\n+```\n(.*?)\n```", blk, re.S)  # "**saw:**"
            mv = re.search(r"\*\*\u505a\u4e86\uff1a\*\* `(.*)`", blk)                      # "**did:**"
            thinks = re.findall(r"````text\n(.*?)\n````", blk, re.S)
            earlier = re.findall(r"\uff08\u56de\u7b54 `(.*?)`\uff09", blk)                 # "(answer `...`)" of earlier calls
            state = (st.group(1) if st else "").replace(" \u2026\uff08\u540e\u9762\u7701\u7565\uff09", " ... (cut)")
            events.append({"event": "move", "actor": int(m.group(2)), "move": mv.group(1) if mv else "{}", "view": state})
            md_think.append({"thinks": thinks, "earlier": earlier, "rescued": "\u8865\u95ee" in blk})
            continue
        if blk.startswith("## \u7ed3\u675f") or blk.startswith("## \u5f00\u5c40"):        # "## end" / "## start"
            js = re.search(r"```\n(\{.*\})\n```", blk, re.S)
            if js:
                try:
                    e = json.loads(js.group(1))
                    events.append(e)
                except ValueError:
                    pass
    head = text.split("\n## ")[0]
    rw = re.search(r"\u5404\u5ea7\u4f4d\u5f97\u5206 (\{[^}]*\})", head)                   # "scores per seat {...}"
    seed = re.search(r"\u79cd\u5b50 (\d+)", head)                                          # "seed N"
    model = re.search(r"\u6a21\u578b (\S+)", head)                                         # "model X"
    fb = re.search(r"\u515c\u5e95\u6b21\u6570 (\d+)", head)                                # "fallbacks N"
    over = next((e for e in events if e.get("event") == "game_over"), {})
    summary = {"rewards": json.loads(rw.group(1)) if rw else over.get("rewards", {}), "fallbacks": int(fb.group(1)) if fb else None,
               "info": {}}
    run = {"seed": int(seed.group(1)) if seed else None, "model": model.group(1) if model else "", "recorded_at": None}
    return {"events": events, "summary": summary, "run": run, "config": {}, "transcript": "", "meta": {}, "md_think": md_think}


def load_source(src):
    if os.path.isdir(src):
        b = common.load_bundle(src)
        meta = os.path.join(src, "meta.json")
        b["meta"] = json.load(open(meta, encoding="utf-8")) if os.path.exists(meta) else {}
        return b
    d = os.path.dirname(os.path.abspath(src))
    if os.path.exists(os.path.join(d, "game_log.jsonl")):
        return load_source(d)
    return _md_source(src)


def recorded_date(src):
    b = load_source(src)
    return str((b.get("meta") or {}).get("recorded_at") or (b.get("run") or {}).get("recorded_at") or DEFAULT_DATE)[:10]


# ------------------------------------------------------------------------------------------------ reading a prompt
POS = r"\((\d+),\s*(\d+)\)"
CH = r"R (\d+), G (\d+), B (\d+)"
RX_ME = re.compile(r"\bYou are (P-Red|P-Blue)\.")
RX_ROW = re.compile(r"^Row (\d+): ([GRB](?: [GRB])*)\s*$", re.M)
RX_MY = re.compile(r"You are at " + POS + r"( \(the goal\))?(?:; goal " + POS + r")?\.\s*Your chips: " + CH)
RX_OTHER = re.compile(r"(P-Red|P-Blue) is at " + POS + r"( \(the goal\))?(?:; goal " + POS + r")?\.\s*(?:P-Red|P-Blue)'s chips: " + CH)
RX_PATH = re.compile(r"Your best path to the goal: (\S+) \((\d+) moves?; chips needed: ([^;)]*)(?:; chips missing: ([^)]*))?\)")
RX_NEG = re.compile(r"Contract negotiation(?:,? attempt (\d+)(?: of \d+)?)?, turn (\d+) of (\d+)")
RX_TURN = re.compile(r"^Turn (\d+) of at most (\d+)\.", re.M)
RX_HIST = re.compile(r"^Turn (\d+): (.*?)\s*End of turn: (.*)$", re.M)
RX_SOFAR = re.compile(r"^This turn so far: (.*)$", re.M)
RX_ASKED = re.compile(r"\b(red|blue|green) chip\b")


def chips_of(groups):
    r, g, b = (int(v) for v in groups)
    return {"R": r, "G": g, "B": b}


def need_of(s):
    """'red 5, green 1' -> {'R': 5, 'G': 1}; 'none' -> {}."""
    out = {}
    for w, n in re.findall(r"(red|blue|green) (\d+)", s or ""):
        out[LETTER[w]] = out.get(LETTER[w], 0) + int(n)
    return out


def parse_view(v):
    p = {"me": None, "rows": [], "pos": None, "goal": None, "chips": None, "other": None, "opos": None, "ochips": None,
         "path": None, "neg": None, "turn": None, "cap": None, "sofar": "", "hist": []}
    m = RX_ME.search(v)
    p["me"] = m.group(1) if m else None
    p["rows"] = [r.replace(" ", "") for _, r in sorted(RX_ROW.findall(v), key=lambda t: int(t[0]))]
    m = RX_MY.search(v)
    if m:
        p["pos"] = (int(m.group(1)), int(m.group(2)))
        p["goal"] = (int(m.group(4)), int(m.group(5))) if m.group(4) else None
        p["chips"] = chips_of(m.group(6, 7, 8))
    m = RX_OTHER.search(v)
    if m:
        p["other"] = m.group(1)
        p["opos"] = (int(m.group(2)), int(m.group(3)))
        p["ochips"] = chips_of(m.group(7, 8, 9))
    m = RX_PATH.search(v)
    if m:
        p["path"] = {"tiles": m.group(1), "len": int(m.group(2)), "need": need_of(m.group(3)), "missing": need_of(m.group(4))}
    m = RX_NEG.search(v)
    if m:
        p["neg"] = (int(m.group(1)) if m.group(1) else None, int(m.group(2)), int(m.group(3)))
    m = RX_TURN.search(v)
    if m:
        p["turn"], p["cap"] = int(m.group(1)), int(m.group(2))
    m = RX_SOFAR.search(v)
    p["sofar"] = m.group(1).strip() if m else ""
    for t, evs, end in RX_HIST.findall(v):
        st = {}
        for who, r, c, a, g, b in re.findall(r"(you|P-Red|P-Blue) at " + POS + r"[^;]*? with " + CH, end):
            st[p["me"] if who == "you" else who] = {"pos": (int(r), int(c)), "chips": chips_of((a, g, b))}
        p["hist"].append((int(t), evs.strip(), st))
    return p


# ------------------------------------------------------------------------------------------------ actions
def canon_color(c):
    c = str(c or "").strip()
    return LETTER.get(c.lower(), c.upper()[:1])


def canon_player(s):
    s = str(s or "").strip().lower()
    return "P-Red" if s in ("p-red", "red", "p_red", "pred") else "P-Blue" if s in ("p-blue", "blue", "p_blue", "pblue") else str(s)


def canon_tile(t):
    if isinstance(t, str):
        m = re.match(r"\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*$", t)
        return [int(m.group(1)), int(m.group(2))] if m else t
    if isinstance(t, (list, tuple)) and len(t) == 2:
        try:
            return [int(t[0]), int(t[1])]
        except (TypeError, ValueError):
            return list(t)
    return t


def canon_bundle(items):
    out = {}
    for it in items or []:
        if isinstance(it, dict):
            c, q = canon_color(it.get("color")), it.get("quantity", 0)
            try:
                q = int(q)
            except (TypeError, ValueError):
                continue
            if q:
                out[c] = out.get(c, 0) + q
    return out


def canon_action(a):
    """An action in a comparable form: colours as letters, players as P-Red / P-Blue, tiles as [r, c], bundles merged by colour."""
    if not isinstance(a, dict):
        return None
    aid, p = a.get("action_id"), a.get("payload") or {}
    if not isinstance(p, dict):
        p = {}
    q = {}
    if aid in ("say", "propose_contract", "accept_contract"):
        q["message"] = str(p.get("message") or "").strip()
        if aid == "propose_contract":
            if "tiles" in p:
                q["tiles"] = sorted((tuple(canon_tile(t.get("tile"))), canon_player(t.get("giver"))) for t in p.get("tiles") or []
                                    if isinstance(t, dict))
            for k in ("red_pays", "blue_pays"):
                if k in p:
                    q[k] = p[k]
    elif aid in ("propose_trade", "propose_arrangement"):
        q["offer"], q["receive"] = canon_bundle(p.get("offer")), canon_bundle(p.get("receive"))
    elif aid == "move":
        q["tile"] = canon_tile(p.get("tile"))
    return {"action_id": aid, "payload": q}


def parse_answer(s):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (s or "").strip())
    for cand in (s, s[s.find("{"): s.rfind("}") + 1] if "{" in s else ""):
        try:
            o = json.loads(cand)
            if isinstance(o, dict):
                return o
        except ValueError:
            pass
    return None


def bundle_text(b, mode, owner, other):
    """{'R': 2} offered by owner -> 'pay for 2 of P-Blue's moves onto red tiles' (arrangement) or '2 red' (trade)."""
    if not b:
        return ""
    parts = [f"{n} {WORD.get(c, c)}" for c, n in b.items()]
    if mode == "arrangement":
        return " and ".join(f"{n} of {other}'s moves onto {WORD.get(c, c)} tiles" for c, n in b.items())
    return " and ".join(parts)


# ------------------------------------------------------------------------------------------------ calls
# the decision each action answers (an answer of another phase is an illegal answer the engine replaced)
PHASE = {"say": "talk", "propose_contract": "talk", "accept_contract": "talk", "skip": "trade", "propose_trade": "trade",
         "propose_arrangement": "trade", "accept": "respond", "reject": "respond", "move": "move", "stay": "move",
         "honor": "honour", "refuse": "honour"}
# alignment costs (lower is better): a call skipped, a step left without a call, a call whose answer the engine did not apply,
# an earlier call kept as a retry, an answer of another phase; credits for an answer equal to the logged move and a prompt equal
# to the logged view
SKIP_CALL, NO_CALL, FALLBACK, RETRY, OTHER_PHASE, ANSWER_HIT, PROMPT_HIT = 3.0, 3.0, 4.0, 1.5, 1.0, -6.0, -3.0
MAX_TRIES = 6                                                   # calls one decision may take (retries included)


def norm_state(s):
    """The state part of a prompt with runs of white space collapsed: the hook logs the full prompt, other call logs only the
    state part (from 'You are seat'), and the game log the full prompt."""
    return re.sub(r"\s+", " ", common.state_part(s or "")).strip()


def column_of(text):
    """The column a prompt names: ('turn', t) for 'Turn t of at most ...', ('talk', k) for 'Contract negotiation, turn k', else None."""
    m = RX_TURN.search(text or "")
    if m:
        return ("turn", int(m.group(1)))
    m = RX_NEG.search(text or "")
    return ("talk", int(m.group(2))) if m else None


def attach_calls(steps, calls):
    """Tie calls to steps (see the module docstring) by the cheapest monotone alignment: every step takes a run of consecutive
    calls (possibly none; earlier calls in the run are retries, the last one is the decision's own) and calls may be left over.
    A call can only go to a step of its seat; a call whose prompt names a turn must name the step's column, and one whose prompt
    carries a state must carry the step's state (white space ignored; a cut-off copy counts). Within that, an answer equal to the
    logged move and a prompt equal to the logged view are preferred; a step whose last call answered something else is shown with
    a note that the engine applied the fallback move. Returns the number of steps that received a call."""
    S, C = len(steps), len(calls)
    if not S or not C:
        return 0
    views = [norm_state(st["_view"]) if st.get("_view") else "" for st in steps]
    want = [canon_action(st["_act"]) for st in steps]
    cinfo = []
    for c in calls:
        p = c.get("prompt") or ""
        a = parse_answer(c.get("answer"))
        cinfo.append({"state": norm_state(p) if "You are seat" in p else "", "col": column_of(p), "act": canon_action(a),
                      "phase": PHASE.get((a or {}).get("action_id"))})
    memo = {}

    def fit(i, k):
        """None when call k cannot be step i's; else (answer is the logged move, prompt is the logged view, phase cost)."""
        if (i, k) in memo:
            return memo[i, k]
        st, ci, r = steps[i], cinfo[k], None
        if calls[k].get("seat") in (None, st["seat"]) and (ci["col"] is None or ci["col"] == st["_col"]):
            same = bool(ci["state"]) and ci["state"] == views[i]
            if same or not ci["state"] or not views[i] or ci["state"].startswith(views[i]) or views[i].startswith(ci["state"]):
                ans = ci["act"] is not None and ci["act"] == want[i]
                r = (ans, same, 0.0 if ans or ci["phase"] in (None, st["_phase"]) else OTHER_PHASE)
        memo[i, k] = r
        return r

    INF = float("inf")
    best = [[INF] * (C + 1) for _ in range(S + 1)]
    back = [[None] * (C + 1) for _ in range(S + 1)]
    best[0][0] = 0.0
    for i in range(S + 1):
        for k in range(C + 1):
            cur = best[i][k]
            if cur == INF:
                continue
            if k < C and cur + SKIP_CALL < best[i][k + 1]:
                best[i][k + 1], back[i][k + 1] = cur + SKIP_CALL, (i, k, None)
            if i == S:
                continue
            skip_step = NO_CALL + 1e-6 * (S - i)        # ties: leave a later step without a call rather than an earlier one
            if cur + skip_step < best[i + 1][k]:
                best[i + 1][k], back[i + 1][k] = cur + skip_step, (i, k, None)
            cost = 0.0
            for L in range(1, MAX_TRIES + 1):
                last = k + L - 1
                if last >= C:
                    break
                f = fit(i, last)
                if f is None:
                    break
                ans, same, ph = f
                tot = cost + (ANSWER_HIT if ans else FALLBACK) + (PROMPT_HIT if same else 0.0) + ph
                if cur + tot < best[i + 1][k + L]:
                    best[i + 1][k + L], back[i + 1][k + L] = cur + tot, (i, k, (k, last))
                cost += RETRY + ph + (2.0 if ans else 0.0)          # call `last` as a retry of a longer run
    blocks, i, k = {}, S, C
    while (i, k) != (0, 0):
        pi, pk, blk = back[i][k]
        if blk is not None:
            blocks[pi] = blk
        i, k = pi, pk

    for i, (k0, k1) in blocks.items():
        st, c = steps[i], calls[k1]
        ans = fit(i, k1)[0]
        parts = [f"[Earlier attempt, answer: {calls[k]['answer'][:300]}]\n{calls[k].get('reasoning') or '(no reasoning returned)'}"
                 for k in range(k0, k1)]
        parts.append(("[Final attempt]\n" if parts else "") + (c.get("reasoning") or ("(no reasoning returned)" if parts else "")))
        if not ans:
            parts.append(f"[The engine applied the fallback move instead of this answer: {c['answer'][:300]}]")
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        st["thinking"] = en("\n\n".join(x for x in parts if x).strip())
        if not st.get("_view") and "You are seat" in (c.get("prompt") or ""):
            st["state"] = en(common.state_part(c["prompt"]))
    return len(blocks)


# ------------------------------------------------------------------------------------------------ build
def build(bundle_dir, calls=None):
    b = load_source(bundle_dir)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    calls = calls or []
    summ, run, cfg = b["summary"] or {}, b["run"] or {}, b["config"] or {}
    info = summ.get("info") or {}
    params = info.get("params") or {}
    warn = []

    # ---- decisions, in log order
    decs = []
    for e in b["events"]:
        if e.get("event") != "move":
            continue
        raw = e.get("move") or "{}"
        act = parse_answer(raw) or {"action_id": str(raw)[:40], "payload": {}}
        view = e.get("view") or ""
        decs.append({"seat": int(e.get("actor", 0)), "raw": raw, "act": act, "aid": act.get("action_id"),
                     "pay": act.get("payload") if isinstance(act.get("payload"), dict) else {}, "view": view, "pv": parse_view(view)})
    if not decs:
        raise SystemExit("no decisions in the log")

    # ---- seats and names
    name_of = {}
    for d in decs:
        if d["pv"]["me"] and d["seat"] not in name_of:
            name_of[d["seat"]] = d["pv"]["me"]
    for s in range(2):
        name_of.setdefault(s, PLAYERS[s])
    seat_of = {v: k for k, v in name_of.items()}
    names = [name_of[s] for s in sorted(name_of)]
    other_of = {names[0]: names[1], names[1]: names[0]}
    for d in decs:
        d["me"] = name_of[d["seat"]]
        d["you"] = other_of[d["me"]]

    rows = next((d["pv"]["rows"] for d in decs if len(d["pv"]["rows"]) >= 2), None) or list(info.get("board_rows") or [])
    if not rows:
        raise SystemExit("board not found in the prompts")
    tile_col = lambda t: rows[t[0]][t[1]] if t and 0 <= t[0] < len(rows) and 0 <= t[1] < len(rows[0]) else "?"
    goal = next((d["pv"]["goal"] for d in decs if d["pv"]["goal"]), None) or (len(rows) - 1, len(rows[0]) - 1)

    # ---- starting state (the first prompt that shows each player)
    start = {}
    for d in decs:
        pv = d["pv"]
        if pv["pos"] and d["me"] not in start:
            start[d["me"]] = {"pos": pv["pos"], "chips": dict(pv["chips"])}
        if pv["opos"] and pv["other"] and pv["other"] not in start and pv["ochips"]:
            start[pv["other"]] = {"pos": pv["opos"], "chips": dict(pv["ochips"])}
    for nm in names:
        if nm not in start:
            raise SystemExit(f"no starting state for {nm}")
    first_path = {}
    for d in decs:
        if d["pv"]["path"] and d["me"] not in first_path:
            first_path[d["me"]] = d["pv"]["path"]

    # ---- phase and column of each decision
    keys, labels = [], []
    attempt, last_k, n_talk, last_turn = 1, 0, 0, 0
    for d in decs:
        pv = d["pv"]
        d["phase"] = PHASE.get(d["aid"]) or ("talk" if pv["neg"] and pv["turn"] is None else "other")
        if d["phase"] == "talk":
            if pv["neg"]:
                k = pv["neg"][1]
                if pv["neg"][0]:
                    attempt = pv["neg"][0]
                elif k < last_k:
                    attempt += 1
            else:
                k = n_talk // 2 + 1
            last_k, n_talk = k, n_talk + 1
            key = ("talk", attempt, k)
            lab = f"Talk {k}" + ("" if attempt == 1 else "b" if attempt == 2 else f"/{attempt}")
        else:
            t = pv["turn"] or max(1, last_turn)
            last_turn = t
            key, lab = ("turn", t), f"Turn {t}"
        if key not in keys:
            keys.append(key); labels.append(lab)
        d["x"], d["key"], d["lab"] = keys.index(key), key, lab
    n = len(keys)
    talk_x = [i for i, k in enumerate(keys) if k[0] == "talk"]
    turn_x = {k[1]: i for i, k in enumerate(keys) if k[0] == "turn"}
    turns = sorted(turn_x)

    # ---- end-of-turn states and the logged events of each turn
    tstate, tevents, inferred = {}, {}, []
    tl = info.get("turn_log") or []
    off = 1 if tl and min(int(r.get("turn", 0)) for r in tl) == 0 else 0     # the log counts turns from 0, the prompts from 1
    for r in tl:
        t = int(r.get("turn", 0)) + off
        tstate[t] = {nm: {"pos": tuple(s["pos"]), "chips": {c: int(s["chips"].get(c, 0)) for c in "RGB"}}
                     for nm, s in (r.get("state") or {}).items()}
        tevents[t] = [en(x) for x in r.get("events") or []]
    if not tl:                                                                 # Markdown source: the history lines of the prompts
        for d in decs:
            for t, evs, st in d["pv"]["hist"]:
                if t not in tstate and len(st) == 2:
                    tstate[t] = {nm: {"pos": s["pos"], "chips": s["chips"]} for nm, s in st.items()}
                    tevents[t] = [en(x) for x in re.split(r"(?<=\.)\s+(?=(?:P-Red|P-Blue|The|No|Contract)\b)", evs) if x.strip()]
    # a played turn that is not in the log (Markdown source, last turn): the latest prompt of that turn plus its own decisions
    for t in turns:
        if t in tstate:
            continue
        cur = {nm: dict(tstate.get(t - 1, {}).get(nm) or start[nm]) for nm in names}
        for d in [d for d in decs if d["key"] == ("turn", t)]:
            pv = d["pv"]
            if pv["pos"]:
                cur[d["me"]] = {"pos": pv["pos"], "chips": dict(pv["chips"])}
            if pv["opos"] and pv["ochips"]:
                cur[d["you"]] = {"pos": pv["opos"], "chips": dict(pv["ochips"])}
        last_by = {}
        for d in decs:
            if d["key"] == ("turn", t) and d["aid"] in ("move", "stay"):
                last_by[d["me"]] = d
        for nm, d in last_by.items():
            if d["aid"] == "move" and d["pv"]["pos"] and cur[nm]["pos"] == d["pv"]["pos"]:
                tile = tuple(canon_tile(d["pay"].get("tile")))
                col = tile_col(tile)
                cur[nm] = {"pos": tile, "chips": dict(cur[nm]["chips"])}
                cur[nm]["chips"][col] = cur[nm]["chips"].get(col, 0) - 1
        tstate[t] = {nm: {"pos": tuple(s["pos"]), "chips": dict(s["chips"])} for nm, s in cur.items()}
        tevents.setdefault(t, [])
        inferred.append(t)

    # cross-check: the first prompt of each turn shows the previous turn's end state
    for t in turns:
        if t - 1 in tstate:
            d = next((d for d in decs if d["key"] == ("turn", t)), None)
            if d and d["pv"]["pos"] and not d["pv"]["sofar"]:
                s = tstate[t - 1][d["me"]]
                if tuple(s["pos"]) != d["pv"]["pos"] or s["chips"] != d["pv"]["chips"]:
                    warn.append(f"turn {t}: {d['me']}'s prompt disagrees with the logged end of turn {t - 1}")

    def state_at(x):
        k = keys[x]
        if k[0] == "turn":
            return tstate[k[1]]
        return {nm: {"pos": tuple(start[nm]["pos"]), "chips": dict(start[nm]["chips"])} for nm in names}

    total = lambda ch: sum(ch.values())
    dist = lambda p: abs(goal[0] - p[0]) + abs(goal[1] - p[1])
    chips_x = {nm: [total(state_at(x)[nm]["chips"]) for x in range(n)] for nm in names}
    dist_x = {nm: [dist(state_at(x)[nm]["pos"]) for x in range(n)] for nm in names}
    end = state_at(n - 1)
    fin_turn = {nm: next((t for t in turns if tuple(tstate[t][nm]["pos"]) == tuple(goal)), None) for nm in names}

    # ---- logged events between decisions: a decision's prompt lists the turn's events so far
    def n_events_in(sofar, evs):
        s = (sofar or "").strip()
        if not s:
            return 0
        for k in range(len(evs), 0, -1):
            if s.startswith(" ".join(evs[:k]).strip()):
                return k
        return 0
    after = {}                                     # decision index -> logged events that followed it in the same turn
    for t in turns:
        idx = [i for i, d in enumerate(decs) if d["key"] == ("turn", t)]
        evs = tevents.get(t, [])
        ks = [n_events_in(decs[i]["pv"]["sofar"], evs) for i in idx] + [len(evs)]
        for j, i in enumerate(idx):
            after[i] = evs[ks[j]: max(ks[j], ks[j + 1])]

    # ---- contract and proposals, read from the decisions
    standing, contract, contract_x, contract_by = {}, None, None, None
    proposals = []                                 # trading-phase proposals with their answers
    pending = None
    for i, d in enumerate(decs):
        if d["aid"] == "propose_contract":
            standing[d["me"]] = (i, d["pay"])
        elif d["aid"] == "accept_contract":
            prop = standing.get(d["you"])
            contract, contract_x, contract_by = (prop[1] if prop else {}), d["x"], d["you"]
        elif d["aid"] in ("propose_trade", "propose_arrangement"):
            pending = {"i": i, "by": d["me"], "mode": "arrangement" if d["aid"] == "propose_arrangement" else "trade",
                       "offer": canon_bundle(d["pay"].get("offer")), "receive": canon_bundle(d["pay"].get("receive")), "answer": None}
            proposals.append(pending)
        elif d["aid"] in ("accept", "reject") and pending and pending["answer"] is None and pending["by"] == d["you"]:
            pending["answer"] = d["aid"]
            d["answers"] = pending
    if info.get("contract_accepted") is False:
        contract = None
    talk_decs = [d for d in decs if d["phase"] == "talk"]
    n_terms = sum(1 for d in talk_decs if d["aid"] == "propose_contract")
    # trading mode and contract type: the logged parameters; without them (Markdown source) the trading prompt's wording
    # ("propose one arrangement" under Pay-for-Partner, "propose one trade" under regular trading) and the contract terms seen
    first_text = (run.get("prompt") or {}).get("text", "") if isinstance(run.get("prompt"), dict) else ""
    trading = params.get("trading")
    if not trading:
        if any(d["aid"] in ("propose_arrangement", "honor", "refuse") for d in decs) or "Pay-for-Partner" in first_text \
                or any(re.search(r"propose one arrangement", d["view"]) for d in decs):
            trading = "pay_for_partner"
        elif any(d["aid"] == "propose_trade" for d in decs) or any(re.search(r"propose one trade", d["view"]) for d in decs):
            trading = "regular"
        else:
            trading = "unknown"
    ctype = params.get("contract")
    if not ctype:
        terms = [d["pay"] for d in decs if d["aid"] == "propose_contract"]
        ctype = ("none" if not talk_decs else "prog_trading" if any("tiles" in t for t in terms)
                 else "prog_points" if any("red_pays" in t or "blue_pays" in t for t in terms) else "unknown")
    honours = [d for d in decs if d["aid"] in ("honor", "refuse")]

    # ---- one step per decision
    def describe(i, d):
        """(head tail, feed text, grid label, kind) for decision i."""
        a, p, me, you = d["aid"], d["pay"], d["me"], d["you"]
        if a == "say":
            msg = en(str(p.get("message") or "").strip())
            short = msg if len(msg) <= 58 else msg[:55].rstrip() + "..."
            return (f"\u201c{short}\u201d" if msg else "says nothing (empty message)",
                    f"\u201c{msg}\u201d (no terms)" if msg else "(empty message, no terms)", "talk", "talk")
        if a == "propose_contract":
            msg = en(str(p.get("message") or "").strip())
            if "tiles" in p:
                ts = [t for t in p.get("tiles") or [] if isinstance(t, dict)]
                by = {}
                for t in ts:
                    tile = tuple(canon_tile(t.get("tile")))
                    by.setdefault(canon_player(t.get("giver")), []).append(f"({tile[0]},{tile[1]}) {WORD.get(tile_col(tile), '?')}")
                terms = "; ".join(f"{g} pays {other_of.get(g, 'the other')}'s chip on " + ", ".join(v) for g, v in by.items())
                head = f"proposes a {len(ts)}-tile contract"
            else:
                terms = f"P-Red pays {p.get('red_pays', 0)} points, P-Blue pays {p.get('blue_pays', 0)} points at the end"
                head = "proposes a points contract"
            return head, f"Proposes contract terms: {terms}." + (f" \u201c{msg}\u201d" if msg else ""), "terms", "propose"
        if a == "accept_contract":
            msg = en(str(p.get("message") or "").strip())
            return (f"accepts {you}'s contract", f"Accepts {you}'s standing proposal: the contract is formed."
                    + (f" \u201c{msg}\u201d" if msg else ""), "signs", "accept")
        if a == "skip":
            return "skips trading", "Skips trading: proposes nothing.", "skip", "pass"
        if a in ("propose_trade", "propose_arrangement"):
            off, rec = canon_bundle(p.get("offer")), canon_bundle(p.get("receive"))
            ans = next((q["answer"] for q in proposals if q["i"] == i), None)
            if a == "propose_arrangement":
                txt = "Proposes an arrangement: " + "; ".join(x for x in (
                    f"it pays for {bundle_text(off, 'arrangement', me, you)}" if off else "",
                    f"{you} pays for {bundle_text(rec, 'arrangement', you, me)}" if rec else "") if x) + "."
                head = "proposes: pays " + (", ".join(f"{v} {WORD.get(c, c)}" for c, v in off.items()) or "nothing") \
                       + ", asks " + (", ".join(f"{v} {WORD.get(c, c)}" for c, v in rec.items()) or "nothing")
            else:
                txt = f"Offers {bundle_text(off, 'trade', me, you) or 'nothing'} for {bundle_text(rec, 'trade', you, me) or 'nothing'}."
                head = f"offers {bundle_text(off, 'trade', me, you) or 'nothing'} for {bundle_text(rec, 'trade', you, me) or 'nothing'}"
            if ans:
                txt += f" {you} {'accepts' if ans == 'accept' else 'rejects'}."
            return head, txt, "offer", "propose"
        if a in ("accept", "reject"):
            verb = "accepts" if a == "accept" else "rejects"
            return f"{verb} {you}'s proposal", f"{verb.capitalize()} {you}'s proposal.", "acc" if a == "accept" else "rej", \
                "accept" if a == "accept" else "reject"
        if a == "move":
            tile = tuple(canon_tile(p.get("tile")))
            col = tile_col(tile)
            t = d["key"][1] if d["key"][0] == "turn" else None
            ok = t is not None and tuple(tstate.get(t, {}).get(me, {}).get("pos", ())) == tile
            tail = f"moves to ({tile[0]},{tile[1]}), a {WORD.get(col, '?')} tile" + (" (the goal)" if tile == tuple(goal) else "")
            if not ok:
                tail = f"tries ({tile[0]},{tile[1]}), a {WORD.get(col, '?')} tile, and stays"
            return tail, None, f"{tile[0]},{tile[1]}", TILE_KIND.get(col, "pass")
        if a == "stay":
            pos = d["pv"]["pos"]
            return (f"stays at ({pos[0]},{pos[1]})" if pos else "stays"), None, "stay", "pass"
        if a in ("honor", "refuse"):
            m = RX_ASKED.search(d["view"].split("\n")[-1] if d["view"] else "") or RX_ASKED.search(d["view"][-600:])
            col = f"a {m.group(1)} chip" if m else "a chip"
            if a == "honor":
                return f"honours: pays {col} for {you}'s move", f"Honours its promise: pays {col} for {you}'s move.", "paid", "accept"
            return f"refuses to pay {col} for {you}'s move", f"Refuses to pay {col} for {you}'s move: the promise is broken.", "no", "reject"
        return a or "unknown action", f"Action {a}.", (a or "?")[:5], "pass"

    steps, feed, grid_info = [], [], []
    for i, d in enumerate(decs):
        tail, text, gl, kind = describe(i, d)
        act = d["raw"] if en_obj(d["act"]) == d["act"] else json.dumps(en_obj(d["act"]), ensure_ascii=False)
        steps.append({"seat": d["seat"], "x": d["x"], "head": f"{d['lab']} \u00b7 {tail}", "kind": kind, "act": act,
                      "state": en(common.state_part(d["view"])) if d["view"] else "Not recorded.", "thinking": "",
                      "_act": d["act"], "_view": d["view"], "_phase": d["phase"],
                      "_col": ("talk", d["key"][2]) if d["key"][0] == "talk" else ("turn", d["key"][1])})
        evs = after.get(i, [])
        if text is None:                                       # moves: the logged line of the move, when there is one
            own = next((e for e in evs if re.match(re.escape(d["me"]) + r"\b", e)), None)
            if own:
                text = own[len(d["me"]):].strip()
                text = text[:1].upper() + text[1:]
                evs = [e for e in evs if e is not own]
            else:
                text = tail[:1].upper() + tail[1:] + "."
        feed.append({"x": d["x"], "step": i, "who": d["me"], "kind": kind, "text": text})
        for e in evs:
            feed.append({"x": d["x"], "step": i, "who": "Game log", "text": e})
        grid_info.append((gl, kind, text))
        # the end of the negotiation
        if d["phase"] == "talk" and (i + 1 == len(decs) or decs[i + 1]["phase"] != "talk"):
            k_turns = len({dd["key"] for dd in talk_decs})
            if contract is not None and contract_x is not None:
                msg = (f"Negotiation over after {k_turns} turn{'s' if k_turns != 1 else ''} ({len(talk_decs)} messages): "
                       f"{other_of.get(contract_by, 'the other player')} accepted {contract_by}'s terms in "
                       f"{labels[contract_x].lower()}, so the contract binds both.")
            else:
                msg = (f"Negotiation over after {k_turns} turn{'s' if k_turns != 1 else ''} ({len(talk_decs)} messages): "
                       + ("no terms were ever proposed, so there is no contract." if n_terms == 0
                          else f"{n_terms} proposal{'s' if n_terms != 1 else ''} of terms, none accepted, so there is no contract."))
            feed.append({"x": d["x"], "step": i, "who": "Game", "text": msg})
    n_tied = attach_calls(steps, calls)
    if calls:
        n_think = sum(1 for st in steps if st["thinking"])
        if n_think < 0.9 * len(steps):
            sys.stderr.write(f"warning: {len(calls)} model calls given, but only {n_tied} of {len(steps)} decisions were tied to a "
                             f"call and {n_think} received reasoning\n")
    md_think = b.get("md_think") or []
    if not calls and len(md_think) == len(steps):
        for st, mt in zip(steps, md_think):
            th = mt["thinks"]
            if not th:
                continue
            parts = [f"[Earlier attempt, answer: {a[:300]}]\n{r}" for a, r in zip(mt["earlier"], th[:-1])]
            parts.append(("[Final attempt]\n" if parts else "") + th[-1])
            if mt["rescued"]:
                parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
            st["thinking"] = en("\n\n".join(parts))
    for st in steps:
        for k in ("_act", "_view", "_phase", "_col"):
            st.pop(k, None)
    has_reason = any(st["thinking"] for st in steps)

    # ---- scores
    rewards = summ.get("rewards") or {}
    scale = info.get("reward_scale") or (GOAL_POINTS + CHIP_POINTS * (32 - 6))
    scores = {}
    for nm in names:
        if (info.get("scores") or {}).get(nm) is not None:
            scores[nm] = info["scores"][nm]
        elif str(seat_of[nm]) in rewards:
            scores[nm] = round(float(rewards[str(seat_of[nm])]) * scale, 6)
            scores[nm] = int(scores[nm]) if float(scores[nm]).is_integer() else scores[nm]
    finished = {nm: tuple(end[nm]["pos"]) == tuple(goal) for nm in names}
    for nm in names:
        rule = GOAL_POINTS + CHIP_POINTS * total(end[nm]["chips"]) if finished[nm] else 0
        if nm in scores and scores[nm] != rule and ctype != "prog_points":
            warn.append(f"{nm}: logged score {scores[nm]} but 20 + 5 x chips rule gives {rule}")
        if str(seat_of[nm]) in rewards and nm in scores and abs(float(rewards[str(seat_of[nm])]) - scores[nm] / scale) > 1e-9:
            warn.append(f"{nm}: reward {rewards[str(seat_of[nm])]} is not score / {scale}")
    baselines = info.get("baselines") or {
        nm: (GOAL_POINTS + CHIP_POINTS * (total(start[nm]["chips"]) - first_path[nm]["len"]) if nm in first_path and not first_path[nm]["missing"] else 0)
        for nm in names}
    missing = {nm: sum((first_path.get(nm) or {}).get("missing", {}).values()) for nm in names}
    bclass = info.get("board_class")
    if not bclass:
        short_of = [nm for nm in names if missing[nm] > 0]
        bclass = "independent" if not short_of else "mutually_dependent" if len(short_of) == 2 else "asymmetric"
    board_id = info.get("paper_board_id")
    reward_of = {nm: (float(rewards[str(seat_of[nm])]) if str(seat_of[nm]) in rewards else scores.get(nm, 0) / scale) for nm in names}
    plays = info.get("players") or {}
    if plays:
        moved_hands = sum(int((plays.get(nm) or {}).get("chips_received", 0) or 0) for nm in names)
    elif contract is None and not honours and (trading == "pay_for_partner" or not any(q["answer"] == "accept" for q in proposals)):
        moved_hands = 0           # by the rules chips change hands only by an accepted trade, an honoured promise or a contract tile
    else:
        moved_hands = None
    kept = info.get("promises_kept", sum(1 for d in honours if d["aid"] == "honor"))
    broken = info.get("promises_broken", sum(1 for d in honours if d["aid"] == "refuse"))
    accepted = [q for q in proposals if q["answer"] == "accept"]
    n_skip = sum(1 for d in decs if d["aid"] == "skip")
    n_turns = len(turns)

    # ---- numbers in words
    def chips_words(ch, nm):
        own = "R" if nm == "P-Red" else "B"
        order = [own] + [c for c in "RBG" if c != own and c != "G"] + ["G"]
        return ", ".join(f"{ch.get(c, 0)} {WORD[c]}" for c in order if ch.get(c, 0) or c in (own, "G"))
    sc = [scores.get(nm) for nm in names]
    both = all(finished.values())

    # ---- title
    c_part = "No contract" if contract is None else ("A points contract" if "tiles" not in contract
                                                     else f"A {len(contract.get('tiles') or [])}-tile contract")
    x_part = "no exchanges" if not proposals else f"{len(accepted)} of {len(proposals)} proposed exchanges accepted"
    if both:
        same_turn = len({fin_turn[nm] for nm in names}) == 1
        own_way = (moved_hands == 0) if moved_hands is not None else (not accepted and contract is None)
        outcome = ("each player pays its own way to the goal" if own_way else "both players reach the goal") \
            + (f" in turn {fin_turn[names[0]]}" if same_turn else "") \
            + (f" and both score {sc[0]}" if sc[0] == sc[1] else f", scoring {sc[0]} and {sc[1]}")
    elif any(finished.values()):
        w = next(nm for nm in names if finished[nm])
        outcome = f"only {w} reaches the goal ({scores.get(w)} to {scores.get(other_of[w])})"
    else:
        outcome = "neither player reaches the goal (0 to 0)"
    title = f"{c_part}, {x_part}: {outcome}"

    # ---- tiles
    def score_note():
        bits = []
        if both and sc[0] == sc[1] and total(end[names[0]]["chips"]) == total(end[names[1]]["chips"]):
            c = total(end[names[0]]["chips"])
            same_turn = fin_turn[names[0]] == fin_turn[names[1]]
            bits.append(f"Both reached the goal{f' in turn {fin_turn[names[0]]}' if same_turn else ''} with {c} chips left: "
                        f"20 + 5 {TIMES} {c} = {sc[0]} each")
        else:
            for nm in names:
                if finished[nm]:
                    c = total(end[nm]["chips"])
                    bits.append(f"{nm} reached the goal in turn {fin_turn[nm]} with {c} chips: 20 + 5 {TIMES} {c} = {scores.get(nm)}"
                                if scores.get(nm) == 20 + 5 * c else f"{nm} reached the goal in turn {fin_turn[nm]}: {scores.get(nm)}")
                else:
                    bits.append(f"{nm} did not reach the goal: {scores.get(nm)}")
        joint = sum(v for v in sc if v is not None)
        r = [reward_of[nm] for nm in names]
        rtxt = f"reward {r[0]:.3f} each" if abs(r[0] - r[1]) < 1e-12 else f"rewards {r[0]:.3f} and {r[1]:.3f}"
        return "; ".join(bits) + f". Joint {joint} of 140; {rtxt} (score \u00f7 {scale})."
    tiles = [{"k": f"Scores, {names[0]} \u00b7 {names[1]}", "v": f"{sc[0]} \u00b7 {sc[1]}", "s": score_note()}]
    cls_label = {"independent": "Independent", "mutually_dependent": "Mutually dependent", "asymmetric": "Asymmetric",
                 "asymmetric_blue": "Asymmetric"}.get(bclass, str(bclass).replace("_", " ").capitalize())
    if bclass == "independent":
        lens = {first_path[nm]["len"] for nm in names if nm in first_path}
        bnote = (f"Each player's own chips pay its whole {str(lens.pop()) + '-move ' if len(lens) == 1 else ''}best path, so neither "
                 f"needs the other. Baselines {baselines.get(names[0])} and {baselines.get(names[1])} already add up to 140, the "
                 "highest joint score, so the two cannot both beat them.")
    elif bclass == "mutually_dependent":
        bnote = (f"Both are short of chips for their best paths ({names[0]} {missing[names[0]]}, {names[1]} {missing[names[1]]} missing), "
                 "so each needs the other; baselines 0.")
    else:
        weak = next((nm for nm in names if missing[nm] > 0), names[1])
        bnote = (f"{other_of[weak]}'s own chips pay its best path; {weak} is {missing[weak]} chip{'s' if missing[weak] != 1 else ''} short "
                 f"and needs help. Baselines {baselines.get(names[0])} and {baselines.get(names[1])}.")
    tiles.append({"k": f"Board {board_id}" if board_id is not None else "Board", "v": cls_label,
                  "s": "Rows " + " \u00b7 ".join(rows) + ". " + bnote})
    k_turns = len({d["key"] for d in talk_decs})
    if ctype != "none":
        if contract is None:
            declines = all(re.search(r"\bno contract\b", str(d["pay"].get("message") or ""), re.I) for d in talk_decs if d["aid"] == "say")
            tiles.append({"k": "Contract", "v": "None",
                          "s": f"{len(talk_decs)} messages over {k_turns} negotiation turns"
                               + (", every one declining a contract" if declines and n_terms == 0 else "")
                               + ("; no terms were ever proposed." if n_terms == 0 else f"; {n_terms} proposals of terms, none accepted.")})
        else:
            if "tiles" in contract:
                ct = [t for t in contract.get("tiles") or [] if isinstance(t, dict)]
                giv = {}
                for t in ct:
                    giv[canon_player(t.get("giver"))] = giv.get(canon_player(t.get("giver")), 0) + 1
                used = (info.get("contract") or {}).get("tiles_used")
                tiles.append({"k": "Contract", "v": f"{len(ct)} tiles",
                              "s": f"{other_of.get(contract_by, '?')} accepted {contract_by}'s terms in {labels[contract_x].lower()}: "
                                   + ", ".join(f"{g} pays on {c} tile{'s' if c != 1 else ''} for {other_of.get(g, 'the other')}"
                                               for g, c in giv.items())
                                   + (f"; {used} of {len(ct)} used in play." if used is not None else ".")})
            else:
                tiles.append({"k": "Contract", "v": "Points",
                              "s": f"P-Red pays {contract.get('red_pays', 0)}, P-Blue pays {contract.get('blue_pays', 0)} points if it finishes."})
    if not proposals:
        # a player on the goal is no longer asked to trade, so count each player's trading turns (all of them skips here)
        k_trade = {nm: sum(1 for d in decs if d["me"] == nm and d["aid"] == "skip") for nm in names}
        if all(k_trade[nm] == n_turns for nm in names):
            who = f"Both players skipped trading in all {n_turns} turns ({n_skip} skips)"
        else:
            cl = []
            for nm in names:
                k = k_trade[nm]
                cl.append(f"{nm} was never asked to trade" if k == 0 else
                          f"{nm} skipped all {k} trading turn{'s' if k != 1 else ''} it had"
                          + (f" (it reached the goal in turn {fin_turn[nm]})" if fin_turn[nm] and k < n_turns else ""))
            who = " and ".join(cl)
        tiles.append({"k": "Exchanges", "v": "0 proposed",
                      "s": f"{who}: no " + ("arrangement, so no promise to keep or break." if trading == "pay_for_partner"
                                            else "trade.")})
    else:
        tiles.append({"k": "Exchanges", "v": f"{len(accepted)} of {len(proposals)}",
                      "s": f"{len(proposals)} proposed, {len(accepted)} accepted"
                           + (f"; promises kept {kept}, broken {broken}." if trading == "pay_for_partner" else ".")})
    tiles = tiles[:4]

    # ---- charts
    def coincide_note(series):
        return " The two players' bars are equal in every column." if series[names[0]] == series[names[1]] else ""
    spent = []
    for nm in names:
        s0, s1 = start[nm]["chips"], end[nm]["chips"]
        spent.append(f"{nm} {'went ' if not spent else ''}from {chips_words(s0, nm)} to {chips_words(s1, nm)}")
    hands = (" No chip changed hands." if moved_hands == 0 else
             f" {moved_hands} chip{'s' if moved_hands != 1 else ''} changed hands." if moved_hands else "")
    chips_n = lambda k: f"{k} chip{'s' if k != 1 else ''}"
    tiles_txt = lambda k: f"{k} tile{'s' if k != 1 else ''}"
    # where play starts and where it ends (the x axis labels only every few columns on a narrow page)
    if both and fin_turn[names[0]] == fin_turn[names[1]]:
        fin_s = f"both reached the goal in turn {fin_turn[names[0]]}"
    else:
        fin_s = ", ".join(f"{nm} reached the goal in turn {fin_turn[nm]}" if fin_turn[nm] else f"{nm} never reached the goal"
                          for nm in names)
    last_note = ", the last column" if turns and fin_turn[names[0]] == fin_turn[names[1]] == turns[-1] else ""
    if talk_x and turns:
        phase_s = (f"{labels[talk_x[0]]} to {labels[talk_x[-1]]} are the contract negotiation, when nothing moves; play runs from "
                   f"Turn {turns[0]} to Turn {turns[-1]}, and {fin_s}{last_note}.")
    elif turns:
        phase_s = f"Play runs from Turn {turns[0]} to Turn {turns[-1]}, and {fin_s}{last_note}."
    else:
        phase_s = ""
    # chips minus tiles to the goal: constant when every chip spent bought one step towards the goal and none changed hands;
    # then the chips staircase also counts down the moves left, and a separate distance chart would repeat it
    gap = {nm: {chips_x[nm][x] - dist_x[nm][x] for x in range(n)} for nm in names}
    steady = moved_hands == 0 and all(len(v) == 1 for v in gap.values())
    moves_s = ""
    if steady:
        gv = {nm: next(iter(gap[nm])) for nm in names}
        g_txt = (f"stays at {gv[names[0]]} for both players" if gv[names[0]] == gv[names[1]]
                 else f"stays at {gv[names[0]]} for {names[0]} and {gv[names[1]]} for {names[1]}")
        st0 = {nm: f"{total(start[nm]['chips'])} chips with {tiles_txt(dist(start[nm]['pos']))} to go" for nm in names}
        en0 = {nm: (f"{total(end[nm]['chips'])} chips on the goal" if finished[nm]
                    else f"{total(end[nm]['chips'])} chips with {tiles_txt(dist(end[nm]['pos']))} to go") for nm in names}
        both_s = lambda d: (f"{d[names[0]]}" if d[names[0]] == d[names[1]] else f"{names[0]} {d[names[0]]}, {names[1]} {d[names[1]]}")
        moves_s = (f" Every chip spent bought one step towards the goal ({goal[0]},{goal[1]}), so the bars also count down the moves "
                   f"left: chips minus tiles to go {g_txt} ({both_s(st0)} at the start, {both_s(en0)} at the end).")
    chart_chips = {
        "title": "Chips in hand",
        "sub": f"Chips each player holds at the end of each column; every move costs one chip of the tile's colour, and a player "
               f"on the goal scores 20 + 5 per chip left. {phase_s} {'; '.join(spent)}.{hands}{moves_s}{coincide_note(chips_x)}",
        "height": 210, "fmt": "int", "lines": [],
        "bars": [{"name": nm, "color": SEAT_COLOR.get(nm, "c1"), "v": chips_x[nm]} for nm in names]}

    # chips still missing for each player's best path, as its last prompt in each column reported it (shown when chips and
    # distance do not move together: then the game's tension is who lacks which chips)
    chart_miss = None
    if not steady:
        miss_x = {nm: [None] * n for nm in names}
        for d in decs:
            pv = d["pv"]
            if pv["path"] is not None:
                miss_x[d["me"]][d["x"]] = sum(pv["path"]["missing"].values())
            elif pv["pos"] is not None and tuple(pv["pos"]) == tuple(goal):
                miss_x[d["me"]][d["x"]] = 0
        for nm in names:
            for x in range(n):
                if miss_x[nm][x] is None and tuple(state_at(x)[nm]["pos"]) == tuple(goal):
                    miss_x[nm][x] = 0
        vals = {nm: [v for v in miss_x[nm] if v is not None] for nm in names}
        if all(vals[nm] for nm in names) and any(v for nm in names for v in vals[nm]):
            helps = (["a contract tile"] if ctype == "prog_trading" else []) \
                + (["an arrangement the partner honours"] if trading == "pay_for_partner" else ["a trade"] if trading == "regular" else [])
            arc = []
            for nm in names:
                a, z = vals[nm][0], vals[nm][-1]
                if not any(vals[nm]):
                    arc.append(f"{nm} was never short")
                elif all(v == a for v in vals[nm]):
                    arc.append(f"{nm} was {chips_n(a)} short throughout")
                else:
                    arc.append(f"{nm} started " + (f"{chips_n(a)} short" if a else "with nothing missing") + " and ended "
                               + (f"{chips_n(z)} short" if z else "with nothing missing")
                               + (f" (at most {max(vals[nm])})" if max(vals[nm]) > max(a, z) else ""))
            chart_miss = {
                "title": "Chips still missing for the best path",
                "sub": "How many chips each player lacked for its cheapest path to the goal, as its last prompt in each column "
                       "reported it (“chips missing”, before its move; 0 = its own chips suffice, or it is on the goal). "
                       + (f"A player short of chips needs the other's: {' or '.join(helps)}. " if helps else "")
                       + "; ".join(arc) + "." + coincide_note(miss_x),
                "height": 180, "fmt": "int", "lines": [],
                "bars": [{"name": nm, "color": SEAT_COLOR.get(nm, "c1"), "v": miss_x[nm]} for nm in names]}
        elif all(vals[nm] for nm in names):
            chart_chips["sub"] += " Neither player was ever short of chips for its best path."

    # grid: per player its offer (contract message or trading proposal), its replies (to the other's proposal and to an honour
    # request; rows only when there are any) and its move, one column per negotiation or game turn. Cells are never overwritten:
    # two decisions in one cell (a reply to a proposal and to an honour request in the same turn) share it.
    has_reply = any(d["phase"] in ("respond", "honour") for d in decs)
    rkinds = ["offer"] + (["reply"] if has_reply else []) + ["move"]
    grows = [f"{nm} {rk}" for nm in names for rk in rkinds]
    row_of = lambda nm, rk: names.index(nm) * len(rkinds) + rkinds.index(rk)
    cells = [[None] * n for _ in grows]
    REFUSALS = ("reject", "refuse")

    def put(r, x, cell, aid):
        old = cells[r][x]
        cell["_aids"] = [aid]
        if old is None:
            cells[r][x] = cell
        else:
            old["t"] = "·".join(t for t in (old.get("t"), cell.get("t")) if t)
            old["tip"] = old["tip"].rstrip() + " " + cell["tip"]
            old["_aids"].append(aid)
            old.pop("o", None)
        tgt = cells[r][x]
        if any(a in REFUSALS for a in tgt["_aids"]):          # a rejection or a broken promise stays visible after a merge
            tgt["c"] = KINDS["reject"][1]
    for i, d in enumerate(decs):
        gl, kind, text = grid_info[i]
        cell = {"c": KINDS[kind][1], "t": gl, "tip": text if len(text) <= 160 else text[:157] + "..."}
        if d["phase"] in ("talk", "trade"):
            rk = "offer"
        elif d["phase"] in ("respond", "honour"):
            rk = "reply"
        else:
            rk = "move"
            if d["aid"] == "move" and d["key"][0] == "turn":
                s_end = tstate[d["key"][1]][d["me"]]
                left = dist(s_end["pos"])
                cell["tip"] = (text.rstrip(".") + f". End of turn: {chips_words(s_end['chips'], d['me'])}; "
                               + ("on the goal." if left == 0 else f"{tiles_txt(left)} from the goal."))
                if "tries" in steps[i]["head"]:
                    cell["o"] = 0.45
        put(row_of(d["me"], rk), d["x"], cell, d["aid"])
    for nm in names:                                              # turns after reaching the goal
        for t in turns:
            x, r = turn_x[t], row_of(nm, "move")
            if cells[r][x] is None and fin_turn[nm] and t > fin_turn[nm]:
                cells[r][x] = {"c": "c3", "o": 0.25, "t": "", "tip": f"On the goal since turn {fin_turn[nm]}; no more moves."}
    n_placed = sum(len(c.get("_aids", [])) for row in cells for c in row if c)
    if n_placed != len(decs):
        warn.append(f"decision grid holds {n_placed} decisions, the log has {len(decs)}")
    for row in cells:
        for c in row:
            if c:
                c.pop("_aids", None)
    used_tiles = [c for c in "RBG" if any(d["aid"] == "move" and tile_col(tuple(canon_tile(d["pay"].get("tile")))) == c for d in decs)]
    glegend = [[KINDS[k][0] if k not in ("move_r", "move_b", "move_g") else KINDS[k][0].replace("Move onto a ", "").capitalize(), KINDS[k][1]]
               for k in KINDS if any(grid_info[i][1] == k for i in range(len(decs)))]
    cols_s = (f"One column per negotiation turn ({labels[talk_x[0]]} to {labels[talk_x[-1]]}) and per game turn "
              f"(Turn {turns[0]} to Turn {turns[-1]})" if talk_x and turns else "One column per game turn")
    grid_sub = (cols_s + ". Offer rows: the contract message in a negotiation turn (talk = a message without terms) and the trading "
                "proposal in a game turn (skip = none). "
                + ("Reply rows: the answer to the other's proposal (acc, rej) and to an honour request (paid; no = a broken "
                   "promise), joined by a dot when both fall in one turn. " if has_reply else "")
                + "Move rows: the tile the player stepped onto, as row,column, in the tile's colour"
                + (" (red tiles in orange)" if "R" in used_tiles else "") + f"; {fin_s}. Hover a column for the messages, "
                "the logged moves and the chips and tiles left.")
    chart_grid = {"title": "Every decision, column by column", "sub": grid_sub, "height": 140,
                  "grid": {"rows": grows, "cells": cells, "rowHeight": 28, "legend": glegend}}
    charts = [chart_chips, chart_grid] + ([chart_miss] if chart_miss else [])
    if proposals or honours:
        per = {"Proposals": [0] * n, "Accepted": [0] * n, "Promises honoured": [0] * n, "Promises refused": [0] * n}
        for q in proposals:
            per["Proposals"][decs[q["i"]]["x"]] += 1
            if q["answer"] == "accept":
                per["Accepted"][decs[q["i"]]["x"]] += 1
        for d in honours:
            per["Promises honoured" if d["aid"] == "honor" else "Promises refused"][d["x"]] += 1
        colors = {"Proposals": "c4", "Accepted": "ink", "Promises honoured": "c3", "Promises refused": "hold"}
        bars = [{"name": k, "color": colors[k], "v": v} for k, v in per.items() if any(v)]
        extra = ""
        if trading == "pay_for_partner" and broken > sum(per["Promises refused"]):
            extra = f" {broken - sum(per['Promises refused'])} more promise(s) broke because the partner lacked the chip (no decision)."
        charts.append({"title": "Exchanges and promises per turn",
                       "sub": "Trading proposals made and accepted in each column, and the honour requests answered (a refused "
                              "promise stays owed and is asked again)." + extra,
                       "height": 180, "fmt": "int", "lines": [], "bars": bars})
    charts = charts[:4]

    # ---- result line
    res = []
    for nm in names:
        res.append(f"{nm} {'on the goal' if finished[nm] else 'not on the goal'} with {chips_words(end[nm]['chips'], nm)}: "
                   f"score {scores.get(nm)}, baseline {baselines.get(nm)}, reward {reward_of[nm]:.3f}")
    reason = info.get("end_reason") or ("both_finished" if both else None)
    reason_txt = {"both_finished": "both players reached the goal", "idle": "3 turns in a row without a move or accepted exchange",
                  "exhausted": "nobody could move any more", "max_turns": "the turn cap"}.get(reason, "")
    feed.append({"x": n - 1, "step": len(steps) - 1, "who": "Result",
                 "text": (f"Game over after {n_turns} turns" + (f" ({reason_txt})" if reason_txt else "") + ". ") + "; ".join(res) + "."})

    # ---- text
    model = (b.get("meta") or {}).get("model") or run.get("model") or ""
    model = model.split("/")[-1] if model else "the model"
    date = str((b.get("meta") or {}).get("recorded_at") or run.get("recorded_at") or DEFAULT_DATE)[:10]
    seed = run.get("seed", cfg.get("seed"))
    variant = os.path.splitext(os.path.basename(cfg.get("variant") or ""))[0]
    s_red, s_blue = start.get("P-Red", {}).get("chips", {}), start.get("P-Blue", {}).get("chips", {})
    contract_sent = {"prog_trading": "Before play they may negotiate a contract that names, tile by tile, who pays the other's chip "
                                     "there (enforced automatically)",
                     "prog_points": "Before play they may negotiate a contract that moves points from a finisher to the other",
                     "none": ""}.get(ctype, "Before play they may negotiate a binding contract")
    trade_sent = ("each turn either may also propose a Pay-for-Partner arrangement, a promise to pay for some of the other's moves "
                  "that can later be broken" if trading == "pay_for_partner" else "each turn either may also propose a trade of chips")
    second = (contract_sent + "; " + trade_sent if contract_sent else trade_sent[:1].upper() + trade_sent[1:]) + "."
    lede = (f"Colored Trails (CT-Bench): P-Red and P-Blue both start on the top-left tile of a {len(rows)}x{len(rows[0])} board of red, "
            f"blue and green tiles and try to reach the green goal at the bottom right; every move costs one chip of the tile's colour, "
            f"and P-Red starts with {s_red.get('R', 0)} red and {s_red.get('G', 0)} green chips, P-Blue with {s_blue.get('B', 0)} blue "
            f"and {s_blue.get('G', 0)} green. {second} A player on the goal scores 20 points plus 5 per chip left, anyone else 0, and "
            "each maximises only its own score.")
    paper = PAPER.get((trading, ctype))
    mode_txt = {"pay_for_partner": "Pay-for-Partner trading", "regular": "regular trading"}.get(trading, trading)
    ctype_txt = {"prog_trading": "a Prog-Trading contract negotiation", "prog_points": "a Prog-Points contract negotiation",
                 "none": "no contract"}.get(ctype, "a contract negotiation")
    foot = (f"Data: the {date} recording of eco_colored_trails (seed {seed}"
            + (f", config {variant}" if variant else "") + f": {mode_txt} with {ctype_txt}; "
            + (f"the paper's board {board_id}, {cls_label.lower()}" if board_id is not None
               else ("an " if cls_label[:1] in "AEIOU" else "a ") + f"{cls_label.lower()} board")
            + f"), both players played by {model}. Every number on this page is computed from its game log: the {len(decs)} decisions, "
            "the logged events of each turn and the end-of-turn positions and chips. "
            + ("Observations and reasoning come from the saved model calls. " if has_reason else
               "This recording was made before reasoning was saved; the game will be re-recorded with reasoning. "
               "Observations are the prompts each player saw (rules and action format cut). ")
            + (f"The end of turn {', '.join(map(str, inferred))} is inferred from the last prompt and move. " if inferred else "")
            + f"Score: 20 + 5 {TIMES} chips left for a player on the goal, 0 elsewhere; reward = score \u00f7 {scale}, the highest score "
            "the rules allow. Baseline: the score from walking one's best path with one's own chips alone (70 when they suffice, else 0). "
            "Game from Wyse, Bustos, Volkova and Kleiman-Weiner (2026), \u201cCommitment To Cooperation With Self-Negotiated "
            "Contracts\u201d (COLM 2026)" + (f", who report {paper}, averaged over six models on their 80 boards." if paper else "."))
    if warn:
        sys.stderr.write("warnings:\n  " + "\n  ".join(warn) + "\n")

    used = []
    for st in steps:
        if st["kind"] not in used:
            used.append(st["kind"])
    out = {"id": GID, "name": "Colored Trails", "title": title, "lede": lede, "seatNames": names, "tiles": tiles, "foot": foot,
           "x": {"n": n, "labels": labels}, "charts": charts, "kinds": {k: KINDS[k] for k in KINDS if k in used}, "steps": steps,
           "feed": feed, "feedTitle": "Negotiation and game transcript",
           "feedSub": "Every message, proposal and move in order, with the game's own log line for each move. The selected step is "
                      "highlighted; later lines are hidden."}
    out["eyebrow"] = (common.EYEBROW.format(gid=GID, date=date) if has_reason
                      else f"social_game_env \u00b7 {GID} \u00b7 {model} \u00b7 recorded {date} \u00b7 reasoning not saved")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out_dir = sys.argv[1], sys.argv[2]
    calls_arg = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    common.write(build(src, calls_arg), out_dir, date=recorded_date(src))
