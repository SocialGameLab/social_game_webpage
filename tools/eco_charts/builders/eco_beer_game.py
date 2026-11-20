"""Replay data for eco_beer_game (the beer distribution game: four supply-chain stages ordering under local information; Jiang,
Hong, Yang, Yang, Jiang, Guo and Xiao, "Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage
Supply Chains with LLM-Based Simulation", ACL 2026).

usage: python3 -I eco_beer_game.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md), a file inside
          one, or the rendered Markdown replay of one (one '## Step N . P<seat>' section per decision: what the seat saw and the
          move it made).
  OUT_DIR where <OUT_DIR>/eco_beer_game.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see load_calls). Calls are tied to
          decisions in time order: a call belongs to the next decision of the same seat (and, when its prompt names it, the same
          round); the call whose answer is the logged order is that decision's call (earlier ones were format retries; their
          reasoning is kept above it). Its reasoning becomes the step's thinking and its prompt the observation.

Each round every stage orders once, the Retailer first (the engine buffers the round), so each round gives one decision per
seat. The per-round record of every stage (incoming order, arrival, shipment, stock, backlog, supply line, cost) is rebuilt
from what the stages were shown in their prompts and the game's shipping rule (rules: Round Procedure, step 3); the last
round, which no prompt reports, is settled with that rule. When the bundle's summary carries the engine's own record
(info.series) the two are compared value by value and every disagreement is printed as a warning; the engine's record is
used for any stage no prompt covers (a scripted stage). Costs, scores and order variances follow the rules file (Reward,
Metrics) and are checked against the summary."""
import json, math, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_beer_game"
DEFAULT_DATE = "2026-11-09"
MINUS, DOT, TIMES, DIV, ARROW = "\u2212", "\u00b7", "\u00d7", "\u00f7", "\u2192"
UP, DOWN = "\u2191", "\u2193"
# One color per stage on the whole page (lines, grid rows, decision blocks); c1-c4 mean nothing else here.
STAGE_COLORS = ("c1", "c2", "c3", "c4", "ink", "hold")
# How an order compares with the order that reached the stage in the same round: a glyph in the grid, words in the heads.
CMP_GLYPH = {"more": UP, "match": "=", "less": DOWN}
CMP_WORD = {"more": "above", "match": "same as", "less": "below"}
TOL = 1e-9
# The paper's main setting (rules: Configuration, eco_beer_game_4p); used only for what neither the summary nor a prompt states.
DEFAULT_PARAMS = {"periods": 20, "stages": 4, "scripted_stages": [], "demand": "uniform", "demand_low": 0, "demand_high": 8,
                  "order_delay": 2, "shipping_delay": 2, "production_delay": 3, "initial_inventory": 12, "pipeline_fill": 0,
                  "capacity": 20, "holding_cost": 0.5, "backlog_cost": 1.0, "information_sharing": False, "max_order": 100,
                  "history": 0, "reward_basis": "own", "cost_cap": 400.0}
DEFAULT_NAMES = ["Retailer", "Wholesaler", "Distributor", "Manufacturer"]
SERIES_KEYS = ("incoming", "received", "shipped", "on_hand", "backlog", "order", "cost", "supply_line")


def warn(msg):
    print("warning: " + msg, file=sys.stderr)


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_PUNCT = {"\u3002": ". ", "\u3001": ", ", "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"', "\u3010": "[", "\u3011": "]",
          "\u300a": '"', "\u300b": '"', "\u3008": "<", "\u3009": ">", "\u301c": "~"}


def en(s):
    """English only on the page: full-width forms become ASCII, CJK punctuation its ASCII counterpart, the rendered replay's
    truncation mark '(truncated)', and any other run of Chinese, Japanese or Korean characters a visible marker."""
    if not isinstance(s, str) or not re.search(r"[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af\uff00-\uffef]", s):
        return s
    s = s.replace("\uff08\u540e\u9762\u7701\u7565\uff09", "(truncated)")
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
    """A rendered Markdown replay: one '## Step N . P<seat>' section per decision (the heading is Chinese) with the state the
    seat saw in a fenced block, the move in backticks and, when saved, the reasoning in a ````text block; the header gives the
    rewards, seed and model."""
    text = open(path, encoding="utf-8").read()
    events, md_calls = [], []
    heads = list(re.finditer(r"^## \u7b2c (\d+) \u6b65 \u00b7 P(\d+)\s*$", text, re.M))
    for k, m in enumerate(heads):
        body = text[m.end(): heads[k + 1].start() if k + 1 < len(heads) else len(text)].split("\n## ", 1)[0]
        view = re.search(r"\*\*\u770b\u5230\uff1a\*\*\s*\n```[a-z]*\n(.*?)\n```", body, re.S)
        move = re.search(r"\*\*\u505a\u4e86\uff1a\*\* `(.*)`\s*$", body, re.M)
        think = re.findall(r"````text\n(.*?)\n````", body, re.S)
        if not move:
            continue
        seat = int(m.group(2))
        events.append({"event": "move", "actor": seat, "turn": int(m.group(1)), "move": move.group(1),
                       "view": en(view.group(1)) if view else ""})
        md_calls.append({"seat": seat, "prompt": "", "answer": move.group(1), "reasoning": think[-1] if think else "", "rescued": False})
    over = re.search(r'```\n(\{"event": "game_over".*?\})\n```', text, re.S)
    if over:
        events.append(json.loads(over.group(1)))
    head = text.split("\n## ")[0]
    rw = re.search(r'(\{"0":[^}]*\})', head)
    seed = re.search(r"\u79cd\u5b50 (\d+)", head)
    model = re.search(r"\u6a21\u578b (\S+)", head)
    summary = {"rewards": json.loads(rw.group(1)) if rw else {}}
    run = {"seed": int(seed.group(1)) if seed else None, "model": model.group(1).rstrip(",\uff0c") if model else ""}
    if not any(c["reasoning"] for c in md_calls):
        md_calls = []
    return {"events": events, "summary": summary, "run": run, "config": {}, "transcript": "", "meta": {}, "md_calls": md_calls,
            "from_md": True}


def load_source(src):
    if isinstance(src, dict):
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


# ------------------------------------------------------------------------------------------------ small helpers
def nf(v):
    """A cost or count: whole numbers without decimals, halves with one (costs are multiples of 0.5)."""
    if v is None:
        return "-"
    s = f"{abs(v):,.2f}".rstrip("0").rstrip(".")
    return (MINUS if v < 0 and s != "0" else "") + s


def v2(v):
    return f"{v:.2f}"


def slash(vals):
    return " / ".join(vals)


def rounds_text(rs):
    """'round 3', 'rounds 3 to 6', 'rounds 3, 5 and 9' (runs of three or more collapsed)."""
    rs = sorted(set(rs))
    if not rs:
        return "no round"
    if len(rs) == 1:
        return f"round {rs[0]}"
    runs, a = [], rs[0]
    for i in range(1, len(rs) + 1):
        if i == len(rs) or rs[i] != rs[i - 1] + 1:
            b = rs[i - 1]
            runs.extend([f"{a}"] if a == b else [f"{a}", f"{b}"] if b == a + 1 else [f"{a} to {b}"])
            if i < len(rs):
                a = rs[i]
    return "rounds " + (runs[0] if len(runs) == 1 else ", ".join(runs[:-1]) + " and " + runs[-1])


def parse_qty(move):
    """The quantity of an order action (string or dict); None when unusable."""
    a = move
    if isinstance(a, str):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", a.strip())
        try:
            a = json.loads(s)
        except ValueError:
            i, j = s.find("{"), s.rfind("}")
            try:
                a = json.loads(s[i:j + 1]) if 0 <= i < j else None
            except ValueError:
                a = None
    if not isinstance(a, dict) or a.get("action_id") != "order" or not isinstance(a.get("payload"), dict):
        return None
    q = a["payload"].get("quantity")
    if isinstance(q, bool) or not isinstance(q, int):
        return None
    return q


def round_of(text):
    m = re.search(r"\bRound (\d+) of (\d+)\b", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def pvar(xs):
    """Population variance (the rules' measure)."""
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def cmp3(a, b):
    """'up', 'down' or 'same' (within TOL) going from a to b."""
    return "same" if abs(b - a) <= TOL else "up" if b > a else "down"


def change_text(a, b):
    """How much b is above a: 'grows 3.01-fold' at a ratio of 1.5 or more, 'grows 2.0%' below it."""
    r = b / a
    return f"grows {r:.2f}-fold" if r >= 1.5 else f"grows {(r - 1) * 100:.1f}%"


def pct(x):
    return f"{x * 100:.0f}%"


def cap1(s):
    return s[:1].upper() + s[1:] if s else s


def sentences(parts):
    """Non-empty parts as sentences: each capitalized and ended with one period."""
    return " ".join(cap1(p.strip().rstrip(".")) + "." for p in parts if p and p.strip())


def join_and(xs):
    xs = list(xs)
    return "" if not xs else xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


# ------------------------------------------------------------------------------------------------ what a stage was shown
def view_facts(view):
    """The numbers in one prompt (rules: Round Procedure, step 1)."""
    f = {"round": round_of(view)[0], "periods": round_of(view)[1], "stage": None, "n_stages": None, "name": None, "seat": None}
    m = re.search(r"You are seat P(\d+)", view)
    f["seat"] = int(m.group(1)) if m else None
    m = re.search(r"You are the (.+?) \(stage (\d+) of (\d+)\)", view)
    if m:
        f["name"], f["stage"], f["n_stages"] = m.group(1), int(m.group(2)) - 1, int(m.group(3))
    m = re.search(r"(?:Customer demand this round|Incoming order this round from [^:\n]+): (\d+) unit", view)
    f["incoming"] = int(m.group(1)) if m else None
    m = re.search(r"Arriving this round from [^:\n]+: (\d+) unit", view)
    f["arriving"] = int(m.group(1)) if m else None
    m = re.search(r"Your stock on hand before this round's arrival: (\d+) unit", view)
    f["stock"] = int(m.group(1)) if m else None
    m = re.search(r"Your backlog \(owed to your customer\): (\d+) unit", view)
    f["backlog"] = int(m.group(1)) if m else None
    m = re.search(r"(?:Shipments from [^\n]*? arriving in later rounds|Your production arriving in later rounds): ([^\n]*)", view)
    f["later"] = [(int(r), int(q)) for r, q in re.findall(r"round (\d+): (\d+)", m.group(1))] if m else None
    m = re.search(r"Units you ordered that [^\n]*? has not shipped yet: (\d+)", view)
    f["unshipped"] = int(m.group(1)) if m else None
    m = re.search(r"in the last (\d+) rounds \(oldest first[^)]*\): \[([^\]]*)\]", view)
    f["recent"] = [int(x) for x in re.findall(r"-?\d+", m.group(2))] if m else None
    return f


def read_params(b, moves):
    """Game parameters: the summary's info.params, else the rules text at the top of a prompt, else the paper's main setting."""
    info = (b.get("summary") or {}).get("info") or {}
    P = dict(DEFAULT_PARAMS)
    P["source"] = "defaults"
    if info.get("params"):
        P.update({k: v for k, v in info["params"].items() if v is not None})
        P["source"] = "summary"
    view = next((e.get("view") or "" for e in moves if "You are playing:" in (e.get("view") or "")), "")
    names = None
    m = re.search(r"supply chain with (\d+) stages: (.*?)\. Goods flow", view)
    if m:
        names = [re.sub(r"\s*\(stage \d+\)$", "", x.strip()) for x in m.group(2).split(", ")]
    if view and P["source"] == "defaults":
        rx = {"capacity": r"at most (\d+) units a round", "order_delay": r"An order reaches the supplier after (\d+) round",
              "shipping_delay": r"a shipment arrives (\d+) rounds? after it is sent",
              "production_delay": r"a production order becomes stock after (\d+) round", "periods": r"The game lasts (\d+) rounds"}
        for k, p in rx.items():
            mm = re.search(p, view)
            if mm:
                P[k] = int(mm.group(1))
        mm = re.search(r"pays ([\d.]+) per unit of stock on hand and ([\d.]+) per unit of backlog", view)
        if mm:
            P["holding_cost"], P["backlog_cost"] = float(mm.group(1)), float(mm.group(2))
        mm = re.search(r"drawn uniformly from (\d+) to (\d+)", view)
        if mm:
            P["demand_low"], P["demand_high"] = int(mm.group(1)), int(mm.group(2))
        P["source"] = "prompt"
    if names:
        P["stages"] = len(names)
    P["names"] = names
    return P


# ------------------------------------------------------------------------------------------------ the per-round record
def settle(f, P):
    """One stage's round from what it was shown (rules: Round Procedure, step 3): it ships its backlog and the incoming order as
    far as its stock plus the arrival allows and at most `capacity` units; the rest stays owed."""
    cap = P["capacity"] if P["capacity"] else float("inf")
    have = f["stock"] + f["arriving"]
    owe = f["backlog"] + f["incoming"]
    sh = min(owe, have, cap)
    on_hand, backlog = have - sh, owe - sh
    return {"shipped": sh, "on_hand": on_hand, "backlog": backlog,
            "cost": P["holding_cost"] * on_hand + P["backlog_cost"] * backlog}


def rebuild(facts, orders, P, S, T):
    """Record {key: [stage][round index]} from the prompts; None where no prompt covers a stage-round."""
    R = {k: [[None] * T for _ in range(S)] for k in SERIES_KEYS + ("stock0", "backlog0")}
    for (s, t), f in facts.items():
        if not 1 <= t <= T or not 0 <= s < S:
            continue
        if None in (f["incoming"], f["arriving"], f["stock"], f["backlog"]):
            warn(f"stage {s} round {t}: the prompt lacks a number; the round is not rebuilt")
            continue
        out = settle(f, P)
        k = t - 1
        R["incoming"][s][k], R["received"][s][k] = f["incoming"], f["arriving"]
        R["stock0"][s][k], R["backlog0"][s][k] = f["stock"], f["backlog"]
        for key in ("shipped", "on_hand", "backlog", "cost"):
            R[key][s][k] = out[key]
        q = orders.get((s, t))
        R["order"][s][k] = q
        pipe = sum(x for _, x in (f["later"] or [])) + (f["unshipped"] or 0)
        R["supply_line"][s][k] = pipe + q if q is not None and f["later"] is not None else None
    return R


def from_series(se, S, T):
    """The engine's own record (summary info.series), padded to T rounds."""
    R = {}
    for k in SERIES_KEYS:
        rows = se.get(k) or []
        R[k] = [[(rows[s][t] if s < len(rows) and t < len(rows[s]) else None) for t in range(T)] for s in range(S)]
    R["demand"] = [(se["demand"][t] if t < len(se.get("demand") or []) else None) for t in range(T)]
    return R


def cross_check(facts, orders, R, P, S, T):
    """Ties between prompts: next round's stock and backlog, the incoming order (the customer's order order_delay rounds
    earlier), the arrival (the supplier's shipment shipping_delay rounds earlier, or production production_delay rounds
    earlier) and the recent-shipments list. Returns (agreeing, compared)."""
    ok = n = 0

    def cmp(a, b, what):
        nonlocal ok, n
        if a is None or b is None:
            return
        n += 1
        if a == b:
            ok += 1
        else:
            warn(f"{what}: prompt {a}, record {b}")
    dO, dS, dP = P["order_delay"], P["shipping_delay"], P["production_delay"]
    for (s, t), f in sorted(facts.items()):
        k = t - 1
        if t > 1:
            cmp(f["stock"], R["on_hand"][s][k - 1], f"stage {s} round {t} stock")
            cmp(f["backlog"], R["backlog"][s][k - 1], f"stage {s} round {t} backlog")
        if s > 0:
            want = R["order"][s - 1][t - dO - 1] if t - dO >= 1 else (0 if not P.get("pipeline_fill") else None)
            cmp(f["incoming"], want, f"stage {s} round {t} incoming order")
        if s < S - 1:
            want = R["shipped"][s + 1][t - dS - 1] if t - dS >= 1 else (0 if not P.get("pipeline_fill") else None)
        else:
            want = R["order"][s][t - dP - 1] if t - dP >= 1 else (0 if not P.get("pipeline_fill") else None)
        cmp(f["arriving"], want, f"stage {s} round {t} arrival")
        if f["recent"] is not None:
            L = len(f["recent"])
            want = [R["shipped"][s][j - 1] if j >= 1 else 0 for j in range(t - L, t)]
            if None not in want:
                cmp(f["recent"], want, f"stage {s} round {t} recent shipments")
    return ok, n


# ------------------------------------------------------------------------------------------------ model calls
def load_calls(path):
    """common.load_calls plus n_prior: the user messages a call sent before its last one. A multi-turn recording sends each
    seat its earlier prompts of the game and its own earlier answers as chat turns, so the model saw more than the last
    prompt (which the Observation tab shows). Logs with a single 'prompt' per call, or none at all, give 0."""
    calls = common.load_calls(path)
    if not calls:
        return calls
    raw = [json.loads(line) for line in open(path, encoding="utf-8")]
    if len(raw) != len(calls):
        warn(f"call log {path}: {len(raw)} lines but {len(calls)} calls; earlier turns not counted")
        return calls
    for c, r in zip(calls, raw):
        msgs = r.get("messages") or []
        c["n_prior"] = max(0, sum(1 for m in msgs if m.get("role") == "user") - 1)
    return calls


def attach_calls(steps, calls):
    """Tie calls to steps in time order. For each step the window is the run of calls from the pointer whose seat and round
    agree with it (unknown values agree with anything); the first call in it whose answer is the logged order is the one, and
    earlier calls in the window were failed attempts whose reasoning is kept above it. Without a matching answer the last call
    of the window is used and the step says the game applied another order (the fallback orders 0)."""
    ptr, tied = 0, 0

    def fits(k, st, strict=False):
        seat, rnd = calls[k].get("seat"), round_of(calls[k].get("prompt"))[0]
        if strict:
            return seat == st["seat"] and rnd == st["round"]
        return seat in (None, st["seat"]) and rnd in (None, st["round"])
    for st in steps:
        if ptr < len(calls) and not fits(ptr, st):
            ahead = next((k for k in range(ptr, min(len(calls), ptr + 40)) if fits(k, st, strict=True)), None)
            if ahead is None:
                continue
            warn(f"calls {ptr}..{ahead - 1} match no decision; skipped")
            ptr = ahead
        j = ptr                                        # every stage orders every round, so one decision's calls are
        while j < len(calls) and fits(j, st):          # consecutive and end at the next stage's call
            j += 1
            if j < len(calls) and calls[j].get("seat") is None and round_of(calls[j].get("prompt"))[0] is None:
                break                                  # nothing ties an anonymous call to this step: one call per step
        window = list(range(ptr, j))
        if not window:
            continue
        hit = next((k for k in window if parse_qty(calls[k].get("answer")) == st["qty"]), None)
        use = hit if hit is not None else window[-1]
        parts = [f"[Earlier attempt, answer: {calls[k]['answer'][:300]}]\n{calls[k].get('reasoning') or '(no reasoning returned)'}"
                 for k in range(ptr, use)]
        c = calls[use]
        parts.append(("[Final attempt]\n" if parts else "") + (c.get("reasoning") or ("(no reasoning returned)" if parts else "")))
        if hit is None:
            parts.append(f"[The game applied the order {st['qty']} instead of this answer: {c['answer'][:300]}]")
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        if c.get("prompt"):
            st["state"] = en(common.state_part(c["prompt"]))
            st["from_prompt"] = True
            if c.get("n_prior"):
                k = c["n_prior"]
                st["state"] = ((f"[Also in this call: the seat's {k} earlier prompts of this game and its own orders, as chat turns]"
                                if k > 1 else "[Also in this call: the seat's earlier prompt of this game and its own order, as "
                                              "chat turns]") + "\n" + st["state"])
        st["n_prior"] = c.get("n_prior") or 0
        ptr = use + 1
        tied += 1
    if ptr < len(calls):
        warn(f"{len(calls) - ptr} calls after the last decision were not tied to any step")
    return tied


# ------------------------------------------------------------------------------------------------ build
def build(bundle_dir, calls=None):
    b = load_source(bundle_dir)
    if isinstance(calls, str):
        calls = load_calls(calls)
    if not calls:
        calls = b.get("md_calls") or []
    events, summary, run = b["events"], b.get("summary") or {}, b.get("run") or {}
    info = summary.get("info") or {}
    moves = [e for e in events if e.get("event") == "move"]
    if not moves:
        raise SystemExit("no move events in " + str(bundle_dir))
    P = read_params(b, moves)

    # ---- what each seat saw and ordered
    facts, orders, seat_stage, stage_name, mrounds = {}, {}, {}, {}, []
    last_seat, rnd = None, 0
    for e in moves:
        f = view_facts(e.get("view") or "")
        s_seat = e.get("actor", f["seat"])
        r = f["round"]
        if r is None:                                   # no round in the prompt: a new round starts when the seat order wraps
            r = rnd + 1 if last_seat is None or s_seat <= last_seat else rnd
        rnd, last_seat = r, s_seat
        mrounds.append(r)
        if f["stage"] is not None:
            seat_stage[s_seat] = f["stage"]
            stage_name[f["stage"]] = f["name"]
        if f["periods"]:
            P["periods"] = f["periods"]
        if f["n_stages"]:
            P["stages"] = f["n_stages"]
        st = seat_stage.get(s_seat, s_seat)
        facts[(st, r)] = f
        orders[(st, r)] = parse_qty(e.get("move"))
    S, T = int(P["stages"]), int(P["periods"])
    names = []
    for s in range(S):
        nm = stage_name.get(s) or ((P.get("names") or [])[s] if P.get("names") and s < len(P["names"]) else None)
        names.append(nm or (DEFAULT_NAMES[s] if S == 4 else f"Stage {s + 1}"))
    seats = sorted(seat_stage) if seat_stage else sorted({e.get("actor", 0) for e in moves})
    seat_names = [names[seat_stage.get(k, k)] for k in seats]
    cols = [STAGE_COLORS[s % len(STAGE_COLORS)] for s in range(S)]
    scripted = [s for s in range(S) if s not in seat_stage.values()] if seat_stage else []

    # ---- the per-round record: rebuilt from the prompts, compared with the engine's own record
    Rb = rebuild(facts, orders, P, S, T)
    se = info.get("series") or {}
    Rs = from_series(se, S, T) if se.get("order") else None
    rec_ok = rec_n = 0
    R = {k: [list(row) for row in Rb[k]] for k in SERIES_KEYS}
    if Rs:
        for k in SERIES_KEYS:
            for s in range(S):
                for t in range(T):
                    a, c = Rb[k][s][t], Rs[k][s][t]
                    if a is None:
                        R[k][s][t] = c                # a stage-round no prompt covers (a scripted stage): the engine's value
                        continue
                    if c is None:
                        continue
                    rec_n += 1
                    if abs(a - c) < 1e-9:
                        rec_ok += 1
                    else:
                        warn(f"{k} {names[s]} round {t + 1}: rebuilt {a}, summary {c}")
    view_ok, view_n = cross_check(facts, orders, R, P, S, T)
    demand = [Rb["incoming"][0][t] if Rb["incoming"][0][t] is not None else (Rs["demand"][t] if Rs else None) for t in range(T)]
    if Rs:
        for t in range(T):
            if Rs["demand"][t] is not None and demand[t] is not None and Rs["demand"][t] != demand[t]:
                warn(f"round {t + 1}: Retailer shown demand {demand[t]}, summary {Rs['demand'][t]}")
    played = [t for t in range(T) if all(R["order"][s][t] is not None for s in range(S))]
    nP = len(played)
    if nP < T:
        warn(f"only {nP} of {T} rounds have every stage's order")

    # ---- costs, scores, variances (rules: Reward, Metrics)
    cost = [sum(R["cost"][s][t] for t in played) for s in range(S)]
    hold_c = [sum(P["holding_cost"] * R["on_hand"][s][t] for t in played) for s in range(S)]
    back_c = [sum(P["backlog_cost"] * R["backlog"][s][t] for t in played) for s in range(S)]
    total = sum(cost)
    cap = float(P.get("cost_cap") or 400.0)
    basis = P.get("reward_basis") or "own"
    score_of = lambda c: 1 - 2 * min(c, cap) / cap
    my_score = [score_of(total / S if basis == "chain" else cost[s]) for s in range(S)]
    raw_rw = summary.get("rewards") or (run.get("result") or {}).get("rewards") or {}
    over = next((e for e in events if e.get("event") == "game_over"), {})
    raw_rw = raw_rw or over.get("rewards") or {}
    rewards = {}
    for k in seats:
        s = seat_stage.get(k, k)
        rw = raw_rw.get(str(k))
        rewards[s] = float(rw) if rw is not None else None
        if rw is not None and abs(float(rw) - my_score[s]) > 1e-9:
            warn(f"seat P{k} ({names[s]}): reward {rw}, computed {my_score[s]}")
    score_ok = all(rewards[s] is not None and abs(rewards[s] - my_score[s]) <= 1e-9 for s in rewards)
    ords = [[R["order"][s][t] for t in played] for s in range(S)]
    ovar = [pvar(o) for o in ords]
    dvals = [demand[t] for t in played if demand[t] is not None]
    dvar, dmean = pvar(dvals), (sum(dvals) / len(dvals) if dvals else 0.0)
    links = [cmp3(ovar[s], ovar[s + 1]) for s in range(S - 1)]   # rises / falls / unchanged on each link
    links_up = [x == "up" for x in links]
    up_pct = (ovar[-1] - ovar[0]) / ovar[0] * 100 if ovar[0] > TOL else None
    for key, mine in (("total_cost", total), ("demand_variance", dvar), ("variance_increases", sum(links_up))):
        if info.get(key) is not None and abs(info[key] - mine) > 1e-9:
            warn(f"{key}: summary {info[key]}, computed {mine}")
    for s, v in enumerate(info.get("order_variances") or []):
        if s < S and abs(v - ovar[s]) > 1e-9:
            warn(f"order variance {names[s]}: summary {v}, computed {ovar[s]}")
    for sd in info.get("stages") or []:
        s = sd.get("stage")
        if s is not None and s < S and sd.get("total_cost") is not None and abs(sd["total_cost"] - cost[s]) > 1e-9:
            warn(f"{names[s]}: summary cost {sd['total_cost']}, computed {cost[s]}")

    dO, dS, dP = P["order_delay"], P["shipping_delay"], P["production_delay"]

    # ---- where the order variance comes from: the start-up against the rest of the game. The start-up lasts until every
    # stage has received its first delivery. From empty pipelines a stage's first delivery can come in round dO + dS + 1
    # (the Manufacturer's production: dP + 1), so round max(dO + dS, dP) + 1 is the first by which every stage can have had
    # one: the start-up is at least that many rounds (one more than the lead time) and at most half the game. The split is
    # reported only when the rest of the game ranks the stages differently from the whole game.
    lo_w = max(dO + dS, dP) + 1
    hi_w = max(lo_w, T // 2)
    first_in = [next((t + 1 for t in played if (R["received"][s][t] or 0) > 0), None) for s in range(S)]
    w_raw = max(first_in) if None not in first_in else None
    W = min(max(w_raw if w_raw is not None else hi_w, lo_w), hi_w)
    w_why = ("until every stage had received its first delivery" if w_raw == W else
             f"up to round {lo_w}, the first by which every stage can have had a delivery" if W == lo_w else
             "the first half of the game")
    early = [t for t in played if t < W]
    late = [t for t in played if t >= W]

    def sq_share(xs_all, xs_early):
        m = sum(xs_all) / len(xs_all) if xs_all else 0.0
        tot = sum((x - m) ** 2 for x in xs_all)
        return sum((x - m) ** 2 for x in xs_early) / tot if tot > TOL else None
    share = [sq_share(ords[s], [R["order"][s][t] for t in early]) for s in range(S)]
    dshare = sq_share(dvals, [demand[t] for t in early if demand[t] is not None])
    late_ovar = [pvar([R["order"][s][t] for t in late]) for s in range(S)]
    late_dvar = pvar([demand[t] for t in late if demand[t] is not None])
    late_links = [cmp3(late_ovar[s], late_ovar[s + 1]) for s in range(S - 1)]
    rm_full, rm_late = cmp3(ovar[0], ovar[-1]), cmp3(late_ovar[0], late_ovar[-1])
    split = len(late) >= 2 and len(early) >= 1 and (late_links != links or rm_late != rm_full)
    big_early = []                                     # per stage: early orders more than one standard deviation from its mean
    for s in range(S):
        m = sum(ords[s]) / len(ords[s]) if ords[s] else 0.0
        big_early.append([(R["order"][s][t], t + 1) for t in early if (R["order"][s][t] - m) ** 2 > ovar[s] + TOL])
    quiet_after = [any(R["order"][s][t] for t in early) and not any(R["order"][s][t] for t in late) for s in range(S)]
    late_rng = f"rounds {W + 1} to {played[-1] + 1}" if late else ""
    late_pct = (late_ovar[-1] - late_ovar[0]) / late_ovar[0] * 100 if late_ovar[0] > TOL else None
    late_vals = slash([v2(v) for v in late_ovar])

    # ---- the last round in which a stage's order can still arrive before the game ends (rules: End of Game)
    last_useful = [T - dP if s == S - 1 else T - (dO + dS) for s in range(S)]

    # ---- per round helpers
    def cmp_of(q, inc):
        if q is None or inc is None:
            return None
        return "match" if q == inc else "more" if q > inc else "less"
    in_word = lambda s: "demand" if s == 0 else "incoming"
    verb = lambda s: "produces" if s == S - 1 else "orders"
    order_word = lambda s: "production" if s == S - 1 else "order"
    kinds = {}

    # ---- steps: one per decision, in log order; a decision block takes its stage's color, or gray for an order of 0
    steps, feed = [], []
    last_of_round = {}
    for mi, e in enumerate(moves):
        r = mrounds[mi]
        last_of_round[r] = mi
    for mi, e in enumerate(moves):
        k, r = e.get("actor", 0), mrounds[mi]
        s = seat_stage.get(k, k)
        f = facts.get((s, r)) or {}
        q = parse_qty(e.get("move"))
        inc = f.get("incoming")
        if q:
            kind = f"s{s}"
            kinds[kind] = [f"{names[s]} {verb(s)}", cols[s]]
        else:
            kind = "zero"
            kinds[kind] = ["Nothing ordered", "muted"]
        c = cmp_of(q, inc)
        head = f"Round {r} {DOT} {verb(s)} {q if q is not None else '?'}"
        if c:
            head += f", {CMP_WORD[c]} {in_word(s)} {inc}"
        elif inc is not None:
            head += f" {DOT} {in_word(s)} {inc}"
        if r > last_useful[s]:
            head += f" {DOT} too late to arrive"
        steps.append({"seat": seats.index(k) if k in seats else 0, "x": r - 1, "head": head, "kind": kind,
                      "act": en(e["move"] if isinstance(e.get("move"), str) else json.dumps(e.get("move"))), "state": en(common.state_part(e.get("view") or "")) or "Not recorded.",
                      "round": r, "qty": q, "stage": s, "cmp": c})
    kinds = {k: kinds[k] for k in sorted(kinds, key=lambda k: (k == "zero", k))}

    # ---- feed: the rules, then each round's settlement after its last decision
    cap_u = P["capacity"]
    if P.get("demand") == "step":
        dem_txt = (f"{P['demand_low']} units a round until round {int(P.get('step_period') or 5) - 1}, then "
                   f"{P['demand_high']}")
    else:
        dem_txt = f"a whole number from {P['demand_low']} to {P['demand_high']} each round"
    feed.append({"x": 0, "step": 0, "who": "Rules",
                 "text": f"{T} rounds. Every stage starts with {P['initial_inventory']} units in stock"
                         + (" and nothing on order" if not P.get("pipeline_fill") else f" and {P['pipeline_fill']} units in every delay slot")
                         + f". An order reaches the supplier after {dO} rounds and its shipment arrives {dS} rounds after it is sent; "
                         f"the Manufacturer's production takes {dP} rounds"
                         + (f"; at most {cap_u} units leave a stage in a round" if cap_u else "")
                         + f". Each round a stage pays {nf(P['holding_cost'])} per unit in stock and {nf(P['backlog_cost'])} per unit "
                         f"it owes. Customer demand is {dem_txt}, seen only by the Retailer. "
                         + (f"Every stage also sees all {S} stages' stock and backlog." if P.get("information_sharing") else
                            "No stage sees another stage's stock, backlog or orders.")})
    for t in played:
        r = t + 1
        bits = []
        d, sold = demand[t], R["shipped"][0][t]
        bits.append(f"Customers want {d}; the Retailer sells {sold}" + (f" and owes {R['backlog'][0][t]}" if R["backlog"][0][t] else "") + ".")
        bits.append("Orders: " + slash([str(R["order"][s][t]) for s in range(S)]) + ".")
        arr = [f"{names[s]} {R['received'][s][t]}" for s in range(S) if R["received"][s][t]]
        if arr:
            bits.append("Arrived: " + ", ".join(arr) + ".")
        ships = [f"{names[s]} {ARROW} {names[s - 1]} {R['shipped'][s][t]}" for s in range(S - 1, 0, -1) if R["shipped"][s][t]]
        bits.append(("Shipped: " + ", ".join(ships) + ".") if ships else "Nothing shipped between stages.")
        ends = []
        for s in range(S):
            oh, bl = R["on_hand"][s][t], R["backlog"][s][t]
            ends.append(f"{names[s]} {oh}" + (f" (owes {bl})" if bl else ""))
        bits.append("Stock after the round: " + ", ".join(ends) + ".")
        bits.append("Cost this round: " + slash([nf(R["cost"][s][t]) for s in range(S)]) + f" (chain {nf(sum(R['cost'][s][t] for s in range(S)))}).")
        mi = last_of_round.get(r)
        feed.append({"x": t, "step": mi if mi is not None else len(steps) - 1, "who": "Result", "text": " ".join(bits)})

    decisions = [(st["stage"], st["round"], st["qty"] or 0, st["cmp"]) for st in steps]
    tied = attach_calls(steps, calls)
    has_reason = any(st.get("thinking") for st in steps)
    from_prompt = sum(1 for st in steps if st.get("from_prompt"))
    chat_steps = sum(1 for st in steps if st.get("n_prior"))       # steps whose call also carried earlier chat turns
    chat_max = max((st.get("n_prior") or 0 for st in steps), default=0)
    for st in steps:
        st.setdefault("thinking", "")
        for key in ("round", "qty", "from_prompt", "stage", "cmp", "n_prior"):
            st.pop(key, None)

    # ---- story numbers
    nsteps = len(decisions)
    zero_dec = [d for d in decisions if d[2] == 0]
    late_dec = [d for d in decisions if d[1] > last_useful[d[0]]]
    zero_late = [d for d in zero_dec if d[1] > last_useful[d[0]]]
    pos_dec = [d for d in decisions if d[2] > 0]
    pos_cmp = {c: sum(1 for d in pos_dec if d[3] == c) for c in CMP_GLYPH}
    units = [sum(o) for o in ords]
    dem_total = sum(dvals)
    short_rounds = [[t + 1 for t in played if R["backlog"][s][t]] for s in range(S)]
    worst_net = [min(R["on_hand"][s][t] - R["backlog"][s][t] for t in played) for s in range(S)]
    peak_sl = []                                       # (peak, rounds at the peak) of each stage's supply line
    for s in range(S):
        vals = [(R["supply_line"][s][t], t + 1) for t in played if R["supply_line"][s][t] is not None]
        top = max((v for v, _ in vals), default=None)
        peak_sl.append((top, [r for v, r in vals if v == top]))
    peak_ord = [max(((R["order"][s][t], t + 1) for t in played), key=lambda p: (p[0], -p[1])) for s in range(S)]
    hold_tot, back_tot = sum(hold_c), sum(back_c)
    end_on = [R["on_hand"][s][played[-1]] for s in range(S)]
    end_bl = [R["backlog"][s][played[-1]] for s in range(S)]
    win = min(range(S), key=lambda s: cost[s])
    link_name = lambda s: f"from the {names[s]} to the {names[s + 1]}"
    link_arrow = lambda s: f"{names[s]} {ARROW} {names[s + 1]}"
    R0, RM = names[0], names[-1]

    # ---- the start-up clause (only when the rest of the game ranks the stages differently)
    def late_desc(nums=True):
        if all(v < late_dvar - TOL for v in late_ovar):
            return ("every stage's orders vary less than demand"
                    + (f" ({late_vals} against {v2(late_dvar)})" if nums else ""))
        if rm_late == "down":
            return f"order variance falls from {v2(late_ovar[0])} at the {R0} to {v2(late_ovar[-1])} at the {RM}"
        if rm_late == "same":
            return f"order variance is {v2(late_ovar[0])} at both the {R0} and the {RM}"
        return f"order variance rises from {v2(late_ovar[0])} at the {R0} to {v2(late_ovar[-1])} at the {RM}"
    startup_title = ""
    if split and rm_full == "up" and rm_late != "up":
        startup_title = f", but only in the start-up (rounds 1 to {W}); from round {W + 1} on " + late_desc(False)
    elif split and rm_full != "up" and rm_late == "up":
        startup_title = (f"; after the start-up (rounds 1 to {W}) it rises from {v2(late_ovar[0])} at the {R0} to "
                         f"{v2(late_ovar[-1])} at the {RM}")
    elif split and rm_full == "up" and late_links.count("up") < links.count("up"):
        k_late = late_links.count("up")
        startup_title = (f", but from round {W + 1} on it rises on " + (f"only {k_late}" if k_late else "none")
                         + f" of {S - 1} links")
    driver_bits = []                                   # stages with most of their variance in the start-up, and the orders
    for s in range(S):
        if share[s] is None or share[s] < 0.5 or not big_early[s]:
            continue
        qs, rs = [q for q, _ in big_early[s]], [r for _, r in big_early[s]]
        bit = f"the {names[s]}'s {join_and(str(q) for q in qs)} in {rounds_text(rs)}"
        if quiet_after[s]:
            bit += f" ({'it produced' if s == S - 1 else 'it ordered'} nothing after round {W})"
        driver_bits.append(bit)
    startup_detail = ""
    if split:
        startup_detail = (f"Rounds 1 to {W} ({w_why}) hold " + slash([pct(x) if x is not None else "n/a" for x in share])
                          + f" ({R0} to {RM}" + (f"; demand {pct(dshare)}" if dshare is not None else "")
                          + ") of each stage's squared deviations from its mean order"
                          + (": " + join_and(driver_bits) if driver_bits else "")
                          + f". Over {late_rng} the order variances are {late_vals}, against {v2(late_dvar)} for demand"
                          + ("" if rm_late == "up" else f": no amplification from the {R0} to the {RM}") + ".")

    # ---- title
    falls = [link_name(s) for s in range(S - 1) if links[s] == "down"]
    flats = [link_name(s) for s in range(S - 1) if links[s] == "same"]
    though = join_and(([f"falls {join_and(falls)}"] if falls else []) + ([f"is unchanged {join_and(flats)}"] if flats else []))
    if all(links_up):
        title = (f"The bullwhip appears: order variance rises at every step up the chain, from {v2(ovar[0])} at the {R0} "
                 f"to {v2(ovar[-1])} at the {RM}, against {v2(dvar)} in customer demand" if not startup_title else
                 "The bullwhip appears: order variance rises at every step up the chain")   # tile 1 has the numbers
    elif ovar[0] <= TOL and ovar[-1] > TOL:
        title = (f"Order variance rises from 0 at the {R0} (constant orders) to {v2(ovar[-1])} at the {RM}"
                 + ("" if startup_title else f", rising on {sum(links_up)} of {S - 1} links"))
    elif rm_full == "up":
        title = (f"Order variance {change_text(ovar[0], ovar[-1])} from the {R0} to the {RM}"
                 + ("" if startup_title else f" ({v2(ovar[0])} {ARROW} {v2(ovar[-1])})")
                 + (f", though it {though}" if though and not startup_title else ""))
    elif rm_full == "down":
        title = f"No bullwhip in this game: order variance falls from {v2(ovar[0])} at the {R0} to {v2(ovar[-1])} at the {RM}"
    else:
        title = f"No bullwhip in this game: order variance is {v2(ovar[0])} at both the {R0} and the {RM}"
    title += startup_title

    # ---- lede
    own = "stock, backlog, the order reaching it, what is arriving and what is still on order"
    sees = (f"sees its own stage ({own}) plus every stage's stock and backlog" if P.get("information_sharing") else
            f"sees only its own stage: {own}")
    if chat_max:                                       # the prompt is not all a call saw: say so where the lede says what it sees
        sees += "; in this recording each model call also carried the seat's earlier prompts and orders as chat history"
    score_of_txt = ("the chain's mean stage cost" if basis == "chain" else "its own total cost")
    demand_lede = ("is stable and its distribution known" if P.get("demand") != "step" else "changes only once")
    lede = (f"{'Four' if S == 4 else str(S)} stages of a supply chain, " + ", ".join(names[:-1]) + f" and {names[-1]}, play {T} "
            f"rounds. Each orders from its supplier (the {RM} schedules production) and {sees}. An order comes back as goods "
            f"{dO + dS} rounds later at the earliest ({dO} to reach the supplier, {dS} to ship; production takes {dP}). Each round "
            f"a stage pays {nf(P['holding_cost'])} per unit in stock and {nf(P['backlog_cost'])} per unit it owes, and its score is "
            f"1 {MINUS} {score_of_txt} {DIV} {nf(cap / 2)}. The source paper (Jiang et al., ACL 2026) asks whether "
            f"language-model stages show the bullwhip effect: orders that swing more and more toward the {RM} although "
            f"customer demand ({dem_txt}) {demand_lede}.")

    # ---- tiles
    if up_pct is not None:
        t1_head = (f"Up {up_pct:.1f}%" if up_pct > TOL * 100 else f"Down {abs(up_pct):.1f}%" if up_pct < -TOL * 100 else "Unchanged")
    elif ovar[-1] > TOL:
        t1_head = f"The {R0} orders the same amount every round"
    else:
        t1_head = "No stage's orders vary"
    other_links = "; ".join(x for x in [("falls on " + ", ".join(link_arrow(s) for s in range(S - 1) if links[s] == "down"))
                                        if "down" in links else "",
                                        ("unchanged on " + ", ".join(link_arrow(s) for s in range(S - 1) if links[s] == "same"))
                                        if "same" in links else ""] if x)
    t1_links = f"rises on {sum(links_up)} of {S - 1} links" + (f" ({other_links})" if other_links and sum(links_up) else "")
    t1_split = ""
    if split:
        t1_split = (f"rounds 1 to {W} (the start-up) hold " + slash([pct(x) if x is not None else "n/a" for x in share])
                    + f" of each stage's squared deviations; {late_rng}: {late_vals}, demand {v2(late_dvar)}")
    late_rounds_txt = (rounds_text(range(last_useful[0] + 1, T + 1))
                       + (f" (the {RM}: {rounds_text(range(last_useful[-1] + 1, T + 1)).split(' ', 1)[1]})"
                          if last_useful[-1] != last_useful[0] and last_useful[-1] < T else ""))
    zero_note = (f"{len(zero_late)} of them in {late_rounds_txt}, when no order could arrive before the end; "
                 f"{len(zero_dec) - len(zero_late)} of the {nsteps - len(late_dec)} earlier decisions"
                 if late_dec else "Decisions that ordered nothing")
    tiles = [
        {"k": f"Order variance, {R0} {ARROW} {RM}", "v": f"{v2(ovar[0])} {ARROW} {v2(ovar[-1])}",
         "s": sentences([t1_head + ("; " + t1_links if max(ovar) > TOL else ""), t1_split, f"demand over the game: {v2(dvar)}",
                         f"sparkline: demand, then {R0} to {RM}"]),
         "spark": [round(dvar, 4)] + [round(v, 4) for v in ovar]},
        {"k": "Cost per stage", "v": f"{nf(min(cost))} to {nf(max(cost))}",
         "s": ", ".join(f"{names[s]} {nf(cost[s])}" for s in range(S)) + " (holding plus backlog)"
              + ("; it rises at every step up the chain" if all(cost[s + 1] > cost[s] for s in range(S - 1)) else
                 f"; the {names[win]} pays least") + ". Scores " + slash([f"{my_score[s]:.4f}".rstrip("0").rstrip(".") for s in range(S)])
              + f" (1 {MINUS} cost {DIV} {nf(cap / 2)})."},
        {"k": "Chain cost", "v": nf(total),
         "s": f"Holding {nf(hold_tot)}, backlog {nf(back_tot)}. "
              + ("With information sharing the paper's all-model chains average 406.80 to 585.88 (Table 4, 32 games each)."
                 if P.get("information_sharing") else
                 "Without information sharing the paper's all-model chains average 486.95 to 751.27 (Table 4, 32 games each).")},
        {"k": "Zero orders", "v": f"{len(zero_dec)} of {nsteps}",
         "s": sentences([zero_note,
                         f"the other {len(pos_dec)}: {pos_cmp['more']} above the order reaching the stage, "
                         f"{pos_cmp['match']} equal, {pos_cmp['less']} below",
                         f"units ordered, {R0} to {RM}: {slash([str(u) for u in units])}; demand {dem_total}"])}]

    # ---- charts
    xn = T
    labels = [f"R{t + 1}" for t in range(xn)]
    grid_rows = ["Customers"] + names
    cells = [[None] * xn for _ in grid_rows]
    for t in range(xn):
        if demand[t] is not None:
            cells[0][t] = {"c": "hold", "t": str(demand[t]), "o": 1,
                           "tip": f"demand {demand[t]} {DOT} sold {R['shipped'][0][t]}" if R["shipped"][0][t] is not None else f"demand {demand[t]}"}
    for s in range(S):
        for t in range(xn):
            q = R["order"][s][t]
            if q is None:
                continue
            inc = R["incoming"][s][t]
            s0 = facts.get((s, t + 1), {}).get("stock")
            b0 = facts.get((s, t + 1), {}).get("backlog")
            arr = R["received"][s][t]
            c = cmp_of(q, inc)
            g = CMP_GLYPH.get(c, "")
            tip = f"{verb(s)} {q}{g}" + (f", {CMP_WORD[c]} {in_word(s)} {inc}" if c else "")
            if s0 is not None:
                tip += f" {DOT} stock {s0} + {arr} arriving {DOT} owes {b0}"
            if t + 1 > last_useful[s]:
                tip += f" {DOT} cannot arrive before the end"
            if s in scripted:
                tip += " (scripted)"
            # full opacity keeps the white digits as legible as the palette allows; orders too late to arrive are faded
            cell = {"c": cols[s] if q else "muted", "t": f"{q}{g}" if q else None, "tip": tip,
                    "o": 0.45 if t + 1 > last_useful[s] else 1}
            cells[s + 1][t] = cell
    fade_txt = (f" Faded cells: {late_rounds_txt}, when an order could no longer arrive before the game ends; it costs the "
                "stage nothing either way." if min(last_useful) < T else "")
    c_grid = {"title": "Orders, round by round",
              "sub": f"One column per round. Top row: customer demand. Stage rows: each stage's order to its supplier (the {RM}: "
                     f"units put into production), in the stage's color; the mark shows how it compares with the order that "
                     f"reached the stage that round (the {R0}: demand): {UP} above, = the same, {DOWN} below. Gray cells are orders "
                     f"of 0. Numbers drop out when the chart is narrow: drag across it to zoom in, or hover a column for every value." + fade_txt,   # the tally of marks is tile 4's
              "grid": {"rows": grid_rows, "cells": cells, "rowHeight": 28,
                       "legend": [["Customer demand", "hold"]] + [[f"{names[s]}'s {order_word(s)}", cols[s]] for s in range(S)]
                                 + [["Order of 0", "muted"]]}}
    var_txt = ", ".join(f"{names[s]} {v2(ovar[s])}" for s in range(S))
    peak_txt = ", ".join(f"{names[s]} {peak_ord[s][0]} in round {peak_ord[s][1]}" if peak_ord[s][0] else f"{names[s]} none"
                         for s in range(S))
    c_ord = {"title": "Orders placed each round",
             "sub": f"What each stage ordered from its supplier each round (the {RM}: put into production), against customer "
                    f"demand (dashed). Order variance over the game: demand {v2(dvar)}, {var_txt}. Largest orders: {peak_txt}."
                    + (" " + startup_detail if startup_detail else ""),
             "height": 240, "fmt": "int",
             "lines": [{"name": "Customer demand", "color": "hold", "v": [demand[t] for t in range(xn)], "dash": True, "step": True}]
                      + [{"name": names[s], "color": cols[s], "v": [R["order"][s][t] for t in range(xn)], "step": True}
                         for s in range(S)]}
    net = [[(R["on_hand"][s][t] - R["backlog"][s][t]) if R["on_hand"][s][t] is not None else None for t in range(xn)] for s in range(S)]
    short_txt = "; ".join(f"the {names[s]} in {rounds_text(short_rounds[s])} (down to {nf(worst_net[s])})" for s in range(S) if short_rounds[s])
    c_net = {"title": "Net inventory after each round",
             "sub": f"Stock on hand minus backlog at the end of each round (the paper's I_t); below zero the stage owes its customer. "
                    f"Every stage starts with {P['initial_inventory']}. "
                    + (f"Short: {short_txt}. " if short_txt else "No stage ever ran short. ")
                    + "Final stock: " + ", ".join(f"{names[s]} {end_on[s]}" + (f" (owes {end_bl[s]})" if end_bl[s] else "") for s in range(S)) + ".",
             "height": 250, "fmt": "int", "zero": True,
             "lines": [{"name": names[s], "color": cols[s], "v": net[s], "end": True} for s in range(S)]}
    sl = [[R["supply_line"][s][t] for t in range(xn)] for s in range(S)]
    c_sl = {"title": "Supply line: ordered, not yet received",
            "sub": f"Units each stage has ordered and not yet received at the end of each round (the paper's N_t). An order needs "
                   f"{dO + dS} rounds to arrive ({dP} for production), so what sits here is stock on its way. Peaks: "
                   + ", ".join(f"{names[s]} {peak_sl[s][0]} ({rounds_text(peak_sl[s][1])})" for s in range(S) if peak_sl[s][0])
                   + ". The paper finds that model stages weigh this pipeline less than their stock.",
            "height": 220, "fmt": "int",
            "lines": [{"name": names[s], "color": cols[s], "v": sl[s]} for s in range(S)]}
    cum = []
    for s in range(S):
        acc, row = 0.0, []
        for t in range(xn):
            if R["cost"][s][t] is None:
                row.append(None)
                continue
            acc += R["cost"][s][t]
            row.append(round(acc, 4))
        cum.append(row)
    c_cost = {"title": "Cost so far",
              "sub": f"Each stage's running total of holding ({nf(P['holding_cost'])} per unit in stock) and backlog ({nf(P['backlog_cost'])} "
                     f"per unit owed) cost; the final value sets its score. Holding makes up {nf(hold_tot)} of the chain's {nf(total)}: "
                     + ", ".join(f"{names[s]} {nf(hold_c[s])} of {nf(cost[s])}" for s in range(S)) + ".",
              "height": 220, "fmt": "price",
              "lines": [{"name": names[s], "color": cols[s], "v": cum[s], "end": True} for s in range(S)]}
    charts = [c_grid, c_ord, c_net, c_sl, c_cost]

    # ---- foot
    model = (b.get("meta") or {}).get("model") or run.get("model") or ""
    model = model.split("/")[-1] if model else "the model"
    date = str((b.get("meta") or {}).get("recorded_at") or run.get("recorded_at") or DEFAULT_DATE)[:10]
    seed = run.get("seed", (b.get("config") or {}).get("seed"))
    variant = os.path.splitext(os.path.basename((b.get("config") or {}).get("variant") or ""))[0]
    cfg_bits = ", ".join(x for x in [f"config {variant}" if variant else "", f"seed {seed}" if seed is not None else "",
                                     f"{S} stages, {T} rounds",
                                     "information sharing" if P.get("information_sharing") else "no information sharing"] if x)
    who = (f"all {S} stages played by {model}" if not scripted else
           ", ".join(names[s] for s in range(S) if s not in scripted) + f" played by {model}, the others scripted")
    src_txt = ("Every number on this page is computed from its " + ("rendered replay (prompts and moves)" if b.get("from_md") else "game log")
               + ": the orders are the logged moves, and each round's "
               "arrivals, shipments, stock, backlog and costs are rebuilt from what the stages were shown and the game's shipping "
               "rule (the last round, which no prompt reports, settled with that rule). ")
    chk = []
    if rec_n:
        chk.append(f"they match the engine's per-round record in the summary in all {rec_n} values" if rec_ok == rec_n else
                   f"they match the engine's per-round record in {rec_ok} of {rec_n} values")
    if view_n:
        chk.append(f"all {view_n} ties between prompts (each order reaching the supplier {dO} rounds later, each shipment arriving "
                   f"{dS} rounds later, stock carried to the next round) hold" if view_ok == view_n else
                   f"{view_ok} of {view_n} ties between prompts hold")
    if score_ok and rewards:
        chk.append("the costs reproduce every logged score")
    chk_txt = ("Checks: " + "; ".join(chk) + ". ") if chk else ""
    if has_reason:
        rs_txt = ("Observations and reasoning come from the saved model calls. " if from_prompt == len(steps) else
                  "Reasoning comes from the saved model calls. ")
    else:
        rs_txt = "This recording was made before reasoning was saved; the game will be re-recorded with reasoning. "
    if chat_max:                                       # a multi-turn recording: every call carried the seat's chat so far
        h = P.get("history") or 0
        rs_txt += (f"Each call also carried the seat's earlier prompts and its own earlier orders as chat history (up to "
                   f"{chat_max} rounds); " + (f"the rules (history: 0) and the paper give only the current round's prompt, and "
                                             f"the Observation tab shows only that. " if not h else
                                             f"the rules (history: {h}) put the seat's last {h} rounds in the prompt itself "
                                             f"and the paper gives only the current round's prompt; the Observation tab shows "
                                             f"only the last prompt. "))
    seat_txt = ("Stage k is seat P(k" + MINUS + "1) in the log and the prompts; each round the stages order in chain order, the "
                f"{names[0]} first, " + (f"but an order reaches the supplier only {dO} rounds later, so no stage sees another's "
                                         "order of the same round. " if dO else
                                         "and with no order delay each stage is told its customer's order of the same round. ")
                ) if not scripted else ""
    foot = (f"Data: the {date} recording of {GID} ({cfg_bits}), {who}. " + src_txt + chk_txt + rs_txt + seat_txt
            + "Order variance is the population variance of a stage's orders over the game (the rules' measure; the paper does not "
            f"say which variance it uses); the bullwhip effect means it rises from each stage to the next. "
            + (f"The start-up is rounds 1 to {W}, {w_why} (pipelines start "
               + ("empty" if not P.get("pipeline_fill") else "full") + f"; it is at least {lo_w} rounds, since from empty "
               f"pipelines round {lo_w} is the first by which every stage can have had a delivery, and at most half the "
               f"game); the page splits the variance there only because {late_rng} rank the stages "
               "differently from the whole game. " if split else "")
            + (f"An order placed after round {last_useful[0]} (production after round {last_useful[-1]}) cannot arrive before "
               "the game ends and costs the ordering stage nothing (rules: End of Game); those cells are faded in the grid. "
               if min(last_useful) < T else "")
            + f"Score = 1 {MINUS} 2 "
            f"{TIMES} min(cost, {nf(cap)}) {DIV} {nf(cap)}. Comparisons, from the paper's 32-game means: total chain cost 486.95 to "
            "751.27 for its all-model chains without information sharing and 406.80 to 585.88 with it (Table 4); order variance "
            "'increased by' 74.2% to 82.3% in its four all-model configurations (section 4.1), which the rules read as the "
            f"{names[-1]} against the {names[0]}" + (f" ({up_pct:.1f}% here" + (f"; {MINUS if late_pct < 0 else '+'}{abs(late_pct):.1f}% over {late_rng}" if split and late_pct is not None else "")
                                                    + ")" if up_pct is not None else "") + ". This page is one game. "
            "Source: Jiang et al., Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains "
            "with LLM-Based Simulation (ACL 2026); rules after Croson and Donohue (2006).")

    d = {"id": GID, "name": "Beer game", "title": title, "lede": lede, "seatNames": seat_names, "tiles": tiles, "foot": foot,
         "x": {"n": xn, "labels": labels}, "charts": charts, "kinds": kinds, "steps": steps, "feed": feed,
         "feedTitle": "Round by round",
         "feedSub": "The rules, then each round's settlement after its last order: the orders placed, what arrived, what was "
                    f"shipped down the chain, the stock left and the round's costs (orders and costs listed {slash(names)}). The "
                    "selected step is highlighted; later rounds are hidden."}
    d["eyebrow"] = (common.EYEBROW.format(gid=GID, date=date) if has_reason
                    else f"social_game_env {DOT} {GID} {DOT} {model} {DOT} recorded {date} {DOT} reasoning not saved")
    build.last_checks = {"record": [rec_ok, rec_n], "views": [view_ok, view_n], "scores": score_ok, "calls_tied": tied,
                         "steps_with_reasoning": sum(1 for st in steps if st["thinking"])}
    return d


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    d = build(src, calls)
    print("checks:", json.dumps(build.last_checks), file=sys.stderr)
    common.write(d, out, date=recorded_date(src))
