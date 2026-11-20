"""Replay data for eco_sealed_auction (repeated sealed-bid auction with independent private values; Vickrey 1961, Milgrom and
Weber 1982; experiments Kagel, Harstad and Levin 1987, Goeree, Holt and Palfrey 2002).

usage: python3 -I eco_sealed_auction.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md, run.log) or a
          file inside one.
  OUT_DIR where <OUT_DIR>/eco_sealed_auction.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). Calls are tied to
          decisions in time order: a call belongs to the next decision of the same seat, phase and round (read from its prompt:
          "You are seat Pk", "Current phase: X", the public state's "round"; a value the prompt does not show agrees with
          anything). Within one decision's calls the first whose answer is the logged action (the bid, or the message text) is
          the decision's call; earlier ones were failed attempts (format retries) and their reasoning is kept above it. Its
          reasoning becomes the step's thinking and its prompt the observation.

Each round every bidder gets a new private value (DRAW), may send talk_passes public messages (TALK, seats in order, logged as
talk events) and then submits one sealed bid (BID, seats asked in order P0, P1, ...; no bidder sees another's bid before the
reveal). The log has no event per bid: all bids are in the round's round_result event (bids in seat order, winner, price). The
private values are not in the log. They are read from the prompts that show them: every call's prompt ("Your private info"), the
prompt saved in run.json, or config fixed_values; a winner's value is also recovered exactly when its score is shown before and
after the round (prompts' public scores, the summary's final rewards). Payoffs (winner: value - price, others 0), equilibrium
benchmarks and all checks are computed exactly (fractions); anything the log cannot determine is shown as unknown, never
guessed. Checks (printed as warnings): winner and price against the pricing rule, bids in range, pending bids and public scores
shown in later prompts, the summary's rewards and winners, the game's own overbid and efficient-round counts. Nothing about one
particular recording (seed, number of bidders or rounds, outcome) is assumed.
When run.log reports that the agent could not parse an answer (or a call failed), the agent played its one listed legal action,
the bid's declared default 0, and the warning is printed once per run: every logged 0 without a matching model answer is then
shown as possibly that fallback (own grid color, step head, title and tiles hedged). Answers forced after the reasoning ran out
(the call log's follow-up answer) are named in the step, the grid cell, the round's result and, when they decide a round the
highest value lost, the "Highest value won" tile. For the rules' Goeree, Holt and Palfrey (2002) configs (two bidders, first
price, their two value pools) the equilibrium bid is the paper's risk-neutral Nash bid table, not (n-1)/n of the value."""
import json, os, re, sys, unicodedata
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_sealed_auction"
NAME = "Sealed-bid auction"
DEFAULT_DATE = "2026-11-09"
MINUS, DOT, TIMES, LDQ, RDQ, NDASH = "−", "·", "×", "“", "”", "–"
TOL = 1                                             # a bid within 1 of the equilibrium bid counts as "at" it (integer bids)
# Band colors (grid cells and step pills, white text on the fill): the most common bands get the fills that keep white text
# legible (blue 4.4:1, orange 3.2:1, green 2.8:1 in light mode; yellow is 2.2:1). Green is also the equilibrium line in the
# share chart. "over" is yellow at first price, where it is rare; at second price "above" cannot occur (the equilibrium bid is
# the value), so there "over" takes orange (see band_colors).
KINDS = {"below": ["Below the equilibrium bid", "c1"],
         "near": ["At the equilibrium bid (within 1)", "c3"],
         "above": ["Above equilibrium, not above value", "c2"],
         "over": ["Above own value", "c4"],
         "unknown": ["Value not logged", "muted"],
         "fallback": ["Bid 0, may be the recorder's fallback", "muted"],
         "talk": ["Public message", "hold"]}
FALLBACK_BID = 0           # the bid action's declared default: the one legal action the agent lists, so its fallback (yaml)
GHP_RN = {(0, 2, 4, 6, 8, 11): [0, 1, 2, 3, 4, 5],    # rules, Goeree, Holt and Palfrey (2002) Tables 1-2: risk-neutral Nash bids
          (0, 3, 5, 7, 9, 12): [0, 1, 2, 3, 4, 5]}    # for the low- and high-values treatments (2 bidders, first price)
HUMAN = {"first": "human bidders in first-price lab auctions bid above the risk-neutral equilibrium (Kagel, Harstad and Levin "
                  "1987; Goeree, Holt and Palfrey 2002)",
         "second": "human bidders in second-price lab auctions bid above their values (Kagel, Harstad and Levin 1987)"}
BAND_TIP = {"below": "below eq", "near": "at eq", "above": "above eq", "over": "above value"}   # grid cell tips (short:
#                                                                 the legend explains the bands; index.html draws tips unwrapped)
# Per-bidder line colors, in this order, skipping every color the page's grid uses for a band and the benchmarks' colors (c3
# equilibrium line, hold: winner and price rows, own-value line, seller revenue), so no color means a bidder in one chart and a
# band in another; only when too few remain do the skipped ones follow (see bidder_colors).
BIDDER_ORDER = ["c1", "c4", "ink", "c2", "muted", "c3", "hold"]
BENCH_COLORS = {"c3", "hold"}
OWN_VALUE_COLOR = "hold"
COLOR_WORD = {"c1": "blue", "c2": "orange", "c3": "green", "c4": "yellow", "hold": "gray", "muted": "light gray"}
NUMW = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_ANY = re.compile("[　-ヿ㐀-鿿가-힯＀-￯]")
_CJK_RUN = re.compile("[぀-ヿ㐀-鿿가-힯]+(?:[^\\n぀-ヿ㐀-鿿가-힯]{0,3}"
                      "[぀-ヿ㐀-鿿가-힯]+)*")
_PUNCT = {"。": ". ", "、": ", ", "「": '"', "」": '"', "『": '"', "』": '"', "【": "[",
          "】": "]", "《": '"', "》": '"', "〈": "<", "〉": ">", "〜": "~"}


def en(s):
    """English only on the page: full-width forms become ASCII, CJK punctuation its ASCII counterpart, and any remaining stretch
    of Chinese, Japanese or Korean text (a model may reason in Chinese) one visible marker."""
    if not isinstance(s, str) or not _CJK_ANY.search(s):
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
    s = re.sub("[　-〿＀-￯]", "", s)
    return re.sub(r"(?<=\S) {2,}(?=\S)", " ", s)


def en_all(o):
    if isinstance(o, str):
        return en(o)
    if isinstance(o, list):
        return [en_all(v) for v in o]
    if isinstance(o, dict):
        return {k: en_all(v) for k, v in o.items()}
    return o


# ------------------------------------------------------------------------------------------------ loading
def load_source(src):
    if isinstance(src, dict):                       # an already loaded bundle (tests)
        return src
    if not os.path.isdir(src):
        src = os.path.dirname(os.path.abspath(src))
    if not os.path.exists(os.path.join(src, "game_log.jsonl")):
        raise SystemExit(f"no game_log.jsonl in {src}")
    b = common.load_bundle(src)
    meta = os.path.join(src, "meta.json")
    b["meta"] = json.load(open(meta, encoding="utf-8")) if os.path.exists(meta) else {}
    rl = os.path.join(src, "run.log")
    b["runlog"] = open(rl, encoding="utf-8", errors="replace").read() if os.path.exists(rl) else ""
    return b


def model_and_date(b):
    """(full model id, short model name, recording date YYYY-MM-DD), from run.json (meta.json as a fallback)."""
    run, meta = b.get("run") or {}, b.get("meta") or {}
    full = str(run.get("model") or meta.get("model") or "")
    date = str(run.get("recorded_at") or meta.get("recorded_at") or DEFAULT_DATE)[:10]
    return full, (full.split("/")[-1] if full else "an unnamed model"), date


def recorded_date(src):
    return model_and_date(load_source(src))[2]


# ------------------------------------------------------------------------------------------------ text helpers
def fn(v, signed=False):
    """A value, bid, payoff or score: an integer with thousands separators, anything else with up to two decimals."""
    v = Fraction(v)
    s = f"{abs(int(v)):,}" if v.denominator == 1 else f"{abs(float(v)):,.2f}".rstrip("0").rstrip(".")
    return (MINUS if v < 0 else ("+" if signed and v > 0 else "")) + s


def num(v):
    """A chart value: an int when whole, else a float (the exact Fraction is kept for the text)."""
    v = Fraction(v)
    return int(v) if v.denominator == 1 else round(float(v), 4)


def pct(x, d=1):
    """A share (Fraction, 1 = 100%) as a percentage."""
    x = Fraction(x) * 100
    return f"{int(x)}%" if x.denominator == 1 else f"{float(x):.{d}f}%"


def frac(v):
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def numword(n):
    return NUMW[n] if 0 <= n < len(NUMW) else str(n)


def cap(s):
    return s[:1].upper() + s[1:]


def plural(n, one, many=None, word=False):
    return f"{numword(n) if word else n} {one if n == 1 else (many or one + 's')}"


def join_and(xs):
    xs = [str(x) for x in xs]
    return "" if not xs else xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def rounds_text(rs):
    rs = sorted(rs)
    if not rs:
        return "no round"
    if len(rs) == 1:
        return f"round {rs[0]}"
    if len(rs) > 2 and rs == list(range(rs[0], rs[-1] + 1)):
        return f"rounds {rs[0]} to {rs[-1]}"
    return "rounds " + join_and(rs)


def bname(s):
    return f"Bidder {s + 1}"


def pool_text(pool):
    """[10, 20, ..., 100] -> '10, 20, ..., 100'; anything else listed in full."""
    if not pool:
        return "an unlogged pool"
    vs = list(pool)
    if len(vs) > 4 and len(set(vs)) == len(vs):
        d = vs[1] - vs[0]
        if d and all(vs[k + 1] - vs[k] == d for k in range(len(vs) - 1)):
            return f"{fn(vs[0])}, {fn(vs[1])}, ..., {fn(vs[-1])}"
    return ", ".join(fn(v) for v in vs)


def _ws(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _num(v):
    """A logged number as a Fraction (None for anything else; bools are not numbers)."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, float):
        return Fraction(str(v))
    if isinstance(v, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", v):
        return Fraction(v.strip())
    return None


# ------------------------------------------------------------------------------------------------ prompts
def _json_after(text, label):
    i = text.find(label)
    if i < 0:
        return None
    s = text[i + len(label):].lstrip()
    try:
        return json.JSONDecoder().raw_decode(s)[0]
    except ValueError:
        return None


def prompt_info(p):
    """What a prompt shows: seat, phase, round, the private value and pending bid, and the public scores at that moment."""
    p = p or ""
    m = re.search(r"Current phase:\s*([A-Za-z_]+)", p)
    pub = _json_after(p, "Public state:")
    priv = _json_after(p, "Your private info:")
    ps = _json_after(p, "Public player state:")
    rnd = pub.get("round") if isinstance(pub, dict) else None
    if rnd is None:
        rm = re.search(r'"round":\s*(\d+)', p[p.find("Public state:"):]) if "Public state:" in p else None
        rnd = int(rm.group(1)) if rm else None
    scores = {}
    if isinstance(ps, dict):
        for k, v in ps.items():
            if str(k).isdigit() and isinstance(v, dict) and _num(v.get("score")) is not None:
                scores[int(k)] = _num(v.get("score"))
    priv = priv if isinstance(priv, dict) else {}
    return {"seat": common.seat_of(p), "phase": m.group(1).upper() if m else None,
            "round": int(rnd) if isinstance(rnd, (int, float)) and not isinstance(rnd, bool) else None,
            "value": _num(priv.get("value")), "pending": _num(priv.get("pending_bid")), "scores": scores}


def _norm_answer(s):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (s or "").strip())
    try:
        return json.loads(s)
    except ValueError:
        i, j = s.find("{"), s.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(s[i:j + 1])
            except ValueError:
                return None
        return None


def _bid_of(a):
    """The bid of a parsed answer, only when it is a legal shape (action bid, payload {"bid": {"amount": int}})."""
    if not isinstance(a, dict) or a.get("action_id") != "bid":
        return None
    b = (a.get("payload") or {}).get("bid") if isinstance(a.get("payload"), dict) else None
    if not isinstance(b, dict) or set(b) != {"amount"}:
        return None
    v = b["amount"]
    return v if isinstance(v, int) and not isinstance(v, bool) else None


# ------------------------------------------------------------------------------------------------ the log
def read_game(events, summary, cfg):
    rules = next((e for e in events if e.get("event") == "rules"), {})
    over = next((e for e in events if e.get("event") == "game_over"), {})
    end = next((e for e in events if e.get("event") == "game_end"), {})
    var = dict(over.get("variables") or {})
    var.update({k: v for k, v in (summary.get("variables") or {}).items() if k not in var})
    cv = cfg.get("variables") or {}

    def par(k, default=None):
        for src in (rules, var, cv):
            if src.get(k) is not None:
                return src[k]
        return default
    rounds, talk, cur, stray = [], [[]], None, []
    known = {"setup", "rules", "new_round", "talk", "round_result", "game_end", "game_over"}
    for e in events:
        ev = e.get("event")
        if ev == "new_round":
            cur = e.get("round")
        elif ev == "round_result":
            bids = [int(x) for x in (e.get("bids") or [])]
            rounds.append({"r": int(e.get("round", cur or len(rounds) + 1)), "bids": bids, "winner": e.get("winner"),
                           "price": _num(e.get("price")), "talk": talk[-1]})
            talk.append([])
        elif ev == "talk":
            seat = e.get("actor", e.get("seat"))
            if seat is None:
                stray.append(e)
                continue
            talk[-1].append({"seat": int(seat), "text": str(e.get("text", ""))})
        elif ev not in known and e.get("_visibility") != "hidden":
            print(f"warning: public event {ev!r} is not shown on the page", file=sys.stderr)
    if not rounds:
        raise SystemExit("no round_result event in the game log")
    if stray:
        print(f"warning: {len(stray)} message(s) without a sender are not shown", file=sys.stderr)
    n = int(par("n_players", cfg.get("n_players") or len(rounds[0]["bids"])))
    pool = par("value_pool")
    pool = [_num(v) for v in pool] if isinstance(pool, list) and pool and all(_num(v) is not None for v in pool) else None
    fixed = cv.get("fixed_values") or var.get("fixed_values") or []
    return {"n": n, "R": int(par("rounds", len(rounds))), "tp": int(par("talk_passes", 0) or 0), "rounds": rounds,
            "pricing": str(par("pricing", "second")), "bid_max": int(par("bid_max", 100)), "pool": pool,
            "fixed": [_num(v) for v in fixed] if isinstance(fixed, list) else [],
            "tail_talk": talk[-1], "end": end, "winners": over.get("winners", summary.get("winners"))}


def check_rounds(G):
    """The pricing rule against the logged bids, winner and price."""
    n, warn = G["n"], []
    for k, rd in enumerate(G["rounds"]):
        r, b, w, p = rd["r"], rd["bids"], rd["winner"], rd["price"]
        if r != k + 1:
            warn.append(f"round_result {k}: round {r} logged, {k + 1} expected")
        if len(b) != n:
            warn.append(f"round {r}: {len(b)} bids logged for {n} bidders")
            continue
        if any(not 0 <= x <= G["bid_max"] for x in b):
            warn.append(f"round {r}: a bid outside 0 to {G['bid_max']}: {b}")
        top = max(b)
        tied = [s for s in range(n) if b[s] == top]
        lower = [x for x in b if x < top]
        second = top if len(tied) > 1 else (max(lower) if lower else top)
        want = top if G["pricing"] == "first" else second
        if w not in tied:
            warn.append(f"round {r}: winner P{w} did not place the highest bid {top} ({b})")
        if p != want:
            warn.append(f"round {r}: price {p} logged, the {G['pricing']}-price rule gives {want} for bids {b}")
        if G["tp"]:
            want_s = [s for _ in range(G["tp"]) for s in range(n)]
            got = [t["seat"] for t in rd["talk"]]
            if got != want_s:
                warn.append(f"round {r}: message senders {got}, {want_s} expected ({G['tp']} pass(es) in seat order)")
    if G["tail_talk"]:
        warn.append(f"{len(G['tail_talk'])} message(s) after the last round_result (a round without a result)")
    if len(G["rounds"]) < G["R"]:
        warn.append(f"only {len(G['rounds'])} of {G['R']} rounds logged: the recording stops early")
    return warn


# ------------------------------------------------------------------------------------------------ benchmarks
def ghp_table(pool, n, pricing):
    """The rules' Goeree, Holt and Palfrey (2002) configs (two bidders, first price, values equally likely from one of the two
    treatments' pools): the paper's risk-neutral Nash bids, {value: bid} (Tables 1-2: $0, 1, 2, 3, 4, 5 in both treatments).
    These discrete-value bids replace (n-1)/n of the value, which assumes continuous uniform values. None for any other config."""
    if pricing != "first" or n != 2 or not pool or len(set(pool)) != len(pool):
        return None
    key = tuple(sorted(pool))
    return {Fraction(v): Fraction(b) for v, b in zip(key, GHP_RN[key])} if key in GHP_RN else None


def eq_bid(v, n, pricing, tab=None):
    """The rules' benchmark bid: (n-1)/n of the value at first price (risk-neutral, uniform values), the value at second price
    (weakly dominant); the paper's table for the GHP configs (see ghp_table)."""
    if tab and Fraction(v) in tab:
        return tab[Fraction(v)]
    return Fraction(n - 1, n) * v if pricing == "first" else Fraction(v)


def band(bid, v, n, pricing, tab=None):
    if bid > v:
        return "over"
    e = eq_bid(v, n, pricing, tab)
    if abs(bid - e) <= TOL:
        return "near"
    return "below" if bid < e else "above"


def band_colors(pricing):
    """KINDS for this pricing rule: at second price "above" cannot occur, so "over" takes its orange."""
    k = {key: list(v) for key, v in KINDS.items()}
    if pricing != "first":
        k["over"][1] = KINDS["above"][1]
    return k


def bidder_colors(n, grid_colors):
    """n line colors for the bidders: BIDDER_ORDER without the colors the grid uses on this page and the benchmark colors,
    then (only if too few remain) the skipped ones."""
    taken = set(grid_colors) | BENCH_COLORS
    order = [c for c in BIDDER_ORDER if c not in taken] + [c for c in BIDDER_ORDER if c in taken]
    return [order[s % len(order)] for s in range(n)]


def eq_round(vals, n, pricing, tab=None):
    """Equilibrium play with these values: every bidder bids eq_bid; the highest value wins (ties at random). Returns
    (expected payoff per seat, seller revenue)."""
    top = max(vals)
    tied = [s for s in range(n) if vals[s] == top]
    if pricing == "first":
        price = eq_bid(top, n, pricing, tab)
    else:
        lower = [v for v in vals if v < top]
        price = top if len(tied) > 1 else (max(lower) if lower else top)
    pay = [(top - price) / len(tied) if s in tied else Fraction(0) for s in range(n)]
    return pay, price


def expected_eq(pool, n, pricing, tab=None):
    """Expected highest value, price and bidders' total payoff per round under equilibrium play, values drawn uniformly from the
    pool (each list entry equally likely), exactly."""
    if not pool:
        return None
    vs = sorted(set(pool))
    F = lambda x: Fraction(sum(1 for v in pool if v <= x), len(pool))
    e_max = e_sec = e_fp = Fraction(0)
    prev_max = prev_sec = Fraction(0)
    for x in vs:
        Fx = F(x)
        cm = Fx ** n                                            # P(highest <= x)
        cs = Fx ** n + n * Fx ** (n - 1) * (1 - Fx)             # P(second highest <= x): at most one value above x
        e_max += x * (cm - prev_max)
        e_sec += x * (cs - prev_sec)
        e_fp += eq_bid(x, n, "first", tab) * (cm - prev_max)    # first price: the highest value's equilibrium bid
        prev_max, prev_sec = cm, cs
    price = e_fp if pricing == "first" else e_sec
    return {"max": e_max, "price": price, "bidders": e_max - price}


# ------------------------------------------------------------------------------------------------ model calls
def call_meta(c):
    info = prompt_info(c.get("prompt"))
    seat = c.get("seat") if c.get("seat") is not None else info["seat"]
    phase = info["phase"]
    if phase is None:                                   # a retry note without the state: placed by its answer
        a = _norm_answer(c.get("answer"))
        phase = {"talk": "TALK", "bid": "BID"}.get(a.get("action_id") if isinstance(a, dict) else None)
    return seat, phase, info["round"], info


def _matches(call, st, loose=False):
    a = _norm_answer(call.get("answer"))
    if st["phase"] == "TALK":
        if not isinstance(a, dict):
            raw = (call.get("answer") or "").strip()
            return bool(raw) and _ws(raw) == _ws(st["text"])
        if a.get("action_id") != "talk":
            return False
        return loose or _ws((a.get("payload") or {}).get("text")) == _ws(st["text"])
    if loose:
        return isinstance(a, dict) and a.get("action_id") == "bid"
    return _bid_of(a) == st["bid"]


def attach_calls(steps, calls, meta):
    """Tie calls to steps in time order (see the module docstring). Returns (steps with a call, steps whose call's answer is not
    the logged action)."""
    def fits(k, st, strict=False):
        seat, ph, rnd = meta[k][:3]
        if strict:
            return seat == st["seat"] and ph == st["phase"] and rnd == st["round"]
        return seat in (None, st["seat"]) and ph in (None, st["phase"]) and rnd in (None, st["round"])
    ptr = tied = mism = 0
    for st in steps:
        if ptr < len(calls) and not fits(ptr, st):     # resynchronise on a call that names this very decision
            ahead = next((k for k in range(ptr, min(len(calls), ptr + 80)) if fits(k, st, strict=True)), None)
            if ahead is None:
                continue
            print(f"warning: calls {ptr}..{ahead - 1} match no decision; skipped", file=sys.stderr)
            ptr = ahead
        j = ptr
        while j < len(calls) and fits(j, st):
            j += 1
            if _matches(calls[j - 1], st):             # a decision's calls end at its accepted answer
                break
        window = list(range(ptr, j))
        if not window:
            continue
        hit = next((k for k in window if _matches(calls[k], st)), None)
        loose = hit if hit is not None else next((k for k in window if _matches(calls[k], st, loose=True)), None)
        use = hit if hit is not None else (loose if loose is not None else window[-1])
        parts = []
        for k in range(ptr, use):
            c = calls[k]
            parts.append(f"[Earlier attempt, answer: {c['answer'][:300]}]\n{c.get('reasoning') or '(no reasoning returned)'}")
        c = calls[use]
        parts.append(("[Final attempt]\n" if parts else "") + (c.get("reasoning") or ("(no reasoning returned)" if parts else "")))
        if hit is None:
            if st["phase"] == "TALK":
                parts.append(f"[The log shows the message {LDQ}{st['text'][:200]}{RDQ}, not this answer: {c['answer'][:300]}]")
            else:
                parts.append(f"[The log shows a bid of {st['bid']}, not this answer: {c['answer'][:300]}. When no reply is a "
                             "legal action the game applies its fallback, a bid of 0.]")
            mism += 1
            st["mismatch"] = True
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
            st["rescued"] = True
        st["hit"] = hit is not None
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        obs = next((k for k in range(use, ptr - 1, -1) if "You are seat" in (calls[k].get("prompt") or "")), None)
        if obs is not None:
            st["state"] = en(common.state_part(calls[obs]["prompt"]))
            st["state_from_prompt"] = True
        a = _norm_answer(c.get("answer"))
        if hit is not None and isinstance(a, dict):
            st["act"] = en(json.dumps(a, ensure_ascii=False))
        ptr = use + 1
        tied += 1
    if ptr < len(calls):
        print(f"warning: {len(calls) - ptr} calls after the last decision were not tied to any step", file=sys.stderr)
    return tied, mism


# ------------------------------------------------------------------------------------------------ end labels
def _nice_ticks(lo, hi, k=4):
    """index.html niceTicks."""
    import math
    span = (hi - lo) or abs(hi) or 1
    p = 10 ** math.floor(math.log10(span / k))
    step = next((m * p for m in (1, 2, 2.5, 5, 10) if span / (m * p) <= k), 10 * p)
    out, v = [], math.ceil(lo / step) * step
    while v <= hi + step * 1e-6:
        out.append(round(v, 8))
        v += step
    return out


def fit_end_labels(lines, n, height, fmt_val, widths=(760, 340)):
    """Keep a line's "end" label (its name and last value, drawn right-aligned 7 px above its last point) only where the page
    can draw it clear of every line and dot, at a desktop and a phone width. Mirrors index.html for a line chart with
    "bars": [] (banded x axis, y axis from 0 when every value is >= 0; margins 86 / 16 px, plot top 14 px, bottom 26 px; 11 px
    monospace, 6.6 px a character; labels stacked at least 13 px apart). Colliding labels are dropped one at a time, dashed
    benchmark lines first. Returns the names whose label was dropped."""
    want = [l for l in lines if l.get("end")]
    for l in lines:
        l.pop("end", None)

    def clashes(on, W):
        iw, ih = W - 102, height - 40
        cw = iw / n
        X = lambda i: 86 + (i + 0.5) * cw
        vals = [v for l in lines for v in l["v"] if v is not None] + [0]
        lo, hi = min(vals), max(vals)
        pad = (hi - lo or abs(hi) or 1) * 0.1
        lo, hi = lo - pad, hi + pad
        if min(vals) >= 0:
            lo = 0
        t = _nice_ticks(lo, hi)
        y0, y1 = min(lo, t[0]), max(hi, t[-1])
        Y = lambda v: 14 + ih - (v - y0) / ((y1 - y0) or 1) * ih
        ends = []
        for l in on:
            i = max(k for k, v in enumerate(l["v"]) if v is not None)
            ends.append([Y(l["v"][i]) - 7, l, i])
        ends.sort(key=lambda e: e[0])
        for k in range(1, len(ends)):
            if ends[k][0] - ends[k - 1][0] < 13:
                ends[k][0] = ends[k - 1][0] + 13
        bad = []
        for yy, l, i in ends:
            text = f"{l['name']} {fmt_val(l['v'][i])}"
            bx0, bx1, by0, by1 = W - 16 - 6.6 * len(text) - 2, W - 16 + 1, yy - 10, yy + 2
            inside = lambda x, y, r=0: bx0 - r <= x <= bx1 + r and by0 - r <= y <= by1 + r
            hit = by0 < 0
            for m in lines:
                pts = [(X(k), Y(v)) for k, v in enumerate(m["v"]) if v is not None]
                for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
                    hit = hit or any(inside(xa + (xb - xa) * j / 40, ya + (yb - ya) * j / 40) for j in range(41))
                if m.get("dots") or len(pts) == 1:
                    hit = hit or any(inside(x, y, 4) for x, y in pts if not (m is l and (x, y) == pts[-1]))
            if hit:
                bad.append(l)
        return bad
    on = list(want)
    while on:
        bad = {id(l) for W in widths for l in clashes(on, W)}
        if not bad:
            break
        victim = next((l for l in on if id(l) in bad and l.get("dash")), None) or next(l for l in reversed(on) if id(l) in bad)
        on = [l for l in on if l is not victim]
    for l in on:
        l["end"] = True
    return [l["name"] for l in want if all(l is not o for o in on)]


# ------------------------------------------------------------------------------------------------ build
def build(src, calls=None):
    b = load_source(src)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    calls = calls or []
    events, summary, run, cfg = b["events"], b.get("summary") or {}, b.get("run") or {}, b.get("config") or {}
    G = read_game(events, summary, cfg)
    n, Rcfg, TP, pricing, BMAX, pool = G["n"], G["R"], G["tp"], G["pricing"], G["bid_max"], G["pool"]
    warns = check_rounds(G)
    raw_rw = summary.get("rewards") or (run.get("result") or {}).get("rewards") or {}
    rewards = [_num(raw_rw.get(str(s))) for s in range(n)]
    full_model, model, date = model_and_date(b)
    seed = run.get("seed", cfg.get("seed"))
    RD = G["rounds"]
    R = len(RD)
    N = max(Rcfg, RD[-1]["r"], R)                       # x axis: every configured round, played or not
    pad = lambda vals: list(vals) + [None] * (N - len(vals))
    B = [bname(s) for s in range(n)]
    first = pricing == "first"
    KIND = band_colors(pricing)
    eqr = Fraction(n - 1, n) if first else Fraction(1)
    tab = ghp_table(pool, n, pricing)                   # the GHP configs: the paper's discrete risk-neutral bids
    EQ = lambda v: eq_bid(v, n, pricing, tab)
    tab_text = ", ".join(f"{fn(b_)} for {fn(v_)}" for v_, b_ in sorted(tab.items())) if tab else ""
    bids = [[RD[i]["bids"][s] for i in range(R)] for s in range(n)]
    price = [RD[i]["price"] for i in range(R)]
    win = [RD[i]["winner"] for i in range(R)]
    rindex = {RD[i]["r"]: i for i in range(R)}

    # ---- steps: per round the messages (logged order), then the bids in seat order
    steps = []
    for i, rd in enumerate(RD):
        for t in rd["talk"]:
            steps.append({"seat": t["seat"], "x": i, "phase": "TALK", "round": rd["r"], "text": t["text"], "kind": "talk",
                          "act": json.dumps({"action_id": "talk", "payload": {"text": t["text"]}}, ensure_ascii=False)})
        for s in range(n):
            steps.append({"seat": s, "x": i, "phase": "BID", "round": rd["r"], "bid": rd["bids"][s],
                          "act": json.dumps({"action_id": "bid", "payload": {"bid": {"amount": rd["bids"][s]}}})})
    meta = [call_meta(c) for c in calls]
    tied, mism = attach_calls(steps, calls, meta) if calls else (0, 0)

    # ---- private values: prompts (calls, run.json), config fixed_values, then scores before and after a win
    val = [[None] * R for _ in range(n)]
    vsrc = [[None] * R for _ in range(n)]

    def put(s, i, v, how):
        if v is None or not 0 <= s < n or i is None:
            return
        if val[s][i] is not None and val[s][i] != v:
            warns.append(f"round {RD[i]['r']}: {bname(s)}'s value is {val[s][i]} in one source and {v} in another ({how})")
            return
        if val[s][i] is None:
            val[s][i], vsrc[s][i] = v, how
    score_at = {}                                       # rounds played before the prompt -> {seat: public score}
    pend = []                                           # (seat, round index, pending bid shown)
    saved = run.get("prompt") or {}
    prompts = [(m[3], "call") for m in meta] + ([(prompt_info(saved.get("text")), "run")] if saved.get("text") else [])
    for info, how in prompts:
        i = rindex.get(info["round"])
        if info["seat"] is not None and i is not None:
            put(info["seat"], i, info["value"], "prompt")
            if info["pending"] is not None:
                pend.append((info["seat"], i, info["pending"]))
        if info["round"] is not None and info["scores"]:
            k = info["round"] - 1
            for s, v in info["scores"].items():
                old = score_at.setdefault(k, {}).get(s)
                if old is not None and old != v:
                    warns.append(f"prompts disagree on {bname(s)}'s score after {k} rounds: {old} and {v}")
                score_at[k][s] = v
    if G["fixed"]:
        for s in range(n):
            for i in range(R):
                put(s, i, G["fixed"][s] if s < len(G["fixed"]) else Fraction(10), "config")
    for s, i, pb in pend:                               # a prompt's pending bid is the bidder's previous bid
        if i > 0 and pb != bids[s][i - 1]:
            warns.append(f"round {RD[i]['r']}: {bname(s)}'s prompt shows pending bid {pb}, the log's round "
                         f"{RD[i - 1]['r']} bid is {bids[s][i - 1]}")

    def pub_score(k, s):
        """A bidder's public score after k rounds: 0 at the start, a prompt's public scores, the summary's reward at the end."""
        if k == 0:
            return Fraction(0), "the rules"
        if score_at.get(k, {}).get(s) is not None:
            return score_at[k][s], "a later prompt"
        if k == R and rewards[s] is not None:
            return rewards[s], "the summary's rewards"
        return None, ""
    for i in range(R):                                  # winner's value = score change + price
        w = win[i]
        if w is None or val[w][i] is not None:
            continue
        a, z = pub_score(i, w)[0], pub_score(i + 1, w)[0]
        if a is not None and z is not None:
            put(w, i, z - a + price[i], "scores")
    for s in range(n):                                  # a bidder's only unknown win: its final reward pins it down
        if rewards[s] is None:
            continue
        mine = [i for i in range(R) if win[i] == s]
        unk = [i for i in mine if val[s][i] is None]
        if len(unk) == 1:
            got = sum(val[s][i] - price[i] for i in mine if i != unk[0])
            put(s, unk[0], rewards[s] - got + price[unk[0]], "scores")
    n_val = sum(1 for s in range(n) for i in range(R) if val[s][i] is not None)
    n_inferred = sum(1 for s in range(n) for i in range(R) if vsrc[s][i] == "scores")
    D = n * R

    # ---- per decision: band (exact when the value is known; else the band every possible value gives), share of value
    possible = sorted(set(pool)) if pool else None
    kind = [[None] * R for _ in range(n)]
    share = [[None] * R for _ in range(n)]
    for s in range(n):
        for i in range(R):
            bd, v = bids[s][i], val[s][i]
            if v is not None:
                kind[s][i] = band(bd, v, n, pricing, tab)
                share[s][i] = Fraction(bd) / v if v > 0 else None
            else:
                ks = {band(bd, x, n, pricing, tab) for x in possible} if possible else set()
                kind[s][i] = ks.pop() if len(ks) == 1 else "unknown"
                if bd == 0 and possible and min(possible) > 0:
                    share[s][i] = Fraction(0)            # 0 is 0% of any value the pool allows
    share_from_zero = sum(1 for s in range(n) for i in range(R) if val[s][i] is None and share[s][i] is not None)

    # ---- payoffs, scores, efficiency, equilibrium benchmarks
    pay = [[None] * R for _ in range(n)]
    for i in range(R):
        for s in range(n):
            pay[s][i] = Fraction(0) if s != win[i] else (val[s][i] - price[i] if val[s][i] is not None else None)
    score = [[None] * R for _ in range(n)]
    for s in range(n):                                  # computed from values and prices, else the public score
        prev = Fraction(0)
        for i in range(R):
            comp = prev + pay[s][i] if prev is not None and pay[s][i] is not None else None
            pub, where = pub_score(i + 1, s)
            if comp is not None and pub is not None and comp != pub:
                warns.append(f"{bname(s)}'s score after round {RD[i]['r']}: {fn(pub)} in {where}, {fn(comp)} from the values "
                             "and prices")
            score[s][i] = prev = comp if comp is not None else pub
    final = [score[s][-1] if score[s][-1] is not None else rewards[s] for s in range(n)]
    all_val = [all(val[s][i] is not None for s in range(n)) for i in range(R)]
    eff = [None] * R
    eq_pay = [[None] * R for _ in range(n)]
    eq_rev = [None] * R
    for i in range(R):
        if all_val[i]:
            vs = [val[s][i] for s in range(n)]
            eff[i] = val[win[i]][i] == max(vs)
            p, rv = eq_round(vs, n, pricing, tab)
            eq_rev[i] = rv
            for s in range(n):
                eq_pay[s][i] = p[s]
    every_val = all(all_val)
    over_n = sum(1 for s in range(n) for i in range(R) if kind[s][i] == "over")
    cnt = {k: sum(1 for s in range(n) for i in range(R) if kind[s][i] == k) for k in KINDS}
    done = R >= Rcfg                                    # every configured round played: the final scores decide the game
    over_known = all(kind[s][i] != "unknown" for s in range(n) for i in range(R))
    end = G["end"]
    eff_n = sum(1 for e in eff if e) if every_val else (int(end["efficient_rounds"]) if end.get("efficient_rounds") is not None
                                                          else None)
    if every_val and end.get("efficient_rounds") is not None and int(end["efficient_rounds"]) != eff_n:
        warns.append(f"game_end efficient_rounds {end['efficient_rounds']}, {eff_n} from the prompts' values")
    if over_known and end.get("overbids") is not None and int(end["overbids"]) != over_n:
        warns.append(f"game_end overbids {end['overbids']}, {over_n} bids above the bidder's value")
    top = max((f for f in final if f is not None), default=None)
    lead = [s for s in range(n) if final[s] is not None and final[s] == top]
    if isinstance(G["winners"], list) and None not in final and sorted(int(w) for w in G["winners"]) != lead:
        warns.append(f"winners {G['winners']} logged, highest scores: seats {lead}")
    for w in warns:
        print("warning:", w, file=sys.stderr)
    E = expected_eq(pool, n, pricing, tab)
    revenue = sum(price)
    wins = [sum(1 for i in range(R) if win[i] == s) for s in range(n)]
    zeros = sum(1 for s in range(n) for i in range(R) if bids[s][i] == 0)
    ties = [i for i in range(R) if sum(1 for s in range(n) if bids[s][i] == max(RD[i]["bids"])) > 1]
    zero_ties = [i for i in ties if max(RD[i]["bids"]) == 0]
    sh_known = [(s, i, share[s][i]) for s in range(n) for i in range(R) if share[s][i] is not None]
    sh_val = [(s, i, x) for s, i, x in sh_known if val[s][i] is not None]       # shares whose value is actually known
    # the mean share: over every bid when every share is known; else only over bids with a known value (a 0 bid's share is
    # known without its value, a positive bid's is not, so mixing them would pull the mean toward 0)
    sh_use = sh_known if len(sh_known) == D else sh_val
    share_ok = 2 * len(sh_val) >= D                     # share statistics only when at least half the values are known
    mean_share = sum(x for _, _, x in sh_use) / len(sh_use) if sh_use else None
    if tab:                                             # the paper's table: the equilibrium share differs from value to value
        eq_sh = [EQ(val[s][i]) / val[s][i] for s, i, _ in sh_use if val[s][i] is not None and val[s][i] > 0]
        eq_ref = sum(eq_sh) / len(eq_sh) if eq_sh else None
    else:
        eq_ref = eqr
    eq_total_real = sum(eq_rev) if every_val else None
    eq_bid_total = [sum(eq_pay[s]) for s in range(n)] if every_val else None
    bidders_total = sum(final) if None not in final else None
    surplus_real = sum(val[win[i]][i] for i in range(R)) if all(val[win[i]][i] is not None for i in range(R)) else None

    # ---- the recorder's fallback and forced answers, per decision. When the agent cannot parse a legal action (or the call
    # fails) it plays a random listed legal action; for the bid's struct field the only listed one is the declared default, a
    # bid of 0. The run log prints that warning once per run, so any logged 0 without a matching call may be the fallback.
    runlog = b.get("runlog") or ""
    fb_parse = runlog.count("Could not parse a legal JSON action")
    fb_rand = len(re.findall(r"using random legal action", runlog))
    fb_any = len(re.findall(r"\[LLM fallback\]", runlog))
    bstep = {(st["seat"], st["x"]): st for st in steps if st["phase"] == "BID"}
    maybe_fb = [[bool(fb_parse or fb_rand) and bids[s][i] == FALLBACK_BID and not bstep[(s, i)].get("hit")
                 for i in range(R)] for s in range(n)]
    forced = [[bool(bstep[(s, i)].get("rescued")) for i in range(R)] for s in range(n)]
    n_fb = sum(map(sum, maybe_fb))
    n_forced = sum(map(sum, forced))
    dk = [["fallback" if maybe_fb[s][i] else kind[s][i] for i in range(R)] for s in range(n)]   # the kind shown on the page
    FB_NOTE = "may be the recorder's fallback bid, played when the model's answer cannot be parsed"

    # ---- observations rebuilt from the log (used where no prompt was saved)
    rule_line = (f"{cap(pricing)}-price sealed-bid auction, {n} bidders, bids 0 to {BMAX}; values are drawn from "
                 f"{pool_text(pool)} each round; the winner pays {'its own bid' if first else 'the second-highest bid'}.")
    history = []
    for i in range(R):
        for st in [st for st in steps if st["x"] == i and st["phase"] == "BID"]:
            s = st["seat"]
            v = val[s][i]
            before = [score[k][i - 1] if i else Fraction(0) for k in range(n)]
            lines = ["(Rebuilt from the game log; the exact prompt for this turn was not saved.)",
                     f"You are seat P{s} ({B[s]} on this page). Your role: PLAYER.",
                     ("Your private info: " + json.dumps({"value": num(v)})) if v is not None else
                     "Your private info: this round's value (not in the log).",
                     f"Current phase: BID, round {RD[i]['r']} of {Rcfg}. {rule_line}",
                     ("Scores so far: " + ", ".join(f"P{k} {fn(before[k])}" for k in range(n)) + ".") if None not in before else
                     "Scores so far: " + ", ".join(f"P{k} " + (fn(before[k]) if before[k] is not None else "unknown")
                                                   for k in range(n)) + " (the log records only the final scores)."]
            said = [t for t in RD[i]["talk"]]
            if said:
                lines += ["Messages this round:"] + [f"- P{t['seat']}: {t['text']}" for t in said]
            lines += (["Earlier rounds:"] + history) if history else ["Earlier rounds: none yet."]
            st.setdefault("state", "\n".join(lines))
        history.append(f"- Round {RD[i]['r']}: bids {RD[i]['bids']}, winner P{win[i]}, price {fn(price[i])}.")
    for st in steps:                                    # talk steps without a prompt
        if "state" not in st:
            i = st["x"]
            st["state"] = "\n".join(["(Rebuilt from the game log; the exact prompt for this turn was not saved.)",
                                     f"You are seat P{st['seat']} ({B[st['seat']]} on this page). Your role: PLAYER.",
                                     f"Current phase: TALK, round {RD[i]['r']} of {Rcfg}. {rule_line}"]
                                    + (["Earlier rounds:"] + history[:i] if i else ["Earlier rounds: none yet."]))
    if saved.get("text"):                               # the run's saved prompt, for the decision it belongs to
        info = prompt_info(saved["text"])
        for st in steps:
            if (st["seat"] == info["seat"] and st["phase"] == info["phase"] and st["round"] == info["round"]
                    and not st.get("state_from_prompt")):
                st["state"] = en(common.state_part(saved["text"]))
                st["state_from_prompt"] = True
                break

    # ---- heads and kinds of the bid steps
    def outcome(s, i):
        top_b = max(RD[i]["bids"])
        k = sum(1 for x in RD[i]["bids"] if x == top_b)
        if s == win[i]:
            w = (f"wins the {k}-way tie at {fn(top_b)}" if k > 2 else "wins the tie" if k == 2 else "wins") + f", pays {fn(price[i])}"
            return w + (f", {fn(pay[s][i], signed=True)}" if pay[s][i] is not None else "")
        if bids[s][i] == top_b:
            return f"loses the tie-break to {B[win[i]]}"
        return f"loses to {B[win[i]]}"
    for st in steps:
        if st["phase"] != "BID":
            tx = _ws(st["text"])
            st["head"] = f"Round {st['round']} {DOT} says {LDQ}{tx[:48].rstrip()}{'...' if len(tx) > 48 else ''}{RDQ}"
            continue
        s, i = st["seat"], st["x"]
        v = val[s][i]
        notes = ([pct(share[s][i])] if v is not None and share[s][i] is not None else []) + \
            (["may be the fallback bid"] if maybe_fb[s][i] else []) + (["forced answer"] if forced[s][i] else [])
        bid_w = f"bids {st['bid']}" + (f" ({', '.join(notes)})" if notes else "")
        st["head"] = (f"Round {st['round']} {DOT} " + (f"value {fn(v)} {DOT} " if v is not None else "") + f"{bid_w} {DOT} "
                      + outcome(s, i))
        st["kind"] = dk[s][i]
    has_reason = any(st.get("thinking") for st in steps)
    rescued = sum(1 for st in steps if "[The reasoning ran out before an answer" in st.get("thinking", ""))
    from_prompt = sum(1 for st in steps if st.get("state_from_prompt"))
    for st in steps:
        st.setdefault("thinking", "")
        for k in ("round", "phase", "bid", "text", "state_from_prompt", "mismatch", "hit", "rescued"):
            st.pop(k, None)
    last_bid_step = {}
    for k, st in enumerate(steps):
        last_bid_step[st["x"]] = k

    # ---- feed
    feed = [{"x": 0, "step": 0, "who": "Rules",
             "text": f"{cap(numword(n))} bidders, {plural(Rcfg, 'round', word=True)}, {pricing} price: each round every bidder "
                     f"privately draws a value from {pool_text(pool)} and submits one sealed bid from 0 to {BMAX}; the highest "
                     f"bid wins and pays {'its own bid' if first else 'the second-highest bid'}; ties are broken at random. "
                     + (f"{cap(plural(TP, 'public message', word=True))} per bidder before each bid." if TP else "No messages.")}]
    step_of_talk, k = {}, 0
    for kk, st in enumerate(steps):
        if st["kind"] == "talk":
            step_of_talk.setdefault(st["x"], []).append(kk)
    for i in range(R):
        for t, kk in zip(RD[i]["talk"], step_of_talk.get(i, [])):
            feed.append({"x": i, "step": kk, "who": B[t["seat"]], "kind": "talk", "text": t["text"]})
        vs = [val[s][i] for s in range(n)]
        top_b = max(RD[i]["bids"])
        kt = sum(1 for x in RD[i]["bids"] if x == top_b)
        txt = ""
        if any(v is not None for v in vs):
            txt += "Values " + " / ".join(fn(v) if v is not None else NDASH for v in vs) + ". "
        txt += "Bids " + " / ".join(str(x) for x in RD[i]["bids"]) + ". "
        txt += (f"{cap(numword(kt))}-way tie at {fn(top_b)}: {B[win[i]]} wins the random tie-break" if kt > 2 else
                f"Tie at {fn(top_b)}: {B[win[i]]} wins the random tie-break" if kt == 2 else f"{B[win[i]]} wins")
        txt += f" and pays {fn(price[i])}"
        w = win[i]
        if pay[w][i] is not None:
            txt += f": {fn(val[w][i])} {MINUS} {fn(price[i])} = {fn(pay[w][i], signed=True)}."
        else:
            txt += "; its value, and so its payoff, is not in the log."
        if all_val[i]:
            hv = max(vs)
            txt += (" The highest value won." if eff[i] else f" The highest value ({fn(hv)}) did not win.")
            txt += (f" Equilibrium bids {' / '.join(fn(EQ(v)) for v in vs)}: price {fn(eq_rev[i])}.")
        if all(score[s][i] is not None for s in range(n)):
            txt += " Scores " + " / ".join(fn(score[s][i]) for s in range(n)) + "."
        fs = [s for s in range(n) if forced[s][i]]
        if fs:
            txt += (" " + join_and(f"{B[s]}'s bid ({bids[s][i]})" for s in fs) + f" {'is an' if len(fs) == 1 else 'are'} "
                    + ("answer" if len(fs) == 1 else "answers") + " forced after the reasoning ran out.")
        nfb_i = sum(1 for s in range(n) if maybe_fb[s][i])
        if nfb_i:
            txt += f" {'Each' if nfb_i > 1 else 'The'} bid of {FALLBACK_BID} {FB_NOTE}."
        feed.append({"x": i, "step": last_bid_step[i], "who": "Result", "text": txt})

    # ---- chart 1: the bids grid (every cell at full opacity, so its white label stays legible; a Winner row names the winner).
    # A column's tooltip is one unwrapped line per row ("Bidder 1: " + tip), so the tips stay short (about 80 characters); the
    # legend and the chart's sub explain the bands, the fallback and forced answers in full.
    rows = B + ["Winner", "Price"]
    gcells = [[] for _ in rows]
    for i in range(R):
        kt = sum(1 for x in RD[i]["bids"] if x == max(RD[i]["bids"]))
        for s in range(n):
            v, bd, k_ = val[s][i], bids[s][i], kind[s][i]
            if v is not None:
                parts = [f"value {fn(v)}", f"bid {bd}" + (f" ({pct(share[s][i])})" if share[s][i] is not None else ""),
                         f"above value (eq {fn(EQ(v))})" if k_ == "over" else f"{BAND_TIP[k_]} {fn(EQ(v))}"]
            else:
                parts = [f"bid {bd}", "value not logged"] + ([BAND_TIP[k_]] if k_ in BAND_TIP else [])
            if maybe_fb[s][i]:
                parts.append("may be fallback")
            if forced[s][i]:
                parts.append("forced answer")
            if s == win[i]:
                parts.append("won" + (f" at {fn(price[i])}" if price[i] != bd else "")
                             + (f", {fn(pay[s][i], signed=True)}" if pay[s][i] is not None else ""))
            else:
                parts.append("lost")
            if score[s][i] is not None:
                parts.append(f"score {fn(score[s][i])}")
            gcells[s].append({"c": KIND[dk[s][i]][1], "t": str(bd), "o": 1, "tip": f" {DOT} ".join(parts)})
        hv = max(val[s][i] for s in range(n)) if all_val[i] else None
        gcells[n].append({"c": "hold", "t": f"B{win[i] + 1}", "o": 1,
                          "tip": f"{B[win[i]]} wins" + (f" a {kt}-way tie-break" if kt > 1 else "")
                                 + (" (highest value)" if eff[i] else f"; the highest value ({fn(hv)}) lost" if eff[i] is False
                                    else "")})
        gcells[n + 1].append({"c": "hold", "t": fn(price[i]), "o": 1,
                              "tip": f"{B[win[i]]} paid {fn(price[i])}"
                                     + (f" {DOT} eq price {fn(eq_rev[i])}" if eq_rev[i] is not None else "")})
    for row in gcells:
        row += [None] * (N - R)
    used = [k for k in ("below", "near", "above", "over", "unknown", "fallback")
            if any(dk[s][i] == k for s in range(n) for i in range(R))]
    lab = {k: KIND[k][0] for k in used}
    if "unknown" in used and "fallback" in used:        # one legend entry per color
        lab["unknown"] = "Value not logged, or a 0 that may be the recorder's fallback"
        used.remove("fallback")
    legend = [[lab[k], KIND[k][1]] for k in used] + [["Winner and price", "hold"]]
    BCOL = bidder_colors(n, {KIND[dk[s][i]][1] for s in range(n) for i in range(R)})
    eq_words = (f"the risk-neutral Nash bid that Goeree, Holt and Palfrey (2002) list for this value pool: {tab_text}" if tab else
                f"{frac(eqr)} of the value, the rules' risk-neutral benchmark for {n} bidders" if first else
                "the value itself, weakly dominant at second price")
    sub1 = (f"One column per round. Bidder rows: the bid, colored by how it compares with the equilibrium bid for the bidder's "
            f"value ({eq_words}): "
            + ", ".join(f"{COLOR_WORD[KIND[k][1]]} {lab[k][:1].lower() + lab[k][1:]}" for k in used) + ". ")
    by_bounds_shown = sum(1 for s in range(n) for i in range(R) if val[s][i] is None and dk[s][i] not in ("unknown", "fallback"))
    if by_bounds_shown:
        lo_v = min(possible)
        sub1 += (f"Values are not in the log; where a bid falls in the same band for every value the pool allows, it gets that "
                 f"band's color (the lowest value, {fn(lo_v)}, has an equilibrium bid of {fn(EQ(lo_v))}). ")
    if n_fb:
        sub1 += (f"Light gray {FALLBACK_BID}: the run log shows the recorder could not parse the model's answer at least once and then played its "
                 f"fallback bid of {FALLBACK_BID}; it does not say in which decisions, so each logged 0 may be the model's bid or "
                 "the fallback. ")
    if n_forced:
        sub1 += (f"In {plural(n_forced, 'cell')} the bid is an answer forced after the reasoning ran out (see the cell's "
                 "details). ")
    sub1 += (f"Winner row: who won (B1 = {B[0]}); Price row: what the winner paid. Hover a column for the details.")
    charts = [{"title": "Bids, round by round", "sub": sub1, "height": (n + 2) * 28 + 24,
               "grid": {"rows": rows, "cells": gcells, "rowHeight": 28, "legend": legend}, "bars": []}]

    # ---- chart 2: bid as a share of value (only when at least half the values are actually known and the shares differ)
    plotted = {x for _, _, x in sh_known}
    if share_ok and len(plotted) > 1:
        lines2 = []
        for s in range(n):
            v2 = [round(float(share[s][i]) * 100, 4) if share[s][i] is not None else None for i in range(R)]
            if any(x is not None for x in v2):
                lines2.append({"name": B[s], "color": BCOL[s], "v": pad(v2), "dots": True})
        if tab:                                         # the paper's bids are not a fixed share: one dashed line per bidder
            for s in range(n):
                e2 = [round(float(EQ(val[s][i]) / val[s][i]) * 100, 4) if val[s][i] else None for i in range(R)]
                if any(x is not None for x in e2):
                    lines2.append({"name": f"{B[s]}, equilibrium", "color": BCOL[s],
                                   "dash": True, "thin": True, "v": pad(e2)})
        else:
            lines2.append({"name": "Equilibrium bid" if first else "Equilibrium bid = value", "color": "c3", "dash": True,
                           "v": [round(float(eqr) * 100, 4)] * N, "end": True})
        if first:
            lines2.append({"name": "Own value", "color": OWN_VALUE_COLOR, "dash": True, "v": [100] * N, "end": True})
        if tab:
            sub2 = ("Each bid divided by the bidder's value that round. Thin dashed, one per bidder in its color: the risk-neutral "
                    f"Nash bid Goeree, Holt and Palfrey (2002) list for that value ({tab_text}), as a share of the value; ")
        else:
            sub2 = (f"Each bid divided by the bidder's value that round. Green dashed: the equilibrium bid, {pct(eqr)} of the "
                    "value " + (f"((n {MINUS} 1)/n with n = {n}); " if first else
                                "(bidding one's value is weakly dominant at second price); "))
        sub2 += ((f"{COLOR_WORD[OWN_VALUE_COLOR]} dashed: the value itself, above which a win loses money. " if first else
                  "above it a win can lose money. ")
                 + f"Mean over {plural(len(sh_use), 'bid')}: {pct(mean_share)}"
                 + (f" (equilibrium: {pct(eq_ref)} on the same bids)" if tab and eq_ref is not None else "") + ". "
                 + cap(HUMAN[pricing]) + ".")
        if share_from_zero:
            sub2 += (f" {cap(plural(share_from_zero, 'bid', word=True))} of 0 {'is' if share_from_zero == 1 else 'are'} shown "
                     "at 0% although the value was not logged: 0 is 0% of any value the pool allows"
                     + ("" if sh_use is sh_known else "; the mean counts only bids whose value is known") + ".")
        if len(plotted) == 1 and len(lines2) > (3 if first else 2):
            sub2 += f" Every bid shown is {pct(next(iter(plotted)))} of value, so the bidders' lines lie on top of each other."
        miss = sum(1 for s in range(n) for i in range(R) if share[s][i] is None and val[s][i] is None)
        zero_v = sum(1 for s in range(n) for i in range(R) if val[s][i] == 0)
        if miss:
            sub2 += f" {cap(plural(miss, 'bid', word=True))} with an unknown value {'is' if miss == 1 else 'are'} left out."
        if zero_v:
            sub2 += (f" {cap(plural(zero_v, 'bid', word=True))} with a value of 0 {'has' if zero_v == 1 else 'have'} no share "
                     "and " + ("is" if zero_v == 1 else "are") + " left out (the grid shows them).")
        fit_end_labels(lines2, N, 220, lambda x: f"{x:,.1f}%")
        charts.append({"title": "Bid as a share of value", "sub": sub2, "height": 220, "fmt": "pct0", "lines": lines2, "bars": []})

    # ---- chart 3: running scores (when every round's score is known)
    if all(score[s][i] is not None for s in range(n) for i in range(R)):
        lines3 = [{"name": B[s], "color": BCOL[s], "v": pad([num(x) for x in score[s]]),
                   "dots": True, "end": True} for s in range(n)]
        sub3 = (f"Each bidder's score after each round: the sum of its payoffs (value {MINUS} price when it wins, 0 otherwise). "
                f"Final: " + ", ".join(f"{B[s]} {fn(final[s])} ({plural(wins[s], 'win')})" for s in range(n)) + ".")
        if eq_bid_total:
            sub3 += (" With the same values and equilibrium bidding they would expect " + " / ".join(fn(x) for x in eq_bid_total)
                     + " (the highest value always wins" + (" and pays its equilibrium bid" if tab else
                                                            " and keeps 1/" + str(n) + " of it" if first else
                                                            " and pays the second-highest value") + ").")
        if fit_end_labels(lines3, N, 220, lambda x: f"{x:,.0f}") and not all(l.get("end") for l in lines3):
            for l in lines3:
                l.pop("end", None)
        charts.append({"title": "Running score", "sub": sub3, "height": 220, "fmt": "int", "lines": lines3, "bars": []})

    # ---- chart 4: where the value went (cumulative bidders' payoff and seller revenue, against equilibrium play). Neutral
    # colors (ink: bidders, gray: seller): blue, green, orange and yellow mean bid bands in the grid.
    cum = lambda xs: [sum(xs[:k + 1]) if None not in xs[:k + 1] else None for k in range(len(xs))]
    bid_tot = [sum(pay[s][i] for s in range(n)) if all(pay[s][i] is not None for s in range(n)) else None for i in range(R)]
    c_bid = cum(bid_tot)
    if c_bid[-1] is None and bidders_total is not None:
        c_bid[-1] = bidders_total                       # the final scores give the total even when a round's payoff is unknown
    c_rev = cum(price)
    n_cb = sum(1 for x in c_bid if x is not None)
    show_bid = 2 * n_cb >= R                            # a bidders' line only when most rounds' running totals are known
    lines4 = []
    if show_bid:
        lines4.append({"name": "Bidders' payoff", "color": "ink", "v": pad([num(x) if x is not None else None for x in c_bid]),
                       "dots": True, "end": True})
    lines4.append({"name": "Seller revenue", "color": "hold", "v": pad([num(x) for x in c_rev]), "dots": True, "end": True})
    pays_w = " and pays its equilibrium bid)." if tab else (f" and pays {frac(eqr)} of it)." if first else
                                                          " and pays the second-highest value).")
    if every_val:
        eqb = cum([sum(eq_pay[s][i] for s in range(n)) for i in range(R)])
        if show_bid:
            lines4.append({"name": "Bidders, equilibrium", "color": "ink", "dash": True, "v": pad([num(x) for x in eqb]),
                           "end": True})
        lines4.append({"name": "Seller, equilibrium", "color": "hold", "dash": True, "v": pad([num(x) for x in cum(eq_rev)]),
                       "end": True})
        bench4 = "Dashed: the same with equilibrium bidding and these values (the highest value wins every round" + pays_w
    elif E:
        if show_bid:
            lines4.append({"name": "Bidders, equilibrium (expected)", "color": "ink", "dash": True,
                           "v": [num(E["bidders"] * (k + 1)) for k in range(N)], "end": True})
        lines4.append({"name": "Seller, equilibrium (expected)", "color": "hold", "dash": True,
                       "v": [num(E["price"] * (k + 1)) for k in range(N)], "end": True})
        bench4 = (f"Dashed: what equilibrium bidding is expected to give with values drawn from the pool: the highest of "
                  f"{numword(n)} values averages {fn(E['max'])}, of which the seller expects {fn(E['price'])} a round and the "
                  f"winner keeps {fn(E['bidders'])}.")
    else:
        bench4 = ""
    eq_end = (sum(sum(eq_pay[s]) for s in range(n)), eq_total_real) if every_val else \
        ((E["bidders"] * R, E["price"] * R) if E else None)
    if show_bid:
        sub4 = ("Running totals: the bidders' payoffs added up (value minus price for each round's winner) and the seller's "
                "revenue (the prices paid). ")
        if c_bid[-1] is not None and any(x is None for x in c_bid[:-1]):
            early = [RD[i]["r"] for i in range(R - 1) if c_bid[i] is not None]
            sub4 += (f"The bidders' running total is known only " + (f"after {rounds_text(early)} (every winner's value so far "
                     "known) and " if early else "") + f"at the end ({fn(c_bid[-1])}, the sum of the final scores): the other "
                     "rounds' winners' values are not in the log. ")
    else:
        sub4 = ("Running total of the seller's revenue (the prices paid). The bidders' payoffs are not drawn: their running "
                f"total is known after only {plural(n_cb, 'round')} of {R} (the winners' values are not in the log)"
                + (f"; at the end it is {fn(c_bid[-1])}, the sum of the final scores" if c_bid[-1] is not None else "")
                + (f", against {fn(eq_end[0])}" + ("" if every_val else " expected") + " with equilibrium bidding"
                   if c_bid[-1] is not None and eq_end else "") + ". ")
    sub4 += bench4
    if eq_end:
        sub4 += (f" After round {RD[-1]['r']}: " + (f"bidders {fn(c_bid[-1])}, " if show_bid and c_bid[-1] is not None else "")
                 + f"seller {fn(c_rev[-1])}; equilibrium bidding: " + (f"bidders {fn(eq_end[0])}, " if show_bid else "")
                 + f"seller {fn(eq_end[1])}" + ("" if every_val else " (expected)") + ".")
    if surplus_real is not None:
        sub4 += f" Together the winners' values add up to {fn(surplus_real)}."
    all4 = [x for l in lines4 for x in l["v"] if x is not None] + [0]
    span4 = max(all4) - min(all4)
    real4 = [l for l in lines4 if not l.get("dash")]
    flat4 = all(max(x for x in l["v"] if x is not None) - min(x for x in l["v"] if x is not None) < Fraction(2, 100) * span4
                for l in real4) if span4 else True
    if not flat4:                                       # a lone flat line against the benchmarks says nothing the tiles do not
        fit_end_labels(lines4, N, 230, lambda x: f"{x:,.0f}")
        charts.append({"title": "Where the value went", "sub": sub4, "height": 230, "fmt": "int", "lines": lines4, "bars": []})

    # ---- tiles
    spark_share = []
    for i in range(R):
        xs = [share[s][i] for s in range(n) if share[s][i] is not None]
        spark_share.append(round(float(sum(xs) / len(xs)) * 100, 4) if len(xs) == n else None)
    nonzero = [(s, i) for s in range(n) for i in range(R) if bids[s][i] != 0]
    lo_pos = min((v for v in possible if v > 0), default=None) if possible else None
    eq_short = (f"the risk-neutral Nash bid Goeree, Holt and Palfrey (2002) list for the value ({tab_text})" if tab else
                f"{frac(eqr)} of the value" if first else "the value itself")
    fb_zero_w = ("the bid the recorder plays when it cannot parse the model's answer (the run log shows this happened at least "
                 "once; how often was not recorded)")
    if 2 * zeros >= D:
        others = ", ".join(f"{bids[s][i]} ({B[s]}, round {RD[i]['r']})" for s, i in nonzero[:4]) + (
            f" and {len(nonzero) - 4} more" if len(nonzero) > 4 else "")
        if n_fb:
            lead1 = (f"{zeros} of {D} logged bids are 0, {fb_zero_w}, so each may be the model's bid or the fallback"
                     if n_fb == zeros else
                     f"Bids of 0 in {zeros} of {D} decisions; {n_fb} of them have no matching model answer and may be the "
                     f"recorder's fallback bid, played when it cannot parse the model's answer")
        else:
            lead1 = f"Bids of 0 in {zeros} of {D} decisions"
        t1 = {"k": "Bids of 0", "v": f"{zeros} of {D}",
              "s": (lead1 + (f"; the other bid{'s' if len(nonzero) > 1 else ''}: {others}" if nonzero else "")
                    + f". The equilibrium bid is {eq_short}"
                    + (f", at least {fn(EQ(lo_pos))} for the lowest positive value in the pool"
                       if lo_pos is not None and not tab else "") + "."),
              "spark": [sum(1 for s in range(n) if bids[s][i] == 0) for i in range(R)]}
    elif share_ok:
        fb_below = sum(1 for s in range(n) for i in range(R) if forced[s][i] and kind[s][i] == "below")
        # the mean without the forced answers (they are not the model's deliberate bids)
        sh_free = [x for s, i, x in sh_use if not forced[s][i]]
        free_w = (f"; the other {len(sh_free)} average {pct(sum(sh_free) / len(sh_free))}"
                  if n_forced and sh_free and len(sh_free) < len(sh_use) else "")
        t1 = {"k": "Bid as a share of value", "v": pct(mean_share),
              "s": (f"Mean over {plural(len(sh_use), 'bid')}; "
                    + (f"the paper's risk-neutral Nash bids average {pct(eq_ref)} of the same values. " if tab else
                       f"the equilibrium bid is {pct(eqr)} of the value. ")
                    + f"Bids below it: {cnt['below']}; at it (within {TOL}): {cnt['near']}; "
                    + (f"above it, up to the value: {cnt['above']}; " if first else "")
                    + f"above the value: {cnt['over']}" + (f"; value unknown: {cnt['unknown']}" if cnt["unknown"] else "") + ". "
                    + (f"{cap(plural(n_forced, 'bid', word=True))} {'is an answer' if n_forced == 1 else 'are answers'} forced "
                       "after the reasoning ran out" + (f" ({numword(fb_below)} of them below it)" if fb_below else "")
                       + free_w + ". " if n_forced else "")
                    + (f"{cap(plural(n_fb, 'bid', word=True))} of 0 may be the recorder's fallback. " if n_fb else "")).strip()}
        sp = [x for x in spark_share if x is not None]
        if len(sp) >= 2:
            t1["spark"] = sp
    else:
        mean_bid = Fraction(sum(sum(bids[s]) for s in range(n)), D)
        t1 = {"k": "Average bid", "v": fn(mean_bid),
              "s": f"Over {plural(D, 'bid')}; " + ("the values were not logged, so bid shading cannot be measured" if not n_val
                                                  else f"only {n_val} of {D} values were logged, too few to measure bid shading")
                   + (f"; {plural(n_fb, 'bid')} of 0 may be the recorder's fallback" if n_fb else "") + ".",
              "spark": [num(Fraction(sum(bids[s][i] for s in range(n)), n)) for i in range(R)]}
    if eq_total_real is not None:
        t2s = (f"Sum of the {plural(R, 'price')}. Equilibrium bidding with the same values would bring {fn(eq_total_real)}"
               + (f", so the seller got {pct(Fraction(revenue) / eq_total_real)} of that." if eq_total_real else "."))
        t2v = f"{fn(revenue)} vs {fn(eq_total_real)}"
    elif E:
        t2s = (f"Sum of the {plural(R, 'price')}. Equilibrium bidding is expected to bring {fn(E['price'] * R)}: "
               f"{fn(E['price'])} a round, " + ("the expected equilibrium bid of the highest" if tab else
                                                 (frac(eqr) + ' of ' if first else '') + "the expected "
                                                 + ('highest' if first else 'second-highest'))
               + f" of {numword(n)} values from the pool.")
        t2v = f"{fn(revenue)} vs {fn(E['price'] * R)}"
    else:
        t2s, t2v = f"Sum of the {plural(R, 'price')}.", fn(revenue)
    t2 = {"k": "Seller revenue vs equilibrium", "v": t2v, "s": t2s}
    if None not in final:
        best = max(final)
        lead = [s for s in range(n) if final[s] == best]
        t3s = ((f"{B[lead[0]]} {'wins' if done else 'leads'}" if len(lead) == 1 else
                join_and(B[s] for s in lead) + (" tie for first" if done else " tie")) + ". Rounds won: "
               + " / ".join(str(w) for w in wins) + f" of {R}.")
        if eq_bid_total:
            t3s += " Equilibrium bidding with these values: " + " / ".join(fn(x) for x in eq_bid_total) + "."
        elif E:
            t3s += (f" Together {fn(bidders_total)}; equilibrium bidding is expected to give {fn(E['bidders'] * R / n)} each "
                    f"({fn(E['bidders'] * R)} together) over {plural(R, 'round')}.")
        t3 = {"k": "Final scores", "v": " / ".join(fn(x) for x in final), "s": t3s}
    else:
        t3 = {"k": "Rounds won", "v": " / ".join(str(w) for w in wins), "s": " / ".join(B) + "."}
    tiles = [t1, t2, t3]
    if eff_n is not None:
        src_w = ("counted from the prompts' values" + (" (the game's own count agrees)" if end.get("efficient_rounds")
                                                         is not None else "")) if every_val else \
            "the game's own count (its hidden end-of-game record); the values themselves are not in the log"
        t4s = (f"Rounds won by the bidder with the round's highest value, {src_w}. Equilibrium bidding gives every round to "
               "the highest value.")
        misses = [i for i in range(R) if eff[i] is False]
        fmiss = [(i, s) for i in misses for s in range(n) if forced[s][i] and val[s][i] == max(val[k][i] for k in range(n))]
        if fmiss and len(misses) == 1:
            i, s = fmiss[0]
            t4s += (f" The one miss, round {RD[i]['r']}, is a forced answer: {B[s]} held the highest value ({fn(val[s][i])}) "
                    f"but bid {bids[s][i]}.")
        elif fmiss:
            t4s += (f" {cap(plural(len({i for i, _ in fmiss}), 'miss', 'misses', word=True))} of {len(misses)} "
                    f"({rounds_text([RD[i]['r'] for i, _ in fmiss])}) come from a forced answer by the bidder holding the "
                    "highest value.")
        tiles.append({"k": "Highest value won", "v": f"{eff_n} of {R}", "s": t4s})

    # ---- title, lede
    if 2 * zeros >= D:
        if n_fb == zeros:
            title = f"{zeros} of {D} logged bids are 0, {fb_zero_w}"
        else:
            title = f"The bidders bid 0 in {zeros} of {D} decisions"
            if n_fb:
                title += f" ({n_fb} of these zeros may be the recorder's fallback bid)"
            if zero_ties:
                title += (f": {numword(len(zero_ties))} of {numword(R)} items go by random tie-break at a price of 0"
                          if len(zero_ties) < R else f": every item goes by random tie-break at a price of 0")
            if eq_total_real is not None:
                title += (f", and the seller collects {fn(revenue)} instead of the {fn(eq_total_real)} equilibrium bidding "
                          "would bring")
            elif E:
                title += f", and the seller collects {fn(revenue)} instead of the {fn(E['price'] * R)} expected at equilibrium"
    elif share_ok and eq_ref is not None:
        gap = mean_share - eq_ref
        ref_w = f"the paper's risk-neutral {pct(eq_ref)}" if tab else pct(eq_ref)
        n_band = sum(1 for s in range(n) for i in range(R) if kind[s][i] != "unknown")
        n_hi, n_lo = cnt["above"] + cnt["over"], cnt["below"]
        over_w = ("none above value" if not over_n else f"{plural(over_n, 'bid', word=True)} above the bidder's own value")
        if abs(gap) <= Fraction(5, 100):                # near the benchmark on average: say which side most bids are on
            if gap >= 0 and 2 * n_hi > n_band:
                title = (f"Bids run slightly above the equilibrium: {pct(mean_share)} of value on average vs {ref_w} "
                         f"({n_hi} of {n_band} above it" + (f", {over_w})" if first else ")"))
            elif gap <= 0 and 2 * n_lo > n_band:
                title = (f"Bids run slightly below the equilibrium: {pct(mean_share)} of value on average vs {ref_w} "
                         f"({n_lo} of {n_band} below it" + (f", {over_w})" if over_n else ")"))
            else:
                title = (f"Bids stay close to the equilibrium: {pct(mean_share)} of value on average vs {ref_w}"
                         + (f", {over_w}" if over_n else ""))
        elif gap < 0:
            title = (f"The bidders shade more than the equilibrium: bids average {pct(mean_share)} of value vs {ref_w}"
                     + (f", {over_w}" if over_n else ""))
        else:
            title = (f"The bidders bid above the equilibrium: {pct(mean_share)} of value on average vs {ref_w}"
                     + (f", {over_w}" if over_n else ""))
        if eff_n is not None:
            title += f"; the highest value wins {eff_n} of {R} rounds"
        if done and None not in final:                  # who won the game
            best = max(final)
            top_s = [s for s in range(n) if final[s] == best]
            title += ("; " + (f"{B[top_s[0]]} wins {plural(wins[top_s[0]], 'round')} and the game ({fn(best)} points)"
                              if len(top_s) == 1 else f"{join_and(B[s] for s in top_s)} tie for first ({fn(best)} points)"))
    else:
        title = (f"The winning bid averages {fn(Fraction(revenue, R) if first else Fraction(sum(max(rd['bids']) for rd in RD), R))} "
                 f"over {plural(R, 'round')}; " + ("the private values were not logged" if not n_val else
                                                   "too few private values were logged to measure bid shading"))
    if R < Rcfg:
        title += f" (the recording stops after round {R} of {Rcfg})"
    lede = (f"{cap(numword(n))} bidders play {plural(Rcfg, 'round', word=True)} of a {pricing}-price sealed-bid auction: each "
            f"round every bidder privately learns its value for the item (drawn from {pool_text(pool)}) and submits one sealed "
            f"bid from 0 to {BMAX}. The highest bid wins and pays {'its own bid' if first else 'the second-highest bid'}, "
            "earning value minus price, ties are broken at random, and all bids are revealed after the round"
            + (f"; before bidding each bidder sends {plural(TP, 'public, non-binding message', 'public, non-binding messages', word=True)}. "
               if TP else ". ")
            + (f"In theory a risk-neutral bidder should bid the Nash bids Goeree, Holt and Palfrey (2002) list for these values: "
               f"{tab_text}." if tab else
               f"In theory each bidder should bid {frac(eqr)} of its value (the rules' risk-neutral equilibrium with "
               f"{numword(n)} bidders)." if first else "In theory each bidder should bid exactly its value, a weakly dominant "
                                                       "strategy at second price."))

    # ---- foot
    if fb_parse or fb_rand:
        fb = ("The recorder's run log reports that at least once " + ("no legal action could be parsed from the model's answer"
              if fb_parse else "a model call failed") + f"; the agent then plays its one listed legal action, a bid of "
              f"{FALLBACK_BID} (the warning is printed once per run, so it can stand for any number of decisions)"
              + (f"; the {plural(n_fb, 'logged bid')} of {FALLBACK_BID} without a matching model answer "
                 f"{'is' if n_fb == 1 else 'are'} marked gray in the grid and in {'its step' if n_fb == 1 else 'their steps'}"
                 if n_fb else "; every logged bid of 0 matches a model answer in the call log") + ". ")
    elif fb_any:
        fb = "The recorder's run log reports agent fallbacks (an API or other failure). "
    else:
        fb = "The recorder's run log reports no fallback action. " if runlog else ""
    cfg_bits = ", ".join(x for x in [f"seed {seed}" if seed is not None else "", plural(n, "bidder"), plural(Rcfg, "round"),
                                     f"{pricing} price", f"bids 0 to {BMAX}", f"values from {pool_text(pool)}",
                                     (f"{plural(TP, 'public message')} per bidder each round" if TP else "no messages")] if x)
    if has_reason:
        rs = (f"Observations and reasoning come from the saved model calls ({tied} of {len(steps)} steps matched to a call"
              + (f"; in {mism} the matched call's answer is not the logged action, which the step notes" if mism else "") + "). "
              + (f"In {plural(rescued, 'decision')} the reasoning ran out before an answer and the action is the answer to a "
                 "follow-up question; those steps, their cells in the bids grid and the round's result say so. "
                 if rescued else ""))
    else:
        rs = ("Reasoning was not saved in this recording (it has no call log), so no step shows the model's reasoning. "
              if not calls else "The call log has no reasoning, so no step shows the model's reasoning. ")
    obs = ("" if from_prompt == len(steps) else
           ("The observation of the first decision is the prompt saved with the run; the others are rebuilt from the log. "
            if from_prompt == 1 and not tied else
            ("The observations are rebuilt from the log. " if not from_prompt else
             f"{from_prompt} of {len(steps)} observations are saved prompts; the others are rebuilt from the log. ")))
    srcs = {vsrc[s][i] for s in range(n) for i in range(R)}
    val_from = [w for k, w in (("prompt", "the saved prompts" if tied else "the prompt saved with the run"),
                               ("config", "the config's fixed_values")) if k in srcs]
    if n_val == D:
        vals_w = ("Private values are not in the game log; every value here is read from "
                  + join_and(w for k, w in (("prompt", "the prompts"), ("config", "the config's fixed_values"),
                                            ("scores", "a winner's score change (value = score change + price)")) if k in srcs)
                  + (f" ({n_inferred} from score changes)" if n_inferred and len(srcs) > 1 else "") + ". ")
    elif n_val:
        shown = [f"{B[s]}'s round-{RD[i]['r']} value, {fn(val[s][i])}" for s in range(n) for i in range(R) if val[s][i] is not None]
        vals_w = (f"Private values are not in the game log; only {n_val} of {D} {'is' if n_val == 1 else 'are'} known here"
                  + (f" ({join_and(shown)}, from {'the prompt saved with the run' if not calls else 'the prompts'}"
                     + (f"; {n_inferred} recovered exactly from a winner's score change" if n_inferred else "") + ")"
                     if n_val <= 3 else " (from the prompts and score changes)")
                  + ", so payoffs per round, bid shading and the equilibrium bids for these values are shown only where the value "
                    "is known; the final scores are the summary's rewards. ")
    else:
        vals_w = ("Private values are not in the game log and no prompt was saved, so payoffs per round and bid shading are "
                  "unknown; the final scores are the summary's rewards. ")
    foot = (f"Data: the {date} recording of eco_sealed_auction ({cfg_bits}); every bidder is played by {model}"
            + (f" ({full_model})" if full_model and full_model != model else "") + ". " + rs + obs
            + "Every number on this page is computed from the game log"
            + (f" and {join_and(val_from)} (private values)" if val_from else "")
            + ": the round_result events (bids in seat order, winner, price)" + (", the talk events (the messages)" if TP else "")
            + ", the summary's rewards and the hidden game_end record. " + vals_w
            + ("Every price agrees with the pricing rule, and every check against the summary agrees. " if not warns else
               f"{cap(plural(len(warns), 'check', word=True))} disagree (printed by the builder). ")
            + fb
            + f"Bidder k is seat P(k{MINUS}1) in the log and the prompts; bidders are asked in that order and none sees another's "
              f"bid before the reveal. Bands: a bid within {TOL} of the equilibrium bid counts as at it (bids are whole numbers)"
            + ("; " if by_bounds_shown or "unknown" in used else ". ")
              + ("a value that is not logged gets the band the bid falls in for every value of the pool (light gray if that "
                 "differs). " if by_bounds_shown or "unknown" in used else "")
            + "Benchmarks (rules): "
            + (f"this is one of the Goeree, Holt and Palfrey (2002) configs (two bidders, first price, values equally likely "
               f"from {pool_text(pool)}), so the equilibrium bid is the paper's risk-neutral Nash bid for each value ({tab_text}; "
               f"Tables 1-2), in place of the (n {MINUS} 1)/n of the value that assumes continuous uniform values; "
               if tab else
               f"at first price the risk-neutral equilibrium bid with independent uniform values is (n {MINUS} 1)/n of the value, "
               f"{frac(eqr)} with {n} bidders (used here as stated, without rounding to whole numbers); "
               if first else "at second price bidding one's value is weakly dominant; ")
            + "equilibrium play gives each item to the highest value"
            + (f", and the expected figures assume values drawn uniformly from the pool (highest of {numword(n)}: "
               f"{fn(E['max'])} on average). " if E and not every_val else ". ")
            + "Human benchmark (rules): bidders overbid in second-price auctions and bid above the risk-neutral equilibrium in "
              "first-price auctions (Kagel, Harstad and Levin 1987; Goeree, Holt and Palfrey 2002 fit risk aversion r = 0.52 to "
              "two-bidder auctions, so no numeric human line is drawn here). "
              "Sources: Vickrey (1961), Journal of Finance; Milgrom and Weber (1982), Econometrica.")

    kinds = {k: v for k, v in KINDS.items() if any(st["kind"] == k for st in steps)}
    d = {"id": GID, "name": NAME, "title": title, "lede": lede, "seatNames": B, "tiles": tiles, "foot": foot,
         "x": {"n": N, "labels": [f"R{i + 1}" for i in range(N)]}, "charts": charts, "kinds": kinds, "steps": steps,
         "feed": feed, "feedTitle": "Round by round",
         "feedSub": ("Each round's public messages, then the reveal: " if TP else "Each round's reveal: ")
                    + "the bids (sealed until every bidder had bid), winner, price"
                    + (", payoffs and scores" if n_val == D else " and, where the values are known, payoffs and scores")
                    + ". The selected step is highlighted; later entries are hidden."}
    d = en_all(d)
    d["eyebrow"] = f"social_game_env {DOT} {GID} {DOT} {model}{', reasoning on' if has_reason else ''} {DOT} recorded {date}"
    return d


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    common.write(build(src, calls), out, date=recorded_date(src))
