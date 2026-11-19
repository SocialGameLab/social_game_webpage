"""Replay data for eco_volunteer (repeated volunteer's dilemma; Diekmann 1985, Franzen 1995, Goeree, Holt and Smith 2017).

usage: python3 -I eco_volunteer.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md, run.log) or a
          file inside one.
  OUT_DIR where <OUT_DIR>/eco_volunteer.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). Calls are tied to
          decisions in time order: a call belongs to the next decision of the same seat, phase and round (read from its prompt:
          "You are seat Pk", "Current phase: X", the public state's "round"; a value the prompt does not show agrees with
          anything, and a prompt without a phase is placed by its answer's action_id). Within one decision's calls the first
          whose answer is the logged action (the message text, or the volunteer decision) is the decision's call; earlier ones
          were failed attempts (format retries) and their reasoning is kept above it. Its reasoning becomes the step's thinking
          and its prompt the observation.

Each round has talk_passes public messages per player (TALK phase, seats in order, logged as talk events in time order) and then
one secret decision per player (DECIDE phase, seats asked in order P0, P1, ...; no player sees another's decision before the
reveal). The log has no event per decision: all decisions are in the round's round_result event (volunteered, in seat order, and
the number of volunteers). Payoffs are not logged; they are recomputed exactly with the rule of the game (a volunteer gets
v - cost, a non-volunteer v, everyone 0 when nobody volunteers) and checked against the logged volunteer counts, the summary's
rewards, the winners and the hidden game_end totals (volunteer_count_total, no_volunteer_rounds); any disagreement is printed as a
warning. The mixed-equilibrium benchmark is computed from the logged v, cost and number of players. With messages, each player's
message is read for a first-person statement of its own choice for that round (see stated_choice; anything else stays
unclassified) and compared with its decision. Nothing about one particular recording (seed, number of players or rounds,
outcome) is assumed."""
import json, os, re, sys, unicodedata
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_volunteer"
NAME = "Volunteer's dilemma"
DEFAULT_DATE = "2026-11-09"
MINUS, TIMES, DOT, LDQ, RDQ = "−", "×", "·", "“", "”"
KINDS = {"volunteer": ["Volunteers", "c1"],
         "abstain": ["Does not volunteer", "hold"],
         "talk": ["Public message", "muted"]}
VOL_C, ABS_C = KINDS["volunteer"][1], KINDS["abstain"][1]
PROVIDED_C, MULTI_C, NOBODY_C = "c3", "c4", "c2"     # outcome row: one volunteer (green), more than needed (yellow), nobody (orange)
EQ_C, ONE_C = "c4", "c3"                             # benchmark lines: the mixed equilibrium (yellow), one volunteer a round (green)
PLAYER_COLORS = ["c1", "c2", "c3", "c4", "hold"]    # players' volunteer-turn lines, by seat (up to five; more are summarised)
NUMW = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_ANY = re.compile("[　-ヿ㐀-鿿가-힯＀-￯]")
_CJK_RUN = re.compile("[぀-ヿ㐀-鿿가-힯]+")
_PUNCT = {"。": ". ", "、": ", ", "「": '"', "」": '"', "『": '"', "』": '"', "【": "[", "】": "]",
          "《": '"', "》": '"', "〈": "<", "〉": ">", "〜": "~"}


def en(s):
    """English only on the page: full-width forms become ASCII, CJK punctuation its ASCII counterpart, and any remaining run of
    Chinese, Japanese or Korean characters (a model may reason or write in Chinese) a visible marker."""
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
def exact(v):
    """A logged number as an exact Fraction (floats through their decimal text, so 0.1 stays 1/10)."""
    if isinstance(v, Fraction):
        return v
    if isinstance(v, float):
        return Fraction(repr(v))
    return Fraction(v)


def fn(v):
    """A payoff or score: an integer with thousands separators, anything else with up to two decimals."""
    v = exact(v)
    s = f"{abs(int(v)):,}" if v.denominator == 1 else f"{abs(float(v)):,.2f}".rstrip("0").rstrip(".")
    return (MINUS if v < 0 else "") + s


def num(v):
    """A chart value: an int when whole, else a float."""
    v = exact(v)
    return int(v) if v.denominator == 1 else round(float(v), 6)


def pct(a, b=1):
    """a/b as a percentage: whole numbers without decimals, otherwise one decimal (more if one would round a value strictly
    between 0% and 100% to either end)."""
    if not b:
        return "n/a"
    x = (exact(a) / exact(b) * 100) if not isinstance(a, float) else round(a / float(b) * 100, 9)   # (float noise: 4.0000000001)
    if float(x) == int(x):
        return f"{int(x)}%"
    for d in range(1, 4):
        s = f"{float(x):.{d}f}"
        if 0 < float(s) < 100 or not 0 < x < 100:
            return s + "%"
    return f"{float(x):.3f}%"


def numword(n):
    return NUMW[n] if 0 <= n < len(NUMW) else str(n)


def cap(s):
    return s[:1].upper() + s[1:]


def plural(n, one, many=None, word=False):
    return f"{numword(n) if word else n} {one if n == 1 else (many or one + 's')}"


def join_and(xs):
    xs = [str(x) for x in xs]
    if not xs:
        return ""
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def rounds_text(rs, cap_first=False):
    """[2, 3, 4] -> 'rounds 2 to 4'; [1] -> 'round 1'; [1, 3, 5] -> 'rounds 1, 3 and 5'; [1, 2, 3, 4, 10] -> 'rounds 1 to 4
    and 10' (runs of three or more consecutive rounds are written as a range)."""
    rs = sorted(rs)
    segs = []
    for r in rs:
        if segs and segs[-1][-1] == r - 1:
            segs[-1].append(r)
        else:
            segs.append([r])
    parts = [f"{g[0]} to {g[-1]}" if len(g) > 2 else str(x) for g in segs for x in ([g[0]] if len(g) > 2 else g)]
    w = "no round" if not rs else f"round {rs[0]}" if len(rs) == 1 else "rounds " + join_and(parts)
    return cap(w) if cap_first else w


def pname(s):
    return f"Player {s + 1}"


def players_text(seats, n):
    """[0, 1, 2] -> 'Players 1 to 3'; [1] -> 'Player 2'; [0, 2] -> 'Players 1 and 3'; every seat -> 'all four players'."""
    s = sorted(seats)
    if not s:
        return "nobody"
    if len(s) == n and n > 1:
        return "both players" if n == 2 else f"all {numword(n)} players"
    if len(s) == 1:
        return pname(s[0])
    if s == list(range(s[0], s[-1] + 1)) and len(s) > 2:
        return f"Players {s[0] + 1} to {s[-1] + 1}"
    return "Players " + join_and(x + 1 for x in s)


def _ws(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def runs(vals):
    """Maximal runs of equal values: [(first round, last round, value)] with 1-based rounds."""
    out = []
    for i, v in enumerate(vals):
        if out and out[-1][2] == v:
            out[-1][1] = i + 1
        else:
            out.append([i + 1, i + 1, v])
    return [tuple(x) for x in out]


# ------------------------------------------------------------------------------------------------ theory
def mixed_equilibrium(n, v, c):
    """The symmetric mixed equilibrium of the one-shot game (rules: Theory Benchmark): each player volunteers with probability
    p = 1 - (c/v)^(1/(n-1)); nobody volunteers with probability (c/v)^(n/(n-1)); each player expects v - c a round (it is
    indifferent, and volunteering always pays v - c). None when 0 < c < v does not hold (then there is no such equilibrium)."""
    if n < 2 or not 0 < c < v:
        return None
    r = float(c) / float(v)
    p = 1 - r ** (1 / (n - 1))
    q0 = r ** (n / (n - 1))
    return {"p": p, "none": q0, "pay": exact(v) - exact(c), "count": n * p}


def rotation_run(vols, n):
    """The longest run of consecutive rounds with exactly one volunteer each in which the volunteers repeat a fixed order that
    covers every seat (the run's first n rounds name each seat once, later rounds repeat that cycle); at least n rounds long.
    Returns (first, last, cycle) with 0-based round indices (the earliest run on a tie), or None."""
    R, best = len(vols), None
    if n < 2:
        return None
    for a in range(R):
        cyc, b = [], a
        while b < R and len(vols[b]) == 1:
            q, k = vols[b][0], b - a
            if k < n:
                if q in cyc:
                    break
                cyc.append(q)
            elif q != cyc[k % n]:
                break
            b += 1
        if b - a >= n and (best is None or b - a > best[1] - best[0] + 1):
            best = (a, b - 1, cyc)
    return best


# ------------------------------------------------------------------------------------------------ said versus did
# A message counts as its sender's own choice for the round only when it says so in the first person ("I will volunteer",
# "I'll not volunteer", "I won't volunteer", "I will abstain", "I'll hold"), about the current round or no named round, and not
# under a condition. A round named right after the verb decides which round is meant (else one named earlier in the same
# sentence); any other round, "next round", "later" and the like, or a condition word (if, unless, when, only, ...) leaves the
# statement unclassified. A message with two classified statements that disagree is unclassified.
_SAY = re.compile(r"(?<![A-Za-z])I(?:\s*['’]ll|\s+will|\s+shall)\s+(?:(not|never)\s+)?(volunteer|abstain|hold)(?![A-Za-z])"
                  r"|(?<![A-Za-z])I\s+won['’]t\s+(volunteer)(?![A-Za-z])")
_CUT = re.compile(r"[.;!?:\n—–]|,|\s(?:and|but|then|so|while|because|or)\s")
_SENT = re.compile(r"[.;!?\n]")
_NUMREF = re.compile(r"\brounds?\s+(\d+)", re.I)
_CUR = re.compile(r"\b(?:this|the current|current)\s+round\b", re.I)
_OTHER = re.compile(r"\b(?:next|last|previous|final|following|future|coming|upcoming|remaining|other|later)\s+(?:round|rounds|one)\b"
                    r"|\b(?:later|afterwards|next time)\b", re.I)
_COND = re.compile(r"\b(?:if|unless|when|whenever|until|only|provided|otherwise|in case|maybe|might|probably)\b", re.I)


def stated_choice(text, r, explicit=False):
    """What a message sent before round r's decision says its sender will do in round r: ("vol" | "abs", the words it was read
    from), or (None, None) when it does not say so in a form read here (see above). explicit: only statements that name round
    r itself count (for a message sent in an earlier round, where a statement without a round is about that round)."""
    found = []
    for m in _SAY.finditer(text or ""):
        neg, verb = (m.group(1), m.group(2)) if m.group(2) else ("not", "volunteer")
        if verb in ("abstain", "hold"):
            if neg:
                continue
            lab = "abs"
        else:
            lab = "abs" if neg else "vol"
        rest = text[m.end():]
        cut = _CUT.search(rest)
        win = rest[:cut.start()] if cut else rest
        head = text[:m.start()]
        pre = head[max((x.end() for x in _SENT.finditer(head)), default=0):]
        if _COND.search(win) or _COND.search(pre):
            continue
        ctx = win if (_NUMREF.search(win) or _CUR.search(win) or _OTHER.search(win)) else pre
        if _OTHER.search(ctx) or any(int(x) != r for x in _NUMREF.findall(ctx)):
            continue
        if explicit and not _NUMREF.findall(ctx):
            continue
        found.append((lab, _ws(text[m.start():m.end()] + win)))
    if len({f[0] for f in found}) != 1:
        return None, None
    return found[0]


# ------------------------------------------------------------------------------------------------ end labels
def _nice_ticks(lo, hi, k=4):
    """index.html niceTicks."""
    import math
    span = (hi - lo) or abs(hi) or 1
    p = 10 ** math.floor(math.log10(span / k))
    step = next((m * p for m in (1, 2, 2.5, 5, 10) if span / (m * p) <= k), 10 * p)
    out, x = [], math.ceil(lo / step) * step
    while x <= hi + step * 1e-6:
        out.append(round(x, 8))
        x += step
    return out


def fit_end_labels(lines, bars, n, height, fmt_val, widths=(760, 340)):
    """Set each line's "end" (its name and last value written above its right end) only where the page can draw it clear of
    every line, dot and bar. Mirrors index.html: margins 86 / 16 px, plot top 14 px and bottom 26 px; a chart with "bars" has a
    banded x axis (bars fill 72% of a column) and a y axis from 0 when all values are >= 0, a line-only chart puts round i at
    86 + i * width / (n - 1) and pads its value range by 10% on both sides; labels 7 px above the end point, right-aligned,
    11 px monospace (6.6 px a character), stacked at least 13 px apart; step lines run across, then up or down. Checked at a
    desktop and a phone width; labels that collide are dropped one at a time, benchmark lines (dashed) before the others.
    Returns the dropped names."""
    want = [l for l in lines if l.get("end")]
    for l in lines:
        l.pop("end", None)

    def clashes(on, W):
        iw, cw, ih = W - 102, (W - 102) / n, height - 40
        X = (lambda i: 86 + (i + 0.5) * cw) if bars else (lambda i: 86 + (iw / 2 if n == 1 else i * iw / (n - 1)))
        vals = ([v for l in lines for v in l["v"] if v is not None] + [v for b in bars for v in b["v"] if v is not None]
                + ([0] if bars else []))
        lo, hi = min(vals), max(vals)
        pad = (hi - lo or abs(hi) or 1) * 0.1
        lo, hi = lo - pad, hi + pad
        if bars and min(vals) >= 0:
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
        rects = []
        for j, b in enumerate(bars):                    # grouped bars: 72% of the column, split between the series
            gw = cw * 0.72
            bw = gw / max(1, len(bars))
            for i, v in enumerate(b["v"]):
                if v is not None:
                    x0 = X(i) - gw / 2 + j * bw
                    rects.append((x0, x0 + bw, min(Y(v), Y(max(y0, min(0, y1)))), max(Y(v), Y(max(y0, min(0, y1))))))
        bad = []
        for yy, l, i in ends:
            text = f"{l['name']} {fmt_val(l['v'][i])}"
            bx0, bx1, by0, by1 = W - 16 - 6.6 * len(text) - 2, W - 16 + 1, yy - 10, yy + 2
            inside = lambda x, y, r=0: bx0 - r <= x <= bx1 + r and by0 - r <= y <= by1 + r
            hit = any(not (r[1] < bx0 or r[0] > bx1 or r[3] < by0 or r[2] > by1) for r in rects)
            for m in lines:
                pts = [(X(k), Y(v)) for k, v in enumerate(m["v"]) if v is not None]
                for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
                    segs = [((xa, ya), (xb, ya)), ((xb, ya), (xb, yb))] if m.get("step") else [((xa, ya), (xb, yb))]
                    for (p0, q0), (p1, q1) in segs:
                        hit = hit or any(inside(p0 + (p1 - p0) * j / 40, q0 + (q1 - q0) * j / 40) for j in range(41))
                if m.get("dots"):
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
    rounds, talk, stray = [], [[]], []
    known = {"setup", "rules", "talk", "round_result", "game_end", "game_over"}
    for e in events:
        ev = e.get("event")
        if ev == "round_result":
            vol = list(e.get("volunteered") or [])
            rounds.append({"r": int(e.get("round", len(rounds) + 1)), "vol_raw": vol,
                           "count_logged": int(e["volunteers"]) if e.get("volunteers") is not None else None,
                           "talk": talk[-1]})
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
    n = int(par("n_players", cfg.get("n_players") or len(rounds[0]["vol_raw"])))
    metrics = over.get("_metrics") or var.get("_metrics") or summary.get("_metrics") or {}
    return {"n": n, "R": int(par("rounds", len(rounds))), "tp": int(par("talk_passes", 0) or 0),
            "v": exact(par("v", 10)), "c": exact(par("cost", 2)), "v_logged": par("v") is not None and par("cost") is not None,
            "rounds": rounds, "tail_talk": talk[-1], "end": end, "metrics": metrics,
            "winners": over.get("winners", summary.get("winners"))}


def verify(G, rewards):
    """Re-derive the volunteer counts, payoffs, scores and the logged totals; returns (decisions, scores, warnings). The page
    shows the logged decisions and the payoffs the rule gives for them."""
    n, v, c, warn = G["n"], G["v"], G["c"], []
    dec, score = [], [Fraction(0)] * n
    for k, rd in enumerate(G["rounds"]):
        r, raw = rd["r"], rd["vol_raw"]
        if r != k + 1:
            warn.append(f"round_result {k}: round {r} logged, {k + 1} expected")
        if len(raw) != n:
            warn.append(f"round {r}: {len(raw)} decisions logged for {n} players")
        if any(not isinstance(x, bool) for x in raw):
            warn.append(f"round {r}: non-boolean decisions {raw} (read as volunteering only when true)")
        d = [raw[s] is True if s < len(raw) else False for s in range(n)]
        dec.append(d)
        cnt = sum(d)
        if rd["count_logged"] is not None and rd["count_logged"] != cnt:
            warn.append(f"round {r}: volunteers {rd['count_logged']} logged, {cnt} counted from the decisions")
        for s in range(n):
            score[s] += (v - c if d[s] else v) if cnt else 0
        if G["tp"]:
            want = [s for _ in range(G["tp"]) for s in range(n)]
            got = [t["seat"] for t in rd["talk"]]
            if got != want:
                warn.append(f"round {r}: message senders {got}, {want} expected ({G['tp']} pass(es) in seat order)")
    if G["tail_talk"]:
        warn.append(f"{len(G['tail_talk'])} message(s) after the last round_result (a round without a result)")
    end, m = G["end"], G["metrics"]
    tot = sum(sum(d) for d in dec)
    nov = sum(1 for d in dec if not any(d))
    for src, name in ((end, "game_end"), (m, "_metrics")):
        if src.get("volunteer_count_total") is not None and int(src["volunteer_count_total"]) != tot:
            warn.append(f"{name} volunteer_count_total {src['volunteer_count_total']}, {tot} counted from the round results")
        if src.get("no_volunteer_rounds") is not None and int(src["no_volunteer_rounds"]) != nov:
            warn.append(f"{name} no_volunteer_rounds {src['no_volunteer_rounds']}, {nov} counted from the round results")
    for s in range(n):
        if rewards[s] is not None and abs(rewards[s] - float(score[s])) > 1e-9:
            warn.append(f"reward of seat {s}: {rewards[s]} in the summary, {fn(score[s])} summed from the payoff rule")
    best = max(score)
    lead = [s for s in range(n) if score[s] == best]
    if isinstance(G["winners"], list) and sorted(int(w) for w in G["winners"]) != lead:
        warn.append(f"winners {G['winners']} logged, highest scores: seats {lead}")
    if len(G["rounds"]) < G["R"]:
        warn.append(f"only {len(G['rounds'])} of {G['R']} rounds logged: the recording stops early")
    return dec, score, warn


# ------------------------------------------------------------------------------------------------ model calls
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


def call_meta(c):
    """Phase and round of a call, from its prompt ('Current phase: X', the public state's "round"); None when not shown. A
    prompt that does not show the phase (a format-retry note) is placed by its answer's action_id ("talk" is the TALK phase,
    "volunteer" DECIDE), so a TALK call never joins a decision's window as an earlier attempt."""
    p = c.get("prompt") or ""
    m = re.search(r"Current phase:\s*([A-Za-z_]+)", p)
    phase = m.group(1).upper() if m else None
    if phase is None:
        a = _norm_answer(c.get("answer"))
        phase = {"talk": "TALK", "volunteer": "DECIDE"}.get(a.get("action_id") if isinstance(a, dict) else None)
    i = p.find("Public state:")
    rm = re.search(r'"round":\s*(\d+)', p[i:]) if i >= 0 else None
    return phase, (int(rm.group(1)) if rm else None)


def _matches(call, st, loose=False):
    """Strict: the logged action (the message text up to whitespace; the volunteer flag as a JSON boolean). Loose: the right
    action_id with any payload."""
    a = _norm_answer(call.get("answer"))
    if st["phase"] == "TALK":
        if not isinstance(a, dict):                 # a discussion prompt may be answered with the bare message
            raw = (call.get("answer") or "").strip()
            return bool(raw) and _ws(raw) == _ws(st["text"])
        if a.get("action_id") != "talk":
            return False
        return loose or _ws((a.get("payload") or {}).get("text")) == _ws(st["text"])
    if not isinstance(a, dict) or a.get("action_id") != "volunteer":
        return False
    p = a.get("payload") or {}
    return loose or (isinstance(p, dict) and isinstance(p.get("volunteer"), bool) and p["volunteer"] == st["vol"])


def attach_calls(steps, calls):
    """Tie calls to steps in time order (see the module docstring). Returns (steps with a call, steps whose call's answer is
    not the logged action)."""
    meta = [call_meta(c) for c in calls]
    seats = [c.get("seat") if c.get("seat") is not None else common.seat_of(c.get("prompt")) for c in calls]

    def fits(k, st, strict=False):
        seat, (ph, rnd) = seats[k], meta[k]
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
            if _matches(calls[j - 1], st):             # a decision's calls end at its accepted answer (later calls of the same
                break                                  # seat, phase and round belong to another decision, e.g. a second pass)
        window = list(range(ptr, j))
        if not window:
            continue
        hit = next((k for k in window if _matches(calls[k], st)), None)
        hit_loose = hit if hit is not None else next((k for k in window if _matches(calls[k], st, loose=True)), None)
        use = hit if hit is not None else (hit_loose if hit_loose is not None else window[-1])
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
                parts.append(f"[The log shows volunteer={'true' if st['vol'] else 'false'}, not this answer: {c['answer'][:300]}. "
                             "When no reply is a legal action the game applies a fallback (the DECIDE phase's default is "
                             "volunteer=false).]")
            mism += 1
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        obs = next((k for k in range(use, ptr - 1, -1) if "You are seat" in (calls[k].get("prompt") or "")), None)
        if obs is not None:                            # a retry's prompt can be only the correction note
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


# ------------------------------------------------------------------------------------------------ build
def build(src, calls=None):
    b = load_source(src)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    calls = calls or []
    events, summary, run_meta, cfg = b["events"], b.get("summary") or {}, b.get("run") or {}, b.get("config") or {}
    G = read_game(events, summary, cfg)
    n, Rcfg, TP, V, C = G["n"], G["R"], G["tp"], G["v"], G["c"]
    if not G["v_logged"]:
        print("warning: v or cost is not in the log, the summary or the config; the rules' defaults (10, 2) are used", file=sys.stderr)
    raw_rw = summary.get("rewards") or (run_meta.get("result") or {}).get("rewards") or {}
    rewards = [float(raw_rw[str(s)]) if str(s) in raw_rw else None for s in range(n)]
    dec, score_chk, warns = verify(G, rewards)
    for w in warns:
        print("warning:", w, file=sys.stderr)
    full_model, model, date = model_and_date(b)
    seed = run_meta.get("seed", cfg.get("seed"))
    RD = G["rounds"]
    R = len(RD)                                         # rounds played
    N = max(Rcfg, RD[-1]["r"], R)                       # x axis: every configured round, played or not
    pad = lambda vals: list(vals) + [None] * (N - len(vals))
    P = [pname(s) for s in range(n)]
    nw = numword(n)
    everyone = "both players" if n == 2 else f"all {nw} players"
    VC = V - C                                          # a volunteer's payoff when the good is provided

    # ---- per round numbers
    vols = [[s for s in range(n) if dec[i][s]] for i in range(R)]
    cnt = [len(x) for x in vols]
    prov = [k > 0 for k in cnt]
    pay = [[((VC if dec[i][s] else V) if prov[i] else Fraction(0)) for i in range(R)] for s in range(n)]
    cum = [[] for _ in range(n)]
    for s in range(n):
        acc = Fraction(0)
        for i in range(R):
            acc += pay[s][i]
            cum[s].append(acc)
    score = [cum[s][-1] for s in range(n)]
    assert score == score_chk
    group = [sum(pay[s][i] for s in range(n)) for i in range(R)]
    best_round = n * V - C                              # one volunteer: the most the group can earn in a round
    turns = [sum(1 for i in range(R) if dec[i][s]) for s in range(n)]
    cumt = [[sum(1 for k in range(i + 1) if dec[k][s]) for i in range(R)] for s in range(n)]   # volunteer turns so far
    T = sum(turns)
    D = n * R                                           # decisions
    prov_n = sum(prov)
    assert all(score[s] == V * prov_n - C * turns[s] for s in range(n))   # what separates the players is their turns
    none_rs = [i + 1 for i in range(R) if not prov[i]]
    one_rs = [i + 1 for i in range(R) if cnt[i] == 1]
    multi_rs = [i + 1 for i in range(R) if cnt[i] >= 2]
    extra = sum(k - 1 for k in cnt if k >= 2)           # volunteer costs paid beyond the one needed
    tot = sum(score)
    tot_best = best_round * R
    loss_none = best_round * len(none_rs)               # a round without a volunteer loses the group's best round, n*v - c
    loss_extra = C * extra
    assert tot_best - loss_none - loss_extra == tot
    ME = mixed_equilibrium(n, V, C)
    rate_r = [Fraction(cnt[i], n) for i in range(R)]
    rate_cum = [Fraction(sum(cnt[:i + 1]), n * (i + 1)) for i in range(R)]
    m_rate = G["metrics"].get("mixed_eq_volunteer_rate")
    if ME and m_rate is not None and abs(float(m_rate) - ME["p"]) > 0.0006:
        print(f"warning: _metrics mixed_eq_volunteer_rate {m_rate}, {ME['p']:.4f} computed from v, cost and n", file=sys.stderr)
        warns.append("mixed_eq_volunteer_rate")
    everyone_short = "both" if n == 2 else f"all {nw}"
    LAB = [f"{P[s]} (P{s})" for s in range(n)]          # seat labels: the page's name and the name the messages use

    # ---- said versus did: what each player's own message said it would do this round (see stated_choice)
    said = [[None] * n for _ in range(R)]               # (label "vol" | "abs", the words) per round and seat
    if TP:
        for i, rd in enumerate(RD):
            for t in rd["talk"]:
                if 0 <= t["seat"] < n:
                    lab, words = stated_choice(t["text"], rd["r"])
                    if lab:                             # with several messages a round, the last statement counts
                        said[i][t["seat"]] = (lab, words)
    K = sum(1 for i in range(R) for s in range(n) if said[i][s])

    def broke(i, s):
        return said[i][s] is not None and (said[i][s][0] == "vol") != dec[i][s]
    mm = [(i, s) for i in range(R) for s in range(n) if broke(i, s)]
    mm_rounds = sorted({i for i, _ in mm})

    def who_of(ss):
        return everyone_short if len(ss) == n and n > 1 else players_text(ss, n)

    def sd_words(i):
        """'all three had said they would not volunteer'; '' when every classified message of round i matched."""
        no = [s for s in range(n) if broke(i, s) and dec[i][s]]
        yes = [s for s in range(n) if broke(i, s) and not dec[i][s]]
        pr = lambda ss: "it" if len(ss) == 1 else "they"
        parts = ([f"{who_of(no)} had said {pr(no)} would not volunteer"] if no else []) + \
                ([f"{who_of(yes)} had said {pr(yes)} would volunteer"] if yes else [])
        return " and ".join(parts)

    def turn_of(i):
        """The seat whose turn round i was in the rotation (see below): a round inside it, or the round right after it ends;
        None otherwise (before the rotation starts, or once a round has broken it, there is no agreed turn to name)."""
        if run and run[0] <= i <= run[1] + 1 and i < R:
            return run[2][(i - run[0]) % n]
        return None

    def refused(i):
        """The seat whose turn round i was, when it broke the rotation (the round right after it) and that player's own message
        that round said it would not volunteer; else None."""
        q = turn_of(i)
        if q is not None and i == run[1] + 1 and said[i][q] and said[i][q][0] == "abs":
            return q
        return None

    def promised(q, i):
        """[(earlier round, 1-based; the words)] in which seat q's own message said it would volunteer in round i, naming
        that round."""
        r, out = RD[i]["r"], []
        for k in range(i):
            for t in RD[k]["talk"]:
                lab, words = stated_choice(t["text"], r, explicit=True) if t["seat"] == q else (None, None)
                if lab == "vol":
                    out.append((RD[k]["r"], words))
                    break
        return out

    def promise_note(i, told=True):
        """For a round that broke the rotation after the player whose turn it was said it would not volunteer: ' In rounds 7
        to 9 Player 1 had said it would volunteer in round 10 (“...”).'; '' when it had not said so. told=False (the text
        before it does not say whose turn it was): a sentence that says so itself, ' Round 10 was Player 1's turn: in rounds
        7 to 9 it had said it would volunteer in round 10 (“...”), and in round 10 it said it would not (“...”).'"""
        q = refused(i)
        if q is None:
            return ""
        prom, r = promised(q, i), RD[i]["r"]
        if told:
            return (f" In {rounds_text([k for k, _ in prom])} {pname(q)} had said it would volunteer in round {r} "
                    f"({LDQ}{prom[-1][1]}{RDQ})." if prom else "")
        now = f"{LDQ}{said[i][q][1]}{RDQ}"
        if not prom:
            return f" Round {r} was {pname(q)}'s turn, and {pname(q)} had said it would not volunteer ({now})."
        return (f" Round {r} was {pname(q)}'s turn: in {rounds_text([k for k, _ in prom])} it had said it would volunteer in "
                f"round {r} ({LDQ}{prom[-1][1]}{RDQ}), and in round {r} it said it would not ({now}).")

    def turn_note(i):
        """'it was Player 2's turn' when a single player of round i, inside the rotation, said it would volunteer and did not
        while the player whose turn it was did, or when the player whose turn it was volunteered although it had said it would
        not; 'it was Player 1's turn, and Player 1 had said it would not volunteer' for a round that broke the rotation after
        the player whose turn it was said so; '' otherwise."""
        q = turn_of(i)
        if q is None:
            return ""
        if i <= run[1]:
            yes = [s for s in range(n) if broke(i, s) and not dec[i][s]]
            no = [s for s in range(n) if broke(i, s) and dec[i][s]]
            return f"it was {pname(q)}'s turn" if (len(yes) == 1 and yes[0] != q) or no == [q] else ""
        if refused(i) is not None:
            return f"it was {pname(q)}'s turn, and {pname(q)} had said it would not volunteer"
        return ""

    def times(k):
        return "once" if k == 1 else "twice" if k == 2 else f"{numword(k)} times"

    def sd_clause(i):
        """'Player 1 does not volunteer although it had said it would' (present tense, for the title)."""
        no = [s for s in range(n) if broke(i, s) and dec[i][s]]
        yes = [s for s in range(n) if broke(i, s) and not dec[i][s]]
        pr = lambda ss: "it" if len(ss) == 1 else "they"
        parts = ([f"{who_of(no)} {'volunteers' if len(no) == 1 else 'volunteer'} although {pr(no)} had said {pr(no)} would not"]
                 if no else []) + \
                ([f"{who_of(yes)} {'does' if len(yes) == 1 else 'do'} not volunteer although {pr(yes)} had said {pr(yes)} would"]
                 if yes else [])
        tn = turn_note(i)
        return " and ".join(parts) + (f" ({tn})" if tn else "")

    # ---- the pattern: the longest rotation (one volunteer a round, a fixed order covering every seat) and what breaks it
    run = rotation_run(vols, n)
    if run and run[1] - run[0] + 1 < max(n, (R + 1) // 2):
        run = None                                      # too short to lead the story

    def is_last(i):
        return i + 1 == R and R == Rcfg

    def where(rs, lead=False):
        """1-based rounds -> 'every round', 'round 10, the last' (lead: 'round 10, the last,'), 'rounds 1 to 9'."""
        rs = sorted(rs)
        if len(rs) == R and R > 1:
            return "every round"
        if len(rs) == 1 and is_last(rs[0] - 1):
            return f"round {rs[0]}, the last" + ("," if lead else "")
        if len(rs) > 1 and is_last(rs[-1] - 1) and rs == list(range(rs[0], rs[-1] + 1)):
            return f"{rounds_text(rs)}, the last {numword(len(rs))}" + ("," if lead else "")
        return rounds_text(rs)

    def order_words(cyc):
        if all(cyc[(k + 1) % n] == (cyc[k] + 1) % n for k in range(n)):
            return "in seat order" + ("" if cyc[0] == 0 else f", starting with {pname(cyc[0])}")
        return "in the order " + ", ".join(pname(q) for q in cyc)

    def round_event(i, expect=None, then=False):
        k, th = cnt[i], (" then" if then else "")
        if k == 0:
            return ("then " if then else "") + "nobody volunteers and everyone gets 0"
        if k == 1:
            q = vols[i][0]
            return f"{pname(q)}{th} volunteers alone" + (f" instead of {pname(expect)}" if expect is not None and expect != q else "")
        return (f"{who_of(vols[i])}{th} volunteer, paying {plural(k - 1, 'cost', word=True)} more than needed")

    told = set()                                        # refusal rounds (see refused) the rotation's story names

    def exceptions(idx, expect_of):
        """Rounds that break the rotation, grouped when consecutive rounds have the same outcome; (text, rounds whose
        said-versus-did breaks the text already names)."""
        groups = []
        for i in idx:
            key = (tuple(vols[i]), expect_of(i) if cnt[i] == 1 else None, refused(i))
            if groups and groups[-1][-1] == i - 1 and groups[-1][0][0] == key:
                groups[-1][1].append(i)
                groups[-1][-1] = i
            else:
                groups.append([(key,), [i], i])
        if len(groups) > 3:
            bits = []
            for k_lo, k_hi, word in ((0, 0, "no volunteer"), (1, 1, "one volunteer"), (2, n, "more than one")):
                c = sum(1 for i in idx if k_lo <= cnt[i] <= k_hi)
                if c:
                    bits.append(f"{numword(c)} with {word}")
            lead = "the first" if idx[-1] < run[0] else "the remaining"
            return f"in {lead} {plural(len(idx), 'round', word=True)}: " + join_and(bits), set()
        parts, covered = [], set()
        for _, g, _ in groups:
            i0 = g[0]
            q = refused(i0)                             # the player whose turn it was said it would not take it
            ev = (round_event(i0, expect_of(i0)) if q is None else
                  f"{pname(q)}, whose turn it was, says it will not volunteer; " + round_event(i0, then=True))
            if q is not None:
                told.add(i0)
            if len(g) > 1 and cnt[i0] >= 2:
                ev += " each round"
            if len(g) == 1 and i0 in mm_rounds:
                ev += f", although {sd_words(i0)}"
                covered.add(i0)
            parts.append(f"in {where([i + 1 for i in g], lead=True)} {ev}")
        return "; ".join(parts), covered

    def multi_clause():
        """The rounds with more than one volunteer, as a clause; (text, covered said-versus-did rounds)."""
        if len(multi_rs) == 1:
            i = multi_rs[0] - 1
            t = (f"{who_of(vols[i])} volunteer in {where(multi_rs)}" if cnt[i] == n else
                 f"{where(multi_rs)} has {numword(cnt[i])} volunteers")
            t += f" ({plural(cnt[i] - 1, 'cost', word=True)} paid beyond the one needed)"
            if i in mm_rounds:
                return t + f", although {sd_words(i)}", {i}
            return t, set()
        if all(cnt[r - 1] == n for r in multi_rs):
            t = f"{everyone} volunteer in {where(multi_rs)}"
        elif len(multi_rs) == R:
            t = "every round has more than one volunteer"
        else:
            t = f"{len(multi_rs)} of the {R} rounds have more than one volunteer"
        return t + f" ({plural(extra, 'cost', word=True)} paid beyond the one needed)", set()

    def count_bits():
        """The rounds by number of volunteers, for tile 2 and the grid note."""
        bits = []
        if one_rs:
            bits.append(f"one volunteer in {where(one_rs)}"
                        + (f", taking turns {order_words(run[2])}"
                           if run and one_rs == list(range(run[0] + 1, run[1] + 2)) else ""))
        byk = {}
        for r in multi_rs:
            byk.setdefault(cnt[r - 1], []).append(r)
        groups = sorted(byk.items()) if len(byk) <= 2 else [(None, multi_rs)]
        for k, rs in groups:
            costs = sum(cnt[r - 1] - 1 for r in rs)
            if k is None:
                lo_k, hi_k = min(byk), max(byk)
                head = f"{numword(lo_k)} to {numword(hi_k)} volunteers"
            else:
                head = f"{everyone} volunteer" if k == n else f"{numword(k)} volunteers"
            bits.append(f"{head} in {where(rs)} ({plural(costs, 'cost', word=True)} more than needed)")
        if none_rs:
            bits.append("nobody volunteers in any round" if len(none_rs) == R else f"no volunteer in {where(none_rs)}")
        return cap("; ".join(bits)) + "."

    # ---- steps: per round the messages (logged order), then the decisions in seat order
    steps, idx_last = [], {}

    def talk_step(t, i, r):
        return {"seat": t["seat"], "x": i, "phase": "TALK", "round": r, "text": t["text"], "_talk": t,
                "head": f"Round {r} {DOT} public message", "kind": "talk",
                "act": json.dumps({"action_id": "talk", "payload": {"text": t["text"]}}, ensure_ascii=False)}
    for i, rd in enumerate(RD):
        r = rd["r"]
        for t in rd["talk"]:
            if not 0 <= t["seat"] < n:
                print(f"warning: round {r}: message from seat {t['seat']} ignored", file=sys.stderr)
                continue
            t["step"] = len(steps)
            steps.append(talk_step(t, i, r))
        for s in range(n):
            mine = dec[i][s]
            others = [q for q in vols[i] if q != s]
            if mine:
                what = ("the only volunteer" if cnt[i] == 1 else
                        f"one of {cnt[i]} volunteers")
                head = f"Round {r} {DOT} volunteers {DOT} {what}, +{fn(pay[s][i])}"
            elif prov[i]:
                who = pname(others[0]) if len(others) == 1 else f"{len(others)} others"
                head = f"Round {r} {DOT} does not volunteer {DOT} {who} volunteered, +{fn(pay[s][i])}"
            else:
                head = f"Round {r} {DOT} does not volunteer {DOT} nobody volunteered, +0"
            if broke(i, s):
                head += f" {DOT} had said it would" + (" not" if mine else "")
            steps.append({"seat": s, "x": i, "phase": "DECIDE", "round": r, "vol": mine,
                          "kind": "volunteer" if mine else "abstain", "head": head,
                          "act": json.dumps({"action_id": "volunteer", "payload": {"volunteer": mine}})})
        idx_last[i] = len(steps) - 1
    tail = []                                           # messages of a round the recording did not finish
    if G["tail_talk"] and R < N:
        for t in G["tail_talk"]:
            if 0 <= t["seat"] < n:
                t["step"] = len(steps)
                steps.append(talk_step(t, R, R + 1))
                tail.append(len(steps) - 1)

    # ---- what each seat saw (rebuilt from the log; the saved prompt where there is one)
    def round_line(k):
        return (f"- Round {RD[k]['r']}: volunteered " + ("[" + ", ".join(f"P{q}" for q in vols[k]) + "]" if vols[k] else "[nobody]")
                + f"; payoffs [{', '.join(fn(pay[q][k]) for q in range(n))}].")

    def observation(st):
        s, r, i = st["seat"], st["round"], st["x"]
        sc = [cum[q][i - 1] if i > 0 else 0 for q in range(n)]
        lines = ["(Rebuilt from the game log; the exact prompt for this turn was not saved.)",
                 f"You are seat P{s} ({P[s]} on this page). Your role: PLAYER.",
                 (f"Current phase: TALK, round {r} of {Rcfg}. Each player sends {plural(TP, 'public, non-binding message', word=True)} "
                  "before the decision." if st["phase"] == "TALK" else
                  f"Current phase: DECIDE, round {r} of {Rcfg}. Every player secretly decides whether to volunteer; the decisions are "
                  "revealed together after the last player has decided."),
                 f"Public state: v {fn(V)}, cost {fn(C)}, {n} players. If at least one player volunteers, each volunteer gets "
                 f"v {MINUS} cost = {fn(VC)} and each other player v = {fn(V)}; if nobody volunteers, everyone gets 0.",
                 "Scores so far: " + ", ".join(f"P{q} {fn(sc[q])}" for q in range(n)) + "."]
        hist = [round_line(k) for k in range(min(i, R))]
        lines += (["Earlier rounds:"] + hist) if hist else ["Earlier rounds: none yet."]
        msgs = RD[i]["talk"] if i < R else G["tail_talk"]
        if st["phase"] == "TALK":
            said_so_far = []
            for t in msgs:
                if t is st["_talk"]:
                    break
                said_so_far.append(t)
            lines += (["Messages this round so far:"] + [f"- P{t['seat']}: {t['text']}" for t in said_so_far]
                      if said_so_far else ["No one has spoken yet this round."])
        elif TP:
            lines += (["Messages this round:"] + [f"- P{t['seat']}: {t['text']}" for t in msgs]) if msgs else ["No messages this round."]
        return en("\n".join(lines))

    for st in steps:
        st["state"] = observation(st)
    prompt = run_meta.get("prompt") or {}
    for st in steps:
        if (prompt.get("text") and prompt.get("seat") == st["seat"] and str(prompt.get("phase", "")).upper() == st["phase"]
                and re.search(r'"round": %d\b' % st["round"], prompt["text"])):
            st["state"] = en(common.state_part(prompt["text"]))      # the prompt saved with the run (the first decision)
            st["state_from_prompt"] = True
            break

    tied, mism = attach_calls(steps, calls) if calls else (0, 0)
    has_reason = any(st.get("thinking") for st in steps)
    from_prompt = sum(1 for st in steps if st.get("state_from_prompt"))
    for st in steps:
        st.setdefault("thinking", "")
        for k in ("phase", "round", "text", "_talk", "vol", "state_from_prompt"):
            st.pop(k, None)

    # ---- words used in several places
    eq_p = pct(ME["p"]) if ME else None
    one_rate = pct(1, n)
    pay_rule = (f"if at least one player volunteers, each volunteer gets {fn(VC)} (v {fn(V)} {MINUS} cost {fn(C)}) and every other "
                f"player {fn(V)}; if nobody volunteers, everyone gets 0")
    if n <= 4:
        name_note = (f"In their messages the players call each other {join_and(f'P{s}' for s in range(n))} "
                     f"({'Players ' + join_and(str(s + 1) for s in range(n))} on this page).")
    else:
        name_note = f"In their messages the players call each other P0 to P{n - 1} (Players 1 to {n} on this page)."

    def vol_words(i):
        """'Player 1 volunteered; Players 2 and 3 did not'."""
        ab = [s for s in range(n) if not dec[i][s]]
        if not vols[i]:
            return "nobody volunteered"
        if not ab:
            return f"{everyone} volunteered"
        return (f"{players_text(vols[i], n)} volunteered; {players_text(ab, n)} did not")

    # ---- feed: the rules, the messages, each round's reveal after its last decision
    feed = [{"x": 0, "step": 0, "who": "Rules",
             "text": f"{cap(nw)} players, {plural(Rcfg, 'round')}, v {fn(V)}, cost {fn(C)}. Each round every player secretly decides "
                     f"whether to volunteer: {pay_rule}. "
                     + (f"Before each decision every player sends {plural(TP, 'public, non-binding message', word=True)}." if TP else
                        "The players cannot send messages.")}]
    for i, rd in enumerate(RD):
        for t in rd["talk"]:
            if "step" in t:
                feed.append({"x": i, "step": t["step"], "who": LAB[t["seat"]], "kind": "talk", "text": LDQ + t["text"] + RDQ})
        if not prov[i]:
            res = "The good was not provided: everyone gets 0."
        elif cnt[i] == 1:
            res = f"Good provided at the lowest cost: {pname(vols[i][0])} gets {fn(VC)}, " + ("the other" if n == 2 else "the others") + f" {fn(V)}."
        else:
            res = (f"Good provided, with {plural(cnt[i] - 1, 'cost', word=True)} more than needed: volunteers get {fn(VC)}"
                   + (f", the others {fn(V)}." if cnt[i] < n else "."))
        sdr, tn, q = "", turn_note(i), refused(i)
        if i in mm_rounds:
            bs = [s for s in range(n) if broke(i, s)]
            if q is not None and q in bs:               # its own words are quoted here already
                tn = f"it was {pname(q)}'s turn"
            quotes = ([f"{LDQ}{said[i][bs[0]][1]}{RDQ}"] if len(bs) == 1 else
                      [f"{pname(s)}: {LDQ}{said[i][s][1]}{RDQ}" for s in bs])
            sdr = f" {cap(sd_words(i))} ({'; '.join(quotes)})" + (f"; {tn}." if tn else ".")
        elif tn:
            sdr = f" {cap(tn)}" + (f" ({LDQ}{said[i][q][1]}{RDQ})" if q is not None else "") + "."
        sdr += promise_note(i)
        feed.append({"x": i, "step": idx_last[i], "who": "Reveal",
                     "text": f"Round {rd['r']}: {vol_words(i)}.{sdr} {res} Group payoff {fn(group[i])} of {fn(best_round)} possible. "
                             "Scores " + " / ".join(fn(cum[s][i]) for s in range(n)) + "."})
    for k in tail:
        st = steps[k]
        feed.append({"x": st["x"], "step": k, "who": LAB[st["seat"]], "kind": "talk",
                     "text": LDQ + json.loads(st["act"])["payload"]["text"] + RDQ})

    # ---- the story: title and the grid's pattern note share these words
    Rw = numword(R)
    seq = [v[0] for v in vols] if all(k == 1 for k in cnt) else None
    vs_eq = f"the volunteering rate is {pct(T, D)}, against {eq_p} in the mixed equilibrium" if ME else f"the volunteering rate is {pct(T, D)}"
    covered = set()
    pattern = None                                      # the rotation and its exceptions (shared by the title and the grid note)
    if run and run[1] - run[0] + 1 == R:
        cyc = run[2]
        title = (f"The {nw} players take turns volunteering {order_words(cyc)}, one volunteer in each of the {Rw} rounds, so the "
                 f"good is always provided at the lowest cost and the group earns the most possible")
        pattern = (f"Exactly one player volunteered in every round, in turn: {' then '.join(pname(q) for q in cyc)}, then again"
                   + (" from the start." if R > n else "."))
    elif run:
        ra, rb, cyc = run
        expect_of = lambda i: cyc[(i - ra) % n]
        bt, bc = exceptions(list(range(ra)), expect_of) if ra > 0 else ("", set())
        at, ac = exceptions(list(range(rb + 1, R)), expect_of) if rb + 1 < R else ("", set())
        covered = bc | ac
        span = rounds_text(range(ra + 1, rb + 2))

        def story(rot):
            if not bt:
                return f"In {span} {rot}; " + at
            if not at:
                return cap(bt) + f"; from round {ra + 1} on, {rot}"
            return cap(bt) + f"; in {span} {rot}; " + at
        title = story(f"the {nw} players take turns volunteering {order_words(cyc)}")
        seat_order = order_words(cyc).startswith("in seat order")
        pattern = story(f"the {nw} players take turns volunteering "
                        + (f"in seat order ({', '.join(pname(q) for q in cyc)}, then again)" if seat_order else
                           f"in the order {', '.join(pname(q) for q in cyc)}, then again")) + "."
    elif seq and len(set(seq)) == 1:
        q = seq[0]
        title = (f"{pname(q)} volunteers alone in all {Rw} rounds: the good is always provided, {pname(q)} pays every cost and "
                 + ("the other player collects" if n == 2 else "the others collect") + f" {fn(V)} a round")
    elif seq:
        title = (f"Exactly one player volunteers in each of the {Rw} rounds, so the good is always provided at the lowest cost; "
                 "volunteer turns by player " + " / ".join(str(x) for x in turns))
    elif prov_n == 0:
        title = f"Nobody volunteers in any of the {Rw} rounds: the good is never provided and every player scores 0"
    elif not none_rs:
        mc, covered = multi_clause()
        title = f"The good is provided in every round, but {mc}; {vs_eq}"
    else:
        title = (f"Nobody volunteers in {len(none_rs)} of the {R} rounds ({rounds_text(none_rs)}), where everyone gets 0; {vs_eq}")
        if multi_rs:
            mc, covered = multi_clause()
            title += f"; {mc}"
    rest_mm = [i for i in mm_rounds if i not in covered]
    if rest_mm:
        title += ("; " + "; ".join(f"in round {i + 1} {sd_clause(i)}" for i in rest_mm) if len(rest_mm) <= 2 else
                  f"; in {plural(len(rest_mm), 'round', word=True)} ({rounds_text([i + 1 for i in rest_mm])}) a player does not "
                  "do what its own message said")
    if R < Rcfg:
        title += f" (the recording stops after round {R} of {Rcfg})"

    if TP and K:
        sd_head = (f"Said versus did: before {K} of the {D} decisions the player's own message said what it would do that round "
                   f"({LDQ}I will volunteer{RDQ}, {LDQ}I'll hold{RDQ}, {LDQ}I will not volunteer in round 4{RDQ} and the like)")
        if not mm:
            sd_note = sd_head + f", and in all {K} it did as it said."
        elif pattern and all(i in covered for i in mm_rounds):
            sd_note = sd_head + f"; in {numword(len(mm))} it did not ({rounds_text([i + 1 for i in mm_rounds])}, above)."
        else:
            sd_note = (sd_head + f"; in {numword(len(mm))} it did not: "
                       + "; ".join(f"in round {i + 1} {sd_words(i)}" + (f" ({turn_note(i)})" if turn_note(i) else "")
                                   for i in mm_rounds[:4])
                       + ("; and others" if len(mm_rounds) > 4 else "") + ".")
    else:
        sd_note = ""

    # ---- chart 1: who volunteered (grid). Volunteer cells are blue with V; others gray and blank; the bottom row counts the
    # volunteers: green for exactly one, yellow for more than needed, orange (0) when nobody volunteered. The hover box shows a
    # whole column on one line per row (the page does not wrap it), so each row is kept short: the said-versus-did note sits in
    # the player's own row only.
    rows = LAB + ["Volunteers"]
    gcells = [[None] * N for _ in rows]
    for i in range(R):
        for s in range(n):
            sd = ""
            if said[i][s]:
                sd = (f"; had said {LDQ}{said[i][s][1]}{RDQ}" if broke(i, s) else "; did as it said")
            if dec[i][s]:
                tip = ("volunteered alone" if cnt[i] == 1 else f"volunteered (one of {cnt[i]})")
                gcells[s][i] = {"c": VOL_C, "t": "V", "tip": f"{tip}: +{fn(pay[s][i])}; score {fn(cum[s][i])}{sd}"}
            else:
                tip = (f"did not volunteer ({players_text(vols[i], n)} did)" if prov[i] else "nobody volunteered")
                gcells[s][i] = {"c": ABS_C, "o": .45, "tip": f"{tip}: +{fn(pay[s][i])}; score {fn(cum[s][i])}{sd}"}
        if not prov[i]:
            gcells[n][i] = {"c": NOBODY_C, "t": "0", "tip": f"nobody, so everyone gets 0; group payoff 0 of {fn(best_round)}"}
        else:
            gcells[n][i] = {"c": PROVIDED_C if cnt[i] == 1 else MULTI_C, "t": str(cnt[i]),
                            "tip": (f"{plural(cnt[i], 'volunteer')}, " + ("the lowest cost" if cnt[i] == 1 else
                                                                         f"{plural(cnt[i] - 1, 'cost')} more than needed")
                                    + f"; group payoff {fn(group[i])} of {fn(best_round)}")}
    for i in range(R, N):
        for q in range(len(rows)):
            gcells[q][i] = {"c": "muted", "t": "", "o": .25, "tip": "not played: the recording stops before this round"}
    legend = [["Volunteered (V)", VOL_C], ["Did not volunteer", ABS_C]]
    legend += [["One volunteer: good provided at the lowest cost", PROVIDED_C]] if one_rs else []
    legend += [["Good provided, more volunteers than needed", MULTI_C]] if multi_rs else []
    legend += [["Nobody volunteered (0)", NOBODY_C]] if none_rs else []
    if pattern:
        gnote = pattern + (promise_note(run[1] + 1, told=run[1] + 1 in told) if run and run[1] + 1 < R else "")
    elif seq:
        solo = sorted(set(seq))
        gnote = ("Exactly one player volunteered in every round: " + (f"{pname(solo[0])} every time." if len(solo) == 1 else
                 "volunteer turns " + ", ".join(f"{P[s]} {turns[s]}" for s in range(n)) + "."))
    else:
        gnote = count_bits()
    colors = (["green for exactly one (1 is enough: everyone gets the benefit for a single cost)"] if one_rs else []) + \
             (["yellow for more than one (the good provided, with costs paid beyond the one needed)"] if multi_rs else []) + \
             (["orange with 0 when nobody volunteered and everyone got 0"] if none_rs else [])
    sub1 = ("One column per round. A player's cell is blue with V when it volunteered and gray when it did not; decisions were "
            "secret until every player had decided. The bottom row counts the volunteers: " + join_and(colors) + ". "
            + f"{gnote} " + (f"{sd_note} " if sd_note else "") + (f"{name_note} " if TP else "")
            + "Hover a column for payoffs, scores" + (" and what each player had said it would do." if sd_note else " and the group payoff."))
    charts = [{"title": "Who volunteered, round by round", "sub": sub1, "height": 30 * len(rows) + 30,
               "grid": {"rows": rows, "cells": gcells, "rowHeight": 28 if n <= 6 else 22, "legend": legend}, "bars": []}]

    # ---- chart 2: volunteering rate against the mixed equilibrium and one volunteer a round
    # The benchmarks are listed (and drawn) first, so the running rate is drawn over a dashed line it runs along.
    lines2 = []
    if ME:
        lines2.append({"name": "Mixed equilibrium", "color": EQ_C, "dash": True, "end": True, "v": [round(ME["p"] * 100, 4)] * N})
    lines2.append({"name": "One volunteer a round", "color": ONE_C, "dash": True, "end": True, "v": [round(100 / n, 4)] * N})
    lines2.append({"name": "Running rate", "color": "ink", "dots": True, "end": True, "v": pad([round(float(x) * 100, 4) for x in rate_cum])})
    bars2 = [{"name": "Share volunteering this round", "color": VOL_C, "v": pad([round(float(x) * 100, 4) for x in rate_r])}]
    fit_end_labels(lines2, bars2, N, 220, lambda x: f"{x:,.1f}%")
    if all(k == 1 for k in cnt):
        rnote = (f"In every round exactly one of the {nw} players volunteered, so the bars, the running rate and the one-volunteer "
                 f"line coincide at {one_rate}.")
    else:
        rr = runs([float(x) for x in rate_r])
        rnote = (f"The rate was {pct(rate_r[0])} in round 1 and {pct(rate_r[-1])} in round {R}; over the game {T} of {D} decisions "
                 f"were to volunteer ({pct(T, D)})." + ("" if len(rr) > 1 else " It did not change from round to round."))
        on_one = next((k for k in range(R) if rate_cum[k] != Fraction(1, n)), R)   # leading rounds on the one-volunteer line
        if max(2, (R + 1) // 2) <= on_one < R:
            k = on_one
            rnote += (f" Through round {k} " + ("the bars and the running rate lie" if all(c == 1 for c in cnt[:k]) else
                                                 "the running rate lies")
                      + f" on the one-volunteer line ({one_rate}); round {k + 1} ({plural(cnt[k], 'volunteer')}) "
                      + ("lifts" if rate_cum[k] > Fraction(1, n) else "lowers") + f" the running rate to {pct(rate_cum[k])}"
                      + (f", and it ends at {pct(rate_cum[-1])}." if k + 1 < R else "."))
    sub2 = ("Bars: the share of players who volunteered in each round. Solid line: the share of all decisions so far that were to "
            "volunteer. Dashed: " + (f"{eq_p}, the symmetric mixed equilibrium, in which each player volunteers at random with "
                                     f"probability 1 {MINUS} (cost/v)^(1/(n{MINUS}1)) = 1 {MINUS} ({fn(C)}/{fn(V)})^(1/{n - 1}); and "
                                     if ME else "")
            + f"{one_rate}, one volunteer a round (1 of {n}): every pure equilibrium has exactly one volunteer, and it gives the "
              f"group the most. " + rnote)
    charts.append({"title": "Volunteering rate against the mixed equilibrium", "sub": sub2, "height": 220, "fmt": "pct0",
                   "bars": bars2, "lines": lines2})

    # ---- chart 3: volunteer turns so far, one step line per player, against an even split of one volunteer a round. The
    # players' scores differ only through these turns (score = v x rounds with the good provided - cost x turns).
    fair_name = f"Fair share (rounds/{n})"
    lines3 = [{"name": fair_name, "color": "ink", "dash": True, "thin": True, "end": True,
               "v": [round((i + 1) / n, 4) for i in range(N)]}]
    tgroups = {}
    for s in range(n):
        tgroups.setdefault(tuple(cumt[s]), []).append(s)
    if n <= len(PLAYER_COLORS):
        for path, seats in sorted(tgroups.items(), key=lambda kv: kv[1][0]):
            nm = LAB[seats[0]] if len(seats) == 1 else cap(players_text(seats, n)) + " (identical)"
            lines3.append({"name": nm, "color": PLAYER_COLORS[seats[0]], "step": True, "dots": True, "end": True, "v": pad(path)})
    else:
        lines3 += [{"name": "Most turns", "color": "c1", "step": True, "dots": True, "end": True,
                    "v": pad([max(cumt[s][i] for s in range(n)) for i in range(R)])},
                   {"name": "Fewest turns", "color": "c2", "step": True, "dots": True, "end": True,
                    "v": pad([min(cumt[s][i] for s in range(n)) for i in range(R)])}]
    dropped = fit_end_labels(lines3, [], N, 230, lambda x: f"{x:,.2f}")
    players3 = lines3[1:]
    if dropped and not all(l.get("end") for l in players3):   # label every player line or none (the legend names them)
        for l in players3:
            l.pop("end", None)
    lo_t, hi_t = min(turns), max(turns)
    fair_end = Fraction(R, n)
    if n <= 6:
        tlist = ", ".join(f"{P[s]} {plural(turns[s], 'turn')} ({fn(score[s])})" for s in range(n))
        tnote = f"Turns and final scores: {tlist}."
    else:
        tnote = f"Turns range from {lo_t} to {hi_t}; final scores from {fn(min(score))} to {fn(max(score))}."
    if lo_t == hi_t:
        tnote += " Every player took the same number of turns, so the costs were shared evenly."
    else:
        tnote += (f" The fair share after {plural(R, 'round')} is {fn(fair_end)}; the most and fewest turns differ by "
                  f"{plural(hi_t - lo_t, 'turn')}, {fn(C * (hi_t - lo_t))} in score.")
    sub3 = (f"Each player's volunteer turns so far: the step rises in each round the player volunteered. Dashed: the fair share, "
            f"one volunteer a round split evenly among the {nw} players (rounds played / {n}). Each turn costs its player {fn(C)}, "
            f"and this is the only thing that separates the scores: a player's score is {fn(V)} {TIMES} the rounds in which the good "
            f"was provided ({prov_n}) {MINUS} {fn(C)} {TIMES} its turns. " + tnote)
    charts.append({"title": "Volunteer turns so far", "sub": sub3, "height": 230, "fmt": "price", "bars": [], "lines": lines3})

    # ---- tiles
    t1s = f"{T} of {D} decisions to volunteer"
    t1s += (f". Mixed equilibrium {eq_p} (each player volunteering at random); one volunteer a round is {one_rate}." if ME else
            f". One volunteer a round is {one_rate}.")
    t1 = {"k": "Volunteering rate", "v": pct(T, D), "s": t1s}
    if len(set(cnt)) > 1:                               # the number of volunteers each round (no line when it never changes)
        t1["spark"] = list(cnt)
        t1["s"] += " The line: volunteers each round."
    t2s = count_bits()
    if ME:
        t2s += (f" In the mixed equilibrium nobody volunteers in {pct(ME['none'])} of rounds "
                f"({float(ME['none']) * R:.2f} of {R} expected).")
    t2 = {"k": "Good provided", "v": f"{prov_n} of {R} rounds", "s": t2s}
    lo_s, hi_s = min(score), max(score)
    win = [s for s in range(n) if score[s] == hi_s]
    if lo_s == hi_s:
        s3 = f"{cap(everyone)} tie" + (", as the log's winners list says." if isinstance(G["winners"], list) and len(G["winners"]) == n else ".")
    else:
        s3 = (f"{players_text(win, n)} " + ("wins" if len(win) == 1 else "tie for the lead")
              + (", as the log's winners list says." if isinstance(G["winners"], list)
                 and sorted(int(w) for w in G["winners"]) == win else "."))
    if n <= 6:
        s3 += " Volunteer turns: " + ", ".join(f"{P[s]} {turns[s]}" for s in range(n)) + f"; each turn costs {fn(C)}."
    s3 += (f" For comparison: never volunteering while someone else always does pays {fn(V)} a round ({fn(V * R)} over "
           f"{plural(R, 'round')}); " + (f"the mixed equilibrium expects {fn(VC)} a round ({fn(VC * R)})." if ME else
                                         f"volunteering every round pays {fn(VC)} a round ({fn(VC * R)})."))
    t3 = {"k": "Final scores", "v": (" / ".join(fn(x) for x in score) if n <= 4 else
                                     (f"{fn(lo_s)} each" if lo_s == hi_s else f"{fn(lo_s)} to {fn(hi_s)}")), "s": s3}
    t4s = f"{pct(tot, tot_best)} of the most the group can earn ({fn(best_round)} a round, with one volunteer)"
    if ME:
        t4s += f"; mixed-equilibrium play expects {fn(n * ME['pay'] * R)}"
    t4s += "." + ("" if not (loss_none or loss_extra) else
                  " Lost: " + join_and(([f"{fn(loss_none)} to rounds with no volunteer"] if loss_none else [])
                                       + ([f"{fn(loss_extra)} to extra volunteers"] if loss_extra else [])) + ".")
    t4 = {"k": "Group payoff", "v": f"{fn(tot)} of {fn(tot_best)}", "s": t4s}
    tiles = [t1, t2, t3, t4]

    # ---- lede, foot
    lede = (f"{cap(nw)} players play {numword(Rcfg)} rounds of a volunteer's dilemma: each round every player secretly decides whether "
            f"to volunteer, and a player's score is the sum of its payoffs. If at least one player volunteers, everyone gets the "
            f"benefit of {fn(V)} and each volunteer pays a cost of {fn(C)}, so a volunteer nets {fn(VC)} and every other player "
            f"{fn(V)}: each would rather someone else paid, but if nobody volunteers, everyone gets 0. "
            + (f"Before each decision every player sends {'one public, non-binding message' if TP == 1 else f'{TP} public, non-binding messages'}."
               if TP else "The players cannot talk; they only see the decisions revealed after each round."))

    runlog = b.get("runlog") or ""
    fallbacks = len(re.findall(r"\[LLM fallback\]", runlog))
    parse_fail = runlog.count("Could not parse a legal JSON action")
    if fallbacks or parse_fail:
        fb = (f"The recorder's run log reports {plural(max(fallbacks, 1 if parse_fail else 0), 'decision')} where no legal action "
              "could be parsed from the model's answer and a random legal action was played instead"
              + ("; the matched calls show which. " if tied else "; it does not say which decision. "))
    else:
        fb = "The recorder's run log reports no fallback action. " if runlog else ""
    cfg_bits = ", ".join(x for x in [f"seed {seed}" if seed is not None else "", plural(n, "player", word=True),
                                     plural(Rcfg, "round"), f"v {fn(V)}", f"cost {fn(C)}",
                                     (f"{plural(TP, 'public message', word=True)} per player before each decision" if TP else "no messages")]
                         if x)
    if has_reason:
        rs = (f"Observations and reasoning come from the saved model calls ({tied} of {len(steps)} steps matched to a call"
              + (f"; in {mism} the matched call's answer is not the logged action, which the step notes" if mism else "") + "). ")
    else:
        rs = (f"Reasoning was not saved in this recording (it has no call log), so no step shows {model}'s reasoning. " if not calls else
              "The call log has no reasoning, so no step shows the model's reasoning. ")
    obs = ("" if from_prompt == len(steps) else
           ("The observation of the first decision is the prompt saved with the run; the others are rebuilt from the log. "
            if from_prompt == 1 and not tied else
            ("The observations are rebuilt from the log. " if not from_prompt else
             f"{from_prompt} of {len(steps)} observations are saved prompts; the others are rebuilt from the log. ")))
    if ME:
        theory = (f"Benchmarks, computed from the logged v, cost and number of players (rules: Theory Benchmark): in the symmetric "
                  f"mixed equilibrium each player volunteers with probability 1 {MINUS} (cost/v)^(1/(n{MINUS}1)) = {ME['p']:.3f} "
                  f"({eq_p}), so on average {ME['count']:.2f} of the {n} players volunteer; nobody volunteers with probability "
                  f"(cost/v)^(n/(n{MINUS}1)) = {ME['none']:.3f} ({pct(ME['none'])}), which rises with the group size (diffusion of "
                  f"responsibility); and each player expects {fn(ME['pay'])} a round (v {MINUS} cost). ")
    else:
        theory = (f"With cost {fn(C)} and v {fn(V)} there is no symmetric mixed equilibrium with volunteering (it needs 0 < cost < v). ")
    theory += (f"The pure equilibria of the one-shot game have exactly one volunteer ({n} of them, one per player); they give the "
               f"group the most, {fn(best_round)} a round. ")
    sd_method = ("" if not TP else
                 "Said versus did: a message counts as its sender's own choice for the round only when it says so in the first "
                 f"person ({LDQ}I will volunteer{RDQ}, {LDQ}I'll not volunteer{RDQ}, {LDQ}I won't volunteer{RDQ}, {LDQ}I will "
                 f"abstain{RDQ}, {LDQ}I'll hold{RDQ}), about that round or no named round and not under a condition (if, unless, "
                 "when, only); a statement about another round, the next round or later is left out, as is a message whose "
                 f"statements disagree. {K} of the {D} decisions have such a message"
                 + (f", and {numword(len(mm))} of them " + ("differs" if len(mm) == 1 else "differ") + " from it" if mm else
                    ", and all of them match it") + ". ")
    foot = (f"Data: the {date} recording of eco_volunteer ({cfg_bits}); every player is played by {model}"
            + (f" ({full_model})" if full_model and full_model != model else "") + ". " + rs + obs
            + "Every number on this page is computed from the game log: the round_result events (decisions in seat order, the "
              "logged number of volunteers) and the talk events (the messages). The log has no event per decision, so each decision "
              "step is rebuilt from its round's result. Payoffs are not logged; they are recomputed with the rule (v "
            + MINUS + " cost for a volunteer, v for the others, 0 for everyone when nobody volunteers) and checked against the logged "
              f"volunteer counts, the summary's rewards, the winners and the game's own totals (volunteer decisions {T}, rounds "
              f"with no volunteer {len(none_rs)})"
            + (". " if not warns else f"; {plural(len(warns), 'check')} disagree (printed by the builder; the page shows the logged decisions). ")
            + fb
            + f"Player k is seat P(k{MINUS}1) in the log, the prompts and the players' messages; each round the players send their "
              "messages in that order, then decide in that order, and none sees another's decision before the reveal. " + sd_method
            + theory
            + "Human benchmark (rules): Franzen (1995) and Goeree, Holt and Smith (2017) find that the volunteering rate falls as "
              "the group grows, in the direction the theory predicts. Sources: Diekmann (1985), Journal of Conflict Resolution; "
              "Franzen (1995); Goeree, Holt and Smith (2017), Journal of Economic Behavior and Organization.")

    kinds = {k: v for k, v in KINDS.items() if any(st["kind"] == k for st in steps)}
    if TP:
        if not K:
            feed_sd = ""
        elif not mm:
            feed_sd = f"Each of the {K} times a player's message said what it would do that round, it did so. "
        else:
            feed_sd = (f"Of the {K} times a player's message said what it would do that round, it did otherwise {times(len(mm))} "
                       f"({rounds_text([i + 1 for i in mm_rounds])}); the reveal says who. ")
        feed_sub = ("Each round's public messages, then the reveal of the decisions (secret until every player had decided), with "
                    f"the payoffs and scores. {name_note} {feed_sd}")
    else:
        feed_sub = "Each round's reveal of the decisions (secret until every player had decided), with the payoffs and scores. "
    d = {"id": GID, "name": NAME, "title": title, "lede": lede, "seatNames": LAB, "tiles": tiles, "foot": foot,
         "x": {"n": N, "labels": [f"R{i + 1}" for i in range(N)]}, "charts": charts, "kinds": kinds, "steps": steps,
         "feed": feed, "feedTitle": "Round by round",
         "feedSub": feed_sub + "A decision step's title shows the outcome its player learned afterwards. The selected step is "
                               "highlighted; later entries are hidden."}
    d = en_all(d)
    d["eyebrow"] = f"social_game_env {DOT} {GID} {DOT} {model}" + (", reasoning on" if has_reason else "") + f" {DOT} recorded {date}"
    return d


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    common.write(build(src, calls), out, date=recorded_date(src))
