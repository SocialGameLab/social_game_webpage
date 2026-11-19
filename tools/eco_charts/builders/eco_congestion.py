"""Replay data for eco_congestion (repeated two-player, two-road congestion game; Rosenthal 1973, Monderer and Shapley 1996).

usage: python3 -I eco_congestion.py SOURCE OUT_DIR [CALLS.jsonl] [--date=YYYY-MM-DD]
  --date  the recording date, for a source that does not carry one (a rendered Markdown replay); otherwise recorded_at from
          meta.json or run.json is used, and with neither the page says the date was not recorded (no date is guessed).
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md), a file inside
          one, or the rendered Markdown replay of one (one "- **event** ... `{json}`" line per logged event).
  OUT_DIR where <OUT_DIR>/eco_congestion.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). Calls are tied to
          decisions in time order: a call belongs to the next decision of the same seat (and, when its prompt shows them, of the
          same phase and round); the call whose answer is the logged road is that decision's call (earlier ones were format
          retries; their reasoning is kept above it). Its reasoning becomes the step's thinking and its prompt the observation.

Each round both drivers choose road A or B in the DECIDE phase, which asks the seats in order (P0, then P1) and keeps each
choice private until both have chosen (rules: Round Procedure, Deviations). The log has no event per choice: both choices,
payoffs and running scores are in the round's round_result event, so each round gives two decisions, P0's first. Every number
shown is taken from the log; payoffs are checked against the logged payoff table and the scores against the summary, and any
disagreement is printed as a warning. Benchmarks (pure equilibria, social optimum, the symmetric mixed equilibrium) are
computed exactly (fractions) from the logged payoff table."""
import json, os, re, sys, unicodedata
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_congestion"
DEFAULT_DATE = None                                  # no guessed date: an undated source is shown as undated, with a warning
ROADS = ("A", "B")
CELLS = ("AA", "AB", "BA", "BB")
KINDS = {"road_a": ["Road A", "c1"], "road_b": ["Road B", "c2"]}
ROAD_COLOR = {"A": "c1", "B": "c2"}
SPLIT_COLOR, SAME_COLOR = "c3", "hold"   # page draws cell text in white: c4 amber is too light for it
DRIVER_COLOR = ("ink", "hold")
MINUS, DOT, TIMES = "\u2212", "\u00b7", "\u00d7"


# ------------------------------------------------------------------------------------------------ English-only text
_CJK = "\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af"
# one marker per stretch of non-English text: runs separated by at most four other characters on one line (a road letter, a
# number, converted punctuation) are one stretch, so a Chinese sentence that mentions road B gives one marker, not three
# (a road letter or short number glued to either end, as in "B\u8def", goes with it)
_CJK_RUN = re.compile(f"(?:(?<![A-Za-z0-9])[A-Za-z0-9]{{1,3}})?[{_CJK}]+(?:[^\\n{_CJK}]{{0,4}}[{_CJK}]+)*(?:[A-Za-z0-9]{{1,3}}(?![A-Za-z0-9]))?")
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
    s = re.sub(r"[\u3000-\u303f\uff00-\uffef]", "", s)
    return re.sub(r"(?<=\S) {2,}(?=\S)", " ", s)              # converted punctuation can leave double spaces


# ------------------------------------------------------------------------------------------------ loading
_HIDDEN_TAG, _PRIVATE_TAG = "\u9690\u85cf", "\u79c1\u5bc6"          # the replay's "hidden" and "private" event tags
_SEED_RE, _MODEL_RE = r"\u79cd\u5b50 (\d+)", r"\u6a21\u578b ([^\s,\uff0c]+)"   # "seed N", "model X" in the replay header


def _md_bundle(path):
    """A rendered Markdown replay (declarative games): events from the '- **event** (tag): `{json}`' lines (the tag marks hidden
    and private events), seat rewards, seed and model from the header, and the model calls listed after the events ('### '
    blocks), if any."""
    text = open(path, encoding="utf-8").read()
    events = []
    for m in re.finditer(r"^- \*\*([A-Za-z_]+)\*\*([^`\n]*)`(\{.*\})`\s*$", text, re.M):
        try:
            body = json.loads(m.group(3))
        except ValueError:
            continue
        body["event"] = m.group(1)
        if _HIDDEN_TAG in m.group(2):
            body["_visibility"] = "hidden"
        elif _PRIVATE_TAG in m.group(2):
            body["_visibility"] = "private"
        events.append(body)
    head = text.split("\n## ")[0].split("\n- **")[0]
    rw = re.search(r'(\{"0":[^}]*\})', head)
    seed = re.search(_SEED_RE, head)
    model = re.search(_MODEL_RE, head)
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
        sm = re.search(r"\bP(\d+)\b", hdr)
        if ans:
            calls.append({"seat": common.seat_of(hdr) if common.seat_of(hdr) is not None else (int(sm.group(1)) if sm else None),
                          "prompt": "", "answer": (ans[-1] if len(ans) > 1 else ans[0]).strip(),
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


def find_date(b, override=None):
    """The recording date (YYYY-MM-DD): --date, else meta.json or run.json recorded_at; None when none of them gives one (a
    rendered Markdown replay has no date), and the page then says the date was not recorded rather than guessing one."""
    d = override or (b.get("meta") or {}).get("recorded_at") or (b.get("run") or {}).get("recorded_at") or DEFAULT_DATE
    return str(d)[:10] if d else None


def recorded_date(src, override=None):
    return find_date(load_source(src), override)


# ------------------------------------------------------------------------------------------------ text helpers
NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def numword(n):
    return NUM[n] if 0 <= n < len(NUM) else str(n)


def dname(seat):
    return f"Driver {seat + 1}"


def g(v):
    """A payoff or score: an integer without decimals, anything else with two (exact Fractions included)."""
    v = Fraction(v) if not isinstance(v, Fraction) else v
    return f"{int(v):,}" if v.denominator == 1 else f"{float(v):,.2f}".rstrip("0").rstrip(".")


def frac(v):
    """5/8 as '5/8', an integer as itself."""
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def pct(v, d=1):
    return f"{float(v) * 100:.{d}f}%"


def span_text(a, z, word="rounds"):
    """Rounds a..z (a <= z): 'round 3', 'rounds 1 and 2' (two in a row read better without 'to'), 'rounds 3 to 8'."""
    return f"{word[:-1]} {a}" if a == z else f"{word} {a} and {z}" if z == a + 1 else f"{word} {a} to {z}"


def rounds_text(rs):
    """[2, 3, 4] -> 'rounds 2 to 4'; [1] -> 'round 1'; [1, 2] -> 'rounds 1 and 2'; [1, 3, 4] -> 'rounds 1, 3 and 4'."""
    rs = sorted(rs)
    if not rs:
        return "no round"
    if len(rs) == 1:
        return f"round {rs[0]}"
    if len(rs) == 2:
        return f"rounds {rs[0]} and {rs[1]}"
    if rs == list(range(rs[0], rs[-1] + 1)):
        return f"rounds {rs[0]} to {rs[-1]}"
    return "rounds " + ", ".join(str(r) for r in rs[:-1]) + " and " + str(rs[-1])


def cells_text(cells):
    cells = list(cells)
    return cells[0] if len(cells) == 1 else ", ".join(cells[:-1]) + " and " + cells[-1]


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


# ------------------------------------------------------------------------------------------------ game theory of the table
def analyse(table):
    """Exact one-shot benchmarks of the 2x2 table {cell: (P0 payoff, P1 payoff)}: pure equilibria (weak and strict), social
    optimum, strictly dominant road per seat and the completely mixed equilibrium (each seat's probability of road A, expected
    payoffs, probability of ending on the same road), when it exists."""
    P = {c: (Fraction(v[0]), Fraction(v[1])) for c, v in table.items()}
    other = {"A": "B", "B": "A"}
    pure, strict = [], []
    for x in ROADS:
        for y in ROADS:
            a0, a1 = P[x + y][0] - P[other[x] + y][0], P[x + y][1] - P[x + other[y]][1]
            if a0 >= 0 and a1 >= 0:
                pure.append(x + y)
                if a0 > 0 and a1 > 0:
                    strict.append(x + y)
    joint = {c: sum(P[c]) for c in CELLS}
    best = max(joint.values())
    dom = []
    for seat in (0, 1):
        d = None
        for r in ROADS:
            if seat == 0 and all(P[r + y][0] > P[other[r] + y][0] for y in ROADS):
                d = r
            if seat == 1 and all(P[x + r][1] > P[x + other[r]][1] for x in ROADS):
                d = r
        dom.append(d)
    mixed = None
    den0 = P["AA"][0] - P["AB"][0] - P["BA"][0] + P["BB"][0]
    den1 = P["AA"][1] - P["BA"][1] - P["AB"][1] + P["BB"][1]
    if den0 and den1:
        q = (P["BB"][0] - P["AB"][0]) / den0            # P1's probability of road A that leaves P0 indifferent
        p = (P["BB"][1] - P["BA"][1]) / den1            # P0's probability of road A that leaves P1 indifferent
        if 0 < p < 1 and 0 < q < 1:
            mixed = {"p": [p, q], "pay": [q * P["AA"][0] + (1 - q) * P["AB"][0], p * P["AA"][1] + (1 - p) * P["BA"][1]],
                     "same": p * q + (1 - p) * (1 - q)}
    sym = P["AB"] == P["BA"][::-1] and P["AA"][0] == P["AA"][1] and P["BB"][0] == P["BB"][1]
    return {"P": P, "pure": pure, "strict": strict, "joint": joint, "best": best, "opt": [c for c in CELLS if joint[c] == best],
            "dom": dom, "mixed": mixed, "sym": sym}


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
    """Set each line's "end" (its name and last value written above its right end) only where the page can draw it clear of
    every line and dot. Mirrors index.html for a line chart with "bars": [] (banded x axis, y axis from 0 when all values are
    >= 0): margins 86 / 16 px, plot top 14 px and bottom 26 px, labels 7 px above the end point, right-aligned, 11 px monospace
    (6.6 px a character), stacked at least 13 px apart. Checked at a desktop and a phone width; labels that collide are dropped
    one at a time, benchmark lines (dashed) before players."""
    want = [l for l in lines if l.get("end")]
    for l in lines:
        l.pop("end", None)

    def clashes(on, W):
        iw, cw, ih = W - 102, (W - 102) / n, height - 40
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
            hit = False
            for m in lines:
                pts = [(X(k), Y(v)) for k, v in enumerate(m["v"]) if v is not None]
                for (xa, ya), (xb, yb) in zip(pts, pts[1:]):          # every line, its own included (a falling end crosses it)
                    hit = hit or any(inside(xa + (xb - xa) * j / 40, ya + (yb - ya) * j / 40) for j in range(41))
                if m.get("dots"):                                       # dots (r 3.5 with a white rim), its own end dot excepted
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
    var = dict(over.get("variables") or {})
    var.update({k: v for k, v in (summary.get("variables") or {}).items() if k not in var})
    cv = cfg.get("variables") or {}

    def par(k, default=None):
        for src in (rules, var, cv):
            if k in src:
                return src[k]
        return default
    table = {}
    for c in CELLS:
        v = par("payoff_" + c)
        if v is None:
            raise SystemExit(f"payoff_{c} is not in the log, the summary or the config")
        table[c] = [v[0], v[1]]
    rounds = [e for e in events if e.get("event") == "round_result"]
    if not rounds:
        raise SystemExit("no round_result event in the game log")
    known = {"setup", "rules", "round_result", "game_end", "game_over"}
    talks, done = [], 0                              # public messages (only when talk_passes > 0), shown in the feed
    for e in events:
        ev = e.get("event")
        if ev == "round_result":
            done += 1
        elif ev in ("talk", "message", "chat", "say"):
            who = e.get("seat", e.get("actor", e.get("speaker", e.get("from"))))
            who = int(re.sub(r"\D", "", str(who)) or -1) if who is not None else -1
            text = e.get("text", e.get("message", e.get("content")))
            if who in (0, 1) and isinstance(text, str):
                talks.append({"seat": who, "text": text, "round": int(e.get("round") or done + 1)})
                continue
            print(f"warning: {ev} event without a seat or text is not shown: {json.dumps(e)[:200]}", file=sys.stderr)
        elif ev not in known and e.get("_visibility") != "hidden":
            print(f"warning: public event {ev!r} is not shown on the page", file=sys.stderr)
    R = int(par("rounds", len(rounds)))
    talk = int(par("talk_passes", 0) or 0)
    if talk and len(talks) != 2 * talk * len(rounds):
        print(f"warning: {len(talks)} messages read, {2 * talk * len(rounds)} expected ({talk} per driver in each round)", file=sys.stderr)
    metrics = over.get("_metrics") or var.get("_metrics") or summary.get("_metrics") or {}
    return {"table": table, "R": R, "talk": talk, "table_id": cv.get("table_id"), "rounds": rounds, "talks": talks, "metrics": metrics,
            "game_end": next((e for e in events if e.get("event") == "game_end"), {}), "winners": over.get("winners", summary.get("winners"))}


def verify(G, A, rewards):
    warn = []
    sc = [0, 0]
    for i, e in enumerate(G["rounds"]):
        if int(e.get("round", i + 1)) != i + 1:
            warn.append(f"round_result {i}: round {e.get('round')} logged, {i + 1} expected")
        cell = e["choice0"] + e["choice1"]
        want = G["table"].get(cell)
        if want is None:
            warn.append(f"round {i + 1}: choices {cell!r} are not two roads")
            continue
        for s in (0, 1):
            if e[f"payoff{s}"] != want[s]:
                warn.append(f"round {i + 1}: payoff{s} {e[f'payoff{s}']} logged, {want[s]} in the table for {cell}")
            sc[s] += e[f"payoff{s}"]
            if e.get(f"score{s}") is not None and e[f"score{s}"] != sc[s]:
                warn.append(f"round {i + 1}: score{s} {e[f'score{s}']} logged, {sc[s]} summed from the payoffs")
    if len(G["rounds"]) != G["R"]:
        warn.append(f"{len(G['rounds'])} rounds logged, {G['R']} configured")
    for s in (0, 1):
        ge = G["game_end"].get(f"score{s}")
        if ge is not None and ge != sc[s]:
            warn.append(f"game_end score{s} {ge}, {sc[s]} summed from the payoffs")
        if rewards[s] is not None and abs(rewards[s] - sc[s]) > 1e-9:
            warn.append(f"reward of seat {s}: {rewards[s]} in the summary, {sc[s]} summed from the payoffs")
    for w in warn:
        print("warning:", w, file=sys.stderr)
    return warn


# ------------------------------------------------------------------------------------------------ model calls
def call_meta(c):
    """Phase and round of a call, from its prompt ('Current phase: X', the public state's "round"); None when not shown. A
    prompt that does not show the phase (a format-retry note) is placed by its answer's action_id ("talk" is the TALK phase,
    "choose" DECIDE), so a TALK call never joins a decision's window as an "earlier attempt"."""
    p = c.get("prompt") or ""
    m = re.search(r"Current phase:\s*([A-Za-z_]+)", p)
    phase = m.group(1).upper() if m else None
    if phase is None:
        a = _norm_answer(c.get("answer"))
        phase = {"talk": "TALK", "choose": "DECIDE"}.get(a.get("action_id") if isinstance(a, dict) else None)
    i = p.find("Public state:")
    rm = re.search(r'"round":\s*(\d+)', p[i:]) if i >= 0 else None
    return phase, int(rm.group(1)) if rm else None


def _matches(call, st):
    """Strict: action_id "choose" with choice exactly "A" or "B" (the legal payloads; "a" or " B" would be an invalid reply that
    the agent retried or replaced, so it is not the applied answer) equal to the logged road."""
    a = _norm_answer(call.get("answer"))
    if not isinstance(a, dict) or a.get("action_id") != "choose":
        return False
    p = a.get("payload") or {}
    return isinstance(p, dict) and p.get("choice") in ROADS and p.get("choice") == st["choice"]


def attach_calls(steps, calls):
    """Tie calls to steps in time order. For each step, the window is the run of calls from the pointer whose seat, phase and
    round agree with the step (unknown values agree with anything); the first call in it whose answer is the logged road is the
    one; earlier calls in the window were failed attempts (format retries) and their reasoning is kept above it. With no matching
    answer the last call of the window is used and the step notes that a different road was applied (the recorder's agent
    replaces an unusable reply after its retries, the engine's default is road A; rules: Invalid Actions). Calls of other phases
    (TALK, when messages are enabled) are skipped with a warning. Returns the number of steps that got a call."""
    meta = [call_meta(c) for c in calls]
    seats = [c.get("seat") if c.get("seat") is not None else common.seat_of(c.get("prompt")) for c in calls]   # a null seat field

    def fits(k, st, strict=False):
        seat, (ph, rnd) = seats[k], meta[k]
        if strict:
            return seat == st["seat"] and ph == st["phase"] and rnd == st["round"]
        return seat in (None, st["seat"]) and ph in (None, st["phase"]) and rnd in (None, st["round"])
    ptr, tied = 0, 0
    for st in steps:
        if ptr < len(calls) and not fits(ptr, st):    # resynchronise on a call that names this very decision
            ahead = next((k for k in range(ptr, min(len(calls), ptr + 40)) if fits(k, st, strict=True)), None)
            if ahead is None:
                continue
            print(f"warning: calls {ptr}..{ahead - 1} match no decision; skipped", file=sys.stderr)
            ptr = ahead
        j = ptr
        while j < len(calls) and fits(j, st):
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
            parts.append(f"[The game applied road {st['choice']} instead of this answer: {c['answer'][:300]}]")
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        # the observation: the latest prompt in the window, up to the tied call, that shows the game state; a retry's prompt can
        # be only the correction note, and then the original prompt of the decision is shown (or the rebuilt state if none is)
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
    return tied


# ------------------------------------------------------------------------------------------------ build
def build(bundle_dir, calls=None, date=None):
    b = load_source(bundle_dir)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    if not calls:
        calls = b.get("md_calls") or []
    events, summary, run, cfg = b["events"], b.get("summary") or {}, b.get("run") or {}, b.get("config") or {}
    G = read_game(events, summary, cfg)
    A = analyse(G["table"])
    P, T, R = A["P"], G["table"], len(G["rounds"])
    raw_rw = summary.get("rewards") or (run.get("result") or {}).get("rewards") or {}
    rewards = [float(raw_rw[str(s)]) if str(s) in raw_rw else None for s in (0, 1)]
    verify(G, A, rewards)

    # ---- per round, straight from the round_result events
    ch = [[e["choice0"] for e in G["rounds"]], [e["choice1"] for e in G["rounds"]]]
    pay = [[e["payoff0"] for e in G["rounds"]], [e["payoff1"] for e in G["rounds"]]]
    score = [[], []]
    for s in (0, 1):
        acc = 0
        for i, e in enumerate(G["rounds"]):
            acc += pay[s][i]
            score[s].append(e.get(f"score{s}", acc))
    cells = [ch[0][i] + ch[1][i] for i in range(R)]
    split = [c[0] != c[1] for c in cells]
    joint = [pay[0][i] + pay[1][i] for i in range(R)]
    S = sum(split)
    split_rs = [i + 1 for i in range(R) if split[i]]
    same_rs = [i + 1 for i in range(R) if not split[i]]
    share = [round(100 * sum(split[:i + 1]) / (i + 1), 4) for i in range(R)]
    a_n = [ch[s].count("A") for s in (0, 1)]        # rounds each driver took road A (the rules' metrics a_count0, a_count1)
    final = [score[0][-1], score[1][-1]]
    total = final[0] + final[1]
    best_total = A["best"] * R
    swaps = sum(1 for i in range(1, R) if split[i] and split[i - 1] and cells[i] != cells[i - 1])
    k0 = R - 1                                       # start of the final run of one cell (0-based)
    while k0 > 0 and cells[k0 - 1] == cells[-1]:
        k0 -= 1
    settled = cells[-1] if R - k0 >= 2 else None
    split_table = sorted(A["pure"]) == ["AB", "BA"]
    dom = A["dom"][0] if A["dom"][0] and A["dom"][0] == A["dom"][1] else None   # a road both should always take (table t3)
    d_n = [ch[s].count(dom) for s in (0, 1)] if dom else None
    mixed = A["mixed"]
    for k, v in (("a_count0", a_n[0]), ("a_count1", a_n[1]), ("split_rounds", S), ("same_road_rounds", R - S)):
        if k in G["metrics"] and G["metrics"][k] != v:
            print(f"warning: _metrics {k} is {G['metrics'][k]}, {v} counted from the round results", file=sys.stderr)
    model_id = ((b.get("meta") or {}).get("model") or run.get("model") or "").split("/")[-1]
    if not model_id:
        print("warning: no model name in meta.json, run.json or the replay header; the page says it was not recorded", file=sys.stderr)
    model = model_id or "a model whose name was not recorded"
    date = find_date(b, date)
    if not date:
        print("warning: no recorded_at in meta.json or run.json (pass --date=YYYY-MM-DD); the page says the date was not recorded",
              file=sys.stderr)
    seed = run.get("seed", cfg.get("seed"))
    better = worse = edge = None                     # symmetric tables: the road that pays more to the driver alone on it
    if A["sym"] and T["AB"][0] != T["BA"][0]:
        better, worse = ("A", "B") if T["AB"][0] > T["BA"][0] else ("B", "A")
        edge = abs(Fraction(T["AB"][0]) - Fraction(T["BA"][0]))

    def rnds(n):
        return f"{numword(n)} round{'s' if n != 1 else ''}"

    def num(v):
        v = Fraction(v)
        return int(v) if v.denominator == 1 else round(float(v), 4)

    def cell_words(c):
        """'Driver 1 on road B, Driver 2 on road A' / 'both on road A'."""
        return f"both on road {c[0]}" if c[0] == c[1] else f"{dname(0)} on road {c[0]}, {dname(1)} on road {c[1]}"

    def cell_plain(c):
        """What a pair of roads is, in words a reader needs no glossary for."""
        eq, op = c in A["pure"], c in A["opt"]
        if eq and op:
            return "an equilibrium (neither driver gains by switching alone) that also gives the most joint payoff"
        if eq:
            return "an equilibrium (neither driver gains by switching alone), but with less joint payoff than the best"
        if op:
            return "the most joint payoff, but not an equilibrium (a driver gains by switching alone)"
        return "neither an equilibrium nor the most joint payoff"

    def short_plain(c):
        eq, op = c in A["pure"], c in A["opt"]
        return ("an equilibrium with the most joint payoff" if eq and op else "an equilibrium" if eq else
                "the most joint payoff, not an equilibrium" if op else "not an equilibrium")

    # ---- steps: two per round, P0 first (the DECIDE phase asks the seats in order; each choice stays secret until both chose)
    tbl_line = "Payoffs [P0, P1] by road pair (P0's road first): " + ", ".join(f"{c} [{g(T[c][0])}, {g(T[c][1])}]" for c in CELLS) + "."
    steps, history = [], []
    prompt = run.get("prompt") or {}
    for i in range(R):
        for s in (0, 1):
            r = i + 1
            me, other = ch[s][i], ch[1 - s][i]
            where = f"both on {me}" if me == other else f"alone ({dname(1 - s)} on {other})"
            st = {"seat": s, "x": i, "kind": "road_" + me.lower(), "round": r, "phase": "DECIDE", "choice": me,
                  "head": f"Round {r} {DOT} road {me} {DOT} {where}, +{g(pay[s][i])}",
                  "act": json.dumps({"action_id": "choose", "payload": {"choice": me}})}
            before = [score[0][i - 1], score[1][i - 1]] if i else [0, 0]
            lines = ["(Rebuilt from the game log; the exact prompt for this turn was not saved.)",
                     f"You are seat P{s} ({dname(s)} on this page). Your role: PLAYER.",
                     f"Current phase: DECIDE, round {r} of {G['R']}. Both players secretly choose road A or road B; the two choices are "
                     "revealed together. " + ("No messages: talk is off." if not G["talk"] else
                                             f"{G['talk']} public message(s) per player before each decision."),
                     tbl_line,
                     f"Scores so far: P0 {g(before[0])}, P1 {g(before[1])}."]
            said = [t for t in G["talks"] if t["round"] == r]
            if said:
                lines += ["Messages this round:"] + [f"- P{t['seat']}: {en(t['text'])}" for t in said]
            lines += ["Earlier rounds:"] + history if history else ["Earlier rounds: none yet."]
            st["state"] = "\n".join(lines)
            if (r == 1 and prompt.get("seat") == s and prompt.get("phase") == "DECIDE"
                    and re.search(r'"round": 1\b', prompt.get("text") or "")):
                st["state"] = en(common.state_part(prompt["text"]))      # the saved prompt of the game's first decision
                st["state_from_prompt"] = True
            steps.append(st)
        history.append(f"- Round {i + 1}: P0 {ch[0][i]}, P1 {ch[1][i]} ({'split' if split[i] else 'same road'}): "
                       f"+{g(pay[0][i])} / +{g(pay[1][i])}.")
    attach_calls(steps, calls)
    has_reason = any(st.get("thinking") for st in steps)
    from_prompt = sum(1 for st in steps if st.get("state_from_prompt"))
    # the engine's DECIDE phase keeps each choice as a private "pending" value and does not clear it after the round, so a saved
    # prompt can show the driver's own previous road as private info; the models do not know what it is (some read it as a
    # default action), so the foot explains it. The claim is checked against the log before it is made.
    pending_note = ""
    pend = []
    for st in steps:
        m = re.search(r"Your private info:\s*(\{[^\n]*\})", st["state"]) if st.get("state_from_prompt") and '"pending"' in st["state"] else None
        if m:
            try:
                v = json.loads(m.group(1)).get("pending")
            except (ValueError, AttributeError):
                v = None
            pend.append((st, v, ch[st["seat"]][st["round"] - 2] if st["round"] > 1 else None))
    if pend and has_reason:
        if all(v == prev for _, v, prev in pend):
            pending_note = ("In the prompts, a driver's private info \"pending\" is its own road from the previous round, which the engine "
                            "does not clear; it never shows the other driver's choice")
        else:
            bad = [f"round {st['round']} P{st['seat']}: {v!r}, previous road {prev!r}" for st, v, prev in pend if v != prev]
            print(f"warning: private \"pending\" is not always the driver's previous road ({'; '.join(bad[:4])})", file=sys.stderr)
            pending_note = ("In some prompts a driver's private info shows a \"pending\" road set by the engine, not always its own previous "
                            "road")
        said = [st for st, _, _ in pend if "pending" in (st.get("thinking") or "").lower()]
        dflt = any(re.search(r"pending[^\n]{0,300}default|default[^\n]{0,300}pending", st["thinking"], re.I) for st in said)
        if said:
            pending_note += (", but the models often puzzle over it" if 2 * len(said) >= len(pend) else
                             ", and the models sometimes puzzle over it") + (" and sometimes treat it as a default action" if dflt else "")
        pending_note += ". "
    for st in steps:
        st.setdefault("thinking", "")
        for k in ("round", "phase", "choice", "state_from_prompt"):
            st.pop(k, None)

    # ---- table words (symmetric tables: what a road pays alone and shared)
    if A["sym"]:
        aA, aB, sA, sB = T["AB"][0], T["BA"][0], T["AA"][0], T["BB"][0]
        table_words = (f"alone, road A pays {g(aA)} and road B {g(aB)}; shared, road A pays {g(sA)} and road B {g(sB)}")
        crowd = (f"Sharing a road costs each driver {g(aA - sA)} on either road" if aA - sA == aB - sB else
                 f"Sharing costs {g(aA - sA)} on road A and {g(aB - sB)} on road B")
    else:
        table_words = "payoffs [Driver 1, Driver 2]: " + ", ".join(f"{c} {g(T[c][0])} / {g(T[c][1])}" for c in CELLS)
        crowd = "The table is not symmetric between the drivers"
    if all(c[0] != c[1] for c in A["opt"]):
        opt_words = "only when they split"
    elif len(A["opt"]) == 1:
        opt_words = cell_words(A["opt"][0])
    else:
        opt_words = "road pairs " + cells_text(A["opt"]) + f", {dname(0)}'s road first"

    # ---- feed: the table, then per round the messages (if any) and the result after the round's second decision
    feed = [{"x": 0, "step": 0, "who": "Rules",
             "text": f"{rnds(G['R'])[:1].upper() + rnds(G['R'])[1:]}, "
                     + ("no messages between the drivers. " if not G["talk"] else f"{G['talk']} public message(s) each before every decision. ")
                     + "Payoffs (Driver 1 / Driver 2) for each road pair, Driver 1's road first: "
                     + ", ".join(f"{c} {g(T[c][0])} / {g(T[c][1])}" for c in CELLS) + "."}]
    for i in range(R):
        c = cells[i]
        for t in [t for t in G["talks"] if t["round"] == i + 1]:
            feed.append({"x": i, "step": 2 * i, "who": dname(t["seat"]), "text": en(t["text"])})
        feed.append({"x": i, "step": 2 * i + 1, "who": "Result",
                     "text": f"{dname(0)} road {c[0]}, {dname(1)} road {c[1]}: " + ("split" if split[i] else "same road")
                             + f" ({c}, {short_plain(c)}). Payoffs {g(pay[0][i])} / {g(pay[1][i])}, together {g(joint[i])} of "
                             f"{g(A['best'])} possible. Scores {g(score[0][i])} / {g(score[1][i])}."})

    # ---- chart 1: the grid. Cell text is drawn in white by the page, so the outcome row uses green and gray (both readable
    # under white text, and not confused with road B's orange), at full opacity, and a one-character label (= same road,
    # ≠ split) that fits a column even at phone width, so the row does not rely on color alone.
    NE = "≠"
    rows = [dname(0), dname(1), "Outcome"]
    gcells = [[], [], []]
    for i in range(R):
        for s in (0, 1):
            gcells[s].append({"c": ROAD_COLOR[ch[s][i]], "t": ch[s][i], "o": 1,
                              "tip": f"road {ch[s][i]}, " + ("alone" if split[i] else f"shared with {dname(1 - s)}")
                                     + f": +{g(pay[s][i])} (score {g(score[s][i])})"})
        good = split[i] if split_table else cells[i] in A["opt"]
        gcells[2].append({"c": SPLIT_COLOR if good else SAME_COLOR, "t": NE if split[i] else "=", "o": 1,
                          "tip": ("split" if split[i] else "same road") + f" ({cells[i]}): {short_plain(cells[i])}; "
                                 f"joint payoff {g(joint[i])} of {g(A['best'])}"})
    if split_table:
        legend = [["Road A", ROAD_COLOR["A"]], ["Road B", ROAD_COLOR["B"]], [f"Split ({NE})", SPLIT_COLOR], ["Same road (=)", SAME_COLOR]]
        out_words = f"green with {NE} when the drivers split onto different roads, gray with = when they shared one"
    else:
        legend = [["Road A", ROAD_COLOR["A"]], ["Road B", ROAD_COLOR["B"]], ["Most joint payoff", SPLIT_COLOR], ["Less joint payoff", SAME_COLOR]]
        out_words = (f"= when both took the same road, {NE} when they split; green when the pair of roads gives the most joint payoff "
                     f"({opt_words}), gray otherwise")
    seg = []                                         # maximal runs of one cell, for the text
    for i in range(R):
        if seg and seg[-1][2] == cells[i]:
            seg[-1][1] = i + 1
        else:
            seg.append([i + 1, i + 1, cells[i]])

    def run_words(a, z, c):
        rr = span_text(a, z, "Rounds")
        i = a - 1
        if c[0] == c[1]:
            return f"{rr}: both on road {c[0]}, {g(pay[0][i])} each"
        return f"{rr}: {dname(0)} alone on road {c[0]} ({g(pay[0][i])}), {dname(1)} alone on road {c[1]} ({g(pay[1][i])})"
    seg_text = "; ".join((w if k == 0 else w[:1].lower() + w[1:]) for k, w in enumerate(run_words(*x) for x in seg)) + "."
    sub1 = (f"One column per round. Driver rows: the road each driver chose (letter in the cell); neither saw the other's road until both "
            f"had chosen. Outcome row: {out_words}. {table_words[:1].upper() + table_words[1:]}. Hover a column for payoffs and running scores.")
    charts = [{"title": "Which road each driver took", "sub": sub1, "height": 120,
               "grid": {"rows": rows, "cells": gcells, "rowHeight": 30, "legend": legend},
               "bars": []}]                     # empty bars: the decision timeline uses the same banded x axis as the grid

    # ---- chart 2: running scores against a benchmark (what the game rewards; a round's payoffs follow from the grid's cell)
    lines2 = [{"name": dname(s), "color": DRIVER_COLOR[s], "v": [num(v) for v in score[s]], "dots": True, "end": True} for s in (0, 1)]
    bench = ""
    if mixed:
        mp = mixed["pay"]
        eq_p = (f"each driver takes road A with probability {frac(mixed['p'][0])}" if mixed["p"][0] == mixed["p"][1] else
                f"{dname(0)} takes road A with probability {frac(mixed['p'][0])}, {dname(1)} with {frac(mixed['p'][1])}")
        if mp[0] == mp[1]:
            lines2.append({"name": "Random play", "color": "muted", "dash": True, "end": True,
                           "v": [num(mp[0] * (i + 1)) for i in range(R)]})
            above = all(f > mp[0] * R for f in final)
            bench = (f" Dashed line, random play: the average score when both drivers randomize as in the mixed equilibrium, where "
                     f"{eq_p}: {g(mp[0])} a round, "
                     f"{g(mp[0] * R)} after {rnds(R)}" + (f", below both final scores ({g(final[0])} and {g(final[1])})." if above else "."))
        else:
            for s in (0, 1):
                lines2.append({"name": f"Random play, {dname(s)}", "color": "muted", "dash": True, "end": True,
                               "v": [num(mp[s] * (i + 1)) for i in range(R)]})
            bench = (f" Dashed lines, random play: the average score when both drivers randomize as in the mixed equilibrium, where "
                     f"{eq_p}: {g(mp[0])} a round for "
                     f"{dname(0)}, {g(mp[1])} for {dname(1)}.")
    elif len(A["pure"]) == 1 and P[A["pure"][0]][0] == P[A["pure"][0]][1]:
        c = A["pure"][0]
        lines2.append({"name": f"Equilibrium play ({c})", "color": "muted", "dash": True, "end": True,
                       "v": [num(P[c][0] * (i + 1)) for i in range(R)]})
        bench = (f" Dashed line: the score of equilibrium play ({cell_words(c)} every round), {g(P[c][0])} a round, "
                 f"{g(P[c][0] * R)} after {rnds(R)}.")
    sub2 = f"Each driver's score after each round: the sum of its payoffs so far. {crowd}. {seg_text}{bench}"
    if score[0] == score[1]:
        sub2 += " The two scores are equal after every round, so the drivers' lines coincide."
    if fit_end_labels(lines2, R, 230, lambda v: f"{v:,.0f}"):
        # a label the page would draw across another line was dropped; label both drivers or neither (the legend names them)
        if not all(l.get("end") for l in lines2[:2]):
            for l in lines2[:2]:
                l.pop("end", None)
        if not any(l.get("end") for l in lines2) and "final scores (" not in sub2:
            sub2 += f" Final scores {g(final[0])} / {g(final[1])}."
    charts.append({"title": "Running score", "sub": sub2, "height": 230, "fmt": "int", "lines": lines2, "bars": []})

    # ---- chart 3: split share (tables where the drivers should split), or each driver's share on the dominant road (t3)
    if dom:
        lines3 = [{"name": dname(s), "color": DRIVER_COLOR[s], "dots": True, "end": True,
                   "v": [round(100 * ch[s][:i + 1].count(dom) / (i + 1), 4) for i in range(R)]} for s in (0, 1)]
        sub3 = (f"After each round, the share of rounds so far in which each driver took road {dom}, the dominant road: it pays more "
                f"whatever the other driver does, so equilibrium play takes it every round (100%). After round {R}: "
                f"{dname(0)} {pct(Fraction(d_n[0], R))}, {dname(1)} {pct(Fraction(d_n[1], R))}."
                + (" The two drivers' lines coincide." if lines3[0]["v"] == lines3[1]["v"] else ""))
        if fit_end_labels(lines3, R, 200, lambda v: f"{v:,.1f}%") and not all(l.get("end") for l in lines3):
            for l in lines3:
                l.pop("end", None)
        charts.append({"title": f"Share of rounds on road {dom} so far", "sub": sub3, "height": 200, "fmt": "pct0", "lines": lines3, "bars": []})
    else:
        lines3 = [{"name": "Split so far", "color": SPLIT_COLOR, "v": share, "area": True, "dots": True, "end": True}]
        sub3 = (f"After each round, the share of the rounds so far in which the drivers took different roads ({pct(Fraction(S, R))} after "
                f"round {R}).")
        if mixed:
            sp = 1 - mixed["same"]
            lines3.append({"name": "Random play", "color": "muted", "dash": True, "v": [round(float(sp) * 100, 4)] * R, "end": True})
            sub3 += (f" Dashed line, random play: {pct(sp)}, the chance of splitting in a round when both drivers randomize as in the mixed "
                     f"equilibrium ({eq_p}).")
        if split_table:
            sub3 += " Both pure equilibria (AB and BA) put the drivers on different roads, so equilibrium play splits every round."
        elif A["pure"]:
            sub3 += f" Pure equilibri{'um' if len(A['pure']) == 1 else 'a'} of this table: {cells_text(A['pure'])}."
        fit_end_labels(lines3, R, 200, lambda v: f"{v:,.1f}%")
        charts.append({"title": "Share of rounds split so far", "sub": sub3, "height": 200, "fmt": "pct0", "lines": lines3, "bars": []})

    # ---- tiles
    if dom:
        nd = sum(d_n)
        tiles = [{"k": f"Decisions on road {dom} (dominant)", "v": f"{nd} of {2 * R}",
                  "s": f"Road {dom} pays more whatever the other driver does, so equilibrium play takes it in every decision.",
                  "spark": [round(100 * (ch[0][:i + 1].count(dom) + ch[1][:i + 1].count(dom)) / (2 * (i + 1)), 4) for i in range(R)]}]
    else:
        rand = ""
        if mixed:
            who = (f"each driver picked road A at random with probability {frac(mixed['p'][0])}" if mixed["p"][0] == mixed["p"][1] else
                   f"{dname(0)} picked road A at random with probability {frac(mixed['p'][0])} and {dname(1)} with {frac(mixed['p'][1])}")
            rand = f". If {who} (the mixed equilibrium), they would split {pct(1 - mixed['same'])} of rounds"
        tiles = [{"k": "Rounds on different roads", "v": f"{S} of {R}",
                  "s": (f"Split in {rounds_text(split_rs)}" if S else "Never split")
                       + (f"; same road in {rounds_text(same_rs)} ({', '.join(sorted(set(cells[r - 1] for r in same_rs)))})" if same_rs else "")
                       + rand + ".",
                  "spark": share}]
    win = G["winners"] if isinstance(G["winners"], list) else []
    lead = 0 if final[0] > final[1] else 1
    gap = abs(final[0] - final[1])
    if gap:
        why = ""
        split_cells = sorted(set(cells[i] for i in range(R) if split[i]))
        if len(split_cells) == 1:                    # every split went the same way: one driver always had the better road
            c = split_cells[0]
            mine, theirs = c[lead], c[1 - lead]
            pm, pt = pay[lead][split.index(True)], pay[1 - lead][split.index(True)]
            why = (f": in {'all ' if S > 1 else ''}{numword(S) if S > 1 else 'the'} split round{'s' if S > 1 else ''} it had road {mine} "
                   f"({g(pm)}) and {dname(1 - lead)} road {theirs} ({g(pt)})")
        elif better:                                 # the gap comes from who had the better road to itself
            nb = [sum(1 for i in range(R) if split[i] and ch[s][i] == better) for s in (0, 1)]
            why = (f": alone on road {better}, the better road to have alone ({g(max(T['AB'][0], T['BA'][0]))}), in {nb[lead]} "
                   f"round{'s' if nb[lead] != 1 else ''}, against {nb[1 - lead]} for {dname(1 - lead)}")
        s2 = f"{dname(0)} / {dname(1)}. {dname(lead)} wins by {g(gap)}{why}."
    else:
        s2 = f"{dname(0)} / {dname(1)}: a tie."
    if win and [int(w) for w in win] != ([lead] if gap else [0, 1]):
        print(f"warning: winners {win} logged, scores {final}", file=sys.stderr)
    road_a = f"Road A taken in {a_n[0]} of {R} round{'s' if R != 1 else ''} by {dname(0)}, {a_n[1]} of {R} by {dname(1)}."
    tile4_has_counts = bool(settled and settled[0] != settled[1]) or bool(dom)
    tiles.append({"k": "Final scores", "v": f"{g(final[0])} / {g(final[1])}", "s": s2 + ("" if tile4_has_counts else " " + road_a)})
    s3 = f"{pct(Fraction(total) / best_total)} of the most two drivers can earn ({g(A['best'])} a round, {opt_words})"
    if mixed:
        s3 += f"; random play (the mixed equilibrium) expects {g((mixed['pay'][0] + mixed['pay'][1]) * R)}"
    tiles.append({"k": "Joint payoff", "v": f"{g(total)} of {g(best_total)}", "s": s3 + "."})
    span = span_text(k0 + 1, R, "Rounds") if k0 else f"All {rnds(R)}"
    if settled and settled[0] != settled[1]:
        note = f"{span}: {settled} ({dname(0)} on {settled[0]}, {dname(1)} on {settled[1]}), {cell_plain(settled)}."
        if better:
            yielder = 0 if settled[0] == worse else 1
            note += (f" {dname(yielder)} gave way: road {better} in {ch[yielder].count(better)} of {R} rounds, against "
                     f"{ch[1 - yielder].count(better)} of {R} for {dname(1 - yielder)}.")
        else:
            note += " " + road_a
        if any(split[:k0]):
            note += f" {swaps} swap{'s' if swaps != 1 else ''} between AB and BA before that."
        tiles.append({"k": "Settled on", "v": f"{settled[0]} / {settled[1]}", "s": note})
    elif settled:
        tiles.append({"k": "Settled on", "v": f"{settled[0]} / {settled[1]}",
                      "s": f"{span}: both on road {settled[0]}, {cell_plain(settled)}."
                           + (f" Road {dom} taken in {d_n[0]} of {R} rounds by {dname(0)}, {d_n[1]} by {dname(1)}." if dom else "")})
    elif dom:
        tiles.append({"k": f"Road {dom} by driver", "v": f"{d_n[0]} / {d_n[1]}",
                      "s": f"Rounds on road {dom}, out of {R}: {dname(0)} / {dname(1)}."})
    elif R > 1:                                      # (one round: nothing to swap, three tiles)
        tiles.append({"k": "Swaps between AB and BA", "v": f"{swaps}",
                      "s": f"Consecutive split rounds in which the drivers traded roads, out of {R - 1} round change{'s' if R != 2 else ''}."})

    # ---- title, lede, foot
    if dom:
        nd = sum(d_n)
        if nd == 2 * R:
            title = f"Both drivers take the dominant road {dom} in {'all ' if R > 1 else ''}{rnds(R)}"
        elif settled == dom + dom and k0 > 0:
            title = f"The drivers take the dominant road {dom} in {nd} of {2 * R} decisions, both of them from round {k0 + 1} on"
        else:
            title = (f"The drivers take the dominant road {dom} in {nd} of {2 * R} decisions ({dname(0)} {d_n[0]} of {R}, "
                     f"{dname(1)} {d_n[1]} of {R})")
    elif settled and settled[0] != settled[1]:
        y = (0 if settled[0] == worse else 1) if better else None   # the driver left with the road that pays less alone
        if k0:
            first = cells[:k0]
            if k0 == 1 and first[0][0] == first[0][1]:
                pre = f"After a round-1 collision on road {first[0][0]}"
            elif all(c[0] == c[1] for c in first):
                pre = f"After {numword(k0)} rounds on the same road"
            else:
                pre = f"After {numword(k0)} unsettled rounds"
            if y is None:
                title = f"{pre}, the drivers split for the last {rnds(R - k0)}: {cell_words(settled)}"
            elif ch[y][k0 - 1] != settled[y]:
                title = f"{pre}, {dname(y)} gives way and takes road {settled[y]} for the last {rnds(R - k0)}"
            else:
                title = f"{pre}, {dname(y)} keeps to road {settled[y]} and leaves road {better} to {dname(1 - y)} for the last {rnds(R - k0)}"
        elif y is not None:
            title = f"{dname(y)} gives way from round 1: road {settled[y]} in all {rnds(R)}, {dname(1 - y)} on road {better}"
        else:
            title = f"The drivers split from the first round and keep the same split all {rnds(R)}: {cell_words(settled)}"
    elif settled:
        title = f"Both drivers end up on road {settled[0]}" + ((f", from round {k0 + 1} to {R}" if R - k0 > 2 else f" in {span_text(k0 + 1, R)}")
                                                              if k0 else f" in all {rnds(R)}")
    else:
        title = f"The drivers split in {S} of {R} round{'s' if R != 1 else ''}" + (f" and swap roads {swaps} time{'s' if swaps != 1 else ''}" if swaps else "")
    talk_words = ("They cannot talk, and each driver scores its own total payoff." if not G["talk"] else
                  "Before each decision they may send public, non-binding messages; each driver scores its own total payoff.")
    if A["sym"] and split_table:
        lede = (f"Two drivers play {rnds(G['R'])} of a road-choice game: each round both pick road A or road B in secret, and a "
                f"shared road is slow (alone, road A pays {g(T['AB'][0])} and road B {g(T['BA'][0])}; shared, {g(T['AA'][0])} and "
                f"{g(T['BB'][0])}). They do best on different roads, "
                + (f"but the driver alone on road {better} earns {g(edge)} more than the one on road {worse}, so each would rather be the "
                   f"one on {better}, and nothing says who should give way. " if better else "but nothing says who should take which. ")
                + talk_words)
    else:
        lede = (f"Two drivers play {rnds(G['R'])} of a road-choice game. Each round both pick road A or road B in secret; "
                f"{table_words}" + (f", so road {dom} pays more whatever the other driver does. " if dom else ". ") + talk_words)
    eq_text = (f"A pure equilibrium is a pair of roads from which neither driver gains by switching alone; this table's: "
               f"{cells_text(A['pure']) if A['pure'] else 'none'}")
    if A["opt"]:
        eq_text += f"; social optimum (most joint payoff, {g(A['best'])}): {cells_text(A['opt'])}"
    if any(A["dom"]):
        eq_text += "; strictly dominant road: " + ", ".join(f"{dname(s)} {A['dom'][s]}" for s in (0, 1) if A["dom"][s])
    if mixed:
        eq_text += (f". In the symmetric mixed equilibrium each driver takes road A with probability {frac(mixed['p'][0])}"
                    if mixed["p"][0] == mixed["p"][1] else
                    f". In the mixed equilibrium {dname(0)} takes road A with probability {frac(mixed['p'][0])} and {dname(1)} with {frac(mixed['p'][1])}")
        eq_text += (f": the drivers land on the same road with probability {frac(mixed['same'])} ({pct(mixed['same'])}) and expect "
                    + (f"{g(mixed['pay'][0])} a round each" if mixed["pay"][0] == mixed["pay"][1] else
                       f"{g(mixed['pay'][0])} and {g(mixed['pay'][1])} a round"))
    cfg_bits = ", ".join(x for x in [f"table {G['table_id']}" if G["table_id"] else "", f"seed {seed}" if seed is not None else "",
                                     f"{G['R']} round{'s' if G['R'] != 1 else ''}",
                                     "no talk" if not G["talk"] else f"{G['talk']} talk pass(es)"] if x)
    foot = (f"Data: {f'the {date} recording' if date else 'an undated recording'} of eco_congestion ({cfg_bits}), both drivers played "
            f"by {model}; every number on this page is computed from its game log (the round_result events: both roads, payoffs and "
            "running scores). "
            + ("Observations and reasoning come from the saved model calls. " + pending_note if has_reason else
               "This recording was made before reasoning was saved; the game will be re-recorded with reasoning. ")
            + ("" if from_prompt == len(steps) else
               ("The observation of the first decision is the saved prompt; the others are rebuilt from the log. " if from_prompt else
                "The observations are rebuilt from the log. "))
            + f"Driver k is seat P(k{MINUS}1) in the log and the prompts; in each round Driver 1 is asked first, and neither sees the "
            "other's road until both have chosen. "
            + eq_text + ". These benchmarks are computed exactly from the logged payoff table. Congestion games always have a pure "
            "equilibrium (Rosenthal 1973); the payoff numbers are the environment's own table, not the paper's.")
    d = {"id": GID, "name": "Congestion", "title": title, "lede": lede, "seatNames": [dname(0), dname(1)], "tiles": tiles, "foot": foot,
         "x": {"n": R, "labels": [f"R{i + 1}" for i in range(R)]}, "charts": charts, "kinds": dict(KINDS), "steps": steps,
         "feed": feed, "feedTitle": "Round by round",
         "feedSub": "The payoff table, then each round's result once both drivers had chosen. Neither saw the other's road before "
                    "choosing; a step's title shows the outcome its driver learned afterwards. The selected step is highlighted; "
                    "later rounds are hidden."}
    # the eyebrow is built here from the run's own model and date (common.EYEBROW names one fixed model)
    who, when = model_id or "model not recorded", f"recorded {date}" if date else "recording date not recorded"
    d["eyebrow"] = (f"social_game_env {DOT} {GID} {DOT} {who}, reasoning on {DOT} {when}" if has_reason
                    else f"social_game_env {DOT} {GID} {DOT} {who} {DOT} {when} {DOT} reasoning not saved")
    return d


if __name__ == "__main__":
    opt = [a for a in sys.argv[1:] if a.startswith("--date=")]
    args = [a for a in sys.argv[1:] if not a.startswith("--date=")]
    given = opt[-1].split("=", 1)[1] if opt else None
    if given is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", given):
        raise SystemExit(f"--date must be YYYY-MM-DD, got {given!r}")
    if len(args) < 2:
        raise SystemExit(__doc__)
    src, out = args[0], args[1]
    calls = common.load_calls(args[2]) if len(args) > 2 else []
    d = build(src, calls, date=given)
    common.write(d, out, date=recorded_date(src, given) or "date not recorded")   # unused: build sets the eyebrow itself
