"""Replay data for eco_min_effort (repeated minimum-effort coordination game; Van Huyck, Battalio and Beil 1990, Goeree and Holt 2005).

usage: python3 -I eco_min_effort.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md, run.log) or a
          file inside one.
  OUT_DIR where <OUT_DIR>/eco_min_effort.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). Calls are tied to
          decisions in time order: a call belongs to the next decision of the same seat, phase and round (read from its prompt:
          "You are seat Pk", "Current phase: X", the public state's "round"; a value the prompt does not show agrees with
          anything). Within one decision's calls the first whose answer is the logged action (the message text, or the effort) is
          the decision's call; earlier ones were failed attempts (format retries) and their reasoning is kept above it. Its
          reasoning becomes the step's thinking and its prompt the observation.

Each round has talk_passes public messages per player (TALK phase, seats in order, logged as talk events in time order) and then
one sealed effort per player (DECIDE phase, seats asked in order P0, P1, ...; no player sees another's effort before the reveal).
The log has no event per effort: all efforts are in the round's round_result event (efforts in seat order, min_effort). Payoffs
are not logged; they are recomputed exactly with the rule of the game, 2 x min - own effort + 6, and checked against the logged
minimum, the summary's rewards, the winners and the hidden game_end totals (sum_min_effort, coordinated_rounds); any disagreement
is printed as a warning. Nothing about one particular recording (seed, number of players or rounds, outcome) is assumed."""
import json, os, re, sys, unicodedata
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_min_effort"
NAME = "Minimum effort"
DEFAULT_DATE = "2026-11-09"
LO, HI = 1, 7                                       # the effort scale (rules: e_i in {1, ..., 7})
MINUS, TIMES, DOT, LDQ, RDQ, ARROW = "−", "×", "·", "“", "”", "→"
# effort buckets (grid cells, step kinds): the payoff-dominant level, high, low, the secure level
KINDS = {"e7": ["Effort 7 (payoff-dominant)", "c3"],
         "e46": ["Effort 4 to 6", "c1"],
         "e23": ["Effort 2 or 3", "c4"],
         "e1": ["Effort 1 (secure)", "c2"],
         "talk": ["Public message", "muted"]}
PATH_COLORS = ["c1", "c4", "hold", "ink", "muted"]  # players' series (green and orange are kept for the 7 and 1 benchmarks)
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
    return re.sub("[　-〿＀-￯]", "", s)


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
def fn(v):
    """A payoff, score or effort: an integer with thousands separators, anything else with up to two decimals."""
    v = Fraction(v)
    s = f"{abs(int(v)):,}" if v.denominator == 1 else f"{abs(float(v)):,.2f}".rstrip("0").rstrip(".")
    return (MINUS if v < 0 else "") + s


def num(v):
    """A chart value: an int when whole, else a float (the exact Fraction is kept for the text)."""
    v = Fraction(v)
    return int(v) if v.denominator == 1 else round(float(v), 6)


def pct(a, b):
    """a/b as a percentage with as many decimals (up to 3) as it takes not to round a value strictly between 0% and 100% to
    either end."""
    if not b:
        return "n/a"
    x = Fraction(a) / Fraction(b) * 100
    if x == int(x):
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


def qty(x, one, many=None):
    """A quantity with its unit, the number formatted like a score: 1 unit, 1,209 units."""
    return f"{fn(x)} {one if x == 1 else (many or one + 's')}"


def join_and(xs):
    xs = [str(x) for x in xs]
    if not xs:
        return ""
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def rounds_text(rs, cap_first=False):
    """[2, 3, 4] -> 'rounds 2 to 4'; [1] -> 'round 1'; [1, 3, 5] -> 'rounds 1, 3 and 5'."""
    rs = sorted(rs)
    w = ("no round" if not rs else f"round {rs[0]}" if len(rs) == 1 else
         f"rounds {rs[0]} to {rs[-1]}" if len(rs) > 2 and rs == list(range(rs[0], rs[-1] + 1)) else "rounds " + join_and(rs))
    return cap(w) if cap_first else w


def pname(s):
    return f"Player {s + 1}"


def players_text(seats, n):
    """[0, 1, 2] -> 'Players 1 to 3'; [1] -> 'Player 2'; [0, 2] -> 'Players 1 and 3'; every seat -> 'all four players'."""
    s = sorted(seats)
    if len(s) == n and n > 1:
        return "both players" if n == 2 else f"all {numword(n)} players"
    if len(s) == 1:
        return pname(s[0])
    if s == list(range(s[0], s[-1] + 1)) and len(s) > 2:
        return f"Players {s[0] + 1} to {s[-1] + 1}"
    return "Players " + join_and(x + 1 for x in s)


def _ws(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def bucket(e):
    return "e7" if e >= HI else ("e46" if e >= 4 else ("e23" if e >= 2 else "e1"))


def runs(vals):
    """Maximal runs of equal values: [(first round, last round, value)] with 1-based rounds."""
    out = []
    for i, v in enumerate(vals):
        if out and out[-1][2] == v:
            out[-1][1] = i + 1
        else:
            out.append([i + 1, i + 1, v])
    return [tuple(x) for x in out]


def ranges_text(rs):
    """[3, 4, 5, 6, 7, 9] -> 'rounds 3 to 7 and 9'; [2, 3, 5] -> 'rounds 2, 3 and 5'; [4] -> 'round 4'."""
    rs = sorted(rs)
    if len(rs) <= 1:
        return rounds_text(rs)
    groups = []
    for r in rs:
        if groups and r == groups[-1][-1] + 1:
            groups[-1].append(r)
        else:
            groups.append([r])
    parts = []
    for g in groups:
        parts += [f"{g[0]} to {g[-1]}"] if len(g) > 2 else [str(x) for x in g]
    return "rounds " + join_and(parts)


_PROMISE = re.compile(r"\b(?:choos(?:e|es|ing)|chose|pick(?:s|ing)?|play(?:s|ing)?|select(?:s|ing)?|stick(?:s|ing)?\s+(?:to|with)|"
                      r"go(?:es|ing)?\s+with|stay(?:s|ing)?\s+(?:at|on|with)|coordinat(?:e|es|ing)\s+on|match(?:es|ing)?\s+at|"
                      r"commit(?:s|ting)?\s+to)\s+(?:an?\s+)?(?:effort\s+(?:level\s+)?(?:of\s+)?)?(\d+)\b|\beffort\s+(?:level\s+)?(?:of\s+)?(\d+)\b",
                      re.I)


def promised(text):
    """The effort levels a message names as a choice ('I will choose 7', 'stick to 7', 'effort 7'); other numbers (payoffs,
    rounds) are not counted."""
    return {int(a or b) for a, b in _PROMISE.findall(text or "")}


def promises_only(text, e):
    """True when a message names effort e as the choice and no other number on the effort scale (outside 'round k'), so a
    hedge such as 'pick 7, or 6 if needed' does not count as a promise of 7."""
    small = {int(x) for x in re.findall(r"(?<!round )(?<!rounds )\b(\d+)\b", text or "", re.I) if LO <= int(x) <= HI}
    return promised(text) == {e} and small <= {e}


def one_way(ms):
    """True when the values only go down or only go up (a constant path counts)."""
    return all(x >= y for x, y in zip(ms, ms[1:])) or all(x <= y for x, y in zip(ms, ms[1:]))


def path_phrase(ms, r0=1):
    """How a run of group minimums (rounds r0, r0 + 1, ...) moves, as words that follow 'The group minimum'. A one-way path is
    'falls/rises from a to b'; any other names its low point (and its high point when above the start) with their rounds."""
    rs = list(range(r0, r0 + len(ms)))
    a, b = ms[0], ms[-1]
    if len(set(ms)) == 1:
        return f"stays at {a}"
    if one_way(ms):
        k = len(ms) - 1
        while k > 0 and ms[k - 1] == b:
            k -= 1
        return (f"{'falls' if b < a else 'rises'} from {a} in round {rs[0]} to {b} "
                + (f"by round {rs[k]}" if k < len(ms) - 1 else f"in round {rs[-1]}"))
    lo, hi = min(ms), max(ms)
    named = []                                          # (first round, words, rounds)
    if lo < a:
        at = [rs[i] for i in range(len(ms)) if ms[i] == lo]
        named.append((at[0], f"drops to {lo} in {ranges_text(at)}", at))
    if hi > a:
        at = [rs[i] for i in range(len(ms)) if ms[i] == hi]
        named.append((at[0], f"rises to {hi} in {ranges_text(at)}", at))
    named.sort()
    parts = [f"starts at {a}"] + [w for _, w, _ in named]
    if not any(rs[-1] in at for _, _, at in named):
        parts.append(f"{'is back at' if b == a else 'ends at'} {b} in round {rs[-1]}")
    return parts[0] + " and " + parts[1] if len(parts) == 2 else ", ".join(parts[:-1]) + ", and " + parts[-1]


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
            efforts = [int(x) for x in (e.get("efforts") or [])]
            rounds.append({"r": int(e.get("round", len(rounds) + 1)), "efforts": efforts,
                           "min_logged": int(e["min_effort"]) if e.get("min_effort") is not None else None,
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
    n = int(par("n_players", cfg.get("n_players") or len(rounds[0]["efforts"])))
    return {"n": n, "R": int(par("rounds", len(rounds))), "tp": int(par("talk_passes", 0) or 0), "rounds": rounds,
            "tail_talk": talk[-1], "end": end, "winners": over.get("winners", summary.get("winners"))}


def verify(G, rewards):
    """Re-derive the minimum, payoffs, scores and the logged totals; returns (scores, warnings). The page shows the logged
    efforts and the payoffs the rule gives for them."""
    n, warn = G["n"], []
    score = [0] * n
    for k, rd in enumerate(G["rounds"]):
        r, ef = rd["r"], rd["efforts"]
        if r != k + 1:
            warn.append(f"round_result {k}: round {r} logged, {k + 1} expected")
        if len(ef) != n:
            warn.append(f"round {r}: {len(ef)} efforts logged for {n} players")
        bad = [x for x in ef if not LO <= x <= HI]
        if bad:
            warn.append(f"round {r}: efforts {bad} outside {LO} to {HI}")
        m = min(ef)
        if rd["min_logged"] is not None and rd["min_logged"] != m:
            warn.append(f"round {r}: min_effort {rd['min_logged']} logged, the efforts' minimum is {m}")
        for s in range(min(n, len(ef))):
            score[s] += 2 * m - ef[s] + 6
        if G["tp"]:
            want = [s for _ in range(G["tp"]) for s in range(n)]
            got = [t["seat"] for t in rd["talk"]]
            if got != want:
                warn.append(f"round {r}: message senders {got}, {want} expected ({G['tp']} pass(es) in seat order)")
    if G["tail_talk"]:
        warn.append(f"{len(G['tail_talk'])} message(s) after the last round_result (a round without a result)")
    end = G["end"]
    sm = sum(min(rd["efforts"]) for rd in G["rounds"])
    co = sum(1 for rd in G["rounds"] if len(set(rd["efforts"])) == 1)
    if end.get("sum_min_effort") is not None and int(end["sum_min_effort"]) != sm:
        warn.append(f"game_end sum_min_effort {end['sum_min_effort']}, {sm} from the round_result events")
    if end.get("coordinated_rounds") is not None and int(end["coordinated_rounds"]) != co:
        warn.append(f"game_end coordinated_rounds {end['coordinated_rounds']}, {co} from the round_result events")
    for s in range(n):
        if rewards[s] is not None and abs(rewards[s] - score[s]) > 1e-9:
            warn.append(f"reward of seat {s}: {rewards[s]} in the summary, {score[s]} summed from the payoff rule")
    best = max(score)
    lead = [s for s in range(n) if score[s] == best]
    if isinstance(G["winners"], list) and sorted(int(w) for w in G["winners"]) != lead:
        warn.append(f"winners {G['winners']} logged, highest scores: seats {lead}")
    if len(G["rounds"]) < G["R"]:
        warn.append(f"only {len(G['rounds'])} of {G['R']} rounds logged: the recording stops early")
    return score, warn


# ------------------------------------------------------------------------------------------------ model calls
def call_meta(c):
    """Phase and round of a call, from its prompt ('Current phase: X', the public state's "round"); None when not shown."""
    p = c.get("prompt") or ""
    m = re.search(r"Current phase:\s*([A-Za-z_]+)", p)
    i = p.find("Public state:")
    rm = re.search(r'"round":\s*(\d+)', p[i:]) if i >= 0 else None
    return (m.group(1).upper() if m else None), (int(rm.group(1)) if rm else None)


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


def _effort_of(payload):
    if not isinstance(payload, dict):
        return None
    v = payload.get("effort")
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v == int(v):
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*\d+\s*", v):
        return int(v)
    return None


def _matches(call, st, loose=False):
    a = _norm_answer(call.get("answer"))
    if st["phase"] == "TALK":
        if not isinstance(a, dict):                 # a discussion prompt may be answered with the bare message
            raw = (call.get("answer") or "").strip()
            return bool(raw) and _ws(raw) == _ws(st["text"])
        if a.get("action_id") != "talk":
            return False
        return loose or _ws((a.get("payload") or {}).get("text")) == _ws(st["text"])
    if not isinstance(a, dict) or a.get("action_id") != "effort":
        return False
    return loose or _effort_of(a.get("payload")) == st["effort"]


def _reject_reason(call, st):
    """Why a decision's earlier attempt was not the accepted one: (words, illegal action id or None)."""
    a = _norm_answer(call.get("answer"))
    if not isinstance(a, dict):
        return "an answer that is not a JSON action", None
    aid = a.get("action_id")
    if aid != ("talk" if st["phase"] == "TALK" else "effort"):
        return "an illegal action id", aid
    if st["phase"] == "DECIDE":
        e = _effort_of(a.get("payload"))
        if e is None or not LO <= e <= HI:
            return f"an effort outside {LO} to {HI}", None
    return "an answer the game did not accept", None


def attach_calls(steps, calls):
    """Tie calls to steps in time order (see the module docstring). Returns a dict: tied (steps with a call), mism (steps whose
    call's answer is not the logged action), retried (steps whose accepted call follows rejected attempts), extra (those
    rejected attempts), reasons ({words: [illegal action ids]}) and skipped (calls tied to no step)."""
    meta = [call_meta(c) for c in calls]
    retried = extra = 0
    reasons = {}

    def fits(k, st, strict=False):
        seat, (ph, rnd) = calls[k].get("seat"), meta[k]
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
            words, aid = _reject_reason(c, st)
            ids = reasons.setdefault(words, [])
            if aid is not None and str(aid) not in ids:
                ids.append(str(aid))
        if use > ptr:
            retried += 1
            extra += use - ptr
        c = calls[use]
        parts.append(("[Final attempt]\n" if parts else "") + (c.get("reasoning") or ("(no reasoning returned)" if parts else "")))
        if hit is None:
            if st["phase"] == "TALK":
                parts.append(f"[The log shows the message {LDQ}{st['text'][:200]}{RDQ}, not this answer: {c['answer'][:300]}]")
            else:
                parts.append(f"[The log shows effort {st['effort']}, not this answer: {c['answer'][:300]}. When no legal action can "
                             "be parsed from the model's reply, the recorder plays a random legal action instead (its run log "
                             "notes each such fallback).]")
            mism += 1
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
        tied += 1
    if ptr < len(calls):
        print(f"warning: {len(calls) - ptr} calls after the last decision were not tied to any step", file=sys.stderr)
    return {"tied": tied, "mism": mism, "retried": retried, "extra": extra, "reasons": reasons,
            "skipped": len(calls) - tied - extra}


# ------------------------------------------------------------------------------------------------ build
def build(src, calls=None):
    b = load_source(src)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    calls = calls or []
    events, summary, run, cfg = b["events"], b.get("summary") or {}, b.get("run") or {}, b.get("config") or {}
    G = read_game(events, summary, cfg)
    n, Rcfg, TP = G["n"], G["R"], G["tp"]
    raw_rw = summary.get("rewards") or (run.get("result") or {}).get("rewards") or {}
    rewards = [float(raw_rw[str(s)]) if str(s) in raw_rw else None for s in range(n)]
    score_chk, warns = verify(G, rewards)
    for w in warns:
        print("warning:", w, file=sys.stderr)
    full_model, model, date = model_and_date(b)
    seed = run.get("seed", cfg.get("seed"))
    RD = G["rounds"]
    R = len(RD)                                         # rounds played
    N = max(Rcfg, RD[-1]["r"], R)                       # x axis: every configured round, played or not
    pad = lambda vals: list(vals) + [None] * (N - len(vals))
    P = [pname(s) for s in range(n)]
    nw = numword(n)
    everyone = "both players" if n == 2 else f"all {nw} players"

    # ---- per round numbers (payoff_i = 2 min - e_i + 6 = (min + 6) - (e_i - min))
    eff = [[rd["efforts"][s] if s < len(rd["efforts"]) else None for rd in RD] for s in range(n)]
    mins = [min(rd["efforts"]) for rd in RD]
    maxs = [max(rd["efforts"]) for rd in RD]
    means = [Fraction(sum(rd["efforts"]), len(rd["efforts"])) for rd in RD]
    coord = [len(set(rd["efforts"])) == 1 for rd in RD]
    pay = [[2 * mins[i] - eff[s][i] + 6 for i in range(R)] for s in range(n)]
    waste = [[eff[s][i] - mins[i] for i in range(R)] for s in range(n)]
    waste_r = [sum(waste[s][i] for s in range(n)) for i in range(R)]
    cum = [[] for _ in range(n)]
    for s in range(n):
        acc = 0
        for i in range(R):
            acc += pay[s][i]
            cum[s].append(acc)
    score = [cum[s][-1] for s in range(n)]
    assert score == score_chk
    best_round, secure_round = 2 * HI - HI + 6, 2 * LO - LO + 6          # 13 and 7 (rules: Theory Benchmark)
    sucker = 2 * LO - HI + 6                                             # effort 7 against a 1: 1
    tot = sum(score)
    tot_best, tot_secure = best_round * n * R, secure_round * n * R
    loss_min = sum(n * (HI - m) for m in mins)                           # lost to a group minimum below 7
    loss_waste = sum(waste_r)                                            # lost to effort above the minimum
    assert tot_best - loss_min - loss_waste == tot
    C = sum(coord)
    co_rs = [i + 1 for i in range(R) if coord[i]]
    mis_rs = [i + 1 for i in range(R) if not coord[i]]
    sum_min = sum(mins)

    # final run in which every player chose one and the same effort
    k0 = R - 1
    if coord[-1]:
        while k0 > 0 and coord[k0 - 1] and mins[k0 - 1] == mins[-1]:
            k0 -= 1
        settled_len, L = R - k0, mins[-1]
    else:
        settled_len, L = 0, None

    # ---- steps: per round the messages (logged order), then the efforts in seat order
    steps, idx_last = [], {}
    for i, rd in enumerate(RD):
        r = rd["r"]
        for t in rd["talk"]:
            if not 0 <= t["seat"] < n:
                print(f"warning: round {r}: message from seat {t['seat']} ignored", file=sys.stderr)
                continue
            t["step"] = len(steps)
            steps.append({"seat": t["seat"], "x": i, "phase": "TALK", "round": r, "text": t["text"], "_talk": t,
                          "head": f"Round {r} {DOT} public message", "kind": "talk",
                          "act": json.dumps({"action_id": "talk", "payload": {"text": t["text"]}}, ensure_ascii=False)})
        for s in range(n):
            e, m, p = eff[s][i], mins[i], pay[s][i]
            where = "at the minimum" if e == m else f"{e - m} above the minimum {m}"
            steps.append({"seat": s, "x": i, "phase": "DECIDE", "round": r, "effort": e, "kind": bucket(e),
                          "head": f"Round {r} {DOT} effort {e} {DOT} {where} {DOT} payoff {p}",
                          "act": json.dumps({"action_id": "effort", "payload": {"effort": e}})})
        idx_last[i] = len(steps) - 1
    tail = []                                           # messages of a round the recording did not finish
    if G["tail_talk"] and R < N:
        for t in G["tail_talk"]:
            if 0 <= t["seat"] < n:
                steps.append({"seat": t["seat"], "x": R, "phase": "TALK", "round": R + 1, "text": t["text"], "_talk": t,
                              "head": f"Round {R + 1} {DOT} public message", "kind": "talk",
                              "act": json.dumps({"action_id": "talk", "payload": {"text": t["text"]}}, ensure_ascii=False)})
                tail.append(len(steps) - 1)

    # ---- what each seat saw (rebuilt from the log; the saved prompt where there is one)
    def round_line(k):
        rd = RD[k]
        return (f"- Round {rd['r']}: efforts {rd['efforts']}, minimum {mins[k]}, payoffs "
                f"[{', '.join(str(pay[s][k]) for s in range(n))}].")

    def observation(st):
        s, r, i = st["seat"], st["round"], st["x"]
        sc = [cum[q][i - 1] if i > 0 else 0 for q in range(n)]
        lines = ["(Rebuilt from the game log; the exact prompt for this turn was not saved.)",
                 f"You are seat P{s} ({P[s]} on this page). Your role: PLAYER.",
                 (f"Current phase: TALK, round {r} of {Rcfg}. Each player sends {plural(TP, 'public, non-binding message', word=True)} before "
                  "the decision." if st["phase"] == "TALK" else
                  f"Current phase: DECIDE, round {r} of {Rcfg}. Every player secretly chooses an effort from {LO} to {HI}; the "
                  "efforts are revealed together after the last player has chosen."),
                 f"Payoff this round = 2 {TIMES} (the lowest effort in the group, yours included) {MINUS} your effort + 6.",
                 "Scores so far: " + ", ".join(f"P{q} {sc[q]}" for q in range(n)) + "."]
        hist = [round_line(k) for k in range(min(i, R))]
        lines += (["Earlier rounds:"] + hist) if hist else ["Earlier rounds: none yet."]
        msgs = RD[i]["talk"] if i < R else G["tail_talk"]
        if st["phase"] == "TALK":
            said = []
            for t in msgs:
                if t is st["_talk"]:
                    break
                said.append(t)
            lines += (["Messages this round so far:"] + [f"- P{t['seat']}: {t['text']}" for t in said]
                      if said else ["No one has spoken yet this round."])
        elif TP:
            lines += (["Messages this round:"] + [f"- P{t['seat']}: {t['text']}" for t in msgs]) if msgs else ["No messages this round."]
        return en("\n".join(lines))

    for st in steps:
        st["state"] = observation(st)
    prompt = run.get("prompt") or {}
    for st in steps:
        if (prompt.get("text") and prompt.get("seat") == st["seat"] and str(prompt.get("phase", "")).upper() == st["phase"]
                and re.search(r'"round": %d\b' % st["round"], prompt["text"])):
            st["state"] = en(common.state_part(prompt["text"]))      # the prompt saved with the run (the first decision)
            st["state_from_prompt"] = True
            break

    cs = attach_calls(steps, calls) if calls else {"tied": 0, "mism": 0, "retried": 0, "extra": 0, "reasons": {}, "skipped": 0}
    tied, mism = cs["tied"], cs["mism"]
    has_reason = any(st.get("thinking") for st in steps)
    from_prompt = sum(1 for st in steps if st.get("state_from_prompt"))

    # The environment stores each effort in the player's private "pending" field and does not clear it after the reveal, so
    # from round 2 on a saved prompt can show the previous round's effort there before the player has chosen. Check that
    # reading against the log before the page explains it.
    pend_seen = pend_prev = pend_think = 0
    for st in steps:
        if not st.get("state_from_prompt"):
            continue
        m = re.search(r"Your private info:\s*(\{[^\n]*\})", st["state"])
        try:
            info = json.loads(m.group(1)) if m else {}
        except ValueError:
            info = {}
        if isinstance(info, dict) and "pending" in info:
            pend_seen += 1
            i = st["x"]
            if i > 0 and str(info["pending"]) == str(eff[st["seat"]][i - 1]):
                pend_prev += 1
            if "pending" in (st.get("thinking") or ""):
                pend_think += 1
    if pend_seen and pend_prev < pend_seen:
        print(f"warning: {pend_seen - pend_prev} of {pend_seen} prompts show a private 'pending' value other than the seat's "
              "previous effort", file=sys.stderr)
    for st in steps:
        st.setdefault("thinking", "")
        for k in ("phase", "round", "text", "_talk", "effort", "state_from_prompt"):
            st.pop(k, None)

    # ---- feed: the rules, the messages, each round's reveal after its last decision
    feed = [{"x": 0, "step": 0, "who": "Rules",
             "text": f"{cap(nw)} players, {Rcfg} rounds. Each round every player secretly picks an effort from {LO} to {HI}; payoff = "
                     f"2 {TIMES} group minimum {MINUS} own effort + 6. All {HI}s pay {best_round} each, all {LO}s pay {secure_round} "
                     f"each, and effort {HI} against a {LO} pays {sucker}. "
                     + (f"Before each decision every player sends {plural(TP, 'public, non-binding message', word=True)}." if TP else
                        "The players cannot send messages.")}]
    for i, rd in enumerate(RD):
        for t in rd["talk"]:
            if "step" in t:
                feed.append({"x": i, "step": t["step"], "who": P[t["seat"]], "kind": "talk", "text": LDQ + t["text"] + RDQ})
        above = [s for s in range(n) if waste[s][i] > 0]
        feed.append({"x": i, "step": idx_last[i], "who": "Reveal",
                     "text": f"Round {rd['r']}: efforts " + " / ".join(str(eff[s][i]) for s in range(n))
                             + (" (Players 1 and 2)" if n == 2 else f" (Players 1 to {n})") + f"; group minimum {mins[i]}. "
                             + ("All efforts equal, no effort wasted. " if coord[i] else
                                "Above the minimum: " + ", ".join(f"{P[s]} (+{waste[s][i]})" for s in above)
                                + f", {qty(waste_r[i], 'unit')} of effort wasted in all. ")
                             + "Payoffs " + " / ".join(str(pay[s][i]) for s in range(n))
                             + "; scores " + " / ".join(fn(cum[s][i]) for s in range(n)) + "."})
    for k in tail:
        st = steps[k]
        feed.append({"x": st["x"], "step": k, "who": P[st["seat"]], "kind": "talk",
                     "text": LDQ + json.loads(st["act"])["payload"]["text"] + RDQ})

    # ---- chart 1: effort grid
    rows = P + ["Minimum", "Wasted"]
    gcells = [[None] * N for _ in rows]
    used = set()
    for i in range(R):
        for s in range(n):
            e = eff[s][i]
            used.add(bucket(e))
            gcells[s][i] = {"c": KINDS[bucket(e)][1], "t": str(e),
                            "tip": f"effort {e}, " + ("at the group minimum" if e == mins[i] else
                                                      f"{e - mins[i]} above the group minimum {mins[i]} (wasted)")
                                   + f"; payoff 2 {TIMES} {mins[i]} {MINUS} {e} + 6 = {pay[s][i]}; score {fn(cum[s][i])}"}
        m = mins[i]
        used.add(bucket(m))
        low = [s for s in range(n) if eff[s][i] == m]
        gcells[n][i] = {"c": KINDS[bucket(m)][1], "t": str(m),
                        "tip": f"group minimum {m}, set by " + (players_text(low, n) if len(low) < n else everyone)
                               + f"; a player at the minimum earns {m + 6}" + (", the most possible" if m == HI else f", {HI - m} less than at {HI}")}
        gcells[n + 1][i] = {"c": "hold" if waste_r[i] else "muted", "t": str(waste_r[i]),
                            "o": .9 if waste_r[i] else .55,
                            "tip": ("0 wasted: all efforts equal (an equilibrium)" if coord[i] else
                                    f"{qty(waste_r[i], 'unit')} of effort above the minimum, summed over the players; mean effort {fn(means[i])}")}
    for i in range(R, N):
        for q in range(len(rows)):
            gcells[q][i] = {"c": "muted", "t": "", "o": .25, "tip": "not played: the recording stops before this round"}
    legend = [[KINDS[k][0], KINDS[k][1]] for k in ("e7", "e46", "e23", "e1") if k in used]
    legend += [["Effort wasted above the minimum", "hold"]] if loss_waste else []
    legend += [["None wasted (all equal)", "muted"]] if C else []
    if C == R and len(set(mins)) == 1:
        gnote = f"Every player chose {mins[0]} in every round, so no effort was wasted."
    else:
        bits = []
        if mis_rs:
            bits.append(f"Efforts differed in {rounds_text(mis_rs)}, wasting {qty(loss_waste, 'unit')} of effort in all")
        if C:
            lv = runs([mins[i] if coord[i] else None for i in range(R)])
            bits.append("all efforts were equal in " + "; ".join(f"{rounds_text(list(range(a, z + 1)))} (all {v})" for a, z, v in lv if v is not None))
        gnote = cap("; ".join(bits)) + "."
    sub1 = (f"One column per round. A player's cell shows the effort it chose (secret until all had chosen); the Minimum row is the "
            f"group's lowest effort, which sets everyone's payoff. Colors: green {HI}, the payoff-dominant level; blue 4 to 6; yellow "
            f"2 or 3; orange {LO}, the secure level. The Wasted row adds up the effort chosen above the minimum: each unit costs its "
            f"chooser one point and raises no one's payoff. " + gnote + " Hover a column for payoffs and scores.")
    charts = [{"title": "Effort per player, round by round", "sub": sub1, "height": 30 * len(rows) + 30,
               "grid": {"rows": rows, "cells": gcells, "rowHeight": 28 if n <= 8 else 22, "legend": legend}, "bars": []}]

    # Charts 2 and 3 only when something moves: if every player chose one and the same effort in every round, their lines and
    # bars would all be flat at that level, and the grid and the tiles already say it.
    flat = C == R and len(set(mins)) == 1

    # ---- chart 2: minimum, mean and highest effort
    if not flat:
        all_int = all(x.denominator == 1 for x in means)
        mr = runs(mins)
        enote = ("Group minimum in " + "; ".join(f"{rounds_text(list(range(a, z + 1)))}: {v}" for a, z, v in mr) + ". "
                 + f"Mean effort {fn(means[0])} in round 1 and {fn(means[-1])} in round {R}.")
        charts.append({"title": "Group minimum against the average and highest effort",
                       "sub": "The gap between the minimum and the higher lines is effort spent above the minimum, which is wasted. "
                              f"Dashed: {HI}, the payoff-dominant level ({best_round} each when all choose it), and {LO}, the secure "
                              f"level that guarantees {secure_round}. Every common effort level is an equilibrium. " + enote,
                       "height": 230, "fmt": "int" if all_int else "price", "bars": [],
                       "lines": [{"name": f"Payoff-dominant ({HI})", "color": "c3", "dash": True, "thin": True, "v": [HI] * N},
                                 {"name": f"Secure ({LO})", "color": "c2", "dash": True, "thin": True, "v": [LO] * N},
                                 {"name": "Highest effort", "color": "hold", "thin": True, "dots": True, "v": pad(maxs)},
                                 {"name": "Average effort", "color": "ink", "thin": True, "dots": True, "v": pad([num(x) for x in means])},
                                 {"name": "Group minimum", "color": "c1", "area": True, "dots": True, "end": True, "v": pad(mins)}]})

    # ---- chart 3: payoff per round (players with identical payoffs share one bar; ink is kept for the 'Minimum + 6' line, so
    # more than three different payoff paths are shown as the lowest, average and highest payoff)
    BAR_COLORS = ["c1", "c4", "hold"]
    if not flat:
        pay_groups = {}
        for s in range(n):
            pay_groups.setdefault(tuple(pay[s]), []).append(s)
        if len(pay_groups) <= len(BAR_COLORS):
            bars3 = []
            for gi, (path, seats) in enumerate(sorted(pay_groups.items(), key=lambda kv: kv[1][0])):
                nm = P[seats[0]] if len(seats) == 1 else cap(players_text(seats, n)) + " (identical)"
                bars3.append({"name": nm, "color": BAR_COLORS[gi], "v": pad(list(path))})
            bar_words = ("Bars: each player's payoff"
                         + (" (players with the same payoff in every round share a bar)" if len(pay_groups) < n else ""))
        else:
            lo_p = [min(pay[s][i] for s in range(n)) for i in range(R)]
            hi_p = [max(pay[s][i] for s in range(n)) for i in range(R)]
            av_p = [Fraction(sum(pay[s][i] for s in range(n)), n) for i in range(R)]
            bars3 = [{"name": "Lowest payoff", "color": "hold", "v": pad(lo_p)},
                     {"name": "Average payoff", "color": "c1", "v": pad([num(x) for x in av_p])},
                     {"name": "Highest payoff", "color": "c4", "v": pad(hi_p)}]
            bar_words = "Bars: the lowest, average and highest payoff among the players (hover a round for each player's)"
        pay_all_int = all(isinstance(v, int) for bb in bars3 for v in bb["v"] if v is not None)
        cap_line = [m + 6 for m in mins]
        pnote = (f"Over the game, the group lost {qty(loss_min, 'point')} to minimums below {HI} and "
                 f"{qty(loss_waste, 'point')} to effort above the minimum, against all-{HI} play.")
        charts.append({"title": "Payoff each round",
                       "sub": f"{bar_words}. A payoff is (minimum + 6) {MINUS} (own effort {MINUS} minimum): the solid line, minimum + "
                              "6, is what a player at the minimum earns, and a bar below it shows wasted effort. Dashed: "
                              f"{best_round} when everyone chooses {HI}, and {secure_round}, the floor that effort {LO} guarantees. "
                              + pnote,
                       "height": 230, "fmt": "int" if pay_all_int else "price", "bars": bars3,
                       "lines": [{"name": f"All choose {HI} ({best_round})", "color": "c3", "dash": True, "thin": True,
                                  "v": [best_round] * N},
                                 {"name": f"Secure floor ({secure_round})", "color": "c2", "dash": True, "thin": True,
                                  "v": [secure_round] * N},
                                 {"name": "Minimum + 6", "color": "ink", "thin": True, "step": True, "end": True, "v": pad(cap_line)}]})

    # ---- chart 4: scores (players with identical paths share one line)
    groups = {}
    for s in range(n):
        groups.setdefault(tuple(cum[s]), []).append(s)
    # A players' line that equals a benchmark at every round would hide the dashed line under it and stack the two end labels:
    # the benchmark is then drawn once, as the players' line, and named in its legend entry.
    bench = [(HI, best_round, "c3"), (LO, secure_round, "c2")]
    bench_v = [[pr * (i + 1) for i in range(N)] for _, pr, _ in bench]
    merged = {}                                         # benchmark index -> (group's seats)
    plines = []
    if len(groups) <= len(PATH_COLORS):
        for gi, (path, seats) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0])):
            v = pad(list(path))
            nm = P[seats[0]] if len(seats) == 1 else cap(players_text(seats, n)) + " (identical)"
            hit = next((bi for bi in range(len(bench)) if bench_v[bi] == v and bi not in merged), None)
            if hit is not None:
                merged[hit] = seats
                e = bench[hit][0]
                nm = (f"{cap(everyone)} = all {e} every round" if len(seats) == n else
                      f"{P[seats[0]] if len(seats) == 1 else cap(players_text(seats, n))} (same as all {e} every round)")
            plines.append({"name": nm, "color": PATH_COLORS[gi], "dots": True, "end": True, "v": v})
    lines4 = [{"name": f"All {e} every round", "color": col, "dash": True, "end": True, "v": bench_v[bi]}
              for bi, (e, _, col) in enumerate(bench) if bi not in merged] + plines
    if len(groups) > len(PATH_COLORS):
        lines4 += [{"name": "Highest score", "color": "ink", "dots": True, "end": True, "v": pad([max(cum[s][i] for s in range(n)) for i in range(R)])},
                   {"name": "Average score", "color": "c1", "dots": True, "end": True,
                    "v": pad([num(Fraction(sum(cum[s][i] for s in range(n)), n)) for i in range(R)])},
                   {"name": "Lowest score", "color": "hold", "dots": True, "end": True, "v": pad([min(cum[s][i] for s in range(n)) for i in range(R)])}]
    sc_all_int = all(isinstance(v, int) for l in lines4 for v in l["v"] if v is not None)
    def on_line(bi):                                    # how the text names a players' line that is also benchmark bi
        e, pr, _ = bench[bi]
        return (f"so the players' line is also the all-{e} benchmark ({pr} a round, {fn(pr * Rcfg)} after {Rcfg} rounds), "
                "drawn once" if bi in merged else f"so the players' line runs on the dashed all-{e} line")
    if flat:
        L0 = mins[0]
        snote = (f"Every payoff was {L0 + 6}"
                 + (f", the most the game allows, {on_line(0)}" if L0 == HI else
                    f", what effort {LO} guarantees, {on_line(1)}" if L0 == LO else
                    f", {HI - L0} a round less than all-{HI} play")
                 + f"; every player ended with {fn(score[0])}.")
    else:
        snote = (f"Every player ended with {fn(score[0])}." if len(groups) == 1 else
                 "Final scores: " + ", ".join(f"{P[s]} {fn(score[s])}" for s in range(n)) + ".")
        for bi, seats in sorted(merged.items()):
            e, pr, _ = bench[bi]
            who = cap(everyone) if len(seats) == n else cap(players_text(seats, n))
            snote += (f" {who} earned {pr} every round, so {'that line is' if len(seats) == 1 else 'their line is'} also the "
                      f"all-{e} benchmark, drawn once.")
    kept = [bench[bi] for bi in range(len(bench)) if bi not in merged]
    dashed = ("" if not kept else
              f"Dashed: the score after each round if everyone chose {HI} every round ({best_round} a round, "
              f"{fn(best_round * Rcfg)} after {Rcfg} rounds) or {LO} every round ({secure_round} a round, {fn(secure_round * Rcfg)}). "
              if len(kept) == 2 else
              f"Dashed: the score after each round if everyone chose {kept[0][0]} every round ({kept[0][1]} a round, "
              f"{fn(kept[0][1] * Rcfg)} after {Rcfg} rounds). ")
    charts.append({"title": "Score so far",
                   "sub": "Each player's running total of payoffs, its score. " + dashed + snote,
                   "height": 230, "fmt": "int" if sc_all_int else "price", "bars": [], "lines": lines4})

    # ---- the talk: did every message name effort HI as the choice (only HI), and was it then chosen in every round?
    def all_promise(rds):
        return all({t["seat"] for t in rd["talk"]} == set(range(n)) and all(promises_only(t["text"], HI) for t in rd["talk"])
                   for rd in rds)
    pscope = None                                       # 'every': every round's messages; 'first': round 1's
    if TP and settled_len == R and L == HI:
        pscope = "every" if all_promise(RD) else ("first" if all_promise(RD[:1]) else None)

    # ---- tiles
    if len(set(mins)) == 1:
        v1 = f"{mins[0]} every round" if R > 1 else f"{mins[0]}"
    elif one_way(mins):                                 # first -> last only for a path that only falls or only rises
        v1 = f"{mins[0]} {ARROW} {mins[-1]}"
    else:
        v1 = f"between {min(mins)} and {max(mins)}"
    mr = runs(mins)
    t1 = {"k": "Group minimum", "v": v1,
          "s": ("" if len(mr) == 1 else cap("; ".join(f"{rounds_text(list(range(a, z + 1)))}: {v}" for a, z, v in mr)) + ". ")
               + f"Sum of the round minimums {sum_min} of {HI * R} possible"
               + (f" ({R} played of {Rcfg})" if R < Rcfg else "") + "."}
    if len(set(mins)) > 1:
        t1["spark"] = mins
    if C == R:
        co_s = (f"Every player chose {mins[0]} in every round" if len(set(mins)) == 1 else
                "All efforts were equal in every round: " + "; ".join(f"{rounds_text(list(range(a, z + 1)))} all {v}" for a, z, v in mr))
    elif C == 0:
        co_s = f"Efforts differed in every round; the most wasted in one round was {qty(max(waste_r), 'unit')} of effort"
    else:
        lv = runs([mins[i] if coord[i] else None for i in range(R)])
        co_s = ("All equal in " + "; ".join(f"{rounds_text(list(range(a, z + 1)))} (all {v})" for a, z, v in lv if v is not None)
                + f"; efforts differed in {rounds_text(mis_rs)}")
    t2 = {"k": "Rounds coordinated", "v": f"{C} of {R}",
          "s": co_s + ". A round counts when every player chose the same effort, an equilibrium of the one-shot game."
               + (f" Every message {'in every round' if pscope == 'every' else 'in round 1'} promised {HI}, and no promise was "
                  "broken." if pscope else "")}
    t3 = {"k": "Total payoff, all players", "v": f"{fn(tot)} of {fn(tot_best)}",
          "s": f"{pct(tot, tot_best)} of all-{HI} play ({best_round} each a round); all-{LO} play earns {fn(tot_secure)}. "
               + ("Nothing was lost." if not (loss_min or loss_waste) else
                  f"Lost to minimums below {HI}: {fn(loss_min)}; to effort above the minimum: {fn(loss_waste)}.")}
    lo_s, hi_s = min(score), max(score)
    win = [s for s in range(n) if score[s] == hi_s]
    if lo_s == hi_s:
        s4 = f"{cap(everyone)} tie" + (", as the log's winners list says." if isinstance(G["winners"], list) and len(G["winners"]) == n else ".")
    else:
        s4 = (f"{players_text(win, n)} " + ("leads" if len(win) == 1 else "tie for the lead") + ". "
              + ("Scores: " + ", ".join(f"{P[s]} {fn(score[s])}" for s in range(n)) + ". " if n <= 6 else "")
              + "A lower score means more effort spent above the minimum.")
    t4 = {"k": "Final scores", "v": f"{fn(lo_s)} each" if lo_s == hi_s else f"{fn(lo_s)} to {fn(hi_s)}",
          "s": s4 + f" All-{HI} play earns {fn(best_round * R)} each over {plural(R, 'round')}."}
    tiles = [t1, t2, t3, t4]

    # ---- title, lede, foot
    Rw = numword(R)
    if settled_len == R:
        if L == HI:
            title = ((f"{cap(everyone)} promise effort {HI} in every round's messages and keep the promise in all {Rw} rounds"
                      if pscope == "every" else
                      f"{cap(everyone)} promise effort {HI} in the first round's messages and choose it in every one of the {Rw} "
                      "rounds" if pscope == "first" else
                      f"{cap(everyone)} choose effort {HI} in every one of the {Rw} rounds")
                     + f": the group coordinates on the payoff-dominant equilibrium from the start, and each player earns "
                       f"{best_round} a round")
        elif L == LO:
            title = (f"{cap(everyone)} choose the secure effort {LO} in every round: coordinated, but on the worst equilibrium, "
                     f"{secure_round} each a round instead of {best_round}")
        else:
            title = (f"{cap(everyone)} choose effort {L} in every round: coordinated below the top, {L + 6} each a round instead of "
                     f"{best_round}")
    elif settled_len >= 2:
        kf = k0                                         # first round of the final stretch with this minimum
        while kf > 0 and mins[kf - 1] == L:
            kf -= 1
        what = (f": coordination unravels to the secure effort, {secure_round} each a round" if L == LO else
                f", the payoff-dominant level, {best_round} each a round" if L == HI else f", {L + 6} each a round")
        if mins[0] != L and one_way(mins[:kf + 1]):
            verb = "falls" if mins[0] > L else "rises"
            title = (f"The group minimum {verb} from {mins[0]} in round 1 to {L}" + (f" by round {kf + 1}" if kf < k0 else "")
                     + f", and from round {k0 + 1} on {everyone} choose {L}" + what)
        elif mins[0] != L:                              # the way there is not one-way: name its low and high points
            title = f"The group minimum {path_phrase(mins[:k0])}; from round {k0 + 1} on {everyone} choose {L}" + what
        else:
            dip = min(mins[:k0])
            title = ((f"After {plural(k0, 'round', word=True)} in which the efforts differ" if not any(coord[:k0]) else
                      f"After {plural(k0, 'unsettled round', word=True)}")
                     + (f" (the group minimum drops to {dip} in {ranges_text([i + 1 for i in range(k0) if mins[i] == dip])})"
                        if dip < L else "")
                     + f", {everyone} choose effort {L} from round {k0 + 1} to {R}" + what)
    else:
        p = 0                                           # rounds 1..p: everyone chose one and the same effort
        while p < R and coord[p] and mins[p] == mins[0]:
            p += 1
        waste_txt = f"the players waste {qty(loss_waste, 'unit')} of effort above the minimum" if loss_waste else ""
        if p >= 2 and 2 * p >= R:                       # a long coordinated start that breaks before the end
            L0 = mins[0]
            title = f"{cap(everyone)} choose {L0} in {rounds_text(list(range(1, p + 1)))}"
            if p == R - 1:
                ef = RD[-1]["efforts"]
                dev = [s for s in range(n) if ef[s] != L0]
                vals = {ef[s] for s in dev}
                if len(vals) == 1:
                    v = next(iter(vals))
                    verb = ("drops" if v < L0 else "rises") if len(dev) == 1 else ("drop" if v < L0 else "rise")
                    title += f"; in round {R} {players_text(dev, n)} {verb} to {v}"
                    wasters = [s for s in range(n) if s not in dev] if v < L0 else dev     # all the waste is in round R
                    if waste_txt and wasters:
                        title += (f", and {players_text(wasters, n)} {'waste' if len(wasters) > 1 else 'wastes'} "
                                  if v < L0 else ", wasting ") + f"{qty(loss_waste, 'unit')} of effort above the minimum"
                        waste_txt = ""
                else:
                    title += (f"; in round {R} the efforts are " + " / ".join(str(x) for x in ef)
                              + f" (minimum {mins[-1]})")
            else:
                Ct = sum(coord[p:])
                title += (f"; then the group minimum {path_phrase(mins[p:], p + 1)}, and the efforts "
                          + (f"all match in {Ct} of the last {plural(R - p, 'round', word=True)}" if Ct else "never all match again"))
            title += f"; {waste_txt}" if waste_txt else ""
        else:
            title = f"The group minimum {path_phrase(mins)}"
            title += (f"; the efforts all match in {'only ' if 0 < C < R / 2 else ''}{C} of {R} rounds" if C else
                      "; the efforts never all match")
            title += f", and {waste_txt}" if waste_txt else ""
    if R < Rcfg:
        title += f" (the recording stops after round {R} of {Rcfg})"
    lede = (f"{cap(nw)} players play {numword(Rcfg)} rounds of a minimum-effort game: each round every player secretly picks an effort "
            f"from {LO} to {HI} and earns 2 {TIMES} (the lowest effort in the group) {MINUS} its own effort + 6, which adds up to its "
            f"score. If everyone picks {HI}, each earns {best_round}, the most possible, but effort above the group minimum is wasted, "
            f"and picking {LO} guarantees {secure_round} whatever the others do. "
            + (f"Before each decision every player sends {'one public, non-binding message' if TP == 1 else f'{TP} public, non-binding messages'}."
               if TP else "The players cannot talk; they only see the efforts revealed after each round."))

    runlog = b.get("runlog") or ""
    fallbacks = len(re.findall(r"\[LLM fallback\]", runlog))
    parse_fail = runlog.count("Could not parse a legal JSON action")
    if fallbacks or parse_fail:
        fb = (f"The recorder's run log reports {plural(max(fallbacks, 1 if parse_fail else 0), 'decision')} where no legal action "
              "could be parsed from the model's answer and a fallback action was played instead"
              + ("; the matched calls show which. " if tied else "; it does not say which decision. "))
    else:
        fb = "The recorder's run log reports no fallback action. " if runlog else ""
    cfg_bits = ", ".join(x for x in [f"seed {seed}" if seed is not None else "", plural(n, "player", word=True),
                                     plural(Rcfg, "round"),
                                     (f"{plural(TP, 'public message', word=True)} per player before each decision" if TP else "no messages")]
                         if x)
    if has_reason:
        retry = ""
        if cs["extra"]:
            why = []
            for words, ids in cs["reasons"].items():
                why.append(words + (", " + join_and(LDQ + x + RDQ for x in ids) if ids else ""))
            retry = (f"; {plural(cs['retried'], 'decision')} needed a retry after " + " or ".join(why) + ", so "
                     + (f"{cs['extra']} calls are rejected attempts" if cs["extra"] > 1 else "one call is a rejected attempt")
                     + ", whose reasoning the step shows above the final attempt's")
        rs = (f"Observations and reasoning come from the saved model calls: {tied} of {len(steps)} steps are matched to one of "
              f"the {len(calls)} calls" + retry
              + (f"; {plural(cs['skipped'], 'call')} match no decision" if cs["skipped"] > 0 else "")
              + (f"; in {mism} the matched call's answer is not the logged action, which the step notes" if mism else "") + ". ")
    else:
        rs = ("Reasoning was not saved in this recording (it has no call log), so no step shows the model's reasoning. " if not calls else
              "The call log has no reasoning, so no step shows the model's reasoning. ")
    obs = ("" if from_prompt == len(steps) else
           ("The observation of the first decision is the prompt saved with the run; the others are rebuilt from the log. "
            if from_prompt == 1 and not tied else
            ("The observations are rebuilt from the log. " if not from_prompt else
             f"{from_prompt} of {len(steps)} observations are saved prompts; the others are rebuilt from the log. ")))
    if not pend_seen:
        pend = ""
    elif pend_prev == pend_seen:
        pend = ('From round 2 on, a player\'s private info in the prompt shows a "pending" field before it has chosen: that is its '
                "effort from the previous round, which the environment stores there and does not clear after the reveal, not a "
                "choice made in advance" + (f" (the reasoning of {pend_think} of the {len(steps)} steps mentions it, mostly unsure what it is)"
                                            if pend_think else "") + ". ")
    else:
        pend = (f'{pend_seen} prompts show a "pending" field in the player\'s private info, set by the environment; in '
                f"{pend_seen - pend_prev} of them it is not the player's previous effort. The page shows the prompts as sent. ")
    foot = (f"Data: the {date} recording of eco_min_effort ({cfg_bits}); every player is played by {model}"
            + (f" ({full_model})" if full_model and full_model != model else "") + ". " + rs + obs + pend
            + "Every number on this page is computed from the game log: the round_result events (efforts in seat order, the logged "
              "minimum) and the talk events (the messages). Payoffs are not logged; they are recomputed with the rule 2 " + TIMES
            + " minimum " + MINUS + " own effort + 6 and checked against the logged minimums, the summary's rewards, the winners and "
              f"the game's own totals (sum of round minimums {sum_min}, coordinated rounds {C})"
            + (". " if not warns else f"; {plural(len(warns), 'check')} disagree (printed by the builder; the page shows the logged efforts). ")
            + fb
            + f"Player k is seat P(k{MINUS}1) in the log and the prompts; each round the players "
            + ("send their messages in that order, then " if TP else "")
            + "choose their efforts in that order, and none sees another's effort before the reveal. Benchmarks from the rules: "
              f"every profile in which all choose the same effort is a Nash equilibrium ({HI - LO + 1} of them); all {HI} is payoff "
              f"dominant ({best_round} each a round, {fn(tot_best)} for the group over {plural(R, 'round')}); all {LO} is the secure "
              f"outcome ({secure_round} each a round, since effort {LO} guarantees {secure_round}); effort {HI} against someone's {LO} "
              f"earns {sucker}. A payoff splits exactly into (minimum + 6) {MINUS} (own effort {MINUS} minimum), so the group's "
              f"shortfall from all-{HI} play is {fn(loss_min)} from minimums below {HI} plus {fn(loss_waste)} from effort above the "
              "minimum. Human benchmark (rules): in Van Huyck, Battalio and Beil (1990), large groups of 14 to 16 players converge to "
              "the minimum effort within a few rounds, while two-player groups often coordinate on high effort. Sources: Van Huyck, "
              "Battalio and Beil (1990), American Economic Review; Goeree and Holt (2005), Games and Economic Behavior.")

    kinds = {k: v for k, v in KINDS.items() if any(st["kind"] == k for st in steps)}
    d = {"id": GID, "name": NAME, "title": title, "lede": lede, "seatNames": P, "tiles": tiles, "foot": foot,
         "x": {"n": N, "labels": [f"R{i + 1}" for i in range(N)]}, "charts": charts, "kinds": kinds, "steps": steps,
         "feed": feed, "feedTitle": "Round by round",
         "feedSub": ("Each round's public messages, then the reveal of the efforts (secret until every player had chosen), with "
                     "the minimum, payoffs and scores. " if TP else
                     "Each round's reveal of the efforts (secret until every player had chosen), with the minimum, payoffs and "
                     "scores. ") + "The selected step is highlighted; later entries are hidden."}
    d = en_all(d)
    d["eyebrow"] = (f"social_game_env {DOT} {GID} {DOT} {model}" + (", reasoning on" if has_reason else "")
                    + f" {DOT} recorded {date}")
    return d


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    common.write(build(src, calls), out, date=recorded_date(src))
