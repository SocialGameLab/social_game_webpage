"""Replay data for eco_public_goods (repeated linear public goods game, the voluntary contribution mechanism; Isaac, Walker and
Thomas 1984, Isaac and Walker 1988, Ledyard 1995).

usage: python3 -I eco_public_goods.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md, run.log) or a
          file inside one.
  OUT_DIR where <OUT_DIR>/eco_public_goods.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). Calls are tied to
          decisions in time order: a call belongs to the next decision of the same seat, phase and round (read from its prompt:
          "You are seat Pk", "Current phase: X", the public state's "round"; a prompt that does not show the phase is placed by
          its answer's action_id; a value that cannot be read agrees with anything). Within one decision's calls the first whose
          answer is the logged action (the message text, or a legal contribution of the logged amount) is the decision's call;
          earlier ones were failed attempts (format retries) and their reasoning is kept above it; the foot counts the retries,
          and one that asked for another amount than the game applied is named in the step title, the grid tips and the
          said-against-gave note. With no such answer the last
          call of the decision is used (what the agent ended with) and the step says the game applied something else. Its reasoning becomes the
          step's thinking and its prompt the observation.

Each round has talk_passes public messages per player (TALK phase, seats in order, logged as talk events in time order) and then
one secret contribution per player (CONTRIBUTE phase, seats asked in order P0, P1, ...; nobody sees another's contribution before
the reveal). The log has no event per contribution: all of them are in the round's round_result event (contributions in seat
order, total_contribution, group_return_each). Payoffs are not logged per player; they are recomputed exactly (fractions) with
the rule of the game, endowment - own contribution + mpcr x total, and checked against the logged totals and group returns, the
hidden game_end total, the summary's rewards and the winners; any disagreement is printed as a warning. A reply the game cannot
use becomes a contribution of 0 (the phase's default and the only contribution the recorder's random fallback can pick), so the
run log's fallback notice is read and reported. Nothing about one particular recording (seed, number of players or rounds,
parameters, outcome) is assumed."""
import json, os, re, sys, unicodedata
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_public_goods"
NAME = "Public goods"
DEFAULT_DATE = "2026-11-09"
DEFAULT_CONTRIB = 0                                 # the CONTRIBUTE phase's default_action (rules: Invalid actions -> amount 0)
MINUS, TIMES, DOT, LDQ, RDQ, DOTS = "\u2212", "\u00d7", "\u00b7", "\u201c", "\u201d", "\u2026"
# contribution buckets (grid cells, step kinds): nothing, part of the endowment, all of it
KINDS = {"full": ["Contributes everything", "c3"],
         "part": ["Contributes part", "c1"],
         "none": ["Contributes nothing", "c2"],
         "talk": ["Public message", "muted"]}
PATH_COLORS = ["c1", "c4", "hold", "ink", "muted"]  # players' series (green and orange are kept for the two benchmarks)
# said-against-gave grid: a kept word is neutral gray so that only the differences stand out, and green (chart 1's "all
# tokens") is not reused with another meaning
MSG_COLORS = {"eq": "hold", "lt": "c2", "gt": "c1", "several": "c4", "none": "muted"}
ARROW = "→"
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
    b["runlog"] = open(rl, encoding="utf-8", errors="replace").read() if os.path.exists(rl) else None
    return b


def model_and_date(b):
    """(full model id, short model name, recording date YYYY-MM-DD), from run.json (meta.json and summary.json as fallbacks)."""
    run, meta, summ = b.get("run") or {}, b.get("meta") or {}, b.get("summary") or {}
    full = str(run.get("model") or meta.get("model") or summ.get("model") or "")
    date = str(run.get("recorded_at") or meta.get("recorded_at") or summ.get("recorded_at") or DEFAULT_DATE)[:10]
    return full, (full.split("/")[-1] if full else "an unnamed model"), date


def recorded_date(src):
    return model_and_date(load_source(src))[2]


# ------------------------------------------------------------------------------------------------ text helpers
def fn(v):
    """A token amount, payoff or score: an integer with thousands separators, anything else with up to four decimals (exact
    for the usual mpcr values such as 0.3 or 0.75)."""
    v = Fraction(v)
    s = f"{abs(int(v)):,}" if v.denominator == 1 else f"{abs(float(v)):,.4f}".rstrip("0").rstrip(".")
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


def join_and(xs):
    xs = [str(x) for x in xs]
    if not xs:
        return ""
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def rounds_text(rs, cap_first=False):
    """[2, 3, 4] -> 'rounds 2 to 4'; [1] -> 'round 1'; [1, 3, 5] -> 'rounds 1, 3 and 5'; [1, 2, 5, 6, 7] -> 'rounds 1, 2 and
    5 to 7'."""
    rs = sorted(set(rs))
    if not rs:
        w = "no round"
    elif len(rs) == 1:
        w = f"round {rs[0]}"
    else:
        parts, a = [], rs[0]
        for k in range(1, len(rs) + 1):
            if k == len(rs) or rs[k] != rs[k - 1] + 1:
                z = rs[k - 1]
                parts += [f"{a} to {z}"] if z - a >= 2 else [str(x) for x in range(a, z + 1)]
                if k < len(rs):
                    a = rs[k]
        w = "rounds " + join_and(parts)
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


def players_short(seats, n):
    """A short series name: [2] -> 'Player 3'; [0, 1, 3] -> 'Players 1, 2, 4'; [0, 1, 2, 3, 4] (of 8) -> 'Players 1-5';
    every seat -> 'All four players'."""
    s = sorted(seats)
    if len(s) == 1:
        return pname(s[0])
    if len(s) == n:
        return cap(players_text(s, n))
    if len(s) > 3 and s == list(range(s[0], s[-1] + 1)):
        return f"Players {s[0] + 1}\u2013{s[-1] + 1}"
    return "Players " + ", ".join(str(x + 1) for x in s)


def _ws(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


TIP_ROW = 90        # a grid tooltip shows one line per row (no wrapping): row name + tip are kept under about this many chars


def short_quote(text, room, at=None):
    """text in curly quotes, starting at the words at (when found; "..." marks the cut before them) and cut at a word to fit
    room characters (an ellipsis marks the cut); "" when room is too small for a useful quote."""
    t = _ws(text)
    tn = t.replace("\u2019", "'").replace("\u2018", "'")      # same length: positions agree
    pre = ""
    if at:
        k = tn.find(at)
        if k > 0:
            t, pre = t[k:], "\u2026"
    room -= 2 + len(pre)
    if room < 12:
        return ""
    if len(t) > room:
        cut = t[:room - 1]
        sp = cut.rfind(" ")
        if sp > room // 2:
            cut = cut[:sp]
        t = cut.rstrip(" ,.;:-\u2014") + "\u2026"
    return LDQ + pre + t + RDQ


def runs(vals):
    """Maximal runs of equal values: [(first round, last round, value)] with 1-based rounds."""
    out = []
    for i, v in enumerate(vals):
        if out and out[-1][2] == v:
            out[-1][1] = i + 1
        else:
            out.append([i + 1, i + 1, v])
    return [tuple(x) for x in out]


# ------------------------------------------------------------------------------------------------ amounts named in messages
# The speaker's own contribution named in a message: a first-person statement about this player's contribution ("I'll
# contribute 10", "I will contribute 0 again", "I'm planning to contribute 5", "I'll keep contributing 10", "I'm contributing
# 0"), with only a few adverbs allowed between the subject and the verb (so "I will not contribute 10" and "I'll match you if you
# contribute 10" name nothing; up to three other words joined by "and" may come first, as in "I'm willing to restart
# cooperation and contribute 10"), followed by a whole number (or "nothing" = 0, "everything" / "all my tokens" / "my full
# endowment" = the endowment). A number followed by "%", "more", "less", a range ("5 to 10", "5-10") or an alternative ("5 or
# 10") names nothing. Proposals for everyone ("let's all contribute 10") are not the speaker's own amount and are not read.
_SUBJ = (r"\bI(?:'ll|\s+will|\s+shall|'m\s+going\s+to|\s+am\s+going\s+to|'m\s+planning\s+to|\s+am\s+planning\s+to|\s+plan\s+to|"
         r"\s+intend\s+to|'m\s+willing\s+to|\s+am\s+willing\s+to|\s+commit\s+to|'m\s+committing\s+to|\s+aim\s+to)")
_ADV = (r"(?:\s+(?:also|again|still|now|then|happily|gladly|definitely|certainly|simply|just|only|keep|continue\s+to|continue))*"
        r"(?:(?:\s+(?!(?:not|never|no|if|unless|you|we|they|everyone|anyone|others?|everybody|anybody)\b)[A-Za-z']+){1,3}\s+and)?")
_VERB = r"\s+(?:contribute|contributing|put\s+in|putting\s+in|invest|investing)"
_PROG = r"\bI(?:'m|\s+am)(?:\s+(?:also|still|now|again))*\s+(?:contributing|putting\s+in|investing)"
_WORDNUM = "|".join(NUMW)
_AMOUNT = (r"\s+(?:(?:only|just|exactly|all|the\s+full|my\s+full|a\s+full)\s+)?(?:(?P<n>\d+|" + _WORDNUM + r")\b"
           r"(?![.,]\d)(?!\s*(?:%|percent|more|less|fewer|extra|additional|-|–|to\s+\d|or\s+\d|and\s+\d))"
           r"|(?P<zero>nothing)\b|(?P<all>everything|all\s+(?:of\s+)?my\s+(?:\d+\s+)?tokens|my\s+(?:whole|entire|full)\s+endowment|fully"
           r"|in\s+full)\b)")
_LATER = re.compile(r"\s*(?:tokens?\s+)?(?:(?:again|each|every|in|for|from|during|the|all|of)\s+){0,3}(?:next|later|future|coming|"
                    r"following|subsequent)\s+(?:rounds?|time)\b|\s*(?:tokens?\s+)?after\s+this\s+round\b", re.I)
_OWN = re.compile(r"(?:" + _SUBJ + _ADV + _VERB + r"|" + _PROG + r")" + _AMOUNT, re.I)


def named_amounts(text, endow):
    """The amounts (0 to the endowment) a message names for the speaker's own contribution (see _OWN), distinct and sorted, each
    with the words that name it. An amount followed at once by words for a later round ("I'll contribute 0 next round", "10
    in later rounds") is not this round's amount."""
    t = str(text or "").replace("’", "'").replace("‘", "'")
    out = {}
    for m in _OWN.finditer(t):
        if _LATER.match(t[m.end():]):
            continue                                # a plan for a later round, not for this round's decision
        if m.group("n") is not None:
            g = m.group("n").lower()
            v = int(g) if g.isdigit() else NUMW.index(g)
        elif m.group("zero"):
            v = 0
        else:
            v = endow
        if 0 <= v <= endow:
            out.setdefault(v, _ws(m.group(0)))
    return sorted(out.items())


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
            cs = [int(x) for x in (e.get("contributions") or [])]
            rounds.append({"r": int(e.get("round", len(rounds) + 1)), "c": cs,
                           "total_logged": e.get("total_contribution"), "ret_logged": e.get("group_return_each"),
                           "talk": talk[-1]})
            talk.append([])
        elif ev == "talk":
            seat = e.get("actor", e.get("seat"))
            if seat is None:
                stray.append(e)
                continue
            talk[-1].append({"seat": int(seat), "text": str(e.get("text") or "")})
        elif ev not in known and e.get("_visibility") != "hidden":
            print(f"warning: public event {ev!r} is not shown on the page", file=sys.stderr)
    if not rounds:
        raise SystemExit("no round_result event in the game log")
    if stray:
        print(f"warning: {len(stray)} message(s) without a sender are not shown", file=sys.stderr)
    n = int(par("n_players", cfg.get("n_players") or len(rounds[0]["c"])))
    E = par("endowment")
    if E is None:
        raise SystemExit("endowment is not in the log, the summary or the config")
    mp = par("mpcr")
    if mp is None:
        raise SystemExit("mpcr is not in the log, the summary or the config")
    return {"n": n, "R": int(par("rounds", len(rounds))), "tp": int(par("talk_passes", 0) or 0), "E": int(E),
            "mpcr": Fraction(str(mp)), "mpcr_raw": mp, "rounds": rounds, "tail_talk": talk[-1], "end": end,
            "winners": over.get("winners", summary.get("winners"))}


def verify(G, rewards):
    """Re-derive totals, group returns, payoffs and scores from the logged contributions; returns (scores, warnings). The page
    shows the logged contributions and the payoffs the rule gives for them."""
    n, E, mp, warn = G["n"], G["E"], G["mpcr"], []
    score = [Fraction(0)] * n
    for k, rd in enumerate(G["rounds"]):
        r, cs = rd["r"], rd["c"]
        if r != k + 1:
            warn.append(f"round_result {k}: round {r} logged, {k + 1} expected")
        if len(cs) != n:
            warn.append(f"round {r}: {len(cs)} contributions logged for {n} players")
        bad = [x for x in cs if not 0 <= x <= E]
        if bad:
            warn.append(f"round {r}: contributions {bad} outside 0 to {E}")
        T = sum(cs)
        if rd["total_logged"] is not None and int(rd["total_logged"]) != T:
            warn.append(f"round {r}: total_contribution {rd['total_logged']} logged, the contributions add up to {T}")
        if rd["ret_logged"] is not None and abs(float(rd["ret_logged"]) - float(mp * T)) > 1e-6:
            warn.append(f"round {r}: group_return_each {rd['ret_logged']} logged, mpcr x total = {float(mp * T)}")
        for s in range(min(n, len(cs))):
            score[s] += E - cs[s] + mp * T
        if G["tp"]:
            want = [s for _ in range(G["tp"]) for s in range(n)]
            got = [t["seat"] for t in rd["talk"]]
            if got != want:
                warn.append(f"round {r}: message senders {got}, {want} expected ({G['tp']} pass(es) in seat order)")
    if G["tail_talk"]:
        warn.append(f"{len(G['tail_talk'])} message(s) after the last round_result (a round without a result)")
    tot = sum(sum(rd["c"]) for rd in G["rounds"])
    if G["end"].get("total_contribution") is not None and int(G["end"]["total_contribution"]) != tot:
        warn.append(f"game_end total_contribution {G['end']['total_contribution']}, {tot} from the round_result events")
    for s in range(n):
        if rewards[s] is not None and abs(rewards[s] - float(score[s])) > 1e-6:
            warn.append(f"reward of seat {s}: {rewards[s]} in the summary, {float(score[s])} summed from the payoff rule")
    best = max(score)
    lead = [s for s in range(n) if score[s] == best]
    if isinstance(G["winners"], list) and sorted(int(w) for w in G["winners"]) != lead:
        warn.append(f"winners {G['winners']} logged, highest scores: seats {lead}")
    if len(G["rounds"]) < G["R"]:
        warn.append(f"only {len(G['rounds'])} of {G['R']} rounds logged: the recording stops early")
    return score, warn


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
    prompt that does not show the phase (an empty prompt, a format-retry note) is placed by its answer's action_id, so a TALK
    call never joins a contribution's window as an earlier attempt."""
    p = c.get("prompt") or ""
    m = re.search(r"Current phase:\s*([A-Za-z_]+)", p)
    phase = m.group(1).upper() if m else None
    if phase is None:
        a = _norm_answer(c.get("answer"))
        phase = {"talk": "TALK", "contribute": "CONTRIBUTE"}.get(a.get("action_id") if isinstance(a, dict) else None)
    i = p.find("Public state:")
    rm = re.search(r'"round":\s*(\d+)', p[i:]) if i >= 0 else None
    return phase, (int(rm.group(1)) if rm else None)


def requested_amount(a):
    """The amount a contribute answer asks for, and whether the game can apply it as it is: payload {"contribute": {"amount": k}}
    with nothing else and k a whole number from 0 to the endowment (checked by the caller). Returns (amount or None, well
    formed)."""
    if not isinstance(a, dict) or a.get("action_id") != "contribute":
        return None, False
    p = a.get("payload")
    if not isinstance(p, dict):
        return None, False
    inner = p.get("contribute")
    v = inner.get("amount") if isinstance(inner, dict) else p.get("amount")
    if isinstance(v, bool):
        return None, False
    if isinstance(v, float) and v == int(v):
        v = int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*\d+\s*", v):
        return int(v), False
    ok = (set(a) <= {"action_id", "payload"} and set(p) == {"contribute"} and isinstance(inner, dict)
          and set(inner) == {"amount"} and isinstance(v, int))
    return (v if isinstance(v, int) else None), ok


def _matches(call, st, E):
    a = _norm_answer(call.get("answer"))
    if st["phase"] == "TALK":
        if not isinstance(a, dict):                 # a discussion prompt may be answered with the bare message
            raw = (call.get("answer") or "").strip()
            return bool(raw) and _ws(raw) == _ws(st["text"])
        if a.get("action_id") != "talk":
            return False
        return _ws((a.get("payload") or {}).get("text")) == _ws(st["text"])
    if not isinstance(a, dict) or a.get("action_id") != "contribute":
        return False
    v, ok = requested_amount(a)
    return ok and 0 <= v <= E and v == st["c"]


def load_calls(path):
    """common.load_calls, plus for each call whether its conversation already held a reply of the model (an assistant
    message), read from the raw log: a format retry without one is a fresh call that does not see the rejected reply."""
    calls = common.load_calls(path)
    if calls:
        raw = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        if len(raw) == len(calls):
            for c, r in zip(calls, raw):
                if isinstance(r.get("messages"), list):
                    c["saw_reply"] = any(isinstance(m, dict) and m.get("role") == "assistant" for m in r["messages"])
    return calls


def _flat_amount(a):
    """A contribute answer whose payload is {"amount": k} (the amount without the {"contribute": ...} wrapper)."""
    return (isinstance(a, dict) and a.get("action_id") == "contribute" and isinstance(a.get("payload"), dict)
            and set(a["payload"]) == {"amount"})


def attach_calls(steps, calls, E):
    """Tie calls to steps in time order (see the module docstring). Returns (steps with a call, contribution steps whose call
    is not a legal contribution of the logged amount: [(step index, requested amount or None, answer)], talk steps whose
    call's text differs, decisions with earlier attempts: [{si, phase, n (earlier attempts), asked (their requested
    amounts), ok (their well-formedness), flat (all {"amount": k}), notice (the text the next prompt adds, when it is the
    previous prompt plus a note, else None), fresh (True when no retry saw an earlier reply, None when not known), hit}])."""
    meta = [call_meta(c) for c in calls]
    retries = []

    def fits(k, st, strict=False):
        seat, (ph, rnd) = calls[k].get("seat"), meta[k]
        if strict:                                      # resynchronising: the seat must be known and equal
            return seat == st["seat"] and ph in (None, st["phase"]) and rnd in (None, st["round"])
        return seat in (None, st["seat"]) and ph in (None, st["phase"]) and rnd in (None, st["round"])
    ptr = tied = 0
    replaced, talk_diff = [], 0
    for si, st in enumerate(steps):
        if ptr < len(calls) and not fits(ptr, st):     # resynchronise on a call that names this very decision
            ahead = next((k for k in range(ptr, min(len(calls), ptr + 80)) if fits(k, st, strict=True)), None)
            if ahead is None:
                continue
            print(f"warning: calls {ptr}..{ahead - 1} match no decision; skipped", file=sys.stderr)
            ptr = ahead
        j = ptr
        while j < len(calls) and fits(j, st):
            j += 1
            if _matches(calls[j - 1], st, E):           # a decision's calls end at its accepted answer (later calls of the same
                break                                   # seat, phase and round belong to another decision, e.g. a second pass)
        window = list(range(ptr, j))
        if not window:
            continue
        hit = next((k for k in window if _matches(calls[k], st, E)), None)
        use = hit if hit is not None else window[-1]   # no accepted answer: the last attempt is what the agent ended with
        if use > ptr:                                   # earlier attempts: format retries
            ans = [_norm_answer(calls[k].get("answer")) for k in range(ptr, use)]
            req = [requested_amount(a) for a in ans]
            notes = set()
            for k in range(ptr, use):
                p0, p1 = calls[k].get("prompt") or "", calls[k + 1].get("prompt") or ""
                notes.add(_ws(p1[len(p0):]) if p0 and p1.startswith(p0) and len(p1) > len(p0) else None)
            seen = [calls[k].get("saw_reply") for k in range(ptr + 1, use + 1)]
            retries.append({"si": si, "phase": st["phase"], "n": use - ptr, "asked": [v for v, _ in req],
                            "ok": [ok for _, ok in req], "flat": all(_flat_amount(a) for a in ans),
                            "notice": notes.pop() if len(notes) == 1 else None,
                            "fresh": (True if all(x is False for x in seen) else None), "hit": hit is not None})
        parts = []
        for k in range(ptr, use):
            c = calls[k]
            parts.append(f"[Earlier attempt, answer: {c['answer'][:300]}]\n{c.get('reasoning') or '(no reasoning returned)'}")
        c = calls[use]
        parts.append(("[Final attempt]\n" if parts else "") + (c.get("reasoning") or ("(no reasoning returned)" if parts else "")))
        if hit is None:
            if st["phase"] == "TALK":
                parts.append(f"[The log shows the message {LDQ}{st['text'][:200]}{RDQ}, not this answer: {c['answer'][:300]}]")
                talk_diff += 1
            else:
                v, ok = requested_amount(_norm_answer(c.get("answer")))
                legal = ok and v is not None and 0 <= v <= E
                if legal:
                    parts.append(f"[The log shows a contribution of {st['c']}, not the {v} this answer asks for.]")
                else:
                    parts.append(f"[This answer is not a legal contribution (an amount from 0 to {E} as "
                                 '{"contribute": {"amount": k}}), so the game applied a contribution of '
                                 f"{st['c']} instead (the phase's default is {DEFAULT_CONTRIB}): {c['answer'][:300]}]")
                replaced.append((si, v if legal or v is not None else None, c.get("answer") or "", legal))
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        obs = next((k for k in range(use, ptr - 1, -1) if "You are seat" in (calls[k].get("prompt") or "")), None)
        if obs is not None:
            st["state"] = en(common.state_part(calls[obs]["prompt"]))
            st["state_from_prompt"] = True
        a = _norm_answer(c.get("answer"))
        if hit is not None and isinstance(a, dict):
            st["act"] = en(json.dumps(a, ensure_ascii=False))
        st["tied"] = True
        ptr = use + 1
        tied += 1
    if ptr < len(calls):
        print(f"warning: {len(calls) - ptr} calls after the last decision were not tied to any step", file=sys.stderr)
    return tied, replaced, talk_diff, retries


# ------------------------------------------------------------------------------------------------ build
def build(src, calls=None):
    b = load_source(src)
    if isinstance(calls, str):
        calls = load_calls(calls)
    calls = calls or []
    events, summary, run, cfg = b["events"], b.get("summary") or {}, b.get("run") or {}, b.get("config") or {}
    G = read_game(events, summary, cfg)
    n, Rcfg, TP, E, mp = G["n"], G["R"], G["tp"], G["E"], G["mpcr"]
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
    mps = fn(mp)                                        # mpcr as text ("0.3")
    full_cap = n * E                                    # most a group can put in in one round

    # ---- per round numbers (payoff_i = endowment - c_i + mpcr x total)
    con = [[rd["c"][s] if s < len(rd["c"]) else 0 for rd in RD] for s in range(n)]
    tots = [sum(con[s][i] for s in range(n)) for i in range(R)]
    ret = [mp * tots[i] for i in range(R)]
    pay = [[E - con[s][i] + ret[i] for i in range(R)] for s in range(n)]
    cum = [[] for _ in range(n)]
    for s in range(n):
        acc = Fraction(0)
        for i in range(R):
            acc += pay[s][i]
            cum[s].append(acc)
    score = [cum[s][-1] for s in range(n)]
    assert score == score_chk
    zeros = [[s for s in range(n) if con[s][i] == 0] for i in range(R)]
    fulls = [[s for s in range(n) if con[s][i] == E] for i in range(R)]
    Z = sum(len(z) for z in zeros)                      # contributions of 0 (free riding)
    F = sum(len(f) for f in fulls)
    tot_c = sum(tots)
    rate = [Fraction(tots[i], full_cap) for i in range(R)]
    none_round, all_round = Fraction(E), mp * n * E
    social_full = mp * n > 1                            # full contribution maximises the group's total
    best_round = max(none_round, all_round)
    tot_pay = sum(score)
    tot_none, tot_all = none_round * n * R, all_round * n * R
    tot_best = best_round * n * R

    runlog = b.get("runlog")
    fb_lines = len(re.findall(r"\[LLM fallback\][^\n]*(?:parse|legal)", runlog or ""))
    other_fb = len(re.findall(r"\[LLM fallback\]", runlog or "")) - fb_lines

    # ---- messages: the amount each player named for its own contribution (per round, all its messages together)
    named = {}                                          # (seat, round index) -> [(amount, words)]
    for i, rd in enumerate(RD):
        for t in rd["talk"]:
            t["amts"] = named_amounts(t["text"], E)
            if t["amts"]:
                named.setdefault((t["seat"], i), [])
                for v, w in t["amts"]:
                    if v not in [x for x, _ in named[(t["seat"], i)]]:
                        named[(t["seat"], i)].append((v, w))
    single = {k: v[0][0] for k, v in named.items() if len(v) == 1}
    n_msgs = sum(len(rd["talk"]) for rd in RD)
    said_eq = [k for k, v in single.items() if con[k[0]][k[1]] == v]
    said_lt = [k for k, v in single.items() if con[k[0]][k[1]] < v]
    said_gt = [k for k, v in single.items() if con[k[0]][k[1]] > v]
    multi = [k for k, v in named.items() if len(v) > 1]

    # ---- steps: per round the messages (logged order), then the contributions in seat order
    steps, idx_last = [], {}
    for i, rd in enumerate(RD):
        r = rd["r"]
        for t in rd["talk"]:
            if not 0 <= t["seat"] < n:
                print(f"warning: round {r}: message from seat {t['seat']} ignored", file=sys.stderr)
                continue
            t["step"] = len(steps)
            am = t["amts"]
            steps.append({"seat": t["seat"], "x": i, "phase": "TALK", "round": r, "text": t["text"], "_talk": t,
                          "head": f"Round {r} {DOT} public message"
                                  + (f" {DOT} says it will contribute {am[0][0]}" if len(am) == 1 else
                                     f" {DOT} names {join_and(str(v) for v, _ in am)}" if am else ""),
                          "kind": "talk",
                          "act": json.dumps({"action_id": "talk", "payload": {"text": t["text"]}}, ensure_ascii=False)})
        for s in range(n):
            c, p = con[s][i], pay[s][i]
            sv = single.get((s, i))
            said = f" (said {sv})" if sv is not None and sv != c else ""
            steps.append({"seat": s, "x": i, "phase": "CONTRIBUTE", "round": r, "c": c,
                          "kind": "none" if c == 0 else ("full" if c == E else "part"),
                          "head": f"Round {r} {DOT} contributes {c} of {E}{said} {DOT} +{fn(p)}",
                          "act": json.dumps({"action_id": "contribute", "payload": {"contribute": {"amount": c}}})})
        idx_last[i] = len(steps) - 1
    tail = []                                           # messages of a round the recording did not finish
    if G["tail_talk"] and R < N:
        for t in G["tail_talk"]:
            if 0 <= t["seat"] < n:
                t["amts"] = named_amounts(t["text"], E)
                steps.append({"seat": t["seat"], "x": R, "phase": "TALK", "round": R + 1, "text": t["text"], "_talk": t,
                              "head": f"Round {R + 1} {DOT} public message", "kind": "talk",
                              "act": json.dumps({"action_id": "talk", "payload": {"text": t["text"]}}, ensure_ascii=False)})
                tail.append(len(steps) - 1)

    # ---- what each seat saw (rebuilt from the log; the saved prompt where there is one)
    pay_rule = f"Payoff this round = {E} {MINUS} your contribution + {mps} {TIMES} (total contribution of all {nw} players)."

    def round_line(k):
        return (f"- Round {RD[k]['r']}: contributions {con_list(k)}, total {tots[k]}, group return each {fn(ret[k])}, payoffs "
                f"[{', '.join(fn(pay[s][k]) for s in range(n))}].")

    def con_list(k):
        return "[" + ", ".join(str(con[s][k]) for s in range(n)) + "]"

    def observation(st):
        s, r, i = st["seat"], st["round"], st["x"]
        sc = [cum[q][i - 1] if i > 0 else 0 for q in range(n)]
        lines = ["(Rebuilt from the game log; the exact prompt for this turn was not saved.)",
                 f"You are seat P{s} ({P[s]} on this page). Your role: PLAYER.",
                 (f"Current phase: TALK, round {r} of {Rcfg}. Each player sends {plural(TP, 'public, non-binding message', word=True)} "
                  "before the decision." if st["phase"] == "TALK" else
                  f"Current phase: CONTRIBUTE, round {r} of {Rcfg}. Every player secretly chooses how many of its {E} tokens to put "
                  "into the group account; all contributions are revealed together after the last player has chosen."),
                 f"Endowment {E} tokens a round, mpcr {mps}. " + pay_rule,
                 "Scores so far: " + ", ".join(f"P{q} {fn(sc[q])}" for q in range(n)) + "."]
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

    tied, replaced, talk_diff, retries = attach_calls(steps, calls, E) if calls else (0, [], 0, [])
    for si, v, ans, legal in replaced:                   # a contribution the model did not choose: say so in the title
        st = steps[si]
        st["head"] = (f"Round {st['round']} {DOT} " + (f"asked for {v}, " if legal else "reply not usable, ")
                      + f"{st['c']} applied {DOT} +{fn(pay[st['seat']][st['x']])}")
    has_reason = any(st.get("thinking") for st in steps)
    from_prompt = sum(1 for st in steps if st.get("state_from_prompt"))
    tied_contrib = sum(1 for st in steps if st.get("tied") and st["phase"] == "CONTRIBUTE")
    n_contrib = sum(1 for st in steps if st["phase"] == "CONTRIBUTE")

    # ---- format retries: a decision whose first reply the game could not use, so it asked again. When a rejected reply
    # asked for another amount than the accepted one, the step title, the grid tips, chart 2's note and the foot say so.
    r_con = [r for r in retries if r["phase"] == "CONTRIBUTE"]
    r_talk = [r for r in retries if r["phase"] == "TALK"]
    chg, changed = {}, []                               # (seat, round index) -> short tip text; [(seat, x, amounts, c, label)]
    for r in r_con:
        st = steps[r["si"]]
        diff = [v for v in r["asked"] if v is not None and v != st["c"]]
        if not (r["hit"] and diff):
            continue
        word = "rejected" if not any(ok for v, ok in zip(r["asked"], r["ok"]) if v in diff) else "unused"
        vs = join_and(sorted(set(diff)))
        which = "first reply" if r["n"] == 1 else "earlier replies"
        st["head"] += f" {DOT} {which.split()[0]} ({word}) {which.split()[1]} asked for {vs}"   # "first (rejected) reply asked for 0"
        chg[(st["seat"], st["x"])] = f"{word} {which}: {vs}"
        changed.append((st["seat"], st["x"], tuple(sorted(set(diff))), st["c"], f"{word} {which}"))
    changed_words = ""                                   # "In rounds 8 and 9 Player 4's rejected first reply asked for 0; ..."
    if changed:
        grp = {}
        for s, x, vs, c, lab in changed:
            grp.setdefault((s, vs, c, lab), []).append(x)
        changed_words = " ".join(
            f"In {rounds_text([x + 1 for x in xs])} {P[s]}'s {lab} asked for {join_and(vs)}; asked again, it gave {c}."
            for (s, vs, c, lab), xs in sorted(grp.items(), key=lambda kv: (min(kv[1]), kv[0][0])))
    retry_words = ""
    if r_con:
        one = all(r["n"] == 1 for r in r_con)
        subj = "the first reply" if one else "one or more replies before the last"
        if all(r["flat"] for r in r_con):
            what = ('put the amount in the wrong place ({"amount": k} instead of {"contribute": {"amount": k}})')
        elif not any(ok for r in r_con for ok in r["ok"]):
            what = 'was not a legal contribution in the form {"contribute": {"amount": k}}'
        else:
            what = "was not the one the game applied"
        notes = {r["notice"] for r in r_con}
        how = (f"the game replied {LDQ}{notes.pop()}{RDQ} and asked again" if len(notes) == 1 and None not in notes else
               "the game asked again")
        if all(r["fresh"] for r in r_con):
            how += " in a fresh call that did not see " + ("the first reply" if one else "the earlier replies")
        retry_words = (f"In {len(r_con)} of the {n_contrib} contribution decisions {subj} {what}; {how}, and the step shows "
                       + ("both replies' reasoning" if one else "the reasoning of every reply")
                       + f" (the blocks headed [Earlier attempt, answer: {DOTS}] and [Final attempt]). ")
        same = sum(1 for r in r_con if r["hit"] and all(v == steps[r["si"]]["c"] for v in r["asked"]))
        lead = ("In each, " if same == len(r_con) else f"In the other {same}, " if same + len(changed) == len(r_con) else
                f"In {same} of them, ")
        retry_words += (changed_words + " " if changed else "") + (
            lead + ("the rejected reply asked for the amount the game then applied. " if one else
                    "the earlier replies asked for the amount the game then applied. ") if same else "")
    if r_talk:
        retry_words += (f"In {len(r_talk)} of the {n_msgs} message decisions an earlier reply was not usable and the model was "
                        "asked again; the step shows the reasoning of every reply. ")
    zero_doubt = bool(fb_lines) and Z > 0 and tied_contrib < n_contrib   # some logged 0s may be replaced replies
    tied_c = {(st["seat"], st["x"]) for st in steps if st.get("tied") and st["phase"] == "CONTRIBUTE"}
    replaced_c = {(steps[si]["seat"], steps[si]["x"]) for si, *_ in replaced}
    empty_talk = sum(1 for rd in RD for t in rd["talk"] if not t["text"].strip())
    # the fallback certainly hit a contribution: every logged message has text (a message fallback would be empty), or the
    # players cannot send messages at all (contribute is then the only action)
    fb_hits_contrib = bool(fb_lines) and empty_talk == 0 and (n_msgs > 0 or not TP)

    def maybe_replaced(s, i):
        """A logged contribution that may be (or, with a call, is) the game's replacement for an unusable reply."""
        return (s, i) in replaced_c or (zero_doubt and con[s][i] == DEFAULT_CONTRIB and (s, i) not in tied_c)
    doubt_words = (" Some logged 0s may be replies the game replaced with 0 (see the note at the bottom)." if zero_doubt else
                   (f" {cap(plural(len(replaced), 'contribution', word=True))} {'is' if len(replaced) == 1 else 'are'} not the "
                    "model's answer but the game's replacement (marked in the step title)." if replaced else ""))
    for st in steps:
        st.setdefault("thinking", "")
        for k in ("phase", "round", "text", "_talk", "c", "state_from_prompt", "tied"):
            st.pop(k, None)

    # ---- feed: the rules, the messages, each round's reveal after its last decision
    feed = [{"x": 0, "step": 0, "who": "Rules",
             "text": f"{cap(nw)} players, {Rcfg} rounds, {E} tokens each a round. Each round every player secretly puts 0 to {E} "
                     f"tokens into a group account and keeps the rest; the account pays every player {mps} {TIMES} the group "
                     f"total. Payoff = {E} {MINUS} own contribution + {mps} {TIMES} total. Nobody contributing pays {fn(none_round)} "
                     f"each a round; everyone contributing all {E} pays {fn(all_round)} each a round. "
                     + (f"Before each decision every player sends {plural(TP, 'public, non-binding message', word=True)}." if TP else
                        "The players cannot send messages.")}]
    for i, rd in enumerate(RD):
        for t in rd["talk"]:
            if "step" in t:
                feed.append({"x": i, "step": t["step"], "who": P[t["seat"]], "kind": "talk", "text": LDQ + t["text"] + RDQ})
        zs = zeros[i]
        feed.append({"x": i, "step": idx_last[i], "who": "Reveal",
                     "text": f"Round {rd['r']}: contributions " + " / ".join(str(con[s][i]) for s in range(n))
                             + f" (Players 1 to {n}), total {tots[i]} of {full_cap} ({pct(tots[i], full_cap)}). The group account "
                             f"pays each player {mps} {TIMES} {tots[i]} = {fn(ret[i])}. "
                             + (f"Nobody contributed. " if len(zs) == n else
                                f"{cap(players_text(zs, n))} contributed nothing. " if zs else "")
                             + "Payoffs " + " / ".join(fn(pay[s][i]) for s in range(n))
                             + "; scores " + " / ".join(fn(cum[s][i]) for s in range(n)) + "."})
    for k in tail:
        st = steps[k]
        feed.append({"x": st["x"], "step": k, "who": P[st["seat"]], "kind": "talk",
                     "text": LDQ + json.loads(st["act"])["payload"]["text"] + RDQ})

    # ---- chart 1: contribution grid (players and the group total)
    rows = P + ["Group total"]
    gcells = [[None] * N for _ in rows]
    used = set()
    for i in range(R):
        for s in range(n):
            c = con[s][i]
            kd = "none" if c == 0 else ("full" if c == E else "part")
            used.add(kd)
            sv = named.get((s, i))
            # a short row (the tooltip shows one row per player, on one line): the formula and the messages are in the feed
            bits = [f"gave {c}, kept {E - c}", f"payoff {fn(pay[s][i])}", f"score {fn(cum[s][i])}"]
            if sv:
                bits.append("said " + join_and(str(v) for v, _ in sv))
            if (s, i) in chg:
                bits.append(chg[(s, i)])
            if (s, i) in replaced_c:
                bits.append("replaced by the game (see the step)")
            elif maybe_replaced(s, i):
                bits.append("may be a replaced reply")
            gcells[s][i] = {"c": KINDS[kd][1], "t": str(c), "tip": f" {DOT} ".join(bits)}
        T = tots[i]
        tk = "none" if T == 0 else ("full" if T == full_cap else "part")
        used.add(tk)
        zs = zeros[i]
        gcells[n][i] = {"c": KINDS[tk][1], "t": str(T),
                        "tip": f"{T} of {full_cap} ({pct(T, full_cap)}) {DOT} each gets {mps} {TIMES} {T} = {fn(ret[i])}"
                               + ("" if not 0 < len(zs) < n else
                                  f" {DOT} {players_text(zs, n)} gave 0" if len(zs) <= 3 else f" {DOT} {len(zs)} players gave 0")}
    for i in range(R, N):
        for q in range(len(rows)):
            gcells[q][i] = {"c": "muted", "t": "", "o": .25, "tip": "not played: the recording stops before this round"}
    legend = [[{"full": f"All {E} tokens", "part": "Part", "none": "Nothing (0)"}[k], KINDS[k][1]]
              for k in ("full", "part", "none") if k in used]
    if Z == n * R:
        gnote = f"Every cell is 0: no player contributed a single token in any of the {R} rounds."
    elif F == n * R:
        gnote = f"Every player contributed all {E} tokens in every round."
    else:
        bits = [f"{plural(Z, 'contribution')} of 0 (free riding), {F} of all {E} tokens and {n * R - Z - F} in between, "
                f"out of {n * R}"]
        tr = runs([tots[i] for i in range(R)])
        if len(tr) <= 4:
            bits.append("group total " + "; ".join(f"{v} in {rounds_text(list(range(a, z + 1)))}" for a, z, v in tr))
        else:
            bits.append(f"group total {tots[0]} in round 1, {tots[-1]} in round {R}, between {min(tots)} and {max(tots)}")
        gnote = cap("; ".join(bits)) + "."
    sub1 = (f"One column per round. A player's cell shows the tokens it put into the group account, out of {E} (secret until all "
            f"had chosen); the last row is the group total, out of {full_cap}. Colors: green all {E}, blue part, orange nothing. "
            + gnote + " Hover a column for payoffs and scores" + (" and what each player's message said." if TP else "."))
    charts = [{"title": "Contribution per player, round by round", "sub": sub1, "height": 30 * len(rows) + 30,
               "grid": {"rows": rows, "cells": gcells, "rowHeight": 28 if n <= 8 else 22, "legend": legend}, "bars": []}]

    # ---- chart 2: the amount each player's message named against what it then contributed
    if TP and n_msgs:
        mcells = [[None] * N for _ in range(n)]
        mused, faded = set(), []
        for i, rd in enumerate(RD):
            texts = {}
            for t in rd["talk"]:
                if 0 <= t["seat"] < n:
                    texts.setdefault(t["seat"], []).append(t["text"])
            for s in range(n):
                c = con[s][i]
                sv = named.get((s, i))
                extra = f" {DOT} {chg[(s, i)]}" if (s, i) in chg else ""

                def tip(prefix, text, at=None):
                    """prefix, then as much of the message as fits on the tooltip row (the feed has every message in full)"""
                    q = short_quote(text, TIP_ROW - len(P[s]) - 2 - len(prefix) - 3, at)
                    return prefix + (f" {DOT} {q}" if q else "")
                joined = " / ".join(texts.get(s, []))
                if not texts.get(s):
                    mcells[s][i] = {"c": "muted", "t": "", "o": .35, "tip": "sent no message this round" + extra}
                    mused.add("nomsg")
                elif not sv:
                    mcells[s][i] = {"c": "muted", "t": "", "o": .55, "tip": tip(f"names no amount {DOT} gave {c}" + extra, joined)}
                    mused.add("none")
                elif len(sv) > 1:
                    mcells[s][i] = {"c": "c4", "t": "", "o": .9,
                                    "tip": tip(f"names {join_and(str(v) for v, _ in sv)} {DOT} gave {c}" + extra, joined)}
                    mused.add("several")
                else:
                    v = sv[0][0]
                    k = "eq" if c == v else ("lt" if c < v else "gt")
                    mused.add(k)
                    doubt = k == "lt" and maybe_replaced(s, i)    # the lower amount may not be the model's choice
                    if doubt:
                        faded.append((s, i))
                    rep_tip = ("" if not doubt else f" {DOT} replaced by the game" if (s, i) in replaced_c else
                               f" {DOT} may be a replaced reply")
                    said_msg = next((t for t in rd["talk"] if t["seat"] == s and any(x == v for x, _ in t["amts"])), None)
                    words = next((w for x, w in said_msg["amts"] if x == v), None) if said_msg else None
                    mcells[s][i] = {"c": MSG_COLORS[k], "t": f"{v}{ARROW}{c}", "o": .4 if doubt else .9,
                                    "tip": tip(f"said {v} {ARROW} gave {c}"
                                               + ("" if k == "eq" else f" ({abs(c - v)} {'less' if k == 'lt' else 'more'})")
                                               + rep_tip + extra,
                                               said_msg["text"] if said_msg else joined, words)}
        for i in range(R, N):
            for s in range(n):
                mcells[s][i] = {"c": "muted", "t": "", "o": .25, "tip": "not played: the recording stops before this round"}
        mleg = [[lab, MSG_COLORS[k]] for k, lab in (("eq", "Gave what it said"), ("lt", "Gave less than it said"),
                                                    ("gt", "Gave more than it said"), ("several", "Names several amounts"),
                                                    ("none", "Names no amount")) if k in mused]
        n_single = len(single)
        if n_single:
            mnote = (f"{n_single} of the {n * R} player-rounds name one amount; the player then gave exactly that amount in "
                     f"{len(said_eq)}, less in {len(said_lt)} and more in {len(said_gt)}.")
            if said_lt:
                lt_rs = sorted({k[1] + 1 for k in said_lt})
                lt_v = sorted({single[k] for k in said_lt})
                lt_c = sorted({con[k[0]][k[1]] for k in said_lt})
                mnote += (f" Gave less: {rounds_text(lt_rs)}, where the messages said "
                          + (f"{lt_v[0]}" if len(lt_v) == 1 else f"{lt_v[0]} to {lt_v[-1]}")
                          + " and the logged contributions are " + (f"{lt_c[0]}" if len(lt_c) == 1 else f"{lt_c[0]} to {lt_c[-1]}")
                          + ".")
            if faded and any(k not in replaced_c for k in faded):
                mnote += (f" Faded: {plural(len(faded), 'lower contribution', word=True)} of {DEFAULT_CONTRIB} that may be the "
                          "recorder's replacement for an unusable reply rather than the model's choice; the log cannot say which "
                          "(see the note at the bottom).")
            elif faded:
                mnote += (f" Faded: {plural(len(faded), 'contribution', word=True)} the game applied in place of the model's "
                          "reply (marked in the step title).")
            else:
                mnote += doubt_words
            if changed_words:
                mnote += " " + changed_words + " The cells show the amounts the game applied."
        else:
            mnote = "No message names an amount for the speaker's own contribution."
        sub2 = (f"Each player's {'message' if TP == 1 else 'messages'} before the round's decision. A cell reads said{ARROW}gave: "
                f"the amount the player said it would contribute itself, read from first-person statements such as "
                f"{LDQ}I'll contribute 10{RDQ} or {LDQ}I'm planning to contribute 0{RDQ} (proposals for everyone, such as "
                f"{LDQ}let's all contribute 10{RDQ}, are not counted), then the contribution it made; gray means it gave what it "
                f"said, orange less, blue more. Messages are public and non-binding. " + mnote + " Hover a column for the words that name "
                "each amount; the feed below has every message in full.")
        charts.append({"title": "What each player said it would contribute, and what it gave", "sub": sub2,
                       "height": 30 * n + 30, "grid": {"rows": P, "cells": mcells, "rowHeight": 28 if n <= 8 else 22,
                                                        "legend": mleg}, "bars": []})

    # ---- chart 3: contribution rate (share of the endowments) against the two benchmarks
    # Every contribution the same number (say all 0): the rate, payoff and benchmark lines would all be flat and say nothing the
    # grid, the tiles and the score chart do not, so charts 3 and 4 are left out.
    one_value = len({x for row in con for x in row}) == 1
    avg_pct = [num(rate[i] * 100) for i in range(R)]
    hi_pct = [num(Fraction(max(con[s][i] for s in range(n)) * 100, E)) for i in range(R)]
    lo_pct = [num(Fraction(min(con[s][i] for s in range(n)) * 100, E)) for i in range(R)]
    diff_rounds = [i for i in range(R) if len(set(con[s][i] for s in range(n))) > 1]   # the players gave different amounts
    spread = bool(diff_rounds)
    # the highest and lowest player lines leave the group line only in diff_rounds; with one such round they would sit on the
    # group line everywhere else and add nothing chart 1 does not show, so they are drawn only when there are several
    show_hl = len(diff_rounds) > 1
    if len(set(tots)) == 1:
        cnote = f"The group gives {pct(tots[0], full_cap)} of its endowments in every round."
    else:
        cnote = (f"Group contribution {pct(tots[0], full_cap)} of the endowments in round 1 and {pct(tots[-1], full_cap)} in round {R}"
                 f" (highest {pct(max(tots), full_cap)}, in {rounds_text([i + 1 for i in range(R) if tots[i] == max(tots)])}; "
                 f"lowest {pct(min(tots), full_cap)}, in {rounds_text([i + 1 for i in range(R) if tots[i] == min(tots)])}).")
    dom_word = ("contributing nothing is each player's dominant strategy and the Nash equilibrium" if mp < 1 else
                "contributing everything is each player's dominant strategy")
    mp_words = (f"; they are also higher with a higher mpcr, such as this run's {mps}" if mp > Fraction(3, 10) else "")
    if TP:
        human = ("In human experiments (rules: Isaac and Walker 1988; Ledyard 1995), face-to-face communication, the condition "
                 "closest to this run's public messages, raises contributions substantially above the level without communication "
                 f"and sustains them over the rounds{mp_words}; the rules give no exact figures, so no human line is drawn. ")
    else:
        human = ("In human experiments without communication (rules: Isaac and Walker 1988; Ledyard 1995), contributions start at a "
                 f"substantial fraction of the endowment and decline over the rounds{mp_words}; the rules give no exact figures, so "
                 "no human line is drawn. ")
    if not one_value:
        lines3 = [{"name": "Everyone gives all (100%)", "color": "c3", "dash": True, "thin": True, "v": [100] * N},
                  {"name": "Nobody gives (0%)", "color": "c2", "dash": True, "thin": True, "v": [0] * N}]
        if show_hl:
            lines3 += [{"name": "Highest player", "color": "hold", "thin": True, "dots": True, "v": pad(hi_pct)},
                       {"name": "Lowest player", "color": "ink", "thin": True, "dots": True, "v": pad(lo_pct)}]
        lines3.append({"name": "Group", "color": "c1", "area": True, "dots": True, "end": True, "v": pad(avg_pct)})
        charts.append({"title": "Contribution rate against the benchmarks",
                       "sub": f"The group total as a share of all endowments ({full_cap} tokens a round)"
                              + (f", with the highest and lowest single contribution as a share of {E}" if show_hl else
                                 f"; the players gave different amounts only in {rounds_text([i + 1 for i in diff_rounds])} (chart 1 "
                                 "shows each player), so no highest or lowest player line is drawn" if spread else
                                 "; within each round every player gave the same amount, so no separate player lines are drawn")
                              + ". Dashed: 100%, everyone contributing everything"
                              + (" (the social optimum, since " + f"{mps} {TIMES} {n} = {fn(mp * n)} > 1)" if social_full else "")
                              + f", and 0%: with mpcr {mps} {'< 1' if mp < 1 else '>= 1'}, {dom_word}. " + human + cnote,
                       "height": 230, "fmt": "pct0", "bars": [], "lines": lines3})

    # ---- chart 4: payoff per round (players with identical payoffs share one bar)
    pay_groups = {}
    for s in range(n):
        pay_groups.setdefault(tuple(pay[s]), []).append(s)
    if len(pay_groups) <= 4:
        bars4 = []
        for gi, (path, seats) in enumerate(sorted(pay_groups.items(), key=lambda kv: kv[1][0])):
            nm = players_short(seats, n)
            bars4.append({"name": nm, "color": PATH_COLORS[gi % len(PATH_COLORS)], "v": pad([num(v) for v in path])})
        bar_words = "Bars: each player's payoff" + (" (players with the same payoff in every round share a bar)" if len(pay_groups) < n else "")
    else:
        lo_p = [min(pay[s][i] for s in range(n)) for i in range(R)]
        hi_p = [max(pay[s][i] for s in range(n)) for i in range(R)]
        av_p = [sum(pay[s][i] for s in range(n)) / n for i in range(R)]
        bars4 = [{"name": "Lowest payoff", "color": "hold", "v": pad([num(v) for v in lo_p])},
                 {"name": "Average payoff", "color": "c1", "v": pad([num(v) for v in av_p])},
                 {"name": "Highest payoff", "color": "c4", "v": pad([num(v) for v in hi_p])}]
        bar_words = "Bars: the lowest, average and highest payoff among the players"
    all_int4 = all(isinstance(v, int) for bb in bars4 for v in bb["v"] if v is not None) and none_round.denominator == 1 \
        and all_round.denominator == 1
    if diff_rounds:
        gap = max(max(pay[s][i] for s in range(n)) - min(pay[s][i] for s in range(n)) for i in diff_rounds)
        pnote = (f"Within a round, a player who gives less earns exactly that much more than the others; the largest gap in one "
                 f"round was {fn(gap)}.")
    else:
        pnote = "Every player gave the same amount in every round, so all payoffs within a round are equal."
    if not one_value:
        charts.append({"title": "Payoff each round",
                       "sub": f"{bar_words}: {E} {MINUS} own contribution + {mps} {TIMES} the group total. Dashed: {fn(none_round)} "
                              f"when nobody contributes and {fn(all_round)} when everyone contributes all {E}. Each token a player gives "
                              f"costs it 1 and returns {mps} to it and {mps} to each of the others. " + pnote,
                       "height": 230, "fmt": "int" if all_int4 else "price", "bars": bars4,
                       "lines": [{"name": f"Everyone gives all ({fn(all_round)})", "color": "c3", "dash": True, "thin": True,
                                  "v": [num(all_round)] * N},
                                 {"name": f"Nobody gives ({fn(none_round)})", "color": "c2", "dash": True, "thin": True,
                                  "v": [num(none_round)] * N}]})

    # ---- chart 5: scores (players with identical paths share one line)
    groups = {}
    for s in range(n):
        groups.setdefault(tuple(cum[s]), []).append(s)
    # the two dashed benchmarks carry no end label (they are named in the legend and the note): end labels are kept for the
    # players, whose final scores sit close together at the right edge
    lines5 = [{"name": "Everyone gives all", "color": "c3", "dash": True,
               "v": [num(all_round * (i + 1)) for i in range(N)]},
              {"name": "Nobody gives", "color": "c2", "dash": True, "v": [num(none_round * (i + 1)) for i in range(N)]}]
    shared = []                                         # groups of players with identical scores (one line each)
    if len(groups) <= len(PATH_COLORS):
        for gi, (path, seats) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0])):
            nm = players_short(seats, n)
            if 1 < len(seats) < n:
                shared.append(seats)
            lines5.append({"name": nm, "color": PATH_COLORS[gi], "dots": True, "end": True, "v": pad([num(v) for v in path])})
    else:
        lines5 += [{"name": "Highest score", "color": "ink", "dots": True, "end": True,
                    "v": pad([num(max(cum[s][i] for s in range(n))) for i in range(R)])},
                   {"name": "Average score", "color": "c1", "dots": True, "end": True,
                    "v": pad([num(sum(cum[s][i] for s in range(n)) / n) for i in range(R)])},
                   {"name": "Lowest score", "color": "hold", "dots": True, "end": True,
                    "v": pad([num(min(cum[s][i] for s in range(n))) for i in range(R)])}]
    sc_all_int = all(isinstance(v, int) for l in lines5 for v in l["v"] if v is not None)
    if len(groups) == 1:
        snote = f"Every player ended with {fn(score[0])}."
    else:
        snote = "Final scores: " + ", ".join(f"{P[s]} {fn(score[s])}" for s in range(n)) + "."
    if shared:
        snote += (" " + " ".join(f"{cap(players_text(g, n))} have identical scores after every round, so they share one line."
                                 for g in shared))
    on_line = [(w, [s for path, seats in groups.items() if list(path) == [ref * (i + 1) for i in range(R)] for s in seats])
               for w, ref in (("nobody-gives", none_round), ("everyone-gives-all", all_round))]
    for w, seats in on_line:
        if seats and len(groups) <= len(PATH_COLORS):
            who = "every player" if len(seats) == n else players_text(seats, n)
            snote += (f" The score of {who} equals the {w} score after every round, so "
                      + ("its line" if len(seats) == 1 else "the players' line" if len(seats) == n else "their line")
                      + f" lies on the dashed {w} line.")
    charts.append({"title": "Score so far",
                   "sub": f"Each player's running total of payoffs, its score. Dashed: the score after each round if everyone contributed "
                          f"all {E} every round ({fn(all_round)} a round, {fn(all_round * Rcfg)} after {Rcfg} rounds) or nobody "
                          f"contributed ({fn(none_round)} a round, {fn(none_round * Rcfg)}). " + snote,
                   "height": 230, "fmt": "int" if sc_all_int else "price", "bars": [], "lines": lines5})

    # ---- tiles
    t1 = {"k": "Tokens contributed", "v": f"{fn(tot_c)} of {fn(full_cap * R)}",
          "s": f"{pct(tot_c, full_cap * R)} of all endowments over {plural(R, 'round')}"
               + (f" ({R} played of {Rcfg})" if R < Rcfg else "")
               + (f"; {pct(tots[0], full_cap)} in round 1, {pct(tots[-1], full_cap)} in round {R}" if R > 1 and len(set(tots)) > 1 else
                  "; the same in every round" if R > 1 else "")
               + (". Human groups (rules): face-to-face communication, the closest condition to this run's messages, raises "
                  "contributions substantially and sustains them; without it they start at a substantial fraction of the "
                  "endowment and decline." if TP else
                  ". Human groups without communication (rules): early contributions are a substantial fraction of the "
                  "endowment and decline over the rounds.")}
    if len(set(tots)) > 1:
        t1["spark"] = [num(rate[i] * 100) for i in range(R)]
    zr = [i + 1 for i in range(R) if len(zeros[i]) == n]
    t2 = {"k": "Contributions of 0", "v": f"{Z} of {n * R}",
          "s": ("Decisions in which a player kept all its tokens"
                + (f", the dominant choice: a token given returns only {mps} to its giver" if mp < 1 else "") + ". "
                + (f"Nobody contributed anything in {rounds_text(zr)}" if zr and len(zr) < R else
                   "Nobody contributed anything in any round" if len(zr) == R else
                   "Every round had at least one contribution above 0")
                + (f"; players giving nothing: {len(zeros[0])} in round 1, {len(zeros[-1])} in round {R}." if R > 1 and len(zr) < R
                   else ".")
                + ((" At least one of these 0s is the recorder's replacement for an unusable reply, not a choice; the log cannot "
                    "say which." if fb_hits_contrib else " Some of these 0s may be the recorder's replacements for unusable replies.")
                   if zero_doubt else ""))}
    lo_s, hi_s = min(score), max(score)
    win = [s for s in range(n) if score[s] == hi_s]
    if lo_s == hi_s:
        s3 = f"{cap(everyone)} tie" + (", as the log's winners list says. " if isinstance(G["winners"], list) and len(G["winners"]) == n else ". ")
    else:
        s3 = (f"{players_text(win, n)} " + ("leads" if len(win) == 1 else "tie for the lead") + " (" + ", ".join(f"{P[s]} {fn(score[s])}" for s in range(n))
              + "): over the game " + ("it" if len(win) == 1 else "they") + " gave the least. " if n <= 6 else
              f"{players_text(win, n)} " + ("leads" if len(win) == 1 else "tie for the lead") + f" with {fn(hi_s)}, "
              + ("the player" if len(win) == 1 else "the players") + " who gave the least over the game. ")
    t3 = {"k": "Final scores", "v": f"{fn(lo_s)} each" if lo_s == hi_s else f"{fn(lo_s)} to {fn(hi_s)}",
          "s": s3 + f"Nobody contributing earns {fn(none_round * R)} each over {plural(R, 'round')}; everyone contributing all "
                    f"{E} earns {fn(all_round * R)} each."}
    tiles = [t1, t2, t3]
    if TP and single:
        nl = len(said_lt)
        tiles.append({"k": "Said one amount, gave less", "v": f"{nl} of {len(single)}",
                      "s": f"Player-rounds whose message named one amount for the speaker's own contribution: the logged contribution "
                           f"was lower in {nl}, equal in {len(said_eq)}, higher in {len(said_gt)}."
                           + (f" All of the lower ones are in {rounds_text(sorted({k[1] + 1 for k in said_lt}))}."
                              if nl and len({k[1] for k in said_lt}) < R else "")
                           + (doubt_words if nl else "")})
    else:
        tiles.append({"k": "Group earnings", "v": f"{fn(tot_pay)} of {fn(tot_best)}",
                      "s": f"{pct(tot_pay, tot_best)} of the most the group can earn ({fn(best_round)} each a round, when "
                           + (f"everyone contributes all {E}); nobody contributing earns {fn(tot_none)}." if social_full else
                              "nobody contributes).")})

    # ---- title, lede, foot
    Rw = numword(R)
    pos_lt = [k for k in said_lt if single[k] > 0]

    def story_title():
        """The game told round by round, when that is short: an opening run of at least two rounds in which every player gave
        the same amount, then one to three clauses, each either a run of rounds in which every player gave the same amount
        (and, when every player's message named one same other amount, what they said) or the only round in which the
        amounts differed, when a single player gave something other than all the others (what it said, and that this round
        decides its finishing place). None when the game does not fit that shape."""
        if R < 3 or len(diff_rounds) > 1:
            return None
        segs = []                                       # [kind, first round index, last round index, data]
        for i in range(R):
            if i in diff_rounds:
                vals = [con[s][i] for s in range(n)]
                odd = [s for s in range(n) if vals.count(vals[s]) == 1]
                if n < 3 or len(set(vals)) != 2 or len(odd) != 1:
                    return None
                segs.append(["lone", i, i, odd[0]])
                continue
            said = [single.get((s, i)) for s in range(n)]
            sk = said[0] if said[0] is not None and said[0] != con[0][i] and all(v == said[0] for v in said) else None
            key = (con[0][i], sk)
            if segs and segs[-1][0] == "all" and segs[-1][3] == key:
                segs[-1][2] = i
            else:
                segs.append(["all", i, i, key])
        if segs[0][0] != "all" or segs[0][2] < 1 or not 2 <= len(segs) <= 4:
            return None
        who = "both" if n == 2 else f"all {nw}"
        c0, s0 = segs[0][3]
        k = numword(segs[0][2] + 1)
        if s0 is not None:
            head = (f"{'Both players say' if n == 2 else 'Everyone says'} {s0} and "
                    f"{'give' if n == 2 else 'gives'} {c0} for {k} rounds")
        elif c0 == 0:
            head = f"{'Neither player gives' if n == 2 else 'Nobody gives'} anything for {k} rounds"
        else:
            head = (f"{'Both players give' if n == 2 else 'Everyone gives'} "
                    + (f"all {E}" if c0 == E else f"{c0} of {E}") + f" for {k} rounds")
        later = []
        for kind, x, y, dat in segs[1:]:
            rt = rounds_text(list(range(x + 1, y + 2)))
            if kind == "all":
                c, sk = dat
                later.append(f"in {rt} {who} say {sk} and give {c}" if sk is not None else
                             f"in {rt} nobody gives anything" if c == 0 else
                             f"in {rt} {who} give " + (f"all {E}" if c == E else str(c)))
            else:
                c = con[dat][x]
                o = next(con[q][x] for q in range(n) if q != dat)
                v = single.get((dat, x))
                cl = (f"in {rt} {P[dat]} says {v} and gives {c}" if v is not None and v != c else
                      f"in {rt} {P[dat]} gives {c} and the others {o}")
                rest = fn(next(score[q] for q in range(n) if q != dat))   # the others' scores are equal (one round differs)
                cl += (f" (it finishes first, {fn(score[dat])} to {rest})" if c < o else
                       f" (it finishes last, {fn(score[dat])} to {rest})")
                later.append(cl)
        return head + "; " + (later[0] if len(later) == 1 else ", ".join(later[:-1]) + ", and " + later[-1])
    story = story_title()
    if Z == n * R:
        title = f"Every logged contribution is 0 in all {Rw} rounds"
        if pos_lt:
            vs = sorted({single[k] for k in pos_lt})
            lt_rounds = sorted({k[1] for k in pos_lt})
            amt = f"{vs[0]}" if len(vs) == 1 else f"{vs[0]} to {vs[-1]}"
            if all(all((s, i) in pos_lt for s in range(n)) for i in lt_rounds):
                title += (f", although in {rounds_text([i + 1 for i in lt_rounds])} "
                          + ("both players said they" if n == 2 else "every player said it") + f" would contribute {amt}")
            else:
                title += (f", although {plural(len(pos_lt), 'message')} in {rounds_text([i + 1 for i in lt_rounds])} say the "
                          f"speaker will contribute {amt}")
        else:
            title += ": the group plays the Nash equilibrium throughout" if mp < 1 else ""
    elif F == n * R:
        title = (f"{cap(everyone)} contribute all {E} tokens in every round: the group earns {fn(all_round)} each a round"
                 + (", the social optimum" if social_full else ""))
    elif story:
        title = story
    else:
        a, z = tots[0], tots[-1]
        tr = runs(tots)
        if R == 1:
            title = f"The group contributes {pct(a, full_cap)} of its endowments in the only round played"
        elif len(tr) > 1 and tr[0][1] >= 2:           # an opening plateau, then something else
            k, later = tr[0][1], tr[1:]
            vs = [v for _, _, v in later]
            verb = "drop to" if all(v < a for v in vs) else ("rise to" if all(v > a for v in vs) else "move to")
            title = f"Contributions stay at {pct(a, full_cap)} of the endowments in {rounds_text(list(range(1, k + 1)))}, then "
            if len(later) <= 3:
                title += verb + " " + join_and(f"{pct(v, full_cap)} in {rounds_text(list(range(x, y + 1)))}" for x, y, v in later)
            else:
                title += (f"range from {pct(min(vs), full_cap)} to {pct(max(vs), full_cap)}, ending at {pct(z, full_cap)} in "
                          f"round {R}")
        elif z < a:
            title = f"Contributions fall from {pct(a, full_cap)} of the endowments in round 1 to {pct(z, full_cap)} in round {R}"
        elif z > a:
            title = f"Contributions rise from {pct(a, full_cap)} of the endowments in round 1 to {pct(z, full_cap)} in round {R}"
        elif len(set(tots)) == 1:
            title = f"Contributions stay at {pct(a, full_cap)} of the endowments in every round"
        else:
            lo_r = [i + 1 for i in range(R) if tots[i] == min(tots)]
            hi_r = [i + 1 for i in range(R) if tots[i] == max(tots)]
            if a == max(tots):
                title = (f"Contributions start and end at {pct(a, full_cap)} of the endowments, with a low of "
                         f"{pct(min(tots), full_cap)} in {rounds_text(lo_r)}")
            elif a == min(tots):
                title = (f"Contributions start and end at {pct(a, full_cap)} of the endowments, with a high of "
                         f"{pct(max(tots), full_cap)} in {rounds_text(hi_r)}")
            else:
                title = (f"Contributions start and end at {pct(a, full_cap)} of the endowments and range from "
                         f"{pct(min(tots), full_cap)} to {pct(max(tots), full_cap)} in between")
        zl, li = zeros[-1], R - 1
        last_said = [single.get((s, li)) for s in range(n)]
        if TP and R > 1 and all(v is not None and con[s][li] < v for s, v in enumerate(last_said)):
            sv, cv = set(last_said), {con[s][li] for s in range(n)}
            if len(sv) == 1 and len(cv) == 1:          # e.g. every player says it will give 10 and gives 0
                title += (f"; in round {R} " + ("both players say they" if n == 2 else "every player says it")
                          + f" will give {last_said[0]} and give{'' if n == 2 else 's'} {con[0][li]}")
            else:
                title += (f"; in round {R} " + ("both players give" if n == 2 else "every player gives")
                          + " less than " + ("their messages" if n == 2 else "its message") + " said")
        elif zl and len(zl) < n:
            title += f"; in round {R} {players_text(zl, n)} give{'s' if len(zl) == 1 else ''} nothing"
        elif len(zl) == n:
            pass                                        # the last round's 0% is already in the title
        elif Z:
            title += f"; {Z} of the {n * R} contributions {'is' if Z == 1 else 'are'} 0"
    if zero_doubt:                                      # the headline's 0s may be the recorder's, not the model's
        title += (f" (the recorder replaced at least one unusable reply with {DEFAULT_CONTRIB}; the log cannot say which or how many)"
                  if fb_hits_contrib else
                  f" (the recorder may have replaced an unusable reply with {DEFAULT_CONTRIB}; the log cannot say which)")
    if R < Rcfg:
        title += f" (the recording stops after round {R} of {Rcfg})"
    lede = (f"{cap(nw)} players play {numword(Rcfg)} rounds of a public goods game: each round every player secretly puts 0 to {E} "
            f"of its {E} tokens into a group account and keeps the rest, and the account pays every player {mps} {TIMES} the group "
            f"total; a player's score is the sum of its round payoffs. "
            + (f"Each token given costs its giver 1 and returns only {mps} to it, so giving nothing is best for each player whatever "
               f"the others do, yet if everyone gives everything each earns {fn(all_round)} a round instead of {fn(none_round)}. "
               if mp < 1 and social_full else
               f"Giving nothing is best for each player whatever the others do. " if mp < 1 else
               f"Each token given returns {mps} to its giver, so giving everything pays. ")
            + (f"Before each decision every player sends {plural(TP, 'public, non-binding message', word=True)}."
               if TP else "The players cannot talk; they only see the contributions revealed after each round."))

    if fb_lines:
        fb = ("This run's log reports that at least one reply could not be parsed as a legal action and a random legal action "
              "was played instead (it prints that notice once per run, so it gives neither the number nor which decisions). "
              "For a contribution that fallback is always 0: the game lists a single legal contribution, its default of "
              f"{DEFAULT_CONTRIB}, and the game itself also applies {DEFAULT_CONTRIB} to a reply it cannot use. ")
        rep_words = (f"{cap(plural(len(replaced), 'contribution', word=True))} {'is' if len(replaced) == 1 else 'are'} not the "
                     "model's answer (marked in the step title). " if replaced else "Every one of them is the model's own answer. ")
        if tied_contrib == n_contrib:
            fb += "The saved calls cover every contribution and show which. " + rep_words
        else:
            if fb_hits_contrib:
                fb += (("Every logged message has text (a message fallback would be empty), so the fallback hit " if n_msgs else
                        "The players cannot send messages, so the fallback hit ")
                       + (f"contributions: at least one of the {Z} logged contributions of 0 is not the model's choice, and the "
                          "log cannot tell which. " if Z else "contributions, yet no contribution is 0; the log does not explain this. "))
            else:
                fb += "A logged 0 may therefore be a deliberate choice or a replaced reply; the log cannot tell which. "
            if tied_contrib:
                fb += f"The saved calls cover {tied_contrib} of the {n_contrib} contributions. " + rep_words
    elif replaced:
        fb = (f"In {plural(len(replaced), 'contribution', word=True)} the model's reply was not a legal contribution of the logged "
              f"amount; the game applies its default ({DEFAULT_CONTRIB}) to a reply it cannot use. "
              + ("That step says so in its title. " if len(replaced) == 1 else "These steps say so in their title. "))
    elif runlog is not None and not other_fb:
        fb = "The recorder's run log reports no fallback action. "
    else:
        fb = ""
    if other_fb:
        fb += "The run log also reports a failed model call (an API error), after which a random legal action was played. "
    cfg_bits = ", ".join(x for x in [f"seed {seed}" if seed is not None else "", plural(n, "player", word=True),
                                     plural(Rcfg, "round"), f"endowment {E}", f"mpcr {mps}",
                                     (f"{plural(TP, 'public message', word=True)} per player before each decision" if TP else "no messages")]
                         if x)
    if has_reason:
        rs = (f"Observations and reasoning come from the saved model calls ({tied} of {len(steps)} steps matched to a call"
              + (f"; in {talk_diff} message step{'s' if talk_diff != 1 else ''} the matched call's text differs from the log" if talk_diff else "")
              + "). ")
    else:
        rs = ("Reasoning was not saved in this recording (it has no call log), so no step shows the model's reasoning. " if not calls else
              "The call log has no reasoning, so no step shows the model's reasoning. ")
    obs = ("" if from_prompt == len(steps) else
           ("The observation of the first decision is the prompt saved with the run; the others are rebuilt from the log. "
            if from_prompt == 1 and not tied else
            ("The observations are rebuilt from the log. " if not from_prompt else
             f"{from_prompt} of {len(steps)} observations are saved prompts; the others are rebuilt from the log. ")))
    amount_rule = (("A message's amount is read only from a first-person statement about the speaker's own contribution "
                    f"({LDQ}I'll contribute 10{RDQ}, {LDQ}I'm planning to contribute 0{RDQ}); proposals for everyone and amounts "
                    "with words such as more, less or a range are not counted. ") if TP and n_msgs else "")
    bench = (f"Benchmarks from the rules: with mpcr {mps} < 1 contributing nothing is a dominant strategy and the Nash equilibrium is "
             f"zero contribution by everyone ({fn(none_round)} each a round, {fn(none_round * R)} over {plural(R, 'round')}); "
             if mp < 1 else f"Benchmarks: with mpcr {mps} >= 1 contributing everything is a dominant strategy; ")
    bench += (f"since {mps} {TIMES} {n} = {fn(mp * n)} > 1, everyone contributing everything maximises the group's total "
              f"({fn(all_round)} each a round, {fn(all_round * R)} each over {plural(R, 'round')}). " if social_full else
              f"since {mps} {TIMES} {n} = {fn(mp * n)} <= 1, contributing nothing also maximises the group's total. ")
    foot = (f"Data: the {date} recording of eco_public_goods ({cfg_bits}); every player is played by {model}"
            + (f" ({full_model})" if full_model and full_model != model else "") + ". " + rs + retry_words + obs
            + "Every number on this page is computed from the game log: the round_result events (contributions in seat order, "
              "total_contribution, group_return_each) and the talk events (the messages). The log has no event per contribution, "
              "so each contribution step is rebuilt from its round's result. Payoffs are not logged per player; they are "
              f"recomputed with the rule {E} {MINUS} own contribution + {mps} {TIMES} total and checked against the logged totals and "
              "group returns, the game's own total contribution, the summary's rewards and the winners"
            + (". " if not warns else f"; {plural(len(warns), 'check')} disagree (printed by the builder; the page shows the logged "
                                      "contributions). ")
            + fb + amount_rule
            + f"Player k is seat P(k{MINUS}1) in the log and the prompts; each round the players send their messages in that order, "
              "then choose their contributions in that order, and none sees another's contribution before the reveal. "
            + bench
            + ("Human benchmark (rules): in Isaac and Walker (1988) and the studies surveyed by Ledyard (1995), face-to-face "
               "communication (the human condition closest to this run's public messages) raises contributions substantially and "
               "sustains them; without communication they start at a substantial fraction of the endowment and decline over the "
               "rounds; and they are higher with a higher mpcr. " if TP else
               "Human benchmark (rules): in Isaac and Walker (1988) and the studies surveyed by Ledyard (1995), contributions without "
               "communication (as in this run) start at a substantial fraction of the endowment and decline over the rounds and are "
               "higher with a higher mpcr; face-to-face communication raises them substantially and sustains them. ")
            + "The rules give no exact figures, so no human line is drawn. Sources: Isaac, Walker and Thomas (1984), Public Choice; Isaac and Walker "
              "(1988), Economic Inquiry; Ledyard (1995), Handbook of Experimental Economics.")

    kinds = {k: v for k, v in KINDS.items() if any(st["kind"] == k for st in steps)}
    d = {"id": GID, "name": NAME, "title": title, "lede": lede, "seatNames": P, "tiles": tiles, "foot": foot,
         "x": {"n": N, "labels": [f"R{i + 1}" for i in range(N)]}, "charts": charts, "kinds": kinds, "steps": steps,
         "feed": feed, "feedTitle": "Round by round",
         "feedSub": ("Each round's public messages, then the reveal of the contributions (secret until every player had chosen), "
                     "with the group total, payoffs and scores. " if TP else
                     "Each round's reveal of the contributions (secret until every player had chosen), with the group total, "
                     "payoffs and scores. ") + "The selected step is highlighted; later entries are hidden."}
    d = en_all(d)
    d["eyebrow"] = f"social_game_env {DOT} {GID} {DOT} {model} {DOT} recorded {date}"
    return d


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    common.write(build(src, calls), out, date=recorded_date(src))
