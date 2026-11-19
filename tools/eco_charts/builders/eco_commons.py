"""Replay data for eco_commons (common-pool fishery with the GovSim mechanics; Hardin 1968, Ostrom 1990, Piatti et al. 2024).

usage: python3 -I eco_commons.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, run.log, transcript.md) or a
          file inside one.
  OUT_DIR where <OUT_DIR>/eco_commons.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). Calls are tied to
          decisions in time order: a call belongs to the next decision of the same seat, phase and month (read from its prompt:
          "You are seat Pk", "Current phase: X", the public state's "month"; a value the prompt does not show agrees with
          anything). Within one decision's calls the first reply the agent accepts is the decision's call (for a harvest: one
          JSON object {"action_id": "harvest", "payload": {"harvest": {"tons": n}}}; for a message: the logged text); earlier
          calls were rejected attempts and their reasoning is kept above it. When no reply was accepted the agent fell back to a
          random legal action, which for a harvest is 0 tons (the only harvest the game lists). The chosen call's reasoning
          becomes the step's thinking and its prompt the observation.

Each month every fisher requests a catch in the HARVEST phase (seats asked in order P0, P1, ...; requests stay private until
all have chosen), the catches are revealed in a month_result event (catches in seat order, total, stock after the harvest),
then, unless the lake collapsed, each fisher sends talk_passes public messages (DISCUSS, logged as talk events) and the stock
regrows (regrow event: min(2 x stock left, capacity)). The log has no event per request, so the harvest decisions of a month are
read from its month_result. Requests are not logged; when a month's total catch is at most the stock minus the number of
fishers the requests cannot have exceeded the stock (rationing always leaves a total above that), so every catch is the request
the game applied. Every number shown is computed from the log and checked (totals, stock arithmetic, regrowth, collapse, final
stock, scores against the summary's rewards, winners); any disagreement is printed as a warning. Benchmarks: a month's total
catch is sustainable when it is at most floor(stock / 2), the most that keeps the lake from shrinking (2 x (S - x) >= S; GovSim's
sustainability threshold, and the rules' Theory Benchmark: "keeps at least half the capacity"); from a full lake that is the rules' 50 a
month (half the capacity left, so the lake refills), 10 a fisher, months x capacity / (2 x fishers) = 120 each. The last
configured month has no future stock to protect, so it is not compared with that limit (gray on the page) and an empty lake
after it is not a collapse that loses months; the group's best is then (months - 1) x capacity / 2 + capacity (650, 130 each).
A harvest step's kind compares the request (when the replies are saved; else the catch) with the fisher's share of the
sustainable limit. The town-hall chart shows the request amounts each message names (the number right after "request" or
"ask for"; see _AMT) and compares them with the speaker's next request (saved replies) or catch (no replies); the town halls
before and after the last harvest are not compared. Charts that would be flat or nearly empty are left out: the stock and
total-catch charts when nothing was caught, every series is constant or at most 2 months were played, and the running-total
chart when at most 2 months were played (the tiles carry those numbers). The x axis shows every configured month (months not
played are muted), except when a single month was played: then it shows that month only, and when the saved replies show it
was rationed the request and catch grids are merged into one ("request→catch" per cell). A recording without replies whose catches are all 0
while its run log reports an unparsable reply is reported as what it most likely is: rejected harvest replies replaced by the
0-ton fallback."""
import json, math, os, re, sys, unicodedata
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_commons"
NAME = "Common-pool fishery"
DEFAULT_DATE = "2026-11-09"
MINUS, TIMES, DOT, LDQ, RDQ, NDASH = "−", "×", "·", "“", "”", "–"
KINDS = {"zero": ["Harvest of 0 tons", "c1"],
         "within": ["Harvest within its sustainable share", "c3"],
         "over": ["Harvest above its sustainable share", "c2"],
         "last": ["Harvest in the last month (no future stock to protect)", "hold"],
         "talk": ["Town-hall message", "muted"]}
# A town-hall message that says the fishers have been catching nothing ("two zero-catch rounds", "we've taken nothing so far").
_NOTHING = re.compile(r"zero[- ]catch|\b(?:caught|taken|catching|got|gotten)\s+(?:nothing|zero|no\s+fish)\b|"
                      r"\bnothing\s+(?:so\s+far|caught|yet)\b", re.I)
# A request amount named in a message: a whole number right after "request" (or "ask for"), with a few filler words allowed in
# between ("request the full 100", "a request of 10", "requesting 20 each"); a number followed by a word for a group total,
# a share or a count of months or people is not a request.
_FILL = r"(?:of|to|for|the|full|just|only|exactly|about|around|roughly|at\s+most|no\s+more\s+than|up\s+to|a|an|my|our|each|another)"
_AMT = re.compile(r"\b(?:request(?:s|ed|ing)?|ask(?:s|ed|ing)?\s+for)\b(?:\s+" + _FILL + r"\b)*\s+(\d+)\b(?![.,]\d)"
                  r"(?!\s*(?:%|percent|total|in\s+total|combined|altogether|overall|months?|rounds?|players?|fishers?|people|"
                  r"of\s+us|of\s+them))", re.I)
FISHER_COLORS = ["c1", "c2", "c4", "ink", "hold", "muted"]     # c3 (green) is kept for the sustainable benchmark
NUMW = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


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
    if isinstance(src, dict):                     # an already loaded bundle (tests)
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
    run, meta = b.get("run") or {}, b.get("meta") or {}
    full = str(run.get("model") or meta.get("model") or "")
    date = str(run.get("recorded_at") or meta.get("recorded_at") or DEFAULT_DATE)[:10]
    return full, (full.split("/")[-1] if full else "an unnamed model"), date


def recorded_date(src):
    return model_and_date(load_source(src))[2]


# ------------------------------------------------------------------------------------------------ text helpers
def fn(v):
    """A stock, catch or score: an integer with thousands separators, anything else with up to two decimals."""
    v = Fraction(v)
    s = f"{abs(int(v)):,}" if v.denominator == 1 else f"{abs(float(v)):,.2f}".rstrip("0").rstrip(".")
    return (MINUS if v < 0 else "") + s


def pct(num, den):
    """num/den as a percentage with as many decimals (up to 3) as it takes not to round a value strictly between 0% and 100%
    to either end."""
    if not den:
        return "n/a"
    x = Fraction(num) / Fraction(den) * 100
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


def plural(n, one, many=None):
    return f"{n} {one if n == 1 else (many or one + 's')}"


def join_and(xs):
    xs = [str(x) for x in xs]
    if not xs:
        return ""
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def months_text(ms, cap_first=False):
    """[2, 3, 4] -> 'months 2 to 4'; [1] -> 'month 1'; [1, 3, 5] -> 'months 1, 3 and 5'."""
    ms = sorted(ms)
    w = ("no month" if not ms else f"month {ms[0]}" if len(ms) == 1 else
         f"months {ms[0]} to {ms[-1]}" if len(ms) > 2 and ms == list(range(ms[0], ms[-1] + 1)) else "months " + join_and(ms))
    return cap(w) if cap_first else w


def fishers_text(seats):
    """[0, 1, 2] -> 'Fishers 1 to 3'; [1] -> 'Fisher 2'; [0, 2] -> 'Fishers 1 and 3'."""
    s = sorted(seats)
    if len(s) == 1:
        return f"Fisher {s[0] + 1}"
    if s == list(range(s[0], s[-1] + 1)) and len(s) > 2:
        return f"Fishers {s[0] + 1} to {s[-1] + 1}"
    return "Fishers " + join_and(x + 1 for x in s)


def named_amounts(text):
    """The request amounts a message names (see _AMT), distinct and sorted."""
    return sorted({int(a) for a in _AMT.findall(str(text or ""))})


def amounts_text(amts):
    """[10] -> 'a request of 10'; [20, 100] -> 'requests of 20 and 100'."""
    return ("a request of " if len(amts) == 1 else "requests of ") + join_and(fn(a) for a in amts)


def requests_text(reqs, before):
    """How a month's requests compare: 'Fisher 3 asks for all 100 fish, the other four for 10 each' when one fisher stands
    out, 'all five ask for 30 each' when they agree, else the list of requests."""
    n = len(reqs)
    whole = lambda r: f"all {fn(r)} fish" if r == before else fn(r)
    if len(set(reqs)) == 1:
        who = "both" if n == 2 else f"all {numword(n)}"
        return f"{who} ask for all {fn(before)} fish" if reqs[0] == before else f"{who} ask for {fn(reqs[0])} each"
    counts = {}
    for r in reqs:
        counts[r] = counts.get(r, 0) + 1
    common_r = max(counts, key=lambda r: (counts[r], -r))
    odd = [s for s in range(n) if reqs[s] != common_r]
    if counts[common_r] >= 2 and len(odd) <= 2:
        rest = n - len(odd)
        return (join_and(f"Fisher {s + 1} asks for {whole(reqs[s])}" for s in odd) + ", the other "
                + (numword(rest) if rest > 1 else "one") + f" for {fn(common_r)}" + (" each" if rest > 1 else ""))
    return "the requests are " + " / ".join(fn(r) for r in reqs)


def num(v):
    """A chart value: an int when whole, else a float (exact Fractions are kept for the text)."""
    v = Fraction(v)
    return int(v) if v.denominator == 1 else round(float(v), 6)


def _ws(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


# ------------------------------------------------------------------------------------------------ the log
def read_game(events, summary, cfg):
    rules = next((e for e in events if e.get("event") == "rules"), {})
    over = next((e for e in events if e.get("event") == "game_over"), {})
    end = next((e for e in events if e.get("event") == "game_end"), {})
    var = dict(over.get("variables") or {})                 # final values (stock and month are the END values)
    var.update({k: v for k, v in (summary.get("variables") or {}).items() if k not in var})
    cv = cfg.get("variables") or {}

    def par(k, default=None):
        for src in (rules, var, cv):
            if src.get(k) is not None:
                return src[k]
        return default
    months, cur, stray = [], None, []
    known = {"setup", "rules", "month_result", "talk", "regrow", "collapse", "game_end", "game_over"}
    for e in events:
        ev = e.get("event")
        if ev == "month_result":
            catches = [int(c) for c in (e.get("catches") or [])]
            cur = {"m": int(e.get("month", len(months) + 1)), "catches": catches,
                   "total": int(e["total_catch"]) if e.get("total_catch") is not None else sum(catches),
                   "after": int(e["stock_after_harvest"]), "talk": [], "regrow": None, "collapse": None}
            months.append(cur)
        elif ev == "talk":
            seat = e.get("actor", e.get("seat"))
            if cur is None or seat is None:
                stray.append(e)
                continue
            cur["talk"].append({"seat": int(seat), "text": str(e.get("text", ""))})
        elif ev == "regrow":
            if cur is not None:
                cur["regrow"] = int(e["stock"])
        elif ev == "collapse":
            if cur is not None:
                cur["collapse"] = int(e.get("stock", cur["after"]))
        elif ev not in known and e.get("_visibility") != "hidden":
            print(f"warning: public event {ev!r} is not shown on the page", file=sys.stderr)
    if not months:
        raise SystemExit("no month_result event in the game log")
    if stray:
        print(f"warning: {len(stray)} message(s) before the first harvest or without a sender are not shown", file=sys.stderr)
    n = int(par("n_players", cfg.get("n_players") or len(months[0]["catches"])))
    C = int(par("capacity", 100))
    init = cv.get("stock")
    return {"n": n, "M": int(par("months", len(months))), "C": C, "T": int(par("collapse_threshold", 5)),
            "tp": int(par("talk_passes", 0) or 0), "init": int(init) if init is not None else C, "months": months,
            "end": end, "winners": over.get("winners", summary.get("winners"))}


def verify(G, rewards):
    """Re-derive the stock arithmetic and the scores; returns the warnings (the page shows the logged values)."""
    n, C, T, warn = G["n"], G["C"], G["T"], []
    prev = G["init"]
    score = [0] * n
    for k, mo in enumerate(G["months"]):
        m = mo["m"]
        if m != k + 1:
            warn.append(f"month_result {k}: month {m} logged, {k + 1} expected")
        if len(mo["catches"]) != n:
            warn.append(f"month {m}: {len(mo['catches'])} catches logged for {n} fishers")
        if sum(mo["catches"]) != mo["total"]:
            warn.append(f"month {m}: total_catch {mo['total']}, the catches add up to {sum(mo['catches'])}")
        mo["before"] = mo["after"] + mo["total"]
        if prev is not None and mo["before"] != prev:
            warn.append(f"month {m}: {prev} fish at the start, but {mo['after']} left + {mo['total']} caught = {mo['before']}")
        for s, c in enumerate(mo["catches"]):
            if not 0 <= c <= mo["before"]:
                warn.append(f"month {m}: seat {s} caught {c} of {mo['before']} fish")
            if s < n:
                score[s] += c
        collapsed = mo["after"] < T
        if collapsed != (mo["collapse"] is not None):
            warn.append(f"month {m}: {mo['after']} fish left (threshold {T}) but the collapse event is "
                        + ("missing" if collapsed else "logged"))
        if collapsed:
            if k != len(G["months"]) - 1:
                warn.append(f"month {m}: the lake collapsed but the log goes on")
            prev = None
            continue
        want = min(2 * mo["after"], C)
        if mo["regrow"] is None:
            if k != len(G["months"]) - 1:
                warn.append(f"month {m}: no regrow event")
            prev = want
        else:
            if mo["regrow"] != want:
                warn.append(f"month {m}: regrew to {mo['regrow']}, min(2 {TIMES} {mo['after']}, {C}) = {want}")
            prev = mo["regrow"]
        if G["tp"] and len(mo["talk"]) != G["tp"] * n:
            warn.append(f"month {m}: {len(mo['talk'])} messages logged, {G['tp']} {TIMES} {n} expected")
    last = G["months"][-1]
    final = last["after"] if last["collapse"] is not None else (last["regrow"] if last["regrow"] is not None else last["after"])
    fs = G["end"].get("final_stock")
    if fs is not None and int(fs) != final:
        warn.append(f"game_end final_stock {fs}, {final} from the month events")
    ge_c = G["end"].get("collapsed")
    if ge_c is not None and bool(ge_c) != (last["collapse"] is not None):
        warn.append(f"game_end collapsed {ge_c}, the month events say {last['collapse'] is not None}")
    for s in range(n):
        if rewards[s] is not None and abs(rewards[s] - score[s]) > 1e-9:
            warn.append(f"reward of seat {s}: {rewards[s]} in the summary, {score[s]} tons summed from the catches")
    best = max(score)
    lead = [s for s in range(n) if score[s] == best]
    if isinstance(G["winners"], list) and sorted(int(w) for w in G["winners"]) != lead:
        warn.append(f"winners {G['winners']} logged, highest scores: seats {lead}")
    if last["collapse"] is None and len(G["months"]) < G["M"]:
        warn.append(f"only {len(G['months'])} of {G['M']} months logged and no collapse: the recording stops early")
    return score, final, warn


# ------------------------------------------------------------------------------------------------ model calls
def call_meta(c):
    """Phase and month of a call, from its prompt ('Current phase: X', the public state's "month"); None when not shown."""
    p = c.get("prompt") or ""
    m = re.search(r"Current phase:\s*([A-Za-z_]+)", p)
    i = p.find("Public state:")
    mm = re.search(r'"month":\s*(\d+)', p[i:]) if i >= 0 else None
    return (m.group(1).upper() if m else None), (int(mm.group(1)) if mm else None)


def _strict_json(ans):
    """The reply as the agent's parser reads it (one bare JSON object with action_id and payload only), else None."""
    try:
        d = json.loads((ans or "").strip())
    except ValueError:
        return None
    if (not isinstance(d, dict) or not isinstance(d.get("action_id"), str) or set(d) - {"action_id", "payload"}
            or not isinstance(d.get("payload", {}), dict)):
        return None
    d.setdefault("payload", {})
    return d


def harvest_reply(ans, stock):
    """What the game makes of a harvest reply: (accepted by the agent's parser, tons applied, why not accepted / note).
    The parser wants exactly {"action_id": "harvest", "payload": {"harvest": {...}}}; the game then checks {"tons": int} with
    0 <= tons <= stock and applies its default, 0 tons, to anything else (rules: HARVEST default_action)."""
    d = _strict_json(ans)
    if d is None:
        return False, None, "not one bare JSON action object"
    if d["action_id"] != "harvest":
        return False, None, f"action {d['action_id']!r} is not available in the harvest"
    if set(d["payload"]) != {"harvest"}:
        return False, None, 'payload is not {"harvest": {"tons": n}}'
    v = d["payload"]["harvest"]
    if (isinstance(v, dict) and set(v) == {"tons"} and isinstance(v["tons"], int) and not isinstance(v["tons"], bool)
            and 0 <= v["tons"] <= stock):
        return True, v["tons"], None
    return True, 0, f"request {json.dumps(v)} is not a whole number of tons from 0 to {stock}; the game applied its default, 0 tons"


def talk_matches(call, text):
    a = (call.get("answer") or "").strip()
    d = _strict_json(a)
    if d is not None:
        return d["action_id"] == "talk" and isinstance(d["payload"].get("text"), str) and _ws(d["payload"]["text"]) == _ws(text)
    return bool(a) and _ws(a) == _ws(text)                  # discussion prompts answer with the bare message


def attach_calls(steps, calls):
    """Tie calls to steps in time order (see the module docstring). Returns (steps with a call, harvests the game replaced)."""
    meta = [call_meta(c) for c in calls]

    def fits(k, st, strict=False):
        seat, (ph, mo) = calls[k].get("seat"), meta[k]
        if strict:
            return seat == st["seat"] and ph == st["phase"] and mo == st["month"]
        return seat in (None, st["seat"]) and ph in (None, st["phase"]) and mo in (None, st["month"])

    def accepted(k, st):
        if st["phase"] == "HARVEST":
            return harvest_reply(calls[k].get("answer"), st["before"])[0]
        return talk_matches(calls[k], st["text"])
    ptr = tied = replaced = 0
    for st in steps:
        if ptr < len(calls) and not fits(ptr, st):         # resynchronise on a call that names this very decision
            ahead = next((k for k in range(ptr, min(len(calls), ptr + 80)) if fits(k, st, strict=True)), None)
            if ahead is None:
                continue
            print(f"warning: calls {ptr}..{ahead - 1} match no decision; skipped", file=sys.stderr)
            ptr = ahead
        j, hit = ptr, None
        while j < len(calls) and fits(j, st):
            j += 1
            if accepted(j - 1, st):                         # the agent stops asking at the first reply it accepts
                hit = j - 1
                break
        window = list(range(ptr, j))
        if not window:
            continue
        use = hit if hit is not None else window[-1]
        parts = []
        for k in window:
            if k == use:
                break
            c = calls[k]
            why = harvest_reply(c.get("answer"), st["before"])[2] if st["phase"] == "HARVEST" else "not the logged message"
            parts.append(f"[Earlier attempt, rejected ({why}); answer: {c['answer'][:300]}]\n"
                         + (c.get("reasoning") or "(no reasoning returned)"))
        c = calls[use]
        parts.append(("[Final attempt]\n" if parts else "") + (c.get("reasoning") or ("(no reasoning returned)" if parts else "")))
        if st["phase"] == "HARVEST":
            if hit is None:
                why = harvest_reply(c.get("answer"), st["before"])[2]
                parts.append(f"[Rejected ({why}). No reply was a legal action, so the agent fell back to a random legal action; "
                             "for a harvest that is 0 tons, the only harvest the game lists.]")
                st["req"], st["req_src"], st["replaced"] = 0, "fallback", True
                st["act"] = json.dumps({"action_id": "harvest", "payload": {"harvest": {"tons": 0}}})
                replaced += 1
            else:
                _, tons, note = harvest_reply(c.get("answer"), st["before"])
                st["req"], st["req_src"] = tons, "call"
                st["act"] = en(json.dumps(_strict_json(c.get("answer")), ensure_ascii=False))
                if note:
                    parts.append(f"[The {note}.]")
                    st["replaced"] = True
                    replaced += 1
        else:
            if hit is None:
                parts.append(f"[The log shows the message {LDQ}{st['text'][:200]}{RDQ}, not this answer: {c['answer'][:300]}]")
            else:
                d = _strict_json(c.get("answer"))
                if d is not None:
                    st["act"] = en(json.dumps(d, ensure_ascii=False))
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        if c.get("prompt"):
            st["state"] = en(common.state_part(c["prompt"]))
            st["state_from_prompt"] = True
        ptr = use + 1
        tied += 1
    if ptr < len(calls):
        print(f"warning: {len(calls) - ptr} calls after the last decision were not tied to any step", file=sys.stderr)
    return tied, replaced


def allocate(reqs, stock):
    """The game's allocation: every request in full when they fit, else floor(request x stock / total) each."""
    tot = sum(reqs)
    return list(reqs) if tot <= stock else [r * stock // tot for r in reqs]


# ------------------------------------------------------------------------------------------------ build
def build(src, calls=None):
    b = load_source(src)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    calls = calls or []
    events, summary, run, cfg = b["events"], b.get("summary") or {}, b.get("run") or {}, b.get("config") or {}
    G = read_game(events, summary, cfg)
    n, Mcfg, C, T, TP = G["n"], G["M"], G["C"], G["T"], G["tp"]
    raw_rw = summary.get("rewards") or (run.get("result") or {}).get("rewards") or {}
    rewards = [float(raw_rw[str(s)]) if str(s) in raw_rw else None for s in range(n)]
    score, final_stock, warns = verify(G, rewards)
    for w in warns:
        print("warning:", w, file=sys.stderr)
    full_model, model, date = model_and_date(b)
    seed = run.get("seed", cfg.get("seed"))
    MO = G["months"]
    Mp = len(MO)                                             # months played
    single = Mp == 1                                         # a single month played: one column, no empty months
    N = MO[-1]["m"] if single else max(Mcfg, MO[-1]["m"])    # x axis: every configured month, played or not (single: that month)
    collapsed = MO[-1]["collapse"] is not None
    k_col = MO[-1]["m"] if collapsed else None
    F = [f"Fisher {s + 1}" for s in range(n)]

    # ---- benchmarks (rules: Theory Benchmark; GovSim's sustainability threshold away from full stock)
    half = Fraction(C, 2)                                    # leave half the capacity and doubling refills the lake
    sus = lambda S: Fraction(int(S) // 2)                    # the most all fishers may take without the lake shrinking
    is_last = lambda m: m >= Mcfg                            # the last configured month: no future stock to protect
    limit = lambda m, S: Fraction(S) if is_last(m) else sus(S)
    safe = lambda S: max(0, S - T)                           # the most they may take without collapsing the lake
    share = lambda m, S: Fraction(limit(m, S), n)            # in the last month: an equal split of the whole stock
    sust_seat = Fraction(Mcfg * C, 2 * n)                    # _metrics.sustainable_per_seat
    sust_total = sust_seat * n
    share_full = Fraction(C, 2 * n)
    opt_total = (Mcfg - 1) * half + C                        # the group's best: sustainable, then the whole lake in the last month
    opt_seat = opt_total / n
    greedy = C // n                                          # each fisher's catch if all request the whole (full) lake
    greedy_left = C - n * greedy                             # what that leaves (the rounding remainder)

    # ---- per month numbers
    for mo in MO:
        mo["last"] = is_last(mo["m"])
        mo["L"], mo["safe"] = limit(mo["m"], mo["before"]), safe(mo["before"])
        mo["sure_unrationed"] = mo["total"] <= mo["before"] - n    # rationing always leaves a total above stock - n
    tot_catch = sum(mo["total"] for mo in MO)
    last_played = MO[-1]["last"]                             # the last configured month was played
    col_last = collapsed and last_played                     # the lake emptied in the last month: no month was lost
    cmp_mo = [mo for mo in MO if not mo["last"]]             # months compared with the sustainable limit
    over_months = [mo["m"] for mo in cmp_mo if mo["total"] > mo["L"]]
    cum = [[] for _ in range(n)]
    for s in range(n):
        acc = 0
        for mo in MO:
            acc += mo["catches"][s] if s < len(mo["catches"]) else 0
            cum[s].append(acc)

    # ---- steps: per month the n harvest requests (seat order), then the town-hall messages in logged order
    steps, idx_last_harvest, idx_last_step = [], {}, {}
    for mo in MO:
        m = mo["m"]
        for s in range(n):
            c = mo["catches"][s] if s < len(mo["catches"]) else 0
            steps.append({"seat": s, "x": m - 1, "phase": "HARVEST", "month": m, "catch": c, "before": mo["before"],
                          "req": c if mo["sure_unrationed"] else None, "req_src": "log" if mo["sure_unrationed"] else None})
        idx_last_harvest[m] = len(steps) - 1
        for t in mo["talk"]:
            if not 0 <= t["seat"] < n:
                print(f"warning: month {m}: message from seat {t['seat']} ignored", file=sys.stderr)
                continue
            amts = named_amounts(t["text"])
            steps.append({"seat": t["seat"], "x": m - 1, "phase": "DISCUSS", "month": m, "text": t["text"], "before": mo["before"],
                          "head": f"Month {m} {DOT} " + (f"message naming {amounts_text(amts)}" if amts else "town-hall message"),
                          "kind": "talk",
                          "act": en(json.dumps({"action_id": "talk", "payload": {"text": t["text"]}}, ensure_ascii=False))})
        idx_last_step[m] = len(steps) - 1

    # ---- what each seat saw (rebuilt from the log; the saved prompt where there is one)
    def history_lines(upto):
        out = []
        for q in MO:
            if q["m"] >= upto:
                break
            out.append(f"- Month {q['m']}: catches {q['catches']}, total {q['total']}, {q['after']} fish left"
                       + (f", regrew to {q['regrow']}." if q["regrow"] is not None else "."))
        return out

    def scores_line(upto, include=False):
        sc = [sum(q["catches"][s] for q in MO if q["m"] < upto or (include and q["m"] == upto)) for s in range(n)]
        return "Scores so far (tons caught): " + ", ".join(f"P{s} {sc[s]}" for s in range(n)) + "."

    def observation(st):
        m, s, mo = st["month"], st["seat"], MO[st["month"] - 1]
        lines = ["(Rebuilt from the game log; the exact prompt for this turn was not saved.)",
                 f"You are seat P{s} ({F[s]} on this page). Your role: PLAYER."]
        if st["phase"] == "HARVEST":
            lines += [f"Current phase: HARVEST, month {m} of {Mcfg}. Every fisher secretly requests a catch from 0 to the stock; the "
                      "requests are revealed together after the last fisher has chosen. If they add up to more than the stock, "
                      "each fisher gets floor(request " + TIMES + " stock / total requested).",
                      f"Lake: {mo['before']} fish (capacity {C}). After the harvest the fish left double, up to {C}; if fewer "
                      f"than {T} are left the lake collapses and the game ends.",
                      scores_line(m)]
            hist = history_lines(m)
            lines += (["Earlier months:"] + hist) if hist else ["Earlier months: none yet."]
            prev = MO[m - 2] if m > 1 else None
            if prev and prev["talk"]:
                lines.append(f"Town hall after month {m - 1}:")
                lines += [f"- P{t['seat']}: {t['text']}" for t in prev["talk"]]
        else:
            lines += [f"Current phase: DISCUSS, the town hall after the month {m} harvest (month {m} of {Mcfg}). Each fisher sends "
                      f"{plural(TP, 'public, non-binding message')}.",
                      f"This month: catches {mo['catches']}, total {mo['total']} of {mo['before']} fish; {mo['after']} left, which "
                      + (f"double, up to {C}, before the next month." if m < Mcfg else
                         f"double, up to {C}; this was the last month's harvest, no harvest follows."),
                      scores_line(m, include=True)]
            said = []
            for t in mo["talk"]:
                if t is st.get("_talk"):
                    break
                said.append(t)
            lines += (["Messages so far in this town hall:"] + [f"- P{t['seat']}: {t['text']}" for t in said]
                      if said else ["No one has spoken yet in this town hall."])
        return en("\n".join(lines))

    for mo in MO:                                            # which logged message each talk step is (for 'said so far')
        k = 0
        for st in steps:
            if st["phase"] == "DISCUSS" and st["month"] == mo["m"]:
                st["_talk"] = mo["talk"][k] if k < len(mo["talk"]) else None
                k += 1
    prompt = run.get("prompt") or {}
    for st in steps:
        st["state"] = observation(st)
    for st in steps:
        if (prompt.get("text") and prompt.get("seat") == st["seat"] and str(prompt.get("phase", "")).upper() == st["phase"]
                and re.search(r'"month": %d\b' % st["month"], prompt["text"])):
            st["state"] = en(common.state_part(prompt["text"]))     # the prompt saved with the run (the first decision)
            st["state_from_prompt"] = True
            break

    tied, replaced = attach_calls(steps, calls) if calls else (0, 0)
    has_reason = any(st.get("thinking") for st in steps)
    from_prompt = sum(1 for st in steps if st.get("state_from_prompt"))

    # ---- requests from the calls: check them against the logged catches with the game's allocation rule
    mismatch = []
    for mo in MO:
        hs = [st for st in steps if st["phase"] == "HARVEST" and st["month"] == mo["m"]]
        if hs and all(st.get("req_src") in ("call", "fallback") for st in hs):
            got = allocate([st["req"] for st in hs], mo["before"])
            if got != mo["catches"][:len(hs)]:
                print(f"warning: month {mo['m']}: requests {[st['req'] for st in hs]} from the calls give catches {got}, "
                      f"the log has {mo['catches']}", file=sys.stderr)
                mismatch.append(mo["m"])
            mo["reqs"] = [st["req"] for st in hs]
            mo["rationed"] = sum(mo["reqs"]) > mo["before"]
        else:
            mo["reqs"] = [st["catch"] for st in hs] if mo["sure_unrationed"] else None
            mo["rationed"] = False if mo["sure_unrationed"] else None

    # ---- heads, kinds and actions of the harvest steps
    def kind_of(c, S, m):
        return "zero" if c == 0 else ("last" if is_last(m) else "within" if c <= share(m, S) else "over")
    for st in steps:
        if st["phase"] != "HARVEST":
            continue
        m, c, S, r = st["month"], st["catch"], st["before"], st.get("req")
        st["kind"] = kind_of(r if r is not None else c, S, m)  # the decision: the request when it is known, else the catch
        sh = fn(share(m, S))
        if st.get("req_src") == "fallback":
            st["head"] = f"Month {m} {DOT} no usable reply, 0 tons applied" + (f", logged catch {fn(c)}" if c != 0 else "")
        elif st.get("replaced"):
            st["head"] = f"Month {m} {DOT} request not legal, 0 tons applied" + (f", logged catch {fn(c)}" if c != 0 else "")
        elif r is not None and r != c:
            st["head"] = (f"Month {m} {DOT} asks {fn(r)}, gets {fn(c)} " +
                          ("(log differs from the reply)" if m in mismatch else "(rationed)"))
        elif c == 0:
            st["head"] = f"Month {m} {DOT} " + ("asks for nothing" if r is not None and st.get("req_src") == "call" else
                                                  "catches nothing")
        else:
            st["head"] = f"Month {m} {DOT} catches {fn(c)} " + ("ton" if c == 1 else "tons") + (
                f" in the last month (an equal split of the stock: {sh})" if st["kind"] == "last" else
                f", within its share of {sh}" if st["kind"] == "within" else f", above its share of {sh}")
        if "act" not in st:
            st["act"] = json.dumps({"action_id": "harvest", "payload": {"harvest": {"tons": r if r is not None else
                                    "not logged (this month may have been rationed)"}}})
    for st in steps:
        st.setdefault("thinking", "")
        for k in ("phase", "month", "catch", "before", "req", "req_src", "text", "state_from_prompt", "replaced", "_talk"):
            st.pop(k, None)

    # ---- feed: the rules, then per month the harvest result, the town hall and the regrowth (or the collapse)
    feed = [{"x": 0, "step": 0, "who": "Rules",
             "text": f"{cap(numword(n))} fishers, {Mcfg} months. The lake holds at most {C} fish and starts with {MO[0]['before']}. "
                     f"Each month every fisher secretly requests a catch; if fewer than {T} fish are left after a harvest the lake "
                     "collapses and the game ends; otherwise the fish left double, up to " + f"{C}. "
                     + (f"After each harvest that leaves the lake standing, every fisher sends {plural(TP, 'public message')} in a "
                        "town hall. " if TP else
                        "There is no town hall: the fishers cannot talk. ")
                     + f"Sustainable benchmark: leave at least half the stock after each harvest, so the lake does not shrink: a "
                       f"total catch of {fn(sus(C))} a month from a full lake ({fn(share_full)} a fisher, {fn(sust_seat)} each over the "
                       f"game). The last month has no future stock to protect: taking the whole lake then raises the group's best to "
                       f"{fn(opt_total)} ({fn(opt_seat)} each)."}]
    for mo in MO:
        m = mo["m"]
        req_bit = ""
        if mo.get("reqs") is not None and mo["rationed"]:
            req_bit = (f" Requests " + " / ".join(fn(r) for r in mo["reqs"]) + f" add up to {fn(sum(mo['reqs']))}, more than the "
                       f"{fn(mo['before'])} fish, so each got a rounded-down share.")
        lim_bit = ("the last month, with no future stock to protect" if mo["last"] else
                   f"sustainable limit {fn(mo['L'])} (half the stock)")
        feed.append({"x": m - 1, "step": idx_last_harvest[m], "who": "Harvest",
                     "text": f"Month {m}: catches " + " / ".join(fn(c) for c in mo["catches"])
                             + f" (Fishers 1 to {n}), total {fn(mo['total'])} of {fn(mo['before'])} fish; {lim_bit}, collapse above "
                               f"{fn(mo['safe'])}. {fn(mo['after'])} fish left." + req_bit})
        k = idx_last_harvest[m]
        for t in mo["talk"]:
            if not 0 <= t["seat"] < n:
                continue
            k += 1
            feed.append({"x": m - 1, "step": k, "who": F[t["seat"]], "kind": "talk", "text": LDQ + en(t["text"]) + RDQ})
        if mo["collapse"] is not None:
            feed.append({"x": m - 1, "step": idx_last_step[m], "who": "Collapse",
                         "text": f"Only {fn(mo['after'])} fish left, fewer than {T}: the lake collapses and the game ends after "
                                 + (f"month {m} of {Mcfg}." if not mo["last"] else
                                    f"month {m}, the last, so no harvest is lost.")})
        elif mo["regrow"] is not None:
            feed.append({"x": m - 1, "step": idx_last_step[m], "who": "Regrowth",
                         "text": f"The {fn(mo['after'])} fish left double to {fn(2 * mo['after'])}"
                                 + (f", capped at the capacity: {fn(mo['regrow'])}" if 2 * mo["after"] > C else f": {fn(mo['regrow'])}")
                                 + (f" at the start of month {m + 1}." if m < Mcfg else " at the end of the game.")})

    # ---- chart 1: catch grid, after a grid of the requests when the saved replies show a rationed month
    over_cells = [(mo["m"], s) for mo in MO for s in range(n) if kind_of(mo["catches"][s], mo["before"], mo["m"]) == "over"]
    pos_cells = sum(1 for mo in MO for s in range(n) if mo["catches"][s] > 0)
    H = n * Mp
    rows = F + ["All fishers"]
    LEG = {"c1": "Nothing caught", "c3": "Within the sustainable share or limit", "c2": "Above it",
           "hold": "Last month (not compared)", "c4": "Lake collapsed"}

    def total_cell(mo, tot, what):
        """Color and words for a month's total (caught or requested) against the whole limit."""
        if what == "caught" and mo["collapse"] is not None and not mo["last"]:
            return "c4", "the lake collapsed"
        if tot == 0:
            return "c1", "nothing " + what
        if mo["last"]:
            return "hold", ("the last month, with no future stock to protect: the whole stock may be taken"
                            + ("; the lake emptied, but no month was lost" if what == "caught" and mo["collapse"] is not None else ""))
        return ("c3", "within the sustainable limit") if tot <= mo["L"] else ("c2", "above the sustainable limit")

    # Hover texts are short: the page prints the row name and the month label above them, and draws them on one line each.
    def share_short(mo):
        sh = fn(share(mo["m"], mo["before"]))
        return f"even split {sh} (last month)" if mo["last"] else f"share {sh}"

    def req_tip(mo, s):
        r, c = mo["reqs"][s], mo["catches"][s]
        return f"asked {fn(r)}, got {fn(c)}" + (", rationed" if r != c else "") + f"; {share_short(mo)}"

    def total_tip(mo):
        """The month's total catch: '... caught 45 of 100, within limit 50; 55 left, regrew to 100'."""
        tot, L = mo["total"], mo["L"]
        t = (f"caught {fn(tot)} of {fn(mo['before'])}"
             + (" (last month)" if mo["last"] else f", {'within' if tot <= L else 'above'} limit {fn(L)}")
             + f"; {fn(mo['after'])} left")
        if mo["collapse"] is not None:
            return t + (", no month lost" if mo["last"] else ", lake collapsed")
        return t + (f", regrew to {fn(mo['regrow'])}" if mo["regrow"] is not None else "")

    empty_tip = f"no harvest (lake collapsed in month {k_col})" if collapsed else "not played (the recording stops)"
    empty_leg = ["No harvest (lake collapsed)" if collapsed else "Not played (the recording stops)", "muted"]

    def empty_months(cells, rows_n):
        for i in range(Mp, N):
            cells[rows_n][i] = {"c": "muted", "t": "", "tip": empty_tip}
        return N > Mp
    charts = []
    rat = [mo for mo in MO if mo.get("rationed")]
    merged = single and bool(rat)                            # one month, rationed: one grid of request -> catch
    if merged:
        mo, i = MO[0], MO[0]["m"] - 1
        mcells = [[None] * N for _ in rows]
        mused, over_req = [], []
        for s, r in enumerate(mo["reqs"]):
            c = mo["catches"][s]
            kd = kind_of(r, mo["before"], mo["m"])
            if KINDS[kd][1] not in mused:
                mused.append(KINDS[kd][1])
            if kd == "over":
                over_req.append(s)
            mcells[s][i] = {"c": KINDS[kd][1], "t": f"{fn(r)}→{fn(c)}" if r != c else fn(r), "tip": req_tip(mo, s)}
        tr = sum(mo["reqs"])
        tcol = total_cell(mo, mo["total"], "caught")[0]
        mcells[n][i] = {"c": tcol, "t": f"{fn(tr)}→{fn(mo['total'])}",
                        "tip": f"{fn(tr)} asked for {fn(mo['before'])} fish → {fn(mo['total'])} caught, {fn(mo['after'])} left"
                               + ("; lake collapsed" if mo["collapse"] is not None and not mo["last"] else
                                  "" if mo["last"] else f"; {'within' if mo['total'] <= mo['L'] else 'above'} limit {fn(mo['L'])}")}
        fleg = {"c1": "Asked for nothing", "c3": "Request within the fisher's share", "c2": "Request above the share",
                "hold": "Last month (not compared)"}
        mleg = [[fleg[c], c] for c in ("c1", "c3", "c2", "hold") if c in mused]
        if tcol not in mused:
            mleg.append([{"c4": "Lake collapsed (bottom row)", "c1": "Nothing caught (bottom row)",
                          "c3": "Total within the limit (bottom row)", "c2": "Total above the limit (bottom row)",
                          "hold": "Last month (bottom row)"}[tcol], tcol])
        msub = (f"Month {mo['m']}, the only month played. Each cell shows the tons a fisher asked for, read from its saved reply (the "
                f"log keeps only the catches), and after the arrow the tons it caught; the bottom row adds them up. The requests "
                f"added up to {fn(tr)} for {fn(mo['before'])} fish, more than the stock, so each fisher got floor(request {TIMES} "
                f"stock / total requested): {fn(mo['total'])} were caught and {fn(mo['after'])} left. "
                + (f"Colors compare each request with an even split of the stock ({fn(share(mo['m'], mo['before']))} a fisher), as "
                   "this last month has no future stock to protect" if mo["last"] else
                   f"Colors compare each request with the fisher's share of the sustainable limit, half the starting stock "
                   f"(rounded down) split {numword(n)} ways ({fn(share(mo['m'], mo['before']))} a fisher): green within it, "
                   "orange above it" + (", blue nothing" if "c1" in mused else ""))
                + (". The bottom row is yellow: the lake collapsed. " if tcol == "c4" else
                   ". The bottom row compares the total catch with the whole limit. ")
                + ("Requests above the share: " + fishers_text(over_req) + "." if over_req else
                   "No request was above the fisher's share."))
        charts.append({"title": "Request and catch per fisher", "sub": msub, "height": 30 * len(rows) + 30,
                       "grid": {"rows": rows, "cells": mcells, "rowHeight": 28, "legend": mleg}, "bars": []})
    elif rat:                                                # requests known (saved replies) and some differ from the catches
        rcells = [[None] * N for _ in rows]
        rused = set()
        for mo in MO:
            i = mo["m"] - 1
            if mo.get("reqs") is None:
                for s in range(n):
                    rcells[s][i] = {"c": "muted", "t": "", "tip": "request not known"}
                rused.add("muted")
                continue
            for s, r in enumerate(mo["reqs"]):
                kd = kind_of(r, mo["before"], mo["m"])
                rused.add(KINDS[kd][1])
                rcells[s][i] = {"c": KINDS[kd][1], "t": fn(r), "tip": req_tip(mo, s)}
            tr = sum(mo["reqs"])
            col, word = total_cell(mo, tr, "requested")
            rused.add(col)
            rcells[n][i] = {"c": col, "t": fn(tr),
                            "tip": f"{fn(tr)} asked for {fn(mo['before'])} fish"
                                   + (" (last month)" if mo["last"] else f" (limit {fn(mo['L'])})")
                                   + (f" → rationed, {fn(mo['total'])} caught" if mo["rationed"] else "")}
        has_empty = empty_months(rcells, n)
        rleg = [[("Asked for nothing" if c == "c1" else LEG[c]), c] for c in ("c1", "c3", "c2", "hold") if c in rused]
        if "muted" in rused or has_empty:
            rleg.append([("Request not known" + (", or " + empty_leg[0][0].lower() + empty_leg[0][1:] if has_empty else "")
                          if "muted" in rused else empty_leg[0]), "muted"])
        charts.append({"title": "Requests per fisher, month by month",
                       "sub": "One column per month. Each fisher's row shows the tons it asked for, read from its saved reply (the log "
                              "keeps only the catches); the bottom row adds them up. When the requests add up to more than the stock, "
                              f"each fisher gets floor(request {TIMES} stock / total requested): "
                              + "; ".join(f"in month {mo['m']} the requests added up to {fn(sum(mo['reqs']))} for "
                                          f"{fn(mo['before'])} fish, and {fn(mo['total'])} were caught" for mo in rat)
                              + ". Colors as in the catch grid below.",
                       "height": 30 * len(rows) + 30, "grid": {"rows": rows, "cells": rcells, "rowHeight": 28, "legend": rleg},
                       "bars": []})
    if not merged:
        gcells = [[None] * N for _ in rows]
        used = set()
        for mo in MO:
            i = mo["m"] - 1
            for s in range(n):
                c = mo["catches"][s]
                kd = kind_of(c, mo["before"], mo["m"])
                used.add(KINDS[kd][1])
                r = mo["reqs"][s] if mo.get("reqs") else None
                gcells[s][i] = {"c": KINDS[kd][1], "t": fn(c),
                                "tip": (f"asked {fn(r)} → caught {fn(c)}" if r is not None and r != c else f"caught {fn(c)}")
                                       + f", {share_short(mo)}, score {fn(cum[s][i])}"}
            col, word = total_cell(mo, mo["total"], "caught")
            used.add(col)
            gcells[n][i] = {"c": col, "t": fn(mo["total"]), "tip": total_tip(mo)}
        has_empty = empty_months(gcells, n)
        legend = [[LEG[c], c] for c in ("c1", "c3", "c2", "hold", "c4") if c in used] + ([empty_leg] if has_empty else [])
        if pos_cells == 0:
            grid_note = f"Every cell is 0: no fisher caught a single fish in any of the {plural(Mp, 'month')}."
        elif not over_cells:
            grid_note = f"No fisher ever caught more than its share; {pos_cells} of the {H} harvests caught something."
        else:
            by = {}
            for m, s in over_cells:
                by.setdefault(s, []).append(m)
            grid_note = ("Catches above the fisher's share: " + "; ".join(f"{F[s]} in {months_text(ms)}" for s, ms in sorted(by.items()))
                         + ".")
        sub1 = (("One column per month" if not single else f"Month {MO[0]['m']}, the only month played")
                + ". Each fisher's row shows the tons it caught; the requests were secret until all had chosen. Colors "
                f"compare a catch with the fisher's share of the sustainable limit, half the month's starting stock (rounded down), the "
                f"most the fishers can take without the lake shrinking, split {numword(n)} ways ({fn(share_full)} a fisher at full stock): "
                f"green within it, orange above it, blue nothing"
                + (f"; the last month (month {Mcfg}) is gray, as no future stock is left to protect and the whole stock may be taken"
                   if "hold" in used else "")
                + ". The bottom row is the month's total against the whole limit"
                + (", yellow when the lake collapsed" if collapsed and not col_last else "") + ". " + grid_note)
        charts.append({"title": "Catch per fisher, month by month" if not single else "Catch per fisher", "sub": sub1,
                       "height": 30 * len(rows) + 30,
                       "grid": {"rows": rows, "cells": gcells, "rowHeight": 28, "legend": legend}, "bars": []})

    # ---- chart 1b: what the town hall said (the request amounts each message names) against the speaker's next harvest
    n_msgs = sum(1 for mo in MO for t in mo["talk"] if 0 <= t["seat"] < n)
    talk_amts = {}                                           # (month, seat) -> amounts named in that seat's messages
    for mo in MO:
        for t in mo["talk"]:
            if 0 <= t["seat"] < n:
                talk_amts.setdefault((mo["m"], t["seat"]), set()).update(named_amounts(t["text"]))
    n_named = sum(1 for mo in MO for t in mo["talk"] if 0 <= t["seat"] < n and named_amounts(t["text"]))
    all_named = sorted({a for v in talk_amts.values() for a in v})
    by_next = MO[1:] + [None]
    req_known = bool(tied)                                   # requests read from the saved replies, not inferred from catches
    said_vs_next = {"one": 0, "kept": 0, "pairs": []}       # messages naming one amount before a harvest that was played
    if n_named:
        tcells = [[None] * N for _ in range(n)]
        tused = set()
        for mo, nxt in zip(MO, by_next):
            i = mo["m"] - 1
            S_next = mo["regrow"] if mo["regrow"] is not None else mo["before"]
            for s in range(n):
                if (mo["m"], s) not in talk_amts:
                    continue
                amts = sorted(talk_amts[(mo["m"], s)])
                msgs = [t["text"] for t in mo["talk"] if t["seat"] == s]
                if nxt is not None:
                    nxt_req = nxt["reqs"][s] if req_known and nxt.get("reqs") is not None else None
                    did = nxt_req if nxt_req is not None else nxt["catches"][s]
                    did_word = ("asked for" if nxt_req is not None else "caught") + f" {fn(did)} in month {nxt['m']}"
                    if len(amts) == 1:
                        said_vs_next["one"] += 1
                        said_vs_next["kept"] += int(did == amts[0])
                        said_vs_next["pairs"].append((amts[0], did))
                else:
                    did_word = ("no harvest followed: the lake collapsed" if collapsed else
                                "no harvest followed: this was the last month" if mo["m"] >= Mcfg else
                                "the recording stops before the next harvest")
                m_next = mo["m"] + 1                         # the harvest the message looks ahead to
                no_next = m_next > Mcfg or mo["collapse"] is not None
                if not amts:
                    col, txt = "muted", "–"
                else:
                    top = amts[-1]
                    col = ("c1" if top == 0 else "hold" if no_next or is_last(m_next) else
                           "c3" if top <= share(m_next, S_next) else "c2")
                    txt = "/".join(fn(a) for a in amts)
                tused.add(col)
                bench = ("" if no_next else
                         f" (the next harvest is the last, with no future stock to protect: an equal split of the {fn(S_next)} "
                         f"fish is {fn(share(m_next, S_next))})" if is_last(m_next) else
                         f" (share of the sustainable limit at the next start, {fn(S_next)} fish: {fn(share(m_next, S_next))})")
                tcells[s][i] = {"c": col, "t": txt,
                                "tip": f"town hall after month {mo['m']}: "
                                       + (f"names {amounts_text(amts)}" if amts else "names no request amount")
                                       + f"{bench}; then {did_word}. Message: {LDQ}{' / '.join(msgs)}{RDQ}"}
        tleg = [lab for lab in (["Names 0", "c1"], ["Names an amount within the share", "c3"], ["Names an amount above it", "c2"],
                                ["Before or after the last harvest (not compared)", "hold"], ["Names no amount", "muted"])
                if lab[1] in tused]
        if said_vs_next["one"]:
            mism = said_vs_next["one"] - said_vs_next["kept"]
            pairs = said_vs_next["pairs"]
            what = "request" if req_known else "catch"
            one, kept = said_vs_next["one"], said_vs_next["kept"]
            all0 = mism and all(d == 0 for a, d in pairs if a != d)
            cmp_note = (f"Of the {plural(one, 'message')} that name a single amount before a harvest that was played, "
                        + (f"the speaker's next {what} equals that amount in all {one}." if not mism else
                           f"the speaker's next {what} never equals that amount" + (f"; it is 0 every time." if all0 else ".")
                           if not kept else
                           f"the speaker's next {what} equals that amount in {kept} and differs in {mism}"
                           + (f", each time a {what} of 0." if all0 else ".")))
        else:
            cmp_note = "No message that names a single amount is followed by a harvest."
        charts.append({"title": "Request named in each town-hall message",
                       "sub": f"Each fisher's message in the town hall after the month's catches were revealed; it looks ahead to "
                              f"the next harvest. The cell shows the request amount the message names: the number right after the "
                              f"word {LDQ}request{RDQ} (as in {LDQ}request 10{RDQ} or {LDQ}a request of 100{RDQ}), a proposal for "
                              f"everyone or the speaker's own plan; hover for the full message. Colors compare the largest amount "
                              f"with one fisher's share of the sustainable limit at the next month's start"
                              + (f"; the town halls before and after the last harvest ({months_text([k for k in (Mcfg - 1, Mcfg) if k >= 1])}) "
                                 "are gray, since that harvest has no future stock to protect" if "hold" in tused else "")
                              + f". {n_named} of the {n_msgs} messages name an amount. " + cmp_note,
                       "height": 30 * n + 30, "grid": {"rows": F, "cells": tcells, "rowHeight": 28, "legend": tleg}, "bars": []})

    # ---- chart 2: the stock (left out when it would be flat or a lone point: the tiles carry the numbers)
    pad = lambda vals: vals + [None] * (N - len(vals))
    const = lambda vals: min(vals) == max(vals)
    befores = [mo["before"] for mo in MO]
    afters = [mo["after"] for mo in MO]
    totals = [mo["total"] for mo in MO]
    few = Mp <= 2                                            # one or two months: every series would be a dot or two
    skipped = []
    every = "every month" if Mp >= Mcfg else "every month played"
    if min(befores) == max(befores) and min(afters) == max(afters) == befores[0]:
        stock_note = f"The lake held {fn(befores[0])} fish at the start of {every} and still {fn(afters[0])} after every harvest."
    else:
        lo_i = min(range(Mp), key=lambda i: afters[i])
        stock_note = ((f"The lake held {fn(befores[0])} fish at the start of {every}" if min(befores) == max(befores) else
                       f"Start-of-month stock from {fn(min(befores))} to {fn(max(befores))}")
                      + f"; the lowest point was {fn(afters[lo_i])} fish after the month {MO[lo_i]['m']} harvest.")
    stock_note += (f" The last harvest, in month {k_col}, left fewer than {T} fish; no month was left to lose." if col_last else
                   f" The lake collapsed in month {k_col}." if collapsed else
                   f" It ended the game with {fn(final_stock)} of {fn(C)} fish.")
    if few or tot_catch == 0 or (const(befores) and const(afters)):
        skipped.append("the stock")
    else:
        charts.append({"title": "Fish in the lake",
                       "sub": f"Fish at the start of each month (before the harvest) and right after the harvest; the fish left then "
                              f"double, up to the capacity {fn(C)}, to give the next month's start. A harvest is sustainable when it "
                              f"leaves at least half of the month's starting stock, so the next start is no lower. Dashed: the "
                              f"capacity; half of it ({fn(half)}), the least to leave for the lake to refill to capacity; and the "
                              f"collapse threshold {fn(T)}: fewer fish left ends the game. " + stock_note,
                       "height": 240, "fmt": "int", "bars": [],
                       "lines": [{"name": "Capacity", "color": "hold", "dash": True, "thin": True, "v": [C] * N},
                                 {"name": f"Half the capacity ({fn(half)}): refills the lake", "color": "c3", "dash": True,
                                  "thin": True, "v": [num(half)] * N},
                                 {"name": f"Collapse threshold ({fn(T)})", "color": "c4", "dash": True, "thin": True, "v": [T] * N},
                                 {"name": "Start of the month", "color": "c1", "area": True, "dots": True, "v": pad(befores),
                                  "end": True},
                                 {"name": "After the harvest", "color": "ink", "thin": True, "dots": True, "v": pad(afters)}]})

    # ---- chart 3: total catch against the limits (left out on the same grounds)
    lim_v = [None if mo["last"] else num(mo["L"]) for mo in MO]
    if tot_catch == 0:
        tot_note = f"The total catch was 0 in every month."
    else:
        big = max(cmp_mo, key=lambda q: q["total"]) if cmp_mo else None
        tot_note = ((f"Largest total{' before the last month' if last_played else ''}: {fn(big['total'])} in month {big['m']} "
                     f"(limit {fn(big['L'])}). " if big else "")
                    + (f"Above the limit in {months_text(over_months)}." if over_months else "Never above the limit." if big else "")
                    + (f" The last month took {fn(MO[-1]['total'])} of {fn(MO[-1]['before'])} fish." if last_played else ""))
    if few or tot_catch == 0 or (const(totals) and const([v for v in lim_v if v is not None] or [0])
                                 and const([mo["safe"] for mo in MO])):
        skipped.append("the total catch")
    else:
        charts.append({"title": "Total catch against the limits",
                       "sub": f"Bars: all fishers' catch each month. Dashed green: the sustainable limit, half the month's starting "
                              f"stock (rounded down; {fn(sus(C))} at full stock), the most the fishers can take without the lake "
                              f"shrinking" + ("; none in the last month, which has no future stock to protect" if last_played else "")
                              + f". Dashed yellow: the most the fishers could take without collapsing the lake, the starting stock "
                                f"minus {fn(T)}. " + tot_note,
                       "height": 230, "fmt": "int",
                       "bars": [{"name": "Total catch", "color": "c1", "v": pad(totals)}],
                       "lines": [{"name": "Sustainable limit", "color": "c3", "dash": True, "v": pad(lim_v), "end": True},
                                 {"name": "Most without collapse", "color": "c4", "dash": True, "v": pad([mo["safe"] for mo in MO]),
                                  "end": True}]})

    # ---- chart 4: cumulative catch per fisher (fishers with identical paths share one line), against the sustainable pace
    if len(set(score)) == 1:
        sc_note = f"Every fisher ended with {fn(score[0])} " + ("ton" if score[0] == 1 else "tons") + "."
    else:
        sc_note = "Final scores: " + ", ".join(f"{F[s]} {fn(score[s])}" for s in range(n)) + "."
    if few:
        skipped.append("the running totals")
    else:
        groups = {}
        for s in range(n):
            groups.setdefault(tuple(cum[s]), []).append(s)
        lines4 = [{"name": "Sustainable pace", "color": "c3", "dash": True, "end": True,
                   "v": [num(share_full * (i + 1)) for i in range(N)]}]
        endgame = Mcfg >= 2 and N == Mcfg
        if endgame:
            lines4.append({"name": f"Whole lake in the last month ({fn(opt_seat)})", "color": "c3", "dash": True, "thin": True,
                           "end": True, "v": [None] * (Mcfg - 2) + [num(share_full * (Mcfg - 1)), num(opt_seat)]})
        for gi, (path, seats) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0])):
            nm = F[seats[0]] if len(seats) == 1 else ((("Both fishers" if n == 2 else "All " + numword(n) + " fishers") if len(seats) == n else fishers_text(seats))
                                                      + " (identical)")
            lines4.append({"name": nm, "color": FISHER_COLORS[seats[0] % len(FISHER_COLORS)], "dots": True, "end": True,
                           "v": pad(list(path))})
        charts.append({"title": "Tons caught so far",
                       "sub": f"Each fisher's running total, its score. Dashed: the sustainable pace, {fn(share_full)} a month (half "
                              f"the full lake split {numword(n)} ways), which ends at {fn(sust_seat)} after {Mcfg} months"
                              + (f"; the thin dashed step at the end takes the whole lake in the last month, when no future stock "
                                 f"is left to protect: {fn(opt_seat)} each, the most the fishers can get together" if endgame else "")
                              + ". " + sc_note,
                       "height": 230, "fmt": "int", "bars": [], "lines": lines4})

    # ---- tiles
    def times_share(r, sh, last):
        """', ten times its sustainable share of 10' / ', above its sustainable share of 12' / '' (at most the share)."""
        what = "an even split of the stock" if last else "its sustainable share"
        if sh <= 0 or r <= sh:
            return ""
        k = Fraction(r) / sh
        return (f", {numword(int(k))} times {what} of {fn(sh)}" if k.denominator == 1 else f", above {what} of {fn(sh)}")

    def request_story(mo):
        """One month's requests in words: 'Fisher 3 asked for all 100 fish, ten times its sustainable share of 10, and got 71;
        the other four asked 10 each, their share, and were cut to 7.'"""
        reqs, cat, S, last = mo["reqs"], mo["catches"], mo["before"], mo["last"]
        sh = share(mo["m"], S)
        asked = lambda r: f"asked for all {fn(r)} fish" if r == S else f"asked {fn(r)}"
        got = lambda r, c: ("" if r == c else f", and got {fn(c)}")
        counts = {}
        for r in reqs:
            counts[r] = counts.get(r, 0) + 1
        common_r = max(counts, key=lambda r: (counts[r], -r))
        odd = [s for s in range(n) if reqs[s] != common_r]
        rest = [s for s in range(n) if reqs[s] == common_r]
        rest_c = {cat[s] for s in rest}
        their = (", their sustainable share" if common_r == sh and not last else "") + times_share(common_r, sh, last).replace(" its ", " their ")
        cut = ("" if rest_c == {common_r} else
               f", and were cut to {fn(next(iter(rest_c)))}" if len(rest_c) == 1 else
               ", and got " + " / ".join(fn(cat[s]) for s in rest))
        if not odd:
            who = "Both" if n == 2 else f"All {numword(n)}"
            return f"{who} {asked(common_r)}" + (" each" if common_r != S else "") + their + cut + "."
        if len(rest) >= 2 and len(odd) <= 2:
            return ("; ".join(f"{F[s]} {asked(reqs[s])}{times_share(reqs[s], sh, last)}{got(reqs[s], cat[s])}" for s in odd)
                    + f"; the other {numword(len(rest))} asked {fn(common_r)} each{their}{cut}.")
        return ("Requests " + " / ".join(fn(r) for r in reqs) + f" ({fishers_text(range(n))}); "
                + ("rationed to " + " / ".join(fn(c) for c in cat) if mo.get("rationed") else "all granted in full") + ".")
    spark = befores + [MO[-1]["after"] if collapsed else final_stock]
    rng = (f"{fn(befores[0])} fish at the start of every month" if min(befores) == max(befores) else
           f"{fn(min(befores))} to {fn(max(befores))} fish at the start of a month")
    if col_last:
        lm = MO[-1]
        t1 = {"k": "Lake", "v": (f"Emptied in month {k_col}, the last" if lm["after"] == 0 else f"Below {T} in month {k_col}, the last"),
              "s": f"The last harvest took {fn(lm['total'])} of {fn(lm['before'])} fish and left {fn(lm['after'])}, fewer than {T}, which "
                   f"collapses the lake; the game ended after month {Mcfg} anyway, so no month was lost. Before it: {rng}."}
    elif collapsed:
        t1 = {"k": "Lake", "v": f"Collapsed, month {k_col}",
              "s": f"The month {k_col} harvest took {fn(MO[-1]['total'])} of {fn(MO[-1]['before'])} fish and left {fn(MO[-1]['after'])}, "
                   f"fewer than {T}; {plural(Mcfg - k_col, 'month')} of the {Mcfg} were lost."}
    else:
        t1 = {"k": "Lake at the end", "v": f"{fn(final_stock)} of {fn(C)}",
              "s": (f"Survived all {Mp} months" if Mp >= Mcfg else f"Not collapsed after {Mp} of {Mcfg} months (the recording stops)")
                   + f": {rng}; lowest after a harvest {fn(min(afters))} (it collapses below {T})."}
    if len(spark) >= 2 and not const(spark):
        t1["spark"] = spark
    t2 = {"k": "Tons caught, all fishers", "v": f"{fn(tot_catch)} of {fn(sust_total)}",
          "s": f"{pct(tot_catch, sust_total)} of the sustainable benchmark ({fn(sus(C))} a month at full stock for {Mcfg} months); "
               f"taking the whole lake in the last month raises the most the fishers can catch together to {fn(opt_total)}."
               + (f" Harvests that caught anything: {pos_cells} of {H}." if pos_cells < H else "")}
    lo, hi = min(score), max(score)
    win = [s for s in range(n) if score[s] == hi]
    t3 = {"k": "Score per fisher", "v": (f"{fn(lo)} each" if lo == hi else f"{fn(lo)} to {fn(hi)}"),
          "s": (("Both fishers tie" if n == 2 else f"All {numword(n)} fishers tie") if len(win) == n else f"{fishers_text(win)} " + ("wins" if len(win) == 1 else "tie for the lead"))
               + f". Sustainable play earns {fn(sust_seat)} each ({Mcfg} {TIMES} {fn(share_full)}), {fn(opt_seat)} with the whole lake "
                 f"taken in the last month; if every fisher requests all {fn(C)} fish of a full lake, each gets {fn(greedy)} and "
               + (f"the lake collapses at once ({fn(greedy_left)} left)." if greedy_left < T else f"only {fn(greedy_left)} fish are left.")}
    if cmp_mo:
        ci = max(range(len(cmp_mo)), key=lambda k: cmp_mo[k]["total"])
        detail = ("No fish were taken in any month." if tot_catch == 0 else
                  f"Over in {months_text(over_months)}." if over_months else
                  f"The closest: {fn(cmp_mo[ci]['total'])} against a limit of {fn(cmp_mo[ci]['L'])}.")
    else:
        detail = "Only the last month was played."
    t4 = {"k": "Months over the limit", "v": (f"{len(over_months)} of {len(cmp_mo)}" if cmp_mo else "none compared"),
          "s": "Months whose total catch exceeded the sustainable limit, half the starting stock (rounded down): a larger catch "
               "shrinks the lake. "
               + (f"The last month is not counted: with no future stock to protect, the whole stock may be taken. " if last_played else "")
               + detail}
    mo1 = MO[0]
    if single and mo1.get("reqs") is not None and mo1["m"] not in mismatch:
        # a single month: its months-over-the-limit count only repeats the Lake tile; show what the fishers asked for instead
        t4 = {"k": f"Requests, month {mo1['m']}", "v": f"{fn(sum(mo1['reqs']))} for {fn(mo1['before'])} fish",
              "s": request_story(mo1)}
    elif single:
        t4 = {"k": f"Catches, month {mo1['m']}", "v": f"{fn(mo1['total'])} of {fn(mo1['before'])} fish",
              "s": "Catches " + " / ".join(fn(c) for c in mo1["catches"]) + f" ({fishers_text(range(n))}). "
                   + ("The requests were not saved; with every fish but " + fn(mo1["after"]) + " taken they may have been "
                      "rationed." if not mo1["sure_unrationed"] else "The total leaves enough fish to rule out rationing, so these "
                      "are the requests.")}
    tiles = [t1, t2, t3, t4]

    # ---- title, lede, foot
    pos_named = [a for a in all_named if a > 0]
    zero_after_pos = sum(1 for a, d in said_vs_next["pairs"] if a > 0 and d == 0)
    runlog = b.get("runlog") or ""
    fb_parse = runlog.count("Could not parse a legal JSON action")
    fb_any = len(re.findall(r"\[LLM fallback\]", runlog))
    # Every town-hall reply logged as a real message: eco_commons has no separate discussion prompt, so the talk replies pass the
    # same JSON parser as the harvests, and a rejected one would be logged as the empty placeholder text.
    talk_logged = sum(1 for mo in MO for t in mo["talk"] if 0 <= t["seat"] < n and t["text"].strip())
    nothing = [(mo["m"], t) for mo in MO for t in mo["talk"] if 0 <= t["seat"] < n and _NOTHING.search(t["text"])]
    likely_rejected = bool(not tied and tot_catch == 0 and fb_parse and talk_logged == n_msgs and (pos_named or nothing))
    named_words = ((f"a request of {fn(pos_named[0])}" if len(pos_named) == 1 else
                    f"requests of {fn(pos_named[0])} to {fn(pos_named[-1])} tons") if pos_named else "")
    quote = ""
    if nothing:
        qm, qt = nothing[0]
        sent = [x for x in re.split(r"(?<=[.!?])\s+", qt["text"]) if _NOTHING.search(x)]
        quote = (f"after month {qm}, {F[qt['seat']]} writes {LDQ}{(sent[0] if sent else qt['text']).strip().rstrip('.')}{RDQ}")
    if collapsed:
        last = MO[-1]
        before_ok = not over_months
        if last.get("reqs") is not None and last["m"] not in mismatch:
            body = (f"{requests_text(last['reqs'], last['before'])}; "
                    + ("the rationed catches take " if last["rationed"] else "together they take ")
                    + f"{fn(last['total'])} of {fn(last['before'])} fish")
        else:
            body = f"the fishers take {fn(last['total'])} of its {fn(last['before'])} fish"
        if col_last:
            title = (("The last harvest empties the lake" if last["after"] == 0 else
                      f"The last harvest leaves {fn(last['after'])} fish, fewer than {T}")
                     + (f" after {plural(k_col - 1, 'sustainable month')}" if k_col > 1 and before_ok else
                        f" after {plural(k_col - 1, 'month')}, {len(over_months)} of them above the sustainable limit" if k_col > 1
                        else "")
                     + f": {body}; with no future stock to protect, no month is lost")
        else:
            after_txt = f", after {plural(k_col - 1, 'month')} within the sustainable limit" if k_col > 1 and before_ok else ""
            if last.get("reqs") is not None and last["m"] not in mismatch:
                short = (f"{requests_text(last['reqs'], last['before'])}, and "
                         + ("the rationed catch leaves " if last["rationed"] else "together they leave ") + f"{fn(last['after'])} fish")
            else:
                short = f"the fishers take {fn(last['total'])} of its {fn(last['before'])} fish and leave {fn(last['after'])}"
            title = f"The lake collapses in month {k_col}: {short}" + after_txt
    elif likely_rejected:
        title = (f"{cap(numword(Mp))} months and not one fish caught, most likely because the harvest replies were rejected and the "
                 f"0-ton fallback applied: " + (f"{n_named} of the {n_msgs} town-hall messages name {named_words}" if pos_named else
                                                f"the town-hall messages say the fishers meant to fish")
                 + (", and the fishers complain about catching nothing" if nothing else ""))
    elif tot_catch == 0 and pos_named:
        title = (f"{cap(numword(Mp))} months and not one fish caught: every logged catch is 0, although {n_named} of the {n_msgs} "
                 f"town-hall messages name {named_words}")
    elif tot_catch == 0:
        title = (f"{cap(numword(Mp))} months and not one fish caught: all {numword(n)} fishers catch 0 in every month"
                 + (f", so the lake stays at {fn(C)} and every score is 0" if min(befores) == C and final_stock == C else
                    ", so every score is 0"))
    else:
        big_last = last_played and MO[-1]["total"] > sus(MO[-1]["before"])
        tail = (f": {fn(tot_catch)} tons in all, {pct(tot_catch, sust_total)} of the sustainable {fn(sust_total)}"
                + (f" (at most {fn(opt_total)} with the whole lake taken in the last month)" if tot_catch > sust_total else ""))
        if not over_months:
            title = (f"The lake survives {'all ' if Mp >= Mcfg else ''}{Mp} months with every "
                     + ("catch before the last month" if big_last else "month's catch") + " within the sustainable limit" + tail)
        else:
            title = (f"The lake survives {'all ' if Mp >= Mcfg else ''}{Mp} months although the catch exceeds the sustainable limit in "
                     f"{plural(len(over_months), 'month')}: {fn(tot_catch)} tons in all, lowest stock {fn(min(afters))}")
    if Mp < Mcfg and not collapsed:
        title += f" (the recording stops after month {Mp} of {Mcfg})"
    talk_words = ((", each fisher posts one public, non-binding message in a town hall" if TP == 1 else
                   f", each fisher posts {TP} public, non-binding messages in a town hall") if TP else " (the fishers cannot talk)")
    lede = (f"{cap(numword(n))} fishers share a lake that holds at most {fn(C)} fish and starts "
            + ("full" if MO[0]["before"] == C else f"with {fn(MO[0]['before'])}") + f", for {Mcfg} months. "
            f"Each month every fisher secretly asks for a catch (requests beyond the stock are scaled down), the catches are "
            f"revealed{talk_words}, and the fish left double, up to {fn(C)}; fewer than {T} left after a harvest collapses the lake "
            f"and ends the game. Each fisher scores the tons it caught: {fn(share_full)} each a month leaves half the lake to regrow "
            f"and earns {fn(sust_seat)} each ({fn(opt_seat)} if the whole lake is taken in the last month, when no future stock is "
            f"left to protect), while grabbing everything empties the lake at once.")
    first_collapse = collapsed and single                    # the game ended at its first harvest: no town hall, no regrowth
    if first_collapse:
        lede += (" In this recording the lake collapsed at the very first harvest, so "
                 + ("the town hall never met, no message was sent and " if TP and not n_msgs else "")
                 + "the fish never regrew.")
    if likely_rejected:
        for st in steps:
            if st["kind"] == "zero":
                st["head"] = f"Month {st['x'] + 1} {DOT} 0 tons applied; the reply was most likely rejected"
        lede += (f" In this recording the harvest replies most likely never reached the game: the run log reports a reply that could "
                 f"not be parsed as a legal action (the agent prints that warning once per run), all {n_msgs} town-hall replies, read "
                 f"by the same parser, were logged as messages, so the rejected reply was a harvest, and a rejected harvest becomes "
                 f"the only harvest the game lists, 0 tons" + (f"; {quote}" if quote else "") + ".")

    unr = [mo["m"] for mo in MO if mo["sure_unrationed"]]
    if tied:
        req_txt = (f"The log keeps catches, not requests; the requests shown come from the saved replies and reproduce every logged "
                   "catch with the allocation rule" + (f" except in {months_text(mismatch)}" if mismatch else "") + ". "
                   + (f"In {plural(replaced, 'harvest')} no usable reply was given and the game applied 0 tons (the agent's fallback, "
                      "or the game's default for an illegal request). " if replaced else
                      "Every harvest request the game applied was the model's own reply. "))
    else:
        req_txt = ("The log keeps catches, not requests or replies, so each harvest action shown is rebuilt from the logged catch. "
                   + (f"In every month the total catch was at most the stock minus {n}, which rules out rationing, so every catch is "
                      "the request the game applied. " if len(unr) == Mp else
                      f"In no month was the total catch at most the stock minus {n}, so rationing cannot be ruled out and the "
                      "requests are not known. " if not unr else
                      f"In {months_text(unr)} the total catch was at most the stock minus {n}, which rules out rationing, so those "
                      "catches are the requests the game applied; in the other months the requests are not known. ")
                   + ("A reply the game cannot use also becomes a request of 0 tons: the agent retries, then falls back to a random "
                      "legal action, and the only harvest the game lists is 0 tons. "
                      + (f"This run's log reports that a reply could not be parsed as a legal action (the agent prints that "
                         f"warning once per run). eco_commons has no separate discussion prompt, so the {n_msgs} town-hall replies "
                         f"went through the same parser, and all {n_msgs} were logged as messages (a rejected one would be logged "
                         f"empty): the rejected reply was a harvest. With every catch 0 while "
                         + (f"{n_named} messages name {named_words}" if pos_named else "the messages say the fishers meant to fish")
                         + ", the harvest replies were most likely all rejected and replaced by 0 tons, not deliberate requests "
                           "of 0; without the replies each one cannot be checked. " if likely_rejected else
                         f"This run's log reports that at least one reply could not be parsed as a legal action (it prints that "
                         "message once per run), and it does not say which; so a catch of 0 here may be a deliberate request or a "
                         "replaced reply. " if fb_parse else
                         ("This run's log reports agent fallbacks of another kind. " if fb_any else
                          "This run's log reports no fallback. ") if runlog else "")
                      if any(c == 0 for mo in MO for c in mo["catches"]) else "")
                   + (("In the one town-hall message that names a single amount" if said_vs_next["one"] == 1 else
                       (f"In all {zero_after_pos}" if zero_after_pos == said_vs_next["one"] else
                        f"In {zero_after_pos} of the {said_vs_next['one']}") + " town-hall messages that name a single amount")
                      + " before a harvest, the amount is positive and the speaker's next catch is 0: either the fisher asked for "
                      "nothing despite the message, or its reply was not usable and 0 tons was applied; without the replies this "
                      "recording cannot tell which. " if zero_after_pos and not likely_rejected else ""))
    cfg_bits = ", ".join(x for x in [f"seed {seed}" if seed is not None else "", f"{n} fishers", f"{Mcfg} months",
                                     f"capacity {fn(C)}", f"collapse below {T}",
                                     (f"{plural(TP, 'town-hall message')} per fisher after each harvest" if TP else "no town hall")] if x)
    skip_txt = ("" if not skipped else
                f"Charts of {join_and(skipped)} are left out, as they would show "
                + (("a single month" if single else "only one or two months") if few else "flat lines")
                + "; the tiles give those numbers. ")
    has_regrow = any(mo["regrow"] is not None for mo in MO)
    ev_names = (["month_result: catches, total, stock after the harvest"] + (["regrow"] if has_regrow else [])
                + (["collapse"] if collapsed else []) + (["talk"] if n_msgs else []))
    checks = (["the stock arithmetic"] + [f"the regrowth min(2 {TIMES} fish left, capacity)"] * has_regrow
              + ["the collapse rule", "the final stock", "the summed catches against the summary's rewards"])
    foot = (f"Data: the {date} recording of eco_commons ({cfg_bits}); every fisher is played by {model}"
            + (f" ({full_model})" if full_model and full_model != model else "") + ". "
            + ("Observations and reasoning come from the saved model calls. " if has_reason else
               "Reasoning was not saved in this recording (it has no call log), so no step shows the model's reasoning. ")
            + ("" if from_prompt == len(steps) else
               ("The observation of the first decision is the prompt saved with the run; the others are rebuilt from the log. "
                if from_prompt == 1 and not tied else
                ("The observations are rebuilt from the log. " if not from_prompt else
                 f"{from_prompt} of {len(steps)} observations are saved prompts; the others are rebuilt from the log. ")))
            + "Every number on this page is computed from the game log (" + "; ".join(ev_names) + ") and checked: "
            + join_and(checks)
            + (". " if not warns else f"; {plural(len(warns), 'logged value')} disagree (the page shows the logged values). ")
            + req_txt + skip_txt
            + f"Fisher k is seat P(k{MINUS}1) in the log and the prompts; each month the seats are asked in that order and no fisher "
              "sees another's request before the reveal. Benchmarks: a month's total catch is sustainable when it leaves at least half "
              "of the month's starting stock, at most floor(stock / 2), so the lake does not shrink (the sustainability threshold as "
              "GovSim defines it); from a full lake that is the rules' "
              f"benchmark, {fn(sus(C))} a month (leaving half the capacity, {fn(half)} fish, refills the lake), {fn(share_full)} a "
              f"fisher, {fn(sust_seat)} each over {Mcfg} months. The last month has no future stock to protect, so it is not "
              f"compared with that limit, and the most the fishers can catch together is {Mcfg - 1} {TIMES} {fn(half)} + {fn(C)} = "
              f"{fn(opt_total)}, {fn(opt_seat)} each. A fisher who maximizes this month's catch requests the whole stock, and if all "
              "do the lake is emptied in the first month. Piatti et al. (2024) found that only the strongest LLM societies sustained "
              "the fishery, with survival rates below 54%. Unlike GovSim, requests above the stock are cut to rounded-down "
              "proportional shares (the remainder stays in the lake) and the game ends at a collapse.")

    kinds = {k: v for k, v in KINDS.items() if any(st["kind"] == k for st in steps)}
    d = {"id": GID, "name": NAME, "title": title, "lede": lede, "seatNames": F, "tiles": tiles, "foot": foot,
         "x": {"n": N, "labels": [f"M{i + 1}" for i in range(N)]}, "charts": charts, "kinds": kinds, "steps": steps,
         "feed": feed, "feedTitle": "Month by month",
         "feedSub": ("The month's harvest (the requests were secret until every fisher had chosen). A month then goes on with "
                     + ("the town hall and the regrowth, but here the lake collapsed after the first harvest, so neither happened. "
                        if TP else "the regrowth, but here the lake collapsed after the first harvest, so it never happened. ")
                     + "The selected step is highlighted; later entries are hidden." if first_collapse else
                     "Each month's harvest (the requests were secret until every fisher had chosen), then the town hall, which meets "
                     "after the catches are revealed, so its messages look ahead to the next harvest (none follows the last month's), "
                     "then the regrowth. The selected step is highlighted; later entries are hidden."
                     if TP else "Each month's harvest (the requests were secret until every fisher had chosen), then the regrowth. "
                                "The selected step is highlighted; later entries are hidden.")}
    d = en_all(d)
    d["eyebrow"] = f"social_game_env {DOT} {GID} {DOT} {model}{', reasoning on' if has_reason else ''} {DOT} recorded {date}"
    return d


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    common.write(build(src, calls), out, date=recorded_date(src))
