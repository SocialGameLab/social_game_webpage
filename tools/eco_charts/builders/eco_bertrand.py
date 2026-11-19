"""Replay data for eco_bertrand (repeated Bertrand price competition; Bertrand 1883, Tirole 1988; experiments: Dufwenberg and
Gneezy 2000; algorithmic collusion: Calvano, Calzolari, Denicolo and Pastorello 2020).

usage: python3 -I eco_bertrand.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md) or a file
          inside one.
  OUT_DIR where <OUT_DIR>/eco_bertrand.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). A call belongs to the
          decision of the same seat, phase and round (read from its prompt: "You are seat Pk", "Current phase: X", the public
          state's "round"; a value the prompt does not show agrees with anything), taken in time order. Within one decision's
          calls the first whose answer is the logged action (the message text, or the price) is the decision's call; earlier
          ones were failed attempts (format retries) and their reasoning is kept above it. Its reasoning becomes the step's
          thinking and its prompt the observation.

Each round has talk_passes public messages per firm (TALK phase, logged as talk events in time order) and then one sealed price
per firm (DECIDE phase, seats asked in order P0, P1, ...; no firm sees another's price before the reveal). The log has no event
per price: all prices are in the round's round_result event (prices in seat order, lowest_price, demand). Profits are not
logged per round; they are recomputed exactly (fractions) with the rule of the game, (lowest price - c) * demand / (number of
firms at the lowest price) for the lowest-price firms and 0 for the others, and checked against the summary's rewards and the
hidden game_end totals (sum_min_price, sum_total_profit); any disagreement is printed as a warning. Benchmarks (monopoly price
and profit, symmetric pure equilibria on the integer price grid) are computed exactly from a, c, p_max and the number of firms."""
import json, os, re, sys, unicodedata
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_bertrand"
NAME = "Bertrand competition"
DEFAULT_DATE = "2026-11-09"
MINUS, TIMES, DOT, LDQ, RDQ = "\u2212", "\u00d7", "\u00b7", "\u201c", "\u201d"
FIRM_COLORS = ["c1", "c2", "c3", "c4"]
KINDS = {"talk": ["Public message", "muted"],
         "sell": ["Lowest price: sells", "c3"],
         "cut": ["Price cut: sells below last round's price", "c4"],
         "undercut": ["Price cut, but undercut: sells nothing", "hold"],
         "out": ["Higher price: sells nothing", "hold"]}
_PROMISE = re.compile(r"\b(?:will|shall|i['\u2019]ll|we['\u2019]ll|set|setting|keep|stay|charge|charging|confirm|confirmed|commit|stick|hold)\b",
                      re.I)
_NEGATION = re.compile(r"\b(?:not|never|no longer|cannot)\b|n['\u2019]t\b", re.I)
NUMW = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_ANY = re.compile("[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af\uff00-\uffef]")
_CJK_RUN = re.compile("[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_PUNCT = {"\u3002": ". ", "\u3001": ", ", "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"', "\u3010": "[", "\u3011": "]",
          "\u300a": '"', "\u300b": '"', "\u3008": "<", "\u3009": ">", "\u301c": "~"}


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
    return re.sub("[\u3000-\u303f\uff00-\uffef]", "", s)


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
    return b


def model_and_date(b):
    run, meta = b.get("run") or {}, b.get("meta") or {}
    full = str(run.get("model") or meta.get("model") or "")
    date = str(run.get("recorded_at") or meta.get("recorded_at") or DEFAULT_DATE)[:10]
    return full, (full.split("/")[-1] if full else "an unnamed model"), date


# ------------------------------------------------------------------------------------------------ text helpers
def g(v):
    """A price, profit or score: an integer with thousands separators, anything else with up to two decimals."""
    v = Fraction(v)
    s = f"{abs(int(v)):,}" if v.denominator == 1 else f"{abs(float(v)):,.2f}".rstrip("0").rstrip(".")
    return (MINUS if v < 0 else "") + s


def sg(v):
    """A round profit with its sign; 'profit 0' for zero."""
    v = Fraction(v)
    return "profit 0" if v == 0 else ("+" if v > 0 else "") + g(v)


def share_pct(num, den):
    """num/den as a percentage with as many decimals (up to 4) as it takes not to round a value below 100% up to 100%."""
    if not den:
        return "n/a"
    x = Fraction(num) / Fraction(den) * 100
    if x == 100:
        return "100%"
    for d in range(1, 5):
        s = f"{float(x):.{d}f}"
        if float(s) != 100.0:
            return (s.rstrip("0").rstrip(".") + "%" if d > 1 else s + "%").replace("-", MINUS)
    return f"{float(x):.4f}%"


def numword(n):
    return NUMW[n] if 0 <= n < len(NUMW) else str(n)


def NUM_CAP(n):
    w = numword(n)
    return w[:1].upper() + w[1:]


def plural(n, one, many=None, word=False):
    return f"{numword(n) if word else n} {one if n == 1 else (many or one + 's')}"


def join_and(xs):
    xs = [str(x) for x in xs]
    if not xs:
        return ""
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def rounds_text(rs, cap=False):
    """[2, 3, 4] -> 'rounds 2 to 4'; [1] -> 'round 1'; [1, 2] -> 'rounds 1 and 2'; [1, 3, 5] -> 'rounds 1, 3 and 5'."""
    rs = sorted(rs)
    w = "no round" if not rs else (f"round {rs[0]}" if len(rs) == 1 else
                                   (f"rounds {rs[0]} to {rs[-1]}" if len(rs) >= 3 and rs == list(range(rs[0], rs[-1] + 1)) else
                                    "rounds " + join_and(rs)))
    return w[:1].upper() + w[1:] if cap else w


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


def _ws(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


# ------------------------------------------------------------------------------------------------ benchmarks
def benchmarks(a, c, p_max, n):
    """Exact one-shot benchmarks on the integer price grid 0..p_max. Monopoly: the price(s) maximizing (p - c)(a - p), the
    market's profit when one firm sells or all tie there. Symmetric pure equilibria: all n firms at p is an equilibrium when no
    firm gains by moving alone (raising sells nothing, cutting to q < p sells the whole demand at q)."""
    prof = {p: Fraction((p - c) * max(0, a - p)) for p in range(0, p_max + 1)}
    pi_m = max(prof.values())
    mono = [p for p in prof if prof[p] == pi_m]
    eq = []
    for p in range(0, p_max + 1):
        stay = prof[p] / n if a - p > 0 else Fraction(0)
        best_dev = max([Fraction(0)] + [prof[q] for q in range(0, p)])
        if stay >= best_dev:
            eq.append(p)
    return {"pi_m": pi_m, "mono": mono, "eq": eq, "eq_profit": {p: (prof[p] if a - p > 0 else Fraction(0)) for p in eq}}


# ------------------------------------------------------------------------------------------------ the log
def read_game(events, summary, cfg):
    rules = next((e for e in events if e.get("event") == "rules"), {})
    over = next((e for e in events if e.get("event") == "game_over"), {})
    var = dict(over.get("variables") or {})
    var.update({k: v for k, v in (summary.get("variables") or {}).items() if k not in var})
    cv = cfg.get("variables") or {}

    def par(k, default=None):
        for src in (rules, var, cv):
            if src.get(k) is not None:
                return src[k]
        return default
    rounds, talk, cur = [], [], []
    for e in events:
        ev = e.get("event")
        if ev == "talk":
            cur.append(e)
        elif ev == "round_result":
            rounds.append(e)
            talk.append(cur)
            cur = []
    if not rounds:
        raise SystemExit("no round_result event in the game log")
    if cur:
        print(f"warning: {len(cur)} message(s) after the last round_result (an unfinished round) are not shown", file=sys.stderr)
    known = {"setup", "rules", "talk", "round_result", "game_end", "game_over"}
    for e in events:
        if e.get("event") not in known and e.get("_visibility") != "hidden":
            print(f"warning: public event {e.get('event')!r} is not shown on the page", file=sys.stderr)
    n = int(par("n_players", len(rounds[0].get("prices") or []) or cfg.get("n_players") or 2))
    return {"a": int(par("a", 100)), "c": int(par("c", 10)), "p_max": int(par("p_max", 100)), "R": int(par("rounds", len(rounds))),
            "talk_passes": int(par("talk_passes", 0) or 0), "n": n, "rounds": rounds, "talk": talk,
            "game_end": next((e for e in events if e.get("event") == "game_end"), {}),
            "winners": over.get("winners", summary.get("winners"))}


def market(G):
    """Per round: prices, market (lowest) price, demand, the lowest-price firms and every firm's profit (exact)."""
    a, c, n = G["a"], G["c"], G["n"]
    out, warn = [], []
    for i, e in enumerate(G["rounds"]):
        pr = [int(p) for p in (e.get("prices") or [])]
        if len(pr) != n:
            warn.append(f"round {i + 1}: {len(pr)} prices logged for {n} firms")
            pr = (pr + [None] * n)[:n]
        if int(e.get("round", i + 1)) != i + 1:
            warn.append(f"round_result {i}: round {e.get('round')} logged, {i + 1} expected")
        valid = [p for p in pr if p is not None]
        mp = min(valid)
        if e.get("lowest_price") is not None and int(e["lowest_price"]) != mp:
            warn.append(f"round {i + 1}: lowest_price {e['lowest_price']} logged, {mp} is the lowest of {pr}")
        D = max(0, a - mp)
        if e.get("demand") is not None and int(e["demand"]) != D:
            warn.append(f"round {i + 1}: demand {e['demand']} logged, max(0, {a} - {mp}) = {D}")
        low = [s for s in range(n) if pr[s] == mp]
        prof = [Fraction(0)] * n
        for s in low:
            prof[s] = Fraction(mp - c) * Fraction(D, len(low))
        out.append({"r": i + 1, "prices": pr, "mp": mp, "D": D, "low": low, "profit": prof, "units": Fraction(D, len(low))})
    return out, warn


def verify(G, M, rewards):
    warn = []
    score = [sum((m["profit"][s] for m in M), Fraction(0)) for s in range(G["n"])]
    if len(M) != G["R"]:
        warn.append(f"{len(M)} rounds logged, {G['R']} configured")
    ge = G["game_end"]
    if ge.get("sum_min_price") is not None and Fraction(ge["sum_min_price"]) != sum(m["mp"] for m in M):
        warn.append(f"game_end sum_min_price {ge['sum_min_price']}, {sum(m['mp'] for m in M)} summed from the round results")
    if ge.get("sum_total_profit") is not None and abs(float(ge["sum_total_profit"]) - float(sum(score))) > 1e-6:
        warn.append(f"game_end sum_total_profit {ge['sum_total_profit']}, {float(sum(score))} recomputed")
    for s in range(G["n"]):
        if rewards[s] is not None and abs(rewards[s] - float(score[s])) > 1e-6:
            warn.append(f"reward of seat {s}: {rewards[s]} in the summary, {float(score[s])} recomputed from the prices")
    best = max(score)
    lead = [s for s in range(G["n"]) if score[s] == best]
    win = G["winners"]
    if isinstance(win, list) and sorted(int(w) for w in win) != lead:
        warn.append(f"winners {win} logged, highest recomputed score(s): seats {lead}")
    return score, warn


# ------------------------------------------------------------------------------------------------ model calls
def call_meta(c):
    """Phase and round of a call, from its prompt ('Current phase: X', the public state's "round"); None when not shown."""
    p = c.get("prompt") or ""
    m = re.search(r"Current phase:\s*([A-Za-z_]+)", p)
    phase = m.group(1).upper() if m else None
    i = p.find("Public state:")
    rm = re.search(r'"round":\s*(\d+)', p[i:]) if i >= 0 else None
    return phase, int(rm.group(1)) if rm else None


def _amount(payload):
    if not isinstance(payload, dict):
        return None
    v = payload.get("price", payload.get("amount"))
    if isinstance(v, dict):
        v = v.get("amount")
    try:
        f = float(v)
        return int(f) if f == int(f) else f
    except (TypeError, ValueError):
        return None


def _matches(call, st, loose=False):
    a = _norm_answer(call.get("answer"))
    if not isinstance(a, dict):
        return False
    if st["phase"] == "TALK":
        if a.get("action_id") != "talk":
            return False
        return loose or _ws((a.get("payload") or {}).get("text")) == _ws(st["text"])
    return a.get("action_id") == "set_price" and _amount(a.get("payload")) == st["price"]


def attach_calls(steps, calls):
    """Tie calls to steps in time order (see the module docstring). Returns the number of steps that got a call and the steps
    whose action is the answer to a follow-up question (their reasoning ran out before an answer)."""
    meta = [call_meta(c) for c in calls]

    def fits(k, st, strict=False):
        seat, (ph, rnd) = calls[k].get("seat"), meta[k]
        if strict:
            return seat == st["seat"] and ph == st["phase"] and rnd == st["round"]
        return seat in (None, st["seat"]) and ph in (None, st["phase"]) and rnd in (None, st["round"])
    ptr, tied, rescued = 0, 0, []
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
            if _matches(calls[j - 1], st):     # a decision's calls end at its accepted answer (later calls of the same seat,
                break                          # phase and round belong to another decision, e.g. a second talk pass)
        window = list(range(ptr, j))
        if not window:
            continue
        hit = next((k for k in window if _matches(calls[k], st)), None)
        if hit is None:
            hit_loose = next((k for k in window if _matches(calls[k], st, loose=True)), None)
        else:
            hit_loose = hit
        use = hit if hit is not None else (hit_loose if hit_loose is not None else window[-1])
        parts = []
        for k in range(ptr, use):
            c = calls[k]
            parts.append(f"[Earlier attempt, answer: {c['answer'][:300]}]\n{c.get('reasoning') or '(no reasoning returned)'}")
        c = calls[use]
        parts.append(("[Final attempt]\n" if parts else "") + (c.get("reasoning") or ("(no reasoning returned)" if parts else "")))
        if hit is None:
            what = f"the message {LDQ}{st['text'][:200]}{RDQ}" if st["phase"] == "TALK" else f"the price {st['price']}"
            parts.append(f"[The log shows {what}, not this answer: {c['answer'][:300]}]")
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
            rescued.append(st)
        st["thinking"] = "\n\n".join(p for p in parts if p).strip()
        if c.get("prompt"):
            st["state"] = common.state_part(c["prompt"])
            st["state_from_prompt"] = True
        a = _norm_answer(c.get("answer"))
        if hit is not None and isinstance(a, dict):
            st["act"] = json.dumps(a, ensure_ascii=False)
        ptr = use + 1
        tied += 1
    if ptr < len(calls):
        print(f"warning: {len(calls) - ptr} calls after the last decision were not tied to any step", file=sys.stderr)
    return tied, rescued


# ------------------------------------------------------------------------------------------------ build
def build(src, calls=None):
    b = load_source(src)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    calls = calls or []
    events, summary, run, cfg = b["events"], b.get("summary") or {}, b.get("run") or {}, b.get("config") or {}
    G = read_game(events, summary, cfg)
    a, c, p_max, n, Rcfg, TP = G["a"], G["c"], G["p_max"], G["n"], G["R"], G["talk_passes"]
    M, warn = market(G)
    R = len(M)
    raw_rw = summary.get("rewards") or (run.get("result") or {}).get("rewards") or {}
    rewards = [float(raw_rw[str(s)]) if str(s) in raw_rw else None for s in range(n)]
    score, w2 = verify(G, M, rewards)
    for w in warn + w2:
        print("warning:", w, file=sys.stderr)
    B = benchmarks(a, c, p_max, n)
    pi_m, mono = B["pi_m"], B["mono"]
    pM = mono[0]
    pM_txt = join_and(mono) if len(mono) > 1 else str(pM)
    eq = B["eq"]
    full_model, model, date = model_and_date(b)
    seed = run.get("seed", cfg.get("seed"))

    F = [f"Firm {s + 1}" for s in range(n)]
    the_firms = "The two firms" if n == 2 else f"The {numword(n)} firms"
    mp = [m["mp"] for m in M]
    cum = [[] for _ in range(n)]
    for s in range(n):
        acc = Fraction(0)
        for m in M:
            acc += m["profit"][s]
            cum[s].append(acc)
    industry = [sum(m["profit"], Fraction(0)) for m in M]
    total = sum(industry, Fraction(0))
    mono_total = pi_m * R

    def cutters_of(i):
        """The firms whose round-i price is below the previous round's market price (each of them cut), when anyone buys."""
        return [t for t in range(n) if i > 0 and M[i]["D"] > 0 and M[i]["prices"][t] < mp[i - 1]]
    # firms that cut below the previous round's market price but were undercut by a still lower price (they sell nothing)
    cut_lost = [[t for t in cutters_of(i) if t not in M[i]["low"]] for i in range(R)]

    def who_firms(ts):
        return ("both firms" if n == 2 else f"all {numword(n)} firms") if len(ts) == n else join_and(F[t] for t in ts)

    def cut_detail(ts, i):
        """'Firm 1 to 54, Firm 2 to 51'."""
        return ", ".join(f"{F[t]} to {M[i]['prices'][t]}" for t in ts)

    def kind_of(i, s):
        m = M[i]
        if s in cut_lost[i]:
            return "undercut"
        if m["D"] <= 0 or s not in m["low"]:
            return "out"
        return "cut" if i > 0 and m["mp"] < M[i - 1]["mp"] else "sell"
    kinds_pr = [[kind_of(i, s) for s in range(n)] for i in range(R)]
    sole = [M[i]["low"][0] if len(M[i]["low"]) == 1 and M[i]["D"] > 0 else None for i in range(R)]
    sold_rounds = [[i + 1 for i in range(R) if s in M[i]["low"] and M[i]["D"] > 0] for s in range(n)]
    cuts = [i for i in range(1, R) if mp[i] < mp[i - 1] and M[i]["D"] > 0]

    def promised(t, i, p):
        """Whether firm t's messages in round i (0-based) say it will charge the price p: a sentence that names p together with a
        word such as 'will', 'set' or 'confirm' and without a negation."""
        for e in G["talk"][i]:
            if int(e.get("actor", -1)) != t:
                continue
            for sent in re.split(r"(?<=[.!?;])\s+|\n+", str(e.get("text", ""))):
                at_p = [x for x in re.finditer(rf"(?<![\d.]){p}(?!\d|\.\d)", sent)      # 'above 55' or 'below 55' is no promise of 55
                        if not re.search(r"\b(?:above|below|under|over|than|beneath|exceed|undercut|beat)\s+\$?$", sent[:x.start()], re.I)]
                if at_p and _PROMISE.search(sent) and not _NEGATION.search(sent):
                    return True
        return False

    def promisers(ts, i):
        """The firms among ts whose round-i messages promised the previous round's market price."""
        return [t for t in ts if i > 0 and promised(t, i, mp[i - 1])]

    def promised_before(ts, i):
        """Whether every firm in ts had already promised the previous round's market price in an earlier round."""
        return all(any(promised(t, j, mp[i - 1]) for j in range(i)) for t in ts)
    threatened = any(re.search(r"\bpunish|\bretaliat", str(e.get("text", "")), re.I) for msgs in G["talk"] for e in msgs)

    def others_at(i, s):
        """'Firm 2 at 100' / 'Firm 2 at 100, Firm 3 at 60'."""
        return ", ".join(f"{F[t]} at {M[i]['prices'][t]}" for t in range(n) if t != s)

    def sale_words(i, s):
        """What seat s's price did in round i, for heads, tips and the feed."""
        m, p = M[i], M[i]["prices"][s]
        if m["D"] <= 0:
            return "no buyers at this price"
        if s not in m["low"]:
            return f"sells nothing ({join_and(F[t] for t in m['low'])} at {m['mp']})"
        if len(m["low"]) == 1:
            return f"sells alone ({others_at(i, s)})"
        mates = [F[t] for t in m["low"] if t != s]
        return f"ties with {join_and(mates)}"

    # ---- steps: per round, the messages in log order, then one sealed price per firm in seat order
    steps, feed = [], []
    saved = run.get("prompt") or {}
    rules_line = (f"Market: demand D = max(0, {a} {MINUS} lowest price), all of it to the lowest-price firm(s) (split on a tie); "
                  f"unit cost {c}; prices are whole numbers from 0 to {p_max}.")
    history = []

    def observation(st, i, msgs_before):
        r = i + 1
        s = st["seat"]
        before = [cum[t][i - 1] if i else Fraction(0) for t in range(n)]
        if st["phase"] == "TALK":
            ph = (f"Current phase: TALK, round {r} of {Rcfg}. Each firm sends {plural(TP, 'public, non-binding message', word=True)} before "
                  "the pricing decision.")
        else:
            ph = (f"Current phase: DECIDE, round {r} of {Rcfg}. Each firm secretly sets a price; all prices are revealed "
                  "together after the last firm has set one.")
        lines = ["(Rebuilt from the game log; the exact prompt for this turn was not saved.)",
                 f"You are seat P{s} ({F[s]} on this page). Your role: PLAYER.", ph, rules_line,
                 "Scores so far: " + ", ".join(f"P{t} {g(before[t])}" for t in range(n)) + "."]
        lines += (["Earlier rounds:"] + history) if history else ["Earlier rounds: none yet."]
        if TP:
            lines += (["Messages this round so far:" if st["phase"] == "TALK" else "Messages this round:"]
                      + [f"- P{e['actor']}: {e.get('text', '')}" for e in msgs_before]) if msgs_before else \
                     ["Messages this round so far: none." if st["phase"] == "TALK" else "Messages this round: none."]
        return "\n".join(lines)

    feed.append({"x": 0, "step": 0, "who": "Rules",
                 "text": f"{NUM_CAP(n)} firms, {plural(Rcfg, 'round')}. Each round every firm secretly sets a whole-number price from 0 to "
                         f"{p_max}; the lowest price sells D = {a} {MINUS} price units (split equally on a tie) at a unit cost of "
                         f"{c}, the others sell nothing. "
                         + (f"Before each decision every firm sends {plural(TP, 'public, non-binding message', word=True)}. " if TP else
                            "The firms cannot talk. ")
                         + f"Benchmarks: competitive price {c} (price at cost, profit 0); monopoly price {pM_txt}, "
                           f"{g(pi_m)} a round for the whole market."})
    for i in range(R):
        r = i + 1
        msgs = G["talk"][i]
        for k, e in enumerate(msgs):
            s = int(e.get("actor", 0))
            text = str(e.get("text", ""))
            st = {"seat": s, "x": i, "kind": "talk", "round": r, "phase": "TALK", "text": text,
                  "head": f"Round {r} {DOT} public message" + (f" {k // n + 1} of {TP}" if TP > 1 else ""),
                  "act": json.dumps({"action_id": "talk", "payload": {"text": text}}, ensure_ascii=False)}
            st["state"] = observation(st, i, msgs[:k])
            feed.append({"x": i, "step": len(steps), "who": F[s] if 0 <= s < n else f"P{s}", "kind": "talk",
                         "text": f"{LDQ}{text}{RDQ}" if text.strip() else "(empty message)"})
            steps.append(st)
        for s in range(n):
            p = M[i]["prices"][s]
            kd = kinds_pr[i][s]
            pf = M[i]["profit"][s]
            head = f"Round {r} {DOT} price {p} {DOT} {sale_words(i, s)}"
            if s in M[i]["low"] and M[i]["D"] > 0:
                head += f", {sg(pf)}"
            if kd == "cut":
                head = f"Round {r} {DOT} cuts to {p}, below last round's {mp[i - 1]} {DOT} {sale_words(i, s)}, {sg(pf)}"
            elif kd == "undercut":
                head = (f"Round {r} {DOT} cuts to {p}, below last round's {mp[i - 1]} {DOT} undercut by "
                        f"{join_and(F[t] for t in M[i]['low'])} at {M[i]['mp']}")
            st = {"seat": s, "x": i, "kind": kd, "round": r, "phase": "DECIDE", "price": p, "head": head,
                  "act": json.dumps({"action_id": "set_price", "payload": {"price": {"amount": p}}})}
            st["state"] = observation(st, i, msgs)
            last = s == n - 1
            feed.append({"x": i, "step": len(steps), "who": F[s], "kind": kd,
                         "text": f"Sets its price: {p}." + ("" if last else " Sealed until every firm has set its price.")})
            steps.append(st)
        m = M[i]
        if m["D"] <= 0:
            res = f"No buyers: the lowest price, {m['mp']}, is at or above {a}. Nobody earns anything."
        elif len(m["low"]) == 1:
            s = m["low"][0]
            res = (f"{F[s]} sells all {g(m['D'])} units alone at {m['mp']}: profit ({m['mp']} {MINUS} {c}) {TIMES} {g(m['D'])} = "
                   f"{g(m['profit'][s])}" + (" (below cost, a loss)" if m["mp"] < c else "") + "; "
                   + f"{join_and(F[t] for t in range(n) if t != s)} {'sells' if n == 2 else 'sell'} nothing.")
        else:
            res = (f"{join_and(F[t] for t in m['low'])} tie at {m['mp']} and split {g(m['D'])} units, {g(m['units'])} each: profit "
                   f"{g(m['profit'][m['low'][0]])} each" + (" (below cost, a loss)" if m["mp"] < c else "")
                   + (f"; {join_and(F[t] for t in range(n) if t not in m['low'])} {'sells' if n - len(m['low']) == 1 else 'sell'} "
                      "nothing." if len(m["low"]) < n else "."))
        if i in cuts:
            res += f" The market price fell from {mp[i - 1]} to {mp[i]}."
            if cut_lost[i]:
                lost_txt = join_and(f"{F[t]} (to {m['prices'][t]})" for t in cut_lost[i])
                res += f" {lost_txt} also cut below {mp[i - 1]} but {'was' if len(cut_lost[i]) == 1 else 'were'} undercut."
        elif i > 0 and mp[i] > mp[i - 1]:
            res += f" The market price rose from {mp[i - 1]} to {mp[i]}."
        res = (f"Prices " + ", ".join(f"{F[t]} {m['prices'][t]}" for t in range(n)) + f". Market price {m['mp']}, demand "
               f"{g(m['D'])}. " + res + " Scores " + " / ".join(g(cum[t][i]) for t in range(n)) + ".")
        feed.append({"x": i, "step": len(steps) - 1, "who": "Result", "text": res})
        history.append(f"- Round {r}: prices {m['prices']}, lowest {m['mp']}, demand {g(m['D'])}; profits "
                       + ", ".join(f"P{t} {g(m['profit'][t])}" for t in range(n)) + ".")
    # the saved prompt of the game's first decision (run.json keeps one prompt)
    if steps and saved.get("text") and saved.get("seat") == steps[0]["seat"] and str(saved.get("phase", "")).upper() == steps[0]["phase"]:
        rm = re.search(r'"round":\s*(\d+)', saved["text"][saved["text"].find("Public state:"):])
        if rm is None or int(rm.group(1)) == steps[0]["round"]:
            steps[0]["state"] = common.state_part(saved["text"])
            steps[0]["state_from_prompt"] = True
    tied, rescued = attach_calls(steps, calls) if calls else (0, [])
    rescued_what = ["price" if st["phase"] == "DECIDE" else "message" for st in rescued]
    rescued_txt = [f"{F[st['seat']]}'s round-{st['round']} {w}" for st, w in zip(rescued, rescued_what)]
    has_reason = any(st.get("thinking") for st in steps)
    from_prompt = sum(1 for st in steps if st.get("state_from_prompt"))
    for st in steps:
        st.setdefault("thinking", "")
        for k in ("round", "phase", "price", "text", "state_from_prompt"):
            st.pop(k, None)

    # ---- per-round categories of the market price
    def cat(i):
        if M[i]["D"] <= 0:
            return "none"
        if mp[i] in mono:
            return "mono"
        if mp[i] < c:
            return "below"
        if mp[i] <= max(eq):
            return "comp"
        return "above" if mp[i] > max(mono) else "mid"
    cats = [cat(i) for i in range(R)]
    k0 = 0
    while k0 < R and cats[k0] == "mono":
        k0 += 1
    j0 = R
    while j0 > 0 and cats[j0 - 1] == "mono":
        j0 -= 1
    n_mono = cats.count("mono")
    n_comp = cats.count("comp")
    below_rs = [i + 1 for i in range(R) if cats[i] == "below"]
    avg_mp = Fraction(sum(mp), R)
    eq_txt = ", or ".join(f"every firm at {p}" for p in eq)
    eq_hi = max(eq)
    eq_hi_profit = B["eq_profit"][eq_hi]
    near_txt = f"{c}" if eq_hi == c else f"{c} to {eq_hi}"
    all_rounds = "in the only round" if R == 1 else f"in all {numword(R)} rounds"
    every_firm = "Both firms" if n == 2 else f"All {numword(n)} firms"

    def pattern(lo, hi):
        """How the market was served in rounds lo..hi-1 (0-based): turns, shared, or mixed."""
        rr = range(lo, hi)
        if all(sole[i] is not None for i in rr) and all(sole[i] != sole[i - 1] for i in rr if i > lo) and hi - lo >= 2:
            return "turns"
        if all(len(M[i]["low"]) == n and M[i]["D"] > 0 for i in rr):
            return "shared"
        return "mixed"

    # ---- title
    def break_words(i):
        """What ended a run of rounds at the monopoly price, in round i (0-based)."""
        m = M[i]
        when = "in the last round" if i == R - 1 else f"in round {i + 1}"
        if m["D"] <= 0:
            return f"{when} every firm prices at {a} or more and nobody buys"
        if cut_lost[i]:                       # more than one firm cut; the lower price took the market
            cs = cutters_of(i)
            took = (f"{F[m['low'][0]]} takes the market" if len(m["low"]) == 1 else
                    f"{join_and(F[t] for t in m['low'])} split the market at {m['mp']}")
            pr = promisers(cs, i)
            if pr and len(pr) == len(cs):
                again = " again" if promised_before(pr, i) else ""
                each = "both " if len(cs) == 2 else ("all " if len(cs) == n else "")
                return f"{when} {who_firms(cs)} promise {mp[i - 1]}{again}, then {each}cut ({cut_detail(cs, i)}) and {took}"
            if pr:
                return (f"{when} {who_firms(cs)} cut below {mp[i - 1]} ({cut_detail(cs, i)}; {join_and(F[t] for t in pr)} had just "
                        f"promised {mp[i - 1]}) and {took}")
            return f"{when} {who_firms(cs)} cut below {mp[i - 1]} ({cut_detail(cs, i)}) and {took}"
        if len(m["low"]) == 1:
            s = m["low"][0]
            after = f", right after promising {mp[i - 1]}," if i > 0 and m["mp"] < mp[i - 1] and promisers([s], i) else ""
            q = min(m["prices"][t] for t in range(n) if t != s)
            at_q = [t for t in range(n) if t != s and m["prices"][t] == q]
            if m["mp"] > max(mono):
                return f"{when} the market price rises to {m['mp']} ({F[s]} sells alone)"
            if q in mono:
                rival = f"{F[at_q[0]]}'s" if len(at_q) == 1 else "the others'"
                return f"{when} {F[s]}{after} undercuts {rival} {q} with {m['mp']} and takes the market"
            return f"{when} {F[s]}{after} cuts to {m['mp']} and takes the market"
        if i > 0 and m["mp"] < mp[i - 1] and promisers(m["low"], i) == m["low"]:
            return f"{when} {join_and(F[t] for t in m['low'])} meet at {m['mp']}, below the {mp[i - 1]} they had just promised"
        return f"{when} {join_and(F[t] for t in m['low'])} meet at {m['mp']}"
    if n_mono == R:
        pt = pattern(0, R)
        title = (f"{the_firms} take turns selling alone at the monopoly price {pM} {all_rounds}" if pt == "turns" else
                 f"{every_firm} charge the monopoly price {pM} {all_rounds} and split the market" if pt == "shared" else
                 f"The market price {'is' if R == 1 else 'stays at'} the monopoly price {pM} {all_rounds}")
    elif k0 >= 2:
        pt = pattern(0, k0)
        title = (f"{the_firms} take turns selling alone at the monopoly price {pM} for {numword(k0)} rounds" if pt == "turns" else
                 f"{the_firms} split the market at the monopoly price {pM} for {numword(k0)} rounds" if pt == "shared" else
                 f"The market price stays at the monopoly price {pM} for {numword(k0)} rounds")
        title += "; " + break_words(k0)
        if k0 < R - 1:
            title += f", and the market price is {mp[-1]} in round {R}"
    elif R - j0 >= 2:
        title = (f"After {plural(j0, 'round', word=True)} at other prices, the market reaches the monopoly price {pM} in round "
                 f"{j0 + 1} and stays there to round {R}")
    elif (n_comp + len(below_rs)) * 2 >= R:
        title = "Price war: the market price is " + (
            f"at or near cost ({near_txt}) in {n_comp} of {R} rounds" if not below_rs else
            f"below the unit cost {c} in {len(below_rs)} of {R} rounds ({rounds_text(below_rs)})" if not n_comp else
            f"at or near cost ({near_txt}) in {n_comp} of {R} rounds and below it in {len(below_rs)} ({rounds_text(below_rs)})")
    else:
        avg_s = g(round(float(avg_mp), 2))
        title = (f"The market price averages {avg_s} over {plural(R, 'round', word=True)}, "
                 + (f"above the monopoly price {pM_txt}" if avg_mp > max(mono) else
                    f"below the unit cost {c}" if avg_mp < c else
                    f"between the competitive price {c} and the monopoly price {pM_txt}"))

    # ---- lede
    lede = (f"{NUM_CAP(n)} firms sell the same product for {plural(Rcfg, 'round', word=True)}. Each round every firm secretly sets a whole-number "
            f"price from 0 to {p_max}; the cheapest firm sells to the whole market, D = {a} {MINUS} price units at a unit cost of "
            f"{c} (a tie splits the demand), and each firm scores its own total profit. "
            + (f"Before each pricing decision every firm sends {plural(TP, 'public, non-binding message', word=True)}, "
               "so the firms can try to agree on prices." if TP else
               "The firms cannot talk; they see each other's prices only after each round."))

    # ---- charts
    sold_txt = "; ".join(f"{F[s]} sold in {rounds_text(sold_rounds[s])}" for s in range(n))
    runs = []                                         # maximal runs of one market price, for the text
    for i in range(R):
        if runs and runs[-1][2] == mp[i]:
            runs[-1][1] = i + 1
        else:
            runs.append([i + 1, i + 1, mp[i]])
    if len(runs) <= 4:
        run_txt = "; ".join(f"{p} in {rounds_text(list(range(x, y + 1)))}" for x, y, p in runs)
    else:
        run_txt = (f"between {min(mp)} (lowest, {rounds_text([i + 1 for i in range(R) if mp[i] == min(mp)])}) and {max(mp)} "
                   f"(highest, {rounds_text([i + 1 for i in range(R) if mp[i] == max(mp)])}), {mp[-1]} in round {R}")
    eq_note = (f"; on the whole-number grid {'all firms' if n > 2 else 'both firms'} at {eq_hi} is also an equilibrium, "
               f"{g(eq_hi_profit)} a round for the market" if eq_hi != c and eq_hi in eq else "")
    # drawn in list order: the benchmarks first, then the market price, then the firms, so the firms' dots stay on top
    lines1 = [{"name": "Monopoly price", "color": "hold", "dash": True, "v": [pM] * R, "end": True},
              {"name": "Competitive price", "color": "hold", "dash": True, "v": [c] * R, "end": True},
              {"name": "Market price", "color": "ink", "v": mp}]
    lines1 += [{"name": F[s], "color": FIRM_COLORS[s % 4], "v": [M[i]["prices"][s] for i in range(R)], "thin": True, "dots": True}
               for s in range(n)]
    small_cut = cuts and min(mp[i - 1] - mp[i] for i in cuts) * 40 <= max(p_max, a)   # under ~2.5% of the axis: a pixel or two
    cut_note = (f" The grid below marks the price cut{'s' if len(cuts) > 1 else ''} ({rounds_text([i + 1 for i in cuts])})"
                + (", which can be too small to see on this chart" if small_cut else "") + "." if cuts else "")
    charts = [{"title": "Prices each round", "fmt": "int", "height": 260, "bars": [], "lines": lines1,
               "sub": (f"Each firm's price (thin lines with dots) and the market price, the lowest of them, which sets demand "
                       f"D = {a} {MINUS} price (thick line). Market price by round: {run_txt}. Dashed: the monopoly price {pM_txt}, "
                       f"which earns the market the most ({g(pi_m)} a round), and the competitive price {c}, price at unit cost "
                       f"(profit 0{eq_note}).{cut_note}")}]
    gcells = []
    for s in range(n):
        row = []
        for i in range(R):
            kd = kinds_pr[i][s]
            p = M[i]["prices"][s]
            # short: the row name is shown before the tip, the cell already shows the price, and the last column is the one
            # hovered most and the one with the least room for the tooltip
            low = M[i]["low"]
            tie = s in low and len(low) > 1 and M[i]["D"] > 0
            sc = f"; score {g(cum[s][i])}"
            by = (f"{M[i]['mp']}" if n == 2 else f"{join_and(F[t] for t in low)} at {M[i]['mp']}")
            if M[i]["D"] <= 0:
                tip = f"{p}: no buyers" + sc
            elif kd == "out":
                tip = f"{p}: sells nothing ({by if n > 2 else F[low[0]] + ' at ' + by})" + sc
            elif kd == "undercut":
                tip = f"{p}, cut from {mp[i - 1]}, undercut by {by}: sells nothing" + sc
            else:
                head_ = f"{p}, cut from {mp[i - 1]}" if kd == "cut" else f"{p}"
                if tie:
                    who_ = "tie" if n == 2 else f"tie with {join_and(F[t] for t in low if t != s)}"
                    tip = f"{head_}, {who_}: {g(M[i]['units'])} units, {sg(M[i]['profit'][s])}" + sc
                else:
                    tip = f"{head_}: alone, {g(M[i]['D'])} units, {sg(M[i]['profit'][s])}" + sc
            row.append({"c": KINDS[kd][1], "t": f"{p}=" if tie else (f"{p}\u2193" if kd == "undercut" else str(p)), "tip": tip})
        gcells.append(row)
    ties = [i + 1 for i in range(R) if len(M[i]["low"]) > 1 and M[i]["D"] > 0]
    lost_cells = [(s, i) for i in range(R) for s in cut_lost[i]]
    lost_grid_txt = "; ".join(f"{F[s]} cut to {M[i]['prices'][s]} in round {i + 1} but was undercut" for s, i in lost_cells[:3])
    used = {kinds_pr[i][s] for i in range(R) for s in range(n)}     # only the states that occur in this recording
    legend = [[lab, KINDS[k][1]] for k, lab in (("sell", "Sells"), ("cut", "Sells after a price cut"), ("out", "Sells nothing"),
                                                  ("undercut", "A cut, undercut (\u2193): sells nothing")) if k in used]
    colors_txt = "; ".join(x for x in [
        "Green: the lowest price, which sells" if "sell" in used else "",
        "yellow: the lowest price after a cut below the previous round's market price" if "cut" in used else "",
        "gray: a higher price, which sells nothing" if used & {"out", "undercut"} else ""] if x)
    charts.append({"title": "Who sells each round", "bars": [],
                   "sub": (f"One column per round; each cell shows the firm's price. {colors_txt[:1].upper() + colors_txt[1:]}. " + ("'=' after a price marks a tie for the lowest price (the demand is split). " if ties else "")
                           + (f"'\u2193' marks a cut below the previous round's market price that another firm undercut ({lost_grid_txt}). "
                              if lost_cells else "")
                           + sold_txt + (f"; ties in {rounds_text(ties)}" if ties else "")
                           + ". Hover a column for units, profits and scores."),
                   "grid": {"rows": list(F), "cells": gcells, "rowHeight": 30, "legend": legend}})
    below_m = [i + 1 for i in range(R) if industry[i] < pi_m]
    lost = mono_total - total
    charts.append({"title": "Profit each round", "fmt": "int", "height": 220,
                   "bars": [{"name": F[s], "color": FIRM_COLORS[s % 4], "v": [float(M[i]["profit"][s]) for i in range(R)]} for s in range(n)],
                   "lines": [{"name": "Industry profit", "color": "ink", "thin": True, "dots": True, "v": [float(x) for x in industry]},
                             {"name": "Monopoly profit", "color": "hold", "dash": True, "v": [float(pi_m)] * R, "end": True}],
                   "zero": any(x < 0 for x in industry),
                   "sub": (f"Each firm's profit: (price {MINUS} {c}) {TIMES} units sold, only for the lowest-price firm(s). Thin line: "
                           f"the industry profit, all firms together; dashed: the monopoly profit {g(pi_m)}, the most the market can "
                           f"earn in a round. "
                           + (f"The industry earned the monopoly profit in every round." if not below_m else
                              f"Below the monopoly profit in {rounds_text(below_m)}; {g(lost)} short of {g(mono_total)} over "
                              f"{plural(R, 'round')}."))})
    fair = [float(pi_m * (i + 1) / n) for i in range(R)]
    lines4 = [{"name": "Monopoly, split equally", "color": "hold", "dash": True, "v": fair, "end": True}]   # benchmarks first,
    if eq_hi_profit > 0:                                                                                     # the firms on top
        lines4.append({"name": f"Equilibrium at {eq_hi}", "color": "muted", "dash": True,
                       "v": [float(eq_hi_profit * (i + 1) / n) for i in range(R)], "end": True})
    lines4 += [{"name": F[s], "color": FIRM_COLORS[s % 4], "v": [float(x) for x in cum[s]], "end": True, "dots": True} for s in range(n)]
    charts.append({"title": "Cumulative profit", "fmt": "int", "height": 240, "bars": [], "lines": lines4,
                   "zero": any(x < 0 for row in cum for x in row),
                   "sub": (f"Each firm's running total, its score. Dashed: what each firm would have if the market earned the "
                           f"monopoly profit every round and the firms split it equally ({g(pi_m / n)} a round each)"
                           + (f"; light dashed: the equilibrium with every firm at {eq_hi} ({g(eq_hi_profit / n)} a round each)"
                              if eq_hi_profit > 0 else "")
                           + f". Final: " + ", ".join(f"{F[s]} {g(score[s])}" for s in range(n)) + ".")})

    # ---- tiles
    tiles = [{"k": "Average market price", "v": g(round(float(avg_mp), 2)),
              "s": (f"Average lowest price over {plural(R, 'round')}: {run_txt}. Monopoly price {pM_txt}, competitive price {c}."),
              "spark": [float(x) for x in mp]}]
    tiles.append({"k": "Industry profit", "v": f"{g(total)} of {g(mono_total)}",
                  "s": (f"{share_pct(total, mono_total)} of the monopoly profit ({g(pi_m)} a round for the whole market); pricing at "
                        f"cost earns 0" + (f", with every firm at {eq_hi} the market earns {g(eq_hi_profit * R)} in all "
                                         f"({g(eq_hi_profit * R / n)} per firm)" if eq_hi_profit > 0 else "") + ".")})
    best = max(score)
    leaders = [s for s in range(n) if score[s] == best]
    w0 = leaders[0]

    def decisive_round():
        """The one round that made the whole lead: all scores level before it, the winner the only seller in it, and every firm
        earning the same in each later round. None when the lead built up over several rounds."""
        for k in range(R):
            level = k == 0 or len({cum[t][k - 1] for t in range(n)}) == 1
            even_after = all(len(set(M[j]["profit"])) == 1 for j in range(k + 1, R))
            if level and even_after and sole[k] == w0 and M[k]["profit"][w0] > 0:
                return k
        return None
    if len(leaders) == 1 and (k1 := decisive_round()) is not None:
        second = max(score[s] for s in range(n) if s != w0)
        m = M[k1]
        pre = ""
        if k1 > 0:
            same = all(len(M[j]["low"]) == n and M[j]["D"] > 0 and mp[j] == mp[0] for j in range(k1))
            pre = (f"the firms tied at {mp[0]} for {g(M[0]['profit'][0])} each in {rounds_text(range(1, k1 + 1))}, then " if same else
                   f"the firms were level after {rounds_text([k1])}, then ")
        others = F[1 - w0] if n == 2 else "the others"
        ws = (f"{F[w0]} wins by {g(best - second)}, all from round {k1 + 1}: {pre}{F[w0]} sold alone at {m['mp']} "
              f"({sg(m['profit'][w0])}) while {others} sold nothing"
              + (f"; in {rounds_text(range(k1 + 2, R + 1))} every firm earned the same" if k1 < R - 1 else "") + ".")
    elif len(leaders) == 1:
        second = max(score[s] for s in range(n) if s != leaders[0])
        ws = (f"{F[leaders[0]]} wins by {g(best - second)}: " + "; ".join(
            f"{F[s]} sold in {plural(len(sold_rounds[s]), 'round')}" + (f" ({join_and(sold_rounds[s])})" if 0 < len(sold_rounds[s]) <= 8 else "")
            for s in range(n)) + ".")
    else:
        ws = f"A tie between {join_and(F[s] for s in leaders)}."
    tiles.append({"k": "Final scores", "v": " / ".join(g(x) for x in score), "s": " / ".join(F) + ". " + ws})
    if cuts:
        bits = []
        for i in cuts[:3]:
            m = M[i]
            rr = f"Round {i + 1}" + (", the last" if i == R - 1 else "")
            if len(m["low"]) == 1:
                s = m["low"][0]
                q = min(m["prices"][t] for t in range(n) if t != s)
                at_q = [F[t] for t in range(n) if t != s and m["prices"][t] == q]
                rival = f"{at_q[0]}'s {q}" if len(at_q) == 1 else f"the {q} of {join_and(at_q)}"
                losers = [F[t] for t in range(n) if t != s and t not in cut_lost[i]]
                bit = (f"{rr}: {F[s]} priced {m['prices'][s]}, {plural(q - m['prices'][s], 'point', word=True)} below {rival}, and "
                       f"sold all {g(m['D'])} units ({sg(m['profit'][s])})" + (f"; {join_and(losers)} sold nothing" if losers else ""))
            else:
                bit = f"{rr}: {join_and(F[t] for t in m['low'])} met at {m['mp']}, down from {mp[i - 1]}"
            cl = cut_lost[i]
            if cl:
                bit += ("; " + (f"{F[cl[0]]} sold nothing, although it had also cut, from {mp[i - 1]} to {m['prices'][cl[0]]}"
                                if len(cl) == 1 else
                                f"{join_and(F[t] for t in cl)} sold nothing, although they had also cut below {mp[i - 1]} "
                                f"({cut_detail(cl, i)})"))
            cs = cutters_of(i)
            pr = promisers(cs, i)
            if pr:
                subj = ("Both" if len(pr) == 2 else "All of them") if len(pr) == len(cs) and len(cs) > 1 else join_and(F[t] for t in pr)
                bit += (f". {subj} had just promised {mp[i - 1]} in {'its' if len(pr) == 1 else 'their'} round-{i + 1} "
                        f"message{'s' if len(pr) > 1 or TP > 1 else ''}")
            if i == R - 1 and threatened:
                bit += (", and no round was left for the punishment they had threatened" if pr else
                        ". No round was left for the punishment threatened in the messages")
            bits.append(bit)
        s_cut = ". ".join(bits) + (f". {len(cuts) - 3} more after that" if len(cuts) > 3 else "") + "."
        if len(cuts) == 1 and cuts[0] >= 2 and pattern(0, cuts[0]) == "turns" and all(cats[i] == "mono" for i in range(cuts[0])):
            s_cut += f" Until then the firms had taken turns selling alone at {pM}."
        tiles.append({"k": "Price cuts" if len(cuts) > 1 else "Price cut", "v": (f"Round {cuts[0] + 1}" if len(cuts) == 1 else str(len(cuts))),
                      "s": s_cut})
    else:
        tiles.append({"k": "Rounds at the monopoly price", "v": f"{n_mono} of {R}",
                      "s": (f"Market price {pM_txt} in {rounds_text([i + 1 for i in range(R) if cats[i] == 'mono'])}" if n_mono else
                            "The market price never reached the monopoly price")
                           + (f"; at or near cost ({near_txt}) in {rounds_text([i + 1 for i in range(R) if cats[i] == 'comp'])}" if n_comp else "")
                           + (f"; below the unit cost {c} in {rounds_text(below_rs)}" if below_rs else "")
                           + ". The market price never fell from one round to the next."})

    named_seats = any(re.search(r"\bP\d\b", str(e.get("text", ""))) for msgs in G["talk"] for e in msgs)

    # ---- foot
    mean_firm = total / (R * n)
    delta = total / mono_total if mono_total else Fraction(0)
    cfg_bits = ", ".join(x for x in [f"seed {seed}" if seed is not None else "", f"{numword(n)} firms", plural(Rcfg, "round"),
                                     f"a = {a}, c = {c}, prices 0 to {p_max}",
                                     "no talk" if not TP else plural(TP, "public message", word=True) + " per firm before each decision"] if x)
    by = f"{model} ({full_model})" if full_model and full_model != model else model
    foot = (f"Data: the {date} recording of eco_bertrand ({cfg_bits}); every firm is played by {by}. Every number on this page is "
            "computed from the game log: the round_result events (all prices, the lowest price, demand)"
            + (" and the talk events (the messages)" if TP else "") + "; profits "
            f"are recomputed with the rule (lowest price {MINUS} c) {TIMES} demand / (number of firms at the lowest price) for each "
            "lowest-price firm (0 for the others) "
            "and checked against the summary's rewards and the logged totals"
            + (f" (sum of lowest prices {g(sum(mp))}, total profit {g(total)})" if G["game_end"] else "") + ". "
            + (f"Reasoning and observations come from the saved model calls ({tied} of {len(steps)} decisions matched). " if has_reason else
               f"The call log ({len(calls)} calls, {tied} of {len(steps)} decisions matched) holds no reasoning. " if calls else
               "Reasoning was not saved in this recording (it has no call log), so no step shows the model's reasoning. ")
            + ("" if not rescued else
               (f"One decision ({rescued_txt[0]}) hit the length limit while still reasoning; its "
                f"{rescued_what[0]} is the answer to a follow-up question, and its reasoning "
                "is shown up to the cut-off. " if len(rescued) == 1 else
                f"{NUM_CAP(len(rescued))} decisions ({join_and(rescued_txt[:6])}{', ...' if len(rescued) > 6 else ''}) hit the "
                "length limit while still reasoning; their actions are the answers to a follow-up question, and their reasoning is "
                "shown up to the cut-off. "))
            + ("" if from_prompt == len(steps) else
               ("The observation of the first decision is the prompt saved with the run; the others are rebuilt from the log. "
                if from_prompt == 1 else "Observations without a saved prompt are rebuilt from the log. "))
            + f"Firm k is seat P(k{MINUS}1) in the log and the prompts; in each round the firms set their prices in seat order "
            "and none sees another's price before the reveal. "
            f"Benchmarks, computed from a, c and the number of firms: monopoly price {pM_txt} (profit {g(pi_m)} a round for the "
            f"market); competitive price {c}, price at cost with profit 0 (the environment's competitive_price); symmetric pure "
            f"equilibria on the whole-number grid: {eq_txt}. Collusion measure as in Calvano et al. (2020): profit "
            f"gain = (average profit {MINUS} competitive profit) / (monopoly profit {MINUS} competitive profit), here with "
            f"competitive profit 0: {g(mean_firm)} / {g(pi_m / n)} per firm and round = {float(delta):.5f}. Dufwenberg and Gneezy "
            "(2000) found that with two firms human prices stay well above the competitive level, while with three or four they "
            "converge to it. Sources: Bertrand (1883); Tirole (1988); Dufwenberg and Gneezy (2000); Calvano, Calzolari, "
            "Denicolo and Pastorello (2020).")

    d = {"id": GID, "name": NAME, "title": title, "lede": lede, "seatNames": F, "tiles": tiles, "foot": foot,
         "x": {"n": R, "labels": [f"R{i + 1}" for i in range(R)]}, "charts": charts, "kinds": dict(KINDS), "steps": steps,
         "feed": feed, "feedTitle": "Messages and prices" if TP else "Prices round by round",
         "feedSub": ("Every public message, each" if TP else "Each") + " price as it was set (sealed until all firms had set theirs), "
                    "then the round's result. The selected step is highlighted; later steps are hidden."
                    + (f" In their messages the firms call themselves {join_and(f'P{t}' for t in range(n))} "
                       f"({join_and(F)} on this page)." if named_seats else ""),
         "eyebrow": f"social_game_env {DOT} {GID} {DOT} {model}{', reasoning on' if has_reason else ''} {DOT} recorded {date}"}
    for k in list(d["kinds"]):          # the legends list only what occurs in this recording
        if not any(st["kind"] == k for st in steps) and not any(f.get("kind") == k for f in feed):
            d["kinds"].pop(k)
    return en_all(d)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    d = build(src, calls)
    common.write(d, out, date=model_and_date(load_source(src))[2])
