"""Replay data for eco_trading_impact (two traders in one market that switches between a bull and a bear regime, as in
eco_trading_regime, plus linear temporary price impact set by the two traders' combined order: Almgren and Chriss 2001, with the
multi-trader price pressure of Carlin, Lobo and Viswanathan 2007, Eq. 1).

usage: python3 -I eco_trading_impact.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md), a file inside
          one, or the rendered Markdown replay of one (one "- **event** ... `{json}`" line per logged event).
  OUT_DIR where <OUT_DIR>/eco_trading_impact.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). A call belongs to the
          decision of the same seat, phase and round (read from its prompt: "You are seat Pk", "Current phase: X", the public
          state's "round"); when a prompt does not show them, calls are taken in time order, seat by seat. Within a decision's
          calls the first one whose answer is the logged move (the logged message text, or the position the game applied) is the
          decision's call; earlier ones were format retries and their reasoning is kept above it. Its reasoning becomes the step's
          thinking, its prompt the observation and its answer the action.

Each round has talk_passes public messages per trader (logged as talk events), then one secret position decision per trader
(rules: Round Procedure). The log has no event per position decision: both positions are read from the round's round_result
event (position0/1, price_change, price, impact0/1, payoff0/1, score0/1, same_direction, signal_seen), so the decisions are
rebuilt as set_position with the applied position, P0 first. Every number shown is taken from the log, and the identities of
the rules are checked on it: price = previous price + change, impact_i = impact x order_i x (order_0 + order_1),
payoff_i = position_i x change - fee x |order_i| - impact_i, score_i = running sum, same_direction = both orders nonzero with the
same sign, and the game_end totals and summary rewards; any disagreement is printed as a warning.

The joint optimum is computed here, not copied (the rules store no reference solution): backward induction over (round, regime,
combined position) for the plan that maximizes the two traders' total expected P&L, risk-neutral. A net change d of the combined
position costs the pair at least fee x |d| + impact x d^2 (both orders in d's direction), so the combined position is the whole
state; a full two-trader dynamic program over (round, regime, position_0, position_1) is run as a check. With the exact signal
the signal is the true regime and the price path does not depend on the traders, so the plan is replayed on the realised
regimes and price moves. Each round of the traders is judged from the combined position they really held (not the replayed
plan's): their move is the best one when its expected total (round payoff with their actual fees and impact, then the plan's
value) equals the best target's, and the shortfall is shown as the expected cost for the pair; a round where only the replayed
path differs is reported as such."""
import itertools, json, os, re, sys, unicodedata
from decimal import Decimal, ROUND_HALF_UP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_trading_impact"
DEFAULT_DATE = "2026-11-09"
SEATS = ["Trader 1", "Trader 2"]               # seat P0, seat P1 (both role PLAYER, symmetric)
POSITIONS = (-2, -1, 0, 1, 2)
STAY, WITH = 0.8, 0.7                          # defaults: regime stays w.p. 0.8; price moves with the regime w.p. 0.7
RECENT = 12                                    # the engine's prompt lists the last 12 public events (rules: Differences)
MINUS, DOT, ARROW, TIMES, PM = "\u2212", "\u00b7", "\u2192", "\u00d7", "\u00b1"
# Colors: c1 / c2 mean market direction everywhere on the page (long / short, bull / bear, buy / sell); the two traders get
# the neutral pair ink / hold, and the pair's total and the joint optimum c3. "Holds" is drawn muted, like a flat grid cell,
# so that gray stays Trader 2's.
KINDS = {"buy": ["Buys: raises position", "c1"], "sell": ["Sells: lowers position", "c2"], "hold": ["Holds position", "muted"],
         "talk": ["Public message", "c4"]}
SEAT_COLORS = ["ink", "hold"]
# Exact short numbers on the impact and P&L charts: every value here is a multiple of 1/8. Each chart takes 'int' or 'price2'
# when that shows all its values exactly (chart_fmt), else NUM_FMT: the page's 'num' format (up to 3 decimals, trailing zeros
# dropped) once common.FMTS and the page's FMT have it, until then 4 fixed decimals.
NUM_FMT = "num" if "num" in common.FMTS else "price4"


def warn(s):
    print("warning: " + s, file=sys.stderr)


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_ANY = re.compile("[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af\uff00-\uffef]")
_CJK_RUN = re.compile("[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_PUNCT = {"\u3002": ". ", "\u3001": ", ", "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"', "\u3010": "[", "\u3011": "]",
          "\u300a": '"', "\u300b": '"', "\u3008": "<", "\u3009": ">", "\u301c": "~"}


def en(s):
    """The page shows English only: full-width forms become ASCII, CJK punctuation its ASCII counterpart, and any remaining run
    of Chinese, Japanese or Korean characters (the model sometimes reasons in Chinese) a visible marker."""
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


# ------------------------------------------------------------------------------------------------ loading
_TAG_HIDDEN, _TAG_PRIVATE = "\u9690\u85cf", "\u79c1\u5bc6"     # the Markdown replay's "hidden" and "private" tags
_SEED, _MODEL = "\u79cd\u5b50", "\u6a21\u578b"                  # its header words for "seed" and "model"


def _seat(v, prompt=""):
    try:
        return int(str(v).strip().lstrip("Pp"))
    except (TypeError, ValueError):
        return common.seat_of(prompt)


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
        if _TAG_HIDDEN in m.group(2):
            body["_visibility"] = "hidden"
        elif _TAG_PRIVATE in m.group(2):
            body["_visibility"] = "private"
        events.append(body)
    head = text.split("\n## ")[0].split("\n- **")[0]
    rw = re.search(r'(\{"0":[^}]*\})', head)
    seed = re.search(_SEED + r" (\d+)", head)
    model = re.search(_MODEL + r" ([^\s,\uff0c]+)", head)
    over = next((e for e in events if e.get("event") == "game_over"), {})
    summary = {"rewards": json.loads(rw.group(1)) if rw else {}, "variables": over.get("variables", {}), "winners": over.get("winners")}
    run = {"seed": int(seed.group(1)) if seed else None, "model": model.group(1) if model else ""}
    calls = []
    for blk in re.split(r"\n(?=### )", text)[1:]:
        hdr = blk.split("\n", 1)[0]
        ans = re.findall(r"^\*\*[^*\n]+\*\* `(.*)`\s*$", blk, re.M)
        th = re.findall(r"````text\n(.*?)\n````", blk, re.S)
        sm = re.search(r"\bP(\d+)\b", hdr)
        if ans:
            calls.append({"seat": int(sm.group(1)) if sm else None, "prompt": "", "answer": ans[-1].strip(),
                          "reasoning": th[-1] if th else "", "rescued": len(ans) > 1})
    return {"events": events, "summary": summary, "run": run, "config": {}, "transcript": "", "meta": {}, "md_calls": calls, "md": True}


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


def recorded_date(src):
    b = load_source(src)
    return str((b.get("meta") or {}).get("recorded_at") or (b.get("run") or {}).get("recorded_at") or DEFAULT_DATE)[:10]


# ------------------------------------------------------------------------------------------------ text helpers
def nstr(v):
    """Exact short number: 13.5, 12.75, 1.125, 0, 108 (values here are multiples of 1/8)."""
    v = round(float(v), 6) + 0.0
    s = f"{abs(v):.4f}".rstrip("0").rstrip(".")
    return (MINUS if v < 0 else "") + s


def sgn(v):
    v = round(float(v), 6) + 0.0
    return ("+" if v > 0 else "") + nstr(v)


def pos(p):
    return ("+" if p > 0 else MINUS if p < 0 else "") + str(abs(int(p)))


def apos(p):
    """Position as it reads in a prompt (ASCII sign)."""
    return f"+{p}" if p > 0 else str(p)


def r6(v):
    return round(float(v), 6) + 0.0


def c3(v):
    """An expected cost to 3 decimals, half up and free of float noise (1.0350000000000006 and 1.035 both give 1.035)."""
    return float(Decimal(repr(round(float(v), 9))).quantize(Decimal("0.001"), ROUND_HALF_UP))


def cstr(v):
    return f"{c3(v):.3f}"


def csum(vs):
    """The total of costs shown to 3 decimals: the sum of the shown parts, so the parts add up to it."""
    return f"{sum(c3(v) for v in vs):.3f}"


def chart_fmt(series):
    """The shortest of the page's fixed-decimal formats that still shows every value of a chart exactly ('int', 'price2',
    else NUM_FMT: the values here are multiples of 1/8, so 0.125 needs three decimals)."""
    vs = [float(v) for s in series for v in s["v"] if v is not None]
    if all(abs(v - round(v)) < 1e-9 for v in vs):
        return "int"
    if all(abs(v * 100 - round(v * 100)) < 1e-6 for v in vs):
        return "price2"
    return NUM_FMT


def and_join(xs):
    return "none" if not xs else xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def rounds_text(rs):
    return and_join([f"R{r}" for r in rs])


def msgs(k):
    return ("no messages" if not k else "one public, non-binding message" if k == 1 else f"{k} public, non-binding messages")


def act_word(prev, p):
    d = p - prev
    if d == 0:
        return f"holds {pos(p)}"
    return f"{'buys' if d > 0 else 'sells'} {abs(d)} ({pos(prev)} {ARROW} {pos(p)})"


# ---- a message's plan, quoted, for the step head
_POS_TOK = re.compile(r"(?<![\w.])[+\-\u2212][1-4](?![\d.])|\b(?:to|at|from) 0(?![\d.])|\bflat\b", re.I)
_NOW = re.compile(r"\bthis (?:final )?round\b|\bnow\b|\bimmediately\b|\bright away\b", re.I)
_FIRST = re.compile(r"\bfirst\b", re.I)
_HOLDW = re.compile(r"\b(?:hold|holding|keep|keeping|stay|staying)\b", re.I)
_WE = re.compile(r"\b(?:let['\u2019]?s|both|we)\b", re.I)
_COND_START = re.compile(r"^(?:then\s+)?(?:if|on (?:a|the)|when|whenever|should|unless|once|after)\b", re.I)
_COND_ANY = re.compile(r"\b(?:if|when|unless|once)\b|\bflips?\b|\bpersists?\b|\bnext round\b|\bthat round\b", re.I)
_REQ = re.compile(r"^(?:please|kindly|you (?:should|can|could|must|may|need to))\b", re.I)   # a request to the other trader
_LEAD = re.compile(r"^(?:(?:agreed|ok(?:ay)?|sure|yes)\s*[,.:!\u2014\u2013-]+\s*|(?:so|and|but|then)\s+)(?=\S)", re.I)
_TAIL = re.compile(r"\s+(?:since|because|as|rather than|so that|which|while the|to (?:avoid|limit|keep|reduce|capture|minimi[sz]e|"
                   r"smooth|lock|cut|save|halve))\b.*$", re.I)
HEAD_ONE, HEAD_TWO = 64, 80                    # longest single quote, longest pair of quotes joined by an ellipsis
LQ, RQ, ELL = "\u201c", "\u201d", "\u2026"


def _plan_parts(text):
    """(fragment, score, contingency) for each clause of a message that names a position. A clause is a sentence, or a part
    of one between semicolons, dashes or a colon; its fragment is the first comma part naming a position, after any leading
    condition ('On a bear flip, ...'), extended by the following comma parts while short, without a leading 'so' or
    'Agreed,' and without a trailing reason ('to avoid ...', 'since ...'). A condition governs the rest of its sentence: once a
    clause, or its first comma part, starts with one ('If bear: I will switch to -2 this round; please hold +2'), every later
    clause of the sentence is contingent too and is quoted with the condition in front ('If bear: please hold +2'). A request
    to the other trader ('please ...', 'you should ...') weighs a little less than the trader's own plan. The words are the
    trader's own."""
    out = []
    for snt in re.split(r"(?<=[.!?])\s+", (text or "").strip()):
        pieces = re.split(r"(\s*[;\u2014\u2013]\s*|:\s+|\s+-\s+)", snt)
        pending = None                                    # (condition text, joiner) governing the later clauses
        for cl, sep in zip(pieces[0::2], pieces[1::2] + [""]):
            cl = cl.strip().rstrip(".!?").strip()
            parts = [p.strip() for p in re.split(r",\s+", cl) if p.strip()]
            if not parts:
                continue
            core = next((i for i, p in enumerate(parts) if _POS_TOK.search(p)), None)
            if core is None:
                if pending is None and _COND_START.match(parts[0]):
                    pending = (parts[0], ": " if len(parts) == 1 and sep.strip().startswith(":") else ", ")
                continue
            cond_part = next((p for p in parts[:core] if _COND_START.match(p)), None)
            frag = _LEAD.sub("", parts[core])
            for p in parts[core + 1:]:
                if len(frag) + 2 + len(p) > HEAD_ONE or _TAIL.match(" " + p):
                    break
                frag += ", " + p
            cut = _TAIL.sub("", frag)
            frag = cut if _POS_TOK.search(cut) else frag
            req = bool(_REQ.match(frag))
            own = bool(cond_part or _COND_START.match(frag))
            if cond_part:
                frag = cond_part + ", " + frag
            elif not own and pending:
                frag = pending[0] + pending[1] + frag
            cond = own or pending is not None
            if own and pending is None:
                pending = (cond_part or parts[core], ", ")
            score = (2 * bool(_NOW.search(frag)) + bool(_FIRST.search(frag)) + bool(_HOLDW.search(frag))
                     + bool(_WE.search(frag)) - req - 3 * cond - bool(_COND_ANY.search(frag)))
            out.append((frag, score, cond or bool(_COND_ANY.search(frag))))
    return out


def _clip(s, n):
    s = s.strip()
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0].rstrip(",;:") + ELL


def talk_head(text):
    """The plan a message states, in its own words: the clause about this round (most weight to 'this round' / 'now', then
    'first', holding, 'let's' / 'both'; least to contingencies), and after an ellipsis the first contingency ('If the signal
    flips ...') or else the next clause naming a position, when both fit. A message naming no position gives its first
    sentence. Only the first letter of the quote is changed (capitalized)."""
    parts = _plan_parts(text)
    if not parts:
        first = re.split(r"(?<=[.!?])\s+", (text or "").strip())[0].rstrip(".!?")
        main = _clip(_LEAD.sub("", first) or first, HEAD_ONE)
        return LQ + main[:1].upper() + main[1:] + RQ
    i = max(range(len(parts)), key=lambda j: (parts[j][1], -j))
    main = _clip(parts[i][0], HEAD_ONE)
    rest = [(j, p) for j, p in enumerate(parts) if j != i and p[0].lower() != parts[i][0].lower()]
    pick = [p for j, p in rest if p[2]] + [p for j, p in rest if not p[2] and j > i]
    extra = next((p[0] for p in pick if len(main) + 3 + len(p[0]) <= HEAD_TWO), None)
    return LQ + main[:1].upper() + main[1:] + (f" {ELL} " + extra if extra else "") + RQ


# ------------------------------------------------------------------------------------------------ the game from the log
def read_game(b):
    events = b["events"]
    rules = next((e for e in events if e.get("event") == "rules"), {})
    cfgv = (b.get("config") or {}).get("variables") or {}
    endv = (b.get("summary") or {}).get("variables") or {}

    def par(k, dflt=None):
        for src in (rules, endv, cfgv):
            if src.get(k) is not None:
                return src[k]
        return dflt
    rows, talks, nres = [], [], 0
    for e in events:
        ev = e.get("event")
        if ev == "round_result":
            nres += 1
            rows.append(e)
        elif ev == "talk" and e.get("_visibility") != "hidden":
            talks.append({"seat": int(e["actor"]), "text": str(e.get("text", "")), "round": int(e.get("round") or nres + 1)})
    rows.sort(key=lambda e: e["round"])
    if not rows:
        raise SystemExit("no round_result events in the log")
    run = b.get("run") or {}
    G = {"rounds": int(par("rounds", len(rows))), "move": float(par("move", 2)), "fee": float(par("fee", 0.5)),
         "impact": float(par("impact", 0.125)), "mode": str(par("signal_mode", "exact")), "talk": int(par("talk_passes", 0) or 0),
         "start_raw": rules.get("start_price", cfgv.get("price", 100)), "rows": rows, "talks": talks, "rules": rules,
         "end": next((e for e in events if e.get("event") == "game_end"), {}), "summary": b.get("summary") or {}, "run": run}
    G["start"] = float(G["start_raw"])
    prompt = ((run.get("prompt") or {}).get("text")) or ""
    m1 = re.search(r"stays the same with probability ([0-9.]+)", prompt)
    m2 = re.search(r"rises by `move` with probability ([0-9.]+)", prompt)
    G["stay"] = float(m1.group(1)) if m1 else STAY
    G["with"] = float(m2.group(1)) if m2 else WITH
    G["probs_from_prompt"] = bool(m1 and m2)
    return G


def verify(G, P, chg, price, imp, pay, score, rewards):
    """Check the rules' identities on the logged numbers (see the module docstring)."""
    n, fee, lam = len(chg) - 1, G["fee"], G["impact"]
    rows = G["rows"]
    if n != G["rounds"]:
        warn(f"{n} round_result events, rules say {G['rounds']} rounds")
    if [r["round"] for r in rows] != list(range(1, n + 1)):
        warn("round numbers of the round_result events are not 1..n")
    for k in range(1, n + 1):
        o = [P[s][k] - P[s][k - 1] for s in (0, 1)]
        if abs(price[k] - (price[k - 1] + chg[k])) > 1e-9:
            warn(f"R{k}: price {price[k]} != {price[k - 1]} + {chg[k]}")
        if abs(abs(chg[k]) - G["move"]) > 1e-9:
            warn(f"R{k}: price change {chg[k]} is not +-move {G['move']}")
        for s in (0, 1):
            ei = lam * o[s] * (o[0] + o[1])
            if abs(imp[s][k] - ei) > 1e-9:
                warn(f"R{k}: impact{s} {imp[s][k]} != impact x order x combined order = {ei}")
            ep = P[s][k] * chg[k] - fee * abs(o[s]) - ei
            if abs(pay[s][k] - ep) > 1e-9:
                warn(f"R{k}: payoff{s} {pay[s][k]} != position x change - fee x |order| - impact = {ep}")
            if abs(score[s][k] - (score[s][k - 1] + pay[s][k])) > 1e-9:
                warn(f"R{k}: score{s} {score[s][k]} != {score[s][k - 1]} + {pay[s][k]}")
        same = o[0] * o[1] > 0
        if "same_direction" in rows[k - 1] and bool(rows[k - 1]["same_direction"]) != same:
            warn(f"R{k}: same_direction {rows[k - 1]['same_direction']} but orders are {o}")
    end = G["end"]
    for s in (0, 1):
        if rewards[s] is not None and abs(rewards[s] - score[s][-1]) > 1e-9:
            warn(f"final score{s} {score[s][-1]} != summary reward {rewards[s]}")
        if end.get(f"total_pnl{s}") is not None and abs(float(end[f"total_pnl{s}"]) - score[s][-1]) > 1e-9:
            warn(f"final score{s} {score[s][-1]} != game_end total_pnl{s} {end[f'total_pnl{s}']}")
        if end.get(f"impact_cost{s}") is not None and abs(float(end[f"impact_cost{s}"]) - sum(imp[s])) > 1e-9:
            warn(f"impact sum{s} {sum(imp[s])} != game_end impact_cost{s} {end[f'impact_cost{s}']}")
    same_n = sum(1 for k in range(1, n + 1) if (P[0][k] - P[0][k - 1]) * (P[1][k] - P[1][k - 1]) > 0)
    if end.get("same_direction_rounds") is not None and int(end["same_direction_rounds"]) != same_n:
        warn(f"{same_n} same-direction rounds != game_end same_direction_rounds {end['same_direction_rounds']}")
    endv = G["summary"].get("variables") or {}
    if endv.get("price") is not None and abs(float(endv["price"]) - price[-1]) > 1e-9:
        warn(f"final price {price[-1]} != summary price {endv['price']}")
    if endv.get("recent_changes") is not None and [float(c) for c in endv["recent_changes"]] != chg[-3:][-len(endv["recent_changes"]):]:
        warn(f"last price changes {chg[-3:]} != summary recent_changes {endv['recent_changes']}")
    if G["talk"]:
        for k in range(1, n + 1):
            for s in (0, 1):
                c = sum(1 for t in G["talks"] if t["round"] == k and t["seat"] == s)
                if c != G["talk"]:
                    warn(f"R{k}: {c} messages from P{s}, talk_passes is {G['talk']}")


# ------------------------------------------------------------------------------------------------ joint optimum
def solve_joint(T, move, fee, lam, stay, with_p):
    """Plan maximizing the two traders' total expected P&L: state (round t, regime r = +1 bull / -1 bear, combined position N in
    -4..4); the cheapest way to change N by d costs fee x |d| + lam x d^2. Ties keep N, then prefer the smaller |N|, then the
    smaller N. Returns (policy {(t, r, N): N'}, value {(r, N): V_1}, action values {(t, r, N, N'): Q}), Q being the expected
    total from round t on when the pair moves from N to N' at the cheapest cost and follows the plan afterwards."""
    NS = range(-4, 5)
    V = {(r, N): 0.0 for r in (1, -1) for N in NS}
    pol, Q = {}, {}
    for t in range(T, 0, -1):
        nV = {}
        for r in (1, -1):
            drift = (2 * (with_p if r == 1 else 1 - with_p) - 1) * move
            for N in NS:
                best = None
                for a in sorted(NS, key=lambda a: (a != N, abs(a), a)):
                    d = a - N
                    val = a * drift - fee * abs(d) - lam * d * d + stay * V[(r, a)] + (1 - stay) * V[(-r, a)]
                    Q[(t, r, N, a)] = val
                    if best is None or val > best[0] + 1e-12:
                        best = (val, a)
                nV[(r, N)], pol[(t, r, N)] = best
        V = nV
    return pol, V, Q


def solve_pair_value(T, move, fee, lam, stay, with_p):
    """The same objective as a full two-trader program over (round, regime, position_0, position_1); start values only (check)."""
    V = {(r, a, b): 0.0 for r in (1, -1) for a in POSITIONS for b in POSITIONS}
    for t in range(T, 0, -1):
        nV = {}
        for r in (1, -1):
            drift = (2 * (with_p if r == 1 else 1 - with_p) - 1) * move
            for p0, p1 in itertools.product(POSITIONS, POSITIONS):
                nV[(r, p0, p1)] = max((a0 + a1) * drift - fee * (abs(a0 - p0) + abs(a1 - p1)) - lam * (a0 - p0 + a1 - p1) ** 2
                                      + stay * V[(r, a0, a1)] + (1 - stay) * V[(-r, a0, a1)]
                                      for a0, a1 in itertools.product(POSITIONS, POSITIONS))
        V = nV
    return V


# ------------------------------------------------------------------------------------------------ calls
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


def answer_position(a):
    if not isinstance(a, dict):
        return None
    if a.get("action_id") == "flat":
        return 0
    if a.get("action_id") == "set_position":
        p = (a.get("payload") or {}).get("position")
        try:
            p = float(p)
        except (TypeError, ValueError):
            return None
        return int(p) if p == int(p) and int(p) in POSITIONS else None
    return None


def call_meta(c):
    """Phase and round of a call, from its prompt ('Current phase: X', the public state's "round"); None when not shown."""
    p = c.get("prompt") or ""
    m = re.search(r"Current phase:\s*([A-Za-z_]+)", p)
    i = p.find("Public state:")
    rm = re.search(r'"round":\s*(\d+)', p[i:]) if i >= 0 else None
    return (m.group(1).upper() if m else None), (int(rm.group(1)) if rm else None)


def _matches(call, st):
    a = _norm_answer(call.get("answer"))
    if not isinstance(a, dict):
        return False
    if st["phase"] == "DECIDE":
        return answer_position(a) == st["position"]
    txt = next((v for v in (a.get("payload") or {}).values() if isinstance(v, str)), None)
    return a.get("action_id") == "talk" and txt is not None and txt.strip() == st["text"].strip()


def attach_calls(steps, calls):
    """Tie calls to steps (see the module docstring). Returns (steps with a call, steps whose call gave the logged move)."""
    if not calls:
        return 0, 0
    for c in calls:
        c["seat"] = _seat(c.get("seat"), c.get("prompt"))
    meta = [call_meta(c) for c in calls]
    keyed = all(c["seat"] is not None and m[0] is not None and m[1] is not None for c, m in zip(calls, meta))
    used = [False] * len(calls)
    tied = hits = ptr = 0
    for st in steps:
        if keyed:
            cand = [k for k in range(len(calls)) if not used[k] and calls[k]["seat"] == st["seat"] and meta[k][0] == st["phase"]
                    and meta[k][1] == st["round"]]
        else:
            cand, j = [], ptr
            while j < len(calls) and calls[j]["seat"] in (None, st["seat"]) and meta[j][0] in (None, st["phase"]) \
                    and meta[j][1] in (None, st["round"]):
                cand.append(j)
                j += 1
                if _matches(calls[j - 1], st):
                    break
        if not cand:
            continue
        hit = next((k for k in cand if _matches(calls[k], st)), None)
        use = hit if hit is not None else (cand[-1] if st["phase"] == "DECIDE" else cand[0])
        before = [k for k in cand if k < use]
        parts = [f"[Earlier attempt, answer: {calls[k]['answer'][:300]}]\n{calls[k].get('reasoning') or '(no reasoning returned)'}"
                 for k in before]
        c = calls[use]
        parts.append(("[Final attempt]\n" if before else "") + (c.get("reasoning") or ("(no reasoning returned)" if before else "")))
        if hit is None:
            parts.append(f"[The game applied {'position ' + apos(st['position']) if st['phase'] == 'DECIDE' else 'the logged message'} "
                         f"instead of this answer: {c['answer'][:300]}]")
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        if c.get("prompt"):
            st["state"] = en(common.state_part(c["prompt"]))
            st["state_from_prompt"] = True
        a = _norm_answer(c.get("answer"))
        if hit is not None and isinstance(a, dict):
            st["act"] = en(json.dumps(a, ensure_ascii=False))
            st["call_hit"] = True
        for k in before + [use]:
            used[k] = True
        ptr = max(ptr, use + 1)
        tied += 1
        hits += hit is not None
    if not all(used):
        warn(f"{used.count(False)} of {len(calls)} calls matched no decision")
    return tied, hits


# ------------------------------------------------------------------------------------------------ observations
def event_line(e):
    ev = e.get("event")
    if ev == "rules":
        return (f"rules: {e.get('rounds')} rounds, fee {e.get('fee')}, impact {e.get('impact')}, move {e.get('move')}, signal "
                f"{e.get('signal_mode')}, start price {e.get('start_price')}, {msgs(int(e.get('talk_passes') or 0)).replace('public, non-binding ', '')} per trader per round")
    if ev == "talk":
        return f"P{e.get('actor')} says: {e.get('text', '')}"
    if ev == "round_result":
        return (f"round {e['round']} result: signal {e.get('signal_seen')}; positions P0 {apos(e['position0'])}, P1 {apos(e['position1'])}"
                f" (same direction {str(bool(e.get('same_direction'))).lower()}); price change {apos(e['price_change'])} to {e['price']}; "
                f"impact costs {e['impact0']} / {e['impact1']}; payoffs {e['payoff0']} / {e['payoff1']}; totals {e['score0']} / {e['score1']}")
    return f"{ev}: " + json.dumps({k: v for k, v in e.items() if k != "event"}, ensure_ascii=False)


def observation(st, G, P, score, chg, price, sig, public, upto):
    """What the trader saw at this step, rebuilt from the log: the fields of the prompt (round, phase, signal, price, last three
    price changes, both positions and scores) and its list of the last 12 public events before the step."""
    k, s, o = st["round"], st["seat"], 1 - st["seat"]
    T, fee, lam = G["rounds"], G["fee"], G["impact"]
    recent = [c for c in chg[max(1, k - 3):k]]
    lines = ["[Rebuilt from the game log; the exact prompt for this turn was not saved.]",
             f"You are seat P{s} ({SEATS[s]}). Round {k} of {T}. " +
             (f"Current phase: TALK. Each trader sends {msgs(G['talk'])} before the decision."
              if st["phase"] == "TALK" else
              "Current phase: DECIDE. Choose your target position from -2 to +2 (flat = 0); both positions are revealed together."),
             f"Regime signal: {sig[k]}" + (" (signal mode exact: the signal is the true regime)" if G["mode"] == "exact" else
                                           f" (signal mode {G['mode']})"),
             f"Price: {nstr(price[k - 1])}. Last price changes: " + (", ".join(apos(int(c)) if float(c).is_integer() else str(c) for c in recent)
                                                                     if recent else "none yet") + ".",
             f"Positions: you (P{s}) {apos(P[s][k - 1])}, P{o} {apos(P[o][k - 1])}. "
             f"P&L so far: you {nstr(score[s][k - 1])}, P{o} {nstr(score[o][k - 1])}.",
             f"Costs: fee {fee:g} per unit of position changed; impact {lam:g} x (your change) x (your change + the other trader's change)."]
    if st["phase"] == "DECIDE" and s == 1:
        lines.append("P0's position for this round is not shown until both have chosen.")
    seen = [e for e in public[:upto]][-RECENT:]
    lines += ["", f"Recent public events (the prompt lists the last {RECENT}):"] + [f"- {event_line(e)}" for e in seen]
    return en("\n".join(lines))


# ------------------------------------------------------------------------------------------------ build
def build(bundle_dir, calls=None):
    b = load_source(bundle_dir)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    if not calls:
        calls = b.get("md_calls") or []
    G = read_game(b)
    rows, talks, fee, lam, move, T = G["rows"], G["talks"], G["fee"], G["impact"], G["move"], G["rounds"]
    n = len(rows)
    exact = G["mode"] == "exact"
    reg_word = "regime" if exact else "signal"

    # ---- per round, straight from the round_result events (index 0 = the start)
    P = [[0] + [int(e[f"position{s}"]) for e in rows] for s in (0, 1)]
    imp = [[0.0] + [float(e[f"impact{s}"]) for e in rows] for s in (0, 1)]
    pay = [[0.0] + [float(e[f"payoff{s}"]) for e in rows] for s in (0, 1)]
    score = [[0.0] + [float(e[f"score{s}"]) for e in rows] for s in (0, 1)]
    chg = [0.0] + [float(e["price_change"]) for e in rows]
    price = [G["start"]] + [float(e["price"]) for e in rows]
    sig = [None] + [str(e.get("signal_seen", "")) for e in rows]
    regime = [0] + [1 if x == "bull" else -1 if x == "bear" else 0 for x in sig[1:]]
    raw_rw = G["summary"].get("rewards") or (G["run"].get("result") or {}).get("rewards") or {}
    rewards = [float(raw_rw[str(s)]) if str(s) in raw_rw else None for s in (0, 1)]
    verify(G, P, chg, price, imp, pay, score, rewards)

    order = [[0] + [P[s][k] - P[s][k - 1] for k in range(1, n + 1)] for s in (0, 1)]
    comb = [order[0][k] + order[1][k] for k in range(n + 1)]
    net = [P[0][k] + P[1][k] for k in range(n + 1)]
    same = [k for k in range(1, n + 1) if order[0][k] * order[1][k] > 0]
    opposite = [k for k in range(1, n + 1) if order[0][k] * order[1][k] < 0]
    fees = [fee * sum(abs(order[s][k]) for k in range(1, n + 1)) for s in (0, 1)]
    gross = [sum(P[s][k] * chg[k] for k in range(1, n + 1)) for s in (0, 1)]
    impact_tot = [sum(imp[s]) for s in (0, 1)]
    against = [k for k in range(1, n + 1) if regime[k] and chg[k] * regime[k] < 0]
    both = [score[0][k] + score[1][k] for k in range(n + 1)]

    # ---- joint optimum (exact signal only)
    opt = None
    if exact and all(regime[1:]):
        pol, V, Q = solve_joint(T, move, fee, lam, G["stay"], G["with"])
        start_val = (V[(1, 0)] + V[(-1, 0)]) / 2
        Vp = solve_pair_value(T, move, fee, lam, G["stay"], G["with"])
        for r in (1, -1):
            if abs(Vp[(r, 0, 0)] - V[(r, 0)]) > 1e-9:
                warn(f"joint optimum: combined-position value {V[(r, 0)]} != two-trader value {Vp[(r, 0, 0)]}")
        N, C, pays = [0], [0.0], [0.0]
        for k in range(1, n + 1):
            a = pol[(k, regime[k], N[-1])]
            d = a - N[-1]
            pays.append(a * chg[k] - fee * abs(d) - lam * d * d)
            C.append(C[-1] + pays[-1])
            N.append(a)
        opt = {"N": N, "C": C, "pay": pays, "start": start_val, "first": V[(regime[1], 0)]}
        print(f"joint optimum: expected total for the pair {start_val:.6f} (first regime known: {opt['first']:.6f}); "
              f"on this path {C[-1]:g}", file=sys.stderr)
        # each round judged from the combined position the traders really held: the optimum's best target from there, and the
        # expected cost for the pair of what they did instead (Q of the best target minus Q of theirs; opposite orders also pay
        # fee x (|order_0| + |order_1| - |combined order|) more than the cheapest way to the same combined position)
        best, cost, waste = [None], [0.0], [0.0]
        for k in range(1, n + 1):
            st0 = (k, regime[k], net[k - 1])
            best.append(pol[st0])
            waste.append(fee * (abs(order[0][k]) + abs(order[1][k]) - abs(comb[k])))
            cost.append(max(0.0, Q[st0 + (pol[st0],)] - (Q[st0 + (net[k],)] - waste[k])))
        opt.update(best=best, cost=cost, waste=waste,
                   good=[k for k in range(1, n + 1) if cost[k] < 1e-9], bad=[k for k in range(1, n + 1) if cost[k] >= 1e-9])
        # rounds where the replayed optimum path differs although the traders' move was the best one from their own position
        opt["drift"] = [k for k in opt["good"] if N[k] != net[k]]
    # the pair's own round cost can exceed fee x |d| + lam x d^2 only when the two orders have opposite signs
    pair_pay = [0.0] + [pay[0][k] + pay[1][k] for k in range(1, n + 1)]

    # ---- steps: per round the messages (log order), then P0's and P1's position decisions (rules: Round Procedure)
    events = b["events"]
    public = [e for e in events if e.get("_visibility") not in ("hidden", "private") and e.get("event") not in ("setup", "game_over", "game_end")]
    rr_idx = [i for i, e in enumerate(public) if e.get("event") == "round_result"]
    talk_idx = [i for i, e in enumerate(public) if e.get("event") == "talk"]
    steps, feed = [], []
    ti = 0
    for k in range(1, n + 1):
        rw = (sig[k] if exact else f"{sig[k]} signal") if sig[k] else "no signal"
        for t in [t for t in talks if t["round"] == k]:
            upto = talk_idx[ti]
            ti += 1
            steps.append({"seat": t["seat"], "x": k, "kind": "talk", "phase": "TALK", "round": k, "text": t["text"], "upto": upto,
                          "head": en(f"Round {k} {DOT} {rw} {DOT} " + (talk_head(t["text"]) if t["text"].strip() else "empty message")),
                          "act": en(json.dumps({"action_id": "talk", "payload": {"text": t["text"]}}, ensure_ascii=False))})
            feed.append({"x": k, "step": len(steps) - 1, "who": SEATS[t["seat"]], "kind": "talk", "text": en(t["text"])})
        for s in (0, 1):
            p, prev, o = P[s][k], P[s][k - 1], 1 - s
            head = f"Round {k} {DOT} {rw} {DOT} {act_word(prev, p)}"
            if order[s][k]:
                head += (f", same way as {SEATS[o]}" if k in same else
                         f", against {SEATS[o]}'s {'sale' if order[o][k] < 0 else 'purchase'}" if k in opposite else
                         f" while {SEATS[o]} holds") + f" {DOT} impact {nstr(imp[s][k])}"
            steps.append({"seat": s, "x": k, "kind": "buy" if p > prev else "sell" if p < prev else "hold", "phase": "DECIDE",
                          "round": k, "position": p, "upto": rr_idx[k - 1], "head": head,
                          "act": json.dumps({"action_id": "set_position", "payload": {"position": p}})})
        c = comb[k]
        push = lam * c
        moves = ", ".join(f"{SEATS[s]} " + (f"holds {pos(P[s][k])}" if not order[s][k] else f"{pos(P[s][k - 1])} {ARROW} {pos(P[s][k])}")
                          for s in (0, 1))
        txt = (f"Signal {sig[k]}. {moves}"
               + (f"; combined order {pos(c)} pushed the trade price {'up' if c > 0 else 'down'} {nstr(abs(push))}" if c else
                  "; the orders cancel, no price push" if order[0][k] or order[1][k] else "; nobody traded")
               + f". Price {sgn(chg[k])} to {nstr(price[k])}" + (f", against the {reg_word}" if k in against else "")
               + f". Payoffs {sgn(pay[0][k])} / {sgn(pay[1][k])}"
               + (f" (impact {nstr(imp[0][k])} / {nstr(imp[1][k])})" if imp[0][k] or imp[1][k] else "")
               + f"; totals {sgn(score[0][k])} / {sgn(score[1][k])}.")
        feed.append({"x": k, "step": len(steps) - 1, "who": "Result", "text": txt})
    if ti != len(talk_idx):
        warn(f"{len(talk_idx) - ti} talk events after the last round were not shown")

    first = G["run"].get("prompt") or {}
    fp_text, fp_phase = first.get("text") or "", str(first.get("phase", "")).upper()
    fp_round = call_meta({"prompt": fp_text})[1]
    fp_used = False
    for st in steps:
        st["state"] = observation(st, G, P, score, chg, price, sig, public, st["upto"])
        if (not fp_used and fp_text and _seat(first.get("seat")) == st["seat"] and fp_phase == st["phase"] and fp_round == st["round"]):
            st["state"] = en(common.state_part(fp_text))             # the saved prompt of the game's first decision
            st["state_from_prompt"] = fp_used = True
    tied, hits = attach_calls(steps, calls)
    n_dec = sum(1 for st in steps if st["phase"] == "DECIDE")
    dec_hits = sum(1 for st in steps if st["phase"] == "DECIDE" and st.get("call_hit"))
    from_prompt = sum(1 for s in steps if s.get("state_from_prompt"))
    for st in steps:
        st.setdefault("thinking", "")
        for key in ("phase", "round", "text", "position", "upto", "state_from_prompt", "call_hit"):
            st.pop(key, None)

    # ---- chart 1: price and the regime
    xn = n + 1
    labels = ["Start"] + [f"R{k}" for k in range(1, n + 1)]
    price_lines = [{"name": "Price", "color": "ink", "v": [r6(v) for v in price], "dots": True}]
    if against:
        price_lines.append({"name": f"Moved against the {reg_word}", "color": "c4", "v": [price[k] if k in against else None for k in range(xn)],
                            "dots": True, "noline": True})
    bull_n, bear_n = sum(1 for k in range(1, xn) if regime[k] > 0), sum(1 for k in range(1, xn) if regime[k] < 0)
    bears = [k for k in range(1, xn) if regime[k] < 0]
    split = (f"bull in every round" if not bears and bull_n == n else f"bear in every round" if bear_n == n else
             f"bull in {bull_n} rounds, bear in {rounds_text(bears)}")
    c_price = {"title": "Price and the regime" if exact else "Price and the regime signal",
               "sub": (f"Price after each round's move ({nstr(price[0])} at the start, {nstr(price[-1])} at the end). The strip shows "
                       + ("the regime of each round, which both traders are told before deciding (exact signal): " if exact else
                          "the signal both traders saw before each round: ")
                       + split + ". "
                       + (f"Yellow: {rounds_text(against)}, where the price moved against the {reg_word} (probability "
                          f"{1 - G['with']:.1f} each round)." if against else f"The price never moved against the {reg_word}.")),
               "height": 220, "fmt": "int", "lines": price_lines,
               "band": {"v": regime, "pos": "Bull regime" if exact else "Bull signal", "neg": "Bear regime" if exact else "Bear signal"}}

    # ---- chart 2: positions grid
    def cell(p, full, tip):
        a = abs(p)
        o = 0.85 if a == 0 else min(0.92, 0.38 + 0.54 * a / full)
        return {"c": "c1" if p > 0 else "c2" if p < 0 else "muted", "t": pos(p), "tip": tip, "o": round(o, 2)}

    g_rows, g_cells = [SEATS[0], SEATS[1], "Combined"], []
    for s in (0, 1):
        r = [None]
        for k in range(1, xn):
            tip = act_word(P[s][k - 1], P[s][k])
            if order[s][k]:
                tip += f"; fee {nstr(fee * abs(order[s][k]))}, impact {nstr(imp[s][k])}"
            r.append(cell(P[s][k], 2, tip + f"; payoff {sgn(pay[s][k])}"))
        g_cells.append(r)
    def after(k):
        return "the start" if k == 0 else f"R{k}"

    def miss(k, cost=True):
        """What the pair did in a round that was not the joint optimum's best move from their own position."""
        txt = (f"went to {pos(net[k])} combined where {pos(opt['best'][k])} was best" if opt["best"][k] != net[k] else
               f"went to {pos(net[k])} combined, as the optimum, but opposite orders paid {nstr(opt['waste'][k])} more in fees")
        return txt + (f" (expected cost {cstr(opt['cost'][k])} for the pair)" if cost else "")

    def miss_short(k):
        """The same for the tile: 'R5 (+4, best +3)'."""
        return (f"R{k} ({pos(net[k])}, best {pos(opt['best'][k])})" if opt["best"][k] != net[k] else
                f"R{k} ({pos(net[k])}, best too, but opposite orders paid {nstr(opt['waste'][k])} more in fees)")

    def drift(k):
        """Why the replayed plan differs in a round where the pair's move was the best one from their own position."""
        if opt["N"][k - 1] == net[k - 1]:
            return f"in R{k}, where the traders' {pos(net[k])} is as good as the plan's {pos(opt['N'][k])}"
        return (f"in R{k} only because it was at {pos(opt['N'][k - 1])} after {after(k - 1)}: from the traders' {pos(net[k - 1])} "
                f"the best move was {pos(opt['best'][k])}" + (", which they made" if opt["best"][k] == net[k] else
                                                             f", and their {pos(net[k])} is as good"))

    r = [None]
    for k in range(1, xn):
        c = comb[k]
        how = ("both traded the same way" if k in same else "orders in opposite directions" if k in opposite else
               f"{SEATS[0] if order[0][k] else SEATS[1]} traded alone" if c else "no trade")
        tip = (f"{pos(net[k])}; combined order {pos(c)} ({how}) pushed the trade price {'up' if c > 0 else 'down'} {nstr(abs(lam * c))}"
               if c else f"{pos(net[k])}; " + ("the two orders cancel, no price push" if order[0][k] else "nobody traded"))
        if opt:
            b_ = opt["best"][k]
            tip += ((f"; the joint optimum's best move from {pos(net[k - 1])} is the same" if b_ == net[k] else
                     f"; as good as the joint optimum's best move from {pos(net[k - 1])}, {pos(b_)}") if k in opt["good"] else
                    (f"; the joint optimum's best move from {pos(net[k - 1])} was {pos(b_)}" if b_ != net[k] else
                     f"; the joint optimum's best move from {pos(net[k - 1])} too, but opposite orders paid {nstr(opt['waste'][k])} more in fees")
                    + f" (expected cost {cstr(opt['cost'][k])} for the pair)")
        r.append(cell(net[k], 4, tip))
    g_cells.append(r)
    if opt:
        g_rows.append("Optimum")
        r = [None]
        for k in range(1, xn):
            tip = f"{pos(opt['N'][k])}; " + ("same as the traders" if opt["N"][k] == net[k] else f"the traders were at {pos(net[k])}")
            if k in opt["drift"] and opt["N"][k - 1] != net[k - 1]:
                tip += (f"; the plan started this round at {pos(opt['N'][k - 1])}, the traders at {pos(net[k - 1])}; from the traders' "
                        f"position the best move was {pos(opt['best'][k])}" + (", as they chose" if opt["best"][k] == net[k] else ""))
            r.append(cell(opt["N"][k], 4, tip))
        g_cells.append(r)
    used_c = {c["c"] for row in g_cells for c in row if c}
    legend = [x for x in (["Long", "c1"], ["Short", "c2"], ["Flat", "muted"]) if x[1] in used_c]
    sub2 = ("Each trader's position after its secret choice, and their sum (Combined). Color shows direction, not the trader; shade "
            "shows size, full at 2 units in a trader's row and at 4 in the Combined" + (" and Optimum rows" if opt else " row") + ". ")
    if opt:
        bad = opt["bad"]
        sub2 += ("Optimum is the combined position of the plan that maximizes the two traders' total expected P&L, replayed on the "
                 f"same regimes. Judged from the position the pair actually held, their combined move was the plan's best in "
                 f"{len(opt['good'])} of {n} rounds" + (", all but " + and_join([f"R{k} ({miss(k, False)})" for k in bad[:3]]
                                                                                 + ([f"{len(bad) - 3} more"] if len(bad) > 3 else []))
                                                        if bad else ", every round") + ". ")
        if opt["drift"]:
            sub2 += "The Optimum row also differs " + ", and ".join([drift(k) for k in opt["drift"][:2]]) + \
                    (f", and in {len(opt['drift']) - 2} more rounds" if len(opt["drift"]) > 2 else "") + ". "
    c_grid = {"title": "Positions, round by round",
              "sub": sub2 + "Hover a column for orders, fees, impact and payoffs.",
              "height": 150,
              "grid": {"rows": g_rows, "cells": g_cells, "rowHeight": 28, "legend": legend}}

    # ---- chart 3: impact cost each round
    # Trader s pays lam x o_s x (o_0 + o_1) = lam x o_s^2 (its cost alone) + lam x o_0 x o_1, so against trading alone each
    # trader pays the same lam x o_0 x o_1 more: positive when the orders go the same way, negative (a saving) when opposite.
    def cross_of(ks):
        return lam * sum(order[0][k] * order[1][k] for k in ks)

    alone_s, paid_s = ([sum(lam * order[s][k] ** 2 for k in same) for s in (0, 1)], [sum(imp[s][k] for k in same) for s in (0, 1)])
    alone_o, paid_o = ([sum(lam * order[s][k] ** 2 for k in opposite) for s in (0, 1)], [sum(imp[s][k] for k in opposite) for s in (0, 1)])
    for s in (0, 1):
        if abs(paid_s[s] - alone_s[s] - cross_of(same)) > 1e-9:
            warn(f"same-way rounds: {SEATS[s]} paid {paid_s[s]}, alone {alone_s[s]}, difference != impact x sum of order products {cross_of(same)}")
        if abs(paid_o[s] - alone_o[s] - cross_of(opposite)) > 1e-9:
            warn(f"opposite rounds: {SEATS[s]} paid {paid_o[s]}, alone {alone_o[s]}, difference != impact x sum of order products {cross_of(opposite)}")
    single = [k for k in range(1, xn) if (order[0][k] == 0) != (order[1][k] == 0)]
    sub3 = (f"Each trader pays its own order (change of position) times the move of the trade price, {lam:g} per unit of the two "
            f"orders combined: a trader alone pays {lam:g} {TIMES} order squared, and the pair together {lam:g} {TIMES} combined order "
            "squared, however it is split. ")
    def paid_vs_alone(paid, alone, word):
        if abs(paid[0] - paid[1]) < 1e-9 and abs(alone[0] - alone[1]) < 1e-9:
            return f"each trader paid {nstr(paid[0])} there, against {nstr(alone[0])} {word}"
        return f"{SEATS[0]} paid {nstr(paid[0])} and {SEATS[1]} {nstr(paid[1])} there, against {nstr(alone[0])} and {nstr(alone[1])} {word}"

    if same:
        sub3 += (f"In {rounds_text(same)} both traded the same way: {paid_vs_alone(paid_s, alone_s, 'had each traded alone')}, so "
                 f"{nstr(cross_of(same))} more each. ")
        if opt:
            as_opt = [k for k in same if opt["best"][k] == net[k]]
            if as_opt:
                sub3 += ("Spreading " + ("those trades" if len(as_opt) == len(same) else f"the trades of {rounds_text(as_opt)}")
                         + " over more rounds would not have helped the pair: from the same positions the joint optimum makes the same "
                         f"combined trade{'s' if len(as_opt) > 1 else ''} ({', '.join(pos(comb[k]) for k in as_opt)}). ")
    if opposite:
        sub3 += (f"In {rounds_text(opposite)} the orders went opposite ways, which saved each trader {nstr(-cross_of(opposite))} "
                 f"against trading alone: {paid_vs_alone(paid_o, alone_o, 'alone')}. ")
    neg = [(s, k) for k in range(1, xn) for s in (0, 1) if imp[s][k] < -1e-12]
    if neg:
        sub3 += ("A negative cost (" + "; ".join(f"{SEATS[s]} in {rounds_text([k for t, k in neg if t == s])}" for s in (0, 1)
                                                 if any(t == s for t, _ in neg)) + ") means that trader earned from "
                 "trading against a larger opposite order. ")
    if single:
        sub3 += f"Only one trader traded in {rounds_text(single)}. "
    sub3 += (f"Totals: {SEATS[0]} {nstr(impact_tot[0])}, {SEATS[1]} {nstr(impact_tot[1])}; fees, not shown here, were "
             + ("larger: " if all(fees[s] > impact_tot[s] for s in (0, 1)) else "") + f"{nstr(fees[0])} and {nstr(fees[1])}.")
    c_imp = {"title": "Impact cost each round", "sub": sub3, "height": 200,
             "bars": [{"name": SEATS[s], "color": SEAT_COLORS[s],
                       "v": [None] + [r6(imp[s][k]) if order[s][k] else None for k in range(1, xn)]} for s in (0, 1)],
             "lines": []}
    c_imp["fmt"] = chart_fmt(c_imp["bars"])
    if neg:
        c_imp["zero"] = True
    if not any(v is not None for bb in c_imp["bars"] for v in bb["v"]):
        c_imp = None

    # ---- chart 4: cumulative P&L, each trader, the pair and the joint optimum
    cum = [{"name": SEATS[s], "color": SEAT_COLORS[s], "v": [r6(v) for v in score[s]], "end": True} for s in (0, 1)]
    cum.append({"name": "Both together", "color": "c3", "v": [r6(v) for v in both], "area": True, "end": True})
    sub4 = (f"Running totals; each trader's own total is its reward. {SEATS[0]}: {sgn(gross[0])} from price moves, fees {nstr(fees[0])}, "
            f"impact {nstr(impact_tot[0])}; {SEATS[1]}: {sgn(gross[1])}, {nstr(fees[1])}, {nstr(impact_tot[1])}. ")
    if opt:
        cum.append({"name": "Joint optimum", "color": "c3", "dash": True, "v": [r6(v) for v in opt["C"]], "end": True})
        dk = [k for k in range(1, xn) if abs(pair_pay[k] - opt["pay"][k]) > 1e-9]
        diffs = [f"R{k} (combined {pos(net[k - 1])} {ARROW} {pos(net[k])}, the plan {pos(opt['N'][k - 1])} {ARROW} {pos(opt['N'][k])}: "
                 f"{sgn(pair_pay[k] - opt['pay'][k])} for the traders)" for k in dk]
        sub4 += (f"Dashed: the joint optimum replayed on the same regimes and price moves, which do not depend on the traders; it "
                 f"expects {MINUS if opt['first'] < 0 else '+'}{abs(opt['first']):.2f} for the pair before the game and makes {sgn(opt['C'][-1])} on this path, against "
                 f"{sgn(both[-1])} for the traders" + (f"; the dashed line matches the pair's line through R{dk[0] - 1}. " if dk and dk[0] > 1 else
                                                       "; the dashed line matches the pair's line throughout. " if not dk else ". "))
        run_ = bool(dk) and dk == list(range(dk[0], dk[-1] + 1))
        sub4 += ("The pair's round payoffs differ from the plan's only in " + and_join(diffs) + ". " if 0 < len(diffs) <= 3 else
                 f"The pair's round payoffs differ from the plan's in {rounds_text(dk)}. " if 0 < len(dk) <= 5 else
                 f"The pair's round payoffs differ from the plan's in {len(dk)} of {n} rounds"
                 + ((f", every round from R{dk[0]} on" if dk[-1] == n else f", R{dk[0]} to R{dk[-1]}") if run_ else "") + ". " if dk else
                 "The pair's round payoffs equal the plan's in every round. ")
        if opt["bad"]:
            sub4 += (f"Judged from the position the pair held, {rounds_text(opt['bad'])} {'was a worse move' if len(opt['bad']) == 1 else 'were worse moves'} "
                     f"(expected cost {csum(opt['cost'][k] for k in opt['bad'])} for the pair)")
            rest = [k for k in dk if k in opt["drift"]]
            sub4 += ("; the plan also differs " + ", and ".join([drift(k) for k in rest[:2]]) if rest else "") + "."
    c_cum = {"title": "Cumulative P&L", "sub": sub4, "height": 250, "fmt": chart_fmt(cum), "zero": True, "lines": cum}

    charts = [c_price, c_grid] + ([c_imp] if c_imp else []) + [c_cum]

    # ---- tiles and text
    model_name = ((b.get("meta") or {}).get("model") or G["run"].get("model") or "").split("/")[-1] or "the model"
    lead = 0 if score[0][-1] > score[1][-1] else 1 if score[1][-1] > score[0][-1] else None
    tiles = [{"k": "Total P&L", "v": f"{sgn(score[0][-1])} / {sgn(score[1][-1])}",
              "s": f"{SEATS[0]} / {SEATS[1]}, each its own reward; together {sgn(both[-1])} (the line)"
                   + (f", {SEATS[lead]} ahead" if lead is not None else ", a tie"),
              "spark": [r6(v) for v in both]},
             {"k": "Impact paid", "v": f"{nstr(impact_tot[0])} / {nstr(impact_tot[1])}",
              "s": f"{SEATS[0]} / {SEATS[1]}; " + "; ".join(
                  ([f"{'of this, ' if not opposite else ''}{nstr(cross_of(same))} each was the extra cost of trading the same way in "
                    f"the same round ({rounds_text(same)})"] if same else [])
                  + ([f"opposite orders saved each {nstr(-cross_of(opposite))} ({rounds_text(opposite)})"] if opposite else [])
                  or ["never both in the same round"])}]
    if opt:
        bad = opt["bad"]
        tiles.append({"k": "Best joint move", "v": f"{len(opt['good'])} of {n} rounds",
                      "s": ("Missed, from the pair's own combined position: " + and_join([miss_short(k) for k in bad[:4]]
                                                                                       + ([f"{len(bad) - 4} more"] if len(bad) > 4 else []))
                            + f"; expected cost {csum(opt['cost'][k] for k in bad)} for the pair" if bad else
                            "The pair's combined move was the joint optimum's best from the position they held, every round")})
    tiles.append({"k": "Price", "v": f"{nstr(price[0])} {ARROW} {nstr(price[-1])}",
                  "s": (f"Bear {reg_word} in {rounds_text(bears)}" if bears and bear_n < n else
                        f"{'Bear' if bears else 'Bull'} {reg_word} in every round") + "; "
                       + (f"moved against the {reg_word} in {rounds_text(against)}" if against else f"never against the {reg_word}")})

    title = (f"Two traders share a market and its price impact, and talk before each of {n} rounds" if G["talk"] else
             f"Two traders share a market and its price impact over {n} rounds")
    lede = (f"Two traders share a market that switches between a bull and a bear regime; "
            + ("both are told the regime before each round. " if exact else "both see the same signal, right 70% of the time. ")
            + f"Each round both secretly pick a position from {MINUS}2 to +2 and earn position {TIMES} price move ({PM}{move:g}, with the "
            f"regime {G['with']:.0%} of the time), minus a fee of {fee:g} per unit changed and a market-impact cost: the trade price "
            f"moves {lam:g} per unit of their combined order and each pays that move on its own change, so trading the same way in "
            "the same round costs both more. "
            + (f"They send {msgs(G['talk'])} each before every decision, and each" if G["talk"] else "Each")
            + f" scores its own total P&L over {T} rounds.")
    seed = G["run"].get("seed")
    date = DEFAULT_DATE if isinstance(bundle_dir, dict) else recorded_date(bundle_dir)
    foot = (f"Source: the {date} recording of {GID} (seed {seed}, {T} rounds, signal {G['mode']}, move {move:g}, fee {fee:g}, impact "
            f"{lam:g}, {msgs(G['talk']).replace('public, non-binding ', '')} per trader per round, {model_name}), "
            f"{'the rendered replay of its game log' if b.get('md') else 'game_log.jsonl'}. Each round's round_result event gives the "
            "signal, both positions, the price change and price, each trader's impact cost, payoff and running score, and they add up "
            f"exactly: payoff = position {TIMES} change {MINUS} {fee:g} {TIMES} |order| {MINUS} impact cost, impact cost = {lam:g} {TIMES} own order {TIMES} "
            f"combined order, so {SEATS[0]} ends at {sgn(gross[0])} {MINUS} {nstr(fees[0])} {MINUS} {nstr(impact_tot[0])} = "
            f"{sgn(score[0][-1])} and {SEATS[1]} at {sgn(gross[1])} {MINUS} {nstr(fees[1])} {MINUS} {nstr(impact_tot[1])} = "
            f"{sgn(score[1][-1])}, the recorded rewards. {SEATS[0]} is seat P0, {SEATS[1]} seat P1. The log has no event per position "
            "decision, so each is placed after the round's messages, P0 first (the two choices are simultaneous), and its action is "
            + (f"the model's answer from the call log where that answer is the position the game applied ({dec_hits} of {n_dec}), "
               "otherwise set_position with the applied position. " if dec_hits else "rebuilt as set_position with the position the game applied. ")
            + ("The observations are the prompts from the call log. " if from_prompt == len(steps) else
               "The observations are rebuilt from earlier events (the first message shows the recorded prompt). " if from_prompt == 1 and not tied else
               f"{from_prompt} of {len(steps)} observations are recorded prompts, the rest rebuilt from earlier events. " if from_prompt else
               "The observations are rebuilt from earlier events. "))
    if opt:
        foot += (f"Joint optimum: backward induction over (round, regime, combined position) for the largest total expected P&L of the "
                 f"pair, risk-neutral, with regime stay probability {G['stay']:g} and move-with-regime probability {G['with']:g}"
                 + (" (read from the recorded prompt)" if G["probs_from_prompt"] else "") + "; computed here, as the rules store no "
                 "reference solution, and checked against a full two-trader program. It is a cooperative benchmark: each trader is in fact "
                 "paid only its own P&L. ")
    foot += (f"Reasoning: from the model-call log, tied to {tied} of {len(steps)} decisions." if calls else
             "This recording was made before reasoning was saved; the game will be re-recorded with reasoning.")

    kinds = {k: v for k, v in KINDS.items() if any(s["kind"] == k for s in steps)}
    return {"id": GID, "name": "Trading with impact", "title": title, "lede": lede, "seatNames": SEATS, "tiles": tiles, "foot": en(foot),
            "x": {"n": xn, "labels": labels}, "charts": charts, "kinds": kinds, "steps": steps, "feed": feed,
            "feedTitle": "Messages and results, round by round" if G["talk"] else "Results, round by round",
            "feedSub": ("Each round: the public messages (non-binding), then the result both traders saw once the two positions were "
                        "revealed together. The selected step is highlighted." if G["talk"] else
                        "One line per round_result event. The selected step is highlighted.")}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    common.write(build(src, calls), out, date=recorded_date(src))
