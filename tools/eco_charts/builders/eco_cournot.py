"""Replay data for eco_cournot (repeated Cournot quantity competition; Cournot 1838, Tirole 1988; experiments: Huck, Normann and
Oechssler 2004, "Two are few and four are many").

usage: python3 -I eco_cournot.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md, run.log) or a
          file inside one.
  OUT_DIR where <OUT_DIR>/eco_cournot.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). A call belongs to the
          decision of the same seat, phase and round (read from its prompt: "You are seat Pk", "Current phase: X", the public
          state's "round"; a value the prompt does not show agrees with anything), taken in time order. A decision's calls run up
          to the first reply the recorder accepts (read_answer mirrors its check: exactly {"action_id": "produce", "payload":
          {"produce": {"quantity": <int 0..q_max>}}}, or {"action_id": "talk", "payload": {"text": ...}}); a prompt without the
          retry note starts the next decision. When no reply is usable the game plays its default (quantity 30, empty message):
          such a step is marked as the game's default, not the model's choice, and shows every try with its reasoning.
          Without a call log, a run-log fallback warning (printed once per run) with no empty message means at least one logged
          30 is a default; the page says so but cannot mark which.

Each round has talk_passes public messages per firm (TALK phase, logged as talk events in time order) and then one sealed
quantity per firm (DECIDE phase, seats asked in order P0, P1, ...; no firm sees another's quantity before the reveal). The log has
no event per quantity: all quantities are in the round's round_result event (quantities in seat order, total_quantity, price).
Profits are not logged per round; they are recomputed exactly with the rule of the game, P = max(0, a - total quantity) and
profit (P - c) * own quantity, and checked against the logged total and price, the summary's rewards and the hidden game_end
totals (sum_total_quantity, sum_total_profit); any disagreement is printed as a warning. Benchmarks (Cournot-Nash, joint-profit
maximum, competitive outcome, and for two firms the pure equilibria and the best collusive splits on the whole-number grid) are
computed exactly (fractions) from a, c, q_max and the number of firms."""
import json, os, re, sys, unicodedata
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_cournot"
NAME = "Cournot competition"
DEFAULT_DATE = "2026-11-09"
MINUS, TIMES, DOT, LDQ, RDQ = "−", "×", "·", "“", "”"
FIRM_COLORS = ["ink", "c2", "c4", "muted"]         # firms in line and bar charts: not c1/c3 (benchmark lines), not hold
                                                   # (the game's default and the price line)
CAT_COLOR = {"collude": "c3", "restrain": "c4", "nash": "c1", "expand": "c2"}
CATS = ("collude", "restrain", "nash", "expand")
DEFAULT_Q = 30      # eco_cournot.yaml: DECIDE default_action and the produce struct default are {"quantity": 30}
NUMW = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


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


# ------------------------------------------------------------------------------------------------ text helpers
def g(v):
    """A quantity, price, profit or score: an integer with thousands separators, anything else with up to two decimals."""
    v = Fraction(v)
    s = f"{abs(int(v)):,}" if v.denominator == 1 else f"{abs(float(v)):,.2f}".rstrip("0").rstrip(".")
    return (MINUS if v < 0 else "") + s


def sg(v):
    """A round profit with its sign; 'profit 0' for zero."""
    v = Fraction(v)
    return "profit 0" if v == 0 else ("+" if v > 0 else "") + g(v)


def fv(v):
    """A chart value: int when whole, float otherwise."""
    v = Fraction(v)
    return int(v) if v.denominator == 1 else float(v)


def fmt_for(*series):
    """'int' when every value is whole, else 'price' (two decimals) so that e.g. 22.5 is not shown rounded."""
    return "int" if all(Fraction(x).denominator == 1 for s in series for x in s if x is not None) else "price"


def share_pct(num, den):
    """num/den as a percentage with as many decimals (up to 4) as it takes not to round a value below 100% up to 100%."""
    if not den:
        return "n/a"
    x = Fraction(num) / Fraction(den) * 100
    for d in range(1, 5):
        s = f"{float(x):.{d}f}"
        if x == 100 or float(s) != 100.0:
            return s.rstrip("0").rstrip(".") + "%" if d > 1 else s + "%"
    return f"{float(x):.4f}%"


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


def rounds_text(rs, capital=False):
    """[2, 3, 4] -> 'rounds 2 to 4'; [1] -> 'round 1'; [1, 3, 5] -> 'rounds 1, 3 and 5'."""
    rs = sorted(rs)
    w = "no round" if not rs else (f"round {rs[0]}" if len(rs) == 1 else
                                   (f"rounds {rs[0]} to {rs[-1]}" if len(rs) > 2 and rs == list(range(rs[0], rs[-1] + 1)) else
                                    "rounds " + join_and(rs)))
    return cap(w) if capital else w


def runs_of(vals):
    """Maximal runs of equal values: [[first round, last round, value], ...] (rounds 1-based)."""
    out = []
    for i, v in enumerate(vals):
        if out and out[-1][2] == v:
            out[-1][1] = i + 1
        else:
            out.append([i + 1, i + 1, v])
    return out


def series_words(vals):
    """'30 in every round' / '30 in rounds 1 to 4; 25 in rounds 5 to 10' / 'from 20 to 35, 30 in round 10'."""
    rr = runs_of(vals)
    if len(rr) == 1:
        return f"{g(rr[0][2])} in every round"
    if len(rr) <= 4:
        return "; ".join(f"{g(v)} in {rounds_text(list(range(x, y + 1)))}" for x, y, v in rr)
    lo, hi = min(vals), max(vals)
    return (f"from {g(lo)} ({rounds_text([i + 1 for i, v in enumerate(vals) if v == lo])}) to {g(hi)} "
            f"({rounds_text([i + 1 for i, v in enumerate(vals) if v == hi])}), {g(vals[-1])} in round {len(vals)}")


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


def mentions(text, v):
    """Whether a message names the number v as a number token ('22.5' but not '122.5' or '22.55'; '30' but not '300' or '30.5')."""
    v = Fraction(v)
    s = str(int(v)) if v.denominator == 1 else f"{float(v):.2f}".rstrip("0").rstrip(".")
    return re.search(r"(?<![\d.])" + re.escape(s) + r"(?!\d|\.\d)", str(text or "")) is not None


# ------------------------------------------------------------------------------------------------ benchmarks
def benchmarks(a, c, q_max, n):
    """Exact one-shot benchmarks of the Cournot market with inverse demand P = max(0, a - Q) and unit cost c.
    Cournot-Nash: q = (a - c)/(n + 1) each. Joint-profit maximum (collusion, monopoly output): Q = (a - c)/2, shared equally.
    Competitive: Q = a - c, P = c, profit 0. For two firms also the pure equilibria on the whole-number grid 0..q_max and the
    whole-number outcomes that maximize joint profit (any split, and equal splits)."""
    A = Fraction(a - c)
    qN = A / (n + 1)
    QN = n * qN
    PN = a - QN
    piN = (PN - c) * qN                           # per firm and round
    QM = A / 2
    PM = a - QM
    PiM = (PM - c) * QM                           # whole market, per round
    B = {"qN": qN, "QN": QN, "PN": PN, "piN": piN, "QM": QM, "PM": PM, "PiM": PiM, "qc": QM / n, "pic": PiM / n,
         "QC": A, "int_ne": None, "int_joint": None, "int_sym": None}
    # the whole-number equal splits that maximize joint profit (any n): with q_c = 22.5 both 22 and 23 earn the most, so a
    # quantity up to the larger one counts as collusive
    sym = {q: n * (max(0, a - n * q) - c) * q for q in range(0, q_max + 1)}
    B["coll_int"] = [q for q in sym if sym[q] == max(sym.values())]
    B["coll_thr"] = max(B["qc"], Fraction(max(B["coll_int"])))
    if n == 2:
        def prof(qi, qj):
            return (max(0, a - qi - qj) - c) * qi
        rng = range(0, q_max + 1)
        br = {}
        for qj in rng:
            vals = [prof(qi, qj) for qi in rng]
            m = max(vals)
            br[qj] = {qi for qi in rng if vals[qi] == m}
        B["int_ne"] = sorted((x, y) for y in rng for x in br[y] if y in br[x])
        best = max(prof(x, y) + prof(y, x) for x in rng for y in rng)
        B["int_joint"] = (best, sorted({x + y for x in rng for y in rng if prof(x, y) + prof(y, x) == best}))
        sb = max(2 * prof(q, q) for q in rng)
        B["int_sym"] = (sb, [q for q in rng if 2 * prof(q, q) == sb])
    return B


def best_reply(a, c, q_max, others):
    """Whole-number best replies to the other firms' total quantity, and the profit they earn."""
    vals = {q: (max(0, a - q - others) - c) * q for q in range(0, q_max + 1)}
    m = max(vals.values())
    return [q for q in vals if vals[q] == m], m


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
    n = int(par("n_players", len(rounds[0].get("quantities") or []) or cfg.get("n_players") or 2))
    return {"a": int(par("a", 100)), "c": int(par("c", 10)), "q_max": int(par("q_max", 90)), "R": int(par("rounds", len(rounds))),
            "talk_passes": int(par("talk_passes", 0) or 0), "n": n, "rounds": rounds, "talk": talk,
            "game_end": next((e for e in events if e.get("event") == "game_end"), {}),
            "winners": over.get("winners", summary.get("winners"))}


def market(G):
    """Per round: quantities, total, price and every firm's profit (exact), each checked against the logged values."""
    a, c, n = G["a"], G["c"], G["n"]
    out, warn = [], []
    for i, e in enumerate(G["rounds"]):
        qs = [int(q) for q in (e.get("quantities") or [])]
        if len(qs) != n:
            warn.append(f"round {i + 1}: {len(qs)} quantities logged for {n} firms")
            qs = (qs + [0] * n)[:n]
        if int(e.get("round", i + 1)) != i + 1:
            warn.append(f"round_result {i}: round {e.get('round')} logged, {i + 1} expected")
        Q = sum(qs)
        if e.get("total_quantity") is not None and int(e["total_quantity"]) != Q:
            warn.append(f"round {i + 1}: total_quantity {e['total_quantity']} logged, {Q} summed from {qs}")
        P = max(0, a - Q)
        if e.get("price") is not None and int(e["price"]) != P:
            warn.append(f"round {i + 1}: price {e['price']} logged, max(0, {a} - {Q}) = {P}")
        prof = [Fraction((P - c) * q) for q in qs]
        out.append({"r": i + 1, "q": qs, "Q": Q, "P": P, "profit": prof, "joint": sum(prof, Fraction(0))})
    return out, warn


def verify(G, M, rewards):
    warn = []
    score = [sum((m["profit"][s] for m in M), Fraction(0)) for s in range(G["n"])]
    if len(M) != G["R"]:
        warn.append(f"{len(M)} rounds logged, {G['R']} configured")
    ge = G["game_end"]
    if ge.get("sum_total_quantity") is not None and Fraction(ge["sum_total_quantity"]) != sum(m["Q"] for m in M):
        warn.append(f"game_end sum_total_quantity {ge['sum_total_quantity']}, {sum(m['Q'] for m in M)} summed from the round results")
    if ge.get("sum_total_profit") is not None and abs(float(ge["sum_total_profit"]) - float(sum(score))) > 1e-6:
        warn.append(f"game_end sum_total_profit {ge['sum_total_profit']}, {float(sum(score))} recomputed")
    for s in range(G["n"]):
        if rewards[s] is not None and abs(rewards[s] - float(score[s])) > 1e-6:
            warn.append(f"reward of seat {s}: {rewards[s]} in the summary, {float(score[s])} recomputed from the quantities")
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


WRAP = f"written without the {LDQ}produce{RDQ} wrapper the game requires"
# Since the parser fix of 2026-11-13 (agents/action_io._wrap_bare_struct, commit 0e3de6a) the agent also accepts
# {"quantity": q} without the wrapper; recordings from that date on are read with that rule.
FIX_DATE = "2026-11-13"
BARE_OK = False


def _num(v):
    """A JSON number as an exact value (int or Fraction), None for anything else (bools included)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        return None
    return v if isinstance(v, int) else Fraction(v)


def read_answer(answer, phase, q_max):
    """How the recorder treats one reply. The agent (agents/action_io.parse_action) runs json.loads on the stripped reply (no
    code fence, no surrounding text) and needs a JSON object with only action_id and payload, the phase's action, and payload
    keys exactly the action's fields ({"produce"} or {"text"}); otherwise it asks again (LLM_JSON_RETRIES tries) and then plays a
    random legal action: for produce the only listed one, the struct default {"quantity": 30}; for talk the empty text. A reply
    that passes but breaks the struct schema (keys other than quantity, a quantity that is not an int) or the bounds 0..q_max is
    replaced by the phase's default_action, produce 30, without another try (engine/env.step).
    Returns {agent_ok, ok, value, why}: value is what the reply names (quantity or text) even when it was not usable, why the
    reason it was not usable."""
    out = {"agent_ok": False, "ok": False, "value": None, "why": None}
    loose = _norm_answer(answer)                       # only for showing what the reply names
    if isinstance(loose, dict) and isinstance(loose.get("payload"), dict):
        pl = loose["payload"]
        if phase == "TALK":
            out["value"] = pl.get("text") if isinstance(pl.get("text"), str) else None
        else:
            v = pl.get("produce", pl.get("quantity"))
            out["value"] = _num(v.get("quantity") if isinstance(v, dict) else v)
    try:
        data = json.loads((answer or "").strip())
    except ValueError:
        out["why"] = "JSON wrapped in other text" if loose is not None else "not JSON"
        return out
    if (not isinstance(data, dict) or not isinstance(data.get("action_id"), str) or set(data) - {"action_id", "payload"}
            or not isinstance(data.get("payload", {}), dict)):
        out["why"] = f"not of the form {{{LDQ}action_id{RDQ}: ..., {LDQ}payload{RDQ}: {{...}}}}"
        return out
    want = "talk" if phase == "TALK" else "produce"
    if data["action_id"] != want:
        out["why"] = f"the action {LDQ}{data['action_id']}{RDQ} is not this phase's action"
        return out
    pl = data.get("payload", {})
    field = "text" if phase == "TALK" else "produce"
    if BARE_OK and phase != "TALK" and set(pl) == {"quantity"}:
        pl = {"produce": pl}
    if set(pl) != {field}:
        out["why"] = (WRAP if phase != "TALK" and "quantity" in pl else
                      f"payload keys {join_and(sorted(map(str, pl))) or 'none'} instead of {field}")
        return out
    out["agent_ok"] = True
    if phase == "TALK":
        if not isinstance(pl["text"], str):
            out["why"] = "the text is not a string"
            return out
        out["ok"] = True
        return out
    v = pl["produce"]
    if not isinstance(v, dict) or set(v) != {"quantity"}:
        out["why"] = f"{LDQ}produce{RDQ} must hold exactly one key, quantity"
    elif _num(v["quantity"]) is None:
        out["why"] = "the quantity is not a number"
    elif not isinstance(v["quantity"], int):
        out["why"] = f"{g(v['quantity'])} is not a whole number" if Fraction(v["quantity"]).denominator != 1 else \
            f"{v['quantity']} is written as a decimal, not a whole number"
    elif not 0 <= v["quantity"] <= q_max:
        out["why"] = f"{v['quantity']} is outside 0 to {q_max}"
    else:
        out["ok"] = True
    return out


RETRY_NOTE = "That was not a legal action"       # appended to the prompt for every new try (agents/llm_agent.act)


def attach_calls(steps, calls, q_max, default_q):
    """Tie calls to steps in time order (see the module docstring). A decision's calls are the consecutive calls of its seat,
    phase and round, up to the first reply the agent accepts; when the prompts carry the retry note, a prompt without it starts
    the next decision. Each step gets src = "chosen" (a usable reply equal to the logged action), "default" (no usable reply;
    the logged action is the game's default) or "mismatch" (anything else, warned), and its answers. Returns the number of
    steps that got a call."""
    meta = [call_meta(c) for c in calls]
    tries = [(c.get("prompt") or "").count(RETRY_NOTE) for c in calls]
    marked = any(tries)

    def fits(k, st, strict=False):
        seat, (ph, rnd) = calls[k].get("seat"), meta[k]
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
            if j > ptr and marked and tries[j] == 0:  # a fresh prompt: the next decision of this seat, phase and round
                break
            j += 1
            if read_answer(calls[j - 1].get("answer"), st["phase"], q_max)["agent_ok"]:
                break
        window = list(range(ptr, j))
        if not window:
            continue
        reads = [read_answer(calls[k].get("answer"), st["phase"], q_max) for k in window]
        last = reads[-1]
        logged = st["text"] if st["phase"] == "TALK" else st["qty"]
        if last["ok"]:
            same = (_ws(last["value"]) == _ws(logged)) if st["phase"] == "TALK" else last["value"] == logged
            st["src"] = "chosen" if same else "mismatch"
        else:
            dflt = "" if st["phase"] == "TALK" else default_q
            st["src"] = "default" if logged == dflt else "mismatch"
        st["answers"] = [(r["value"], r["why"]) for r in reads]
        st["by_env"] = last["agent_ok"] and not last["ok"]   # passed the agent's check, replaced by the game's default action
        m = len(window)
        parts = []
        for t, (k, r) in enumerate(zip(window, reads)):
            c = calls[k]
            if m > 1 or not r["ok"]:
                verdict = "accepted" if r["ok"] else f"not usable: {r['why']}"
                parts.append(f"[Try {t + 1} of {m}, answer {c['answer'][:300]} ({verdict})]")
            parts.append(c.get("reasoning") or ("(no reasoning returned)" if m > 1 or not r["ok"] else ""))
        if st["src"] == "default":
            what = "message, so the game logged an empty one" if st["phase"] == "TALK" else \
                f"quantity, so the game played its default, {default_q}"
            parts.append(f"[No usable {what}.]" if not st["by_env"] else
                         f"[The reply passed the format check, but {last['why']}; the game replaced it with its default, {default_q}.]")
        elif st["src"] == "mismatch":
            what = f"the message {LDQ}{st['text'][:200]}{RDQ}" if st["phase"] == "TALK" else f"the quantity {st['qty']}"
            parts.append(f"[The log shows {what}, which this answer does not explain: {calls[window[-1]]['answer'][:300]}]")
            print(f"warning: round {st['round']} seat {st['seat']} {st['phase']}: logged {logged!r}, last answer "
                  f"{calls[window[-1]]['answer'][:120]!r}", file=sys.stderr)
        c = calls[window[-1]]
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        st["thinking"] = "\n\n".join(p for p in parts if p).strip()
        if c.get("prompt"):
            st["state"] = common.state_part(c["prompt"])
            st["state_from_prompt"] = True
        if st["src"] == "chosen":
            st["act"] = json.dumps(json.loads(c["answer"].strip()), ensure_ascii=False)
        elif st["src"] == "default":                  # not the model's answer: say so, and list what it answered
            rej = [None if v is None else (v if isinstance(v, (int, str)) else float(v)) for v, _ in st["answers"]]
            st["act"] = json.dumps({"action_id": "game_default",
                                    "payload": ({"game_default": "empty message", "rejected_messages": len(rej)}
                                                if st["phase"] == "TALK" else
                                                {"game_default_quantity": default_q, "rejected_answers": rej})},
                                   ensure_ascii=False)
        ptr = window[-1] + 1
        tied += 1
    if ptr < len(calls):
        print(f"warning: {len(calls) - ptr} calls after the last decision were not tied to any step", file=sys.stderr)
    return tied


# ------------------------------------------------------------------------------------------------ what the messages say
_SENT = re.compile(r"[;!?\n]|\.(?!\d)")
_COND_START = re.compile(r"^\W*(?:if|unless|otherwise|whenever|should\s+(?:either|you|one|anyone))\b", re.I)
_SKIP_CLAUSE = re.compile(r"\b(?:if|unless|otherwise|whenever|revert\w*|reversion|switch\w*|retaliat\w*|deviated|should have|"
                          r"last round|previous round|instead of)\b", re.I)
_VERB_NUM = re.compile(r"\b(?:produce|producing|target|targeting|try|continue with|continue at|stick to|stick with|stay at|"
                       r"go with|aim for|commit to)\s+(?:(?!rounds?\b)[a-z]+\s+){0,2}?(\d+(?:\.\d+)?)(?!\d|%|\.\d)", re.I)
_EACH_NUM = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s+(?:units?\s+)?each\b", re.I)


def said_qty(text, q_max):
    """The quantity a message names for the sender's own production this round, or None. Read from the text: sentences that
    start with a condition and clauses with a condition, a threat or a past deviation ('if you deviate I will switch to 45',
    'I deviated last round by producing 30') are skipped; in the rest the first number after a production verb ('I will
    produce 45', 'let's continue with 30 each', 'target 22.5') or before 'each' ('22.5 each'), within 0..q_max, is taken."""
    for sent in _SENT.split(text or ""):
        if _COND_START.match(sent):
            continue
        for cl in re.split(r"[,:]", sent):
            if _SKIP_CLAUSE.search(cl):
                continue
            for rx in (_VERB_NUM, _EACH_NUM):
                for mm in rx.finditer(cl):
                    if re.search(r"\b(?:rounds?|total(?: of)?)\s*$", cl[:mm.start(1)], re.I):
                        continue
                    v = Fraction(mm.group(1))
                    if 0 <= v <= q_max:
                        return v
    return None


# ------------------------------------------------------------------------------------------------ build
def build(src, calls=None):
    b = load_source(src)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    calls = calls or []
    events, summary, run, cfg = b["events"], b.get("summary") or {}, b.get("run") or {}, b.get("config") or {}
    G = read_game(events, summary, cfg)
    a, c, q_max, n, Rcfg, TP = G["a"], G["c"], G["q_max"], G["n"], G["R"], G["talk_passes"]
    M, warn = market(G)
    R = len(M)
    raw_rw = summary.get("rewards") or (run.get("result") or {}).get("rewards") or {}
    rewards = [float(raw_rw[str(s)]) if str(s) in raw_rw else None for s in range(n)]
    score, w2 = verify(G, M, rewards)
    for w in warn + w2:
        print("warning:", w, file=sys.stderr)
    B = benchmarks(a, c, q_max, n)
    qN, QN, PN, piN, QM, PM, PiM, qc, pic = (B[k] for k in ("qN", "QN", "PN", "piN", "QM", "PM", "PiM", "qc", "pic"))
    full_model, model, date = model_and_date(b)
    global BARE_OK
    BARE_OK = date >= FIX_DATE
    seed = run.get("seed", cfg.get("seed"))
    fallbacks = len(re.findall(r"\[LLM fallback\] Could not parse", b.get("runlog") or ""))
    dq = DEFAULT_Q

    F = [f"Firm {s + 1}" for s in range(n)]
    both = "both firms" if n == 2 else f"all {numword(n)} firms"
    the_firms = "The two firms" if n == 2 else f"The {numword(n)} firms"
    q = [[M[i]["q"][s] for i in range(R)] for s in range(n)]
    Qs = [m["Q"] for m in M]
    Ps = [m["P"] for m in M]
    joint = [m["joint"] for m in M]
    total = sum(joint, Fraction(0))
    cum = [[] for _ in range(n)]
    for s in range(n):
        acc = Fraction(0)
        for m in M:
            acc += m["profit"][s]
            cum[s].append(acc)

    def cat(x, ref_nash, ref_coll):
        """Where a quantity (or a total) sits: at Cournot-Nash (within less than 1 of it, so only the exact value when it is whole),
        at or below the collusive level, between the two, or above Cournot-Nash."""
        x = Fraction(x)
        if abs(x - ref_nash) < 1:
            return "nash"
        if x <= ref_coll:
            return "collude"
        return "restrain" if x < ref_nash else "expand"
    thr = B["coll_thr"]                               # a firm's quantity up to the best whole-number equal split counts as collusive
    fcat = [[cat(q[s][i], qN, thr) for i in range(R)] for s in range(n)]
    tcat = [cat(Qs[i], QN, QM) for i in range(R)]

    def tcat_of(Q):
        return cat(Q, QN, QM)
    nash_int = sorted({x for x in range(q_max + 1) if abs(x - qN) < 1})
    qN_txt = g(qN) if qN.denominator == 1 else f"{g(qN)} ({join_and(nash_int)} on the whole-number grid)"
    ci = B["coll_int"]
    KINDS = {"talk": ["Public message", "muted"],
             "fallback": [f"Game default {dq} (no usable answer)", "hold"],
             "collude": [f"Collusive split or less (q ≤ {g(thr)}" + (f"; {g(qc)} exactly)" if thr != qc else ")"),
                         CAT_COLOR["collude"]],
             "restrain": [f"Between collusive and Cournot-Nash", CAT_COLOR["restrain"]],
             "nash": [f"Cournot-Nash quantity ({g(qN) if qN.denominator == 1 else join_and(nash_int)})", CAT_COLOR["nash"]],
             "expand": [f"Above Cournot-Nash", CAT_COLOR["expand"]]}
    TOT_WORDS = {"collude": f"at or below the collusive output {g(QM)}", "restrain": f"between the collusive output {g(QM)} and the "
                 f"Cournot-Nash output {g(QN)}", "nash": "the Cournot-Nash output", "expand": f"above the Cournot-Nash output {g(QN)}"}

    def others_total(i, s):
        return Qs[i] - q[s][i]

    def br_short(i):
        """The round's best replies, for the result line: 'Best replies to the rival's quantity: Firm 1 33 or 34 (1,122
        instead of 1,110); Firm 2 played one.'"""
        parts, miss = [], 0
        for s in range(n):
            brs, bp = best_reply(a, c, q_max, others_total(i, s))
            if q[s][i] in brs:
                parts.append(f"{F[s]} played one")
            else:
                miss += 1
                parts.append(f"{F[s]} {' or '.join(map(str, brs))} ({g(bp)} instead of {g(M[i]['profit'][s])})")
        to = "the rival's quantity" if n == 2 else "the others' total"
        if not miss:
            return f" Each quantity was a best reply to {to}."
        return f" Best replies to {to}: " + "; ".join(parts) + "."

    # ---- steps: per round, the messages in log order, then one sealed quantity per firm in seat order
    steps, feed, fidx = [], [], {}
    saved = run.get("prompt") or {}
    rules_line = (f"Market: price P = max(0, {a} {MINUS} total quantity); unit cost {c}; each firm chooses a whole-number quantity "
                  f"from 0 to {q_max} and earns (P {MINUS} {c}) {TIMES} its own quantity, which can be negative.")
    history = []

    def observation(st, i, msgs_before):
        r = i + 1
        s = st["seat"]
        before = [cum[t][i - 1] if i else Fraction(0) for t in range(n)]
        if st["phase"] == "TALK":
            ph = (f"Current phase: TALK, round {r} of {Rcfg}. Each firm sends {plural(TP, 'public, non-binding message', word=True)} "
                  "before the production decision.")
        else:
            ph = (f"Current phase: DECIDE, round {r} of {Rcfg}. Each firm secretly chooses its quantity; all quantities, the price "
                  "and the profits are revealed together after the last firm has chosen.")
        lines = ["(Rebuilt from the game log; the exact prompt for this turn was not saved.)",
                 f"You are seat P{s} ({F[s]} on this page). Your role: PLAYER.", ph, rules_line,
                 "Scores so far: " + ", ".join(f"P{t} {g(before[t])}" for t in range(n)) + "."]
        lines += (["Earlier rounds:"] + history) if history else ["Earlier rounds: none yet."]
        if TP:
            lines += (["Messages this round so far:" if st["phase"] == "TALK" else "Messages this round:"]
                      + [f"- P{e.get('actor')}: {e.get('text', '')}" for e in msgs_before]) if msgs_before else \
                     ["Messages this round so far: none." if st["phase"] == "TALK" else "Messages this round: none."]
        return "\n".join(lines)

    feed.append({"x": 0, "step": 0, "who": "Rules",
                 "text": f"{cap(numword(n))} firms, {Rcfg} rounds. Each round every firm secretly chooses a whole-number quantity "
                         f"from 0 to {q_max}; the price is P = max(0, {a} {MINUS} total quantity) and each firm earns (P {MINUS} {c}) "
                         f"{TIMES} its own quantity. "
                         + (f"Before each decision every firm sends {plural(TP, 'public, non-binding message', word=True)}. " if TP else
                            "The firms cannot talk. ")
                         + f"Benchmarks: Cournot-Nash {g(qN)} each (total {g(QN)}, price {g(PN)}, {g(piN)} each a round); collusion "
                           f"total {g(QM)} (price {g(PM)}, {g(PiM)} a round for the market, {g(pic)} each); competitive total "
                           f"{g(B['QC'])} (price {c}, profit 0). When a firm's answer cannot be used, the game plays its default "
                           f"quantity, {dq}."})
    dec = {}                                          # (round, seat) -> the DECIDE step
    for i in range(R):
        r = i + 1
        msgs = G["talk"][i]
        m = M[i]
        for k, e in enumerate(msgs):
            s = int(e.get("actor", 0))
            text = str(e.get("text", ""))
            st = {"seat": s, "x": i, "kind": "talk", "round": r, "phase": "TALK", "text": text,
                  "head": f"Round {r} {DOT} public message" + (f" {k // n + 1} of {TP}" if TP > 1 else ""),
                  "act": json.dumps({"action_id": "talk", "payload": {"text": text}}, ensure_ascii=False)}
            st["state"] = observation(st, i, msgs[:k])
            fidx[len(steps)] = len(feed)
            feed.append({"x": i, "step": len(steps), "who": F[s] if 0 <= s < n else f"P{s}", "kind": "talk",
                         "text": f"{LDQ}{text}{RDQ}" if text.strip() else "(empty message)"})
            steps.append(st)
        for s in range(n):
            qq = q[s][i]
            st = {"seat": s, "x": i, "kind": fcat[s][i], "round": r, "phase": "DECIDE", "qty": qq,
                  "act": json.dumps({"action_id": "produce", "payload": {"produce": {"quantity": qq}}})}
            st["state"] = observation(st, i, msgs)
            dec[(r, s)] = st
            fidx[len(steps)] = len(feed)
            feed.append({"x": i, "step": len(steps), "who": F[s]})
            steps.append(st)
        res = (f"Quantities " + ", ".join(f"{F[t]} {m['q'][t]}" for t in range(n)) + f": total {m['Q']}, price "
               f"max(0, {a} {MINUS} {m['Q']}) = {m['P']}. Profit (P {MINUS} {c}) {TIMES} q: "
               + ", ".join(f"{F[t]} {g(m['profit'][t])}" for t in range(n))
               + f"; together {g(m['joint'])} (Cournot-Nash {g(n * piN)}, collusive maximum {g(PiM)}).")
        if i > 0 and Qs[i] != Qs[i - 1]:
            res += f" Total output {'rose' if Qs[i] > Qs[i - 1] else 'fell'} from {Qs[i - 1]} to {Qs[i]}."
        res += br_short(i)
        res += " Scores " + " / ".join(g(cum[t][i]) for t in range(n)) + "."
        feed.append({"x": i, "step": len(steps) - 1, "who": "Result", "text": res})
        history.append(f"- Round {r}: quantities {m['q']}, total {m['Q']}, price {m['P']}; profits "
                       + ", ".join(f"P{t} {g(m['profit'][t])}" for t in range(n)) + ".")
    # the saved prompt of the game's first decision (run.json keeps one prompt)
    if steps and saved.get("text") and saved.get("seat") == steps[0]["seat"] and str(saved.get("phase", "")).upper() == steps[0]["phase"]:
        rm = re.search(r'"round":\s*(\d+)', saved["text"][saved["text"].find("Public state:"):])
        if rm is None or int(rm.group(1)) == steps[0]["round"]:
            steps[0]["state"] = common.state_part(saved["text"])
            steps[0]["state_from_prompt"] = True
    tied = attach_calls(steps, calls, q_max, dq) if calls else 0

    # ---- which logged actions are the game's default (only the call log can say)
    dflt = [[(dec[(i + 1, s)]["src"] == "default") if "src" in dec[(i + 1, s)] else None for i in range(R)] for s in range(n)]
    known = bool(calls) and all(v is not None for row in dflt for v in row)
    nd = sum(1 for row in dflt for v in row if v)
    talk_dflt = [st for st in steps if st["phase"] == "TALK" and st.get("src") == "default"]
    mism = [st for st in steps if st.get("src") == "mismatch"]
    empty_msgs = sum(1 for t in G["talk"] for e in t if not str(e.get("text", "")).strip())
    n_dq = sum(1 for s in range(n) for i in range(R) if q[s][i] == dq)
    # without a call log: a run-log warning with no empty message means at least one logged default quantity is not a choice
    unsure = fallbacks > 0 and not known and nd == 0 and not talk_dflt
    sure_q = unsure and empty_msgs == 0 and n_dq > 0
    dcells = [(s, i) for s in range(n) for i in range(R) if dflt[s][i]]
    whys = sorted({w for s, i in dcells for _, w in dec[(i + 1, s)]["answers"] if w})
    cause = ("the answers used a format the game rejects" if whys == [WRAP] else
             "the answers could not be used" if whys else "")

    def answers_txt(st):
        """'its three answers (23, 23 and 23) were not usable: written without the "produce" wrapper the game requires'."""
        ans = st.get("answers") or []
        vals = [("no quantity" if v is None else g(v)) for v, _ in ans]
        ws = sorted({w for _, w in ans if w})
        if st.get("by_env"):
            return f"its answer {vals[-1]} passed the format check, but {ans[-1][1]}"
        k = len(ans)
        return (f"{'its answer' if k == 1 else f'its {numword(k)} answers'} ({join_and(vals)}) "
                f"{'was' if k == 1 else 'were'} not usable ({join_and(ws)})")

    def tries_txt(st):
        """For a chosen quantity that took more than one try: 'accepted on try 3; the answers before it, 30 and 30, were not
        usable (...)'."""
        ans = st.get("answers") or []
        if len(ans) < 2:
            return ""
        early = ans[:-1]
        return (f"accepted on try {len(ans)}; {'the answer' if len(early) == 1 else 'the answers'} before it, "
                f"{join_and('no quantity' if v is None else g(v) for v, _ in early)}, {'was' if len(early) == 1 else 'were'} "
                f"not usable ({join_and(sorted({w for _, w in early if w}))})")

    for k, st in enumerate(steps):
        fe = feed[fidx[k]]
        i, s = st["x"], st["seat"]
        if st["phase"] == "TALK":
            if st.get("src") == "default":
                st["head"] += f" {DOT} game default (empty)"
                fe["text"] = f"(empty message: the game's default, because {answers_txt(st)})"
            continue
        qq, m = st["qty"], M[i]
        chg = f" ({'up' if qq > q[s][i - 1] else 'down'} from {q[s][i - 1]})" if i > 0 and q[s][i - 1] != qq else ""
        sealed = "" if s == n - 1 else f" Sealed until {both} have chosen."
        if st.get("src") == "default":
            st["kind"] = "fallback"
            st["head"] = f"Round {i + 1} {DOT} game default {qq} (no usable answer) {DOT} price {m['P']}, {sg(m['profit'][s])}"
            fe["text"] = f"Game default {qq}{chg}: {answers_txt(st)}.{sealed}"
        else:
            st["head"] = f"Round {i + 1} {DOT} produces {qq} {DOT} price {m['P']}, {sg(m['profit'][s])}"
            tt = tries_txt(st)
            fe["text"] = f"Produces {qq}{chg}" + (f", {tt}" if tt else "") + f".{sealed}"
        fe["kind"] = st["kind"]
    max_tries = max([len(st.get("answers") or []) for st in steps] + [0])
    def short_answers(st):
        """For a grid cell: 'answers 23, 23, 30 rejected' / 'answer 95 replaced: 95 is outside 0 to 90'."""
        ans = st.get("answers") or []
        vals = ", ".join("none" if v is None else g(v) for v, _ in ans)
        if st.get("by_env"):
            return f"answer {vals.split(', ')[-1]} replaced: {ans[-1][1]}"
        return f"{'answer' if len(ans) == 1 else 'answers'} {vals} rejected"
    ans_short = {(i, s): short_answers(dec[(i + 1, s)]) for i in range(R) for s in range(n) if dflt[s][i]}
    all_ans = [st.get("answers") or [] for st in steps if st["phase"] == "DECIDE"]
    n_ans = sum(len(x) for x in all_ans)
    n_rej = sum(1 for x in all_ans for v, w in x if w)
    rej_dec = {(st["round"], st["seat"]): [v for v, _ in st.get("answers") or []] for st in steps if st["phase"] == "DECIDE"}
    has_reason = any(st.get("thinking") for st in steps)
    from_prompt = sum(1 for st in steps if st.get("state_from_prompt"))
    first = [[((dec[(i + 1, s)].get("answers") or [(None, None)])[0][0]) for i in range(R)] for s in range(n)]
    for st in steps:
        st.setdefault("thinking", "")
        for k in ("round", "phase", "qty", "text", "state_from_prompt", "src", "answers", "by_env"):
            st.pop(k, None)

    # ---- what each firm's messages say it will produce (the last message of the round that names a quantity)
    said = [[None] * R for _ in range(n)]
    said_msg = [[None] * R for _ in range(n)]
    for i in range(R):
        for e in G["talk"][i]:
            s = int(e.get("actor", 0))
            if not 0 <= s < n:
                continue
            v = said_qty(e.get("text"), q_max)
            if v is not None or said_msg[s][i] is None:
                said_msg[s][i] = str(e.get("text", ""))
            if v is not None:
                said[s][i] = v

    def said_run():
        """The longest run of rounds in which every firm says the same quantity and some firm's logged quantity differs."""
        best = None
        for x, y, v in runs_of([tuple(said[s][i] for s in range(n)) for i in range(R)]):
            if None in v or len(set(v)) != 1:
                continue
            if any(q[s][i] != v[0] for i in range(x - 1, y) for s in range(n)) and (best is None or y - x > best[1] - best[0]):
                best = (x, y, v[0])
        return best

    # ---- summary numbers
    avgQ = Fraction(sum(Qs), R)
    phi = (QN - avgQ) / (QN - QM) if QN != QM else Fraction(0)          # 1 = collusive output, 0 = Cournot-Nash, < 0 beyond Nash
    nash_rounds = [i + 1 for i in range(R) if all(fcat[s][i] == "nash" for s in range(n))]
    chosen_nash = [r for r in nash_rounds if not any(dflt[s][r - 1] for s in range(n))]
    all_nash = len(nash_rounds) == R
    nmsg = sum(len(t) for t in G["talk"])
    men_c = [i + 1 for i in range(R) if any(mentions(e.get("text"), qc) for e in G["talk"][i])]
    men_n = [i + 1 for i in range(R) if any(mentions(e.get("text"), qN) for e in G["talk"][i])]
    flat_q = [x for s in range(n) for x in q[s]]
    lo_q, hi_q = min(flat_q), max(flat_q)
    same_q = all(q[s] == q[0] for s in range(n))
    pair = [tuple(M[i]["q"]) for i in range(R)]
    varied = len(runs_of(pair)) > 1                  # False: every round has the same quantities, so per-round charts are flat
    k0 = R - 1                                       # start of the final run of one quantity profile (0-based)
    while k0 > 0 and pair[k0 - 1] == pair[-1]:
        k0 -= 1
    settled = R - k0 >= 3
    sr = said_run() if TP else None
    dq_word = f"{dq}s"

    def profile(i):
        return " / ".join(str(x) for x in M[i]["q"])

    def tot_words(Q):
        """Where a total sits, naming a total within one unit of the collusive output as such (e.g. 23 + 23 = 46)."""
        if Q != QM and abs(Q - QM) <= 1 and tcat_of(Q) != "nash":
            dq_ = abs(Fraction(Q) - QM)
            return f"{'one unit' if dq_ == 1 else g(dq_) + ' units'} {'above' if Q > QM else 'below'} the collusive output {g(QM)}"
        return TOT_WORDS[tcat_of(Q)]

    def blames():
        """Messages that admit or blame a deviation in the previous round: (round, sender, target seat). 'You deviated last
        round' blames the rival (two firms only), 'I apologize for deviating last round' / 'I deviated last round' the sender."""
        out = []
        for i in range(1, R):
            for e in G["talk"][i]:
                snd = int(e.get("actor", 0))
                for sent in re.split(r"[.;!?\n]", str(e.get("text", ""))):
                    if not re.search(r"\b(?:last|previous) round\b|\bround " + str(i) + r"\b", sent, re.I):
                        continue
                    if n == 2 and re.search(r"\b[Yy]ou\b[^.;!?]*\b(?:deviat|defect)\w*", sent):
                        out.append((i + 1, snd, 1 - snd))
                        break
                    if re.search(r"\bI\b[^.;!?]*\b(?:deviat|defect)\w*", sent) and not _COND_START.match(sent):
                        out.append((i + 1, snd, snd))
                        break
        return out

    def story():
        """What happened, in one or two sentences, when the call log shows defaults: what the messages agree on and what the
        firms first answer, how many answers the game rejected, which blamed deviations were the default, and the last round's
        announcements against what was logged."""
        out = []
        agreed = {}
        for i in range(R):
            v = said[0][i]
            if v is not None and all(said[s][i] == v for s in range(n)) and cat(v, qN, thr) == "collude":
                agreed.setdefault(v, []).append(i)
        if agreed:
            v, rs = max(agreed.items(), key=lambda kv: len(kv[1]))
            fa = sorted({first[s][i] for i in rs for s in range(n) if first[s][i] is not None})
            out.append(f"In this recording the messages agree on {g(v)} each ({rounds_text([i + 1 for i in rs])})"
                       + (f" and {both} first answer {join_and(g(x) for x in fa)}" if fa else "")
                       + f", but the game rejects {n_rej} of the {n_ans} quantity answers and plays its default {dq} for "
                         f"{nd} of the {n * R} quantities.")
        else:
            out.append(f"In this recording the game rejects {n_rej} of the {n_ans} quantity answers and plays its default {dq} for "
                       f"{nd} of the {n * R} quantities.")
        bl = blames()
        if bl:
            hit = [b for b in bl if dflt[b[2]][b[0] - 2]]
            rr = rounds_text(sorted({b[0] for b in bl}))
            if len(hit) == len(bl):
                out.append(f"In {rr} the messages admit or blame a deviation in the round before; "
                           + ("that quantity was" if len(bl) == 1 else "each of those quantities was")
                           + f" the game's default {dq}, not an answer the game accepted.")
            elif hit:
                out.append(f"In {rr} the messages admit or blame a deviation in the round before; {len(hit)} of those "
                           f"{len(bl)} quantities {'was' if len(hit) == 1 else 'were'} the game's default {dq}.")
        i = R - 1
        last = []
        for s in range(n):
            v = said[s][i]
            if v is None or (abs(Fraction(q[s][i]) - v) < 1 and not dflt[s][i]):
                continue
            if dflt[s][i]:
                av = [x for x in rej_dec.get((R, s), [])]
                last.append(f"{F[s]} says {g(v)}"
                            + (f", answers {join_and('no quantity' if x is None else g(x) for x in av)}," if av else "")
                            + f" and gets the default {dq}")
            else:
                last.append(f"{F[s]} says {g(v)} and chooses {q[s][i]}")
        if last:
            out.append(f"In the last round " + "; ".join(last) + ".")
        return " ".join(out)

    # ---- title
    say_txt = (f"{both} say {g(sr[2])} in {rounds_text(list(range(sr[0], sr[1] + 1)))}" if sr else "")
    if nd:
        lead = None                                  # a run of rounds in which every firm's first answer is the same quantity
        for x, y, v in runs_of([tuple(first[s][i] for s in range(n)) for i in range(R)]):
            if None in v or len(set(v)) != 1 or y == x:
                continue
            diff = sum(1 for i in range(x - 1, y) for s in range(n) if q[s][i] != v[0])
            if diff and (lead is None or diff > lead[0]):
                lead = (diff, x, y, v[0])
        core = f"{nd} of the {n * R} logged quantities are the game's default {dq}"
        title = ((f"{'Both firms' + chr(39) if n == 2 else 'Every firm' + chr(39) + 's'} first answer is {g(lead[3])} in "
                  f"{rounds_text(list(range(lead[1], lead[2] + 1)))}, but {core}" if lead else f"{core}, not the firms' choice")
                 + (f": {cause}" if cause else ""))
    elif sure_q and all_nash and nash_int == [dq] and lo_q == hi_q == dq:
        title = (f"Every logged quantity is {dq}, the Cournot-Nash quantity"
                 + (f", although {say_txt}" if say_txt else "")
                 + f"; at least one of these {dq_word} is the game's default, not a choice")
    else:
        if all_nash and len(nash_int) == 1:
            title = f"{cap(both)} produce the Cournot-Nash quantity, {g(qN)}, in all {numword(R)} rounds"
            if say_txt:
                title += f", although {say_txt}"
            elif TP and men_c and men_c[0] == 1:
                title += f", although the first round's messages mention the collusive share, {g(qc)}"
            else:
                title += f": price {g(PN)} and {g(piN)} profit each a round"
        elif all(t == "nash" for t in tcat):
            title = f"Total output stays at the Cournot-Nash level, {g(QN)}, in all {numword(R)} rounds"
            if settled:
                title += (f"; the firms split it {profile(R - 1)} in every round" if k0 == 0 else
                          f"; from round {k0 + 1} the firms split it {profile(R - 1)}")
        elif (R >= 3 and len({tuple(q[s][i] for s in range(n)) for i in range(R - 1)}) == 1
              and tuple(q[s][R - 1] for s in range(n)) != tuple(q[s][0] for s in range(n))):
            title = (f"{the_firms} produce {profile(0)} in rounds 1 to {R - 1}: total {Qs[0]}, {tot_words(Qs[0])}, price {Ps[0]}; "
                     f"in the last round they switch to {profile(R - 1)}: total {Qs[-1]}, {tot_words(Qs[-1])}, price {Ps[-1]}")
        elif settled and k0 == 0:
            title = (f"{the_firms} produce {profile(R - 1)} in every round: total {Qs[-1]}, {tot_words(Qs[-1])}, "
                     f"price {Ps[-1]}")
        elif settled:
            title = (f"After {plural(k0, 'round', word=True)} of other quantities, {the_firms.lower()} settle on {profile(R - 1)} "
                     f"from round {k0 + 1} to {R}: total {Qs[-1]}, {tot_words(Qs[-1])}, price {Ps[-1]}")
        else:
            where = (f"above the Cournot-Nash output {g(QN)}" if avgQ > QN else f"below the collusive output {g(QM)}" if avgQ < QM else
                     f"between the collusive output {g(QM)} and the Cournot-Nash output {g(QN)}")
            title = (f"Total output averages {g(round(avgQ, 2))} over {numword(R)} rounds, {where}; the price ranges from "
                     f"{min(Ps)} to {max(Ps)}")
        if sure_q:
            title += f"; at least one logged {dq} is the game's default, not a choice"

    # ---- lede
    lede = (f"{cap(numword(n))} firms sell the same product for {numword(Rcfg)} rounds. Each round {both} secretly choose how much "
            f"to produce, a whole number from 0 to {q_max}; the price is {a} minus the total output, and each firm earns "
            f"(price {MINUS} {c}) {TIMES} its own quantity, which adds up to its score. "
            + (f"Before each decision every firm sends {plural(TP, 'public, non-binding message', word=True)}, so the firms can try "
               "to agree to hold output down." if TP else
               "The firms cannot talk; they see each other's quantities only after each round.")
            + f" A quantity the game cannot read from the model's answer is replaced by the game's default, {dq}.")
    if nd and TP:
        lede += " " + story()

    # ---- charts
    legend = [[f"Collusive split or less (≤ {g(thr)})", CAT_COLOR["collude"]], ["Between collusive and Cournot-Nash", CAT_COLOR["restrain"]],
              ["Cournot-Nash", CAT_COLOR["nash"]], ["Above Cournot-Nash", CAT_COLOR["expand"]]]
    if nd:
        legend.append([f"Game default {dq} (no usable answer)", "hold"])
    unsure_tip = f"; may be the game's default (see the note below)" if sure_q else ""
    SHORT = {"collude": "collusive split or less", "restrain": "between collusive and Cournot-Nash", "nash": "Cournot-Nash",
             "expand": "above Cournot-Nash"}

    def quote(txt, k=64):
        """A message cut to about k characters at a word boundary (the feed shows it in full)."""
        t = _ws(txt)
        if len(t) <= k:
            return f"{LDQ}{t}{RDQ}"
        cut = t[:k].rsplit(" ", 1)[0] if " " in t[:k] else t[:k]
        return f"{LDQ}{cut.rstrip(',;:.')}…{RDQ}"
    rows, gcells = [], []
    for s in range(n):
        if TP:
            row = []
            for i in range(R):
                v, txt = said[s][i], said_msg[s][i]
                if txt is None:
                    row.append(None)
                elif v is None:
                    row.append({"c": "muted", "o": 0.4, "t": MINUS,
                                "tip": f"names no quantity: {quote(txt)}" if txt.strip() else "empty message"})
                else:
                    row.append({"c": CAT_COLOR[cat(v, qN, thr)], "o": 0.7, "t": g(v),
                                "tip": (f"names {g(v)} ("
                                        + ("kept" if abs(Fraction(q[s][i]) - v) < 1 and not dflt[s][i] else f"logged {q[s][i]}"
                                           + (", default" if dflt[s][i] else ""))
                                        + f"): {quote(txt)}")})
            rows.append(f"{F[s]} says")
            gcells.append(row)
        row = []
        for i in range(R):
            mk = f"({M[i]['P']} {MINUS} {c}) {TIMES} {q[s][i]} = {g(M[i]['profit'][s])}; score {g(cum[s][i])}"
            st = dec[(i + 1, s)]
            if dflt[s][i]:
                row.append({"c": "hold", "t": f"{q[s][i]}*",
                            "tip": f"game default {q[s][i]} ({ans_short[(i, s)]}); {mk}"})
            else:
                row.append({"c": CAT_COLOR[fcat[s][i]], "t": str(q[s][i]),
                            "tip": (f"{q[s][i]}, {SHORT[fcat[s][i]]}; {mk}" + (unsure_tip if q[s][i] == dq else ""))})
        rows.append(F[s])
        gcells.append(row)
    rows.append("Market price")
    gcells.append([{"c": CAT_COLOR[tcat[i]], "t": str(Ps[i]),
                    "tip": (f"total {Qs[i]} (" + " + ".join(str(x) for x in M[i]["q"]) + f"), {SHORT[tcat[i]]}; "
                            f"{a} {MINUS} {Qs[i]} = {Ps[i]}; joint profit {g(joint[i])} of {g(PiM)}")} for i in range(R)])
    lg = "logged quantity" if (nd or sure_q) else "quantity"
    firm_txt = (f"{cap(both)}' {lg} was {series_words(q[0])}" if same_q and n == 2 else
                f"Every firm's {lg} was {series_words(q[0])}" if same_q else
                "; ".join(f"{F[s]}: {series_words(q[s])}" for s in range(n)))
    grid_sub = ("One column per round. "
                + (f"For each firm the lighter row shows the quantity its message names for its own production that round (read "
                   f"from the text; conditional threats such as {LDQ}if you deviate I will produce 45{RDQ} are not counted, "
                   f"{MINUS} means no quantity named), the row below it the logged quantity; " if TP else
                   "Each firm's row shows its quantity; ")
                + f"the bottom row is the market price. Colors place a quantity against the benchmarks: green at or below the "
                  f"collusive split, {g(qc)} each (half the joint-profit output {g(QM)})"
                + ((f", or {g(thr)} on the whole-number grid, where {join_and(ci)} each earn the same" if len(ci) > 1 else
                    f", or {g(thr)}, the best equal split on the whole-number grid") if thr != qc else "")
                + f"; yellow between that and the Cournot-Nash quantity {g(qN)}, blue at Cournot-Nash, orange above it; the price "
                  f"row is colored by total output against {g(QM)} and {g(QN)}. "
                + (f"Gray {dq}*: no usable answer, so the game played its default quantity, {dq} ({nd} of {n * R}); the tooltip "
                   "shows what the firm answered. " if nd else "")
                + (f"At least one logged {dq} is the game's default rather than a choice, but without the call log it cannot be "
                   "marked. " if sure_q else "")
                + ("" if nd else f"{firm_txt}. ")
                + ("" if varied else
                   f"Every round has the same quantities ({profile(0)}, total {Qs[0]}, price {Ps[0]}), so the charts of each firm's "
                   "quantity, total output and profit per round are left out: they would be flat lines. ")
                + "Hover a column for the round: each message (shortened; the feed has it in full), each firm's profit and score, "
                  f"and the market. The feed's round result names each firm's best reply to {'the rival' + chr(39) + 's quantity' if n == 2 else 'the others' + chr(39) + ' total'}.")
    charts = [{"title": "What each firm says and the quantity logged" if TP else "Quantity each round", "bars": [],
               "sub": grid_sub, "grid": {"rows": rows, "cells": gcells, "rowHeight": 30, "legend": legend}}]
    if varied:
        lines2 = [{"name": F[s], "color": FIRM_COLORS[s % 4], "v": q[s], "dots": True} for s in range(n)]
        if nd:                                       # one series for every default (always the same quantity); the sub names the firms
            lines2.append({"name": f"Game default {dq}", "color": "hold", "noline": True, "dots": True,
                           "v": [dq if any(dflt[s][i] for s in range(n)) else None for i in range(R)]})
        lines2 += [{"name": "Cournot-Nash", "color": CAT_COLOR["nash"], "dash": True, "v": [fv(qN)] * R, "end": True},
                   {"name": "Collusive share", "color": CAT_COLOR["collude"], "dash": True, "v": [fv(qc)] * R, "end": True}]
        charts.append({"title": "Each firm's quantity against the benchmarks", "height": 220, "bars": [], "lines": lines2,
                       "fmt": fmt_for(*[l["v"] for l in lines2]),
                       "sub": (f"Dashed: the Cournot-Nash quantity {qN_txt}, each firm's best reply when the other also "
                               f"plays it, and the collusive share {g(qc)}, half of the output {g(QM)} that maximizes joint profit. "
                               + (f"Large gray dots: the game's default {dq}, played because the firm's answer could not be used ("
                                  + "; ".join(f"{F[s]} in {rounds_text([i + 1 for i in range(R) if dflt[s][i]])}"
                                              for s in range(n) if any(dflt[s])) + "). " if nd else "")
                               + ("The firms' quantities are identical in every round, so their lines lie on top of each other. "
                                  if same_q and n > 1 else "")
                               + (f"Every quantity was {lo_q}." if lo_q == hi_q else f"Quantities ranged from {lo_q} to {hi_q}."))})
        lines3 = [{"name": "Total output", "color": "ink", "v": Qs, "dots": True, "end": True},
                  {"name": "Market price", "color": "hold", "v": Ps, "dots": True, "thin": True, "end": True},
                  {"name": "Collusive output", "color": CAT_COLOR["collude"], "dash": True, "v": [fv(QM)] * R, "end": True},
                  {"name": "Cournot-Nash output", "color": CAT_COLOR["nash"], "dash": True, "v": [fv(QN)] * R, "end": True}]
        charts.append({"title": "Total output and the price it sets", "height": 230, "bars": [], "lines": lines3,
                       "fmt": fmt_for(*[l["v"] for l in lines3]),
                       "sub": (f"Total output (thick) and the market price {a} {MINUS} total (thin gray), so the two lines mirror "
                               f"each other. Dashed: the collusive output {g(QM)}, which sets the price {g(PM)}, and the Cournot-Nash "
                               f"output {g(QN)}, price {g(PN)}; the competitive output {g(B['QC'])} would push the price down to the "
                               f"unit cost {c}. Total output by round: {series_words(Qs)}; collusion index (Cournot-Nash output "
                               f"{MINUS} output) / (Cournot-Nash output {MINUS} collusive output), averaged over the game: "
                               f"{float(phi):.2f} (1 collusive, 0 Cournot-Nash, below 0 beyond it)"
                               + (f", from the logged quantities, {nd} of which are the game's default {dq}" if nd else "") + ".")})
        prof_vals = [[fv(M[i]["profit"][s]) for i in range(R)] for s in range(n)]
        below_n = [i + 1 for i in range(R) if joint[i] < n * piN]
        above_n = [i + 1 for i in range(R) if joint[i] > n * piN]
        lines4 = [{"name": "Cournot-Nash profit", "color": CAT_COLOR["nash"], "dash": True, "v": [fv(piN)] * R, "end": True},
                  {"name": "Collusive share", "color": CAT_COLOR["collude"], "dash": True, "v": [fv(pic)] * R, "end": True}]
        charts.append({"title": "Profit each round", "height": 220, "lines": lines4,
                       "bars": [{"name": F[s], "color": FIRM_COLORS[s % 4], "v": prof_vals[s]} for s in range(n)],
                       "fmt": fmt_for(*prof_vals, *[l["v"] for l in lines4]),
                       "zero": any(x < 0 for row in prof_vals for x in row),
                       "sub": (f"Each firm's profit, (price {MINUS} {c}) {TIMES} its quantity. Dashed: {g(piN)}, each firm's profit "
                               f"when {both} play Cournot-Nash, and {g(pic)}, an equal share of the collusive maximum {g(PiM)}. "
                               + (f"Joint profit equals the Cournot-Nash level in every round." if not below_n and not above_n else
                                  (f"Joint profit above the Cournot-Nash level in {rounds_text(above_n)}" if above_n else "")
                                  + ("; " if above_n and below_n else "")
                                  + (f"below it in {rounds_text(below_n)}" if below_n else "") + "."))})
    lines5 = [{"name": F[s], "color": FIRM_COLORS[s % 4], "v": [fv(x) for x in cum[s]], "dots": True, "end": True} for s in range(n)]
    lines5 += [{"name": "Cournot-Nash path", "color": CAT_COLOR["nash"], "dash": True, "v": [fv(piN * (i + 1)) for i in range(R)], "end": True},
               {"name": "Collusive path", "color": CAT_COLOR["collude"], "dash": True, "v": [fv(pic * (i + 1)) for i in range(R)], "end": True}]
    gap = pic * R - max(score)
    charts.append({"title": "Cumulative profit", "height": 240, "bars": [], "lines": lines5,
                   "fmt": fmt_for(*[l["v"] for l in lines5]),
                   "zero": any(x < 0 for row in cum for x in row),
                   "sub": (f"Each firm's running total, its score. Dashed: the score after each round if {both} played Cournot-Nash "
                           f"every round ({g(piN)} a round each) or split the collusive maximum equally ({g(pic)} a round each). "
                           + ("The firms' lines and the Cournot-Nash path lie on top of each other. "
                              if all(cum[s] == [piN * (i + 1) for i in range(R)] for s in range(n)) else "")
                           + "Final: " + ", ".join(f"{F[s]} {g(score[s])}" for s in range(n))
                           + f", against {g(piN * R)} at Cournot-Nash and {g(pic * R)} with equal collusion"
                           + ((f", which {both} miss by {g(gap)}" if len(set(score)) == 1 else
                               f", which the leader misses by {g(gap)}") if gap > 0 else "") + ".")})

    # ---- tiles
    tq = (f"{series_words(Qs)[:1].upper()}{series_words(Qs)[1:]}" if len(runs_of(Qs)) <= 4 else
          f"Average over {plural(R, 'round')}; {series_words(Qs)}")
    tiles = [{"k": "Total output", "v": g(round(avgQ, 2)),
              "s": (f"{tq}. Collusive output {g(QM)}, Cournot-Nash {g(QN)}, competitive {g(B['QC'])}; collusion index "
                    f"{float(phi):.2f} (1 collusive, 0 Cournot-Nash)"
                    + (f", from the logged quantities, {nd} of which are the game's default {dq}" if nd else "") + "."),
              "spark": [fv(x) for x in Qs]}]
    tiles.append({"k": "Joint profit", "v": f"{g(total)} of {g(PiM * R)}",
                  "s": (f"{share_pct(total, PiM * R)} of the collusive maximum ({g(PiM)} a round for the market). Cournot-Nash play in "
                        f"every round earns {g(n * piN * R)} ({share_pct(n * piN * R, PiM * R)}); the competitive outcome earns 0.")})
    best = max(score)
    leaders = [s for s in range(n) if score[s] == best]
    if len(leaders) == 1:
        L = leaders[0]
        S = max((s for s in range(n) if s != L), key=lambda s: score[s])
        ws = f"{F[L]} wins by {g(best - score[S])}"
        # where the margin comes from: the rounds in which the two firms' profits differ (same price, so the larger quantity
        # earns more), and whether the larger quantity there was the game's default
        D = [i for i in range(R) if M[i]["profit"][L] != M[i]["profit"][S]]
        won_in = {L: [i for i in D if M[i]["profit"][L] > M[i]["profit"][S]], S: [i for i in D if M[i]["profit"][S] > M[i]["profit"][L]]}
        hi_dflt = [i for i in D if dflt[L if i in won_in[L] else S][i]]
        if nd and D and hi_dflt:
            def who(t):
                rs = won_in[t]
                if not rs:
                    return ""
                return (f"{rounds_text([i + 1 for i in rs])} ("
                        + ", ".join(f"+{g(abs(M[i]['profit'][L] - M[i]['profit'][S]))}" for i in rs) + f") for {F[t]}")
            split = join_and(x for x in (who(L), who(S)) if x)
            other_eq = len(D) < R
            if len(hi_dflt) == len(D) and all(not dflt[S if i in won_in[L] else L][i] for i in D):
                ws += (f", all from rounds where the game's default {dq} met a lower chosen quantity: {split}."
                       + (f" In the other rounds {'both firms' if n == 2 else F[L] + ' and ' + F[S]} earned the same." if other_eq else ""))
            else:
                ws += (f": {split}; in {rounds_text([i + 1 for i in hi_dflt])} the larger quantity was the game's default {dq}."
                       + (f" In the other rounds {'both firms' if n == 2 else F[L] + ' and ' + F[S]} earned the same." if other_eq else ""))
        else:
            ws += "."
            hi_firm = max(range(n), key=lambda s: sum(q[s]))
            if sum(q[hi_firm]) > min(sum(q[s]) for s in range(n)):
                ws += (" Total " + ("logged " if nd or sure_q else "") + "quantity over the game: "
                       + ", ".join(f"{F[s]} {sum(q[s])}" for s in range(n))
                       + (f", defaults {dq} included" if nd else "") + ".")
    else:
        win = G["winners"] if isinstance(G["winners"], list) else []
        named = sorted(int(w) for w in win) == leaders
        ws = (f"A tie: the log names {'both firms' if len(leaders) == n == 2 else join_and(F[s] for s in leaders)} as winners."
              if named else f"A tie between {join_and(F[s] for s in leaders)}.")
    tiles.append({"k": "Final scores", "v": " / ".join(g(x) for x in score), "s": " / ".join(F) + ". " + ws})
    if nd:
        other = [r for r in nash_rounds if r not in chosen_nash]
        k_in = sum(1 for r in other for s in range(n) if dflt[s][r - 1])
        nv = f"{len(chosen_nash)} of {R}"
        s4 = ((f"{cap(both)} chose {g(qN)} in {rounds_text(chosen_nash)}." if chosen_nash else
               f"No round with {both} choosing the Cournot-Nash quantity.")
              + (f" {rounds_text(other, capital=True)} also log the Cournot-Nash quantity for {both}, but {k_in} of those quantities "
                 f"{'is' if k_in == 1 else 'are'} the game's default {dq}, so they are not counted." if other else "")
              + f" {nd} of the {n * R} logged quantities are defaults: "
              + "; ".join(f"{F[s]} in {rounds_text([i + 1 for i in range(R) if dflt[s][i]])}" for s in range(n) if any(dflt[s])) + ".")
    elif sure_q:
        in_nash = all(i + 1 in nash_rounds for s in range(n) for i in range(R) if q[s][i] == dq)
        ub = len(nash_rounds) - (1 if in_nash else 0)
        nv = f"at most {ub} of {R}"
        s4 = ((f"Every logged quantity is {dq}" if lo_q == hi_q == dq else
               f"{cap(both)} log the Cournot-Nash quantity in {rounds_text(nash_rounds)}" if nash_rounds else
               f"No round with {both} at the Cournot-Nash quantity")
              + f", but at least one logged {dq} is the game's default rather than a choice (see the note below), so at most "
                f"{plural(ub, 'round', word=True)} can have {both} choosing it. The log does not say which round.")
    else:
        nv = f"{len(nash_rounds)} of {R}"
        s4 = (f"{cap(both)} produced {g(qN)} in every round." if all_nash and len(nash_int) == 1 else
              (f"{cap(both)} at the Cournot-Nash quantity in {rounds_text(nash_rounds)}." if nash_rounds else
               f"No round with {both} at the Cournot-Nash quantity."))
        if not (all_nash and len(nash_int) == 1):
            s4 += f" Every quantity was {lo_q}." if lo_q == hi_q else f" Quantities ranged from {lo_q} to {hi_q}."
    if TP:
        low = sorted({v for s in range(n) for v in said[s] if v is not None and v < qN and abs(v - qN) >= 1})
        both_low = {v: [i + 1 for i in range(R) if all(said[s][i] == v for s in range(n))] for v in low}
        both_low = {v: rs for v, rs in both_low.items() if rs}
        s4 += (f" {plural(nmsg, 'public message')} in all; "
               + ("; ".join(f"{both} say {g(v)} in {rounds_text(rs)}" for v, rs in both_low.items()) if both_low else
                  f"the collusive share {g(qc)} is mentioned in " + (rounds_text(men_c) if men_c else "no round's messages"))
               + ".")
    tiles.append({"k": "Rounds at Cournot-Nash", "v": nv, "s": s4})

    # ---- foot
    cfg_bits = ", ".join(x for x in [f"seed {seed}" if seed is not None else "", f"{numword(n)} firms", plural(Rcfg, "round"),
                                     f"a = {a}, c = {c}, quantities 0 to {q_max}",
                                     "no talk" if not TP else plural(TP, "public message", word=True) + " per firm before each decision"] if x)
    by = f"{model} ({full_model})" if full_model and full_model != model else model
    mean_firm = total / (R * n)
    gain = (mean_firm - piN) / (pic - piN) if pic != piN else Fraction(0)
    grid_txt = ""
    if B["int_ne"] is not None:
        ne = B["int_ne"]
        jb, jQ = B["int_joint"]
        sb, sq = B["int_sym"]
        grid_txt = (f" On the whole-number grid the pure equilibria are {join_and('(' + ', '.join(map(str, p)) + ')' for p in ne)}; "
                    f"joint profit peaks at {g(jb)} with total output {join_and(jQ)}"
                    + (f", and an equal split ({' or '.join(map(str, sq))} each) earns {g(sb)} together, {g(Fraction(sb, 2))} each"
                       if sb != jb else "") + ".")
    rule_txt = ('The game uses an answer only if it is exactly one JSON object such as {"action_id": "produce", "payload": '
                f'{{"produce": {{"quantity": 25}}}}}}'
                + (' (or, since the parser fix of 2026-11-13, the same without the wrapper: {"action_id": "produce", "payload": '
                   '{"quantity": 25}})' if BARE_OK else '')
                + f' with a whole number from 0 to {q_max}; it asks again after an unusable answer, '
                + (f"up to {numword(max_tries)} tries in all here, " if max_tries > 1 else "")
                + f"and after the last try plays its default, which for a quantity is always {dq} and for a message the empty "
                  f"text. A well-formed answer whose quantity is not a whole number from 0 to {q_max} is replaced by {dq} at once. ")
    if calls:
        fb_txt = (rule_txt
                  + (f"The call log shows {nd} of the {n * R} quantity decisions with no usable answer ({cause}); they are marked "
                     f"{dq}* on the grid and {LDQ}Game default{RDQ} in the steps, and left out of the Cournot-Nash count. " if nd else
                     "The call log shows a usable answer for every matched quantity decision. ")
                  + (f"{cap(plural(len(talk_dflt), 'message', word=True))} {'is' if len(talk_dflt) == 1 else 'are'} the game's empty "
                     "default. " if talk_dflt else "")
                  + ("The run log's fallback warning is printed only once per run, so it does not count them. " if fallbacks else "")
                  + (f"{cap(plural(len(mism), 'step', word=True))} {'has' if len(mism) == 1 else 'have'} a logged action that the "
                     "matched answers do not explain. " if mism else ""))
    elif fallbacks:
        fb_txt = (rule_txt
                  + "The run log shows that at least one answer could not be used; the recorder prints this warning only once per "
                    "run, so it does not say how many. "
                  + (f"No message is empty, so at least one logged {dq} is the game's default, not the model's choice; without the "
                     "call log the page cannot tell which. " if sure_q else
                     f"{cap(plural(empty_msgs, 'message', word=True))} {'is' if empty_msgs == 1 else 'are'} empty and may be the "
                     f"default; a logged {dq} may be one as well. " if empty_msgs else ""))
    else:
        fb_txt = ""
    foot = (f"Data: the {date} recording of eco_cournot ({cfg_bits}); {'both firms are' if n == 2 else 'every firm is'} played by {by}. Every number on this page is "
            "computed from the game log: the round_result events (quantities, total quantity, price)"
            + (" and the talk events (the messages)" if TP else "") + f"; profits are recomputed with the rule (P {MINUS} c) "
            f"{TIMES} own quantity, P = max(0, a {MINUS} total), and checked against the logged price, the summary's rewards and the "
            "logged totals"
            + (f" (total quantity {g(sum(Qs))}, total profit {g(total)})" if G["game_end"] else "") + ". "
            + (f"Reasoning and observations come from the saved model calls ({tied} of {len(steps)} decisions matched; a decision "
               "with several tries shows every try). " if has_reason else
               f"The call log ({len(calls)} calls, {tied} of {len(steps)} decisions matched) holds no reasoning. " if calls else
               "Reasoning was not saved in this recording (it has no call log), so no step shows the model's reasoning. ")
            + ("" if from_prompt == len(steps) else
               ("The observation of the first decision is the prompt saved with the run; the others are rebuilt from the log. "
                if from_prompt == 1 else "Observations without a saved prompt are rebuilt from the log. "))
            + fb_txt
            + (f"The quantity a message names is read from its text: the first number after a production verb or before "
               f"{LDQ}each{RDQ}, skipping conditional threats and past rounds. " if TP else "")
            + f"Firm k is seat P(k{MINUS}1) in the log and the prompts; in each round the firms choose in seat order and none sees "
            "another's quantity before the reveal. "
            f"Benchmarks, computed from a, c and the number of firms: Cournot-Nash quantity (a {MINUS} c)/{n + 1} = {g(qN)} each, "
            f"total {g(QN)}, price {g(PN)}, profit {g(piN)} each a round; collusion (the output that maximizes joint profit, "
            f"(a {MINUS} c)/2 = {g(QM)}), price {g(PM)}, {g(PiM)} a round for the market, {g(pic)} each; competitive output "
            f"{g(B['QC'])}, price {c}, profit 0.{grid_txt} Profit gain, (average profit per firm and round {MINUS} Cournot-Nash "
            f"profit) / (collusive share {MINUS} Cournot-Nash profit) = ({g(round(mean_firm, 2))} {MINUS} {g(piN)}) / ({g(pic)} "
            f"{MINUS} {g(piN)}) = {float(gain):.3f} (1 collusive, 0 Cournot-Nash). Huck, Normann and Oechssler (2004) found that with two firms human outcomes are often more "
            "collusive than the Cournot-Nash prediction, while with four or more firms they move to the Nash quantity or beyond. "
            "Sources: Cournot (1838); Tirole (1988); Huck, Normann and Oechssler (2004).")

    kinds = {k: v for k, v in KINDS.items() if any(st["kind"] == k for st in steps)}
    d = {"id": GID, "name": NAME, "title": title, "lede": lede, "seatNames": F, "tiles": tiles, "foot": foot,
         "x": {"n": R, "labels": [f"R{i + 1}" for i in range(R)]}, "charts": charts, "kinds": kinds, "steps": steps,
         "feed": feed, "feedTitle": "Messages and quantities" if TP else "Quantities round by round",
         "feedSub": ("Every public message, each" if TP else "Each") + " quantity as it was chosen (sealed until all firms had chosen), "
                    "then the round's result. The selected step is highlighted; later steps are hidden.",
         "eyebrow": f"social_game_env {DOT} {GID} {DOT} {model}" + (", reasoning on" if has_reason else "") + f" {DOT} recorded {date}"}
    return en_all(d)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    d = build(src, calls)
    common.write(d, out, date=model_and_date(load_source(src))[2])
