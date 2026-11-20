"""Replay data for eco_trading_regime (one trader sets a position each round in a market that switches between a bull and a bear
regime; Hamilton 1989 regime switching, risk attitude as the entropic certainty equivalent of the RQE paper).

usage: python3 -I eco_trading_regime.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md), a file inside
          one, or the rendered Markdown replay of one (one "- **event** ... `{json}`" line per logged event).
  OUT_DIR where <OUT_DIR>/eco_trading_regime.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). The calls are split
          in time order into one consecutive group per decision by a minimum-cost alignment (attach_calls): a call whose prompt
          shows another round (the 'Public state' JSON, or 'Round N of M' in a free-text prompt) or that belongs to another seat
          cannot join a decision; where the prompt shows no round (an empty prompt, the Markdown replay's calls) a group has at
          most 1 + LLM_JSON_RETRIES = 3 calls and ends at its first reply that parses to a legal position, as LLMAgent retries only
          unparseable replies. The group's call whose answer is the applied position is the decision's call (earlier ones were
          format retries; their reasoning is kept above it); with none, its last call is shown with an 'applied instead' note. Its
          reasoning becomes the step's thinking, its prompt the observation and its answer the action. A warning is printed when
          the calls do not split into exactly one group per decision.

The log has no event per decision: every round has one round_result event with the signal the trader saw, the position the game
applied, the price change, the round payoff, the price and the running score. So each round is one decision, and without a call
log the action is rebuilt as set_position with the applied position. Every number shown is taken from the log and checked
(price = previous price + change, payoff = position x change - fee x |position change|, score = running sum, the final score
against summary.json and the game_end event); any disagreement is printed as a warning.

Reference policies are computed here, not copied: backward dynamic programming over (round, regime, position) with the entropic
certainty equivalent for risk aversion tau (tau 0 = risk-neutral expectation), from the logged rounds, move and fee and the
stated regime probabilities (read from the recorded prompt when present). With the exact signal the signal is the true regime,
so these policies can be replayed on the realised regimes and price changes, which do not depend on the trader. Their expected
totals (exact forward propagation) are checked against the config's reference values.

Behavior (the rules' offline analysis): the three behavioral metrics of the rules (trend chasing, disposition effect, signal
use) from the log, next to the optimum's values (exact signal: computed by forward propagation and checked against the rules'
table; noisy70: quoted from the rules), and a risk-attitude fit: the tau interval over which the DP policy makes every logged
decision from the position the trader held. With noisy70 the chance of a move against the signal is
acc x (1 - with) + (1 - acc) x with (0.42), and the references are the config's (Bayes optimum, following the signal +-1)."""
import json, math, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_trading_regime"
DEFAULT_DATE = "2026-11-09"
POSITIONS = (-2, -1, 0, 1, 2)
TAUS = (0.0, 0.2, 0.3, 0.5)                  # the risk attitudes of the rules' reference table
STAY, WITH = 0.8, 0.7                         # defaults: regime stays w.p. 0.8; price moves with the regime w.p. 0.7
MINUS, DOT, MULT = "\u2212", "\u00b7", "\u00d7"
SIDE_KIND = {1: "long", -1: "short", 0: "flat"}
# Flat is "hold", not "muted": grid labels are drawn in white, and on muted gray they fall to about 2:1 contrast in the light theme.
KINDS = {"long": ["Long position", "c1"], "short": ["Short position", "c2"], "flat": ["Flat (no position)", "hold"]}
LLM_JSON_RETRIES = 2                          # LLMAgent retries an unparseable reply twice, then plays a random legal action
MAX_TRIES = 1 + LLM_JSON_RETRIES              # so a decision whose round the prompt does not show has at most 3 calls
TAU_HI, TAU_STEP = 2.0, 0.005                 # the risk-attitude fit scans tau over [0, 2]
WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
# Optimum values of the rules' behavioral metrics with noisy70 (Bayes policy, default parameters, 10 rounds; the rules computed
# them with a script outside the repository, so they are quoted, not recomputed). The exact-signal values are recomputed here.
NOISY_METRICS = {"signal": 0.686, "trend": 0.175, "disp": -0.213}
NOISY_REFS = {"opt": 2.539, "half": -1.368}   # rules' noisy70 reference totals, default parameters and 10 rounds only


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
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
    return re.sub(r"[\u3000-\u303f\uff00-\uffef]", "", s)


# ------------------------------------------------------------------------------------------------ loading
_HIDDEN_TAG = "\u9690\u85cf"          # the Markdown replay's "hidden" event tag
_SEED_RE, _MODEL_RE = r"\u79cd\u5b50 (\d+)", r"\u6a21\u578b ([^\s,\uff0c]+)"   # "seed N", "model X" in the replay header


def _md_bundle(path):
    """A rendered Markdown replay (declarative games): events from the '- **event** (tag): `{json}`' lines, seat rewards from the
    header, and the model calls listed after the events ('### ' blocks), if any."""
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
        events.append(body)
    head = text.split("\n- **")[0]
    rw = re.search(r'(\{"0":[^}]*\})', head)
    over = next((e for e in events if e.get("event") == "game_over"), {})
    summary = {"rewards": json.loads(rw.group(1)) if rw else {}, "variables": over.get("variables", {}), "winners": over.get("winners")}
    seed, model = re.search(_SEED_RE, head), re.search(_MODEL_RE, head)
    calls = []
    for blk in re.split(r"\n(?=### )", text)[1:]:
        ans = re.findall(r"^\*\*[^*\n]+\*\* `(.*)`\s*$", blk, re.M)
        th = re.findall(r"````text\n(.*?)\n````", blk, re.S)
        if ans:
            calls.append({"seat": 0, "prompt": "", "answer": ans[-1].strip(), "reasoning": th[-1] if th else "", "rescued": len(ans) > 1})
    return {"events": events, "summary": summary, "run": {}, "config": {}, "transcript": "", "meta": {"model": model.group(1) if model else ""},
            "md_calls": calls, "md_seed": int(seed.group(1)) if seed else None, "md": True}


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
    """The recording date (YYYY-MM-DD) from meta.json or run.json. When neither has recorded_at (a Markdown replay, a bundle without
    meta.json), DEFAULT_DATE is used and a warning says so, so an old date never reaches the page silently."""
    b = load_source(src)
    got = (b.get("meta") or {}).get("recorded_at") or (b.get("run") or {}).get("recorded_at")
    if not got:
        print(f"warning: no recorded_at in meta.json or run.json; the page will say recorded {DEFAULT_DATE}", file=sys.stderr)
        return DEFAULT_DATE
    return str(got)[:10]


# ------------------------------------------------------------------------------------------------ text helpers
def num(v, d=1):
    """Signed number with a real minus sign: +3.0, -4.0 (U+2212), 0.0."""
    v = round(float(v), 10)
    return ("+" if v > 0 else MINUS if v < 0 else "") + f"{abs(v):.{d}f}"


def pos(p):
    return ("+" if p > 0 else MINUS if p < 0 else "") + str(abs(int(p)))


def apos(p):
    """Position as it reads in a prompt (ASCII sign)."""
    return f"+{p}" if p > 0 else str(p)


def verb(prev, p):
    if p == prev:
        return "stays flat" if p == 0 else f"keeps {pos(p)}"
    if prev == 0:
        return f"opens {'long' if p > 0 else 'short'} {pos(p)}"
    if p == 0:
        return f"closes {pos(prev)} to flat"
    if (p > 0) != (prev > 0):
        return f"reverses to {pos(p)}"
    return f"adds to {pos(p)}" if abs(p) > abs(prev) else f"cuts to {pos(p)}"


def tau_name(t):
    return f"DP \u03c4 {t:g}"


def rounds_text(rs):
    rs = [f"R{r}" for r in rs]
    return "none" if not rs else rs[0] if len(rs) == 1 else ", ".join(rs[:-1]) + " and " + rs[-1]


def rounds_span(rs):
    """Like rounds_text, with runs of three or more consecutive rounds written as 'R2 to R10'."""
    runs = []
    for r in sorted(rs):
        if runs and r == runs[-1][1] + 1:
            runs[-1][1] = r
        else:
            runs.append([r, r])
    items = [x for a, b in runs for x in ([f"R{a} to R{b}"] if b - a >= 2 else [f"R{k}" for k in range(a, b + 1)])]
    return "none" if not items else items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def r4(v):
    return round(float(v) + 0.0, 4) + 0.0


# ------------------------------------------------------------------------------------------------ the game from the log
def read_game(b):
    events = b["events"]
    rules = next((e for e in events if e.get("event") == "rules"), {})
    cfgv = (b.get("config") or {}).get("variables") or {}
    endv = (b.get("summary") or {}).get("variables") or {}
    par = lambda k, dflt=None: rules.get(k, cfgv.get(k, endv.get(k, dflt)))
    rows = sorted((e for e in events if e.get("event") == "round_result"), key=lambda e: e["round"])
    if not rows:
        raise SystemExit("no round_result events in the log")
    G = {"rounds": int(par("rounds", len(rows))), "move": float(par("move", 2)), "fee": float(par("fee", 0.5)),
         "mode": str(par("signal_mode", "exact")), "start": float(rules.get("start_price", cfgv.get("price", 100))), "rows": rows,
         "end": next((e for e in events if e.get("event") == "game_end"), {}), "cfg": cfgv, "summary": b.get("summary") or {},
         "run": b.get("run") or {}}
    prompt = ((G["run"].get("prompt") or {}).get("text")) or ""
    m1 = re.search(r"stays the same with probability ([0-9.]+)", prompt)
    m2 = re.search(r"rises by `move` with probability ([0-9.]+)", prompt)
    G["stay"] = float(m1.group(1)) if m1 else STAY
    G["with"] = float(m2.group(1)) if m2 else WITH
    G["probs_from_prompt"] = bool(m1 and m2)
    return G


def verify(G, pos_, chg, price, pay, score):
    warn = lambda s: print("warning: " + s, file=sys.stderr)
    n = len(pos_) - 1
    if n != G["rounds"]:
        warn(f"{n} round_result events, rules say {G['rounds']} rounds")
    for k in range(1, n + 1):
        if abs(price[k] - (price[k - 1] + chg[k])) > 1e-9:
            warn(f"R{k}: price {price[k]} != {price[k - 1]} + {chg[k]}")
        exp = pos_[k] * chg[k] - G["fee"] * abs(pos_[k] - pos_[k - 1])
        if abs(pay[k] - exp) > 1e-9:
            warn(f"R{k}: payoff {pay[k]} != position x change - fee x |change of position| = {exp}")
        if abs(score[k] - (score[k - 1] + pay[k])) > 1e-9:
            warn(f"R{k}: score {score[k]} != {score[k - 1]} + {pay[k]}")
        if abs(chg[k]) != G["move"]:
            warn(f"R{k}: price change {chg[k]} is not +-move {G['move']}")
    rw = (G["summary"].get("rewards") or (G["run"].get("result") or {}).get("rewards") or {}).get("0")
    if rw is not None and abs(float(rw) - score[-1]) > 1e-9:
        warn(f"final score {score[-1]} != summary reward {rw}")
    if G["end"].get("total_pnl") is not None and abs(float(G["end"]["total_pnl"]) - score[-1]) > 1e-9:
        warn(f"final score {score[-1]} != game_end total_pnl {G['end']['total_pnl']}")
    trades = sum(1 for k in range(1, n + 1) if pos_[k] != pos_[k - 1])
    if G["end"].get("n_trades") is not None and int(G["end"]["n_trades"]) != trades:
        warn(f"{trades} position changes != game_end n_trades {G['end']['n_trades']}")


# ------------------------------------------------------------------------------------------------ dynamic programming
def solve(T, move, fee, tau, stay, with_p):
    """Optimal policy for the exact-signal game: state (round t, regime r = +1 bull / -1 bear, current position p); value of a
    target a = certainty equivalent of (a x price change - fee x |a - p| + continuation value), the change and the next regime
    being independent given r. tau 0: expectation; tau > 0: -(1/tau) ln E exp(-tau x ...). Ties keep the current position, then
    prefer the smaller position. Returns (policy {(t, r, p): a}, start value = mean over the two first regimes of V_1(r, 0))."""
    V = {(r, p): 0.0 for r in (1, -1) for p in POSITIONS}
    pol = {}
    for t in range(T, 0, -1):
        nV = {}
        for r in (1, -1):
            pu = with_p if r == 1 else 1 - with_p
            for p in POSITIONS:
                best = None
                for a in sorted(POSITIONS, key=lambda a: (a != p, abs(a), a)):
                    outs = [(pu * stay, a * move, r), (pu * (1 - stay), a * move, -r),
                            ((1 - pu) * stay, -a * move, r), ((1 - pu) * (1 - stay), -a * move, -r)]
                    xs = [(q, g - fee * abs(a - p) + V[(r2, a)]) for q, g, r2 in outs]
                    if tau == 0:
                        val = sum(q * x for q, x in xs)
                    else:
                        lo = min(x for _, x in xs)
                        val = lo - math.log(sum(q * math.exp(-tau * (x - lo)) for q, x in xs)) / tau
                    if best is None or val > best[0] + 1e-12:
                        best = (val, a)
                nV[(r, p)], pol[(t, r, p)] = best
        V = nV
    return pol, (V[(1, 0)] + V[(-1, 0)]) / 2


def expected_path(pol, T, move, fee, stay, with_p):
    """Exact risk-neutral expected cumulative P&L of a policy after each round (forward propagation of the joint distribution of
    the regime and the position; the first regime is bull or bear with probability 1/2)."""
    dist = {(1, 0): 0.5, (-1, 0): 0.5}
    acc, out = 0.0, [0.0]
    for t in range(1, T + 1):
        nd = {}
        for (r, p), q in dist.items():
            a = pol[(t, r, p)]
            drift = (2 * (with_p if r == 1 else 1 - with_p) - 1) * move
            acc += q * (a * drift - fee * abs(a - p))
            for r2, q2 in ((r, stay), (-r, 1 - stay)):
                nd[(r2, a)] = nd.get((r2, a), 0.0) + q * q2
        dist = nd
        out.append(acc)
    return out


def replay(policy_fn, regimes, chg, fee):
    """Positions and cumulative P&L of a policy on the realised regimes and price changes (index 0 = start)."""
    P, C = [0], [0.0]
    for k in range(1, len(chg)):
        a = policy_fn(k, regimes[k], P[-1])
        C.append(C[-1] + a * chg[k] - fee * abs(a - P[-1]))
        P.append(a)
    return P, C


# ------------------------------------------------------------------------------------------------ behavioral analysis
def _reduces(old, new):
    """The rules' 'reduce': the new position is smaller in the old position's direction (a reversal included)."""
    return old != 0 and new * (1 if old > 0 else -1) < abs(old)


def behavior(pos_, chg, sign):
    """The rules' three behavioral metrics from the log, as (value or None, numerator/denominator pairs):
    trend chasing = P(increase | last change up) - P(increase | last change down), rounds 2 to T;
    disposition = P(reduce | the held position gained in the last round) - P(reduce | it lost), rounds 2 to T;
    signal use = share of non-flat rounds whose sign agrees with the signal (sign = +1 bull / -1 bear signal)."""
    n = len(pos_) - 1
    up = [k for k in range(2, n + 1) if chg[k - 1] > 0]
    dn = [k for k in range(2, n + 1) if chg[k - 1] < 0]
    inc_up, inc_dn = sum(pos_[k] > pos_[k - 1] for k in up), sum(pos_[k] > pos_[k - 1] for k in dn)
    gain = [k for k in range(2, n + 1) if pos_[k - 1] * chg[k - 1] > 0]
    loss = [k for k in range(2, n + 1) if pos_[k - 1] * chg[k - 1] < 0]
    red_g = [k for k in gain if _reduces(pos_[k - 1], pos_[k])]
    red_l = [k for k in loss if _reduces(pos_[k - 1], pos_[k])]
    held = [k for k in range(1, n + 1) if pos_[k]]
    agree = sum(1 for k in held if pos_[k] * sign[k] > 0)
    return {"trend": inc_up / len(up) - inc_dn / len(dn) if up and dn else None, "trend_n": (inc_up, len(up), inc_dn, len(dn)),
            "disp": len(red_g) / len(gain) - len(red_l) / len(loss) if gain and loss else None,
            "disp_n": (len(red_g), len(gain), len(red_l), len(loss)), "reductions": red_g + red_l,
            "signal": agree / len(held) if held else None, "signal_n": (agree, len(held))}


def behavior_reference(pol, T, move, stay, with_p):
    """Exact values of the three metrics for a policy of the exact-signal game: ratios of expected counts, the joint distribution
    of (regime, position held, last price change) being propagated forward round by round (as the rules compute them)."""
    dist = {(1, 0, 0): 0.5, (-1, 0, 0): 0.5}          # (regime of this round, position held before it, last change sign)
    cnt = dict.fromkeys(("up", "inc_up", "dn", "inc_dn", "gain", "red_g", "loss", "red_l", "held", "agree"), 0.0)
    for t in range(1, T + 1):
        nd = {}
        for (r, p, d), q in dist.items():
            a = pol[(t, r, p)]
            if t >= 2:
                cnt["up" if d > 0 else "dn"] += q
                cnt["inc_up" if d > 0 else "inc_dn"] += q * (a > p)
                if p * d:
                    cnt["gain" if p * d > 0 else "loss"] += q
                    cnt["red_g" if p * d > 0 else "red_l"] += q * _reduces(p, a)
            if a:
                cnt["held"] += q
                cnt["agree"] += q * (a * r > 0)
            pu = with_p if r == 1 else 1 - with_p
            for d2, qd in ((1, pu), (-1, 1 - pu)):
                for r2, qr in ((r, stay), (-r, 1 - stay)):
                    nd[(r2, a, d2)] = nd.get((r2, a, d2), 0.0) + q * qd * qr
        dist = nd
    f = lambda a, b: cnt[a] / cnt[b] if cnt[b] > 1e-12 else None
    sub = lambda x, y: None if x is None or y is None else x - y
    return {"trend": sub(f("inc_up", "up"), f("inc_dn", "dn")), "disp": sub(f("red_g", "gain"), f("red_l", "loss")),
            "signal": f("agree", "held")}


def tau_fit(T, move, fee, stay, with_p, regime, pos_):
    """For which risk aversions tau is every logged decision the DP-optimal one, given the position the trader held? tau is scanned
    over [0, TAU_HI] in steps of TAU_STEP and the edges of each run are refined by bisection. Returns (runs, top): top is the largest
    number of matching decisions over the scan (n when some tau explains every decision) and runs the tau intervals reaching it, as
    (lo, hi), hi None when the run reaches TAU_HI."""
    n = len(pos_) - 1
    memo = {}

    def matches(tau):
        if tau not in memo:
            pol, _ = solve(T, move, fee, tau, stay, with_p)
            memo[tau] = sum(1 for k in range(1, n + 1) if pol[(k, regime[k], pos_[k - 1])] == pos_[k])
        return memo[tau]

    grid = [round(i * TAU_STEP, 6) for i in range(int(round(TAU_HI / TAU_STEP)) + 1)]
    cnt = [matches(t) for t in grid]
    top = max(cnt)

    def edge(lo, hi, want):                     # matches(lo) == want != matches(hi): the boundary between them
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if matches(mid) == want else (lo, mid)
        return (lo + hi) / 2

    runs, i = [], 0
    while i < len(grid):
        if cnt[i] != top:
            i += 1
            continue
        j = i
        while j + 1 < len(grid) and cnt[j + 1] == top:
            j += 1
        lo = 0.0 if i == 0 else edge(grid[i], grid[i - 1], top)
        hi = None if j == len(grid) - 1 else edge(grid[j], grid[j + 1], top)
        runs.append((lo, hi))
        i = j + 1
    return runs, top


def tau_text(runs):
    """'below 0.083', 'from 0.15 to 0.27', '0.43 or more' (lower edges rounded up, upper edges down, to 3 decimals)."""
    out = []
    for lo, hi in runs:
        lo_s = f"{math.ceil(lo * 1000 - 1e-9) / 1000:g}"
        hi_s = None if hi is None else f"{math.floor(hi * 1000 + 1e-9) / 1000:g}"
        if lo == 0 and hi is None:
            out.append(f"every τ from 0 to {TAU_HI:g}")
        elif lo == 0:
            out.append(f"τ below {hi_s}")
        elif hi is None:
            out.append(f"τ {lo_s} or more")
        else:
            out.append(f"τ from {lo_s} to {hi_s}")
    return " or ".join(out)


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
            return int(p) if p == int(p) and int(p) in POSITIONS else None
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def prompt_state(prompt):
    """(round, public state dict, public player state dict) parsed from a DECIDE prompt; None parts when not shown."""
    def js(label):
        m = re.search(label + r":\s*(\{.*\})\s*$", prompt or "", re.M)
        try:
            return json.loads(m.group(1)) if m else None
        except ValueError:
            return None
    ps, pp = js("Public state"), js("Public player state")
    return (ps or {}).get("round"), ps, pp


def call_round(prompt):
    """The round a call's prompt is about: the generic prompt's 'Public state' JSON, else a free-text 'Round N of M', else a
    'Round N' when the prompt names exactly one round. None when the prompt is empty or shows no round."""
    r = prompt_state(prompt)[0]
    if r is not None:
        try:
            return int(r)
        except (TypeError, ValueError):
            return None
    m = re.search(r"\bRound (\d+) of \d+\b", prompt or "")
    if m:
        return int(m.group(1))
    found = set(re.findall(r"\bRound (\d+)\b", prompt or ""))
    return int(found.pop()) if len(found) == 1 else None


# Costs of the call-to-decision alignment (attach_calls). A decision's calls are consecutive: zero or more replies that do not
# parse to a legal position (retried), then either a legal one, which the game applies, or, after MAX_TRIES calls, a fallback.
C_SKIP_CALL, C_NO_CALL = 3.0, 3.0             # a call tied to no decision; a decision without a call
C_LEGAL_RETRIED = 4.0                          # a legal reply followed by another call of the same decision (LLMAgent never retries one)
C_WRONG_LEGAL = 4.0                            # the last reply is legal but the game applied another position
C_FALLBACK, C_SHORT_FALLBACK = 0.5, 1.5        # no legal reply: after the full MAX_TRIES calls, or after fewer


def attach_calls(steps, calls):
    """Tie calls to decisions (see the module docstring). The calls are split, in time order, into one consecutive group per
    decision by a minimum-cost alignment: a call of another seat, or whose prompt shows another round, cannot belong to a decision;
    a group whose round is not shown in every prompt has at most MAX_TRIES calls and should end at its first legal reply, which must
    be the applied position. So prompts without a round (an empty prompt, the Markdown replay's calls, a custom prompt) cannot pull
    a later round's call into a decision. Returns the number of decisions that got a call; prints a warning when the calls do not
    split into exactly one group per decision or a group's answer is not the applied position."""
    m, n = len(calls), len(steps)
    rnd = [call_round(c.get("prompt")) for c in calls]
    ans = [answer_position(_norm_answer(c.get("answer"))) for c in calls]
    seat_ok = [c.get("seat") in (None, 0) for c in calls]
    INF = float("inf")

    def groups_from(i, st):
        """(size, cost) of every feasible group of calls i, i+1, ... for decision st."""
        out = []
        for g in range(1, m - i + 1):
            k = i + g - 1
            if not seat_ok[k] or (rnd[k] is not None and rnd[k] != st["round"]):
                break
            if g > MAX_TRIES and any(rnd[j] is None for j in range(i, i + g)):
                break
            cost = sum(C_LEGAL_RETRIED for j in range(i, k) if ans[j] is not None)
            if ans[k] == st["position"]:
                pass
            elif ans[k] is not None:
                cost += C_WRONG_LEGAL
            else:
                cost += C_FALLBACK if g >= MAX_TRIES else C_SHORT_FALLBACK
            out.append((g, cost))
        return out

    best = [[INF] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]
    best[0][0] = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            b0 = best[i][j]
            if b0 == INF:
                continue
            moves = []
            if j < n:
                moves += [((i + g, j + 1), b0 + c, ("group", g)) for g, c in groups_from(i, steps[j])]
                # a hair cheaper later on: when a call fits two decisions equally well (same applied position, no round in the
                # prompt), the decision left without a call is the later one, as when the log stops early ...
                moves.append(((i, j + 1), b0 + C_NO_CALL - 1e-3 * j / max(1, n), ("none", 0)))
            if i < m:                                    # ... and the call left over is the later one
                moves.append(((i + 1, j), b0 + C_SKIP_CALL - 1e-3 * i / max(1, m), ("skip", 0)))
            for (i2, j2), c, how in moves:
                if c < best[i2][j2] - 1e-12:
                    best[i2][j2], back[i2][j2] = c, (i, j, how)
    plan, skipped, i, j = [None] * n, [], m, n
    while (i, j) != (0, 0):
        pi, pj, (how, g) = back[i][j]
        if how == "group":
            plan[pj] = list(range(pi, pi + g))
        elif how == "skip":
            skipped.append(pi)
        i, j = pi, pj

    warn = lambda s: print("warning: " + s, file=sys.stderr)
    missing = [st["round"] for st, grp in zip(steps, plan) if not grp]
    if skipped or missing:
        warn(f"the {m} calls do not split into exactly one group per decision ({n} decisions): "
             + "; ".join(x for x in (f"calls {sorted(skipped)} tied to no decision" if skipped else "",
                                     f"no call for {rounds_text(missing)}" if missing else "") if x))
    tied = 0
    for st, grp in zip(steps, plan):
        if not grp:
            continue
        hit = next((k for k in grp if ans[k] == st["position"]), None)
        use = hit if hit is not None else grp[-1]
        if hit is None and ans[use] is not None:
            warn(f"R{st['round']}: the tied call answers {apos(ans[use])} but the game applied {apos(st['position'])}")
        parts = []
        for k in grp:
            c = calls[k]
            if k < use:
                parts.append(f"[Earlier attempt, answer: {c['answer'][:300]}]\n{c.get('reasoning') or '(no reasoning returned)'}")
            elif k == use:
                parts.append(("[Final attempt]\n" if k > grp[0] else "") + (c.get("reasoning") or ("(no reasoning returned)" if k > grp[0] else "")))
                if hit is None:
                    parts.append(f"[The game applied position {apos(st['position'])} instead of this answer: {c['answer'][:300]}]")
                if c.get("rescued"):
                    parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
            else:
                parts.append(f"[A later call for the same decision, answer: {c['answer'][:300]}]\n{c.get('reasoning') or '(no reasoning returned)'}")
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        c = calls[use]
        if c.get("prompt"):
            st["state"] = en(common.state_part(c["prompt"]))
            st["state_from_prompt"] = True
            st["private"] = private_info(c["prompt"])
            check_prompt(c["prompt"], st)
        a = _norm_answer(c.get("answer"))
        if hit is not None and isinstance(a, dict):
            st["act"] = en(json.dumps(a, ensure_ascii=False))
            st["act_from_call"] = True
        tied += 1
    return tied


def private_info(prompt):
    """The 'Your private info: {...}' JSON of a generic prompt; None when the prompt has none (or it does not parse)."""
    m = re.search(r"Your private info:\s*(\{.*\})\s*$", prompt or "", re.M)
    try:
        v = json.loads(m.group(1)) if m else None
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


def check_prompt(prompt, st):
    """The prompt's numbers must agree with the state rebuilt from the log; disagreements are printed."""
    rnd, ps, pp = prompt_state(prompt)
    if ps is None:
        return
    want = {"round": st["round"], "price": st["seen"]["price"], "regime_signal": st["seen"]["signal"]}
    for k, v in want.items():
        if k in ps and ps[k] != v and not (isinstance(v, float) and abs(float(ps[k]) - v) < 1e-9):
            print(f"warning: R{st['round']}: prompt {k} {ps[k]!r} != log {v!r}", file=sys.stderr)
    me = (pp or {}).get("0") or {}
    for k in ("position", "score"):
        if k in me and abs(float(me[k]) - st["seen"][k]) > 1e-9:
            print(f"warning: R{st['round']}: prompt {k} {me[k]!r} != log {st['seen'][k]!r}", file=sys.stderr)


# ------------------------------------------------------------------------------------------------ build
# The rules' reference table (start certainty equivalents), valid for the default parameters only: 10 rounds, move 2, fee 0.5,
# stay 0.8, move with the regime 0.7. Used only to check the solver, never shown unless the solver reproduces it.
RULES_CE = {0.0: 11.40, 0.2: 1.91, 0.3: 0.039, 0.5: 0.0}
RULES_CE_TOL = {0.0: 0.005, 0.2: 0.005, 0.3: 0.0005, 0.5: 0.0005}


def observation(k, T, G, rows, pos_, score):
    """What the trader saw before round k, rebuilt from the log: the fields of the DECIDE prompt (signal, price, last three price
    changes, position, score) and the earlier rounds, which the prompt lists among its recent public events. Numbers as logged."""
    e = rows[k - 1]
    prev = rows[k - 2] if k > 1 else None
    price = prev["price"] if prev else G["start_raw"]
    changes = [(j, rows[j - 1]["price_change"]) for j in range(max(1, k - 3), k)]
    lines = [f"Round {k} of {T} (DECIDE): choose your target position, -2 to +2.",
             f"Regime signal: {e.get('signal_seen')}" + (" (signal mode exact: the signal is the true regime)" if G["mode"] == "exact"
                                                         else f" (signal mode {G['mode']})"),
             f"Price: {json.dumps(price)}",
             "Last three price changes: " + (", ".join(json.dumps(c) for _, c in changes) + f" (rounds {changes[0][0]} to {changes[-1][0]})"
                                             if len(changes) > 1 else f"{json.dumps(changes[0][1])} (round 1)" if changes else "none yet"),
             f"Your position: {pos_[k - 1]}",
             f"Your score (P&L so far): {json.dumps(prev['score']) if prev else 0}",
             f"Fee {G['fee']:g} per unit of position changed; price step {G['move']:g}."]
    if k > 1:
        lines += ["", "Earlier rounds (recent public events in the prompt):",
                  "  round  signal  position  change  payoff  price  score"]
        for j in range(1, k):
            r = rows[j - 1]
            lines.append(f"  R{j:<5} {r.get('signal_seen', ''):<7} {json.dumps(r['position']):>8} {json.dumps(r['price_change']):>7} "
                         f"{json.dumps(r['payoff']):>7} {json.dumps(r['price']):>6} {json.dumps(r['score']):>6}")
    return "\n".join(lines)


def build(bundle_dir, calls=None, date=None):
    """date: the recording date when the caller already has it (else read with recorded_date, DEFAULT_DATE for a loaded dict)."""
    b = load_source(bundle_dir)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    if not calls:
        calls = b.get("md_calls") or []
    G = read_game(b)
    rules = next((e for e in b["events"] if e.get("event") == "rules"), {})
    G["start_raw"] = rules.get("start_price", G["start"])
    rows, fee, move, T = G["rows"], G["fee"], G["move"], G["rounds"]
    n = len(rows)
    T = max(T, n)                                     # (verify warns when the log has more rounds than the rules say)
    exact = G["mode"] == "exact"
    reg_word = "regime" if exact else "signal"
    mq = re.fullmatch(r"noisy(\d+)", G["mode"])
    acc = 1.0 if exact else int(mq.group(1)) / 100 if mq else None     # P(the signal is the true regime)
    if acc is None:
        print(f"warning: unknown signal mode {G['mode']!r}: its accuracy is not known", file=sys.stderr)
    # P(the price moves against the signal) = P(signal right) P(move against the regime) + P(signal wrong) P(move with the regime)
    p_against = None if acc is None else acc * (1 - G["with"]) + (1 - acc) * G["with"]
    defaults = (move, fee, G["stay"], G["with"]) == (2, 0.5, 0.8, 0.7)

    # ---- per round, straight from the round_result events (index 0 = the start)
    pos_ = [0] + [int(e["position"]) for e in rows]
    chg = [0.0] + [float(e["price_change"]) for e in rows]
    price = [G["start"]] + [float(e["price"]) for e in rows]
    pay = [0.0] + [float(e["payoff"]) for e in rows]
    score = [0.0] + [float(e["score"]) for e in rows]
    sig = [None] + [str(e.get("signal_seen", "")) for e in rows]
    regime = [0] + [1 if s == "bull" else -1 if s == "bear" else 0 for s in sig[1:]]
    verify(G, pos_, chg, price, pay, score)
    against = [k for k in range(1, n + 1) if regime[k] and chg[k] * regime[k] < 0]
    trades = [k for k in range(1, n + 1) if pos_[k] != pos_[k - 1]]
    fees = sum(fee * abs(pos_[k] - pos_[k - 1]) for k in range(1, n + 1))
    gross = sum(pos_[k] * chg[k] for k in range(1, n + 1))
    beh = behavior(pos_, chg, regime)

    # ---- reference policies
    refs, rn_exp, rules_ok, cfg_ok, beh_ref, fit_runs, fit_top, cexp = [], None, False, False, None, [], 0, None
    ref_opt = ref_half = None
    if exact:
        for tau in TAUS:
            pol, start = solve(T, move, fee, tau, G["stay"], G["with"])
            P, C = replay(lambda k, r, p, pol=pol: pol[(k, r, p)], regime, chg, fee)
            ex = expected_path(pol, T, move, fee, G["stay"], G["with"])
            refs.append({"tau": tau, "name": tau_name(tau), "P": P, "C": C, "exp": ex, "start": start, "pol": pol})
        rn_exp = refs[0]["exp"]
        print("DP start values: " + ", ".join(f"tau {r['tau']:g}: {r['start']:.4f} (risk-neutral expected total {r['exp'][-1]:.4f})"
                                             for r in refs), file=sys.stderr)
        if (T, move, fee, G["stay"], G["with"]) == (10, 2, 0.5, 0.8, 0.7):
            bad = [r for r in refs if abs(r["start"] - RULES_CE[r["tau"]]) > RULES_CE_TOL[r["tau"]]]
            for r in bad:
                print(f"warning: DP tau {r['tau']:g} start value {r['start']:.4f} != rules table {RULES_CE[r['tau']]}", file=sys.stderr)
            rules_ok = not bad
        ref_opt = G["cfg"].get("ref_optimal_risk_neutral_expected")
        if ref_opt is not None:
            cfg_ok = abs(rn_exp[-1] - float(ref_opt)) < 1e-6
            if not cfg_ok:
                print(f"warning: DP risk-neutral expected total {rn_exp[-1]:.4f} != config reference {ref_opt}", file=sys.stderr)
        half = {(t, r, p): r for t in range(1, T + 1) for r in (1, -1) for p in POSITIONS}
        h_exp = expected_path(half, T, move, fee, G["stay"], G["with"])[-1]
        ref_half = G["cfg"].get("ref_follow_signal_half_expected")
        if ref_half is not None and abs(h_exp - float(ref_half)) > 1e-6:
            print(f"warning: expected total of +-1 with the regime {h_exp:.4f} != config reference {ref_half}", file=sys.stderr)
        # the rules' behavioral metrics of the risk-neutral optimum, computed exactly and checked against the rules' table
        beh_ref = behavior_reference(refs[0]["pol"], T, move, G["stay"], G["with"])
        if defaults:
            for key, want in (("trend", -0.08), ("disp", 0.0), ("signal", 1.0)):
                if beh_ref[key] is None or abs(beh_ref[key] - want) > 5e-4:
                    print(f"warning: optimum's {key} metric {beh_ref[key]} != rules table {want}", file=sys.stderr)
        if all(regime[1:]):
            fit_runs, fit_top = tau_fit(T, move, fee, G["stay"], G["with"], regime, pos_)
            print(f"tau fit: {fit_top} of {n} decisions optimal for " + ", ".join(f"[{lo:.4f}, {'open' if hi is None else f'{hi:.4f}'}]"
                                                                               for lo, hi in fit_runs), file=sys.stderr)
        # what the logged positions were expected to earn given the regimes that occurred (the price draws being the only luck)
        cexp = sum(pos_[k] * (2 * G["with"] - 1) * move * regime[k] - fee * abs(pos_[k] - pos_[k - 1]) for k in range(1, n + 1))
    else:                                             # noisy signal: the regime is never seen; compare with following the signal
        noisy_default = defaults and T == 10 and acc == 0.7
        ref_opt = G["cfg"].get("ref_optimal_risk_neutral_expected", NOISY_REFS["opt"] if noisy_default else None)
        ref_half = G["cfg"].get("ref_follow_signal_half_expected", NOISY_REFS["half"] if noisy_default else None)
        beh_ref = dict(NOISY_METRICS) if noisy_default else None
        P, C = replay(lambda k, r, p: r, regime, chg, fee)
        refs.append({"tau": None, "name": "Signal ±1", "P": P, "C": C, "exp": None, "start": None})
    opt = refs[0] if exact else None
    model_is_opt = bool(opt) and all(opt["P"][k] == pos_[k] for k in range(1, n + 1))
    # decisions equal to the risk-neutral optimum's from the position the model actually held
    same = sum(1 for k in range(1, n + 1) if opt and regime[k] and opt["pol"][(k, regime[k], pos_[k - 1])] == pos_[k])

    # ---- steps: one decision per round, made before that round's price move
    first_prompt = G["run"].get("prompt") or {}
    fp_text = first_prompt.get("text") or ""
    fp_round = prompt_state(fp_text)[0] if first_prompt.get("phase", "DECIDE") == "DECIDE" else None
    note = ("[Rebuilt from the game log; this recording kept the full prompt of round 1 only.]" if fp_round == 1 else
            "[Rebuilt from the game log; this recording kept no prompts.]")
    steps = []
    for k in range(1, n + 1):
        p, prev = pos_[k], pos_[k - 1]
        st = {"seat": 0, "x": k, "round": k, "position": p,
              "seen": {"price": price[k - 1], "signal": sig[k], "position": prev, "score": score[k - 1]},
              "head": f"Round {k} {DOT} {sig[k]} {DOT} {verb(prev, p)}", "kind": SIDE_KIND[(p > 0) - (p < 0)],
              "act": json.dumps({"action_id": "set_position", "payload": {"position": p}}),
              "state": observation(k, T, G, rows, pos_, score) + "\n\n" + note, "thinking": ""}
        if k == fp_round:
            st["state"] = en(common.state_part(fp_text))
            st["state_from_run"] = True
            st["private"] = private_info(fp_text)
            check_prompt(fp_text, st)
        steps.append(st)
    tied = attach_calls(steps, calls) if calls else 0
    # where each step's observation and action come from, read before the bookkeeping keys are dropped (for the foot)
    obs_call = [st["round"] for st in steps if st.get("state_from_prompt")]
    obs_run = [st["round"] for st in steps if st.get("state_from_run") and not st.get("state_from_prompt")]
    obs_reb = [st["round"] for st in steps if not st.get("state_from_prompt") and not st.get("state_from_run")]
    act_call = [st["round"] for st in steps if st.get("act_from_call")]
    act_reb = [st["round"] for st in steps if not st.get("act_from_call")]
    if obs_call:                                      # the call log kept prompts: the round-1-only note no longer holds
        for st in steps:
            if st["round"] in obs_reb and st["state"].endswith(note):
                st["state"] = st["state"][:-len(note)] + "[Rebuilt from the game log; no model call with a prompt is tied to this round.]"
    # the generic prompt's private info carries a 'pending' field the rules do not list; check what it holds
    pend = [(st["round"], st["private"]["pending"]) for st in steps if isinstance(st.get("private"), dict) and "pending" in st["private"]]
    pend_bad = [r for r, v in pend if not (isinstance(v, (int, float)) and not isinstance(v, bool) and v == pos_[r - 1])]
    for r in pend_bad:
        print(f"warning: R{r}: prompt private info pending {dict(pend)[r]!r} != position carried into the round {pos_[r - 1]}",
              file=sys.stderr)
    for st in steps:
        for key in ("round", "position", "seen", "state_from_prompt", "state_from_run", "act_from_call", "private"):
            st.pop(key, None)

    # ---- chart 1: price, regime strip, trades (at the price seen when deciding), moves against the regime
    xn = n + 1
    labels = ["Start"] + [f"R{k}" for k in range(1, n + 1)]
    marks = [{"x": k - 1, "y": r4(price[k - 1]), "kind": "buy" if pos_[k] > pos_[k - 1] else "sell"} for k in trades]

    def trade_text(k):
        prev, p, at = pos_[k - 1], pos_[k], f"{price[k - 1]:g}"
        if prev == 0:
            what = "buy" if p > 0 else "short sale"
        elif p == 0:
            what = "close to flat"
        elif (p > 0) != (prev > 0):
            what = f"reversal to {'long' if p > 0 else 'short'}"
        else:
            what = "buy" if p > prev else "sale"
        return f"{what} at {at} before R{k}"
    tt = [trade_text(k) for k in trades]
    tt = ", ".join(tt) if len(tt) <= 4 else ", ".join(tt[:3]) + f" and {len(tt) - 3} more"
    ag = [price[k] if k in against else None for k in range(xn)]
    price_lines = [{"name": "Price", "color": "ink", "v": [r4(v) for v in price], "dots": True}]
    if against:
        price_lines.append({"name": f"Moved against the {reg_word}", "color": "c4", "v": ag, "dots": True, "noline": True})
    c_price = {"title": "Price and the hidden regime" if exact else "Price and the regime signal",
               "sub": (f"Price after each round's move ({G['start']:g} at the start). The strip shows the "
                       + ("regime in force during the round; with the exact signal the trader is told it before deciding. " if exact else
                          "signal the trader saw before the round. ")
                       + f"Yellow dots: rounds in which the price moved against the {reg_word}"
                       + (f", which happens with probability {p_against:.2g}" if p_against is not None else "")
                       + (f" (signal right {acc:.0%} {MULT} move against the regime {1 - G['with']:.0%}, plus signal wrong "
                          f"{1 - acc:.0%} {MULT} move with the regime {G['with']:.0%})" if not exact and p_against is not None else "") + ". "
                       + (f"Triangles: where the trader changed position, at the price it saw just before that round's move ({tt})."
                          if trades else "The trader never changed position, so there are no trade triangles.")),
               "height": 250, "fmt": "int", "lines": price_lines,
               "band": {"v": regime, "pos": "Bull regime" if exact else "Bull signal", "neg": "Bear regime" if exact else "Bear signal"},
               "markers": marks}

    # ---- the rules' behavioral metrics, next to the optimum's values
    def val_s(v, signed=True):
        return "n/a" if v is None else num(v, 2) if signed else f"{v:.2f}"

    def ref_s(v, signed=True):
        if v is None:
            return "n/a"
        r = round(v, 3) + 0.0
        return ("+" if signed and r > 0 else MINUS if r < 0 else "") + f"{abs(r):g}"
    br = beh_ref or {}
    opt_word = "risk-neutral optimum" if exact else "Bayes optimum"
    trend_note = f"; it never increased its position in rounds 2 to {n}" if beh["trend"] is not None and not (beh["trend_n"][0] or beh["trend_n"][2]) else ""
    met = (f"Behavioral metrics of the rules" + (f" ({opt_word} in brackets)" if beh_ref else "") + ": signal use "
           + val_s(beh["signal"], False) + (f" ({ref_s(br.get('signal'), False)})" if beh_ref else "")
           + ", trend chasing " + val_s(beh["trend"]) + (f" ({ref_s(br.get('trend'))}{trend_note})" if beh_ref else
                                                          f" ({trend_note[2:]})" if trend_note else "")
           + ", disposition effect " + val_s(beh["disp"]) + (f" ({ref_s(br.get('disp'))})" if beh_ref else "") + ".")
    reds = sorted(beh["reductions"])

    def red_text(k):
        prev, p = pos_[k - 1], pos_[k]
        what = "close to flat" if p == 0 else "reversal" if (p > 0) != (prev > 0) else "cut"
        flip = f" when the {reg_word} flipped" if k > 1 and sig[k] != sig[k - 1] else ""
        return f"the R{k} {what}{flip}", pos_[k - 1] * chg[k - 1] > 0
    if len(reds) == 1:
        txt, won = red_text(reds[0])
        met += (f" Its only reduction, {txt}, came after its position had {'gained' if won else 'lost'} in R{reds[0] - 1}, "
                f"so it counts as selling a {'winner' if won else 'loser'}.")
    elif reds:
        met += " Its reductions: " + "; ".join(f"{t} (after a {'gain' if w else 'loss'})" for t, w in map(red_text, reds)) + "."
    if fit_runs and fit_top == n:
        fit_line = f"Every decision is DP-optimal only for {tau_text(fit_runs)}."
    elif fit_runs:
        fit_line = f"No τ from 0 to {TAU_HI:g} makes all {n} decisions optimal; {tau_text(fit_runs)} makes {fit_top}."
    else:
        fit_line = ""

    # ---- chart 2: positions of the model and of the reference policies
    def cell(p, tip):
        return {"c": KINDS[SIDE_KIND[(p > 0) - (p < 0)]][1], "t": pos(p), "tip": tip, "o": 1}

    model_cells = [None] + [cell(pos_[k], f"{verb(pos_[k - 1], pos_[k])} {DOT} {sig[k]} {DOT} payoff {num(pay[k])}") for k in range(1, xn)]
    ref_cells = [[None] + [cell(r["P"][k], f"{pos(r['P'][k])} {DOT} payoff {num(r['C'][k] - r['C'][k - 1])}") for k in range(1, xn)]
                 for r in refs]
    c_grid = {"title": "Position each round: the model and the optimal policies" if exact else "Position each round",
              "sub": (("DP τ x is the dynamic-programming optimal policy for risk aversion τ (entropic certainty equivalent as "
                       "in the RQE paper; τ 0 is risk-neutral), replayed on the same regimes. Labels give the position in units; "
                       "hover a column for each round's payoff. " + (fit_line + " " if fit_line else ""))
                      if exact else
                      ("Signal ±1 holds 1 unit in the direction of the signal every round"
                       + (f", which expects {num(ref_half, 2)} after fees (the rules' reference): a losing baseline, not an optimum"
                          if ref_half is not None else "") + ". Labels give the position in units. ")) + met,
              "height": 160, "fmt": "int",
              "grid": {"rows": ["Model"] + [r["name"] for r in refs], "cells": [model_cells] + ref_cells,
                       "legend": [["Long", "c1"], ["Short", "c2"], ["Flat", KINDS["flat"][1]]]}}

    # ---- chart 3: cumulative P&L on this path
    cum_lines = [{"name": "Model (= DP τ 0)" if model_is_opt else "Model", "color": "ink", "v": [r4(v) for v in score],
                  "area": True, "end": True}]
    if exact:
        if not model_is_opt:
            cum_lines.append({"name": refs[0]["name"], "color": "c1", "v": [r4(v) for v in refs[0]["C"]], "end": True})
        risky = [r for r in refs[1:] if any(r["P"][k] for k in range(1, xn))]
        best = max(risky, key=lambda r: r["tau"]) if risky else None   # the most risk-averse policy that still trades
        if best:
            cum_lines.append({"name": best["name"], "color": "c3", "v": [r4(v) for v in best["C"]], "end": True})
        cum_lines.append({"name": "Expected, DP τ 0", "color": "hold", "dash": True, "v": [r4(v) for v in rn_exp], "end": True})
        others = []
        for r in refs[1:]:
            if r is best:
                continue
            if not any(r["P"][k] for k in range(1, xn)):
                others.append(f"{r['name']} never trades (the zero line)")
            else:
                ratio = [r["P"][k] * 2 == pos_[k] for k in range(1, xn)]
                others.append(f"{r['name']} ends at {num(r['C'][-1])}" + (", half the model's, as it holds half the model's position "
                                                                          "every round" if all(ratio) else ""))
        ahead = ""
        if best and best["C"][-1] > score[-1] + 1e-9:
            BP = best["P"]
            k0 = next((k for k in range(1, xn) if pos_[k - 1] * pos_[k] < 0 and BP[k] == 0 and BP[k - 1] != 0), None)
            run = 0
            while k0 is not None and k0 + run < xn and chg[k0 + run] * pos_[k0] < 0:
                run += 1
            vs = f"({num(best['C'][-1])} against {num(score[-1])})"
            if k0 is not None and run:
                mv = "rise" if pos_[k0] < 0 else "fall"
                ahead = (f" {best['name']} ends ahead {vs} because it closed its {pos(BP[k0 - 1])} to flat instead of reversing in "
                         f"R{k0}, just before {WORDS[run] if run < len(WORDS) else run} {mv}{'s' if run > 1 else ''}.")
            elif all(abs(BP[k]) <= abs(pos_[k]) for k in range(1, xn)) and against:
                ahead = f" {best['name']} ends ahead {vs}: its smaller position lost less on the {len(against)} moves against the regime."
            else:
                ahead = f" {best['name']} ends ahead {vs}."
        exp_txt = ", ".join(f"τ {r['tau']:g} {r['exp'][-1]:.2f}" for r in refs)
        c_cum = {"title": "Cumulative P&L on this price path",
                 "sub": ("Each policy replayed on the realised regimes and price moves, which do not depend on the trader; "
                         + "; ".join(others) + "." + ahead + " Dashed: what the risk-neutral optimum expects before the game. "
                         f"Expected totals over all price paths: {exp_txt}."),
                 "height": 240, "fmt": "price", "zero": True, "lines": cum_lines}
    else:
        cum_lines.append({"name": "Follow signal ±1", "color": "c3", "v": [r4(v) for v in refs[0]["C"]], "end": True})
        c_cum = {"title": "Cumulative P&L",
                 "sub": ("The model against holding 1 unit in the direction of the signal every round, on the same price moves"
                         + (f"; that baseline expects {num(ref_half, 2)} over all price paths" if ref_half is not None else "")
                         + (f", the Bayes optimum {num(ref_opt, 2)}" if ref_opt is not None else "") + " (the rules' references)."
                         if ref_half is not None or ref_opt is not None else
                         "The model against holding 1 unit in the direction of the signal every round, on the same price moves."),
                 "height": 240, "fmt": "price", "zero": True, "lines": cum_lines}

    # ---- round-by-round feed
    feed = []
    for k in range(1, n + 1):
        p, prev = pos_[k], pos_[k - 1]
        f = fee * abs(p - prev)
        mv = f"price {num(chg[k], 0)} to {price[k]:g}" + (f", against the {reg_word}" if k in against else "")
        txt = f"Signal {sig[k]}; {verb(prev, p)}" + (f" (fee {f:.1f})" if f else "") + f"; {mv}; payoff {num(pay[k])}; P&L {num(score[k])}"
        feed.append({"x": k, "step": k - 1, "who": "Trader", "kind": steps[k - 1]["kind"], "text": txt})

    # ---- title, tiles and text
    model_name = ((b.get("meta") or {}).get("model") or G["run"].get("model") or "").split("/")[-1] or "the model"
    k_ag = len(against)
    e_ag = None if p_against is None else round(p_against * n, 6)
    mv_ag = f"{WORDS[k_ag] if k_ag < len(WORDS) else k_ag} move{'s' if k_ag != 1 else ''} against the {reg_word}"
    if exact:
        res = f"{num(score[-1])} ({num(rn_exp[-1])} expected)"
        if model_is_opt:
            full = all(pos_[k] == max(POSITIONS) * regime[k] for k in range(1, n + 1))
            title = (("Followed the regime at full size, the risk-neutral optimum, in all" if full else
                      "Played the risk-neutral optimum in all") + f" {n} rounds; "
                     + (f"no move against the regime: {res}" if k_ag == 0 else
                        f"{mv_ag} left {res}" if k_ag > e_ag else
                        f"only {mv_ag}: {res}" if k_ag < e_ag else f"{mv_ag}, as many as expected: {res}"))
        else:
            if fit_runs and fit_top == n:               # a risk-averse optimum explains every decision: lead with it
                title = (f"Stayed flat in all {n} rounds, the DP optimum only for {tau_text(fit_runs)}" if not any(pos_[1:]) else
                         f"Played the DP optimum for {tau_text(fit_runs)} in all {n} rounds")
            else:
                title = (f"Stayed flat in all {n} rounds" if not any(pos_[1:]) else
                         f"Matched the risk-neutral optimum in {same} of {n} decisions")
            title += (f"; ended at {num(score[-1])}, the risk-neutral optimum made {num(opt['C'][-1])} on this path "
                      f"({num(rn_exp[-1])} expected)")
        s1 = (f"Same positions as the optimum; {num(cexp)} was expected given these regimes, {num(rn_exp[-1])} before the game"
              if model_is_opt else
              f"{num(cexp)} expected for its positions given these regimes; the optimum expects {num(rn_exp[-1])} before the game")
    else:
        title = (f"Traded {n} rounds on a signal that is right {acc:.0%} of the time and ended at {num(score[-1])}" if acc is not None
                 else f"Traded {n} rounds on a {G['mode']} signal and ended at {num(score[-1])}")
        if ref_opt is not None:
            title += f"; the Bayes optimum expects {num(ref_opt, 2)}"
        s1 = (f"Bayes optimum expects {num(ref_opt, 2)}" if ref_opt is not None else "Sum of the round payoffs")
    tiles = [{"k": "Total P&L", "v": num(score[-1]), "s": s1, "spark": [r4(v) for v in score]}]
    if exact:
        if fit_runs and fit_top == n:
            lo0 = fit_runs[0][0] == 0 and len(fit_runs) == 1
            fit_note = (f"Optimal only for {tau_text(fit_runs)}: the model acted risk-neutral" if lo0 and fit_runs[0][1] is not None else
                        f"Optimal for {tau_text(fit_runs)}" + ("" if lo0 else ": the model acted risk-averse"))
        elif fit_runs:
            fit_note = f"No τ makes all {n} optimal; {tau_text(fit_runs)} fits {fit_top}"
        else:
            fit_note = "Decisions equal to the risk-neutral DP policy's, from the position held"
        tiles.append({"k": "Same as the optimum", "v": f"{same} of {n} rounds", "s": fit_note})
    tiles.append({"k": f"Moves against the {reg_word}", "v": f"{k_ag} of {n}",
                  "s": rounds_text(against) + ("" if p_against is None else f"; {e_ag:g} expected at {p_against:.0%} a round")})
    tiles.append({"k": "Trades and fees", "v": f"{len(trades)} trade{'s' if len(trades) != 1 else ''}",
                  "s": ("Never changed position" if not trades else
                        f"Fees {fees:.1f}: " + "; ".join(f"{pos(pos_[k - 1])} to {pos(pos_[k])} in R{k}" for k in trades) if len(trades) <= 3
                        else f"Fees {fees:.1f} at {fee:g} per unit changed, in {rounds_text(trades)}")})

    kinds = {k: v for k, v in KINDS.items() if any(s["kind"] == k for s in steps)}
    lede = (f"A single trader picks a position from {MINUS}2 (short) to +2 (long) every round while the market switches between a bull "
            f"and a bear regime, staying in the same regime with probability {G['stay']:g}. In a bull regime the price rises {move:g} "
            f"with probability {G['with']:g} and otherwise falls {move:g} (a bear regime is the reverse), and every unit of position "
            f"changed costs a fee of {fee:g}. The score is the total P&L over {T} rounds; "
            + ("here the trader is told the true regime before each round." if exact else
               f"here the trader sees a signal that is right {acc:.0%} of the time." if acc is not None else
               f"here the trader sees a {G['mode']} signal."))
    seed = G["run"].get("seed", b.get("md_seed"))
    if date is None:
        date = DEFAULT_DATE if isinstance(bundle_dir, dict) else recorded_date(bundle_dir)
    # where the steps' observations and actions come from
    if not obs_call and not act_call:                 # no call tied (or no call log): everything but a recorded round-1 prompt is rebuilt
        prov = ("The log has no event per decision, so each action is rebuilt as set_position with the applied position"
                + (", and the observations from round 2 on are rebuilt from earlier events (round 1 shows the recorded prompt)"
                   if fp_round == 1 else "") + ". ")
    elif len(obs_call) == n and len(act_call) == n:
        prov = ("The game log has no event per decision; each step's observation and action are the prompt the model received and its "
                "answer, from the model-call log, and every answer is the position the game applied. ")
    else:
        acts = ("each action is the model's recorded answer, the position the game applied" if not act_reb else
                f"the action is the model's recorded answer in {rounds_span(act_call)} and is rebuilt as set_position with the applied "
                f"position in {rounds_span(act_reb)}" if act_call else
                "each action is rebuilt as set_position with the applied position")
        if len(obs_call) == n:
            obs = "each observation is the prompt the model received, from the model-call log"
        elif not obs_call and not obs_run:
            obs = "each observation is rebuilt from earlier events"
        else:
            parts = ([f"the prompt the model received (model-call log) in {rounds_span(obs_call)}"] if obs_call else []) \
                + ([f"the recorded prompt in {rounds_span(obs_run)}"] if obs_run else []) \
                + ([f"rebuilt from earlier events in {rounds_span(obs_reb)}"] if obs_reb else [])
            obs = "the observation is " + (parts[0] if len(parts) == 1 else f"{parts[0]} and {parts[1]}"
                                             if len(parts) == 2 and " and " not in parts[0] else ", ".join(parts[:-1]) + ", and " + parts[-1])
        prov = f"The game log has no event per decision; {acts}; {obs}. "
    if pend and not pend_bad:
        pr = [r for r, _ in pend]
        where = ("in every round" if pr == list(range(1, n + 1)) else f"from round {pr[0]} on" if pr == list(range(pr[0], n + 1))
                 else f"in {rounds_span(pr)}")
        prov += (f"'pending' in the prompt's private info ({where}) equals the position carried into the round; the rules do not "
                 "list it among what the trader sees. ")
    foot = (f"Source: the {date} recording of {GID} ({T} rounds, signal {G['mode']}, fee {fee:g}, move {move:g}"
            + (f", seed {seed}" if seed is not None else "") + f", {model_name}), {'the rendered replay of its game log' if b.get('md') else 'game_log.jsonl'}. Each round's round_result event "
            f"gives the signal, the position the game applied, the price change, the payoff and the running score, and they add up "
            f"exactly: {num(gross)} from price moves minus {fees:.1f} in fees is {num(score[-1])}, the recorded reward. " + prov)
    if exact:
        foot += (f"With the exact signal the signal is the true regime, so the strip shows the hidden regime. DP policies: backward "
                 f"induction over (round, regime, position) with regime stay probability {G['stay']:g} and move-with-regime "
                 f"probability {G['with']:g}" + (" (read from the recorded prompt)" if G["probs_from_prompt"] else "")
                 + "; start certainty equivalents " + ", ".join(f"τ {r['tau']:g} {abs(r['start']):.2f}" if abs(r['start']) < 0.005
                                                                else f"τ {r['tau']:g} {r['start']:.2f}" for r in refs)
                 + (", as in the rules' table" if rules_ok else "")
                 + (f"; the risk-neutral expected total equals the config's reference {float(G['cfg']['ref_optimal_risk_neutral_expected']):g}"
                    if cfg_ok else "") + ". "
                 + f"Expected given these regimes: each round's position times {(2 * G['with'] - 1) * move:g} in the regime's direction "
                 "(the mean move), minus fees. "
                 + (f"Risk-attitude fit: the DP policy for τ from 0 to {TAU_HI:g} (steps of {TAU_STEP:g}, edges refined by "
                    "bisection) is compared with every decision, from the position the model held. " if fit_runs else ""))
    foot += ("Behavioral metrics as defined in the rules, rounds 2 to the end: trend chasing = P(increase | last change up) "
             f"{MINUS} P(increase | last change down); disposition effect = P(reduce | the held position gained last round) {MINUS} "
             "P(reduce | it lost), a reversal counting as a reduction; signal use = the share of non-flat rounds (all rounds) on the "
             "signal's side"
             + ("; the optimum's values are exact (expected counts propagated round by round). " if exact and beh_ref else
                "; the Bayes optimum's values are quoted from the rules (10 rounds, default parameters). " if beh_ref else ". "))
    foot += (f"Reasoning: from the model-call log, tied to {tied} of {n} decisions." if calls else
             "This recording was made before reasoning was saved; the game will be re-recorded with reasoning.")

    return {"id": GID, "name": "Regime trading", "title": title, "lede": lede, "seatNames": ["Trader"], "tiles": tiles, "foot": en(foot),
            "x": {"n": xn, "labels": labels}, "charts": [c_price, c_grid, c_cum], "kinds": kinds, "steps": steps,
            "feed": feed, "feedTitle": "Round by round",
            "feedSub": "One line per round_result event: what the trader did, how the price moved and what it earned. The selected round is highlighted."}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    date = recorded_date(src)                         # read once, so a missing date is warned about once
    common.write(build(src, calls, date=date), out, date=date)
