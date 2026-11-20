"""Replay data for eco_finrl_daily (daily trading of one real stock with five position levels, 0/25/50/75/100%; execution and
accounting as FinRL-Meta's StockTradingEnv, re-implemented for one stock; Liu et al., "FinRL-Meta", NeurIPS 2022 Datasets and
Benchmarks).

usage: python3 -I eco_finrl_daily.py BUNDLE OUT_DIR [CALLS.jsonl]
  BUNDLE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md), or a file
          inside one.
  OUT_DIR where <OUT_DIR>/eco_finrl_daily.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer, followup_answer when the answer came from a
          follow-up question, finish, usage, t_start). Calls are grouped into runs of consecutive calls whose prompts show the
          same 20 closes, weight and level (a day's prompt is unique, since the window moves one close a day, and the recorder
          asks again with the identical prompt after a rejected reply); each decision takes the first run with its prompt after
          the previous decision's run. In the run, the deciding call is the first whose answer the game accepts, read exactly as
          the engine reads it (engine_level), and it must give the logged level; earlier calls were rejected attempts, kept
          above it with their reasoning. For a move the game logged as a fallback ("fallback": true), every call in the run is
          a rejected attempt. Calls that started before the recording did (an earlier attempt appended to the same log) are
          left out. The reasoning becomes the step's thinking and the prompt the observation.

test only:  python3 -I eco_finrl_daily.py --test-from-calls CALLS.jsonl OUT_DIR
          builds the page from a call log alone (for example the partial log of a run that stopped, or the growing log of a
          recording still running, whose half-written last line is left out). The stand-in bundle is assembled in memory (see
          TEST HELPER at the end of this file); the page says in its eyebrow, title and foot that it is a test build. Nothing is
          written but OUT_DIR/eco_finrl_daily.json.

One round is one trading day and there is one seat. Each prompt shows the 20 dividend-adjusted closes ending today, the stock's
weight in the portfolio (whole percent) and the level held. The page's day record is rebuilt from the prompts and the moves:
day d's close is the last close in day d's prompt (consecutive prompts must agree on the 19 closes they share), the level
applied on day d is the level the next prompt says is held (the move's level is checked against it), and the portfolio is
replayed with the rules' accounting (Round Procedure: $1,000,000 start, whole shares, target = floor(L x V / (P x 1.001)),
0.1% fee on each buy and sell, the level already held makes no trade). A move the recorder flags as a fallback (no reply the
game accepts; rules: Actions, the level held is kept) is shown as 'No usable answer', and the number of flags is checked
against the summary's count. The prompts show closes to the cent while the engine trades at the unrounded adjusted close, so the
rebuilt portfolio can hold one share more or fewer than the engine's on some days and its values differ slightly; every weight
the trader was shown is checked against the replay, and the summary's metrics (return, drawdown, volatility, Sharpe, trades,
turnover, fees, levels, final cash, shares and equity, buy-and-hold) and the reward are checked against the rebuilt ones, with
every disagreement printed as a warning. The rebuild draws only the per-day curves and the day-by-day entries; every total the
page quotes (title, tiles, chart notes, the End entry) is the summary's value when it has one.
Dates: the prompts show none. When the summary gives the first and last day and the day count, the days are labelled with the
NYSE trading days of 2025-2026 (weekdays without exchange holidays), used only if that calendar gives exactly the summary's
first day, last day and number of days."""
import datetime, json, math, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_finrl_daily"
MINUS, DOT, TIMES, DIV, ARROW, NDASH, SQRT = "−", "·", "×", "÷", "→", "–", "√"
MAX_BYTES = 15_000_000                 # the page is one JSON file; its host takes up to 16 MB
# Rule constants (rules: Configuration, "fixed by the rules"); the summary's values win when it has them.
START_CASH, FEE, LEVELS, CLIP, YEAR = 1_000_000.0, 0.001, (0, 25, 50, 75, 100), 1.0, 252
DEFAULT_WINDOW = ("2025-06-02", "2026-10-19", 348)           # the five configs' window (rules: Default Configs)
COMPANY = {"AAPL": "Apple", "JPM": "JPMorgan", "JNJ": "Johnson & Johnson", "CVX": "Chevron", "CAT": "Caterpillar"}
# Other stocks and funds the reasoning may name while it guesses which stock it sees (the prompt names none): the four other
# stocks of the game plus well-known US large caps and index funds, each with its ticker. Short tickers that are also common
# words or letters (V, MA, BA, GS, PG, KO, COST) are left out; matching is case-sensitive.
OTHER_NAMES = {**{c: [t, c] for t, c in COMPANY.items()},
               "Microsoft": ["Microsoft", "MSFT"], "Amazon": ["Amazon", "AMZN"], "Tesla": ["Tesla", "TSLA"],
               "Alphabet": ["Alphabet", "Google", "GOOGL", "GOOG"], "Meta": ["Meta", "Facebook"],
               "NVIDIA": ["NVIDIA", "Nvidia", "NVDA"], "Netflix": ["Netflix", "NFLX"], "AMD": ["AMD"], "Intel": ["Intel", "INTC"],
               "IBM": ["IBM"], "Oracle": ["Oracle", "ORCL"], "Salesforce": ["Salesforce", "CRM"], "Adobe": ["Adobe", "ADBE"],
               "Broadcom": ["Broadcom", "AVGO"], "Qualcomm": ["Qualcomm", "QCOM"], "Costco": ["Costco"],
               "Walmart": ["Walmart", "WMT"], "Home Depot": ["Home Depot"], "Visa": ["Visa"], "Mastercard": ["Mastercard"],
               "Boeing": ["Boeing"], "Disney": ["Disney"], "PayPal": ["PayPal", "PYPL"], "Goldman Sachs": ["Goldman Sachs"],
               "Exxon": ["Exxon", "XOM"], "Pfizer": ["Pfizer", "PFE"], "UnitedHealth": ["UnitedHealth", "UNH"],
               "Berkshire Hathaway": ["Berkshire"], "Procter & Gamble": ["Procter"], "Coca-Cola": ["Coca-Cola"],
               "PepsiCo": ["PepsiCo", "PEP"], "SPY": ["SPY"], "QQQ": ["QQQ"]}
# Reference results from the rules, valid only for the full default window (rules: Reward, "What this reward favours", and
# Phenomenon): reward when holding one level every day (0, 25, 50, 75, 100%); a uniformly random level every day, 30 seeds
# (mean, min, max); buy-and-hold cumulative return, Sharpe and max drawdown.
REF = {"AAPL": {"fixed": (0, 0.163, 0.303, 0.425, 0.535), "random": (0.093, -0.098, 0.281), "bh": (0.707, 1.737, 0.138)},
       "JPM": {"fixed": (0, 0.071, 0.138, 0.200, 0.259), "random": (-0.025, -0.170, 0.163), "bh": (0.296, 1.000, 0.155)},
       "JNJ": {"fixed": (0, 0.157, 0.293, 0.412, 0.519), "random": (0.139, 0.026, 0.237), "bh": (0.680, 2.049, 0.110)},
       "CVX": {"fixed": (0, 0.163, 0.303, 0.426, 0.536), "random": (0.133, -0.053, 0.322), "bh": (0.709, 1.825, 0.208)},
       "CAT": {"fixed": (0, 0.249, 0.448, 0.614, 0.756), "random": (0.196, -0.069, 0.716), "bh": (1.130, 1.693, 0.317)}}
# NYSE full-day closures in 2025 and 2026 (the only years the game's window can cover; rules: Configuration bounds).
NYSE_HOLIDAYS = {datetime.date(*d) for d in (
    (2025, 1, 1), (2025, 1, 9), (2025, 1, 20), (2025, 2, 17), (2025, 4, 18), (2025, 5, 26), (2025, 6, 19), (2025, 7, 4),
    (2025, 9, 1), (2025, 11, 27), (2025, 12, 25),
    (2026, 1, 1), (2026, 1, 19), (2026, 2, 16), (2026, 4, 3), (2026, 5, 25), (2026, 6, 19), (2026, 7, 3), (2026, 9, 7),
    (2026, 11, 26), (2026, 12, 25))}
# How the foot names the summary values it checked (by key or key prefix).
CHECK_WORDS = [("equity", "final value"), ("cash", "cash"), ("shares", "shares"), ("level_days", "days per level"),
               ("level", "final level"), ("cum_return", "return"), ("log_return", "log return"), ("max_drawdown", "max drawdown"),
               ("ann_vol", "volatility"), ("sharpe", "Sharpe"), ("trades", "trades"), ("turnover", "turnover"),
               ("traded_value", "traded value"), ("fees", "fees"), ("avg_target", "mean level"), ("days_done", "days played"),
               ("buy_hold", "buy-and-hold metrics"), ("excess", "excess over buy-and-hold"), ("reward", "reward")]
KINDS = {"buy": ["Raises level " + "▲", "c1"], "sell": ["Lowers level " + "▼", "c2"], "keep": ["Keeps level", "hold"],
         "fallback": ["No usable answer, level kept", "muted"]}


def warn(msg):
    print("warning: " + msg, file=sys.stderr)


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_RUN = re.compile(r"[぀-ヿ㐀-鿿가-힯]+")
_PUNCT = {"。": ". ", "、": ", ", "「": '"', "」": '"', "『": '"', "』": '"', "【": "[",
          "】": "]", "《": '"', "》": '"', "〈": "<", "〉": ">", "〜": "~"}


def en(s):
    """English only on the page: full-width forms become ASCII, CJK punctuation its ASCII counterpart, and any other run of
    Chinese, Japanese or Korean characters a visible marker."""
    if not isinstance(s, str) or not re.search(r"[　-ヿ㐀-鿿가-힯＀-￯]", s):
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
    return re.sub(r"[　-〿＀-￯]", "", s)


# ------------------------------------------------------------------------------------------------ small helpers
def money(v, signed=False, cents=False):
    """$1,234,567 (or $1,234.56 with cents) with a true minus sign; '+' only when signed."""
    s = f"${abs(v):,.2f}" if cents else f"${abs(v):,.0f}"
    zero = s in ("$0", "$0.00")
    if v < 0 and not zero:
        return MINUS + s
    return ("+" + s) if signed and v > 0 and not zero else s


def spct(v, d=1):
    """A signed percentage of a fraction: +46.7%, −9.8%."""
    s = f"{abs(100 * v):.{d}f}"
    if float(s) == 0:
        return f"{0:.{d}f}%"
    return ("+" if v > 0 else MINUS) + s + "%"


def upct(v, d=0):
    """An unsigned percentage of a fraction: 82%."""
    return f"{100 * v:.{d}f}%"


def r3(v):
    """A reward or log return to 3 decimals with a true minus sign."""
    s = f"{abs(v):.3f}"
    return (MINUS if v < 0 and float(s) != 0 else "") + s


def px(v):
    return f"${v:,.2f}"


def nd(n, word):
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def times(n):
    return {0: "no times", 1: "once", 2: "twice"}.get(n, f"{n:,} times")


def and_list(xs):
    xs = list(xs)
    if len(xs) <= 1:
        return "".join(xs)
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------ loading
def load_source(src):
    if isinstance(src, dict):
        return src
    if not os.path.isdir(src):
        d = os.path.dirname(os.path.abspath(src))
        if os.path.exists(os.path.join(d, "game_log.jsonl")):
            return load_source(d)
        raise SystemExit(f"not a demo bundle: {src}")
    b = common.load_bundle(src)
    meta = os.path.join(src, "meta.json")
    b["meta"] = json.load(open(meta, encoding="utf-8")) if os.path.exists(meta) else {}
    return b


def _lenient_lines(path):
    """The JSON lines of a log that may still be growing: a last line that does not parse (half written) is left out."""
    lines = [line for line in open(path, encoding="utf-8") if line.strip()]
    raw = []
    for k, line in enumerate(lines):
        try:
            raw.append(json.loads(line))
        except ValueError:
            if k == len(lines) - 1:
                warn("the call log's last line is incomplete (still being written); left out")
            else:
                raise
    return raw


def load_calls(path, lenient=False):
    """The model-call log, read as common.load_calls reads it (prompt = last message, answer = follow-up answer when there is
    one, else the answer), plus why the call stopped ('finish'), its output tokens, start time and model. Same lines, same
    order. With `lenient` (a log still being written) a half-written last line is left out."""
    if not path or not os.path.exists(path):
        return []
    raw = _lenient_lines(path) if lenient else [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    out = []
    for r in raw:
        prompt = r.get("prompt")
        if prompt is None:
            msgs = r.get("messages") or []
            prompt = msgs[-1].get("content", "") if msgs else ""
        fu = r.get("followup_answer")
        if fu in ("None", "null"):                           # a stringified null is no follow-up
            fu = None
        u = r.get("usage") or {}
        if isinstance(u, str):                               # some logs store the usage dict as its Python repr
            m = re.search(r"'completion_tokens': (\d+)", u)
            u = {"completion_tokens": int(m.group(1))} if m else {}
        out.append({"seat": r.get("seat", common.seat_of(prompt)), "prompt": prompt or "",
                    "answer": (fu or r.get("answer") or "").strip(), "first_answer": (r.get("answer") or "").strip(),
                    "reasoning": r.get("reasoning") or "", "rescued": bool(fu),
                    "finish": r.get("finish") or r.get("finish_reason"), "out_tokens": u.get("completion_tokens"),
                    "t_start": _f(r.get("t_start")), "model": r.get("model")})
    return out


def recorded_date(b, calls=None):
    """The recording date: meta.json or run.json recorded_at, else the timestamp that opens the source run's name, else the first
    call's start time (UTC)."""
    meta, run = b.get("meta") or {}, b.get("run") or {}
    for v in (meta.get("recorded_at"), run.get("recorded_at")):
        if v:
            return str(v)[:10]
    for v in (meta.get("source_run"), run.get("source_run")):
        m = re.match(r"(\d{4})(\d{2})(\d{2})T\d{6}Z", str(v or ""))
        if m:
            return "-".join(m.groups())
    t0 = next((c["t_start"] for c in (calls or []) if c.get("t_start")), None)
    if t0:
        return datetime.datetime.fromtimestamp(t0, datetime.timezone.utc).strftime("%Y-%m-%d")
    return "date not recorded"


# ------------------------------------------------------------------------------------------------ what the trader saw
RX_CLOSES = re.compile(r"Closing prices of the stock over the last (\d+) trading days[^\n]*:\n((?:[ \t]*-?[\d,]+(?:\.\d+)?[ \t]*\n?)+)")
RX_WEIGHT = re.compile(r"Current position: (\d+)% of the portfolio is in the stock and (\d+)% is in cash")
RX_LEVEL = re.compile(r"Level you hold now: (\d+)%")


def parse_view(text):
    """The numbers a prompt shows: the closes (oldest first, the last is today), the stock weight and cash share (whole
    percent) and the level held. Missing parts are None."""
    v = {"closes": None, "weight": None, "cash": None, "level": None}
    m = RX_CLOSES.search(text or "")
    if m:
        cl = [float(x.replace(",", "")) for x in m.group(2).split()]
        if cl:
            v["closes"] = cl
            if int(m.group(1)) != len(cl):
                warn(f"a prompt announces {m.group(1)} closes but lists {len(cl)}")
    m = RX_WEIGHT.search(text or "")
    if m:
        v["weight"], v["cash"] = int(m.group(1)), int(m.group(2))
    m = RX_LEVEL.search(text or "")
    if m:
        v["level"] = int(m.group(1))
    return v


def view_key(text):
    v = parse_view(text)
    if v["closes"] is None:
        return None
    return (tuple(v["closes"]), v["weight"], v["level"])


def _percent(p):
    """`percent` as the engine reads it (whole_number, then the five levels): a JSON number with a whole value (50.0 counts as
    50; booleans and strings do not) equal to one of the levels, else None."""
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        return None
    try:
        f = float(p)
    except OverflowError:
        return None
    if not math.isfinite(f) or f != math.floor(f) or int(f) not in LEVELS:
        return None
    return int(f)


def engine_level(text):
    """The level the game reads from a reply or a logged move, exactly as the engine parses it (rules: Actions; the engine's
    parse_action and _check, and the game's _canonical): json.loads of the whole text (surrounding white space aside), a JSON
    object whose keys lie within action_id and payload, action_id 'set_position', a payload whose only key is percent, and
    percent one of the five levels (_percent). None when the game rejects it."""
    if not isinstance(text, str):
        return None
    try:
        data = json.loads(text.strip())
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict) or set(data) - {"action_id", "payload"} or data.get("action_id") != "set_position":
        return None
    payload = data.get("payload", {})
    if not isinstance(payload, dict) or set(payload) != {"percent"}:
        return None
    return _percent(payload["percent"])


def reject_reason(text):
    """Why the game rejects a reply, in a few words, with the level it seems to mean when one can be read leniently (a code
    fence stripped, the first {...} span, extra keys ignored). Used only to describe a rejected reply, never to tie a call to a
    decision."""
    s = (text or "").strip()
    if not s:
        return "empty reply"
    why = []
    try:
        data = json.loads(s)
    except (ValueError, RecursionError):
        data = None
        fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
        i, j = s.find("{"), s.rfind("}")
        for cand, w in ((fenced, "JSON inside a code fence"), (s[i:j + 1] if 0 <= i < j else "", "JSON with other text around it")):
            if cand and cand != s:
                try:
                    data = json.loads(cand)
                    why.append(w)
                    break
                except (ValueError, RecursionError):
                    pass
        if data is None:
            return "not JSON"
    if not isinstance(data, dict):
        return "; ".join(why + ["not a JSON object"])
    extra = sorted(str(k) for k in set(data) - {"action_id", "payload"})
    if extra:
        why.append("extra key " + ", ".join(extra))
    if data.get("action_id") != "set_position":
        why.append(f"action_id {json.dumps(data.get('action_id'))[:40]}")
    payload = data.get("payload")
    p = None
    if not isinstance(payload, dict):
        why.append("no payload object")
    else:
        extra_p = sorted(str(k) for k in set(payload) - {"percent"})
        if extra_p:
            why.append("extra payload key " + ", ".join(extra_p))
        if "percent" not in payload:
            why.append("no percent")
        else:
            p = _percent(payload["percent"])
            if p is None:
                why.append(f"percent {json.dumps(payload['percent'])[:20]} is not one of the five levels")
    if not why:
        return "accepted"
    return "; ".join(why) + (f"; it reads as {p}%" if p is not None else "")


# ------------------------------------------------------------------------------------------------ calendar
def trading_days(a, b):
    """NYSE trading days from a to b inclusive, or None when the range leaves 2025-2026 (holidays known only there)."""
    if a.year < 2025 or b.year > 2026 or b < a:
        return None
    out, d = [], a
    while d <= b:
        if d.weekday() < 5 and d not in NYSE_HOLIDAYS:
            out.append(d)
        d += datetime.timedelta(days=1)
    return out


def _date(s):
    try:
        return datetime.date.fromisoformat(str(s)[:10])
    except (TypeError, ValueError):
        return None


def day_dates(first, last, days, T):
    """ISO dates of the T decision days, or (None, reason). Used only when the calendar from `first` to `last` starts and ends on
    those days and has exactly `days` days (and at least T)."""
    a, b = _date(first), _date(last)
    if a is None or b is None:
        return None, "the summary gives no first and last day"
    cal = trading_days(a, b)
    if not cal:
        return None, f"the window {a} to {b} leaves the calendar this page knows (2025-2026)"
    if cal[0] != a or cal[-1] != b:
        return None, f"{a} or {b} is not an NYSE trading day in this page's calendar"
    if days is not None and len(cal) != int(days):
        return None, f"the calendar has {len(cal)} trading days from {a} to {b}, the summary {days}"
    if len(cal) < T:
        return None, f"{T} decisions but only {len(cal)} trading days from {a} to {b}"
    return [d.isoformat() for d in cal[:T]], ""


# ------------------------------------------------------------------------------------------------ accounting (rules: Round Procedure)
def replay(prices, levels, v0=START_CASH, fee=FEE):
    """Day rows after each decision: the rules' trade at the close (target = floor(L/100 x V / (P (1 + fee))); a sale adds
    n P (1 - fee) to cash, a purchase takes n P (1 + fee), also capped at floor(cash / P); the level already held makes no
    trade), then the mark V = cash + shares x P."""
    cash, sh, held, rows = float(v0), 0, 0, []
    for p, lv in zip(prices, levels):
        v_pre = cash + sh * p
        w_pre = sh * p / v_pre if v_pre > 0 else 0.0
        n = 0
        if lv != held:
            tgt = int(math.floor(lv / 100.0 * v_pre / (p * (1 + fee))))
            if tgt < sh:
                n = tgt - sh
                cash += (sh - tgt) * p * (1 - fee)
                sh = tgt
            elif tgt > sh:
                n = min(tgt - sh, int(math.floor(cash / p)))
                cash -= n * p * (1 + fee)
                sh += n
        v = cash + sh * p
        rows.append({"price": p, "held0": held, "level": lv, "v_pre": v_pre, "w_pre": w_pre, "n": n, "traded": abs(n) * p,
                     "fee": abs(n) * p * fee, "cash": cash, "shares": sh, "value": v, "w": sh * p / v if v > 0 else 0.0})
        held = lv
    return rows


def metrics(values, v0=START_CASH):
    """The rules' metrics of a value path (V_1..V_T, after each day's trade): return and log return on V_0, daily returns from
    V_0 (the first one is the first trade's fee), annualized volatility and Sharpe (sample deviation, sqrt 252, risk-free 0),
    max drawdown from the running peak with V_0 included, and the drawdown path."""
    seq = [float(v0)] + list(values)
    r = [seq[i] / seq[i - 1] - 1 for i in range(1, len(seq))]
    n = len(r)
    mean = sum(r) / n if n else 0.0
    sd = math.sqrt(sum((x - mean) ** 2 for x in r) / (n - 1)) if n > 1 else 0.0
    peak, pk_i, mdd, mdd_at, dd = seq[0], 0, 0.0, (0, 0), []
    for i, v in enumerate(seq):
        if v > peak:
            peak, pk_i = v, i
        d = 1 - v / peak
        if i:
            dd.append(-d)
        if d > mdd:
            mdd, mdd_at = d, (pk_i, i)
    return {"cum": seq[-1] / v0 - 1, "log": math.log(seq[-1] / v0), "vol": sd * math.sqrt(YEAR),
            "sharpe": mean / sd * math.sqrt(YEAR) if sd > 0 else 0.0, "mdd": mdd, "mdd_at": mdd_at, "dd": dd, "r": r}


def info_get(info, *keys):
    """The first of `keys` found in the summary's info, at the top level or under 'metrics'."""
    for k in keys:
        for d in (info, info.get("metrics") if isinstance(info.get("metrics"), dict) else None):
            if isinstance(d, dict) and d.get(k) is not None:
                return d[k]
    return None


# ------------------------------------------------------------------------------------------------ model calls
def run_start(b):
    """When the recording started (epoch seconds), or None: the UTC stamp that opens the source run's name (the recorder names
    the run directory when play starts), else recorded_at."""
    meta, run = b.get("meta") or {}, b.get("run") or {}
    for v in (meta.get("source_run"), run.get("source_run")):
        m = re.match(r"(\d{8}T\d{6}Z)", str(v or ""))
        if m:
            return datetime.datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
    for v in (meta.get("recorded_at"), run.get("recorded_at")):
        try:
            t = datetime.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        return (t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)).timestamp()
    return None


def call_text(c, deciding):
    """A call's reasoning as shown: a visible cut mark where it hit the output limit (finish 'length') and, when its answer came
    from a follow-up question (rescued), a leading note saying so. Returns (text, the token limit it hit or None)."""
    finish, toks = c.get("finish"), c.get("out_tokens")
    cut = finish == "length" if finish else bool(c.get("rescued"))
    lim_n = toks if cut and finish == "length" and toks else None
    lim = f"the {lim_n:,}-token output limit" if lim_n else "the token limit"
    text = (c.get("reasoning") or "").rstrip()
    if text and cut:
        text += f" ... [cut off here at {lim}]"
    if c.get("rescued"):
        what = "the action is the model's answer" if deciding else "its answer is the model's reply"
        note = (f"[This reasoning was cut off at {lim} before it reached an answer; {what} to a follow-up question.]" if cut else
                f"[This reasoning ended without a usable answer; {what} to a follow-up question.]")
        text = note + ("\n\n" + text if text else "")
    return text, lim_n


def _span(ks):
    """Call numbers (1-based) of a list of call indices: 'call 7' or 'calls 7..12'."""
    return f"call {ks[0] + 1}" if len(ks) == 1 else f"calls {ks[0] + 1}..{ks[-1] + 1}"


def attach_calls(steps, calls, start=None):
    """Tie calls to steps. Calls that started before the recording did (`start`, epoch seconds; an earlier attempt appended to the
    same log) are left out. The rest are grouped into runs: maximal stretches of consecutive calls whose prompts show the same
    closes, weight and level (the view key). Each step takes the first run with its view's key after the previous step's run,
    so any number of unmatched calls in between is skipped (and reported); a key found in two separate runs is reported too.
    In the run, the deciding call is the first whose answer the game accepts (engine_level, read exactly as the engine reads
    it), and it must give the logged level; the calls before it were rejected attempts, kept above it with their reasoning and
    why the game rejected them. For a step the game logged as a fallback, every call in the run is a rejected attempt and the
    reasoning ends with a note that the game kept the level held. Returns counts: tied, rescued (actions that came from a
    follow-up question), limits (the token limit each of those hit), retried (decisions accepted after a rejected reply)."""
    out = {"tied": 0, "rescued": 0, "limits": [], "retried": 0}
    if not calls:                                            # no call log: the steps have no reasoning
        return out
    keep = list(range(len(calls)))
    if start is not None:
        early = [k for k in keep if calls[k].get("t_start") is not None and calls[k]["t_start"] < start - 2]
        if len(early) == len(calls) and calls:
            warn("every call in the log starts before the recording's start time; the start time is not used")
        elif early:
            warn(f"{nd(len(early), 'call')} ({_span(early)}) started before this recording did (an earlier attempt in the same "
                 "log); left out")
            gone = set(early)
            keep = [k for k in keep if k not in gone]
    runs = []                                                # [view key or None, [call indices]]
    for k in keep:
        kk = view_key(calls[k].get("prompt")) if calls[k].get("seat") in (None, 0) else None
        if runs and kk is not None and runs[-1][0] == kk:
            runs[-1][1].append(k)
        else:
            runs.append([kk, [k]])
    by_key = {}
    for ri, (kk, _) in enumerate(runs):
        if kk is not None:
            by_key.setdefault(kk, []).append(ri)
    dup = [ris for ris in by_key.values() if len(ris) > 1]
    if dup:
        warn(f"{nd(len(dup), 'prompt')} appear in two or more separate runs of calls (for example {_span(runs[dup[0][0]][1])} "
             f"and {_span(runs[dup[0][1]][1])}); each decision takes the first run after the previous decision's calls")
    pos = -1                                                 # index in `runs` of the previous step's run
    for st in steps:
        if st["key"] is None:
            warn(f"day {st['day']}: its view shows no closes, so no call is tied to it")
            continue
        ri = next((r for r in by_key.get(st["key"], []) if r > pos), None)
        if ri is None:
            warn(f"no call for day {st['day']}")
            continue
        skipped = [k for r in range(pos + 1, ri) for k in runs[r][1]]
        if skipped:
            warn(f"{nd(len(skipped), 'call')} before day {st['day']}'s ({_span(skipped)}) match no decision; skipped")
        pos = ri
        window = runs[ri][1]
        read = [engine_level(calls[k].get("answer")) for k in window]
        notes = []
        if st["fb"]:
            use, attempts = None, window
            ok = [k for k, lv in zip(window, read) if lv is not None]
            if ok:
                warn(f"day {st['day']} is logged as a fallback, but the game accepts the answer of call {ok[0] + 1}")
            # the recorder keeps the first 2,000 characters of the last rejected reply with the fallback
            if st.get("reply") is not None and (calls[window[-1]].get("answer") or "")[:2000] != st["reply"]:
                warn(f"day {st['day']}: the reply the game logged with its fallback is not the answer of call {window[-1] + 1}")
            notes.append(f"[No usable answer after {nd(len(window), 'model call')}; the game kept the level held "
                         "(rules: Actions)]")
        else:
            j = next((i for i, lv in enumerate(read) if lv is not None), None)
            if j is None:
                use, attempts = None, window
                warn(f"day {st['day']}: the game accepts none of the answers of {_span(window)}, yet logged {st['act']} "
                     "without a fallback flag")
                notes.append(f"[The game accepts none of these answers, yet it logged {st['act']}; this call log may not belong "
                             "to this recording]")
            else:
                use, attempts = window[j], window[:j]
                if read[j] != st["move_level"]:
                    warn(f"day {st['day']}: the first answer the game accepts reads {read[j]}%, the logged move "
                         f"{st['move_level']}%")
                    notes.append(f"[The game logged another action than this answer: {st['act']}]")
                if j + 1 < len(window):
                    warn(f"day {st['day']}: {nd(len(window) - j - 1, 'call')} after the deciding one repeat its prompt; left out")
        n_att = len(attempts) + (use is not None)
        parts = []
        for i, k in enumerate(attempts):
            c = calls[k]
            text, _ = call_text(c, False)
            ans = c.get("answer") or ""
            parts.append(f"[Attempt {i + 1} of {n_att}, rejected by the game ({reject_reason(ans)}); answer: "
                         f"{ans[:300] + (' ...' if len(ans) > 300 else '') if ans else '(empty)'}]\n"
                         + (text or "(no reasoning returned)"))
        if use is not None:
            c = calls[use]
            text, lim_n = call_text(c, True)
            if c.get("rescued"):
                out["rescued"] += 1
                out["limits"].append(lim_n)
            if attempts:
                out["retried"] += 1
                parts.append(f"[Attempt {n_att} of {n_att}, accepted]\n" + (text or "(no reasoning returned)"))
            else:
                parts.append(text)
        st["thinking"] = en("\n\n".join(p for p in parts + notes if p).strip())
        st["chars"] = sum(len(calls[k].get("reasoning") or "") for k in window)
        shown = calls[use if use is not None else window[-1]]
        if shown.get("prompt"):
            st["state"] = en(common.state_part(shown["prompt"]))
        out["tied"] += 1
    rest = [k for r in range(pos + 1, len(runs)) for k in runs[r][1]]
    if rest:
        warn(f"{nd(len(rest), 'call')} after the last decision ({_span(rest)}) were not tied to any step")
    return out


def fit_thinking(steps, budget, full):
    """Keep the page within its size limit: when all reasoning together is longer than `budget` characters, shorten the longest
    ones (one common cap, the largest that fits) by removing their middle, with a visible mark. Always cuts from each step's
    full reasoning (`full`). Returns the number of shortened steps and the characters left out."""
    lens = [len(t) for t in full]
    if sum(lens) <= budget:
        for s, t in zip(steps, full):
            s["thinking"] = t
        return 0, 0
    lo, hi = 2000, max(lens)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(x, mid) for x in lens) <= budget:
            lo = mid
        else:
            hi = mid - 1
    cap, n, removed = lo, 0, 0
    for s, x, t in zip(steps, lens, full):
        if x <= cap:
            s["thinking"] = t
            continue
        head, tail = int(cap * 0.45), int(cap * 0.55)
        gone = x - head - tail
        s["thinking"] = (t[:head].rstrip() + f"\n\n[... {gone:,} characters of this reasoning are left out here to keep the page "
                         "within its size limit; the beginning and the end are shown ...]\n\n" + t[-tail:].lstrip())
        n += 1
        removed += gone
    return n, removed


# ------------------------------------------------------------------------------------------------ the day record
def rebuild_days(moves):
    """Per decision day: the view's numbers, the close (aligned from every prompt's window), the move's level (read as the engine
    reads it), whether the recorder flagged the move as a fallback (no reply the game accepted; the move then keeps the level
    held) with the last rejected reply, and the level the game applied (the next prompt's level held; for the last day the
    move's level, or the level held when it is unusable). Returns (days, W, checks)."""
    T = len(moves)
    views = [parse_view(e.get("view") or "") for e in moves]
    W = next((len(v["closes"]) for v in views if v["closes"]), 0)
    chk = {"shift_n": 0, "shift_ok": 0, "applied_n": 0, "applied_ok": 0, "pre": None}
    # closes: day i's window covers days i - W + 1 .. i; the earliest prompt also shows W - 1 days before the first decision
    close = [None] * T
    pre = {}
    for i, v in enumerate(views):
        cl = v["closes"]
        if not cl:
            continue
        for j, p in enumerate(cl):
            d = i - (len(cl) - 1) + j
            if d < 0:
                if d in pre and abs(pre[d] - p) > 1e-9:
                    warn(f"prompts disagree on the close {-d} days before the first decision: {pre[d]} and {p}")
                pre.setdefault(d, p)
                continue
            if close[d] is not None and abs(close[d] - p) > 1e-9:
                warn(f"prompts disagree on day {d + 1}'s close: {close[d]} and {p}")
                continue
            close[d] = p if close[d] is None else close[d]
        if i and views[i - 1]["closes"]:
            chk["shift_n"] += 1
            if views[i - 1]["closes"][1:] == cl[:-1]:
                chk["shift_ok"] += 1
            else:
                warn(f"day {i + 1}'s prompt does not continue day {i}'s closes")
    chk["pre"] = [pre[d] for d in sorted(pre)]
    days = []
    for i, e in enumerate(moves):
        v = views[i]
        move = e.get("move")
        if isinstance(move, dict):
            move = json.dumps(move, sort_keys=True)
        mv = engine_level(move)
        fb = bool(e.get("fallback"))
        held0 = v["level"] if v["level"] is not None else (days[-1]["applied"] if days else 0)
        if mv is None:
            warn(f"day {i + 1}: the logged move is not an action the game accepts: {str(move)[:120]!r}")
        elif fb and mv != held0:
            warn(f"day {i + 1}: logged as a fallback, but its move {mv}% is not the level held ({held0}%)")
        nxt = views[i + 1]["level"] if i + 1 < T else None
        if nxt is not None:
            applied = nxt
            chk["applied_n"] += 1
            if mv is not None and mv == nxt or mv is None and nxt == held0:
                chk["applied_ok"] += 1
            else:
                warn(f"day {i + 1}: the move says {mv}, the next prompt holds {nxt}")
        else:
            applied = mv if mv is not None else held0
        sma = sum(v["closes"]) / len(v["closes"]) if v["closes"] else None
        days.append({"day": i + 1, "view": v, "move_level": mv, "held0": held0, "applied": applied, "close": close[i], "sma": sma,
                     "move": move or "", "text": e.get("view") or "", "fallback": fb,
                     "reply": e.get("reply") if fb and isinstance(e.get("reply"), str) else None})
    return days, W, chk


def summary_checks(info, rb, rbh, rm, bm, levels, rewards, reward_rb, T):
    """Compare the rebuilt numbers with the summary's metrics and reward; every disagreement is a warning. Returns (n, ok, notes)
    with notes holding the largest relative difference of the final value."""
    pairs = []

    def add(name, theirs, mine, tol_rel=2e-4, tol_abs=1e-6, exact=False):
        if theirs is None:
            return
        try:
            t = float(theirs)
        except (TypeError, ValueError):
            return
        ok = (t == mine) if exact else abs(t - mine) <= max(tol_abs, tol_rel * max(abs(t), abs(mine)))
        pairs.append((name, t, mine, ok))
    days_done = len(levels)
    last = rb[-1]
    add("equity", info_get(info, "equity"), last["value"])
    add("cash", info_get(info, "cash"), last["cash"], tol_rel=0, tol_abs=max(1.0, 3e-4 * last["value"]))
    add("shares", info_get(info, "shares"), last["shares"], tol_rel=0, tol_abs=2)
    lv_end = _f(info_get(info, "level"))
    if lv_end is not None:                                   # a percent (0..100) or a fraction (0..1)
        add("level", lv_end * 100 if 0 < lv_end <= 1 else lv_end, levels[-1], exact=True)
    add("cum_return", info_get(info, "cum_return"), rm["cum"], tol_rel=0, tol_abs=2e-4)
    add("log_return", info_get(info, "log_return"), rm["log"], tol_rel=0, tol_abs=2e-4)
    add("max_drawdown", info_get(info, "max_drawdown"), rm["mdd"], tol_rel=0, tol_abs=2e-4)
    add("ann_vol", info_get(info, "ann_vol"), rm["vol"], tol_rel=2e-3, tol_abs=2e-4)
    add("sharpe", info_get(info, "sharpe"), rm["sharpe"], tol_rel=2e-3, tol_abs=2e-3)
    add("trades", info_get(info, "trades"), sum(1 for r in rb if r["n"]), exact=True)
    add("turnover", info_get(info, "turnover"), sum(r["traded"] / r["v_pre"] for r in rb if r["n"]), tol_rel=1e-3)
    add("traded_value", info_get(info, "traded_value"), sum(r["traded"] for r in rb), tol_rel=1e-3)
    add("fees", info_get(info, "fees"), sum(r["fee"] for r in rb), tol_rel=1e-3)
    add("avg_target", info_get(info, "avg_target"), sum(levels) / 100.0 / days_done, tol_rel=0, tol_abs=1e-9)
    add("days_done", info_get(info, "days_done"), days_done, exact=True)
    ld = info_get(info, "level_days")
    if isinstance(ld, dict):                                 # keys are levels in percent ("25") or as fractions ("0.25")
        ks = {k: _f(str(k).rstrip("%")) for k in ld}
        frac = all(x is not None and x <= 1 for x in ks.values())
        for k, v in ld.items():
            if ks[k] is None:
                continue
            lv = round(ks[k] * 100) if frac else round(ks[k])
            add(f"level_days[{k}]", v, sum(1 for x in levels if x == lv), exact=True)
    bh = info_get(info, "buy_hold")
    if isinstance(bh, dict):
        add("buy_hold.cum_return", bh.get("cum_return"), bm["cum"], tol_rel=0, tol_abs=2e-4)
        add("buy_hold.log_return", bh.get("log_return"), bm["log"], tol_rel=0, tol_abs=2e-4)
        add("buy_hold.max_drawdown", bh.get("max_drawdown"), bm["mdd"], tol_rel=0, tol_abs=2e-4)
        add("buy_hold.ann_vol", bh.get("ann_vol"), bm["vol"], tol_rel=2e-3, tol_abs=2e-4)
        add("buy_hold.sharpe", bh.get("sharpe"), bm["sharpe"], tol_rel=2e-3, tol_abs=2e-3)
    add("excess_log_return", info_get(info, "excess_log_return"), rm["log"] - bm["log"], tol_rel=0, tol_abs=3e-4)
    add("excess_return", info_get(info, "excess_return"), rm["cum"] - bm["cum"], tol_rel=0, tol_abs=3e-4)
    if rewards.get(0) is not None:
        add("reward", rewards[0], reward_rb, tol_rel=0, tol_abs=2e-4)
    for name, t, mine, ok in pairs:
        if not ok:
            warn(f"summary {name} {t}, rebuilt {mine}")
    eq = _f(info_get(info, "equity"))
    rel = (eq - last["value"], abs(eq - last["value"]) / eq) if eq else None
    return len(pairs), sum(1 for p in pairs if p[3]), rel, [p[0] for p in pairs], [p[0] for p in pairs if not p[3]]


# ------------------------------------------------------------------------------------------------ build
def build(src, calls=None):
    b = load_source(src)
    if isinstance(calls, str):
        calls = load_calls(calls)
    calls = calls or []
    events, summary, run = b["events"], b.get("summary") or {}, b.get("run") or {}
    meta, config = b.get("meta") or {}, b.get("config") or {}
    info = summary.get("info") or {}
    test_build = bool(run.get("test_bundle_from_calls"))
    moves = [e for e in events if e.get("event") == "move"]
    if not moves:
        raise SystemExit("no move events in " + str(src if isinstance(src, str) else "the stand-in bundle"))
    if any(e.get("actor", 0) != 0 for e in moves):
        warn("a move by a seat other than P0; this game has one seat")
    days, W, chk = rebuild_days(moves)
    T = len(days)
    missing = [d["day"] for d in days if d["close"] is None]
    if missing:
        raise SystemExit(f"no close for days {missing[:10]}: the prompts do not cover them")

    params = info.get("params") or {}
    ticker = str(params.get("ticker") or info_get(info, "ticker") or "").upper() or None
    if not ticker:
        m = re.search(r"eco_finrl_daily_([a-z]+)\.ya?ml", str(config.get("variant") or ""))
        ticker = m.group(1).upper() if m else None
    v0 = float(info_get(info, "initial_cash", "start_cash") or START_CASH)
    fee = float(info_get(info, "fee", "transaction_cost") or params.get("fee") or FEE)
    first_day, last_day = info_get(info, "first_day"), info_get(info, "last_day")
    n_days = info_get(info, "days")
    assumed = run.get("assume_window")
    if not first_day and assumed:                            # test builds only: the rules' default window, said on the page
        first_day, last_day, n_days = assumed
    dates, why_nodates = day_dates(first_day, last_day, n_days, T) if first_day else (None, "the summary gives no first day")
    if dates is None:
        warn(f"no dates on the page: {why_nodates}")
    total_days = int(n_days) if n_days else None
    cut_short = bool(total_days and T < total_days)

    prices = [d["close"] for d in days]
    levels = [d["applied"] for d in days]
    rb = replay(prices, levels, v0, fee)
    rbh = replay(prices, [100] * T, v0, fee)
    mean_level = sum(levels) / T
    rfix = replay(prices, [mean_level] * T, v0, fee)
    rm, bm, fm = metrics([r["value"] for r in rb], v0), metrics([r["value"] for r in rbh], v0), metrics([r["value"] for r in rfix], v0)
    reward_rb = max(-CLIP, min(CLIP, rm["log"]))

    # what the trader was shown against the replay (the level shown as held is not checked here: the applied levels are read
    # from the next prompts, so it would agree by construction; the moves are checked against those prompts in rebuild_days)
    w_n = w_ok = w_near = 0
    for d, r in zip(days, rb):
        v = d["view"]
        if v["weight"] is not None:
            w_n += 1
            mine = 100 * r["w_pre"]
            if abs(v["weight"] - round(mine)) == 0 or abs(v["weight"] - mine) <= 0.5 + 1e-9:
                w_ok += 1
            elif abs(v["weight"] - mine) <= 1.0 + 1e-9:
                w_near += 1
                warn(f"day {d['day']}: shown weight {v['weight']}%, rebuilt {mine:.3f}% (rounding)")
            else:
                warn(f"day {d['day']}: shown weight {v['weight']}%, rebuilt {mine:.3f}%")

    over = next((e for e in events if e.get("event") == "game_over"), {})
    raw_rw = summary.get("rewards") or (run.get("result") or {}).get("rewards") or over.get("rewards") or {}
    rewards = {int(k): float(v) for k, v in raw_rw.items() if v is not None}
    s_n, s_ok, eq_rel, s_names, s_bad = summary_checks(info, rb, rbh, rm, bm, levels, rewards, reward_rb, T)

    def words(names):
        out = []
        for nm in names:
            w_ = next((v for k, v in CHECK_WORDS if nm == k or nm.startswith(k)), nm)
            if w_ not in out:
                out.append(w_)
        return out
    s_what, s_bad_what = words(s_names), words(s_bad)

    # every total quoted in text: the summary's (the engine's) value when present, else the rebuilt one; the rebuild only draws
    # the per-day curves and the day-by-day entries
    def num_or(t, mine):
        return float(t) if isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) else mine
    bh_info = info_get(info, "buy_hold")
    bh_info = bh_info if isinstance(bh_info, dict) else {}

    def hv(key, mine):
        return num_or(info_get(info, key), mine)

    def hb(key, mine):
        return num_or(bh_info.get(key), mine)
    cum, logr = hv("cum_return", rm["cum"]), hv("log_return", rm["log"])
    mdd, vol, shp = hv("max_drawdown", rm["mdd"]), hv("ann_vol", rm["vol"]), hv("sharpe", rm["sharpe"])
    b_cum, b_log = hb("cum_return", bm["cum"]), hb("log_return", bm["log"])
    b_mdd, b_vol, b_shp = hb("max_drawdown", bm["mdd"]), hb("ann_vol", bm["vol"]), hb("sharpe", bm["sharpe"])
    reward = rewards.get(0, reward_rb)
    n_trades = int(hv("trades", sum(1 for r in rb if r["n"])))      # days with a trade
    fees = hv("fees", sum(r["fee"] for r in rb))
    turnover = hv("turnover", sum(r["traded"] / r["v_pre"] for r in rb if r["n"]))
    end_v = hv("equity", rb[-1]["value"])
    b_end_v = v0 * (1 + b_cum)                               # buy-and-hold's value on the last day played
    # days the level was raised or lowered: exact, from the moves (a new level almost always trades; the trades count above is
    # the engine's)
    n_up = sum(1 for r in rb if r["level"] > r["held0"])
    n_down = sum(1 for r in rb if r["level"] < r["held0"])
    lvl_days = {L: sum(1 for x in levels if x == L) for L in LEVELS}

    # ---- labels
    def dlabel(i):
        return dates[i] if dates else f"Day {i + 1}"
    labels = [dlabel(i) for i in range(T)]

    # ---- steps and feed
    steps, feed = [], []
    rule_txt = (f"One trader, one stock{' (' + ticker + ')' if ticker else ''}, {money(v0)} in cash to start. Each trading day it sees the "
                f"last {W} dividend-adjusted closes (the last is today's), the stock's weight in its portfolio and the level it "
                f"holds, never the stock's name or a date, and picks 0, 25, 50, 75 or 100% of the portfolio's value to hold in the "
                f"stock. A new level trades whole shares at today's close with a {100 * fee:g}% fee; the level already held makes no "
                f"trade. Score: ln(final value {DIV} {money(v0)}), clipped to [{MINUS}1, 1]."
                + (f" {total_days} trading days{', ' + dates[0] + ' to ' + str(last_day) if dates else ''}." if total_days else ""))
    feed.append({"x": 0, "step": 0, "who": "Rules", "text": rule_txt})
    for i, (d, r, rbh_i) in enumerate(zip(days, rb, rbh)):
        a, h0 = r["level"], r["held0"]
        prev_close = days[i - 1]["close"] if i else (chk["pre"][-1] if chk["pre"] else None)
        chg = (r["price"] / prev_close - 1) if prev_close else None
        # the recorder's fallback flag decides (rules: Actions: no reply the game accepts, so it keeps the level held); a logged
        # move the game would not accept, with the level kept, is read the same way (warned in rebuild_days)
        fb = d["fallback"] or (d["move_level"] is None and a == h0)
        if fb:
            kind, verb = "fallback", f"no usable answer, keeps {h0}%"
        elif a > h0:
            kind, verb = "buy", f"raises {h0}% {ARROW} {a}%" + (f", buys {r['n']:,} shares" if r["n"] > 0 else ", no whole share to buy")
        elif a < h0:
            kind, verb = "sell", f"lowers {h0}% {ARROW} {a}%" + (f", sells {-r['n']:,} shares" if r["n"] < 0 else ", no whole share to sell")
        else:
            kind = "keep"
            verb = f"keeps {a}%" + (f" (weight {100 * r['w_pre']:.0f}%)" if a not in (0, 100) and abs(100 * r["w_pre"] - a) >= 0.5 else "")
        head = f"Day {d['day']}" + (f" {DOT} {dates[i]}" if dates else "") + f" {DOT} close {px(r['price'])} {DOT} {verb}"
        st = {"seat": 0, "x": i, "head": head, "kind": kind, "act": en(d["move"]) or "(no move logged)",
              "state": en(common.state_part(d["text"])) or "Not recorded.", "day": d["day"], "move_level": d["move_level"],
              "key": view_key(d["text"]), "fb": fb, "reply": d["reply"]}
        steps.append(st)
        # feed: the day's trade and the portfolio at the close
        txt = f"Close {px(r['price'])}" + (f" ({spct(chg, 2)} on the day)" if chg is not None else "") + ". "
        if kind == "buy" and r["n"] > 0:
            txt += (f"Raises {h0}% {ARROW} {a}%: buys {r['n']:,} shares for {money(r['traded'])} plus a {money(r['fee'], cents=True)} fee. ")
        elif kind == "sell" and r["n"] < 0:
            txt += (f"Lowers {h0}% {ARROW} {a}%: sells {-r['n']:,} shares for {money(r['traded'])} less a {money(r['fee'], cents=True)} fee. ")
        elif kind in ("buy", "sell"):
            txt += f"Moves {h0}% {ARROW} {a}% without a whole share to trade. "
        elif kind == "fallback":
            txt += (f"No usable answer, so the game keeps {h0}%: no trade"
                    + (f"; the stock is {100 * r['w_pre']:.1f}% of the portfolio" if h0 not in (0, 100) else "") + ". ")
        else:
            txt += f"Keeps {a}%: no trade" + (f"; the stock is {100 * r['w_pre']:.1f}% of the portfolio" if a not in (0, 100) else "") + ". "
        txt += (f"Now {r['shares']:,} shares and {money(r['cash'])} cash: value {money(r['value'])} ({spct(r['value'] / v0 - 1, 2)}); "
                f"buy-and-hold {money(rbh_i['value'])} ({spct(rbh_i['value'] / v0 - 1, 2)}).")
        feed.append({"x": i, "step": i, "who": f"Day {d['day']}", "kind": kind, "text": txt})
    end_txt = ((f"The recording stops after day {T} of {total_days}. Value then {money(end_v)}" if cut_short else
                f"Final value {money(end_v)}") + f": return {spct(cum, 2)}, log return {r3(logr)}"
               + (f", reward {r3(reward)}" if rewards.get(0) is not None else
                  "; the game did not reach its last day, so it gave no reward" if cut_short else f" (reward {r3(reward_rb)} by the rules)")
               + f". Buy-and-hold: {money(b_end_v)}, {spct(b_cum, 2)}, log return {r3(b_log)}.")
    rb_end, rbh_end = money(rb[-1]["value"]), money(rbh[-1]["value"])
    if info_get(info, "equity") is not None and (rb_end != money(end_v) or rbh_end != money(b_end_v)):
        end_txt += (f" These are the engine's values; the day-by-day entries are rebuilt from cent-rounded closes and end at "
                    f"{rb_end} and {rbh_end}.")
    feed.append({"x": T - 1, "step": T - 1, "who": "End" if not cut_short else "Stopped", "text": end_txt})

    tie = attach_calls(steps, calls, run_start(b))
    tied, n_rescued, limits, n_retried = tie["tied"], tie["rescued"], tie["limits"], tie["retried"]
    n_fallback = sum(1 for st in steps if st["fb"])
    n_flagged = sum(1 for d in days if d["fallback"])
    s_fb = summary.get("fallbacks", (run.get("extras") or {}).get("fallbacks"))
    s_fb = s_fb if isinstance(s_fb, int) and not isinstance(s_fb, bool) else None
    if s_fb is not None and s_fb != n_flagged:
        warn(f"the log flags {n_flagged} fallback moves, the summary counts {s_fb}")
    has_reason = any(st.get("thinking") for st in steps)
    reason_chars = sum(st.get("chars", 0) for st in steps)
    # guesses at the stock's identity in the reasoning: decisions naming this stock, and decisions naming any other (by company,
    # ticker or name; OTHER_NAMES)
    mention = None
    if ticker and has_reason:
        def rx(names):
            return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z0-9])")
        own = [ticker] + ([COMPANY[ticker]] if ticker in COMPANY else [])
        rx_own = rx(own)
        others = {c: rx(ns) for c, ns in OTHER_NAMES.items() if not set(ns) & set(own)}
        n_own = n_other = 0
        per = {}
        for st in steps:
            th = st.get("thinking") or ""
            n_own += bool(rx_own.search(th))
            hit = [c for c, r_ in others.items() if r_.search(th)]
            n_other += bool(hit)
            for c in hit:
                per[c] = per.get(c, 0) + 1
        top = [c for c, _ in sorted(per.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]
        mention = (n_own, own, n_other, top, len(others))

    # ---- facts for the text
    def dname(i):
        """A day by number and, when known, date (i is 0-based)."""
        return f"day {i + 1}" + (f" ({dates[i]})" if dates else "")
    pk, tr = rm["mdd_at"]                    # indices into [V_0, V_1, ...]: 0 is the start
    bpk, btr = bm["mdd_at"]

    def at(k):
        """Where a peak was: 'at the start' or 'on day N (date)' (k indexes [V_0, V_1, ...])."""
        return "at the start" if k == 0 else "on " + dname(k - 1)
    below = [i for i in range(T) if levels[i] < 100]
    nxt_moves = [prices[i + 1] / prices[i] - 1 for i in below if i + 1 < T]
    under_growth = math.prod(1 + x for x in nxt_moves) - 1 if nxt_moves else None
    up_after = sum(1 for x in nxt_moves if x > 0)
    down_after = sum(1 for x in nxt_moves if x < 0)
    stock_ret = prices[-1] / prices[0] - 1

    def longest(level):
        """(length, first index, last index) of the longest run of days at `level`, or None."""
        best, k = None, 0
        while k < T:
            if levels[k] != level:
                k += 1
                continue
            j = k
            while j + 1 < T and levels[j + 1] == level:
                j += 1
            if best is None or j - k + 1 > best[0]:
                best = (j - k + 1, k, j)
            k = j + 1
        return best
    run100, run0 = longest(100), longest(0)

    # ---- title, lede
    window_txt = (f"{dates[0]} to {str(last_day)[:10]}" if dates and last_day else f"{dates[0]} to {dates[-1]}" if dates else "")
    who = (run.get("model") or meta.get("model") or next((c.get("model") for c in calls if c.get("model")), None) or "the model").split("/")[-1]
    beat = cum > b_cum + 1e-12
    span = f"for {T} of {total_days} days (the recording stopped there)" if cut_short else f"for {nd(T, 'day')}"
    shallower = mdd < b_mdd and spct(-mdd) != spct(-b_mdd)
    gap = abs(100 * (cum - b_cum))
    title = (f"Holding 0 to 100% of {ticker or 'one stock'} day by day {span}, the model {'stands at' if cut_short else 'ends'} "
             + (f"{spct(cum)}, the same as buy-and-hold" if gap < 0.05 else f"{spct(cum)} against {spct(b_cum)} for buy-and-hold"
                + (f", with a shallower worst drawdown: {spct(-mdd)} against {spct(-b_mdd)}" if shallower and not beat
                   else f", {gap:.1f} points {'ahead' if beat else 'behind'}")))
    if test_build:
        title = "TEST BUILD from a partial call log, not the final recording. " + title
    lede = (f"One trader replays {nd(total_days or T, 'trading day')} of real daily closes of one stock"
            + (f" ({ticker}, {window_txt})" if ticker and window_txt else f" ({ticker})" if ticker else f" ({window_txt})" if window_txt else "")
            + f", starting with {money(v0)} in cash. Each day it sees only the last {W} closing prices, the stock's current weight in "
            f"its portfolio and the level it holds (no name, no date, no news) and picks how much of the portfolio to keep in the "
            f"stock: 0, 25, 50, 75 or 100%. A new level is bought or sold in whole shares at that day's close for a {100 * fee:g}% "
            f"fee; picking the level it already holds makes no trade, so holding 100% every day is exactly buy-and-hold. The score is "
            f"the log return of the whole run, ln(final value {DIV} {money(v0)}), clipped to [{MINUS}1, 1].")

    # ---- tiles
    step_ = max(1, T // 120)
    curve = [round(100 * (r["value"] / v0 - 1), 3) for r in rb]
    spark = curve[::step_] + ([curve[-1]] if (T - 1) % step_ else [])
    t1 = {"k": "Model's return", "v": spct(cum), "spark": spark,
          "s": (f"Buy-and-hold {spct(b_cum)} over the same {nd(T, 'day')}. Log return {r3(logr)} vs {r3(b_log)}"
                + (f"; reward {r3(reward)}" if rewards.get(0) is not None else "") + ".")}
    t2 = {"k": "Worst drawdown", "v": spct(-mdd),
          "s": ((f"From its peak {at(pk)} to {dname(tr - 1)}. " if tr else "The value never fell below an earlier peak. ")
                + f"Buy-and-hold {spct(-b_mdd)}. Annualized volatility {upct(vol, 1)} vs {upct(b_vol, 1)}.")}
    t3 = {"k": "Sharpe ratio", "v": f"{shp:.2f}".replace("-", MINUS),
          "s": "Buy-and-hold " + f"{b_shp:.2f}".replace("-", MINUS) + f". Daily returns, annualized with {SQRT}252, risk-free rate 0."}
    lv_txt = ", ".join(f"{L}% on {lvl_days[L]}" for L in sorted(LEVELS, reverse=True) if lvl_days[L])
    t4 = {"k": "Mean level held", "v": upct(mean_level / 100),
          "s": f"Days at each level: {lv_txt}. Trades on {n_trades} of {nd(T, 'day')}, {money(fees)} in fees."}
    tiles = [t1, t2, t3, t4]

    # ---- charts
    band = [1 if L == 100 else -1 if L == 0 else 0 for L in levels]
    markers = [{"x": i, "y": prices[i], "kind": "buy" if r["level"] > r["held0"] else "sell"}
               for i, r in enumerate(rb) if r["level"] != r["held0"]]
    hi_i = max(range(T), key=lambda i: prices[i])
    lo_i = min(range(T), key=lambda i: prices[i])
    c_price = {"title": (f"{ticker} close and the model's trades" if ticker else "Close price and the model's trades"),
               "sub": (f"Dividend-adjusted close, the last of the {W} closes in each day's prompt: {px(prices[0])} on day 1 and "
                       f"{px(prices[-1])} on day {T}, {spct(stock_ret)}; low {px(prices[lo_i])} on {dname(lo_i)}, high "
                       f"{px(prices[hi_i])} on {dname(hi_i)}. Thin gray: the average of those {W} closes. "
                       f"{chr(0x25B2)} a day the model raised its level ({n_up}), {chr(0x25BC)} a day it lowered it ({n_down}). "
                       f"Strip: blue while it holds 100% ({lvl_days[100]} days), orange while it holds 0%, all cash "
                       f"({lvl_days[0]} days); no strip at 25 to 75% ({T - lvl_days[100] - lvl_days[0]} days)."),
               "height": 280, "fmt": "price",
               "lines": [{"name": f"{W}-day average", "color": "hold", "thin": True, "v": [round(d["sma"], 4) if d["sma"] is not None else None for d in days]},
                         {"name": "Close", "color": "ink", "v": prices}],
               "band": {"v": band, "pos": "100% in the stock", "neg": "0%, all cash"}, "markers": markers}
    under_txt = ""
    if under_growth is not None and len(nxt_moves) >= 3:
        flat = len(nxt_moves) - up_after - down_after
        under_txt = (f" After the {nd(len(nxt_moves), 'decision')} below 100% (the last day aside) the stock rose the next day "
                     f"{times(up_after)} and fell {times(down_after)}" + (f" ({flat} unchanged)" if flat else "")
                     + f"; those next-day moves compound to {spct(under_growth)}"
                     + (", gains the model held only in part." if under_growth > 0 else
                        ", losses the model was partly out of." if under_growth < 0 else "."))
    c_level = {"title": "Level held after each day's decision",
               "sub": (f"Area: the level the model held after deciding (0, 25, 50, 75 or 100% of the portfolio's value). Thin line: "
                       f"the stock's actual share of the portfolio at that day's close; between trades it drifts with the price, "
                       f"because keeping a level makes no trade. Mean level {upct(mean_level / 100)}; "
                       + (f"longest stretch at 100%: {nd(run100[0], 'day')}, {dname(run100[1])} to {dname(run100[2])}; "
                          if run100 and run100[0] > 1 else "" if run100 else "never at 100%; ")
                       + (f"longest stretch at 0%: {nd(run0[0], 'day')}, {dname(run0[1])} to {dname(run0[2])}; "
                          if run0 and run0[0] > 1 else "")
                       + (f"trades on {n_trades} days ({n_up} raising the level, {n_down} lowering it)" if n_up + n_down == n_trades
                          else f"level raised on {nd(n_up, 'day')} and lowered on {n_down}, trades on {nd(n_trades, 'day')}")
                       + f", turnover {turnover:.1f} times the portfolio."
                       + under_txt),
               "height": 190, "fmt": "pct0",
               "lines": [{"name": "Level held", "color": "c1", "step": True, "area": True, "v": levels},
                         {"name": "Stock weight at the close", "color": "ink", "thin": True, "v": [round(100 * r["w"], 3) for r in rb]}]}
    fix_name = f"Fixed {mean_level:.0f}%"
    show_fix = f"{mean_level:.0f}" not in ("0", "100")      # a fixed 0% is cash; a fixed 100% is buy-and-hold itself
    c_value = {"title": "Portfolio value against buy-and-hold",
               "sub": (f"Return on the {money(v0)} start, valued at each close after the day's trade. Buy-and-hold: 100% from day 1, "
                       f"never traded again (the same as choosing 100% every day). "
                       + (f"{fix_name}: buying the model's mean level on day 1 and never trading, the same average exposure without "
                          f"any timing. " if show_fix else "")
                       + f"Day {T}: model {spct(cum, 2)} ({money(end_v)}), buy-and-hold {spct(b_cum, 2)} ({money(b_end_v)})"
                       + (f", fixed {spct(fm['cum'], 2)}" if show_fix else "")
                       + f"; the model paid {money(fees)} in fees on {nd(n_trades, 'day')} with a trade."),
               "height": 240, "fmt": "pct", "zero": True,
               "lines": [{"name": "Buy-and-hold", "color": "hold", "dash": True, "end": True, "v": [round(100 * (r["value"] / v0 - 1), 4) for r in rbh]}]
               + ([{"name": fix_name, "color": "c3", "thin": True, "end": True, "v": [round(100 * (r["value"] / v0 - 1), 4) for r in rfix]}]
                  if show_fix else [])
               + [{"name": "Model", "color": "c1", "area": True, "end": True, "v": [round(100 * (r["value"] / v0 - 1), 4) for r in rb]}]}
    c_dd = {"title": "Drawdown: distance below the highest value so far",
            "sub": (f"Each portfolio's value against its running peak (the {money(v0)} start included). "
                    + (f"Worst for the model: {spct(-mdd)}, from its peak {at(pk)} to {dname(tr - 1)}. " if tr else
                       "The model's value never fell below an earlier peak. ")
                    + (f"Worst for buy-and-hold: {spct(-b_mdd)}, from its peak {at(bpk)} to {dname(btr - 1)}." if btr else
                       "Buy-and-hold never fell below an earlier peak.")),
            "height": 200, "fmt": "pct", "zero": True,
            "lines": [{"name": "Buy-and-hold", "color": "hold", "dash": True, "v": [round(100 * x, 4) for x in bm["dd"]]},
                      {"name": "Model", "color": "c2", "area": True, "v": [round(100 * x, 4) for x in rm["dd"]]}]}
    charts = [c_price, c_level, c_value, c_dd]

    # ---- foot
    date = recorded_date(b, calls)
    seed = run.get("seed", config.get("seed"))
    variant = os.path.splitext(os.path.basename(str(config.get("variant") or "")))[0]
    full_window = bool(dates and dates[0] == DEFAULT_WINDOW[0] and str(last_day)[:10] == DEFAULT_WINDOW[1] and not cut_short)
    data_txt = (("TEST BUILD: assembled from a partial model-call log, not from the recorded bundle; the final page is built from the "
                 "bundle when the recording ends. " if test_build else "")
                + f"Data: the {date} recording of {GID}"
                + (" (" + ", ".join(x for x in [f"config {variant}" if variant else "", f"seed {seed}, which has no effect on this "
                                                 "deterministic replay" if seed is not None else ""] if x) + ")" if variant or seed is not None else "")
                + f", one seat played by {who}" + (" with reasoning on" if has_reason else "") + ". "
                + (f"Stock: {ticker}" + (f" ({COMPANY[ticker]})" if ticker in COMPANY else "") + ", dividend-adjusted daily closes. "
                   if ticker else "The stock is not named in this data. ")
                + ((f"Days played: {dates[0]} to {dates[-1]}" + (f", {T} of the window's {total_days}" if cut_short else "") + ". "
                    + ("The dates assume the rules' default window (2025-06-02 to 2026-10-19, 348 trading days), because a call log "
                       "states no window; the trader never saw a date. " if test_build and assumed else
                       "The trader never saw the ticker or a date; the dates here are the NYSE trading days from the summary's first "
                       "to its last day, which give exactly its day count. ")) if dates else f"Days are numbered, not dated ({why_nodates}). ")
                + f"The day record is rebuilt from the prompts and the moves: day d's close is the last of the {W} closes in day d's "
                f"prompt, the level is the one the next prompt says is held, and the portfolio is replayed with the rules' accounting "
                f"({money(v0)} start, whole shares, target floor(level {TIMES} value {DIV} (close {TIMES} {1 + fee:g})), "
                f"{100 * fee:g}% fee on every purchase and sale, no trade when the level is unchanged). The prompts give closes to "
                f"the cent while the engine trades at the unrounded adjusted close, so the per-day values (the curves and the "
                f"day-by-day entries) are rebuilt from cent-rounded closes: on some days the rebuilt portfolio holds one share more "
                f"or fewer than the engine's, and a day's value can be off by up to about $120 (about 0.01% of the portfolio). "
                + ("Every total quoted (title, tiles, chart notes and the End entry: return, final value, drawdown, volatility, "
                   "Sharpe, trades, turnover, fees and buy-and-hold) is the engine's, from the summary"
                   + (f"; the {fix_name.lower()} line is computed here with the same accounting" if show_fix else "") + ". "
                   if s_n else "This build has no summary, so the totals quoted are the rebuild's too. "))
    checks = []
    if chk["shift_n"]:
        checks.append(f"{chk['shift_ok']} of {chk['shift_n']} consecutive prompts agree on the {W - 1} closes they share")
    if w_n:
        checks.append(f"{w_ok} of {w_n} stock weights shown match the replay" + (f" ({w_near} more within a percent)" if w_near else ""))
    if chk["applied_n"]:
        checks.append(f"{chk['applied_ok']} of {chk['applied_n']} logged moves are the level the next prompt holds")
    if s_fb is not None:
        checks.append("no move is flagged as a fallback, and the summary counts none" if s_fb == n_flagged == 0 else
                      f"{nd(n_flagged, 'move')} flagged as {'a fallback' if n_flagged == 1 else 'fallbacks'}, "
                      + ("as many as the summary counts" if s_fb == n_flagged else f"but the summary counts {s_fb}"))
    if s_n:
        checks.append(f"{s_ok} of {s_n} values in the summary ({and_list(s_what)}) agree with the rebuild"
                      + (f" (not: {and_list(s_bad_what)})" if s_bad_what else "")
                      + ("" if eq_rel is None else "; the rebuilt final value equals the engine's to the cent" if abs(eq_rel[0]) < 0.005 else
                         f"; the rebuilt final value differs from the engine's by {money(abs(eq_rel[0]), cents=True)} "
                         f"({100 * eq_rel[1]:.2g}%)"))
    chk_txt = ("Checks: " + "; ".join(checks) + ". ") if checks else ""
    met_txt = (f"Metrics as in the rules: return = final value {DIV} start {MINUS} 1; daily returns from the {money(v0)} start (the "
               f"first is the first trade's fee); volatility and Sharpe = standard deviation and mean {DIV} standard deviation of "
               f"daily returns, {TIMES} {SQRT}252, risk-free rate 0; drawdown from the running peak. ")
    ref_txt = ""
    if full_window and ticker in REF:
        R = REF[ticker]
        ref_txt = (f"For scale (rules, {ticker} over this window): holding 25, 50 or 75% every day scores {r3(R['fixed'][1])}, "
                   f"{r3(R['fixed'][2])} or {r3(R['fixed'][3])} and 100% (buy-and-hold) {r3(R['fixed'][4])}; a uniformly random "
                   f"level every day scores {r3(R['random'][0])} on average ({r3(R['random'][1])} to {r3(R['random'][2])} over 30 "
                   f"seeds); buy-and-hold's Sharpe is {r3(R['bh'][1])} and its worst drawdown {100 * R['bh'][2]:.1f}%. Because the "
                   f"window rises, the reward grows with the level held and 100% every day is the best fixed policy. ")
    if has_reason:
        rs_txt = (f"Reasoning was saved for {tied} of {T} decisions ({reason_chars:,} characters) and is shown with each step; "
                  f"the observation is the prompt of that call.")
        if n_rescued:
            lims = sorted({x for x in limits if x})
            rs_txt += (f" In {n_rescued} decision{'s' if n_rescued != 1 else ''} the reasoning ran out "
                       + (f"at the {and_list(f'{x:,}' for x in lims)}-token output limit " if lims else "")
                       + "before an answer, and the action is the model's answer to a follow-up question (marked in the reasoning).")
        if n_retried:
            rs_txt += (f" In {nd(n_retried, 'decision')} the game rejected a reply and accepted a retry with the same prompt; the "
                       "rejected attempts are shown above the accepted one, with the reason.")
        if mention and (mention[0] or mention[2]):
            n_own, own, n_other, top, n_names = mention
            rs_txt += (f" The prompt never names the stock; the reasoning {'often' if 4 * max(n_own, n_other) >= tied else 'sometimes'} "
                       f"tries to guess it, naming {' or '.join(own)} in {n_own} of the {tied} decisions and other stocks or funds"
                       + (f" (most often {and_list(top)})" if top else "")
                       + f" in {n_other} (counted with a fixed list of {n_names} well-known names).")
        rs_txt += " "
    else:
        rs_txt = "No model-call log was given, so the steps have no reasoning. "
    fb_txt = (f"On {nd(n_fallback, 'day')} the game accepted none of the model's replies and kept the level held (rules: "
              f"Actions); {'this step is' if n_fallback == 1 else 'these steps are'} marked 'No usable answer'"
              + (" and show every rejected reply with its reasoning" if has_reason else "") + ". " if n_fallback else "")
    pre_txt = fb_txt + (f"The first prompt also shows the {len(chk['pre'])} closes before day 1, which are not charted. " if chk["pre"] else "")
    src_txt = ("Source: the execution and accounting rules of FinRL-Meta's StockTradingEnv (Liu et al., FinRL-Meta: Market Environments "
               "and Benchmarks for Data-Driven Financial Reinforcement Learning, NeurIPS 2022 Datasets and Benchmarks; 0.1% transaction "
               "cost, log-return reward, buy-and-hold baseline), re-implemented for one stock; the interface (20 closes, five levels) "
               "comes from an earlier unpublished experiment of the game's authors, so none of the paper's results apply. Prices: "
               "Yahoo Finance daily bars, dividend-adjusted.")
    d = {"id": GID, "name": "FinRL daily trading", "title": en(title), "lede": lede, "seatNames": ["Trader"], "tiles": tiles,
         "x": {"n": T, "labels": labels}, "charts": charts, "kinds": dict(KINDS), "steps": steps, "feed": feed,
         "feedTitle": "Day by day",
         "feedSub": ("Each day's close, the model's decision and the trade it made (whole shares at the close, 0.1% fee), then the "
                     "portfolio at the close against buy-and-hold. The selected step is highlighted; later entries are hidden.")}
    for st in steps:
        st.setdefault("thinking", "")
        for k in ("day", "move_level", "key", "chars", "fb", "reply"):
            st.pop(k, None)
    mtxt = who if who != "the model" else "model not recorded"
    d["eyebrow"] = (f"social_game_env {DOT} {GID} {DOT} {mtxt}" + (", reasoning on" if has_reason else "")
                    + f" {DOT} recorded {date}" + (f" {DOT} test build from a partial call log" if test_build else ""))
    # the page is one JSON file the host takes up to 16 MB: shorten the longest reasonings, visibly, if it would be larger
    n_short, gone, full = 0, 0, [st["thinking"] for st in steps]
    for _ in range(8):
        d["foot"] = en(data_txt + chk_txt + met_txt + ref_txt + rs_txt
                       + (f"To keep the page within its size limit, the {n_short} longest reasonings are shortened in the middle "
                          f"({gone:,} characters left out, marked where they were cut; the call log has them whole). " if n_short else "")
                       + pre_txt + src_txt)
        size = len(json.dumps(d, ensure_ascii=False).encode("utf-8"))
        if size <= MAX_BYTES:
            break
        total = sum(len(st["thinking"]) for st in steps)
        n_short, gone = fit_thinking(steps, max(100_000, total - int((size - MAX_BYTES) * 1.02) - 20_000), full)
    else:
        warn(f"the page is still {size:,} bytes after shortening the reasoning")
    print(f"checks: prompt shifts {chk['shift_ok']}/{chk['shift_n']}, weights {w_ok}/{w_n} (+{w_near} near), "
          f"moves applied {chk['applied_ok']}/{chk['applied_n']}, summary {s_ok}/{s_n}"
          + (f" (final value off by {eq_rel[0]:+.2f} dollars, {100 * eq_rel[1]:.2g}%)" if eq_rel is not None else "")
          + f"; fallbacks {n_fallback} (flagged {n_flagged}, summary {s_fb if s_fb is not None else 'n/a'})"
          + f"; calls tied {tied}/{T}, retried {n_retried}, rescued {n_rescued}; dates {'yes' if dates else 'no'}; {size:,} bytes")
    return d


# ------------------------------------------------------------------------------------------------ TEST HELPER (not for real pages)
def bundle_from_calls(path):
    """TEST ONLY. A stand-in bundle assembled in memory from a model-call log, for running the whole pipeline before the real
    bundle exists, in the recorder's format: one move per run of consecutive calls with the same prompt, the prompt as the view.
    The move is the first answer the game accepts (engine_level; the recorder stops asking there), serialized as the engine
    logs actions; when the game accepts none, the move keeps the level the prompt says is held and is flagged as a fallback
    with the last reply, as the recorder does. A last run without an accepted answer may have been cut off mid-retry and is
    left out. No summary and no game_over. run.json is marked test_bundle_from_calls, which the page shows in its eyebrow,
    title and foot, and carries the rules' default window (2025-06-02 to 2026-10-19, 348 days) as an assumption for the date
    labels, which the foot also states. It is never written to disk."""
    calls = load_calls(path, lenient=True)
    events = [{"_visibility": "hidden", "event": "setup", "roles": {"0": "player"}}]
    k, held = 0, 0
    while k < len(calls):
        j = k
        while j + 1 < len(calls) and calls[j + 1]["prompt"] == calls[k]["prompt"]:
            j += 1
        group = calls[k:j + 1]
        lv = next((x for x in (engine_level(c["answer"]) for c in group) if x is not None), None)
        shown = parse_view(group[0]["prompt"])["level"]
        held = shown if shown is not None else held
        ev = {"actor": 0, "event": "move", "turn": len(events), "view": group[0]["prompt"]}
        if lv is not None:
            ev["move"] = json.dumps({"action_id": "set_position", "payload": {"percent": lv}}, sort_keys=True)
            held = lv
        elif j + 1 >= len(calls):
            warn(f"the log ends with {nd(len(group), 'call')} for a decision with no accepted answer; it may have been cut off "
                 "mid-retry, so it is left out")
            break
        else:
            ev |= {"move": json.dumps({"action_id": "set_position", "payload": {"percent": held}}, sort_keys=True),
                   "fallback": True, "reply": group[-1]["answer"][:2000]}
        events.append(ev)
        k = j + 1
    t0 = next((c["t_start"] for c in calls if c.get("t_start")), None)
    run = {"model": next((c.get("model") for c in calls if c.get("model")), None), "test_bundle_from_calls": True,
           "assume_window": DEFAULT_WINDOW,
           "recorded_at": datetime.datetime.fromtimestamp(t0, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t0 else None}
    return {"events": events, "summary": {}, "run": run, "config": {}, "transcript": "", "meta": {}}, calls


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--test-from-calls":
        tb, tcalls = bundle_from_calls(sys.argv[2])
        common.write(build(tb, tcalls), sys.argv[3], date=recorded_date(tb, tcalls))
    else:
        if len(sys.argv) < 3:
            raise SystemExit(__doc__)
        src, out = sys.argv[1], sys.argv[2]
        cl_ = load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
        common.write(build(src, cl_), out, date=recorded_date(load_source(src), cl_))
