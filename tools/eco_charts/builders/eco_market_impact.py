"""Replay data for eco_market_impact (FinRL-Meta MACE stock trading with an Almgren-Chriss market-impact cost, one trader;
Abbade and Costa, "Realistic Market Impact Modeling for Reinforcement Learning Trading Environments", arXiv:2603.29086, 2026).

usage: python3 -I eco_market_impact.py BUNDLE OUT_DIR [CALLS.jsonl]
  BUNDLE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md), or a file
          inside one.
  OUT_DIR where <OUT_DIR>/eco_market_impact.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer, followup_answer when the answer came from a
          follow-up question, finish, usage). Calls are tied to decisions in time order: a call belongs to the next decision
          whose prompt shows the same decision number and portfolio value (unknown prompts agree with anything); the call whose
          answer reads as the logged orders is that decision's call, earlier calls in the same window were retries and their
          reasoning is kept above it. Its reasoning becomes the step's thinking.

test only:  python3 -I eco_market_impact.py --test-from-calls CALLS.jsonl OUT_DIR
          builds the page from a call log alone (the partial log of a run that stopped, or the growing log of a recording still
          running, whose half-written last line is left out). The stand-in bundle is assembled in memory (TEST HELPER at the
          end of this file); the page says in its eyebrow, title and foot that it is a test build. Nothing is written but
          OUT_DIR/eco_market_impact.json.

One round is one trading day and there is one seat. Decision k is made at the close of day k-1 (day 0 is the first day of the
replay) and its orders execute at the next close, day k. The prompt of decision k shows that day's portfolio (value, cash,
peak, drawdown, log return), the T-bill rate, one row per stock (close including the trader's own permanent impact, 1-day
return, 20-day volatility and average volume, shares held with their share of the portfolio, the position cap in shares, the
permanent impact in bp and the days since the stock was last traded) and the previous decision's execution (signed shares per
stock, total cost, buys not executed for lack of cash). The page's x axis is the trading day: x = k-1 is the day of decision k,
and, for a finished game, one more point for the last close, whose value comes from the summary (no prompt shows it).
Everything else is read from the prompts and checked against itself: share counts chain from day to day through the
executions, value = cash + stocks, each weight = shares x close / value, the cap = floor(cap share x value / close), the peak,
drawdown and log return follow from the values, the per-stock day counters reset exactly on the days the stock traded, and
every execution agrees with the order that caused it (the rules' Actions, with the cap rebuilt at the execution price from
the decision's prompt, see exec_caps); the summary's metrics are checked against the rebuilt ones. Every disagreement is printed
as a warning. The summary's 'days' is the backend's max_step, the number of decisions (trading days - 1)."""
import datetime, json, math, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_market_impact"
MINUS, DOT, TIMES, DIV, ARROW, NDASH, APPROX = "−", "·", "×", "÷", "→", "–", "≈"
MAX_BYTES = 15_000_000                 # the page is one JSON file; its host takes up to 16 MB
STOCK_COLORS = ("c1", "c2", "c3", "c4", "ink")
KINDS = {"buy": ["Buy orders only", "c1"], "sell": ["Sell orders only", "c2"], "both": ["Buys and sells", "c4"],
         "none": ["No orders", "hold"], "fallback": ["No usable answer, no orders", "muted"]}
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
# The five-stock setting (rules: Data, Default Configs): AAPL, CAT, CVX, JNJ, JPM with the DIA benchmark; labels S01.. follow
# the alphabetical ticker order (rules: Differences From the Source, "Anonymous stocks").
DOW5 = ("AAPL", "CAT", "CVX", "JNJ", "JPM")
COMPANY = {"AAPL": "Apple", "CAT": "Caterpillar", "CVX": "Chevron", "JNJ": "Johnson & Johnson", "JPM": "JPMorgan Chase"}
# The out-of-sample window of every config (rules: End of Game): first day, last day and the number of decisions, in the unit of
# the summary's 'days' (the backend's max_step = trading days - 1 = 396; the window has 397 trading days).
DEFAULT_WINDOW = ("2024-06-03", "2025-12-31", 396)
# Scripted policies from the rules (Phenomenon), valid only for the full window: (total return %, cost bp of the initial capital,
# of which impact bp, log-return reward); random = uniform fractions in [-1, 1] for every stock every day, seeds 0, 1, 2.
REF = {
    "dow5": {"label": "the five-stock Almgren-Chriss setting with $1 billion", "every": (52.16, 21.8, 14.8, 0.4198),
             "random": ((15.14, -11.41, -2.02), (2961, 2641, 2717), (2200, 1953, 2003), (0.1410, -0.1212, -0.0204)),
             "bench": ("DIA", 27.68)},
    "main": {"label": "the main NASDAQ-100 Almgren-Chriss setting with $1 billion", "every": (27.34, 41.9, 14.4, 0.2417),
             "once": (23.68, 18.3, 12.0, 0.2125),
             "random": ((-2.32, 1.93, -3.58), (2651, 2774, 2697), (1692, 1780, 1725), (-0.0234, 0.0191, -0.0365)),
             "bench": ("QQEW", 19.09)},
    "10bps": {"label": "the 10 bp fee setting with $1 billion", "every": (27.05, 55.0, 0.0, 0.2394),
              "random": ((0.90, 5.51, -0.24), (1952, 2024, 1980), (0, 0, 0), (0.0090, 0.0536, -0.0024)), "bench": ("QQEW", 19.09)},
    "1e7": {"label": "the Almgren-Chriss setting with $10 million", "every": (29.29, 32.3, 0.3, 0.2569),
            "random": ((10.51, 16.00, 10.14), (1075, 1121, 1094), (29, 31, 29), (0.0999, 0.1484, 0.0966)), "bench": ("QQEW", 19.09)},
}
RULES_CASH_ONLY = 6.86                 # cash only over the out-of-sample window (rules: Phenomenon; same T-bill series and days)
# The cost lines of the default parameters, as the prompt's variant line states them (backend describe_cost_model)
DEFAULT_COST_AC = ("Almgren-Chriss cost model: a trade of x shares (q = |x|) at price P, with that day's volume V and 20-day "
                   "volatility s, costs (0.5*1*s*q/V + 0.0005 + 1*s*q/V)*q*P and moves the stock's price permanently by "
                   "1*s*(x/V)*P; half of it decays every 5 trading days.")
DEFAULT_COST_FEE = "Fixed-fee cost model: every trade costs 10 bp of its value and does not move the price."
# NYSE full-day closures 2024 to 2026 (the out-of-sample window lies in 2024-2025).
NYSE_HOLIDAYS = {datetime.date(*d) for d in (
    (2024, 1, 1), (2024, 1, 15), (2024, 2, 19), (2024, 3, 29), (2024, 5, 27), (2024, 6, 19), (2024, 7, 4), (2024, 9, 2),
    (2024, 11, 28), (2024, 12, 25),
    (2025, 1, 1), (2025, 1, 9), (2025, 1, 20), (2025, 2, 17), (2025, 4, 18), (2025, 5, 26), (2025, 6, 19), (2025, 7, 4),
    (2025, 9, 1), (2025, 11, 27), (2025, 12, 25),
    (2026, 1, 1), (2026, 1, 19), (2026, 2, 16), (2026, 4, 3), (2026, 5, 25), (2026, 6, 19), (2026, 7, 3), (2026, 9, 7),
    (2026, 11, 26), (2026, 12, 25))}
TICKER_RX = re.compile(r"\b(AAPL|Apple|CAT|Caterpillar|CVX|Chevron|JNJ|Johnson|JPM|JPMorgan)\b")
# A statement in the reasoning that calls the refused-buy rule (rules: Actions, step 5: a buy the cash cannot cover is skipped but
# its price impact still applies) an exploit: a declarative sentence (not a question) with 'exploit' and words specific to refused
# buys. Quoted only verbatim, and only from a decision whose buys were in fact refused; no count is drawn from it.
EXPLOIT_RX = re.compile(r"\bexploit", re.I)
EXPLOIT_CTX_RX = re.compile(r"afford|not executed|still appl|exceeds? (?:your )?cash", re.I)


def exploit_quote(text):
    for s in re.split(r"(?<=[.!?])\s+|(?<=[.!?][\"'])\s+|\n+", text or ""):
        s = s.strip().strip("\"' ")
        if 30 <= len(s) <= 240 and not s.endswith("?") and EXPLOIT_RX.search(s) and EXPLOIT_CTX_RX.search(s):
            return s.replace('"', "'")
    return None


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
def num(s):
    return float(str(s).replace(",", "").replace(MINUS, "-"))


def usd(v, signed=False):
    """Dollars: $15.04 billion from $10 billion up, $1,288.1M from $10 million, $2.01M from $1 million, else whole dollars;
    true minus; '+' only when signed."""
    a = abs(v)
    s = (f"${a / 1e9:,.2f} billion" if a >= 1e10 else f"${a / 1e6:,.1f}M" if a >= 1e7 else f"${a / 1e6:,.2f}M" if a >= 1e6
         else f"${a:,.0f}")
    zero = a < 0.5
    if v < 0 and not zero:
        return MINUS + s
    return ("+" + s) if signed and v > 0 and not zero else s


def capital_words(v):
    for unit, word in ((1e9, "billion"), (1e6, "million")):
        if v >= unit and abs(v / unit - round(v / unit, 3)) < 1e-9:
            return f"${v / unit:g} {word}"
    return usd(v)


def spct(v, d=1):
    """A signed percentage of a fraction: +28.8%, −3.1%."""
    s = f"{abs(100 * v):.{d}f}"
    if float(s) == 0:
        return f"{0:.{d}f}%"
    return ("+" if v > 0 else MINUS) + s + "%"


def upct(v, d=1):
    return f"{100 * v:.{d}f}%"


def r3(v):
    s = f"{abs(v):.3f}"
    return (MINUS if v < 0 and float(s) != 0 else "") + s


def r4(v):
    s = f"{abs(v):.4f}"
    return (MINUS if v < 0 and float(s) != 0 else "") + s


def bp(v, d=1):
    s = f"{abs(v):,.{d}f}"
    return (MINUS if v < 0 and float(s.replace(",", "")) != 0 else "") + s + " bp"


def frac(a):
    """An order fraction: +0.5, +0.215, −1."""
    s = f"{abs(a):.3f}".rstrip("0").rstrip(".")
    return ("+" if a > 0 else MINUS if a < 0 else "") + s


def sh(q, signed=True):
    s = f"{abs(q):,}"
    return (("+" if q > 0 else MINUS if q < 0 else "") if signed else (MINUS if q < 0 else "")) + s


def nd(n, word, plural=None):
    return f"{n:,} {word if n == 1 else (plural or word + 's')}"


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


def info_get(info, *keys):
    """The first of `keys` found in the summary's info, at the top level, under 'metrics' or under 'params'."""
    for k in keys:
        for d in (info, info.get("metrics"), info.get("params")):
            if isinstance(d, dict) and d.get(k) is not None:
                return d[k]
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
    one, else the answer), plus why the call stopped ('finish'), its output tokens, start time and model."""
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
        if fu in ("None", "null"):
            fu = None
        u = r.get("usage") or {}
        if isinstance(u, str):
            m = re.search(r"'completion_tokens': (\d+)", u)
            u = {"completion_tokens": int(m.group(1))} if m else {}
        out.append({"seat": r.get("seat", common.seat_of(prompt)), "prompt": prompt or "",
                    "answer": (fu or r.get("answer") or "").strip(), "first_answer": (r.get("answer") or "").strip(),
                    "reasoning": r.get("reasoning") or "", "rescued": bool(fu),
                    "finish": r.get("finish") or r.get("finish_reason"), "out_tokens": u.get("completion_tokens"),
                    "t_start": _f(r.get("t_start")), "model": r.get("model")})
    return out


def recorded_date(b, calls=None):
    """meta.json or run.json recorded_at, else the timestamp that opens the source run's name, else the first call's start (UTC)."""
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
RX_DEC = re.compile(r"Decision (\d+) of (\d+)")
_N = r"-?[\d,]+(?:\.\d+)?"
RX_PORT = re.compile(rf"Portfolio value ({_N}): cash ({_N}) \((-?[\d.]+)%\), stocks ({_N}) \((-?[\d.]+)%\)\."
                     rf"(?: Peak so far ({_N}), drawdown (-?[\d.]+)%)?")
RX_TBILL = re.compile(r"T-bill rate (-?[\d.]+)% a year")
RX_LOGRET = re.compile(r"Log return so far:[^\n=]*= ([+-]?\d+(?:\.\d+)?)")
RX_HEAD = re.compile(r"^Stocks: (.*)$", re.M)
RX_ROW = re.compile(r"^(S\d+) \|(.*)$", re.M)
RX_LAST = re.compile(r"^Last execution: (.*)$", re.M)
RX_VAR = re.compile(r"^Variant: (.*)$", re.M)
DEFAULT_COLS = ["label", "close", "ret", "vol", "adv", "shares", "cap", "imp", "since"]


def col_key(name):
    """The table column a header name stands for; the specific names first ('cap in shares at this close' is not the close,
    'close incl. your impact' is not the impact)."""
    n = name.lower()
    if n.startswith("label"):
        return "label"
    for key, word in (("shares", "your shares"), ("cap", "cap in shares"), ("since", "days since"), ("close", "close"),
                      ("ret", "return"), ("vol", "volatility"), ("adv", "average volume"), ("imp", "impact")):
        if word in n:
            return key
    return None


def parse_row(cells, cols):
    r = {"listed": True}
    joined = " ".join(cells).lower()
    if "not listed" in joined or len(cells) < 2:
        r["listed"] = False
        return r
    for k, c in zip(cols[1:], cells):
        if k is None:
            continue
        c = c.strip()
        try:
            if k == "shares":
                m = re.match(r"(-?[\d,]+)(?: \((-?[\d.]+)%(?:, (-?[\d.]+))?\))?", c)
                if m:
                    r["shares"] = int(m.group(1).replace(",", ""))
                    r["weight"] = float(m.group(2)) if m.group(2) else None
                    r["sh_adv"] = float(m.group(3)) if m.group(3) else None
            elif k in ("ret", "vol"):
                r[k] = None if "n/a" in c else float(c.rstrip("%").replace(MINUS, "-"))
            elif k in ("cap", "since"):
                r[k] = int(c.replace(",", ""))
            else:
                r[k] = num(c)
        except ValueError:
            r[k] = None
    return r


def parse_exec(s):
    """The 'Last execution' line: {trades: {label: signed shares}, cost, rejected: [labels], other: [clauses], none_yet}."""
    if s is None:
        return None
    s = s.strip()
    out = {"trades": {}, "cost": None, "rejected": [], "other": [], "none_yet": False}
    if s.startswith("none yet"):
        out.update(none_yet=True, cost=0.0)
        return out
    for p in [p.strip() for p in s.rstrip(".").split(";")]:
        if not p or p.startswith("no trades"):
            continue
        m = re.fullmatch(rf"cost ({_N})", p)
        if m:
            out["cost"] = num(m.group(1))
            continue
        m = re.fullmatch(r"buys not executed for lack of cash: (.*)", p)
        if m:
            out["rejected"] = re.findall(r"S\d+", m.group(1))
            continue
        tr = re.findall(r"(S\d+) ([+\-−][\d,]+)", p)
        if tr and re.fullmatch(r"(?:S\d+ [+\-−][\d,]+(?:, )?)+", p):
            for lab, q in tr:
                out["trades"][lab] = out["trades"].get(lab, 0) + int(q.replace(",", "").replace(MINUS, "-"))
            continue
        out["other"].append(p)
    return out


def parse_view(text):
    """The numbers one prompt shows; missing parts are None."""
    t = text or ""
    v = {"dec": None, "T": None, "V": None, "cash": None, "cash_pct": None, "stk": None, "stk_pct": None, "peak": None,
         "dd": None, "irx": None, "lr": None, "rows": {}, "labels": [], "exec": None, "variant": None, "cols": None}
    m = RX_DEC.search(t)
    if m:
        v["dec"], v["T"] = int(m.group(1)), int(m.group(2))
    m = RX_PORT.search(t)
    if m:
        g = m.groups()
        v["V"], v["cash"], v["cash_pct"], v["stk"], v["stk_pct"] = num(g[0]), num(g[1]), float(g[2]), num(g[3]), float(g[4])
        v["peak"] = num(g[5]) if g[5] else None
        v["dd"] = float(g[6]) if g[6] else None
    m = RX_TBILL.search(t)
    if m:
        v["irx"] = float(m.group(1))
    m = RX_LOGRET.search(t)
    if m:
        v["lr"] = float(m.group(1))
    m = RX_HEAD.search(t)
    cols = DEFAULT_COLS
    if m:
        names = [x.strip() for x in m.group(1).split(" | ")]
        cols = [col_key(x) for x in names]
        if cols and cols[0] != "label":
            cols = DEFAULT_COLS
    v["cols"] = cols
    for m in RX_ROW.finditer(t):
        v["rows"][m.group(1)] = parse_row(m.group(2).split("|"), cols)
        v["labels"].append(m.group(1))
    m = RX_LAST.search(t)
    if m:
        v["exec"] = parse_exec(m.group(1))
    m = RX_VAR.search(t)
    if m:
        v["variant"] = m.group(1).strip()
    return v


def parse_variant(text):
    """The variant line: cost model, its half-spread or fee rate, the caps, the impact half-life and the reward."""
    t = text or ""
    out = {"model": None, "model_key": None, "eps": None, "cap_pct": None, "order_pct": None, "half_life": None,
           "reward_txt": None, "reward_mode": None, "cost_txt": None}
    m = re.match(r"(.+?) cost model", t)
    name = m.group(1).strip() if m else ""
    out["model"] = name or None
    low = t.lower()
    if "almgren" in low:
        out["model_key"] = "almgren_chriss"
        m = re.search(r"costs \(([^)]*)\)", t)
        if m:
            consts = [x for x in (p.strip() for p in m.group(1).split("+")) if re.fullmatch(r"\d*\.?\d+", x)]
            out["eps"] = float(consts[0]) if len(consts) == 1 else None
    elif "obizhaeva" in low:
        out["model_key"] = "obizhaeva_wang"
    elif "square" in low:
        out["model_key"] = "square_root"
    elif "fee" in low or "baseline" in low:
        out["model_key"] = "baseline"
        m = re.search(r"(\d+(?:\.\d+)?) ?bp", t)
        if m:
            out["eps"] = float(m.group(1)) / 1e4
    m = re.search(r"[Pp]osition cap (\d+(?:\.\d+)?)% of portfolio value", t)
    out["cap_pct"] = float(m.group(1)) if m else None
    m = re.search(r"order cap (\d+(?:\.\d+)?)% of", t)
    out["order_pct"] = float(m.group(1)) if m else None
    m = re.search(r"decays every (\d+(?:\.\d+)?) trading days", t)
    out["half_life"] = float(m.group(1)) if m else None
    m = re.search(r"Reward: (.*)$", t)
    if m:
        out["reward_txt"] = m.group(1).strip()
        r = out["reward_txt"].lower()
        out["reward_mode"] = ("log_return" if r.startswith("ln(") or "log return" in r else
                              "official_dsr" if "paper" in r or "official" in r else "moody_dsr" if "moody" in r else None)
    i = t.find(" Position cap")
    out["cost_txt"] = t[:i].strip() if i > 0 else None
    return out


def parse_orders(move, labels=None):
    """The non-zero orders {label: fraction} of a 'trade' action (string or dict), as the rules read it: payload exactly
    {'orders': map}, labels S01.. (known ones when `labels` is given), numbers from -1 to 1 (no booleans); None for anything else."""
    a = move
    if isinstance(a, str):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", a.strip())
        try:
            a = json.loads(s)
        except ValueError:
            i, j = s.find("{"), s.rfind("}")
            try:
                a = json.loads(s[i:j + 1]) if 0 <= i < j else None
            except ValueError:
                a = None
    if not isinstance(a, dict) or a.get("action_id") != "trade" or not isinstance(a.get("payload"), dict):
        return None
    p = a["payload"]
    if set(p) != {"orders"} or not isinstance(p["orders"], dict):
        return None
    out = {}
    for k, val in p["orders"].items():
        if not re.fullmatch(r"S\d+", str(k)) or (labels and k not in labels):
            return None
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or not -1 <= val <= 1:
            return None
        if val != 0:
            out[k] = float(val)
    return out


def view_key(text):
    """(decision number, portfolio value) of a prompt: unique per decision. None when the prompt shows neither."""
    m1, m2 = RX_DEC.search(text or ""), RX_PORT.search(text or "")
    if not m1 and not m2:
        return None
    return (int(m1.group(1)) if m1 else None, m2.group(1) if m2 else None)


# ------------------------------------------------------------------------------------------------ calendar
def trading_days(a, b):
    """NYSE trading days from a to b inclusive, or None when the range leaves 2024-2026 (holidays known only there)."""
    if a.year < 2024 or b.year > 2026 or b < a:
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


def day_dates(first, last, decisions, n):
    """(ISO dates of the first n days, reason, trading days in the window); dates None when they cannot be trusted. Used only
    when the calendar from `first` to `last` starts and ends on those days and has exactly `decisions` + 1 days (the summary's
    'days' counts decisions, the backend's max_step = trading days - 1), and at least n."""
    a, b = _date(first), _date(last)
    if a is None or b is None:
        return None, "the summary gives no first and last day", None
    cal = trading_days(a, b)
    if not cal:
        return None, f"the window {a} to {b} leaves the calendar this page knows (2024-2026)", None
    if cal[0] != a or cal[-1] != b:
        return None, f"{a} or {b} is not an NYSE trading day in this page's calendar", None
    if decisions is not None and len(cal) != int(decisions) + 1:
        return None, (f"the calendar has {len(cal)} trading days from {a} to {b}, but {int(decisions)} decisions need "
                      f"{int(decisions) + 1}"), len(cal)
    if len(cal) < n:
        return None, f"{n} days on the page but only {len(cal)} trading days from {a} to {b}", len(cal)
    return [d.isoformat() for d in cal[:n]], "", len(cal)


# ------------------------------------------------------------------------------------------------ model calls
def attach_calls(steps, calls):
    """Tie calls to steps in time order. A step's window is the run of consecutive calls, from the pointer, whose prompt shows the
    same decision number and portfolio value as the step's view (unknown prompts agree with anything); the first call in it
    whose answer reads as the logged orders is the step's call, and earlier calls in the window were failed attempts whose
    reasoning is kept above it. Without a matching answer the window's last call is used and the step says that the game applied
    another action. A call that hit the output limit (finish 'length') ends with a visible cut mark; when its action came from a
    follow-up question (rescued), a note saying so leads its reasoning. Returns (tied, rescued, limits)."""
    keys = [view_key(c.get("prompt")) for c in calls]
    ptr, tied, rescued, limits = 0, 0, 0, []

    def fits(k, st, nxt, strict=False):
        if calls[k].get("seat") not in (None, st["seat"]):
            return False
        if st["key"] is not None or strict:
            return keys[k] is not None and keys[k] == st["key"]
        return keys[k] is None or keys[k] != nxt
    for si, st in enumerate(steps):
        nxt = steps[si + 1]["key"] if si + 1 < len(steps) else None
        if ptr < len(calls) and not fits(ptr, st, nxt):
            ahead = next((k for k in range(ptr, min(len(calls), ptr + 40)) if fits(k, st, nxt, strict=True)), None)
            if ahead is None:
                warn(f"no call for decision {st['dec']}")
                continue
            warn(f"calls {ptr + 1}..{ahead} match no decision; skipped")
            ptr = ahead
        j = ptr
        while j < len(calls) and fits(j, st, nxt):
            j += 1
        window = list(range(ptr, j))
        if not window:
            continue
        if st["orders"] is not None:
            hit = next((k for k in window if parse_orders(calls[k].get("answer"), st["labels"]) == st["orders"]), None)
        else:                                                # an unusable move: the call whose answer is the logged text,
            hit = next((k for k in window if (calls[k].get("answer") or "").strip() == (st["act"] or "").strip()), None)
            if hit is None:                                   # or one that the rules cannot read either
                hit = next((k for k in window if parse_orders(calls[k].get("answer"), st["labels"]) is None), None)
        use = hit if hit is not None else window[-1]
        parts = [f"[Earlier attempt, answer: {calls[k]['answer'][:300]}]\n{calls[k].get('reasoning') or '(no reasoning returned)'}"
                 for k in range(ptr, use)]
        c = calls[use]
        finish, toks = c.get("finish"), c.get("out_tokens")
        cut = finish == "length" if finish else bool(c.get("rescued"))
        lim = f"the {toks:,}-token output limit" if cut and finish == "length" and toks else "the token limit"
        text = (c.get("reasoning") or "").rstrip()
        if text and cut:
            text += f" ... [cut off here at {lim}]"
        if c.get("rescued"):
            note = (f"[This reasoning was cut off at {lim} before it reached an answer; the action is the model's answer to a "
                    "follow-up question.]" if cut else
                    "[This reasoning ended without a usable answer; the action is the model's answer to a follow-up question.]")
            text = note + ("\n\n" + text if text else "")
            rescued += 1
            limits.append(toks if cut and finish == "length" and toks else None)
        parts.append(("[Final attempt]\n" if parts else "") + (text or ("(no reasoning returned)" if parts else "")))
        if hit is None:
            parts.append(f"[The game applied another action than this answer: {c['answer'][:300]}]")
            warn(f"decision {st['dec']}: no call answered the logged action {st['act'][:200]}")
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        st["matched"], st["answer_orders"] = hit is not None, parse_orders(c.get("answer"), st["labels"])
        st["chars"] = len(c.get("reasoning") or "")
        st["reasoning_raw"] = c.get("reasoning") or ""
        if c.get("prompt") and not st.get("has_view"):
            st["state"] = en(common.state_part(c["prompt"]))
        ptr = use + 1
        tied += 1
    if ptr < len(calls):
        warn(f"{len(calls) - ptr} calls after the last decision were not tied to any step")
    return tied, rescued, limits


def fit_thinking(steps, budget, full):
    """Keep the page within its size limit: when all reasoning together is longer than `budget` characters, shorten the longest
    ones (one common cap, the largest that fits) by removing their middle, with a visible mark. Returns (shortened, removed)."""
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


# ------------------------------------------------------------------------------------------------ the record
def rebuild(views, labels, cap_pct, V0):
    """Per day x (the day of decision x + 1): what the prompt shows, plus the consistency checks of the module docstring.
    Returns the check counts."""
    m = len(views)
    chk = {k: 0 for k in ("chain_n", "chain_ok", "ident_n", "ident_ok", "weight_n", "weight_ok", "cap_n", "cap_ok", "peak_n",
                          "peak_ok", "lr_n", "lr_ok", "since_n", "since_ok")}
    peak = V0
    for x, v in enumerate(views):
        V = v["V"]
        if V is None:
            warn(f"day {x}: the prompt shows no portfolio value")
            continue
        # value = cash + stocks; stocks = sum of shares x close; each weight; the cap
        if v["cash"] is not None and v["stk"] is not None:
            chk["ident_n"] += 1
            tot = sum((r.get("shares") or 0) * (r.get("close") or 0) for r in v["rows"].values() if r.get("listed"))
            if abs(v["cash"] + v["stk"] - V) <= 0.02 + 1e-9 * V and abs(tot - v["stk"]) <= 1e-4 * V + 1:
                chk["ident_ok"] += 1
            else:
                warn(f"decision {x + 1}: value {V} vs cash + stocks {v['cash'] + v['stk']}, stocks {v['stk']} vs shares x closes {tot}")
        for lab, r in v["rows"].items():
            if not r.get("listed") or not r.get("close"):
                continue
            if r.get("weight") is not None and r.get("shares") is not None:
                chk["weight_n"] += 1
                w = 100 * r["shares"] * r["close"] / V
                if abs(w - r["weight"]) <= 0.005 + 100 * r["shares"] * 0.005 / V + 1e-9:
                    chk["weight_ok"] += 1
                else:
                    warn(f"decision {x + 1} {lab}: weight {r['weight']}%, shares x close / value {w:.4f}%")
            if r.get("cap") is not None and cap_pct:
                chk["cap_n"] += 1
                want = cap_pct / 100 * V / r["close"]
                tol = cap_pct / 100 * V * 0.005 / r["close"] ** 2 + 1.5      # the close is shown to the cent
                if abs(r["cap"] - want) <= tol:
                    chk["cap_ok"] += 1
                else:
                    warn(f"decision {x + 1} {lab}: cap {r['cap']:,}, {cap_pct:g}% of value / close = {want:,.1f}")
        # peak, drawdown, log return
        peak = max(peak, V)
        if v["peak"] is not None:
            chk["peak_n"] += 1
            if abs(v["peak"] - peak) <= 0.01 + 1e-9 * peak and (v["dd"] is None or abs(100 * (1 - V / peak) - v["dd"]) <= 0.0051):
                chk["peak_ok"] += 1
            else:
                warn(f"decision {x + 1}: peak {v['peak']} / drawdown {v['dd']}%, rebuilt {peak} / {100 * (1 - V / peak):.4f}%")
        if v["lr"] is not None:
            chk["lr_n"] += 1
            if abs(math.log(V / V0) - v["lr"]) <= 0.000051:
                chk["lr_ok"] += 1
            else:
                warn(f"decision {x + 1}: log return {v['lr']}, ln(value / start) {math.log(V / V0):.6f}")
        # share counts chain through the executions; the per-stock day counters reset on a trade
        if x:
            prev, ex = views[x - 1], v["exec"] or {}
            tr = ex.get("trades") or {}
            for lab in labels:
                r0, r1 = prev["rows"].get(lab) or {}, v["rows"].get(lab) or {}
                if r0.get("shares") is not None and r1.get("shares") is not None:
                    chk["chain_n"] += 1
                    if r0["shares"] + tr.get(lab, 0) == r1["shares"]:
                        chk["chain_ok"] += 1
                    else:
                        warn(f"decision {x + 1} {lab}: {r0['shares']:,} held, executed {tr.get(lab, 0):+,}, now {r1['shares']:,}")
                if r0.get("since") is not None and r1.get("since") is not None:
                    chk["since_n"] += 1
                    want = 0 if tr.get(lab) else min(10, r0["since"] + 1)
                    if r1["since"] == want:
                        chk["since_ok"] += 1
                    else:
                        warn(f"decision {x + 1} {lab}: days since the last trade {r1['since']}, expected {want}")
    return chk


def want_shares(a, cap, held):
    """Signed shares the rules ask for before the volume cap (rules: Actions, steps 2-3): d = int(a x cap); sell min(-d, held)
    when d < 0, else sell a position above the cap down to it, else buy min(d, cap - held)."""
    d = int(a * cap)
    if d < 0:
        return -min(-d, held)
    if held > cap:
        return cap - held
    return min(d, cap - held)


def want_range(a, caps, held):
    """(lowest, highest) signed wanted shares over the integer caps lo..hi the order may have used. Within each branch of
    want_shares the wish moves monotonically with the cap, so the ends and the branch edges (cap = held, the smallest cap with
    d != 0) bound it."""
    lo, hi = caps
    pts = {lo, hi}
    for c in (held - 1, held, math.ceil(1 / abs(a)) - 1 if a else lo, math.ceil(1 / abs(a)) if a else lo):
        if lo <= c <= hi:
            pts.add(c)
    ws = [want_shares(a, c, held) for c in pts]
    return min(ws), max(ws)


def classify(a, q, held, caps, rejected, order_pct):
    """What happened to one stock's order at the execution close (rules: Actions, steps 1-5), from the order fraction a, the
    executed signed shares q, the shares held before and the range (lo, hi) of position caps the order may have used at the
    execution price (rebuilt from the prompts, see exec_caps). An order is cut by the volume cap when it filled fewer shares
    than the smallest wish in that range (the volume cap is the only other limit in the rules); 'excess' (more than the
    largest) disagrees with the rules. Without a cap range (no cap share known) only the sign tests apply."""
    w_lo, w_hi = want_range(a, caps, held) if (a and caps) else (None, None)
    if a > 0:
        if rejected:
            return "cash"
        if q < 0:
            return "trim"
        if q == 0:
            if caps and int(a * caps[1]) == 0:
                return "tiny"
            if caps and held >= caps[0]:
                return "at_cap"
            return "zero"
        if w_lo is not None and q < w_lo:
            return "volume"
        return "excess" if w_hi is not None and q > max(w_hi, 0) else "filled"
    if a < 0:
        if held <= 0:
            return "nothing"
        if q > 0:
            return "unexpected"
        if q == 0:
            return "tiny" if caps and int(-a * caps[1]) == 0 else "zero"
        if w_hi is not None and q > w_hi:                    # a smaller sell than the smallest the rules ask for
            return "volume"
        return "excess" if w_lo is not None and q < w_lo else "filled"
    if q < 0:
        return "trim"
    if q > 0:
        return "unexpected"
    return None


HIDDEN_IMPACT_BAND = 0.005             # when the prompt hides the impact column: the execution price within 50 bp of the close


def exec_caps(v0, v1, labels, cap_pct):
    """The position caps the orders of the decision at prompt v0 used at the next close (prompt v1), as (lo, hi) per stock,
    and whether the impact column made the price exact. The rules compute the cap before the day's trades (Actions, step 1):
    cap = int(cap share x V / P) with P = the execution close plus the impact before the trade, V = cash + shares x P. The
    backend shows the impact as perm / (close incl. impact) x 10,000, so the market close is close x (1 - bp / 10,000), and
    P = close_1 x (1 - bp_1 / 10,000) + close_0 x bp_0 / 10,000; the cash and the shares are those of prompt v0. The range
    covers the rounding of the shown numbers (closes to the cent, impact to 0.1 bp, cash to the cent)."""
    if not cap_pct or v0.get("cash") is None:
        return {}, False
    P, eP, exact = {}, {}, True
    for lab in labels:
        r0, r1 = v0["rows"].get(lab) or {}, v1["rows"].get(lab) or {}
        c1 = r1.get("close") if r1.get("listed") else None
        if not c1:
            continue
        b1 = r1.get("imp")
        c0, b0 = (r0.get("close") or 0.0), (r0.get("imp") if r0.get("listed") and r0.get("close") else 0.0)
        if b1 is None or b0 is None:
            P[lab], eP[lab], exact = c1, 0.005 + HIDDEN_IMPACT_BAND * c1, False
            continue
        P[lab] = c1 * (1 - b1 / 1e4) + c0 * b0 / 1e4
        eP[lab] = 0.005 * (1 + abs(b1) / 1e4 + abs(b0) / 1e4) + 0.05e-4 * (c1 + c0) + 1e-6 * P[lab]
    held = {lab: (v0["rows"].get(lab) or {}).get("shares") or 0 for lab in labels}
    if any(held[lab] and lab not in P for lab in labels):
        return {}, False
    V = v0["cash"] + sum(held[lab] * P[lab] for lab in P)
    eV = 0.005 + sum(held[lab] * eP[lab] for lab in P)
    k = cap_pct / 100
    out = {}
    for lab in P:
        if P[lab] - eP[lab] <= 0:
            continue
        out[lab] = (max(0, int(k * (V - eV) / (P[lab] + eP[lab]))), int(k * (V + eV) / (P[lab] - eP[lab])))
    return out, exact


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
    views = [parse_view(e.get("view") or "") for e in moves]
    m = len(moves)
    for i, v in enumerate(views):
        if v["dec"] is not None and v["dec"] != i + 1:
            warn(f"move {i + 1}'s prompt says decision {v['dec']}")
    labels = []
    for v in views:
        for lab in v["labels"]:
            if lab not in labels:
                labels.append(lab)
    labels.sort(key=lambda s: int(s[1:]))
    if not labels:
        raise SystemExit("the prompts list no stocks")
    n = len(labels)
    var = parse_variant(next((v["variant"] for v in views if v["variant"]), ""))
    params = info.get("params") if isinstance(info.get("params"), dict) else {}
    cap_pct = var["cap_pct"] if var["cap_pct"] is not None else (100 * float(params["max_stock_pct"]) if params.get("max_stock_pct") else None)
    order_pct = var["order_pct"] if var["order_pct"] is not None else (100 * float(params["max_trade_volume_pct"]) if params.get("max_trade_volume_pct") else None)
    half_life = var["half_life"] if var["half_life"] is not None else _f(params.get("perm_half_life_days"))
    model_key = var["model_key"] or params.get("impact_model")
    reward_mode = str(info_get(info, "reward_mode") or var["reward_mode"] or params.get("reward") or "log_return")
    V0 = _f(info_get(info, "initial_capital"))
    if V0 is None:
        if views[0]["dec"] in (1, None) and views[0]["V"] and views[0]["stk"] == 0:
            V0 = views[0]["V"]
        else:
            V0 = 1e9
            warn("no initial capital in the summary or the first prompt; assuming $1 billion (rules: Configuration)")
    T = next((v["T"] for v in views if v["T"]), None)
    if T is None:                                             # the summary's 'days' is the backend's max_step: decisions
        days_info = _f(info_get(info, "days"))
        T = int(days_info) if days_info else m
    over = next((e for e in events if e.get("event") == "game_over"), None)
    raw_rw = summary.get("rewards") or (run.get("result") or {}).get("rewards") or (over or {}).get("rewards") or {}
    rewards = {int(k): float(v) for k, v in raw_rw.items() if v is not None}

    # ---- the final value (after the last decision's execution): only the summary has it
    complete = m >= T and (over is not None or bool(info))
    V_end = _f(info_get(info, "portfolio_value"))
    end_from = "summary"
    if V_end is None and complete and reward_mode == "log_return" and rewards.get(0) is not None and abs(rewards[0]) < 1 - 1e-12:
        V_end, end_from = V0 * math.exp(rewards[0]), "reward"
    has_end = complete and V_end is not None
    if m > T:
        warn(f"{m} moves for {T} decisions")
    cut_short = m < T
    nx = m + (1 if has_end else 0)

    chk = rebuild(views, labels, cap_pct, V0)

    # ---- tickers: from the summary's symbol list, the config or data file, or (test builds) the rules' five-stock setting
    tickers, tick_from = None, None
    syms = info_get(info, "symbols", "tickers")
    if syms is None and isinstance(info.get("stocks"), list):
        syms = info["stocks"]
    if isinstance(syms, list) and len(syms) == n and all(isinstance(s, str) for s in syms):
        tickers, tick_from = list(syms), "summary"
    else:
        src_txt = " ".join(str(x) for x in (config.get("variant"), info_get(info, "data_file"), info_get(info, "dataset"),
                                             run.get("source_run"), meta.get("source_run")) if x)
        if "dow5" in src_txt.lower() and n == 5:
            tickers, tick_from = list(DOW5), "config"
        elif test_build and n == 5 and cap_pct == 40:
            tickers, tick_from = list(DOW5), "assumed"
    if tickers and tick_from != "summary" and sorted(tickers) != tickers:
        warn("tickers are not in alphabetical order")
    tk = {lab: (tickers[i] if tickers else None) for i, lab in enumerate(labels)}

    def lname(lab):
        return f"{lab} {tk[lab]}" if tk.get(lab) else lab
    # the rules' scripted results (REF) hold only for their configs' cost line and caps
    cost_line = var["cost_txt"] or ""
    std_caps = order_pct == 10 and cap_pct == (40 if n == 5 else 2)
    setting = None
    if not std_caps or cost_line not in (DEFAULT_COST_AC, DEFAULT_COST_FEE):
        pass
    elif n == 5 and cost_line == DEFAULT_COST_AC and abs(V0 - 1e9) < 1:
        setting = "dow5"
    elif n == 89 and cost_line == DEFAULT_COST_AC and abs(V0 - 1e9) < 1:
        setting = "main"
    elif n == 89 and cost_line == DEFAULT_COST_FEE and abs(V0 - 1e9) < 1:
        setting = "10bps"
    elif n == 89 and cost_line == DEFAULT_COST_AC and abs(V0 - 1e7) < 1:
        setting = "1e7"

    # ---- dates
    first_day, last_day = info_get(info, "first_date"), info_get(info, "last_date")
    n_dec = info_get(info, "days")                            # decisions (max_step), one fewer than the window's trading days
    assumed = run.get("assume_window")
    if not first_day and assumed:
        first_day, last_day, n_dec = assumed
    dates, why_nodates, n_window = (day_dates(first_day, last_day, n_dec, nx) if first_day else
                                    (None, "the summary gives no first day", None))
    if dates is None:
        warn(f"no dates on the page: {why_nodates}")
    full_window = bool(dates and str(first_day)[:10] == DEFAULT_WINDOW[0] and str(last_day)[:10] == DEFAULT_WINDOW[1])

    def xl(x):
        if dates:
            return dates[x]
        return "End" if x >= m else f"D{x + 1}"

    def dname(x):
        """A day by decision number and, when known, date (x is the page index)."""
        if x >= m:
            return "the last close" + (f" ({dates[x]})" if dates else "")
        return f"decision {x + 1}" + (f" ({dates[x]})" if dates else "")

    def dname_in(x):
        """The same, for use inside parentheses: 'decision 232, 2025-05-06'."""
        if x >= m:
            return "the last close" + (f", {dates[x]}" if dates else "")
        return f"decision {x + 1}" + (f", {dates[x]}" if dates else "")

    def exname(x):
        """What happened at close x (1 .. m) by its cause: the prompt of day x reports the execution of decision x's orders
        (decision x + 1 is only made at that close)."""
        when = f"executed on {dates[x]}" if dates else "executed at the final close" if x >= m else "executed at the next close"
        return f"from decision {x}'s orders, {when}"

    # ---- per-day series (x = 0 .. m-1 from the prompts)
    V = [v["V"] for v in views]
    if any(x is None for x in V):
        bad = [i + 1 for i, x in enumerate(V) if x is None]
        raise SystemExit(f"no portfolio value in the prompts of decisions {bad[:10]}")
    rows = [v["rows"] for v in views]
    execs = [v["exec"] or {"trades": {}, "cost": 0.0, "rejected": [], "other": [], "none_yet": False} for v in views]
    for x, ex in enumerate(execs):
        if ex.get("other"):
            warn(f"decision {x + 1}: unread clauses in the execution line: {ex['other']}")
        if x and ex.get("cost") is None:
            warn(f"decision {x + 1}: the execution line shows no cost")
    cost = [ex.get("cost") or 0.0 for ex in execs]
    # market close without the trader's own permanent impact: the backend shows the impact as perm / (close incl. impact) x
    # 10,000, so the market close is close x (1 - bp / 10,000)
    raw = {lab: [None] * m for lab in labels}
    for x in range(m):
        for lab in labels:
            r = rows[x].get(lab) or {}
            if r.get("listed") and r.get("close"):
                raw[lab][x] = r["close"] * (1 - (r.get("imp") or 0.0) / 1e4)

    # orders per decision, and what happened to them at the next close
    orders = [parse_orders(e.get("move"), labels) for e in moves]
    status = [{} for _ in range(m)]                           # status[i][lab] for decision i + 1, None when no order and no trade
    filled = [{} for _ in range(m)]
    tv = [0.0] * m                                            # traded value at day x (shares x the close shown that day)
    cap_how = {"exact": 0, "hidden": 0, "shown": 0}           # how the caps of the classified orders were found
    for i in range(m):
        o = orders[i] or {}
        if i + 1 >= m:
            for lab, a in o.items():
                status[i][lab] = "unknown"
            continue
        ex = execs[i + 1]
        tr = ex.get("trades") or {}
        ecaps, exact = exec_caps(views[i], views[i + 1], labels, cap_pct)
        for lab in labels:
            a = o.get(lab, 0.0)
            q = tr.get(lab, 0)
            r0, r1 = rows[i].get(lab) or {}, rows[i + 1].get(lab) or {}
            if a == 0 and q == 0:
                continue
            caps = ecaps.get(lab)
            if caps is not None:
                cap_how["exact" if exact else "hidden"] += 1
            elif r1.get("cap") is not None:                   # no cap share known: the cap shown after the trade, within 5%
                caps = (int(0.95 * r1["cap"]), int(1.05 * r1["cap"]) + 1)
                cap_how["shown"] += 1
            st_ = classify(a, q, r0.get("shares") or 0, caps, lab in (ex.get("rejected") or []), order_pct)
            if r0.get("listed") is False and a and q == 0:
                st_ = "unlisted"
            status[i][lab] = st_
            filled[i][lab] = q
            if q and r1.get("close"):
                tv[i + 1] += abs(q) * r1["close"]
            if st_ == "unexpected":
                warn(f"decision {i + 1} {lab}: {'bought' if q > 0 else 'sold'} {abs(q):,} shares against order {a}")
            elif st_ == "excess":
                warn(f"decision {i + 1} {lab}: order {a} filled {q:+,} shares, more than the rules allow "
                     f"(cap {caps[0]:,} to {caps[1]:,}, {r0.get('shares') or 0:,} held)")
        for lab in ex.get("rejected") or []:
            if o.get(lab, 0) <= 0:
                warn(f"decision {i + 1} {lab}: a buy not executed for lack of cash without a buy order")
    st_all = [s for d in status for s in d.values()]
    n_volume = sum(1 for s in st_all if s == "volume")
    n_cash = sum(1 for s in st_all if s == "cash")
    n_trim = sum(1 for s in st_all if s == "trim")
    n_nothing = sum(1 for s in st_all if s in ("nothing", "zero", "at_cap", "unlisted", "tiny"))
    ord_n = sum(1 for i in range(m - 1) for lab, s in status[i].items() if (orders[i] or {}).get(lab))
    ord_ok = sum(1 for i in range(m - 1) for lab, s in status[i].items()
                 if (orders[i] or {}).get(lab) and s not in ("unexpected", "zero", "excess"))
    for i in range(m - 1):
        for lab, s in status[i].items():
            if s == "zero":
                warn(f"decision {i + 1} {lab}: order {orders[i].get(lab)} filled no shares, and the rules give no reason here")
    # buys refused for lack of cash still move the price (rules: Actions, step 5): the impact they leave at that close and its
    # value on the shares held. In dollars per share the impact is close x bp / 10,000 (each day's bp is measured against that
    # day's close, so bp figures of two days cannot be subtracted); the refused stock has no executed trade that day, so its
    # impact after the day minus the decayed impact before is the refused order's shift
    dec_f = 0.5 ** (1 / half_life) if half_life else 1.0
    phantom = [0.0] * m
    phantom_bp = {}
    for x in range(1, m):
        for lab in execs[x].get("rejected") or []:
            r0, r1 = rows[x - 1].get(lab) or {}, rows[x].get(lab) or {}
            if r1.get("imp") is None or not r1.get("close"):
                continue
            if r0.get("listed") and r0.get("close"):
                if r0.get("imp") is None:
                    continue
                b0, perm0 = r0["imp"], r0["close"] * r0["imp"] / 1e4
            else:                                             # listed that day: no impact before
                b0, perm0 = 0.0, 0.0
            phantom[x] += (r1.get("shares") or 0) * (r1["close"] * r1["imp"] / 1e4 - dec_f * perm0)
            phantom_bp[(x, lab)] = (b0, r1["imp"])
    rej_dec = [x - 1 for x in range(1, m) if execs[x].get("rejected")]       # decision indices whose buys were refused
    phantom_tot = sum(phantom)
    exec_days = [x for x in range(1, m) if execs[x].get("trades")]
    n_buys = sum(1 for x in range(1, m) for q in execs[x]["trades"].values() if q > 0)
    n_sells = sum(1 for x in range(1, m) for q in execs[x]["trades"].values() if q < 0)
    n_rej = sum(len(execs[x].get("rejected") or []) for x in range(1, m))
    cum_cost, acc = [], 0.0
    for x in range(m):
        acc += cost[x]
        cum_cost.append(acc)
    tot_cost_seen = acc
    eps = var["eps"]
    hs = [eps * tv[x] if eps is not None else None for x in range(m)]   # the half-spread (or fixed fee) part, estimated
    hs_tot = sum(h for h in hs if h is not None) if eps is not None else None
    imp_tot = tot_cost_seen - hs_tot if hs_tot is not None else None
    tv_tot = sum(tv)

    # ---- the final close from the summary
    info_cost = _f(info_get(info, "total_cost"))
    info_lin, info_imp = _f(info_get(info, "linear_cost")), _f(info_get(info, "impact_cost"))
    # the summary adds unrounded costs, the prompts show each day's cost to the cent: up to half a cent per execution line
    cost_tol = 0.005 * sum(1 for x in range(1, m) if execs[x].get("cost") or execs[x].get("trades")) + 0.01
    last_cost = info_cost - tot_cost_seen if (info_cost is not None and has_end) else None
    if last_cost is not None and last_cost <= cost_tol:       # within the rounding: no cost on the final close
        last_cost = 0.0
    total_cost = info_cost if (info_cost is not None and has_end) else tot_cost_seen
    Vs = V + ([V_end] if has_end else [])

    # value paths: model, model with its costs added back, equal-weight buy-and-hold without costs, cash at the T-bill rate
    base = 1 if m >= 2 else None
    basket = [lab for lab in labels if base is not None and raw[lab][base]]
    bh = [None] * nx
    if basket:
        bh[0] = V0
        last_p = {lab: raw[lab][base] for lab in basket}
        for x in range(base, m):
            for lab in basket:
                if raw[lab][x]:
                    last_p[lab] = raw[lab][x]
            bh[x] = V0 * sum(last_p[lab] / raw[lab][base] for lab in basket) / len(basket)
    cash_only = [None] * nx
    cash_only[0] = V0
    for x in range(1, m):
        r_ = views[x]["irx"]
        cash_only[x] = cash_only[x - 1] * (1 + r_ / 100) ** (1 / 252) if (r_ is not None and cash_only[x - 1]) else None
    if m > 1 and cash_only[m - 1] is None:                    # a config that hides the T-bill line: no cash line
        cash_only = [None] * nx
    with_cost = [V[x] + cum_cost[x] for x in range(m)] + ([V_end + total_cost] if has_end and info_cost is not None else [None] * (nx - m))

    def ret(v):
        return None if v is None else round(100 * (v / V0 - 1), 4)

    def mdd_of(seq):
        peak, pk, out, at = V0, -1, 0.0, (-1, -1)
        for i, v in enumerate(seq):
            if v is None:
                continue
            if v > peak:
                peak, pk = v, i
            d = 1 - v / peak
            if d > out:
                out, at = d, (pk, i)
        return out, at
    mdd, mdd_at = mdd_of(Vs)
    bh_mdd, bh_at = mdd_of(bh)

    # ---- the summary's metrics against the rebuild
    s_pairs = []

    def add(name, theirs, mine, tol, kind="eq"):
        t = _f(theirs)
        if t is None or mine is None:
            return
        ok = abs(t - mine) <= tol if kind == "eq" else t >= mine - tol
        s_pairs.append((name, ok))
        if not ok:
            warn(f"summary {name} {t}, rebuilt {mine} ({'equal' if kind == 'eq' else 'at least'} expected)")
    if has_end and info:
        lr_end = math.log(V_end / V0)
        add("log return", info_get(info, "log_return"), lr_end, 2e-6)
        add("total return", info_get(info, "total_return"), V_end / V0 - 1, 2e-6)
        if reward_mode == "log_return":
            add("reward", rewards.get(0, info_get(info, "reward")), max(-1.0, min(1.0, lr_end)), 2e-6)
        add("max drawdown", info_get(info, "max_drawdown"), mdd, 2e-5)
        add("total cost", info_cost, tot_cost_seen, cost_tol, kind="ge")
        add("buys", info_get(info, "buys"), n_buys, 0, kind="ge")
        add("sells", info_get(info, "sells"), n_sells, 0, kind="ge")
        add("rejected buys", info_get(info, "rejected_buys"), n_rej, 0, kind="ge")
        add("days traded", info_get(info, "days_traded"), len(exec_days), 0, kind="ge")
        add("days", info_get(info, "days"), T, 0)                  # the backend's max_step: the number of decisions
        add("fallbacks", summary.get("fallbacks"), sum(1 for o in orders if o is None), 0, kind="ge")
        if info_lin is not None and info_imp is not None and info_cost is not None:
            add("cost split", info_lin + info_imp, info_cost, max(0.05, 1e-9 * info_cost))
        if info_lin is not None and hs_tot is not None:
            add("half-spread part", info_lin, hs_tot, 1e-3 * max(info_lin, 1.0) + 0.05, kind="ge")
    s_n, s_ok = len(s_pairs), sum(1 for p in s_pairs if p[1])
    s_bad = [p[0] for p in s_pairs if not p[1]]

    # ---- steps and feed
    steps, feed = [], []
    cost_txt = (var["cost_txt"] or "the variant's trading cost").rstrip(".")
    rule_txt = (f"One trader, {NUM_WORDS.get(n, str(n))} stocks" + (f" ({and_list(f'{lab} = {tk[lab]}' for lab in labels)})" if tickers and n <= 6 else "")
                + f", {capital_words(V0)} in cash to start, {T} decisions" + (f" ({dates[0]} to {str(last_day)[:10]})" if dates and full_window else "")
                + f". Each day it gives every stock an order fraction from {MINUS}1 to 1, a share of the position cap"
                + (f" ({cap_pct:g}% of portfolio value)" if cap_pct else "") + " to buy or sell at the next close; sells execute first"
                + (f", every order is capped at {order_pct:g}% of that day's volume" if order_pct else "")
                + ", and a buy the cash cannot cover is skipped. " + en(cost_txt) + "."
                + (f" Reward: {en(var['reward_txt'])}" if var["reward_txt"] else ""))
    feed.append({"x": 0, "step": 0, "who": "Rules", "text": rule_txt})

    def orders_words(o, many=6):
        if o is None:
            return "no usable answer, no orders"
        if not o:
            return "no orders"
        buys = [lab for lab in labels if o.get(lab, 0) > 0]
        sells = [lab for lab in labels if o.get(lab, 0) < 0]
        if len(o) > many:
            return " and ".join(x for x in ([f"buys {len(buys)} stocks"] if buys else []) + ([f"sells {len(sells)}"] if sells else []))
        parts = []
        if buys:
            parts.append("buys " + ("all " + NUM_WORDS.get(n, str(n)) if len(buys) == n and n > 1 else ", ".join(buys)))
        if sells:
            parts.append("sells " + ("all " + NUM_WORDS.get(n, str(n)) if len(sells) == n and n > 1 else ", ".join(sells)))
        return "; ".join(parts)

    for i, e in enumerate(moves):
        o = orders[i]
        kind = ("fallback" if o is None else "none" if not o else
                "both" if any(a > 0 for a in o.values()) and any(a < 0 for a in o.values()) else
                "buy" if any(a > 0 for a in o.values()) else "sell")
        res = ""
        if i + 1 < m:
            ex = execs[i + 1]
            tr = ex.get("trades") or {}
            rej = ex.get("rejected") or []
            trims = [lab for lab, s in status[i].items() if s == "trim"]
            buys_o = [lab for lab in labels if (o or {}).get(lab, 0) > 0]
            bits = []
            if tr:
                bits.append(f"cost {usd(ex.get('cost') or 0.0)}")
            elif o and rej and set(rej) >= set(buys_o) and not any(a < 0 for a in o.values()):
                bits.append("not executed: no cash")
            elif o:
                bits.append("nothing executed")
            if rej and (tr or any(a < 0 for a in (o or {}).values()) or set(rej) < set(buys_o)):
                bits.append(f"{', '.join(rej) if len(rej) <= 3 else nd(len(rej), 'buy')} not bought: no cash")
            if trims:
                bits.append(f"{', '.join(trims)} trimmed to the cap")
            res = f" {DOT} ".join(bits)
        head = f"Decision {i + 1}" + (f" {DOT} {dates[i]}" if dates else "") + f" {DOT} {orders_words(o)}" + (f" {DOT} {res}" if res else "")
        st = {"seat": 0, "x": i, "head": en(head), "kind": kind, "act": en(e.get("move") if isinstance(e.get("move"), str) else json.dumps(e.get("move"))) or "(no move logged)",
              "state": en(common.state_part(e.get("view") or "")) or "Not recorded.", "dec": i + 1, "orders": o, "labels": labels,
              "key": view_key(e.get("view") or ""), "has_view": bool(e.get("view"))}
        steps.append(st)
        # feed: the orders, what executed at the next close, and the portfolio after it
        if o is None:
            txt = "No usable answer: the game trades nothing (rules: Actions, the fallback is an empty order map)."
        elif not o:
            txt = "No orders."
        else:
            txt = "Orders: " + ", ".join(f"{lab} {frac(a)}" for lab, a in sorted(o.items(), key=lambda kv: int(kv[0][1:]))) + "."
        if i + 1 < m:
            ex = execs[i + 1]
            tr = ex.get("trades") or {}
            sell_l = [lab for lab in labels if tr.get(lab, 0) < 0]
            buy_l = [lab for lab in labels if tr.get(lab, 0) > 0]

            def one(lab):
                q = tr[lab]
                p = (rows[i + 1].get(lab) or {}).get("close")
                s_ = status[i].get(lab)
                note = {"volume": f"cut by the {order_pct:g}% volume cap" if order_pct else "", "trim": "trimmed to the cap, no order"}.get(s_, "")
                inner = ", ".join(x for x in [f"{APPROX}{usd(abs(q) * p)}" if p else "", note] if x)
                return f"{lab} {sh(q)}" + (f" ({inner})" if inner else "")
            parts = []
            if sell_l:
                parts.append("sold " + ", ".join(one(lab) for lab in sell_l))
            if buy_l:
                parts.append("bought " + ", ".join(one(lab) for lab in buy_l))
            if tr:
                c_ = ex.get("cost") or 0.0
                txt += (f" At the next close: {'; '.join(parts)}; cost {usd(c_)}"
                        + (f" ({bp(1e4 * c_ / tv[i + 1])} of the {APPROX}{usd(tv[i + 1])} traded)" if tv[i + 1] > 0 else "") + ".")
            elif o:
                txt += " Nothing executed at the next close."
            rej = ex.get("rejected") or []
            if rej:
                moved = [(lab, phantom_bp[(i + 1, lab)]) for lab in rej if (i + 1, lab) in phantom_bp]
                txt += (f" Not executed for lack of cash: {', '.join(rej)}"
                        + (f"; the refused order{'s' if len(rej) > 1 else ''} still moved the price (own impact "
                           + ", ".join(f"{lab} {bp(b0)} {ARROW} {bp(b1)}" for lab, (b0, b1) in moved)
                           + f"), worth {APPROX}{usd(phantom[i + 1])} on the shares held" if moved else "") + ".")
            nos = [lab for lab, s in status[i].items() if s in ("nothing", "zero", "at_cap", "unlisted", "tiny")]
            if nos:
                why_f = {"nothing": "nothing to sell", "at_cap": "already at the cap", "zero": "no shares",
                         "unlisted": "not listed yet", "tiny": "the fraction rounds to 0 shares"}
                txt += " Filled no shares: " + ", ".join(f"{lab} ({why_f[status[i][lab]]})" for lab in nos) + "."
            v1 = V[i + 1]
            txt += (f" Value then {usd(v1)} ({spct(v1 / V0 - 1, 2)}), cash {upct((views[i + 1]['cash_pct'] or 0) / 100)}"
                    + (f"; equal-weight buy-and-hold {spct(bh[i + 1] / V0 - 1, 2)}" if bh[i + 1] else "") + ".")
        elif o:
            txt += (" These orders execute at the final close, which no prompt shows; the summary gives the end value."
                    if has_end else " The recording stops before these orders execute.")
        st["feed_at"] = len(feed)
        feed.append({"x": i, "step": i, "who": f"Decision {i + 1}", "kind": kind, "text": en(txt)})
    if has_end:
        end_txt = (f"Final value {usd(V_end)} ({spct(V_end / V0 - 1, 2)}), log return {r4(math.log(V_end / V0))}"
                   + (f", reward {r4(rewards[0])}" if rewards.get(0) is not None else "") + "."
                   + (f" The last execution cost {usd(last_cost)}; {usd(total_cost)} in all." if last_cost else
                      f" Trading costs {usd(total_cost)} in all." if info_cost is not None else ""))
        feed.append({"x": m, "step": m - 1, "who": "End", "text": end_txt})
    elif cut_short:
        feed.append({"x": m - 1, "step": m - 1, "who": "Stopped",
                     "text": f"The recording stops at decision {m} of {T}. Value then {usd(V[-1])} ({spct(V[-1] / V0 - 1, 2)}); "
                             + (f"equal-weight buy-and-hold {spct(bh[m - 1] / V0 - 1, 2)}. " if bh[m - 1] else "")
                             + "The game did not reach its end, so it gave no reward."})

    tied, n_rescued, limits = attach_calls(steps, calls)
    for st in steps:
        if st["kind"] == "none" and st.get("matched") is False and st.get("answer_orders") is None:
            st["kind"] = "fallback"
            st["head"] = st["head"].replace(f" {DOT} no orders", f" {DOT} no usable answer, no orders", 1)
            f = feed[st["feed_at"]]
            f["kind"] = "fallback"
            f["text"] = f["text"].replace("No orders.", "No usable answer: the game trades nothing (rules: Actions, the fallback "
                                          "is an empty order map).", 1)
            st["thinking"] = st["thinking"].replace("[The game applied another action than this answer:",
                                                    "[The answer could not be read as an order map, so the game applied its fallback, no orders:", 1)
    n_fallback = sum(1 for st in steps if st["kind"] == "fallback")
    has_reason = any(st.get("thinking") for st in steps)
    reason_chars = sum(st.get("chars", 0) for st in steps)
    mention = sum(1 for st in steps if TICKER_RX.search(st.get("reasoning_raw") or "")) if (tickers and tick_from and set(tickers) == set(DOW5)) else None
    # decisions whose buys were refused for lack of cash and whose reasoning calls the refused-buy impact an exploit (verbatim)
    exploit = [(i, exploit_quote(steps[i].get("reasoning_raw"))) for i in rej_dec]
    exploit = [(i, q) for i, q in exploit if q]
    if exploit:
        i0, q0 = exploit[0]
        f = feed[steps[i0]["feed_at"]]
        f["text"] = f["text"] + en(f' Its reasoning before these orders: "{q0}"')

    # ---- facts for the text
    xe = nx - 1                                               # last point of the model's value
    xc = m - 1                                                # last point the prompts cover
    cum = Vs[xe] / V0 - 1
    cum_c = V[xc] / V0 - 1
    bh_c = bh[xc] / V0 - 1 if bh[xc] else None
    cash_c = cash_only[xc] / V0 - 1 if cash_only[xc] else None
    pchg = {lab: (raw[lab][xc] / next(p for p in raw[lab] if p) - 1) if raw[lab][xc] and any(raw[lab]) else None for lab in labels}
    mean_w = {lab: (sum((rows[x].get(lab) or {}).get("weight") or 0.0 for x in range(1, m)) / (m - 1)) if m > 1 else 0.0 for lab in labels}
    mean_cash = sum((views[x]["cash_pct"] or 0.0) for x in range(1, m)) / (m - 1) if m > 1 else 100.0
    invested = 100 - mean_cash
    near_equal = n > 1 and m > 2 and all(abs(mean_w[lab] - invested / n) <= max(3.0, 0.15 * invested / n) for lab in labels)
    max_w = max(((x, lab, (rows[x].get(lab) or {}).get("weight") or 0.0) for x in range(m) for lab in labels), key=lambda t: t[2])
    max_cash = max(range(1, m), key=lambda x: views[x]["cash_pct"] or 0.0) if m > 1 else 0
    imp_pk = max(((x, lab, (rows[x].get(lab) or {}).get("imp")) for x in range(m) for lab in labels
                  if (rows[x].get(lab) or {}).get("imp") is not None), key=lambda t: abs(t[2]), default=None)
    imp_shown = any(r.get("imp") is not None for x in range(m) for r in rows[x].values())
    own_imp = [sum((r.get("shares") or 0) * r["close"] * (r.get("imp") or 0) / 1e4
                   for r in rows[x].values() if r.get("listed") and r.get("close")) for x in range(m)]
    big_cost = max(range(m), key=lambda x: cost[x]) if m else 0
    orders_any = sum(1 for o in orders if o)
    orders_none = sum(1 for o in orders if o == {})
    orders_all = sum(1 for o in orders if o and len(o) == n)
    turnover = tv_tot / V0
    who = (run.get("model") or meta.get("model") or next((c.get("model") for c in calls if c.get("model")), None) or "the model").split("/")[-1]
    nw = NUM_WORDS.get(n, str(n))
    stocks_word = f"the {nw} stocks" if n <= 10 else f"the {n} stocks"
    exec_seen = m - 1                                         # executions a prompt reports (decisions 1 .. m-1)

    # ---- title, lede
    style = (f"Managing {stocks_word}" if exec_seen < 1 else
             f"Holding {stocks_word} at near-equal weights and trading on {len(exec_days)} of {nd(exec_seen, 'day')}" if near_equal else
             f"Trading {stocks_word} on {len(exec_days)} of {nd(exec_seen, 'day')}")
    # the title compares like with like: the same stocks held at equal weights, and the rules' scripted '+1 every day' policy
    # in the same setting and window; the benchmark index (another basket) goes to the first tile
    every = REF[setting]["every"] if (setting in REF and full_window) else None
    if has_end:
        title = (f"{style}, the model ends at {spct(cum)}" + (f" (reward {r3(rewards[0])})" if rewards.get(0) is not None else "")
                 + f" after paying {upct(total_cost / V0, 2)} of its capital in trading costs"
                 + (f"; an equal-weight buy-and-hold without costs had {spct(bh_c)} one close before the end, the model {spct(cum_c)}"
                    if bh_c is not None else "")
                 + (f"; the rules' scripted policy of ordering +1 on every stock every day ends at {spct(every[0] / 100)}"
                    + (f" (reward {r3(every[3])})" if reward_mode == "log_return" else "") if every else ""))
    else:
        title = (f"{style}, the model stands at {spct(cum_c)} after {m} of {T} decisions"
                 + (f" against {spct(bh_c)} for an equal-weight buy-and-hold without costs" if bh_c is not None else "")
                 + f", after paying {upct(tot_cost_seen / V0, 2)} of its capital in trading costs")
    if test_build:
        title = "TEST BUILD from a partial call log, not the final recording. " + title
    ac = model_key == "almgren_chriss"
    lede = (f"One trader manages a long-only portfolio of {NUM_WORDS.get(n, str(n))} real stocks"
            + (f" ({and_list(tickers)}, shown to it only as {labels[0]} to {labels[-1]})" if tickers and n <= 6 else f" (shown as {labels[0]} to {labels[-1]}, without names)")
            + f", replaying {T + 1} trading days of daily closes" + (f" from {dates[0]} to {str(last_day)[:10]}" if dates and full_window else "")
            + f" with {capital_words(V0)} in cash to start. Each day it gives every stock an order fraction from {MINUS}1 to 1, a share "
            f"of the position cap" + (f" ({cap_pct:g}% of portfolio value)" if cap_pct else "") + " to buy or sell at the next close"
            + (f", and no order may exceed {order_pct:g}% of that day's volume" if order_pct else "") + ". "
            + (f"Under the Almgren-Chriss model every trade pays a {100 * eps:g}% half-spread plus an impact charge that grows with the "
               f"order's share of the day's volume and the stock's volatility; each trade also moves the stock's price, a shift that "
               f"decays by half every {half_life:g} trading days. " if ac and eps is not None and half_life else
               f"Every trade pays the variant's trading cost ({en(var['model'] or 'see the foot')}). ")
            + ("The score is the log return of the whole replay, ln(final value " + DIV + f" {capital_words(V0)}), clipped to "
               f"[{MINUS}1, 1], so costs, impact and interest all count." if reward_mode == "log_return" else
               f"The score: {en(var['reward_txt'] or reward_mode)}."))

    # ---- tiles
    curve = [ret(v) for v in Vs]
    step_ = max(1, len(curve) // 120)
    spark = curve[::step_] + ([curve[-1]] if (len(curve) - 1) % step_ else [])
    s1 = [f"Final value {usd(V_end)}." if has_end else
          f"Value {usd(Vs[xe])} at decision {m} of {T}." if cut_short else f"Value {usd(Vs[xe])} at the last decision."]
    if bh_c is not None and has_end:
        s1.append(f"On {dname(xc)}, the last close a prompt shows: model {spct(cum_c)}, equal-weight buy-and-hold {spct(bh_c)} "
                  f"without costs" + (f", cash at the T-bill rate {spct(cash_c)}" if cash_c is not None else "") + ".")
    elif bh_c is not None:
        s1.append(f"Over the same days: equal-weight buy-and-hold {spct(bh_c)} without costs"
                  + (f", cash at the T-bill rate {spct(cash_c)}" if cash_c is not None else "") + ".")
    lr_t = f"Log return {r4(math.log(Vs[xe] / V0))}" if reward_mode == "log_return" else ""
    if rewards.get(0) is not None:
        s1.append((lr_t + ", reward " if lr_t else "Reward ") + r4(rewards[0]) + ".")
    elif cut_short:
        s1.append((lr_t + "; no reward: " if lr_t else "No reward: ") + "the game did not reach its end.")
    elif lr_t:
        s1.append(lr_t + ".")
    bench = _f(info_get(info, "benchmark_return")) if has_end else None
    bname = params.get("benchmark") or (REF[setting]["bench"][0] if setting in REF else None)
    if bench is not None:
        s1.append(f"The {bname + ' ' if bname else ''}benchmark returned {spct(bench)} over the same days (summary).")
    t1 = {"k": "Model's return", "v": spct(cum), "spark": spark, "s": " ".join(s1)}
    cost_s = (f"{bp(1e4 * total_cost / V0)} of the {capital_words(V0)} start" + (f", {usd(last_cost)} of it on the final close" if last_cost else "")
              + f". Paid on {len(exec_days)} of {nd(exec_seen, 'execution')} the prompts report")
    if info_lin is not None and info_imp is not None and has_end:
        cost_s += f": {bp(1e4 * info_lin / V0)} half-spread, {bp(1e4 * info_imp / V0)} size-dependent impact (summary)"
    elif hs_tot is not None and ac:
        cost_s += (f": {APPROX}{bp(1e4 * hs_tot / V0)} half-spread ({100 * eps:g}% of the {APPROX}{usd(tv_tot)} traded), "
                   f"{APPROX}{bp(1e4 * imp_tot / V0)} size-dependent impact")
    if last_cost and last_cost > cost[big_cost]:
        cost_s += f". Costliest day: {usd(last_cost)}, {exname(m)}."
    else:
        cost_s += f". Costliest day: {usd(cost[big_cost])}, {exname(big_cost)}." if cost[big_cost] > 0 else "."
    if exec_seen < 1 and not has_end:
        cost_s = "No execution yet: the first orders execute at the next close, which no prompt shows."
    t2 = {"k": "Trading costs", "v": usd(total_cost), "s": cost_s}
    dt_info = _f(info_get(info, "days_traded")) if has_end else None
    t3 = {"k": "Days with trades",
          "v": f"{int(dt_info)} of {T}" if dt_info is not None else f"{len(exec_days)} of {exec_seen}" if exec_seen else "none yet",
          "s": (f"Orders on {orders_any} of {nd(m, 'decision')}; they execute at the next close, which no prompt shows yet."
                if exec_seen < 1 and not has_end else
                f"Orders on {orders_any} of {nd(m, 'decision')}" + (f" (every stock on {orders_all})" if orders_all and n > 1 else "")
                + f", none on {orders_none}" + (f", no usable answer on {n_fallback}" if n_fallback else "") + ". "
                + (f"{nd(n_volume, 'order')} cut by the {order_pct:g}% volume cap; " if order_pct else "")
                + f"{nd(n_cash, 'buy')} not executed for lack of cash"
                + (f", whose price impact still applied ({APPROX}{usd(phantom_tot)} added to the holdings' value on those days)"
                   if n_cash and phantom_bp else "")
                + (f"; {nd(n_trim, 'position')} trimmed to the cap" if n_trim else "")
                + f". Traded {APPROX}{usd(tv_tot)}, {turnover:.2f} times the starting capital"
                + (" (to the last prompt)" if has_end else "") + ".")}
    a_vol, a_shp = _f(info_get(info, "annualized_volatility")), _f(info_get(info, "annualized_sharpe"))
    s4 = [(("From the start" if mdd_at[0] < 0 else f"From its peak on {dname(mdd_at[0])}") + f" to {dname(mdd_at[1])}."
           if mdd_at[1] >= 0 and mdd > 0 else "The value never fell below an earlier peak.")]
    if bh_c is not None:
        s4.append(f"Equal-weight buy-and-hold {spct(-bh_mdd)} over the days the prompts cover.")
    if has_end and a_vol is not None and a_shp is not None:
        s4.append(f"Annualized volatility {upct(a_vol)}, Sharpe {f'{a_shp:.2f}'.replace('-', MINUS)} (summary).")
    t4 = {"k": "Worst drawdown", "v": spct(-mdd), "s": " ".join(s4)}
    tiles = [t1, t2, t3, t4]

    # ---- charts
    labels_x = [xl(x) for x in range(nx)]
    pad = [None] * (nx - m)
    end_note = (f" The last point ({xl(m)}) is the final value from the summary" + (" (from the reward)" if end_from == "reward" else "")
                + "; no prompt shows that close, so the other lines stop one close earlier." if has_end else "")
    c_value = {"title": "Portfolio value against buy-and-hold",
               "sub": (f"Return on the {capital_words(V0)} start at each close, after that day's execution, interest and impact decay, as "
                       f"the prompts show it (valued at closes that include the model's own price impact). Buy-and-hold: "
                       + (f"{upct(1 / len(basket), 0)} of the start in each of the {len(basket)} stocks" if basket else "equal weights")
                       + f" at the first execution close ({xl(1) if m > 1 else 'none'}), never traded, no costs, "
                       + ("at market closes without the model's impact" if imp_shown else
                          "at the prompts' closes (this variant's prompt hides the model's impact, so they include it)")
                       + f". Cash: the start kept in cash at the T-bill rate the prompts show. With costs added back: the "
                       f"model's value plus the trading costs paid so far, so the gap to the model is those costs ("
                       f"{bp(1e4 * cum_cost[xc] / V0)} of the start by {dname_in(xc)}). On {dname(xc)}: model {spct(cum_c, 2)}"
                       + (f", buy-and-hold {spct(bh_c, 2)}" if bh_c is not None else "") + (f", cash {spct(cash_c, 2)}" if cash_c is not None else "")
                       + (f". That day {usd(own_imp[xc])} of the model's value was its own permanent impact on the prices it holds."
                          if imp_shown else ".")
                       + end_note),
               "height": 260, "fmt": "pct", "zero": True,
               "lines": ([{"name": "Equal-weight buy-and-hold", "color": "hold", "dash": True, "end": True, "v": [ret(v) for v in bh]}] if basket else [])
               + ([{"name": "Cash at the T-bill rate", "color": "c3", "thin": True, "end": True, "v": [ret(v) for v in cash_only]}]
                  if any(v is not None for v in cash_only) else [])
               + [{"name": "Model, costs added back", "color": "ink", "thin": True, "v": [ret(v) for v in with_cost]},
                  {"name": "Model", "color": "c1", "area": True, "end": True, "v": curve}]}
    hl = labels if n <= 5 else sorted(sorted(labels, key=lambda lab: -mean_w[lab])[:5], key=lambda lab: int(lab[1:]))
    col = {lab: STOCK_COLORS[k] for k, lab in enumerate(hl)}
    others = [lab for lab in labels if lab not in hl]
    p_lines = []
    for lab in hl:
        b0 = next((p for p in raw[lab] if p), None)
        p_lines.append({"name": lname(lab), "color": col[lab], "end": True,
                        "v": [round(100 * (p / b0 - 1), 4) if p and b0 else None for p in raw[lab]] + pad})
    if others:
        avg = []
        for x in range(m):
            vals = [raw[lab][x] / next(p for p in raw[lab] if p) for lab in labels if raw[lab][x]]
            avg.append(round(100 * (sum(vals) / len(vals) - 1), 4) if vals else None)
        p_lines.append({"name": f"All {n} stocks, average", "color": "hold", "dash": True, "end": True, "v": avg + pad})
    ranked = sorted([lab for lab in labels if pchg[lab] is not None], key=lambda lab: -pchg[lab])
    c_price = {"title": "Stock prices, change since the first close",
               "sub": ((f"Each stock's market close without the model's own impact (the prompt's close {TIMES} (1 {MINUS} impact in "
                        f"bp {DIV} 10,000))" if imp_shown else "Each stock's close as the prompts show it (this variant's prompt hides "
                        "the model's own impact, so the closes include it)") + f", from {xl(0)} to {dname(xc)}: " + ", ".join(f"{lname(lab)} {spct(pchg[lab])}" for lab in ranked[:8])
                       + (f" (the {len(others)} other stocks are averaged in the dashed line)" if others else "") + ". "
                       + (f"The model's own impact moved a price by at most {bp(abs(imp_pk[2]))} ({lname(imp_pk[1])}, "
                          f"{'down' if imp_pk[2] < 0 else 'up'}, at the close "
                          + (f"of {dates[imp_pk[0]]} " if dates else "") + f"where decision {imp_pk[0]}'s orders executed; the "
                          f"impact adds up its earlier trades and refused buys); half of it fades every {half_life:g} trading days. "
                          if imp_pk and abs(imp_pk[2]) > 0 and half_life and imp_pk[0] >= 1 else "")
                       + "Buy-and-hold in the chart above is the average of these stocks from the first execution close."),
               "height": 240, "fmt": "pct", "zero": True, "lines": p_lines}

    # orders grid: one row per stock, one column per decision
    why = {"nothing": "nothing held, so nothing to sell", "at_cap": "already at the cap", "zero": "filled no shares",
           "unlisted": "not listed yet: no volume, no shares", "cash": "not executed: the cash did not cover it",
           "volume": f"cut by the order cap ({order_pct:g}% of that day's volume)" if order_pct else "cut by the order cap",
           "trim": "position above the cap, sold down to it", "filled": "filled", "tiny": "the fraction rounds to 0 shares",
           "unknown": ("executes at the final close, which no prompt shows" if has_end else
                       "the recording stops before this order executes"),
           "unexpected": "a trade against the order's direction (unexpected)",
           "excess": "filled more shares than the rules allow (unexpected)"}
    cells = []
    grid_labels = labels if n <= 8 else hl
    for lab in grid_labels:
        row_c = []
        for x in range(nx):
            if x >= m:
                row_c.append(None)
                continue
            o = orders[x] or {}
            a = o.get(lab, 0.0)
            s_ = status[x].get(lab)
            if s_ is None:
                row_c.append(None)
                continue
            r0 = rows[x].get(lab) or {}
            r1 = (rows[x + 1].get(lab) or {}) if x + 1 < m else {}
            q = filled[x].get(lab)
            tip = (f"Decision {x + 1}: " + (f"order {frac(a)} of the cap ({r0.get('cap', 0):,} shares at this close)" if a else "no order")
                   + f"; {why.get(s_, s_)}")
            if q:
                p1 = r1.get("close")
                tip += f": {sh(q)} shares" + (f", {APPROX}{usd(abs(q) * p1)}" if p1 else "")
            if s_ == "cash" and (x + 1, lab) in phantom_bp:
                b0, b1 = phantom_bp[(x + 1, lab)]
                tip += f", but the order still moved the price: own impact {bp(b0)} {ARROW} {bp(b1)}"
            if r1.get("shares") is not None:
                tip += (f". After: {r1['shares']:,} shares, {r1.get('weight', 0):.2f}% of the portfolio"
                        + (f" (cap {cap_pct:g}%)" if cap_pct else "") + (f", own impact {bp(r1['imp'])}" if r1.get("imp") is not None else ""))
            if r0.get("close"):
                tip += f". Close at the decision {r0['close']:,.2f}" + (f", days since its last trade: {r0['since']}" if r0.get("since") is not None else "")
            o_ = round(0.3 + 0.7 * min(1.0, abs(a)) ** 0.5, 3)
            if s_ == "cash":
                cell = {"c": "c4", "o": 0.95, "t": "$", "tip": tip}
            elif s_ in ("nothing", "zero", "at_cap", "unlisted", "tiny"):
                cell = {"c": "hold", "o": 0.7, "t": "0", "tip": tip}
            elif s_ == "trim" and a == 0:
                cell = {"c": "ink", "o": 0.85, "t": "cap", "tip": tip}
            else:
                cell = {"c": "c1" if a > 0 else "c2", "o": o_, "t": frac(a), "tip": tip}
            row_c.append(cell)
        cells.append(row_c)
    row_names = [lname(lab) for lab in grid_labels]
    if n > 8:                                                 # the other stocks in one row: net direction of their orders
        row_c = []
        for x in range(nx):
            o = (orders[x] or {}) if x < m else {}
            ob = [lab for lab in others if o.get(lab, 0) > 0]
            os_ = [lab for lab in others if o.get(lab, 0) < 0]
            if not ob and not os_:
                row_c.append(None)
                continue
            row_c.append({"c": "c4" if ob and os_ else "c1" if ob else "c2", "o": 0.8,
                          "tip": f"Decision {x + 1}: buy orders on {len(ob)}, sell orders on {len(os_)} of the {len(others)} other stocks"})
        cells.append(row_c)
        row_names.append(f"{len(others)} others")
    big_ph = max(range(1, m), key=lambda x: phantom[x]) if m > 1 and any(phantom) else None
    n_buy_o = sum(1 for o in orders if o for a in o.values() if a > 0)
    n_sell_o = sum(1 for o in orders if o for a in o.values() if a < 0)
    c_grid = {"title": "The model's orders, stock by stock",
              "sub": (f"One column per decision: the order fraction for each stock, a share of the position cap"
                      + (f" ({cap_pct:g}% of portfolio value)" if cap_pct else "") + ", executed at the next close. Blue: buy order, "
                      f"orange: sell order, darker for a larger fraction; yellow ($): a buy not executed because the cash did not cover it; "
                      f"gray (0): an order that filled no shares; dark (cap): no order, but the position was over the cap and was sold "
                      f"down to it; blank: no order. Hover for the shares filled and the position after. {nd(n_buy_o, 'buy order')} and "
                      f"{nd(n_sell_o, 'sell order')} in {nd(orders_any, 'decision')}"
                      + (f"; {nd(n_volume, 'order')} filled less than asked although the cap and the cash allowed more, so the "
                         f"{order_pct:g}% volume cap bound (those trades took that share of the day's volume)" if order_pct else "")
                      + f"; {nd(n_cash, 'buy')} not executed for lack of cash" + (f"; {nd(n_nothing, 'order')} filled no shares" if n_nothing else "")
                      + (f"; {nd(n_trim, 'trim')} to the cap" if n_trim else "") + "."
                      + (f" A refused buy still moves the price (rules: Actions, kept from the official code): the {nd(n_cash, 'refused buy')} "
                         f"in {nd(len(rej_dec), 'decision')} raised the model's own impact on those stocks and with it the value of its "
                         f"holdings, by {APPROX}{usd(phantom_tot)} in all on those days" + (f" ({APPROX}{usd(phantom[big_ph])} from the "
                         f"orders of decision {big_ph})" if big_ph is not None and len(rej_dec) > 1 else "")
                         + f", gains that then fade by half every {half_life:g} trading days." if n_cash and phantom_bp and half_life else "")
                      + (f" The reasoning of decision {exploit[0][0] + 1}, whose buys were refused, says: \"{exploit[0][1]}\""
                         if exploit else "")),
              "grid": {"rows": row_names, "cells": cells, "rowHeight": 24,
                       "legend": [["Buy order", "c1"], ["Sell order", "c2"], ["Buy not executed: no cash", "c4"],
                                  ["Filled no shares", "hold"], ["Trimmed to the cap", "ink"]]}}

    # holdings
    h_lines = []
    for lab in hl:
        h_lines.append({"name": lname(lab), "color": col[lab],
                        "v": [((rows[x].get(lab) or {}).get("weight")) for x in range(m)] + pad})
    if others:
        h_lines.append({"name": f"Other {len(others)} stocks together", "color": "hold", "thin": True,
                        "v": [round(sum((rows[x].get(lab) or {}).get("weight") or 0.0 for lab in others), 4) for x in range(m)] + pad})
    h_lines.append({"name": "Cash", "color": "hold", "dash": True, "v": [views[x]["cash_pct"] for x in range(m)] + pad})
    if cap_pct:
        h_lines.append({"name": f"Position cap {cap_pct:g}%", "color": "muted", "dash": True, "thin": True, "v": [cap_pct] * nx})
    build_up = next((x for x in range(1, m) if (views[x]["cash_pct"] or 100) < 5), None)
    c_hold = {"title": "Holdings: each stock's share of the portfolio",
              "sub": (f"Shares {TIMES} close {DIV} portfolio value at each close, as the prompts show it, with cash dashed"
                      + (f" and the {cap_pct:g}% position cap" if cap_pct else "") + ". "
                      + (f"Cash first fell below 5% on {dname(build_up)}. " if build_up is not None else "")
                      + "Mean over the days after the first execution: " + ", ".join(f"{lname(lab)} {mean_w[lab]:.1f}%" for lab in hl)
                      + (f", the {len(others)} others {sum(mean_w[lab] for lab in others):.1f}% together" if others else "")
                      + f", cash {mean_cash:.1f}%. Largest position: {lname(max_w[1])} at {max_w[2]:.2f}% on {dname(max_w[0])}"
                      + (f"; most cash: {views[max_cash]['cash_pct']:.2f}% on {dname(max_cash)}" if m > 1 else "") + "."),
              "height": 240, "fmt": "pct0", "lines": h_lines}
    charts = [c_value, c_price, c_grid, c_hold]

    # ---- foot
    date = recorded_date(b, calls)
    seed = run.get("seed", config.get("seed"))
    variant = os.path.splitext(os.path.basename(str(config.get("variant") or "")))[0]
    data_txt = (("TEST BUILD: assembled from a partial model-call log, not from the recorded bundle; there is no summary and no "
                 "final value yet, and the final page is built from the bundle when the recording ends. " if test_build else "")
                + f"Data: the {date} recording of {GID}"
                + (" (" + ", ".join(x for x in [f"config {variant}" if variant else "", f"seed {seed}, which has no effect on this "
                                                 "deterministic replay" if seed is not None else ""] if x) + ")" if variant or seed is not None else "")
                + f", one seat played by {who}" + (" with reasoning on" if has_reason else "") + ". "
                + (f"Stocks: {and_list(f'{lab} = {tk[lab]} ({COMPANY.get(tk[lab], tk[lab])})' for lab in labels)}"
                   + (", from the summary" if tick_from == "summary" else ", the rules' five-stock setting (labels in alphabetical "
                      "ticker order)" + (", identified from the prompt (five stocks, 40% cap) because a call log names no config"
                                         if tick_from == "assumed" else ""))
                   + "; the trader saw only the labels. " if tickers and n <= 6 else
                   f"{n} stocks, shown to the trader as {labels[0]} to {labels[-1]}; the page does not name them. ")
                + (f"Days: {dates[0]} to {dates[-1]}" + (f", {nx} of the window's {n_window}" if cut_short and n_window else "") + ". "
                   + (f"The dates assume the rules' window ({DEFAULT_WINDOW[0]} to {DEFAULT_WINDOW[1]}, {DEFAULT_WINDOW[2] + 1} trading "
                      f"days for {DEFAULT_WINDOW[2]} decisions), because a call log states no window; the trader never saw a date. "
                      if test_build and assumed else
                      "The trader never saw a date; the dates are the NYSE trading days from the summary's first to its last day, one "
                      "more than its day count, which counts decisions. ") if dates else f"Days are numbered by decision, not dated ({why_nodates}). ")
                + f"Setting, as the prompt states it: {en(cost_txt)}."
                + (f" Position cap {cap_pct:g}% of portfolio value per stock" if cap_pct else "")
                + (f"; order cap {order_pct:g}% of the execution day's volume" if order_pct else "") + ". "
                + (f"Reward: {en(var['reward_txt'])} " if var["reward_txt"] else f"Reward: {reward_mode}. "))
    how_txt = (f"Decision k is made at a close and executes at the next; the x axis is the trading day, labelled by the day of each "
               f"decision" + (", plus the final close" if has_end else "") + ". Values, cash, holdings, closes, impact, executions and "
               f"costs are the prompts' numbers"
               + (f"; the final value is the summary's{' (derived from the reward)' if end_from == 'reward' else ''}" if has_end else "")
               + (f". Market closes without the model's impact = close {TIMES} (1 {MINUS} impact in bp {DIV} 10,000), as the prompt "
                  f"gives the impact relative to the close that includes it" if imp_shown else
                  ". The prompt hides the model's impact, so the closes used for buy-and-hold and the stock prices include it")
               + f". Traded value = shares "
               f"{TIMES} the close shown after the trade"
               + (f", and the half-spread = {100 * eps:g}% of it, the rest of each day's cost being the size-dependent Almgren-Chriss "
                  f"impact terms (both {APPROX}: the execution price also carries the impact before the trade)" if ac and eps is not None else "")
               + ". An order counts as cut by the volume cap when it was not refused for cash and filled fewer shares than the rules' "
               "steps 1-3 ask for at the execution close, with the cap rebuilt from the cash and shares of the decision's prompt and "
               "the execution price (that day's market close plus the impact before the trade), even at the low end of what the "
               "rounding of the prompts' numbers allows; the rules leave no other limit"
               + (f"; where a prompt hides the impact, the execution price is taken within {HIDDEN_IMPACT_BAND * 1e4:g} bp of the close"
                  if cap_how["hidden"] else "")
               + ("; where no cap share is stated, the cap shown after the trade, within 5%, stands in" if cap_how["shown"] else "")
               + ". ")
    checks = []
    if chk["chain_n"]:
        checks.append(f"{chk['chain_ok']:,} of {chk['chain_n']:,} share counts follow from the day before and the execution")
    if chk["ident_n"]:
        checks.append(f"{chk['ident_ok']:,} of {chk['ident_n']:,} values equal cash plus shares {TIMES} closes")
    if chk["weight_n"]:
        checks.append(f"{chk['weight_ok']:,} of {chk['weight_n']:,} weights and {chk['cap_ok']:,} of {chk['cap_n']:,} caps match the rules' formulas")
    if chk["peak_n"] or chk["lr_n"]:
        checks.append(f"{chk['peak_ok']:,} of {chk['peak_n']:,} peaks and drawdowns and {chk['lr_ok']:,} of {chk['lr_n']:,} log returns follow from the values")
    if chk["since_n"]:
        checks.append(f"{chk['since_ok']:,} of {chk['since_n']:,} per-stock day counters reset exactly on the days the stock traded")
    if ord_n:
        checks.append(f"{ord_ok:,} of {ord_n:,} orders agree with what executed")
    if s_n:
        checks.append(f"{s_ok} of {s_n} summary values agree with the rebuild" + (f" (not: {and_list(s_bad)})" if s_bad else ""))
    chk_txt = ("Checks: " + "; ".join(checks) + ". ") if checks else ""
    ref_txt = ""
    if setting in REF and full_window and not cut_short:
        R = REF[setting]
        e_ = R["every"]
        sp = lambda v: spct(v / 100, 2)                      # noqa: E731  (a percentage given in percent)
        ref_txt = (f"For scale (rules, scripted policies in {R['label']} over the same window): +1 on every stock every day, the "
                   f"official example's buy-and-hold action, returns {sp(e_[0])} at {e_[1]:g} bp of costs ({e_[2]:g} bp impact)"
                   + (f", reward {r4(e_[3])}" if reward_mode == "log_return" else "")
                   + (f"; buying once and holding returns {sp(R['once'][0])} at {R['once'][1]:g} bp" if "once" in R else "")
                   + f"; uniformly random fractions return {', '.join(sp(v) for v in R['random'][0])} (seeds 0, 1, 2) at "
                   f"{min(R['random'][1]):,} to {max(R['random'][1]):,} bp; cash only returns {sp(RULES_CASH_ONLY)}"
                   + (f" (this page's cash line: {spct(cash_c, 2)})" if cash_c is not None else "")
                   + f"; the benchmark {R['bench'][0]} returned {sp(R['bench'][1])}. ")
    if n == 5 and tickers and set(tickers) == set(DOW5):
        ref_txt += ("This five-stock setting is not in the paper (the rules add it as a short prompt for quick model runs); the "
                    "paper's experiments trade the NASDAQ-100 with trained RL agents. ")
    if has_reason:
        rs_txt = (f"Reasoning was saved for {tied} of {m} decisions ({reason_chars:,} characters) and is shown with each step; "
                  f"the observation is the prompt of that decision.")
        if n_rescued:
            lims = sorted({x for x in limits if x})
            rs_txt += (f" In {n_rescued} decision{'s' if n_rescued != 1 else ''} the reasoning ran out "
                       + (f"at the {and_list(f'{x:,}' for x in lims)}-token output limit " if lims else "")
                       + "before an answer, and the action is the model's answer to a follow-up question (marked in the reasoning).")
        if mention is not None:
            rs_txt += (f" The prompt names no stock; the reasoning names at least one of the five companies (by ticker or name) in "
                       f"{mention} of the {tied} decisions.")
        rs_txt += " "
    else:
        rs_txt = "No model-call log was given, so the steps have no reasoning. "
    fb_txt = (f"{nd(n_fallback, 'answer')} could not be read as an order map; the game then trades nothing (rules: Actions). "
              if n_fallback else "")
    src_txt = ("Source: Abbade and Costa, Realistic Market Impact Modeling for Reinforcement Learning Trading Environments (arXiv:2603.29086, "
               "2026), and FinRL-Meta's MACEStockTradingEnv, ported to Python with the same execution, costs and impact; prices are "
               "Yahoo Finance daily closes, dividend-adjusted.")
    d = {"id": GID, "name": "Market-impact trading (FinRL-Meta MACE)", "title": en(title), "lede": en(lede), "seatNames": ["Trader"],
         "tiles": tiles, "x": {"n": nx, "labels": labels_x}, "charts": charts, "kinds": dict(KINDS), "steps": steps, "feed": feed,
         "feedTitle": "Decision by decision",
         "feedSub": ("Each decision's order fractions, what executed at the next close (sells first, then buys), its cost, and the "
                     "portfolio at that close against buy-and-hold. The selected step is highlighted; later entries are hidden.")}
    for st in steps:
        st.setdefault("thinking", "")
        for k in ("dec", "orders", "labels", "key", "has_view", "chars", "feed_at", "reasoning_raw", "matched", "answer_orders"):
            st.pop(k, None)
    mtxt = who if who != "the model" else "model not recorded"
    d["eyebrow"] = (f"social_game_env {DOT} {GID} {DOT} {mtxt}" + (", reasoning on" if has_reason else "")
                    + f" {DOT} recorded {date}" + (f" {DOT} test build from a partial call log" if test_build else ""))
    n_short, gone, full = 0, 0, [st["thinking"] for st in steps]
    size = 0
    for _ in range(8):
        d["foot"] = en(data_txt + how_txt + chk_txt + ref_txt + rs_txt
                       + (f"To keep the page within its size limit, the {n_short} longest reasonings are shortened in the middle "
                          f"({gone:,} characters left out, marked where they were cut; the call log has them whole). " if n_short else "")
                       + fb_txt + src_txt)
        size = len(json.dumps(d, ensure_ascii=False).encode("utf-8"))
        if size <= MAX_BYTES:
            break
        total = sum(len(st["thinking"]) for st in steps)
        n_short, gone = fit_thinking(steps, max(100_000, total - int((size - MAX_BYTES) * 1.02) - 20_000), full)
    else:
        warn(f"the page is still {size:,} bytes after shortening the reasoning")
    print(f"checks: chain {chk['chain_ok']}/{chk['chain_n']}, identity {chk['ident_ok']}/{chk['ident_n']}, weights "
          f"{chk['weight_ok']}/{chk['weight_n']}, caps {chk['cap_ok']}/{chk['cap_n']}, peaks {chk['peak_ok']}/{chk['peak_n']}, "
          f"log returns {chk['lr_ok']}/{chk['lr_n']}, day counters {chk['since_ok']}/{chk['since_n']}, orders {ord_ok}/{ord_n}, "
          f"summary {s_ok}/{s_n}; calls tied {tied}/{m}, rescued {n_rescued}; end point {'yes' if has_end else 'no'}; "
          f"dates {'yes' if dates else 'no'}; {size:,} bytes")
    return d


# ------------------------------------------------------------------------------------------------ TEST HELPER (not for real pages)
def bundle_from_calls(path):
    """TEST ONLY. A stand-in bundle assembled in memory from a model-call log, for running the whole pipeline before the real
    bundle exists: one move per run of consecutive calls with the same decision and portfolio value (its last call's answer, or
    the follow-up answer when rescued, re-serialized as the engine logs actions; an unusable answer is kept as text), the prompt
    as the view, no summary and no game_over. run.json is marked test_bundle_from_calls, which the page shows in its eyebrow,
    title and foot; when the prompts count 396 decisions it also carries the rules' window (2024-06-03 to 2025-12-31, 396
    decisions over 397 trading days, in the unit of the summary's 'days') as an assumption for the date labels, which the foot
    states. It is never written to disk."""
    calls = load_calls(path, lenient=True)
    events = [{"event": "setup", "_visibility": "hidden", "roles": {"0": "TRADER"}}]
    k = 0
    while k < len(calls):
        j = k
        while j + 1 < len(calls) and view_key(calls[j + 1]["prompt"]) == view_key(calls[k]["prompt"]):
            j += 1
        last = calls[j]
        o = parse_orders(last["answer"])
        move = json.dumps({"action_id": "trade", "payload": {"orders": o}}) if o is not None else last["answer"]
        events.append({"event": "move", "actor": 0, "turn": len(events), "move": move, "view": last["prompt"],
                       "_audience": [0], "_visibility": "private"})
        k = j + 1
    t0 = next((c["t_start"] for c in calls if c.get("t_start")), None)
    T = next((parse_view(c["prompt"])["T"] for c in calls if RX_DEC.search(c["prompt"] or "")), None)
    run = {"model": next((c.get("model") for c in calls if c.get("model")), None), "test_bundle_from_calls": True,
           "recorded_at": datetime.datetime.fromtimestamp(t0, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t0 else None}
    if T == DEFAULT_WINDOW[2]:
        run["assume_window"] = DEFAULT_WINDOW
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
