"""Replay data for eco_inventory_collusion (episodic, inventory-constrained price competition; Friedrich, Pasztor and Ramponi,
"Learning Collusion in Episodic, Inventory-Constrained Markets", AAMAS 2025).

usage: python3 -I eco_inventory_collusion.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md), a file inside
          one, or the rendered Markdown replay of one (one '## Step N . P<seat>' section per decision: what the seat saw and
          the move it made).
  OUT_DIR where <OUT_DIR>/eco_inventory_collusion.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). Calls are tied to
          decisions in time order: a call belongs to the next decision of the same seat (and, when its prompt names it, the same
          round); the call whose answer is the logged price is that decision's call (earlier ones were format retries; their
          reasoning is kept above it). Its reasoning becomes the step's thinking and its prompt the observation.

Each round every seller sets one price at the same time (the engine asks P0, then P1, and buffers the round), so each round
gives one decision per seller, P0's first. The per-round market record (prices used, demand, units sold, profit, stock left)
is the summary's info.series; it is checked against every number the sellers were shown in their prompts and against the
game's demand formula (rules: Round Procedure), and every disagreement is printed as a warning. Without a summary (a
Markdown source) the record is rebuilt from the prompts, and rounds no prompt reports (the last one) are computed with the
demand formula. Profit gains and the collusion index follow the rules file (Reward; Metrics): the paper's benchmark profits,
the generalized mean with gamma 0.5 floored at 0."""
import datetime, json, math, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_inventory_collusion"
DEFAULT_DATE = None        # no fallback: a source without a recording date stops the build (see recorded_date)
SELLER_COLORS = ("c1", "c2", "c3", "c4", "ink", "hold")
KINDS = {"open": ["Opening price", "muted"], "raise": ["Raises price", "c3"], "cut": ["Cuts price", "c4"], "keep": ["Keeps price", "hold"]}
MINUS, DOT, TIMES, DIV = "\u2212", "\u00b7", "\u00d7", "\u00f7"
# The paper's main setting (rules: Configuration, Reward); used only when the source carries no parameters (a Markdown replay).
DEFAULT_PARAMS = {"sellers": 2, "horizon": 20, "inventory_per_step": [440], "unlimited_inventory": False, "quality": 2.0,
                  "outside_quality": 0.0, "marginal_cost": 1.0, "mu": 0.25, "demand_scale": 1000}
DEFAULT_GRID = [1.647, 1.67, 1.693, 1.716, 1.739, 1.763, 1.786, 1.809, 1.832, 1.855, 1.879, 1.902, 1.925, 1.948, 1.971]
PAPER = {"nash": 1.693, "monopoly": 1.925, "q_nash_cap": 470, "q_monopoly": 365}


def warn(msg):
    print("warning: " + msg, file=sys.stderr)


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_PUNCT = {"\u3002": ". ", "\u3001": ", ", "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"', "\u3010": "[", "\u3011": "]",
          "\u300a": '"', "\u300b": '"', "\u3008": "<", "\u3009": ">", "\u301c": "~"}


def en(s):
    """English only on the page: full-width forms become ASCII, CJK punctuation its ASCII counterpart, the rendered replay's
    truncation mark '(truncated)', and any other run of Chinese, Japanese or Korean characters a visible marker."""
    if not isinstance(s, str) or not re.search(r"[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af\uff00-\uffef]", s):
        return s
    s = s.replace("\uff08\u540e\u9762\u7701\u7565\uff09", "(truncated)")
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
def _md_bundle(path):
    """A rendered Markdown replay (move games): one '## Step N . P<seat>' section per decision (the heading is Chinese) with the
    state the seat saw in a fenced block, the move in backticks and, when saved, the reasoning in a ````text block; the header
    gives the rewards, seed and model."""
    text = open(path, encoding="utf-8").read()
    events, md_calls = [], []
    heads = list(re.finditer(r"^## \u7b2c (\d+) \u6b65 \u00b7 P(\d+)\s*$", text, re.M))
    for k, m in enumerate(heads):
        body = text[m.end(): heads[k + 1].start() if k + 1 < len(heads) else len(text)].split("\n## ", 1)[0]
        view = re.search(r"\*\*\u770b\u5230\uff1a\*\*\s*\n```[a-z]*\n(.*?)\n```", body, re.S)
        move = re.search(r"\*\*\u505a\u4e86\uff1a\*\* `(.*)`\s*$", body, re.M)
        think = re.findall(r"````text\n(.*?)\n````", body, re.S)
        if not move:
            continue
        seat = int(m.group(2))
        events.append({"event": "move", "actor": seat, "turn": int(m.group(1)), "move": move.group(1),
                       "view": en(view.group(1)) if view else ""})
        md_calls.append({"seat": seat, "prompt": "", "answer": move.group(1), "reasoning": think[-1] if think else "", "rescued": False})
    over = re.search(r'```\n(\{"event": "game_over".*?\})\n```', text, re.S)
    if over:
        events.append(json.loads(over.group(1)))
    head = text.split("\n## ")[0]
    rw = re.search(r'(\{"0":[^}]*\})', head)
    seed = re.search(r"\u79cd\u5b50 (\d+)", head)
    model = re.search(r"\u6a21\u578b (\S+)", head)
    summary = {"rewards": json.loads(rw.group(1)) if rw else {}}
    run = {"seed": int(seed.group(1)) if seed else None, "model": model.group(1).rstrip(",\uff0c") if model else ""}
    if not any(c["reasoning"] for c in md_calls):
        md_calls = []
    # the replay carries no date; the bundle it was rendered from may sit beside it (replays_<date>/raw/<game>/)
    meta = {}
    raw = os.path.join(os.path.dirname(os.path.abspath(path)), "raw", GID)
    if os.path.exists(os.path.join(raw, "meta.json")):
        rmeta = json.load(open(os.path.join(raw, "meta.json"), encoding="utf-8"))
        rrun = json.load(open(os.path.join(raw, "run.json"), encoding="utf-8")) if os.path.exists(os.path.join(raw, "run.json")) else {}
        if rrun.get("seed") == run["seed"] and (rmeta.get("model") or rrun.get("model")) == run["model"]:
            meta = rmeta
    return {"events": events, "summary": summary, "run": run, "config": {}, "transcript": "", "meta": meta, "md_calls": md_calls,
            "from_md": True}


def load_source(src):
    if isinstance(src, dict):
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


def load_calls(path):
    """common.load_calls plus what it drops from each line: why the call stopped ('finish'), the output tokens it used, its
    start time and model. Same lines, same order."""
    out = common.load_calls(path)
    if not out:
        return out
    raw = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if len(raw) != len(out):
        warn(f"call log: {len(raw)} lines but {len(out)} calls; finish reasons and token counts are not used")
        return out
    for c, r in zip(out, raw):
        u = r.get("usage") or {}
        c.update(finish=r.get("finish") or r.get("finish_reason"), out_tokens=u.get("completion_tokens"),
                 t_start=r.get("t_start"), model=r.get("model"))
    return out


def recorded_date(src, calls=None):
    """The recording date: meta.json or run.json recorded_at, else the timestamp that opens the source run's name, else the first
    call's start time. With none of them the build stops: the page must not show a guessed date."""
    b = load_source(src)
    meta, run = b.get("meta") or {}, b.get("run") or {}
    for v in (meta.get("recorded_at"), run.get("recorded_at")):
        if v:
            return str(v)[:10]
    for v in (meta.get("source_run"), run.get("source_run")):
        m = re.match(r"(\d{4})(\d{2})(\d{2})T\d{6}Z", str(v or ""))
        if m:
            return "-".join(m.groups())
    t = next((c.get("t_start") for c in calls or [] if isinstance(c.get("t_start"), (int, float))), None)
    if t is not None:
        return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d")
    raise SystemExit("no recording date in the source (meta.json or run.json recorded_at, the source run's name, or the call "
                     "log's start times); refusing to guess one")


def recorded_model(b, calls=None):
    """The model that played, without its provider prefix ('openai/deepseek-flash' -> 'deepseek-flash'); None when unrecorded."""
    model = (b.get("meta") or {}).get("model") or (b.get("run") or {}).get("model") or \
        next((c.get("model") for c in calls or [] if c.get("model")), None)
    return str(model).split("/")[-1] if model else None


# ------------------------------------------------------------------------------------------------ small helpers
def p3(v):
    return f"{v:.3f}"


def i0(v):
    return f"{int(v):,}"


def m2(v):
    return f"{v:,.2f}"


def bn(v):
    """A benchmark profit as the rules give it: up to 3 decimals, no trailing zeros (304.92, 337.625)."""
    return f"{v:,.3f}".rstrip("0").rstrip(".")


def sg(v, d=3):
    """Signed number with a true minus sign."""
    s = f"{abs(v):.{d}f}"
    return (MINUS if v < 0 and float(s) != 0 else "+" if v > 0 and float(s) != 0 else "") + s


def num(s):
    return float(s.replace(",", ""))


def rounds_text(rs):
    rs = sorted(rs)
    if len(rs) == 1:
        return f"round {rs[0]}"
    if rs == list(range(rs[0], rs[-1] + 1)) and len(rs) > 2:
        return f"rounds {rs[0]} to {rs[-1]}"
    return "rounds " + ", ".join(map(str, rs[:-1])) + " and " + str(rs[-1])


def slash(vals):
    return " / ".join(vals)


def parse_price(move):
    """The price of a set_price action (string or dict), rounded to 3 decimals as the engine does; None when unusable."""
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
    if not isinstance(a, dict) or a.get("action_id") != "set_price" or not isinstance(a.get("payload"), dict):
        return None
    p = a["payload"].get("price")
    if isinstance(p, bool) or not isinstance(p, (int, float, str)):
        return None
    try:
        p = float(p)
    except ValueError:
        return None
    return round(p, 3) if math.isfinite(p) else None


def round_of(text):
    m = re.search(r"\bRound (\d+) of (\d+)\b", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


# ------------------------------------------------------------------------------------------------ market model (rules: Round Procedure)
def market_demand(prices, stocks, P):
    """floor(lambda * share) for each seller with stock at the start of the round, 0 for a sold-out one (choice substitution)."""
    a, a0, mu, lam = P["quality"], P["outside_quality"], P["mu"], P["demand_scale"]
    live = [i for i, x in enumerate(stocks) if P["unlimited_inventory"] or x > 0]
    den = sum(math.exp((a - prices[i]) / mu) for i in live) + math.exp(a0 / mu)
    return [int(math.floor(lam * math.exp((a - prices[i]) / mu) / den)) if i in live else 0 for i in range(len(prices))]


def read_params(b, moves):
    """Market parameters: the summary's info.params, else the rules text at the top of a prompt, else the paper's main setting."""
    info = (b.get("summary") or {}).get("info") or {}
    P = dict(DEFAULT_PARAMS)
    src = "defaults"
    if info.get("params"):
        P.update({k: v for k, v in info["params"].items() if v is not None})
        src = "summary"
    else:
        view = next((e.get("view") or "" for e in moves if "You are playing:" in (e.get("view") or "")), "")
        m = re.search(r"floor\((\d+) x share\), where share = exp\(\(([\d.]+) - own price\) / ([\d.]+)\).*?plus exp\(([\d.-]+) / [\d.]+\)", view)
        if m:
            P.update(demand_scale=int(m.group(1)), quality=float(m.group(2)), mu=float(m.group(3)), outside_quality=float(m.group(4)))
            src = "prompt"
        m = re.search(r"earns \(price - ([\d.]+)\) per unit", view)
        if m:
            P["marginal_cost"] = float(m.group(1))
        m = re.search(r"(\d+) sellers offer .*? for (\d+) rounds", view)
        if m:
            P["sellers"], P["horizon"] = int(m.group(1)), int(m.group(2))
    n = int(P.get("sellers") or 2)
    inv = list(P.get("inventory_per_step") or [440])
    P["inventory_per_step"] = (inv * n)[:n] if len(inv) < n else inv[:n]
    grid = info.get("grid")
    if not grid:
        view = next((e.get("view") or "" for e in moves if "Prices must be one of these" in (e.get("view") or "")), "")
        m = re.search(r"Prices must be one of these \d+ values: ([\d., ]+)\.", view)
        grid = [float(x) for x in m.group(1).split(",")] if m else list(DEFAULT_GRID)
    P["grid"] = [round(float(x), 3) for x in grid]
    P["source"] = src
    return P


def read_bench(b, P):
    """The benchmark prices and per-round profits behind the profit gain: the summary's info.benchmark, else the paper's
    convention for the main setting (rules: Reward): R^N = (pN - c) min(470, I/T), R^M = (pM - c) 365."""
    info = (b.get("summary") or {}).get("info") or {}
    bm = info.get("benchmark") or {}
    n = P["sellers"]
    if bm.get("nash_profit_per_step") and bm.get("monopoly_profit_per_step"):
        return {"method": bm.get("method") or "paper", "pN": list(bm.get("nash_prices") or [info.get("nash_grid_price")] * n),
                "pM": list(bm.get("monopoly_prices") or [info.get("monopoly_grid_price")] * n),
                "RN": [float(x) for x in bm["nash_profit_per_step"]], "RM": [float(x) for x in bm["monopoly_profit_per_step"]],
                "qN": bm.get("nash_quantity_per_step"), "qM": bm.get("monopoly_quantity_per_step"),
                "realN": bm.get("realized_nash_profit"), "realM": bm.get("realized_monopoly_profit"), "from": "summary"}
    c, inv0 = P["marginal_cost"], P["inventory_per_step"][0]
    qN = min(PAPER["q_nash_cap"], inv0)
    return {"method": "paper", "pN": [PAPER["nash"]] * n, "pM": [PAPER["monopoly"]] * n,
            "RN": [round((PAPER["nash"] - c) * qN, 6)] * n, "RM": [round((PAPER["monopoly"] - c) * PAPER["q_monopoly"], 6)] * n,
            "qN": qN, "qM": PAPER["q_monopoly"], "realN": None, "realM": None, "from": "paper defaults"}


# ------------------------------------------------------------------------------------------------ what the sellers were shown
_HIST_RX = re.compile(r"^- Round (\d+): prices (.*?); you sold ([\d,]+) units for a profit of ([\d,.-]+)\.", re.M)


def view_facts(view, seat, n):
    """Numbers a prompt reports: own and rivals' remaining stock (and the start stock), past rounds' prices and the seat's own
    sales and profit, the total profit so far."""
    f = {"round": round_of(view)[0], "stock": {}, "start": None, "hist": [], "total": None}
    m = re.search(r"Your remaining stock: ([\d,]+) of ([\d,]+) units", view)
    if m:
        f["stock"][seat], f["start"] = int(num(m.group(1))), int(num(m.group(2)))
    for m in re.finditer(r"Seller P(\d+) remaining stock: ([\d,]+) of ([\d,]+) units", view):
        f["stock"][int(m.group(1))] = int(num(m.group(2)))
    for m in _HIST_RX.finditer(view):
        pr = {int(k): float(v) for k, v in re.findall(r"P(\d+) ([\d.]+)", m.group(2))}
        f["hist"].append({"round": int(m.group(1)), "prices": pr, "sold": int(num(m.group(3))), "profit": num(m.group(4)),
                          "profit_txt": m.group(4)})
    m = re.search(r"Your total profit so far: ([\d,.-]+)\.", view)
    if m:
        f["total"], f["total_txt"] = num(m.group(1)), m.group(1)
    return f


# ------------------------------------------------------------------------------------------------ the market record
def read_record(b, moves, P):
    """Per-round record {prices, demand, sold, profit, stock_end} (lists over sellers). From the summary's info.series when
    present; otherwise rebuilt: prices from the moves, sales, profit and stock from the next round's prompts, and rounds no
    prompt reports computed with the demand formula. Returns (rounds, how) with how in {'summary', 'rebuilt'}."""
    n, T = P["sellers"], int(P["horizon"])
    start = [int(x) * T for x in P["inventory_per_step"]]
    se = ((b.get("summary") or {}).get("info") or {}).get("series") or {}
    if se.get("prices"):
        R = []
        for k in range(len(se["prices"])):
            R.append({"round": se["round"][k] if se.get("round") else k + 1,
                      "prices": [None if v is None else round(float(v), 3) for v in se["prices"][k]] if se["prices"][k] else [None] * n,
                      "demand": list(se["demand"][k]) if se.get("demand") else None, "sold": list(se["sold"][k]),
                      # raw floats: the engine formats the unrounded value (f"{v:,.2f}"), so pre-rounding here could
                      # show a profit one cent off what the seller was told (354.79499... is 354.79, not 354.80)
                      "profit": [float(v) for v in se["profit"][k]], "stock_end": list(se["stock_end"][k])})
        return R, start, "summary"
    # rebuild from the prompts
    facts, prices = {}, {}
    last_seat, rnd = None, 0
    for e in moves:
        s = e["actor"]
        r = round_of(e.get("view"))[0]
        if r is None:
            rnd = rnd + 1 if last_seat is None or s <= last_seat else rnd
            r = rnd
        rnd, last_seat = r, s
        prices.setdefault(r, [None] * n)[s] = parse_price(e.get("move"))
        facts[(r, s)] = view_facts(e.get("view") or "", s, n)
    R, stock = [], list(start)
    for r in sorted(prices):
        pr = prices[r]
        if any(p is None for p in pr):
            warn(f"round {r}: a price is missing or unreadable; the round is left out")
            break
        dem = market_demand(pr, stock, P)
        sold = dem if P["unlimited_inventory"] else [min(d, x) for d, x in zip(dem, stock)]
        nxt = [facts.get((r + 1, s)) for s in range(n)]
        for s in range(n):
            h = next((h for h in (nxt[s] or {}).get("hist", []) if h["round"] == r), None)
            if h and h["sold"] != sold[s]:
                warn(f"round {r}: P{s} was told it sold {h['sold']}, the demand formula gives {sold[s]}; using the prompt")
                sold[s] = h["sold"]
        prof = [(pr[s] - P["marginal_cost"]) * sold[s] for s in range(n)]
        stock = stock if P["unlimited_inventory"] else [stock[s] - sold[s] for s in range(n)]
        R.append({"round": r, "prices": list(pr), "demand": dem, "sold": sold, "profit": prof, "stock_end": list(stock)})
    return R, start, "rebuilt"


def check_record(R, start, moves, P, rewards, gains, info):
    """Cross-checks; returns counts for the footnote. Every disagreement is printed as a warning."""
    n = P["sellers"]
    out = {"demand_ok": 0, "demand_n": 0, "view_ok": 0, "view_n": 0, "moves_ok": 0, "moves_n": 0, "hist_ok": 0, "hist_n": 0}
    stock = list(start)
    for k, rd in enumerate(R):
        if rd["prices"] is None or any(p is None for p in rd["prices"]):
            stock = list(rd["stock_end"])
            continue
        dem = market_demand(rd["prices"], stock, P)
        sold = dem if P["unlimited_inventory"] else [min(d, x) for d, x in zip(dem, stock)]
        for s in range(n):
            out["demand_n"] += 1
            exp_profit = (rd["prices"][s] - P["marginal_cost"]) * sold[s]
            if sold[s] == rd["sold"][s] and abs(exp_profit - rd["profit"][s]) < 1e-6 and \
                    (rd["demand"] is None or rd["demand"][s] == (dem[s] if stock[s] > 0 or P["unlimited_inventory"] else 0)):
                out["demand_ok"] += 1
            else:
                warn(f"round {rd['round']} P{s}: record sold {rd['sold'][s]} profit {rd['profit'][s]}, formula {sold[s]} / {exp_profit}")
        stock = list(rd["stock_end"])
    byround = {rd["round"]: rd for rd in R}
    cum = [[0.0] * n]
    for rd in R:
        cum.append([cum[-1][s] + rd["profit"][s] for s in range(n)])
    for e in moves:
        s, view = e["actor"], e.get("view") or ""
        f = view_facts(view, s, n)
        r = f["round"]
        if r is None:
            continue
        rd = byround.get(r)
        if rd:
            out["moves_n"] += 1
            used = rd["prices"][s]
            fd = info.get("forced_deviation") or {}
            if parse_price(e.get("move")) == used or (fd.get("round") == r and fd.get("seat") == s):
                out["moves_ok"] += 1
            else:
                warn(f"round {r} P{s}: move {e.get('move')} but the record shows price {used}")
        prev = byround.get(r - 1)
        for k, v in f["stock"].items():
            out["view_n"] += 1
            want = prev["stock_end"][k] if prev else start[k]
            if v == want:
                out["view_ok"] += 1
            else:
                warn(f"round {r} P{s} prompt: P{k} stock {v}, record {want}")
        for h in f["hist"]:
            hr = byround.get(h["round"])
            if not hr:
                continue
            # the profit must print exactly as the seller saw it (the page shows m2 of the record's value)
            ok = all(abs(hr["prices"][k] - v) < 5e-4 for k, v in h["prices"].items()) and h["sold"] == hr["sold"][s] and \
                m2(hr["profit"][s]) == h["profit_txt"]
            out["view_n"] += 1
            out["view_ok"] += ok
            out["hist_n"] += 1
            out["hist_ok"] += ok
            if not ok:
                warn(f"round {r} P{s} prompt: round {h['round']} {h}, record {hr}")
        if f["total"] is not None and r - 1 < len(cum):
            out["view_n"] += 1
            ok = m2(cum[r - 1][s]) == f["total_txt"]
            out["view_ok"] += ok
            if not ok:
                warn(f"round {r} P{s} prompt: total profit {f['total']}, record {cum[r - 1][s]:.3f}")
    for s in range(n):
        rw = rewards[s]
        if rw is not None and abs(rw - max(-1.0, min(1.0, gains[s]))) > 1e-9:
            warn(f"P{s}: reward {rw} but profit gain {gains[s]} (clipped {max(-1.0, min(1.0, gains[s]))})")
    for sel in info.get("sellers") or []:
        s = sel.get("seat")
        if s is not None and s < n and abs(sel.get("profit", cum[-1][s]) - cum[-1][s]) > 1e-6:
            warn(f"P{s}: summary profit {sel.get('profit')}, record total {cum[-1][s]}")
    return out


# ------------------------------------------------------------------------------------------------ scores (rules: Reward, Metrics)
def collusion_index(gains, gamma=0.5):
    """The paper's index: generalized mean (gamma 0.5) of sign(d)|d|^gamma, floored at 0 before the outer power."""
    m = sum(math.copysign(abs(d) ** gamma, d) for d in gains) / len(gains)
    return max(0.0, m) ** (1 / gamma)


# ------------------------------------------------------------------------------------------------ model calls
def _call_round(c):
    return round_of(c.get("prompt"))[0]


def attach_calls(steps, calls):
    """Tie calls to steps in time order. For each step the window is the run of calls from the pointer whose seat and round
    agree with it (unknown values agree with anything); the first call in it whose answer is the logged price is the one, and
    earlier calls in the window were failed attempts whose reasoning is kept above it. Without a matching answer the last call
    of the window is used and the step says the game applied another price (the recorder's fallback holds the last price).
    A call whose reasoning hit the output limit (finish 'length') ends with a visible cut mark; when its price came from a
    follow-up question (rescued), a note saying so leads its reasoning, so the page's 'conclusion' (the last paragraph) is
    still the model's own last words. Returns (tied, rescued, limits): the number of steps with a call, of steps whose price
    came from a follow-up question, and for each of those the output-token count at which its reasoning was cut (None when
    the log does not say it hit the limit)."""
    ptr, tied, rescued, limits = 0, 0, 0, []

    def fits(k, st, strict=False):
        seat, rnd = calls[k].get("seat"), _call_round(calls[k])
        if strict:
            return seat == st["seat"] and rnd == st["round"]
        return seat in (None, st["seat"]) and rnd in (None, st["round"])
    for st in steps:
        if ptr < len(calls) and not fits(ptr, st):
            ahead = next((k for k in range(ptr, min(len(calls), ptr + 40)) if fits(k, st, strict=True)), None)
            if ahead is None:
                continue
            warn(f"calls {ptr}..{ahead - 1} match no decision; skipped")
            ptr = ahead
        j = ptr                                        # every seller prices every round, so a seat's calls for one decision
        while j < len(calls) and fits(j, st):          # are consecutive and end at the next seller's call
            j += 1
        window = list(range(ptr, j))
        if not window:
            continue
        hit = next((k for k in window if parse_price(calls[k].get("answer")) == st["price"]), None)
        use = hit if hit is not None else window[-1]
        parts = [f"[Earlier attempt, answer: {calls[k]['answer'][:300]}]\n{calls[k].get('reasoning') or '(no reasoning returned)'}"
                 for k in range(ptr, use)]
        c = calls[use]
        finish, toks = c.get("finish"), c.get("out_tokens")
        cut = finish == "length" if finish else bool(c.get("rescued"))   # without a finish reason a rescue means it ran out
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
            parts.append(f"[The game applied the price {p3(st['price'])} instead of this answer: {c['answer'][:300]}]")
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        if c.get("prompt"):
            st["state"] = en(common.state_part(c["prompt"]))
            st["from_prompt"] = True
        ptr = use + 1
        tied += 1
    if ptr < len(calls):
        warn(f"{len(calls) - ptr} calls after the last decision were not tied to any step")
    return tied, rescued, limits


# ------------------------------------------------------------------------------------------------ build
def build(bundle_dir, calls=None):
    b = load_source(bundle_dir)
    if isinstance(calls, str):
        calls = load_calls(calls)
    if not calls:
        calls = b.get("md_calls") or []
    events, summary, run = b["events"], b.get("summary") or {}, b.get("run") or {}
    info = summary.get("info") or {}
    moves = [e for e in events if e.get("event") == "move"]
    if not moves:
        raise SystemExit("no move events in " + str(bundle_dir))
    P = read_params(b, moves)
    B = read_bench(b, P)
    n, T, grid = P["sellers"], int(P["horizon"]), P["grid"]
    R, start, how = read_record(b, moves, P)
    names = [f"Seller P{s}" for s in range(n)]           # the seat names the prompts and the log use
    cols = [SELLER_COLORS[s % len(SELLER_COLORS)] for s in range(n)]
    pN, pM = B["pN"][0], B["pM"][0]
    raw_rw = summary.get("rewards") or (run.get("result") or {}).get("rewards") or {}
    over = next((e for e in events if e.get("event") == "game_over"), {})
    raw_rw = raw_rw or over.get("rewards") or {}
    rewards = [float(raw_rw[str(s)]) if str(s) in raw_rw else None for s in range(n)]

    # ---- per round, per seller
    played = [rd for rd in R if rd["prices"] and all(p is not None for p in rd["prices"])]
    nR = len(played)
    xn = max(T, len(R))
    price = [[None] * xn for _ in range(n)]
    sold = [[None] * xn for _ in range(n)]
    stock = [[None] * xn for _ in range(n)]
    for k, rd in enumerate(R):
        for s in range(n):
            if rd["prices"] and rd["prices"][s] is not None:
                price[s][k] = rd["prices"][s]
            sold[s][k] = rd["sold"][s]
            stock[s][k] = rd["stock_end"][s]
    profit = [sum(rd["profit"][s] for rd in R) for s in range(n)]
    units = [sum(rd["sold"][s] for rd in R) for s in range(n)]
    left = [R[-1]["stock_end"][s] for s in range(n)]
    sold_out = [next((rd["round"] for rd in R if rd["stock_end"][s] == 0), None) if not P["unlimited_inventory"] else None for s in range(n)]
    PiN = [B["RN"][s] * T for s in range(n)]
    PiM = [B["RM"][s] * T for s in range(n)]
    gains = [(profit[s] - PiN[s]) / (PiM[s] - PiN[s]) for s in range(n)]
    ci = collusion_index(gains)
    ci_ar = sum(gains) / n
    # score banked so far: the excess over the competitive benchmark after t rounds, over the whole episode's width
    # (cumulative profit - t R^N) / (T (R^M - R^N)); this is the final score if every later round earned exactly R^N, and
    # after round T it is the score. (Dividing by t instead lets one round's luck dominate the first points.)
    banked = [[None] * xn for _ in range(n)]
    acc = [0.0] * n
    for k, rd in enumerate(R):
        t = k + 1
        acc = [acc[s] + rd["profit"][s] for s in range(n)]
        for s in range(n):
            banked[s][k] = round((acc[s] - t * B["RN"][s]) / (PiM[s] - PiN[s]), 6)
    if len(R) == T:
        for s in range(n):
            if abs(banked[s][T - 1] - gains[s]) > 1e-5:
                warn(f"P{s}: banked score after round {T} is {banked[s][T - 1]}, the score is {gains[s]}")
    # stock against the even pace: stock left - (start - per-round stock x rounds); after round T it is the unsold stock
    pace_dev = [[None] * xn for _ in range(n)]
    if not P["unlimited_inventory"]:
        for k, rd in enumerate(R):
            for s in range(n):
                pace_dev[s][k] = rd["stock_end"][s] - (start[s] - P["inventory_per_step"][s] * (k + 1))
    mean_p = [sum(rd["prices"][s] for rd in played) / nR for s in range(n)]
    level = [(mean_p[s] - pN) / (pM - pN) for s in range(n)]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    conv = sum(abs(rd["prices"][i] - rd["prices"][j]) for rd in played for i, j in pairs) / (nR * len(pairs)) / (pM - pN) if pairs else 0.0
    for key, mine in (("collusion_index", ci), ("convergence", conv)):
        if info.get(key) is not None and abs(info[key] - mine) > 1e-9:
            warn(f"{key}: summary {info[key]}, computed {mine}")
    for s, sel in enumerate(info.get("sellers") or []):
        if s < n and sel.get("mean_price") is not None and abs(sel["mean_price"] - mean_p[s]) > 1e-9:
            warn(f"P{s}: summary mean price {sel['mean_price']}, computed {mean_p[s]}")
    checks = check_record(R, start, moves, P, rewards, gains, info)

    # ---- steps: one per decision, in log order
    byround = {rd["round"]: rd for rd in R}
    pos = {rd["round"]: k for k, rd in enumerate(R)}
    steps, feed = [], []
    mr, last_seat, rnd = [], None, 0                      # the round of each move: from its prompt, else by seat order
    for e in moves:
        r = round_of(e.get("view"))[0]
        if r is None:
            r = rnd + 1 if last_seat is None or e["actor"] <= last_seat else rnd
        rnd, last_seat = r, e["actor"]
        mr.append(r)
    fd = info.get("forced_deviation") or {}

    def zone(p):
        tags = []
        if abs(p - grid[0]) < 5e-4:
            tags.append("the lowest grid price")
        if abs(p - grid[-1]) < 5e-4:
            tags.append("the highest grid price")
        if abs(p - pN) < 5e-4:
            tags.append("the competitive (Nash) price")
        if abs(p - pM) < 5e-4:
            tags.append("the collusive (monopoly) price")
        return ", " + " and ".join(tags) if tags else ""
    for mi, e in enumerate(moves):
        s, r = e["actor"], mr[mi]
        p = parse_price(e.get("move"))
        prev = byround.get(r - 1)
        own_prev = prev["prices"][s] if prev else None
        st_start = prev["stock_end"][s] if prev else start[s]
        if p is None:
            kind, head = "keep", f"Round {r} {DOT} unreadable action"
        elif own_prev is None:
            kind, head = "open", f"Round {r} {DOT} opens at {p3(p)}"
        elif p > own_prev + 5e-4:
            kind, head = "raise", f"Round {r} {DOT} raises to {p3(p)} (from {p3(own_prev)})"
        elif p < own_prev - 5e-4:
            kind, head = "cut", f"Round {r} {DOT} cuts to {p3(p)} (from {p3(own_prev)})"
        else:
            kind, head = "keep", f"Round {r} {DOT} keeps {p3(p)}"
        sold_out_now = st_start == 0 and not P["unlimited_inventory"]
        if sold_out_now:
            head += " (sold out)"
        forced = fd.get("round") == r and fd.get("seat") == s
        if forced:
            head += f" {DOT} replaced by {p3(fd.get('forced'))}"
        st = {"seat": s, "x": r - 1, "head": head, "kind": kind, "act": en(e.get("move") or ""),
              "state": en(common.state_part(e.get("view") or "")) or "Not recorded.", "round": r, "price": p}
        steps.append(st)
        # feed entry for this decision
        if p is None:
            text = "Unreadable action."
        elif own_prev is None:
            text = f"Opens at {p3(p)}{zone(p)}. No prices have been seen yet."
        else:
            verb = {"raise": f"Raises from {p3(own_prev)} to", "cut": f"Cuts from {p3(own_prev)} to", "keep": "Keeps"}[kind]
            rivals = [k for k in range(n) if k != s]
            q = min(prev["prices"][k] for k in rivals)
            who = names[rivals[0]] if n == 2 else "the cheapest rival"
            if p < q - 5e-4:
                rel = f"this undercuts it by {q - p:.3f}"
            elif p > q + 5e-4:
                rel = f"this is {p - q:.3f} above it"
            else:
                rel = "this matches it"
            text = f"{verb} {p3(p)}{zone(p)}. {who[0].upper() + who[1:]} charged {p3(q)} last round: {rel}."
        if sold_out_now:
            text += " Sold out: this price affects no one's demand."
        if forced:
            text += f" The game replaced it with {p3(fd.get('forced'))} (forced deviation)."
        feed.append({"x": r - 1, "step": len(steps) - 1, "who": names[s], "kind": kind, "text": text})
        # market result after the round's last decision
        rd = byround.get(r)
        if rd and (mi + 1 == len(moves) or mr[mi + 1] != r):
            bits = []
            for k in range(n):
                was = R[pos[r] - 1]["stock_end"][k] if pos[r] else start[k]
                if was == 0 and not P["unlimited_inventory"]:
                    bits.append(f"{names[k]} (sold out, quoting {p3(rd['prices'][k])}) sells nothing")
                    continue
                dm = rd["demand"][k] if rd["demand"] else None
                part = f"{names[k]} at {p3(rd['prices'][k])} sells {i0(rd['sold'][k])} units"
                if dm is not None and dm > rd["sold"][k]:
                    part += f" of {i0(dm)} demanded and sells out"
                bits.append(part + f", profit {m2(rd['profit'][k])}")
            feed.append({"x": r - 1, "step": len(steps) - 1, "who": "Market",
                         "text": "; ".join(bits) + ". Stock left: " + slash([i0(v) for v in rd["stock_end"]]) + "."})
    # rounds after every seller sold out pass without decisions: their record has no step
    tied, n_rescued, limits = attach_calls(steps, calls)
    has_reason = any(st.get("thinking") for st in steps)
    from_prompt = sum(1 for st in steps if st.get("from_prompt"))
    for st in steps:
        st.setdefault("thinking", "")
        for k in ("round", "price", "from_prompt"):
            st.pop(k, None)

    # ---- story features
    floor = grid[0]
    floor_rs = [rd["round"] for rd in played if all(abs(p - floor) < 5e-4 for p in rd["prices"])]
    run_a = run_b = None
    for r0 in floor_rs:                                   # the longest run of rounds with every seller at the floor
        r1 = r0
        while r1 + 1 in floor_rs:
            r1 += 1
        if run_a is None or r1 - r0 > run_b - run_a:
            run_a, run_b = r0, r1
    every = "both sellers" if n == 2 else f"all {n} sellers"
    Every = every[0].upper() + every[1:]
    win = max(range(n), key=lambda s: gains[s])
    floor_war = run_a is not None and run_b > run_a
    last_r = played[-1]["round"]

    def longest_run(rs, min_len=3):
        """The longest run of consecutive rounds in rs (at least min_len long, the earliest on a tie), or None."""
        best, rs = None, sorted(rs)
        for k, r0 in enumerate(rs):
            if k and rs[k - 1] == r0 - 1:
                continue
            r1 = r0
            while r1 + 1 in rs:
                r1 += 1
            if r1 - r0 + 1 >= min_len and (best is None or r1 - r0 > best[1] - best[0]):
                best = (r0, r1)
        return best

    def inside_floor_war(span):
        return floor_war and run_a <= span[0] and span[1] <= run_b

    def gridstep(p):
        return min(range(len(grid)), key=lambda i: abs(grid[i] - p))
    # the longest stretch with every seller below the competitive price (the floor war above is one kind of it)
    below = longest_run([rd["round"] for rd in played if all(p < pN - 5e-4 for p in rd["prices"])])
    below_txt = below_title = below_tile = ""
    if below and not inside_floor_war(below):
        ba, bb = below
        span = [rd for rd in played if ba <= rd["round"] <= bb]
        L = len(span)
        modes = []                                        # each seller's most frequent price in the stretch, with its count
        for s in range(n):
            ps = [rd["prices"][s] for rd in span]
            m = max(sorted(set(ps)), key=lambda v: ps.count(v))
            modes.append((m, ps.count(m), s))
        modes.sort()

        def where(m, prev):
            if abs(m - floor) < 5e-4:
                return f"at the grid floor {p3(m)}"
            if prev is not None:
                k = gridstep(m) - gridstep(prev)
                if k in (1, 2):
                    return f"{'one grid step' if k == 1 else 'two grid steps'} above, at {p3(m)}"
            return f"at {p3(m)}"
        mostly = "" if all(c == L for _, c, _ in modes) else "mostly "
        if len({m for m, _, _ in modes}) == 1:
            who_title = f"{every} {mostly}sit {where(modes[0][0], None)}"
            who_sub = f"{every} {where(modes[0][0], None)}" + (" every round" if not mostly else
                                                              ", " + ", ".join(f"{names[s]} in {c} of the {L}" for _, c, s in modes))
        else:
            descs, prev = [], None
            for m, c, s in modes:
                descs.append((s, c, where(m, prev)))
                prev = m
            who_title = (f"{names[descs[0][0]]} {mostly}sits {descs[0][2]}"
                         + "".join(f"{' and' if k == len(descs) - 1 else ','} {names[s]} {w}" for k, (s, c, w) in enumerate(descs) if k))
            who_sub = ", ".join(f"{names[s]} {w}{',' if ',' in w else ''} " + ("every round" if c == L else f"in {c} of the {L}")
                                for s, c, w in descs)
        before = [rd for rd in played if rd["round"] < ba]
        settles = bb == last_r
        tail = "settle below the competitive level" if settles else f"stay below the competitive level for {L} rounds"
        if not before:
            lead = "Prices sit below the competitive level from the start" if settles else \
                f"Prices start below the competitive level and stay there for {L} rounds"
        else:
            bp = [p for rd in before for p in rd["prices"]]
            moves_before = sum(1 for k in range(1, len(before)) for s in range(n)
                               if abs(before[k]["prices"][s] - before[k - 1]["prices"][s]) > 5e-4)
            swing = len(before) >= 2 and gridstep(max(bp)) - gridstep(min(bp)) >= 3 and 2 * moves_before >= n * (len(before) - 1)
            if swing:
                lead = f"Prices swing early, then {tail}"
            elif (sum(bp) / len(bp) - pN) / (pM - pN) >= 0.4:
                lead = f"Prices start well above the competitive level, then {tail}"
            else:
                lead = f"Prices {tail}"
        when = f"from round {ba}" if settles else f"in rounds {ba} to {bb}"
        below_title = f"{lead}: {when} {who_title}"
        below_txt = (f" In rounds {ba} to {bb} {every} price below the competitive {p3(pN)} every round: {who_sub}.")
        below_tile = f" {Every} below the competitive price in every round from {ba} to {bb}."
    if floor_war:
        before = [p for rd in played if rd["round"] < run_a for p in rd["prices"]]
        lead = ("Prices start well above the competitive level, then collapse into a price war"
                if before and (sum(before) / len(before) - pN) / (pM - pN) >= 0.4 else "The sellers fight a price war")
        title = (f"{lead}: {every} charge the grid floor, {p3(floor)}, in rounds {run_a} to {run_b}; collusion index {ci:.3f}")
    elif ci >= 0.5:
        title = f"The sellers hold prices near the collusive level: collusion index {ci:.2f}, average prices {slash([p3(v) for v in mean_p])}"
    elif below_title:
        title = f"{below_title}; collusion index {ci:.3f}"
    elif max(level) < 0.2:
        title = f"The sellers compete prices down to the competitive level: average prices {slash([p3(v) for v in mean_p])}; collusion index {ci:.3f}"
    else:
        title = (f"Prices between competition and collusion: average prices {slash([p3(v) for v in mean_p])} against the "
                 f"competitive {p3(pN)} and collusive {p3(pM)}; collusion index {ci:.3f}")

    stock0 = start[0] if len(set(start)) == 1 else None
    pace = P["inventory_per_step"][0] if len(set(P["inventory_per_step"])) == 1 else None
    num_words = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}
    lede = (f"{num_words.get(n, str(n))} sellers of a perishable good (think seats on one flight) each start with "
            + (f"{i0(stock0)} units" if stock0 is not None else "a fixed stock") +
            f" and, for {T} rounds, set a price from a grid of {len(grid)} values between {p3(grid[0])} and {p3(grid[-1])}, all at the "
            f"same time and without talking. Customers split between the sellers by a logit demand rule, so the cheaper seller "
            f"sells more; each unit costs {P['marginal_cost']:g}, and stock left after round {T} is worthless. Each seller's score is "
            f"its profit gain: 0 at the competitive profit (both at the Nash price {p3(pN)}) and 1 at the collusive profit (both at "
            f"the monopoly price {p3(pM)}), as in the paper by Friedrich et al. (2025).")

    # ---- tiles
    tiles = [
        {"k": "Collusion index", "v": f"{ci:.3f}",
         "s": f"0 = competitive, 1 = collusive: the paper's mean of the {'two' if n == 2 else str(n)} scores. The paper's "
              "trained agents reach 0.43 (PPO) and 0.23 (DQN), averaged over 100 trained pairs."},
        {"k": "Scores (profit gain)", "v": slash([sg(g) for g in gains]),
         "s": f"{slash(names)}. Profit {slash([m2(v) for v in profit])}; the benchmarks are {m2(PiN[0])} (score 0) and "
              f"{m2(PiM[0])} (score 1). {names[win]} wins."},
        {"k": "Average price", "v": slash([p3(v) for v in mean_p]),
         "s": f"{slash(names)} over {nR} rounds, against the competitive {p3(pN)} and collusive {p3(pM)}"
              + (f". Both at the floor {p3(floor)} in {rounds_text(floor_rs)}." if floor_rs and n == 2 else
                 f". All at the floor {p3(floor)} in {rounds_text(floor_rs)}." if floor_rs else ".") + below_tile},
        {"k": "Unsold stock", "v": slash([i0(v) for v in left]),
         "s": f"Left after round {T}, worth nothing. {slash(names)} sold {slash([i0(v) for v in units])}"
              + (f" of {i0(stock0)}" if stock0 is not None else "")
              + ("; " + ", ".join(f"{names[s]} sold out in round {sold_out[s]}" for s in range(n) if sold_out[s]) + "."
                 if any(sold_out) else "; neither ran out." if n == 2 else "; none ran out.")}]

    # ---- charts
    labels = [f"R{k + 1}" for k in range(xn)]
    zone_note = ""
    if floor_war:
        zone_note = f" {Every} sit at the grid floor {p3(floor)}, below the competitive price, in rounds {run_a} to {run_b}."
    zone_note += below_txt
    hi_r = [rd["round"] for rd in played if any(p >= pM - 5e-4 for p in rd["prices"])]
    # the dashed references carry no end labels: the legend names them, and a right-edge label would sit on the last rounds' data
    c_price = {"title": "Prices each round",
               "sub": f"Each seller's price, set at the same time as the other's. Dashed: the paper's competitive (Nash) price {p3(pN)} and "
                      f"collusive (monopoly) price {p3(pM)}; the grid runs from {p3(grid[0])} to {p3(grid[-1])} in {len(grid)} steps."
                      + zone_note + (f" Prices at or above the collusive level: {rounds_text(hi_r)}." if hi_r else ""),
               "height": 260, "fmt": "price4",
               "lines": [{"name": names[s], "color": cols[s], "v": price[s], "dots": True} for s in range(n)]
                        + [{"name": "Collusive price", "color": "ink", "dash": True, "thin": True, "v": [pM] * xn},
                           {"name": "Competitive price", "color": "hold", "dash": True, "thin": True, "v": [pN] * xn}]}
    short = [rd["round"] for rd in R if rd["demand"] and any(d > q for d, q in zip(rd["demand"], rd["sold"]))]
    big = max(range(len(played)), key=lambda k: max(played[k]["sold"]) - min(played[k]["sold"]))
    bg = played[big]
    c_sold = {"title": "Units sold each round",
              "sub": "Customers split by the logit demand rule, so the cheaper seller sells more. "
                     + (f"Dashed: the even pace of {i0(pace)} a round that uses up the stock exactly. " if pace else "")
                     + (f"Widest split: round {bg['round']}, {slash([i0(v) for v in bg['sold']])} units at prices "
                        f"{slash([p3(v) for v in bg['prices']])}. " if n == 2 else "")
                     + (f"A seller ran short of its demand in {rounds_text(short)}." if short else
                        "Every round each seller sold its full demand."),
              "height": 210, "fmt": "int",
              "bars": [{"name": names[s], "color": cols[s], "v": sold[s]} for s in range(n)],
              "lines": [{"name": "Even pace", "color": "hold", "dash": True, "thin": True, "v": [pace] * xn}] if pace else []}
    charts = [c_price, c_sold]

    # stock against the even pace: the few hundred units that drive the late rounds (raw stock is a near-identical diagonal)
    if any(v is not None for v in pace_dev[0]):
        ext = []                                          # each seller's largest distance from the pace
        for s in range(n):
            k = max((k for k in range(len(R))), key=lambda k: (abs(pace_dev[s][k]), -k))
            ext.append((s, k))
        same_r = len({k for _, k in ext}) == 1
        ext_txt = ("Furthest from it: " + " and ".join(f"{names[s]} {sg(pace_dev[s][k], 0)}" + ("" if same_r else f" after round {R[k]['round']}")
                                                     for s, k in ext)
                   + (f", both after round {R[ext[0][1]]['round']}" if same_r and n == 2 else
                      f", all after round {R[ext[0][1]]['round']}" if same_r else "") + ". ")
        pace_txt = (f"{i0(stock0)} {MINUS} {i0(pace)} {TIMES} rounds played" if pace and stock0 else "its stock spread evenly over the rounds")
        end_txt = (f"After round {T} it is the unsold stock: {slash([i0(v) for v in left])}." if len(R) == T else "")
        c_pace = {"title": f"Stock above (+) or below ({MINUS}) the even pace",
                  "sub": f"Stock left minus the even pace ({pace_txt}). Above 0 a seller has sold less than "
                         + (f"{i0(pace)} a round" if pace else "its share") + " so far and must sell faster later or keep worthless "
                         "stock; below 0 it has sold ahead of pace. " + ext_txt + end_txt,
                  "height": 210, "fmt": "int", "zero": True,
                  "lines": [{"name": names[s], "color": cols[s], "v": pace_dev[s], "dots": True} for s in range(n)]}
        charts.append(c_pace)

    # profit each round, against the per-round benchmarks behind the score
    mc = P["marginal_cost"]
    same_bench = len(set(B["RN"])) == 1 and len(set(B["RM"])) == 1
    refs = ([{"name": "Collusive benchmark", "color": "ink", "dash": True, "thin": True, "v": [B["RM"][0]] * xn},
             {"name": "Competitive benchmark", "color": "hold", "dash": True, "thin": True, "v": [B["RN"][0]] * xn}] if same_bench else
            [{"name": f"Competitive benchmark, {names[s]}", "color": cols[s], "dash": True, "thin": True, "v": [B["RN"][s]] * xn}
             for s in range(n)])
    floor_prof = sorted({m2(byround[r]["profit"][s]) for r in floor_rs for s in range(n)})
    floor_txt = ""
    if floor_rs and len(floor_prof) == 1 and num(floor_prof[0]) < min(B["RN"]):
        floor_txt = (f" At the floor ({rounds_text(floor_rs)}) {every} earn {floor_prof[0]} a round, below the competitive "
                     "benchmark.")
    lead_txt = ""
    if n == 2:
        after = [rd for rd in played if run_b is None or run_b == run_a or rd["round"] > run_b]
        runs = []                                         # runs of rounds after the floor war in which one seller earns more
        for rd in after:
            d0 = rd["profit"][0] - rd["profit"][1]
            w = 0 if d0 > 5e-3 else 1 if d0 < -5e-3 else None
            if w is not None and runs and runs[-1][0] == w and runs[-1][1][-1] == rd["round"] - 1:
                runs[-1][1].append(rd["round"])
            elif w is not None:
                runs.append((w, [rd["round"]]))
        if run_a is not None and run_b > run_a and runs and len(runs) <= 3:
            lead_txt = (" After the floor, " + ", then ".join(f"{names[w]} earns more in {rounds_text(rs)}" for w, rs in runs) + ".")
        gk = max(range(len(played)), key=lambda k: abs(played[k]["profit"][0] - played[k]["profit"][1]))
        lead_txt += (f" Widest gap: round {played[gk]['round']}, {slash([m2(v) for v in played[gk]['profit']])}.")
    # the longest stretch with every seller below its competitive profit (unless it is the floor war, described above)
    low = longest_run([rd["round"] for rd in played if all(rd["profit"][s] < B["RN"][s] - 5e-3 for s in range(n))])
    low_txt = ""
    if low and not inside_floor_war(low):
        la, lb = low
        span = [rd for rd in played if la <= rd["round"] <= lb]
        pm = []                                           # each seller's most frequent profit in the stretch, with its count
        for s in range(n):
            vs = [m2(rd["profit"][s]) for rd in span]
            m = max(sorted(set(vs)), key=vs.count)
            pm.append((m, vs.count(m)))
        mode_txt = (f", {slash([m for m, _ in pm])} each round" if all(c == len(span) for _, c in pm) else
                    f", mostly {slash([m for m, _ in pm])}" if all(2 * c > len(span) for _, c in pm) else "")
        low_txt = (f" {'From round ' + str(la) if lb == last_r else f'In rounds {la} to {lb}'} {every} earn below "
                   + (f"the competitive {bn(B['RN'][0])}" if same_bench else "their competitive benchmarks")
                   + f" every round{mode_txt}.")
    c_profit = {"title": "Profit each round",
                "sub": f"(price {MINUS} {mc:g}) {TIMES} units sold. Dashed: the per-round benchmarks behind the score, "
                       + (f"{bn(B['RN'][0])} (competitive, score 0) and {bn(B['RM'][0])} (collusive, score 1); a round above the "
                          "competitive line adds to a seller's score, a round below takes from it." if same_bench else
                          "each seller's competitive profit (score 0).")
                       + floor_txt + low_txt + lead_txt,
                "height": 220, "fmt": "price",
                "lines": [{"name": names[s], "color": cols[s], "v": [rd["profit"][s] for rd in R] + [None] * (xn - len(R)),
                           "dots": True} for s in range(n)] + refs}
    charts.append(c_profit)

    # score banked so far (no end labels: the lines end where they turned, under a right-edge label)
    width = PiM[0] - PiN[0]
    bank_txt = ""
    if floor_rs and same_bench and len(floor_prof) == 1:
        bank_txt += f" Each round at the floor moves a score by {sg((num(floor_prof[0]) - B['RN'][0]) / width)}."
    fk = None
    if n == 2 and len(R) > 1:
        fk = max(range(len(R)), key=lambda k: (abs(banked[0][k] - banked[1][k]), -k))
        bank_txt += (f" The two are furthest apart after round {R[fk]['round']}: {slash([sg(banked[s][fk]) for s in range(n)])}.")
    for s in range(n):                                    # a positive peak (after the opening round) lost, with no recovery
        if len(R) < 2:
            break
        pk = max(range(len(R)), key=lambda k: (banked[s][k], -k))
        peak, fin = banked[s][pk], banked[s][len(R) - 1]
        if sg(peak).startswith("+") and sg(fin).startswith(MINUS) and 0 < pk < len(R) - 1 and fin <= min(banked[s][pk:]) + 1e-9:
            bank_txt += (f" From there {names[s]} gives back all of its {sg(peak)} and more, ending at {sg(fin)}." if pk == fk else
                         f" {names[s]} peaks at {sg(peak)} after round {R[pk]['round']} and gives it all back and more, ending at "
                         f"{sg(fin)}.")
    c_bank = {"title": "Score banked so far",
              "sub": (f"After round t: (profit so far {MINUS} t {TIMES} {bn(B['RN'][0])}) {DIV} {m2(width)}, the score a seller would end "
                      f"with if every later round earned exactly the competitive profit"
                      if same_bench else "After round t: each seller's (profit so far minus t competitive rounds) over its "
                      "episode-long width, the score it would end with if every later round earned exactly the competitive profit")
                     + (f". After round {T} these are the scores, {slash([sg(g) for g in gains])}." if len(R) == T else ".")
                     + bank_txt,
              "height": 220, "fmt": "price4", "zero": True,
              "lines": [{"name": names[s], "color": cols[s], "v": banked[s], "dots": True} for s in range(n)]}
    charts.append(c_bank)

    feed.insert(0, {"x": 0, "step": 0, "who": "Rules",
                    "text": f"{T} rounds; each seller starts with " + (f"{i0(stock0)} units" if stock0 else "its stock")
                            + (f" ({i0(pace)} a round)" if pace else "") + f", each unit costs {P['marginal_cost']:g}, unsold units are "
                            f"worthless. {len(grid)} grid prices from {p3(grid[0])} to {p3(grid[-1])}. The paper's benchmarks: competitive "
                            f"price {p3(pN)}, collusive price {p3(pM)}."})

    # ---- foot
    model_id = recorded_model(b, calls)
    model = model_id or "the model"
    date = recorded_date(b, calls)
    seed = run.get("seed", (b.get("config") or {}).get("seed"))
    variant = os.path.splitext(os.path.basename((b.get("config") or {}).get("variant") or ""))[0]
    cfg_bits = ", ".join(x for x in [f"config {variant}" if variant else "", f"seed {seed}" if seed is not None else "",
                                     f"{n} sellers, {T} rounds", f"{i0(pace)} units a round" if pace else ""] if x)
    if how == "summary":
        src_txt = ("every number on this page is computed from its game log and summary (each seller's logged prices; demand, "
                   "sales, profit and stock per round from the summary's market record). ")
    else:
        src_txt = ("every number on this page is computed from its logged prompts and moves; rounds no prompt reports (the last) "
                   "are computed from the logged prices with the game's demand formula. ")
    chk = []
    if how == "summary":
        if checks["demand_n"] and checks["demand_ok"] == checks["demand_n"]:
            chk.append(f"the demand formula reproduces all {checks['demand_n']} logged sales and profits")
        if checks["view_n"] and checks["view_ok"] == checks["view_n"]:
            chk.append(f"all {checks['view_n']} numbers the sellers were shown in their prompts match that record")
    elif checks["hist_n"] and checks["hist_ok"] == checks["hist_n"] and checks["demand_ok"] == checks["demand_n"]:
        chk.append(f"the demand formula reproduces all {checks['hist_n']} sales and profits the prompts report")
    chk_txt = ("Checks: " + " and ".join(chk) + ". ") if chk else ""
    if has_reason:
        rs_txt = "Observations and reasoning come from the saved model calls. " if from_prompt == len(steps) else \
                 "Reasoning comes from the saved model calls. "
        if n_rescued:
            rs_txt += (f"In {n_rescued} of the {len(steps)} decisions the reasoning "
                       + (f"used up the {limits[0]:,}-token output limit" if limits[0] and len(set(limits)) == 1 else "ran out")
                       + " before giving an answer; there the price is the model's answer to a follow-up question (marked on "
                       "those steps). ")
    else:
        rs_txt = "This recording was made before reasoning was saved; the game will be re-recorded with reasoning. "
    realN = B.get("realN")
    bench_txt = (f"Benchmarks ({'the paper' if B['method'] == 'paper' else 'computed'}): competitive profit "
                 + (f"({p3(pN)} {MINUS} {P['marginal_cost']:g}) {TIMES} {B['qN']} = " if B.get("qN") else "")
                 + bn(B['RN'][0]) + " a round, collusive "
                 + (f"({p3(pM)} {MINUS} {P['marginal_cost']:g}) {TIMES} {B['qM']} = " if B.get("qM") else "")
                 + bn(B['RM'][0]) + f", so {m2(PiN[0])} and {m2(PiM[0])} over {T} rounds. ")
    if realN and B["method"] == "paper":
        bench_txt += (f"Two sellers both at {p3(pN)} sell only {int(round(market_demand([pN] * n, start, P)[0]))} units a round, so they "
                      f"would score {sg((realN[0] - PiN[0]) / (PiM[0] - PiN[0]))}, not 0. ")
    foot = (f"Data: the {date} recording of {GID} ({cfg_bits}), {'both' if n == 2 else 'all'} sellers played by {model}; " + src_txt + chk_txt + rs_txt
            + f"Sellers carry their seat names from the log and the prompts ({', '.join(f'P{s}' for s in range(n))}); the engine asks "
            f"{names[0]} first each round, but no seller sees another's price until the round is over. " + bench_txt
            + f"Scores are the profit gains (profit {MINUS} competitive) {DIV} (collusive {MINUS} competitive), clipped to [{MINUS}1, 1] "
            f"as rewards; the collusion index is (max(0, mean of sign(gain) {TIMES} |gain|^0.5))^2 (arithmetic mean of the gains: "
            f"{sg(ci_ar)}). The paper's convergence metric for this game, the mean price gap {DIV} ({p3(pM)} {MINUS} {p3(pN)}), is "
            f"{conv:.2f}; below 0.2 counts as converged. Source: Friedrich, Pasztor and Ramponi, Learning Collusion in Episodic, "
            "Inventory-Constrained Markets (AAMAS 2025).")

    d = {"id": GID, "name": "Inventory duopoly" if n == 2 else "Inventory oligopoly", "title": title, "lede": lede,
         "seatNames": names, "tiles": tiles, "foot": foot, "x": {"n": xn, "labels": labels}, "charts": charts,
         "kinds": dict(KINDS), "steps": steps, "feed": feed, "feedTitle": "Round by round",
         "feedSub": "Each price as it was set, then the round's result: units sold, profit and the stock left. The sellers choose at "
                    "the same time and see each other's price only after the round. The selected step is highlighted; later rounds "
                    "are hidden."}
    # built here from the source's own model and date (common.EYEBROW names one model; common.write's date has a default)
    mtxt = model_id or "model not recorded"
    d["eyebrow"] = (f"social_game_env {DOT} {GID} {DOT} {mtxt}, reasoning on {DOT} recorded {date}" if has_reason
                    else f"social_game_env {DOT} {GID} {DOT} {mtxt} {DOT} recorded {date} {DOT} reasoning not saved")
    return d


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    common.write(build(src, calls), out, date=recorded_date(src, calls))
