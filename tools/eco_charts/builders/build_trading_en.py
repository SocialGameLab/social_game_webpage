"""Build one JSON per trading game for the replay page (English) from social-game-env demo runs.

usage: build_trading_en.py DEMO_DIR OUT_DIR [--no-history] [GAME ...]

Reads demo/<game>/<game>/ (run.json and its event log). Every `move` event is one decision; the state its model saw is
the move's view from "You are seat" up to the action format, cut at 4,000 characters as on the replay pages. With
--no-history it builds the second recording (<game>_no_history) of the games that have one, into <game>__no_history.json.
Reasoning is not read. Without GAME arguments it builds all seven games.
"""
import json, math, os, re, sys

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DEMO, OUT, ONLY = ARGS[0], ARGS[1], set(ARGS[2:])
NOHIST = "--no-history" in sys.argv[1:]
os.makedirs(OUT, exist_ok=True)
EYEBROW = "social_game_env · {} · deepseek-flash, reasoning on · recorded {}{}"
CUT = re.compile(r"(Legal actions:|sample actions:|Action schema|Respond with exactly one JSON)", re.S)
DATE = {}


def run_dir(game):
    return os.path.join(DEMO, game, game + ("_no_history" if NOHIST else ""))


def steps_of(game):
    """Each decision's seat, the state its model saw and its move, and the final rewards by seat."""
    rec = json.load(open(os.path.join(run_dir(game), "run.json"), encoding="utf-8"))
    DATE[game] = rec["recorded_at"][:10]
    out = []
    for line in open(os.path.join(run_dir(game), rec["events_file"]), encoding="utf-8"):
        e = json.loads(line) if line.strip() else {}
        if e.get("event") != "move":
            continue
        v = e.get("view") or ""
        i = v.find("You are seat"); s = v[i:] if i >= 0 else v[-3000:]
        m = CUT.search(s); s = (s[:m.start()] if m else s).strip()
        if len(s) > 4000:
            s = s[:4000] + " …(truncated)"
        out.append({"seat": int(e.get("actor") or 0), "state": s, "act": e.get("move") or "", "thinking": ""})
    return out, {str(k): v for k, v in ((rec.get("result") or {}).get("rewards") or {}).items()}


def num(s):
    return float(s.replace(",", "").replace("$", "").rstrip("."))


def aid(s):
    try:
        return json.loads(s)
    except Exception:
        return {"action_id": "?", "payload": {}}


def f(x, d=2):
    return None if x is None else round(x, d)


def usd(v, signed=False):
    """−$1,234.50, or +$1,234.50 when signed (profits)."""
    return ("−" if v < 0 else "+" if signed and v > 0 else "") + f"${abs(v):,.2f}"


def usd_short(v):
    """Large cash amounts in millions: −$1.89M."""
    return ("−" if v < 0 else "") + f"${abs(v) / 1e6:,.2f}M" if abs(v) >= 1e6 else usd(v)


def sgn(v, d=1, unit="%"):
    return ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v):.{d}f}{unit}"


def write(game, d):
    d["id"] = game
    d["eyebrow"] = EYEBROW.format(game, DATE.get(game, ""), ", no chat history" if NOHIST else "")
    if NOHIST:
        d["foot"] = ("This run was recorded without chat history: each prompt was the seat’s only input, as the source specifies. "
                     "The other run of this game gave each seat its own recent turns as chat history. ") + d.get("foot", "")
    d.setdefault("seatNames", ["Model"])
    sp = d.pop("spark", None)
    if sp is not None:
        vals = [v for v in sp if v is not None]; step = max(1, len(vals) // 120)
        d["tiles"][0]["spark"] = vals[::step] + ([vals[-1]] if (len(vals) - 1) % step else [])
    txt = json.dumps(d, ensure_ascii=False)
    open(f"{OUT}/{game}{'__no_history' if NOHIST else ''}.json", "w", encoding="utf-8").write(txt)
    print(game, "steps", len(d["steps"]), "x", d["x"]["n"], "charts", len(d["charts"]))


def book_of(S, ask_pat, bid_pat, sep, scale=1.0):
    out = {}
    for side, pat in (("asks", ask_pat), ("bids", bid_pat)):
        m = re.search(pat, S); rows = []
        if m:
            for part in m.group(1).rstrip(".").split(sep):
                q = re.search(r"\$?([\d,]+(?:\.\d+)?) x ([\d,]+)", part)
                if q: rows.append([num(q.group(1)) * scale, num(q.group(2))])
        out[side] = rows
    return out if out["asks"] or out["bids"] else None


# ---------------------------------------------------------------- InvestorBench
def build_eco_investorbench():
    st, rw = steps_of("eco_investorbench")
    days = []
    for s in st:
        S = s["state"]
        g = lambda pat: (re.search(pat, S) or [None, None])[1]
        days.append({"date": g(r"Date: (\d{4}-\d\d-\d\d)"), "day": int(g(r"Trading day (\d+) of")), "close": float(g(r"Adjusted close today: (\d+(?:\.\d+)?)")),
                     "momentum": g(r"past 3 trading days is ([+-]\d+(?:\.\d+)?)%"), "news": re.findall(r"^\d+\. (.*)$", S, re.M),
                     "action": json.loads(s["act"])["action_id"], "thinking": "",
                     "pos_ret": g(r"it returned ([+-]?\d+(?:\.\d+)?)% today"), "cum": g(r"Your cumulative return so far: ([+-]?\d+(?:\.\d+)?)%")})
    n = len(days); close = [x["close"] for x in days]

    def ib_head(i):
        x = days[i]; side = "long" if x["action"] == "buy" else "short"
        h = (f"{x['date']} · day {x['day']} · close {x['close']:.2f}" + (f", 3-day momentum {x['momentum']}%" if x["momentum"] is not None else "")
             + (f", {len(x['news'])} news item{'s' if len(x['news']) != 1 else ''}" if x["news"] else ", no news") + f" · goes {side} for the next day")
        if i + 1 < n and days[i + 1]["pos_ret"] is not None:
            h += f"; that day the {side} position returned {days[i + 1]['pos_ret']}%, {days[i + 1]['cum']}% in all"
        return h
    r = [0.0] + [math.log(close[i] / close[i - 1]) * 100 for i in range(1, n)]
    held = [0] + [1 if days[i - 1]["action"] == "buy" else -1 for i in range(1, n)]
    model, acc = [], 0.0
    for i in range(n):
        acc += held[i] * r[i]; model.append(f(acc))
    hold = [f(math.log(c / close[0]) * 100) for c in close]
    hits = sum(1 for i in range(n - 1) if (1 if days[i]["action"] == "buy" else -1) * r[i + 1] > 0)
    switch = [i for i in range(n) if i == 0 or days[i]["action"] != days[i - 1]["action"]]
    score = rw["0"]
    nl = sum(x["action"] == "buy" for x in days)
    write("eco_investorbench", {"spark": model,
        "name": "InvestorBench", "title": f"Trading HON stock for {n} days",
        "lede": "After each close the model sees the day's price, the last three days' momentum and up to three news items, then goes long (buy) or short (sell) for the next day. Returns add up as daily log returns, exactly as the game scores them.",
        "tiles": [{"k": "Model return", "v": sgn(model[-1]), "s": f"Game score {score:.3f} (1 = every day called right)"},
                  {"k": "Buy and hold", "v": sgn(hold[-1]), "s": "Bought on day one and kept"},
                  {"k": "Direction hit rate", "v": f"{100 * hits / (n - 1):.0f}%", "s": f"Decision matched the next day's move on {hits} of {n - 1} days"},
                  {"k": "Position flips", "v": f"{len(switch) - 1}", "s": f"Long {nl} days, short {n - nl} days, never flat"}],
        "foot": "Game score = the model's cumulative log return divided by the return of perfect foresight (167.06% for this stock and window), so it lies between −1 and 1; buy and hold scores 0.206. The model sees the date and the ticker and may remember this real price history; that has not been tested.",
        "x": {"n": n, "labels": [x["date"] for x in days]},
        "charts": [
            {"title": "Price and position", "sub": "Adjusted close. The strip below shows the side held each day; triangles mark the days the model flipped.", "height": 280, "fmt": "price",
             "lines": [{"name": "Close", "color": "ink", "v": close}], "band": {"v": held, "pos": "Long", "neg": "Short"},
             "markers": [{"x": i, "y": close[i], "kind": days[i]["action"]} for i in switch]},
            {"title": "Cumulative return", "sub": "The model's long/short calls against buy and hold, in percentage points of log return.", "height": 220, "fmt": "pct", "zero": True,
             "lines": [{"name": "Buy and hold", "color": "hold", "dash": True, "v": hold, "end": True}, {"name": "Model", "color": "ink", "v": model, "end": True, "area": True}]}],
        "kinds": {"buy": ["Long ▲ buy", "c1"], "sell": ["Short ▼ sell", "c2"]},
        "steps": [{"seat": 0, "x": i, "head": ib_head(i), "kind": x["action"], "act": json.dumps({"action_id": x["action"], "payload": {}}),
                   "state": f"Date: {x['date']}. Adjusted close: {x['close']:.2f}. Momentum (3 days): {x['momentum']}%.\n" + "\n".join(f"{k + 1}. {t}" for k, t in enumerate(x["news"])),
                   "thinking": x["thinking"]} for i, x in enumerate(days)]})


# ---------------------------------------------------------------- two-trader liquidation
def build_eco_liquidation_2p():
    st, rw = steps_of("eco_liquidation_2p")
    R = int(re.search(r"Round \d+ of (\d+)", st[0]["state"]).group(1))
    rounds = sorted({int(re.search(r"Round (\d+) of", s["state"]).group(1)) for s in st})
    last_r = max(rounds); xn = last_r + 1
    block = num(re.search(r"Your block: ([\d,]+) shares", st[0]["state"]).group(1))
    price = [None] * xn; holdv = {0: [None] * xn, 1: [None] * xn}; sold = {0: [None] * xn, 1: [None] * xn}; execp = {0: [None] * xn, 1: [None] * xn}
    steps = []; last = {}
    for s in st:
        S = s["state"]; rd = int(re.search(r"Round (\d+) of", S).group(1))
        price[rd] = num(re.search(r"Market price now: \$([\d,.]+)", S).group(1))
        holdv[s["seat"]][rd] = num(re.search(r"You still hold ([\d,]+) shares", S).group(1)) if "You still hold" in S else block
        m = re.search(r"Your recent sales \(round: shares at execution price\): (.*)", S)
        if m:
            for r_, q, p in re.findall(r"(\d+): ([\d,]+) at \$([\d.]+\d)", m.group(1)):
                sold[s["seat"]][int(r_)] = num(q); execp[s["seat"]][int(r_)] = float(p)
        fr = aid(s["act"]).get("payload", {}).get("fraction")
        last[s["seat"]] = (rd, fr)
        steps.append({"seat": s["seat"], "x": rd, "head": f"Round {rd} · sells {fr * 100:.1f}% of what is left" if fr is not None else f"Round {rd}",
                      "kind": "sell", "act": s["act"], "state": S, "thinking": s["thinking"]})
    # A trader that stops deciding before the last round sold out in its last one; no later view of its own shows that
    # sale, so it is its order: the fraction it chose of the shares it still held.
    for k, (rd, fr) in last.items():
        if fr is not None and rd + 1 < xn and holdv[k][rd] is not None and holdv[k][rd + 1] is None:
            q = float(round(holdv[k][rd] * fr))
            holdv[k][rd + 1] = holdv[k][rd] - q
            if sold[k][rd] is None:
                sold[k][rd] = q or None
    for k in (0, 1):
        holdv[k][0] = block
        for i in range(1, xn):
            if holdv[k][i] is None: holdv[k][i] = holdv[k][i - 1]
            if sold[k][i] is None and i < xn - 1 and holdv[k][i + 1] is not None:
                sold[k][i] = max(0.0, holdv[k][i] - holdv[k][i + 1]) or None
    for stp in steps:
        k, rd = stp["seat"], stp["x"]; fr = aid(stp["act"]).get("payload", {}).get("fraction")
        if fr is None or holdv[k][rd] is None:
            continue
        q, px, left = sold[k][rd], execp[k][rd], holdv[k][rd + 1] if rd + 1 < xn else None
        stp["head"] = (f"Round {rd}" + (f" · price ${price[rd]:,.4f}" if price[rd] is not None else "") + f" · sells {fr * 100:.1f}% of its {holdv[k][rd]:,.0f} shares" + (f": {q:,.0f} shares" if q else "")
                       + (f" at ${px:,.4f}" if px is not None else "") + (f" · {left:,.0f} left" if left is not None else ""))
    write("eco_liquidation_2p", {"seatNames": ["Trader 1", "Trader 2"], "spark": holdv[0],
        "name": "Two-trader liquidation", "title": f"Two traders unloading 500,000 shares each (up to {R} rounds)",
        "lede": "Each trader must sell a 500,000-share block. Every round both decide at the same time what fraction of their remaining shares to sell. Selling fast pushes the execution price down, and part of that impact stays in the price; holding exposes the trader to price risk. Score = 1 − cost ÷ the cost of dumping the whole block in round one.",
        "tiles": [{"k": "Trader 1 score", "v": f"{rw['0']:.3f}", "s": "1 − cost ÷ cost of selling all at once"},
                  {"k": "Trader 2 score", "v": f"{rw['1']:.3f}", "s": "Same scale"},
                  {"k": "Sold in round 1", "v": f"{sold[0][1] or 0:,.0f} / {sold[1][1] or 0:,.0f}", "s": "Trader 1 / trader 2, shares"},
                  {"k": "Rounds used", "v": f"{last_r}", "s": f"Out of {R} allowed"}],
        "foot": "Holdings are the shares left at the start of each round. Sales and execution prices come from the trade record each trader sees. Execution price = market price − fixed cost − temporary impact of both traders' combined sales that round. A trader's last sale, which no later view of its own shows, is its last order: the fraction it chose of the shares it still held.",
        "x": {"n": xn, "labels": [f"R{i}" for i in range(xn)]},
        "charts": [
            {"title": "Shares still held", "sub": "Each trader's remaining block at the start of every round.", "height": 230, "fmt": "int",
             "lines": [{"name": "Trader 1", "color": "c1", "v": holdv[0], "end": True, "area": True}, {"name": "Trader 2", "color": "c2", "v": holdv[1], "end": True, "area": True}]},
            {"title": "Shares sold per round", "sub": "The more both sell in the same round, the lower that round's execution price.", "height": 210, "fmt": "int",
             "bars": [{"name": "Trader 1", "color": "c1", "v": sold[0]}, {"name": "Trader 2", "color": "c2", "v": sold[1]}], "lines": []},
            {"title": "Market price and execution prices", "sub": "Line: market price at the start of each round. Dots: what each trader actually received.", "height": 220, "fmt": "price2",
             "lines": [{"name": "Market price", "color": "ink", "v": price}, {"name": "Trader 1 fills", "color": "c1", "v": execp[0], "dots": True, "noline": True},
                       {"name": "Trader 2 fills", "color": "c2", "v": execp[1], "dots": True, "noline": True}]}],
        "kinds": {"sell": ["Sell", "c2"]}, "steps": steps})


# ---------------------------------------------------------------- ABIDES daily investor
def build_eco_abides():
    st, rw = steps_of("eco_abides")
    xn = len(st); lab, last, prof, shares, cash, bb, ba, marks, steps = [], [], [], [], [], [], [], [], []
    for i, s in enumerate(st):
        S = s["state"]
        lab.append(re.search(r"Time (\d\d:\d\d):\d\d", S).group(1))
        m = re.search(r"Market: last trade \$([\d,.]+\d)", S); last.append(num(m.group(1)) if m else None)
        m = re.search(r"\(profit ([+-]?)\$([\d,.]+\d)\)", S); prof.append((-1 if m.group(1) == "-" else 1) * num(m.group(2)) if m else None)
        m = re.search(r"Your account: cash -?\$[\d,.]+, (-?[\d,]+) shares", S); shares.append(num(m.group(1)) if m else None)
        m = re.search(r"Your account: cash (-?)\$([\d,.]+\d)", S); cash.append((-1 if m.group(1) else 1) * num(m.group(2)) if m else None)
        bk = book_of(S, r"Asks \(best first[^)]*\): (.*)", r"Bids \(best first[^)]*\): (.*)", ", ")
        bb.append(bk["bids"][0][0] if bk and bk["bids"] else None); ba.append(bk["asks"][0][0] if bk and bk["asks"] else None)
        k = aid(s["act"])["action_id"]
        if k in ("buy", "sell") and last[-1] is not None: marks.append({"x": i, "y": last[-1], "kind": k})
        steps.append({"seat": 0, "x": i, "head": f"{lab[-1]} · decision {i + 1}", "kind": k, "act": s["act"], "state": S, "thinking": s["thinking"],
                      "book": dict(bk, last=last[-1], unit="$") if bk else None})
    close_at = (re.search(r"until the close at (\d\d:\d\d)", st[0]["state"]) or [None, None])[1]
    mkt = lambda i: (f"last ${last[i]:,.2f}" if last[i] is not None else "no trade yet") + (f", bid ${bb[i]:,.2f} / ask ${ba[i]:,.2f}" if bb[i] and ba[i] else "")
    for i, stp in enumerate(steps):
        k = aid(stp["act"])["action_id"]
        if i + 1 < xn and None not in (shares[i], shares[i + 1], cash[i], cash[i + 1], prof[i + 1]):
            ds, dc = shares[i + 1] - shares[i], cash[i + 1] - cash[i]
            did = (f"buys {ds:,.0f} shares at ${-dc / ds:,.2f}" if ds > 0 else f"sells {-ds:,.0f} shares at ${dc / -ds:,.2f}" if ds < 0
                   else "holds" if k == "hold" else f"{k} order not filled")
            stp["head"] = f"{lab[i]} · {mkt(i)} · {did} · now {shares[i + 1]:,.0f} shares, cash {usd_short(cash[i + 1])}, profit {usd(prof[i + 1], True)}"
        elif k != "hold":
            stp["head"] = f"{lab[i]} · {mkt(i)} · {k} order " + ("sent at the close, so it is not filled" if lab[i] == close_at else "sent as the game ends")
    cnt = {k: sum(aid(s["act"])["action_id"] == k for s in st) for k in ("buy", "sell", "hold")}
    pr = [v for v in prof if v is not None]
    # The exchange never checks cash (as in ABIDES-Gym), so buys run on unlimited credit; say how far this run went.
    lev = ""
    if shares[-1] and last[-1] and cash[-1] is not None and last[0]:
        pos = shares[-1] * last[-1]
        first_neg = next((i for i, c in enumerate(cash) if c is not None and c < 0), None)
        lev = (f" The exchange does not check cash, as in ABIDES-Gym, so buys run on credit: "
               + (f"after decision {first_neg} the cash was {'−' if cash[first_neg] < 0 else ''}${abs(cash[first_neg]):,.2f}, and " if first_neg else "")
               + f"at the last decision the trader held {shares[-1]:,.0f} shares worth about ${pos:,.0f}, {pos / 10000:,.0f} times its $10,000 starting cash, "
               + f"while the last trade moved {sgn((last[-1] / last[0] - 1) * 100, 3)} over the game.")
    write("eco_abides", {"spark": prof,
        "name": "ABIDES", "title": "A day as an investor on the ABIDES exchange",
        "lede": "The trader starts with $10,000 in cash. Through the session it decides at intervals to buy 100 shares, sell 100 shares or hold, always with market orders. It sees its account, the last trade and five levels of bids and asks. The score is the change in account value by the close.",
        "tiles": [{"k": "Score", "v": f"{rw['0']:.4f}", "s": "Change in account value by the close"},
                  {"k": "Final P&L", "v": (f"{'−' if pr[-1] < 0 else '+'}${abs(pr[-1]):,.2f}" if pr else "—"), "s": "Marked at the last trade, minus $10,000"},
                  {"k": "Final position", "v": f"{shares[-1]:,.0f} shares" if shares[-1] is not None else "—", "s": f"{cnt['buy']} buys, {cnt['sell']} sell, {cnt['hold']} holds"},
                  {"k": "Price range", "v": f"${min(v for v in last if v):,.2f}–${max(v for v in last if v):,.2f}", "s": "Last trades seen during the game"}],
        "foot": "ABIDES-Gym task markets-daily_investor-v0; every other trader on the exchange is simulated. Each decision is a 100-share market order, so buys fill at the ask and sells at the bid." + lev,
        "x": {"n": xn, "labels": lab},
        "charts": [
            {"title": "Last trade, best bid and best ask", "sub": "Thick line: last trade. Thin lines: best ask and best bid. Triangles: the model's buys and sells.", "height": 250, "fmt": "price2",
             "lines": [{"name": "Best ask", "color": "c2", "v": ba, "thin": True}, {"name": "Best bid", "color": "c1", "v": bb, "thin": True}, {"name": "Last trade", "color": "ink", "v": last}], "markers": marks},
            {"title": "Shares held", "sub": "Position at each decision.", "height": 170, "fmt": "int", "lines": [{"name": "Shares", "color": "c1", "v": shares, "area": True, "step": True}]},
            {"title": "Profit and loss", "sub": "Account value at the last trade minus the starting $10,000.", "height": 170, "fmt": "money", "zero": True,
             "lines": [{"name": "P&L", "color": "c1", "v": prof, "area": True}]}],
        "book": {"title": "Order book at this step", "sub": "Five levels on each side; longer bars mean more shares resting at that price."},
        "kinds": {"buy": ["Buy 100", "c1"], "sell": ["Sell 100", "c2"], "hold": ["Hold", "muted"]}, "steps": steps})


# ---------------------------------------------------------------- MarS execution
def build_eco_mars():
    st, rw = steps_of("eco_mars")
    RN = int(re.search(r"round \d+ of (\d+)", st[0]["state"]).group(1))
    rds = [int(re.search(r"round (\d+) of", s["state"]).group(1)) for s in st]
    xn = max(rds) + 1
    mid = [None] * xn; fill = [None] * xn; avg = [None] * xn; steps = []
    for s, rd in zip(st, rds):
        S = s["state"]
        m = re.search(r"mid ([\d,]+) \(([+-]?[\d.]+) bps vs arrival\)", S); mid[rd] = float(m.group(2)) if m else None
        m = re.search(r"Filled so far: ([\d,]+) shares \(([\d.]+)%\)", S); fill[rd] = float(m.group(2)) if m else 0.0
        m = re.search(r"at average price [\d,]+ \(([+-]?[\d.]+) bps vs arrival", S); avg[rd] = float(m.group(1)) if m else None
        lp = re.search(r"last price ([\d,]+)", S)
        bk = book_of(S, r"asks \(price x shares\): (.*)", r"bids \(price x shares\): (.*)", " | ")
        p = aid(s["act"]).get("payload", {})
        steps.append({"seat": 0, "x": rd, "head": f"Round {rd} of {RN} · bids for {p.get('volume', 0):,} at {p.get('price', 0):,}", "kind": "buy", "act": s["act"], "state": S,
                      "thinking": s["thinking"], "book": dict(bk, last=num(lp.group(1)) if lp else None, unit="", mine=p.get("price")) if bk else None})
    for j, stp in enumerate(steps):
        p = aid(stp["act"]).get("payload", {})
        h = f"Round {stp['x']} of {RN}" + (f" · mid {mid[stp['x']]:+.1f} bps against arrival" if mid[stp["x"]] is not None else "") + f" · bids for {p.get('volume', 0):,} shares at {p.get('price', 0):,}"
        m = re.search(r"Filled so far: ([\d,]+) shares \(([\d.]+)%\) at average price ([\d,]+(?:\.\d+)?) \(([+-]?[\d.]+) bps", steps[j + 1]["state"]) if j + 1 < len(steps) else None
        if m:
            h += f" · by the next decision {m.group(1)} shares ({m.group(2)}%) are filled, average {m.group(3)} ({m.group(4)} bps against arrival)"
        stp["head"] = h
    mand = re.search(r"buy ([\d,]+) shares by the close", st[0]["state"])
    fl = [v for v in fill if v is not None]; av = [v for v in avg if v is not None]
    write("eco_mars", {"spark": fill,
        "name": "MarS", "title": "Executing a large buy order in the MarS order-flow market",
        "lede": "The model has five minutes to buy a fixed number of shares and may re-quote its order (price and size) every 15 seconds. Every other order comes from MarS's generative market model. The closer the average fill is to the arrival mid price, the better; costs are in basis points.",
        "tiles": [{"k": "Score", "v": f"{rw['0']:.3f}", "s": "Higher when execution is cheaper"},
                  {"k": "Mandate", "v": f"{mand.group(1) if mand else '—'} shares", "s": "Must be bought by the close"},
                  {"k": "Filled", "v": f"{fl[-1]:.1f}%", "s": "At the last decision"},
                  {"k": "Average fill", "v": (("+" if av and av[-1] > 0 else "") + f"{av[-1]:.1f} bps") if av else "—", "s": "At the last decision, against the arrival mid; positive = paid more"}],
        "foot": "This game took only four decisions: the model quoted the full size at once and most of it filled early. Prices are in the simulator's own tick units. The 11-09 recording cost 249 bps, against 41–48 bps for an even time-sliced schedule.",
        "x": {"n": xn, "labels": [f"R{i}" for i in range(xn)]},
        "charts": [
            {"title": "Share of the mandate filled", "sub": "Filled shares as a share of the mandate at each decision.", "height": 180, "fmt": "pct0",
             "bars": [{"name": "Filled", "color": "c1", "v": fill}], "lines": []},
            {"title": "Prices against arrival (bps)", "sub": "Mid price and the model's average fill, both relative to the mid at arrival; positive means more expensive.", "height": 200, "fmt": "bps", "zero": True,
             "lines": [{"name": "Mid", "color": "ink", "v": mid, "dots": True}, {"name": "Average fill", "color": "c1", "v": avg, "dots": True}]}],
        "book": {"title": "Order book at this step", "sub": "Asks on top, bids below; the dashed frame marks the model's own bid price."},
        "kinds": {"buy": ["Buy order", "c1"]}, "steps": steps})


# ---------------------------------------------------------------- TwinMarket
def build_eco_twinmarket_lite():
    st, rw = steps_of("eco_twinmarket_lite")
    D = int(re.search(r"trading day \d+ of (\d+)", st[0]["state"]).group(1))
    xn = D; idx = [None] * xn; ret = {k: [None] * xn for k in range(4)}; steps = []
    for s in st:
        S = s["state"]; dd = int(re.search(r"Today is trading day (\d+) of", S).group(1))
        m = re.search(r"Market index \([^)]*\): ([\d.]+)", S); idx[dd - 1] = float(m.group(1)) if m else idx[dd - 1]
        m = re.search(r"cumulative return ([+-]?[\d.]+)%", S); ret[s["seat"]][dd - 1] = float(m.group(1)) if m else None
        k = aid(s["act"])["action_id"]
        steps.append({"seat": s["seat"], "x": dd - 1, "head": f"Day {dd} of {D}", "kind": k, "act": s["act"], "state": S, "thinking": s["thinking"]})
    for stp in steps:
        k, x = stp["seat"], stp["x"]; orders = aid(stp["act"]).get("payload", {}).get("orders") or []
        did = ", ".join(f"{o.get('side', '?')}s {o.get('index', '?')} ({o.get('percent', 0):g}% of its assets, limit {o.get('price', 0):g})" for o in orders) or "holds"
        now, nxt = ret[k][x], ret[k][x + 1] if x + 1 < xn else None
        stp["head"] = f"Day {x + 1} of {D}" + (f" · index {idx[x]:.4f}" if idx[x] is not None else "") + f" · {did}" + (f" · its return {now:+.2f}%" + (f" → {nxt:+.2f}% the next day" if nxt is not None else "") if now is not None else "")
    write("eco_twinmarket_lite", {"seatNames": [f"Investor {k + 1}" for k in range(4)], "spark": idx,
        "name": "TwinMarket", "title": f"TwinMarket: 4 model investors among 96 simulated ones, {D} trading days",
        "lede": "A hundred investors trade ten sector indices. Four are played by the model, each with its own style (value, trend-following and so on). After each close they see their account, every index's price, moving averages and the system's recommendations, and decide what to buy or sell.",
        "tiles": [{"k": f"Investor {k + 1} return", "v": (sgn([v for v in ret[k] if v is not None][-1], 2) if any(v is not None for v in ret[k]) else "—"),
                   "s": f"Before the last day's trades; score {rw.get(str(k), 0):.4f}"} for k in range(4)],
        "foot": "Each point is what the investor saw before that day's open, that is the state after the previous close.",
        "x": {"n": xn, "labels": [f"Day {i + 1}" for i in range(xn)]},
        "charts": [
            {"title": "Market index", "sub": "Cap-weighted composite of the ten sector indices, 1.0 at the start.", "height": 220, "fmt": "price4",
             "lines": [{"name": "Composite", "color": "ink", "v": idx, "area": True}]},
            {"title": "Returns of the four model investors", "sub": "Each account against its starting value, in percent.", "height": 240, "fmt": "pct", "zero": True,
             "lines": [{"name": f"Investor {k + 1}", "color": f"c{k + 1}", "v": ret[k], "end": True} for k in range(4)]}],
        "kinds": {"trade": ["Trade", "c1"], "hold": ["Hold", "muted"]}, "steps": steps})


# ---------------------------------------------------------------- MarketSim
def build_eco_marketsim_talk():
    st, rw = steps_of("eco_marketsim_talk")
    HDR = r"Day (\d+) of (\d+), decision (\d+) of the day \(([^)]*)\), (?:(\w+) stage|decision)"
    keys, info = [], {}
    for s in st:
        m = re.search(HDR, s["state"]); key = (int(m.group(1)), int(m.group(3)))
        if key not in keys: keys.append(key)
        info.setdefault(key, s["state"])
    xn = len(keys); candles = [None] * xn; lastp = [None] * xn; flags = []; sur = {k: [None] * xn for k in range(4)}; prev_t = {}
    for x, key in enumerate(keys):
        S = info[key]; m = re.search(HDR, S); t_now = re.match(r"(\d\d:\d\d:\d\d)", m.group(4))
        p = re.search(r"last trade (\d+) cents", S) or re.search(r"[Ll]ast traded price (\d+) cents", S)
        lastp[x] = int(p.group(1)) / 100 if p else None
        sm = re.search(r"Today's trades at \d+ evenly spaced points \([^)]*\): (.*?); change from", S)
        samples = [(t, int(pr) / 100) for t, pr, _ in re.findall(r"(\d\d:\d\d:\d\d) (\d+) x (\d+)", sm.group(1))] if sm else []
        since = prev_t.get(key[0]); win = [pr for t, pr in samples if since is None or t > since]
        if win and lastp[x] is not None: candles[x] = [win[0], max(win + [lastp[x]]), min(win + [lastp[x]]), lastp[x]]
        if t_now: prev_t[key[0]] = t_now.group(1)
        nw = re.search(r"News released since the last decision:\s*\n- (?:\[[^\]]*\] )?(.*)", S)
        if nw: flags.append({"x": x, "text": nw.group(1)[:220]})
    steps = []
    for s in st:
        S = s["state"]; m = re.search(HDR, S); x = keys.index((int(m.group(1)), int(m.group(3))))
        q = re.search(r"surplus rate so far ([+-]?[\d.]+)%", S)
        if q: sur[s["seat"]][x] = float(q.group(1))
        bk = book_of(S, r"Asks \(price x shares\): (.*)", r"Bids \(price x shares\): (.*)", ", ", 0.01)
        g_ = aid(s["act"]).get("payload", {})
        stage = "public comment" if m.group(5) == "communication" else (
            "guidance: target price " + (f"${g_['price'] / 100:,.2f}" if isinstance(g_.get("price"), (int, float)) else "none")
            + ("; strategy weights " + ", ".join(f"{a} {b:g}" for a, b in (g_.get("weights") or {}).items()) if g_.get("weights") else ""))
        steps.append({"seat": s["seat"], "x": x, "head": f"Day {m.group(1)}, decision {m.group(3)} ({m.group(4)})" + (f" · last trade ${lastp[x]:,.2f}" if lastp[x] is not None else "") + f" · {stage}",
                      "kind": aid(s["act"])["action_id"], "act": s["act"], "state": S, "thinking": s["thinking"], "book": dict(bk, last=lastp[x], unit="$") if bk else None})
    first = next(v for v in lastp if v is not None); lastv = [v for v in lastp if v is not None][-1]
    write("eco_marketsim_talk", {"seatNames": [f"Manager {k + 1}" for k in range(4)], "spark": lastp,
        "name": "MarketSim", "title": "MarketSim: 4 model fund managers in a market of 12,000 traders",
        "lede": "Each fund manager directs 295 traders; the market also holds 12,000 value, momentum and noise traders and market makers. Every decision has two stages: a public comment, then guidance to the manager's own traders (a target price and strategy weights). News shocks such as tariffs, an AI launch and earnings arrive during the game.",
        "tiles": [{"k": f"Manager {k + 1}", "v": (sgn([v for v in sur[k] if v is not None][-1], 3) if any(v is not None for v in sur[k]) else "—"),
                   "s": f"Surplus rate before the last decision; score {rw.get(str(k), 0):.4f}"} for k in range(4)],
        "foot": f"The axis counts decisions as day-decision ({xn} in total). Each candle covers the trades between two decisions (open, high, low, close from the 20 sampled trades shown to the model and the last trade); dashed yellow lines mark news releases. The price went from ${first:,.2f} to ${lastv:,.2f}.",
        "x": {"n": xn, "labels": [f"{a}-{b}" for a, b in keys]},
        "charts": [
            {"title": "Price, one candle per decision interval", "sub": "Pre-open decisions have no trades, so their slot is empty. Hover a yellow line to read the news.", "height": 260, "fmt": "price2",
             "candles": candles, "flags": flags, "lines": []},
            {"title": "Surplus rate of the four managers", "sub": "Each account against its starting value, in percent.", "height": 230, "fmt": "pct3", "zero": True,
             "lines": [{"name": f"Manager {k + 1}", "color": f"c{k + 1}", "v": sur[k], "end": True} for k in range(4)]}],
        "book": {"title": "Order book at this step", "sub": "Five levels on each side, in dollars; longer bars mean more shares."},
        "kinds": {"post": ["Public comment", "c4"], "guide": ["Guidance", "c1"]}, "steps": steps})


# ---------------------------------------------------------------- QFinZero
def build_eco_qfinzero():
    st, rw = steps_of("eco_qfinzero")
    B = int(re.search(r"bar \d+ of (\d+)", st[0]["state"]).group(1))
    xn = B; dates = [None] * xn; ohlc = {}; eq = [None] * xn; marks, steps = [], []
    for s in st:
        S = s["state"]; m = re.search(r"bar (\d+) of \d+, trading date (\d{4}-\d\d-\d\d)", S); b = int(m.group(1)) - 1; dates[b] = m.group(2)
        cur = re.search(r"Current bars \(as traded\): (.*)", S)
        if cur:
            for sym, o, h, l, c in re.findall(r"(\w+) O ([\d.]+) H ([\d.]+) L ([\d.]+) C ([\d.]+)", cur.group(1)):
                ohlc.setdefault(sym, [None] * xn)[b] = [float(o), float(h), float(l), float(c)]
        e = re.search(r"equity ([\d.]+)", S); eq[b] = float(e.group(1)) if e else eq[b]
        a = aid(s["act"]); k = a["action_id"]; p = a.get("payload", {})
        kind = "order" if k == "submit_order" else "cancel" if k == "cancel_order" else "advance" if k == "advance" else "tool"
        if k == "submit_order":
            side = str(p.get("side", "")).lower(); marks.append({"x": b, "sym": p.get("symbol"), "kind": "buy" if side.startswith("b") else "sell"})
        if k == "submit_order":
            did = (f"sends a {str(p.get('order_type', '')).lower()} order to {str(p.get('side', '')).lower()} {p.get('qty', 0):,} {p.get('symbol', '')}"
                   + (f" at ${p['limit_price']:,.2f}" if p.get("limit_price") else "")
                   + (f" (last close ${ohlc[p['symbol']][b][3]:,.2f})" if (ohlc.get(p.get("symbol")) or [None] * xn)[b] else ""))
        elif k == "cancel_order":
            did = f"cancels order {p.get('order_id', '')}"
        elif k == "advance":
            did = f"moves the clock {p.get('bars', 1)} bar" + ("s" if p.get("bars", 1) != 1 else "")
        elif k.startswith("query_"):
            did = f"looks up {k[6:].replace('_', ' ')}" + (f" of {p['symbol']}" if p.get("symbol") else "") + (f", {p['start']} to {p['end']}" if p.get("start") else "")
        else:
            did = k.replace("_", " ")
        steps.append({"seat": 0, "x": b, "head": f"{m.group(2)} · bar {b + 1} · {did}", "kind": kind, "act": s["act"], "state": S, "thinking": s["thinking"]})
    for j, stp in enumerate(steps[:-1]):  # after the clock moves, what the account holds
        if aid(stp["act"])["action_id"] == "advance":
            nxt = steps[j + 1]["state"]
            e_ = re.search(r"equity ([\d.]+)", nxt); pos = re.search(r"Positions: (.*)", nxt)
            stp["head"] += (f" · equity then {float(e_.group(1)):,.2f}" if e_ else "") + (f"; positions: {pos.group(1).rstrip('.')[:90]}" if pos else "")
    apb = int(re.search(r"(\d+) actions left at this bar", st[0]["state"]).group(1))
    charts = []
    for sym in [s for s in ("ACME", "BOLT", "CRUX") if s in ohlc]:
        o = ohlc[sym]
        charts.append({"title": f"{sym} daily candles", "sub": "One candle per trading day; triangles are the model's buy and sell orders that day.", "height": 220, "fmt": "price2",
                       "candles": o, "lines": [], "markers": [{"x": mk["x"], "y": (o[mk["x"]][2] if mk["kind"] == "buy" else o[mk["x"]][1]) if o[mk["x"]] else None, "kind": mk["kind"]}
                                                                for mk in marks if mk["sym"] == sym]})
    charts.append({"title": "Account equity", "sub": "Cash plus the value of open positions; 100,000 at the start.", "height": 200, "fmt": "money0", "lines": [{"name": "Equity", "color": "c1", "v": eq, "area": True}]})
    cnt = {}
    for s in steps: cnt[s["kind"]] = cnt.get(s["kind"], 0) + 1
    lastq = [v for v in eq if v is not None][-1]
    write("eco_qfinzero", {"spark": eq,
        "name": "QFinZero", "title": f"QFinZero: a tool-using trader over {B} trading days",
        "lede": f"The model trades through tools: it queries daily and minute bars, news, earnings and macro data, then submits or cancels orders. Data is locked to the current date, so it cannot see the future. Each day allows up to {apb} actions; then, or when the model advances, the clock moves to the next day. ACME, BOLT and CRUX are tradable.",
        "tiles": [{"k": "Score", "v": f"{rw['0']:.4f}", "s": "From final equity against the start"},
                  {"k": "Equity at the last decision", "v": f"{lastq:,.0f}", "s": f"From 100,000, {sgn((lastq / 100000 - 1) * 100, 2)}"},
                  {"k": "Orders / cancels", "v": f"{cnt.get('order', 0)} / {cnt.get('cancel', 0)}", "s": "Orders submitted and cancelled"},
                  {"k": "Data queries", "v": f"{cnt.get('tool', 0)}", "s": f"Advanced the clock {cnt.get('advance', 0)} times"}],
        "foot": "Tickers are the dataset's pseudonyms; candles come from the current bars shown to the model each day.",
        "x": {"n": xn, "labels": [d or "" for d in dates]},
        "charts": charts,
        "kinds": {"order": ["Order", "c1"], "cancel": ["Cancel", "c2"], "advance": ["Next day", "muted"], "tool": ["Data query", "c3"]}, "steps": steps})


BUILDERS = {"eco_investorbench": build_eco_investorbench, "eco_liquidation_2p": build_eco_liquidation_2p, "eco_abides": build_eco_abides, "eco_mars": build_eco_mars, "eco_twinmarket_lite": build_eco_twinmarket_lite, "eco_marketsim_talk": build_eco_marketsim_talk, "eco_qfinzero": build_eco_qfinzero}
for game, build in BUILDERS.items():
    if (not ONLY or game in ONLY) and os.path.isfile(os.path.join(run_dir(game), "run.json")):
        build()
