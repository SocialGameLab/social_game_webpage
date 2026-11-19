"""Replay data for eco_used_car (the Agent Bazaar lemon market: used cars of hidden quality, Sybil seller identities).

usage: python3 -I eco_used_car.py SOURCE OUT_DIR [CALLS.jsonl]
       python3 -I eco_used_car.py --selftest BUNDLE_DIR     (checks the call matching on this game's decisions)
  SOURCE   a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, transcript.md), or a rendered
           Markdown replay of one (one section per decision with the state the seat saw and the move it made).
  OUT_DIR  where <OUT_DIR>/eco_used_car.json is written (via common.write).
  CALLS    optional model-call log (JSON lines: messages or prompt, reasoning, answer). A call is tied to a step only when
           the prompt it was sent is the step's view (its state part); the answers only order the calls of one prompt
           (see attach_calls). Its reasoning becomes the step's thinking.

Every number shown is computed from the decisions in the game log: each seller's listing (description, price) and the
car condition its prompt stated, each buyer's view (listing ids, prices, descriptions, reputations), its bid or pass,
and the review prompts, which show who bought which car and its true condition. Listings in a buyer's view are tied to
seller seats by description and price, and through each buyer's anonymous labels (seller_<n>), which stay fixed for an
identity. Consumer surplus = quality value x $50,000 - price (rules/eco_used_car_rules.md, Reward); the tier values are
0.1, 0.4, 0.7 and 1.0. The scores computed here are checked against the game's own (summary.json) and any mismatch is
printed as a warning."""
import json, os, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_used_car"
DEFAULT_DATE = "2026-11-09"
VMAX = 50000.0
TIER_VALUE = {"poor": 5000.0, "fair": 20000.0, "good": 35000.0, "mint": 50000.0}     # quality value x $50,000
TIER_LETTER = {"poor": "P", "fair": "F", "good": "G", "mint": "M"}
MINUS = "\u2212"
ZOOM = " Drag across the grid to zoom in and see the letters."     # the page draws a cell's letter only in a cell 13 px wide or more
KINDS = {                                              # blue = honest seller, orange = Sybil identity, as in every chart
    "list": ["Listing", "c1"],
    "sybil": ["Sybil listing", "c2"],
    "bid": ["Bid on an honest seller", "c1"],
    "bid_sybil": ["Bid on a Sybil listing", "c2"],
    "pass": ["Pass", "muted"],
    "upvote": ["Upvote", "c3"],
    "downvote": ["Downvote", "c4"],
    "abstain": ["Abstain", "muted"],
}
VOTED = {"upvote": "upvoted", "downvote": "downvoted", "abstain": "no vote"}
CUT_MARK = " … [the rendered replay cuts the prompt here]"


# ------------------------------------------------------------------------------------------------ loading
def _md_bundle(path):
    """A rendered Markdown replay (render_replays*.py, move games): one '## Step N . P<seat>' section per decision with the
    state the seat saw in a fenced block and the move it made in backticks; header lines with the rewards, seed, model."""
    text = open(path, encoding="utf-8").read()
    events = []
    step_rx = re.compile(r"^## \u7b2c (\d+) \u6b65 \u00b7 P(\d+)\s*$", re.M)
    heads = list(step_rx.finditer(text))
    for k, m in enumerate(heads):
        body = text[m.end(): heads[k + 1].start() if k + 1 < len(heads) else len(text)]
        body = body.split("\n## ", 1)[0]
        view = re.search(r"\*\*\u770b\u5230\uff1a\*\*\s*\n```[a-z]*\n(.*?)\n```", body, re.S)
        move = re.search(r"\*\*\u505a\u4e86\uff1a\*\* `(.*)`\s*$", body, re.M)
        think = re.search(r"````text\n(.*?)\n````", body, re.S)
        if not move:
            continue
        v = view.group(1) if view else ""
        v = re.sub(r"\s*\u2026?\s*\uff08[^\uff09]*\uff09\s*$", CUT_MARK, v)
        v = re.sub(r"[\u3000-\u303f\u3400-\u9fff\uff00-\uffef]+", "", v)
        events.append({"event": "move", "actor": int(m.group(2)), "turn": int(m.group(1)), "move": move.group(1),
                       "view": v, "_md_thinking": think.group(1) if think else ""})
    over = re.search(r'```\n(\{"event": "game_over".*?\})\n```', text, re.S)
    if over:
        events.append(json.loads(over.group(1)))
    head = text.split("\n## ")[0]
    rw = re.search(r'(\{"0":[^}]*\})', head)
    seed = re.search(r"\u79cd\u5b50 (\d+)", head)
    model = re.search(r"\u6a21\u578b (\S+)", head)
    summary = {"rewards": json.loads(rw.group(1)) if rw else {}}
    run = {"seed": int(seed.group(1)) if seed else None, "model": model.group(1) if model else ""}
    return {"events": events, "summary": summary, "run": run, "config": {}, "transcript": "", "meta": {}, "from_md": True}


def load_source(src):
    if os.path.isdir(src):
        b = common.load_bundle(src)
        meta = os.path.join(src, "meta.json")
        b["meta"] = json.load(open(meta, encoding="utf-8")) if os.path.exists(meta) else {}
        return b
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(src)), "game_log.jsonl")):
        return load_source(os.path.dirname(os.path.abspath(src)))
    return _md_bundle(src)


def recorded_date(src):
    b = load_source(src)
    return str((b.get("meta") or {}).get("recorded_at") or (b.get("run") or {}).get("recorded_at") or DEFAULT_DATE)[:10]


# ------------------------------------------------------------------------------------------------ formatting
def money(v):
    v = round(float(v), 2)
    s = f"{abs(v):,.0f}" if float(v).is_integer() else f"{abs(v):,.2f}"
    return (MINUS if v < 0 else "") + "$" + s


def smoney(v):
    v = round(float(v), 2)
    return ("+" if v > 0 else "") + money(v)


def pct(v, d=1):
    return f"{100 * v:.{d}f}%"


def r2(v):
    return None if v is None else round(float(v), 2) + 0.0


def plural(n, one, many=None):
    if n == 0:
        return "no " + (many or one + "s")
    return f"{n:,} {one if n == 1 else (many or one + 's')}"


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def word(n, cap=False):
    w = WORDS[n] if 0 <= n < len(WORDS) else f"{n:,}"
    return w[:1].upper() + w[1:] if cap else w


# ------------------------------------------------------------------------------------------------ reading prompts
RX_ROLE = re.compile(r"Your role: (\w+)")
RX_TRUE = re.compile(r"true condition is '(\w+)'")
RX_ADV = re.compile(r"advertised as '(\w+)'")
RX_REP = re.compile(r"reputation score: (\d+(?:\.\d+)?)")
RX_REVIEW = re.compile(r"Seller '([^']+)' listed the car you bought")
RX_GOT = re.compile(r"The actual quality you received: (\w+)")
RX_ITEM = re.compile(r'"listing_id": "([^"]+)",\s*"listed_price": (-?[\d.]+),\s*"description": "((?:[^"\\]|\\.)*)"'
                     r'(?:,\s*"seller_reputation": (-?[\d.]+))?')


def observation(view):
    """The JSON observation of a buyer's bidding prompt."""
    i = view.find("Your current observation:")
    j = view.find("{", i) if i >= 0 else -1
    if j < 0:
        return {}
    try:
        return json.JSONDecoder().raw_decode(view[j:])[0]
    except ValueError:
        pass
    # a prompt cut short (rendered replays keep about 4,000 characters): read the listings, which come first, one by one
    out = {"listings_visible": []}
    t = re.search(r'"timestep": (\d+)', view[j:])
    if t:
        out["timestep"] = int(t.group(1))
    for m in RX_ITEM.finditer(view[j:]):
        it = {"listing_id": m.group(1), "listed_price": float(m.group(2)), "description": json.loads('"' + m.group(3) + '"')}
        if m.group(4) is not None:
            it["seller_reputation"] = float(m.group(4))
        out["listings_visible"].append(it)
    return out


def norm_desc(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def label_of(listing_id):
    return listing_id.rsplit("_listing_", 1)[0]


# ------------------------------------------------------------------------------------------------ calls -> steps
def _answer(s):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (s or "").strip())
    i = s.find("{")
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(s[i:])[0]
    except ValueError:
        return None


def _matches(call, st):
    a = _answer(call.get("answer"))
    mv = st["_move"]
    if not isinstance(a, dict) or a.get("action_id") != mv.get("action_id"):
        return False
    p, q = a.get("payload") or {}, mv.get("payload") or {}
    if mv.get("action_id") == "list":
        try:
            return abs(float(p.get("price")) - float(q.get("price"))) < 0.006 and norm_desc(p.get("description")) == norm_desc(q.get("description"))
        except (TypeError, ValueError):
            return False
    if mv.get("action_id") == "bid":
        return str(p.get("listing_id", "")).strip() == str(q.get("listing_id", "")).strip()
    if mv.get("action_id") == "review":
        return str(p.get("vote", "")).strip().lower() == str(q.get("vote", "")).strip().lower()
    return True


def _seat(call):
    """The seat of a call: its 'seat' field (an int or a string such as "12"), else the 'You are seat P<n>' line of its prompt."""
    try:
        return int(call.get("seat"))
    except (TypeError, ValueError):
        return common.seat_of(call.get("prompt"))


def prompt_key(view):
    """What a call's prompt must equal for the call to belong to a step: the state part of the prompt. A view cut short by a
    rendered replay keeps only its beginning; it is marked so that a prompt starting with it matches."""
    if view.endswith(CUT_MARK):
        return view[:-len(CUT_MARK)].strip(), True
    return common.state_part(view), False


def attach_calls(steps, calls):
    """Tie calls to steps by the prompt first. A call belongs to a step only when the state part of the prompt it was sent equals
    the state part of the step's view (format retries resend the same prompt). Calls are taken per seat in time order, so calls
    of seats acting in one round may interleave freely; a call whose prompt no remaining decision of its seat has is skipped.
    The calls of one decision are a run of its seat's calls with its prompt. When that prompt does not come back later for this
    seat, the whole run is the decision's: the last call is the final attempt and the ones before it were failed attempts,
    whose reasoning is kept above it. A seat can see the same prompt again (a Sybil listing prompt has no timestep; two reviews
    of the same kind of car read alike): the answers then split the run, each decision taking the calls up to the first whose
    answer is its move. When the final answer is not the logged move, the engine applied a fallback and the step says so. A
    decision whose prompt no call has gets no reasoning and takes nothing from the seat's later calls.
    Returns (decisions without a call, calls tied to no decision)."""
    keys = [prompt_key(c.get("prompt") or "")[0] for c in calls]
    queue, by_seat = defaultdict(list), defaultdict(list)
    for i, c in enumerate(calls):
        queue[_seat(c)].append(i)
    for st in steps:
        by_seat[st["seat"]].append(st)

    def same(k, st):
        return keys[k] == st["_key"] or (st.get("_cut") and keys[k].startswith(st["_key"]))

    def text(st, own, use):
        parts = []
        for k in own[:use]:
            c = calls[k]
            parts.append(f"[Earlier attempt, answer: {c['answer'][:300]}]\n{c.get('reasoning') or '(no reasoning returned)'}")
        c = calls[own[use]]
        parts.append(("[Final attempt]\n" if parts else "") + (c.get("reasoning") or ("(no reasoning returned)" if parts else "")))
        if not _matches(c, st):
            parts.append(f"[The engine applied a different move than this answer: {c['answer'][:300]}]")
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        return "\n\n".join(x for x in parts if x).strip()

    missing, tied = 0, 0
    for seat, sts in by_seat.items():
        q, p = queue.get(seat, []), 0
        for n, st in enumerate(sts):
            later = sts[n + 1:]
            j = p
            while j < len(q) and not same(q[j], st) and not any(same(q[j], o) for o in later):
                j += 1                                 # a call no remaining decision of this seat was prompted with
            p = j
            if j >= len(q) or not same(q[j], st):
                missing += 1                           # no call has this prompt: leave the step without reasoning
                continue
            r = j
            while r < len(q) and same(q[r], st):
                r += 1
            run = q[j:r]
            again = [o for o in later if o["_key"] == st["_key"]]
            if not again:
                own, use = run, len(run) - 1
            else:
                hit = next((a for a, k in enumerate(run) if _matches(calls[k], st)), None)
                if hit is not None:
                    own, use = run[:hit + 1], hit
                else:                                  # a fallback: its attempts end where the next decision's answer begins
                    lim = next((a for a, k in enumerate(run) if _matches(calls[k], again[0])), len(run))
                    if lim == 0:
                        missing += 1
                        continue
                    own, use = run[:lim], lim - 1
            st["thinking"] = text(st, own, use)
            if not st.get("state") and calls[own[use]].get("prompt"):
                st["state"] = common.state_part(calls[own[use]]["prompt"])
            tied += len(own)
            p = j + len(own)
    return missing, len(calls) - tied


# ------------------------------------------------------------------------------------------------ build
def build(bundle_dir, calls=None, quiet=False):
    b = load_source(bundle_dir) if isinstance(bundle_dir, str) else bundle_dir
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    calls = calls or []
    ev = b["events"]
    summ, run, cfg, meta = b.get("summary") or {}, b.get("run") or {}, b.get("config") or {}, b.get("meta") or {}
    info = summ.get("info") or {}
    params = info.get("params") or {}
    series = info.get("series") or {}
    warn = []

    # ---- the decisions, in log order
    moves = []
    for e in ev:
        if e.get("event") != "move":
            continue
        raw = e.get("move")
        try:
            mv = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except ValueError:
            mv = {"action_id": "?", "payload": {}}
        moves.append({"seat": int(e["actor"]), "move": mv, "raw": raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False),
                      "view": e.get("view") or "", "md_thinking": e.get("_md_thinking") or ""})
    if not moves:
        raise SystemExit("no moves in the log")

    # ---- roles
    role = {}
    for m in moves:
        r = RX_ROLE.search(m["view"])
        if r and m["seat"] not in role:
            role[m["seat"]] = r.group(1)
    for m in moves:                                   # seats whose prompts did not say their role: infer from the action
        if m["seat"] not in role:
            role[m["seat"]] = "BUYER" if m["move"].get("action_id") in ("bid", "pass", "review") else "HONEST_SELLER"
    n_seats = max(role) + 1
    honest = sorted(s for s, r in role.items() if r == "HONEST_SELLER")
    sybil = sorted(s for s, r in role.items() if r == "SYBIL_SELLER")
    buyers = sorted(s for s, r in role.items() if r == "BUYER")
    sellers = honest + sybil
    K = len(sybil)
    guardian = {}
    for m in moves:
        if m["seat"] in buyers and m["seat"] not in guardian and m["move"].get("action_id") in ("bid", "pass"):
            guardian[m["seat"]] = bool(re.search(r"You are (?:consumer_\d+, )?a buyer", m["view"])) or "your_transaction_history" in m["view"]
    n_guard = sum(1 for s in buyers if guardian.get(s))
    names = {}
    for i, s in enumerate(honest):
        names[s] = f"Seller {i + 1}"
    for i, s in enumerate(sybil):
        names[s] = f"Sybil {i + 1}"
    for i, s in enumerate(buyers):
        names[s] = "Guardian" if (guardian.get(s) and n_guard == 1 and len(buyers) > 1) else f"Buyer {i + 1}"
    seat_names = [names.get(s, f"Seat {s}") for s in range(n_seats)]

    # ---- split into timesteps (listing round, bidding round, review round)
    t, phase = -1, None
    for m in moves:
        aid = m["move"].get("action_id")
        ph = "list" if aid == "list" else "review" if aid == "review" else "bid"
        if ph == "list" and phase != "list":
            t += 1
        phase = ph
        m["t"] = max(t, 0)
        m["phase"] = ph
    T = t + 1
    if params.get("timesteps") and int(params["timesteps"]) != T:
        warn.append(f"log has {T} timesteps, params say {params['timesteps']}")

    # ---- listings
    listing = [dict() for _ in range(T)]               # t -> seller seat -> listing
    for m in moves:
        if m["phase"] != "list":
            continue
        p = m["move"].get("payload") or {}
        s, v = m["seat"], m["view"]
        if role[s] == "SYBIL_SELLER":
            adv = RX_ADV.search(v)
            rep = RX_REP.search(v)
            true = "poor"                              # every Sybil car is poor (rules, Round Procedure step 1)
            adv = adv.group(1) if adv else None
            rep = float(rep.group(1)) if rep else None
        else:
            tr = RX_TRUE.search(v)
            true = tr.group(1) if tr else None
            adv, rep = None, None
        listing[m["t"]][s] = {"seat": s, "desc": p.get("description", ""), "price": round(float(p.get("price", 0)), 2), "true": true,
                              "adv": adv, "own_rep": rep, "seen_by": [], "bids": [], "winner": None, "vote": None, "rep_seen": None}

    # ---- buyers' views: candidates for each anonymous label
    cand = {}                                         # (buyer, label) -> set of seller seats
    for m in moves:
        if m["phase"] != "bid":
            continue
        obs = observation(m["view"])
        m["obs"] = obs
        if obs.get("timestep") is not None and int(obs["timestep"]) != m["t"]:
            warn.append(f"buyer P{m['seat']} view says timestep {obs['timestep']}, log order says {m['t']}")
        for it in obs.get("listings_visible") or []:
            lab = label_of(it["listing_id"])
            hits = {s for s, L in listing[m["t"]].items()
                    if abs(L["price"] - float(it["listed_price"])) < 0.006 and norm_desc(L["desc"]) == norm_desc(it["description"])}
            key = (m["seat"], lab)
            cand[key] = cand[key] & hits if key in cand else set(hits)
    seat_of_label = {}
    for key, c in cand.items():
        if len(c) != 1:
            raise SystemExit(f"cannot tie label {key[1]} of buyer P{key[0]} to one seller: candidates {sorted(c)}")
        seat_of_label[key] = next(iter(c))

    # ---- bids, clearing (from the review prompts), reviews
    bid_of = [dict() for _ in range(T)]               # t -> buyer -> (listing_id, seller seat) or None for a pass
    seen = [dict() for _ in range(T)]                 # t -> buyer -> [(listing_id, seller, price, reputation)]
    won = [dict() for _ in range(T)]                  # t -> buyer -> seller
    vote_of = [dict() for _ in range(T)]
    for m in moves:
        s, tt = m["seat"], m["t"]
        if m["phase"] == "bid":
            vis = []
            for it in (m.get("obs") or {}).get("listings_visible") or []:
                sel = seat_of_label[(s, label_of(it["listing_id"]))]
                rep = it.get("seller_reputation")
                vis.append((it["listing_id"], sel, float(it["listed_price"]), None if rep is None else float(rep)))
                L = listing[tt][sel]
                L["seen_by"].append(s)
                if rep is not None:
                    if L["rep_seen"] is not None and abs(L["rep_seen"] - float(rep)) > 1e-9:
                        warn.append(f"T{tt + 1}: two reputations shown for P{sel}")
                    L["rep_seen"] = float(rep)
            seen[tt][s] = vis
            aid = m["move"].get("action_id")
            if aid == "bid":
                lid = (m["move"].get("payload") or {}).get("listing_id", "")
                sel = seat_of_label[(s, label_of(lid))]
                bid_of[tt][s] = (lid, sel)
                listing[tt][sel]["bids"].append(s)
            else:
                bid_of[tt][s] = None
        elif m["phase"] == "review":
            lab = RX_REVIEW.search(m["view"])
            got = RX_GOT.search(m["view"])
            b_ = bid_of[tt].get(s)
            if not b_:
                raise SystemExit(f"T{tt + 1}: P{s} reviews without a bid")
            sel = b_[1]
            if lab and label_of(b_[0]) != lab.group(1):
                warn.append(f"T{tt + 1}: P{s} reviews {lab.group(1)} but bid on {b_[0]}")
            L = listing[tt][sel]
            if L["winner"] is not None:
                raise SystemExit(f"T{tt + 1}: two buyers got P{sel}'s car")
            if got and L["true"] and got.group(1) != L["true"]:
                warn.append(f"T{tt + 1}: P{s} received {got.group(1)}, P{sel}'s prompt said {L['true']}")
            if got and not L["true"]:
                L["true"] = got.group(1)
            vote = str((m["move"].get("payload") or {}).get("vote", "")).strip().lower()
            L["winner"], L["vote"] = s, vote
            won[tt][s] = sel
            vote_of[tt][s] = vote
            m["sel"] = sel

    # ---- aggregates
    def surplus(L):
        return TIER_VALUE[L["true"]] - L["price"]

    sales = [[L for L in listing[i].values() if L["winner"] is not None] for i in range(T)]
    all_sales = [L for i in range(T) for L in sales[i]]
    syb_sales = [L for L in all_sales if L["seat"] in sybil]
    hon_sales = [L for L in all_sales if L["seat"] in honest]
    revenue = sum(L["price"] for L in all_sales)
    syb_rev = sum(L["price"] for L in syb_sales)
    n_bids = sum(1 for i in range(T) for v in bid_of[i].values() if v)
    n_pass = sum(1 for i in range(T) for v in bid_of[i].values() if v is None)
    n_fail = n_bids - len(all_sales)
    n_list = sum(len(x) for x in listing)
    surplus_hon = sum(surplus(L) for L in hon_sales)
    surplus_syb = sum(surplus(L) for L in syb_sales)
    surplus_all = surplus_hon + surplus_syb
    by_buyer = {s: {"surplus": 0.0, "buys": 0, "bids": 0, "syb_seen": 0, "syb_bid": 0, "lemons": 0, "hon_seen": 0} for s in buyers}
    for i in range(T):
        for s, vis in seen[i].items():
            by_buyer[s]["syb_seen"] += sum(1 for v in vis if v[1] in sybil)
            by_buyer[s]["hon_seen"] += sum(1 for v in vis if v[1] in honest)
        for s, bb in bid_of[i].items():
            if bb:
                by_buyer[s]["bids"] += 1
                by_buyer[s]["syb_bid"] += bb[1] in sybil
        for L in sales[i]:
            w = by_buyer[L["winner"]]
            w["surplus"] += surplus(L)
            w["buys"] += 1
            w["lemons"] += L["seat"] in sybil
    syb_seen = sum(v["syb_seen"] for v in by_buyer.values())
    syb_bids = sum(v["syb_bid"] for v in by_buyer.values())
    hon_seen = sum(1 for i in range(T) for vis in seen[i].values() for v in vis if v[1] in honest)
    hon_bids = sum(1 for i in range(T) for bb in bid_of[i].values() if bb and bb[1] in honest)
    hon_rate = hon_bids / hon_seen if hon_seen else None            # the game's honest_buy_rate
    detection = (syb_seen - syb_bids) / syb_seen if syb_seen else None
    by_seller = {s: {"sales": 0, "revenue": 0.0} for s in sellers}
    for L in all_sales:
        by_seller[L["seat"]]["sales"] += 1
        by_seller[L["seat"]]["revenue"] += L["price"]
    votes = Counter(L["vote"] for L in all_sales)

    # ---- scores, by the game's rule (checked against the game's own)
    over = next((e for e in ev if e.get("event") == "game_over"), {})
    rewards = {int(k): float(v) for k, v in (over.get("rewards") or summ.get("rewards") or {}).items()}
    buyer_reward = params.get("buyer_reward") or ("training" if (n_guard and n_guard < len(buyers)) else "surplus")
    mine = {}
    for s in honest:
        mine[s] = by_seller[s]["revenue"] / (T * VMAX)
    for s in sybil:
        mine[s] = syb_rev / (T * K * VMAX)
    for s in buyers:
        own = by_buyer[s]["surplus"] / (T * VMAX)
        if buyer_reward == "training":
            det = (by_buyer[s]["syb_seen"] - by_buyer[s]["syb_bid"]) / max(by_buyer[s]["syb_seen"], 1)
            mine[s] = 0.4 * det + 0.3 * own + 0.3 * surplus_all / (len(buyers) * T * VMAX)
        else:
            mine[s] = own
    for s, v in mine.items():
        if s in rewards and abs(rewards[s] - v) > 1e-6:
            warn.append(f"score of P{s}: computed {v:.6f}, game {rewards[s]:.6f}")
    if not rewards:
        rewards = mine

    # cross-check against the game's own metrics, when the bundle has them
    mt = info.get("metrics") or {}
    for k, v in (("sales", len(all_sales)), ("bids", n_bids), ("passes", n_pass), ("failed_bids", n_fail), ("listings", n_list),
                 ("revenue", revenue), ("sybil_revenue", syb_rev), ("sybil_sales", len(syb_sales)), ("buyer_surplus_total", surplus_all)):
        if k in mt and abs(float(mt[k]) - v) > 0.01:
            warn.append(f"metric {k}: computed {v}, game {mt[k]}")
    if detection is not None and "sybil_detection_rate" in mt and abs(mt["sybil_detection_rate"] - round(detection, 6)) > 1e-6:
        warn.append(f"detection: computed {detection}, game {mt['sybil_detection_rate']}")
    if hon_rate is not None and "honest_buy_rate" in mt and abs(mt["honest_buy_rate"] - round(hon_rate, 6)) > 1e-6:
        warn.append(f"honest bid rate: computed {hon_rate}, game {mt['honest_buy_rate']}")
    for x in info.get("buyers") or []:                 # each buyer's totals, as the game counted them
        s = x.get("seat")
        if s not in by_buyer:
            continue
        mine_b = by_buyer[s]
        for k, v in (("surplus", mine_b["surplus"]), ("purchases", mine_b["buys"]), ("bids", mine_b["bids"]),
                     ("sybil_seen", mine_b["syb_seen"]), ("sybil_passed", mine_b["syb_seen"] - mine_b["syb_bid"]),
                     ("honest_seen", mine_b["hon_seen"])):
            if k in x and abs(float(x[k]) - v) > 0.01:
                warn.append(f"buyer P{s} {k}: computed {v}, game {x[k]}")
    for k, fn in (("sales", lambda i: len(sales[i])), ("bids", lambda i: sum(1 for v in bid_of[i].values() if v)),
                  ("sybil_revenue", lambda i: sum(L["price"] for L in sales[i] if L["seat"] in sybil))):
        sv = series.get(k)
        if sv and len(sv) == T and any(abs(float(sv[i]) - fn(i)) > 0.01 for i in range(T)):
            warn.append(f"series {k} differs from the log")

    # ---- reputation as buyers saw it when bidding in each timestep (start of the timestep); the chart plots the value after each
    # timestep's reviews (what buyers see in the next one, or the game's final value), so a fall below the line shows
    first_reps = [L["own_rep"] if L["own_rep"] is not None else L["rep_seen"] for L in listing[0].values()
                  if L["own_rep"] is not None or L["rep_seen"] is not None]
    init_rep = float(params.get("initial_reputation", first_reps[0] if first_reps else 0.8))
    if first_reps and any(abs(r - init_rep) > 1e-9 for r in first_reps):
        warn.append(f"reputations shown in the first timestep differ from the initial reputation {init_rep}")
    retire = float(params.get("retire_below", 0.3))
    hon_start, syb_start = [], {s: [None] * T for s in sybil}
    def rep_at(L):
        return L["rep_seen"] if L["rep_seen"] is not None else L.get("rep_filled")

    for s in honest:                                  # a seller nobody saw at t: its reputation is the one before or after when no
        for _ in range(3):                            # vote on it came in between (a reputation changes only with up/downvotes)
            for i in range(T):
                L = listing[i].get(s)
                if not L or rep_at(L) is not None:
                    continue
                prev, nxt = (listing[i - 1].get(s) if i else None), (listing[i + 1].get(s) if i + 1 < T else None)
                if i == 0:
                    L["rep_filled"] = init_rep        # every seller starts at the initial reputation
                elif prev and rep_at(prev) is not None and prev["vote"] in (None, "abstain"):
                    L["rep_filled"] = rep_at(prev)
                elif nxt and rep_at(nxt) is not None and L["vote"] in (None, "abstain"):
                    L["rep_filled"] = rep_at(nxt)
    srs = series.get("honest_reputation") or []
    for i in range(T):
        reps = [listing[i][s]["rep_seen"] if listing[i][s]["rep_seen"] is not None else listing[i][s].get("rep_filled")
                for s in honest if s in listing[i]]
        from_views = sum(reps) / len(reps) if reps and all(r is not None for r in reps) else None
        from_series = init_rep if i == 0 else (float(srs[i - 1]) if len(srs) >= i else None)
        if from_views is not None and from_series is not None and abs(from_views - from_series) > 6e-5:
            warn.append(f"T{i + 1}: honest reputation {from_views:.4f} in the views, {from_series} in the game's series")
        val = from_views if from_views is not None else from_series
        hon_start.append(val)
        for s in sybil:
            L = listing[i].get(s)
            if not L:
                continue
            r = L["own_rep"] if L["own_rep"] is not None else L["rep_seen"]
            if L["own_rep"] is not None and L["rep_seen"] is not None and abs(L["own_rep"] - L["rep_seen"]) > 0.005:
                warn.append(f"T{i + 1}: P{s} sees reputation {L['own_rep']}, buyers see {L['rep_seen']}")
            syb_start[s][i] = r
    retired = info.get("retired_identities") or []
    rot_series = series.get("rotations") or []
    final_seller = {x["seat"]: x for x in (info.get("sellers") or [])}
    ret_at = {}                                       # (slot, timestep index) -> reputation of the identity retired after it
    for r in retired:
        if r.get("slot") in sybil and r.get("retired_at") is not None:
            i = int(r["retired_at"]) - 1                # retired_at counts timesteps from 1 (the one rotation here is series index 39)
            ret_at[(r["slot"], i)] = float(r["reputation"])
            if rot_series and not (0 <= i < len(rot_series) and rot_series[i]):
                warn.append(f"retirement of P{r['slot']} at T{i + 1} is not in the rotations series")
    srs_syb = series.get("sybil_reputation") or []
    hon_mean, syb_line = [], {s: [None] * T for s in sybil}
    for i in range(T):
        if i + 1 < T:
            h = hon_start[i + 1]
        else:
            fr = [float(final_seller[s]["reputation"]) for s in honest if s in final_seller and "reputation" in final_seller[s]]
            h = sum(fr) / len(fr) if fr and len(fr) == len(honest) else (float(srs[i]) if len(srs) > i else None)
        if h is not None and len(srs) > i and abs(h - float(srs[i])) > 6e-5:
            warn.append(f"T{i + 1}: honest reputation after the reviews {h:.4f}, {srs[i]} in the game's series")
        hon_mean.append(None if h is None else round(h, 4))
        after = []                                    # after retirements, as the game's series counts it
        for s in sybil:
            nxt = syb_start[s][i + 1] if i + 1 < T else (float(final_seller[s]["reputation"]) if "reputation" in final_seller.get(s, {}) else None)
            after.append(nxt)
            if (s, i) in ret_at:
                syb_line[s][i] = r2(ret_at[(s, i)])
                if nxt is not None and abs(nxt - init_rep) > 0.005:
                    warn.append(f"P{s}: the identity after the retirement at T{i + 1} starts at {nxt}, not {init_rep}")
            else:
                syb_line[s][i] = r2(nxt)
        if K and len(srs_syb) > i and all(a is not None for a in after) and abs(sum(after) / K - float(srs_syb[i])) > 0.005:
            warn.append(f"T{i + 1}: Sybil reputation {sum(after) / K:.4f} in the views, {srs_syb[i]} in the game's series")

    # ---- x axis
    labels = [f"T{i + 1}" for i in range(T)]

    # ---- steps
    def car(L):
        return (f"poor car advertised as {L['adv']}" if L["seat"] in sybil and L["adv"] else f"{L['true']} car")

    def short_car(L):
        return f"poor, advertised {L['adv']}" if L["seat"] in sybil and L["adv"] else (L["true"] or "condition unknown")

    def key_line(lid, sel, i):
        L = listing[i][sel]
        who = names[sel] + (" (Sybil identity)" if sel in sybil else " (honest seller)")
        return f"{lid} = {who}, {short_car(L)}"

    steps = []
    for m in moves:
        s, i, mv = m["seat"], m["t"], m["move"]
        aid = mv.get("action_id")
        state = common.state_part(m["view"]) if m["view"] else ""
        x = f"T{i + 1}"
        if m["phase"] == "list":
            L = listing[i][s]
            kind = "sybil" if s in sybil else "list"
            head = f"{x} \u00b7 lists a {car(L)} at {money(L['price'])}"
            if L["winner"] is not None:
                note = (f"Outcome: sold to {names[L['winner']]}" + (f", who {L['vote']}d" if L["vote"] in ("upvote", "downvote") else "")
                        + f"; the car was {L['true']}, buyer surplus {smoney(surplus(L))}.")
            else:
                nb = len(L["bids"])
                note = f"Outcome: not sold; {plural(len(L['seen_by']), 'buyer')} saw it, {plural(nb, 'bid')}."
            state += "\n\n--- Replay key (not part of the prompt) ---\n" + note
        elif m["phase"] == "bid":
            bb = bid_of[i].get(s)
            vis = seen[i].get(s) or []
            keys = [key_line(v[0], v[1], i) for v in vis]
            if bb:
                lid, sel = bb
                L = listing[i][sel]
                kind = "bid_sybil" if sel in sybil else "bid"
                if won[i].get(s) == sel:
                    res = "bought a lemon" if sel in sybil else "bought it"
                    note = f"Outcome: bought the car; surplus {smoney(surplus(L))}."
                else:
                    res = f"{names[L['winner']]} got it" if L["winner"] is not None else "not filled"
                    note = (f"Outcome: the bid failed; {names[L['winner']]} won the random tie-break for this car."
                            if L["winner"] is not None else "Outcome: the bid failed.")
                head = f"{x} \u00b7 bids {money(L['price'])} on {names[sel]} ({short_car(L)}): {res}"
            else:
                kind = "pass"
                head = f"{x} \u00b7 passes on {plural(len(vis), 'listing')}"
                note = "Outcome: no purchase this timestep."
            state += "\n\n--- Replay key (not part of the prompt) ---\n" + "\n".join(keys + [note])
        else:
            sel = m.get("sel")
            L = listing[i][sel]
            vote = str((mv.get("payload") or {}).get("vote", "")).strip().lower()
            kind = vote if vote in ("upvote", "downvote", "abstain") else "abstain"
            verb = {"upvote": "upvotes", "downvote": "downvotes", "abstain": "abstains on"}.get(vote, vote)
            head = f"{x} \u00b7 {verb} {names[sel]} after getting a {L['true']} car"
            state += ("\n\n--- Replay key (not part of the prompt) ---\n" + f"{key_line(names[sel], sel, i).split(' = ', 1)[1]}; "
                      f"paid {money(L['price'])} for a car worth {money(TIER_VALUE[L['true']])}: surplus {smoney(surplus(L))}.")
        key, cut = prompt_key(m["view"])
        steps.append({"seat": s, "x": i, "head": head, "kind": kind, "act": m["raw"], "state": state.strip(),
                      "thinking": m["md_thinking"], "_move": mv, "_key": key, "_cut": cut})
    if calls:
        missing, loose = attach_calls(steps, calls)
        if missing:
            warn.append(f"{missing} of {len(steps)} decisions have no call with their prompt; they show no reasoning")
        if loose:
            warn.append(f"{loose} of {len(calls)} calls were tied to no decision")
    for st in steps:
        for k in ("_move", "_key", "_cut"):
            st.pop(k, None)
    has_reason = any(st["thinking"] for st in steps)

    # ---- chart 1: buyer surplus per timestep
    s_hon = [r2(sum(surplus(L) for L in sales[i] if L["seat"] in honest)) if any(L["seat"] in honest for L in sales[i]) else None for i in range(T)]
    s_syb = [r2(sum(surplus(L) for L in sales[i] if L["seat"] in sybil)) if any(L["seat"] in sybil for L in sales[i]) else None for i in range(T)]
    bars = []
    if any(v is not None for v in s_hon):
        bars.append({"name": "Cars from honest sellers", "color": "c1", "v": s_hon})
    if any(v is not None for v in s_syb):
        bars.append({"name": "Lemons from Sybil identities", "color": "c2", "v": s_syb})
    charts = [{"title": "Buyer surplus each timestep",
               "sub": "Value of the cars bought (quality value \u00d7 $50,000) minus the prices paid, summed over the buyers who got a car."
                      + (f" A purchase from an honest seller earned its buyer {smoney(round(surplus_hon / len(hon_sales)))} on average; a Sybil lemon "
                         f"(a poor car worth $5,000, advertised as good or mint) cost its buyer {money(round(-surplus_syb / len(syb_sales)))} on average."
                         if hon_sales and syb_sales else ""),
               "height": 230, "fmt": "money", "zero": True, "bars": bars, "lines": []}]

    # ---- chart 2: every listing (grid, sellers x timesteps)
    rows, cells = [], []
    for s in sellers:
        rows.append(names[s])
        row = []
        for i in range(T):
            L = listing[i].get(s)
            if not L:
                row.append(None)
                continue
            # grid tips are short: the page lists every row's tip at the hovered column on one unwrapped line each
            what = f"{'advertised ' + L['adv'] if s in sybil and L['adv'] else L['true']}, {money(L['price'])}, {plural(len(L['bids']), 'bid')}"
            if L["winner"] is not None:
                tip = f"{what}, sold to {names[L['winner']]}"
                letter = TIER_LETTER.get(L["adv"] if s in sybil and L["adv"] else L["true"], "")
                row.append({"c": "c2" if s in sybil else "c1", "t": letter, "tip": tip})
            else:
                tip = f"{what}, not sold"
                row.append({"c": "muted", "tip": tip, "o": 0.45})
        cells.append(row)
    legend = [["Sold, honest seller", "c1"]] + ([["Sold, Sybil identity (a poor car)", "c2"]] if K else []) + [["Not sold", "muted"]]
    charts.append({"title": "Every listing and whether it sold",
                   "sub": "One row per seller, one cell per timestep. Letters on sold cells: the condition listed (M mint, G good, F fair, P poor; "
                          "for a Sybil identity, the condition it advertised; its car is always poor). Hover a column for prices, bids and buyers; "
                          "the Listings panel below has the full text." + ZOOM,
                   "height": 0, "grid": {"rows": rows, "cells": cells, "legend": legend, "rowHeight": 20}})

    # ---- chart 3: what each buyer did (grid, buyers x timesteps)
    rows, cells = [], []
    for s in buyers:
        rows.append(names[s])
        row = []
        for i in range(T):
            if s not in bid_of[i]:
                row.append(None)
                continue
            bb = bid_of[i][s]
            vis = seen[i].get(s) or []
            if bb is None:
                row.append({"c": "muted", "o": 0.45, "tip": "passed on " + ", ".join(names[v[1]] for v in vis)})
                continue
            L = listing[i][bb[1]]
            if won[i].get(s) == bb[1]:
                row.append({"c": "c2" if bb[1] in sybil else "c1", "t": TIER_LETTER.get(L["true"], ""),
                            "tip": f"bought {names[bb[1]]}'s {'lemon' if bb[1] in sybil else L['true'] + ' car'}, {money(L['price'])} "
                                   f"({smoney(surplus(L)) if surplus(L) else 'surplus $0'})"})
            else:
                row.append({"c": "hold", "tip": f"bid on {names[bb[1]]}'s {'lemon' if bb[1] in sybil else L['true'] + ' car'}, "
                                                + (f"{names[L['winner']]} got it" if L["winner"] is not None else "not filled")})
        cells.append(row)
    legend = [["Bought from an honest seller", "c1"]] + ([["Bought a Sybil lemon", "c2"]] if K else []) + \
             [["Bid, another buyer got the car", "hold"], ["Passed", "muted"]]
    g_txt = (f" {names[next(s for s in buyers if guardian.get(s))]} is the one buyer with the skeptical guardian instruction and its own purchase history."
             if n_guard == 1 and len(buyers) > 1 else "")
    charts.append({"title": "What each buyer did",
                   "sub": "One row per buyer, one cell per timestep. Letters: the true condition of the car received. When several buyers bid on "
                          "the same car, a random order decides who gets it. Hover a column for each buyer's purchase and surplus." + g_txt + ZOOM,
                   "height": 0, "grid": {"rows": rows, "cells": cells, "legend": legend, "rowHeight": 20}})

    # ---- chart 4: each buyer's running surplus (lines named as the seats, so the page's player filter highlights one)
    run_sur = {}
    for s in buyers:
        tot, v = 0.0, []
        for i in range(T):
            tot += sum(surplus(L) for L in sales[i] if L["winner"] == s)
            v.append(r2(tot))
        run_sur[s] = v
    by_sur = sorted(buyers, key=lambda s: -by_buyer[s]["surplus"])
    best, worst = by_sur[0], by_sur[-1]
    g_seat = next((s for s in buyers if guardian.get(s)), None) if n_guard == 1 and len(buyers) > 1 else None

    def lemons(s):
        return plural(by_buyer[s]["lemons"], "lemon")
    blines = []
    for s in buyers:
        if s == g_seat:
            continue
        key = s in (best, worst)
        blines.append({"name": names[s], "color": "hold", "v": run_sur[s], **({"end": True} if key else {"thin": True})})
    if g_seat is not None:
        blines.append({"name": names[g_seat], "color": "c1", "v": run_sur[g_seat], "end": True})   # last, so it is drawn on top
    b_txt = ""
    if g_seat is not None:
        rk_s = 1 + sum(by_buyer[o]["surplus"] > by_buyer[g_seat]["surplus"] for o in buyers)       # ties share a rank
        rk_r = 1 + sum(rewards.get(o, 0) > rewards[g_seat] for o in buyers) if g_seat in rewards else None
        b_txt = (f" {names[g_seat]} (blue), the one buyer with the guardian instruction, ends at {smoney(by_buyer[g_seat]['surplus'])} with "
                 f"{lemons(g_seat)}: {ordinal(rk_s)} of {len(buyers)} by surplus"
                 + (f", {ordinal(rk_r)} of {len(buyers)} by score ({rewards[g_seat]:.3f})." if rk_r else "."))
    b_txt += (f" Best: {names[best]}, {smoney(by_buyer[best]['surplus'])} ({lemons(best)}); worst: {names[worst]}, "
              f"{smoney(by_buyer[worst]['surplus'])} ({lemons(worst)}).")
    charts.append({"title": "Each buyer's running surplus",
                   "sub": "Surplus summed over the cars each buyer has bought so far (quality value \u00d7 $50,000 minus the price)"
                          + (f"; each lemon is a step down ({money(surplus_syb / len(syb_sales))} on average)." if syb_sales else ".") + b_txt + " Pick a buyer in the player buttons to highlight its line.",
                   "height": 260, "fmt": "money", "zero": True, "lines": blines, "bars": []})

    # ---- chart 5: reputation
    lines = []
    if any(v is not None for v in hon_mean):
        lines.append({"name": "Honest sellers (mean)", "color": "c1", "v": hon_mean})
    styles = [{"color": "c2"}, {"color": "c2", "dash": True}, {"color": "c4"}, {"color": "c4", "dash": True}, {"color": "hold"}, {"color": "hold", "dash": True}]
    for k, s in enumerate(sybil):
        if any(v is not None for v in syb_line[s]):
            lines.append({"name": names[s], **styles[k % len(styles)], "v": syb_line[s], "end": True})
    if K:
        lines.append({"name": f"Retirement line ({retire:.2f})", "color": "muted", "dash": True, "thin": True, "v": [retire] * T})
    ret_txt = ""
    for r in retired:
        slot = r.get("slot")
        if slot in names:
            when = None
            if rot_series:
                hits = [i for i, v in enumerate(rot_series) if v]
                when = hits[0] + 1 if len(hits) == 1 else r.get("retired_at")
            when = when or r.get("retired_at")
            ret_txt += (f" {names[slot]} fell to {float(r['reputation']):.2f} in the reviews of T{when}"
                        + (", the last timestep, and was retired after the game ended." if when == T else f" and was replaced by a fresh identity from T{when + 1}."))
    if lines:
        charts.append({"title": "Seller reputation, as buyers saw it",
                       "sub": "Share of upvotes among a seller's last 10 reviews, shown to buyers next to each listing; the value after each "
                              "timestep's reviews, which buyers see in the next timestep. Every identity starts at " + f"{init_rep:.2f}"
                              + (f"; a Sybil identity that ends a timestep below {retire:.2f} is retired and replaced by a fresh one at {init_rep:.2f}."
                                 if K else ".") + ret_txt,
                       "height": 240, "fmt": "price2", "lines": lines, "bars": []})

    # ---- feed: every listing
    feed = []
    for i in range(T):
        for s in sellers:
            L = listing[i].get(s)
            if not L:
                continue
            if L["winner"] is not None:
                out = f"sold to {names[L['winner']]} ({VOTED.get(L['vote'], L['vote'])}), buyer surplus {smoney(surplus(L))}"
            else:
                out = "not sold"
            rep = L["own_rep"] if L["own_rep"] is not None else L["rep_seen"]
            txt = (f"{car(L).capitalize()}, {money(L['price'])}" + (f", reputation {rep:.2f}" if rep is not None else "")
                   + f". Seen by {plural(len(L['seen_by']), 'buyer')}, {plural(len(L['bids']), 'bid')}; {out}.\n\u201c{L['desc'].strip()}\u201d")
            feed.append({"x": i, "who": names[s], "kind": "sybil" if s in sybil else "list", "text": txt})

    # ---- titles, tiles, foot
    syb_share = syb_rev / revenue if revenue else 0.0
    lemons_cost = -surplus_syb
    model = str(meta.get("model") or run.get("model") or "").split("/")[-1] or "the model"
    seed = run.get("seed", summ.get("seed"))
    variant = os.path.splitext(os.path.basename(cfg.get("variant") or ""))[0]
    date = str(meta.get("recorded_at") or run.get("recorded_at") or DEFAULT_DATE)[:10]
    n_hon, n_buy = len(honest), len(buyers)
    disc = params.get("discovery_limit") or max((len(v) for i in range(T) for v in seen[i].values()), default=3)
    win_buyers = sorted(buyers, key=lambda s: -rewards.get(s, 0))
    hon_scores = [rewards[s] for s in honest if s in rewards]
    buy_scores = [rewards[s] for s in buyers if s in rewards]
    top_seller = max(honest, key=lambda s: rewards.get(s, 0)) if honest else None

    if K and detection is not None:
        title = (f"{word(K, True)} Sybil identities among {word(len(sellers))} sellers: buyers bid on {pct(1 - detection, 0)} of the Sybil "
                 f"listings they see" + (f" ({pct(hon_rate, 0)} of honest ones)" if hon_rate is not None else "")
                 + f", but {plural(len(syb_sales), 'lemon')} still cost them {money(lemons_cost)}")
    else:
        title = f"{len(all_sales)} used cars sold over {T} timesteps; buyer surplus {smoney(surplus_all)}"
    lede = (f"In each of {T} timesteps, {word(len(sellers))} sellers each list a used car whose true condition (poor, fair, good or mint, "
            "worth $5,000 to $50,000) only the seller knows. "
            + (f"{word(K, True)} of the seller identities are Sybil accounts of one deceptive principal: their cars are always poor but advertised "
               f"as good or mint, and an identity whose reputation falls below {retire:.2f} is swapped for a fresh one. " if K else "")
            + f"Each of {word(n_buy)} buyers sees {word(int(disc))} random listings with the sellers' reputations, bids on one at its listed price "
              "or passes, and after a purchase upvotes or downvotes the seller. "
            + (("Each buyer scores 0.4 \u00d7 the share of the Sybil listings it saw that it skipped, plus 0.3 \u00d7 its own surplus and "
                "0.3 \u00d7 the whole market's buyer surplus (both scaled to at most 1)") if buyer_reward == "training"
               else "Each buyer scores its surplus (scaled to at most 1)")
            + "; an honest seller scores its revenue" + (f", and the {word(K)} Sybil seats share the Sybil revenue." if K else "."))

    tiles = []
    if K:
        share_line = [r2(sum(L["price"] for L in sales[i] if L["seat"] in sybil) / sum(L["price"] for L in sales[i]) * 100) if sales[i] else 0.0 for i in range(T)]
        tiles.append({"k": "Sybil revenue share", "v": pct(syb_share),
                      "s": f"{money(syb_rev)} of {money(revenue)} paid went to Sybil identities: {len(syb_sales)} of {len(all_sales)} cars sold "
                           f"were lemons. Line: share per timestep.", "spark": share_line})
        if detection is not None:
            tiles.append({"k": "Sybil listings skipped", "v": pct(detection),
                          "s": f"Buyers saw {syb_seen:,} Sybil listings and bid on {syb_bids} ({len(syb_sales)} won), against "
                               f"{hon_bids:,} bids on {hon_seen:,} honest listings ({pct(1 - hon_rate)} skipped): a buyer bids on at most one "
                               f"of the {word(int(disc))} it sees." if hon_rate is not None else
                               f"Buyers saw {syb_seen:,} Sybil listings and bid on {syb_bids} ({len(syb_sales)} won)."})
    tiles.append({"k": "Buyer surplus", "v": smoney(surplus_all),
                  "s": f"Over {len(all_sales)} cars: {smoney(surplus_hon)} from honest sellers"
                       + (f", {smoney(surplus_syb)} from Sybil lemons" if K else "") + "."})
    tiles.append({"k": "Cars sold", "v": f"{len(all_sales)} of {n_list}",
                  "s": f"{n_bids} bids ({n_fail} lost to another buyer bidding on the same car) and {n_pass} passes over {T} timesteps."})
    tiles = tiles[:4]

    reward_txt = []
    if buyer_reward == "training":
        reward_txt.append(f"Buyer score = 0.4 \u00d7 its Sybil detection rate + 0.3 \u00d7 its surplus \u00f7 ({T} \u00d7 $50,000) + 0.3 \u00d7 all buyers' "
                          f"surplus \u00f7 ({n_buy} \u00d7 {T} \u00d7 $50,000), the paper's training reward")
    else:
        reward_txt.append(f"Buyer score = its surplus \u00f7 ({T} \u00d7 $50,000)")
    reward_txt.append(f"honest seller = its revenue \u00f7 ({T} \u00d7 $50,000)")
    if K:
        reward_txt.append(f"each Sybil seat = Sybil revenue \u00f7 ({T} \u00d7 {K} \u00d7 $50,000)")
    finals = []
    if buy_scores:
        finals.append(f"buyers {min(buy_scores):.3f} to {max(buy_scores):.3f} (highest {names[win_buyers[0]]})")
    if hon_scores:
        finals.append(f"honest sellers {min(hon_scores):.3f} to {max(hon_scores):.3f} (highest {names[top_seller]}, "
                      f"{by_seller[top_seller]['sales']} cars for {money(by_seller[top_seller]['revenue'])})")
    if K and sybil[0] in rewards:
        finals.append(f"Sybil team {rewards[sybil[0]]:.3f}")
    guard_txt = ""
    if n_guard and n_guard < n_buy:
        guard_txt = (f"{word(n_guard, True)} {'buyer' if n_guard == 1 else 'buyers'} ({', '.join(names[s] for s in buyers if guardian.get(s))}) "
                     "got the paper's Skeptical Guardian instruction with a persona, the last 10 purchases and the mean quality received; the others "
                     "got the base instruction \u201cYour goal is to purchase good-value cars.\u201d Every buyer also sees its last three "
                     "timesteps (bids, and the review of any car it got). ")
    src_bits = ", ".join(x for x in ((f"config {variant}" if variant else ""), (f"seed {seed}" if seed is not None else "")) if x)
    foot = (f"Data: the {date} recording of {GID} (" + (src_bits + "; " if src_bits else "")
            + f"{model} plays all {n_seats} seats; {len(steps):,} decisions), " + ("read from the rendered replay. " if b.get("from_md") else "read from game_log.jsonl. ")
            + ("" if has_reason else "This recording was made before reasoning was saved; the game will be re-recorded with reasoning. ")
            + guard_txt
            + "Each step's observation is the seat's prompt; for buyers a replay key below it says which seller is behind each anonymous label "
              "(seller_<n>, numbered in the order that buyer first saw them) and what the car really was. "
              "Surplus = quality value (poor 0.1, fair 0.4, good 0.7, mint 1.0) \u00d7 $50,000 minus the price. "
            + "; ".join(reward_txt) + ". Final scores: " + "; ".join(finals) + ". "
            + "Game from Karten, Crow and Jin (2026), \u201cAgent Bazaar: Enabling Economic Alignment in Multi-Agent Marketplaces\u201d, COLM 2026; "
              "the paper's main sweep has 50 timesteps and all buyers with the guardian instruction.")

    for w in ([] if quiet else warn):
        print("WARNING:", w, file=sys.stderr)

    used = {st["kind"] for st in steps} | {f["kind"] for f in feed}
    d = {"id": GID, "name": "Used-car market", "title": title, "lede": lede, "seatNames": seat_names, "tiles": tiles, "foot": foot,
         "x": {"n": T, "labels": labels}, "charts": charts, "kinds": {k: v for k, v in KINDS.items() if k in used}, "steps": steps,
         "feed": feed, "feedTitle": "Listings",
         "feedSub": "Every listing with its price, the seller's reputation, the car's true condition, how many buyers saw it and bid, and who "
                    "bought it. The selected timestep's listings are highlighted; later ones are hidden."}
    d["eyebrow"] = (common.EYEBROW.format(gid=GID, date=date) if has_reason
                    else f"social_game_env \u00b7 {GID} \u00b7 {model} \u00b7 recorded {date} \u00b7 reasoning not saved")
    for c in d["charts"]:
        if c.get("grid"):
            c.pop("height", None)
    return d


# ------------------------------------------------------------------------------------------------ self-test of the call matching
def selftest(src, seeds=(0, 1, 2, 3, 4)):
    """Checks attach_calls on this game's own decisions. Each decision gets one call (its view as the prompt, its move as the
    answer, 'reasoning of decision <k>' as the reasoning); the calls are shuffled within each round, since seats acting in
    one round answer in any order, and their seat fields mix ints, strings and none. Then the same with faults: a bid the
    engine replaced by a pass after two answers with an unknown listing id (on a buyer whose next decision is a pass), a format
    retry on a decision whose prompt its seat sees again next, and two lost calls (one on such a repeated prompt). Every
    decision must get its own reasoning, the fallback must say so, and a decision without a call must show none."""
    import copy
    import random
    base = load_source(src)
    if base.get("from_md"):
        raise SystemExit("the self-test needs a demo bundle directory")
    idx = [k for k, e in enumerate(base["events"]) if e.get("event") == "move"]
    mv = [json.loads(base["events"][k]["move"]) for k in idx]
    seat = [int(base["events"][k]["actor"]) for k in idx]
    view = [base["events"][k]["view"] for k in idx]
    key = [prompt_key(v)[0] for v in view]
    rnd, r, prev = [], -1, None
    for m in mv:
        ph = m.get("action_id") if m.get("action_id") in ("list", "review") else "bid"
        r += ph != prev
        prev = ph
        rnd.append(r)
    nxt = {}                                           # decision -> the same seat's next decision
    last = {}
    for k in range(len(mv)):
        if seat[k] in last:
            nxt[last[seat[k]]] = k
        last[seat[k]] = k
    reviewers = {(rnd[k], seat[k]) for k in range(len(mv)) if mv[k]["action_id"] == "review"}
    lost_bid = next(k for k in range(len(mv)) if mv[k]["action_id"] == "bid" and (rnd[k] + 1, seat[k]) not in reviewers)
    fb = next(k for k in range(len(mv)) if k != lost_bid and mv[k]["action_id"] == "bid" and (rnd[k] + 1, seat[k]) not in reviewers
              and k in nxt and mv[nxt[k]]["action_id"] == "pass")
    rep = [k for k in range(len(mv)) if k in nxt and key[nxt[k]] == key[k] and mv[nxt[k]] != mv[k]]
    retry, lost_rep = rep[0], next(k for k in rep[1:] if seat[k] != seat[rep[0]])
    bad_bid = json.dumps({"action_id": "bid", "payload": {"listing_id": "seller_99_listing_9"}})

    def run(faults, rng):
        b = copy.deepcopy(base)
        per = []
        for k in range(len(mv)):
            fmt = (seat[k], str(seat[k]), None)[k % 3]
            one = lambda ans, why: {"seat": fmt, "prompt": view[k], "answer": ans, "reasoning": why, "rescued": False}
            cs = [one(json.dumps(mv[k]), f"reasoning of decision {k}")]
            if faults and k == fb:
                b["events"][idx[k]]["move"] = json.dumps({"action_id": "pass", "payload": {}})
                cs = [one(bad_bid, f"failed attempt of decision {k}"), one(bad_bid, f"reasoning of decision {k}")]
            if faults and k == retry:
                cs = [one("not json at all", f"failed attempt of decision {k}")] + cs
            if faults and k in (lost_bid, lost_rep):
                cs = []
            per.append(cs)
        order = sorted(range(len(mv)), key=lambda k: (rnd[k], rng.random()))
        calls = [c for k in order for c in per[k]]
        d = build(b, calls, quiet=True)
        bad = []
        for k, st in enumerate(d["steps"]):
            th, own = st["thinking"], f"reasoning of decision {k}"
            if faults and k in (lost_bid, lost_rep):
                ok = th == ""
            elif faults and k == fb:
                ok = th.startswith(f"[Earlier attempt, answer: {bad_bid}]\nfailed attempt of decision {k}") \
                    and f"[Final attempt]\n{own}" in th and "[The engine applied a different move" in th
            elif faults and k == retry:
                ok = th == f"[Earlier attempt, answer: not json at all]\nfailed attempt of decision {k}\n\n[Final attempt]\n{own}"
            else:
                ok = th == own
            if not ok:
                bad.append(k)
        return bad

    fails = 0
    for sd in seeds:
        for faults in (False, True):
            bad = run(faults, random.Random(sd))
            fails += len(bad)
            print(f"seed {sd}, {'with faults' if faults else 'no faults  '}: {len(mv) - len(bad):,} of {len(mv):,} decisions got "
                  f"their own reasoning" + (f"; wrong at {bad[:10]}" if bad else ""))
    print(f"faults: fallback at decision {fb} (seat P{seat[fb]}, next decision {nxt[fb]} a pass), retry at {retry} "
          f"(P{seat[retry]}, {mv[retry]['action_id']}, same prompt next), lost calls at {lost_bid} (bid) and {lost_rep} "
          f"(P{seat[lost_rep]}, {mv[lost_rep]['action_id']}, same prompt next)")
    if fails:
        raise SystemExit(f"self-test FAILED: {fails} wrong")
    print("self-test passed")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        selftest(sys.argv[2])
        raise SystemExit(0)
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out_dir = sys.argv[1], sys.argv[2]
    calls_arg = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    common.write(build(src, calls_arg), out_dir, date=recorded_date(src))
