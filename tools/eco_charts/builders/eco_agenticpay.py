"""Replay data for eco_agenticpay (AgenticPay: one buyer and one seller negotiate the total price of one product).

usage: python3 -I eco_agenticpay.py BUNDLE_DIR_OR_MD OUT_DIR [CALLS.jsonl]

BUNDLE_DIR_OR_MD is a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, transcript.md), a file inside
one, or the rendered Markdown replay (one JSON line per logged event, and the model calls when they were saved). CALLS.jsonl is
the optional model-call log (see common.load_calls); each call is tied to the message it produced by order, seat and answer,
and its reasoning, prompt and raw answer then fill that step. Every number shown comes from the log; the scores are the
logged ones (recomputed here only to confirm the score weights quoted in the tiles and the footnote).

The shared axis is Setup, one slot per message (R1 B, R1 S, ...), Result. The two end slots hold the hidden setup and the
outcome (the deal dot), so no message sits on the edge of a plot and the deal dot never covers the price it closed on. No
chart has bars: the page starts the y axis of any chart with a `bars` key at 0, which would squash the prices."""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_agenticpay"
DEFAULT_DATE = "2026-11-09"
# Algorithm 1 of the paper with its section 5.1 weights (the game's defaults): deal bonus, quality weight, efficiency bonus,
# discount per round, failure penalty. A config may override them through the hidden variables of the same names.
WEIGHTS = (("deal_bonus", 30.0), ("quality_weight", 55.0), ("efficiency_bonus", 15.0), ("gamma", 0.99), ("failure_penalty", 15.0))
KINDS = {"b_offer": ["Buyer offer", "c1"], "s_offer": ["Seller offer", "c2"], "accept": ["Accept (same price)", "c3"],
         "hold": ["No new price", "muted"]}
# product fields that hold a public reference price (list price, original price, book value, market reference, asking price)
REF_KEY = re.compile(r"price|value|reference|asking|rate|msrp", re.I)
PER_UNIT_TOKENS = {"unit", "each", "per", "piece", "item"}
QTY_KEY = re.compile(r"quantity|qty|units|count", re.I)


# ------------------------------------------------------------------ sources
def _md_text(path):
    """The rendered replay, with its continuation pages (<game>_2.md, ...) when the replay was split."""
    base, parts = re.sub(r"\.md$", "", path), []
    parts.append(open(path, encoding="utf-8").read())
    k = 2
    while os.path.exists(f"{base}_{k}.md"):
        parts.append(open(f"{base}_{k}.md", encoding="utf-8").read()); k += 1
    return "\n".join(parts)


def _parse_md(path):
    """Rebuild a bundle from the rendered Markdown replay (render_replays2.py): one '- **event**...`{json}`' line per logged
    event, the header with the seed, model and rewards, and the model calls ('### ... You are seat Pk' blocks with the
    answer and a ````text reasoning block)."""
    text = _md_text(path)
    events = []
    for m in re.finditer(r"^- \*\*([A-Za-z_]+)\*\*[^`\n]*`(\{.*\})`\s*$", text, re.M):
        try:
            e = json.loads(m.group(2))
        except ValueError:      # event lines are cut at 1,500 characters; a cut line is skipped
            continue
        e["event"] = m.group(1); events.append(e)
    rw = re.search(r"各座位得分 (\{.*?\})", text)          # "scores per seat {...}"
    seed = re.search(r"种子 (\d+)", text)                              # "seed N"
    model = re.search(r"模型 (\S+)", text)                             # "model X"
    over = next((e for e in events if e["event"] == "game_over"), {})
    run = {"seed": int(seed.group(1)) if seed else None, "model": model.group(1) if model else "", "recorded_at": None}
    summary = {"rewards": json.loads(rw.group(1)) if rw else {}, "variables": over.get("variables", {})}
    calls = []
    for blk in re.split(r"\n(?=### 第 \d+ 次调用)", text)[1:]:   # "### call N"
        seat = re.search(r"You are seat P(\d+)", blk.split("\n", 1)[0])
        ans = re.search(r"\*\*回答：\*\* `(.*)`", blk)                  # "**answer:** `...`"
        th = re.search(r"````text\n(.*?)\n````", blk, re.S)
        calls.append({"seat": int(seat.group(1)) if seat else None, "prompt": "", "answer": (ans.group(1) if ans else "").strip(),
                      "reasoning": th.group(1) if th else "", "rescued": False})
    return {"events": events, "summary": summary, "run": run, "config": {}, "transcript": "", "from_md": True}, calls


def load_source(src):
    if os.path.isdir(src):
        return common.load_bundle(src), []
    d = os.path.dirname(os.path.abspath(src))
    if os.path.exists(os.path.join(d, "game_log.jsonl")):
        return common.load_bundle(d), []
    return _parse_md(src)


# ------------------------------------------------------------------ formatting
def money(v):
    if v is None:
        return "none"
    v = float(v)
    return f"${v:,.0f}" if v.is_integer() else f"${v:,.2f}"


def num(v):
    return None if v is None else float(v)


def fx(v, dp=2):
    """A number with a typographic minus sign."""
    return f"{v:.{dp}f}".replace("-", "−")


def pct(v):
    return f"{100 * v:.0f}%"


def kv_text(d):
    """A product or context dictionary as one readable line: 'name X; brand Y; features a, b' (the name first)."""
    out = []
    for k, v in sorted((d or {}).items(), key=lambda kv: kv[0] != "name"):
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        elif isinstance(v, dict):
            v = ", ".join(f"{a} {b}" for a, b in v.items())
        out.append(f"{str(k).replace('_', ' ')} {v}")
    return "; ".join(out)


def _as_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and re.fullmatch(r"\s*\$?\s*[\d,]+(\.\d+)?\s*", v):
        return float(v.replace("$", "").replace(",", ""))
    return None


def public_references(product):
    """The public reference prices inside the product description (list price, original price, book value, market
    reference, asking price), in the order given: {label, value (a number or None), text, per_unit}. A per-unit price is
    flagged by its key (unit, each, per, ...) or, when the product names a quantity, decided later by its scale."""
    out, qty = [], None
    for k, v in (product or {}).items():
        if QTY_KEY.search(str(k)) and _as_number(v):
            qty = _as_number(v)
    for k, v in (product or {}).items():
        key = str(k)
        if not REF_KEY.search(key):
            continue
        lab = "List price" if key == "price" else key.replace("_", " ").capitalize()
        x = _as_number(v)
        if x is None and isinstance(v, str) and re.search(r"\d", v):          # a range or a phrase: shown in the note only
            out.append({"label": lab, "value": None, "text": f"{lab.lower()} {v.strip()}", "per_unit": False, "qty": qty})
        elif x is not None and x > 0:
            per = bool(PER_UNIT_TOKENS & set(re.split(r"[_\s]+", key.lower())))
            out.append({"label": lab, "value": x, "text": None, "per_unit": per, "qty": qty})
    return out


def parse_answer(s):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (s or "").strip())
    for cand in (s, s[s.find("{"): s.rfind("}") + 1] if "{" in s else ""):
        try:
            o = json.loads(cand)
            if isinstance(o, dict):
                return o
        except ValueError:
            pass
    return None


def answer_price(a):
    p = (a or {}).get("payload") or {}
    off = p.get("offer") if isinstance(p, dict) else None
    return off.get("price") if isinstance(off, dict) else None


def answer_matches(ans, msg):
    a = parse_answer(ans)
    if not a:
        return False
    if msg.get("price") is None:
        return a.get("action_id") == "hold"
    try:
        return a.get("action_id") == "offer" and abs(float(answer_price(a)) - float(msg["price"])) < 1e-9
    except (TypeError, ValueError):
        return False


def valid_move(ans):
    """Whether an answer is a move the engine would have played as it is (a hold, or an offer with a positive price)."""
    a = parse_answer(ans)
    if not a:
        return False
    if a.get("action_id") == "hold":
        return True
    try:
        return a.get("action_id") == "offer" and float(answer_price(a)) > 0
    except (TypeError, ValueError):
        return False


def tie_calls(msgs, calls):
    """For each message, the model call that produced it, or None. Calls are in time order and the two sides alternate, so
    the calls for one message are a consecutive run of that seat's calls (format retries, then the answer that was used);
    the last call in the run whose answer is the logged move is taken, else the last call of the run (a replaced answer).
    A call whose seat is unknown (None) may belong to either side, so it joins a run only while the run has not yet found
    the logged move, and never when it is itself a different valid move (that is the next message's call)."""
    out, ptr = [], 0
    for m in msgs:
        while ptr < len(calls) and calls[ptr]["seat"] is not None and calls[ptr]["seat"] != m["seat"]:
            ptr += 1
        run = []
        while ptr < len(calls):
            c = calls[ptr]
            if c["seat"] is not None and c["seat"] != m["seat"]:
                break
            if c["seat"] is None and run:
                if any(answer_matches(r["answer"], m) for r in run):
                    break
                if valid_move(c["answer"]) and not answer_matches(c["answer"], m):
                    break
            run.append(c); ptr += 1
        hit = [c for c in run if answer_matches(c["answer"], m)]
        out.append((hit or run or [None])[-1])
    return out


# ------------------------------------------------------------------ build
def build(bundle_dir, calls=None):
    B, md_calls = load_source(bundle_dir)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    from_log = bool(calls)                      # the calls come from a model-call log, not from the rendered replay
    calls = list(calls or md_calls or [])
    ev = B["events"]
    first = lambda name: next((e for e in ev if e.get("event") == name), {})
    setup, inst, end, over = first("setup"), first("instance"), first("game_end"), first("game_over")
    S, run, cfg = B["summary"] or {}, B["run"] or {}, B["config"] or {}
    V = dict(over.get("variables") or {}); V.update(S.get("variables") or {})
    metrics = dict(S.get("_metrics") or S.get("metrics") or {}); metrics.update(end.get("_metrics") or {})
    rewards = S.get("rewards") or (run.get("result") or {}).get("rewards") or {}
    roles = {int(k): v for k, v in (setup.get("roles") or S.get("roles") or {}).items()}
    if inst.get("buyer_seat") is not None:
        roles[int(inst["buyer_seat"])] = "BUYER"
    if inst.get("seller_seat") is not None:
        roles[int(inst["seller_seat"])] = "SELLER"
    n_players = max(2, len(roles))
    seat_names = ["Buyer" if roles.get(k) == "BUYER" else "Seller" for k in range(n_players)]
    bseat = next(k for k in range(n_players) if roles.get(k) == "BUYER")
    sseat = next(k for k in range(n_players) if roles.get(k) == "SELLER")

    pmax, pmin = num(inst.get("buyer_max_price")), num(inst.get("seller_min_price"))
    ask0, zone = num(inst.get("initial_asking_price")), num(inst.get("zone"))
    if zone is None and pmax is not None and pmin is not None:
        zone = pmax - pmin
    R = int(V.get("max_rounds") or (cfg.get("variables") or {}).get("max_rounds") or 20)
    tol = float(V.get("price_tolerance") or 0.0)
    cross = bool(V.get("deal_on_cross"))
    product, context = V.get("product") or {}, V.get("environment") or {}
    pname = product.get("name") or str(inst.get("task") or "the product").replace("_", " ")
    outcome = end.get("outcome") or V.get("outcome") or "TIMEOUT"
    deal_price, deal_round = num(end.get("deal_price", V.get("deal_price"))), end.get("deal_round")
    bscore, sscore, gscore = num(end.get("buyer_score")), num(end.get("seller_score")), num(end.get("global_score"))
    brew = num(end.get("buyer_reward", rewards.get(str(bseat))))
    srew = num(end.get("seller_reward", rewards.get(str(sseat))))
    msgs = [e for e in ev if e.get("event") == "message"]
    if not msgs:
        raise SystemExit("no message events in " + str(bundle_dir))
    model = re.sub(r"^.*/", "", str(run.get("model") or "deepseek-flash"))

    # scoring: the paper's Algorithm 1, or the official code's round accounting under code_scoring (rules, Reward)
    Dw, Ww, Ew, gam, Fw = (float(V.get(k, inst.get(k, dflt))) for k, dflt in WEIGHTS)
    code = bool(inst.get("code_scoring", V.get("code_scoring", (cfg.get("variables") or {}).get("code_scoring"))))
    top = Dw + Ww + Ew
    disc = lambda t: gam ** (t if code else t - 1)                       # discount of a deal closed in round t
    fail = lambda t: -Fw * (1 - gam ** (t if code else R - 1))          # failure score after t rounds played

    def deal_value(price, t):
        """(buyer points, seller points) of a deal at this price closed in round t."""
        if not zone or price is None:
            return None, None
        if not pmin - 1e-9 <= price <= pmax + 1e-9:
            return fail(t), fail(t)
        rb_, rs_ = (pmax - price) / zone, (price - pmin) / zone
        return disc(t) * (Dw + Ww * rb_ + Ew), disc(t) * (Dw + Ww * rs_ + Ew)

    # the buyer's private info as its own (first) prompt shows it; the run record keeps the first prompt of the game
    rp = run.get("prompt") or {}
    rp_text, rp_seat = rp.get("text") or "", rp.get("seat")
    m = re.search(r"Your private info: (\{.*\})\n", rp_text)
    buyer_priv = {}
    if m and rp_seat == bseat:
        try:
            buyer_priv = json.loads(m.group(1))
        except ValueError:
            pass
    pers = {bseat: inst.get("buyer_personality") or "", sseat: inst.get("seller_personality") or ""}

    def priv_text(seat):
        if seat == bseat:
            s = f"maximum price {money(pmax)} (confidential)"
            if buyer_priv.get("user_requirement"):
                s += f"; requirement: {buyer_priv['user_requirement']}"
            if buyer_priv.get("user_profile"):
                s += f"; profile: {buyer_priv['user_profile']}"
        else:
            s = f"minimum acceptable price {money(pmin)} (confidential); initial asking price {money(ask0)}"
        return s + (f"; personality: {pers[seat]}" if pers.get(seat) else "")

    def rebuilt_state(i, bp, sp):
        mm = msgs[i]; seat = mm["seat"]
        lines = ["Rebuilt from the game log: the prompt for this step was not saved (instructions text omitted).",
                 f"You are seat P{seat}. Your role: {roles.get(seat)}.",
                 f"Your private info: {priv_text(seat)}.",
                 f"Round {mm.get('round')} of {R}. Latest prices: buyer {money(bp)}, seller {money(sp)}.",
                 f"Product (public): {kv_text(product)}." if product else None,
                 f"Market context (public): {kv_text(context)}." if context else None,
                 "Conversation so far:" + ("" if i else " none yet.")]
        for q in msgs[:i]:
            lines.append(f"- Round {q.get('round')}, {q.get('role', '').capitalize()}: "
                         f"{money(q['price']) if q.get('price') is not None else 'no new price'}. {q.get('talk') or ''}".rstrip())
        return "\n".join(x for x in lines if x is not None)

    # private context for the step headings: the viewer sees both sides' limits, the players never do
    def vs_max(p):
        if pmax is None:
            return ""
        return "its private max" if abs(pmax - p) < 1e-9 else f"{money(abs(pmax - p))} {'under' if p < pmax else 'over'} its private max"

    def vs_min(p):
        if pmin is None:
            return ""
        return "its private minimum" if abs(p - pmin) < 1e-9 else f"{money(abs(p - pmin))} {'above' if p > pmin else 'under'} its private minimum"

    def vs_ask(p):
        if ask0 is None:
            return ""
        return "its private initial ask" if abs(p - ask0) < 1e-9 else f"{money(abs(p - ask0))} {'under' if p < ask0 else 'over'} its private initial ask"

    paren = lambda s: f" ({s})" if s else ""

    tied = tie_calls(msgs, calls) if calls else [None] * len(msgs)
    n = len(msgs)
    X = n + 2                                   # Setup, one slot per message, Result
    labels, steps, feed = ["Setup"], [], []
    bline, sline, rounds = [], [], []
    bp = sp = None
    used, rebuilt, saved, any_thinking, deal_i, closed_check = [], 0, [], False, None, None
    log_prompts, log_thinking = 0, 0             # steps whose observation / reasoning came from the model-call log
    round_gaps = []                             # |buyer - seller| at the end of each round, as the engine's min_gap
    for i, mm in enumerate(msgs):
        seat, rd = mm["seat"], mm.get("round")
        buyer = seat == bseat
        other_w = "seller" if buyer else "buyer"
        price, talk = num(mm.get("price")), mm.get("talk") or ""
        own_before, other = (bp, sp) if buyer else (sp, bp)
        c = tied[i]
        if c and c.get("prompt"):
            state = common.state_part(c["prompt"]); saved.append(i); log_prompts += from_log
        elif i == 0 and rp_text and rp_seat == seat and '"dialogue": []' in rp_text:
            state = common.state_part(rp_text); saved.append(i)
        else:
            state = rebuilt_state(i, bp, sp); rebuilt += 1
        # what the move was, with where it sits against the mover's private numbers
        if price is None:
            kind = "hold"
            head = f"Round {rd} · no new price" + (f" (still {money(own_before)})" if own_before is not None else "")
        elif other is not None and abs(price - other) <= tol:
            kind = "accept"
            head = f"Round {rd} · accepts the {other_w}'s {money(other)}"
            if own_before is not None and abs(price - own_before) <= tol:
                # the other side had just matched this side's own standing price; this message only confirms it (the
                # seller after a buyer who matched its ask: the deal is checked only after the seller's reply)
                head = (f"Round {rd} · confirms {money(price)}, its own {'offer' if buyer else 'ask'} the {other_w} matched"
                        + paren(vs_max(price) if buyer else vs_min(price)))
            elif own_before is None:
                head += (" without a counter-offer" + paren(vs_max(other)) if buyer else
                         " without countering" + (f" (its private initial ask was {money(ask0)})" if ask0 is not None else ""))
            else:
                head += paren(vs_max(other) if buyer else vs_min(other))
        else:
            kind = "b_offer" if buyer else "s_offer"
            if own_before is None:
                head = (f"Round {rd} · opens at {money(price)}" + paren(vs_max(price)) if buyer else
                        f"Round {rd} · opens with an ask of {money(price)}" + paren(vs_min(price) if pmin is not None and price < pmin else vs_ask(price)))
            elif price == own_before:
                head = f"Round {rd} · repeats {money(price)}"
            else:
                word = ("raises" if price > own_before else "lowers") + (" the offer" if buyer else " the ask")
                head = f"Round {rd} · {word} to {money(price)} ({'+' if price > own_before else '−'}{money(abs(price - own_before))})"
                if other is not None and abs(own_before - other) <= tol:       # it walks away from a price the other side matched
                    head += f" after the {other_w} matched its {money(own_before)}"
                if buyer and pmax is not None and price > pmax:
                    head += f", {vs_max(price)}"
                elif not buyer and pmin is not None and price < pmin:
                    head += f", {vs_min(price)}"
        if price is not None:
            if buyer:
                bp = price
            else:
                sp = price
        # the round check after the seller's message, as the engine does it
        if not buyer and bp is not None and sp is not None:
            round_gaps.append(abs(bp - sp))
        closes = (not buyer) and bp is not None and sp is not None and (abs(bp - sp) <= tol or (cross and sp <= bp))
        if closes and closed_check is None:
            closed_check = (rd, (bp + sp) / 2 if abs(bp - sp) <= tol else sp)
            deal_i = i
            head += " · deal" if kind == "accept" else f" · deal at {money(closed_check[1])}"
        if kind not in used:
            used.append(kind)
        if c is not None and answer_matches(c["answer"], mm):
            act = json.dumps(parse_answer(c["answer"]), ensure_ascii=False)
        else:
            act = json.dumps({"action_id": "hold", "payload": {"talk": talk}} if price is None else
                             {"action_id": "offer", "payload": {"talk": talk, "offer": {"price": mm["price"]}}}, ensure_ascii=False)
        thinking = (c or {}).get("reasoning") or ""
        any_thinking = any_thinking or bool(thinking)
        log_thinking += from_log and bool(thinking)
        labels.append(f"R{rd} {'B' if buyer else 'S'}")
        steps.append({"seat": seat, "x": i + 1, "head": head, "kind": kind, "act": act, "state": state, "thinking": thinking})
        feed.append({"x": i + 1, "step": i, "who": seat_names[seat], "kind": kind,
                     "text": (f"{money(price)} · " if price is not None else "No new price · ") + talk})
        bline.append(bp); sline.append(sp); rounds.append(rd)
    labels.append("Result")

    # the logged result must be the one the messages imply
    if outcome != "TIMEOUT":
        if closed_check is None or deal_round is None or closed_check[0] != deal_round or abs(closed_check[1] - deal_price) > 1e-9:
            raise SystemExit(f"logged deal {deal_price} in round {deal_round} does not follow from the messages ({closed_check})")
    elif closed_check is not None:
        raise SystemExit("the messages close a deal but the log says TIMEOUT")
    rb = (pmax - deal_price) / zone if outcome == "DEAL" else None
    rs = (deal_price - pmin) / zone if outcome == "DEAL" else None
    d = disc(deal_round) if outcome != "TIMEOUT" else None
    fail_now = fail(deal_round if outcome == "DEAL_OUT_OF_ZONE" else R)
    # do these weights and discounts reproduce the logged scores? (then the tiles and the footnote may quote them)
    if outcome == "DEAL":
        weights_ok = (abs(d * (Dw + rb * Ww + Ew) - bscore) < 1e-6 and abs(d * (Dw + rs * Ww + Ew) - sscore) < 1e-6
                      and abs(d * (Dw + 4 * rb * rs * Ww + Ew) - gscore) < 1e-6)
    else:
        weights_ok = abs(fail_now - bscore) < 1e-6 and abs(fail_now - sscore) < 1e-6
    logged_gap = num(end.get("min_gap", metrics.get("min_gap", V.get("min_gap"))))
    min_gap = logged_gap if logged_gap is not None else (min(round_gaps) if round_gaps else None)
    first_b = next((num(q["price"]) for q in msgs if q["seat"] == bseat and q.get("price") is not None), None)
    took_first = (outcome == "DEAL" and deal_i is not None and steps[deal_i]["kind"] == "accept"
                  and sum(1 for q in msgs if q["seat"] == bseat and q.get("price") is not None) == 1 and deal_price == first_b)
    seller_no_counter = (deal_i is not None and steps[deal_i]["kind"] == "accept"
                         and not any(q["seat"] == sseat and q.get("price") is not None for q in msgs[:deal_i]))

    # public reference prices: drawn when on the scale of the negotiation, else only named in the zone note
    stated = [v for v in (num(q.get("price")) for q in msgs) if v is not None]
    core = [v for v in [pmin, pmax, ask0] + stated if v is not None]
    lo_c, hi_c = min(core), max(core)
    span = max(hi_c - lo_c, zone or 0, 1e-9)
    refs_drawn, ref_notes = [], []
    for r in public_references(product):
        v = r["value"]
        if v is None:
            ref_notes.append(r["text"]); continue
        on_scale = lo_c - span / 2 <= v <= hi_c + span / 2
        per = r["per_unit"] or (not on_scale and r["qty"] and lo_c - span / 2 <= v * r["qty"] <= hi_c + span / 2)
        lab = r["label"].lower()
        if per:
            ref_notes.append(f"{lab} {money(v)}" + ("" if re.search(r"unit|each|per", lab) else " per unit"))
        else:
            ref_notes.append(f"{'public ' if 'public' not in lab else ''}{lab} {money(v)}")
            if on_scale:
                refs_drawn.append((r["label"], v))

    # ---------------- text
    if outcome == "DEAL":
        title = f"{pname}: deal at {money(deal_price)} in round {deal_round} of {R}"
    elif outcome == "DEAL_OUT_OF_ZONE":
        title = f"{pname}: agreed at {money(deal_price)}, outside the bargaining zone"
    else:
        title = f"{pname}: no deal after {R} rounds"
    speed = "×0.99 for every round played" if code else "×0.99 for every round after the first"
    same = "the same" if tol <= 0 else f"within {money(tol)} of each other"
    lede = (f"A buyer and a seller negotiate the total price of one product for up to {R} rounds: each round the buyer sends a "
            "short message with one price (or none) and the seller replies. Each side's limit (the buyer's maximum, the seller's "
            f"minimum) is private, and a deal closes when, after the seller's reply, the two latest prices are {same}"
            + (" (or the seller's price is at or below the buyer's)" if cross else "") + ". Each side scores for closing "
            f"a deal, for its share of the bargaining zone between the two limits, and for speed ({speed}).")
    zone_tile = {"k": "Bargaining zone", "v": f"{money(pmin)}–{money(pmax)}",
                 "s": "Seller's minimum to buyer's maximum, both private" + (f"; {', '.join(ref_notes)}" if ref_notes else "")}

    def breakdown(share):
        """A side's deal score as its parts: (discount ×) deal bonus + zone-share points + efficiency bonus."""
        parts = f"{Dw:g} deal bonus + {Ww * share:.2f} for {pct(share)} of the zone + {Ew:g} efficiency bonus"
        return parts if abs(d - 1) < 1e-12 else f"{d:.3f} × ({parts})"

    if outcome == "DEAL":
        when = ("no late-deal discount" if abs(d - 1) < 1e-12 else f"late-deal discount ×{d:.3f}") if weights_ok else "scores as logged"
        how = f"Round {deal_round} of {R}, {when}" + ("; the seller took the buyer's opening offer" if took_first else "")
        tiles = [{"k": "Deal price", "v": money(deal_price), "s": how},
                 {"k": "Buyer score", "v": f"{bscore:.2f}",
                  "s": (breakdown(rb) if weights_ok else f"{pct(rb)} of the zone") + f"; reward {brew:.3f}"},
                 {"k": "Seller score", "v": f"{sscore:.2f}",
                  "s": (breakdown(rs) if weights_ok else f"{pct(rs)} of the zone") + f"; reward {srew:.3f}"}, zone_tile]
    elif outcome == "DEAL_OUT_OF_ZONE":
        tiles = [{"k": "Agreed price", "v": money(deal_price), "s": f"Round {deal_round}; outside the zone, so scored as a failure"},
                 {"k": "Scores", "v": f"{fx(bscore)} each", "s": f"Failure penalty; rewards {fx(brew, 4)} and {fx(srew, 4)}"}, zone_tile]
    else:
        tiles = [{"k": "Outcome", "v": "No deal", "s": f"{R} rounds without matching prices"},
                 {"k": "Scores", "v": f"{fx(bscore)} each", "s": f"Failure penalty; rewards {fx(brew, 4)} and {fx(srew, 4)}"},
                 {"k": "Closest gap", "v": money(min_gap) if min_gap is not None else "—",
                  "s": "Smallest gap between the two latest prices at the end of a round, after the seller's reply"},
                 zone_tile]

    src_bits = [f"seed {run.get('seed')}" if run.get("seed") is not None else None,
                f"task {inst.get('task')}, category {inst.get('category')}" if inst.get("task") else None,
                f"env commit {run.get('env_commit')}" if run.get("env_commit") else None]
    date = str(run.get("recorded_at") or DEFAULT_DATE)[:10]
    source = "the rendered replay" if B.get("from_md") else "game_log.jsonl"
    if log_prompts or log_thinking:             # what the model-call log supplied to the steps
        got = " and ".join(w for w, k in (("prompt", log_prompts), ("reasoning", log_thinking)) if k)
        source += (f" and the model-call log (each step's {got})" if all(k in (0, n) for k in (log_prompts, log_thinking))
                   else f" and the model-call log (the {got.replace('prompt', 'prompts')} it saved)")
    foot = [f"Source: {source} of the {date} recording ({', '.join(x for x in src_bits if x)}); {model} plays both sides. "
            "On the axis, R1 B is the buyer's message in round 1 and R1 S the seller's reply; Setup and Result at the two ends "
            "hold the hidden setup and the outcome."]
    if not any_thinking:
        foot.append("This recording was made before reasoning was saved; the game will be re-recorded with reasoning.")
    if rebuilt:
        if saved == [0]:
            foot.append("The observation of the first step is the prompt the buyer was sent; the " + (
                "other one is rebuilt from the game log, because its prompt was not saved." if n == 2 else
                "others are rebuilt from the game log, because those prompts were not saved."))
        elif not saved:
            foot.append("Observations are rebuilt from the game log, because the prompts were not saved.")
        else:
            foot.append(f"{rebuilt} of {n} observations are rebuilt from the game log, because their prompts were not saved.")
    t_exp = "t" if code else "(t−1)"
    fail_txt = (f"−{Fw:g} × (1 − {gam:g}^rounds played)" if code else f"−{Fw:g} × (1 − {gam:g}^{R - 1}) = {fx(fail(R))}")
    if weights_ok and outcome == "DEAL":
        foot.append(("Scores follow the official code's round accounting (code_scoring)" if code else
                     "Scores follow the paper's Algorithm 1 (Liu, Gu and Song 2026)") +
                    f": a deal inside the zone in round t gives a side {gam:g}^{t_exp} × ({Dw:g} + {Ww:g} × its share of the zone "
                    f"+ {Ew:g}) points, a timeout or a deal outside the zone {fail_txt}; reward = points ÷ {top:g}. "
                    f"The joint GlobalScore of this game is {gscore:.2f}.")
    elif weights_ok:
        foot.append(f"A timeout or a deal outside the zone scores {fail_txt} points for each side"
                    + (f", {fx(bscore)} here (the official code's round accounting, code_scoring)" if code else " (the paper's Algorithm 1)")
                    + f"; reward = points ÷ {top:g}.")
    if outcome == "DEAL" and bscore > sscore:
        foot.append("In the paper sellers usually out-score buyers (Table 4); here " +
                    ("the seller accepted the buyer's first offer without ever naming a price of its own, so the buyer came out ahead."
                     if took_first and seller_no_counter else "the buyer came out ahead."))

    # ---------------- charts
    pad = lambda v: [None] + v + [None]                  # the Setup and Result slots carry no message
    # reference lines; lines at the same price are merged into one ("Buyer max = initial ask") so none hides another
    refs = [(nm, v, col, desc) for nm, v, col, desc in (
        ("Buyer max", pmax, "c1", "the buyer's maximum"), ("Seller min", pmin, "c2", "the seller's minimum"),
        ("Initial ask", ask0, "hold", "the seller's initial ask")) if v is not None]
    n_private = len(refs)
    refs += [(lab, v, "muted", f"the public {lab.lower()}") for lab, v in refs_drawn]
    groups = []
    for r in refs:
        g = next((g for g in groups if abs(g[0][1] - r[1]) < 1e-9), None)
        (g.append(r) if g else groups.append([r]))
    lines = []
    for g in groups:
        name = " = ".join([g[0][0]] + [r[0][0].lower() + r[0][1:] for r in g[1:]])
        lines.append({"name": name, "color": g[0][2], "v": [g[0][1]] * X, "dash": True, "thin": True, "end": True})
    merged = [g for g in groups if len(g) > 1]
    # the deal: a segment at the agreed price from the closing message to the Result slot, drawn under the two sides'
    # lines so the closing price stays visible on top of it (a dot there would cover it, and the Result slot is the edge)
    if outcome != "TIMEOUT":
        lines.append({"name": "Deal" if outcome == "DEAL" else "Agreed, outside the zone", "color": "c3" if outcome == "DEAL" else "c4",
                      "v": [None] * (deal_i + 1) + [deal_price] * (X - deal_i - 1), "end": True})
    side_end = outcome == "TIMEOUT"                      # without a deal, label where each side stopped
    lines += [{"name": name, "color": col, "v": pad(v), "step": True, **({"end": True} if side_end else {})}
              for name, col, v in (("Buyer", "c1", bline), ("Seller", "c2", sline)) if any(x is not None for x in v)]
    and_join = lambda xs: ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else "".join(xs)
    priv, pub = [r[3] for r in refs[:n_private]], [r[3] for r in refs[n_private:]]
    cap = lambda s: s[:1].upper() + s[1:]
    price_sub = ("In dollars. Solid steps: each side's latest stated price after every message. Dashed: "
                 + and_join(priv) + (", each known only to its owner" if priv else "") + (", and " + and_join(pub) if pub else "") + "."
                 + "".join(f" {cap(' and '.join(r[3] for r in g))} are both {money(g[0][1])}, drawn as one line." for g in merged)
                 + (f" {'Green' if outcome == 'DEAL' else 'Yellow'}: the agreed price, from the message that closed the deal "
                    "to the Result column." if outcome != "TIMEOUT" else ""))
    allv = [v for ln in lines for v in ln["v"] if v is not None]
    fmt = "int" if all(float(v).is_integer() for v in allv) else "price"
    charts = [{"title": "Prices on the table against the private limits", "sub": price_sub, "height": 250, "fmt": fmt,
               "lines": lines}]

    # what each standing price is worth in points; only when the prices moved (one price repeated says nothing new) and
    # when the weights and discounts above reproduce the logged scores
    if zone and weights_ok and len({round(v, 6) for v in stated}) >= 2:
        own_b = [deal_value(bline[i], rounds[i])[0] for i in range(n)]
        acc_b = [deal_value(sline[i], rounds[i])[0] for i in range(n)]
        own_s = [deal_value(sline[i], rounds[i])[1] for i in range(n)]
        acc_s = [deal_value(bline[i], rounds[i])[1] for i in range(n)]
        r4 = lambda v: [None if x is None else round(x, 4) for x in v]
        # after a deal the solid lines run on to the Result slot at the logged final scores (the value of the deal price in the
        # deal round, so the same number) and carry the end labels
        fin = {"Buyer at own price": bscore, "Seller at own price": sscore} if outcome == "DEAL" else {}
        vl = []
        for nm, col, v, dash in (("Buyer at own price", "c1", own_b, False), ("Buyer if accepting", "c1", acc_b, True),
                                 ("Seller at own price", "c2", own_s, False), ("Seller if accepting", "c2", acc_s, True)):
            if not any(x is not None for x in v):
                continue
            ln = {"name": nm, "color": col, "v": pad(r4(v)), "step": True}
            if dash:
                ln["dash"] = True
            if nm in fin:
                ln["v"][-1] = round(fin[nm], 4); ln["end"] = True
            vl.append(ln)
        vl.insert(0, {"name": "No deal", "color": "hold", "v": [round(fail(R), 4)] * X, "dash": True, "thin": True, "end": True})
        charts.append({"title": "What a deal at each standing price is worth, in points",
                       "sub": f"Points (of {top:g}) a side would score if a deal closed in the current round at its own latest price "
                              "(solid) or at the other side's latest price, by accepting it (dashed). A price outside the zone "
                              f"would score as a failure; gray: the no-deal score, {fx(fail(R))}."
                              + (" The solid lines end in the Result column at the logged final scores." if outcome == "DEAL" else ""),
                       "height": 230, "fmt": "price", "lines": vl})

    # ---------------- feed: hidden setup, the messages, the result
    setup_txt = (f"Task {inst.get('task')} ({inst.get('category')}): {pname}"
                 + (f", {', '.join(ref_notes)}" if ref_notes else "")
                 + f". Buyer's maximum {money(pmax)}; seller's minimum {money(pmin)} and initial asking price {money(ask0)}; "
                   f"zone {money(zone)}." + (f" Context: {', '.join(str(v) for v in context.values())}." if context else ""))
    if outcome == "DEAL":
        parts = lambda share: ((f" ({Dw:g} + {Ww * share:.2f} + {Ew:g})" if abs(d - 1) < 1e-12 else
                                f" ({d:.3f} × ({Dw:g} + {Ww * share:.2f} + {Ew:g}))") if weights_ok else "")
        res_txt = (f"Deal at {money(deal_price)} in round {deal_round} after {n} messages. BuyerScore {bscore:.2f}{parts(rb)}, "
                   f"SellerScore {sscore:.2f}{parts(rs)}, GlobalScore {gscore:.2f}; rewards {brew:.3f} (buyer) and {srew:.3f} (seller).")
    elif outcome == "DEAL_OUT_OF_ZONE":
        res_txt = f"Agreed at {money(deal_price)} in round {deal_round}, outside the zone: both score {fx(bscore)}."
    else:
        res_txt = (f"No deal after {R} rounds: both score {fx(bscore)}."
                   + (f" Closest gap at the end of a round: {money(min_gap)}." if min_gap is not None else ""))
    feed = ([{"x": 0, "step": 0, "who": "Hidden setup", "text": setup_txt}] + feed
            + [{"x": X - 1, "step": n - 1, "who": "Result", "text": res_txt}])

    out = {"id": GID, "name": "AgenticPay", "title": title, "lede": lede, "seatNames": seat_names, "tiles": tiles,
           "foot": " ".join(foot), "x": {"n": X, "labels": labels}, "charts": charts,
           "kinds": {k: KINDS[k] for k in KINDS if k in used}, "steps": steps, "feed": feed,
           "feedTitle": "Negotiation transcript",
           "feedSub": "Every message with the price it states. The first and last lines are the hidden setup and the result, "
                      "which the players never see."}
    if not any_thinking:
        out["eyebrow"] = f"social_game_env · {GID} · {model}, reasoning not saved · recorded {date}"
    return out


def record_date(src):
    """The recording date (run.json recorded_at), for common.write."""
    run = load_source(src)[0]["run"] or {}
    return str(run.get("recorded_at") or DEFAULT_DATE)[:10]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    calls_arg = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else None
    data = build(sys.argv[1], calls_arg)
    common.write(data, sys.argv[2], date=record_date(sys.argv[1]))
