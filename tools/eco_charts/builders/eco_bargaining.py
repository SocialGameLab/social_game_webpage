"""Replay data for eco_bargaining (bilateral price bargaining: a buyer with a private budget, a seller with a private cost).

usage: python3 -I eco_bargaining.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE    a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, transcript.md), or a rendered
            Markdown replay of one (one "- **event**: `{json}`" line per logged event, optionally followed by the model calls).
  OUT_DIR   where <OUT_DIR>/eco_bargaining.json is written (via common.write).
  CALLS     optional model-call log (JSON lines: messages or prompt, reasoning, answer). Each call is tied to the step whose
            seat and move it matches, in time order; its reasoning becomes the step's thinking and its prompt the observation.

Every number shown comes from the game log. Rewards follow the game's rule (rules/eco_bargaining_rules.md, Reward):
buyer = clip((B - P) / |B - C|, -1, 1), seller = clip((P - C) / |B - C|, -1, 1) for a deal at P; 0 for both after a quit
or a timeout; -1 for the buyer (0 for the seller) after a buyer offer above the budget (OVERSHOOT)."""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_bargaining"
DEFAULT_DATE = "2026-11-09"
# Each side's private instructions, as set at SETUP (rules, "Differences From the Source" / Prompts). The buyer's text is
# replaced by the one in the saved prompt when the bundle has it.
INSTRUCTIONS = {
    "BUYER": "You can only buy at a price within your budget; otherwise, you should quit negotiating. Your goal is a deal at "
             "a price as low as possible: spend as little money as possible, not just stay within your budget.",
    "SELLER": "You may not ask or accept a price below your cost; otherwise, you should quit negotiating. Your goal is a deal "
              "at a price as high as possible.",
}
KINDS = {"bid": ["Buyer bid", "c1"], "ask": ["Seller ask", "c2"], "wait": ["Wait, no new offer", "muted"],
         "deal": ["Accept (deal)", "c3"], "quit": ["Quit", "c4"], "other": ["Other", "hold"]}
ROLE_NAME = {"BUYER": "Buyer", "SELLER": "Seller"}
ROLE_COLOR = {"BUYER": "c1", "SELLER": "c2"}
MINUS = "\u2212"
# The paper's mean buyer reward per seller persona, (trained buyer, untrained base buyer): rules, Phenomenon (Liu et al.
# 2026, Tables 7 and 9-11; 128 held-out products x 4 rollouts each).
PAPER = {"default": (0.766, 0.013), "begging": (0.695, -0.017), "insulting": (0.499, -0.158), "unyielding": (0.117, -0.369)}
INSTANCE_SET = {"train": "training instance", "heldout": "held-out instance"}
# An offer this close to a private-limit line, in pixels, covers it: the line dots have a radius of 3.5 px plus a
# 1.5 px stroke. The page floors the price chart's y axis at 0 ("bars": [] and no negative value) and adds 10% headroom
# above the top value, and the plot is the chart height minus 40 px of margins (240 px of 280).
PRICE_H = 280
NEAR_PX = 5.0


# ------------------------------------------------------------------------------------------------ loading
def _md_bundle(path):
    """A rendered Markdown replay (render_replays*.py, declarative games): events from the '- **event**...`{json}`' lines,
    seat rewards / seed / model from the header, and the model calls listed after the events, if any."""
    text = open(path, encoding="utf-8").read()
    events = []
    for m in re.finditer(r"^- \*\*([A-Za-z_]+)\*\*([^`\n]*)`(\{.*\})`\s*$", text, re.M):
        ev, tag, raw = m.group(1), m.group(2), m.group(3)
        try:
            body = json.loads(raw)
        except ValueError:              # JSON cut at 1,500 characters: keep the fields a step needs
            body = {}
            for k, pat in (("move", r'"move": "(\w+)"'), ("role", r'"role": "(\w+)"'), ("price", r'"price": (-?[\d.]+)'),
                           ("round", r'"round": (\d+)'), ("seat", r'"seat": (\d+)'), ("talk", r'"talk": "(.*)')):
                q = re.search(pat, raw)
                if q:
                    v = q.group(1)
                    body[k] = float(v) if k == "price" else int(v) if k in ("round", "seat") else v
        body["event"] = ev
        if "\u9690\u85cf" in tag:       # the renderer's "hidden information" tag
            body["_visibility"] = "hidden"
        events.append(body)
    head = text.split("\n## ")[0]
    rw = re.search(r'(\{"0":[^}]*\})', head)
    seed = re.search(r"\u79cd\u5b50 (\d+)", head)
    model = re.search(r"\u6a21\u578b (\S+)", head)
    over = next((e for e in events if e.get("event") == "game_over"), {})
    setup = next((e for e in events if e.get("event") == "setup"), {})
    summary = {"rewards": json.loads(rw.group(1)) if rw else {}, "roles": setup.get("roles", {}), "variables": over.get("variables", {})}
    run = {"seed": int(seed.group(1)) if seed else None, "model": model.group(1) if model else ""}
    calls = []
    for blk in re.split(r"\n(?=### )", text)[1:]:
        hdr = blk.split("\n", 1)[0]
        ans = re.findall(r"^\*\*[^*\n]+\*\* `(.*)`\s*$", blk, re.M)
        th = re.findall(r"````text\n(.*?)\n````", blk, re.S)
        if not ans:
            continue
        calls.append({"seat": common.seat_of(hdr), "prompt": "", "answer": (ans[-1] if len(ans) > 1 else ans[0]).strip(),
                      "reasoning": th[-1] if th else "", "rescued": len(ans) > 1})
    return {"events": events, "summary": summary, "run": run, "config": {}, "transcript": "", "md_calls": calls}


def load_source(src):
    if os.path.isdir(src):
        b = common.load_bundle(src)
        meta = os.path.join(src, "meta.json")
        b["meta"] = json.load(open(meta, encoding="utf-8")) if os.path.exists(meta) else {}
        return b
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(src)), "game_log.jsonl")):
        return load_source(os.path.dirname(os.path.abspath(src)))
    b = _md_bundle(src)
    b["meta"] = {}
    return b


# ------------------------------------------------------------------------------------------------ helpers
def money(v):
    return (MINUS if v < 0 else "") + f"${abs(v):,.2f}"


def num(v):
    return f"{v:,.2f}"


def short(v):
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"


def rew(v):
    return (MINUS if v < -5e-4 else "") + f"{abs(v):.3f}"


def clip(v):
    return max(-1.0, min(1.0, v))


def r2(v):
    return None if v is None else round(v, 2) + 0.0


def rounds_phrase(rs):
    rs = sorted(set(rs))
    if len(rs) == 1:
        return f"round {rs[0]}"
    if len(rs) > 2 and rs == list(range(rs[0], rs[-1] + 1)):
        return f"rounds {rs[0]} to {rs[-1]}"
    return "rounds " + ", ".join(str(r) for r in rs[:-1]) + " and " + str(rs[-1])


def rew_range(vals):
    vs = sorted(set(round(v, 3) + 0.0 for v in vals))
    return rew(vs[0]) if len(vs) == 1 else f"{rew(vs[0])} to {rew(vs[-1])}"


def rew_capped(v):
    """A reward as the rule pays it (clipped to [-1, 1]), saying so when the clip changed it."""
    if v > 1 + 1e-9:
        return f"{rew(1.0)} (capped at 1)"
    if v < -1 - 1e-9:
        return f"{rew(-1.0)} (capped at {MINUS}1)"
    return rew(v)


def and_list(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _norm_answer(s):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (s or "").strip())
    try:
        return json.loads(s)
    except ValueError:
        return None


def _matches(call, st):
    a = _norm_answer(call.get("answer"))
    if not isinstance(a, dict) or a.get("action_id") != st["action_id"]:
        return False
    p = a.get("payload") or {}
    if st["price"] is not None and st["action_id"] in ("buy", "sell", "deal_at"):
        try:
            if abs(float((p.get("offer") or {}).get("price")) - float(st["price"])) > 1e-9:
                return False
        except (TypeError, ValueError):
            return False
    return True


def attach_calls(steps, calls):
    """Tie calls to steps in time order. For each step, the window is the run of calls from the pointer whose seat is the
    step's seat (or unknown); the first call in it whose answer is the step's move is the one; earlier calls in the window
    were failed attempts (format retries) and their reasoning is kept above it. With no matching answer the last call of
    the window is used and the step notes that the engine applied a different move.
    A call whose answer is one of the next two steps' moves (and not this step's) belongs to that later step, so the
    window ends before it. Without this, a call log with no seats would let one replaced move (say a seller ask below
    cost, played as a wait) swallow every later call. Sets st["_prompt"] when the state comes from the saved prompt."""
    ptr = 0
    fits = lambda k, s: calls[k].get("seat") in (None, s["seat"])
    for i, st in enumerate(steps):
        j = ptr
        while j < len(calls) and fits(j, st):
            j += 1
        for nx in steps[i + 1:i + 3]:
            cut = next((k for k in range(ptr + 1, j) if fits(k, nx) and _matches(calls[k], nx) and not _matches(calls[k], st)), None)
            if cut is not None:
                j = cut
        window = list(range(ptr, j))
        if not window:
            continue
        hit = next((k for k in window if _matches(calls[k], st)), None)
        use = hit if hit is not None else window[-1]
        parts = []
        for k in range(ptr, use):
            c = calls[k]
            parts.append(f"[Earlier attempt, answer: {c['answer'][:300]}]\n{c.get('reasoning') or '(no reasoning returned)'}")
        c = calls[use]
        if parts:
            parts.append("[Final attempt]\n" + (c.get("reasoning") or "(no reasoning returned)"))
        else:
            parts.append(c.get("reasoning") or "")
        if hit is None:
            parts.append(f"[The engine applied a different move than this answer: {c['answer'][:300]}]")
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        st["thinking"] = "\n\n".join(p for p in parts if p).strip()
        if c.get("prompt"):
            st["state"] = common.state_part(c["prompt"])
            st["_prompt"] = True
        a = _norm_answer(c.get("answer"))
        if hit is not None and isinstance(a, dict):
            st["act"] = json.dumps(a, ensure_ascii=False)
        ptr = use + 1


# ------------------------------------------------------------------------------------------------ build
def build(bundle_dir, calls=None):
    b = load_source(bundle_dir)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    if not calls:
        calls = b.get("md_calls") or []
    ev = b["events"]
    summ, run, cfg = b["summary"], b["run"], b["config"]
    inst = next((e for e in ev if e.get("event") == "instance"), {})
    end = next((e for e in ev if e.get("event") == "game_end"), {})
    setup = next((e for e in ev if e.get("event") == "setup"), {})
    var = summ.get("variables") or next((e for e in ev if e.get("event") == "game_over"), {}).get("variables", {})
    roles = {int(k): v for k, v in (setup.get("roles") or summ.get("roles") or {}).items()}
    if not roles and inst:
        roles = {inst["buyer_seat"]: "BUYER", inst["seller_seat"]: "SELLER"}
    seat_of_role = {v: k for k, v in roles.items()}
    n_seats = max(roles) + 1 if roles else 2
    seat_names = [ROLE_NAME.get(roles.get(s, ""), f"Seat {s}") for s in range(n_seats)]

    B, C, L = float(inst["budget"]), float(inst["cost"]), float(inst.get("list_price", var.get("list_price")))
    code = inst.get("codename", var.get("codename", "the product"))
    R = int(var.get("max_rounds") or (cfg.get("variables") or {}).get("max_rounds") or 6)
    gap = abs(B - C)
    mi = B > C
    persona_id = inst.get("persona") or (cfg.get("variables") or {}).get("seller_persona_id") or "default"
    persona_txt = (cfg.get("variables") or {}).get("seller_persona") or ""
    buyer_instr = INSTRUCTIONS["BUYER"]
    first_prompt = (run.get("prompt") or {}) if isinstance(run.get("prompt"), dict) else {}
    pm = re.search(r"Your private info: (\{.*\})", first_prompt.get("text") or "")
    if pm:
        try:
            buyer_instr = json.loads(pm.group(1)).get("instructions") or buyer_instr
        except ValueError:
            pass

    # ---- messages (one model decision each), in log order
    msgs = []
    for e in ev:
        kind_ev = e.get("event")
        if e.get("_visibility") == "hidden" or "seat" not in e or kind_ev in ("setup", "instance", "game_end", "game_over"):
            continue
        role = e.get("role") or roles.get(e["seat"], "")
        if kind_ev == "offer":
            mv = e.get("move") or ("buy" if role == "BUYER" else "sell")
            kind = "bid" if mv == "buy" else "ask"
        elif kind_ev == "await":
            mv, kind = "await", "wait"
        elif kind_ev in ("deal", "deal_at"):
            mv, kind = kind_ev, "deal"
        elif kind_ev == "quit":
            mv, kind = "quit", "quit"
        else:
            mv, kind = kind_ev, "other"
        price = e.get("price")
        msgs.append({"seat": int(e["seat"]), "role": role, "round": int(e.get("round") or 1), "kind": kind, "action_id": mv,
                     "price": None if price is None else float(price), "raw_price": price, "talk": e.get("talk") or ""})
    if not msgs:
        raise SystemExit("no messages in the log")

    outcome = end.get("outcome") or var.get("outcome") or ""
    deal_price = end.get("deal_price", var.get("deal_price"))
    deal_price = None if deal_price is None else float(deal_price)
    rewards = summ.get("rewards") or {}
    r_buy = end.get("buyer_reward", rewards.get(str(seat_of_role.get("BUYER"))))
    r_sel = end.get("seller_reward", rewards.get(str(seat_of_role.get("SELLER"))))
    # ---- the shared x axis: one column per round, or, when the game lasted at most two rounds, one column per message
    # (a round axis would leave one or two columns: single-point charts). A one-message game (a buyer who quits or
    # overshoots at once) gets a leading "Start" column, so the private limits still draw as lines.
    n_rounds = max(m["round"] for m in msgs)
    per_msg = n_rounds <= 2
    if per_msg:
        lead = 1 if len(msgs) == 1 else 0
        labels = (["Start"] if lead else []) + [f"R{m['round']} {ROLE_NAME.get(m['role'], 'Seat %d' % m['seat'])}" for m in msgs]
        for k, m in enumerate(msgs):
            m["col"] = k + lead
    else:
        labels = [f"Round {i + 1}" for i in range(n_rounds)]
        for m in msgs:
            m["col"] = m["round"] - 1
    n = len(labels)
    col_round = {m["col"]: m["round"] for m in msgs}

    # ---- prices on the table per column, concessions, and the state each player saw
    cur = {"BUYER": None, "SELLER": None}          # price on the table: latest offer, or the price accepted
    offer = {"BUYER": None, "SELLER": None}        # latest own offer (a bid or an ask), never an acceptance
    first = {"BUYER": None, "SELLER": None}
    line = {"BUYER": [None] * n, "SELLER": [None] * n}
    offer_line = {"BUYER": [None] * n, "SELLER": [None] * n}
    conc = {"BUYER": [None] * n, "SELLER": [None] * n}
    spoke = {"BUYER": set(), "SELLER": set()}
    deal_line = [None] * n
    waits, history, steps, feed = [], [], [], []
    deal_round = deal_by = None

    def say(m):
        p = m["price"]
        if m["kind"] == "bid":
            return f"bid {money(p)}"
        if m["kind"] == "ask":
            return f"ask {money(p)}"
        if m["kind"] == "wait":
            return "wait, no new offer"
        if m["kind"] == "deal":
            return f"accept {money(p)} (deal)" if p is not None else "accept (deal)"
        if m["kind"] == "quit":
            return "quit"
        return m["action_id"]

    def observation(m):
        role = m["role"]
        other = "SELLER" if role == "BUYER" else "BUYER"
        last_b, last_s = cur["BUYER"], cur["SELLER"]
        standing = cur[other]
        lines = ["(Rebuilt from the game log; the exact prompt for this turn was not saved.)",
                 f"You are seat P{m['seat']}. Your role: {role}.",
                 (f"Your private info: budget {money(B)}." if role == "BUYER"
                  else f"Your private info: cost {money(C)}; persona: {persona_id}" + (f" ({persona_txt})" if persona_txt else " (no persona text)") + "."),
                 "Your instructions: " + (buyer_instr if role == "BUYER" else INSTRUCTIONS["SELLER"]),
                 f"Current phase: {role}_TURN, round {m['round']} of {R}.",
                 f"Product {code}, list price {money(L)}.",
                 f"Last buyer offer: {money(last_b) if last_b is not None else 'none'}. Last seller offer: {money(last_s) if last_s is not None else 'none'}. "
                 f"Standing offer (the price \"deal\" would close at): {money(standing) if standing is not None else 'none'}.",
                 "Messages so far:" if history else "Messages so far: none."]
        lines += history
        return "\n".join(lines)

    first_seen = False
    for k, m in enumerate(msgs):
        role, x = m["role"], m["col"]
        m["standing"] = cur.get("SELLER" if role == "BUYER" else "BUYER")     # what "deal" would have closed at
        state, from_prompt = observation(m), False
        if (not first_seen and first_prompt.get("seat") == m["seat"] and first_prompt.get("phase") == f"{role}_TURN"
                and re.search(r'"round": %d\b' % m["round"], first_prompt.get("text") or "")):
            state, from_prompt = common.state_part(first_prompt["text"]), True   # the saved prompt of the game's first decision
        first_seen = True
        prev_own = cur.get(role)
        if m["kind"] in ("bid", "ask") and m["price"] is not None:
            cur[role] = offer[role] = m["price"]
            if first[role] is None:
                first[role] = m["price"]
        elif m["kind"] == "deal":
            if m["price"] is None:
                m["price"] = deal_price if deal_price is not None else cur["SELLER" if role == "BUYER" else "BUYER"]
            cur[role] = m["price"]
            deal_line[x] = r2(m["price"])
            deal_round, deal_by = m["round"], role
        elif m["kind"] == "wait":
            waits.append((m["round"], role, cur.get(role)))
        spoke.setdefault(role, set()).add(x)
        if role in line:
            if prev_own is not None and cur[role] is not None and m["kind"] in ("bid", "ask", "deal", "wait"):
                d = cur[role] - prev_own
                conc[role][x] = r2(d if role == "BUYER" else -d)
        # head, act, feed
        p = m["price"]
        if m["kind"] == "bid":
            head = f"Round {m['round']} \u00b7 bids {money(p)}"
            text = f"Bids {money(p)}."
            payload = {"talk": m["talk"], "offer": {"price": m["raw_price"]}}
        elif m["kind"] == "ask":
            head = f"Round {m['round']} \u00b7 asks {money(p)}"
            text = f"Asks {money(p)}."
            payload = {"talk": m["talk"], "offer": {"price": m["raw_price"]}}
        elif m["kind"] == "wait":
            own = cur.get(role)
            what = "bid" if role == "BUYER" else "ask"
            head = f"Round {m['round']} \u00b7 waits" + (f"; its {what} of {money(own)} stands" if own is not None else "")
            text = "Waits, no new offer" + (f" (its {what} of {money(own)} still stands)." if own is not None else ".")
            payload = {"talk": m["talk"]}
        elif m["kind"] == "deal":
            head = f"Round {m['round']} \u00b7 accepts {money(p)}"
            text = f"Accepts the {'seller' if role == 'BUYER' else 'buyer'}'s {money(p)}: deal."
            payload = {"talk": m["talk"]} if m["action_id"] == "deal" else {"talk": m["talk"], "offer": {"price": m["raw_price"]}}
        elif m["kind"] == "quit":
            head = f"Round {m['round']} \u00b7 quits"
            text = "Quits: the negotiation ends with no deal."
            payload = {"talk": m["talk"]}
        else:
            head = f"Round {m['round']} \u00b7 {m['action_id']}"
            text = m["action_id"]
            payload = {"talk": m["talk"]}
        talk = m["talk"].strip()
        feed.append({"x": x, "step": k, "who": ROLE_NAME.get(role, f"Seat {m['seat']}"), "kind": m["kind"],
                     "text": text + (f" \u201c{talk}\u201d" if talk else " (no talk)")})
        steps.append({"seat": m["seat"], "x": x, "head": head, "kind": m["kind"],
                      "act": json.dumps({"action_id": m["action_id"], "payload": payload}, ensure_ascii=False),
                      "state": state, "thinking": "", "action_id": m["action_id"], "price": m["raw_price"], "_prompt": from_prompt})
        history.append(f"- Round {m['round']}, {ROLE_NAME.get(role, role)}: {say(m)}." + (f" Talk: \"{talk}\"" if talk else ""))
        # the price on the table in this column (the last message of the column wins: end of round, or this message)
        for rl in ("BUYER", "SELLER"):
            line[rl][x] = r2(cur[rl])
            offer_line[rl][x] = r2(offer[rl])
    # carry the table forward over columns where a side was silent (game ended before its turn)
    for rl in ("BUYER", "SELLER"):
        for i in range(1, n):
            if line[rl][i] is None and line[rl][i - 1] is not None and i not in spoke[rl]:
                line[rl][i] = line[rl][i - 1]

    attach_calls(steps, calls)
    prompted = [bool(st.get("_prompt")) for st in steps]     # the observation is a saved prompt, not rebuilt
    n_prompt = sum(prompted)
    for st in steps:
        st.pop("action_id", None); st.pop("price", None); st.pop("_prompt", None)

    # ---- outcome text
    last = msgs[-1]
    overshoot = outcome == "OVERSHOOT"
    final_round_label = "in the final round" if (deal_round == R) else f"in round {deal_round}"
    if outcome == "DEAL" and deal_price is not None:
        P = deal_price
        if mi and P <= B:
            title = f"Deal at {money(P)} {final_round_label}, {money(B - P)} under the buyer's private budget"
        else:
            title = f"Deal at {money(P)} {final_round_label}, {money(P - B)} above the buyer's budget"
    elif outcome in ("BUYER_QUIT", "SELLER_QUIT"):
        who = "buyer" if outcome == "BUYER_QUIT" else "seller"
        title = f"No deal: the {who} walks away in round {last['round']}" + (" (no price suited both sides)" if not mi else "")
    elif outcome == "TIMEOUT":
        title = f"No deal after {R} rounds"
    elif overshoot:
        title = f"The buyer bids {money(last['price'])}, above its {money(B)} budget, and loses"
    else:
        title = f"Bargaining over {code}: {outcome or 'unfinished'}"

    # ---- tiles
    bid_path = [m["price"] for m in msgs if m["kind"] == "bid" and m["price"] is not None]
    tiles = []
    if outcome == "DEAL" and deal_price is not None:
        by = "seller" if deal_by == "SELLER" else "buyer"
        whose = "buyer's bid" if deal_by == "SELLER" else "seller's ask"
        last_msg = deal_round == R and deal_by == "SELLER"
        tiles.append({"k": "Deal price", "v": money(deal_price),
                      "s": f"The {by} accepted the {whose} in round {deal_round} of {R}" + (", the last message allowed" if last_msg else "")
                           + (". Line: the buyer's bids." if len(bid_path) > 1 else "."),
                      **({"spark": bid_path} if len(bid_path) > 1 else {})})
    else:
        lab = {"BUYER_QUIT": "The buyer quit", "SELLER_QUIT": "The seller quit", "OVERSHOOT": "The buyer bid above its budget"}
        tiles.append({"k": "Outcome", "v": "No deal", "s": f"All {R} rounds passed without a deal" if outcome == "TIMEOUT"
                      else f"{lab.get(outcome, outcome or 'Unfinished')} in round {last['round']} of {R}"})
    if r_buy is not None:
        if outcome == "DEAL" and deal_price is not None and mi:
            s = (f"Kept {money(B - deal_price)} of the {money(gap)} between its budget and the seller's cost" if deal_price <= B
                 else f"Paid {money(deal_price - B)} more than its budget")
        elif overshoot:
            s = "A bid above the budget ends the game with " + MINUS + "1"
        elif outcome == "DEAL":
            s = "No price suited both sides, so any deal costs the buyer " + MINUS + "1"
        else:
            s = "No deal: 0 for both sides"
        tiles.append({"k": "Buyer reward", "v": rew(float(r_buy)), "s": s})
    if r_sel is not None:
        if outcome == "DEAL" and deal_price is not None:
            s = f"Sold {money(deal_price - C)} above its cost of {money(C)}"
        else:
            s = "No deal: the seller scores 0"
        tiles.append({"k": "Seller reward", "v": rew(float(r_sel)), "s": s})
    if first["BUYER"] is not None:
        ratio = first["BUYER"] / B
        tiles.append({"k": "Buyer's opening bid", "v": money(first["BUYER"]),
                      "s": f"{ratio * 100:.1f}% of its {money(B)} budget (the paper's first-offer ratio, {ratio:.3f})"})
    tiles = tiles[:4]

    # ---- charts
    wait_note = " ".join(f"In round {r} the {ROLE_NAME[rl].lower()} waited, so its {'bid' if rl == 'BUYER' else 'ask'} of {money(v)} stood."
                         for r, rl, v in waits if v is not None and rl in ROLE_NAME)
    # An offer within a few pixels of the other side's private limit draws on top of that limit's dashed line and hides
    # it (seed 42, 2026-11-09: the seller's 550.00 ask, 0.35 under the 550.35 budget; 2026-11-11: also its 540.00 ask,
    # 10.35 under, about 3 px on the rendered axis). Mark those columns with yellow dots and say in one sentence which
    # side of the limit each offer was on and what accepting it would have paid.
    top = max([abs(L), abs(B), abs(C)] + [abs(m["price"]) for m in msgs if m["price"] is not None])
    near = NEAR_PX / (PRICE_H - 40) * 1.1 * top          # NEAR_PX on the page's y axis (0 to 1.1 x top over the plot)
    deal_col = next((m["col"] for m in msgs if m["kind"] == "deal"), None)
    hidden_lines, hidden_notes = [], []
    for rl, ref, ref_name in (("SELLER", B, "budget"), ("BUYER", C, "cost")):
        cols = [i for i in range(n) if i != deal_col and offer_line[rl][i] is not None and line[rl][i] == offer_line[rl][i]
                and abs(offer_line[rl][i] - ref) <= near]
        if not cols:
            continue
        prices = [offer_line[rl][i] for i in cols]
        uniq = sorted(set(prices), key=prices.index)                                # in order of first appearance
        inside = {p: (ref - p) if rl == "SELLER" else (p - ref) for p in uniq}     # >= 0: the other side could accept it
        who, what = ("seller", "ask") if rl == "SELLER" else ("buyer", "bid")
        in_w, out_w = ("under", "above") if rl == "SELLER" else ("above", "under")
        rr = [col_round[i] for i in cols]
        one = len(uniq) == 1
        them = "it" if one else "them"

        def rel(p):
            d = inside[p]
            return ("exactly on", "") if abs(d) < 0.005 else (money(abs(d)), in_w if d > 0 else out_w)
        if one:
            amt, w = rel(uniq[0])
            s = f"Yellow dots: the {who}'s {money(uniq[0])} {what}, {(amt + ' ' + w).strip()} the {ref_name}, hides its dashed line"
        else:                                    # the dots show the rounds; the note names the prices in order
            rels = [rel(p) for p in uniq]
            words = {w for _, w in rels}
            offs = (and_list(a for a, _ in rels) + " " + rels[0][1] if len(words) == 1 and "" not in words
                    else and_list((a + " " + w).strip() for a, w in rels))
            s = f"Yellow dots: the {who}'s {and_list(money(p) for p in uniq)} {what}s, {offs} the {ref_name}, hide its dashed line"
        if rl == "SELLER" and all(v >= 0 for v in inside.values()) and mi:
            flagged = set(uniq)
            took = next((m for m in msgs if m["role"] == "BUYER" and m["kind"] == "deal" and m["price"] is not None), None)
            missed = [m for m in msgs if m["role"] == "BUYER" and m["kind"] != "deal" and m["standing"] is not None
                      and r2(m["standing"]) in flagged]
            if missed:
                seen = sorted(set(r2(m["standing"]) for m in missed), key=[r2(m["standing"]) for m in missed].index)
                ref_ask = "it" if one else money(seen[0]) if len(seen) == 1 else "them"
                s += (f"; accepting {ref_ask} in {rounds_phrase([m['round'] for m in missed])} would have paid the buyer "
                      + rew_range([clip((B - m["standing"]) / gap) for m in missed]))
            if took is not None:
                p_t, r_t = took["price"], took["round"]
                lead = ", and it" if missed else "; the buyer"
                r_b = rew(float(r_buy)) if missed else f"a reward of {rew(float(r_buy))}"   # "paid the buyer X ..., and it ... for Y"
                if r2(p_t) in flagged:
                    s += f"{lead} accepted {'that ask' if one else money(p_t)} in round {r_t} for {r_b}"
                else:
                    # the buyer took an ask that is not marked (farther from the budget): name it, never "it"
                    k_t = next(k for k, m in enumerate(msgs) if m is took)
                    src = next((m for m in reversed(msgs[:k_t])
                                if m["role"] == "SELLER" and m["kind"] == "ask" and r2(m["price"]) == r2(p_t)), None)
                    later = "later " if src is not None and src["round"] > max(rr) else ""
                    s += f"{lead} accepted the seller's {later}ask of {money(p_t)} in round {r_t} instead, for {r_b}"
            elif missed:
                if missed[-1]["kind"] == "quit":
                    s += ", but it quit instead (0 for both)"
                elif outcome == "DEAL" and deal_price is not None:
                    s += f", but it kept bargaining and closed at {money(deal_price)} instead, for {rew(float(r_buy))}"
                elif overshoot:
                    s += f", but it kept bargaining and then bid above its budget ({MINUS}1)"
                else:
                    s += ", but it kept bargaining, and the game ended with no deal (0 for both)"
        elif rl == "SELLER" and all(v < 0 for v in inside.values()) and gap > 0:
            s += f"; accepting {them} would have paid the buyer {rew_range([clip((B - p) / gap) for p in uniq])}"
        elif rl == "BUYER" and all(v >= 0 for v in inside.values()) and gap > 0:
            s += f"; the seller could have accepted {them} for a reward of {rew_range([clip((p - C) / gap) for p in uniq])}"
        elif rl == "BUYER" and all(v < 0 for v in inside.values()):
            s += f"; below its cost, the seller could not accept {them}"
        hidden_notes.append(s + ".")
        # named for why the dots are drawn (the offer covers the dashed line), not for which side of the limit it is on,
        # which the note says: a name like "Ask within the budget" would also fit unmarked asks farther inside
        hidden_lines.append({"name": f"{what.capitalize()} on the {ref_name} line", "color": "c4",
                             "v": [line[rl][i] if i in cols else None for i in range(n)], "dots": True, "noline": True})

    # The private limits are named in the legend and valued in the subtitle; no end labels, which crowded the last
    # column's dots. Drawing order: limits, the two sides, the yellow marks, the deal dot on top.
    price_lines = [
        {"name": "List price", "color": "hold", "dash": True, "thin": True, "v": [round(L, 2)] * n},
        {"name": "Buyer's budget", "color": "c1", "dash": True, "thin": True, "v": [round(B, 2)] * n},
        {"name": "Seller's cost", "color": "c2", "dash": True, "thin": True, "v": [round(C, 2)] * n},
    ]
    for rl in ("BUYER", "SELLER"):
        if any(v is not None for v in line[rl]):
            price_lines.append({"name": ROLE_NAME[rl], "color": ROLE_COLOR[rl], "v": line[rl], "dots": True,
                                **({"step": True} if per_msg else {})})
    price_lines += hidden_lines
    if any(v is not None for v in deal_line):
        price_lines.append({"name": "Deal", "color": "c3", "v": deal_line, "dots": True, "noline": True})
    spoke_any = any(line[rl][i] is not None for rl in line for i in range(n))
    chart_price = {
        "title": "Bids and asks against the two private limits",
        "sub": (("Each solid line is that side's price on the table " + ("after each message" if per_msg else "at the end of the round")
                 + ": its latest offer, or the price it accepted. ") if spoke_any else "Neither side named a price. ")
               + (wait_note + " " if wait_note else "")
               + f"Dashed lines: the public list price ({money(L)}), the buyer's budget ({money(B)}) and the seller's cost ({money(C)})"
               + ("; the budget is below the cost, so no price suits both sides." if not mi else ".")
               + (" Green dot: the deal." if any(v is not None for v in deal_line) else "")
               + "".join(" " + h for h in hidden_notes),
        # "bars": [] lays the shared x axis out in bands (one column per round or message), the same as the bar chart, the
        # message grid and the decision timeline, so the columns line up across all panels
        "height": PRICE_H, "fmt": "price2", "lines": price_lines, "bars": []}

    def moved(rl):
        f, l_ = first[rl], line[rl][-1]
        if f is None or l_ is None:
            return None
        return f, l_
    totals = []
    for rl in ("BUYER", "SELLER"):
        mv = moved(rl)
        if mv and any(v is not None for v in conc[rl]):        # a side with a single price has no total move to report
            d = (mv[1] - mv[0]) if rl == "BUYER" else (mv[0] - mv[1])
            totals.append(f"{ROLE_NAME[rl].lower()} {num(d)} ({num(mv[0])} to {num(mv[1])})")
    conc_bars = [{"name": ROLE_NAME[rl], "color": ROLE_COLOR[rl], "v": conc[rl]} for rl in ("BUYER", "SELLER") if any(v is not None for v in conc[rl])]
    charts = [chart_price]
    if sum(v is not None for bb in conc_bars for v in bb["v"]) >= 2:     # one bar alone is not worth a chart
        neg = any(v is not None and v < 0 for bb in conc_bars for v in bb["v"])
        charts.append({
            "title": "Concessions per message" if per_msg else "Concessions per round",
            "sub": "How far each side moved from its own previous price, in dollars: the buyer's raise and the seller's cut (0 = held its price"
                   + ("; below 0 = moved away from the other side" if neg else "") + ")."
                   + (" In total: " + ", ".join(totals) + "." if totals else ""),
            "height": 200, "fmt": "price2", "bars": conc_bars, "lines": [], **({"zero": True} if neg else {})})

    def if_accepted(m):
        p = m["price"]
        if m["kind"] == "bid":
            if p > B:
                return f"bid {num(p)}: above the budget, the game ends with buyer {MINUS}1"
            if p < C:
                return f"bid {num(p)}: below the seller's cost, it may not accept"
        if m["kind"] in ("bid", "ask") and gap > 0:
            return f"{m['kind']} {num(p)}; if accepted: buyer {rew_capped((B - p) / gap)}, seller {rew_capped((p - C) / gap)}"
        if m["kind"] == "deal" and p is not None and gap > 0:
            return f"accepts {num(p)}: buyer {rew_capped((B - p) / gap)}, seller {rew_capped((p - C) / gap)}"
        if m["kind"] == "wait":
            return "waits, no new offer"
        return m["kind"]
    # rows in seat order, the same order as the decision timeline's lanes and the seat buttons
    rows = [roles.get(s, "") for s in range(n_seats)]
    cells = [[None] * n for _ in rows]
    for m in msgs:
        if m["role"] not in rows:
            continue
        t = short(m["price"]) if m["kind"] in ("bid", "ask") else {"wait": "wait", "deal": "deal", "quit": "quit"}.get(m["kind"], m["kind"])
        cells[rows.index(m["role"])][m["col"]] = {"c": KINDS[m["kind"]][1], "t": t, "tip": if_accepted(m)}
    used = []
    for m in msgs:
        if m["kind"] not in used:
            used.append(m["kind"])
    charts.append({
        "title": "Messages one by one" if per_msg else "Messages round by round",
        "sub": "One cell per message: the price offered, or the move. Hover a cell for what that price would have paid each side if it had been accepted.",
        "height": 90, "grid": {"rows": seat_names, "cells": cells, "rowHeight": 30,
                               "legend": [[KINDS[k][0], KINDS[k][1]] for k in used]}})

    # ---- result line in the transcript (shown only at the last step)
    if outcome == "DEAL" and deal_price is not None:
        res = f"Deal at {money(deal_price)} in round {deal_round}. Buyer reward {rew(float(r_buy))}, seller reward {rew(float(r_sel))}."
        if mi and deal_price <= B:
            res += f" The buyer kept {money(B - deal_price)} of the {money(gap)} between its budget and the seller's cost; the seller took {money(deal_price - C)}."
    else:
        res = f"No deal ({outcome.replace('_', ' ').lower()}). Buyer reward {rew(float(r_buy or 0))}, seller reward {rew(float(r_sel or 0))}."
    feed.append({"x": msgs[-1]["col"], "step": len(steps) - 1, "who": "Result", "text": res})

    # ---- text
    model = (b.get("meta") or {}).get("model") or run.get("model") or ""
    model = model.split("/")[-1] if model else "the model"
    date = ((b.get("meta") or {}).get("recorded_at") or run.get("recorded_at") or DEFAULT_DATE)[:10]
    seed = run.get("seed", (cfg or {}).get("seed"))
    has_reason = any(s["thinking"] for s in steps)
    n_think = sum(1 for s in steps if s["thinking"])
    first_who = ROLE_NAME.get(msgs[0]["role"], "first player").lower()
    if n_prompt == len(steps):
        obs = "every observation is the saved prompt."
    elif n_prompt == 0:
        obs = "every observation is rebuilt from the game log."
    elif n_prompt == 1 and prompted[0]:
        obs = f"the observation of the {first_who}'s first message is the saved prompt; the later ones are rebuilt from the log."
    else:
        obs = f"the observations of {n_prompt} of {len(steps)} messages are the saved prompts; the others are rebuilt from the log and say so."
    if not has_reason:
        prov = "This recording was made before reasoning was saved; the game will be re-recorded with reasoning. " + obs[0].upper() + obs[1:] + " "
    elif n_prompt == len(steps):
        prov = "Observations and reasoning come from the saved model calls. "
    else:
        prov = ("Reasoning comes from the saved model calls" + (f" ({n_think} of {len(steps)} messages)" if n_think < len(steps) else "")
                + ". " + obs[0].upper() + obs[1:] + " ")
    persona_phrase = "" if persona_id == "default" else f", against a seller told to play the {persona_id} persona"
    lede = (f"A buyer and a seller haggle over one product ({code}, list price {money(L)}) for at most {R} rounds{persona_phrase}. "
            "The buyer speaks first, and each message is a short public talk text plus one move: offer a price, accept the other side's latest offer, wait, or quit. "
            f"Only the buyer knows its budget ({money(B)}) and only the seller its cost ({money(C)}); each side's score is the share of the gap between budget and cost that it captures.")
    inst_set = inst.get("instance_set") or ""
    inst_word = INSTANCE_SET.get(inst_set, f"{inst_set} instance" if inst_set else "instance")
    paper = PAPER.get(persona_id)
    if paper:
        trained, base = paper
        cite = (f"Game and reward from Liu et al. (2026). Against the {persona_id} seller they report a mean buyer reward of "
                f"{rew(base)} for their untrained base model and {rew(trained)} after RL training: means over 512 games "
                "(128 held-out products, 4 rollouts each) with their own buyer and seller models, so this one game cannot be compared "
                "with them")
        if r_buy is not None and rew(float(r_buy)) == rew(base):
            cite += f"; this game's buyer reward of {rew(float(r_buy))} equals their base-model mean only by chance"
        cite += "."
    else:
        cite = "Game and reward from Liu et al. (2026)."
    foot = (f"Data: the {date} recording of eco_bargaining (seed {seed}, {persona_id} seller persona, {inst_word} {code}, "
            + ("a mutual-interest instance: budget above cost" if mi else "a conflict instance: budget below cost, so walking away is right")
            + f"), both sides played by {model}; every number about this game is computed from its game log. "
            + prov
            + ("Columns: one per message, since the game lasted " + ("one round" if n_rounds == 1 else "two rounds") + ". " if per_msg else "")
            + "Rewards: buyer = (budget " + MINUS + " price) \u00f7 (budget " + MINUS + " cost), seller = (price " + MINUS + " cost) \u00f7 (budget "
            + MINUS + " cost), each clipped to [" + MINUS + f"1, 1]; a quit or {R} rounds without a deal give both 0, and a buyer bid above the budget gives the buyer "
            + MINUS + "1. " + cite)
    d = {"id": GID, "name": "Bargaining", "title": title, "lede": lede, "seatNames": seat_names, "tiles": tiles, "foot": foot,
         "x": {"n": n, "labels": labels}, "charts": charts, "kinds": {k: KINDS[k] for k in used}, "steps": steps,
         "feed": feed, "feedTitle": "Negotiation transcript",
         "feedSub": "Every message both sides saw: the move, the price and the talk text. The selected step is highlighted; later messages are hidden."}
    d["eyebrow"] = (common.EYEBROW.format(gid=GID, date=date) if has_reason
                    else f"social_game_env \u00b7 {GID} \u00b7 {model} \u00b7 recorded {date} \u00b7 reasoning not saved")
    return d


def recorded_date(src):
    b = load_source(src)
    return ((b.get("meta") or {}).get("recorded_at") or b["run"].get("recorded_at") or DEFAULT_DATE)[:10]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    common.write(build(src, calls), out, date=recorded_date(src))
