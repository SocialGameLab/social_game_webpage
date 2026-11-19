"""Replay data for eco_gossip_reciprocity (the repeated donation game with public gossip of Zhu et al., ICML 2026, "Talk, Judge,
Cooperate": each round one donor and one recipient who never meet again; the donor cooperates (pays cost, recipient gains
benefit) or defects; the recipient then publishes a toned message about the donor that every seat can read).

usage: python3 -I eco_gossip_reciprocity.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json), a file inside one, or the
          rendered Markdown replay of one (one "- **event** ... `{json}`" line per logged event).
  OUT_DIR where <OUT_DIR>/eco_gossip_reciprocity.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). Calls are tied to
          decisions in time order: a call belongs to the next decision of the same seat and the same phase (donation or
          message; see attach_calls), and the call whose answer is the logged move is that decision's call (earlier ones were
          format retries; their reasoning is kept above it). Its reasoning becomes the step's thinking and its prompt the
          observation; the step's `concl` (optional field) is the end of that reasoning without the closing checks of the answer
          format (see conclusion()), for the page's 'Conclusion of its reasoning' box once the page reads it.
       python3 -I eco_gossip_reciprocity.py --selftest SOURCE
          runs the regression test of the call matching on synthetic call logs over SOURCE.

The x axis is the rounds (R1 .. R36 with 9 seats). Each round has two decisions: the donor's donate move and, with gossip on,
the recipient's message (or 0/1 signal). The logged moves are rebuilt exactly from the log: the donation event gives the choice,
the donor's private reflection event its justification, the public gossip event the message, the hidden gossip_tone event the
tone, the recipient's reflection event its justification. Every number shown is computed from the log; the metrics follow
rules/eco_gossip_reciprocity_rules.md (Metrics): cooperation rate, image score, net payoff, payoff per round, the discounted
return indexed by each seat's own interactions, gossip accuracy."""
import json, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_gossip_reciprocity"
FEED_TITLE = "Donations and public messages"
MINUS, TIMES, DIV, ARROW, DOTS = "\u2212", "\u00d7", "\u00f7", "\u2192", "\u2026"
NEG_TONES = ("mocking", "complaint", "criticism")
TONES = ("praising", "neutral") + NEG_TONES
KINDS = {"cooperate": ["Cooperates", "c3"], "defect": ["Defects", "c2"],
         "praising": ["Praising", "c1"], "neutral": ["Neutral", "hold"], "negative": ["Negative tone", "c4"], "silent": ["No message", "muted"],
         "signal1": ["Signal 1", "c1"], "signal0": ["Signal 0", "c4"]}


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_PUNCT = {"\u3002": ". ", "\u3001": ", ", "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"', "\u3010": "[", "\u3011": "]",
          "\u300a": '"', "\u300b": '"', "\u3008": "<", "\u3009": ">", "\u301c": "~"}


def en(s):
    """The page shows English only: full-width forms become ASCII, CJK punctuation its ASCII counterpart, and any remaining run of
    Chinese, Japanese or Korean characters (a model may write or reason in Chinese) a visible marker."""
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
def _md_bundle(path):
    """A rendered Markdown replay (declarative games): events from the '- **event** (tag): `{json}`' lines (the tag marks private
    and hidden events), seat rewards, seed and model from the header, and the model calls listed after the events ('### '
    blocks), if any."""
    text = open(path, encoding="utf-8").read()
    events = []
    for m in re.finditer(r"^- \*\*([A-Za-z_]+)\*\*([^`\n]*)`(\{.*\})`\s*$", text, re.M):
        try:
            body = json.loads(m.group(3))
        except ValueError:
            continue
        body["event"] = m.group(1)
        if "\u9690\u85cf" in m.group(2):            # "hidden"
            body["_visibility"] = "hidden"
        elif "\u79c1\u5bc6" in m.group(2):          # "private"
            body["_visibility"] = "private"
        events.append(body)
    head = text.split("\n## ")[0].split("\n- **")[0]
    rw = re.search(r'(\{"0":[^}]*\})', head)
    seed = re.search(r"\u79cd\u5b50 (\d+)", head)    # "seed N"
    model = re.search(r"\u6a21\u578b (\S+)", head)   # "model X"
    over = next((e for e in events if e.get("event") == "game_over"), {})
    setup = next((e for e in events if e.get("event") == "setup"), {})
    summary = {"rewards": json.loads(rw.group(1)) if rw else {}, "roles": setup.get("roles", {}), "variables": over.get("variables", {})}
    run = {"seed": int(seed.group(1)) if seed else None, "model": model.group(1) if model else ""}
    calls = []
    for blk in re.split(r"\n(?=### )", text)[1:]:
        hdr = blk.split("\n", 1)[0]
        ans = re.findall(r"^\*\*[^*\n]+\*\* `(.*)`\s*$", blk, re.M)
        th = re.findall(r"````text\n(.*?)\n````", blk, re.S)
        if ans:
            calls.append({"seat": common.seat_of(hdr), "prompt": "", "answer": (ans[-1] if len(ans) > 1 else ans[0]).strip(),
                          "reasoning": th[-1] if th else "", "rescued": len(ans) > 1})
    return {"events": events, "summary": summary, "run": run, "config": {}, "transcript": "", "meta": {}, "md_calls": calls}


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
    """The recording date (YYYY-MM-DD) from meta.json or run.json; None when neither has one (a rendered Markdown replay carries
    no date). Never a guessed default: the page then says the date was not saved."""
    b = load_source(src)
    d = (b.get("meta") or {}).get("recorded_at") or (b.get("run") or {}).get("recorded_at")
    return str(d)[:10] if d else None


# ------------------------------------------------------------------------------------------------ text helpers
def P(s):
    return f"P{s}"


NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def numword(n):
    return NUM[n] if 0 <= n < len(NUM) else str(n)


def num(v, d=2):
    """A signed number with a real minus sign."""
    return (MINUS if v < -0.5 * 10 ** -d else "") + f"{abs(v):.{d}f}"


def plist(seats):
    names = [P(s) for s in seats]
    if not names:
        return "nobody"
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def clip(s, n):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n - 1].rstrip() + DOTS


def tone_kind(tone, message):
    if not (message or "").strip():
        return "silent"
    return tone if tone in ("praising", "neutral") else "negative" if tone in NEG_TONES else "neutral"


def accurate(tone, coop, message):
    """The rules' gossip accuracy: accurate = praising or neutral after a cooperation, or neutral, mocking, complaint or criticism
    after a defection; misleading = praising a defection or mocking, complaining about or criticizing a cooperation; an empty
    message is silent (neither). Returns True, False or None."""
    if not (message or "").strip() or tone not in TONES:
        return None
    if coop:
        return tone in ("praising", "neutral")
    return tone != "praising"


# ------------------------------------------------------------------------------------------------ calls
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
    if st["action_id"] == "donate":
        return p.get("cooperate") is st["cooperate"]
    if st["action_id"] == "gossip":
        return p.get("tone") == st["tone"] and " ".join(str(p.get("message") or "").split()) == " ".join((st["message"] or "").split())
    if st["action_id"] == "signal":
        return p.get("value") == st["value"]
    return False


_PROMPT_PHASE = {"DONATE": "donate", "GOSSIP": "message", "SIGNAL": "message"}
_ANSWER_PHASE = {"donate": "donate", "gossip": "message", "signal": "message"}


def _step_phase(st):
    return "donate" if st["action_id"] == "donate" else "message"


def _call_phase(call):
    """Which decision a call was made for: 'donate' (the donor's choice), 'message' (the recipient's gossip or signal) or None
    when neither the prompt ('Current phase: DONATE' / 'GOSSIP' / 'SIGNAL') nor the answer's action_id says so. The prompt is
    what the engine asked and wins over the answer."""
    m = re.search(r"Current phase:\s*([A-Z_]+)", call.get("prompt") or "")
    if m and m.group(1) in _PROMPT_PHASE:
        return _PROMPT_PHASE[m.group(1)]
    a = _norm_answer(call.get("answer"))
    return _ANSWER_PHASE.get(a.get("action_id")) if isinstance(a, dict) else None


def _call_seat(call):
    s = call.get("seat")
    return s if s is not None else common.seat_of(call.get("prompt"))


# A sentence about the answer's form, not the decision: DeepSeek's reasoning usually ends with checks such as 'Need action_id likely
# "donate". Ensure no markdown.' or with a draft of the JSON. Also bare fillers ('Good.', 'Done.').
_FORM = re.compile(r"json|action[_ ]?id|markdown|payload|exactly one|\bvalid\b|double quotes?|no extra|escape|\boutput|\brespond|\banswer\b|"
                   r"\bcraft|formulat|\bwords?\b|\b150\b|\bmention|\bconcise|\bkeys?\b|legal actions?|sample|\bformat|\bfinal\b|\bfields?\b|"
                   r"boolean|instruction|short explanation|sees (?:the )?justification|starts? with (?:a )?justification|"
                   r"justification (?:short|hidden|unseen|no other|private|not (?:seen|public|shown))|\{\"|\"\}|\":", re.I)
_FILLER = re.compile(r"^(?:good|done|yes|ok(?:ay)?|fine|great|sure|that'?s (?:it|fine|good)|"
                     r"let'?s (?:produce|go|do it|write|finalize|proceed))[.!]?$", re.I)
_SENT = re.compile(r"\S.*?(?:[.!?][\"'\u201d)\]]*(?=\s+[A-Z\"'(\[{\u201c]|\s*$)|$)", re.S)
CONCL_MIN, CONCL_MAX = 80, 700


def _form_only(s):
    s = s.strip()
    return bool(_FORM.search(s) or _FILLER.match(s))


def conclusion(reasoning, justification=""):
    """The step's 'conclusion of its reasoning' (per-step field `concl`; the page's own pick is the last paragraph, which here is
    usually a check of the answer format). The last run of consecutive sentences, within one paragraph, none of which is about the
    answer's form, of at least CONCL_MIN characters (its last CONCL_MAX characters, '...' when cut); the text is the model's own,
    unchanged. When no such run exists, the move's justification, labeled as such. None for an empty reasoning."""
    if not (reasoning or "").strip():
        return None
    for p in reversed([p for p in re.split(r"\n\s*\n", reasoning) if p.strip()]):
        sp = [(m.start(), m.end()) for m in _SENT.finditer(p) if m.group().strip()]
        k = len(sp) - 1
        while k >= 0:
            if _form_only(p[sp[k][0]:sp[k][1]]):
                k -= 1
                continue
            end, i = sp[k][1], k
            while i > 0 and not _form_only(p[sp[i - 1][0]:sp[i - 1][1]]) and end - sp[i - 1][0] <= CONCL_MAX:
                i -= 1
            text = p[sp[i][0]:end].strip()
            cut = i > 0 and not _form_only(p[sp[i - 1][0]:sp[i - 1][1]])
            if len(text) > CONCL_MAX:
                text, cut = text[-CONCL_MAX:], True
            if len(text) >= CONCL_MIN:
                return (DOTS + " " if cut else "") + text
            k = i - 1
    j = " ".join((justification or "").split())
    return f"(The reasoning ends with checks of the answer's format; the move's justification:) {j}" if j else None


def attach_calls(steps, calls):
    """Tie calls to steps in time order. For each step the window is the run of calls from the pointer that belong to this
    decision: same seat (or seat unknown) and same phase (or phase unknown). The window ends at the first call of another seat
    or of the other phase (the hub seat of the official order often gossips in one round and donates in the next, so seat
    alone does not separate two decisions), at the first call whose answer is the logged move (that call is the decision's), and,
    for a call with no seat, at a call that is not this move but is the next decision's logged move. Calls before the matching
    one were failed attempts (format retries); their reasoning is kept above it. With no matching answer the last call of the
    bounded window is used (never a call made for another decision) and the step notes that the engine applied a different
    move (an invalid donation becomes defect, an invalid message the empty neutral one, an invalid signal 0; rules: Actions)."""
    ptr = 0
    for i, st in enumerate(steps):
        ph, nxt = _step_phase(st), (steps[i + 1] if i + 1 < len(steps) else None)
        window, hit, j = [], None, ptr
        while j < len(calls):
            c = calls[j]
            seat, cph = _call_seat(c), _call_phase(c)
            if seat not in (None, st["seat"]) or (cph is not None and cph != ph):
                break
            if _matches(c, st):
                window.append(j)
                hit = j
                break
            if seat is None and nxt is not None and _matches(c, nxt):
                break                                   # not this move but the next decision's: that call is the next one's
            window.append(j)
            j += 1
        if not window:
            continue
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
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        a = _norm_answer(c.get("answer"))
        just = ((a.get("payload") or {}).get("justification") if isinstance(a, dict) and isinstance(a.get("payload"), dict) else None) \
            or (json.loads(st["act"])["payload"].get("justification") if st.get("act") else "")
        concl = conclusion(en(c.get("reasoning") or ""), en(str(just or "")))
        if concl:
            notes = (["[The engine applied a different move than this answer.]"] if hit is None else []) \
                + (["[The reasoning ran out before an answer; the action is the answer to a follow-up question.]"] if c.get("rescued") else [])
            st["concl"] = "\n\n".join([concl] + notes)
        if c.get("prompt"):
            st["state"] = en(common.state_part(c["prompt"]))
            st["state_from_prompt"] = True
        if hit is not None and isinstance(a, dict):
            st["act"] = en(json.dumps(a, ensure_ascii=False))
        ptr = use + 1


# ------------------------------------------------------------------------------------------------ build
def build(bundle_dir, calls=None):
    b = load_source(bundle_dir)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    if not calls:
        calls = b.get("md_calls") or []
    events, summary, run, cfg = b["events"], b.get("summary") or {}, b.get("run") or {}, b.get("config") or {}
    meta = b.get("meta") or {}
    over = next((e for e in events if e.get("event") == "game_over"), {})
    var = dict(over.get("variables") or {})
    var.update({k: v for k, v in (summary.get("variables") or {}).items() if k not in var})
    rules = next((e for e in events if e.get("event") == "rules"), {})
    gv = lambda k, d=None: rules.get(k, var.get(k, (cfg.get("variables") or {}).get(k, d)))
    N = int(gv("n_players", cfg.get("n_players", 9)))
    cost, benefit, endow = gv("cost", 1), gv("benefit", 5), gv("endowment", 10)
    gamma = float(gv("discount", 0.99))
    horizon, mode = gv("horizon", "infinite"), gv("gossip_mode", "open")
    hint = rules.get("equilibrium_hint", var.get("equilibrium_hint_text"))
    trust = rules.get("trust_reminder", var.get("trust_reminder_text"))
    conv = rules.get("signal_convention", var.get("signal_convention_text"))
    rounds_total = rules.get("rounds_total", var.get("rounds_total"))
    maxw = gv("max_message_words", 150)
    roles = {int(k): v for k, v in (summary.get("roles") or next((e for e in events if e.get("event") == "setup"), {}).get("roles", {})).items()}
    greedy = sorted(s for s, r in roles.items() if r == "GREEDY")
    names = [P(s) + (" (greedy)" if s in greedy else "") for s in range(N)]
    reflection_on = any(e.get("event") == "reflection" and e.get("_visibility") == "private" for e in events)

    # ---- one pass over the log: rounds, stocks, the decisions and what each seat knew when it made them
    stock = [endow] * N
    priv = {s: [] for s in range(N)}           # each seat's private records, rendered, in log order
    pool = []                                   # the public message pool, rendered
    rounds, steps, warn = [], [], []
    cur = None                                  # the round in progress
    first_prompt = run.get("prompt") or {}

    def params_line():
        hz = (f"{rounds_total} rounds announced" if rounds_total is not None else "open-ended (no number of rounds announced)")
        gm = {"open": f"open gossip (the recipient publishes a message of at most {maxw} words; its tone is not published)",
              "binary": "binary signals (the recipient publishes 0 or 1)", "none": "no gossip"}.get(mode, f"gossip mode {mode}")
        notes = [n for n, t in (("equilibrium hint", hint), ("signal convention", conv), ("trust reminder", trust)) if t]
        return (f"Game: cost {cost}, benefit {benefit}, endowment {endow}, discount {gamma:g} per own interaction, {hz}, {gm}. "
                + ("Standing notes shown: " + ", ".join(notes) + "." if notes else "No standing notes."))

    def observation(seat, phase, r):
        lines = ["(Rebuilt from the game log; the exact prompt of this turn was not saved.)",
                 f"You are seat {P(seat)}. Your role: {roles.get(seat, 'PLAYER')}. Your stock: {stock[seat]:g}.", params_line()]
        if phase == "DONATE":
            lines.append(f"Current phase: DONATE. You are the donor and {P(r['recipient'])} is the recipient "
                         f"(stocks now: you {r['donor_stock']:g}, {P(r['recipient'])} {r['recipient_stock']:g}). "
                         f"Cooperate pays {cost} so that {P(r['recipient'])} gains {benefit}; defect transfers nothing. "
                         "The current round number is not shown.")
        else:
            what = "a message (tone and text) about" if mode == "open" else "a 0/1 signal about"
            lines.append(f"Current phase: {'GOSSIP' if mode == 'open' else 'SIGNAL'}. You are the recipient; publish {what} your donor "
                         f"{P(r['donor'])}, who {'cooperated' if r['coop'] else 'defected'} this round (see your records).")
        lines.append("Your private records (oldest first):" if priv[seat] else "Your private records: none yet.")
        lines += priv[seat]
        lines.append(f"Public messages ({len(pool)} so far, every seat reads them):" if pool else "Public messages: none yet.")
        lines += pool
        return "\n".join(lines)

    for e in events:
        ev = e.get("event")
        if ev == "pairing":
            d, rc = int(e["donor"]), int(e["recipient"])
            cur = {"donor": d, "recipient": rc, "donor_stock": e.get("donor_stock", stock[d]), "recipient_stock": e.get("recipient_stock", stock[rc]),
                   "coop": None, "justif_d": None, "justif_r": None, "tone": None, "message": None, "signal": None}
            if (cur["donor_stock"], cur["recipient_stock"]) != (stock[d], stock[rc]):
                warn.append(f"pairing {P(d)}->{P(rc)}: logged stocks {cur['donor_stock']}/{cur['recipient_stock']}, tracked {stock[d]}/{stock[rc]}")
            # the donor decides now: it sees this pairing on top of its records
            st = {"seat": d, "kind": None, "action_id": "donate", "state": observation(d, "DONATE", cur), "thinking": "", "rd": cur}
            steps.append(st)
            cur["step_d"] = len(steps) - 1
        elif ev == "donation" and cur is not None:
            r = int(e["round"])
            coop = e.get("action") == "cooperate"
            d, rc = cur["donor"], cur["recipient"]
            before = (stock[d], stock[rc])
            if coop:
                stock[d] -= cost
                stock[rc] += benefit
            if (e.get("donor_stock_after"), e.get("recipient_stock_after")) != (stock[d], stock[rc]):
                warn.append(f"round {r}: logged stocks after {e.get('donor_stock_after')}/{e.get('recipient_stock_after')}, tracked {stock[d]}/{stock[rc]}")
            cur.update(round=r, coop=coop, before=before, after=(stock[d], stock[rc]), share=e.get("cost_share_of_donor_stock"))
            rounds.append(cur)
            st = steps[cur["step_d"]]
            st.update(kind="cooperate" if coop else "defect", cooperate=coop, round=r)
            sh = e.get("cost_share_of_donor_stock")
            shtxt = f" (the cost was {sh * 100:.1f}% of {P(d)}'s stock)" if isinstance(sh, (int, float)) and coop else ""
            priv[d].append(f"- Round {r}: you were the donor for {P(rc)} and {'cooperated: you paid ' + str(cost) if coop else 'defected: nothing moved'}; "
                           f"stocks after: you {stock[d]:g}, {P(rc)} {stock[rc]:g}.")
            priv[rc].append(f"- Round {r}: {P(d)} was your donor and {'cooperated: you received ' + str(benefit) + shtxt if coop else 'defected: you received nothing'}; "
                            f"stocks after: {P(d)} {stock[d]:g}, you {stock[rc]:g}.")
            if mode in ("open", "binary"):
                st = {"seat": rc, "kind": None, "action_id": "gossip" if mode == "open" else "signal", "round": r,
                      "state": observation(rc, "GOSSIP", cur), "thinking": "", "rd": cur}
                steps.append(st)
                cur["step_r"] = len(steps) - 1
        elif ev == "reflection" and cur is not None:
            a = int(e.get("author", -1))
            role = e.get("role")
            j = en(e.get("justification") or "")
            if role == "donor" and a == cur["donor"]:
                cur["justif_d"] = j
            elif role == "recipient" and a == cur["recipient"]:
                cur["justif_r"] = j
            if e.get("_visibility") != "hidden" and 0 <= a < N:
                priv[a].append(f"  Your justification (round {e.get('round')}, as {role}): {j}")
        elif ev == "gossip" and cur is not None:
            cur["message"] = en(e.get("message") or "")
            if int(e.get("donor", -1)) != cur["donor"] or int(e.get("author", -1)) != cur["recipient"]:
                warn.append(f"round {e.get('round')}: gossip by {e.get('author')} about {e.get('donor')} does not match the pairing")
            pool.append(f"- Round {e.get('round')}, {P(e.get('author'))} about {P(e.get('donor'))}: "
                        + (cur["message"] if cur["message"].strip() else "(empty message)"))
        elif ev == "signal" and cur is not None:
            v = e.get("signal", e.get("value"))
            cur["signal"] = int(v) if v is not None else None
            pool.append(f"- Round {e.get('round')}, {P(e.get('author'))} about {P(e.get('donor'))}: signal {v}")
        elif ev == "gossip_tone" and cur is not None:
            cur["tone"] = e.get("tone")
    if not rounds:
        raise SystemExit("no donation in the game log")
    R = len(rounds)

    # ---- the moves, rebuilt exactly
    for r in rounds:
        st = steps[r["step_d"]]
        st["act"] = json.dumps({"action_id": "donate", "payload": {"justification": r["justif_d"] or "", "cooperate": r["coop"]}}, ensure_ascii=False)
        st["head"] = f"Round {r['round']} \u00b7 donor for {P(r['recipient'])} \u00b7 " + ("cooperates" if r["coop"] else "defects")
        if r["justif_d"] is None:
            st["head"] += " (no justification logged)"
        if "step_r" in r:
            st = steps[r["step_r"]]
            if mode == "open":
                k = tone_kind(r["tone"], r["message"])
                st.update(kind=k, tone=r["tone"], message=r["message"])
                st["act"] = json.dumps({"action_id": "gossip", "payload": {"justification": r["justif_r"] or "", "tone": r["tone"], "message": r["message"]}},
                                       ensure_ascii=False)
                st["head"] = f"Round {r['round']} \u00b7 about donor {P(r['donor'])} \u00b7 " + ("no message" if k == "silent" else f"{r['tone']}")
            else:
                v = r["signal"]
                st.update(kind="signal1" if v == 1 else "signal0", value=v)
                st["act"] = json.dumps({"action_id": "signal", "payload": {"justification": r["justif_r"] or "", "value": v}}, ensure_ascii=False)
                st["head"] = f"Round {r['round']} \u00b7 signal about donor {P(r['donor'])} \u00b7 {v}"
            st["x"] = r["round"] - 1
        steps[r["step_d"]]["x"] = r["round"] - 1

    # the saved prompt of the game's first decision replaces the rebuilt observation
    s0 = steps[0]
    if first_prompt.get("seat") == s0["seat"] and first_prompt.get("phase") == "DONATE" and '"message_pool": []' in (first_prompt.get("text") or ""):
        s0["state"] = en(common.state_part(first_prompt["text"]))
        s0["state_from_prompt"] = True

    # ---- metrics (rules: Metrics)
    coops = sum(r["coop"] for r in rounds)
    final = stock[:]
    game_end = next((e for e in events if e.get("event") == "game_end"), {})
    if game_end.get("final_resources") is not None and list(game_end["final_resources"]) != final:
        warn.append(f"final stocks {game_end['final_resources']} logged, {final} tracked")
    image = [0] * N
    hist = {s: [] for s in range(N)}           # per seat, its own interactions in order: (round, role, payoff)
    for r in rounds:
        image[r["donor"]] += 1 if r["coop"] else -1
        hist[r["donor"]].append((r["round"], "donor", -cost if r["coop"] else 0))
        hist[r["recipient"]].append((r["round"], "recipient", benefit if r["coop"] else 0))
    if game_end.get("image_score") is not None and list(game_end["image_score"]) != image:
        warn.append(f"image scores {game_end['image_score']} logged, {image} computed")
    if game_end.get("cooperation_rate") is not None and abs(game_end["cooperation_rate"] - coops / R) > 1e-9:
        warn.append(f"cooperation rate {game_end['cooperation_rate']} logged, {coops / R} computed")
    alternating = all(all(a[1] != b_[1] for a, b_ in zip(hist[s], hist[s][1:])) for s in range(N))
    disc = [sum(gamma ** k * p for k, (_, _, p) in enumerate(hist[s])) for s in range(N)]
    disc_full = [sum(gamma ** k * (-cost if role == "donor" else benefit) for k, (_, role, _) in enumerate(hist[s])) for s in range(N)]
    donor_first = [s for s in range(N) if hist[s] and hist[s][0][1] == "donor"]
    recip_first = [s for s in range(N) if hist[s] and hist[s][0][1] == "recipient"]
    n_donor = {s: sum(1 for h in hist[s] if h[1] == "donor") for s in range(N)}
    n_recip = {s: sum(1 for h in hist[s] if h[1] == "recipient") for s in range(N)}
    norm = benefit * -(-(N - 1) // 2)        # the most any seat can gain (rules: Reward)
    raw_rw = summary.get("rewards") or {}
    rewards = [float(raw_rw.get(str(s), raw_rw.get(s, (final[s] - endow) / norm))) for s in range(N)]
    for s in range(N):
        want = (final[s] - endow) / norm
        if abs(want - rewards[s]) > 1e-9:
            warn.append(f"{P(s)}: reward {rewards[s]} logged, {want} by the rule")
    net = [final[s] - endow for s in range(N)]
    per_round = sum(net) / (2 * R)
    tones = {t: 0 for t in TONES}
    silent = acc = mis = 0
    for r in rounds:
        if mode != "open" or "step_r" not in r:
            continue
        if not (r["message"] or "").strip():
            silent += 1
            continue
        tones[r["tone"]] = tones.get(r["tone"], 0) + 1
        a = accurate(r["tone"], r["coop"], r["message"])
        acc += a is True
        mis += a is False
    sig = {(c, v): 0 for c in (True, False) for v in (0, 1)}
    for r in rounds:
        if mode == "binary" and r["signal"] in (0, 1):
            sig[(r["coop"], r["signal"])] += 1
    for w in warn:
        print("warning:", w, file=sys.stderr)

    # ---- the schedule as it played out: runs of rounds that all involve one seat (the official order meets seat after seat)
    blocks, i = [], 0
    while i < R:
        best = None
        for s in (rounds[i]["donor"], rounds[i]["recipient"]):
            j = i
            while j < R and s in (rounds[j]["donor"], rounds[j]["recipient"]):
                j += 1
            if best is None or j > best[1]:
                best = (s, j)
        blocks.append((best[0], i + 1, best[1]))
        i = best[1]

    # ---- attach the model calls (reasoning, the exact prompt) when given
    attach_calls(steps, calls)
    has_reason = any(s.get("thinking") for s in steps)
    from_prompt = sum(1 for s in steps if s.get("state_from_prompt"))

    # ---- feed: every donation (private to the pair) and every public message, in order
    feed = []
    for k, st in enumerate(steps):
        r = st["rd"]
        if st["action_id"] == "donate":
            d, rc = r["donor"], r["recipient"]
            if r["coop"]:
                txt = (f"Cooperates with {P(rc)}: pays {cost}, {P(rc)} gains {benefit}. Stocks {P(d)} {r['before'][0]:g} {ARROW} {r['after'][0]:g}, "
                       f"{P(rc)} {r['before'][1]:g} {ARROW} {r['after'][1]:g}. Seen only by the two of them.")
            else:
                txt = f"Defects on {P(rc)}: nothing moves (stocks {P(d)} {r['after'][0]:g}, {P(rc)} {r['after'][1]:g}). Seen only by the two of them."
        elif st["action_id"] == "gossip":
            txt = (f"About {P(r['donor'])}, to everyone: \u201c{r['message']}\u201d" if r["message"].strip() else
                   f"Publishes an empty message about {P(r['donor'])}.")
        else:
            txt = f"Publishes signal {r['signal']} about {P(r['donor'])}."
        # the move's written justification in full (the Decision pane cuts long ones): the only stated reason in a recording
        # without saved reasoning; never shown to another seat
        j = r["justif_d"] if st["action_id"] == "donate" else r["justif_r"]
        if j and j.strip():
            txt += f"\nJustification (no other seat saw it): {j.strip()}"
        elif reflection_on:
            txt += "\nNo justification was logged for this move."
        feed.append({"x": st["x"], "step": k, "who": P(st["seat"]), "kind": st["kind"], "text": txt})

    # ---- chart 1: the public record of every seat, round by round (rows: seats, one column per round). The Decisions timeline
    # already shows who acted and how; this board shows what every seat could read about each seat when the round's donor chose.
    used = [k for k in KINDS if any(st["kind"] == k for st in steps)]
    if mode in ("open", "binary"):
        def record(qs):
            """A seat's public record (the earlier rounds in which it was donor): (category, short text)."""
            if mode == "open":
                ks = [tone_kind(q["tone"], q["message"]) for q in qs]
                cat = "neg" if any(k in ("negative", "silent") for k in ks) else "mixed" if "neutral" in ks else "praise"
                toned = [q["tone"] for q in qs if (q["message"] or "").strip()]
                order = list(TONES) + sorted(set(toned) - set(TONES), key=str)
                parts = [f"{toned.count(t)} {t}" for t in order if toned.count(t)]
                ne = len(qs) - len(toned)
                parts += [f"{ne} empty"] if ne else []
                mis = sum(1 for q in qs if accurate(q["tone"], q["coop"], q["message"]) is False)
                return cat, ", ".join(parts) + (f" ({mis} misleading)" if mis else "")
            vals = [q["signal"] for q in qs]
            return ("neg" if any(v != 1 for v in vals) else "praise"), "signals " + ", ".join(str(v) for v in vals)
        about = {s: [] for s in range(N)}          # per seat, the rounds whose message is about it, published so far
        recs = []                                  # per round: {seat: [rounds]} as the donor chose
        for r in rounds:
            recs.append({s: list(about[s]) for s in range(N)})
            if "step_r" in r:
                about[r["donor"]].append(r)
        maxn = max([len(v) for rec in recs for v in rec.values()] + [1])
        cat_color = {"praise": "c1", "mixed": "hold", "neg": "c4"}
        cells = [[None] * R for _ in range(N)]
        cats_used, all_from = set(), None
        for r, rec in zip(rounds, recs):
            x = r["round"] - 1
            d, rc = r["donor"], r["recipient"]
            if all_from is None and all(rec[s] for s in range(N)):
                all_from = r["round"]
            for s in range(N):
                role = (f" \u00b7 donor for {P(rc)}: {'cooperated' if r['coop'] else 'defected'}" if s == d else
                        f" \u00b7 recipient from {P(d)}" if s == rc else "")
                if not rec[s]:
                    cells[s][x] = {"c": "muted", "o": .12, "tip": "nothing published yet" + role}
                    continue
                cat, txt = record(rec[s])
                cats_used.add(cat)
                n = len(rec[s])
                o = round(.3 + .65 * (n - 1) / (maxn - 1), 2) if maxn > 1 else .9
                cells[s][x] = {"c": cat_color[cat], "o": o, "tip": txt + role}
        if mode == "open":
            labels = {"praise": f"Only praise (darker: more messages, 1 to {maxn})", "mixed": "Praise and neutral messages",
                      "neg": "At least one mocking, complaint, criticism or empty message"}
        else:
            labels = {"praise": f"Only signals 1 (darker: more, 1 to {maxn})", "neg": "At least one signal 0"}
        legend = [[labels[k], cat_color[k]] for k in ("praise", "mixed", "neg") if k in cats_used]
        what = "messages" if mode == "open" else "signals"
        # name only the colors that occur (the legend lists the same ones)
        if mode == "open":
            col_txt = {"praise": "blue: only praise", "mixed": "gray: at least one neutral message, none negative or empty",
                       "neg": "yellow: at least one mocking, complaint, criticism or empty message"}
            only, basis = "praise", ", by the tone each author chose (the players read only the text)"
        else:
            col_txt = {"praise": "blue: only signals 1", "neg": "yellow: at least one signal 0"}
            only, basis = "signals 1", ""
        shown = [col_txt[k] for k in ("praise", "mixed", "neg") if k in cats_used]
        if cats_used == {"praise"}:
            colors = f"Every record held {only} only{basis}: blue, darker with more {what} (1 to {maxn}); blank: nothing about the seat yet. "
        elif shown:
            colors = (shown[0][0].upper() + "; ".join(shown)[1:] + f"; darker with more {what} (1 to {maxn}); blank: nothing about the seat yet. "
                      + ("The colors follow the tone each author chose; the players read only the text. " if basis else ""))
        else:
            colors = "Blank: nothing about the seat yet. "
        sub1 = (f"Every seat could read every public {what[:-1]}. One row per seat, one column per round: the {what} about that seat published "
                f"before the round's donor chose. They come from the seats it had been donor for, so a record grows only after the seat has "
                f"been donor. " + colors
                + (f"Every seat had a record from round {all_from} on. " if all_from else "")
                + "Hover a column for every seat's record and the round's pair.")
        title1 = "What everyone could read about each seat" if mode == "open" else "The public signals about each seat"
        charts = [{"title": title1, "sub": sub1, "height": 240,
                   "grid": {"rows": names, "cells": cells, "legend": legend, "rowHeight": 24}, "bars": []}]
    else:
        # no gossip: nothing is public, so the board shows the pairs (the timeline shows only the donors)
        def stock_tip(s, r):
            return f"stock {r['before'][0 if s == r['donor'] else 1]:g} {ARROW} {r['after'][0 if s == r['donor'] else 1]:g}"
        cells = [[None] * R for _ in range(N)]
        for r in rounds:
            x, d, rc = r["round"] - 1, r["donor"], r["recipient"]
            cells[d][x] = {"c": KINDS["cooperate" if r["coop"] else "defect"][1], "t": str(rc),
                           "tip": f"donor for {P(rc)}, {'cooperated' if r['coop'] else 'defected'}; {stock_tip(d, r)}"}
            cells[rc][x] = {"c": KINDS["cooperate" if r["coop"] else "defect"][1], "o": .45, "t": str(d),
                            "tip": f"recipient from {P(d)}, {'got ' + str(benefit) if r['coop'] else 'got nothing'}; {stock_tip(rc, r)}"}
        legend = [["Donor cooperated", "c3"]] + ([["Donor defected", "c2"]] if coops < R else []) + [["Recipient (lighter)", "c3"]]
        sub1 = ("One column per round, one row per seat; nothing is published, so a seat knows only its own pairs. Darker cell: the donor "
                f"(green when it cooperated, paying {cost} so the recipient gained {benefit}" + ("" if coops == R else ", orange when it defected")
                + "); lighter cell: the recipient. Hover a column to see the pair and both stocks.")
        charts = [{"title": "Who gave to whom", "sub": sub1, "height": 240,
                   "grid": {"rows": names, "cells": cells, "legend": legend, "rowHeight": 24}, "bars": []}]

    # ---- chart 2: what the donor could read about its recipient before deciding (the public record, by the author's tone)
    if mode in ("open", "binary"):
        cnt = {k: [0] * R for k in ("praising", "neutral", "negative", "silent", "signal1", "signal0")}
        for r in rounds:
            x = r["round"] - 1
            for q in rounds:
                if q["round"] >= r["round"] or q["donor"] != r["recipient"] or "step_r" not in q:
                    continue
                k = (tone_kind(q["tone"], q["message"]) if mode == "open" else ("signal1" if q["signal"] == 1 else "signal0"))
                cnt[k][x] += 1
        label = {"praising": "Praising messages", "neutral": "Neutral messages", "negative": "Mocking, complaint or criticism",
                 "silent": "Empty messages", "signal1": "Signals 1", "signal0": "Signals 0"}
        bars = [{"name": label[k], "color": KINDS[k][1], "v": cnt[k]} for k in cnt if any(cnt[k])]
        none_known = [r["round"] for r in rounds if not any(cnt[k][r["round"] - 1] for k in cnt)]
        coop_blind = [r["round"] for r in rounds if r["round"] in none_known and r["coop"]]
        if bars:
            sub2 = ("Before choosing, the donor could read every earlier public message. Bars: how many of them were about this round's recipient "
                    "(its row in the chart above; written by the seats that recipient had been donor to), "
                    + ("by the tone their authors chose. " if mode == "open" else "by signal. ")
                    + f"In {len(none_known)} of {R} rounds the recipient had no public record yet"
                    + ((f"; the donor cooperated in all {len(coop_blind)} anyway." if len(coop_blind) == len(none_known) else
                        f"; the donor cooperated in {len(coop_blind)} of them.") if none_known else ".")
                    + f" The most any recipient had: {max(sum(cnt[k][x] for k in cnt) for x in range(R))}.")
            charts.append({"title": "What the donor could read about the recipient", "sub": sub2, "height": 200, "fmt": "int", "bars": bars, "lines": []})

    # ---- chart 3: every seat's stock after each round, and the gap between the highest and the lowest
    path = [[None] * R for _ in range(N)]
    sk = [endow] * N
    for r in rounds:
        if r["coop"]:
            sk[r["donor"]] -= cost
            sk[r["recipient"]] += benefit
        for s in range(N):
            path[s][r["round"] - 1] = sk[s]
    spread = [max(path[s][x] for s in range(N)) - min(path[s][x] for s in range(N)) for x in range(R)]
    hi_x = max(range(R), key=lambda x: spread[x])
    gap = spread[hi_x]
    rich = [s for s in range(N) if path[s][hi_x] == max(path[t][hi_x] for t in range(N))]
    poor = [s for s in range(N) if path[s][hi_x] == min(path[t][hi_x] for t in range(N))]
    same_end = len(set(final)) == 1
    sub3 = (f"Each seat's stock after every round (pick a player above the charts to highlight it; hover for every seat's value). "
            f"A seat loses {cost} each time it cooperates as donor and gains {benefit} each time its donor cooperates. "
            f"Dashed: the gap between the highest and the lowest stock, widest after round {hi_x + 1}: {plist(rich)} at "
            f"{max(path[s][hi_x] for s in range(N)):g}, {plist(poor)} at {min(path[s][hi_x] for s in range(N)):g} ({gap:g} apart)"
            + (", purely a matter of the order of the pairs, since every donor cooperated. " if coops == R else ". ")
            + (f"Every seat ended at {final[0]:g} (gap 0)." if same_end else f"Final stocks range from {min(final):g} to {max(final):g}."))
    lines = [{"name": names[s], "color": "c1", "v": path[s], "step": True} for s in range(N)]
    lines.append({"name": f"Gap, highest {MINUS} lowest", "color": "ink", "v": spread, "dash": True, "step": True, "end": True})
    charts.append({"title": "Stocks round by round", "sub": sub3, "height": 260, "fmt": "int", "lines": lines, "bars": []})

    # ---- the treatment the players were told, and where the paper puts it (rules: Phenomenon)
    paper_cell = N == 9 and abs(gamma - 0.99) < 1e-12 and cost == 1 and benefit == 5 and not greedy and not trust
    main_setting = paper_cell and rounds_total is None and mode == "open" and bool(hint) and reflection_on
    rt = (f"{rounds_total:g}" if isinstance(rounds_total, (int, float)) else str(rounds_total)) if rounds_total is not None else None
    told = ("Players are told the game continues indefinitely" if rt is None else f"Players are told the game lasts exactly {rt} rounds")
    told += f" and to maximize their payoff discounted by {gamma:g} per interaction of their own"
    aids = [t for t, on in (("a note on equilibrium reasoning", hint), ("their own earlier justifications", reflection_on)) if on]
    told += (", with " + " and ".join(aids) + " in view. " if aids else ". ")
    spe = "defection is the only subgame-perfect equilibrium (Proposition 3.2)"
    if main_setting:
        told += ("That is the paper's main setting (Table 2), where its eight models' cooperation rates range from 60% to 100%; "
                 f"with a known last round, {spe} and the paper's reasoning models almost never cooperate. ")
    elif rt is not None:
        told += (f"With a known last round, {spe}"
                 + ({"open": "; in the paper's runs of this setting (Table 5) its reasoning models cooperate in at most 2% of rounds. ",
                     "none": "; in the paper's runs of this setting without gossip (Table 4) seven of its eight models never cooperate. "}
                    .get(mode, ". ") if paper_cell else ". "))
    elif mode == "none" and paper_cell:
        told += "Without gossip (the paper's Table 1) its four reasoning models defect in every round. "
    else:
        told += f"With a known last round, {spe}. "

    # ---- tiles
    cum, c_ = [], 0
    for r in rounds:
        c_ += r["coop"]
        cum.append(c_)
    nd = sorted(set(n_donor.values()))
    tiles = [{"k": "Cooperation", "v": f"{coops} of {R}", "spark": cum,
              "s": f"Donations made ({coops / R * 100:.0f}%, the paper's cooperation rate); each seat was donor "
                   + (f"{nd[0]} times" if len(nd) == 1 else f"{nd[0]} to {nd[-1]} times")
                   + ("; the paper's eight models: 60% to 100% in this setting" if main_setting else "")}]
    if len(set(rewards)) == 1:
        s0_ = 0
        tiles.append({"k": "Reward, every seat", "v": f"{rewards[0]:.3f}",
                      "s": f"Stock {endow} {ARROW} {final[s0_]:g}: got {benefit} {TIMES} {sum(1 for h in hist[s0_] if h[1] == 'recipient' and h[2] > 0)}, "
                           f"paid {cost} {TIMES} {sum(1 for h in hist[s0_] if h[1] == 'donor' and h[2] < 0)}; reward = ({final[s0_]:g} {MINUS} {endow}) {DIV} {benefit * -(-(N - 1) // 2)}"})
    else:
        best = max(range(N), key=lambda s: rewards[s])
        tiles.append({"k": "Rewards", "v": f"{num(min(rewards), 3)} to {num(max(rewards), 3)}",
                      "s": f"Mean {num(sum(rewards) / N, 3)}; best {P(best)} (stock {final[best]:g}); reward = (final stock {MINUS} {endow}) {DIV} {benefit * -(-(N - 1) // 2)}"})
    if mode == "open":
        nmsg = sum(tones.values())
        neg = sum(tones[t] for t in NEG_TONES)
        if nmsg and tones["praising"] == nmsg and not silent:
            v = f"{nmsg} of {nmsg} praising"
        else:
            v = f"{tones['praising']} praising \u00b7 {neg} negative"
        acc_txt = (f"accuracy {acc / (acc + mis) * 100:.0f}%: {acc} messages fit the donor's choice, {mis} misleading" if acc + mis else "no toned message")
        tiles.append({"k": "Gossip", "v": v, "s": f"Tone chosen by the recipient; {acc_txt}"
                      + (f"; {tones['neutral']} neutral" if tones["neutral"] else "") + (f"; {silent} empty" if silent else "")})
    elif mode == "binary":
        tiles.append({"k": "Signals", "v": f"{sig[(True, 1)] + sig[(False, 1)]} ones \u00b7 {sig[(True, 0)] + sig[(False, 0)]} zeros",
                      "s": f"After a cooperation: {sig[(True, 1)]} ones, {sig[(True, 0)]} zeros; after a defection: {sig[(False, 1)]} ones, {sig[(False, 0)]} zeros"})
    mean_disc = sum(disc) / N
    mean_full = sum(disc_full) / N
    paper = N == 9 and R == 36 and abs(gamma - 0.99) < 1e-12 and cost == 1 and benefit == 5
    df_txt = (f"{len(donor_first)} seats that gave first {num(sum(disc[s] for s in donor_first) / len(donor_first))}, "
              f"{len(recip_first)} that received first {num(sum(disc[s] for s in recip_first) / len(recip_first))}"
              if donor_first and recip_first else "")
    tiles.append({"k": "Discounted return", "v": num(mean_disc),
                  "s": f"Mean over seats, discount {gamma:g} per own interaction" + (f" ({df_txt})" if df_txt else "") + "; "
                       + (f"full cooperation on these pairings would give {num(mean_full)}" + (", the paper's Table 2 value" if paper else "")
                          if abs(mean_full - mean_disc) > 5e-3 else "the full-cooperation value" + (" of the paper's Table 2" if paper else ""))})

    # ---- text
    pre = "Told the game is open-ended, " if rt is None else f"Told the game lasts {rt} rounds, "
    if coops == R and mode == "open" and tones["praising"] == R:
        title = f"{pre}every donor cooperates in all {R} rounds, and every message praises the donor"
    elif coops == R:
        title = f"{pre}every donor cooperates in all {R} rounds"
    elif coops == 0:
        title = f"{pre}no donor ever cooperates in the {R} rounds"
    else:
        title = f"{pre}{coops} of {R} donors cooperate and the rest defect"
    model = meta.get("model") or run.get("model") or ""
    model = model.split("/")[-1] if model else "the model"
    date = recorded_date(b)
    if not date:
        print("warning: neither meta.json nor run.json gives recorded_at; the page says the recording date was not saved", file=sys.stderr)
    seed = run.get("seed", cfg.get("seed"))
    nr = -(-(N - 1) // 2)
    lede = (f"{numword(N).capitalize()} seats play the repeated donation game of Zhu et al. (ICML 2026): each round pairs two seats that never "
            f"meet again, and the donor either pays {cost} so the recipient gains {benefit} or defects; only the two of them see the choice"
            + ({"open": ", but the recipient then publishes a message about the donor that every seat reads, so help can be repaid only by "
                        "others, through reputation. ",
                "binary": ", but the recipient then publishes a 0/1 signal about the donor that every seat reads, so help can be repaid only "
                          "by others, through reputation. ",
                "none": ", and nothing is published, so a donor can be repaid only by seats it never helped. "}.get(mode, ". "))
            + told
            + f"Each seat scores (final stock {MINUS} {endow}) {DIV} {benefit * nr}.")
    struct = ""
    if len(blocks) >= 2 and blocks[0][2] - blocks[0][1] >= 2:
        b0, b1 = blocks[0], blocks[1]
        struct = (f"The pairs came seat by seat: {P(b0[0])} met {numword(b0[2] - b0[1] + 1)} others in rounds {b0[1]} to {b0[2]}, {P(b1[0])} "
                  f"{numword(b1[2] - b1[1] + 1)} more in rounds {b1[1]} to {b1[2]}, and so on")
        struct += ("; every seat alternated between donor and recipient. " if alternating else ". ")
    foot = ((f"Data: the {date} recording" if date else "Data: a recording (its date was not saved)") + f" of eco_gossip_reciprocity (seed {seed}, {N} seats, cost {cost}, benefit {benefit}, endowment {endow}, "
            f"discount {gamma:g}, {'open-ended framing' if rounds_total is None else str(rounds_total) + ' rounds announced'}, "
            f"{ {'open': 'open gossip', 'binary': 'binary signals', 'none': 'no gossip'}.get(mode, 'gossip ' + str(mode))}, "
            f"equilibrium hint {'on' if hint else 'off'}, reflection memory {'on' if reflection_on else 'off'}), every seat played by {model}; "
            "every number on this page is computed from its game log. "
            + ("Observations and reasoning come from the saved model calls. " if has_reason else
               "This recording was made before reasoning was saved; the game will be re-recorded with reasoning. ")
            + ("Each move's written justification (part of the move, never shown to other seats) appears with the action and in full "
               f"under the move in the “{FEED_TITLE}” panel. ")
            + ("" if from_prompt == len(steps) else
               ("The first donor's observation is the saved prompt; the others are rebuilt from the log. " if from_prompt else
                "The observations are rebuilt from the log. "))
            + struct
            + "The players never saw the current round number or the tone labels. Metrics as in the paper and its official code: cooperation rate = "
              f"share of donations made; payoff per round = net payoff per seat per interaction ({num(per_round)} here); image score = cooperations "
              f"{MINUS} defections as donor ({', '.join(sorted(set(str(v) for v in image)))} here); discounted return = sum over a seat's own interactions k of "
              f"{gamma:g}^(k{MINUS}1) {TIMES} its payoff in that interaction.")
    for st in steps:
        for k in ("action_id", "cooperate", "tone", "message", "value", "round", "state_from_prompt", "rd"):
            st.pop(k, None)
    d = {"id": GID, "name": "Gossip and reciprocity", "title": title, "lede": lede, "seatNames": names, "tiles": tiles, "foot": foot,
         "x": {"n": R, "labels": [f"R{r['round']}" for r in rounds]}, "charts": charts, "kinds": {k: KINDS[k] for k in KINDS if k in used},
         "steps": steps, "feed": feed, "feedTitle": FEED_TITLE,
         "feedSub": ("Each round: the donor's choice, seen only by the pair, then the recipient's public message, which every seat could read "
                     "before its later decisions. Under each move: its author's justification, which no other seat saw. The colored tag is "
                     "the tone the author chose (hidden from the players). The selected step is highlighted; later entries are hidden.")
                    if mode == "open" else
                    "Each round: the donor's choice, seen only by the pair, then what the recipient published; under each move, its author's "
                    "justification, which no other seat saw. The selected step is highlighted; later entries are hidden."}
    d["eyebrow"] = (common.EYEBROW.format(gid=GID, date=date or "(date not saved)") if has_reason
                    else f"social_game_env \u00b7 {GID} \u00b7 {model} \u00b7 recorded {date or '(date not saved)'} \u00b7 reasoning not saved")
    return d


# ------------------------------------------------------------------------------------------------ regression test of attach_calls
def _selftest(src):
    """Synthetic call logs over a real recording (python3 -I eco_gossip_reciprocity.py --selftest SOURCE): every call carries its
    step number in its reasoning and prompt, so a call tied to the wrong decision shows. Cases: (1) the hub seat's gossip falls
    back (its answer is not the logged message) and the same seat's donation follows; (2) a format retry before a move; (3) the
    same log without seats or prompts (calls as listed in a rendered Markdown replay), with one unparsable and one non-matching
    answer, and one decision with no call at all; (4) a no-gossip game (all decisions are donations), calls without seats: one
    fallback and one missing call, each followed by a different move."""
    import contextlib, copy, io

    def run(bundle, calls):
        with contextlib.redirect_stderr(io.StringIO()):
            return build(bundle, calls)["steps"]

    def make_calls(steps, seat=True, prompt=True):
        out = []
        for i, st in enumerate(steps):
            ph = "DONATE" if json.loads(st["act"])["action_id"] == "donate" else "GOSSIP"
            out.append({"seat": st["seat"] if seat else None, "answer": st["act"], "reasoning": f"reasoning-{i}.", "rescued": False,
                        "prompt": (f"You are seat P{st['seat']}. marker-{i}.\nCurrent phase: {ph} -- test\nLegal actions:\n- test" if prompt else "")})
        return out

    def check(steps, expect, label):
        bad = []
        for i, st in enumerate(steps):
            want = expect.get(i, f"reasoning-{i}.")
            got = st.get("thinking") or ""
            if (want == "" and got) or (want and not got.startswith(want) and f"[Final attempt]\n{want}" not in got):
                bad.append(f"step {i}: want {want!r}, got {got[:60]!r}")
        if bad:
            raise SystemExit(f"selftest {label} FAILED:\n  " + "\n  ".join(bad[:8]))

    bundle = load_source(src)
    base = run(bundle, [])
    hub = next(k for k in range(len(base) - 1) if base[k]["seat"] == base[k + 1]["seat"]
               and json.loads(base[k]["act"])["action_id"] == "gossip" and json.loads(base[k + 1]["act"])["action_id"] == "donate")
    wrong_gossip = json.dumps({"action_id": "gossip", "payload": {"justification": "x", "tone": "praising", "message": "word " * 200}})

    # (1) hub-seat gossip fallback, then the same seat's donation
    calls = make_calls(base)
    calls[hub]["answer"] = wrong_gossip
    st = run(bundle, calls)
    check(st, {}, "hub fallback")
    assert "[The engine applied a different move" in st[hub]["thinking"], "hub: fallback note missing"
    assert f"marker-{hub}." in st[hub]["state"] and f"marker-{hub + 1}." in st[hub + 1]["state"], "hub: wrong observation"
    assert "different move" not in st[hub + 1]["thinking"], "hub: the donation was marked as a fallback"
    assert json.loads(st[hub + 1]["act"])["action_id"] == "donate", "hub: donation step lost its move"

    # (2) a format retry before step 5
    calls = make_calls(base)
    calls.insert(5, dict(calls[5], answer="not json", reasoning="retry-5."))
    st = run(bundle, calls)
    check(st, {}, "retry")
    assert st[5]["thinking"].startswith("[Earlier attempt") and "retry-5." in st[5]["thinking"], "retry: earlier attempt missing"

    # (3) no seats, no prompts: an unparsable answer at step 10, a non-matching message at the hub, no call at all for step 20
    calls = make_calls(base, seat=False, prompt=False)
    calls[10]["answer"] = "I will praise them."
    calls[hub]["answer"] = wrong_gossip
    del calls[20]
    st = run(bundle, calls)
    check(st, {20: ""}, "seatless")
    assert "different move" in st[10]["thinking"] and "different move" in st[hub]["thinking"], "seatless: fallback notes missing"

    # (4) no gossip (donations only), no seats: round 7 defects; step 5 falls back, step 12 has no call
    nb = copy.deepcopy(bundle)
    evs = []
    for e in nb["events"]:
        if e.get("event") in ("gossip", "gossip_tone") or (e.get("event") == "reflection" and e.get("role") == "recipient"):
            continue
        if e.get("event") == "rules":
            e["gossip_mode"] = "none"
        if e.get("event") == "donation" and e.get("round") in (7, 14):
            e["action"] = "defect"
        evs.append(e)
    nb["events"] = evs
    nbase = run(nb, [])
    assert all(json.loads(x["act"])["action_id"] == "donate" for x in nbase) and len(nbase) == len(base) // 2, "no-gossip: steps"
    calls = make_calls(nbase, seat=False, prompt=False)
    calls[5]["answer"] = "cooperate"
    del calls[12]
    st = run(nb, calls)
    check(st, {12: ""}, "no gossip")
    assert "different move" in st[5]["thinking"], "no gossip: fallback note missing"
    return hub


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        print("selftest ok (hub step", _selftest(sys.argv[2]), ")")
        raise SystemExit(0)
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    common.write(build(src, calls), out, date=recorded_date(src) or "(date not saved)")   # build() sets the eyebrow itself
