"""Replay data for eco_network_adoption (repeated 2x2 technology-adoption game with network externalities; story from Katz and
Shapiro 1985 and Farrell and Saloner 1985, payoff tables drawn by the project's own generator).

usage: python3 -I eco_network_adoption.py SOURCE OUT_DIR [CALLS.jsonl] [--date YYYY-MM-DD]
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md), a file inside
          one, or the rendered Markdown replay of one (one "- **event** ... `{json}`" line per logged event).
  OUT_DIR where <OUT_DIR>/eco_network_adoption.json is written (via common.write).
  --date  the recording date, needed only when the source has no recorded_at (meta.json or run.json); the Markdown replay has
          none. A source without a date and no --date is an error; no date is assumed.
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). When it is named it
          must exist, hold calls and match at least one decision; otherwise the build stops. A call belongs to the
          decision of the same seat, phase and round (read from its prompt: "You are seat Pk", "Current phase: X", the public
          state's "round"); when a prompt does not show them, calls are taken in time order, one run of same-seat calls per
          decision, with the phase read from the answer's action_id. Within a decision's calls the last one whose answer is
          exactly the logged move is the decision's call (earlier ones were format retries; their reasoning is kept above it).
          Its reasoning becomes the step's thinking and its prompt the observation.

Each round has one decision per seat (rules: Round Procedure; the DECIDE phase asks seat P0, then seat P1, and P1 cannot see
P0's pending choice), plus talk_passes public messages per seat before it when talk is on. The log has no event per decision:
both choices are read from the round's result record (round_result: choice0, choice1, payoff0, payoff1, score0, score1). Every
number shown is taken from the log; payoffs and running scores are re-derived from the payoff table and any disagreement is
printed as a warning, as are the reference values (equilibria, risk dominance, mixed equilibrium) when they differ from the
ones stored in the config."""
import json, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_network_adoption"
MINUS, TIMES, DOT = "\u2212", "\u00d7", "\u00b7"
CELLS = ("AA", "AB", "BA", "BB")
SEATS = ["Row player", "Column player"]       # seat P0 picks the row of the payoff table, seat P1 the column (rules: Players)
SHORT = ["Row", "Column"]
# The three tables of the default configs (rules: Payoff Tables), used only to name the table when the source has no config.
KNOWN_TABLES = {"t1": {"AA": (87, 94), "AB": (30, 17), "BA": (42, 21), "BB": (59, 25)},
                "t2": {"AA": (376, 472), "AB": (131, 35), "BA": (283, 116), "BB": (302, 417)},
                "t3": {"AA": (96, 96), "AB": (41, 48), "BA": (21, 56), "BB": (78, 87)}}


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_ANY = re.compile("[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af\uff00-\uffef]")
_CJK_RUN = re.compile("[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_PUNCT = {"\u3002": ". ", "\u3001": ", ", "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"', "\u3010": "[", "\u3011": "]",
          "\u300a": '"', "\u300b": '"', "\u3008": "<", "\u3009": ">", "\u301c": "~"}


def en(s):
    """The page shows English only: full-width forms become ASCII, CJK punctuation its ASCII counterpart, and any remaining run
    of Chinese, Japanese or Korean characters (the model sometimes reasons in Chinese) a visible marker."""
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
    return re.sub("[\u3000-\u303f\uff00-\uffef]", "", s)


# ------------------------------------------------------------------------------------------------ loading
_TAG_HIDDEN, _TAG_PRIVATE = "\u9690\u85cf", "\u79c1\u5bc6"     # the Markdown replay's "hidden" and "private" tags
_SEED, _MODEL = "\u79cd\u5b50", "\u6a21\u578b"                  # its header words for "seed" and "model"


def _md_bundle(path):
    """A rendered Markdown replay (declarative games): events from the '- **event** (tag): `{json}`' lines (the tag marks hidden
    and private events), seat rewards, seed and model from the header, and the model calls listed after the events ('### '
    blocks), if any."""
    text = open(path, encoding="utf-8").read()
    events = []
    for m in re.finditer(r"^- \*\*([A-Za-z_]+)\*\*([^`\n]*)`(\{.*\})`\s*$", text, re.M):
        try:
            body = json.loads(m.group(3))
        except ValueError:
            continue
        body["event"] = m.group(1)
        if _TAG_HIDDEN in m.group(2):
            body["_visibility"] = "hidden"
        elif _TAG_PRIVATE in m.group(2):
            body["_visibility"] = "private"
        events.append(body)
    head = text.split("\n## ")[0].split("\n- **")[0]
    rw = re.search(r'(\{"0":[^}]*\})', head)
    seed = re.search(_SEED + r" (\d+)", head)
    model = re.search(_MODEL + r" ([^\s,\uff0c]+)", head)
    over = next((e for e in events if e.get("event") == "game_over"), {})
    setup = next((e for e in events if e.get("event") == "setup"), {})
    summary = {"rewards": json.loads(rw.group(1)) if rw else {}, "roles": setup.get("roles", {}),
               "variables": over.get("variables", {}), "winners": over.get("winners")}
    run = {"seed": int(seed.group(1)) if seed else None, "model": model.group(1) if model else ""}
    calls = []
    for blk in re.split(r"\n(?=### )", text)[1:]:
        hdr = blk.split("\n", 1)[0]
        ans = re.findall(r"^\*\*[^*\n]+\*\* `(.*)`\s*$", blk, re.M)
        th = re.findall(r"````text\n(.*?)\n````", blk, re.S)
        sm = re.search(r"\bP(\d+)\b", hdr)
        if ans:
            calls.append({"seat": common.seat_of(hdr) if common.seat_of(hdr) is not None else (int(sm.group(1)) if sm else None),
                          "prompt": "", "answer": (ans[-1] if len(ans) > 1 else ans[0]).strip(),
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


def recorded_date(b, override=None):
    """The recording date (YYYY-MM-DD) from meta.json or run.json, or the --date given on the command line. A source without
    one (the rendered Markdown replay has none) is an error unless --date is given: no date is guessed."""
    d = override or (b.get("meta") or {}).get("recorded_at") or (b.get("run") or {}).get("recorded_at")
    if not d or not re.match(r"\d{4}-\d{2}-\d{2}", str(d)):
        raise SystemExit("no recorded_at in meta.json or run.json" + (f" (got {d!r})" if d else "")
                         + "; pass --date YYYY-MM-DD for a source without one")
    return str(d)[:10]


# ------------------------------------------------------------------------------------------------ small helpers
def num(v):
    """696.0 -> 696; keeps real fractions."""
    v = float(v)
    return int(v) if v.is_integer() else v


def fint(v):
    v = num(v)
    s = f"{abs(v):,}" if isinstance(v, int) else f"{abs(v):,.2f}"
    return (MINUS + s) if v < 0 else s


def signed(v):
    v = num(v)
    return ("+" if v > 0 else "") + fint(v)


def rounds_word(n):
    return f"{n} round" + ("" if n == 1 else "s")


def outcome_of(c0, c1):
    return "AA" if c0 == c1 == "A" else "BB" if c0 == c1 == "B" else "MIS"


CHOICE_WORD = {"A": "switch to the new standard", "B": "stay with the old standard"}


# ------------------------------------------------------------------------------------------------ the one-shot game (reference)
def theory(T):
    """Pure equilibria, payoff and risk dominance (Harsanyi and Selten 1988: larger product of the deviation losses) and the
    mixed equilibrium of the one-shot table T = {cell: (row payoff, column payoff)}."""
    a, b, c, d = T["AA"][0], T["AB"][0], T["BA"][0], T["BB"][0]          # row player's payoffs
    e, f, g, h = T["AA"][1], T["AB"][1], T["BA"][1], T["BB"][1]          # column player's payoffs
    pure = [k for k, ok in (("AA", a >= c and e >= f), ("AB", b >= d and f >= e), ("BA", c >= a and g >= h), ("BB", d >= b and h >= g)) if ok]
    pd = "AA" if a > d and e > h else "BB" if d > a and h > e else None
    prod_aa, prod_bb = (a - c) * (e - f), (d - b) * (h - g)
    rd = "AA" if prod_aa > prod_bb else "BB" if prod_bb > prod_aa else None
    den_p, den_q = (e - f) + (h - g), (a - c) + (d - b)
    p = (h - g) / den_p if den_p else None          # row player's probability of A that leaves the column player indifferent
    q = (d - b) / den_q if den_q else None          # column player's probability of A that leaves the row player indifferent
    mixed = (p, q) if p is not None and q is not None and 0 < p < 1 and 0 < q < 1 else None
    return {"pure": pure, "payoff_dominant": pd, "risk_dominant": rd, "prod_aa": prod_aa, "prod_bb": prod_bb, "mixed": mixed}


# ------------------------------------------------------------------------------------------------ the log
def read_game(events, summary, cfg):
    rules = next((e for e in events if e.get("event") == "rules"), {})
    var = dict((next((e for e in events if e.get("event") == "game_over"), {}) or {}).get("variables") or {})
    var.update({k: v for k, v in (summary.get("variables") or {}).items() if k not in var})
    cv = cfg.get("variables") or {}

    def par(k, default=None):
        for src in (rules, var, cv):
            if src.get(k) is not None:
                return src[k]
        return default
    T = {}
    for cell in CELLS:
        v = par("payoff_" + cell)
        if v is None:
            raise SystemExit(f"payoff_{cell} is in neither the rules event, the final variables nor the config")
        T[cell] = (num(v[0]), num(v[1]))
    P = {"T": T, "R": int(par("rounds", 8)), "talk": int(par("talk_passes", 0) or 0), "table_id": cv.get("table_id"), "cv": cv}
    known = next((k for k, t in KNOWN_TABLES.items() if all(tuple(t[c]) == T[c] for c in CELLS)), None)
    if P["table_id"] and known and P["table_id"] != known:
        print(f"warning: config says table {P['table_id']}, the payoffs are those of {known}", file=sys.stderr)
    P["table_id"] = P["table_id"] or known

    rounds, talks = [], []
    for e in events:
        ev = e.get("event")
        if ev == "round_result":
            r = {"round": int(e["round"]), "c": [str(e["choice0"]), str(e["choice1"])], "pay": [num(e["payoff0"]), num(e["payoff1"])],
                 "score": [num(e["score0"]), num(e["score1"])]}
            r["out"] = outcome_of(*r["c"])
            r["cell"] = r["c"][0] + r["c"][1]
            rounds.append(r)
        elif ev in ("talk", "message", "chat", "say"):             # public messages (only when talk_passes > 0)
            # the eco matrix-game template logs {"actor": k, "event": "talk", "text": ...} with no round (eco_trading_impact)
            seat = _seat_int(e.get("seat", e.get("actor", e.get("speaker", e.get("from")))))
            text = e.get("text", e.get("message", e.get("content")))
            if seat is None or text is None:
                print(f"warning: {ev} event without a speaker or text is not shown: {json.dumps(e)[:200]}", file=sys.stderr)
                continue
            talks.append({"seat": seat, "text": str(text), "round": int(e.get("round") or len(rounds) + 1)})
    if P["talk"] > 0:
        want = P["talk"] * 2 * len(rounds)
        if not talks:
            print(f"warning: talk_passes is {P['talk']} but no message event was read from the log; the page shows no messages",
                  file=sys.stderr)
        elif len([t for t in talks if t["round"] <= len(rounds)]) != want:
            print(f"warning: {len(talks)} messages read, {want} expected ({P['talk']} per seat in each of {len(rounds)} rounds)",
                  file=sys.stderr)
    return P, rounds, talks


def _seat_int(v):
    """0, "0" or "P0" -> 0; None when absent or unreadable."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return v
    m = re.fullmatch(r"\s*P?(\d+)\s*", str(v))
    return int(m.group(1)) if m else None


def verify(P, rounds, rewards, game_end):
    """Warnings for any logged number the payoff table does not reproduce (the page shows the logged values)."""
    T, warn, acc = P["T"], [], [0, 0]
    for k, r in enumerate(rounds):
        if r["round"] != k + 1:
            warn.append(f"round record {k + 1} is numbered {r['round']}")
        if r["cell"] not in T:
            warn.append(f"round {r['round']}: choices {r['c']} are not A/B")
            continue
        for s in (0, 1):
            if T[r["cell"]][s] != r["pay"][s]:
                warn.append(f"round {r['round']} seat {s}: payoff {r['pay'][s]} logged, {T[r['cell']][s]} in cell {r['cell']}")
            acc[s] += r["pay"][s]
            if acc[s] != r["score"][s]:
                warn.append(f"round {r['round']} seat {s}: score {r['score'][s]} logged, {acc[s]} summed")
    if len(rounds) != P["R"]:
        warn.append(f"{len(rounds)} rounds logged, the config says {P['R']}")
    for s in (0, 1):
        if rewards is not None and rounds and rewards[s] != rounds[-1]["score"][s]:
            warn.append(f"seat {s}: reward {rewards[s]}, last logged score {rounds[-1]['score'][s]}")
        if game_end.get(f"score{s}") is not None and rounds and num(game_end[f"score{s}"]) != rounds[-1]["score"][s]:
            warn.append(f"seat {s}: game_end score {game_end[f'score{s}']}, last logged score {rounds[-1]['score'][s]}")
    th, cv = theory(T), P["cv"]
    if cv.get("ref_pure_nash") is not None and sorted(cv["ref_pure_nash"]) != sorted(th["pure"]):
        warn.append(f"config ref_pure_nash {cv['ref_pure_nash']}, computed {th['pure']}")
    for k, mine in (("ref_payoff_dominant", th["payoff_dominant"]), ("ref_risk_dominant", th["risk_dominant"])):
        if cv.get(k) is not None and cv[k] != mine:
            warn.append(f"config {k} {cv[k]}, computed {mine}")
    if th["mixed"]:
        for k, mine in (("ref_mixed_a_rate0", th["mixed"][0]), ("ref_mixed_a_rate1", th["mixed"][1])):
            if cv.get(k) is not None and abs(float(cv[k]) - mine) > 5e-5:
                warn.append(f"config {k} {cv[k]}, computed {mine:.4f}")
    return warn


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


def call_meta(c):
    """Phase and round of a call, from its prompt ('Current phase: X', the public state's "round"); None when not shown."""
    p = c.get("prompt") or ""
    m = re.search(r"Current phase:\s*([A-Za-z_]+)", p)
    i = p.find("Public state:")
    rm = re.search(r'"round":\s*(\d+)', p[i:]) if i >= 0 else None
    return (m.group(1).upper() if m else None), (int(rm.group(1)) if rm else None)


def _matches(call, st):
    """Strict: a DECIDE answer must be action_id "choose" with choice exactly "A" or "B" (the legal payloads; anything else is an
    invalid reply the engine replaces), and a TALK answer action_id "talk" with the logged text."""
    a = _norm_answer(call.get("answer"))
    if not isinstance(a, dict):
        return False
    p = a.get("payload") if isinstance(a.get("payload"), dict) else {}
    if st["phase"] == "DECIDE":
        ch = p.get("choice")
        return a.get("action_id") == "choose" and ch in ("A", "B") and ch == st["choice"]
    txt = p.get("text", next((v for v in p.values() if isinstance(v, str)), None))
    return a.get("action_id") == "talk" and isinstance(txt, str) and " ".join(txt.split()) == " ".join(st["text"].split())


def _answer_phase(call):
    """The phase a call answered, from its action_id ("talk" or "choose"); None when the answer does not say."""
    a = _norm_answer(call.get("answer"))
    aid = a.get("action_id") if isinstance(a, dict) else None
    return {"talk": "TALK", "choose": "DECIDE"}.get(aid)


def attach_calls(steps, calls):
    """Tie calls to steps (see the module docstring). Returns (steps with a call, steps whose call answered the logged move).
    A decision's calls are its format retries followed by the accepted answer, so the LAST matching call is the decision's call
    and the earlier ones are shown as earlier attempts. Without seat, phase and round in every prompt the calls are read in time
    order: a decision takes the whole run of consecutive calls of its seat (the seats alternate, P0 then P1, in every phase), and
    the phase of a call whose prompt does not show it is read from its answer, so a TALK call never becomes a DECIDE candidate."""
    if not calls:
        return 0, 0
    meta = []
    for c in calls:
        ph, rd = call_meta(c)
        meta.append((ph or _answer_phase(c), rd))
    keyed = all(c.get("seat") is not None and m[0] is not None and m[1] is not None for c, m in zip(calls, meta))
    used = [False] * len(calls)
    tied = hits = ptr = 0
    for st in steps:
        if keyed:
            cand = [k for k in range(len(calls)) if not used[k] and calls[k]["seat"] == st["seat"] and meta[k][0] == st["phase"]
                    and meta[k][1] == st["round"]]
        else:
            cand, j = [], ptr
            while j < len(calls) and calls[j].get("seat") in (None, st["seat"]) and meta[j][0] in (None, st["phase"]) \
                    and meta[j][1] in (None, st["round"]):
                cand.append(j)
                j += 1
                # a seat-less call, or a TALK call (a seat may send several messages in a row), ends the run at its first match
                if (calls[j - 1].get("seat") is None or st["phase"] == "TALK") and _matches(calls[j - 1], st):
                    break
        if not cand:
            continue
        match = [k for k in cand if _matches(calls[k], st)]
        if st["phase"] == "TALK":                   # in keyed mode cand holds all of the seat's messages of the round
            hit = match[0] if match else None
            use = hit if hit is not None else cand[0]
            run = [k for k in cand if k <= use]
        else:
            hit = match[-1] if match else None
            use = hit if hit is not None else cand[-1]
            run = cand
        before, after = [k for k in run if k < use], [k for k in run if k > use]
        parts = [f"[Earlier attempt, answer: {calls[k]['answer'][:300]}]\n{calls[k].get('reasoning') or '(no reasoning returned)'}" for k in before]
        c = calls[use]
        parts.append(("[Final attempt]\n" if before else "") + (c.get("reasoning") or ("(no reasoning returned)" if before else "")))
        if hit is None:
            what = f"choice {st['choice']}" if st["phase"] == "DECIDE" else "the logged message"
            parts.append(f"[The game applied {what} instead of this answer: {c['answer'][:300]}]")
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        parts += [f"[A later call for this decision, answer: {calls[k]['answer'][:300]}]\n{calls[k].get('reasoning') or ''}"
                  for k in after]
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        if c.get("prompt"):
            st["state"] = en(common.state_part(c["prompt"]))
            st["state_from_prompt"] = True
        a = _norm_answer(c.get("answer"))
        if hit is not None and isinstance(a, dict):
            st["act"] = en(json.dumps(a, ensure_ascii=False))
        for k in run:
            used[k] = True
        if not keyed:
            ptr = run[-1] + 1
        st["tied"], st["hit"] = True, hit is not None
        tied += 1
        hits += hit is not None
    if not all(used):
        print(f"warning: {used.count(False)} of {len(calls)} calls matched no decision", file=sys.stderr)
    return tied, hits


# ------------------------------------------------------------------------------------------------ the stale "pending" field
_HINT = re.compile(r"\b(suggests?|confirms?|hint\w*|commit\w*|precommit\w*)\b", re.I)


def private_info(state):
    """The seat's private info as shown in a saved prompt ('Your private info: {...}'); None when the line is absent."""
    m = re.search(r"^Your private info:\s*(\{.*\})\s*$", state or "", re.M)
    if not m:
        return None
    try:
        v = json.loads(m.group(1))
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


def pending_note(steps, rounds):
    """One foot sentence on the per-seat private field "pending" when a saved DECIDE prompt shows it. The choose action stores the
    seat's choice there and the round's resolution reads it but does not clear it (games/eco_network_adoption.yaml: choose ->
    set_player private.pending; DECIDE on_complete), so from round 2 each prompt shows the seat's own previous choice as
    private info. The sentence says what it is, how we know it is not the other player's choice, and how replies read it."""
    seen = []                                       # (round, seat, value or None, thinking) of each saved DECIDE prompt
    for st in steps:
        if st["phase"] == "DECIDE" and st.get("state_from_prompt"):
            pi = private_info(st["state"])
            if pi is not None:
                seen.append((st["round"], st["seat"], pi.get("pending"), st.get("thinking") or ""))
    shown = [x for x in seen if x[2] is not None]
    if not shown:
        return ""
    prev = {r["round"]: r["c"] for r in rounds}
    own = all(r >= 2 and r - 1 in prev and v == prev[r - 1][s] for r, s, v, _ in shown)
    if not own:
        print("warning: a saved prompt shows a private 'pending' value that is not the seat's own previous choice: "
              + ", ".join(f"round {r} P{s} {v!r}" for r, s, v, _ in shown
                          if not (r >= 2 and r - 1 in prev and v == prev[r - 1][s])), file=sys.stderr)
        return (f"{len(shown)} of the {len(seen)} saved prompts show a private info field \"pending\" that the engine leaves set "
                "from an earlier choice and the prompt does not explain; it does not always equal the seat's own previous choice, "
                "so read the prompts before reading the choices. ")
    first = min(r for r, _, _, _ in shown)
    vals = {v for _, _, v, _ in shown}
    again = f"choose {next(iter(vals))} again" if len(vals) == 1 else "repeat its previous choice"
    # evidence that it is not the other player's current choice: the column player's round-1 prompt (the row player has chosen)
    col_r1_empty = any(r == 1 and s == 1 and v is None for r, s, v, _ in seen)
    mention = sum(1 for _, _, _, th in shown if "pending" in th)
    hints = []
    for r, s, _, th in shown:
        for sent in re.split(r"(?<=[.?!])\s+|\n+", th):
            sent = sent.strip()
            if "pending" in sent and _HINT.search(sent) and not sent.endswith("?") and len(sent) <= 160:
                hints.append((r, s, sent.rstrip(".")))
                break
    every = all(v is not None for r, _, v, _ in seen if r >= first)
    text = (f"From round {first} " + ("each prompt's" if every else "the prompts'")
            + " private info shows {\"pending\": <that seat's own choice in the previous round>}: the engine stores each choice "
            "there and does not clear it after the round, and the prompt does not say what it is. ")
    if col_r1_empty:
        text += ("It is not the other player's choice: in round 1 the column player's private info is empty although the row "
                 "player had already chosen. ")
    said = (f"all {mention} replies that saw it mention it" if mention == len(shown) else
            f"{mention} of the {len(shown)} replies that saw it mention it")
    if hints:                                       # a keyword match finds examples, not an exact count: no number is claimed
        r, s, sent = hints[0]
        text += (f"{said[0].upper() + said[1:]}, and " + ("some read" if len(hints) > 1 else "one reads")
                 + f" it as a hint or a commitment to {again} (for example the {SHORT[s].lower()} player in round {r}: \"{sent}\"). ")
    elif mention:
        text += f"{said[0].upper() + said[1:]}. "
    r1 = [x for x in seen if x[0] == 1]
    if first >= 2 and len(r1) == 2 and rounds and rounds[0]["round"] == 1:
        text += f"Round 1, played without it, ended {rounds[0]['cell']}. "
    return text


# ------------------------------------------------------------------------------------------------ build
def build(bundle_dir, calls=None, date=None):
    b = load_source(bundle_dir)
    date = recorded_date(b, date)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    if not calls:
        calls = b.get("md_calls") or []
    events, summary, run, cfg = b["events"], b.get("summary") or {}, b.get("run") or {}, b.get("config") or {}
    P, rounds, talks = read_game(events, summary, cfg)
    if not rounds:
        raise SystemExit("no finished round in the game log")
    T, R, Rcfg = P["T"], len(rounds), P["R"]
    game_end = next((e for e in events if e.get("event") == "game_end"), {})
    raw_rw = summary.get("rewards") or {}
    rewards = [num(raw_rw.get(str(s), raw_rw.get(s))) for s in (0, 1)] if raw_rw else None
    warnings = verify(P, rounds, rewards, game_end)
    for w in warnings:
        print("warning:", w, file=sys.stderr)
    final = rounds[-1]["score"]
    th = theory(T)
    tid = P["table_id"]

    # ---- steps: per round the talk messages (if any), then P0's and P1's choices (rules: Round Procedure)
    steps, feed = [], []
    for r in rounds:
        x = r["round"] - 1
        for t in [t for t in talks if t["round"] == r["round"]]:
            steps.append({"seat": t["seat"], "x": x, "kind": "talk", "phase": "TALK", "round": r["round"], "text": t["text"],
                          "head": f"Round {r['round']} {DOT} public message",
                          "act": json.dumps({"action_id": "talk", "payload": {"text": t["text"]}}, ensure_ascii=False)})
            feed.append({"x": x, "step": len(steps) - 1, "who": SEATS[t["seat"]], "kind": "talk", "text": en(t["text"])})
        for s in (0, 1):
            ch = r["c"][s]
            steps.append({"seat": s, "x": x, "kind": "switch" if ch == "A" else "stay", "phase": "DECIDE", "round": r["round"], "choice": ch,
                          "head": f"Round {r['round']} {DOT} chooses {ch}" + {"A": ", the new standard", "B": ", the old standard"}.get(ch, ""),
                          "act": json.dumps({"action_id": "choose", "payload": {"choice": ch}})})

    # ---- what each seat saw (rebuilt from the log; the saved prompt where there is one)
    table_line = ("Payoff table, [P0, P1] for each pair of choices (P0's first): "
                  + ", ".join(f"{c} [{fint(T[c][0])}, {fint(T[c][1])}]" for c in CELLS) + ".")

    def observation(st):
        r, s = st["round"], st["seat"]
        past = [q for q in rounds if q["round"] < r]
        sc = past[-1]["score"] if past else [0, 0]
        lines = ["(Rebuilt from the game log; the exact prompt for this turn was not saved.)",
                 f"You are seat P{s}, the {'row' if s == 0 else 'column'} player. Round {r} of {Rcfg}.",
                 ("Current phase: DECIDE. Both players secretly choose A (switch to the new standard) or B (stay with the old standard); "
                  "the choices are revealed together." if st["phase"] == "DECIDE" else
                  f"Current phase: TALK. Each player sends {P['talk']} public, non-binding message(s) before the decision."),
                 table_line,
                 f"Scores so far: P0 {fint(sc[0])}, P1 {fint(sc[1])}."]
        if past:
            lines.append("Earlier rounds:")
            lines += [f"- Round {q['round']}: P0 chose {q['c'][0]}, P1 chose {q['c'][1]}; payoffs {fint(q['pay'][0])} / {fint(q['pay'][1])}."
                      for q in past]
        else:
            lines.append("Earlier rounds: none.")
        said = [t for t in talks if t["round"] < r or (t["round"] == r and st["phase"] == "DECIDE")]
        if said:
            lines.append("Messages: " + " | ".join(f"round {t['round']} P{t['seat']}: {t['text']}" for t in said))
        if st["phase"] == "DECIDE" and s == 1:
            lines.append("P0's choice for this round is not shown until both have chosen.")
        return en("\n".join(lines))

    prompt = run.get("prompt") or {}
    for st in steps:
        st["state"] = observation(st)
        if (prompt.get("text") and prompt.get("seat") == st["seat"] and str(prompt.get("phase", "")).upper() == st["phase"]
                and re.search(r'"round": %d\b' % st["round"], prompt["text"]) and not any(q.get("state_from_prompt") for q in steps)):
            st["state"] = en(common.state_part(prompt["text"]))        # the saved prompt of the game's first decision
            st["state_from_prompt"] = True

    tied, hits = attach_calls(steps, calls)
    if calls and not tied:                          # a wrong call log would otherwise give a quiet "reasoning not saved" page
        raise SystemExit(f"{len(calls)} model calls were given but none matched a decision of this game")
    pnote = pending_note(steps, rounds)
    has_reason = any(s.get("thinking") for s in steps)
    from_prompt = sum(1 for s in steps if s.get("state_from_prompt"))
    dec = [s for s in steps if s["phase"] == "DECIDE"]
    dec_tied = sum(1 for s in dec if s.get("tied"))
    dec_replaced = sum(1 for s in dec if s.get("tied") and not s.get("hit"))
    for st in steps:
        st.setdefault("thinking", "")
        for k in ("phase", "round", "choice", "text", "state_from_prompt", "tied", "hit"):
            st.pop(k, None)

    # ---- counts
    n_out = {k: sum(r["out"] == k for r in rounds) for k in ("AA", "BB", "MIS")}
    a_count = [sum(r["c"][s] == "A" for r in rounds) for s in (0, 1)]
    last = rounds[-1]["out"]
    settle = R
    while settle > 1 and rounds[settle - 2]["out"] == last:
        settle -= 1                                                  # first round of the final run of equal outcomes
    settled = last in ("AA", "BB") and settle > 1 and R - settle + 1 >= 3   # "settle" only for a final run of 3+ rounds
    first_aa = next((r["round"] for r in rounds if r["out"] == "AA"), None)
    bb_path = [T["BB"][s] * R for s in (0, 1)]
    aa_max = [T["AA"][s] * R for s in (0, 1)]
    worst = [min(T["AB"][s], T["BA"][s]) for s in (0, 1)]             # each player's worse mismatch payoff
    counts_phrase = f"AA in {n_out['AA']}, BB in {n_out['BB']} and mismatch in {n_out['MIS']} of {R} rounds"
    # the majority outcome with its exceptional rounds named, e.g. "AA in 7 of 8 rounds, mismatch in round 8 (AB)"
    OUT_WORD = {"AA": "AA", "BB": "BB", "MIS": "mismatch"}
    top = max(("AA", "BB", "MIS"), key=lambda k: n_out[k])
    others = [r for r in rounds if r["out"] != top]
    majority_phrase = None
    if 2 * n_out[top] > R and others and len(others) <= 3:
        groups = []
        for k in ("AA", "BB", "MIS"):
            labels = [f"{r['round']} ({r['cell']})" if k == "MIS" else str(r["round"]) for r in others if r["out"] == k]
            if labels:
                groups.append(f"{OUT_WORD[k]} in round{'s' if len(labels) > 1 else ''} "
                              + (", ".join(labels[:-1]) + " and " + labels[-1] if len(labels) > 1 else labels[0]))
        majority_phrase = f"{OUT_WORD[top]} in {n_out[top]} of {R} rounds, " + ", ".join(groups)
    pd_rd = th["payoff_dominant"] == "AA" and th["risk_dominant"] == "AA"      # t1, t3: the new standard is also the safer one
    lock_table = th["payoff_dominant"] == "AA" and th["risk_dominant"] == "BB"  # t2: the old standard is the safer one

    # ---- charts
    OUT_STYLE = {"AA": ("c3", "Both on the new standard"), "BB": ("hold", "Both stay with the old standard"),
                 "MIS": ("c4", "Mismatch")}
    mis_worse = all(max(T["AB"][s], T["BA"][s]) < min(T["AA"][s], T["BB"][s]) for s in (0, 1))   # rules: AA > BB > mismatch
    mis_word = "worse for both than AA or BB" if mis_worse else "the players chose differently"
    rows = [SHORT[0], SHORT[1], "Outcome"]
    cells = [[None] * R for _ in rows]
    for r in rounds:
        i = r["round"] - 1
        for s in (0, 1):
            ch = r["c"][s]
            cells[s][i] = {"c": "c3" if ch == "A" else "hold" if ch == "B" else "muted", "t": ch,
                           "tip": f"chooses {ch}" + (f" ({CHOICE_WORD[ch]})" if ch in CHOICE_WORD else "")
                                  + f", gets {fint(r['pay'][s])}"}
        col, word = OUT_STYLE[r["out"]]
        cells[2][i] = {"c": col, "t": r["cell"],
                       "tip": f"{r['cell']}: {word.lower() if r['out'] != 'MIS' else 'mismatch, ' + mis_word}, "
                              f"{fint(r['pay'][0])} / {fint(r['pay'][1])}; scores {fint(r['score'][0])} / {fint(r['score'][1])}"}
    grid_sub = ("Both choose at the same time each round: A switches to the new standard, B stays with the old one. Outcome is the cell "
                "of the payoff table the round landed on, with payoffs [row, column]: "
                + ", ".join(f"{c} {fint(T[c][0])} / {fint(T[c][1])}" for c in CELLS)
                + " (AB means the row player chose A and the column player B). "
                + ("Here every round landed on AA." if n_out["AA"] == R else
                   f"AA {rounds_word(n_out['AA'])}, BB {rounds_word(n_out['BB'])}, mismatch {rounds_word(n_out['MIS'])}."))
    # the legend names only the colors drawn (the sub lists all four cells of the payoff table)
    used_c = {cell["c"] for row in cells for cell in row if cell}
    legend = [e for e in [["A, or both on the new standard (AA)", "c3"], ["B, or both on the old standard (BB)", "hold"],
                          [f"Mismatch (AB or BA): {mis_word}", "c4"]] if e[1] in used_c]
    charts = [{"title": "Choices, round by round", "sub": grid_sub, "height": 120,
               "grid": {"rows": rows, "cells": cells, "rowHeight": 30, "legend": legend}}]

    pay = [[None] * R, [None] * R]
    cum = [[None] * R, [None] * R]
    for r in rounds:
        for s in (0, 1):
            pay[s][r["round"] - 1] = r["pay"][s]
            cum[s][r["round"] - 1] = r["score"][s]
    if len({tuple(r["pay"]) for r in rounds}) > 1:      # a per-round chart only when the payoffs change from round to round
        charts.append({"title": "Payoff per round", "sub": "Bars: what each player got in the round. Dashed: what it would have got had "
                       f"both stayed with the old standard (BB: {fint(T['BB'][0])} for the row player, {fint(T['BB'][1])} for the column "
                       f"player). Both switching (AA) pays {fint(T['AA'][0])} and {fint(T['AA'][1])}; a mismatch pays at most "
                       f"{fint(max(T['AB'][0], T['BA'][0]))} and {fint(max(T['AB'][1], T['BA'][1]))}.",
                       "height": 210, "fmt": "int",
                       "bars": [{"name": SEATS[0], "color": "c1", "v": pay[0]}, {"name": SEATS[1], "color": "c2", "v": pay[1]}],
                       "lines": [{"name": "Row player if both stay", "color": "c1", "dash": True, "thin": True, "v": [T["BB"][0]] * R},
                                 {"name": "Column player if both stay", "color": "c2", "dash": True, "thin": True,
                                  "v": [T["BB"][1]] * R}]})
    # Running totals measured against lock-in (both on BB every round = 0), so the stakes are one band per player: this game
    # above (solid; ends at the "Gain over lock-in" tile) and the worst case below (dashed). In absolute points the reference
    # paths of t1 end at 240, 200 and 136, too close for their end labels to clear each other's lines. Only the lock-in line has an
    # end label: the page sets labels above a line's end, which a falling worst-case line would cross; the legend names them.
    gain = [[None if v is None else v - T["BB"][s] * (i + 1) for i, v in enumerate(cum[s])] for s in (0, 1)]
    wgap = [worst[s] - T["BB"][s] for s in (0, 1)]
    best = [T["AA"][s] - T["BB"][s] for s in (0, 1)]
    cum_sub = ("Each player's running total minus what both staying with the old standard every round (BB, lock-in) would have paid by "
               "then, so lock-in is the gray line at 0. Solid: this game; it ends at the Gain over lock-in tile. Dashed: the worst case, "
               f"the player's worse mismatch payoff every round instead of BB: {R} {TIMES} ({fint(worst[0])} {MINUS} {fint(T['BB'][0])}) = "
               f"{fint(wgap[0] * R)} for the row player and {R} {TIMES} ({fint(worst[1])} {MINUS} {fint(T['BB'][1])}) = {fint(wgap[1] * R)} "
               f"for the column player. AA in every round, the most this table allows, ends at {signed(best[0] * R)} and "
               f"{signed(best[1] * R)}" + ("; here that is the solid lines." if n_out["AA"] == R else "."))
    # The lock-in label sits just above the zero line at the right end (about 5-16 px, roughly 9% of the value range at this height,
    # across up to 3 slots on a phone): keep it only when no other line passes through that band in the last 4 rounds.
    others_v = gain + [[wgap[s] * (i + 1) for i in range(R)] for s in (0, 1)]
    flat = [v for vs in others_v for v in vs if v is not None] + [0]
    band = 0.12 * ((max(flat) - min(flat)) or 1)
    lab_ok = True
    for vs in others_v:
        pts = [v for v in vs[max(0, R - 5):] if v is not None]
        for lo, hi in zip([pts[0]] + pts[:-1], pts):
            if min(lo, hi) <= band and max(lo, hi) > 0:
                lab_ok = False
    charts.append({"title": "Cumulative score against lock-in", "sub": cum_sub, "height": 250, "fmt": "int",
                   "lines": [{"name": "Lock-in: both stay", "color": "hold", "dash": True, "thin": True, "end": lab_ok, "v": [0] * R},
                             {"name": "Row: worst case", "color": "c1", "dash": True, "thin": True,
                              "v": [wgap[0] * (i + 1) for i in range(R)]},
                             {"name": "Column: worst case", "color": "c2", "dash": True, "thin": True,
                              "v": [wgap[1] * (i + 1) for i in range(R)]},
                             {"name": SEATS[0], "color": "c1", "v": gain[0], "dots": True},
                             {"name": SEATS[1], "color": "c2", "v": gain[1], "dots": True}]})

    # ---- tiles
    if not dec_tied:                                # no saved replies: an A may be the engine's default for an invalid reply
        invalid_s = ("A is also the engine's default for an invalid reply; this recording did not save how many replies were invalid"
                     if any(a_count) else "This recording did not save how many replies were invalid")
    elif dec_replaced:
        invalid_s = f"{dec_replaced} of the {dec_tied} choices with a saved reply replaced an invalid reply"
    elif dec_tied == len(dec):
        invalid_s = "Every choice is the model's own valid reply"
    else:
        invalid_s = f"All {dec_tied} choices with a saved reply are the model's own; {len(dec) - dec_tied} have none"
    if n_out["AA"] == R:
        coord_s = invalid_s
    else:
        non_aa = [r for r in rounds if r["out"] != "AA"]
        bits = []
        for k in ("BB", "MIS"):
            rs = [r for r in non_aa if r["out"] == k]
            if not rs:
                continue
            if len(non_aa) <= 3:                     # few exceptions: name the rounds
                lab = [f"{r['round']} ({r['cell']})" if k == "MIS" else str(r["round"]) for r in rs]
                bits.append(f"{OUT_WORD[k]} in round{'s' if len(lab) > 1 else ''} "
                            + (", ".join(lab[:-1]) + " and " + lab[-1] if len(lab) > 1 else lab[0]))
            else:
                bits.append(f"{OUT_WORD[k]} in {rounds_word(len(rs)) if len(rs) < R else 'all ' + rounds_word(R)}")
        bits[0] = bits[0][0].upper() + bits[0][1:]
        coord_s = ("; ".join(bits) + (f"; settled on {last} from round {settle}" if settled else "")
                   + (f"; first AA in round {first_aa}" if first_aa and first_aa > 1 and not (settled and last == "AA" and settle == first_aa)
                      else "") + ". " + invalid_s)
    at_max = [final[s] == aa_max[s] for s in (0, 1)]
    if all(at_max):
        score_s = f"Row / column player: {R} {TIMES} {fint(T['AA'][0])} and {R} {TIMES} {fint(T['AA'][1])}, the most this table allows"
    else:
        score_s = (f"Row / column player; {100 * final[0] / aa_max[0]:.0f}% and {100 * final[1] / aa_max[1]:.0f}% of the most this table "
                   f"allows (AA every round: {fint(aa_max[0])} / {fint(aa_max[1])})")
    mixed = th["mixed"]

    def pct(x):
        v = 100 * x
        return f"{v:.0f}%" if v >= 1 else f"{v:.1f}%"
    # switching pays for the row player when the column player switches with probability above mixed[1], and vice versa
    thr = (f"switching pays for the row player if the column player switches with probability above {pct(mixed[1])}, and for the "
           f"column player above {pct(mixed[0])}") if mixed else None
    if pd_rd:
        th_v, th_s = "AA", ("Payoff and risk dominant: " + thr if thr else "Both payoff and risk dominant; BB is the other pure equilibrium")
    elif lock_table:
        th_v, th_s = "AA or BB", ("AA pays more, BB is risk dominant: " + thr if thr else "AA is payoff dominant, BB risk dominant")
    else:
        th_v, th_s = " and ".join(th["pure"]) or "none", (thr[0].upper() + thr[1:] if thr else "Pure equilibria of the one-shot game")
    tiles = [{"k": "Coordination", "v": f"AA in {n_out['AA']} of {R}", "s": coord_s},
             {"k": "Final scores", "v": f"{fint(final[0])} / {fint(final[1])}", "s": score_s},
             {"k": "Gain over lock-in", "v": f"{signed(final[0] - bb_path[0])} / {signed(final[1] - bb_path[1])}",
              "s": f"Against both staying with the old standard (BB) for all {R} rounds: {fint(bb_path[0])} / {fint(bb_path[1])}"},
             {"k": "Theory" + (f" for table {tid}" if tid else ""), "v": th_v, "s": th_s}]

    # ---- text
    if n_out["AA"] == R:
        title = (f"AA in all {R} rounds: both players are on the new standard from round 1"
                 + (", as theory predicts for this table" if pd_rd else
                    ", although the old one is the safer equilibrium in this table" if lock_table else ""))
    elif n_out["BB"] == R:
        title = f"Locked in: both players stay with the old standard in all {R} rounds"
    elif n_out["MIS"] == R:
        title = f"No coordination: the players choose differently in all {R} rounds"
    elif settled:
        title = f"The players settle on the {'new' if last == 'AA' else 'old'} standard from round {settle}: {counts_phrase}"
    elif majority_phrase:
        title = ({"AA": "Mostly on the new standard", "BB": "Mostly on the old standard", "MIS": "Mostly mismatched"}[top]
                 + ": " + majority_phrase)
    else:
        title = f"No settled standard: {counts_phrase}"
    other = "t2" if tid in ("t1", "t3") else None
    if pd_rd:
        pred = (f" In this table{f' ({tid})' if tid else ''} switching is both the better and the safer equilibrium, so AA is the predicted "
                "outcome" + (f"; the test of lock-in is table {other}, where staying is the safer one." if other else "."))
    elif lock_table:
        pred = (f" In this table{f' ({tid})' if tid else ''} switching is the better equilibrium but staying is the safer one (risk "
                "dominant), so this is the table where lock-in on the old standard is expected.")
    else:
        pred = ""
    lede = (f"Two players start on an old technology standard and, in each of {Rcfg} rounds, secretly choose A (switch to a new standard) "
            "or B (stay with the old one); the two choices are revealed together. A standard is worth more when the other player uses it "
            "too: both switching pays each player the most, both staying less, and choosing differently the least; each player scores the "
            "sum of its payoffs over the rounds, and "
            + ("the players cannot talk." if P["talk"] == 0 else
               f"before each choice each player sends {P['talk']} public, non-binding message(s).")
            + pred)
    model = (b.get("meta") or {}).get("model") or run.get("model") or ""
    model = model.split("/")[-1] if model else "the model"
    seed = run.get("seed", cfg.get("seed"))
    foot = (f"Data: the {date} recording of eco_network_adoption (seed {seed}" + (f", table {tid}" if tid else "")
            + f", {Rcfg} rounds, " + ("no talk" if P["talk"] == 0 else f"{P['talk']} message(s) per player per round")
            + f"), both seats played by {model}; every number on this page is computed from its game log"
            + (", and every payoff and running score agrees with the payoff table. " if not warnings else
               f"; {len(warnings)} logged value(s) disagree with the payoff table (the page shows the logged values). ")
            + (f"The config turns on {P['talk']} message(s) per player per round, but the log has no message event, so none is shown. "
               if P["talk"] > 0 and not talks else "")
            + ("Observations and reasoning come from the saved model calls. " if has_reason else
               "This recording was made before reasoning was saved; the game will be re-recorded with reasoning. ")
            + ("" if from_prompt == len(steps) else
               ("The observation of the first decision is the saved prompt; the others are rebuilt from the log. " if from_prompt else
                "The observations are rebuilt from the log. "))
            + ("" if tied else "The log keeps each round's two choices, not the model's replies, so each action shown is rebuilt from the "
               "logged choice, and the log does not say whether any reply was invalid (an invalid reply becomes A on the engine and "
               "training paths and a random legal choice in the local agent). ")
            + (f"{hits} of the {tied} moves with a saved model call are the model's own answers; for the others the game applied a "
               "different move. " if tied and hits < tied else "")
            + (f"{len(steps) - tied} of the {len(steps)} moves have no saved model call. " if tied and tied < len(steps) else "")
            + pnote
            + "The row player is seat P0 and the column player seat P1 in the log and the prompts; payoff_XY = [row, column] when the row "
            "player chooses X and the column player Y. Reference solution of the one-shot game: pure equilibria "
            + " and ".join(th["pure"]) + "; "
            + (f"{th['payoff_dominant']} is both payoff and risk dominant" if th["payoff_dominant"] and th["payoff_dominant"] == th["risk_dominant"]
               else (f"{th['payoff_dominant']} is payoff dominant" if th["payoff_dominant"] else "no cell is payoff dominant")
               + (f", {th['risk_dominant']} risk dominant" if th["risk_dominant"] else ""))
            + f" (Harsanyi and Selten 1988: deviation-loss products {fint(th['prod_aa'])} for AA, {fint(th['prod_bb'])} for BB)"
            + (f"; in the mixed equilibrium the row player chooses A with probability {mixed[0]:.4f} and the column player with "
               f"{mixed[1]:.4f}" if mixed else "")
            + f". The row player chose A in {a_count[0]} of {R} rounds, the column player in {a_count[1]}. "
            "The story is from Katz and Shapiro (1985) and Farrell and Saloner (1985); the 2x2 payoffs are the project's own random draws, "
            "not numbers from either paper.")
    kinds_all = {"switch": ["Chooses A: switch to new", "c3"], "stay": ["Chooses B: stay with old", "hold"], "talk": ["Public message", "muted"]}
    d = {"id": GID, "name": "Network adoption", "title": title, "lede": lede, "seatNames": SEATS, "tiles": tiles, "foot": foot,
         "x": {"n": R, "labels": [f"R{r['round']}" for r in rounds]}, "charts": charts,
         "kinds": {k: v for k, v in kinds_all.items() if any(s["kind"] == k for s in steps)}, "steps": steps}
    if feed:
        d.update({"feed": feed, "feedTitle": "Messages", "feedSub": "Public, non-binding messages sent before each round's choices. "
                  "The selected step is highlighted; later entries are hidden."})
    d["eyebrow"] = (common.EYEBROW.format(gid=GID, date=date) if has_reason
                    else f"social_game_env {DOT} {GID} {DOT} {model} {DOT} recorded {date} {DOT} reasoning not saved")
    return d

if __name__ == "__main__":
    args, date_opt = list(sys.argv[1:]), None
    if "--date" in args:
        k = args.index("--date")
        if k + 1 >= len(args):
            raise SystemExit("--date needs a value YYYY-MM-DD")
        date_opt = args[k + 1]
        del args[k:k + 2]
    if len(args) < 2:
        raise SystemExit(__doc__)
    src, out = args[0], args[1]
    calls = []
    if len(args) > 2:                               # a call log was named: it must exist and hold calls (no quiet fallback)
        if not os.path.isfile(args[2]):
            raise SystemExit(f"call log {args[2]} not found")
        calls = common.load_calls(args[2])
        if not calls:
            raise SystemExit(f"call log {args[2]} holds no model call")
    bundle = load_source(src)
    date = recorded_date(bundle, date_opt)
    common.write(build(bundle, calls, date=date), out, date=date)
