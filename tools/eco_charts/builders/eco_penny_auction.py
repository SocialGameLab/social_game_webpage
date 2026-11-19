"""Replay data for eco_penny_auction (penny auction: a pay-per-bid auction with one registered bid per countdown; the equilibrium
duration model of Wang and Yu, "Predicting Real-World Penny Auction Durations by Integrating Game Theory and Machine Learning",
AAAI-24).

usage: python3 -I eco_penny_auction.py SOURCE OUT_DIR [CALLS.jsonl]
       python3 -I eco_penny_auction.py --selftest
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md), a file inside
          one, or the rendered Markdown replay of one (one '## Step N . P<seat>' section per decision: what the seat saw and the
          move it made; the heading is Chinese).
  OUT_DIR where <OUT_DIR>/eco_penny_auction.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). Calls are tied to
          decisions in time order: a call belongs to the next decision of the same seat (and, when its prompt names it, the same
          round); the call whose answer is the logged move is that decision's call (earlier ones were format retries; their
          reasoning is kept above it). Its reasoning becomes the step's thinking and its prompt the observation.
  --selftest  checks the GT1 model against the numbers in rules/eco_penny_auction_rules.md; rebuilds a synthetic auction
          (opening bid placed by the game, several rounds, a winner) from the engine record and from the prompts alone, with the
          opening bid marked in every way the engine or the prompts might mark it and not at all; and checks a one-bid auction,
          an unsold old-rule auction, a round with missing decisions, the reasoning quotes (they follow the action taken) and a
          two-bid auction with reasoning (opening bid in the lede and tiles, the short-auction charts, the cap note).

The x axis is the auction's rounds (one countdown each). Under the rule of 2026-11-10 the game places the opening bid for a
random bidder before anyone decides; that bid gets its own column, "Open". It is recognised by where it sits, not by a format:
the first registered bid is the opening bid when it comes before the first round in which a seat decided (or in that round when
that round's prompts already count it); the engine's mark (no candidates) and the prompts' wording are cross-checked, and a
registered round without decision events is reported. The first recording under that rule is the 2026-11-11 one (seed 42).
The reasoning: these models end it with sentences about the answer format, which the page would show as the conclusion; the
builder then appends the decision sentences before them, quoted verbatim and labelled, and quotes one stated reason per decision
in the feed: a sentence that argues for the action taken (never for the other one, never a question), with the decision
sentence after it. The cap sentence of the prompt is checked too: whether the seats asked if it gives the outcome, and whether
they then solved the auction backwards from it by the parity of the bids left (the subgame-perfect end game the rules file
describes) or rested a pass on reading it as the outcome.
A short auction (take under a quarter of the item's value) draws price and take without the item's value and the gain bars
(the gains go in the text), and the fee chart is left out for three bids or fewer.
Each round is rebuilt from what the seats were shown (round number, price, leader, their own bids and fees, the recent bid
history) and, when the bundle carries it, from the engine's own record (summary or game_over info: history, bidders, duration,
price, ended, winner). The two are compared and every disagreement is printed as a warning; the engine's record wins. The
rewards are checked against the money flow of the rules file (Reward). The GT1 curve is the paper's Eq. 3 evaluated on the
logged item value, fee and increment, with the authors' code for where it ends (rules: Differences From the Source)."""
import json, math, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_penny_auction"
DEFAULT_DATE = "2026-11-09"
MINUS, DOT, TIMES, DIV = "−", "·", "×", "÷"
KINDS = {"bid_reg": ["Bid, registered", "c1"], "bid_lost": ["Bid, another bid drawn", "c4"], "pass": ["Pass", "muted"]}
BIDDER_COLORS = ("c1", "c2", "c3", "c4", "ink", "hold")
LABEL = r"Bidder [A-Z]{1,2}"


def warn(msg):
    print("warning: " + msg, file=sys.stderr)


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_RUN = re.compile(r"[぀-ヿ㐀-鿿가-힯]+")
_PUNCT = {"。": ". ", "、": ", ", "「": '"', "」": '"', "『": '"', "』": '"', "【": "[", "】": "]",
          "《": '"', "》": '"', "〈": "<", "〉": ">", "〜": "~"}


def en(s):
    """English only on the page: full-width forms become ASCII, CJK punctuation its ASCII counterpart, the rendered replay's
    truncation mark '(truncated)', and any other run of Chinese, Japanese or Korean characters a visible marker."""
    if not isinstance(s, str) or not re.search(r"[　-ヿ㐀-鿿가-힯＀-￯]", s):
        return s
    s = s.replace("（后面省略）", "(truncated)")
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


# ------------------------------------------------------------------------------------------------ loading
def _md_bundle(path):
    """A rendered Markdown replay: one '## Step N . P<seat>' section per decision (the heading is Chinese) with the state the
    seat saw in a fenced block, the move in backticks and, when saved, the reasoning in a ````text block; the setup and
    game_over events in fenced blocks; the header gives the rewards, seed and model."""
    text = open(path, encoding="utf-8").read()
    events, md_calls = [], []
    setup = re.search(r'```\n(\{"event": "setup".*?\})\n```', text, re.S)
    if setup:
        events.append(json.loads(setup.group(1)))
    heads = list(re.finditer(r"^## 第 (\d+) 步 · P(\d+)\s*$", text, re.M))
    for k, m in enumerate(heads):
        body = text[m.end(): heads[k + 1].start() if k + 1 < len(heads) else len(text)].split("\n## ", 1)[0]
        view = re.search(r"\*\*看到：\*\*\s*\n```[a-z]*\n(.*?)\n```", body, re.S)
        move = re.search(r"\*\*做了：\*\* `(.*)`\s*$", body, re.M)
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
    seed = re.search(r"种子 (\d+)", head)
    model = re.search(r"模型 (\S+)", head)
    summary = {"rewards": json.loads(rw.group(1)) if rw else {}}
    run = {"seed": int(seed.group(1)) if seed else None, "model": model.group(1).rstrip(",，") if model else ""}
    if not any(c["reasoning"] for c in md_calls):
        md_calls = []
    return {"events": events, "summary": summary, "run": run, "config": {}, "transcript": "", "meta": {}, "md_calls": md_calls}


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
    b = load_source(src)
    return str((b.get("meta") or {}).get("recorded_at") or (b.get("run") or {}).get("recorded_at") or DEFAULT_DATE)[:10]


# ------------------------------------------------------------------------------------------------ the paper's GT1 model
def duration_limit(v, b, d):
    """T = floor((v - b) / d): the last bid whose winner can still net >= 0 (authors' code, with a 1e-9 guard); None if d = 0."""
    return math.floor((v - b) / d + 1e-9) if d > 0 else None


def gt1_u(t, v, b, d):
    """GT1 (Eq. 3): the probability that the t-th bid is placed once t - 1 bids are in. u_1 = 1 (only auctions with a first bid
    are studied); u_t = 1 - b / (v - (t - 1) d) for 2 <= t <= T; 0 beyond T."""
    if t <= 1:
        return 1.0
    T = duration_limit(v, b, d)
    if T is not None and t > T:
        return 0.0
    den = v - (t - 1) * d
    return max(0.0, 1.0 - b / den) if den > 0 else 0.0


def gt1_summary(v, b, d, cap):
    """Mean and median number of bids under GT1 (p_n = (1 - u_{n+1}) prod_{t<=n} u_t, Eq. 1; durations past the cap lumped
    into the cap) and T."""
    T = duration_limit(v, b, d)
    last = min(T, cap) if (T is not None and cap) else (T if T is not None else cap)
    if last is None:
        return None
    p, alive = [], 1.0
    for n in range(last + 1):
        if n == last and cap and last == cap:
            p.append(alive)
            break
        un = gt1_u(n + 1, v, b, d)
        p.append(alive * (1 - un))
        alive *= un
    mean = sum(n * x for n, x in enumerate(p))
    acc, median = 0.0, last
    for n, x in enumerate(p):
        acc += x
        if acc >= 0.5 - 1e-12:
            median = n
            break
    return {"mean": mean, "median": median, "T": T, "last": last}


def per_bidder(u, m):
    """The symmetric per-bidder probability when m bidders may bid and at least one bids with probability u."""
    if m <= 0:
        return None
    return 1.0 - (1.0 - u) ** (1.0 / m) if u < 1 else 1.0


# ------------------------------------------------------------------------------------------------ reading prompts
def money(s):
    return float(s.replace(",", "").replace("$", ""))


def view_facts(view):
    """Everything a seat's prompt states about the auction, or None where it says nothing."""
    f = {}
    v = view or ""
    m = re.search(r"Item: (.+?)\. Retail price \$([\d,]+(?:\.\d+)?)", v)
    if m:
        f["item"], f["v"] = m.group(1), money(m.group(2))
    m = re.search(r"Bid fee \$([\d,]+(?:\.\d+)?) per registered bid", v)
    f["b"] = money(m.group(1)) if m else None
    m = re.search(r"raises the price by \$([\d,]+(?:\.\d+)?)", v)
    f["d"] = money(m.group(1)) if m else None
    m = re.search(r"Bidders: (\d+)", v)
    f["n_bidders"] = int(m.group(1)) if m else None
    m = re.search(r"you are (" + LABEL + r")\b", v)
    f["label"] = m.group(1) if m else None
    m = re.search(r"stopped after ([\d,]+) registered bids \(price \$([\d,]+(?:\.\d+)?)\)", v)
    f["cap"], f["cap_price"] = (int(m.group(1).replace(",", "")), money(m.group(2))) if m else (None, None)
    rl = re.search(r"^Round ([\d,]+)\b[:.]?(.*)$", v, re.M)
    f["round"] = int(rl.group(1).replace(",", "")) if rl else None
    line = rl.group(2) if rl else ""
    f["no_bid_yet"] = bool(re.search(r"no bid has been placed yet", line))
    m = re.search(r"the price is \$([\d,]+(?:\.\d+)?)", line) or re.search(r"price (?:is |now )?\$([\d,]+(?:\.\d+)?)", line)
    f["price"] = money(m.group(1)) if m else None
    m = re.search(r"([\d,]+) (?:registered )?bids? (?:so far|have been registered|registered so far|placed so far)", line)
    f["bids"] = int(m.group(1).replace(",", "")) if m else (0 if f["no_bid_yet"] else None)
    f["leader"] = None
    for sent in re.split(r"(?<=[.;])\s+", line):
        if re.search(r"\blead", sent):
            lm = re.search(LABEL, sent)
            if lm:
                f["leader"] = lm.group(0)
                break
    m = re.search(r"If your bid is registered now the price becomes \$([\d,]+(?:\.\d+)?)", v)
    f["price_if_bid"] = money(m.group(1)) if m else None
    m = re.search(r"your net gain is (−|-)?\$(-)?([\d,]+(?:\.\d+)?)", v)
    f["gain_if_last"] = ((-1 if (m.group(1) or m.group(2)) else 1) * money(m.group(3))) if m else None
    m = re.search(r"You have placed ([\d,]+) registered bid\(s\) and paid \$([\d,]+(?:\.\d+)?)", v)
    f["own_bids"], f["own_fees"] = (int(m.group(1).replace(",", "")), money(m.group(2))) if m else (None, None)
    # recent registered bids (round, bidder, price after the bid), from whatever list the prompt gives after the round line
    hist = []
    tail = v[rl.end():] if rl else ""
    for hm in re.finditer(r"[Rr]ound ([\d,]+)[^\n;$]*?(" + LABEL + r"|\byou\b)[^\n;$]*?\$([\d,]+\.\d+)", tail):
        who = hm.group(2)
        entry = tail[hm.start():].split(";", 1)[0].split("\n", 1)[0]       # the whole entry, notes after the price included
        hist.append({"round": int(hm.group(1).replace(",", "")), "bidder": f["label"] if who == "you" else who, "price": money(hm.group(3)),
                     "opening": bool(re.search(r"opening|at random|placed by the game|automatic", entry, re.I))})
    f["history"] = hist
    return f


def parse_move(s):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (s or "").strip())
    try:
        a = json.loads(s)
    except ValueError:
        return None
    return a.get("action_id") if isinstance(a, dict) else None


def round_of(prompt):
    m = re.search(r"^Round ([\d,]+)\b", prompt or "", re.M)
    return int(m.group(1).replace(",", "")) if m else None


# ------------------------------------------------------------------------------------------------ quoting the reasoning
# The page's "Conclusion of its reasoning" box shows the last paragraph over 40 characters. These models end their reasoning
# with sentences about the answer format ("Let's ensure the JSON format exactly"), so the builder finds the decision itself:
# the sentences before the format talk, quoted verbatim, and a one-sentence reason for the feed.
_FORMAT = re.compile(r"\b(JSON|output|outputs|respond|response|format|payload|action_id|markdown|extra text|no extra|exactly one|code block)\b", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"'])\s+")
# what makes a sentence a reason in this game: who opens and who is outbid, dominance, the parity of the bids left, timing,
# a stated cause
_PARITY = (r"\bparity\b|\b(?:odd|even)\s+(?:k|n|r|t|number|remaining|bids?|states?)\b|"
           r"\b(?:k|n|r|t|remaining)\s*=?\s*\d+\s*(?:is\s+)?\(?(?:odd|even)\b|\b(?:for|at)\s+(?:odd|even)\b")
_REASON = [(r"first bidder|first-mover|first mover|first to bid|opens? the auction", 2),
           (r"outbid|a target\b|sucker|triggers others|lose (?:the |your |my |our |its )?(?:fee|leadership)", 2),
           (_PARITY, 1.5),
           (r"\bdominant\b|\bdominates\b|strictly better|at least as good|positive expected|negative expected", 1),
           (r"\bpremature\b|\btoo early\b|\bwait (?:until|for)\b", 1),
           (r"\bbecause\b|\bso\b|\bhence\b|\btherefore\b|\bthus\b|\bsince\b", .5)]
OPENER_THEME = re.compile(r"first bidder|first-mover|first mover|first to bid|outbid|a target\b|sucker|triggers others", re.I)
_BIDPASS = re.compile(r"\b(bid|bids|bidding|pass|passes|passing)\b", re.I)
# which action a sentence argues for: an action word followed (no punctuation between, at most five words) by better /
# optimal / dominates ...; or prefer / should / the answer is ... followed by an action word; or a whole-sentence decision
# ("So pass."). "bidding strictly dominates passing" argues for bidding: the action after "dominates" or "better than" loses.
_ACT = r"(bid|bids|bidding|pass|passes|passing)"
_GOOD = r"(?:better|best|optimal|preferable|dominant|dominates|dominate|superior|profitable|correct)"
_BAD = r"(?:worse|bad|negative|premature|dominated|suboptimal|a mistake|a loss)"
_FAV_AFTER = re.compile(rf"\b{_ACT}\b((?: +[\w$.=]+){{0,5}}?) +(?:{_GOOD}|(?P<bad>{_BAD}))\b", re.I)
_FAV_BEFORE = re.compile(rf"\b(?:prefer|prefers|should|better to|best to|optimal to|(?:optimal )?(?:action|answer|choice|decision) (?:is|should be)|"
                         rf"decide to|choose to|(?:we|i)(?: \([^)]*\))? (?:will|must|would rather))(?: \w+){{0,2}}? {_ACT}\b", re.I)
_FAV_WHOLE = re.compile(rf"^(?:so|thus|therefore|hence|then)?[, ]*(?:we |i )?(?:should |will |must )?{_ACT}\W*$", re.I)


def _act_of(word):
    return "bid" if word.lower().startswith("bid") else "pass"


def stance(s):
    """(actions the sentence argues for, actions it argues against). Conditions ("if we bid ...") and negations count for
    nothing and for the other side respectively."""
    pro, con = set(), set()
    def conditional(m):
        return bool(re.search(r"\b(if|whether|when|unless)\b", re.split(r"[,;:]", s[:m.start()])[-1], re.I))
    for m in _FAV_AFTER.finditer(s):
        if conditional(m):
            continue
        neg = re.search(r"\b(not|never|no)\b|n't", m.group(2) or "")
        (con if bool(m.group("bad")) != bool(neg) else pro).add(_act_of(m.group(1)))
    for m in _FAV_BEFORE.finditer(s):
        if conditional(m):
            continue
        neg = re.search(r"\b(not|never)\b|n't", m.group(0))
        (con if neg else pro).add(_act_of(m.group(1)))
    m = _FAV_WHOLE.match(s.strip())
    if m:
        pro.add(_act_of(m.group(1)))
    return pro, con


def supports(s, action):
    """+1 when the sentence argues for the action (or against the other), -1 when it argues for the other, 0 otherwise."""
    if action not in ("bid", "pass"):
        return 0
    other = "pass" if action == "bid" else "bid"
    pro, con = stance(s)
    yes, no = action in pro or other in con, other in pro or action in con
    return 0 if yes == no else (1 if yes else -1)


def _paras(t):
    return [p.strip() for p in re.split(r"\n\s*\n", t or "") if p.strip()]


def _sents(p):
    return [s.strip() for s in _SENT.split(p) if s.strip()]


def _is_format(s):
    return bool(_FORMAT.search(s)) and len(s) < 200


def _is_tail_noise(s):
    """A short closing sentence with no bid or pass in it ("No extra.", "Done.")."""
    return len(s) < 25 and not _BIDPASS.search(s)


def _is_question(s):
    return "?" in s or bool(re.match(r"(?:but )?(?:what if|wait\b)", s, re.I))


def ends_in_format_talk(t):
    """True when the paragraph the page would show as the conclusion only talks about the answer format (closing short
    sentences with no bid or pass in them count as format talk)."""
    ps = [p for p in _paras(t) if len(p) > 40]
    if not ps:
        return False
    ss = _sents(ps[-1])
    while ss and (_is_format(ss[-1]) or _is_tail_noise(ss[-1])):
        ss.pop()
    return not ss


def decision_quote(t, lo=340, hi=580):
    """The last sentences of the reasoning, about lo-hi characters, without the sentences about the answer format and
    without short closing sentences that name neither bid nor pass."""
    out, total = [], 0
    for p in reversed(_paras(t)):
        for s in reversed([s for s in _sents(p) if not _is_format(s)]):
            if not out and _is_tail_noise(s):
                continue
            if total and total + len(s) + 1 > hi:
                return " ".join(out)
            out.insert(0, s)
            total += len(s) + 1
            if total >= lo:
                return " ".join(out)
    return " ".join(out)


def reason_quote(t, action=None, tail=0.4, only=None):
    """The stated reason for the decision: one sentence (30-220 characters) from the last 40% of the reasoning, plus the
    decision sentence right after it when there is one ("Thus the answer is definitely bid."). Questions and sentences that
    argue for the other action are left out (a long reasoning weighs both). Score: the strongest reason pattern (_REASON;
    +0.5 for a stated cause), +1.5 when the sentence argues for the action taken, +1.5 when the next sentence states that
    decision, +1 in the last paragraph that is not about the answer format; the latest wins ties. With only (a regex), only
    sentences that match it are candidates. (score, quote) or None."""
    t = t or ""
    paras, pos, cut = [], 0, len(t) * (1 - tail)
    for p in _paras(t):
        i = t.find(p, pos)
        pos = i + len(p)
        paras.append((i, p))
    last = next((k for k in range(len(paras) - 1, -1, -1) if any(not _is_format(s) and not _is_tail_noise(s) for s in _sents(paras[k][1]))), None)
    best = None
    for k, (i, p) in enumerate(paras):
        if i < cut:
            continue
        ss = _sents(p)
        for j, s in enumerate(ss):
            if _is_format(s) or _is_question(s) or not 30 <= len(s) <= 220 or not re.search(
                    r"\b(bid|bids|bidder|bidders|bidding|pass|passing|leader|outbid)\b", s, re.I) or (only is not None and not only.search(s)):
                continue
            sup = supports(s, action)
            if sup < 0:
                continue
            nxt = ss[j + 1] if j + 1 < len(ss) else ""
            follow = bool(nxt) and not _is_format(nxt) and not _is_question(nxt) and len(nxt) <= 80 and supports(nxt, action) > 0
            hits = [w for pat, w in _REASON if re.search(pat, s, re.I)]
            base = max([w for w in hits if w > .5] or [0]) + (.5 if .5 in hits else 0)
            if not base and not sup:
                continue
            sc = base + 1.5 * sup + (1.5 if follow else 0) + (1 if k == last else 0)
            if best is None or sc >= best[0]:
                best = (sc, s + (" " + nxt if follow else ""))
    return best


def first_question(t, pat, maxlen=120):
    """The first question (a sentence ending in '?', at most maxlen characters) that matches pat."""
    for p in _paras(t):
        for s in _sents(p):
            if s.endswith("?") and len(s) <= maxlen and re.search(pat, s, re.I):
                return s
    return None


# ------------------------------------------------------------------------------------------------ calls -> steps
def attach_calls(steps, calls):
    """Tie calls to steps in time order. For each step the window is the run of calls from the pointer whose seat and round
    agree with it (unknown values agree with anything); the first call in it whose answer is the logged move is the one, and
    earlier calls in the window were failed attempts whose reasoning is kept above it. Without a matching answer the last call
    of the window is used and the step says the game applied another move (the fallback is pass)."""
    ptr = 0

    def fits(k, st, strict=False):
        seat, rnd = calls[k].get("seat"), round_of(calls[k].get("prompt"))
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
        j = ptr
        while j < len(calls) and fits(j, st):
            j += 1
            if j < len(calls) and calls[j].get("seat") is None and round_of(calls[j].get("prompt")) is None:
                break
        window = list(range(ptr, j))
        if not window:
            continue
        hit = next((k for k in window if parse_move(calls[k].get("answer")) == st["action"]), None)
        use = hit if hit is not None else window[-1]
        parts = [f"[Earlier attempt, answer: {calls[k]['answer'][:300]}]\n{calls[k].get('reasoning') or '(no reasoning returned)'}"
                 for k in range(ptr, use)]
        c = calls[use]
        parts.append(("[Final attempt]\n" if parts else "") + (c.get("reasoning") or ("(no reasoning returned)" if parts else "")))
        st["_raw"] = c.get("reasoning") or ""
        if st["_raw"] and ends_in_format_talk(st["_raw"]):
            dq = decision_quote(st["_raw"])
            if dq:          # one paragraph (single line break), so the page's conclusion box shows it instead of the format talk
                parts.append("[Its decision, quoted verbatim from the end of the reasoning above; answer-format sentences left out]\n" + dq)
        if hit is None:
            parts.append(f"[The game applied '{st['action']}' instead of this answer: {c['answer'][:300]}]")
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        if c.get("prompt"):
            st["state"] = en(common.state_part(c["prompt"]))
        ptr = use + 1
    if ptr < len(calls):
        warn(f"{len(calls) - ptr} calls after the last decision were not tied to any step")


# ------------------------------------------------------------------------------------------------ text helpers
def usd(v, signed=False):
    if v is None:
        return "-"
    s = f"${abs(v):,.2f}"
    if v < -5e-10:
        return MINUS + s
    return ("+" if signed and v > 5e-10 else "") + s


def r2(v):
    return None if v is None else round(v + 0.0, 2)


def score(v):
    return "0" if abs(v) < 5e-10 else (MINUS if v < 0 else "") + f"{abs(v):.3f}"


def and_join(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1] if xs else ""


NUMW = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def numword(n):
    return NUMW[n] if 0 <= n < len(NUMW) else str(n)


def pct(v):
    return f"{100 * v:.1f}%"


def nbid(k):
    return f"{k:,} bid" + ("" if k == 1 else "s")


def noun_of(item):
    return item.replace("a ", "", 1) if item.startswith("a ") else item


# ------------------------------------------------------------------------------------------------ build
def build(bundle_dir, calls=None):
    b = load_source(bundle_dir)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    if not calls:
        calls = b.get("md_calls") or []
    events, summary, run, cfg = b["events"], b.get("summary") or {}, b.get("run") or {}, b.get("config") or {}
    over = next((e for e in events if e.get("event") == "game_over"), {})
    info = dict(summary.get("info") or {})
    for src in (over.get("info"), summary.get("variables"), over.get("variables")):
        for k, v in (src or {}).items():
            info.setdefault(k, v)

    moves = [e for e in events if e.get("event") == "move"]
    if not moves:
        raise SystemExit("no decision in the game log")
    facts = []
    for e in moves:
        f = view_facts(e.get("view") or "")
        f["seat"], f["action"], f["_view"] = int(e["actor"]), parse_move(e.get("move")) or "pass", e.get("view") or ""
        facts.append(f)
    if run.get("prompt") and not any(f.get("v") for f in facts):
        facts[0].update({k: v for k, v in view_facts(run["prompt"].get("text") or "").items() if v is not None})

    def first(key):
        return next((f[key] for f in facts if f.get(key) is not None), None)
    params = info.get("params") or {}
    v = float(params.get("retail_price") or first("v") or 0)
    fee = float(params.get("bid_fee") if params.get("bid_fee") is not None else first("b"))
    inc = float(params.get("bid_increment") if params.get("bid_increment") is not None else first("d"))
    cap = int(info.get("cap") or params.get("max_rounds") or first("cap") or 0) or None
    item = first("item") or params.get("product") or "the item"
    if not v:
        raise SystemExit("the retail price is in neither the prompts nor the engine record")

    # ---- bidders: seats (label from their own prompts) and, from the engine record, scripted bidders
    seat_label = {}
    for f in facts:
        if f.get("label"):
            if seat_label.get(f["seat"], f["label"]) != f["label"]:
                warn(f"seat {f['seat']} is called {seat_label[f['seat']]} and {f['label']}")
            seat_label.setdefault(f["seat"], f["label"])
    rec_bidders = info.get("bidders") or []
    for bd in rec_bidders:
        if bd.get("seat") is not None and bd.get("label"):
            if seat_label.get(int(bd["seat"]), bd["label"]) != bd["label"]:
                warn(f"seat {bd['seat']}: prompts say {seat_label[int(bd['seat'])]}, engine record {bd['label']}")
            seat_label[int(bd["seat"])] = bd["label"]
    n_seats = max([f["seat"] for f in facts] + [int(k) for k in (summary.get("rewards") or {}).keys()] + list(seat_label)) + 1
    for s in range(n_seats):
        seat_label.setdefault(s, f"Seat P{s}")
    scripted = [bd["label"] for bd in rec_bidders if bd.get("seat") is None and bd.get("label")]
    n_bidders = int(first("n_bidders") or len(rec_bidders) or n_seats)
    labels = [seat_label[s] for s in range(n_seats)] + scripted
    seat_of = {lab: s for s, lab in seat_label.items()}
    if len(labels) < n_bidders:
        warn(f"{n_bidders} bidders in the auction, {len(labels)} known by name (scripted bidders unnamed without the engine record)")

    # ---- registered bids: the engine record if present, else what the prompts show
    rec_hist = []
    for h in info.get("history") or []:
        cand = h.get("candidates")
        ncand = len(cand) if isinstance(cand, (list, tuple)) else (int(cand) if cand is not None else None)
        rec_hist.append({"round": int(h["round"]) if h.get("round") is not None else None, "bidder": h.get("bidder"),
                         "price": float(h["price"]) if h.get("price") is not None else None,
                         "candidates": cand, "opening": ncand == 0, "marked": ncand == 0})
    rounds_seen = sorted({f["round"] for f in facts if f.get("round") is not None})
    if not rounds_seen:
        raise SystemExit("no round number in the prompts")
    by_round = {}
    for f in facts:
        by_round.setdefault(f["round"], []).append(f)
    start = {}                                   # what the prompts of each round state about its start
    for r, fs in by_round.items():
        for key in ("price", "bids", "leader", "no_bid_yet"):
            vals = {f[key] for f in fs if f.get(key) is not None}
            if len(vals) > 1:
                warn(f"round {r}: seats were shown different {key}: {sorted(map(str, vals))}")
            if vals:
                start.setdefault(r, {})[key] = sorted(vals, key=str)[0]
        st = start.setdefault(r, {})
        if st.get("price") is None:
            pib = next((f["price_if_bid"] for f in fs if f.get("price_if_bid") is not None), None)
            if pib is not None:
                st["price"] = round(pib - inc, 10)
        if st.get("bids") is None and st.get("price") is not None and inc > 0:
            st["bids"] = int(round(st["price"] / inc))
        if st.get("no_bid_yet"):
            st["leader"] = None

    # ---- the opening bid. Under the rule of 2026-11-10 the game places it before anyone decides. It is recognised by its
    # place in the auction, not by a format: the first registered bid is the opening bid when it comes before the first round
    # in which a seat decided (or in that round, when that round's prompts already count a bid). The engine marking it (no
    # candidates) or the prompts calling it the opening bid are taken too, and a disagreement between them is printed.
    r0 = rounds_seen[0]

    def before_decisions(rnd):
        return rnd is not None and (rnd < r0 or (rnd == r0 and (start.get(r0, {}).get("bids") or 0) >= 1))
    p_hist, seen_h = [], set()                   # every registered bid the prompts list, once each, in auction order
    for f in facts:
        for h in f.get("history") or []:
            key = (h["round"], h["bidder"], h["price"])
            if key not in seen_h:
                seen_h.add(key)
                p_hist.append(h)
    p_hist.sort(key=lambda h: (h["price"], h["round"]))
    opening = None
    if rec_hist:
        h0 = rec_hist[0]
        if before_decisions(h0["round"]) or h0["marked"]:
            if not before_decisions(h0["round"]):
                warn(f"the engine marks the bid of round {h0['round']} as placed by the game, but the seats decided from round {r0} on")
            elif not h0["marked"] and h0["candidates"] is not None:
                warn(f"the bid of round {h0['round']} comes before any seat decided, so it is taken as the opening bid, "
                     f"although the engine lists candidates for it: {h0['candidates']!r}")
            opening = dict(h0, opening=True)
    if opening is None and p_hist:
        h0 = p_hist[0]
        if before_decisions(h0["round"]) or h0["opening"]:
            if not before_decisions(h0["round"]):
                warn(f"the prompts call the bid of round {h0['round']} the opening bid, but the seats decided from round {r0} on")
            opening = dict(h0, opening=True)
        elif any(h["opening"] for h in p_hist):
            warn("the prompts call a bid the opening bid that is not the first registered bid; ignored")
    if opening is None and start.get(r0, {}).get("bids") == 1 and start[r0].get("leader"):
        opening = {"round": r0 - 1 if r0 > 1 else None, "bidder": start[r0]["leader"], "price": inc, "opening": True}
    if opening and rec_hist and p_hist and not rec_hist[0]["marked"] and not before_decisions(rec_hist[0]["round"]):
        warn("the prompts show an opening bid the engine record does not")
    is_opening = lambda h: opening is not None and h["round"] == opening.get("round") and h["bidder"] == opening.get("bidder") and (
        h["price"] is None or opening.get("price") is None or abs(h["price"] - opening["price"]) < 1e-9)

    reg = {}                                     # round -> registered bidder label, from the prompts
    for h in p_hist:
        if not is_opening(h) and h["round"] not in reg:
            reg[h["round"]] = h["bidder"]
    for k, r in enumerate(rounds_seen):        # the leader at the start of round r + 1 placed the bid of round r
        nxt = rounds_seen[k + 1] if k + 1 < len(rounds_seen) else None
        if nxt == r + 1 and start.get(nxt, {}).get("leader"):
            if reg.get(r, start[nxt]["leader"]) != start[nxt]["leader"]:
                warn(f"round {r}: history line says {reg[r]}, next round's leader is {start[nxt]['leader']}")
            reg[r] = start[nxt]["leader"]
    if rec_hist:
        rec_reg = {h["round"]: h["bidder"] for h in rec_hist if not is_opening(h)}
        for r, lab in reg.items():
            if r in rec_reg and rec_reg[r] != lab:
                warn(f"round {r}: the prompts point to {lab}, the engine record to {rec_reg[r]}")
        reg = {**reg, **rec_reg}

    # ---- slots on the x axis: the opening bid (if any), then every round from the first to the last one seen or recorded
    last_round = max(rounds_seen + [r for r in reg if r is not None])
    first_round = min(rounds_seen + [r for r in reg if r is not None])
    slots = []
    if opening:
        slots.append({"round": opening.get("round"), "opening": True, "registered": opening.get("bidder"), "moves": {}})
    for r in range(first_round, last_round + 1):
        slots.append({"round": r, "opening": False, "registered": reg.get(r), "moves": {}})
    xi = {s["round"]: i for i, s in enumerate(slots) if not s["opening"]}
    for f in facts:
        slots[xi[f["round"]]]["moves"][f["seat"]] = f["action"]

    # ---- walk the auction: price, leader and bids before each slot; who bid; fees
    ended = info.get("ended")
    bids, leader = 0, None
    fees_after = {lab: [] for lab in labels}
    nbids = {lab: 0 for lab in labels}
    for i, sl in enumerate(slots):
        sl["bids_before"], sl["price_before"], sl["leader"] = bids, round(bids * inc, 10), leader
        stt = start.get(sl["round"], {}) if not sl["opening"] else {}
        if stt.get("bids") is not None and stt["bids"] != bids:
            warn(f"round {sl['round']}: prompts say {stt['bids']} bids so far, the rebuilt auction {bids}")
        if stt.get("price") is not None and abs(stt["price"] - sl["price_before"]) > 1e-6:
            warn(f"round {sl['round']}: prompts say price {stt['price']}, the rebuilt auction {sl['price_before']}")
        if stt and (stt.get("leader") or None) != leader and (stt.get("leader") or stt.get("no_bid_yet")):
            warn(f"round {sl['round']}: prompts say leader {stt.get('leader')}, the rebuilt auction {leader}")
        bidders_here = [seat_label[s] for s, a in sl["moves"].items() if a == "bid"]
        if sl["registered"] is None and bidders_here and not sl["opening"]:
            if len(bidders_here) == 1 and n_bidders == n_seats:
                sl["registered"] = bidders_here[0]
            elif i == len(slots) - 1 and ended not in ("cap",):
                warn(f"round {sl['round']}: {and_join(bidders_here)} bid but the registered bid is not in the log")
        if sl["registered"] is not None and not sl["opening"] and sl["registered"] in seat_of:
            if sl["moves"].get(seat_of[sl["registered"]]) not in (None, "bid"):
                warn(f"round {sl['round']}: {sl['registered']} is registered but its seat passed")
        if not sl["opening"]:                    # every seat that does not lead decides in every round
            missing = [seat_label[s] for s in range(n_seats) if seat_label[s] != leader and s not in sl["moves"]]
            if sl["registered"] and not sl["moves"]:
                warn(f"round {sl['round']}: a bid by {sl['registered']} is registered but no seat has a decision event in that round")
            elif missing:
                warn(f"round {sl['round']}: no decision event for {and_join(missing)}")
        if sl["registered"]:
            bids += 1
            leader = sl["registered"]
            nbids[leader] = nbids.get(leader, 0) + 1
        for lab in labels:
            fees_after[lab].append(nbids.get(lab, 0) * fee)
        sl["price_after"] = round(bids * inc, 10)
    duration = int(info["duration"]) if info.get("duration") is not None else bids
    if duration != bids:
        warn(f"the engine record says {duration} bids, the rebuilt auction {bids}")
    price = float(info["price"]) if info.get("price") is not None else round(bids * inc, 10)
    winner = info.get("winner") if info.get("winner") is not None else leader
    if not ended:
        ended = "unsold" if bids == 0 else ("cap" if cap and bids >= cap else "no_bid")
    sold = winner is not None and ended != "unsold"

    # own bids and fees as the seats saw them, against the rebuilt auction
    for f in facts:
        lab = seat_label[f["seat"]]
        i = xi[f["round"]]
        if f.get("own_bids") is not None:
            before = nbids_before = sum(1 for s in slots[:i] if s["registered"] == lab)
            if f["own_bids"] != before:
                warn(f"round {f['round']}: {lab} was told {f['own_bids']} own bids, the rebuilt auction has {nbids_before}")
            if abs(f["own_fees"] - before * fee) > 1e-6:
                warn(f"round {f['round']}: {lab} was told it paid {f['own_fees']}, {before} bids x {fee} = {before * fee}")

    # ---- rewards against the money flow (rules: Reward)
    raw = summary.get("rewards") or over.get("rewards") or {}
    rewards = [float(raw.get(str(s), raw.get(s, 0.0))) for s in range(n_seats)]
    net = {}
    for lab in labels:
        net[lab] = ((v - price) if sold and lab == winner else 0.0) - nbids.get(lab, 0) * fee
    for s in range(n_seats):
        want = max(-1.0, min(1.0, net[seat_label[s]] / v))
        if abs(want - rewards[s]) > 1e-6:
            warn(f"{seat_label[s]} (P{s}): reward {rewards[s]} logged, {want} by the rules")
    fees_total = bids * fee
    revenue = fees_total + (price if sold else 0.0)
    for key, mine in (("fees_total", fees_total), ("revenue", revenue)):
        if info.get(key) is not None and abs(float(info[key]) - mine) > 1e-6:
            warn(f"{key}: engine record {info[key]}, rebuilt {mine}")

    # ---- the paper's GT1 model on this item
    g1 = gt1_summary(v, fee, inc, cap)
    m_gt1 = ((info.get("models") or {}).get("gt1") or {})
    if g1 and m_gt1.get("expected_duration") is not None and abs(float(m_gt1["expected_duration"]) - g1["mean"]) > 1e-6:
        warn(f"GT1 mean duration: engine record {m_gt1['expected_duration']}, computed {g1['mean']}")

    # ---- steps
    steps = []
    for e, f in zip(moves, facts):
        sl = slots[xi[f["round"]]]
        lab = seat_label[f["seat"]]
        if f["action"] == "bid":
            kind = "bid_reg" if sl["registered"] == lab else "bid_lost"
            what = ("bids; its bid is registered" if kind == "bid_reg" else
                    f"bids; {sl['registered']}'s bid is drawn" if sl["registered"] else "bids; not registered")
        else:
            kind, what = "pass", "passes"
        steps.append({"seat": f["seat"], "x": xi[f["round"]], "round": f["round"], "action": f["action"], "kind": kind,
                      "head": f"Round {f['round']} {DOT} price {usd(sl['price_before'])} {DOT} {what}",
                      "act": en(json.dumps({"action_id": f["action"], "payload": {}})),
                      "state": en(common.state_part(e.get("view") or "")), "thinking": ""})
    attach_calls(steps, calls)
    has_reason = any(s.get("thinking") for s in steps)
    raws = [s.get("_raw") or "" for s in steps]           # the final call's reasoning, for the quotes below
    acts = [s["action"] for s in steps]
    reasons = [reason_quote(r, a) if r else None for r, a in zip(raws, acts)]
    for s in steps:
        s.pop("round", None)
        s.pop("action", None)
        s.pop("_raw", None)

    # ---- x axis
    n = len(slots)
    xl = ["Open" if s["opening"] else f"R{s['round']}" for s in slots]
    one = n == 1 and not slots[0]["opening"]
    r_first = slots[0]

    # ---- chart: what a bid could win (bars when the auction is one or two rounds long, lines over rounds otherwise)
    gain = []                                   # v - price after this round's bid - fee: the net of a bid nobody tops
    for sl in slots:
        stated = next((f["gain_if_last"] for f in by_round.get(sl["round"], []) if f.get("gain_if_last") is not None), None)
        mine = v - (sl["price_before"] + inc) - fee
        if stated is not None and abs(stated - mine) > 0.005 + 1e-9:
            warn(f"round {sl['round']}: prompt says a bid nets {stated}, computed {mine}")
        gain.append(r2(stated if stated is not None and not sl["opening"] else mine))
    take = [r2(sl["bids_before"] * (fee + inc)) for sl in slots]    # fees paid so far plus the price
    charts = []
    if bids == 0:                                # nobody opened (the rule before 2026-11-10): what opening risked
        g0 = gain[-1]
        u2 = gt1_u(2, v, fee, inc)
        odds = 0 < u2 < 1
        charts.append({"title": "Opening the auction: what it could have won or lost",
                       "sub": (f"Round {r_first['round']}, price {usd(0)}, no leader yet. Opening costs the {usd(fee)} fee and raises the price to "
                               f"{usd(inc)}. Had nobody bid after it, the opener would have bought the {usd(v)} {noun_of(item)} for {usd(inc)} and "
                               f"netted {usd(g0, True)}, the figure every bidder's prompt gave; outbid, it would have lost the {usd(fee)} (bidding "
                               "again is worth $0 on average under GT1). "
                               + (f"Under the paper's GT1 equilibrium another bid follows {pct(u2)} of the time: {pct(1 - u2)} {TIMES} {usd(g0)} = "
                                  f"{usd((1 - u2) * g0)} against {pct(u2)} {TIMES} {usd(fee)} = {usd(u2 * fee)}, so opening is worth exactly "
                                  f"{usd((1 - u2) * g0 - u2 * fee + 0.0)}." if odds else "")),
                       "height": 210, "fmt": "money", "zero": True, "lines": [],
                       "bars": [{"name": f"Nobody bids after it: {usd(g0, True)}" + (f" (GT1: {pct(1 - u2)})" if odds else ""), "color": "c3", "v": [r2(g0)] * n},
                                {"name": f"Outbid: {usd(-fee)}, the fee" + (f" (GT1: {pct(u2)})" if odds else ""), "color": "c2", "v": [r2(-fee)] * n}]})
    else:
        T_ = duration_limit(v, fee, inc) if inc > 0 else None
        head = (f"At the start of each round. Price: {usd(inc)} per registered bid so far. Auctioneer's take: every fee paid so far "
                f"({usd(fee)} per bid) plus the price. ")
        net_def = f"({usd(v)} {MINUS} new price {MINUS} {usd(fee)} fee)"
        turns = f"; it turns negative after bid {T_:,}" if T_ is not None else ""
        tail_ = (f"The auction ended at {usd(price)} after {nbid(bids)}; the take reached {usd(revenue)}, "
                 f"{revenue / v:.2f} {TIMES} the item's value.")
        price_v = [r2(s["price_before"]) for s in slots]
        if revenue < 0.25 * v:
            # a short auction: on an axis up to the item's value price and take would be flat lines and the gain bars would
            # all look the same, so the gains are given in words and price and take get the full height
            gains_txt = (" / ".join(usd(g_) for g_ in gain) + " (" + " / ".join(xl) + ")" if n <= 6 else
                         f"{usd(gain[0])} at {xl[0]} down to {usd(gain[-1])} at {xl[-1]}")
            charts.append({"title": "Price and the auctioneer's take, round by round",
                           "sub": head + f"A bid placed in these rounds would have netted its bidder {gains_txt} if nobody bid after it "
                                         f"{net_def}, far more than the price and the take drawn here" + turns + ". " + tail_,
                           "height": 220, "fmt": "price2",
                           "lines": [{"name": "Auctioneer's take ($)", "color": "c2", "v": take, "end": True, "dots": n <= 12},
                                     {"name": "Price ($)", "color": "ink", "v": price_v, "end": True, "dots": n <= 12}]})
        else:
            charts.append({"title": "Price, the auctioneer's take, and what a bid could still win",
                           "sub": head + ("Bars: what a bid placed in that round would net its bidder if nobody bid after it "
                                          f"{net_def}{turns}. ") + tail_,
                           "height": 250, "fmt": "price2",
                           "bars": [{"name": "Net gain if nobody bids after it ($)", "color": "c3", "v": gain}],
                           "lines": [{"name": "Retail price ($)", "color": "hold", "dash": True, "v": [r2(v)] * n},
                                     {"name": "Auctioneer's take ($)", "color": "c2", "v": take},
                                     {"name": "Price ($)", "color": "ink", "v": price_v}]})

    # ---- chart: each bidder's choice per round
    rows, cells = [], []
    for lab in labels:
        s_ = seat_of.get(lab)
        row = []
        for sl in slots:
            if sl["opening"]:
                row.append({"c": "c2", "t": "Open", "tip": f"The game placed the opening bid for {lab} at random; price {usd(inc)}"}
                           if sl["registered"] == lab else None)
                continue
            if sl["leader"] == lab:
                row.append({"c": "c3", "t": "Lead", "o": .75, "tip": f"Leading at {usd(sl['price_before'])}: no turn this round"})
                continue
            a = sl["moves"].get(s_) if s_ is not None else None
            if a == "bid" or (s_ is None and sl["registered"] == lab):
                if sl["registered"] == lab:
                    row.append({"c": "c1", "t": "Bid", "tip": f"Bid and was registered: paid {usd(fee)}, price {usd(sl['price_after'])}, now leads"})
                else:
                    row.append({"c": "c4", "t": "Bid", "tip": (f"Bid, but {sl['registered']}'s bid was drawn: paid nothing" if sl["registered"]
                                                              else "Bid; which bid was registered is not in the log")})
            elif a == "pass":
                row.append({"c": "muted", "t": "Pass", "tip": f"Passed at {usd(sl['price_before'])}"})
            elif s_ is None:
                row.append({"c": "hold", "t": "?", "o": .35, "tip": "Scripted bidder: its choice is not in the log unless its bid was registered"})
            else:
                row.append(None)
        rows.append(lab + ("" if lab in seat_of else " (scripted)"))
        cells.append(row)
    legend = [["Bid, registered", "c1"], ["Bid, another bid drawn", "c4"], ["Pass", "muted"], ["Leading (no turn)", "c3"]]
    if any(sl["opening"] for sl in slots):
        legend.insert(0, ["Opening bid placed by the game", "c2"])
    if scripted:
        legend.append(["Not logged (scripted)", "hold"])
    used = {c["c"] for row in cells for c in row if c}
    legend = [it for it in legend if it[1] in used]
    if one:
        last = slots[0]
        passed = [seat_label[s] for s, a in sorted(last["moves"].items()) if a == "pass"]
        all_passed = len(passed) == len(last["moves"])
        sub = (f"Round {last['round']} is the only round: nobody led yet, so all {numword(len(last['moves']))} bidders decided at the same time, "
               "none seeing the others' choices. " + (f"{and_join(passed)} all passed, so the countdown ran out with no bid and the auction ended "
                                                      "with the item unsold." if all_passed else ""))
        r_steps = [k for k, s_ in enumerate(steps) if s_["x"] == 0]
        # with no leader yet the question is who opens: when every seat gave a reason about the first bidder being outbid,
        # that reason is the one quoted
        themed = {k: reason_quote(raws[k], acts[k], only=OPENER_THEME) if raws[k] else None for k in r_steps}
        if all_passed and r_steps and all(themed.values()):
            for k in r_steps:
                reasons[k] = themed[k]
            sub += (f" {'All ' + numword(len(r_steps)) if len(r_steps) > 2 else 'Both'} reasoned that the first bidder is the one who gets "
                    "outbid, so each waited for someone else to open; each one's own words are quoted in the feed.")
        elif any(reasons[k] for k in r_steps):
            sub += " Each bidder's stated reason is quoted in the feed."
    else:
        sub = ("One column per round (one countdown). Every bidder except the leader decides at the same time; if several bid, one bid is "
               f"drawn at random and only that bidder pays the {usd(fee)} fee and becomes the leader. Hover a cell for the price.")
    charts.append({"title": "Each bidder's choice, round by round", "sub": sub, "height": 140,
                   "grid": {"rows": rows, "cells": cells, "legend": legend, "rowHeight": 34 if n <= 3 else 28 if len(rows) <= 4 else 22}})

    # ---- chart: fees paid by each bidder (only when somebody paid)
    # (left out for three bids or fewer: the lines are flat and lie on top of each other, and the Winner and Scores tiles and
    # the feed's final money already give every bidder's fees)
    if bids > 3:
        lines, dashed = [], []
        for k, lab in enumerate(labels):
            vv = [r2(x) for x in fees_after[lab]]
            # a line that runs on top of an earlier one for two slots or more is dashed, so the one under it shows through
            over = any(any(vv[i] == o["v"][i] and vv[i + 1] == o["v"][i + 1] for i in range(n - 1)) for o in lines)
            lines.append({"name": lab, "color": BIDDER_COLORS[k % len(BIDDER_COLORS)], "v": vv, "step": True, "end": True,
                          **({"dash": True} if over else {})})
            if over:
                dashed.append(lab)
        charts.append({"title": "Fees paid by each bidder",
                       "sub": (f"Running total after each round: {usd(fee)} for every registered bid; bids that were not drawn cost nothing. "
                               + (f"{winner} also paid the final price of {usd(price)} for the item. " if sold else "")
                               + (f"Dashed where {and_join(dashed)} paid as much as a bidder drawn before {'it' if len(dashed) == 1 else 'them'}."
                                  if dashed else "")).strip(),
                       "height": 210, "fmt": "price2", "lines": lines})

    # ---- chart: the paper's GT1 model against what the seats did. The model predicts from the second bid on (Eq. 3); the first
    # bid is the paper's sample-selection assumption (u_1 = 1, only auctions with a first bid are studied), not a prediction, so
    # no model value is drawn for it (the opening bid placed by the game, or round 1 under the old rule). Without any slot the
    # model predicts, the chart is left out and the tiles say why.
    share, someone, each, tally = [], [], [], []
    for sl in slots:
        k1 = sl["bids_before"] + 1
        dec = list(sl["moves"].values())
        tally.append((sum(a == "bid" for a in dec), len(dec)))
        share.append(round(100 * tally[-1][0] / len(dec), 2) if dec else None)
        if sl["opening"] or k1 < 2:
            someone.append(None)
            each.append(None)
            continue
        u = gt1_u(k1, v, fee, inc)
        m = n_bidders - (1 if sl["leader"] else 0)
        someone.append(round(100 * u, 2))
        each.append(round(100 * per_bidder(u, m), 2) if m else None)
    if any(x is not None for x in someone):
        short = n <= 3
        same = all(a_ == b_ for a_, b_ in zip(someone, each) if a_ is not None)
        what_u = "the paper's GT1 equilibrium on this item (Eq. 3), the chance that the next bid comes at all (u)"
        what_m = (f"each non-leader's own chance, 1 {MINUS} (1 {MINUS} u)^(1/m) for the m bidders who may bid (the rule the game's scripted "
                  "equilibrium bidders follow; the paper models only u)")
        counts = "; ".join(f"{xl[i]}: {b_} of {d_}" for i, (b_, d_) in enumerate(tally) if d_)
        if short:
            sub = (f"Dark bar: {what_u}" + ("; with one bidder free to bid it is also that bidder's own chance. " if same else f". Gray bar: {what_m}. ")
                   + f"Blue bar: the share of the seats deciding that chose to bid ({counts}). ")
        else:
            sub = (f"Dashed dark line: {what_u}" + ("; with one bidder free to bid it is also that bidder's own chance. " if same else
                                                   f". Dashed gray line: {what_m}. ")
                   + "Bars: the share of the seats deciding that chose to bid. ")
        if any(sl["opening"] for sl in slots):
            sub += (f"Nothing is drawn at Open: the game placed that bid, which the paper's models assume anyway (u{chr(0x2081)} = 1). ")
        elif r_first["bids_before"] == 0:
            sub += (f"No model value in round {r_first['round']}: the paper's models assume the first bid comes (u{chr(0x2081)} = 1); "
                    f"there {tally[0][0]} of {tally[0][1]} bidders bid. ")
        sub += "Under GT1 every bid has zero expected value, so the bidders are exactly indifferent between bidding and passing."
        model_series = [{"name": "GT1: a bid comes (u)" if same else "GT1: someone bids (u)", "color": "ink", "dash": True, "v": someone, "dots": n <= 12}]
        if not same:
            model_series.append({"name": "GT1: each non-leader bids", "color": "hold", "dash": True, "v": each, "dots": n <= 12})
        seat_bars = [{"name": "Seats that chose to bid", "color": "c1", "v": share}] if any(x is not None for x in share) else []
        if short:
            charts.append({"title": "Chance of a bid: the paper's GT1 model against the bidders", "sub": sub, "height": 210, "fmt": "pct0",
                           "bars": [{k: v_ for k, v_ in m_.items() if k in ("name", "color", "v")} for m_ in model_series] + seat_bars,
                           "lines": []})
        else:
            charts.append({"title": "Chance of a bid: the paper's GT1 model against the bidders", "sub": sub, "height": 220, "fmt": "pct0",
                           "bars": seat_bars, "lines": model_series})

    # ---- feed
    feed = []
    per_decision = n <= 30
    step_ix = {}
    for k, s in enumerate(steps):
        step_ix.setdefault(s["x"], []).append(k)
    for i, sl in enumerate(slots):
        ks = step_ix.get(i, [])
        if sl["opening"]:
            feed.append({"x": i, "who": "Opening", "kind": "bid_reg",
                         "text": f"The game places the opening bid for {sl['registered']}, drawn at random: the price is now {usd(inc)} and "
                                 f"{sl['registered']} leads."})
            continue
        if per_decision:
            for k in ks:
                s = steps[k]
                lab = seat_label[s["seat"]]
                feed.append({"x": i, "step": k, "who": lab, "kind": s["kind"],
                             "text": ("Bids" if s["kind"] != "pass" else "Passes") + f" at {usd(sl['price_before'])}"
                                     + (", without seeing the other bidders' choices." if len(ks) > 1 else ".")
                                     + (f" Its reason: \u201c{reasons[k][1]}\u201d" if reasons[k] else "")})
        bidders_here = [seat_label[s] for s, a in sorted(sl["moves"].items()) if a == "bid"]
        if sl["registered"]:
            lead_in = (f"only {sl['registered']} bids, so its bid is registered" if bidders_here == [sl["registered"]] else
                       f"{and_join(bidders_here)} bid and {sl['registered']}'s bid is drawn" if len(bidders_here) > 1 and sl["registered"] in bidders_here else
                       f"{sl['registered']}'s bid is registered")
            res = (f"Round {sl['round']}: {lead_in}. {sl['registered']} pays {usd(fee)}, the price rises to {usd(sl['price_after'])} and "
                   f"{sl['registered']} now leads." + (f" Bid {sl['bids_before'] + 1} reaches the cap: the auction stops." if cap and sl["bids_before"] + 1 >= cap else ""))
            kind = "bid_reg"
        else:
            if sl["leader"]:
                res = (f"Round {sl['round']}: nobody bids, so the countdown runs out. {sl['leader']} buys the {item.replace('a ', '', 1) if item.startswith('a ') else item} "
                       f"for {usd(sl['price_before'])} after {nbid(sl['bids_before'])}.")
            else:
                res = (f"Round {sl['round']}: nobody bids, so the countdown runs out without a single bid. The "
                       f"{item.replace('a ', '', 1) if item.startswith('a ') else item} stays unsold, nobody pays anything and every score is 0.")
            kind = "pass"
        if i == n - 1 and sold:
            res += " Final money: " + "; ".join(f"{lab} {usd(net[lab], True)}" for lab in labels) + "."
        feed.append({"x": i, "step": ks[-1] if ks else None, "who": "Result", "kind": kind, "text": res})
        if feed[-1]["step"] is None:
            feed[-1].pop("step")

    # ---- tiles
    seat_names = [seat_label[s] for s in range(n_seats)]
    g_note = (f"The paper's GT1 equilibrium expects {g1['mean']:.1f} bids on this item (median {g1['median']})" if g1 else "")
    if sold:
        tiles = [{"k": "Winner", "v": winner, "s": f"Bought the item for {usd(price)} after {nbid(bids)}; net {usd(net[winner], True)} "
                                                  f"after its {usd(nbids.get(winner, 0) * fee)} in fees"},
                 {"k": "Registered bids", "v": f"{bids:,}", "s": (g_note + "; " if g_note else "") + f"{n} rounds in this auction"
                                                              + (" (stopped at the cap)" if ended == "cap" else "")},
                 {"k": "Auctioneer's take", "v": usd(revenue), "s": f"{revenue / v:.2f} {TIMES} the item's value: fees {usd(fees_total)} + price {usd(price)}"},
                 {"k": "Scores", "v": " / ".join(score(r) for r in rewards), "s": " / ".join(seat_names) + f": net money {DIV} {usd(v)}"}]
        op = slots[0]["registered"] if slots[0]["opening"] else None
        if op in seat_of and nbids.get(op, 0) >= 1:     # a seat that paid for a bid it never chose
            tiles[-1]["s"] += (f". {op}'s {usd(net[op])} is the fee of the opening bid the game placed for it" if op != winner and nbids[op] == 1 else
                               f". {op}'s fees include the {usd(fee)} of the opening bid the game placed for it")
    else:
        dec = list(r_first["moves"].values())
        tiles = [{"k": "Outcome", "v": "Unsold",
                  "s": (f"All {numword(len(dec))} bidders passed in round {r_first['round']}, so the auction ended before any bid"
                        if dec and all(a == "pass" for a in dec) else "No bid was ever registered")},
                 {"k": "Registered bids", "v": "0",
                  "s": ((f"{sum(a == 'bid' for a in dec)} of {len(dec)} bid. " if dec else "")
                        + f"The paper's models (GT1, GT2, GT3) assume a first bid (u{chr(0x2081)} = 1), so they give an auction with no bid "
                          "probability 0, and the paper's per-auction loss (NLL) cannot be computed for it"
                        + (f"; on this item GT1 expects {g1['mean']:.1f} bids (median {g1['median']})" if g1 else ""))},
                 {"k": "Scores", "v": " / ".join(score(r) for r in rewards), "s": " / ".join(seat_names) + ": nobody won and nobody paid a fee"},
                 {"k": "The opening bid", "v": f"{usd(gain[0], True)} / {usd(-fee)}",
                  "s": (f"Its net if nobody had bid after it, or the fee it loses when outbid"
                        + (f"; GT1 gives these {pct(1 - gt1_u(2, v, fee, inc))} and {pct(gt1_u(2, v, fee, inc))}, an expected {usd(0)}"
                           if 0 < gt1_u(2, v, fee, inc) < 1 else ""))}]

    # ---- text
    noun = item.replace("a ", "", 1) if item.startswith("a ") else item
    if sold:
        title = f"{winner} wins the {noun} for {usd(price)} after {nbid(bids)}" + (", stopped at the cap" if ended == "cap" else "")
    elif one and all(a == "pass" for a in r_first["moves"].values()):
        title = f"All {numword(len(r_first['moves']))} bidders declined to place the first bid, so the {noun} went unsold"
    else:
        title = f"No bid was registered and the {noun} went unsold"
    opened = any(sl["opening"] for sl in slots)
    lede = (f"{numword(n_bidders).capitalize()} bidders compete for {item} worth {usd(v)} in a penny auction: every registered bid costs its "
            f"bidder a {usd(fee)} fee and raises the price by {usd(inc)}. "
            + (f"The game opens the auction with one bid already registered for a bidder drawn at random, who pays its fee; the seats "
               f"decide from round {r0} on. In each of those rounds" if opened else "In each round")
            + " all bidders except the current leader decide at the same time whether to bid, and one of the bids is registered at "
            "random; when a round passes without a bid, the leader buys at the current price. Each bidder scores its net money (the "
            f"item's value minus the price if it wins, minus all its fees) {DIV} {usd(v)}.")
    model = (b.get("meta") or {}).get("model") or run.get("model") or ""
    model = model.split("/")[-1] if model else "the model"
    date = str((b.get("meta") or {}).get("recorded_at") or run.get("recorded_at") or DEFAULT_DATE)[:10]
    seed = run.get("seed", cfg.get("seed"))
    variant = os.path.basename(str(cfg.get("variant") or "")).replace(".yaml", "")
    who = "; ".join(f"{seat_label[s]} is seat P{s}" for s in range(n_seats))
    old_rule = not opened
    if old_rule and date >= "2026-11-10":
        warn(f"recorded {date}, under the rule that the game places the opening bid, but no opening bid was found")
    # the cap sentence of the prompt: did the seats ask whether it reports how this auction ends, and how did they use it?
    # Either they read it as a hard stop and solved the auction backwards from it (the parity of the bids left decides), or
    # some pass rests on reading it as the outcome (a prompt-clarity problem).
    cap_note = ""
    cap_sent = next((m.group(0).strip() for f in facts for m in [re.search(r"The auction is stopped after [^\n]*", f.get("_view") or "")] if m), None)
    if cap and cap_sent and any(raws):
        cap_pat = rf"\b(?:{cap}|{cap:,}) registered bids"
        seat_raw = {}
        for st_, r_ in zip(steps, raws):
            if r_:
                seat_raw.setdefault(st_["seat"], []).append(r_)
        asked, counts_, leaned = {}, {}, {}
        for s_, rs in sorted(seat_raw.items()):
            txt = "\n\n".join(rs)
            counts_[s_] = len(re.findall(cap_pat, txt))
            q = first_question(txt, r"\b(outcome|history|future info|already (?:simulated|determined|occurred|happened)|predetermined)\b")
            if q:
                asked[s_] = q
            tail = txt[int(len(txt) * .6):]
            lean = next((x for p_ in _paras(tail) for x in _sents(p_) if re.search(rf"\b{cap}\b", x) and re.search(r"\bpass\b", x)
                         and re.search(r"\b(optimal|best|better)\b", x) and re.search(r"\b(believe|assume|reach|will)\b", x) and len(x) <= 200), None)
            if lean:
                leaned[s_] = lean
        # decisions whose reasoning ends on the parity of the bids left (the subgame-perfect end game: whoever places bid T,
        # the last one that can pay, wins for sure, so with T - k bids to go a non-leader bids when T - k is odd)
        T_ = g1["T"] if g1 else None
        par = [k for k, r_ in enumerate(raws) if r_ and T_ and any(re.search(_PARITY, x, re.I) for p_ in _paras(r_[int(len(r_) * .6):]) for x in _sents(p_))]
        rule = lambda k: "bid" if slots[steps[k]["x"]]["bids_before"] < T_ and (T_ - slots[steps[k]["x"]]["bids_before"]) % 2 == 1 else "pass"
        quotes = "; ".join(f"{seat_label[s_]}: \u201c{q}\u201d" for s_, q in asked.items())
        if par:
            par_seats = sorted({steps[k]["seat"] for k in par})
            def group(ss):
                return ("All " + numword(len(ss)) + " seats" if len(ss) == n_seats > 2 else "Both seats" if len(ss) == n_seats == 2
                        else and_join([seat_label[x] for x in ss]))
            # backwards from the cap when each of these seats set its horizon there (k = 134, horizon 134, 134 - 2 ...)
            at_cap = all(re.search(rf"\b[A-Za-z]\s*=\s*{cap}\b|horizon[^.\n]{{0,20}}\b{cap}\b|\b{cap}\s*-\s*\d|\bcap\s*=\s*{cap}\b",
                                   "\n".join(seat_raw[x]), re.I) for x in par_seats)
            anchor = cap if at_cap and (cap - T_) % 2 == 0 else T_
            solved = (f"treated it as a hard stop and solved the auction backwards from "
                      + (f"bid {cap:,}: with an even number of bids left a non-leader passes, with an odd number it bids" if anchor == cap else
                         f"its end: with an even number of bids left before bid {T_:,} a non-leader passes, with an odd number it bids"))
            if asked and set(asked) == set(par_seats):
                cap_note = (f"{group(par_seats)} first asked whether the cap sentence of the prompt (\u201c{cap_sent}\u201d) announced how the "
                            f"auction ends ({quotes}), then {solved}. ")
            else:
                cap_note = ((f"{group(sorted(asked))} asked whether the cap sentence of the prompt (\u201c{cap_sent}\u201d) announced how the "
                             f"auction ends ({quotes}). " if asked else "")
                            + f"{group(par_seats)} {solved}. ")
            def what(ks):                       # "both round-2 bids", "Bidder B's round-3 pass"
                out = []
                for x in sorted({steps[k]["x"] for k in ks}):
                    here = [k for k in ks if steps[k]["x"] == x]
                    deciders = [k for k in range(len(steps)) if steps[k]["x"] == x]
                    a_ = {acts[k] for k in here}
                    noun_ = lambda a__, m: ("bid" if a__ == "bid" else "pass") + ("" if m == 1 else ("s" if a__ == "bid" else "es"))
                    if len(here) == len(deciders) > 1 and len(a_) == 1:
                        out.append(("both" if len(here) == 2 else "all " + numword(len(here))) + f" {xl[x].replace('R', 'round-')} "
                                   + noun_(acts[here[0]], len(here)))
                    else:
                        out += [f"{seat_label[steps[k]['seat']]}'s {xl[x].replace('R', 'round-')} {noun_(acts[k], 1)}" for k in here]
                return out
            agree = [k for k in par if acts[k] == rule(k)]
            against = [k for k in par if acts[k] != rule(k)]
            if agree:
                w_ = what(agree)
                cap_note += (and_join(w_)[0].upper() + and_join(w_)[1:]) + (" follow" if len(agree) > 1 else " follows") + " that rule"
                cap_note += ("; " + and_join(what(against)) + (" go" if len(against) > 1 else " goes") + " against it" if against else "")
            others = [k for k in range(len(steps)) if raws[k] and k not in par]
            for k in others:
                w_ = what([k])[0]
                cap_note += (("; " if agree else "") + f"{w_} gives a different reason"
                             + (f" (\u201c{reasons[k][1]}\u201d)" if reasons[k] else "")
                             + ("" if acts[k] == rule(k) else ", against that rule"))
            cap_note += ". " if agree or others else ""
            if leaned:
                cap_note += ("; ".join(f"{seat_label[s_]}'s pass partly rests on reading the cap sentence as the outcome (\u201c{x}\u201d)"
                                       for s_, x in leaned.items()) + "; this is a prompt-clarity problem. ")
            if all(acts[k] == rule(k) for k in range(len(steps))) and ended == "no_bid":
                per = []
                for x in sorted({s_["x"] for s_ in steps}):
                    ks = [k for k in range(len(steps)) if steps[k]["x"] == x]
                    if len({acts[k] for k in ks}) == 1:
                        per.append(("both " if len(ks) == 2 else "all " + numword(len(ks)) + " " if len(ks) > 2 else "")
                                   + ("bid" if acts[ks[0]] == "bid" else "passed") + f" at {nbid(slots[x]['bids_before'])}")
                cap_note += ("The observed play matches it" + (f" ({', '.join(per)})" if per else "") + f", and the auction ended after {nbid(bids)}. ")
            cap_note += (f"The rules file notes this end game: whoever places bid T = {T_:,}, the last one that can still pay, wins for sure, so "
                         "a subgame-perfect analysis has everyone bid then and unravels backwards from there"
                         + (f" (bids {T_ + 1:,} to {cap:,} cannot pay, so starting from the cap comes to the same)" if cap > T_ and anchor == cap else "")
                         + "; the paper's models keep the indifference condition instead, as the authors' code does, under which every "
                         "bid is worth exactly zero.")
        elif len(asked) >= 2 or (asked and len(asked) == len(seat_raw)):
            who_ = [seat_label[s_] for s_ in asked]
            cap_note = ((f"Every seat" if len(asked) == len(seat_raw) == n_seats else and_join(who_))
                        + f" puzzled over the cap sentence of its prompt (\u201c{cap_sent}\u201d), asking whether it reported how this "
                        "auction would end: " + quotes + ". "
                        + f"{and_join([seat_label[s_] for s_ in asked])} mention \u201c{cap:,} registered bids\u201d "
                        + and_join([f"{counts_[s_]}" for s_ in asked]) + f" times in {'its' if len(asked) == 1 else 'their'} reasoning"
                        + ("; " + "; ".join(f"{seat_label[s_]}'s pass partly rests on that reading (\u201c{x}\u201d)" for s_, x in leaned.items())
                           if leaned else "")
                        + ". This is a prompt-clarity problem.")
    foot = (f"Data: the {date} recording of eco_penny_auction (" + (f"config {variant}, " if variant else "") + (f"seed {seed}, " if seed is not None else "")
            + f"{item} worth {usd(v)}, fee {usd(fee)}, increment {usd(inc)}, {n_bidders} bidders"
            + (f", stopped after {cap:,} bids at the latest" if cap else "") + f"), every seat played by {model}; every number on this page is "
            "computed from its game log and the bidders' prompts. "
            + ("Observations and reasoning come from the saved model calls. " if has_reason else
               "This recording was made before reasoning was saved; the game will be re-recorded with reasoning. ")
            + f"{who} (the letters are drawn at random each episode). "
            + (f"GT1 is the expected-utility equilibrium of Wang and Yu (AAAI-24, Eq. 3): after t {MINUS} 1 bids the next one comes with "
               f"probability u = 1 {MINUS} fee {DIV} (value {MINUS} (t {MINUS} 1) {TIMES} increment), up to bid T = {g1['T']:,}, the last one that "
               f"can still pay, and the first bid is assumed (u{chr(0x2081)} = 1: the paper studies only auctions that got one); on this item that "
               f"gives {g1['mean']:.2f} bids on average (median "
               f"{g1['median']}), and the auctioneer expects to collect exactly the item's value. " if g1 and g1["T"] is not None else "")
            + ("This auction ran under the old opening rule, which left the first bid to the bidders. Since 2026-11-10 the game places the "
               "opening bid for a random bidder, as the paper assumes, and the game will be re-recorded. "
               if old_rule else
               f"This auction ran under the opening rule of 2026-11-10: the game placed the opening bid for a random bidder, as the paper "
               f"assumes (u{chr(0x2081)} = 1). ")
            + cap_note)
    d = {"id": GID, "name": "Penny auction", "title": title, "lede": lede, "seatNames": seat_names, "tiles": tiles, "foot": foot.strip(),
         "x": {"n": n, "labels": xl}, "charts": charts,
         "kinds": {k: KINDS[k] for k in KINDS if any(s["kind"] == k for s in steps) or any(e.get("kind") == k for e in feed)},
         "steps": steps, "feed": feed, "feedTitle": "Round by round",
         "feedSub": "Every decision and what each round produced. Decisions within a round are simultaneous and secret; the selected step "
                    "is highlighted and later entries are hidden."}
    if not has_reason:
        d["eyebrow"] = f"social_game_env {DOT} {GID} {DOT} {model} {DOT} recorded {date} {DOT} reasoning not saved"
    return d


# ------------------------------------------------------------------------------------------------ self test
def _synthetic(with_info=True, plan=None, draws=None, cand="zero", keyword=True, opening=True):
    """A small auction written in the prompt format of the recordings plus a leader statement: 3 seats, v 20, fee 0.75,
    increment 0.15. With opening (the rule of 2026-11-10) the game places the opening bid for Bidder B in round 1 and the seats
    decide from round 2 on; the engine's history entry for it has candidates 0 (cand "zero"), None ("none"), no such key
    ("missing") or a list ("list"), and keyword says whether the prompts call it the opening bid. Without opening the seats
    decide from round 1 on (the old rule)."""
    v, fee, inc, cap = 20.0, 0.75, 0.15, 134
    lab = {0: "Bidder B", 1: "Bidder A", 2: "Bidder C"}
    plan = plan or [{1: "bid", 2: "pass"}, {0: "bid", 2: "bid"}, {0: "pass", 1: "bid"}, {0: "pass", 2: "pass"}]   # rounds 2..5
    draws = draws or [1, 2, 1, None]
    if opening:
        leader, bids, own, r_start = 0, 1, {0: 1, 1: 0, 2: 0}, 2
        h0 = {"round": 1, "bidder": lab[0], "price": inc}
        if cand != "missing":
            h0["candidates"] = {"zero": 0, "none": None, "list": [lab[0]]}[cand]
        hist = [h0]
    else:
        leader, bids, own, r_start, hist = None, 0, {0: 0, 1: 0, 2: 0}, 1, []
    events, turn = [{"event": "setup", "roles": {str(s): "player" for s in lab}}], 0
    for k, (pl, dr) in enumerate(zip(plan, draws)):
        r = k + r_start
        for s, a in sorted(pl.items()):
            turn += 1
            p = bids * inc
            state = (f"Round {r}: {bids} bids so far; the price is ${p:.2f}; the leader is {lab[leader]}.\n" if leader is not None else
                     f"Round {r}: no bid has been placed yet; the price is $0.00. If nobody bids now the item stays unsold.\n")
            view = (f"You are seat P{s}. Your role: BIDDER.\nItem: a game controller. Retail price ${v:.2f}. Bid fee ${fee:.2f} per registered bid. "
                    f"Each registered bid raises the price by ${inc:.2f}.\nBidders: 3. In this auction you are {lab[s]}.\n"
                    f"The auction is stopped after {cap} registered bids (price ${cap * inc:.2f}); the leader then buys at that price.\n"
                    + state +
                    f"If your bid is registered now the price becomes ${p + inc:.2f}; if nobody bids after that, you buy the item for "
                    f"${p + inc:.2f} and your net gain is ${v - p - inc - fee:.2f} for this bid.\n"
                    f"You have placed {own[s]} registered bid(s) and paid ${own[s] * fee:.2f} in fees so far (non-refundable).\n"
                    + ("Recent registered bids: " + "; ".join(f"round {h['round']}: {h['bidder']}, price ${h['price']:.2f}"
                                                              + (" (opening bid, placed at random)" if keyword and h is hist[0] and opening else "")
                                                              for h in hist[-10:]) + ".\n" if hist else "")
                    + "Legal actions:\n- bid\n- pass")
            events.append({"event": "move", "actor": s, "turn": turn, "move": json.dumps({"action_id": a, "payload": {}}), "view": view})
        if dr is not None:
            bids += 1
            leader = dr
            own[dr] += 1
            hist.append({"round": r, "bidder": lab[dr], "price": round(bids * inc, 2), "candidates": [lab[s] for s, a in pl.items() if a == "bid"]})
    price = bids * inc
    net = {s: ((v - price) if s == leader else 0.0) - own[s] * fee for s in lab}
    rewards = {str(s): net[s] / v for s in lab}
    events.append({"event": "game_over", "rewards": rewards, "truncated": False, "winners": [leader] if leader is not None else []})
    info = {"duration": bids, "price": price, "cap": cap, "ended": "no_bid" if bids else "unsold",
            "winner": lab[leader] if leader is not None else None, "history": hist,
            "fees_total": bids * fee, "revenue": bids * fee + (price if bids else 0.0),
            "bidders": [{"label": lab[s], "seat": s, "bids": own[s], "fees": own[s] * fee} for s in lab]}
    summary = {"rewards": rewards}
    if with_info:
        summary["info"] = info
    return {"events": events, "summary": summary, "run": {"seed": 7, "model": "openai/test"}, "config": {}, "meta": {}}


def _quiet_build(bundle, calls=None):
    """build() with its warnings collected instead of printed."""
    got, keep = [], globals()["warn"]
    globals()["warn"] = got.append
    try:
        return build(bundle, calls), got
    finally:
        globals()["warn"] = keep


def _selftest():
    g = gt1_summary(20, .75, .15, 134)
    assert g["T"] == 128 and abs(g["mean"] - 22.2222) < 1e-3 and g["median"] == 17, g
    g = gt1_summary(382.21, .75, .15, 2549)
    assert g["T"] == 2543 and abs(g["mean"] - 424.68) < 0.01 and g["median"] == 330, g
    g = gt1_summary(699, .75, .15, 4660)
    assert g["T"] == 4655 and abs(g["mean"] - 776.67) < 0.01 and g["median"] == 603, g
    assert abs(gt1_u(2, 20, .75, .15) - (1 - .75 / 19.85)) < 1e-12 and gt1_u(1, 20, .75, .15) == 1 and gt1_u(129, 20, .75, .15) == 0
    strip = lambda d: json.dumps({k: d[k] for k in ("x", "charts", "tiles", "title")}, sort_keys=True)
    kinds = ["bid_reg", "pass", "bid_lost", "bid_reg", "pass", "bid_reg", "pass", "pass"]

    # 1. the opening bid however the engine and the prompts mark it (or not at all): always its own column, never a warning
    #    except when the engine lists candidates for a bid placed before anyone decided
    ref = None
    for cand in ("zero", "none", "missing", "list"):
        for keyword in (True, False):
            for with_info in (True, False):
                d, w = _quiet_build(_synthetic(with_info, cand=cand, keyword=keyword))
                tag = (cand, keyword, with_info)
                assert common.check(d) == [], (tag, common.check(d))
                assert d["x"]["labels"] == ["Open", "R2", "R3", "R4", "R5"], (tag, d["x"]["labels"])
                assert d["title"] == "Bidder A wins the game controller for $0.60 after 4 bids", (tag, d["title"])
                assert [s["kind"] for s in d["steps"]] == kinds, (tag, [s["kind"] for s in d["steps"]])
                assert d["feed"][0]["who"] == "Opening" and d["feed"][0]["x"] == 0, (tag, d["feed"][0])
                want_w = 1 if (cand == "list" and with_info) else 0
                assert len(w) == want_w and all("candidates" in x for x in w), (tag, w)
                ref = ref or strip(d)
                assert strip(d) == ref, ("the rebuilt auction depends on how the opening bid is marked", tag)

    # 2. a short auction under the new rule: the opening bid, then both non-leaders pass (columns Open, R2)
    d, w = _quiet_build(_synthetic(True, plan=[{1: "pass", 2: "pass"}], draws=[None]))
    assert not w, w
    assert d["x"]["labels"] == ["Open", "R2"] and common.check(d) == [], (d["x"]["labels"], common.check(d))
    assert d["title"] == "Bidder B wins the game controller for $0.15 after 1 bid", d["title"]
    ch = next(c for c in d["charts"] if c["title"].startswith("Chance of a bid"))
    assert [b["color"] for b in ch["bars"]] == ["ink", "hold", "c1"], ch["bars"]
    assert len({json.dumps(b["v"]) for b in ch["bars"]}) == 3 and ch["bars"][0]["v"][0] is None, ch["bars"]
    assert ch["sub"].startswith("Dark bar:") and "Gray bar:" in ch["sub"] and "Blue bar:" in ch["sub"] and "Dashed" not in ch["sub"], ch["sub"]
    assert "R2: 0 of 2" in ch["sub"] and "Nothing is drawn at Open" in ch["sub"], ch["sub"]

    # 3. the old rule, all three pass in round 1: no model chart (the first bid is the models' assumption, not a prediction),
    #    the opening chart shows signed outcomes on a zero line
    d, w = _quiet_build(_synthetic(True, plan=[{0: "pass", 1: "pass", 2: "pass"}], draws=[None], opening=False))
    assert not w, w
    assert d["x"]["labels"] == ["R1"] and common.check(d) == [], (d["x"]["labels"], common.check(d))
    assert not any(c["title"].startswith("Chance of a bid") for c in d["charts"]), [c["title"] for c in d["charts"]]
    c0 = d["charts"][0]
    assert c0["title"] == "Opening the auction: what it could have won or lost" and c0.get("zero"), c0["title"]
    assert [b["v"][0] for b in c0["bars"]] == [19.1, -0.75] and [b["color"] for b in c0["bars"]] == ["c3", "c2"], c0["bars"]
    assert "(GT1: 3.8%)" in c0["bars"][0]["name"] and "(GT1: 96.2%)" in c0["bars"][1]["name"], c0["bars"]
    assert "so opening is worth exactly $0.00." in c0["sub"], c0["sub"]
    t = next(t for t in d["tiles"] if t["k"] == "Registered bids")
    assert t["s"].startswith("0 of 3 bid. The paper's models (GT1, GT2, GT3) assume a first bid"), t["s"]

    # 4. a round with a registered bid but no decision events is reported
    b = _synthetic(True)
    b["events"] = [e for e in b["events"] if not (e.get("event") == "move" and "Round 3:" in e.get("view", ""))]
    d, w = _quiet_build(b)
    assert any("round 3: a bid by Bidder C is registered but no seat has a decision event" in x for x in w), w

    # 5. the reasoning quotes: format talk at the end is skipped, a reason is picked
    t = ("Early analysis of the rules. " * 12 + "\n\nIf we bid now we become leader and the others outbid us, so we lose the fee. So passing is better.\n\n"
         "Thus pass.\n\nLet's ensure the JSON format exactly: {\"action_id\": \"pass\", \"payload\": {}}.")
    assert ends_in_format_talk(t) and decision_quote(t).endswith("So passing is better. Thus pass."), decision_quote(t)
    assert reason_quote(t, "pass")[1] == "If we bid now we become leader and the others outbid us, so we lose the fee. So passing is better.", reason_quote(t, "pass")

    # 6. the quote follows the action: a late sentence for the other action (about a later round) and questions are left out,
    #    "No extra." is format talk, and a 38-character verdict qualifies
    filler = "Early analysis of the rules and the value of each state. " * 30
    t_bid = (filler + "\n\nAt k=2 both pass. Why would anyone bid at k=2? So you prefer to pass and let the other become leader, then outbid "
             "them later. Thus both pass is indeed equilibrium.\n\nIf we pass, and other bids, other wins at .30, we get 0. So bidding "
             "strictly dominates passing. Thus the answer is definitely bid.\n\nSo final JSON: {\"action_id\": \"bid\", \"payload\": {}}.\n\n"
             "Need output exactly one JSON object. No extra.")
    assert reason_quote(t_bid, "bid")[1] == "So bidding strictly dominates passing. Thus the answer is definitely bid.", reason_quote(t_bid, "bid")
    assert reason_quote(t_bid, "pass")[1].startswith("So you prefer to pass"), reason_quote(t_bid, "pass")
    assert ends_in_format_talk(t_bid) and decision_quote(t_bid).endswith("So bidding strictly dominates passing. Thus the answer is definitely bid.")
    assert stance("So bidding strictly dominates passing.") == ({"bid"}, set()) and stance("If we choose to bid, we pay the fee.") == (set(), set())
    assert stance("Pass strictly dominates Bid!") == ({"pass"}, set()) and stance("The expected value of bidding now is negative.") == (set(), {"bid"})

    # 7. a two-bid auction under the new rule, with reasoning: the opening bid in the lede and the Scores tile, no fee chart, price
    #    and take without the gain bars, and the cap note on seats that solved the auction backwards by parity
    b = _synthetic(True, plan=[{1: "bid", 2: "bid"}, {0: "pass", 1: "pass"}], draws=[2, None])
    why = {(1, 2): ("That seems maybe outcome predetermined? Horizon = 134.", "For odd k, both bid; for even k, both pass. So bidding strictly dominates passing. Thus the answer is definitely bid."),
           (2, 2): ("Is this revealing final outcome? Backward from n=134.", "At odd k the non-leaders bid. So bidding is optimal because the other will bid too."),
           (0, 3): ("This may be actual outcome?", "Remaining = 134 - 2 = 132 even. Thus current leader C is on the winning parity. We should pass and let C win."),
           (1, 3): ("Is the cap the outcome?", "Bidding now at .30 would be premature and likely lead to paying many fees. So the optimal action is to pass now.")}
    calls = []
    for e in b["events"]:
        if e.get("event") == "move":
            q, end = why[(e["actor"], round_of(e["view"]))]
            calls.append({"seat": e["actor"], "prompt": e["view"], "answer": e["move"], "rescued": False,
                          "reasoning": q + "\n\n" + filler + "\n\n" + end + "\n\nNeed output exactly one JSON object. No extra."})
    d, w = _quiet_build(b, calls)
    d.setdefault("eyebrow", "test")                     # common.write sets it for pages with reasoning
    assert not w and common.check(d) == [], (w, common.check(d))
    assert "The game opens the auction with one bid already registered" in d["lede"], d["lede"]
    assert d["tiles"][-1]["s"].endswith("Bidder B's " + MINUS + "$0.75 is the fee of the opening bid the game placed for it"), d["tiles"][-1]
    assert not any(c["title"] == "Fees paid by each bidder" for c in d["charts"]), [c["title"] for c in d["charts"]]
    c0 = d["charts"][0]
    assert not c0.get("bars") and [l_["name"] for l_ in c0["lines"]] == ["Auctioneer's take ($)", "Price ($)"], c0
    assert "$19.10 / $18.95 / $18.80 (Open / R2 / R3)" in c0["sub"], c0["sub"]
    for frag in ("solved the auction backwards from bid 134", "Both round-2 bids and Bidder B's round-3 pass follow that rule",
                 "Bidder A's round-3 pass gives a different reason", "both bid at 1 bid, both passed at 2 bids", "unravels backwards"):
        assert frag in d["foot"], (frag, d["foot"])
    assert "check the next recording" not in d["foot"], d["foot"]
    for c_ in calls:                                     # a seat that never set its horizon at the cap: backwards from T instead
        c_["reasoning"] = c_["reasoning"].replace(" Horizon = 134.", "")
    d2, _ = _quiet_build(b, calls)
    assert "solved the auction backwards from its end: with an even number of bids left before bid 128" in d2["foot"], d2["foot"]
    fq = [f["text"] for f in d["feed"] if f.get("step") is not None and f["who"] != "Result"]
    assert fq[0].endswith("\u201cSo bidding strictly dominates passing. Thus the answer is definitely bid.\u201d"), fq
    assert fq[2].endswith("\u201cThus current leader C is on the winning parity. We should pass and let C win.\u201d"), fq
    print("selftest ok: GT1 22.22 / 424.68 / 776.67 bids (medians 17, 330, 603); the opening bid is found however it is marked "
          "(16 variants, engine record and prompts alone agree); a one-bid auction draws three distinct model bars; an old-rule "
          "unsold auction draws no model chart; a registered round without decisions is reported; reasoning quotes skip format talk "
          "and follow the action; a two-bid auction states the opening bid, drops the gain bars and the fee chart, and reports the "
          "seats' parity reading of the cap")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        _selftest()
        raise SystemExit(0)
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    common.write(build(src, calls), out, date=recorded_date(src))
