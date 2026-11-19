"""Replay data for eco_bayesian_persuasion (Bayesian persuasion after Li et al., "Verbalized Bayesian Persuasion", ICML 2026:
an informed sender pledges a policy, then signals; a receiver who never sees the state acts on the signal).

usage: python3 -I eco_bayesian_persuasion.py SOURCE OUT_DIR [CALLS.jsonl]
       python3 -I eco_bayesian_persuasion.py --selftest SOURCE   (call alignment on synthetic call logs; exit status = failed cases)
  SOURCE   a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md), a file inside
           one, or the rendered Markdown replay of one (one "- **event** ... `{json}`" line per logged event).
  OUT_DIR  where <OUT_DIR>/eco_bayesian_persuasion.json is written (via common.write).
  CALLS    optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). Calls are queued per
           seat in time order and tied to that seat's decisions (pledge, write, signal, decide) in order: a decision's attempts run
           up to the first call whose answer parses to a move of that decision (its action_id), whatever the payload. The engine
           asks again after output it cannot read and after an action_id that is not a legal action ("That was not a legal
           action. Reply with one valid JSON action."), and replaces a legal action_id with an illegal payload by the phase
           default (rules: Actions). A call that belongs to another decision (by the phase and the public-state round of its
           prompt, or, with no prompt saved, by its answer's action) ends them, and is skipped when it is a leftover of an
           earlier one. The final call's reasoning comes first in the step's thinking, the re-asked attempts' reasoning after
           it; its prompt becomes the observation and its answer the action (with a note when the engine applied a different
           move).

The x axis is the rounds. Each round has up to four decisions: the sender's pledge (made before the state is drawn), its
message, its label, and the receiver's action. Every number shown is computed from the game log: the state, label and
decision of each round (private state and payoff events, public round_result events), both payoffs (private payoff events,
checked against the payoff tables and against game_end and summary.json; any mismatch is printed as a warning). Lie rate and
honesty follow the paper's definitions (rules/eco_bayesian_persuasion_rules.md, Metrics); the theoretical references are the
config's ref_* values (the paper's Appendix C.1)."""
import json, os, re, sys, unicodedata
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_bayesian_persuasion"
DEFAULT_DATE = "2026-11-09"
MINUS, TIMES, DIV, MID = "−", "×", "÷", "·"
LQ, RQ = "“", "”"

# Per scenario: role names, the noun for the state, words for the labels and actions. The payoff tables, prior and Appendix
# C.1 references below are the rules file's (Reward; Configuration) and are used only when the source does not carry them (a
# rendered Markdown replay keeps the events but not the config).
SCEN = {
    "rel": {"sender": "Professor", "receiver": "HR officer", "noun": "student", "signal_noun": "Letter", "policy": "how it will write its letters",
            "decides": "decides whether to hire", "fav_event": "hire", "verb": {"hire": "Hires", "do_not_hire": "Does not hire"},
            "short": {"hire": "Hire", "do_not_hire": "No hire"}, "gerund": {"hire": "hiring", "do_not_hire": "not hiring"},
            "hi_pp": "recommended", "hi_verb": "recommends", "lo_pp": "not recommended", "hi_noun": "recommendation", "lo_noun": "non-recommendation",
            "fav_past": "hired", "oth_past": "did not hire", "never": "never hiring", "never_does": "never hires",
            "sig_short": {"recommend": "Rec.", "not_recommend": "Not rec."}, "promise": "to recommend only strong students",
            "vars": {"favored_state": "strong", "other_state": "weak", "high_signal": "recommend", "low_signal": "not_recommend",
                     "favored_action": "hire", "other_action": "do_not_hire", "prior_action": "do_not_hire", "prior_favored_state": 0.3333,
                     "signal_labels": ["not_recommend", "recommend"], "action_labels": ["do_not_hire", "hire"],
                     "sender_payoffs": {"strong": {"hire": 1, "do_not_hire": 0}, "weak": {"hire": 1, "do_not_hire": 0}},
                     "receiver_payoffs": {"strong": {"hire": 1, "do_not_hire": 0}, "weak": {"hire": -1, "do_not_hire": 0}},
                     "ref_lie_probability": 0.5, "ref_optimal_payoffs": [0.6667, 0.0], "ref_no_info_payoffs": [0.0, 0.0],
                     "ref_full_info_payoffs": [0.3333, 0.3333], "reward_scale": {"SENDER": 1, "RECEIVER": 1}}},
    "cor": {"sender": "Prosecutor", "receiver": "Judge", "noun": "defendant", "signal_noun": "Report", "policy": "how it will report on defendants",
            "decides": "decides whether to convict", "fav_event": "conviction", "verb": {"convict": "Convicts", "acquit": "Acquits"},
            "short": {"convict": "Convict", "acquit": "Acquit"}, "gerund": {"convict": "convicting", "acquit": "acquitting"},
            "hi_pp": "reported guilty", "hi_verb": "reports as guilty", "lo_pp": "reported not guilty", "hi_noun": "guilty report", "lo_noun": "not-guilty report",
            "fav_past": "convicted", "oth_past": "acquitted", "never": "always acquitting", "never_does": "never convicts",
            "sig_short": {"guilty": "Guilty", "not_guilty": "Not guilty"}, "promise": "to report only guilty defendants as guilty",
            "vars": {"favored_state": "guilty", "other_state": "innocent", "high_signal": "guilty", "low_signal": "not_guilty",
                     "favored_action": "convict", "other_action": "acquit", "prior_action": "acquit", "prior_favored_state": 0.3,
                     "signal_labels": ["guilty", "not_guilty"], "action_labels": ["acquit", "convict"],
                     "sender_payoffs": {"guilty": {"convict": 1, "acquit": 0}, "innocent": {"convict": 1, "acquit": 0}},
                     "receiver_payoffs": {"guilty": {"convict": 1, "acquit": 0}, "innocent": {"convict": 0, "acquit": 1}},
                     "ref_lie_probability": 0.4286, "ref_optimal_payoffs": [0.6, 0.7], "ref_no_info_payoffs": [0.0, 0.7],
                     "ref_full_info_payoffs": [0.3, 1.0], "reward_scale": {"SENDER": 1, "RECEIVER": 1}}},
    "lae": {"sender": "Police", "receiver": "Driver", "noun": "mile", "signal_noun": "Notice", "policy": "how it will announce patrols",
            "decides": "decides whether to obey the speed limit", "fav_event": "obeyed mile", "verb": {"obey": "Obeys the limit", "speed": "Speeds"},
            "short": {"obey": "Obey", "speed": "Speed"}, "gerund": {"obey": "obeying the limit on", "speed": "speeding on"},
            "hi_pp": "announced as patrolled", "hi_verb": "announces as patrolled", "lo_pp": "announced as unpatrolled", "hi_noun": "patrol notice", "lo_noun": "no-patrol notice",
            "fav_past": "obeyed", "oth_past": "sped", "never": "always obeying", "never_does": "never obeys the limit",
            "sig_short": {"patrol": "Patrol", "no_patrol": "No patrol"}, "promise": "to announce patrols only on patrolled miles",
            "vars": {"favored_state": "patrolled", "other_state": "unpatrolled", "high_signal": "patrol", "low_signal": "no_patrol",
                     "favored_action": "obey", "other_action": "speed", "prior_action": "obey", "prior_favored_state": 0.6667,
                     "signal_labels": ["no_patrol", "patrol"], "action_labels": ["obey", "speed"],
                     "sender_payoffs": {"patrolled": {"obey": 1, "speed": 0}, "unpatrolled": {"obey": 1, "speed": 0}},
                     "receiver_payoffs": {"patrolled": {"obey": 0, "speed": -0.5}, "unpatrolled": {"obey": 0, "speed": 1.5}},
                     "ref_lie_probability": 0.6667, "ref_optimal_payoffs": [0.8889, 0.1667], "ref_no_info_payoffs": [0.0, 0.1667],
                     "ref_full_info_payoffs": [0.6667, 0.5], "reward_scale": {"SENDER": 1, "RECEIVER": 1.5}}},
}
# The paper's own results for the five-round setting S3 (Figure 4 / Table 1, read from the figure files; rules file,
# Phenomenon): sender mean payoff, receiver mean payoff, lie probability. Shown only for a five-round game of that scenario.
PAPER_S3 = {"rel": (0.48, 0.19, 0.22)}
KIND_COLORS = {"pledge": "hold", "message": "muted", "true_high": "c1", "lie": "c4", "true_low": "c2", "withheld": "ink",
               "act_fav": "c1", "act_oth": "c2"}
PHASE = {"pledge": "PLEDGE", "write": "WRITE", "signal": "SIGNAL", "decide": "DECIDE"}


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_PUNCT = {"\u3002": ". ", "\u3001": ", ", "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"', "\u3010": "[", "\u3011": "]",
          "\u300a": '"', "\u300b": '"', "\u3008": "<", "\u3009": ">", "\u301c": "~"}


def en(s):
    """The page shows English only: full-width forms become ASCII, CJK punctuation its ASCII counterpart, and any remaining run
    of Chinese, Japanese or Korean characters (the model sometimes reasons in Chinese) a visible marker."""
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
    """A rendered Markdown replay (declarative games): events from the '- **event** (tag): `{json}`' lines (the tag marks
    private and hidden events; a line cut short, like the long game_over one, is skipped), seat rewards, seed and model from
    the header, and the model calls listed after the events ('### ' blocks), if any."""
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
    setup = next((e for e in events if e.get("event") == "setup"), {})
    summary = {"rewards": json.loads(rw.group(1)) if rw else {}, "roles": setup.get("roles", {}), "variables": {}}
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
    b = load_source(src)
    return str((b.get("meta") or {}).get("recorded_at") or (b.get("run") or {}).get("recorded_at") or DEFAULT_DATE)[:10]


# ------------------------------------------------------------------------------------------------ text helpers
NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def numword(n):
    return NUM[n] if 0 <= n < len(NUM) else str(n)


def disp(label):
    """'not_recommend' -> 'not recommend'."""
    return str(label).replace("_", " ")


def pay(v):
    """A payoff: +1, 0, −1, +1.5."""
    v = float(v)
    if abs(v) < 1e-12:
        return "0"
    s = f"{abs(v):g}"
    return ("+" if v > 0 else MINUS) + s


def dec2(v):
    v = float(v)
    return (MINUS if v < -0.005 else "") + f"{abs(v):.2f}"


def frac(p):
    fr = Fraction(float(p)).limit_denominator(12)
    return f"{fr.numerator}/{fr.denominator}" if abs(float(fr) - float(p)) < 1e-3 and fr.denominator > 1 else f"{float(p):g}"


def article(word):
    return "an" if str(word)[:1].lower() in "aeiou" else "a"


def rounds_text(rs):
    """[3] -> 'Round 3'; [1, 4, 5] -> 'Rounds 1, 4 and 5'."""
    rs = list(rs)
    if not rs:
        return "no round"
    if len(rs) == 1:
        return f"Round {rs[0]}"
    return "Rounds " + ", ".join(str(r) for r in rs[:-1]) + " and " + str(rs[-1])


def plural(n, one, many=None):
    """'1 round', '3 lies', 'no lies'."""
    return f"{'no' if n == 0 else n} {one if n == 1 else (many or one + 's')}"


def quote(t):
    return f"{LQ}{t}{RQ}"


def num(v):
    """A total: 1, \u22121, 0, 1.5."""
    v = float(v)
    return (MINUS if v < 0 else "") + f"{abs(v):g}"


def of_the(k, n, word, words=False):
    """'1 of the 4 recommendations', 'the non-recommendation', 'both ...', 'all three true recommendations'."""
    pl = word + "s"
    if k == n:
        return f"the {word}" if n == 1 else f"both {pl}" if n == 2 else f"all {numword(n) if words else n} {pl}"
    return f"{numword(k) if words else k} of the {n} {word if n == 1 else pl}"


def low(name):
    """A role name inside a sentence: 'Professor' -> 'professor'; acronyms stay ('HR officer')."""
    return name if name[:2].isupper() else name[:1].lower() + name[1:]


def trim(t, n=200):
    """At most n characters, cut at a word boundary with an ellipsis."""
    t = " ".join(str(t or "").split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def truthful_pledge(text, fav, oth):
    """Whether a pledge promises the truthful scheme: the high label only for the favored state ('recommend only students
    whose true quality is strong', 'only if I am certain they are strong'), with no word of mixing (sometimes, a probability,
    a share, some of the other state). Anything else counts as not detected, and the page then just quotes the pledge."""
    t = " ".join(str(text or "").split()).lower()
    mixing = (r"\b(sometimes|occasional\w*|probabilit\w*|percent|half|random\w*|mix\w*|at times|not always|chance (of|that))\b|%|\d\s*/\s*\d"
              + rf"|\b(some|a few|a couple of)( of the)? {re.escape(oth.lower())}")
    if not t or re.search(mixing, t):
        return False
    return bool(re.search(r"(?<!not )\bonly\b[^.;:!?]{0,80}\b" + re.escape(fav.lower()) + r"\b", t))


_PROB = re.compile(r"\bprobability(?: of)?(?: about| roughly| around| approximately)?\s+(?P<d>0?\.\d+)\b"
                   r"|(?P<pc>\d{1,2}(?:\.\d+)?)\s*(?:%|percent\b)", re.I)


def mixed_pledge(text, fav, oth, verb=None):
    """A pledge that states a mixed policy: a sentence naming the other state and the sender's verb, with a probability
    after the verb ('For weak students, I will recommend them with probability 0.45', 'I will recommend 40% of weak ones');
    the order keeps a sentence about the prior out ('Since 67% of students are weak, I will recommend sometimes'). Returns
    {'p': that probability, 'every': whether the pledge also promises the high label for every favored-state draw
    ('recommend every strong student')}, or None."""
    t = " ".join(str(text or "").split())
    for sent in re.split(r"(?<=[.;!?])\s+", t):
        v = re.search(rf"\b{re.escape(verb)}", sent, re.I) if verb else None
        m = _PROB.search(sent, v.end() if v else 0)       # the probability after the verb: "recommend ... with probability p"
        if not m or not re.search(rf"\b{re.escape(oth)}\b", sent, re.I) or (verb and not v):
            continue
        p = float(m.group("d")) if m.group("d") else float(m.group("pc")) / 100
        if 0 < p < 1:
            f_ = re.escape(fav)
            every = re.search(rf"\b(every|all|always|each)\b[^.;:!?]{{0,40}}\b{f_}\b|\b{f_}\b[^.;:!?]{{0,20}}\balways\b", t, re.I)
            return {"p": p, "every": bool(every)}
    return None


def admits_breach(text):
    """Whether a pledge owns up to a broken promise ('I acknowledge my last recommendation was wrong ... I broke your trust')."""
    t = str(text or "").lower()
    return bool(re.search(r"\b(acknowledg\w*|admit\w*|apologi\w*|confess\w*)\b", t)
                and re.search(r"\b(wrong\w*|broke|break\w*|mislead\w*|misled|lie|lied|mistake\w*)\b", t))


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


def _norm_text(t):
    return re.sub(r"\s+", " ", str(t or "")).strip()


def _matches(call, st):
    a = _answer(call.get("answer"))
    mv = st["_move"]
    if not isinstance(a, dict) or a.get("action_id") != mv["action_id"]:
        return False
    p, q = a.get("payload") or {}, mv["payload"]
    key = next(iter(q))
    got, want = _norm_text(p.get(key)), _norm_text(q[key])
    if key in ("text", "message"):
        return got == want or (min(len(got), len(want)) >= 20 and (got.startswith(want) or want.startswith(got)))
    return got == want


PHASE_ORDER = {"PLEDGE": 0, "WRITE": 1, "SIGNAL": 2, "DECIDE": 3}


def _phase(call):
    m = re.search(r"^Current phase: ([A-Z_]+)", call.get("prompt") or "", re.M)
    return m.group(1) if m else None


def _round(call):
    """The round shown in the prompt's public state. Only that line counts: the private messages listed above it carry the
    rounds of earlier events (the sender's states and payoffs, the receiver's payoffs)."""
    m = re.search(r"^Public state: (\{.*\})\s*$", call.get("prompt") or "", re.M)
    if not m:
        return None
    try:
        r = json.loads(m.group(1)).get("round")
        return int(r) if r is not None else None
    except (ValueError, TypeError, AttributeError):
        r = re.match(r'\{[^\n]*?"round": (\d+)', m.group(1))
        return int(r.group(1)) if r else None


def _aid(call):
    a = _answer(call.get("answer"))
    return a.get("action_id") if isinstance(a, dict) else None


def attach_calls(steps, calls):
    """Tie calls to steps, per seat in time order. A decision's attempts are that seat's next calls up to and including the
    first one whose answer parses to a move of this decision (its action_id), whatever the payload: the engine asks again
    after output it cannot read and after an action_id that is not a legal action, and replaces a move with an illegal
    payload by the phase default (rules: Actions). The final attempt's reasoning comes first; the re-asked attempts follow,
    each with the reason it was re-asked. A call that belongs to another decision ends the attempts: its prompt shows
    another phase or round, or (prompts not saved) its answer is another of the seat's moves. Such a call is skipped when it
    is a leftover of an earlier decision (an earlier round or phase; without a prompt, a move the seat made more recently than
    it will next make it), so a missing or extra call shifts nothing after it. When the final answer is not the logged move,
    the engine applied another move and the step says so."""
    queue = {}
    for i, c in enumerate(calls):
        queue.setdefault(c.get("seat"), []).append(i)
    seq = {}                                              # each seat's decisions, in order: [action_id, ...]
    for st in steps:
        st["_j"] = len(seq.setdefault(st["seat"], []))
        seq[st["seat"]].append(st["_move"]["action_id"])
    ptr = {}
    for st in steps:
        q, p, s = queue.get(st["seat"], []), ptr.get(st["seat"], 0), seq[st["seat"]]
        aid, ph, j = st["_move"]["action_id"], PHASE[st["_move"]["action_id"]], st["_j"]
        here = (st["_r"]["round"], PHASE_ORDER[ph])
        run, k = [], p
        while k < len(q) and len(run) < 6:
            c = calls[q[k]]
            cp, cr, ca = _phase(c), _round(c), _aid(c)
            stale = ahead = False
            if cp is not None and cr is not None:         # the prompt says which decision it is
                key = (cr, PHASE_ORDER.get(cp, -1))
                stale, ahead = key < here, key > here
            elif cp is not None or cr is not None:        # only half of it: another decision, order unknown
                ahead = (cp is not None and cp != ph) or (cr is not None and cr != here[0])
            elif ca in PHASE and ca != aid:               # no prompt: another of the seat's moves
                back = next((d for d in range(1, j + 1) if s[j - d] == ca), None)
                fwd = next((d for d in range(1, len(s) - j) if s[j + d] == ca), None)
                stale, ahead = back is not None and (fwd is None or back < fwd), True
            if stale and not run:
                k += 1
                p = k
                continue
            if stale or ahead:
                break
            run.append(q[k])
            k += 1
            if ca == aid:
                break
        ptr[st["seat"]] = p
        if not run:
            continue
        hit = next((j for j, ci in enumerate(run) if _matches(calls[ci], st)), None)
        use = hit if hit is not None else len(run) - 1
        earlier = []
        for ci in run[:use]:
            c = calls[ci]
            why = ("its answer was not a legal action, so the engine asked again" if isinstance(_answer(c.get("answer")), dict)
                   else "its answer could not be read, so the engine asked again")
            earlier.append(f"[Earlier attempt: {why}. Its answer: {c['answer'][:300]}]\n{c.get('reasoning') or '(no reasoning returned)'}")
        c = calls[run[use]]
        parts = [(f"[Final attempt; {plural(len(earlier), 'earlier attempt')} below]\n" if earlier else "")
                 + (c.get("reasoning") or ("(no reasoning returned)" if earlier else ""))]
        if hit is None:
            parts.append(f"[The engine applied a different move than this answer: {c['answer'][:300]}]")
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        parts += earlier
        st["thinking"] = en("\n\n".join(x for x in parts if x).strip())
        if c.get("prompt"):
            st["state"] = en(common.state_part(c["prompt"]))
            st["_from_prompt"] = True
        a = _answer(c.get("answer"))
        if hit is not None and isinstance(a, dict):
            st["act"] = en(json.dumps(a, ensure_ascii=False))
        ptr[st["seat"]] = p + use + 1


def selftest(src):
    """Synthetic call logs made from the game log, in the generic engine's prompt layout (private messages, which carry the
    rounds of earlier events, above the public state), each call's reasoning naming its own decision. Every case must tie
    every decision to its own call, except the one decision whose call a case leaves out. Prints one line per case."""
    b = load_source(src)
    ev = b["events"]
    base = ((b.get("run") or {}).get("prompt") or {}).get("text") or ""
    if "Public state: {" not in base:
        base = 'You are seat P1. Your role: SENDER.\nPrivate messages to you: []\nCurrent phase: PLEDGE -- x\nPublic state: {"rounds": 5, "round": 1}\nLegal actions:\n- x'
    move_of = {"pledge": "pledge", "message": "write", "signal": "signal", "round_result": "decide"}   # logged event -> decision
    info = next((e for e in ev if e.get("event") == "setup_info"), {})
    s_seat, r_seat = info.get("sender_seat", 1), info.get("receiver_seat", 0)

    def prompt(st, ph, rnd, upto):
        priv = [{k: v for k, v in e.items() if not k.startswith("_")} for e in ev[:upto] if e.get("_audience") == [st]]
        t = re.sub(r"You are seat P\d+\.", f"You are seat P{st}.", base)
        t = re.sub(r"(?m)^Current phase: [A-Z_]+", f"Current phase: {ph}", t)
        t = re.sub(r'(?m)^(Public state: \{[^\n]*?"round": )\d+', lambda m: m.group(1) + str(rnd), t)
        return re.sub(r"(?m)^Private messages to you: .*$", lambda m: "Private messages to you: " + json.dumps(priv), t)

    def make(prompts=True, drop=None, illegal=None, retry=None, extra=None, badid=None):
        out = []
        for i, e in enumerate(ev):
            r, kind = e.get("round"), e.get("event")
            if kind not in move_of:
                continue
            aid = move_of[kind]
            st = r_seat if aid == "decide" else s_seat
            pay_ = {"pledge": {"text": e.get("text")}, "write": {"message": e.get("text")}, "signal": {"signal": e.get("signal")},
                    "decide": {"action": e.get("decision")}}[aid]
            if (r, aid) == drop:
                continue
            if (r, aid) == illegal:
                pay_ = {"signal": "maybe"} if aid == "signal" else {"action": "maybe"}
            P = prompt(st, PHASE[aid], r, i) if prompts else ""
            if (r, aid) == retry:
                out.append({"seat": st, "prompt": P, "answer": "I think I should", "reasoning": f"R{r} {aid} bad", "rescued": False})
            if (r, aid) == badid:                           # an action_id that is no legal action (the label), then the re-ask
                wrong = str(next(iter(pay_.values()))) if aid in ("signal", "decide") else "maybe"
                out.append({"seat": st, "prompt": P, "answer": json.dumps({"action_id": wrong, "payload": pay_}),
                            "reasoning": f"R{r} {aid} bad", "rescued": False})
                P = P + ("\nThat was not a legal action. Reply with one valid JSON action." if P else "")
            out.append({"seat": st, "prompt": P, "answer": json.dumps({"action_id": aid, "payload": pay_}), "reasoning": f"R{r} {aid}",
                        "rescued": False})
            if (r, aid) == extra:                           # a stray second answer to the same decision
                out.append(dict(out[-1], reasoning=f"R{r} {aid} stray"))
        return out

    cases = [("prompts", {}), ("prompts, format retry in R2 write", {"retry": (2, "write")}),
             ("prompts, R1 write call missing", {"drop": (1, "write")}), ("prompts, stray extra R2 signal call", {"extra": (2, "signal")}),
             ("prompts, illegal R3 signal", {"illegal": (3, "signal")}),
             ("no prompts", {"prompts": False}), ("no prompts, format retry in R2 write", {"prompts": False, "retry": (2, "write")}),
             ("no prompts, R1 write call missing", {"prompts": False, "drop": (1, "write")}),
             ("no prompts, stray extra R2 signal call", {"prompts": False, "extra": (2, "signal")}),
             ("no prompts, illegal R3 signal", {"prompts": False, "illegal": (3, "signal")}),
             ("no prompts, illegal R3 decision", {"prompts": False, "illegal": (3, "decide")}),
             ("prompts, illegal action_id re-asked in R4 decision", {"badid": (4, "decide")}),
             ("prompts, illegal action_id re-asked in R2 signal", {"badid": (2, "signal")}),
             ("no prompts, illegal action_id re-asked in R4 decision", {"prompts": False, "badid": (4, "decide")})]
    bad = 0
    for name, kw in cases:
        d = build(b, make(**kw))
        wrong = []
        for i, st in enumerate(d["steps"]):
            r = st["x"] + 1
            aid = {"pledge": "pledge", "message": "write"}.get(st["kind"], "decide" if st["kind"].startswith("act_") else "signal")
            want = f"R{r} {aid}"
            th = st["thinking"]
            if (r, aid) == kw.get("drop"):
                ok = th == ""
            else:                                           # the final attempt's reasoning comes first, re-asked ones after it
                own = (th.split("\n", 1)[1] if th.startswith("[Final attempt") else th).split("\n\n[")[0]
                ok = own == want and ((r, aid) != kw.get("illegal") or "applied a different move" in th) \
                    and ((r, aid) != kw.get("retry") or ("could not be read" in th and f"R{r} {aid} bad" in th)) \
                    and ((r, aid) != kw.get("badid") or ("not a legal action" in th and f"R{r} {aid} bad" in th))
            if not ok:
                wrong.append(f"step {i} ({want}): {th[:60]!r}")
        bad += bool(wrong)
        print(("ok   " if not wrong else "FAIL ") + name + ("" if not wrong else ": " + "; ".join(wrong[:4])))
    return bad


# ------------------------------------------------------------------------------------------------ build
def build(bundle_dir, calls=None):
    b = load_source(bundle_dir)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    if not calls:
        calls = b.get("md_calls") or []
    events, summary, run, cfg = b["events"], b.get("summary") or {}, b.get("run") or {}, b.get("config") or {}
    info = next((e for e in events if e.get("event") == "setup_info"), {})
    sid = info.get("scenario_id") or (summary.get("variables") or {}).get("scenario_id") or "rel"
    sc = SCEN.get(sid, SCEN["rel"])
    # variables: the game's own (game_over, summary, config) first; the rules defaults of the scenario only for what is missing
    over = next((e for e in events if e.get("event") == "game_over"), {})
    var = dict(sc["vars"])
    for src in ((cfg.get("variables") or {}), (summary.get("variables") or {}), (over.get("variables") or {})):
        var.update({k: v for k, v in src.items() if v is not None})
    var.update({k: info[k] for k in ("rounds", "commitment", "use_label", "message_max_words") if info.get(k) is not None})
    fav, oth = var["favored_state"], var["other_state"]
    hi, lo = var["high_signal"], var["low_signal"]
    fa, oa = var["favored_action"], var["other_action"]
    SP, RP = var["sender_payoffs"], var["receiver_payoffs"]
    prior = float(var["prior_favored_state"])
    R = int(info.get("rounds") or var.get("rounds") or 0)
    commitment = info.get("commitment", var.get("commitment"))
    roles = summary.get("roles") or next((e.get("roles") for e in events if e.get("event") == "setup"), {}) or {}
    seat_of_role = {r: int(s) for s, r in roles.items()}
    s_seat = info.get("sender_seat", seat_of_role.get("SENDER"))
    r_seat = info.get("receiver_seat", seat_of_role.get("RECEIVER"))
    n_seats = len(roles) or (int(s_seat is not None) + int(r_seat is not None))
    SN, RN = sc["sender"], sc["receiver"]                 # as names ("Professor", "HR officer")
    sn, rn = low(SN), low(RN)                             # inside a sentence ("the professor", "the HR officer")
    noun = sc["noun"]
    seat_names = ["Model"] * max(1, n_seats)
    if s_seat is not None:
        seat_names[s_seat] = SN
    if r_seat is not None:
        seat_names[r_seat] = RN
    if n_seats == 1:                                      # one seated role: its index is 0
        s_seat = 0 if s_seat is not None else None
        r_seat = 0 if r_seat is not None else None
        seat_names = [SN if s_seat == 0 else RN]

    # ---- rounds from the log
    rd = {}

    def at(r):
        return rd.setdefault(int(r), {"round": int(r), "pledge": None, "pledge_actor": None, "state": None, "message": None, "signal": None,
                                      "decision": None, "s_pay": None, "r_pay": None})
    for e in events:
        ev = e.get("event")
        if ev == "pledge":
            x = at(e.get("round", 1))
            x["pledge"], x["pledge_actor"] = e.get("text", ""), e.get("actor")
        elif ev == "scheme":                              # a scripted sender's scheme, made public as its pledge (no decision)
            x = at(e.get("round", 1) or 1)
            x["pledge"] = x["pledge"] or e.get("text") or e.get("pledge") or ""
        elif ev == "state":
            at(e["round"])["state"] = e["state"]
        elif ev == "message":
            at(e["round"])["message"] = e.get("text", "")
        elif ev == "signal":
            at(e["round"])["signal"] = e["signal"]
        elif ev == "payoff":
            x = at(e["round"])
            aud = e.get("_audience")
            to_sender = (aud == [s_seat]) if aud else ("state" in e)
            if to_sender:
                x["s_pay"] = e["payoff"]
                x["state"] = x["state"] or e.get("state")
            else:
                x["r_pay"] = e["payoff"]
            x["decision"] = x["decision"] or e.get("decision")
            x["signal"] = x["signal"] or e.get("signal")
        elif ev == "round_result":
            x = at(e["round"])
            x["decision"], x["signal"] = e.get("decision", x["decision"]), e.get("signal", x["signal"])
    rounds = [rd[r] for r in sorted(rd) if rd[r]["decision"] is not None]
    if not rounds:
        raise SystemExit("no finished round in the game log")
    R = R or len(rounds)
    warn = []
    if len(rounds) != R:
        warn.append(f"{len(rounds)} finished rounds in the log, {R} configured")
    for x in rounds:
        if x["state"] is not None:
            want_s, want_r = SP[x["state"]][x["decision"]], RP[x["state"]][x["decision"]]
            if x["s_pay"] is None:
                x["s_pay"] = want_s
            if x["r_pay"] is None:
                x["r_pay"] = want_r
            if x["s_pay"] != want_s or x["r_pay"] != want_r:
                warn.append(f"round {x['round']}: payoffs {x['s_pay']}/{x['r_pay']} logged, {want_s}/{want_r} by the tables")
    game_end = next((e for e in events if e.get("event") == "game_end"), {})
    s_score = sum(x["s_pay"] or 0 for x in rounds)
    r_score = sum(x["r_pay"] or 0 for x in rounds)
    if game_end and (game_end.get("sender_score") != s_score or game_end.get("receiver_score") != r_score):
        warn.append(f"scores {s_score}/{r_score} from the payoffs, {game_end.get('sender_score')}/{game_end.get('receiver_score')} logged")
    scale = var.get("reward_scale") or sc["vars"]["reward_scale"]
    raw_rw = summary.get("rewards") or {}
    rw_s = float(raw_rw.get(str(s_seat), s_score / R / scale["SENDER"])) if s_seat is not None else s_score / R / scale["SENDER"]
    rw_r = float(raw_rw.get(str(r_seat), r_score / R / scale["RECEIVER"])) if r_seat is not None else r_score / R / scale["RECEIVER"]
    if abs(rw_s - s_score / R / scale["SENDER"]) > 1e-9 or abs(rw_r - r_score / R / scale["RECEIVER"]) > 1e-9:
        warn.append(f"rewards {rw_s}/{rw_r} logged, {s_score / R / scale['SENDER']}/{r_score / R / scale['RECEIVER']} by the rule")
    for w in warn:
        print("warning:", w, file=sys.stderr)

    # ---- per-round classification (the paper's definitions)
    for x in rounds:
        st, sg, dc = x["state"], x["signal"], x["decision"]
        x["cls"] = (None if sg is None or st is None else
                    "true_high" if (sg == hi and st == fav) else "lie" if (sg == hi and st == oth) else
                    "true_low" if (sg == lo and st == oth) else "withheld")
        x["follow"] = None if sg is None else (dc == fa) == (sg == hi)
        cf = (fa if sg == hi else oa) if sg is not None else dc
        x["cf_s"], x["cf_r"] = SP[st][cf] if st else None, RP[st][cf] if st else None
        x["reveals"] = st is not None and RP[fav][dc] != RP[oth][dc]
    lie_r = [x["round"] for x in rounds if x["cls"] == "lie"]
    honest_r = [x["round"] for x in rounds if x["cls"] == "true_high"]
    withheld_r = [x["round"] for x in rounds if x["cls"] == "withheld"]
    labelled_oth = [x["round"] for x in rounds if x["state"] == oth and x["signal"] is not None]
    labelled_fav = [x["round"] for x in rounds if x["state"] == fav and x["signal"] is not None]
    lie_rate = len(lie_r) / len(labelled_oth) if labelled_oth else None
    honesty = len(honest_r) / len(labelled_fav) if labelled_fav else None
    hi_r = [x["round"] for x in rounds if x["signal"] == hi]
    lo_r = [x["round"] for x in rounds if x["signal"] == lo]
    hire_r = [x["round"] for x in rounds if x["decision"] == fa]
    follow_r = [x["round"] for x in rounds if x["follow"]]
    hire_after_hi = [r for r in hire_r if r in hi_r]
    no_after_lo = [r for r in lo_r if r not in hire_r]
    ignored_true = [x["round"] for x in rounds if x["cls"] == "true_high" and x["decision"] != fa]
    cf_s = sum(x["cf_s"] for x in rounds if x["cf_s"] is not None)
    cf_r = sum(x["cf_r"] for x in rounds if x["cf_r"] is not None)
    pa = var.get("prior_action") or oa
    never_r = sum(RP[x["state"]][pa] for x in rounds if x["state"])
    has_labels = any(x["signal"] is not None for x in rounds) and bool(info.get("use_label", var.get("use_label", True)))
    ref_lie = var.get("ref_lie_probability")
    ref_opt, ref_no, ref_full = var.get("ref_optimal_payoffs"), var.get("ref_no_info_payoffs"), var.get("ref_full_info_payoffs")
    paper = PAPER_S3.get(sid) if R == 5 else None
    sig_noun = sc["signal_noun"]
    short = sc["short"]

    # ---- steps (pledge before the draw, message, label, action), in log order
    steps = []
    for x in rounds:
        r, xi = x["round"], x["round"] - 1
        if x["pledge_actor"] is not None and s_seat is not None:
            steps.append({"seat": s_seat, "x": xi, "kind": "pledge", "head": f"Round {r} {MID} pledge (the {noun} is not drawn yet)",
                          "_move": {"action_id": "pledge", "payload": {"text": x["pledge"]}}, "_r": x, "_ph": "pledge"})
        if x["message"] is not None and s_seat is not None:
            steps.append({"seat": s_seat, "x": xi, "kind": "message", "head": f"Round {r} {MID} {noun} {x['state']} {MID} writes a message",
                          "_move": {"action_id": "write", "payload": {"message": x["message"]}}, "_r": x, "_ph": "write"})
        if x["signal"] is not None and s_seat is not None and info.get("scripted_role") != "SENDER":
            tag = {"lie": " (a lie)", "withheld": f" (a {fav} {noun})"}.get(x["cls"], "")
            steps.append({"seat": s_seat, "x": xi, "kind": x["cls"], "head": f"Round {r} {MID} {noun} {x['state']} {MID} sends {disp(x['signal'])}{tag}",
                          "_move": {"action_id": "signal", "payload": {"signal": x["signal"]}}, "_r": x, "_ph": "signal"})
        if r_seat is not None and info.get("scripted_role") != "RECEIVER":
            after = f"after {disp(x['signal'])}" if x["signal"] is not None else "after the message"
            steps.append({"seat": r_seat, "x": xi, "kind": "act_fav" if x["decision"] == fa else "act_oth",
                          "head": f"Round {r} {MID} {after} {MID} {sc['verb'].get(x['decision'], disp(x['decision'])).lower()} {MID} payoff {pay(x['r_pay'])}",
                          "_move": {"action_id": "decide", "payload": {"action": x["decision"]}}, "_r": x, "_ph": "decide"})
    for st in steps:
        st["act"] = json.dumps(st["_move"], ensure_ascii=False)

    # ---- what each player saw (rebuilt from the log; the saved prompt of the first decision where there is one)
    def table(P):
        return "; ".join(f"{s}: " + ", ".join(f"{a} {P[s][a]:g}" for a in var["action_labels"]) for s in (fav, oth))

    def history(upto, me_sender):
        lines = []
        for y in rounds:
            if y["round"] >= upto:
                break
            parts = []
            if y["pledge"] is not None and (commitment == "pledge_each_round" or y["round"] == 1):
                parts.append(f'pledge "{y["pledge"]}"')
            if me_sender:
                parts.insert(0, f"{noun} {y['state']} (private)")
            if y["signal"] is not None:
                parts.append(f"label {y['signal']}")
            parts.append(("you chose " if not me_sender else f"{rn} chose ") + y["decision"])
            parts.append(f"your payoff {(y['s_pay'] if me_sender else y['r_pay']):g} (private)")
            lines.append(f"- Round {y['round']}: " + "; ".join(parts) + ".")
        return lines

    for st in steps:
        x, ph = st["_r"], st["_ph"]
        r = x["round"]
        me_sender = ph != "decide"
        done = [y for y in rounds if y["round"] < r]
        score = sum((y["s_pay"] if me_sender else y["r_pay"]) or 0 for y in done)
        lines = ["(Rebuilt from the game log; the exact prompt for this turn was not saved.)",
                 f"You are seat P{st['seat']}. Your role: {'SENDER' if me_sender else 'RECEIVER'}, the {sn if me_sender else rn}.",
                 f"Round {r} of {R}. " + {
                     "pledge": f"Current phase: PLEDGE. Announce, in words, the signaling policy you intend to follow (public, not enforced; at most {var.get('pledge_max_words', 80)} words). This round's {noun} is not drawn yet.",
                     "write": f"Current phase: WRITE. Write this round's message to the {rn} (at most {var.get('message_max_words', 120)} words); it is shown during this round only.",
                     "signal": f"Current phase: SIGNAL. Send this round's label, one of {json.dumps(var['signal_labels'])} ({hi} claims {fav}).",
                     "decide": f"Current phase: DECIDE. Choose your action, one of {json.dumps(var['action_labels'])}."}[ph],
                 f"Public: P({fav}) = {prior:g}. Your payoffs: {table(SP if me_sender else RP)}. The {rn if me_sender else sn}'s payoffs: {table(RP if me_sender else SP)}.",
                 f"Your score so far: {score:g}."]
        hist = history(r, me_sender)
        lines.append("Earlier rounds (pledges, labels and decisions are public; your payoffs are private):" if hist else "Earlier rounds: none.")
        lines += hist
        now = []
        if x["pledge"] is not None and ph != "pledge" and (commitment == "pledge_each_round" or r == 1):
            now.append(f'- Pledge in force: "{x["pledge"]}"')
        if me_sender and ph in ("write", "signal") and x["state"]:
            now.append(f"- The {noun} this round is {x['state']} (private to you).")
        if ph in ("signal", "decide") and x["message"] is not None:
            now.append(f'- {"Your message" if me_sender else "The message"} (shown this round only): "{x["message"]}"')
        if ph == "decide" and x["signal"] is not None:
            now.append(f"- Label: {x['signal']}")
        if now:
            lines.append("This round:")
            lines += now
        st["state"] = "\n".join(lines)
    prompt = run.get("prompt") or {}
    if steps and prompt.get("text") and prompt.get("seat") == steps[0]["seat"] and prompt.get("phase") == PHASE[steps[0]["_ph"]] \
            and _round({"prompt": prompt["text"]}) == steps[0]["_r"]["round"]:
        steps[0]["state"] = en(common.state_part(prompt["text"]))
        steps[0]["_from_prompt"] = True

    # ---- feed
    hi_d, lo_d = disp(hi), disp(lo)
    hi_pp, lo_pp, hi_n, lo_n = sc["hi_pp"], sc["lo_pp"], sc["hi_noun"], sc["lo_noun"]
    sig_l = sig_noun.lower()

    def relation(x):
        if x["signal"] is None:
            return ""
        if x["follow"]:
            return f", in line with the {sig_l}"
        return f", against the {hi_n if x['signal'] == hi else lo_n}"

    feed = []
    for k, st in enumerate(steps):
        x, ph = st["_r"], st["_ph"]
        if ph == "pledge":
            text = f"Pledges, before this round's {noun} is drawn (public, not binding): {quote(x['pledge'])}"
        elif ph == "write":
            text = f"Learns privately that the {noun} is {x['state']}, then writes (shown to the {rn} during this round only): {quote(x['message'])}"
        elif ph == "signal":
            text = f"Sends the label {disp(x['signal'])}. " + {
                "true_high": f"True: the {noun} is {fav}.", "lie": f"A lie: the {noun} is {oth}.",
                "true_low": f"True: the {noun} is {oth}.", "withheld": f"Not a lie, but the {noun} is {fav}: a {hi_n} withheld."}[x["cls"]]
        else:
            verb = sc["verb"].get(x["decision"], f"Chooses {disp(x['decision'])}")
            learn = (f"The {rn}'s {pay(x['r_pay'])} tells it the {noun} was {x['state']}." if x["reveals"] else
                     f"The {rn}'s {pay(x['r_pay'])} would be the same for any {noun}, so it learns nothing about this one.")
            text = f"{verb}{relation(x)}. Payoffs: {sn} {pay(x['s_pay'])}, {rn} {pay(x['r_pay'])} (each player is told only its own). {learn}"
        feed.append({"x": st["x"], "step": k, "who": seat_names[st["seat"]], "kind": st["kind"], "text": text})
    if steps:
        feed.append({"x": steps[-1]["x"], "step": len(steps) - 1, "who": "Result",
                     "text": f"Game over after {plural(R, 'round')}. {SN}: total payoff {num(s_score)}, reward {dec2(rw_s)}. "
                             f"{RN}: total payoff {num(r_score)}, reward {dec2(rw_r)}. A reward is the mean payoff per round"
                             + ("." if scale["SENDER"] == 1 and scale["RECEIVER"] == 1 else f" {DIV} the role's fixed scale.")})

    attach_calls(steps, calls)
    has_reason = any(s.get("thinking") for s in steps)
    from_prompt = sum(1 for s in steps if s.get("_from_prompt"))
    for st in steps:
        st.setdefault("thinking", "")
        for k in [k for k in st if k.startswith("_")]:
            st.pop(k)

    # ---- kinds (only those used)
    kind_label = {"pledge": "Pledge", "message": "Message", "true_high": f"{sig_noun}: {hi_d} (true)", "lie": f"{sig_noun}: {hi_d} (a lie)",
                  "true_low": f"{sig_noun}: {lo_d} (true)", "withheld": f"{sig_noun}: {lo_d} ({fav} {noun})",
                  "act_fav": f"{RN}: {disp(fa)}", "act_oth": f"{RN}: {disp(oa)}"}
    kinds = {k: [kind_label[k], KIND_COLORS[k]] for k in KIND_COLORS if any(s["kind"] == k for s in steps)}

    # ---- chart 1: the round-by-round grid
    n = R
    labels = [f"Round {i + 1}" for i in range(n)]
    byr = {x["round"]: x for x in rounds}
    cls_word = {"true_high": ("True", "c3", f"True: {article(fav)} {fav} {noun} {hi_pp}"),
                "lie": ("Lie", "c4", f"Lie: {article(oth)} {oth} {noun} {hi_pp}"),
                "true_low": ("True", "c3", f"True: {article(oth)} {oth} {noun} {lo_pp}"),
                "withheld": ("Withheld", "hold", f"Withheld: {article(fav)} {fav} {noun} {lo_pp}")}
    # The page draws cell text in white; yellow and green fills are drawn at full opacity (o 1) so that text keeps what
    # contrast those light fills allow.
    row_pl, row_state, row_sig, row_cls, row_dec = [None] * n, [None] * n, [None] * n, [None] * n, [None] * n
    # Each pledge cell names what the pledge promises: "Only strong" (the high label only for the favored state), "Mixed 0.45"
    # (a stated probability for the other state), or "Other" (read as neither); the tip says it in words and quotes it.
    truthful = {r: truthful_pledge(x["pledge"], fav, oth) for r, x in byr.items() if x["pledge"]}
    hi_inf = re.sub(r"^(\w+?)s\b", r"\1", sc["hi_verb"])          # "recommends" -> "recommend"
    mixed = {r: m for r, m in ((r, mixed_pledge(byr[r]["pledge"], fav, oth, hi_inf.split()[0])) for r in truthful if not truthful[r]) if m}
    read = {r: "only" if truthful[r] else "mixed" if r in mixed else None for r in truthful}

    def mix_policy(m):
        return (f"every {fav} {noun} and {oth} ones" if m["every"] else f"{oth} {noun}s") + f" with probability {m['p']:.2f}"
    for r, x in byr.items():
        i = r - 1
        if not 0 <= i < n:
            continue
        if x["pledge"]:
            if read[r] == "only":
                t_, tip_ = f"Only {fav}", f"Promises {sc['promise']}: "
            elif read[r] == "mixed":
                t_ = f"Mixed {mixed[r]['p']:.2f}"
                tip_ = (f"Promises to {hi_inf} {mix_policy(mixed[r])}"
                        + (f" (sender-optimal: {dec2(ref_lie)})" if ref_lie is not None else "") + ": ")
            else:
                t_, tip_ = "Other", f"Read as neither a promise {sc['promise']} nor a stated mixed policy: "
            row_pl[i] = {"c": "hold", "t": t_, "tip": tip_ + quote(trim(x["pledge"]))}
        elif commitment == "pledge" and byr.get(1, {}).get("pledge"):
            row_pl[i] = {"c": "hold", "o": .35, "tip": "No new pledge; the one made before Round 1 stands"}
        if x["state"]:
            row_state[i] = {"c": "c1" if x["state"] == fav else "c2", "t": x["state"].capitalize(),
                            "tip": f"{x['state'].capitalize()} {noun} (only the {sn} knew)"}
        if x["signal"] is not None:
            row_sig[i] = {"c": "c1" if x["signal"] == hi else "c2", "t": sc["sig_short"].get(x["signal"], disp(x["signal"]).capitalize()),
                          "tip": f"Label {disp(x['signal'])} (public)"}
            w = cls_word[x["cls"]]
            row_cls[i] = {"c": w[1], "t": w[0], "tip": w[2]}
            if w[1] in ("c3", "c4"):
                row_cls[i]["o"] = 1
        row_dec[i] = {"c": "c1" if x["decision"] == fa else "c2", "t": short.get(x["decision"], disp(x["decision"])),
                      "tip": f"{sc['verb'].get(x['decision'], disp(x['decision']))}{relation(x)}; payoffs: {sn} {pay(x['s_pay'])}, {rn} {pay(x['r_pay'])}"}
    dec_row = "HR decision" if RN.startswith("HR") else "Decision"
    has_pledges = any(row_pl)
    g_rows, g_cells = [noun.capitalize(), sig_noun, f"{sig_noun} was", dec_row], [row_state, row_sig, row_cls, row_dec]
    if not has_labels:
        g_rows, g_cells = [g_rows[0], g_rows[3]], [row_state, row_dec]
    if has_pledges:
        g_rows, g_cells = ["Pledge"] + g_rows, [row_pl] + g_cells
    # what the pledges promised, against what the labels did (said only when every pledge in question is read safely)
    lie_after = [r for r in lie_r if truthful.get(r)]
    admitted = [r + 1 for r in lie_after if r + 1 in byr and admits_breach(byr[r + 1]["pledge"])]
    one = len(lie_after) == 1

    def mix_words(rs):
        ps = sorted({f"{mixed[r]['p']:.2f}" for r in rs})
        return (f"a mixed policy ({oth} {noun}s {hi_pp} with probability {ps[0]})" if len(ps) == 1 else
                f"mixed policies ({oth} {noun}s {hi_pp} with probability " + ", ".join(ps[:-1]) + f" or {ps[-1]})")

    def pledges_of(rs):
        return f"{rounds_text(rs)} pledge" if len(rs) == 1 else f"pledges of {rounds_text(rs)}"
    # a label goes against its round's pledge: a lie after "only", a withheld high label after "every <favored> ... and mixed"
    broke = [r for r in sorted(read) if (read[r] == "only" and byr[r]["cls"] == "lie")
             or (read[r] == "mixed" and mixed[r]["every"] and byr[r]["cls"] == "withheld")]
    all_read = commitment == "pledge_each_round" and len(truthful) == len(rounds) and all(read.values())
    if all_read and all(v == "only" for v in read.values()):
        p_clause = (f"Every pledge promised {sc['promise']}"
                    + ((f"; the {rounds_text(lie_after)} lie came right after such a pledge" if one else
                        f"; the lies in {rounds_text(lie_after)} each came right after such a pledge") if lie_after else
                       f", and no {sig_l} broke it" if has_labels else "")
                    + ((f", and the Round {admitted[0]} pledge admits it" if one else
                        f", and the pledge{'s' if len(admitted) > 1 else ''} of {rounds_text(admitted)} "
                        f"admit{'' if len(admitted) > 1 else 's'} the lie before {'them' if len(admitted) > 1 else 'it'}")
                       if admitted else "") + ".")
    elif all_read:                                        # some pledges state a mixed policy, the others "only"
        M = sorted(r for r, v in read.items() if v == "mixed")
        O = sorted(r for r, v in read.items() if v == "only")
        p_clause = f"The {pledges_of(M)} promised {mix_words(M)}" if O else f"Every pledge promised {mix_words(M)}"
        if O:
            tail = O == list(range(O[0], R + 1)) and M == list(range(1, O[0]))
            p_clause += (f"; from Round {O[0]} on every pledge promised {sc['promise']}" if tail else
                         f"; the {pledges_of(O)} promised {sc['promise']}")
        p_clause += ((f"; the {sig_l}s went against their round's pledge in {rounds_text(broke)}." if broke else
                      f", and no {sig_l} broke its round's pledge.") if has_labels else ".")
    elif commitment == "pledge" and truthful.get(1) and lie_r:
        p_clause = f"The pledge before Round 1 promised {sc['promise']}; the {sig_l}s broke it in {rounds_text(lie_r)}."
    elif commitment == "pledge" and 1 in mixed and has_labels:
        bad = [x["round"] for x in rounds if mixed[1]["every"] and x["cls"] == "withheld"]
        p_clause = (f"The pledge before Round 1 promised {mix_words([1])}"
                    + (f"; the {sig_l}s went against it in {rounds_text(bad)}." if bad else f", and no {sig_l} broke it."))
    elif lie_after:
        p_clause = (f"The {rounds_text(lie_after)} lie came right after a pledge {sc['promise']}." if one else
                    f"The lies in {rounds_text(lie_after)} each came right after a pledge {sc['promise']}.")
    else:
        p_clause = ""
    legend = [[f"{fav.capitalize()} {MID} {hi_d} {MID} {disp(fa)}", "c1"], [f"{oth.capitalize()} {MID} {lo_d} {MID} {disp(oa)}", "c2"]]
    if has_labels:
        legend.append([f"{sig_noun} true to the {noun}", "c3"])
        if lie_r:
            legend.append([cls_word["lie"][2], "c4"])
        if withheld_r:
            legend.append([cls_word["withheld"][2], "hold"])
    if has_labels:
        if hire_r and len(hire_r) < R:
            hire_part = (f"the {rn} {sc['fav_past']} only in {rounds_text(hire_r)}"
                         + ((", after the lie" if len(lie_r) == 1 else ", after the lies") if hire_r == lie_r else ""))
        else:
            hire_part = f"the {rn} {sc['fav_past']} every time" if hire_r else f"the {rn} never chose {disp(fa)}"
        g_tail = (f"The {sn} {hi_pp} {len(hi_r)} of the {R} {noun}s"
                  + (f", {numword(len(lie_r))} of them {oth} ({rounds_text(lie_r)})" if lie_r else ", all of them truthfully")
                  + f"; {hire_part}.")
    else:
        g_tail = ""
    hover = f"Hover a column for {'the pledge and ' if has_pledges else ''}the payoffs."
    saw = "the pledge, the message and the label" if has_pledges else "the message and the label"
    g_sub = (f"Blue is the {sn}'s side ({fav}, {hi_d}, {disp(fa)}) and orange the other ({oth}, {lo_d}, {disp(oa)}): where the "
             f"{noun.capitalize()}, {sig_noun} and {dec_row} cells of a round are all one color, the {sig_l} told the truth and the "
             f"decision fit the {noun}. Only the {sn} saw the {noun}; the {rn} saw {saw}. {hover} " + g_tail
             if has_labels else
             f"Blue is the {sn}'s side ({fav}, {disp(fa)}) and orange the other ({oth}, {disp(oa)}). Only the {sn} saw the {noun}. {hover}")
    g_sub = (g_sub + " " + p_clause).strip()
    g_title = (f"Round by round: the pledge, the {noun}, the {sig_l} and the {rn}'s decision" if has_pledges else
               f"Round by round: the {noun}, the {sig_l} and the {rn}'s decision")
    # "bars": [] lays the shared x axis out in bands (one column per round), so the decision timeline lines up with the grid.
    charts = [{"title": g_title, "sub": g_sub, "height": 150, "bars": [],
               "grid": {"rows": g_rows, "cells": g_cells, "legend": legend, "rowHeight": 30}}]

    # ---- what following the labels is worth: the total with these draws, and per round in expectation if the sender kept the
    # scheme it showed (its lie rate and honesty), against the prior action; the prior as the fraction the config rounds (1/3)
    def per(v):
        return "0" if abs(v) < 0.005 else (MINUS if v < 0 else "") + f"{abs(v):.2f}"
    pf = float(Fraction(prior).limit_denominator(12))
    pf = pf if abs(pf - prior) < 1e-3 else prior
    n_fav = sum(1 for x in rounds if x["state"] == fav)
    exp_follow = ""
    if lie_rate is not None and honesty is not None:
        e_f = (pf * (honesty * RP[fav][fa] + (1 - honesty) * RP[fav][oa])
               + (1 - pf) * (lie_rate * RP[oth][fa] + (1 - lie_rate) * RP[oth][oa]))
        e_n = pf * RP[fav][pa] + (1 - pf) * RP[oth][pa]
        same = abs(e_f - e_n) < 0.005
        exp_follow = (f"at the {sn}'s {lie_rate:.2f} lie rate" + ("" if honesty == 1 else f" and {honesty:.2f} honesty")
                      + f", following is worth {per(e_f)} per round in expectation, "
                      + (f"no better than {sc['never']}" if same else
                         f"{'more' if e_f > e_n else 'less'} than {sc['never']} ({per(e_n)})"))

    # ---- chart 2: running totals, and the same labels followed
    cum_s, cum_r, cf_cs, cf_cr, acc = [None] * n, [None] * n, [None] * n, [None] * n, [0, 0, 0, 0]
    mean_s, mean_r = [None] * n, [None] * n
    for x in rounds:
        i = x["round"] - 1
        if not 0 <= i < n:
            continue
        acc = [acc[0] + (x["s_pay"] or 0), acc[1] + (x["r_pay"] or 0), acc[2] + (x["cf_s"] or 0), acc[3] + (x["cf_r"] or 0)]
        cum_s[i], cum_r[i], cf_cs[i], cf_cr[i] = acc
        mean_s[i], mean_r[i] = round(acc[0] / (i + 1), 4), round(acc[1] / (i + 1), 4)
    # Two series that coincide are drawn as one (the second would hide the first): the players' totals, or the totals had the
    # labels been followed (in the recommendation letter game both equal the strong students hired).
    same_act, same_cf = cum_s == cum_r, has_labels and cf_cs == cf_cr
    lines2 = ([{"name": "Both players", "color": "ink", "v": cum_s, "dots": True, "end": True}] if same_act else
              [{"name": SN, "color": "c1", "v": cum_s, "dots": True, "end": True},
               {"name": RN, "color": "c2", "v": cum_r, "dots": True, "end": True}])
    if same_cf:
        lines2.append({"name": f"Both, if the {sig_l}s were followed", "color": "hold", "dash": True, "thin": True, "v": cf_cs})
    elif has_labels:
        lines2 += [{"name": f"{SN} if the {sig_l}s were followed", "color": "c1", "dash": True, "thin": True, "v": cf_cs},
                   {"name": f"{RN} if it followed the {sig_l}s", "color": "c2", "dash": True, "thin": True, "v": cf_cr}]
    const_s = len({v for v in cum_s if v is not None}) == 1
    sub2 = ("Total payoff after each round. Solid: what happened"
            + ((f" (one line: the two totals are equal, {num(cum_s[-1] or 0)} throughout)" if const_s else
                " (one line: the two totals are equal)") if same_act else "") + ". "
            + (f"Dashed: the same {sig_l}s, had the {rn} {sc['fav_past']} after every {hi_n} and not otherwise"
               + (" (one line: the two totals would have been equal)" if same_cf else "") + ": it would have totalled "
               f"{pay(cf_r)} instead of {pay(r_score)} ({per(cf_r / R)} per round), and the {sn} {pay(cf_s)} instead of {pay(s_score)}. "
               if has_labels else "")
            + f"{sc['never'].capitalize()}, the best choice without information, would have given the {rn} {pay(never_r)}."
            + (f" That {pay(cf_r)} comes with {n_fav} {fav} {noun}s in {R} ({R * pf:.2f} expected); {exp_follow}." if has_labels and exp_follow else ""))
    ints = all(float(v).is_integer() for v in (cum_s + cum_r + cf_cs + cf_cr) if v is not None)
    chart2 = {"title": f"Running payoff, and what following the {sig_l}s would have paid", "sub": sub2, "height": 230,
              "fmt": "int" if ints else "price2", "zero": True, "lines": lines2, "bars": []}
    charts.append(chart2)

    # ---- chart 3: mean payoff per round against the paper's benchmarks (config ref_*)
    lines3 = [{"name": SN, "color": "c1", "v": mean_s, "dots": True, "end": True}, {"name": RN, "color": "c2", "v": mean_r, "dots": True, "end": True}]
    # One dashed line per benchmark value, in the player's color (gray when it is both players'). A benchmark at 0 would lie
    # on the zero line this chart draws, where it cannot be seen, so the zero line stands for it and the subtitle says so.
    refs = {}
    for nm, pair in (("no information", ref_no), ("full information", ref_full), ("sender-optimal", ref_opt)):
        if not pair:
            continue
        s_v, r_v = round(float(pair[0]), 4), round(float(pair[1]), 4)
        if s_v == r_v:
            refs.setdefault(s_v, []).append((nm, "both"))
        else:
            refs.setdefault(s_v, []).append((nm, "S"))
            refs.setdefault(r_v, []).append((nm, "R"))
    who_name = {"both": "both", "S": sn, "R": rn}

    def ref_name(items):
        whos = {w for _, w in items}
        if len(whos) == 1:
            nms = [nm for nm, _ in items]
            t = (" and ".join(nms) if len(nms) <= 2 else ", ".join(nms[:-1]) + " and " + nms[-1]) + ", " + who_name[next(iter(whos))]
        else:
            t = "; ".join(f"{nm}, {who_name[w]}" for nm, w in items)
        return t[:1].upper() + t[1:]
    on_zero = ""
    for v in sorted(refs, reverse=True):
        who = {w for _, w in refs[v]}
        if abs(v) < 1e-9:
            on_zero = ref_name(refs[v])
            continue
        lines3.append({"name": f"{ref_name(refs[v])} ({dec2(v)})", "color": "c1" if who == {"S"} else "c2" if who == {"R"} else "hold",
                       "dash": True, "thin": True, "v": [v] * n})
    pr = lambda p: f"({dec2(p[0])}, {dec2(p[1])})"
    sub3 = ("Running mean payoff per round; the final value is each player's reward"
            + ("" if scale["SENDER"] == 1 and scale["RECEIVER"] == 1 else " before dividing by the role's fixed scale") + ". "
            + (f"Dashed: the benchmarks of the paper's Appendix C.1 (stored in the game's config), as ({sn}, {rn}): no information {pr(ref_no)}, "
               f"full information {pr(ref_full)} and the sender-optimal scheme {pr(ref_opt)}, which {sc['hi_verb']} every {fav} {noun} and "
               f"each {oth} one with probability {dec2(ref_lie)}. " if ref_no and ref_full and ref_opt and ref_lie is not None else "")
            + (f"The solid zero line also marks the benchmarks at 0 ({on_zero[:1].lower() + on_zero[1:]}). " if on_zero else "")
            + (f"In this five-round setting the paper's agents reached ({paper[0]:.2f}, {paper[1]:.2f}). " if paper else "")
            + f"Here: ({dec2(s_score / R)}, {dec2(r_score / R)})"
            + (f"; the {rn} ends below the {dec2(never_r / R)} it could guarantee by {sc['never']}." if r_score < never_r else "."))
    # When neither player's running mean moves (no hire at all, say), this chart would be flat lines against constants: it is
    # left out, and its numbers join the running-payoff chart's subtitle.
    flat3 = all(len({v for v in mv if v is not None}) == 1 for mv in (mean_s, mean_r))
    if not flat3:
        charts.append({"title": "Mean payoff per round so far, against the theory", "sub": sub3, "height": 230, "fmt": "price2", "zero": True,
                       "lines": lines3, "bars": []})
    else:
        bench = [f"{nm} {pr(pair)}" for nm, pair in (("no information", ref_no), ("full information", ref_full), ("sender-optimal", ref_opt)) if pair]
        chart2["sub"] += (f" Mean payoff per round, as ({sn}, {rn}): here ({dec2(s_score / R)}, {dec2(r_score / R)})"
                          + (f"; the paper's Appendix C.1 benchmarks are " + (", ".join(bench[:-1]) + " and " + bench[-1] if len(bench) > 1 else bench[0])
                             if bench else "")
                          + (f"; the paper's agents reached ({paper[0]:.2f}, {paper[1]:.2f}) in this five-round setting" if paper else "")
                          + (f"; the {rn} ends below the {dec2(never_r / R)} it could guarantee by {sc['never']}." if r_score < never_r else "."))

    # ---- tiles
    sp_const = {SP[fav][fa], SP[oth][fa]}
    flat = len(sp_const) == 1 and SP[fav][oa] == 0 and SP[oth][oa] == 0
    detail = "; ".join(f"Round {r}, {article(byr[r]['state'])} {byr[r]['state']} {noun}" for r in hire_r)
    spark = [float(v) for v in cum_s if v is not None]
    moves = len(set(spark)) > 1                           # a flat sparkline says nothing: drawn only when the total moves
    t1 = (f"{plural(len(hire_r), sc['fav_event'])} in {plural(R, 'round')}" + (f" ({detail})" if hire_r and len(hire_r) <= 2 else "")
          + (f", paid {next(iter(sp_const)):g} each" if flat and hire_r else "")
          + (f". Sender-optimal scheme: {dec2(ref_opt[0])} per round" if ref_opt else "") + (". Line: running total." if moves else "."))
    tiles = [{"k": f"{SN}'s reward", "v": dec2(rw_s), "s": t1[:1].upper() + t1[1:]}]
    if moves:
        tiles[0]["spark"] = spark
    if len(hire_r) == 1:
        rr = hire_r[0]
        r_hires = f"Its only {sc['fav_event']} was {article(byr[rr]['state'])} {byr[rr]['state']} {noun} (Round {rr}, {pay(byr[rr]['r_pay'])})"
    elif not hire_r:
        r_hires = f"It never {sc['fav_past']}"
    else:
        r_hires = f"It {sc['fav_past']} in {rounds_text(hire_r)}, total payoff {pay(r_score)}"
    tiles.append({"k": f"{RN}'s reward", "v": dec2(rw_r),
                  "s": r_hires + (f". Following every {sig_l} would have totalled {pay(cf_r)} with these {noun}s ({per(cf_r / R)} per round); "
                                  f"{sc['never']}, {pay(never_r)}" if has_labels else f". {sc['never'].capitalize()} would have totalled {pay(never_r)}")
                       + (f". {exp_follow[:1].upper() + exp_follow[1:]}" if has_labels and exp_follow else "") + "."})
    if has_labels:
        if lie_rate is not None:
            tiles.append({"k": "Lie rate", "v": f"{lie_rate:.2f}",
                          "s": f"{hi_pp.capitalize()} {of_the(len(lie_r), len(labelled_oth), oth + ' ' + noun)}"
                               + (f" ({rounds_text(lie_r)})" if lie_r else "")
                               + (f" and {of_the(len(honest_r), len(labelled_fav), fav + ' one')}" if labelled_fav else "")
                               + (f". Sender-optimal: {dec2(ref_lie)}" if ref_lie is not None else "")
                               + (f"; the paper's agents: {paper[2]:.2f}" if paper else "") + "."})
        tiles.append({"k": f"{RN} followed the {sig_l}", "v": f"{len(follow_r)} of {R}",
                      "s": f"{sc['fav_past'].capitalize()} after {of_the(len(hire_after_hi), len(hi_r), hi_n)}"
                           + (" (the lie)" if hire_after_hi and hire_after_hi == lie_r and len(lie_r) == 1 else
                              " (the lies)" if hire_after_hi and hire_after_hi == lie_r else "")
                           + (f" and {sc['oth_past']} after {of_the(len(no_after_lo), len(lo_r), lo_n)}" if lo_r else "")
                           + (f". It passed on {of_the(len(ignored_true), len(honest_r), 'true ' + hi_n, words=True)} ({rounds_text(ignored_true)})"
                              if ignored_true else "") + "."})

    # ---- title, lede, foot
    if lie_r and hire_r and set(hire_r) == set(lie_r):
        one = len(lie_r) == 1
        title = (f"The {sn}'s {'one lie is' if one else numword(len(lie_r)) + ' lies are'} the {rn}'s only {sc['fav_event']}"
                 + (f", {article(oth)} {oth} {noun} in Round {lie_r[0]}" if one else f", {oth} {noun}s in {rounds_text(lie_r)}"))
        k_ = len(ignored_true)
        if k_ and sid == "rel":
            title += (f"; the {numword(k_)} {fav} {noun}{'s' if k_ > 1 else ''} the {sn} truthfully {hi_pp} "
                      + ("is turned down" if k_ == 1 else "are both turned down" if k_ == 2 else "are all turned down"))
        elif k_:
            title += f"; the {rn} goes against {numword(k_)} true {hi_n}{'s' if k_ > 1 else ''}"
        title += "."
    elif has_labels and not lie_r and not withheld_r and len(follow_r) == R:
        title = f"The {sn} tells the truth every round and the {rn} follows every {sig_l}: full information"
    elif has_labels and not lie_r and not withheld_r and not hire_r:
        # every label true, yet no favored action: the receiver's follows are only its default after the low label
        k_ = len(ignored_true)
        title = (f"The {sn} tells the truth every round, yet the {rn} {sc['never_does']}: "
                 + (f"it turns down {of_the(k_, k_, f'truthfully {hi_pp} {fav} {noun}', words=True)}" if sid == "rel" else
                    f"it goes against {of_the(k_, k_, f'true {hi_n}', words=True)}"))
    else:
        title = (f"{SN} {dec2(rw_s)}, {rn} {dec2(rw_r)}"
                 + ((f": {plural(len(lie_r), 'lie')} about {plural(len(labelled_oth), oth + ' ' + noun)}" if labelled_oth else
                     f": no {oth} {noun} drawn") + f", and the {rn} followed {len(follow_r)} of {R} {sig_l}s" if has_labels else ""))
    s_pay_txt = (f"The {sn} earns {next(iter(sp_const)):g} for every {sc['fav_event']}" if flat else
                 f"The {sn} is paid by its own table of {noun} and action")
    r_parts = [f"{pay(RP[s][a])} for {sc['gerund'].get(a, disp(a))} {article(s)} {s} {noun}" for s in (fav, oth) for a in (fa, oa) if RP[s][a] != 0]
    if any(RP[s][a] == 0 for s in (fav, oth) for a in (fa, oa)):
        r_parts.append("0 otherwise")
    r_pay_txt = (", ".join(r_parts[:-1]) + " and " + r_parts[-1]) if len(r_parts) > 1 else (r_parts[0] if r_parts else "nothing")
    when = {"pledge_each_round": "Before each draw", "pledge": "Before the first round"}.get(commitment)
    labels_q = f"{LQ}{hi_d}{RQ} or {LQ}{lo_d}{RQ}"
    say = (f"a short message and the label {labels_q}" if (var.get("message_max_words") or 0) > 0 and has_labels else
           f"the label {labels_q}" if has_labels else "a free-text message")
    lede = (f"Each round a new {noun} is drawn, {fav} with probability {frac(prior)} and {oth} otherwise, and only the {sn} learns which. "
            + (f"{when} the {sn} publicly pledges {sc['policy']} (the pledge is not enforced); then it sends the {rn} {say}, and the {rn}, "
               f"who never sees the {noun}, {sc['decides']}. " if when else
               f"The {sn} sends the {rn} {say}, and the {rn}, who never sees the {noun}, {sc['decides']}. ")
            + f"{s_pay_txt} and the {rn} {r_pay_txt}; each player's reward is its mean payoff over the {numword(R)} rounds"
            + ("." if scale["SENDER"] == 1 and scale["RECEIVER"] == 1 else ", divided by a fixed scale."))
    model = (b.get("meta") or {}).get("model") or run.get("model") or ""
    model = model.split("/")[-1] if model else "the model"
    date = str((b.get("meta") or {}).get("recorded_at") or run.get("recorded_at") or DEFAULT_DATE)[:10]
    seed = run.get("seed", cfg.get("seed"))
    scen_name = {"rel": "recommendation letter", "cor": "courtroom", "lae": "law enforcement"}.get(sid, sid)
    cfg_txt = (f"{scen_name}, {plural(R, 'round')}, " + {"pledge_each_round": "a pledge before every round",
                                                         "pledge": "one pledge before round 1"}.get(commitment, "no pledge")
               + (", message and label" if (var.get("message_max_words") or 0) > 0 and has_labels else ", label only" if has_labels else ", message only"))
    setting = ("five-round setting S3, with each round's pledge as the verbalized commitment" if R > 1 else
               "one-round setting S1 (polarized labels)" if has_labels else "one-round setting S2 (free text)")
    foot = (f"Data: the {date} recording of {GID} ({cfg_txt}; seed {seed}), "
            + ("both seats" if n_seats == 2 else "the one seat") + f" played by {model}; every number on this page is computed from its game log. "
            + ("Observations and reasoning come from the saved model calls. " if has_reason else
               "This recording was made before reasoning was saved; the game will be re-recorded with reasoning. "
               "Each action is rebuilt from the move the log records (the raw answers were not saved). ")
            + ("" if from_prompt == len(steps) else
               ("The observation of the first decision is the saved prompt; the others are rebuilt from the log and show only what that "
                "player could see (a message only in its own round, the state only to the sender). " if from_prompt else
                "The observations are rebuilt from the log and show only what that player could see (a message only in its own round, "
                "the state only to the sender). "))
            + (f"Seat P{s_seat} is the {sn} (sender), seat P{r_seat} the {rn} (receiver). " if s_seat is not None and r_seat is not None else "")
            + f"Lie rate = {oth} {noun}s {hi_pp} {DIV} {oth} {noun}s and honesty = {fav} {noun}s {hi_pp} {DIV} {fav} {noun}s"
            + (f" (here {dec2(honesty)})" if honesty is not None else "") + ", as the paper defines them. "
            f"The benchmarks are the paper's Appendix C.1 values, stored in the game's config. Game: Li et al., {LQ}Verbalized Bayesian "
            f"Persuasion{RQ} (ICML 2026), {setting}"
            + (f"; the paper reports {sn} {paper[0]:.2f}, {rn} {paper[1]:.2f} and lie probability {paper[2]:.2f} there (Figure 4, Table 1)."
               if paper else "."))
    d = {"id": GID, "name": "Bayesian persuasion", "title": title, "lede": lede, "seatNames": seat_names, "tiles": tiles[:4], "foot": foot,
         "x": {"n": n, "labels": labels}, "charts": charts, "kinds": kinds, "steps": steps, "feed": feed,
         "feedTitle": "Pledges, messages, labels and decisions",
         "feedSub": f"Everything the two players did, in order. Pledges, labels and decisions were public; each message was shown to the {rn} "
                    f"during its own round only; the {noun} was known only to the {sn}, and each payoff only to its own player. "
                    f"The selected step is highlighted; later entries are hidden."}
    d["eyebrow"] = (common.EYEBROW.format(gid=GID, date=date) if has_reason
                    else f"social_game_env {MID} {GID} {MID} {model} {MID} recorded {date} {MID} reasoning not saved")
    return d

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        raise SystemExit(selftest(sys.argv[2]))
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    common.write(build(src, calls), out, date=recorded_date(src))
