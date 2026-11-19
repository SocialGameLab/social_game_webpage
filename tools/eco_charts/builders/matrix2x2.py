"""Replay data for the repeated 2x2 matrix games of the earlier eco batch: Selten and Chmura (2008) games 1-12 (eco_sc1 ..
eco_sc12) and Goeree, Holt and Palfrey (2003) game 4 (eco_ghp4).

usage: python3 -I matrix2x2.py GAME_ID BUNDLE_DIR OUT_DIR [CALLS.jsonl]
       python3 -I matrix2x2.py --test      (the announcement reader's regression cases; they also run on every build)
  GAME_ID     eco_sc1 .. eco_sc12 or eco_ghp4 (selects the published reference numbers below; must match the bundle's run.json)
  BUNDLE_DIR  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, transcript.md)
  OUT_DIR     where <OUT_DIR>/<GAME_ID>.json is written (via common.write)
  CALLS       optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). A call
              belongs to the decision of the same seat, phase and round, read from its prompt ("You are seat Pk",
              "Current phase: X", the public state's "round"); when a prompt does not show them, calls are taken in time
              order, seat by seat. Among a decision's calls the first one whose answer is the logged move (the choice, or
              the message text) is the decision's call; earlier ones were format retries and their reasoning is kept above
              it. Its reasoning becomes the step's thinking and its prompt the observation.

Game (rules docs, eco_sc1.yaml / eco_ghp4.yaml): each round, if talk_passes > 0, a TALK phase in which the seats send
talk_passes public, non-binding messages each (sequential: P0, then P1; logged as "talk" events with "actor"), then a DECIDE
phase in which both seats secretly choose A or B (simultaneous; P0 is the row player, P1 the column player). The log has no
event per choice: both choices, payoffs and running scores are in the round's round_result event, so each round gives the
talk steps (log order) and then two choice steps, P0's first. Every number shown is taken from the log; payoffs are checked
against the logged payoff table, running scores against the payoffs and the final rewards against the summary, and any
disagreement is printed as a warning. The mixed equilibrium is computed exactly (fractions) from the logged table and checked
against the published one; the human benchmark rates are the published constants of each game (rules doc: Human Benchmark)."""
import json, os, re, sys, unicodedata
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

CELLS = ("AA", "AB", "BA", "BB")
OTHER = {"A": "B", "B": "A"}
DOT, TIMES, ARROW = "·", "×", "→"
ROLE = ("row", "column")
SEATS = ["Row player", "Column player"]          # seat P0 picks the row of the payoff table, seat P1 the column
CH_COLOR = {"A": "c1", "B": "c2"}                # choices (grid cells, step kinds)
PL_COLOR = ("c3", "c4")                          # players (bars, lines, the outcome row: whose aim the round met)
NASH_COLOR = "ink"                                # the equilibrium reference line shared by both players
TIP_W = 40                                        # widest grid hover line, in characters (see the grid in build())

SC = "Selten and Chmura (2008)"
GHP = "Goeree, Holt and Palfrey (2003)"
SC_PROTOCOL = "laboratory play without messages, with a new random partner every round"
GHP_PROTOCOL = "laboratory play without messages"


def _ref(src, k, table, human, nash):
    return {"src": src, "k": k, "table": {c: tuple(v) for c, v in zip(CELLS, table)}, "human": human, "nash": nash}


# Published payoff tables, human rates of choosing A (row, column) and mixed-equilibrium rates as published (rules docs:
# Payoff Matrix, Human Benchmark). Properties of the games, not of any recording.
REF = {
    "eco_sc1": _ref(SC, 1, [(10, 8), (0, 18), (9, 9), (10, 8)], (0.079, 0.690), (0.091, 0.909)),
    "eco_sc2": _ref(SC, 2, [(9, 4), (0, 13), (6, 7), (8, 5)], (0.217, 0.527), (0.182, 0.727)),
    "eco_sc3": _ref(SC, 3, [(8, 6), (0, 14), (7, 7), (10, 4)], (0.163, 0.793), (0.273, 0.909)),
    "eco_sc4": _ref(SC, 4, [(7, 4), (0, 11), (5, 6), (9, 2)], (0.286, 0.736), (0.364, 0.818)),
    "eco_sc5": _ref(SC, 5, [(7, 2), (0, 9), (4, 5), (8, 1)], (0.327, 0.664), (0.364, 0.727)),
    "eco_sc6": _ref(SC, 6, [(7, 1), (1, 7), (3, 5), (8, 0)], (0.445, 0.596), (0.455, 0.636)),
    "eco_sc7": _ref(SC, 7, [(10, 12), (4, 22), (9, 9), (14, 8)], (0.141, 0.564), (0.091, 0.909)),
    "eco_sc8": _ref(SC, 8, [(9, 7), (3, 16), (6, 7), (11, 5)], (0.250, 0.586), (0.182, 0.727)),
    "eco_sc9": _ref(SC, 9, [(8, 9), (3, 17), (7, 7), (13, 4)], (0.254, 0.827), (0.273, 0.909)),
    "eco_sc10": _ref(SC, 10, [(7, 6), (2, 13), (5, 6), (11, 2)], (0.366, 0.699), (0.364, 0.818)),
    "eco_sc11": _ref(SC, 11, [(7, 4), (2, 11), (4, 5), (10, 1)], (0.331, 0.652), (0.364, 0.727)),
    "eco_sc12": _ref(SC, 12, [(7, 3), (3, 9), (3, 5), (10, 0)], (0.439, 0.604), (0.455, 0.636)),
    "eco_ghp4": _ref(GHP, 4, [(200, 160), (160, 10), (370, 200), (10, 370)], (0.470, 0.650), (0.531, 0.469)),
}


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_RUN = re.compile("[぀-ヿ㐀-鿿가-힯]+")
_PUNCT = {"。": ". ", "、": ", ", "「": '"', "」": '"', "『": '"', "』": '"', "【": "[", "】": "]",
          "《": '"', "》": '"', "〈": "<", "〉": ">", "〜": "~"}


def en(s):
    """English only on the page: full-width forms become ASCII, CJK punctuation its ASCII counterpart, and any remaining run of
    Chinese, Japanese or Korean characters (the model sometimes reasons in Chinese) a visible marker."""
    if not isinstance(s, str) or not re.search("[　-ヿ㐀-鿿가-힯＀-￯]", s):
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
    return re.sub("[　-〿＀-￯]", "", s)


# ------------------------------------------------------------------------------------------------ formatting
NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def numword(n):
    return NUM[n] if 0 <= n < len(NUM) else str(n)


def g(v):
    """A payoff or score: an integer without decimals, anything else with up to two (exact Fractions included)."""
    v = Fraction(v)
    return f"{int(v):,}" if v.denominator == 1 else f"{float(v):,.2f}".rstrip("0").rstrip(".")


def frac(v):
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def pct(v, d=1):
    return f"{float(v) * 100:.{d}f}%"


def pct_short(v):
    """12.5% stays 12.5%, 50% and 0% lose the '.0' (shares of a few rounds are often exact halves; no rounding to even)."""
    t = f"{float(Fraction(v) * 100):.1f}"
    return (t[:-2] if t.endswith(".0") else t) + "%"


def plural(n, word, many=None):
    return f"{n} {word if n == 1 else (many or word + 's')}"


def pluralw(n, word, many=None):
    """plural() with the count spelled out (prose)."""
    return f"{numword(n)} {word if n == 1 else (many or word + 's')}"


def rounds_text(rs):
    """[2, 3, 4] -> 'rounds 2 to 4'; [1] -> 'round 1'; [1, 3, 4] -> 'rounds 1, 3 and 4'."""
    rs = sorted(rs)
    if not rs:
        return "no round"
    if len(rs) == 1:
        return f"round {rs[0]}"
    if len(rs) > 2 and rs == list(range(rs[0], rs[-1] + 1)):
        return f"rounds {rs[0]} to {rs[-1]}"
    return "rounds " + ", ".join(str(r) for r in rs[:-1]) + " and " + str(rs[-1])


def listing(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


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


# ------------------------------------------------------------------------------------------------ game theory of the table
def analyse(T):
    """Exact one-shot benchmarks of the table {cell: (row payoff, column payoff)} (cell = row's letter, then column's)."""
    P = {c: (Fraction(T[c][0]), Fraction(T[c][1])) for c in CELLS}
    br_row = {y: [x for x in "AB" if P[x + y][0] >= P[OTHER[x] + y][0]] for y in "AB"}    # row's best replies to column y
    br_col = {x: [y for y in "AB" if P[x + y][1] >= P[x + OTHER[y]][1]] for x in "AB"}    # column's best replies to row x
    pure = [x + y for x in "AB" for y in "AB" if x in br_row[y] and y in br_col[x]]

    def aim(br):
        if br == {"A": ["A"], "B": ["B"]}:
            return "match"
        if br == {"A": ["B"], "B": ["A"]}:
            return "mismatch"
        return None
    mixed = None
    d0 = P["AA"][0] - P["AB"][0] - P["BA"][0] + P["BB"][0]
    d1 = P["AA"][1] - P["AB"][1] - P["BA"][1] + P["BB"][1]
    if d0 and d1:
        q = (P["BB"][0] - P["AB"][0]) / d0           # column's probability of A that leaves the row player indifferent
        p = (P["BB"][1] - P["BA"][1]) / d1           # row's probability of A that leaves the column player indifferent
        if 0 < p < 1 and 0 < q < 1:
            mixed = {"p": (p, q), "v": (q * P["AA"][0] + (1 - q) * P["AB"][0], p * P["AA"][1] + (1 - p) * P["BA"][1])}
    sums = {c: P[c][0] + P[c][1] for c in CELLS}
    return {"P": P, "br_row": br_row, "br_col": br_col, "pure": pure, "aims": (aim(br_row), aim(br_col)), "mixed": mixed,
            "sums": sums, "const": sums["AA"] if len(set(sums.values())) == 1 else None}


# ------------------------------------------------------------------------------------------------ announcements in messages
# One or two adverbs may sit between the subject and the verb ("I will genuinely play A", "I am really going to play B"), any
# word in -ly except the hedges ("I'll probably play A" is not an announcement); "not" / "never" are not adverbs here, so
# "I will not play A" announces nothing.
_HEDGE = r"probably|likely|possibly|potentially|hopefully|presumably|tentatively|occasionally|randomly|usually|mostly|partly|unlikely"
_ADV = (r"(?:(?:again|also|still|now|then|just|definitely|indeed|simply|genuinely|really|truly|honestly|actually|certainly|"
        r"absolutely|surely|firmly|once\s+more|here)|(?!(?:" + _HEDGE + r")\b)[a-z]+ly)\s+")
_VERB = (r"(?:play|choose|pick|select|go\s+with|stick\s+(?:with|to)|stay\s+with|keep\s+(?:playing|choosing|picking)|"
         r"continue\s+(?:with|playing|choosing|picking)|continue\s+to\s+(?:play|choose|pick)|switch\s+to|return\s+to|start\s+with|"
         r"be\s+(?:playing|choosing|picking))")
_LETTER = r"\s+[\"'“‘]?([AB])[\"'”’]?(?![\w/])"
_INTENT = re.compile(
    # I will / I'll / I shall / I am (really) going to / I plan to ... play A
    r"\bI(?:['’]ll|\s+will|\s+shall|(?:['’]m|\s+am)\s+(?:" + _ADV + r"){0,2}going\s+to|(?:['’]m|\s+am)\s+planning\s+to|"
    r"\s+plan\s+to|\s+intend\s+to)\s+(?:" + _ADV + r"){0,2}" + _VERB + _LETTER
    # I'm playing A, I am sticking with B
    + r"|\bI(?:['’]m|\s+am)\s+(?:" + _ADV + r"){0,2}(?:playing|choosing|picking|going\s+with|sticking\s+(?:with|to)|staying\s+with)"
    + _LETTER
    # I am committed to (playing) A, I remain committed to B
    + r"|\bI(?:['’]m|\s+am|\s+remain)\s+(?:" + _ADV + r"){0,2}committed\s+to(?:\s+(?:playing|choosing|picking))?" + _LETTER
    # "... and will play A" with I as the subject ("I am committed to our agreement and will play A"; checked in announced())
    + r"|\b(?P<coord>and)\s+(?:" + _ADV + r"){0,2}(?:will|shall)\s+(?:" + _ADV + r"){0,2}" + _VERB + _LETTER)
_COND = re.compile(r"(?i)\b(?:if|unless|provided|as\s+long\s+as|as\s+soon\s+as|only\s+when|in\s+case|whenever|otherwise)\b")
# "while you ..." is a condition ("I'll keep choosing A while you play B") unless the statement is pinned to one round
# ("I will choose A in round 5 while you choose B" proposes the round's cell).
_WHILE = re.compile(r"(?i)\bwhile\s+you\b")
_ONE_ROUND = re.compile(r"(?i)\b(?:this\s+(?:round|time)|(?:in|for)\s+(?:round\s+\d+|the\s+(?:final|last)\s+round))\b")
_NOT_THIS = re.compile(r"(?i)^\s*(?:(?:or|and|/)\s*[\"'“‘]?[AB]\b|[^.;]{0,40}(?:%|probabilit|percent|of\s+the\s+time|"
                       r"sometimes|at\s+random|randomly|\bmix|next\s+round|later\s+round|future\s+round))")
_ROUND_N = re.compile(r"(?i)^[^.;]{0,30}?\b(?:in|for)\s+round\s+(\d+)\b")
_SUBJ = re.compile(r"\b(I|[Yy]ou|[Ww]e|[Tt]hey|[Hh]e|[Ss]he|[Ii]t)\b")
# a condition on the other player's letter in the same round, which the log can check ('as long as you play A')
_SIMPLE_COND = re.compile(r"^(?:as\s+long\s+as|if|provided(?:\s+that)?|while)\s+you\s+(?:keep\s+(?:playing|choosing|picking)|"
                          r"play|choose|pick)\s+[\"'“‘]?([AB])[\"'”’]?$")

# ", as" starts a new clause ("I'll play A, as I said") but ", as long as" / ", as soon as" is a condition of what precedes it.
_JOIN = re.compile(r",\s*(?:so|and|but|because|since|which|as(?!\s+(?:long|soon)\s+as)|while)\b", re.I)


def _cond_words(seg, m):
    """The words of a condition from its keyword to the end of its segment ('as long as you play A'), at most 60 characters."""
    s = " ".join(seg[m.start():].split()).strip(" ,")
    s = s[:1].lower() + s[1:]
    return s if len(s) <= 60 else s[:57].rsplit(" ", 1)[0] + "..."


def statements(text, rnd=None):
    """Every first-person statement of a letter in the message, in text order, as (letter, kind, condition): kind 'plain'
    (counts as an announcement), 'cond' (skipped because a condition governs it; condition: its words, e.g. 'as long as you
    play A') or 'other' (skipped: probabilities, mixing, later rounds, another round number). See announced()."""
    out = []
    for clause, end in re.findall(r"([^.;:!?\n—–]+)([.;:!?\n—–]|$)", text or ""):
        if end == "?":
            continue
        for m in _INTENT.finditer(clause):
            if m.group("coord"):                          # "... and will play A": the clause's subject must be I
                subj = _SUBJ.findall(clause[:m.start()])
                if not subj or subj[-1] != "I":
                    continue
            letter = next(x for x in m.groups() if x in ("A", "B"))
            pre = _JOIN.split(clause[:m.start()])[-1]
            post = _JOIN.split(clause[m.end():])[0]
            rn = _ROUND_N.match(clause[m.end():])
            if _NOT_THIS.match(clause[m.end():]) or (rnd is not None and rn and int(rn.group(1)) != rnd):
                out.append((letter, "other", None))
                continue
            c = _COND.search(pre) or _WHILE.search(pre)
            if c:
                out.append((letter, "cond", _cond_words(pre, c)))
                continue
            c = _COND.search(post)
            if c:
                out.append((letter, "cond", _cond_words(post, c)))
                continue
            w = _WHILE.search(post)
            if w and not _ONE_ROUND.search(post[:w.start()]):
                out.append((letter, "cond", _cond_words(post, w)))
                continue
            out.append((letter, "plain", None))
    return out


def announced(text, rnd=None):
    """The choice a message plainly announces for this round ('I'll play A', 'I will genuinely choose B', "I'm playing A",
    'I am committed to playing A', 'I will start with A'), read conservatively. The text is cut into clauses at . ; : ! ? and
    dashes; questions are skipped. A statement is skipped when a condition (if, unless, as long as, as soon as, a durable
    'while you ...') governs it: in its clause before it, or after it up to the next ', so' / ', and' / ', but' ... ("If you play
    A, I'll play B", "I'll play B if you play A" and "I will play B every round, as long as you play A" are skipped; "I'll play
    A, so if you play B we both gain" is not). Statements about probabilities, mixing, later rounds or another round number
    (rnd: this round's number) are skipped too. The message counts only when every remaining statement names the same
    letter; None when it announces nothing clear."""
    found = {x for x, kind, _ in statements(text, rnd) if kind == "plain"}
    return found.pop() if len(found) == 1 else None


def conditional(text, rnd=None):
    """The statements of a letter that announced() skipped because a condition governs them, as [(letter, condition words)]
    in text order (repeats dropped); [] when there are none. Shown on the page, never counted as announcements."""
    out = []
    for x, kind, cw in statements(text, rnd):
        if kind == "cond" and (x, cw) not in out:
            out.append((x, cw))
    return out


# Regression cases for announced(): (message, round, expected letter or None). Most are DeepSeek messages from the 2026-11-11
# recordings (game, round, seat noted), the rest pin down the conservative reading.
ANNOUNCE_CASES = [
    ("This round I am genuinely going to play A. I want to try for the higher payoff.", 5, "A"),            # sc1 R5 P0
    ("This round I will genuinely play A. Let's both play A and build some trust.", 3, "A"),                  # sc8 R3 P1
    ("I know I've said one thing and done another before, but this round I am genuinely going to play A.", 5, "A"),  # sc8 R5 P0
    ("I will really play B this round.", 2, "B"),
    ("I propose we both play A every round. I will start with A. If you deviate, I will switch to B permanently.", 1, "A"),  # sc9 R1 P0
    ("Let's both play A in round 7. I am committed to playing A. If we both do, we can then agree on a fair split.", 7, "A"),  # sc12 R7 P0
    ("I'm committed to B this round.", 2, "B"),                                                                # sc7 R2 P0
    ("I am committed to our agreement and will play A in round 3. Let's stick to the plan.", 3, "A"),        # sc12 R3 P1
    ("I will continue to play B this round.", 4, "B"),
    # conditional promises are skipped, with or without a comma before "as long as" (ghp4 R2 and R3 P0)
    ("I played B as promised. I will continue to play B every round, including the final round, as long as you play A.", 2, None),
    ("I confirm I will continue playing B every round, including the final round, as long as you play A.", 3, None),
    ("I will play B every round, as long as you play A.", 2, None),
    ("I will play B every round as long as you play A.", 2, None),
    ("Agreed. I will play A every round as long as you do.", 1, None),                                         # sc9 R1 P1
    ("I'll keep choosing A while you choose B; if you switch to A, I'll choose B.", 3, None),
    ("I will choose B while you choose A.", 2, None),
    ("To make it right, I will choose A in round 5 while you choose B, giving you 11 and me only 2.", 5, "A"),  # sc11 R5 P0
    ("If you play A, I'll play B.", 1, None),
    ("I'll play B if you play A.", 1, None),
    ("I'll play A, so if you play B we both gain.", 1, "A"),
    ("I'll play A, as I said.", 2, "A"),
    ("I will play A with probability 4/11 and B with probability 7/11 this round.", 2, None),                 # sc5 R2 P0
    ("I will probably play A.", 1, None),
    ("I will not play A this round.", 1, None),
    ("I will play A in round 6.", 5, None),
    ("I will play A in round 5.", 5, "A"),
    ("I will play A this round. I am committed to B.", 1, None),
    ("What will you play? I'll play B.", 1, "B"),
    ("I remain committed to A/A as long as you do.", 4, None),
    ("You said you would play A and will play B.", 2, None),
]


# Regression cases for conditional(): (message, round, expected [(letter, condition words)]).
CONDITIONAL_CASES = [
    ("I played B as promised. I will continue to play B every round, including the final round, as long as you play A. "
     "If you play B at any point, I will switch to A for all remaining rounds.", 2,
     [("B", "as long as you play A"), ("A", "if you play B at any point")]),                                 # ghp4 R2 P0
    ("I will play B every round as long as you play A.", 2, [("B", "as long as you play A")]),
    ("I'll keep choosing A while you choose B; if you switch to A, I'll choose B.", 3,
     [("A", "while you choose B"), ("B", "if you switch to A")]),
    ("If you play A, I'll play B.", 1, [("B", "if you play A")]),
    ("I will play A this round.", 1, []),
    ("I will play A with probability 4/11 and B with probability 7/11 this round.", 2, []),
    ("I will play A in round 6 if you play B.", 5, []),
]


def test_announced():
    bad = [(t, r, want, announced(t, r)) for t, r, want in ANNOUNCE_CASES if announced(t, r) != want]
    bad += [(t, r, want, conditional(t, r)) for t, r, want in CONDITIONAL_CASES if conditional(t, r) != want]
    if bad:
        raise SystemExit("announcement reader regression:\n  " + "\n  ".join(f"{t!r} (round {r}): expected {w}, got {g}" for t, r, w, g in bad))
    return len(ANNOUNCE_CASES) + len(CONDITIONAL_CASES)


# ------------------------------------------------------------------------------------------------ the log
def read_game(events, summary, cfg):
    rules = next((e for e in events if e.get("event") == "rules"), {})
    over = next((e for e in events if e.get("event") == "game_over"), {})
    var = dict(over.get("variables") or {})
    var.update({k: v for k, v in (summary.get("variables") or {}).items() if k not in var})
    cv = cfg.get("variables") or {}

    def par(k, default=None):
        for src in (rules, var, cv):
            if src.get(k) is not None:
                return src[k]
        return default
    T = {}
    for c in CELLS:
        v = par("payoff_" + c)
        if v is None:
            raise SystemExit(f"payoff_{c} is in neither the rules event, the final variables nor the config")
        T[c] = (v[0], v[1])
    rounds, talks, unknown = [], [], set()
    for e in events:
        ev = e.get("event")
        if ev == "round_result":
            r = {"round": int(e.get("round", len(rounds) + 1)), "c": [str(e["choice0"]).strip().upper(), str(e["choice1"]).strip().upper()],
                 "pay": [e["payoff0"], e["payoff1"]], "score": [e.get("score0"), e.get("score1")]}
            r["cell"] = r["c"][0] + r["c"][1]
            rounds.append(r)
        elif ev == "talk":
            seat = e.get("actor", e.get("seat"))
            if seat is None:
                print(f"warning: talk event without a speaker: {e}", file=sys.stderr)
                continue
            talks.append({"seat": int(seat), "text": str(e.get("text") or ""), "round": int(e.get("round") or len(rounds) + 1)})
        elif ev not in ("setup", "rules", "game_end", "game_over") and e.get("_visibility") != "hidden":
            unknown.add(ev)
    for ev in sorted(unknown, key=str):
        print(f"warning: public event {ev!r} is not shown on the page", file=sys.stderr)
    return {"T": T, "R": int(par("rounds", len(rounds)) or len(rounds)), "talk": int(par("talk_passes", 0) or 0), "rounds": rounds,
            "talks": talks, "game_end": next((e for e in events if e.get("event") == "game_end"), {}),
            "winners": over.get("winners", summary.get("winners"))}


def verify(G, ref, A, rewards):
    warn, acc, T = [], [0, 0], G["T"]
    for k, r in enumerate(G["rounds"]):
        if r["round"] != k + 1:
            warn.append(f"round_result {k + 1} is numbered {r['round']}")
        if r["cell"] not in T:
            warn.append(f"round {r['round']}: choices {r['c']} are not A/B")
            continue
        for s in (0, 1):
            if Fraction(T[r["cell"]][s]) != Fraction(r["pay"][s]):
                warn.append(f"round {r['round']} seat {s}: payoff {r['pay'][s]} logged, {T[r['cell']][s]} in the table for {r['cell']}")
            acc[s] += Fraction(r["pay"][s])
            if r["score"][s] is not None and Fraction(r["score"][s]) != acc[s]:
                warn.append(f"round {r['round']} seat {s}: score {r['score'][s]} logged, {acc[s]} summed from the payoffs")
    if len(G["rounds"]) != G["R"]:
        warn.append(f"{len(G['rounds'])} rounds logged, {G['R']} configured")
    for s in (0, 1):
        ge = G["game_end"].get(f"score{s}")
        if ge is not None and Fraction(ge) != acc[s]:
            warn.append(f"game_end score{s} {ge}, {acc[s]} summed from the payoffs")
        if rewards[s] is not None and abs(float(rewards[s]) - float(acc[s])) > 1e-9:
            warn.append(f"reward of seat {s}: {rewards[s]} in the summary, {acc[s]} summed from the payoffs")
    if any(tuple(Fraction(x) for x in T[c]) != tuple(Fraction(x) for x in ref["table"][c]) for c in CELLS):
        warn.append(f"the logged payoff table {T} is not the published one {ref['table']}; the human benchmark may not apply")
    if A["mixed"]:
        for s in (0, 1):
            if abs(float(A["mixed"]["p"][s]) - ref["nash"][s]) > 0.0005 + 1e-12:
                warn.append(f"mixed equilibrium of seat {s}: {float(A['mixed']['p'][s]):.4f} computed, {ref['nash'][s]} published")
    else:
        warn.append("the logged table has no completely mixed equilibrium")
    for t in G["talks"]:
        if t["round"] > len(G["rounds"]) + 1:
            warn.append(f"talk in round {t['round']} after the last logged round")
    for w in warn:
        print("warning:", w, file=sys.stderr)
    return warn


# ------------------------------------------------------------------------------------------------ model calls
def call_meta(c):
    """Phase and round of a call, from its prompt ('Current phase: X', the public state's "round"); None when not shown."""
    p = c.get("prompt") or ""
    m = re.search(r"Current phase:\s*([A-Za-z_]+)", p)
    i = p.find("Public state:")
    rm = re.search(r'"round":\s*(\d+)', p[i:]) if i >= 0 else None
    return (m.group(1).upper() if m else None), (int(rm.group(1)) if rm else None)


def _same_text(a, b):
    a, b = " ".join((a or "").split()), " ".join((b or "").split())
    return bool(a) and (a == b or (len(a) >= 20 and (a.startswith(b) or b.startswith(a)) and min(len(a), len(b)) >= 20))


def _matches(call, st):
    a = _norm_answer(call.get("answer"))
    if not isinstance(a, dict):
        return False
    p = a.get("payload") if isinstance(a.get("payload"), dict) else {}
    if st["phase"] == "DECIDE":
        ch = p.get("choice")
        return a.get("action_id") == "choose" and isinstance(ch, str) and ch.strip().upper() == st["choice"]
    txt = p.get("text", next((v for v in p.values() if isinstance(v, str)), None))
    return a.get("action_id") == "talk" and isinstance(txt, str) and (_same_text(txt, st["text"]) or txt.strip() == st["text"].strip())


def attach_calls(steps, calls):
    """Tie calls to steps (see the module docstring). Returns (steps with a call, steps whose call answered the logged move)."""
    if not calls:
        return 0, 0
    meta = [call_meta(c) for c in calls]
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
                if _matches(calls[j - 1], st):
                    break
        if not cand:
            continue
        hit = next((k for k in cand if _matches(calls[k], st)), None)
        use = hit if hit is not None else (cand[-1] if st["phase"] == "DECIDE" else cand[0])
        before = [k for k in cand if k < use]
        parts = [f"[Earlier attempt, answer: {calls[k]['answer'][:300]}]\n{calls[k].get('reasoning') or '(no reasoning returned)'}"
                 for k in before]
        c = calls[use]
        parts.append(("[Final attempt]\n" if before else "") + (c.get("reasoning") or ("(no reasoning returned)" if before else "")))
        if hit is None:
            what = f"choice {st['choice']}" if st["phase"] == "DECIDE" else "a different message"
            parts.append(f"[The game applied {what} instead of this answer: {c['answer'][:300]}]")
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        if c.get("prompt"):
            st["state"] = en(common.state_part(c["prompt"]))
            st["state_from_prompt"] = True
        a = _norm_answer(c.get("answer"))
        if hit is not None and isinstance(a, dict):
            st["act"] = en(json.dumps(a, ensure_ascii=False))
        for k in before + [use]:
            used[k] = True
        ptr = max(ptr, use + 1)
        tied += 1
        hits += hit is not None
    if not all(used):
        print(f"warning: {used.count(False)} of {len(calls)} calls matched no decision", file=sys.stderr)
    return tied, hits


# ------------------------------------------------------------------------------------------------ build
def load(bundle_dir):
    b = common.load_bundle(bundle_dir)
    meta = os.path.join(bundle_dir, "meta.json")
    b["meta"] = json.load(open(meta, encoding="utf-8")) if os.path.exists(meta) else {}
    return b


def recorded_date(b, bundle_dir):
    d = (b.get("run") or {}).get("recorded_at") or (b.get("meta") or {}).get("recorded_at")
    if d:
        return str(d)[:10]
    import datetime
    f = os.path.join(bundle_dir, "game_log.jsonl")
    return datetime.date.fromtimestamp(os.path.getmtime(f)).isoformat()


def build(gid, bundle_dir, calls=None):
    if gid not in REF:
        raise SystemExit(f"unknown game {gid!r}; expected one of {', '.join(REF)}")
    ref = REF[gid]
    test_announced()                                     # the announcement reader must pass its regression cases
    b = load(bundle_dir)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    calls = calls or []
    events, summary, run, cfg = b["events"], b.get("summary") or {}, b.get("run") or {}, b.get("config") or {}
    if run.get("game") and run["game"] != gid:
        print(f"warning: run.json is for {run['game']}, building {gid}", file=sys.stderr)
    G = read_game(events, summary, cfg)
    if not G["rounds"]:
        raise SystemExit("no round_result event in the game log")
    T = G["T"]
    A = analyse(T)
    P, mixed = A["P"], A["mixed"]
    raw_rw = summary.get("rewards") or (run.get("result") or {}).get("rewards") or {}
    rewards = [raw_rw.get(str(s), raw_rw.get(s)) for s in (0, 1)]
    warnings = verify(G, ref, A, rewards)

    rounds, talks, talk_on = G["rounds"], G["talks"], G["talk"] > 0 or bool(G["talks"])
    R = len(rounds)
    X = max([R] + [t["round"] for t in talks])           # a cut-off game may have messages of an unfinished round
    ch = [[r["c"][s] for r in rounds] for s in (0, 1)]
    pay = [[Fraction(r["pay"][s]) for r in rounds] for s in (0, 1)]
    score = [[], []]
    for s in (0, 1):
        acc = Fraction(0)
        for i, r in enumerate(rounds):
            acc += pay[s][i]
            score[s].append(Fraction(r["score"][s]) if r["score"][s] is not None else acc)
    final = [score[0][-1], score[1][-1]]
    cells = [r["cell"] for r in rounds]
    alt = [[P[OTHER[ch[0][i]] + ch[1][i]][0] for i in range(R)], [P[ch[0][i] + OTHER[ch[1][i]]][1] for i in range(R)]]
    a_cnt = [sum(1 for c in ch[s] if c == "A") for s in (0, 1)]
    share = [[round(100 * sum(1 for c in ch[s][:i + 1] if c == "A") / (i + 1), 4) for i in range(R)] for s in (0, 1)]
    aims = A["aims"]
    matched = [c[0] == c[1] for c in cells]
    n_match = sum(matched)
    # whose aim the round met: the player whose choice was a best reply to the other's (in these games exactly one)
    best = [[pay[0][i] >= alt[0][i] for i in range(R)], [pay[1][i] >= alt[1][i] for i in range(R)]]
    model_full = (b.get("meta") or {}).get("model") or run.get("model") or ""
    model = model_full.split("/")[-1] if model_full else "an unnamed model"
    date = recorded_date(b, bundle_dir)
    seed = run.get("seed", cfg.get("seed"))
    gname = f"Selten-Chmura game {ref['k']}" if ref["src"] == SC else f"GHP game {ref['k']}"
    src_short = "Selten and Chmura" if ref["src"] == SC else "Goeree, Holt and Palfrey"
    protocol = SC_PROTOCOL if ref["src"] == SC else GHP_PROTOCOL

    # ---- announcements: the choice each seat's message(s) in a round plainly announced (the last clear one before choosing)
    ann = [[None] * X, [None] * X]
    cond = [[None] * X, [None] * X]                      # conditional statements of a round without a plain announcement
    for t in talks:
        t["ann"] = announced(t["text"], t["round"])
        t["cond"] = [] if t["ann"] else conditional(t["text"], t["round"])
        if t["ann"]:
            ann[t["seat"]][t["round"] - 1] = t["ann"]
    for s in (0, 1):
        for i in range(X):
            cs = []
            for t in talks:
                if t["seat"] == s and t["round"] == i + 1:
                    cs += [q for q in t["cond"] if q not in cs]
            if cs and not ann[s][i]:
                cond[s][i] = cs
    said = [[i for i in range(R) if ann[s][i]] for s in (0, 1)]
    kept = [[i for i in said[s] if ann[s][i] == ch[s][i]] for s in (0, 1)]
    broke = [[i for i in said[s] if ann[s][i] != ch[s][i]] for s in (0, 1)]
    conds = [[i for i in range(R) if cond[s][i]] for s in (0, 1)]

    def cond_text(cs):
        return "; ".join(f"{x} {w}" for x, w in cs)

    def aim_word(s):
        return {"match": "matching", "mismatch": "mismatching"}.get(aims[s], "")

    # ---- steps: per round the messages (log order), then the row player's and the column player's choice
    steps, feed = [], []
    tbl = ", ".join(f"{c} [{g(T[c][0])}, {g(T[c][1])}]" for c in CELLS)
    feed.append({"x": 0, "step": 0, "who": "Rules",
                 "text": f"{G['R']} rounds. Payoffs (row / column) by cell, the row player's letter first: "
                         + ", ".join(f"{c} {g(T[c][0])} / {g(T[c][1])}" for c in CELLS) + ". "
                         + (f"Each player sends {pluralw(G['talk'], 'public message')} before every choice; messages are not binding."
                            if talk_on else "No messages between the players.")})
    for x in range(X):
        r = x + 1
        for t in [t for t in talks if t["round"] == r]:
            empty = not t["text"].strip()
            head = (f"Round {r} {DOT} " + ("empty message" if empty else "message")
                    + (f" {DOT} announces {t['ann']}" if t["ann"] else f" {DOT} conditional only, not counted" if t["cond"] else ""))
            steps.append({"seat": t["seat"], "x": x, "kind": "talk", "phase": "TALK", "round": r, "text": t["text"], "head": head,
                          "act": en(json.dumps({"action_id": "talk", "payload": {"text": t["text"]}}, ensure_ascii=False))})
            feed.append({"x": x, "step": len(steps) - 1, "who": SEATS[t["seat"]], "kind": "talk",
                         "text": ("(an empty message)" if empty else en(t["text"]))
                                 + (f"  [announces {t['ann']}]" if t["ann"] else
                                    en(f"  [conditional only, not counted: {cond_text(t['cond'])}]") if t["cond"] else "")})
        if x >= R:
            continue
        for s in (0, 1):
            c = ch[s][x]
            head = f"Round {r} {DOT} chooses {c} {DOT} +{g(pay[s][x])}"
            if ann[s][x] and ann[s][x] != c:
                head += f" {DOT} had announced {ann[s][x]}"
            steps.append({"seat": s, "x": x, "kind": "choose_" + c.lower(), "phase": "DECIDE", "round": r, "choice": c, "head": head,
                          "act": json.dumps({"action_id": "choose", "payload": {"choice": c}})})
        c = cells[x]
        whose = [k for k in (0, 1) if best[k][x]]
        feed.append({"x": x, "step": len(steps) - 1, "who": "Result",
                     "text": f"Row {c[0]}, column {c[1]}: cell {c}, " + ("match" if matched[x] else "mismatch")
                             + (f", the {ROLE[whose[0]]} player's aim" if len(whose) == 1 else "")
                             + f". Payoffs {g(pay[0][x])} / {g(pay[1][x])}; scores {g(score[0][x])} / {g(score[1][x])}."
                             + "".join(f" The {ROLE[k]} player had announced {ann[k][x]}." for k in (0, 1) if ann[k][x] and ann[k][x] != ch[k][x])})

    # ---- what each seat saw (rebuilt from the log; the saved prompt or the call's prompt where there is one)
    def observation(st):
        r, s = st["round"], st["seat"]
        past = rounds[:r - 1]
        sc = [g(past[-1]["score"][k] if past[-1]["score"][k] is not None else score[k][r - 2]) for k in (0, 1)] if past else ["0", "0"]
        lines = ["(Rebuilt from the game log; the exact prompt for this turn was not saved.)",
                 f"You are seat P{s}, the {ROLE[s]} player. Round {r} of {G['R']}.",
                 ("Current phase: DECIDE. Both players secretly choose A or B; the choices are revealed together." if st["phase"] == "DECIDE"
                  else f"Current phase: TALK. Each player sends {pluralw(G['talk'], 'public, non-binding message')} before the decision."),
                 f"Payoffs [P0, P1] by cell (P0's letter first): {tbl}.",
                 f"Scores so far: P0 {sc[0]}, P1 {sc[1]}."]
        if past:
            lines.append("Earlier rounds:")
            lines += [f"- Round {q['round']}: P0 {q['c'][0]}, P1 {q['c'][1]}; payoffs {g(q['pay'][0])} / {g(q['pay'][1])}." for q in past]
        else:
            lines.append("Earlier rounds: none.")
        if st["phase"] == "DECIDE":
            before = [t for t in talks if t["round"] <= r]
        else:                                            # messages sent before this one (talk is sequential: P0, then P1)
            before = [t for k, t in enumerate(talks) if k < st["_ti"]]
        if before:
            lines.append("Messages so far:")
            lines += [f"- Round {t['round']}, P{t['seat']}: {t['text']}" for t in before]
        if st["phase"] == "DECIDE" and s == 1:
            lines.append("P0's choice for this round is not shown until both have chosen.")
        return en("\n".join(lines))

    ti = 0
    for st in steps:                                     # index of each talk step's message, for "messages so far"
        if st["phase"] == "TALK":
            st["_ti"] = ti
            ti += 1
    prompt = run.get("prompt") or {}
    for st in steps:
        st["state"] = observation(st)
    first = next((st for st in steps if prompt.get("text") and prompt.get("seat") == st["seat"]
                  and str(prompt.get("phase", "")).upper() == st["phase"] and re.search(r'"round": %d\b' % st["round"], prompt["text"])), None)
    if first is not None:
        first["state"] = en(common.state_part(prompt["text"]))       # the saved prompt of the game's first decision
        first["state_from_prompt"] = True

    tied, hits = attach_calls(steps, calls)
    has_reason = any(st.get("thinking") for st in steps)
    from_prompt = sum(1 for st in steps if st.get("state_from_prompt"))
    for st in steps:
        st.setdefault("thinking", "")
        for k in ("phase", "round", "choice", "text", "state_from_prompt", "_ti"):
            st.pop(k, None)

    # ---- references
    nash = [mixed["p"][s] for s in (0, 1)] if mixed else [None, None]
    human = list(ref["human"])
    eqv = list(mixed["v"]) if mixed else [None, None]
    # whole-number payoffs, scores and references are shown without decimals; as soon as one value on the payoff and score charts
    # is fractional (the mixed-equilibrium averages and paces usually are) both charts hover with two decimals, the precision of
    # the subtitles and tiles, so a reference never hovers as a rounded whole number
    on_charts = pay[0] + pay[1] + score[0] + score[1] + ([eqv[s] * (i + 1) for s in (0, 1) for i in range(R)] if mixed else [])
    fmt_pay = "int" if all(Fraction(v).denominator == 1 for v in on_charts) else "price2"

    # ---- charts
    def aim_label(s):
        a = aims[s]
        if a == "match":
            return f"Match: the {ROLE[s]} player's aim"
        if a == "mismatch":
            return f"Mismatch: the {ROLE[s]} player's aim"
        return f"The {ROLE[s]} player's best reply"

    # The grid's hover box shows one line per row, "<row name>: <tip>", unwrapped in 12px monospace (about 7.2px a character);
    # at 40 characters it still fits the chart panel on a 360px-wide phone, so every tip is kept to that (optional parts are
    # dropped when they would not fit).
    fits = lambda name, tip: len(name) + 2 + len(tip) <= TIP_W
    choice_tip = [[f"{ch[s][i]} {'+' if pay[s][i] >= 0 else ''}{g(pay[s][i])} ({OTHER[ch[s][i]]}: {g(alt[s][i])})" for i in range(R)]
                  for s in (0, 1)]
    with_total = all(fits(f"{ROLE[s].capitalize()} chose" if talk_on else ROLE[s].capitalize(),
                          choice_tip[s][i] + f", total {g(score[s][i])}") for s in (0, 1) for i in range(R))   # all or none
    rows, gcells, gray = [], [], {"cond": 0, "none": 0}
    for s in (0, 1):
        if talk_on:
            name = f"{ROLE[s].capitalize()} said"
            rows.append(name)
            rc = []
            for i in range(X):
                msgs = [t["text"] for t in talks if t["seat"] == s and t["round"] == i + 1]
                if not msgs:
                    rc.append(None)
                    continue
                if ann[s][i]:                            # whether it was kept is the 'chose' row's letter right below
                    a = ann[s][i]
                    tip = f"announced {a}" + ((", kept" if a == ch[s][i] else ", broken") if i < R else "; log ends")
                    rc.append({"c": CH_COLOR[a], "t": a, "tip": tip})
                elif cond[s][i]:                         # gray: only conditional statements (the first one, in its own words)
                    x0, w0 = cond[s][i][0]
                    tip = en(f'{x0} "{w0}"')
                    if not fits(name, tip):
                        letters = list(dict.fromkeys(x for x, _ in cond[s][i]))
                        tip = "conditional " + " or ".join(letters)
                    rc.append({"c": "muted", "tip": tip})
                    gray["cond"] += 1
                else:                                    # gray, no label: the fill says it
                    rc.append({"c": "muted", "tip": "an empty message" if not msgs[-1].strip() else "no clear announcement"})
                    gray["none"] += 1
            gcells.append(rc)
        name = f"{ROLE[s].capitalize()} chose" if talk_on else ROLE[s].capitalize()
        rows.append(name)
        rc = []
        for i in range(X):
            if i >= R:
                rc.append(None)
                continue
            c = ch[s][i]
            rc.append({"c": CH_COLOR[c], "t": c, "tip": choice_tip[s][i] + (f", total {g(score[s][i])}" if with_total else "")})
        gcells.append(rc)
    rows.append("Outcome")
    rc = []
    for i in range(X):
        if i >= R:
            rc.append(None)
            continue
        whose = [k for k in (0, 1) if best[k][i]]
        col = PL_COLOR[whose[0]] if len(whose) == 1 else ("hold" if whose else "muted")
        what = f", {ROLE[whose[0]]}'s aim" if len(whose) == 1 else (", both best replies" if whose else "")
        # no letters drawn on these cells: the page writes cell text in white, too faint on the green and amber fills; the
        # cell is in the hover and is the two letters of the rows above
        rc.append({"c": col, "tip": f"{cells[i]} {g(pay[0][i])} / {g(pay[1][i])}" + what})
    gcells.append(rc)
    wide = [f"{n}: {c['tip']}" for n, row in zip(rows, gcells) for c in row if c and not fits(n, c["tip"])]
    if wide:
        print(f"warning: {len(wide)} grid hover lines exceed {TIP_W} characters (too wide on a phone), e.g. {wide[0]!r}", file=sys.stderr)
    legend = [["A", CH_COLOR["A"]], ["B", CH_COLOR["B"]], [aim_label(0), PL_COLOR[0]], [aim_label(1), PL_COLOR[1]]]
    if any(len([k for k in (0, 1) if best[k][i]]) == 2 for i in range(R)):
        legend.append(["Both best replies (an equilibrium cell)", "hold"])
    if talk_on and (gray["cond"] or gray["none"]):
        legend.append(["Conditional only, not counted" if not gray["none"] else
                       "No clear or only conditional announcement" if gray["cond"] else "No clear announcement", "muted"])
    sub1 = ("One column per round; both players choose at the same time and see the other's choice only afterwards. "
            + ("'Said' rows: the choice the player's message that round plainly announced, as in \"I'll play A\"; gray when it "
               "announced none, or only under a condition such as 'if' or 'as long as' (hover for the condition), which is not "
               "counted; questions and talk of probabilities do not count either. 'Chose' rows: the actual choice. "
               if talk_on else "")
            + "Outcome: colored by whose aim the round's cell met"
            + (f" (the row player gains by {aim_word(0)}, the column player by {aim_word(1)})" if aims[0] and aims[1] else "")
            + ". Hover a round for the cell (row player's letter first), the payoffs and, in brackets, what the other letter "
              "would have paid" + (", with each player's running total." if with_total else "."))
    charts = [{"title": "Choices, round by round", "sub": sub1, "height": 120,
               "grid": {"rows": rows, "cells": gcells, "rowHeight": 26 if talk_on else 30, "legend": legend}, "bars": []}]

    pad = lambda v: v + [None] * (X - len(v))
    num = lambda v: int(v) if Fraction(v).denominator == 1 else round(float(v), 4)
    lines2, same_eq = [], bool(mixed) and eqv[0] == eqv[1]
    if same_eq:                                          # one reference line when both expect the same
        lines2 = [{"name": "Equilibrium average (each)", "color": NASH_COLOR, "dash": True, "thin": True, "v": pad([num(eqv[0])] * R)}]
    elif mixed:
        lines2 = [{"name": f"{ROLE[s].capitalize()}: equilibrium average", "color": PL_COLOR[s], "dash": True, "thin": True,
                   "v": pad([num(eqv[s])] * R)} for s in (0, 1)]
    if A["const"] is not None:
        total_words = f"Every cell pays {g(A['const'])} in total (constant sum), so one player's gain is the other's loss."
    else:
        lo, hi = min(A["sums"].values()), max(A["sums"].values())
        total_words = (f"Cell totals differ (not constant sum): from {g(lo)} ({listing([c for c in CELLS if A['sums'][c] == lo])}) "
                       f"to {g(hi)} ({listing([c for c in CELLS if A['sums'][c] == hi])}).")
    sub2 = (f"Bars: each player's payoff in the round. {total_words}"
            + ((f" Dashed: the average a round when both mix as in the mixed equilibrium, {g(eqv[0])} for each player." if same_eq else
                f" Dashed: the average a round when both mix as in the mixed equilibrium, {g(eqv[0])} for the row player and "
                f"{g(eqv[1])} for the column player.") if mixed else "")
            + f" Actual averages over {plural(R, 'round')}: {g(final[0] / R)} and {g(final[1] / R)}.")
    charts.append({"title": "Payoff each round", "sub": sub2, "height": 210, "fmt": fmt_pay,
                   "bars": [{"name": SEATS[s], "color": PL_COLOR[s], "v": pad([num(v) for v in pay[s]])} for s in (0, 1)],
                   "lines": lines2})

    if score[0] == score[1]:                             # identical running totals: one line, or the row player's would be hidden
        lines3 = [{"name": "Both players (identical)", "color": "ink", "v": pad([num(v) for v in score[0]]), "dots": True, "end": True}]
    else:
        lines3 = [{"name": SEATS[s], "color": PL_COLOR[s], "v": pad([num(v) for v in score[s]]), "dots": True, "end": True} for s in (0, 1)]
    if same_eq:
        lines3.append({"name": "Equilibrium pace (each)", "color": NASH_COLOR, "dash": True, "thin": True,
                       "v": pad([num(eqv[0] * (i + 1)) for i in range(R)])})
    elif mixed:
        lines3 += [{"name": f"{ROLE[s].capitalize()}: equilibrium pace", "color": PL_COLOR[s], "dash": True, "thin": True,
                    "v": pad([num(eqv[s] * (i + 1)) for i in range(R)])} for s in (0, 1)]
    lead = 0 if final[0] > final[1] else 1
    sub3 = (f"Each player's running total (its reward). Final {g(final[0])} for the row player and {g(final[1])} for the column player"
            + (", a tie." if final[0] == final[1] else f": the {ROLE[lead]} player ends {g(abs(final[0] - final[1]))} ahead.")
            + (" The two totals are equal after every round, so one line shows both." if score[0] == score[1] else "")
            + ((f" Dashed: the expected pace in the mixed equilibrium, ending at {g(eqv[0] * R)} for each player." if same_eq else
                f" Dashed: the expected pace in the mixed equilibrium, ending at {g(eqv[0] * R)} and {g(eqv[1] * R)}.") if mixed else ""))
    charts.append({"title": "Cumulative score", "sub": sub3, "height": 220, "fmt": fmt_pay, "lines": lines3})

    # one chart for both players' share of A (two charts were flat lines whenever a player kept one letter); the references are
    # drawn in the player's color: thin dashed the mixed equilibrium, thicker dashed the human rate
    # identical shares: one line, or the row player's would be hidden under the column player's (as in the score chart)
    diff = [i for i in range(R) if share[0][i] != share[1][i]]
    if not diff:
        ls = [{"name": "Both players (identical)", "color": "ink", "v": pad(share[0]), "dots": True, "end": True}]
    else:
        ls = [{"name": SEATS[s], "color": PL_COLOR[s], "v": pad(share[s]), "dots": True, "end": True} for s in (0, 1)]
    if mixed:
        ls += [{"name": f"{ROLE[s].capitalize()}: mixed equilibrium", "color": PL_COLOR[s], "dash": True, "thin": True,
                "v": [round(float(nash[s]) * 100, 4)] * X} for s in (0, 1)]
    ls += [{"name": f"{ROLE[s].capitalize()}: humans", "color": PL_COLOR[s], "dash": True,
            "v": [round(human[s] * 100, 4)] * X} for s in (0, 1)]
    sub = (f"After each round, the share of rounds so far in which each player chose A; by the end the row player {a_cnt[0]} of {R} "
           f"({pct_short(Fraction(a_cnt[0], R))}) and the column player {a_cnt[1]} of {R} ({pct_short(Fraction(a_cnt[1], R))})."
           + (" The two shares are equal after every round, so one line shows both." if not diff else
              f" The two shares are equal after every round except {rounds_text([i + 1 for i in diff])}; elsewhere the column "
              "player's line covers the row player's." if R >= 4 and len(diff) <= R // 4 else "")
           + (f" Thin dashed, in each player's color: the mixed equilibrium, row {frac(nash[0])} = {pct(nash[0])} and column "
              f"{frac(nash[1])} = {pct(nash[1])}, each the rate that leaves the other player indifferent between A and B." if mixed else "")
           + f" Thicker dashed: humans in {ref['src']}, row {pct(human[0])} and column {pct(human[1])}, from {protocol}."
           + (" This recording had messages, so the human rates are a reference rather than a like-for-like comparison." if talk_on else ""))
    charts.append({"title": "Share of rounds choosing A", "sub": sub, "height": 230, "fmt": "pct0", "lines": ls})

    # ---- tiles
    counts = {c: cells.count(c) for c in CELLS}
    tops = [c for c in CELLS if counts[c] == max(counts.values())]    # several when tied: none is named the most frequent
    tiles = [{"k": "Chose A (row / column)", "v": f"{pct_short(Fraction(a_cnt[0], R))} / {pct_short(Fraction(a_cnt[1], R))}",
              "s": f"{a_cnt[0]} and {a_cnt[1]} of {R} rounds. "
                   + (f"Mixed equilibrium {pct(nash[0])} / {pct(nash[1])}; " if mixed else "")
                   + f"humans {pct(human[0])} / {pct(human[1])} ({src_short})."}]
    s2 = ("Row / column player: a tie." if final[0] == final[1] else
          f"Row / column player: the {ROLE[lead]} player wins by {g(abs(final[0] - final[1]))}.")
    if mixed:
        s2 += (f" Mixed-equilibrium play expects {g(eqv[0] * R)} each over {plural(R, 'round')}." if eqv[0] == eqv[1] else
               f" Mixed-equilibrium play expects {g(eqv[0] * R)} / {g(eqv[1] * R)} over {plural(R, 'round')}.")
    win = G["winners"] if isinstance(G["winners"], list) else None
    if win is not None and sorted(int(w) for w in win) != ([lead] if final[0] != final[1] else [0, 1]):
        print(f"warning: winners {win} logged, final scores {final}", file=sys.stderr)
    tiles.append({"k": "Final scores", "v": f"{g(final[0])} / {g(final[1])}", "s": s2})
    s3 = ", ".join(f"{c} {counts[c]}" for c in CELLS) + ". "
    if aims[0] and aims[1]:
        s3 += (f"Matched in {plural(n_match, 'round')} (the {ROLE[0 if aims[0] == 'match' else 1]} player's aim), "
               f"mismatched in {R - n_match} (the {ROLE[0 if aims[0] == 'mismatch' else 1]} player's).")
    n_top = counts[tops[0]]
    if len(tops) == 1:
        v3 = f"{tops[0]} in {n_top} of {R}"
    else:
        v3 = (f"All four, {n_top} each" if len(tops) == len(CELLS) else f"{' / '.join(tops)}, {n_top} each")
        s3 = ("No single cell came up most often: " if len(tops) == len(CELLS) else
              f"No single cell came up most often ({listing(tops)} tie): ") + s3
    tiles.append({"k": "Most frequent cell", "v": v3, "s": s3})
    if talk_on and (said[0] or said[1] or conds[0] or conds[1]):
        def kept_words(s):
            if said[0] and said[1] and not broke[0] and not broke[1]:
                return "both players kept every one" if s == 0 else ""
            if not said[s]:
                return f"the {ROLE[s]} player announced no clear choice"
            if not broke[s]:
                return f"the {ROLE[s]} player kept every one"
            return f"the {ROLE[s]} player broke its word in {rounds_text([i + 1 for i in broke[s]])}"
        def cond_words(s):
            """Rounds whose messages were only conditional: left out of the counts, named here; a simple condition on the other
            player's letter this round ('as long as you play A') is checked against that player's choice."""
            rs = conds[s]
            if not rs:
                return ""
            first = {cond[s][i][0] for i in rs}
            w = (f"Not counted: the {ROLE[s]} player's conditional "
                 + ("statement" if len(rs) == 1 else "statements") + f" in {rounds_text([i + 1 for i in rs])}")
            if len(first) == 1:
                x0, w0 = next(iter(first))
                w += en(f' ({x0} "{w0}")')
                m = [_SIMPLE_COND.match(w0) for i in rs]
                if all(m) and all(ch[1 - s][i] == q.group(1) for i, q in zip(rs, m)):
                    k = sum(1 for i in rs if ch[s][i] == x0)
                    w += (f"; the condition held and it chose {x0} " + ("each time" if len(rs) > 1 else "") if k == len(rs) else
                          f"; the condition held and it chose {x0} in {k} of {len(rs)}").rstrip()
            return w + "."
        kv = [f"{len(kept[s])} of {len(said[s])}" if said[s] else "none" for s in (0, 1)]
        tiles.append({"k": "Kept their announcement", "v": f"{kv[0]} / {kv[1]}",
                      "s": "Row / column player: rounds in which the choice matched what the player's message had plainly announced; "
                           + ", ".join(w for w in (kept_words(0), kept_words(1)) if w) + "."
                           + "".join(" " + cond_words(s) for s in (0, 1) if conds[s])})
    else:
        sw = [sum(1 for i in range(1, R) if ch[s][i] != ch[s][i - 1]) for s in (0, 1)]
        tiles.append({"k": "Choice changes (row / column)", "v": f"{sw[0]} / {sw[1]}",
                      "s": f"Rounds in which a player chose differently from its previous round, of {R - 1} possible"
                           + (f"; random play at the mixed equilibrium would change {pct(2 * nash[0] * (1 - nash[0]))} / "
                              f"{pct(2 * nash[1] * (1 - nash[1]))} of the time." if mixed else ".")})

    # ---- title, lede, foot
    k0 = R - 1
    while k0 > 0 and cells[k0 - 1] == cells[-1]:
        k0 -= 1                                          # first round (0-based) of the final run of one cell
    score_clause = ("the scores tie at " + g(final[0]) if final[0] == final[1] else
                    f"the {ROLE[lead]} player ends ahead {g(max(final))} to {g(min(final))}")
    # promise-breakers. Repeated breakers: at least four announcements, of which at least two and at least 3/8 broken (three of
    # eight); liars, the stronger case: at least half broken. Both repeated breakers -> a title about both; otherwise a liar gets
    # the title, and a repeated breaker who is no liar a clause in whichever title the play gets. More broken first.
    often = sorted((s for s in (0, 1) if len(said[s]) >= 4 and len(broke[s]) >= 2 and 8 * len(broke[s]) >= 3 * len(said[s])),
                   key=lambda s: (-len(broke[s]), s))
    liars = [s for s in often if 2 * len(broke[s]) >= len(said[s])]

    def broke_words(s):
        rs, letters = [i + 1 for i in broke[s]], {ann[s][i] for i in broke[s]}
        if len(letters) == 1:
            x = next(iter(letters))
            return f"the {ROLE[s]} player announced {x} in {rounds_text(rs)} but chose {OTHER[x]}" + (" each time" if len(rs) > 1 else "")
        return f"the {ROLE[s]} player broke its word in {rounds_text(rs)}"
    lone = [s for s in often if s not in liars] if len(often) == 1 else []
    if lone:                                             # "...; the row player announced A in rounds 1, 4 and 5 but chose B each
        s0 = lone[0]                                     # time, and ends ahead 73 to 71"
        tail = "; " + broke_words(s0) + (f", and ends ahead {g(max(final))} to {g(min(final))}" if final[0] != final[1] and lead == s0
                                         else f"; {score_clause}")
    else:
        tail = f"; {score_clause}"
    runs, i = [], 0                                      # maximal runs of one cell, (first, last) 0-based
    while i < R:
        j = i
        while j + 1 < R and cells[j + 1] == cells[i]:
            j += 1
        runs.append((i, j))
        i = j + 1
    hold = max(runs, key=lambda q: (q[1] - q[0], -q[0]))   # the longest run, the earliest of equals
    n_hold = hold[1] - hold[0] + 1

    def a_words(n):
        return "never chooses A" if n == 0 else ("chooses A in every round" if n == R else f"chooses A in {numword(n)} of {numword(R)} rounds")
    if len(set(cells)) == 1 and R > 1:
        c = cells[0]
        title = (f"The players repeat {c} in all {numword(R)} rounds ({g(T[c][0])} / {g(T[c][1])} each round): the row player always "
                 f"chooses {c[0]} and the column player always {c[1]}")
    elif len(often) == 2:
        a, b2 = often
        title = (f"Both players {'often' if len(liars) == 2 else 'repeatedly'} break their word: the {ROLE[a]} player "
                 f"{numword(len(broke[a]))} of its {numword(len(said[a]))} announcements, the {ROLE[b2]} player "
                 f"{numword(len(broke[b2]))} of {numword(len(said[b2]))}; {score_clause}")
    elif liars:
        liar, o = liars[0], 1 - liars[0]
        other = ""
        if said[o]:                                      # the other player's record too, so the title does not imply it kept its word
            other = (f"; the {ROLE[o]} player broke {numword(len(broke[o]))} of {numword(len(said[o]))}" if broke[o] else
                     f"; the {ROLE[o]} player kept " + (f"all {numword(len(said[o]))}" if len(said[o]) > 1 else "its one"))
        title = (f"The {ROLE[liar]} player announces a choice in {numword(len(said[liar]))} rounds and breaks it in "
                 f"{numword(len(broke[liar]))}{other}; {score_clause}")
    elif R - k0 >= 3 and k0 > 1 and cells.count(cells[-1]) >= R - 2:
        # the final run's cell also fills all but one or two of the rounds before it: not "shifting play"
        c = cells[-1]
        title = (f"The players land on {c} in {numword(cells.count(c))} of {numword(R)} rounds "
                 f"(all but {rounds_text([i + 1 for i in range(R) if cells[i] != c])}){tail}")
    elif n_hold >= 3 and 2 * n_hold >= R and hold[1] < R - 1:
        # one cell held for at least half the game, then a change: say who changed, and whether against its announcement
        c, e = cells[hold[1]], hold[1] + 1
        nxt = cells[e]
        movers = [s for s in (0, 1) if nxt[s] != c[s]]
        broken = [s for s in movers if ann[s][e] == c[s]]
        if len(movers) == 2:
            to = f"to {nxt[0]}" if nxt[0] == nxt[1] else f"to {nxt}"
            who = (f"both players break their word and switch {to}" if len(broken) == 2 else f"both players switch {to}")
        else:
            m0 = movers[0]
            who = (f"the {ROLE[m0]} player breaks its word and switches to {nxt[m0]}" if broken else
                   f"the {ROLE[m0]} player switches to {nxt[m0]}")
        span = f"for {numword(n_hold)} rounds" if hold[0] == 0 else f"for {numword(n_hold)} rounds from round {hold[0] + 1}"
        when = "in the last round" if e == R - 1 else f"in round {e + 1}"
        title = f"The players hold {c} {span} until {who} {when}; {score_clause}"
    elif R - k0 >= 3 and k0 > 1:
        title = f"After {pluralw(k0, 'round')} of shifting play the players settle on {cells[-1]} for the last {numword(R - k0)}{tail}"
    elif R - k0 >= 3 and k0 == 1:
        title = f"After {cells[0]} in round 1 the players settle on {cells[-1]} for the last {numword(R - k0)}{tail}"
    else:
        mid = [0 < a_cnt[k] < R for k in (0, 1)]
        title = (f"The row player chooses A in {numword(a_cnt[0])} of {numword(R)} rounds and the column player in "
                 f"{numword(a_cnt[1])}{tail}" if all(mid) else
                 f"The row player {a_words(a_cnt[0])} and the column player {a_words(a_cnt[1])}{tail}")
    title = title[:1].upper() + title[1:]

    game_words = f"game {ref['k']} of {ref['src']}"
    tbl_words = ", ".join(f"{c} {g(T[c][0])} / {g(T[c][1])}" for c in CELLS)
    if aims[0] and aims[1]:
        logic = (f"the row player gains by {aim_word(0)} the column player's letter and the column player by {aim_word(1)}, so no pair of "
                 "choices is stable" + (" and the only equilibrium is mixed" if not A["pure"] and mixed else ""))
    else:
        logic = ("pure equilibria: " + listing(A["pure"])) if A["pure"] else "there is no pure equilibrium"
    lede = (f"Two players repeat {game_words} for {numword(G['R'])} rounds: each round the row "
            f"player and the column player secretly choose A or B, and the choices are revealed together. Payoffs, row / column, by cell "
            f"(row's letter first): {tbl_words}; {logic}. "
            + (f"Before each choice each player sends {pluralw(G['talk'], 'public, non-binding message')}, and each scores the sum of its own payoffs."
               if talk_on else "They cannot talk, and each scores the sum of its own payoffs."))
    cfg_bits = ", ".join(x for x in [f"seed {seed}" if seed is not None else "", plural(G["R"], "round"),
                                     "no messages" if not talk_on else f"{plural(G['talk'], 'message')} per player per round"] if x)
    eq_text = ""
    if mixed:
        eq_text = (f"Mixed equilibrium, computed exactly from the logged table: the row player chooses A with probability {frac(nash[0])} "
                   f"({pct(nash[0])}), the column player with {frac(nash[1])} ({pct(nash[1])}); each makes the other indifferent, and the "
                   + (f"expected payoff is {g(eqv[0])} a round for each. " if eqv[0] == eqv[1] else
                      f"expected payoffs are {g(eqv[0])} and {g(eqv[1])} a round. "))
    if ref["src"] == SC and ref["k"] > 6:
        eq_text += f"It is the same equilibrium as in game {ref['k'] - 6}; only the cell totals differ. "
    risk = ""
    if ref["src"] == GHP:
        cA, cB = sorted({T["AA"][1], T["BA"][1]}), sorted({T["AB"][1], T["BB"][1]})
        risk = (f"For the column player A is the safe letter ({g(cA[0])} or {g(cA[-1])}) and B the risky one ({g(cB[0])} or {g(cB[-1])}), "
                "which makes play sensitive to risk attitudes. ")
    if has_reason:
        obs = ("Reasoning comes from the saved model calls"
               + ("" if tied == len(steps) else f" ({tied} of {len(steps)} decisions have a call)")
               + (f"; in {tied - hits} the game applied a different move than the model's answer, as noted in its reasoning" if tied > hits else "")
               + ". ")
    else:
        obs = "Reasoning was not saved in this recording, so the Reasoning tab is empty. "
    if from_prompt == len(steps):
        obs += "Each observation is the prompt the model saw. "
    elif from_prompt == 1 and not tied:
        obs += "The observation of the first decision is the saved prompt; the others are rebuilt from the log. "
    elif from_prompt:
        obs += f"{from_prompt} of {len(steps)} observations are the prompts the model saw; the others are rebuilt from the log. "
    else:
        obs += "The observations are rebuilt from the log. "
    foot = (f"Data: the {date} recording of {gid} ({cfg_bits}), both seats played by {model}"
            + (f" ({model_full})" if model_full and model_full != model else "") + "; every number on this page is computed from its game log "
            "(round_result events: both letters, payoffs and running scores; talk events: the messages)"
            + (", and every payoff and running score agrees with the payoff table. " if not warnings else
               ". Checks against the payoff table and the summary found: " + "; ".join(warnings[:4]) + (" (and more)" if len(warnings) > 4 else "") + ". ")
            + (f"The log stops after round {R} of {G['R']}. " if R < G["R"] else "") + obs
            + "The row player is seat P0 and the column player seat P1 in the log and the prompts; P0 is asked first, and neither sees "
            "the other's letter until both have chosen. "
            + f"Payoff table (row / column): {tbl_words}. " + eq_text + risk
            + f"Human benchmark: {ref['src']}, {protocol}: row player A {pct(human[0])}, column player A {pct(human[1])}"
            + (", so it is comparable only to recordings without messages." if talk_on else ".")
            + (" Announcements are read from the message text: a plain first-person statement of a letter for this round ('I will "
               "play A', 'I am committed to playing B') that no condition governs ('if', 'unless', 'as long as', a lasting 'while you "
               "...'); messages without one are left out of the kept/broken counts, and a message whose statements of a letter all "
               "carry such a condition is shown in gray with its condition and is not counted either." if talk_on else ""))
    kinds_all = {"choose_a": ["Chooses A", CH_COLOR["A"]], "choose_b": ["Chooses B", CH_COLOR["B"]], "talk": ["Public message", "muted"]}
    used = {st["kind"] for st in steps}
    d = {"id": gid, "name": gname, "title": title, "lede": lede, "seatNames": list(SEATS), "tiles": tiles, "foot": foot,
         "x": {"n": X, "labels": [f"R{i + 1}" for i in range(X)]}, "charts": charts,
         "kinds": {k: v for k, v in kinds_all.items() if k in used}, "steps": steps, "feed": feed,
         "feedTitle": "Messages and results" if talk_on else "Round by round",
         "feedSub": ("The payoff table, then each round's public messages in the order sent and the round's result, shown after the "
                     "round's second choice (the letters were secret until then). The selected step is highlighted; later ones are hidden."
                     if talk_on else
                     "The payoff table, then each round's result, shown after the round's second choice (the letters were secret until "
                     "then). The selected step is highlighted; later rounds are hidden."),
         "eyebrow": f"social_game_env {DOT} {gid} {DOT} {model}" + (", reasoning on" if has_reason else "") + f" {DOT} recorded {date}"}
    self_check(d, G, R, X, final, a_cnt)
    return d, date


def self_check(d, G, R, X, final, a_cnt):
    """Internal consistency of what is about to be written (independent recounts from the raw events)."""
    n_talk = len(G["talks"])
    assert len(d["steps"]) == n_talk + 2 * R, (len(d["steps"]), n_talk, R)
    xs = [s["x"] for s in d["steps"]]
    assert xs == sorted(xs), "steps out of round order"
    for s in (0, 1):
        assert sum(1 for st in d["steps"] if st["seat"] == s and st["kind"] == "choose_a") == a_cnt[s]
        assert sum(Fraction(r["pay"][s]) for r in G["rounds"]) == final[s]
    assert d["x"]["n"] == X


if __name__ == "__main__":
    if sys.argv[1:] == ["--test"]:
        print(f"announcement reader: {test_announced()} regression cases pass")
        raise SystemExit(0)
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    gid, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
    calls = common.load_calls(sys.argv[4]) if len(sys.argv) > 4 else []
    if len(sys.argv) > 4 and not calls:
        print(f"warning: no calls read from {sys.argv[4]}", file=sys.stderr)
    data, date = build(gid, src, calls)
    common.write(data, out, date=date)
