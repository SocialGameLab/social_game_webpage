"""Replay data for eco_signal_catdog (Cat & Dog cooperative signaling: Hayler et al., "Zero-Shot Coordination among LLM
Agents", ICLR 2026 workshop, after the toy game of Hu et al., "Off-Belief Learning", ICML 2021).

usage: python3 -I eco_signal_catdog.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md), a file inside
          one, or the rendered Markdown replay of one (one "- **event** ... `{json}`" line per logged event, model calls in
          "### " blocks after them).
  OUT_DIR where <OUT_DIR>/eco_signal_catdog.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: seat, prompt or messages, reasoning, answer; see common.load_calls). Without it the
          calls listed in a Markdown replay are used. Calls are queued per seat in time order and tied to that seat's decisions
          in order: a decision takes calls up to the first one whose answer is a legal action id of the seat's role (earlier
          ones were format retries; their reasoning is kept above the final one). A call whose prompt shows a later round is
          left for that round.

Game (rules/eco_signal_catdog_rules.md): each round Alice privately sees a cat or a dog and plays light_on, light_off,
remove_wall (-5) or quit (+1, Bob does not act); Bob sees the light (and the animal if the wall is gone) and plays
guess_cat, guess_dog (+10 right, -10 wrong) or bail (+0.5). Both seats receive the team score. A role with no seat is a
script (the _1p configs).

Axis: a one-round game (the default config) is drawn stage by stage (the animal is drawn, Alice acts, Bob acts, the round is
scored), so the information flow of the single round can be followed; a game of several rounds is drawn round by round.
Every value shown comes from the log: the animal from the private animal event, actions, light, wall, points and team score
from the public round_result events. Each round's points are re-derived from the payoff rules and the running team score is
re-added; any mismatch is printed as a warning (stderr).

Reasoning: each decision's feed line carries a few verbatim sentences of its reasoning, picked by keyword (what the light
means, the wall, bailing, the stated reason; see quotes_of), so that how each player decided is visible outside the raw
text. The page's Decision tab shows the last paragraph of the reasoning as its conclusion; when that paragraph only restates
the answer format ("Need output exactly JSON ..."), it is joined to the paragraph before it with a single line break
(join_format_tail; no word is changed), so the conclusion shown contains the model's decision."""
import json, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_signal_catdog"
DEFAULT_DATE = "2026-11-09"
MINUS, DOT, ARROW = "−", "·", "→"

# Payoffs (rules: Reward; the same numbers are in the game's prompt text). They are fixed by the game, not config values.
RIGHT, WRONG, BOB_BAIL, WALL_COST, ALICE_QUIT = 10, -10, 0.5, 5, 1
ALICE_ACTS = ("light_on", "light_off", "remove_wall", "quit")
BOB_ACTS = ("guess_cat", "guess_dog", "bail")

# Colors: the animal and Bob's guess share one color per animal (cat blue, dog orange), so a right guess is a blue-blue or
# orange-orange column; a lit bulb is yellow; doing nothing (light off, bail, quit) is gray; the removed wall and points won
# green; points lost orange. A color with two meanings on one page has both in its legend entry (legend_of).
# Action labels carry no player name: the timeline lane, the ticket and the feed already name the player.
ANIMAL_C = {"cat": "c1", "dog": "c2"}
KINDS = {"light_on": ["Light on", "c4"], "light_off": ["Light off", "hold"], "remove_wall": ["Remove the wall", "c3"],
         "quit": ["Quit", "hold"], "guess_cat": ["Guess cat", "c1"], "guess_dog": ["Guess dog", "c2"],
         "bail": ["Bail", "hold"]}
ALICE_CELL = {"light_on": ("Light on", "c4"), "light_off": ("Light off", "hold"), "remove_wall": ("Wall down", "c3"), "quit": ("Quit", "hold")}
BOB_CELL = {"guess_cat": ("Guess cat", "c1"), "guess_dog": ("Guess dog", "c2"), "bail": ("Bail", "hold")}
ALICE_VERB = {"light_on": "turns the light on", "light_off": "leaves the light off", "remove_wall": "removes the wall",
              "quit": "quits"}
ALICE_NOTE = {"light_on": "free; Bob will see a lit bulb but not the animal",
              "light_off": "free: she does nothing, so Bob will see the bulb off and not the animal",
              "remove_wall": "costs the team 5 points; Bob will see the animal",
              "quit": "the round ends at once with +1; Bob does not act"}
ALICE_PAST = {"light_on": "turned the light on (free); the wall stayed up", "light_off": "left the light off (free); the wall stayed up",
              "remove_wall": "removed the wall for 5 points", "quit": "quit for the sure +1"}
STAGES = ["Draw", "Alice acts", "Bob acts", "Score"]


def warn(msg):
    print(f"warning ({GID}): {msg}", file=sys.stderr)


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_ANY = re.compile("[　-ヿ㐀-鿿가-힯＀-￯]")
_CJK_RUN = re.compile("[぀-ヿ㐀-鿿가-힯]+")
_PUNCT = {"。": ". ", "、": ", ", "「": '"', "」": '"', "『": '"', "』": '"', "【": "[", "】": "]",
          "《": '"', "》": '"', "〈": "<", "〉": ">", "〜": "~"}


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
    return re.sub("[　-〿＀-￯]", "", s)


# ------------------------------------------------------------------------------------------------ loading
_TAG_HIDDEN, _TAG_PRIVATE = "隐藏", "私密"     # the Markdown replay's "hidden" and "private" tags
_SEED, _MODEL = "种子", "模型"                  # its header words for "seed" and "model"


def _md_bundle(path):
    """A rendered Markdown replay: events from the '- **event** (tag): `{json}`' lines (the tag marks hidden and private
    events), seat rewards, seed and model from the header, and the model calls listed after the events ('### ' blocks)."""
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
    model = re.search(_MODEL + r" ([^\s,，]+)", head)
    over = next((e for e in events if e.get("event") == "game_over"), {})
    setup = next((e for e in events if e.get("event") == "setup"), {})
    summary = {"rewards": json.loads(rw.group(1)) if rw else {}, "roles": setup.get("roles", {}), "variables": over.get("variables", {})}
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
    return {"events": events, "summary": summary, "run": run, "config": {}, "transcript": "", "meta": {}, "md_calls": calls,
            "md_path": path}


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


# ------------------------------------------------------------------------------------------------ small text helpers
def pts(v):
    """+10, −10, +0.5, 0."""
    v = float(v)
    if abs(v) < 1e-12:
        return "0"
    return ("+" if v > 0 else MINUS) + f"{abs(v):g}"


def num(v):
    v = float(v)
    return (MINUS if v < 0 else "") + f"{abs(v):g}"


def action_id(answer):
    """The action id of a model answer, or None when it is not a JSON action."""
    if not answer:
        return None
    s = answer.strip()
    m = re.search(r"\{.*\}", s, re.S)
    try:
        j = json.loads(m.group(0) if m else s)
    except (ValueError, AttributeError):
        return None
    return j.get("action_id") if isinstance(j, dict) else None


def plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


# ------------------------------------------------------------------------------------------------ reasoning: conclusion, quotes
_FORMAT = re.compile(r"\b(JSON|output|payload|format)\b", re.I)


def join_format_tail(t):
    """The page takes the last paragraph longer than 40 characters (paragraphs split at blank lines) as the conclusion of the
    reasoning. When that paragraph only restates the answer format (under 240 characters, mentions JSON, output, payload or
    format), it is joined to the previous long paragraph with a single line break. Only whitespace changes. Returns
    (text, joined)."""
    parts = re.split(r"(\n[ \t]*\n\s*)", t or "")          # paragraphs at even indices, separators at odd ones
    long_ = [i for i in range(0, len(parts), 2) if len(parts[i].strip()) > 40]
    if len(long_) < 2:
        return t, False
    last, prev = long_[-1], long_[-2]
    p = parts[last].strip()
    if len(p) >= 240 or not _FORMAT.search(p):
        return t, False
    for j in range(prev + 1, last, 2):
        parts[j] = "\n"
    return "".join(parts), True


def sentences(t):
    """The reasoning cut into sentences, in order; every line (and list item) starts a new sentence."""
    out = []
    for line in (t or "").split("\n"):
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
        for s in re.split(r"(?<=[.?!])\s+(?=[A-Z\"“'(\[])", line):
            if s.strip():
                out.append(s.strip())
    return out


_STRONG = re.compile(r"no unique|any mapping|arbitrar|multiple equilibri|which mapping|could mean (?:cat|dog) or|"
                     r"no (?:pre-?agreed|shared|common|agreed) (?:code|convention|mapping|agreement|protocol)", re.I)
_MEANING_WORDS = [r"no unique", r"any mapping", r"arbitrar", r"multiple equilibri", r"could mean", r"mapping", r"convention",
                  r"either"]
_FIVE = re.compile(r"(?<![\d.\-−])\+?5(?!\d|\.\d)")        # a 5 (the wall's points), not 0.5, -5, 15 or 5.5


def _theme_hits(ss):
    """Sentence index per theme: 'meaning' (the light's meaning is arbitrary: the sentence with the most such keywords),
    'wall' (the first sentence on the wall and its 5 points), 'bail' (the first that weighs bailing against a guess's
    chances), 'why' (the last 'because/since' sentence naming a reason such as order, a default or payoff dominance), 'memory'
    (the first 'I recall / I've seen')."""
    hits = {}
    best = (0, None)
    for i, s in enumerate(ss):
        if _STRONG.search(s):
            sc = sum(1 for w in _MEANING_WORDS if re.search(w, s, re.I))
            if sc > best[0]:
                best = (sc, i)
    if best[1] is not None:
        hits["meaning"] = best[1]
    for i, s in enumerate(ss):
        if "wall" not in hits and re.search(r"wall", s, re.I) and _FIVE.search(s):
            hits["wall"] = i
        if "bail" not in hits and re.search(r"\bbail", s, re.I) and re.search(r"\bp\s*[<>]|probabilit|50/50|expected|chance", s, re.I):
            hits["bail"] = i
        if (re.search(r"\b(because|since)\b", s, re.I) and not _FORMAT.search(s)
                and re.search(r"\b(first|natural\w*|default|payoff-dominant|focal|salient|alphabetic\w*)\b", s, re.I)):
            hits["why"] = i
    mem = [i for i, s in enumerate(ss) if re.search(r"\bI (?:recall|remember)\b", s)] or \
          [i for i, s in enumerate(ss) if re.search(r"\bI(?:'ve| have) seen\b", s)]
    if mem:
        hits["memory"] = mem[0]
    return hits


THEMES = {"ALICE": ["meaning", "wall", "why", "memory"], "BOB": ["meaning", "bail", "why", "memory"]}
LEADS = {"meaning": "On what the light means", "wall": "On removing the wall", "bail": "On bailing",
         "why": "Why {she} chose", "memory": "What {she} recalls"}


def quotes_of(text, role, cap=4):
    """Up to `cap` verbatim quotes from one decision's reasoning, in the order of the reasoning: [(theme, lead, quote)]. Each quote is
    a theme's sentence plus up to three short sentences right after it (each at most 80 characters, at most 260 in all),
    stopping at another quote's sentence or at one about the answer format."""
    ss = sentences(text)
    hits = _theme_hits(ss)
    order = [t for t in THEMES[role] if t in hits][:cap]
    anchors = {hits[t] for t in order}
    she = "she" if role == "ALICE" else "he"
    out = []
    for t in sorted(order, key=lambda t: hits[t]):
        i = hits[t]
        q = ss[i]
        for j in range(i + 1, min(len(ss), i + 4)):
            nxt = ss[j]
            if j in anchors or len(nxt) > 80 or _FORMAT.search(nxt) or len(q) + 1 + len(nxt) > 260:
                break
            q += " " + nxt
        out.append((t, LEADS[t].format(she=she), q))
    return out


def quote_block(qs, role):
    if not qs:
        return ""
    who = "her" if role == "ALICE" else "his"
    return f"\nFrom {who} reasoning, verbatim:\n" + "\n".join(f"• {lead}: “{q}”" for _, lead, q in qs)


def legend_of(entries):
    """[[label, color]] entries in first-seen order, one per color; a color with several meanings lists all of them. Gray
    entries ('nothing done') are collected into one."""
    by, nothing = {}, []
    for lab, col in entries:
        if col == "hold":
            if lab not in nothing:
                nothing.append(lab)
            by.setdefault("hold", [])
            continue
        by.setdefault(col, [])
        if lab not in by[col]:
            by[col].append(lab)
    out = []
    for col, labs in by.items():
        if col == "hold":
            out.append([f"Nothing done ({', '.join(nothing)})", "hold"])
        else:
            out.append([labs[0] + "".join(" / " + l[0].lower() + l[1:] for l in labs[1:]), col])
    return out


# ------------------------------------------------------------------------------------------------ the game record
def game_record(b):
    ev = b["events"]
    setup = next((e for e in ev if e.get("event") == "setup"), {})
    info = next((e for e in ev if e.get("event") == "setup_info"), {})
    roles = {int(k): v for k, v in (setup.get("roles") or (b.get("summary") or {}).get("roles") or {}).items()}
    if not roles:
        roles = {p["seat"]: p["role"] for p in (b.get("run") or {}).get("players", [])}
    alice = info.get("alice_seat", next((s for s, r in roles.items() if r == "ALICE"), None))
    bob = info.get("bob_seat", next((s for s, r in roles.items() if r == "BOB"), None))
    animals = {e["round"]: e["animal"] for e in ev if e.get("event") == "animal"}
    results = sorted((e for e in ev if e.get("event") == "round_result"), key=lambda e: e["round"])
    over = next((e for e in ev if e.get("event") == "game_over"), {})
    end = next((e for e in ev if e.get("event") == "game_end"), {})
    n_rounds = info.get("rounds") or end.get("rounds") or (over.get("variables") or {}).get("rounds") or len(results)
    rounds, prev = [], 0
    for e in results:
        r, a = e["round"], e.get("alice_action")
        animal = animals.get(r) or e.get("revealed_animal")
        bob_a = e.get("bob_action") if a != "quit" else None
        right = None if bob_a in (None, "bail") or animal is None else bob_a == "guess_" + animal
        if a == "quit":
            want = ALICE_QUIT
        elif bob_a == "bail":
            want = BOB_BAIL
        else:
            want = RIGHT if right else WRONG
        if a == "remove_wall":
            want -= WALL_COST
        if animal is not None and abs(want - float(e["reward"])) > 1e-9:
            warn(f"round {r}: logged points {e['reward']}, the rules give {want}")
        if abs(prev + float(e["reward"]) - float(e["team_score"])) > 1e-9:
            warn(f"round {r}: team score {e['team_score']} is not {prev} + {e['reward']}")
        rounds.append({"r": r, "animal": animal, "alice": a, "bob": bob_a, "right": right, "reward": float(e["reward"]),
                       "before": prev, "score": float(e["team_score"]), "light": e.get("light"), "wall": e.get("wall"),
                       "revealed": e.get("revealed_animal"), "event": e})
        prev = float(e["team_score"])
    final = rounds[-1]["score"] if rounds else 0.0
    rw = (b.get("summary") or {}).get("rewards") or {}
    for s, v in rw.items():
        if abs(float(v) - final) > 1e-9:
            warn(f"seat {s} reward {v} differs from the final team score {final}")
    n_players = (over.get("variables") or {}).get("n_players") or len(roles) or 1
    return {"roles": roles, "alice": alice, "bob": bob, "rounds": rounds, "n_rounds": int(n_rounds), "final": final,
            "cfd": bool(info.get("cfd")), "scripted": info.get("scripted_role"), "convention": info.get("script_convention"),
            "n_players": n_players, "unscored": sorted(set(animals) - {x["r"] for x in rounds})}


# ------------------------------------------------------------------------------------------------ decisions and calls
def decisions_of(g):
    out = []
    for x in g["rounds"]:
        if g["alice"] is not None:
            out.append({"seat": g["alice"], "role": "ALICE", "round": x, "action": x["alice"]})
        if g["bob"] is not None and x["bob"] is not None:
            out.append({"seat": g["bob"], "role": "BOB", "round": x, "action": x["bob"]})
    return out


def _prompt_round(p):
    m = re.search(r'Public state: \{[^\n]*?"round": (\d+)', p or "")
    return int(m.group(1)) if m else None


def attach_calls(decs, calls):
    """Tie each decision to its model calls (see the module docstring). Returns the number of calls left over."""
    queues = {}
    for c in calls:
        queues.setdefault(c.get("seat"), []).append(c)
    for d in decs:
        q = queues.get(d["seat"], [])
        legal = ALICE_ACTS if d["role"] == "ALICE" else BOB_ACTS
        took = []
        while q:
            pr = _prompt_round(q[0].get("prompt"))
            if pr is not None and pr > d["round"]["r"]:
                break
            c = q.pop(0)
            took.append(c)
            if action_id(c.get("answer")) in legal:
                break
        d["calls"] = took
    return sum(len(q) for q in queues.values())


def rebuilt_view(g, d):
    """The decision's observation in the engine's prompt layout, rebuilt from the log (the prompt was not saved)."""
    x, seat, r = d["round"], d["seat"], d["round"]["r"]
    alice = d["role"] == "ALICE"
    priv = {"animal": x["animal"]} if alice else {}
    if g["cfd"]:
        priv["cfd_instructions"] = "(the paper's coordination-friendly definition; its text is not in the log)"
    msgs = [{"event": "animal", "round": y["r"], "animal": y["animal"]} for y in g["rounds"] if y["r"] <= r] if alice else []
    if alice:
        state = {"rounds": g["n_rounds"], "round": r, "light": "off", "wall": "standing", "revealed_animal": None,
                 "team_score": x["before"], "n_players": g["n_players"]}
    else:
        state = {"rounds": g["n_rounds"], "round": r, "light": x["light"], "wall": x["wall"], "revealed_animal": x["revealed"],
                 "team_score": x["before"], "n_players": g["n_players"]}
    state = {k: (int(v) if isinstance(v, float) and v.is_integer() else v) for k, v in state.items()}
    seats = sorted(g["roles"])
    public = [dict([("event", "round_result")] + [(k, v) for k, v in y["event"].items() if k != "event" and not k.startswith("_")])
              for y in g["rounds"] if y["r"] < r][-12:]
    phase = "ALICE_TURN -- Alice sees the animal and takes one action." if alice else "BOB_TURN"
    lines = ["(Rebuilt from the game log; the exact prompt for this turn was not saved.)",
             f"You are seat P{seat}. Your role: {d['role']} (team PLAYERS).",
             f"Your private info: {json.dumps(priv)}",
             f"Private messages to you: {json.dumps(msgs)}",
             f"Current phase: {phase}",
             f"Public state: {json.dumps(state)}",
             f"Player count: {g['n_players']}",
             f"Alive seats: {json.dumps({str(s): True for s in seats})}",
             f"Public player state: {json.dumps({str(s): {} for s in seats})}",
             f"Recent public events: {json.dumps(public)}"]
    return "\n".join(lines)


def run_prompt(b):
    """The prompt the bundle saved for the game's first decision (run.json), with its seat and phase."""
    p = (b.get("run") or {}).get("prompt") or {}
    return (p.get("seat"), p.get("phase"), p.get("text")) if p.get("text") else (None, None, None)


# ------------------------------------------------------------------------------------------------ build
def build(src, calls=None):
    b = load_source(src)
    g = game_record(b)
    if not g["rounds"]:
        raise SystemExit(f"{GID}: no scored round in the log")
    calls = calls if calls else b.get("md_calls", [])
    decs = decisions_of(g)
    left = attach_calls(decs, calls)
    if left:
        warn(f"{left} model call(s) could not be tied to a decision")
    rp_seat, rp_phase, rp_text = run_prompt(b)

    rounds, R = g["rounds"], len(g["rounds"])
    stage = g["n_rounds"] == 1 and R == 1
    roles = g["roles"]
    # Lanes in play order (Alice, then Bob), whatever seat numbers the seed gave them; the foot names the seats.
    lane = sorted(roles, key=lambda s: (0 if roles[s] == "ALICE" else 1, s))
    seat_names = [("Alice" if roles[s] == "ALICE" else "Bob") for s in lane] or ["Alice"]
    idx = {s: k for k, s in enumerate(lane)}

    # ---- steps, one per decision
    steps, n_reason, n_saved, rebuilt_ok, n_joined, n_quoted = [], 0, 0, False, 0, 0
    for k, d in enumerate(decs):
        x = d["round"]
        cs = d.get("calls") or []
        final = cs[-1] if cs else None
        answer = (final or {}).get("answer") or ""
        if action_id(answer) != d["action"]:
            if answer:
                warn(f"round {x['r']} {d['role']}: the model answered {answer[:60]!r}, the log records {d['action']}")
            act = json.dumps({"action_id": d["action"], "payload": {}})
        else:
            act = en(answer)
        th = [c.get("reasoning") or "" for c in cs]
        if len(th) > 1:
            parts = [f"[Attempt {i + 1}, answer rejected: {(c.get('answer') or '')[:120]}]\n{t}" for i, (c, t) in enumerate(zip(cs[:-1], th[:-1])) if t]
            thinking = "\n\n".join(parts + ([f"[Final attempt]\n{th[-1]}"] if th[-1] else []))
        else:
            thinking = th[0] if th else ""
        thinking = en(thinking)
        n_reason += bool(thinking)
        d["quotes"] = quotes_of(en(th[-1] if th and th[-1] else thinking), d["role"])
        n_quoted += bool(d["quotes"])
        thinking, joined = join_format_tail(thinking)
        n_joined += joined
        prompt = (final or {}).get("prompt") or ""
        if prompt:
            state = common.state_part(en(prompt)); n_saved += 1
        elif k == 0 and rp_text and rp_seat == d["seat"] and rp_phase == ("ALICE_TURN" if d["role"] == "ALICE" else "BOB_TURN"):
            state = common.state_part(en(rp_text)); n_saved += 1
            mine = rebuilt_view(g, d).split("\n", 1)[1]
            rebuilt_ok = mine.strip() == state.strip()
            if not rebuilt_ok:
                warn("the rebuilt first observation differs from the saved prompt")
        else:
            state = rebuilt_view(g, d)
        a = x["animal"]
        rp = "" if stage else f"Round {x['r']} {DOT} "
        if d["role"] == "ALICE":
            head = f"{rp}Sees a {a} {DOT} {ALICE_VERB[d['action']]}"
            xi = 1 if stage else x["r"] - 1
        else:
            if x["wall"] == "removed":
                saw = f"Wall down, sees the {x['revealed'] or a}"
            else:
                saw = f"Light {x['light']}, wall up"
            verdict = "bails" if d["action"] == "bail" else f"guesses {d['action'][6:]} ({'right' if x['right'] else 'wrong'})"
            head = f"{rp}{saw} {DOT} {verdict}"
            xi = 2 if stage else x["r"] - 1
        if rp:
            head = rp + head[len(rp)].lower() + head[len(rp) + 1:]
        steps.append({"seat": idx.get(d["seat"], 0), "x": xi, "head": head, "kind": d["action"], "act": act, "state": state,
                      "thinking": thinking})

    used = {d["action"] for d in decs} | {x["alice"] for x in rounds} | {x["bob"] for x in rounds if x["bob"]}
    kinds = {k: v for k, v in KINDS.items() if k in used}

    out = build_stage(g, steps, decs) if stage else build_rounds(g, steps, decs)
    out.update({"id": GID, "name": "Cat & Dog", "seatNames": seat_names, "kinds": kinds, "steps": steps})

    # ---- foot: source, definitions
    run, meta = b.get("run") or {}, b.get("meta") or {}
    date = recorded_date(b)
    model = str(run.get("model") or meta.get("model") or "the model")
    seed = run.get("seed")
    run_id = run.get("source_run") or meta.get("source_run")
    if b.get("md_path"):
        src_txt = f"the {date} test recording of {GID}" + (" with reasoning" if n_reason else "") + ", rendered as a Markdown replay"
    else:
        src_txt = f"the {date} recording of {GID}"
    paren = ([f"run {run_id}"] if run_id and not b.get("md_path") else []) + ([f"seed {seed}"] if seed is not None else [])
    setting = [plural(g["n_rounds"], "round")]
    if g["scripted"]:
        conv = {"light_on_cat": "light on means cat", "light_on_dog": "light on means dog"}.get(g["convention"], str(g["convention"]))
        seated = "Alice" if g["scripted"] == "BOB" else "Bob"
        setting.append(f"{seated} played by {model}, {'Bob' if g['scripted'] == 'BOB' else 'Alice'} by a script with the convention "
                       f"“{conv}”, which the model is not told")
    else:
        setting.append(f"both roles played by {model}")
    setting.append("with the paper's coordination-friendly definition (CFD) in both prompts" if g["cfd"] else "no CFD")
    seats_txt = ", ".join(f"seat P{s} was {('Alice' if r == 'ALICE' else 'Bob')}" for s, r in sorted(roles.items()))
    main = g["n_rounds"] == 1 and not g["scripted"] and not g["cfd"] and len(roles) == 2
    foot = [f"Source: {src_txt}" + (f" ({', '.join(paren)})" if paren else "") + f": {', '.join(setting)}"
            + (" (the paper's main setting)" if main else "") + f". In this game {seats_txt}"
            + (" (the seed decides which)." if len(roles) > 1 else ".")]
    foot.append("Every value shown comes from its game log: the animal from Alice's private animal event, and the actions, light, wall, "
                "points and team score from the public round_result event of each round; each round's points were checked against the payoff rules.")
    if n_reason == 0:
        foot.append("This recording was made before reasoning was saved; the game will be re-recorded with reasoning.")
    elif n_reason < len(steps):
        foot.append(f"Reasoning was saved for {n_reason} of the {len(steps)} decisions.")
    else:
        foot.append("The full reasoning of every decision was saved (Reasoning tab).")
    if n_quoted:
        foot.append("The quotes in the transcript are verbatim sentences of that decision's reasoning, picked by keyword: what the "
                    "light means, the wall, bailing, the reason given for the choice.")
    if n_joined:
        foot.append("Where the reasoning ends with a paragraph that only restates the answer format, that paragraph follows the one "
                    "before it after a single line break instead of a blank line (no words changed), so that the Decision tab's "
                    "conclusion shows the model's decision.")
    if n_saved == 0:
        foot.append("The prompts were not saved, so each observation is rebuilt from the log in the engine's prompt layout, showing only what that player could see.")
    elif n_saved < len(steps):
        foot.append((f"The first observation is the prompt the bundle saved" if n_saved == 1 else f"{n_saved} observations are saved prompts")
                    + "; the others are rebuilt from the log in the engine's prompt layout"
                    + (" (the rebuild of the first one matches the saved prompt exactly)." if rebuilt_ok else "."))
    if g["unscored"]:
        foot.append(f"The log ends before round {g['unscored'][0]} was scored; that round is not shown.")
    foot.append("Game: Hayler et al., “Zero-Shot Coordination among LLM Agents” (ICLR 2026 workshop), after the cat-and-dog game of "
                "Hu et al., “Off-Belief Learning” (ICML 2021). In the paper, two models each playing with itself scored "
                f"{MINUS}0.75 and 4.50 per one-round game on average over 100 games without the CFD and 5.00 with it (Table 1)"
                + ("; one game that scores +10 shows that this pair's guesses at a convention matched once, not that they reliably do." if stage and rounds[0]["right"] and rounds[0]["alice"] in ("light_on", "light_off") else "."))
    out["foot"] = " ".join(foot)
    if n_reason == 0:
        out["eyebrow"] = f"social_game_env {DOT} {GID} {DOT} deepseek-flash, reasoning not saved {DOT} recorded {date}"
    return out


def lede(multi):
    each = "Each round " if multi else ""
    return (f"Alice and Bob are one team, separated by a wall. {each}Alice alone sees an animal, a cat or a dog, and can turn on a light on "
            "Bob's side (free), leave it off, remove the wall for 5 points so that Bob sees the animal, or quit for a sure +1; Bob then "
            f"guesses the animal (+10 if right, {MINUS}10 if wrong) or bails (+0.5), and both players receive the team score. Nothing fixes "
            "what the light means: “light on means cat” and “light on means dog” each pay +10 only if both players happen "
            "to pick the same one, so for partners who never agreed on anything, removing the wall (a sure +5) is the safe play.")


def round_points_text(x):
    a, b = x["alice"], x["bob"]
    if a == "quit":
        return f"Alice's quit +{ALICE_QUIT}"
    base = f"Bob's bail +{BOB_BAIL:g}" if b == "bail" else ("right guess +10" if x["right"] else f"wrong guess {MINUS}10")
    return base + (f", wall {MINUS}{WALL_COST}" if a == "remove_wall" else "")


def bob_sees(x):
    if x["alice"] == "quit":
        return None
    if x["wall"] == "removed":
        an = x["revealed"] or x["animal"]
        return (f"{an.capitalize()}, no wall", ANIMAL_C.get(an, "hold"), f"Bob sees the {an} (the wall is gone) and the light {x['light']}.")
    return (f"Bulb {x['light']}", "c4" if x["light"] == "on" else "hold",
            f"Bob sees the light {x['light']} and the wall standing; not the animal.")


# ------------------------------------------------------------------------------------------------ one round, stage by stage
def build_stage(g, steps, decs):
    x = g["rounds"][0]
    a, al, bo = x["animal"], x["alice"], x["bob"]
    A = a.capitalize()
    n = 4
    cells = [[None] * n for _ in range(5)]
    hidden = x["wall"] != "removed"
    # Bob's column: the animal is drawn faded and unlabeled while the wall hides it (or when he never acts).
    if al == "quit":
        bob_an = {"c": ANIMAL_C[a], "o": .3, "tip": f"Hidden from Bob ({A.lower()}): the round ended at Alice's quit and Bob does not act."}
    elif hidden:
        bob_an = {"c": ANIMAL_C[a], "o": .3, "tip": f"Hidden from Bob: the wall stands, so he sees only the light, not the {a}."}
    else:
        bob_an = {"c": ANIMAL_C[a], "t": A, "tip": f"Bob sees the {a}: the wall is gone."}
    cells[0] = [{"c": ANIMAL_C[a], "t": A, "tip": f"A {a} is drawn (cat or dog, 1/2 each). Only Alice sees it."},
                {"c": ANIMAL_C[a], "t": A, "tip": f"Alice sees the {a}; the wall hides it from Bob."},
                bob_an,
                {"c": ANIMAL_C[a], "t": A, "tip": f"The animal was a {a}. After the round both players see the actions and the points."}]
    t, c = ALICE_CELL[al]
    cells[1][1] = {"c": c, "t": t, "tip": f"Alice {ALICE_VERB[al]} ({ALICE_NOTE[al]})."}
    bs = bob_sees(x)
    if bs:
        cells[2][2] = {"c": bs[1], "t": bs[0], "tip": bs[2]}
    if bo:
        t, c = BOB_CELL[bo]
        tip = f"Bob bails (+{BOB_BAIL:g})." if bo == "bail" else f"Bob guesses {bo[6:]}: {'right' if x['right'] else 'wrong'} ({pts(RIGHT if x['right'] else WRONG)})."
        cells[3][2] = {"c": c, "t": t, "tip": tip}
    cells[4][3] = {"c": "c3" if x["reward"] > 0 else "c2" if x["reward"] < 0 else "hold", "t": pts(x["reward"]),
                   "tip": f"Round score {pts(x['reward'])} ({round_points_text(x)}). Team score {num(x['score'])}; both players receive it."}
    if al == "quit":
        cells[2][2] = {"c": "hold", "t": "No turn", "tip": "Alice quit, so Bob does not act."}
    # Legend: every meaning a color has in this grid, so that no cell is left unexplained (orange is the dog and also lost
    # points; green the removed wall and also points won).
    ent = [[f"{A} (the animal" + (", and Bob's guess)" if bo == "guess_" + a else ")"), ANIMAL_C[a]]]
    if bo and bo.startswith("guess_") and bo != "guess_" + a:
        ent.append([f"{bo[6:].capitalize()} (Bob's guess)", ANIMAL_C[bo[6:]]])
    if al == "light_on":
        ent.append(["Light on", "c4"])
    ent += [[lab, "hold"] for lab, on in (("light off", al == "light_off" or (bs and x["light"] == "off" and hidden)),
                                         ("bail", bo == "bail"), ("quit", al == "quit")) if on]
    if al == "remove_wall":
        ent.append(["Wall removed", "c3"])
    if x["reward"] > 0:
        ent.append(["Points won", "c3"])
    elif x["reward"] < 0:
        ent.append(["Points lost", "c2"])
    legend = legend_of(ent)
    grid = {"title": "The round, stage by stage",
            "sub": "Read left to right. Top row: the animal, which only Alice sees"
                   + ("; the faded cell is the animal hidden from Bob. " if hidden or al == "quit" else " until she removes the wall. ")
                   + ("Then what Alice does, what Bob can see when he acts, his guess and the points. " if al != "quit" else
                      "Then what Alice does and the points. ")
                   + ("Bob never sees the animal: he has only the light, and the meaning of the light was never agreed. " if hidden and al != "quit" else "")
                   + "Hover a cell for details; click a column to open that decision.",
            "height": 180, "bars": [],
            "grid": {"rows": ["Animal", "Alice", "Bob sees", "Bob", "Points"], "cells": cells, "legend": legend, "rowHeight": 32}}

    # ---- what each option was worth at each player's turn (bars), the option taken (dot), the team score (line)
    # Alice's turn: a light scores what Bob then does with it (right +10, wrong -10, bail +0.5); removing the wall +5 (Bob
    # then guesses what he sees); quitting +1. Bob's turn: his three options given the actual animal, each 5 lower after a
    # removed wall. The series order puts Bob's three bars in the middle of his column (the page keeps a slot per series).
    cost = WALL_COST if al == "remove_wall" else 0
    on_bob = (lambda v: v - cost) if bo else (lambda v: None)
    col = lambda va, vb: [None, va, vb, None]
    bars = [{"name": "Alice removes the wall", "color": "c1", "v": col(RIGHT - WALL_COST, None)},
            {"name": "Right guess", "color": "c3", "v": col(RIGHT, on_bob(RIGHT))},
            {"name": "Wrong guess", "color": "c2", "v": col(WRONG, on_bob(WRONG))},
            {"name": "Bob bails", "color": "hold", "v": col(BOB_BAIL, on_bob(BOB_BAIL))},
            {"name": "Alice quits", "color": "muted", "v": col(ALICE_QUIT, None)}]
    if al in ("light_on", "light_off"):
        took_a = BOB_BAIL if bo == "bail" else RIGHT if x["right"] else WRONG
    else:
        took_a = RIGHT - WALL_COST if al == "remove_wall" else ALICE_QUIT
    before = x["before"]
    lines = [{"name": "Team score", "color": "ink", "v": [before, before, before, x["score"]], "step": True},
             {"name": "Option taken", "color": "ink", "v": col(took_a, x["reward"] if bo else None), "noline": True, "dots": True}]
    allv = [v for s in bars + lines for v in s["v"] if v is not None]
    fmt = "int" if all(float(v).is_integer() for v in allv) else "price"
    line = {"title": "What each option was worth, at each player's turn",
            "sub": (f"Bars: the round's score for each option, given that the animal was a {a}. At Alice's turn a light scores "
                    f"+10 if Bob reads it right, {MINUS}10 if he reads it wrong and +0.5 if he bails; removing the wall scores +5 "
                    "(Bob then sees the animal) and quitting +1. "
                    + (f"At Bob's turn: a right guess, a wrong guess or bailing" + (", each 5 lower because the wall was removed. " if cost else ". ")
                       if bo else "Bob has no turn after Alice quits. ")
                    + "Dot: the option taken, at the height of its bar. Line: the team score, added when the round ends."),
            "height": 240, "fmt": fmt, "zero": True, "lines": lines, "bars": bars}

    # ---- tiles
    score_s = {True: "Bob guessed right without seeing the animal; +10 is the most one round can score.",
               False: "Bob guessed wrong: the light did not mean to him what it meant to Alice."}
    if al == "quit":
        s0 = "Alice quit at once: the sure +1."
    elif bo == "bail":
        s0 = "Bob bailed: the sure +0.5."
    elif al == "remove_wall":
        s0 = f"Bob saw the animal; the wall cost 5 of the 10 points." if x["right"] else "Bob saw the animal and still guessed wrong."
    else:
        s0 = score_s[bool(x["right"])]
    # One line from each player's reasoning where the keyword search found it (the quotes themselves are in the transcript).
    themes = {d["role"]: {q[0] for q in d.get("quotes") or []} for d in decs}
    a_note = (" Her reasoning priced the wall at a sure +5, then chose the light (quoted below)."
              if al in ("light_on", "light_off") and "wall" in themes.get("ALICE", ()) else "")
    tiles = [{"k": "Team score", "v": pts(x["score"]), "s": s0 + " Both players receive it."},
             {"k": "Alice's move", "v": ALICE_CELL[al][0], "s": f"She saw a {a} and {ALICE_PAST[al]}." + a_note}]
    if bo:
        if x["wall"] == "removed":
            seen = f"He saw the {x['revealed'] or a} through the removed wall and " + ("bailed." if bo == "bail" else f"guessed {bo[6:]}.")
        else:
            bulb = "lit" if x["light"] == "on" else "dark"
            seen = (f"He saw the light {x['light']} and the wall standing, not the animal, and "
                    + ("bailed." if bo == "bail" else f"read the {bulb} bulb as \u201c{bo[6:]}\u201d."))
        if bo != "bail" and "bail" in themes.get("BOB", ()):
            seen += " His reasoning weighed bailing's sure +0.5, then guessed (quoted below)."
        tiles.append({"k": "Bob's guess", "v": "Bail" if bo == "bail" else f"{bo[6:].capitalize()}, {'right' if x['right'] else 'wrong'}",
                      "s": seen})
    if al == "remove_wall":
        tiles.append({"k": "Cost of the wall", "v": pts(-WALL_COST),
                      "s": "Paid so that Bob could see the animal: the price of not relying on what the light might mean."})
    else:
        tiles.append({"k": "Sure alternative", "v": pts(RIGHT - WALL_COST),
                      "s": ("Removing the wall pays +5 with any Bob who guesses what he sees; had Bob read the light the other way, "
                            f"this round would have scored {pts(WRONG)}." if al in ("light_on", "light_off") and bo and bo != "bail" and x["right"]
                            else "Removing the wall pays +5 with any Bob who guesses what he sees, whatever the two think the light means.")})

    # ---- title
    if al in ("light_on", "light_off") and bo and bo != "bail":
        title = (f"Alice sees a {a} and {ALICE_VERB[al]}; Bob reads the light as \u201c{bo[6:]}\u201d: "
                 + ("+10 on a convention the two never agreed on" if x["right"] else f"{MINUS}10, the two read the light differently"))
    elif al == "remove_wall":
        title = f"Alice removes the wall so Bob sees the {a}: " + ("the safe +5" if x["right"] else
                                                                   f"but Bob bails, {pts(x['reward'])}" if bo == "bail" else
                                                                   f"yet Bob guesses wrong, {pts(x['reward'])}")
    elif al == "quit":
        title = f"Alice sees a {a} and quits at once: +1"
    else:
        title = f"Alice sees a {a} and {ALICE_VERB[al]}; Bob bails: +0.5"

    if g["scripted"] == "BOB":
        conv = {"light_on_cat": "light on means cat", "light_on_dog": "light on means dog"}.get(g["convention"], str(g["convention"]))
        title = f"Against a scripted Bob (\u201c{conv}\u201d): " + title
    elif g["cfd"]:
        title = "With the coordination-friendly definition: " + title

    # ---- transcript
    sidx = {(d["role"]): k for k, d in enumerate(decs)}
    dq = {d["role"]: d.get("quotes") or [] for d in decs}
    last = len(steps) - 1
    feed = [{"x": 0, "step": sidx.get("ALICE", 0), "who": "Chance", "text": f"The animal is a {a}. Only Alice sees it; the wall hides it from Bob."}]
    if True:                                     # Alice always acts (a seat or the script)
        e = {"x": 1, "who": "Alice" + ("" if g["alice"] is not None else " (script)"), "kind": al,
             "text": f"{ALICE_VERB[al].capitalize()} ({ALICE_NOTE[al]})." + quote_block(dq.get("ALICE"), "ALICE")}
        if "ALICE" in sidx:
            e["step"] = sidx["ALICE"]
        feed.append(e)
    if bo:
        e = {"x": 2, "who": "Bob" + ("" if g["bob"] is not None else " (script)"), "kind": bo,
             "text": (bs[2][4:].capitalize() if bs else "") + (" Bails." if bo == "bail" else f" Guesses {bo[6:]}.")
             + quote_block(dq.get("BOB"), "BOB")}
        if "BOB" in sidx:
            e["step"] = sidx["BOB"]
        feed.append(e)
    feed.append({"x": 3, "step": last, "who": "Result",
                 "text": (f"{'Right' if x['right'] else 'Wrong'}: the animal was a {a}. " if bo and bo != "bail" else "")
                 + f"Round score {pts(x['reward'])} ({round_points_text(x)}); team score {num(x['score'])}, the same for both players."})
    return {"title": title, "lede": lede(False), "tiles": tiles, "x": {"n": n, "labels": STAGES}, "charts": [grid, line],
            "feed": feed, "feedTitle": "The round, step by step",
            "feedSub": "The first line is hidden from Bob: only Alice sees the animal. Quoted lines are verbatim sentences from that "
                       "player's reasoning. The lines of the selected decision are highlighted."}


# ------------------------------------------------------------------------------------------------ several rounds
def build_rounds(g, steps, decs):
    rs = g["rounds"]
    n = len(rs)
    labels = [f"R{x['r']}" for x in rs]
    row_an, row_al, row_bob, row_pt = [], [], [], []
    for x in rs:
        a = x["animal"]
        row_an.append({"c": ANIMAL_C.get(a, "hold"), "t": (a or "?").capitalize(), "tip": f"A {a} (only Alice sees it)"})
        t, c = ALICE_CELL[x["alice"]]
        row_al.append({"c": c, "t": t, "tip": f"Alice {ALICE_VERB[x['alice']]} ({ALICE_NOTE[x['alice']]})"})
        if x["bob"]:
            t, c = BOB_CELL[x["bob"]]
            row_bob.append({"c": c, "t": t, "o": .9 if x["bob"] == "bail" or x["right"] else .45,
                            "tip": "Bob bails" if x["bob"] == "bail" else f"Bob guesses {x['bob'][6:]}: {'right' if x['right'] else 'wrong'}"})
        else:
            row_bob.append(None)
        row_pt.append({"c": "c3" if x["reward"] > 0 else "c2" if x["reward"] < 0 else "hold", "t": pts(x["reward"]),
                       "tip": f"{pts(x['reward'])} ({round_points_text(x)}); team score {num(x['score'])}"})
    used = lambda f: any(f(x) for x in rs)
    ent = [[an.capitalize(), ANIMAL_C[an]] for an in ("cat", "dog") if used(lambda x: an in (x["animal"], (x["bob"] or "")[6:]))]
    ent += [["Light on", "c4"]] if used(lambda x: x["alice"] == "light_on") else []
    ent += [[lab, "hold"] for lab, f in (("light off", lambda x: x["alice"] == "light_off"), ("bail", lambda x: x["bob"] == "bail"),
                                         ("quit", lambda x: x["alice"] == "quit")) if used(f)]
    ent += [["Wall removed", "c3"]] if used(lambda x: x["alice"] == "remove_wall") else []
    ent += [["Points won", "c3"]] if used(lambda x: x["reward"] > 0) else []
    ent += [["Points lost", "c2"]] if used(lambda x: x["reward"] < 0) else []
    grid = {"title": "Round by round: the animal, Alice's move, Bob's guess and the points",
            "sub": "Cat blue, dog orange: a guess in the animal's color is right (faded cells are wrong guesses). Yellow: the light on; gray: "
                   "nothing done (light off, bail or quit); green: the wall removed, or points won"
                   + ("; orange in the Points row: points lost" if used(lambda x: x["reward"] < 0) else "") + ". Hover a column for details.",
            "height": 170, "bars": [],
            "grid": {"rows": ["Animal", "Alice", "Bob", "Points"], "cells": [row_an, row_al, row_bob, row_pt],
                     "legend": legend_of(ent), "rowHeight": 30}}
    score = [x["score"] for x in rs]
    k = [i + 1 for i in range(n)]
    fmt = "int" if all(float(v).is_integer() for v in score + [x["reward"] for x in rs]) else "price"
    line = {"title": "Team score against two benchmarks",
            "sub": f"Solid: the running team score. Dashed: removing the wall every round (+{RIGHT - WALL_COST} a round with any Bob who "
                   "guesses what he sees) and a convention both players share from the first round (+10 a round).",
            "height": 240, "fmt": fmt, "zero": True, "bars": [],
            "lines": [{"name": "Shared convention", "color": "c3", "v": [RIGHT * i for i in k], "dash": True, "thin": True, "end": True},
                      {"name": "Wall every round", "color": "hold", "v": [(RIGHT - WALL_COST) * i for i in k], "dash": True, "thin": True, "end": True},
                      {"name": "Team score", "color": "c1", "v": score, "dots": True, "end": True, "area": True}]}
    # ---- counts behind the paper's p(Alice's action | animal) and Bob's accuracy
    on = {an: sum(1 for x in rs if x["animal"] == an and x["alice"] == "light_on") for an in ("cat", "dog")}
    tot = {an: sum(1 for x in rs if x["animal"] == an) for an in ("cat", "dog")}
    walls = sum(1 for x in rs if x["alice"] == "remove_wall")
    guesses = [x for x in rs if x["bob"] and x["bob"] != "bail"]
    right = sum(1 for x in guesses if x["right"])
    tiles = [{"k": "Team score", "v": pts(g["final"]), "s": f"After {plural(n, 'round')}; a shared convention from round 1 would give {RIGHT * n}, "
              f"removing the wall every round {(RIGHT - WALL_COST) * n}.", "spark": [0.0] + score},
             {"k": "Light on", "v": f"{on['cat']}/{tot['cat']} cats {DOT} {on['dog']}/{tot['dog']} dogs",
              "s": "Rounds in which Alice turned the light on, by animal: the light carries information only if these differ."},
             {"k": "Bob's guesses right", "v": f"{right} of {len(guesses)}", "s": f"{plural(sum(1 for x in rs if x['bob'] == 'bail'), 'bail')}; "
              f"{plural(sum(1 for x in rs if x['alice'] == 'quit'), 'round')} ended by Alice's quit."},
             {"k": "Wall removed", "v": f"{walls} of {n}", "s": "Rounds in which Alice paid 5 so that Bob saw the animal."}]
    title = f"{plural(n, 'round')} of Cat & Dog: team score {num(g['final'])} (+{RIGHT * n} with a shared convention, +{(RIGHT - WALL_COST) * n} by always removing the wall)"
    sidx, dq = {}, {}
    for k_, d in enumerate(decs):
        sidx[(d["round"]["r"], d["role"])] = k_
        dq[(d["round"]["r"], d["role"])] = (d.get("quotes") or [])[:2]      # two per decision keep a long game's feed short
    feed = []
    for i, x in enumerate(rs):
        last = max([v for (r, _), v in sidx.items() if r == x["r"]], default=None)
        e = {"x": i, "who": "Alice" + ("" if g["alice"] is not None else " (script)"), "kind": x["alice"],
             "text": f"Sees a {x['animal']} and {ALICE_VERB[x['alice']]}." + quote_block(dq.get((x["r"], "ALICE")), "ALICE")}
        if (x["r"], "ALICE") in sidx:
            e["step"] = sidx[(x["r"], "ALICE")]
        feed.append(e)
        if x["bob"]:
            bs = bob_sees(x)
            e = {"x": i, "who": "Bob" + ("" if g["bob"] is not None else " (script)"), "kind": x["bob"],
                 "text": (bs[2][4:].capitalize() if bs else "") + (" Bails." if x["bob"] == "bail" else f" Guesses {x['bob'][6:]} ({'right' if x['right'] else 'wrong'}).")
                 + quote_block(dq.get((x["r"], "BOB")), "BOB")}
            if (x["r"], "BOB") in sidx:
                e["step"] = sidx[(x["r"], "BOB")]
            feed.append(e)
        e = {"x": i, "who": "Result", "text": f"Round {x['r']}: {pts(x['reward'])} ({round_points_text(x)}); team score {num(x['score'])}."}
        if last is not None:
            e["step"] = last
        feed.append(e)
    return {"title": title, "lede": lede(True), "tiles": tiles, "x": {"n": n, "labels": labels}, "charts": [grid, line],
            "feed": feed, "feedTitle": "Round by round",
            "feedSub": "Every move with what the other player could see; the animal itself is shown to Bob only when the wall is removed. "
                       "Quoted lines are verbatim sentences from that player's reasoning."}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    common.write(build(src, calls), out, date=recorded_date(src))
