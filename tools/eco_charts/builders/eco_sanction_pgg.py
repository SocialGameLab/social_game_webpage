"""Replay data for eco_sanction_pgg (public goods game with a choice between a sanctioning and a sanction-free institution;
Guzman Piedrahita et al. 2025, after Gurerk, Irlenbusch and Rockenbach 2006).

usage: python3 -I eco_sanction_pgg.py SOURCE OUT_DIR [CALLS.jsonl]
  SOURCE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md), a file inside
          one, or the rendered Markdown replay of one (one "- **event** ... `{json}`" line per logged event).
  OUT_DIR where <OUT_DIR>/eco_sanction_pgg.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer; see common.load_calls). Calls are tied to
          decisions in time order: a call belongs to the next decision of the same seat (and, when its prompt shows them, of the
          same phase and round); the call whose answer is the logged move is that decision's call (earlier ones were format
          retries; their reasoning is kept above it). Its reasoning becomes the step's thinking and its prompt the observation.

Every round has up to three decisions per seat, asked phase by phase in seat order (rules: Round Procedure): the institution
choice, the contribution, and, for members of an SI with at least two members, the sanction lists. The log has no separate
event for the first two; they are read from the round's result record (round_result, one entry per seat). The sanction lists
are the hidden 'sanction' events. Every number shown is taken from the log; the payoffs are re-derived from the rules
(equations 1 to 3) and any disagreement is printed as a warning."""
import json, math, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_sanction_pgg"
MINUS, TIMES, DIV, ARROW = "−", "×", "÷", "→"
PHASES = ("INSTITUTION", "CONTRIBUTE", "SANCTION")


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_RUN = re.compile(r"[぀-ヿ㐀-鿿가-힯]+")
_PUNCT = {"。": ". ", "、": ", ", "「": '"', "」": '"', "『": '"', "』": '"', "【": "[", "】": "]",
          "《": '"', "》": '"', "〈": "<", "〉": ">", "〜": "~"}


def en(s):
    """The page shows English only: full-width forms become ASCII, CJK punctuation its ASCII counterpart, and any remaining run
    of Chinese, Japanese or Korean characters (the model sometimes reasons in Chinese) a visible marker."""
    if not isinstance(s, str) or not re.search(r"[　-ヿ㐀-鿿가-힯＀-￯]", s):
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
    return re.sub(r"[　-〿＀-￯]", "", s)


# ------------------------------------------------------------------------------------------------ loading
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
        if "隐藏" in m.group(2):            # the "hidden" tag
            body["_visibility"] = "hidden"
        elif "私密" in m.group(2):          # the "private" tag
            body["_visibility"] = "private"
        events.append(body)
    head = text.split("\n## ")[0].split("\n- **")[0]
    rw = re.search(r'(\{"0":[^}]*\})', head)
    seed = re.search(r"种子 (\d+)", head)    # "seed N"
    model = re.search(r"模型 ([^\s,，]+)", head)   # "model X"
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


def recorded_date(src):
    """The recording date (YYYY-MM-DD) from meta.json or run.json; None when neither has one (the rendered Markdown replays)."""
    b = load_source(src)
    d = (b.get("meta") or {}).get("recorded_at") or (b.get("run") or {}).get("recorded_at")
    return str(d)[:10] if d else None


# ------------------------------------------------------------------------------------------------ text helpers
NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
       "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def numword(n):
    return NUM[n] if 0 <= n < len(NUM) else str(n)


def pname(seat):
    return f"Player {seat + 1}"


def plist(seats, n=None):
    """[4, 5] -> 'Players 5 and 6'; [4] -> 'Player 5'; all n seats -> 'all seven players'."""
    seats = sorted(seats)
    if n is not None and len(seats) == n and n > 2:
        return f"all {numword(n)} players"
    names = [str(s + 1) for s in seats]
    if not names:
        return "no player"
    if len(names) == 1:
        return f"Player {names[0]}"
    return "Players " + ", ".join(names[:-1]) + " and " + names[-1]


def alist(agents):
    """[0, 2] -> 'agents 0 and 2'; [3] -> 'agent 3'."""
    agents = sorted(agents)
    if len(agents) == 1:
        return f"agent {agents[0]}"
    return "agents " + ", ".join(str(a) for a in agents[:-1]) + " and " + str(agents[-1])


def apairs(x, seats):
    """Seats of one round -> 'agent 0 (Player 5) and agent 2 (Player 6)', ordered by agent number."""
    items = [f"agent {x['agent_of'][s]} ({pname(s)})" for s in sorted(seats, key=lambda t: x["agent_of"][t])]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def target(x, k):
    """Agent number k of one round -> 'Player 3 (agent 0)' (the page's rows are players; agent numbers are redrawn every round)."""
    return f"{pname(x['seat_of'][k])} (agent {k})" if k in x["seat_of"] else f"agent {k}"


def targets(x, lst):
    """A sanction list (tokens per agent number) -> 'Player 3 (agent 0) with 3 tokens and Player 5 (agent 4) with 1 token'."""
    items = [f"{target(x, k)} with {tokens(v)}" for k, v in enumerate(lst) if v]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def f2(v):
    """Two decimals with a thousands separator and a true minus sign."""
    s = f"{abs(v):,.2f}"
    return (MINUS + s) if v < 0 and s.strip("0.,") else s


def tokens(n):
    return f"{n:g} token" + ("" if n == 1 else "s")


def grouped(vals, fmt, prep="for"):
    """{seat: value} -> '0 from Players 5 and 6, 20 from the other five' (smaller groups first); one group: '52.00 each'."""
    by = {}
    for s, v in vals.items():
        by.setdefault(v, []).append(s)
    if len(by) == 1:
        v, ss = next(iter(by.items()))
        return f"{fmt(v)} each" if len(ss) > 1 else f"{fmt(v)} {prep} {pname(ss[0])}"
    groups = sorted(by.items(), key=lambda t: (len(t[1]), -t[0] if isinstance(t[0], (int, float)) else 0))
    parts = [f"{fmt(v)} {prep} {plist(ss)}" for v, ss in groups[:-1]]
    v, ss = groups[-1]
    parts.append(f"{fmt(v)} {prep} " + (f"the other {numword(len(ss))}" if len(ss) > 1 else pname(ss[0])))
    return ", ".join(parts)


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


# ------------------------------------------------------------------------------------------------ the log
def read_game(events, summary, cfg):
    rules = next((e for e in events if e.get("event") == "rules"), {})
    var = dict((next((e for e in events if e.get("event") == "game_over"), {}) or {}).get("variables") or {})
    var.update({k: v for k, v in (summary.get("variables") or {}).items() if k not in var})
    cv = cfg.get("variables") or {}

    def par(k, default):
        for src in (rules, var, cv):
            if src.get(k) is not None:
                return src[k]
        return default
    P = {"rounds": int(par("rounds", 15)), "n": int(par("n_players", cfg.get("n_players", 7))), "E": int(par("endowment", 20)),
         "S": int(par("sanction_endowment", 20)), "alpha": float(par("multiplier", 1.6)), "pc": float(par("punish_cost", 1)),
         "pe": float(par("punish_effect", 3)), "rc": float(par("reward_cost", 1)), "re": float(par("reward_effect", 1)),
         "init": float(par("initial_tokens", 1000)), "H": int(par("history_rounds", 5)), "cap": str(par("sanction_cap", "tokens"))}
    rounds = {}

    def rd(r):
        return rounds.setdefault(int(r), {"round": int(r), "si_list": None, "sanctions": [], "my": {}, "result": None})
    for e in events:
        ev = e.get("event")
        if ev == "si_contributions":
            rd(e["round"])["si_list"] = [(int(a["agent"]), int(a["contribution"])) for a in e.get("contributions") or []]
        elif ev == "sanction":
            rd(e["round"])["sanctions"].append({"seat": int(e["seat"]), "agent": int(e["agent"]),
                                                "punish": [int(x) for x in e["punish"]], "reward": [int(x) for x in e["reward"]]})
        elif ev == "my_round":
            rd(e["round"])["my"][int(e["seat"])] = e
        elif ev == "round_result":
            rd(e["round"])["result"] = e
    out = []
    for r in sorted(rounds):
        x = rounds[r]
        recs = {}
        if x["result"]:
            for a in x["result"].get("agents") or []:
                recs[int(a["seat"])] = dict(a)
        for s, m in x["my"].items():                 # a missing round_result entry falls back to the seat's own record
            if s not in recs:
                recs[s] = dict(m, agent=m.get("agent_number"))
        if len(recs) < P["n"]:
            continue                                 # an unfinished round
        x["seats"] = recs
        x["agent_of"] = {s: int(a["agent"]) for s, a in recs.items()}
        x["seat_of"] = {v: s for s, v in x["agent_of"].items()}
        x["si"] = [s for s in range(P["n"]) if recs[s]["institution"] == "SI"]
        x["sfi"] = [s for s in range(P["n"]) if recs[s]["institution"] != "SI"]
        out.append(x)
    return P, out


def verify(P, rounds, earnings, rewards):
    """Recompute every payoff from the rules (eq. 1 to 3) and compare with the log; returns a list of disagreements."""
    warn, n = [], P["n"]
    cum = {s: P["init"] for s in range(n)}
    for x in rounds:
        r, recs = x["round"], x["seats"]
        recv_p, recv_r = {s: 0 for s in range(n)}, {s: 0 for s in range(n)}
        for sa in x["sanctions"]:
            for k, (p, w) in enumerate(zip(sa["punish"], sa["reward"])):
                if (p or w) and k in x["seat_of"]:
                    recv_p[x["seat_of"][k]] += p
                    recv_r[x["seat_of"][k]] += w
        for inst, mem in (("SI", x["si"]), ("SFI", x["sfi"])):
            if not mem:
                continue
            T = sum(recs[s]["contribution"] for s in mem)
            for s in mem:
                a = recs[s]
                s1 = P["E"] - a["contribution"] + P["alpha"] * T / len(mem)
                if inst == "SI":
                    s2 = (P["S"] - P["pc"] * a["punish_tokens_given"] - P["rc"] * a["reward_tokens_given"]
                          + P["re"] * a["reward_tokens_received"] - P["pe"] * a["punish_tokens_received"])
                else:
                    s2 = P["S"]
                for k, want in (("stage1_payoff", s1), ("stage2_payoff", s2), ("round_payoff", s1 + s2)):
                    if abs(float(a[k]) - want) > 1e-6:
                        warn.append(f"round {r}, seat {s}: {k} {a[k]} logged, {want} by the rules")
                if a["punish_tokens_received"] != recv_p[s] or a["reward_tokens_received"] != recv_r[s]:
                    warn.append(f"round {r}, seat {s}: received tokens differ from the sanction lists")
                cum[s] += float(a["round_payoff"])
                if abs(cum[s] - float(a["cumulative_payoff"])) > 1e-6:
                    warn.append(f"round {r}, seat {s}: cumulative {a['cumulative_payoff']} logged, {cum[s]} summed")
        if x["si_list"] is not None and sorted(x["si_list"]) != sorted((x["agent_of"][s], recs[s]["contribution"]) for s in x["si"]):
            warn.append(f"round {r}: si_contributions differs from the round result")
        given = sorted(sa["seat"] for sa in x["sanctions"])
        if given != (sorted(x["si"]) if len(x["si"]) >= 2 else []):
            warn.append(f"round {r}: sanction turns {given}, SI members {x['si']}")
    B = P["rounds"] * (P["E"] + P["S"])
    for s in range(n):
        e = cum[s] - P["init"]
        if earnings is not None and abs(e - earnings[s]) > 1e-6:
            warn.append(f"seat {s}: earnings {earnings[s]} logged, {e} summed")
        want = max(-1.0, min(1.0, e / B - 1))
        if rewards is not None and abs(want - rewards[s]) > 1e-9:
            warn.append(f"seat {s}: reward {rewards[s]} logged, {want} by the rule")
    return warn


# ------------------------------------------------------------------------------------------------ calls
def call_meta(c):
    """Phase and round of a call, from its prompt ('Current phase: X', the public state's "round"); None when not shown."""
    p = c.get("prompt") or ""
    phase = None
    m = re.search(r"Current phase:\s*([A-Za-z_]+)", p)
    if m:
        t = m.group(1).upper()
        phase = next((ph for ph in PHASES if ph[:7] in t), t)
    i = p.find("Public state:")
    rm = re.search(r'"round":\s*(\d+)', p[i:]) if i >= 0 else None
    return phase, int(rm.group(1)) if rm else None


def _matches(call, st):
    a = _norm_answer(call.get("answer"))
    if not isinstance(a, dict):
        return False
    aid, p = a.get("action_id"), a.get("payload") or {}
    mv = st["move"]
    if st["phase"] == "INSTITUTION":
        return aid == "choose_institution" and p.get("institution") == mv["institution"]
    if st["phase"] == "CONTRIBUTE":
        v = p.get("contribute")
        amt = v.get("amount") if isinstance(v, dict) else None
        return (aid == "contribute" and isinstance(amt, (int, float)) and not isinstance(amt, bool) and amt == mv["amount"])
    return aid == "sanction" and p.get("punish") == mv["punish"] and p.get("reward") == mv["reward"]


def attach_calls(steps, calls):
    """Tie calls to steps in time order. For each step, the window is the run of calls from the pointer whose seat, phase and
    round agree with the step (unknown values agree with anything); the first call in it whose answer is the logged move is the
    one; earlier calls in the window were failed attempts (format retries) and their reasoning is kept above it. With no matching
    answer the last call of the window is used and the step notes that the engine applied its fallback (SFI, half the
    endowment, or no tokens; rules: Actions). Returns the number of steps that got a call."""
    meta = [call_meta(c) for c in calls]

    def fits(k, st, strict=False):
        seat, (ph, rnd) = calls[k].get("seat"), meta[k]
        if strict:
            return seat == st["seat"] and ph == st["phase"] and rnd == st["round"]
        return seat in (None, st["seat"]) and ph in (None, st["phase"]) and rnd in (None, st["round"])
    ptr, tied = 0, 0
    for st in steps:
        if ptr < len(calls) and not fits(ptr, st):    # resynchronise on a call that names this very decision
            ahead = next((k for k in range(ptr, min(len(calls), ptr + 60)) if fits(k, st, strict=True)), None)
            if ahead is None:
                continue
            print(f"warning: calls {ptr}..{ahead - 1} match no decision; skipped", file=sys.stderr)
            ptr = ahead
        j = ptr
        while j < len(calls) and fits(j, st):
            j += 1
        window = list(range(ptr, j))
        if not window:
            continue
        hit = next((k for k in window if _matches(calls[k], st)), None)
        use = hit if hit is not None else window[-1]
        parts = []
        for k in range(ptr, use):
            c = calls[k]
            # the engine's reply to a rejected answer is the next call's prompt ("That was not a legal action. ...")
            why = ("rejected as not a legal action" if "not a legal action" in (calls[k + 1].get("prompt") or "")
                   else "not accepted")
            parts.append(f"[Earlier attempt, {why}: {c['answer'][:300]}]\n{c.get('reasoning') or '(no reasoning returned)'}")
        c = calls[use]
        if parts:
            parts.append("[Final attempt]\n" + (c.get("reasoning") or "(no reasoning returned)"))
        else:
            parts.append(c.get("reasoning") or "")
        if hit is None:
            parts.append(f"[The engine applied its fallback instead of this answer: {c['answer'][:300]}]")
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        if c.get("prompt"):
            st["state"] = en(common.state_part(c["prompt"]))
            st["state_from_prompt"] = True
        a = _norm_answer(c.get("answer"))
        if hit is not None and isinstance(a, dict):
            st["act"] = en(json.dumps(a, ensure_ascii=False))
        ptr = use + 1
        tied += 1
    return tied


# ------------------------------------------------------------------------------------------------ build
def build(bundle_dir, calls=None):
    b = load_source(bundle_dir)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    if not calls:
        calls = b.get("md_calls") or []
    events, summary, run, cfg = b["events"], b.get("summary") or {}, b.get("run") or {}, b.get("config") or {}
    P, rounds = read_game(events, summary, cfg)
    if not rounds:
        raise SystemExit("no finished round in the game log")
    N, R, E, S = P["n"], len(rounds), P["E"], P["S"]
    HI, LO = math.ceil(0.75 * E), math.floor(0.25 * E)          # paper section 3.2: high contributor >= 75%, free rider <= 25%
    game_end = next((e for e in events if e.get("event") == "game_end"), {})
    raw_rw = summary.get("rewards") or {str(k): v for k, v in enumerate(game_end.get("rewards") or [])}
    rewards = [float(raw_rw.get(str(s), raw_rw.get(s, 0.0))) for s in range(N)]
    last_cum = {s: float(rounds[-1]["seats"][s]["cumulative_payoff"]) for s in range(N)}
    earnings = [last_cum[s] - P["init"] for s in range(N)]
    for w in verify(P, rounds, game_end.get("earnings"), rewards if raw_rw else None):
        print("warning:", w, file=sys.stderr)
    for h in ((summary.get("variables") or {}).get("history") or []):    # the last rounds' public records, as the players saw them
        x = next((y for y in rounds if y["round"] == h.get("round")), None)
        for ag in (h.get("agents") or []) if x else []:
            a = x["seats"][x["seat_of"][int(ag["agent"])]]
            want = (a["institution"], a["contribution"], a["punish_tokens_given"], a["reward_tokens_given"],
                    a["punish_tokens_received"] * P["pe"], a["reward_tokens_received"] * P["re"])
            got = (ag["institution"], ag["contribution"], ag["punish_tokens_given"], ag["reward_tokens_given"],
                   ag["punish_effect_received"], ag["reward_effect_received"])
            pays = all(abs(float(ag[k]) - float(a[k])) < 0.0051 for k in ("stage1_payoff", "stage2_payoff", "round_payoff"))
            if want != got or not pays:
                print(f"warning: round {h['round']} agent {ag['agent']}: public record {got} differs from the round result {want}", file=sys.stderr)

    def cls(c):
        return "high" if c >= HI else "low" if c <= LO else "mid"
    CLS_COLOR = {"high": "c3", "mid": "c4", "low": "c2"}
    CLS_LABEL = {"high": f"{HI} to {E} tokens", "mid": f"{LO + 1} to {HI - 1} tokens", "low": f"0 to {LO} tokens"}
    # The sanction kinds' colors are set once the steps are known (see "sanction colors" below): the palette has seven tokens and the
    # institution and contribution kinds already use five of them, so punish / reward / both take only colors no present kind uses.
    KINDS = {"si": ["Chooses SI", "c1"], "sfi": ["Chooses SFI", "hold"],
             "high": [f"Contributes {HI} to {E}", "c3"], "mid": [f"Contributes {LO + 1} to {HI - 1}", "c4"], "low": [f"Contributes 0 to {LO}", "c2"],
             "none": ["No sanction tokens", "muted"], "punish": ["Punishes", "ink"], "reward": ["Rewards", "c4"], "both": ["Punishes and rewards", "c3"]}

    # ---- what each seat saw (rebuilt from the log; the saved prompt where there is one)
    def own_line(x, s):
        a = x["seats"][s]
        pr, rr = a["punish_tokens_received"], a["reward_tokens_received"]
        return (f"- Round {x['round']}: {a['institution']}, contributed {a['contribution']}; stage 1 {f2(a['stage1_payoff'])} + stage 2 "
                f"{f2(a['stage2_payoff'])} = {f2(a['round_payoff'])}; gave {a['punish_tokens_given']} punishment / {a['reward_tokens_given']} "
                f"reward tokens; received {pr} / {rr}" + (f" (effect {MINUS}{pr * P['pe']:g} / +{rr * P['re']:g})" if pr or rr else "")
                + f"; rank {a['rank']} of {N}.")

    def anon_lines(x):
        """The round's anonymous record: agents with identical records grouped, sanction fields only when not zero."""
        by = {}
        for s in range(N):
            a = x["seats"][s]
            key = (a["institution"], a["contribution"], a["punish_tokens_given"], a["reward_tokens_given"],
                   a["punish_tokens_received"] * P["pe"], a["reward_tokens_received"] * P["re"],
                   f2(a["stage1_payoff"]), f2(a["stage2_payoff"]), f2(a["round_payoff"]))
            by.setdefault(key, []).append(x["agent_of"][s])
        parts = []
        for key, ags in sorted(by.items(), key=lambda t: min(t[1])):
            inst, c, pg, rg, pe, re_, s1, s2, tot = key
            who = f"all {N} agents" if len(ags) == N else alist(ags)
            t = f"{who}: {inst}, contributed {c}"
            if pg or rg:
                t += f", assigned {pg} punishment / {rg} reward tokens"
            if pe or re_:
                t += f", received effects {MINUS}{pe:g} / +{re_:g}"
            parts.append(t + f", payoff {s1} + {s2} = {tot}")
        return [f"- Round {x['round']}: " + "; ".join(parts) + "."]

    prompt = run.get("prompt") or {}

    def observation(ri, s, phase):
        x = rounds[ri]
        a = x["seats"][s]
        prev = rounds[ri - 1]["seats"][s] if ri else None
        score = float(prev["cumulative_payoff"]) if prev else P["init"]
        lines = ["(Rebuilt from the game log; the exact prompt for this turn was not saved.)",
                 f"You are seat P{s} ({pname(s)} on this page). Round {x['round']} of {P['rounds']}. Your anonymous agent number this round: {a['agent']}.",
                 f"Your account: {f2(score)} tokens ({f2(P['init'])} at the start)"
                 + (f"; your rank by cumulative payoff after round {x['round'] - 1}: {prev['rank']} of {N} (ties ranked by agent number)." if prev else "; no rank yet.")]
        if phase == "INSTITUTION":
            lines.append("Current phase: INSTITUTION. Choose the SI (sanctions and rewards possible) or the SFI (none) for this round. "
                         "Choices are not revealed until the round ends.")
        elif phase == "CONTRIBUTE":
            lines.append(f"You are in the {a['institution']} this round. Current phase: CONTRIBUTE. Put an integer from 0 to {E} of your {E} tokens "
                         f"into the {a['institution']}'s project; the project pays every member {P['alpha']:g} {TIMES} the members' total {DIV} "
                         "the number of members. Group sizes are not shown.")
        else:
            others = [ag for ag, _ in sorted(x["si_list"] or []) if ag != a["agent"]]
            lst = ", ".join(f"{ag}: {c}" for ag, c in sorted(x["si_list"] or []))
            lines.append(f"You are in the SI and contributed {a['contribution']}; your stage-1 payoff so far: {f2(a['stage1_payoff'])}.")
            lines.append(f"SI contributions this round (si_contributions, agent: tokens): {lst}. You are agent {a['agent']}.")
            lines.append(f"Current phase: SANCTION. You may assign up to {S} tokens in total to the other SI members ({alist(others)}): a punishment "
                         f"token costs you {P['pc']:g} and takes {P['pe']:g} from the receiver; a reward token costs you {P['rc']:g} and gives "
                         f"{P['re']:g}. Tokens you do not use stay in your account. Sanctions take effect after every SI member has decided.")
        win = rounds[max(0, ri - P["H"]):ri]
        if win:
            span = f"the last {numword(len(win))} rounds" if len(win) > 1 else "the last round"
            lines.append(f"Your own records of {span} (my_history):")
            lines += [own_line(y, s) for y in win]
            lines.append(f"Anonymous records of {span} (history, by the agent numbers used in that round):")
            for y in win:
                lines += anon_lines(y)
        else:
            lines.append("No earlier rounds yet (history and my_history are empty).")
        return "\n".join(lines)

    # ---- steps: per round, the institution choices, the contributions, the sanction lists, each in seat order
    steps, feed = [], []
    for ri, x in enumerate(rounds):
        r, recs = x["round"], x["seats"]
        x["first_step"] = len(steps)
        for s in range(N):
            inst = recs[s]["institution"]
            steps.append({"seat": s, "x": ri, "kind": "si" if inst == "SI" else "sfi", "phase": "INSTITUTION", "round": r,
                          "move": {"institution": inst}, "head": f"Round {r} · chooses the {inst}",
                          "act": json.dumps({"action_id": "choose_institution", "payload": {"institution": inst}})})
        for s in range(N):
            c = int(recs[s]["contribution"])
            steps.append({"seat": s, "x": ri, "kind": cls(c), "phase": "CONTRIBUTE", "round": r, "move": {"amount": c},
                          "head": f"Round {r} · contributes {c} of {E} to the {recs[s]['institution']}",
                          "act": json.dumps({"action_id": "contribute", "payload": {"contribute": {"amount": c}}})})
        x["last_contrib_step"] = len(steps) - 1
        for sa in sorted(x["sanctions"], key=lambda t: t["seat"]):
            pt, wt = sum(sa["punish"]), sum(sa["reward"])
            kind = "both" if pt and wt else "punish" if pt else "reward" if wt else "none"
            bits = []
            if pt:
                bits.append("punishes " + targets(x, sa["punish"]))
            if wt:
                bits.append("rewards " + targets(x, sa["reward"]))
            steps.append({"seat": sa["seat"], "x": ri, "kind": kind, "phase": "SANCTION", "round": r,
                          "move": {"punish": sa["punish"], "reward": sa["reward"]},
                          "head": f"Round {r} · " + ("; ".join(bits) if bits else "assigns no tokens"),
                          "act": json.dumps({"action_id": "sanction", "payload": {"punish": sa["punish"], "reward": sa["reward"]}})})
        x["last_step"] = len(steps) - 1

    # ---- sanction colors: punish is 'ink' (no institution or contribution kind uses it); reward and both take the first of c4, c3,
    # hold whose kind is absent from this game (c1 is always taken: only SI members sanction). With no color left, a decision that
    # both punishes and rewards is shown as punishing, and reward falls back to the SFI's gray (the SFI block sits at the bottom of
    # its slot, a sanction block at the top of an SI slot).
    present = {st["kind"] for st in steps}
    taken = {KINDS[k][1] for k in ("si", "sfi", "high", "mid", "low") if k in present}
    free = [c for c in ("c4", "c3", "hold") if c not in taken]
    if "reward" in present:
        KINDS["reward"][1] = free.pop(0) if free else "hold"
    free = [c for c in free if c != "hold"]           # 'both' never takes the gray that resembles 'none' (muted)
    merge_both = "both" in present and not free
    if "both" in present and free:
        KINDS["both"][1] = free.pop(0)
    if merge_both:
        KINDS["punish"][0] = "Punishes (some also reward)"
        for st in steps:
            if st["kind"] == "both":
                st["kind"] = "punish"

    def sanction_kind(pt, wt):
        return "none" if not (pt or wt) else "punish" if pt and (merge_both or not wt) else "both" if pt else "reward"
    first_prompt = False
    for st in steps:
        st["state"] = observation(st["x"], st["seat"], st["phase"])
        if (not first_prompt and prompt.get("seat") == st["seat"] and prompt.get("phase") == st["phase"]
                and re.search(r'Public state: \{[^\n]*"round": %d\b' % st["round"], prompt.get("text") or "")):
            st["state"] = en(common.state_part(prompt["text"]))      # the saved prompt of the game's first decision
            st["state_from_prompt"] = True
        first_prompt = True

    # ---- per-round facts
    for ri, x in enumerate(rounds):
        recs = x["seats"]
        x["contrib"] = {s: int(recs[s]["contribution"]) for s in range(N)}
        x["pay"] = {s: float(recs[s]["round_payoff"]) for s in range(N)}
        x["ptok"] = sum(sum(sa["punish"]) for sa in x["sanctions"])
        x["rtok"] = sum(sum(sa["reward"]) for sa in x["sanctions"])
        x["low_si"] = [s for s in x["si"] if x["contrib"][s] <= LO]

    def all_low(x):
        """Every SI member of the round at LO or less: '0' when all gave nothing, else 'LO or less'; None otherwise."""
        if not x["si"] or len(x["low_si"]) < len(x["si"]):
            return None
        return "0" if all(x["contrib"][s] == 0 for s in x["si"]) else f"{LO} or less"

    # ---- the end game: free riders in the SI of the game's last round, and nobody punished them (rules: the round number and the
    # number of rounds are public, so every player knew it was the last round; punishment could no longer change later behavior).
    # When every SI member was a free rider there was no contributor left to bear the cost of punishing (no second-order dilemma).
    # Shown in the last round's feed entry only.
    lastx = rounds[-1]
    endgame = ""
    if (lastx["round"] == P["rounds"] and len(lastx["si"]) >= 2 and lastx["low_si"]
            and not any(lastx["seats"][s]["punish_tokens_received"] for s in lastx["low_si"])):
        hi_si = [s for s in lastx["si"] if lastx["contrib"][s] >= HI]
        k = len(lastx["low_si"])
        amt = all_low(lastx)
        if amt:
            endgame = (f"Round {lastx['round']} was the last round and all {numword(len(lastx['si']))} SI members had contributed {amt}: "
                       + ("nobody who had paid in was left to punish anyone." if amt == "0" else
                          "there was no high contributor left to punish the free riders."))
        else:
            endgame = (f"Round {lastx['round']} was the last round, and the public state showed it (round {lastx['round']} of {P['rounds']}): "
                       f"punishing could no longer change anyone's later contributions, and each punishment token would have cost its giver "
                       f"{P['pc']:g} (the second-order dilemma). No one punished the free rider{'s' if k > 1 else ''}")
            if hi_si:
                adv = (sum(lastx["pay"][s] for s in lastx["low_si"]) / k - sum(lastx["pay"][s] for s in hi_si) / len(hi_si))
                if adv > 0:
                    endgame += f", who earned {f2(adv)} more than the high contributors that round"
            endgame += "."

    # ---- feed: stage 1 (institutions, contributions, project) and stage 2 (sanctions, payoffs) of every round
    def joined(mem, inst):
        who = f"All {numword(N)} players" if len(mem) == N else plist(mem)
        return f"{who} joined the {inst}"
    for ri, x in enumerate(rounds):
        recs, r = x["seats"], x["round"]
        parts = []
        for inst, mem in (("SI", x["si"]), ("SFI", x["sfi"])):
            if not mem:
                continue
            T = sum(x["contrib"][s] for s in mem)
            share = P["alpha"] * T / len(mem)
            cs = {s: x["contrib"][s] for s in mem}
            s1 = {s: round(float(recs[s]["stage1_payoff"]), 9) for s in mem}
            t = joined(mem, inst)
            t += (f" and contributed {cs[mem[0]]}" + (" each" if len(mem) > 1 else "") + "." if len(set(cs.values())) == 1
                  else ". Contributions: " + grouped(cs, str, "from") + ".")
            t += f" Project: {P['alpha']:g} {TIMES} {T} {DIV} {len(mem)} = {f2(share)} to each member"
            t += (f", so stage 1 paid {f2(s1[mem[0]])}" + (" each" if len(mem) > 1 else "") + "." if len(set(s1.values())) == 1
                  else "; stage 1: " + grouped(s1, f2, "for") + ".")
            parts.append(t)
        if not x["si"]:
            parts.append("Nobody joined the SI, so there is no sanction stage.")
        if x["si_list"] is not None and len(x["si"]) >= 2:
            lows = [ag for ag, c in sorted(x["si_list"]) if c <= LO]
            amt = all_low(x) if len(lows) == len(x["si_list"]) else None
            parts.append("SI members then saw the contributions listed by agent number: " + ", ".join(f"{ag}: {c}" for ag, c in sorted(x["si_list"]))
                         + (f"; all {numword(len(lows))} had contributed {amt}." if amt else
                            f"; {apairs(x, [x['seat_of'][ag] for ag in lows])} had contributed {LO} or less." if lows else "."))
        feed.append({"x": ri, "step": x["last_contrib_step"], "who": "Stage 1", "text": " ".join(parts)})
        parts = []
        if len(x["si"]) >= 2:
            givers = [sa for sa in x["sanctions"] if any(sa["punish"]) or any(sa["reward"])]
            if not givers:
                amt = all_low(x)
                if amt and not (x is lastx and endgame):      # the last round's end-game sentence says it
                    t = f"No SI member assigned a single token; all {numword(len(x['si']))} had contributed {amt}, and each kept the {S}-token second endowment"
                else:
                    t = f"No SI member assigned a single token; each kept the {S}-token second endowment"
                if x["low_si"] and not amt:
                    t += f", although {apairs(x, x['low_si'])} had contributed {LO} or less and could have been punished"
                parts.append(t + ".")
            else:
                for sa in givers:
                    bits = ([f"punished {targets(x, sa['punish'])}"] if any(sa["punish"]) else []) \
                        + ([f"rewarded {targets(x, sa['reward'])}"] if any(sa["reward"]) else [])
                    parts.append(f"{pname(sa['seat'])} (agent {sa['agent']}) " + "; ".join(bits) + ".")
                quiet = [sa["seat"] for sa in x["sanctions"] if not (any(sa["punish"]) or any(sa["reward"]))]
                if quiet:
                    parts.append(f"{plist(quiet)} assigned no tokens.")
            if x is lastx and endgame:
                parts.append(endgame)
        elif x["si"]:
            parts.append(f"{pname(x['si'][0])} was alone in the SI: no sanction stage, and it kept the {S} tokens.")
        if x["sfi"]:
            parts.append(f"SFI members kept their second endowment of {S}.")
        parts.append("Round payoff: " + grouped({s: round(x["pay"][s], 9) for s in range(N)}, f2, "for") + ".")
        if ri == R - 1:
            parts.append(f"Final earnings (beyond the {f2(P['init'])} starting tokens): "
                         + grouped({s: round(earnings[s], 9) for s in range(N)}, f2, "for") + "; rewards: "
                         + grouped({s: round(rewards[s], 9) for s in range(N)}, lambda v: f"{v:.3f}", "for") + ".")
        feed.append({"x": ri, "step": x["last_step"], "who": "Stage 2" if len(x["si"]) >= 2 else "Result", "text": " ".join(parts)})

    # ---- attach the model calls, if any
    tied = attach_calls(steps, calls) if calls else 0
    has_reason = any(s.get("thinking") for s in steps)
    from_prompt = sum(1 for s in steps if s.get("state_from_prompt"))
    for st in steps:
        st.setdefault("thinking", "")
        for k in ("phase", "round", "move", "state_from_prompt"):
            st.pop(k, None)

    # ---- aggregates
    labels = [f"R{x['round']}" for x in rounds]
    total = N * R
    si_n = sum(len(x["si"]) for x in rounds)
    allc = [x["contrib"][s] for x in rounds for s in range(N)]
    avg_c = sum(allc) / len(allc)
    n_high = sum(1 for c in allc if c >= HI)
    n_low = sum(1 for c in allc if c <= LO)
    ptok = sum(x["ptok"] for x in rounds)
    rtok = sum(x["rtok"] for x in rounds)
    n_sanc = sum(len(x["sanctions"]) for x in rounds)
    avg_pay = [sum(x["pay"].values()) / N for x in rounds]
    full_round = lambda x: all(x["contrib"][s] == E for s in range(N))
    lead = 0
    while lead < R and full_round(rounds[lead]):
        lead += 1
    all_si = si_n == total
    coop = P["alpha"] * E + S                     # every seat, everyone contributes everything in one institution
    nash = E + S                                  # nobody contributes, nobody sanctions

    by_round = {x["round"]: x for x in rounds}
    low_si_rounds = [x for x in rounds if x["low_si"] and len(x["si"]) >= 2]
    lows_at = [(x["round"], s) for x in rounds for s in range(N) if x["contrib"][s] <= LO]

    # ---- chart 1: payoff per round, group average and by contribution level, against the two benchmarks
    # A class dot is drawn only in rounds where contributions fell into more than one class: with one class its average is the group
    # average, and the dot would only sit on the bar top (and on the benchmark line).
    cls_pay = {}
    split = [len({cls(x["contrib"][s]) for s in range(N)}) > 1 for x in rounds]
    for k in ("high", "mid", "low"):
        v = []
        for ri, x in enumerate(rounds):
            ss = [s for s in range(N) if cls(x["contrib"][s]) == k]
            v.append(round(sum(x["pay"][s] for s in ss) / len(ss), 4) if ss and split[ri] else None)
        if any(t is not None for t in v):
            cls_pay[k] = v
    lines = [{"name": {"high": f"Contributed {HI} to {E}", "mid": f"Contributed {LO + 1} to {HI - 1}", "low": f"Contributed 0 to {LO}"}[k],
              "color": CLS_COLOR[k], "v": v, "noline": True, "dots": True} for k, v in cls_pay.items()]
    lines += [{"name": f"All contribute ({coop:g})", "color": "ink", "dash": True, "thin": True, "v": [coop] * R},
              {"name": f"None contribute ({nash:g})", "color": "muted", "dash": True, "thin": True, "v": [nash] * R}]
    n_split = sum(split)
    sub1 = (f"Bars: the average round payoff (stage 1 + stage 2) of the {numword(N)} players. "
            + ((f"Dots, in the round{'s' if n_split > 1 else ''} where contributions differed"
                + (f" (round {rounds[split.index(True)]['round']})" if n_split == 1 else "") + ": the average payoff of the players who "
                f"contributed {HI} or more (the paper's high contributors)" + (f", of those who contributed {LO + 1} to {HI - 1}," if "mid" in cls_pay else "")
                + f" and of those who contributed {LO} or less (free riders). ") if cls_pay else "")
            + f"Dashed: {coop:g}, what each player earns when all contribute everything in one institution (dark), "
            f"and {nash:g}, what each earns when nobody contributes (light). ")
    gap = [ri for ri in range(R) if "high" in cls_pay and "low" in cls_pay and cls_pay["high"][ri] is not None and cls_pay["low"][ri] is not None]
    if gap:
        ri = gap[-1]
        x = rounds[ri]
        hi_v, lo_v = cls_pay["high"][ri], cls_pay["low"][ri]
        lows = [s for s in range(N) if x["contrib"][s] <= LO]
        hit = sum(x["seats"][s]["punish_tokens_received"] for s in lows)
        fr = "the free rider" + ("s" if len(lows) > 1 else "")
        hc = "the high contributor" + ("s" if sum(1 for s in range(N) if x["contrib"][s] >= HI) > 1 else "")
        sub1 += f"In round {x['round']} {fr} earned {f2(lo_v)} and {hc} {f2(hi_v)}: "
        if lo_v > hi_v:
            sub1 += f"free riding paid {f2(lo_v - hi_v)} more"
            if hit:
                sub1 += f", even after the {tokens(hit)} of punishment {fr} received."
            elif x["low_si"] and len(x["si"]) >= 2:
                sub1 += (f", and no one punished {fr}." if len(x["low_si"]) == len(lows)
                         else f", and no one punished {fr} in the SI.")
            else:
                sub1 += "."
        elif lo_v < hi_v:
            sub1 += (f"after {tokens(hit)} of punishment {fr} earned {f2(hi_v - lo_v)} less." if hit
                     else f"{fr} earned {f2(hi_v - lo_v)} less.")
        else:
            sub1 += "both earned the same."
    elif lead == R:
        sub1 += f"Every round paid {f2(coop)} to every player, the social optimum."
    chart1 = {"title": "Round payoff: the group, the contributors" + (" and the free riders" if "low" in cls_pay else ""),
              "sub": sub1.strip(), "height": 210, "fmt": "price",
              "bars": [{"name": "Group average", "color": "c1", "v": [round(v, 4) for v in avg_pay]}], "lines": lines}

    # ---- chart 2: contributions, player by player (pale: in the SFI)
    cells = []
    for s in range(N):
        row = []
        for x in rounds:
            a, c = x["seats"][s], x["contrib"][s]
            cell = {"c": CLS_COLOR[cls(c)], "t": str(c), "tip": f"{a['institution']}, contributed {c}, round payoff {f2(a['round_payoff'])}"}
            if a["institution"] != "SI":
                cell["o"] = .45
            row.append(cell)
        cells.append(row)
    seen = [k for k in ("high", "mid", "low") if any(cls(c) == k for c in allc)]
    legend2 = [[CLS_LABEL[k] + {"high": " (high contributor)", "mid": "", "low": " (free riding)"}[k], CLS_COLOR[k]] for k in seen]
    if not all_si:
        legend2.append(["Pale cell: in the SFI", "hold"])
    if lead == R:
        story = f"Every player contributed all {E} tokens in every round."
    else:
        story = (f"Everyone contributed all {E} in rounds 1 {'and' if lead == 2 else 'to'} {lead}. " if lead > 1 else
                 f"Everyone contributed all {E} in round 1. " if lead == 1 else "")
        byr = {}
        for r_, s in lows_at:
            byr.setdefault(r_, []).append(s)
        few = len(byr) <= 3                          # a sentence per round with low contributions, however many players
        if byr and few:
            for r_, ss in sorted(byr.items()):
                y = by_round[r_]
                story += (f"In round {r_}{', the last,' if r_ == rounds[-1]['round'] else ''} {plist(ss, N)} contributed "
                          + ("nothing" if all(y["contrib"][s] == 0 for s in ss) else f"{LO} or less")
                          + (f" while everyone else contributed {E}" if len(ss) < N and all(y["contrib"][t] == E for t in range(N) if t not in ss)
                             else "") + ". ")
        else:
            story += f"{n_low} of {total} contributions were {LO} or less and {n_high} were {HI} or more. "
        n_mid = total - n_low - n_high
        if n_mid and few:
            story += f"{n_mid} contribution{'s were' if n_mid > 1 else ' was'} between {LO + 1} and {HI - 1}. "
    sub2 = (f"Tokens each player put into its institution's project each round, out of {E} (the number in the cell). "
            + (f"All {numword(N)} players chose the SI in every round. " if all_si else f"{si_n} of {total} choices were the SI; pale cells are SFI rounds. ")
            + story + "Players never saw each other's seats, only agent numbers redrawn every round.")
    chart2 = {"title": "Contributions, player by player", "sub": sub2, "height": 200,
              "grid": {"rows": [pname(s) for s in range(N)], "cells": cells, "legend": legend2, "rowHeight": 26}}

    # ---- chart 3: the sanction stage, with the free riders the SI members could see. Colors as on the rest of the page: orange is
    # free riding (as in the contributions grid), gray (muted) is zero, the decision colors are the timeline's sanction kinds. When
    # nobody assigned a token all game, the per-player rows would all be gray, so the grid collapses to two rows.
    def zero(tip):
        return {"c": "muted", "t": "0", "tip": tip}

    def cell_color(kind):
        col = KINDS[kind][1]
        return {"c": col, "o": .55} if col == "ink" else {"c": col}      # white text stays readable on ink in both themes
    collapse = ptok + rtok == 0
    top, prow, tot = [], [[] for _ in range(N)], []
    for x in rounds:
        if len(x["si"]) >= 2:
            k = len(x["low_si"])
            top.append({"c": "c2", "t": str(k), "tip": f"{k} SI member{'s' if k > 1 else ''} at {LO} or less: {apairs(x, x['low_si'])}"}
                       if k else zero(f"No SI member contributed {LO} or less"))
            tot.append(zero(f"None of the {numword(len(x['si']))} SI members assigned a token"))
        else:
            top.append(None)
            tot.append(None)
        given = {sa["seat"]: sa for sa in x["sanctions"]}
        for s_ in range(N):
            sa = given.get(s_)
            if not sa:
                prow[s_].append(None)
                continue
            pt, wt = sum(sa["punish"]), sum(sa["reward"])
            a = x["seats"][s_]
            rec = f"received {a['punish_tokens_received']} punishment and {a['reward_tokens_received']} reward tokens"
            if pt or wt:
                tip = "; ".join(([f"punished {targets(x, sa['punish'])}"] if pt else [])
                                + ([f"rewarded {targets(x, sa['reward'])}"] if wt else [])) + "; " + rec
                prow[s_].append(dict(cell_color(sanction_kind(pt, wt)), t=(f"P{pt}" if pt else "") + (f"R{wt}" if wt else ""), tip=tip))
            else:
                prow[s_].append(zero("Assigned no tokens; " + rec))
    legend3 = [[f"Free riders: SI members at {LO} or less (number)", "c2"]]
    for kind, lab in (("punish", "Punished (P tokens)" + (", some also rewarded" if merge_both else "")), ("reward", "Rewarded (R tokens)"),
                      ("both", "Punished and rewarded")):
        if kind in present and not (kind == "both" and merge_both):
            legend3.append([lab, KINDS[kind][1]])
    legend3.append(["0: none" if not collapse else "0: no free riders, no tokens", "muted"])

    def pa(x, seats):
        """Seats of one round -> 'Player 6 (agent 6)' or 'Player 5 (agent 0) and Player 6 (agent 2)', by player number."""
        items = [f"{pname(t)} (agent {x['agent_of'][t]})" for t in sorted(seats)]
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    def si_story(x):
        """One sentence on the free riders an SI round's list showed, and what came of it (the reasons are in the feed)."""
        r, si, lows = x["round"], x["si"], x["low_si"]
        when = f"In round {r}" + (", the last," if r == P["rounds"] else "")
        amt = all_low(x)
        if amt:
            return f"{when} all {numword(len(si))}" + ("" if len(si) == N else " SI members") + f" gave {amt}."
        t = f"{when} {pa(x, lows)} gave " + ("0" if all(x["contrib"][s] == 0 for s in lows) else f"{LO} or less")
        rest = [s for s in si if s not in lows]
        if all(x["contrib"][s] == E for s in rest):
            t += f" while the other {numword(len(rest))} gave {E}" if len(rest) > 1 else f" while {pa(x, rest)} gave {E}"
        left = P["rounds"] - r
        if left > 0:
            t += f", with {numword(left)} round{'s' if left > 1 else ''} still to play"
        hit = sum(x["seats"][s]["punish_tokens_received"] for s in lows)
        return t + (", and nobody punished." if not hit else f"; {'they' if len(lows) > 1 else 'it'} received {tokens(hit)} of punishment.")
    rule3 = (f"SI members see the contributions listed by agent number and may each spend up to {S} tokens: a punishment token costs "
             f"{P['pc']:g} and takes {P['pe']:g} from the target, a reward token costs {P['rc']:g} and gives {P['re']:g}.")
    if collapse:
        tail = f"Top row: SI members at {LO} or less; bottom row: tokens assigned, none in all {n_sanc} sanction decisions."
    else:
        tail = (f"Top row: SI members at {LO} or less; other rows: each SI member's decision (P punishment tokens, R reward tokens, 0 none), hover "
                f"for the targets. Over the game {tokens(ptok)} of punishment and {tokens(rtok)} of reward were assigned in {n_sanc} sanction "
                f"decisions (punishment-to-reward ratio {'n/a' if not rtok else f'{ptok / rtok:.2f}'}; the paper's LLMs 0.00 to 0.88, its humans 1.66).")
    if low_si_rounds and len(low_si_rounds) <= 3:
        tail += " " + " ".join(si_story(x) for x in low_si_rounds)
    elif low_si_rounds:
        tail += f" The list showed free riders in {numword(len(low_si_rounds))} rounds."
    sub3 = rule3 + " " + tail
    if collapse:
        rows3, cells3, rh3 = ["Free riders", "Tokens given"], [top, tot], 28
    else:
        rows3, cells3, rh3 = ["Free riders"] + [pname(s_) for s_ in range(N)], [top] + prow, 24
    chart3 = {"title": "Sanction stage: who could be punished, and who was", "sub": sub3, "height": len(rows3) * rh3 + 24,
              "grid": {"rows": rows3, "cells": cells3, "legend": legend3, "rowHeight": rh3}}

    charts = [chart1, chart2, chart3]
    if not all_si:                                 # institution sizes matter only when the SFI was used
        charts.insert(1, {"title": "Institution sizes", "sub": f"Members of the SI and of the SFI in each round ({si_n} of {total} choices were the SI). "
                          "Choices are revealed only when the round ends.", "height": 190, "fmt": "int",
                          "bars": [{"name": "SI members", "color": "c1", "v": [len(x["si"]) for x in rounds]},
                                   {"name": "SFI members", "color": "hold", "v": [len(x["sfi"]) for x in rounds]}], "lines": []})

    # ---- tiles
    B = P["rounds"] * (E + S)
    ev = {s: round(earnings[s], 9) for s in range(N)}
    if len(set(ev.values())) == 1:
        t1 = {"k": "Earnings per player", "v": f"{f2(earnings[0])} each", "s": f"Reward {rewards[0]:.3f} for every player. "}
    else:
        by = {}
        for s_ in range(N):
            by.setdefault(ev[s_], []).append(s_)
        grp = sorted(by.items(), key=lambda t: (len(t[1]), -t[0]))
        txt = [f"{plist(ss)}: {f2(v)} (reward {rewards[ss[0]]:.3f})" for v, ss in grp[:-1]]
        v, ss = grp[-1]
        txt.append((f"the other {numword(len(ss))}" if len(ss) > 1 else pname(ss[0])) + f": {f2(v)} ({rewards[ss[0]]:.3f})")
        t1 = {"k": "Earnings per player", "v": f"{f2(min(earnings))} to {f2(max(earnings))}", "s": "; ".join(txt) + ". "}
    t1["s"] += (f"Full cooperation every round: {f2(P['rounds'] * coop)}; no contributions: {f2(B)}. "
                f"The paper's models: 554 to 721 on average")
    t1["spark"] = [round(v, 4) for v in avg_pay]
    t2 = {"k": "Chose the SI", "v": f"{100 * si_n / total:.0f}%",
          "s": f"{si_n} of {total} institution choices" + (" (every player, every round)" if all_si else "")
               + "; the paper's models: 28% to 100%; the humans of G\u00fcrerk et al.: 92.9% in the final rounds"}
    t3 = {"k": "Average contribution", "v": f"{avg_c:.2f} of {E}",
          "s": f"{n_high} of {total} contributions at {HI} or more, {n_low} at {LO} or less"
               + (f", all in round {lows_at[0][0]}" if lows_at and len({r_ for r_, _ in lows_at}) == 1 else "")
               + "; the paper's models: 5.4 to 18.7; its humans: 18.3"}
    if ptok == 0 and rtok == 0:
        t4 = {"k": "Sanction tokens", "v": "None",
              "s": f"Not one punishment or reward token in {n_sanc} sanction decisions"
                   + ((f", not even against the {numword(len(low_si_rounds[0]['low_si']))} free rider{'s' if len(low_si_rounds[0]['low_si']) > 1 else ''}"
                       f" of round {low_si_rounds[0]['round']}") if len(low_si_rounds) == 1 else
                      f", although free riders were in the SI in {numword(len(low_si_rounds))} rounds" if low_si_rounds else "")}
    else:
        t4 = {"k": "Sanction tokens", "v": f"{ptok} P \u00b7 {rtok} R",
              "s": f"{ptok} punishment and {rtok} reward tokens in {n_sanc} sanction decisions; the paper's LLMs gave punishment-to-reward ratios of 0.00 to 0.88, its humans 1.66"}
    tiles = [t1, t2, t3, t4]

    # ---- title, lede, foot
    inst_phrase = " in the sanctioning institution" if all_si else ""

    def avg1(v):
        """An average contribution for the title: whole numbers without decimals ('0', '17'), others with one ('16.3')."""
        return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"

    def dev_phrase(x):
        """What the players below the full endowment did in one round: 'Player 6 free-rides unpunished in round 14',
        'all seven contribute nothing in round 15'."""
        devs = [s for s in range(N) if x["contrib"][s] < E]
        many = len(devs) > 1
        allz = all(x["contrib"][s] == 0 for s in devs)
        lowall = all(x["contrib"][s] <= LO for s in devs)
        subj = f"all {numword(N)}" if len(devs) == N else plist(devs)
        if lowall and len(devs) < N:
            pred = "free-ride" if many else "free-rides"
        else:
            pred = ("contribute" if many else "contributes") + (" nothing" if allz else f" {LO} or less" if lowall else " less")
        if lowall and len(x["si"]) >= 2 and all(s in x["si"] for s in devs) and any(s not in devs for s in x["si"]):
            pun = [s for s in devs if x["seats"][s]["punish_tokens_received"]]
            pred += (" unpunished" if not pun else (" and are punished" if many else " and is punished") if len(pun) == len(devs) else "")
        return f"{subj} {pred} in round {x['round']}"
    if lead == R:
        title = (f"All {numword(N)} players join the sanctioning institution" if all_si else f"All {numword(N)} players") \
                + f" and contribute everything in all {R} rounds" + ("; nobody ever sanctions" if ptok + rtok == 0 else "")
    elif lead == R - 1 and R > 1:
        last = rounds[-1]
        dev = [s for s in range(N) if last["contrib"][s] < E]
        allz = all(last["contrib"][s] == 0 for s in dev)
        title = (f"{numword(lead).capitalize()} rounds of full cooperation{inst_phrase}, then {plist(dev)} "
                 + ("contribute" if len(dev) > 1 else "contributes") + (" nothing" if allz else " less") + " in the last round")
        if (len(last["si"]) >= 2 and all(s in last["si"] for s in dev)
                and not any(last["seats"][s]["punish_tokens_received"] for s in dev)):
            title += f" and nobody punishes {'them' if len(dev) > 1 else 'it'}"
    elif lead == R - 2 and lead >= 1 and not full_round(rounds[-1]):
        # a long lead, then two single-round events (e.g. one unpunished free rider, then nobody contributing in the last round)
        title = (f"{numword(lead).capitalize()} round{'s' if lead > 1 else ''} of full cooperation{inst_phrase}; "
                 f"{dev_phrase(rounds[-2])}, and {dev_phrase(rounds[-1])}")
    elif lead >= 1:
        # after the full-cooperation lead: 'falls' only when the last round is the lowest; a dip that recovers says so
        avgr = [sum(x["contrib"].values()) / N for x in rounds]
        after = avgr[lead:]
        low = min(after)
        k_low = rounds[lead + after.index(low)]["round"]
        title = f"Full cooperation{inst_phrase} for {numword(lead)} round{'s' if lead > 1 else ''}, then the average contribution "
        if avgr[-1] <= low and max(avgr[lead:-1] or [0]) >= E:      # broke, was back at full, and the last round is the lowest
            title += (f"dips to {avg1(avgr[lead])} of {E} in round {rounds[lead]['round']}, is back at {E}, "
                      f"and falls to {avg1(avgr[-1])} in round {R}")
        elif avgr[-1] <= low:
            title += f"falls to {avg1(avgr[-1])} of {E} by round {R}"
        elif full_round(rounds[-1]):
            back = R - 1
            while back > lead and full_round(rounds[back - 1]):
                back -= 1
            title += (f"dips to {avg1(low)} of {E} in round {k_low} and is back at {E} of {E} "
                      + (f"in round {rounds[back]['round']}, the last" if back == R - 1 else f"from round {rounds[back]['round']} on"))
        elif avgr[-1] >= avgr[lead]:
            title += f"dips to {avg1(low)} of {E} in round {k_low} and recovers to {avg1(avgr[-1])} by round {R}"
        else:
            title += f"drops to {avg1(low)} of {E} in round {k_low} and ends at {avg1(avgr[-1])} in round {R}"
    else:
        title = (f"An average contribution of {avg_c:.1f} of {E}, {100 * si_n / total:.0f}% of choices in the sanctioning institution, "
                 f"{ptok + rtok} sanction tokens")
    lede = (f"{numword(N).capitalize()} players play {P['rounds']} rounds of a public goods game. Each round every player first joins the Sanctioning "
            f"Institution (SI) or the Sanction-Free Institution (SFI), then puts 0 to {E} of its {E} tokens into its institution's project, which "
            f"pays every member {P['alpha']:g} {TIMES} the members' total {DIV} their number; SI members then get {S} more tokens and may spend them to "
            f"punish (cost {P['pc']:g}, {MINUS}{P['pe']:g} to the target) or reward (cost {P['rc']:g}, +{P['re']:g}) other members, while SFI "
            f"members simply keep them. Each player's score is its total earnings: keeping one's tokens while the others contribute pays most in "
            f"any single round unless the other SI members punish it, yet everyone contributing everything earns {coop:g} a round against {nash:g} when nobody does.")
    meta = b.get("meta") or {}
    model = meta.get("model") or run.get("model") or ""
    model = model.split("/")[-1] if model else ""
    date = recorded_date(b)
    if not date:
        print("warning: neither meta.json nor run.json gives recorded_at; the page says the recording date is unknown", file=sys.stderr)
    seed = run.get("seed", cfg.get("seed"))
    cm = re.search(r"_(eco_sanction_pgg\w*?)_\d+$", str(meta.get("source_run") or run.get("source_run") or ""))
    cfg_name = f"config {cm.group(1)}, " if cm else ""
    foot = ((f"Data: the {date} recording" if date else "Data: a recording (date not saved)") + f" of eco_sanction_pgg ({cfg_name}seed {seed}, {N} players, {P['rounds']} rounds, endowment {E} + "
            f"sanction endowment {S}, multiplier {P['alpha']:g}, punishment cost {P['pc']:g} / effect {P['pe']:g}, reward cost {P['rc']:g} / effect "
            f"{P['re']:g}, at most {S} tokens per sanction turn), every player played by {model or 'the model'}; every number on this page is computed from its "
            "game log. "
            + ("Observations and reasoning come from the saved model calls. " if has_reason else
               "This recording was made before reasoning was saved; the game will be re-recorded with reasoning. ")
            + ("" if from_prompt == len(steps) else
               ("The observation of the first decision is the saved prompt; the others are rebuilt from the log. " if from_prompt else
                "The observations are rebuilt from the log. "))
            + "Institution choices and contributions are read from each round's result record and shown in seat order, the order the engine "
            "asks; the sanction lists are logged one by one. "
            f"Player k is seat P(k{MINUS}1) in the log and in the prompts; during the game players saw only anonymous agent numbers, redrawn every "
            f"round. Earnings exclude the {f2(P['init'])} starting tokens; reward = earnings {DIV} {B} {MINUS} 1, clipped to [{MINUS}1, 1] "
            f"({B} = {P['rounds']} rounds {TIMES} {E + S}, the earnings when nobody contributes). High contributor: {HI} tokens or more; free rider: "
            f"{LO} or fewer (75% and 25% of the endowment, paper section 3.2). Paper: Guzman Piedrahita, Yang, Sachan, Ramponi, Sch\u00f6lkopf and Jin "
            "(2025), \"Corrupted by Reasoning: Reasoning Language Models Become Free-Riders in Public Goods Games\" (COLM 2025), following "
            "G\u00fcrerk, Irlenbusch and Rockenbach (2006); the model and human ranges quoted above are from its Table 1 and section 4.")
    d = {"id": GID, "name": "Sanctioned public goods", "title": title, "lede": lede, "seatNames": [pname(s) for s in range(N)], "tiles": tiles,
         "foot": foot, "x": {"n": R, "labels": labels}, "charts": charts,
         "kinds": {k: KINDS[k] for k in KINDS if any(s["kind"] == k for s in steps)}, "steps": steps, "feed": feed,
         "feedTitle": "Round by round",
         "feedSub": "Stage 1 (institutions, contributions, the project) and stage 2 (sanctions, payoffs) of every round, with the anonymous agent "
                    "numbers the players saw. The selected step is highlighted; later entries are hidden."}
    when = f"recorded {date}" if date else "recording date unknown"
    who = model or "model not recorded"
    d["eyebrow"] = (f"social_game_env · {GID} · {who}, reasoning on · {when}" if has_reason
                    else f"social_game_env · {GID} · {who} · {when} · reasoning not saved")
    if calls:
        print(f"calls: {len(calls)} read, {tied} of {len(steps)} decisions tied to a call", file=sys.stderr)
    return d


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
    common.write(build(src, calls), out, date=recorded_date(src) or "date unknown")
