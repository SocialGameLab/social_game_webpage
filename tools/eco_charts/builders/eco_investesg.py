"""Replay data for eco_investesg (InvestESG: climate investment as an intertemporal social dilemma between companies and
ESG-conscious investors; Hou, Yuan, Leibo and Jaques, ICLR 2025).

usage: python3 -I eco_investesg.py BUNDLE OUT_DIR [CALLS.jsonl]
  BUNDLE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md), or a file
          inside one.
  OUT_DIR where <OUT_DIR>/eco_investesg.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer, followup_answer when the answer came from
          a follow-up question, usage; see common.load_calls). Calls are tied to decisions in time order: a call belongs to the
          next decision of the same seat and year (both read from its prompt); the call whose answer the engine reads (strictly:
          bare JSON, as base.parse_action) as the logged action is that decision's call, and on a fallback move (fallback: true)
          the call whose answer is the rejected reply the recorder logged (earlier calls in the same window were retries; their
          reasoning is kept above it). Its reasoning becomes the step's thinking and its prompt the observation. The log is
          appended to, so a recording restarted into the same output directory leaves the earlier attempt's calls in it; the
          runs are told apart (call numbers restart at 1, the (year, seat) order goes back) and only the run this game came
          from is used (the one starting at the game's run stamp, else the one whose answers match the most decisions).

test only:  python3 -I eco_investesg.py --test-from-calls CALLS.jsonl OUT_DIR
          builds the page from a call log alone (for example the growing log of a recording still running, whose half-written
          last line is left out): the stand-in
          bundle is assembled in memory (see TEST HELPER at the end of this file) and the page says in its eyebrow, title and
          foot that it is a test build from a partial call log. Nothing is written but OUT_DIR/eco_investesg.json.

One round is one year. Every seat decides once a year, companies first, then investors, all from the same snapshot (the round
is buffered; rules: Round Procedure). The page's year record (capital, investor flows, spending, climate risk, disasters,
margins, holdings, rewards) is computed by replaying the logged actions through the rules' transition (equations 1 to 11 with
the official code's fixed disaster draws, NumPy default_rng(100 t), (500 t), (1000 t), re-implemented here in the standard
library and checked against NumPy). Before it is used the replay is checked three ways and every disagreement is printed as a
warning: (1) the transition reproduces the rules' fixed-policy table (status quo and universal 0.5%); (2) every number the
seats were shown in their prompts (every company's capital, margin and ESG score, every investor's cash and holdings, the
seat's own totals and its history lines) matches the replay; (3) the summary's metrics, per-company and per-investor totals,
yearly series and the rewards match. If the prompts disagree with the replay (a configuration the replay does not cover),
the record is rebuilt from the prompts and the summary instead and the page says so."""
import datetime, json, math, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_investesg"
MINUS, DOT, TIMES, DIV, ARROW, ALPHA, NDASH = "−", "·", "×", "÷", "→", "α", "–"
HAZARDS = ("extreme heat", "heavy precipitation", "drought")
# The rules' Configuration defaults; used only for what neither the summary nor the prompts state.
DEFAULT_PARAMS = {"companies": 5, "investors": 3, "scripted_companies": 0, "scripted_investors": 0, "years": 100,
                  "company_capital": 10.0, "investor_capital": 16.7, "esg_preference": [0.0], "esg_disclosure": True,
                  "allow_greenwash": False, "greenwash_coef": 2.0, "allow_resilience": False, "resilience_rate": 3.0,
                  "action_cap": 0.05, "market_growth": 0.10, "growth_sd": 0.0, "damage": 0.07, "uncertain_damage": False,
                  "disaster_draws": "fixed", "hazard_p0": [0.28, 0.13, 0.17], "hazard_growth": [0.0083, 0.0018, 0.003],
                  "hazard_lambda": [0.0222, 0.0326, 0.038], "climate_observable": False, "investment_rounding": "exact",
                  "real_data_seeding": False, "strict_bankruptcy": False, "lock_in": 1, "first_year": 2021}
# Options the replay does not reproduce (chance drawn from the episode seed, scripted agents, locked or seeded decisions).
UNSUPPORTED = (("disaster_draws", lambda v: v not in (None, "fixed")), ("uncertain_damage", bool), ("growth_sd", bool),
               ("real_data_seeding", bool), ("strict_bankruptcy", bool), ("lock_in", lambda v: (v or 1) > 1),
               ("scripted_companies", bool), ("scripted_investors", bool))
# The rules' fixed-policy table (Reward), rows 1 and 2: every company spends nothing / 0.5% on mitigation, three alpha-0
# investors holding every company. (final climate risk, market wealth, disasters, total mitigation, ending capital per company)
RULES_TABLE = {0.0: (1.037, 238, 116, 0.0, 47.7), 0.005: (0.512, 10321, 60, 1050.8, 2064.2)}
# Per-step reasoning is kept whole unless the page's JSON would pass this many bytes (the replay page loads one JSON file per
# game and its host takes files of at most 16 MB); then the longest are shortened in the middle, visibly marked.
MAX_BYTES = 15_000_000


def warn(msg):
    print("warning: " + msg, file=sys.stderr)


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_RUN = re.compile(r"[぀-ヿ㐀-鿿가-힯]+")
_PUNCT = {"。": ". ", "、": ", ", "「": '"', "」": '"', "『": '"', "』": '"', "【": "[", "】": "]",
          "《": '"', "》": '"', "〈": "<", "〉": ">", "〜": "~"}


def en(s):
    """English only on the page: full-width forms become ASCII, CJK punctuation its ASCII counterpart, and any other run of
    Chinese, Japanese or Korean characters a visible marker."""
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
def load_source(src):
    if isinstance(src, dict):
        return src
    if not os.path.isdir(src):
        d = os.path.dirname(os.path.abspath(src))
        if os.path.exists(os.path.join(d, "game_log.jsonl")):
            return load_source(d)
        raise SystemExit(f"not a demo bundle: {src}")
    b = common.load_bundle(src)
    meta = os.path.join(src, "meta.json")
    b["meta"] = json.load(open(meta, encoding="utf-8")) if os.path.exists(meta) else {}
    return b


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _lenient_lines(path):
    """The JSON lines of a log that may still be growing: a last line that does not parse (half written) is left out."""
    lines = [line for line in open(path, encoding="utf-8") if line.strip()]
    raw = []
    for k, line in enumerate(lines):
        try:
            raw.append(json.loads(line))
        except ValueError:
            if k == len(lines) - 1:
                warn("the call log's last line is incomplete (still being written); left out")
            else:
                raise
    return raw


def load_calls(path, lenient=False):
    """common.load_calls plus what it drops from each line: why the call stopped ('finish'), the output tokens it used, its
    start time and model. Same lines, same order. With `lenient` (test builds from a live log) a half-written last line is
    left out; each call is then read exactly as common.load_calls reads it."""
    if lenient and path and os.path.exists(path):
        raw = _lenient_lines(path)
        out = []
        for c in raw:
            prompt = c.get("prompt")
            if prompt is None:
                msgs = c.get("messages") or []
                prompt = msgs[-1]["content"] if msgs else ""
            out.append({"seat": c.get("seat", common.seat_of(prompt)), "prompt": prompt,
                        "answer": (c.get("followup_answer") or c.get("answer") or "").strip(),
                        "reasoning": c.get("reasoning") or "", "rescued": bool(c.get("followup_answer"))})
    else:
        out = common.load_calls(path)
        raw = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()] if out else []
    if not out:
        return out
    if len(raw) != len(out):
        warn(f"call log: {len(raw)} lines but {len(out)} calls; finish reasons and token counts are not used")
        return out
    for c, r in zip(out, raw):
        u = r.get("usage") or {}
        if isinstance(u, str):                       # some logs store the usage dict as its Python repr
            m = re.search(r"'completion_tokens': (\d+)", u)
            u = {"completion_tokens": int(m.group(1))} if m else {}
        c.update(finish=r.get("finish") or r.get("finish_reason"), out_tokens=u.get("completion_tokens"),
                 t_start=_f(r.get("t_start")), model=r.get("model"),
                 call=r.get("call") if isinstance(r.get("call"), int) and not isinstance(r.get("call"), bool) else None)
        if r.get("followup_answer") in ("None", "null"):  # a stringified null is no follow-up
            c["rescued"] = False
            c["answer"] = (r.get("answer") or "").strip()
    return out


def recorded_date(b, calls=None):
    """The recording date: meta.json or run.json recorded_at, else the timestamp that opens the source run's name, else the first
    call's start time. With none of them the build stops: the page must not show a guessed date."""
    meta, run = b.get("meta") or {}, b.get("run") or {}
    for v in (meta.get("recorded_at"), run.get("recorded_at")):
        if v:
            return str(v)[:10]
    for v in (meta.get("source_run"), run.get("source_run")):
        m = re.match(r"(\d{4})(\d{2})(\d{2})T\d{6}Z", str(v or ""))
        if m:
            return "-".join(m.groups())
    t = next((c.get("t_start") for c in calls or [] if isinstance(c.get("t_start"), (int, float))), None)
    if t is not None:
        return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d")
    raise SystemExit("no recording date in the source (meta.json or run.json recorded_at, the source run's name, or the call "
                     "log's start times); refusing to guess one")


def recorded_model(b, calls=None):
    """The model that played, without its provider prefix ('openai/deepseek-flash' -> 'deepseek-flash'); None when unrecorded."""
    model = (b.get("meta") or {}).get("model") or (b.get("run") or {}).get("model") or \
        next((c.get("model") for c in calls or [] if c.get("model")), None)
    return str(model).split("/")[-1] if model else None


# ------------------------------------------------------------------------------------------------ small helpers
def num(s):
    return float(str(s).replace(",", "").replace(MINUS, "-"))


def decimals(s):
    s = str(s)
    return len(s.split(".")[1]) if "." in s else 0


def tn(v, d=2):
    """Trillions as the prompts print them (2 decimals, thousands separated), with a true minus sign."""
    if v is None:
        return "n/a"
    s = f"{abs(v):,.{d}f}"
    return (MINUS if v < 0 and float(s.replace(",", "")) != 0 else "") + s


def tn0(v):
    """A large amount with as few digits as still say something: 2,064 / 238 / 47.7 / 9.85."""
    if v is None:
        return "n/a"
    a = abs(v)
    s = f"{a:,.0f}" if a >= 1000 else f"{a:,.0f}" if a >= 100 else f"{a:.1f}" if a >= 10 else f"{a:.2f}"
    return (MINUS if v < 0 and float(s.replace(",", "")) != 0 else "") + s


def sg(v, d=2, unit=""):
    if v is None:
        return "n/a"
    s = f"{abs(v):,.{d}f}"
    z = float(s.replace(",", "")) == 0
    return ("" if z else MINUS if v < 0 else "+") + s + unit


def pc(v, d=1):
    """A share in percent (v is a fraction)."""
    return "n/a" if v is None else f"{100 * v:.{d}f}%"


def mp(m):
    """A mitigation percentage as the seat wrote it, rounded to 3 decimals, without trailing zeros: 5%, 0.5%, 0.001%."""
    s = f"{m:.3f}".rstrip("0").rstrip(".")
    return s + "%"


def and_list(xs):
    xs = list(xs)
    if not xs:
        return ""
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def years_text(ys):
    """Calendar years as runs: 2021, 2023 to 2030 and 2040."""
    ys = sorted(ys)
    runs, k = [], 0
    while k < len(ys):
        j = k
        while j + 1 < len(ys) and ys[j + 1] == ys[j] + 1:
            j += 1
        runs.append(str(ys[k]) if j == k else f"{ys[k]} and {ys[j]}" if j == k + 1 else f"{ys[k]} to {ys[j]}")
        k = j + 1
    return and_list(runs)


NUM_WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def nw(n):
    return NUM_WORDS.get(n, str(n))


def cap1(s):
    return s[:1].upper() + s[1:] if s else s


# ------------------------------------------------------------------------------------------------ actions (rules: Actions)
# The engine (ecosim base.parse_action, _check and InvestESG._canonical) reads an answer strictly: json.loads of the text the
# client returns (already stripped), nothing else (no code fences, no JSON inside prose); top-level keys only action_id and
# payload; the payload exactly the action's fields. The builder reads answers the same way, for the logged moves and for tying
# calls to decisions, so a rejected attempt is never taken for the decision.
PCT_DECIMALS = 3                                             # allocations are percentages rounded to 0.001% (engine)


def _json_strict(s):
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s.strip())
    except (ValueError, RecursionError):
        return None


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _alloc_keys(P):
    return ["mitigation"] + (["greenwash"] if P["allow_greenwash"] else []) + (["resilience"] if P["allow_resilience"] else [])


def _cap_pct(P):
    return round(float(P["action_cap"]) * 100, PCT_DECIMALS)


def _short(v, n=40):
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else repr(v)
    return s if len(s) <= n else s[:n - 3] + "..."


def why_illegal(move, role, P, live=None):
    """Why the engine rejects an answer, as (a short phrase for a step's title, a sentence); None when it is legal."""
    text = move if isinstance(move, str) else json.dumps(move) if isinstance(move, dict) else ""
    if not text.strip():
        return "an empty answer", "The answer was empty"
    a = _json_strict(text)
    if a is None:
        fenced = text.strip().startswith("```")
        return ("not bare JSON" if fenced else "not valid JSON",
                "The answer was a JSON code block (the game reads bare JSON only)" if fenced else "The answer was not valid JSON")
    if not isinstance(a, dict):
        return "not an action object", "The answer was JSON but not an action object"
    extra = sorted(set(a) - {"action_id", "payload"})
    if extra:
        return "extra keys", f"The answer had keys the game does not accept ({', '.join(map(str, extra))})"
    want = "allocate" if role == "COMPANY" else "invest"
    if a.get("action_id") != want:
        return (f"action {_short(a.get('action_id'), 20)}",
                f"The answer named the action {_short(a.get('action_id'))}, not {want}")
    p = a.get("payload", {})
    if not isinstance(p, dict):
        return "no payload object", "The answer's payload was not an object"
    if role == "COMPANY":
        keys, cap = _alloc_keys(P), _cap_pct(P)
        if set(p) != set(keys):
            return ("wrong fields", f"The answer gave the fields {', '.join(sorted(map(str, p))) or 'none'} instead of "
                    + and_list(keys))
        for k in keys:
            v = p[k]
            if not _is_num(v):
                return f"{k} not a number", f"The answer's {k} was {_short(v)}, not a number"
            if v > cap:
                return f"{k} {v:g}%, over the {cap:g}% cap", f"The answer asked for {k} {v:g}% of capital, over the {cap:g}% cap"
            if v < 0:
                return f"{k} {v:g}%, below 0", f"The answer asked for {k} {v:g}% of capital, below 0"
        return None
    if set(p) != {"companies"}:
        return "wrong fields", f"The answer gave the fields {', '.join(sorted(map(str, p))) or 'none'} instead of companies"
    cs = p["companies"]
    if not isinstance(cs, list):
        return "companies not a list", f"The answer's companies was {_short(cs)}, not a list"
    seen = set()
    for c in cs:
        ok = _is_num(c) and c == math.floor(c) and 0 <= c < int(P["companies"])
        if not ok:
            return f"no company {_short(c, 12)}", f"The answer named {_short(c)}, which is not a company number"
        c = int(c)
        if live is not None and not live[c]:
            return f"Company {c} is bankrupt", f"The answer named Company {c}, which is bankrupt"
        if c in seen:
            return f"Company {c} twice", f"The answer named Company {c} twice"
        seen.add(c)
    return None


def fallback_action(role, P):
    """The engine's fallback (InvestESG._fallback): no spending for a company, all cash for an investor."""
    if role == "COMPANY":
        return {"action_id": "allocate", "payload": {k: 0.0 for k in _alloc_keys(P)}}
    return {"action_id": "invest", "payload": {"companies": []}}


def parse_action(move, role, P, live=None):
    """The action as the engine reads it: ('allocate', (m, g, r)) in percent of capital rounded to 3 decimals, or ('invest',
    sorted company numbers); None when the engine rejects it (it then applies the fallback: no spending, or all cash)."""
    if why_illegal(move, role, P, live) is not None:
        return None
    p = _json_strict(move)["payload"]
    if role == "COMPANY":
        return ("allocate", tuple(round(float(p.get(k, 0.0)), PCT_DECIMALS) for k in ("mitigation", "greenwash", "resilience")))
    return ("invest", tuple(sorted(int(c) for c in p["companies"])))


# ------------------------------------------------------------------------------------------------ what a seat was shown
NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"        # 2,064.20 / 20.92 / 0.001 (never a trailing comma)
SIGNED = r"([+\-−]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?))"
RX_SEAT = re.compile(r"You are seat P(\d+)\. Your role: (\w+)")
RX_YEAR = re.compile(r"\bYear (\d{4}) \(period (\d+)(?: of (\d+))?\)")
RX_CO = re.compile(r"^- Company (\d+): (.*)$", re.M)
RX_INV = re.compile(r"^- Investor (\d+)(?: \(ESG preference alpha = ([\d.]+)\))?: (.*)$", re.M)
RX_OWN_CO = re.compile(r"You are Company (\d+)\. Your capital: " + NUM + r" trillion, of which investors' money " + NUM + r"\. "
                       r"Your cumulative spending: (.*?)\. Your total capital gains so far: " + SIGNED + " trillion")
RX_OWN_INV = re.compile(r"You are Investor (\d+)(?: \(ESG preference alpha = ([\d.]+)\))?\. Your capital: " + NUM + r" trillion "
                        r"\(cash " + NUM + r"\)\. Your cumulative reward so far: " + SIGNED + r" \(returns " + SIGNED +
                        r", ESG part " + SIGNED + r"\)")
RX_HIST_CO = re.compile(r"^- (\d{4}): capital at decision " + NUM + r" \(of which investors' money " + NUM + r"\); spent (.*?); "
                        r"disasters (\d+); margin " + SIGNED + r"%; gain " + SIGNED + r"; ESG score ([\d.]+); capital at end "
                        r"" + NUM + r"\.", re.M)
RX_HIST_INV = re.compile(r"^- (\d{4}): capital " + NUM + r"; (.*?); portfolio return " + SIGNED + r"%; portfolio ESG score "
                         r"([\d.]+); reward " + SIGNED + r"; capital at end " + NUM + r"\.", re.M)
RX_SETTINGS = re.compile(r"Settings: (\d+) companies with ([\d.]+) trillion each at the start; (\d+) investors with ([\d.]+) "
                         r"trillion each; (\d+) years; market growth ([\d.]+)% a year; loss per disaster ([\d.]+)% of capital; "
                         r"spending cap ([\d.]+)% per category; ESG disclosure (on|off); (.*?)\.\n")


def _spend_parts(text):
    out = {}
    for k, v in re.findall(r"(mitigation|greenwash\w*|resilience) " + NUM + r"", text or ""):
        out["greenwash" if k.startswith("greenwash") else k] = v
    return out


def view_facts(view):
    """Everything a prompt reports, as (number, printed text) pairs where the text matters for the check's tolerance."""
    f = {"seat": None, "role": None, "year": None, "period": None, "T": None, "co": {}, "inv": {}, "own": None, "hist": []}
    m = RX_SEAT.search(view or "")
    if m:
        f["seat"], f["role"] = int(m.group(1)), m.group(2).upper()
    m = RX_YEAR.search(view or "")
    if m:
        f["year"], f["period"] = int(m.group(1)), int(m.group(2))
        f["T"] = int(m.group(3)) if m.group(3) else None
    head = (view or "").split("\nYou are Company ")[0].split("\nYou are Investor ")[0]
    for m in RX_CO.finditer(head):
        body = m.group(2)
        c = {"bankrupt": "bankrupt" in body.lower()}
        for key, rx in (("capital", r"capital " + NUM + r""), ("margin", r"last year's margin " + SIGNED + "%"),
                        ("esg", r"ESG score ([\d.]+)"), ("loss", r"loss (?:share )?per disaster ([\d.]+)%")):
            mm = re.search(rx, body)
            if mm:
                c[key] = mm.group(1).rstrip(".")
        f["co"][int(m.group(1))] = c
    for m in RX_INV.finditer(head):
        body = m.group(3)
        c = {"alpha": m.group(2)}
        mm = re.search(r"cash " + NUM + r"", body)
        if mm:
            c["cash"] = mm.group(1).rstrip(".")
        hp = body.split("holdings", 1)[1] if "holdings" in body else ""
        c["holdings"] = {int(i): v.rstrip(".") for i, v in re.findall(r"Company (\d+) " + NUM + r"", hp)}
        f["inv"][int(m.group(1))] = c
    m = RX_OWN_CO.search(view or "")
    if m:
        f["own"] = {"kind": "company", "i": int(m.group(1)), "capital": m.group(2), "inv_money": m.group(3),
                    "cum": _spend_parts(m.group(4)), "gain": m.group(5)}
        for h in RX_HIST_CO.finditer(view):
            f["hist"].append({"year": int(h.group(1)), "Kint": h.group(2), "inv_money": h.group(3), "spent": _spend_parts(h.group(4)),
                              "X": int(h.group(5)), "margin": h.group(6), "gain": h.group(7), "esg": h.group(8), "K_end": h.group(9)})
    m = RX_OWN_INV.search(view or "")
    if m:
        f["own"] = {"kind": "investor", "j": int(m.group(1)), "alpha": m.group(2), "capital": m.group(3), "cash": m.group(4),
                    "U": m.group(5), "Uret": m.group(6), "Uesg": m.group(7)}
        for h in RX_HIST_INV.finditer(view):
            mid = h.group(3)
            held = [int(x) for x in re.findall(r"\d+", mid.split("(")[0])] if "ompan" in mid else []
            each = re.search(r"\(" + NUM + r"(?: each)?\)", mid)
            f["hist"].append({"year": int(h.group(1)), "KI": h.group(2), "held": held, "each": each.group(1) if each else None,
                              "ret": h.group(4), "port_esg": h.group(5), "reward": h.group(6), "cap_end": h.group(7), "mid": mid})
    return f


# ------------------------------------------------------------------------------------------------ parameters
def read_params(b, views):
    """The game's parameters: the summary's info.params, else what the prompts say, else the rules' defaults (marked)."""
    info = (b.get("summary") or {}).get("info") or {}
    P = dict(DEFAULT_PARAMS)
    src = {k: "rules default" for k in P}
    for k, v in (info.get("params") or {}).items():
        if v is not None:
            P[k] = v
            src[k] = "summary"
    text = next((v for v in views if "Settings:" in v), "")
    m = RX_SETTINGS.search(text)
    if m:
        got = {"companies": int(m.group(1)), "company_capital": float(m.group(2)), "investors": int(m.group(3)),
               "investor_capital": float(m.group(4)), "years": int(m.group(5)), "market_growth": float(m.group(6)) / 100,
               "damage": float(m.group(7)) / 100, "action_cap": float(m.group(8)) / 100, "esg_disclosure": m.group(9) == "on",
               "allow_greenwash": "greenwashing allowed" in m.group(10) or "greenwashing on" in m.group(10),
               "allow_resilience": "resilience spending allowed" in m.group(10) or "resilience allowed" in m.group(10)}
        bm = re.search(r"beta = ([\d.]+)", m.group(10))
        if bm:
            got["greenwash_coef"] = float(bm.group(1))
        for k, v in got.items():
            if src[k] == "summary":
                if (abs(float(P[k]) - float(v)) > 1e-9 if isinstance(v, float) else P[k] != v):
                    warn(f"parameter {k}: summary {P[k]}, prompt {v}")
            else:
                P[k], src[k] = v, "prompt"
        hz = re.search(r"probabilities start at ([\d.]+)%, ([\d.]+)%, ([\d.]+)%", text)
        if hz and src["hazard_p0"] != "summary":
            P["hazard_p0"], src["hazard_p0"] = [float(x) / 100 for x in hz.groups()], "prompt"
        y0 = re.search(r"over \d+ one-year periods \((\d{4}) to (\d{4})\)", text)
        if y0:
            P["first_year"] = int(y0.group(1))
    # each investor's alpha: the prompts' investor lines, else the summary's per-investor record, else the parameter
    N = int(P["investors"])
    alphas = [None] * N
    for v in views:
        for j, c in view_facts(v)["inv"].items():
            if c.get("alpha") is not None and j < N:
                alphas[j] = float(c["alpha"])
        if all(a is not None for a in alphas):
            break
    for j, iv in enumerate(info.get("investors") or []):
        if j < N and iv.get("alpha") is not None:
            if alphas[j] is not None and abs(alphas[j] - float(iv["alpha"])) > 1e-9:
                warn(f"Investor {j}: alpha {alphas[j]} in the prompts, {iv['alpha']} in the summary")
            alphas[j] = float(iv["alpha"]) if alphas[j] is None else alphas[j]
    pref = P.get("esg_preference")
    pref = pref if isinstance(pref, list) else [pref]
    P["alphas"] = [a if a is not None else float(pref[j] if j < len(pref) else pref[0]) for j, a in enumerate(alphas)]
    P["src"] = src
    return P


def replay_supported(P):
    return [k for k, bad in UNSUPPORTED if bad(P.get(k))]


# ------------------------------------------------------------------------------------------------ disaster draws
# The official code's fixed draws (rules: Round Procedure, step 5): in period t the first uniform of NumPy's default_rng(100 t),
# default_rng(500 t) and default_rng(1000 t). NumPy's SeedSequence and PCG64, re-implemented; checked against NumPy 2.4 for
# t = 0..129 (all 390 numbers identical).
_M32, _M64, _M128 = 0xFFFFFFFF, (1 << 64) - 1, (1 << 128) - 1
_PCG_MULT = (2549297995355413924 << 64) | 4865540595714422341


def _first_uniform(seed):
    words, n = [], int(seed)
    if n == 0:
        words = [0]
    while n > 0:
        words.append(n & _M32)
        n >>= 32
    hc = [0x43b0d7e5]

    def hashmix(v):
        v = (v ^ hc[0]) & _M32
        hc[0] = (hc[0] * 0x931e8875) & _M32
        v = (v * hc[0]) & _M32
        return v ^ (v >> 16)

    def mix(x, y):
        r = (0xca01f9dd * x - 0x4973f715 * y) & _M32
        return r ^ (r >> 16)
    pool = [hashmix(words[i]) if i < len(words) else hashmix(0) for i in range(4)]
    for s in range(4):
        for d in range(4):
            if s != d:
                pool[d] = mix(pool[d], hashmix(pool[s]))
    for s in range(4, len(words)):
        for d in range(4):
            pool[d] = mix(pool[d], hashmix(words[s]))
    h, st = 0x8b51f9dd, []
    for i in range(8):
        v = (pool[i % 4] ^ h) & _M32
        h = (h * 0x58f38ded) & _M32
        v = (v * h) & _M32
        st.append(v ^ (v >> 16))
    u64 = [st[2 * k] | (st[2 * k + 1] << 32) for k in range(4)]
    inc = ((((u64[2] << 64) | u64[3]) << 1) | 1) & _M128
    s = (0 * _PCG_MULT + inc) & _M128
    s = (s + ((u64[0] << 64) | u64[1])) & _M128
    s = (s * _PCG_MULT + inc) & _M128
    s = (s * _PCG_MULT + inc) & _M128
    x, rot = ((s >> 64) ^ s) & _M64, s >> 122
    return ((((x >> rot) | (x << ((64 - rot) & 63))) & _M64) >> 11) * (1.0 / 9007199254740992.0)


_DRAWS = {}


def draws(t):
    if t not in _DRAWS:
        _DRAWS[t] = (_first_uniform(100 * t), _first_uniform(500 * t), _first_uniform(1000 * t))
    return _DRAWS[t]


# ------------------------------------------------------------------------------------------------ the transition
def hazard_probs(P, t, U_m):
    return [P["hazard_p0"][e] + P["hazard_growth"][e] * t / (1 + P["hazard_lambda"][e] * U_m) for e in range(3)]


def risk_of(ps):
    r = 1.0
    for p in ps:
        r *= 1 - p
    return 1 - r


def simulate(P, acts, years):
    """Replay `years` years. acts[k] = {"alloc": {i: (m, g, r) in percent}, "invest": {j: (companies)}}; a missing entry is the
    fallback (no spending, all cash). Returns one record per year (rules: Round Procedure, steps 1 to 7)."""
    M, N = int(P["companies"]), int(P["investors"])
    gam, L0, eta = P["market_growth"], P["damage"], P["resilience_rate"]
    beta = P["greenwash_coef"] if P["allow_greenwash"] else 0.0
    K = [float(P["company_capital"])] * M
    cash = [float(P["investor_capital"])] * N
    H = [[0.0] * M for _ in range(N)]
    cum = [[0.0, 0.0, 0.0] for _ in range(M)]
    L = [L0] * M
    bankrupt = [False] * M
    G, U, Uret, Uesg = [0.0] * M, [0.0] * N, [0.0] * N, [0.0] * N
    margin_prev, Q_prev = [None] * M, [None] * M
    rows = []
    for k in range(years):
        t = k
        a = acts[k] if k < len(acts) else {}
        row = {"k": k, "year": P["first_year"] + k, "K_start": K[:], "cash_start": cash[:], "H_start": [h[:] for h in H],
               "margin_prev": margin_prev[:], "Q_prev": Q_prev[:], "live": [not x for x in bankrupt]}
        KI = []
        for j in range(N):                                  # 1. investors collect
            for i in range(M):
                if H[j][i]:
                    K[i] -= H[j][i]
                    cash[j] += H[j][i]
                    H[j][i] = 0.0
            KI.append(cash[j])
        choice, amt = [], []
        for j in range(N):                                  # 2. investors redistribute
            S = [i for i in (a.get("invest") or {}).get(j, ()) if not bankrupt[i]]
            x = 0.0
            if S:
                if P["investment_rounding"] == "floor":
                    x = float(math.floor(KI[j] / len(S)))
                    cash[j] = KI[j] - x * len(S)
                else:
                    x = KI[j] / len(S)
                    cash[j] = 0.0
                for i in S:
                    H[j][i] = x
                    K[i] += x
            choice.append(tuple(S))
            amt.append(x)
        Kint = K[:]
        inv_money = [sum(amt[j] for j in range(N) if i in choice[j]) for i in range(M)]
        share, Q = [], []
        for i in range(M):                                  # 3. companies spend
            sh = (0.0, 0.0, 0.0) if bankrupt[i] else tuple(v / 100 for v in (a.get("alloc") or {}).get(i, (0.0, 0.0, 0.0)))
            share.append(sh)
            for c in range(3):
                cum[i][c] += sh[c] * Kint[i]
            if Kint[i] > 0:
                L[i] = L0 * math.exp(-eta * cum[i][2] / Kint[i])
            Q.append(sh[0] + beta * sh[1])
        U_m = sum(c[0] for c in cum)
        ps = hazard_probs(P, t, U_m)                        # 4. climate risk
        d = draws(t)
        hits = [d[e] < ps[e] for e in range(3)]             # 5. disasters
        X = sum(hits)
        gain, margin = [0.0] * M, [0.0] * M
        for i in range(M):                                  # 6. company capital
            if bankrupt[i]:
                continue
            Kn = Kint[i] * (1 - sum(share[i])) * (1 + gam) * (1 - L[i] * X)
            gain[i] = Kn - Kint[i]
            margin[i] = gain[i] / Kint[i] if Kint[i] else 0.0
            G[i] += gain[i]
            if Kn <= 0:
                bankrupt[i] = True
                Kn = 0.0
            K[i] = Kn
        ret, esg_part, rew, port_esg, cap_end = [], [], [], [], []
        for j in range(N):                                  # 7. investor holdings and reward
            for i in range(M):
                H[j][i] = max(0.0, H[j][i] * (1 + margin[i]))
            new = sum(H[j]) + cash[j]
            r = (new - KI[j]) / KI[j] if KI[j] > 0 else 0.0
            pe = sum(H[j][i] * Q[i] for i in range(M)) / new if new > 0 and KI[j] > 0 else 0.0
            e = P["alphas"][j] * pe
            ret.append(r)
            esg_part.append(e)
            rew.append(r + e)
            port_esg.append(pe)
            cap_end.append(new)
            U[j] += r + e
            Uret[j] += r
            Uesg[j] += e
        margin_prev, Q_prev = margin[:], Q[:]
        row.update(KI=KI, choice=choice, amt=amt, Kint=Kint, inv_money=inv_money, share=share,
                   spend=[tuple(s * Kint[i] for s in share[i]) for i in range(M)], cum=[c[:] for c in cum], L=L[:], Q=Q,
                   U_m=U_m, ps=ps, risk=risk_of(ps), draw=d, hits=hits, X=X, K_end=K[:], gain=gain, margin=margin,
                   bankrupt=bankrupt[:], H_end=[h[:] for h in H], cash_end=cash[:], ret=ret, esg_part=esg_part, reward=rew,
                   port_esg=port_esg, cap_end=cap_end, G=G[:], U=U[:], Uret=Uret[:], Uesg=Uesg[:],
                   inv_money_end=[sum(H[j][i] for j in range(N)) for i in range(M)], wealth=sum(K) + sum(cash))
        rows.append(row)
    return rows


def policy_acts(P, years, mitigation, invest="all"):
    """Fixed policies for reference runs: every company the same mitigation percentage (a number, or a list per year), every
    investor holding every company (or the logged choices: a list per year of {j: companies})."""
    M, N = int(P["companies"]), int(P["investors"])
    out = []
    for k in range(years):
        m = mitigation[k] if isinstance(mitigation, list) else mitigation
        alloc = {i: (m[i] if isinstance(m, (list, tuple)) else m, 0.0, 0.0) for i in range(M)}
        inv = {j: tuple(range(M)) for j in range(N)} if invest == "all" else invest[k]
        out.append({"alloc": alloc, "invest": inv})
    return out


def self_test(P):
    """The transition against the rules' fixed-policy table (5 companies of 10, 3 alpha-0 investors of 16.7 holding every
    company, 100 years, exact split, the code's draws). Returns a list of problems ([] when every number agrees)."""
    Q = dict(DEFAULT_PARAMS, alphas=[0.0, 0.0, 0.0], first_year=2021)
    probs = []
    for m, (risk, wealth, ev, mit, cap) in RULES_TABLE.items():
        R = simulate(Q, policy_acts(Q, 100, 100 * m), 100)
        got = (R[-1]["risk"], R[-1]["wealth"], sum(r["X"] for r in R), R[-1]["U_m"], R[-1]["K_end"][0])
        if not (abs(got[0] - risk) < 5e-4 and abs(got[1] - wealth) < 0.5 and got[2] == ev and abs(got[3] - mit) < 0.05
                and abs(got[4] - cap) < 0.05):
            probs.append(f"mitigation {100 * m}%: replay {got}, rules {(risk, wealth, ev, mit, cap)}")
    return probs


def company_reward(G, P):
    K0 = float(P["company_capital"])
    W0 = int(P["companies"]) * K0 + int(P["investors"]) * float(P["investor_capital"])
    gmax = W0 * ((1 + P["market_growth"]) ** int(P["years"]) - 1)
    return math.log(1 + G / K0) / math.log(1 + gmax / K0) if G >= 0 else max(-1.0, G / K0)


def investor_reward(U, alpha, P):
    qmax = P["action_cap"] * (1 + P["greenwash_coef"]) if P["allow_greenwash"] else P["action_cap"]
    return max(-1.0, min(1.0, U / (int(P["years"]) * (P["market_growth"] + alpha * qmax))))


# ------------------------------------------------------------------------------------------------ checks
class Tally:
    def __init__(self):
        self.n, self.bad, self.groups = 0, 0, {}

    def eq(self, group, shown, value, where):
        """A printed number against the replay: equal to the printed precision (half a unit in the last digit)."""
        if shown is None or value is None:
            return
        tol = 0.5 * 10 ** (-decimals(shown)) + 1e-7
        ok = abs(num(shown) - value) <= tol
        self.n += 1
        g = self.groups.setdefault(group, [0, 0])
        g[0] += 1
        if not ok:
            self.bad += 1
            g[1] += 1
            if self.bad <= 25:
                warn(f"{where}: shown {shown}, replay {value:.6f}")


def check_views(rec, moves, P, tally):
    """Every number in every prompt against the replay's state at the start of that year (and the seat's history lines)."""
    M, N = int(P["companies"]), int(P["investors"])
    by_year = {r["year"]: r for r in rec}
    for e in moves:
        f = e["facts"]
        y = f["year"]
        k = y - P["first_year"] if y is not None else None
        if k is None or k > len(rec):
            continue
        if k == len(rec):                                   # a year the replay did not play (cut short): start = last end
            if not rec:
                continue
            last = rec[-1]
            st = {"K": last["K_end"], "cash": last["cash_end"], "H": last["H_end"], "margin": last["margin"], "Q": last["Q"],
                  "G": last["G"], "cum": last["cum"], "U": last["U"], "Uret": last["Uret"], "Uesg": last["Uesg"]}
        else:
            r = rec[k]
            prev = rec[k - 1] if k else None
            st = {"K": r["K_start"], "cash": r["cash_start"], "H": r["H_start"], "margin": r["margin_prev"], "Q": r["Q_prev"],
                  "G": prev["G"] if prev else [0.0] * M, "cum": prev["cum"] if prev else [[0.0] * 3 for _ in range(M)],
                  "U": prev["U"] if prev else [0.0] * N, "Uret": prev["Uret"] if prev else [0.0] * N,
                  "Uesg": prev["Uesg"] if prev else [0.0] * N}
        where = f"{y} P{f['seat']} prompt"
        for i, c in f["co"].items():
            if i >= M:
                continue
            tally.eq("company lines", c.get("capital"), st["K"][i], f"{where}, Company {i} capital")
            if st["margin"][i] is not None:
                tally.eq("company lines", c.get("margin"), 100 * st["margin"][i], f"{where}, Company {i} margin")
                tally.eq("company lines", c.get("esg"), 100 * st["Q"][i], f"{where}, Company {i} ESG score")
        for j, c in f["inv"].items():
            if j >= N:
                continue
            tally.eq("investor lines", c.get("cash"), st["cash"][j], f"{where}, Investor {j} cash")
            held = {i for i in range(M) if st["H"][j][i] > 0.005}
            if set(c["holdings"]) != held and not (set(c["holdings"]) <= held and
                                                    all(st["H"][j][i] < 0.005 for i in held - set(c["holdings"]))):
                tally.n += 1
                tally.bad += 1
                warn(f"{where}, Investor {j} holdings {sorted(c['holdings'])}, replay {sorted(held)}")
            for i, v in c["holdings"].items():
                tally.eq("investor lines", v, st["H"][j][i], f"{where}, Investor {j} holding in Company {i}")
        o = f["own"]
        if o and o["kind"] == "company":
            i = o["i"]
            tally.eq("own lines", o["capital"], st["K"][i], f"{where}, own capital")
            tally.eq("own lines", o["inv_money"], sum(st["H"][j][i] for j in range(N)), f"{where}, own investors' money")
            for c, key in enumerate(("mitigation", "greenwash", "resilience")):
                tally.eq("own lines", o["cum"].get(key), st["cum"][i][c], f"{where}, own cumulative {key}")
            tally.eq("own lines", o["gain"], st["G"][i], f"{where}, own total gains")
            for h in f["hist"]:
                r = by_year.get(h["year"])
                if not r:
                    continue
                w = f"{where}, history {h['year']}"
                tally.eq("history lines", h["Kint"], r["Kint"][i], w + " capital at decision")
                tally.eq("history lines", h["inv_money"], r["inv_money"][i], w + " investors' money")
                for c, key in enumerate(("mitigation", "greenwash", "resilience")):
                    tally.eq("history lines", h["spent"].get(key), 100 * r["share"][i][c], w + f" {key} %")
                tally.eq("history lines", str(h["X"]), r["X"], w + " disasters")
                tally.eq("history lines", h["margin"], 100 * r["margin"][i], w + " margin")
                tally.eq("history lines", h["gain"], r["gain"][i], w + " gain")
                tally.eq("history lines", h["esg"], 100 * r["Q"][i], w + " ESG score")
                tally.eq("history lines", h["K_end"], r["K_end"][i], w + " capital at end")
        elif o and o["kind"] == "investor":
            j = o["j"]
            tally.eq("own lines", o["capital"], sum(st["H"][j]) + st["cash"][j], f"{where}, own capital")
            tally.eq("own lines", o["cash"], st["cash"][j], f"{where}, own cash")
            tally.eq("own lines", o["U"], st["U"][j], f"{where}, own cumulative reward")
            tally.eq("own lines", o["Uret"], st["Uret"][j], f"{where}, own cumulative return part")
            tally.eq("own lines", o["Uesg"], st["Uesg"][j], f"{where}, own cumulative ESG part")
            for h in f["hist"]:
                r = by_year.get(h["year"])
                if not r:
                    continue
                w = f"{where}, history {h['year']}"
                tally.eq("history lines", h["KI"], r["KI"][j], w + " capital")
                if tuple(h["held"]) != tuple(r["choice"][j]):
                    tally.n += 1
                    tally.bad += 1
                    warn(f"{w}: held {h['held']}, replay {r['choice'][j]}")
                tally.eq("history lines", h["each"], r["amt"][j] if r["choice"][j] else None, w + " amount each")
                tally.eq("history lines", h["ret"], 100 * r["ret"][j], w + " return")
                tally.eq("history lines", h["port_esg"], 100 * r["port_esg"][j], w + " portfolio ESG score")
                tally.eq("history lines", h["reward"], r["reward"][j], w + " reward")
                tally.eq("history lines", h["cap_end"], r["cap_end"][j], w + " capital at end")


def _series(se, key, n):
    """A yearly series from info.series, stored either as {key: [values]} or as [{key: value}, ...]."""
    if isinstance(se, dict):
        v = se.get(key)
        return list(v)[:n] if isinstance(v, list) else None
    if isinstance(se, list) and se and isinstance(se[0], dict) and key in se[0]:
        return [x.get(key) for x in se][:n]
    return None


def check_summary(rec, info, rewards, P, tally, seat_of_co=None, seat_of_inv=None):
    """The summary's metrics (rules: Metrics) and the rewards against the replay; returns notes on what was compared."""
    notes = []
    if not rec:
        return notes
    M, N, last = int(P["companies"]), int(P["investors"]), rec[-1]

    def eq(group, want, got, what, tol):
        if want is None or got is None:
            return
        if not _is_num(want) or not _is_num(got):            # a field of another shape (a list, a string): not compared
            warn(f"summary {what}: {_short(want)} is not a number; not compared")
            return
        tally.n += 1
        g = tally.groups.setdefault(group, [0, 0])
        g[0] += 1
        if abs(float(want) - float(got)) > tol:
            tally.bad += 1
            g[1] += 1
            warn(f"summary {what}: {want}, replay {got}")
    if info:
        eq("summary metrics", info.get("climate_risk"), last["risk"], "climate_risk", 1e-6)
        eq("summary metrics", info.get("total_mitigation"), last["U_m"], "total_mitigation", 1e-6 * max(1, last["U_m"]))
        eq("summary metrics", info.get("climate_events"), sum(r["X"] for r in rec), "climate_events", 0)
        eq("summary metrics", info.get("market_wealth"), last["wealth"], "market_wealth", 1e-6 * max(1, last["wealth"]))
        eq("summary metrics", info.get("bankruptcies"), sum(last["bankrupt"]), "bankruptcies", 0)
        full = len(rec) == int(P["years"])
        for i, c in enumerate(info.get("companies") or []):
            if i >= M:
                break
            vals = [("capital", last["K_end"][i]), ("total_gain", last["G"][i]), ("cumulative_mitigation", last["cum"][i][0]),
                    ("cumulative_greenwash", last["cum"][i][1]), ("cumulative_resilience", last["cum"][i][2]),
                    ("investors_money", last["inv_money_end"][i]), ("investment_received", sum(r["inv_money"][i] for r in rec))]
            if full:
                vals.append(("reward", company_reward(last["G"][i], P)))
            for key, val in vals:
                eq("summary companies", c.get(key), val, f"Company {i} {key}", 1e-6 * max(1, abs(val)))
        for j, c in enumerate(info.get("investors") or []):
            if j >= N:
                break
            vals = [("capital", last["cap_end"][j]), ("cash", last["cash_end"][j]), ("total_utility", last["U"][j]),
                    ("total_return", last["Uret"][j]), ("total_esg", last["Uesg"][j])]
            if full:
                vals.append(("reward", investor_reward(last["U"][j], P["alphas"][j], P)))
            for key, val in vals:
                eq("summary investors", c.get(key), val, f"Investor {j} {key}", 1e-6 * max(1, abs(val)))
            # per company (the engine writes these as lists): money placed in each company over the years (its row of the
            # investment matrix) and the holding in each company at the end
            per = (("invested", lambda i: sum(r["amt"][j] for r in rec if i in r["choice"][j])),
                   ("holdings", lambda i: last["H_end"][j][i]))
            for key, mine in per:
                v = c.get(key)
                if isinstance(v, list):
                    for i, x in enumerate(v[:M]):
                        val = mine(i)
                        eq("summary investors", x, val, f"Investor {j} {key}[{i}]", 1e-6 * max(1, abs(val)))
                elif v is not None:
                    tot = sum(mine(i) for i in range(M))
                    eq("summary investors", v, tot, f"Investor {j} {key} (total)", 1e-6 * max(1, abs(tot)))
        im = info.get("investment_matrix")
        if isinstance(im, list):
            for j, row in enumerate(im[:N]):
                for i, v in enumerate((row or [])[:M]):
                    mine = sum(r["amt"][j] for r in rec if i in r["choice"][j])
                    eq("summary investment matrix", v, mine, f"investment_matrix[{j}][{i}]", 1e-6 * max(1, abs(mine)))
        se = info.get("series")
        n = len(rec)
        for key, mine in (("climate_risk", [r["risk"] for r in rec]), ("market_wealth", [r["wealth"] for r in rec]),
                          ("total_mitigation", [r["U_m"] for r in rec])):
            v = _series(se, key, n)
            if v:
                for k, (a, b) in enumerate(zip(v, mine)):
                    eq("summary series", a, b, f"series {key} {rec[k]['year']}", 1e-6 * max(1, abs(b)))
        ev = _series(se, "events", n)
        if ev and all(x is not None for x in ev):
            per = [r["X"] for r in rec]
            cum = [sum(per[:k + 1]) for k in range(n)]
            use = per if [int(x) for x in ev] == per else cum if [int(x) for x in ev] == cum else per
            for k, (a, b) in enumerate(zip(ev, use)):
                eq("summary series", a, b, f"series events {rec[k]['year']}", 0)
            notes.append("per year" if use is per else "cumulative")
    if len(rec) == int(P["years"]):
        sc, si = seat_of_co or {i: i for i in range(M)}, seat_of_inv or {j: M + j for j in range(N)}
        for i in range(M):
            eq("rewards", rewards.get(sc.get(i)), company_reward(last["G"][i], P), f"reward of Company {i}", 1e-6)
        for j in range(N):
            eq("rewards", rewards.get(si.get(j)), investor_reward(last["U"][j], P["alphas"][j], P), f"reward of Investor {j}", 1e-6)
    return notes


# ------------------------------------------------------------------------------------------------ record from the prompts
def record_from_views(P, moves, acts, years, info):
    """Fallback when the replay disagrees with the prompts: each year from what the next year's prompts report (public lines,
    each company's and investor's own history line) and, for the last year, the summary. Unknown values stay None."""
    M, N = int(P["companies"]), int(P["investors"])
    nxt = {}                                                 # year -> facts of the prompts of the following year
    for e in moves:
        f = e["facts"]
        if f["year"] is not None:
            nxt.setdefault(f["year"] - 1, []).append(f)
    rows, cumX = [], 0
    for k in range(years):
        y = P["first_year"] + k
        a = acts[k] if k < len(acts) else {}
        fs = nxt.get(y, [])
        row = {"k": k, "year": y, "Kint": [None] * M, "inv_money": [None] * M, "K_end": [None] * M, "margin": [None] * M,
               "gain": [None] * M, "Q": [None] * M, "share": [(0.0, 0.0, 0.0)] * M, "spend": [(None, None, None)] * M,
               "choice": [tuple((a.get("invest") or {}).get(j, ())) for j in range(N)], "amt": [None] * N, "KI": [None] * N,
               "ret": [None] * N, "esg_part": [None] * N, "reward": [None] * N, "port_esg": [None] * N, "cap_end": [None] * N,
               "H_end": [[None] * M for _ in range(N)], "cash_end": [None] * N, "G": [None] * M, "U": [None] * N,
               "Uret": [None] * N, "Uesg": [None] * N, "cum": [[None] * 3 for _ in range(M)], "X": None, "hits": None,
               "bankrupt": [False] * M, "inv_money_end": [None] * M, "live": [True] * M}
        row["share"] = [tuple(v / 100 for v in (a.get("alloc") or {}).get(i, (0.0, 0.0, 0.0))) for i in range(M)]
        for f in fs:
            for i, c in f["co"].items():
                if i < M and c.get("capital"):
                    row["K_end"][i] = num(c["capital"])
                    row["margin"][i] = num(c["margin"]) / 100 if c.get("margin") else None
                    row["Q"][i] = num(c["esg"]) / 100 if c.get("esg") else None
                    row["bankrupt"][i] = c["bankrupt"]
            for j, c in f["inv"].items():
                if j < N and c.get("cash"):
                    row["cash_end"][j] = num(c["cash"])
                    row["H_end"][j] = [num(c["holdings"][i]) if i in c["holdings"] else 0.0 for i in range(M)]
            o = f["own"]
            for h in f["hist"]:
                if h["year"] != y:
                    continue
                if o["kind"] == "company":
                    i = o["i"]
                    row["Kint"][i], row["inv_money"][i] = num(h["Kint"]), num(h["inv_money"])
                    row["gain"][i], row["X"] = num(h["gain"]), h["X"]
                else:
                    j = o["j"]
                    row["KI"][j], row["ret"][j] = num(h["KI"]), num(h["ret"]) / 100
                    row["port_esg"][j], row["reward"][j], row["cap_end"][j] = num(h["port_esg"]) / 100, num(h["reward"]), num(h["cap_end"])
                    row["amt"][j] = num(h["each"]) if h["each"] else 0.0
            if o and o["kind"] == "company":
                row["G"][o["i"]] = num(o["gain"])
                row["cum"][o["i"]] = [num(o["cum"].get(key, "0")) for key in ("mitigation", "greenwash", "resilience")]
            elif o:
                j = o["j"]
                row["U"][j], row["Uret"][j], row["Uesg"][j] = num(o["U"]), num(o["Uret"]), num(o["Uesg"])
        if k == years - 1 and info:                          # the last year: the summary's final state
            for i, c in enumerate(info.get("companies") or []):
                if i < M:
                    row["K_end"][i] = c.get("capital", row["K_end"][i])
                    row["G"][i] = c.get("total_gain", row["G"][i])
                    row["cum"][i] = [c.get("cumulative_mitigation"), c.get("cumulative_greenwash"), c.get("cumulative_resilience")]
                    row["bankrupt"][i] = bool(c.get("bankrupt"))
            for j, c in enumerate(info.get("investors") or []):
                if j < N:
                    row["cap_end"][j], row["cash_end"][j] = c.get("capital"), c.get("cash")
                    row["U"][j], row["Uret"][j], row["Uesg"][j] = c.get("total_utility"), c.get("total_return"), c.get("total_esg")
        for i in range(M):
            if row["Kint"][i] is not None:
                row["spend"][i] = tuple(s * row["Kint"][i] for s in row["share"][i])
            row["inv_money_end"][i] = sum(row["H_end"][j][i] for j in range(N)) if all(row["H_end"][j][i] is not None
                                                                                         for j in range(N)) else None
        row["U_m"] = sum(c[0] for c in row["cum"]) if all(c[0] is not None for c in row["cum"]) else None
        se = info.get("series") if info else None
        rs = _series(se, "climate_risk", years)
        row["risk"] = rs[k] if rs and k < len(rs) else (risk_of(hazard_probs(P, k, row["U_m"])) if row["U_m"] is not None else None)
        if row["X"] is None:
            ev = _series(se, "events", years)
            if ev and k < len(ev) and all(x is not None for x in ev):
                row["X"] = int(ev[k]) if sum(int(x) for x in ev) == (info.get("climate_events") or -1) else int(ev[k]) - cumX
        cumX += row["X"] or 0
        row["wealth"] = (sum(row["K_end"]) + sum(row["cash_end"])) if None not in row["K_end"] + row["cash_end"] else None
        rows.append(row)
    while rows and (None in rows[-1]["K_end"] + rows[-1]["cap_end"] + rows[-1]["cash_end"] or rows[-1]["risk"] is None):
        warn(f"{rows[-1]['year']}: no prompt or summary reports its outcome; the record ends before it")
        rows.pop()
    return rows


# ------------------------------------------------------------------------------------------------ model calls
REPLY_CHARS = 2000                                           # the recorder keeps a rejected reply's first 2,000 characters


def _call_year(c):
    m = RX_YEAR.search(c.get("prompt") or "")
    return int(m.group(1)) if m else None


def _same_reply(answer, reply):
    return (answer or "").strip()[:REPLY_CHARS] == (reply or "").strip()[:REPLY_CHARS]


def _answers_step(c, st):
    """Whether a call's answer is the step's decision: on a fallback step the rejected reply the recorder logged, otherwise an
    answer the engine reads as the logged action."""
    if st.get("reply") is not None:
        return _same_reply(c.get("answer"), st["reply"])
    return st["parsed"] is not None and parse_action(c.get("answer"), st["role"], st["P"], st.get("live")) == st["parsed"]


def split_runs(calls):
    """The call log of a game is appended to: a recording restarted into the same output directory adds its calls after the
    earlier attempt's. Each recording process numbers its calls from 1; and within one recording the year of the prompts never
    goes back and a seat's calls for a year come in one stretch (its retries follow at once), so a seat and year seen again
    after other decisions means a new run. Any of these signs starts a new run. Returns the runs as (start, end) index ranges."""
    bounds, prev, done = [0], None, set()
    for k, c in enumerate(calls):
        y, s, n = _call_year(c), c.get("seat"), c.get("call")
        key = (s, y) if y is not None and s is not None else None
        if k:
            n0 = calls[k - 1].get("call")
            if ((n is not None and n0 is not None and n <= n0)
                    or (key and prev and (y < prev[1] or (key != prev and key in done)))):
                bounds.append(k)
                done, prev = set(), None
        if key:
            if prev and key != prev:
                done.add(prev)
            prev = key
    bounds.append(len(calls))
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def _ts(v):
    """A recording time (ISO 'YYYY-mm-ddTHH:MM:SSZ' or a run name opening with 'YYYYmmddTHHMMSSZ') as epoch seconds."""
    for fmt, s in (("%Y-%m-%dT%H:%M:%S", str(v or "")[:19]), ("%Y%m%dT%H%M%SZ", str(v or "")[:16])):
        try:
            return datetime.datetime.strptime(s, fmt).replace(tzinfo=datetime.timezone.utc).timestamp()
        except ValueError:
            pass
    return None


def pick_run(steps, calls, start_ts=None):
    """When the call log holds several recordings (see split_runs), the calls of the one this game came from: the first run that
    starts at or after the game's own start time (its run directory's stamp), else the run whose answers match the most of the
    game's decisions (the later on a tie). Returns (calls, note for the page or None)."""
    runs = split_runs(calls)
    if len(runs) <= 1:
        return calls, None

    def score(a, b):
        idx = {}
        for k in range(a, b):
            idx.setdefault((calls[k].get("seat"), _call_year(calls[k])), []).append(k)
        return sum(1 for st in steps if any(_answers_step(calls[k], st) for k in idx.get((st["seat"], st["year"]), ())))
    scores = [score(a, b) for a, b in runs]
    best = max(range(len(runs)), key=lambda i: (scores[i], i))
    by_time = None
    if start_ts is not None:
        t0 = [next((calls[k]["t_start"] for k in range(a, b) if calls[k].get("t_start") is not None), None) for a, b in runs]
        by_time = next((i for i, t in enumerate(t0) if t is not None and t >= start_ts - 1), None)
    use = by_time if by_time is not None else best
    if scores[use] < scores[best]:
        warn(f"call log run {use + 1} starts with this game but answers {scores[use]} of its decisions, run {best + 1} "
             f"answers {scores[best]}")
    a, b = runs[use]
    desc = ", ".join(f"calls {x + 1} to {y}" for x, y in runs)
    warn(f"the call log holds {len(runs)} recording runs ({desc}); run {use + 1} is this game's "
         + ("(it starts with the game's run)" if use == by_time else "(its answers match the most decisions)")
         + f"; {len(calls) - (b - a)} calls of the other runs are left out")
    note = (f"The call log holds {len(runs)} recording runs of this game (the recording was restarted into the same output "
            f"directory); the reasoning comes from the run this game was recorded in (calls {a + 1} to {b} of {len(calls)}). ")
    return calls[a:b], note


def attach_calls(steps, calls):
    """Tie calls to steps in time order. A step's window is the run of consecutive calls, from the pointer, of the same seat and
    year (both read from the call's prompt; unknown values agree with anything). The step's call is the first in the window
    whose answer the engine reads (strictly, as base.parse_action does) as the logged action; on a fallback step, the last
    whose answer is the rejected reply the recorder logged. Earlier calls in the window were failed attempts whose reasoning is
    kept above it. Without a match the window's last call is used and the step says that the game applied another action.
    A call whose reasoning hit the output limit (finish 'length') ends with a visible cut mark; when its action came from a
    follow-up question (rescued), a note saying so leads its reasoning. Returns (tied, rescued, limits)."""
    ptr, tied, rescued, limits = 0, 0, 0, []
    year_of = _call_year

    def fits(k, st, strict=False):
        seat, y = calls[k].get("seat"), year_of(calls[k])
        if strict:
            return seat == st["seat"] and y == st["year"]
        return seat in (None, st["seat"]) and y in (None, st["year"])
    for st in steps:
        if ptr < len(calls) and not fits(ptr, st):
            ahead = next((k for k in range(ptr, min(len(calls), ptr + 40)) if fits(k, st, strict=True)), None)
            if ahead is None:
                warn(f"no call for P{st['seat']} {st['year']}")
                continue
            warn(f"calls {ptr}..{ahead - 1} match no decision; skipped")
            ptr = ahead
        j = ptr
        while j < len(calls) and fits(j, st):
            j += 1
        window = list(range(ptr, j))
        if not window:
            continue
        fell = st.get("reply") is not None                   # a fallback: every attempt was rejected, the last one is logged
        order = reversed(window) if fell else window
        hit = next((k for k in order if _answers_step(calls[k], st)), None)
        use = hit if hit is not None else window[-1]
        parts = [f"[Earlier attempt, answer: {calls[k]['answer'][:300]}]\n{calls[k].get('reasoning') or '(no reasoning returned)'}"
                 for k in range(ptr, use)]
        c = calls[use]
        finish, toks = c.get("finish"), c.get("out_tokens")
        cut = finish == "length" if finish else bool(c.get("rescued"))
        lim = f"the {toks:,}-token output limit" if cut and finish == "length" and toks else "the token limit"
        text = (c.get("reasoning") or "").rstrip()
        if text and cut:
            text += f" ... [cut off here at {lim}]"
        if c.get("rescued"):
            note = (f"[This reasoning was cut off at {lim} before it reached an answer; the action is the model's answer to a "
                    "follow-up question.]" if cut else
                    "[This reasoning ended without a usable answer; the action is the model's answer to a follow-up question.]")
            text = note + ("\n\n" + text if text else "")
            rescued += 1
            limits.append(toks if cut and finish == "length" and toks else None)
        parts.append(("[Final attempt]\n" if parts else "") + (text or ("(no reasoning returned)" if parts else "")))
        if fell:
            if hit is None:
                warn(f"P{st['seat']} {st['year']}: no call's answer is the rejected reply the recorder logged")
                parts.append(f"[The recorder logged this rejected answer: {st['reply'][:300]}]")
            parts.append(f"[{st['why']}. The game rejected it and applied the fallback: {st['fallback_txt']}.]")
        elif hit is None:
            parts.append(f"[The game applied another action than this answer: {c['answer'][:300]}]")
            warn(f"P{st['seat']} {st['year']}: no call answered the logged action {st['act']}")
        st["thinking"] = "\n\n".join(p for p in parts if p).strip()
        st["rescued"] = bool(c.get("rescued"))
        if c.get("prompt"):
            st["state"] = en(common.state_part(c["prompt"]))
            st["from_prompt"] = True
        ptr = (window[-1] if fell else use) + 1
        tied += 1
    if ptr < len(calls):
        warn(f"{len(calls) - ptr} calls after the last decision were not tied to any step")
    return tied, rescued, limits


def fit_thinking(steps, budget, full):
    """Keep the page within the size its host takes: when all reasoning together is longer than `budget` characters, shorten the
    longest ones (one common cap, the largest that fits) by removing their middle, with a visible mark. Always cuts from each
    step's full reasoning (`full`, one text per step), so repeated calls do not cut a cut text. Returns the number of shortened
    steps and the characters left out."""
    lens = [len(t) for t in full]
    if sum(lens) <= budget:
        for s, t in zip(steps, full):
            s["thinking"] = t
        return 0, 0
    lo, hi = 2000, max(lens)
    while lo < hi:                                           # the largest cap with sum(min(len, cap)) <= budget
        mid = (lo + hi + 1) // 2
        if sum(min(x, mid) for x in lens) <= budget:
            lo = mid
        else:
            hi = mid - 1
    cap, n, removed = lo, 0, 0
    for s, x, t in zip(steps, lens, full):
        if x <= cap:
            s["thinking"] = t
            continue
        head, tail = int(cap * 0.45), int(cap * 0.55)
        gone = x - head - tail
        s["thinking"] = (t[:head].rstrip() + f"\n\n[... {gone:,} characters of this reasoning are left out here to keep the page "
                         "within its size limit; the beginning and the end are shown ...]\n\n" + t[-tail:].lstrip())
        n += 1
        removed += gone
    return n, removed


# ------------------------------------------------------------------------------------------------ build
def shown_top(vals, d):
    """The indices whose value, printed with d decimals as the prompts print it, is the largest (None values skipped)."""
    r = [None if v is None else round(v, d) for v in vals]
    live = [x for x in r if x is not None]
    if not live:
        return None
    best = max(live)
    return tuple(i for i, x in enumerate(r) if x is not None and x >= best)


# where an investor put its money in a year, against what it saw (last year's ESG scores and margins): (color, opacity, legend)
CATS = ("esg", "esg_in", "margin", "margin_in", "pick", "all", "cash")
CAT_STYLE = {"esg": ("c3", 0.9, "Exactly last year's top ESG scorer(s)"),
             "esg_in": ("c3", 0.55, "Some, not all, of last year's top ESG scorers (faded green)"),
             "margin": ("c4", 0.9, "Exactly last year's best margin(s)"),
             "margin_in": ("c4", 0.55, "Some, not all, of last year's best margins (faded yellow)"),
             "pick": ("ink", 0.9, "Another selection"), "all": ("hold", 0.9, "Every company"), "cash": ("muted", 0.9, "Cash only")}


def cat_note(A, k, ct, S):
    """A feed note on an investor's selection in year k: how it relates to last year's top ESG scorers and best margins."""
    te, tm = A["top_esg"][k], A["top_margin"][k]
    if ct == "esg":
        return (" That is exactly the compan" + ("ies" if len(S) > 1 else "y") + " with last year's top ESG score"
                + (f" ({A['top_esg_v'][k]:.3f})." if A["top_esg_v"][k] is not None else "."))
    if ct == "margin":
        return (" That is exactly the compan" + ("ies" if len(S) > 1 else "y") + " with last year's best margin"
                + (f" ({sg(A['top_margin_v'][k])}%)." if A["top_margin_v"][k] is not None else "."))
    if ct == "esg_in":
        return (f" Only companies with last year's top ESG score ({A['top_esg_v'][k]:.3f}, shared by "
                f"{companies_txt(te)}), not all of them.")
    if ct == "margin_in":
        return (f" Only companies with last year's best margin ({sg(A['top_margin_v'][k])}%, shared by "
                f"{companies_txt(tm)}), not all of them.")
    return ""


def amount(v):
    """A spending amount in trillions: 2 decimals, or 4 when it is tiny (a token mitigation of 0.001% is 0.0002)."""
    if v is None:
        return "?"
    return f"{v:.4f}" if 0 < abs(v) < 0.005 else tn(v)


def share_txt(v):
    """A share of a total (fraction) in whole percent; a positive share under 0.5% is 'under 1%'."""
    return "0%" if v <= 0 else "under 1%" if v < 0.005 else f"{100 * v:.0f}%"


def companies_txt(ids, cap=True):
    """'Company 2' or 'Companies 1 and 4' (names, so always capitalized; `cap` is kept for call compatibility)."""
    ids = list(ids)
    return ("Company " + str(ids[0])) if len(ids) == 1 else "Companies " + and_list([str(i) for i in ids])


def build(src, calls=None):
    b = load_source(src)
    if isinstance(calls, str):
        calls = load_calls(calls)
    calls = calls or []
    run = b.get("run") or {}
    test_build = bool(run.get("test_bundle_from_calls"))
    events, summary = b["events"], b.get("summary") or {}
    info = summary.get("info") or {}
    moves = [e for e in events if e.get("event") == "move"]
    if not moves:
        raise SystemExit("no move events in " + str(src))
    for e in moves:
        e["facts"] = view_facts(e.get("view") or "")
    P = read_params(b, [e.get("view") or "" for e in moves])
    M, N, T, Y0 = int(P["companies"]), int(P["investors"]), int(P["years"]), int(P["first_year"])
    cap_pct = 100 * P["action_cap"]
    alphas = P["alphas"]

    # ---- who is who: companies are seats 0..M-1, investors M..M+N-1 (rules: Players and Roles); the prompts confirm it
    role_of, ent_of = {}, {}
    for s in range(M):
        role_of[s], ent_of[s] = "COMPANY", s
    for j in range(N):
        role_of[M + j], ent_of[M + j] = "INVESTOR", j
    for e in moves:
        f, s = e["facts"], e.get("actor")
        o = f["own"]
        if f["role"] and role_of.get(s) != f["role"]:
            warn(f"P{s}: role {f['role']} in its prompt, {role_of.get(s)} by seat order")
            role_of[s] = f["role"]
        if o:
            ent = o["i"] if o["kind"] == "company" else o["j"]
            if ent_of.get(s) != ent:
                warn(f"P{s}: entity {ent} in its prompt, {ent_of.get(s)} by seat order")
                ent_of[s] = ent
    co_seat = {ent_of[s]: s for s in role_of if role_of[s] == "COMPANY"}
    inv_seat = {ent_of[s]: s for s in role_of if role_of[s] == "INVESTOR"}
    # short role names: the page draws them as lane and row labels in an 86-pixel margin
    seat_names = [f"Company {ent_of[s]}" if role_of.get(s) == "COMPANY" else f"Investor {ent_of[s]}" for s in range(M + N)]

    # ---- the moves: one per seat per year, read as the engine reads them
    acts, move_year = {}, []
    last_seat, year = None, Y0 - 1
    for e in moves:
        s, f = e["actor"], e["facts"]
        y = f["year"]
        if y is None:                                       # by seat order: a year ends when the seat number falls
            y = year + 1 if last_seat is None or s <= last_seat else year
        year, last_seat = y, s
        move_year.append(y)
        k = y - Y0
        a = acts.setdefault(k, {"alloc": {}, "invest": {}, "illegal": set(), "seats": set()})
        live = [not f["co"].get(i, {}).get("bankrupt", False) for i in range(M)]
        pa = parse_action(e.get("move"), role_of[s], P, live)
        e["parsed"], e["live"] = pa, live
        # an illegal answer: the recorder logs the fallback the game applied as the move, with fallback: true and the rejected
        # reply; a stand-in test bundle logs the raw answer itself as the move, which the engine would reject
        fb = bool(e.get("fallback"))
        e["illegal"] = fb or pa is None
        if e["illegal"]:
            reply = e.get("reply") if fb else e.get("move")
            e["reply_txt"] = reply if isinstance(reply, str) else "" if reply is None else json.dumps(reply)
            why = why_illegal(e["reply_txt"], role_of[s], P, live)
            if why is None:
                warn(f"P{s} {y}: the move is flagged as a fallback but its reply {_short(e['reply_txt'], 80)} reads as legal")
                why = ("not accepted", "The game did not accept the answer")
            e["why"] = why
            a["illegal"].add(s)
        a["seats"].add(s)
        if pa is None:
            pass
        elif pa[0] == "allocate":
            a["alloc"][ent_of[s]] = pa[1]
        else:
            a["invest"][ent_of[s]] = pa[1]
    n_years_seen = max(acts) + 1
    played = 0                                               # years with every seat's decision, from the first on
    while played in acts and len(acts[played]["seats"]) >= M + N:
        played += 1
    over = next((e for e in events if e.get("event") == "game_over"), None)
    cut_short = played < T
    raw_rw = summary.get("rewards") or (run.get("result") or {}).get("rewards") or (over or {}).get("rewards") or {}
    rewards = {int(k): float(v) for k, v in raw_rw.items()}
    acts_list = [acts.get(k, {}) for k in range(played)]
    n_illegal = sum(1 for e in moves if e["illegal"])
    n_flagged = sum(1 for e in moves if e.get("fallback"))
    if isinstance(summary.get("fallbacks"), int) and summary["fallbacks"] != n_flagged:
        warn(f"the summary counts {summary['fallbacks']} fallbacks, the log flags {n_flagged} moves")
    if n_illegal != n_flagged and not test_build:
        warn(f"{n_illegal - n_flagged} logged moves are not legal actions but carry no fallback flag")

    # ---- the year record: the replay, checked; or the prompts
    unsupported = replay_supported(P)
    st_probs = self_test(P)
    for p in st_probs:
        warn("self-test: " + p)
    tally, tsum = Tally(), Tally()                           # the prompts' numbers; the summary's values and the rewards
    rec, how = None, "replay"
    if not unsupported and not st_probs:
        rec = simulate(P, acts_list, played)
        check_views(rec, moves, P, tally)
        if tally.bad:
            warn(f"{tally.bad} of {tally.n} numbers in the prompts disagree with the replay; rebuilding the record from the prompts")
        else:                                                # the prompts are what happened; the summary is compared, not trusted
            check_summary(rec, info, rewards, P, tsum, co_seat, inv_seat)
            if tsum.bad:
                warn(f"{tsum.bad} of {tsum.n} summary values disagree with the replay (kept: it matches all {tally.n} prompt numbers)")
    if rec is None or tally.bad:
        if unsupported:
            warn("the replay does not cover " + ", ".join(unsupported) + "; the record comes from the prompts")
        rec, how = record_from_views(P, moves, acts_list, played, info), "views"
    n = len(rec)
    xn = max(T, n_years_seen)
    labels = [str(Y0 + k) for k in range(xn)]

    # ---- reference runs on the same draws (the replay's own transition)
    ref = {}
    if how == "replay":
        ref["none"] = simulate(P, policy_acts(P, T, 0.0), T)
        ref["coop"] = simulate(P, policy_acts(P, T, 0.5), T)
        ref["nomit_same"] = simulate(P, [{"alloc": {}, "invest": acts_list[k]["invest"]} for k in range(n)], n)
    no_mit_risk = [risk_of(hazard_probs(P, k, 0.0)) for k in range(xn)]

    A = analyse(P, rec, ref, no_mit_risk, rewards, cut_short)

    # ---- kinds, steps and the feed
    def co_kind(m):
        return "none" if m == 0 else "token" if m < 0.5 else "cap" if m >= cap_pct - 1e-9 else "some"
    kinds_all = {"none": ["Company: no mitigation", "c2"], "token": ["Company: a token, under 0.5%", "c4"],
                 "some": [f"Company: 0.5% to under {mp(cap_pct)}", "c1"], "cap": [f"Company: the {mp(cap_pct)} cap", "c3"],
                 "all": ["Investor: every company", "hold"], "pick": ["Investor: a selection", "ink"],
                 "cash": ["Investor: cash only", "muted"]}
    steps, feed = [], []
    feed.append({"x": 0, "step": 0, "who": "Rules", "text": (
        f"{nw(M).capitalize()} companies with {tn(P['company_capital'])} trillion each and {nw(N)} investors with "
        f"{tn(P['investor_capital'])} trillion each (" + ", ".join(f"Investor {j} {ALPHA} {alphas[j]:g}" for j in range(N))
        + f"), {T} years from {Y0}. Capital left after spending grows {pc(P['market_growth'], 0)} a year; each disaster takes "
        f"{pc(P['damage'], 0)} of every company's capital. Disaster probabilities start at "
        + " / ".join(pc(p, 0) for p in P["hazard_p0"]) + f" ({', '.join(HAZARDS)}) and rise every year; all companies' "
        f"mitigation so far slows the rise. Spending cap {mp(cap_pct)} of capital a year per category. All decisions of a year "
        "are made at the same time, so investors choose on last year's ESG scores and margins."
        + ("" if P["esg_disclosure"] else " No ESG disclosure: scores are not shown."))})
    by_k = {r["k"]: r for r in rec}
    prev_m = {}
    for mi, e in enumerate(moves):
        s, f = e["actor"], e["facts"]
        y = move_year[mi]
        k = y - Y0
        role, ent, pa = role_of[s], ent_of[s], e["parsed"]
        r = by_k.get(k)
        illegal = e["illegal"]
        fb_txt = "no spending" if role == "COMPANY" else "all cash"
        if role == "COMPANY":
            m, g, rr = pa[1] if pa else (0.0, 0.0, 0.0)
            kind = co_kind(m)
            extra = "".join(f", {lab} {mp(v)}" for lab, v in (("greenwashing", g), ("resilience", rr)) if v)
            spend = m / 100 * r["Kint"][ent] if r and r["Kint"][ent] is not None else None
            was = prev_m.get(ent)
            if illegal:
                head = f"{y} {DOT} illegal answer ({e['why'][0]}) {DOT} fallback: " + ("no mitigation" if m == 0 else f"{mp(m)}")
            elif m == 0:
                head = f"{y} {DOT} no mitigation" + extra
            else:
                head = (f"{y} {DOT} mitigation {mp(m)}" + (" (the cap)" if kind == "cap" else "")
                        + (f", {amount(spend)} tn" if spend is not None else "") + extra)
            if was is not None and abs(was - m) > 1e-9:
                head += f" {DOT} was {mp(was)}"
            prev_m[ent] = m
            txt = f"No mitigation{extra}." if m == 0 else f"{mp(m)} of its capital on mitigation{extra}"
            if m and spend is not None:
                txt += (f": {amount(spend)} of the {tn(r['Kint'][ent])} trillion it manages this year"
                        + (f" ({tn(r['inv_money'][ent])} of it investors' money)." if r["inv_money"][ent] else "."))
            elif m:
                txt += "."
            c = f["co"].get(ent, {})
            if k > 0 and c.get("esg") is not None:
                txt += f" Its score last year: ESG {c['esg']}, margin {c.get('margin', '?').replace('-', MINUS)}%."
            if r is not None and k > 0:
                esg_in = [j for j in range(N) if alphas[j] > 0 and ent in r["choice"][j]]
                if esg_in and m < 0.5 and A["mit"][ent][k - 1] >= 0.5:
                    txt += (f" The ESG investor{'s' if len(esg_in) > 1 else ''} ({and_list([str(j) for j in esg_in])}) "
                            f"{'hold' if len(esg_in) > 1 else 'holds'} it this year on that score.")
        else:
            S = pa[1] if pa else ()
            n_live = sum(1 for i in range(M) if not f["co"].get(i, {}).get("bankrupt"))
            kind = "cash" if not S else "all" if len(S) == n_live else "pick"
            if illegal:
                head = f"{y} {DOT} illegal answer ({e['why'][0]}) {DOT} fallback: " + ("all cash" if not S else companies_txt(S))
            elif not S:
                head = f"{y} {DOT} keeps all in cash"
            elif kind == "all":
                head = f"{y} {DOT} every company ({nw(len(S))})"
            else:
                head = f"{y} {DOT} " + (f"Company {S[0]} only" if len(S) == 1 else companies_txt(S))
            amt = r["amt"][ent] if r else None
            if not S:
                txt = "Keeps everything in cash, which earns nothing."
            else:
                txt = (("Every company" if kind == "all" else companies_txt(S))
                       + (f": {tn(amt)} trillion each" if amt and len(S) > 1 else f": {tn(amt)} trillion" if amt else "") + ".")
            if r is not None and 0 < k < n:
                txt += cat_note(A, k, A["cats"][ent][k], S)
            if r is not None and r["ret"][ent] is not None and S:
                txt += f" Return this year {sg(100 * r['ret'][ent])}%, reward {sg(r['reward'][ent], 4)}."
        if illegal:
            # the rejected answer and why, in place of the first sentence that describes the applied fallback
            for first in ("No mitigation.", "Keeps everything in cash, which earns nothing."):
                if txt.startswith(first):
                    txt = txt[len(first):].strip()
                    break
            txt = (f"{e['why'][1]}; the game rejected it and applied the fallback: {fb_txt}"
                   + (", which earns nothing." if fb_txt == "all cash" else ".") + (" " + txt if txt else ""))
        act = e.get("move") or ""
        if illegal and pa is None:                           # a test build logs the raw answer: show the applied fallback,
            act = json.dumps(fallback_action(role, P), sort_keys=True)   # as the recorder logs it
        st = {"seat": s, "x": k, "head": head, "kind": kind, "act": en(act),
              "state": en(common.state_part(e.get("view") or "")) or "Not recorded.", "year": y, "role": role, "P": P,
              "parsed": pa, "live": e["live"]}
        if illegal:
            st.update(reply=e["reply_txt"], why=e["why"][1], fallback_txt=fb_txt)
        steps.append(st)
        feed.append({"x": k, "step": len(steps) - 1, "who": f"Company {ent}" if role == "COMPANY" else f"Investor {ent}",
                     "kind": kind, "text": txt})
        if r and (mi + 1 == len(moves) or move_year[mi + 1] != y):  # the year's result, after its last decision
            if r["X"] is None:
                dtxt = "Disasters not reported."
            elif r["X"] == 0:
                dtxt = "No disaster."
            else:
                which = [HAZARDS[h] for h in range(3) if r["hits"] and r["hits"][h]]
                dtxt = ((f"{cap1(and_list(which))}: " if which else "") + f"{nw(r['X'])} disaster{'s' if r['X'] > 1 else ''}, "
                        f"each taking {pc(P['damage'], 0)} of every company's capital.")
            txt = dtxt
            if r["risk"] is not None:
                txt += f" Climate risk {pc(r['risk'])} (with no mitigation ever: {pc(no_mit_risk[k])})."
            mg = ", ".join(f"C{i} {sg(100 * r['margin'][i])}%" for i in range(M) if r["margin"][i] is not None)
            ke = ", ".join(f"C{i} {tn(r['K_end'][i])}" for i in range(M) if r["K_end"][i] is not None)
            txt += (f" Margins: {mg}." if mg else "") + (f" Capital at the end of {y}: {ke}." if ke else "")
            if r.get("wealth") is not None:
                txt += f" Market wealth {tn(r['wealth'])} trillion."
            feed.append({"x": k, "step": len(steps) - 1, "who": "Climate", "text": txt})
    kinds = {k: v for k, v in kinds_all.items() if any(st["kind"] == k for st in steps)}

    meta_ = b.get("meta") or {}
    start_ts = next((t for t in (_ts(meta_.get("source_run")), _ts(run.get("source_run")), _ts(meta_.get("recorded_at")),
                                 _ts(run.get("recorded_at"))) if t is not None), None)
    calls, run_note = pick_run(steps, calls, start_ts)
    tied, n_rescued, limits = attach_calls(steps, calls)
    has_reason = any(st.get("thinking") for st in steps)
    think_chars = sum(len(st.get("thinking") or "") for st in steps)
    from_prompt = sum(1 for st in steps if st.get("from_prompt"))
    for st in steps:
        st["thinking"] = en(st.get("thinking") or "")
        for key in ("year", "role", "P", "parsed", "from_prompt", "rescued", "live", "reply", "why", "fallback_txt"):
            st.pop(key, None)

    # ---- charts
    def pad(v):
        return list(v) + [None] * (xn - len(v))
    mit, gw, rs_ = A["mit"], A["gw"], A["rs"]
    rows_c, cells_c = [], []
    for i in range(M):                                       # 1. mitigation, company by company
        rows_c.append(f"Company {i}")
        row = []
        for k in range(xn):
            if k >= n:
                row.append(None)
                continue
            r = rec[k]
            m, g, rr = mit[i][k], gw[i][k], rs_[i][k]
            kd = co_kind(m)
            o = 0.92 if kd in ("cap", "none") else 0.5 + 0.4 * min(1.0, m / cap_pct) if kd == "some" else 0.8
            extra = "".join(f", {lab} {mp(v)}" for lab, v in (("greenwashing", g), ("resilience", rr)) if v)
            tip = (f"{mp(m)} on mitigation{extra}"
                   + (f" = {amount(m / 100 * r['Kint'][i])} of {tn(r['Kint'][i])} tn" if r["Kint"][i] is not None and m else "")
                   + (f"; margin {sg(100 * r['margin'][i])}%" if r["margin"][i] is not None else "")
                   + (f"; capital at end {tn(r['K_end'][i])}" if r["K_end"][i] is not None else ""))
            row.append({"c": kinds_all[kd][1], "o": round(o, 3), "t": f"{m:.3f}".rstrip("0").rstrip("."), "tip": tip})
        cells_c.append(row)
    leg_c = [[cap1(v[0].replace("Company: ", "")), v[1]] for kk, v in kinds_all.items() if kk in ("none", "token", "some", "cap")
             and any(co_kind(mit[i][k]) == kk for i in range(M) for k in range(n))]
    rows_i, cells_i = [], []
    for j in range(N):                                       # 2. where the investors put their money
        rows_i.append(f"Investor {j}")
        row = []
        for k in range(xn):
            if k >= n:
                row.append(None)
                continue
            r = rec[k]
            S, ct = r["choice"][j], A["cats"][j][k]
            t = "cash" if not S else "all" if ct == "all" else ",".join(str(i) for i in S)
            tip = (f"{ALPHA} {alphas[j]:g}: " + ("cash" if not S else ("every company" if ct == "all" else companies_txt(S)))
                   + (f", {tn(r['amt'][j])} tn each" if S and r["amt"][j] else ""))
            if r["ret"][j] is not None and S:
                tip += f"; return {sg(100 * r['ret'][j])}%, reward {sg(r['reward'][j], 4)}"
            if k:
                note = cat_note(A, k, ct, S).strip()
                tip += (". " + note[:-1]) if note else ""
            col, op, _ = CAT_STYLE[ct]
            row.append({"c": col, "o": op, "t": t, "tip": tip})
        cells_i.append(row)
    leg_i = [[lab, col] for key, (col, _, lab) in CAT_STYLE.items()
             if any(A["cats"][j][k] == key for j in range(N) for k in range(n))]
    co_cols = ["c1", "c2", "c3", "c4", "ink", "hold", "muted"]
    # capital compounds about a thousandfold over a century, so a linear axis in trillions would flatten most years into the
    # baseline: the chart shows each company's share of all companies' capital (the trillions are in the feed and the tips)
    tot_k = [sum(r["K_end"]) if None not in r["K_end"] else None for r in rec]
    cap_lines = [{"name": f"Company {i}", "color": co_cols[i % len(co_cols)],
                  "v": pad([round(100 * r["K_end"][i] / tk, 4) if tk else None for r, tk in zip(rec, tot_k)])} for i in range(M)]
    cap_lines.append({"name": f"Equal shares ({100 / M:g}%)", "color": "muted", "dash": True, "thin": True, "v": [100 / M] * xn})
    risk_lines = [{"name": "This game", "color": "c2", "v": pad([None if r["risk"] is None else round(100 * r["risk"], 4)
                                                                 for r in rec]), "area": True},
                  {"name": "Nobody mitigates", "color": "hold", "dash": True, "thin": True,
                   "v": [round(100 * v, 4) for v in no_mit_risk]}]
    if ref.get("coop"):
        risk_lines.append({"name": "Every company 0.5% a year", "color": "c3", "dash": True, "thin": True,
                           "v": pad([round(100 * r["risk"], 4) for r in ref["coop"]])})
    band = {"v": [(-1 if rec[k]["X"] else 1) if k < n and rec[k]["X"] is not None else 0 for k in range(xn)],
            "pos": "No disaster that year", "neg": "One or more disasters"}

    rw = {"co": [rewards.get(co_seat.get(i)) for i in range(M)], "inv": [rewards.get(inv_seat.get(j)) for j in range(N)]}
    S_ = story(P, A, rec, ref, no_mit_risk, rw, cut_short, n_illegal)
    charts = [
        {"title": "Mitigation, company by company", "sub": S_["sub_mit"],
         "grid": {"rows": rows_c, "cells": cells_c, "legend": leg_c, "rowHeight": 24}},
        {"title": "Where the investors put their money", "sub": S_["sub_inv"],
         "grid": {"rows": rows_i, "cells": cells_i, "legend": leg_i, "rowHeight": 24}},
        {"title": "Capital: each company's share of all companies' capital", "sub": S_["sub_cap"], "height": 250,
         "fmt": "pct0", "lines": cap_lines},
        {"title": "Climate risk: the chance of at least one disaster in the year", "sub": S_["sub_risk"], "height": 240,
         "fmt": "pct0", "lines": risk_lines, "band": band},
    ]

    # ---- foot
    model_id = recorded_model(b, calls)
    date = recorded_date(b, calls)
    seed = run.get("seed", (b.get("config") or {}).get("seed"))
    variant = os.path.splitext(os.path.basename((b.get("config") or {}).get("variant") or ""))[0]
    seats_txt = (", ".join(f"Company {i} is P{co_seat[i]}" for i in sorted(co_seat)) + ", "
                 + ", ".join(f"Investor {j} is P{inv_seat[j]}" for j in sorted(inv_seat)))
    cfg = ", ".join(x for x in [f"config {variant}" if variant else "", f"seed {seed}" if seed is not None else "",
                                f"{M} companies and {N} investors ({ALPHA} = {', '.join(f'{a:g}' for a in alphas)})",
                                f"{T} years", "exact investment split" if P["investment_rounding"] == "exact" else
                                "whole-trillion investment split", "the official code's fixed disaster draws"
                                if P["disaster_draws"] == "fixed" else "disaster draws from the episode seed"] if x)
    if test_build:
        data_txt = (f"TEST BUILD, not a finished recording: this page was assembled from the call log of the {date} recording of "
                    f"{GID} while it was still running ({len(steps)} decisions, years {Y0} to {Y0 + n_years_seen - 1}), as a "
                    f"stand-in for its demo bundle; there is no summary yet. ")
    else:
        data_txt = f"Data: the {date} recording of {GID} ({cfg}), every seat played by {model_id or 'the model'}. "
    data_txt += f"Seats: {seats_txt}. "
    if cut_short:
        data_txt += (f"The run stops after {played} of {T} years" + (" (no game_over event)" if over is None else "")
                     + "; the charts end there. ")
    if n_illegal:
        data_txt += (f"{n_illegal} answer{'s were' if n_illegal > 1 else ' was'} not a legal action"
                     + ("" if test_build else " even after the retries") + "; the game applied the fallback (no spending, or "
                     "all cash), marked on "
                     + ("that step" if n_illegal == 1 else "those steps") + " with the rejected answer. ")
    else:
        data_txt += "Every decision was a legal action (no fallback was applied). "
    if how == "replay":
        shown, summ = tally.n, tsum.n
        chk = ("Every number on the page comes from replaying the logged actions through the rules' transition (equations 1 to "
               "11) with the official code's fixed disaster draws (NumPy default_rng(100t), (500t) and (1000t), re-implemented); "
               "the replay reproduces the rules' fixed-policy table (nobody mitigates: climate risk 1.037, 116 disasters, market "
               "238 trillion; everyone at 0.5%: 0.512, 60, 10,321). "
               + (f"Checks: all {shown:,} numbers the seats were shown in their prompts (capital, margins, ESG scores, cash, "
                  "holdings, own totals and history lines) match the replay" if shown else
                  "The log holds no prompts to check the replay against"))
        if summ and not tsum.bad:
            chk += f", and so do all {summ:,} values of the summary (metrics, per-seat totals, yearly series and rewards). "
        elif summ:
            bad_g = [g_ for g_, v in tsum.groups.items() if v[1]]
            chk += (f"; of the summary's {summ:,} values {tsum.bad} disagree with it ({', '.join(bad_g)}; listed in the build "
                    "log), the rest agree. ")
        else:
            chk += ". "
    else:
        chk = ("The record comes from the prompts: each year from what the next year's prompts report (public lines and each "
               "seat's own history line) and the last year from the summary; the replay of the rules' transition was not used "
               + ("(it does not cover " + ", ".join(unsupported) + "). " if unsupported else
                  f"({tally.bad} of {tally.n} compared numbers disagreed with it; see the build log). "))
    if has_reason:
        rs_txt = (f"Reasoning was saved for {tied} of the {len(steps)} decisions ({think_chars:,} characters) and is shown with "
                  "each step; " + ("the observations are the prompts of those calls. " if from_prompt == len(steps) else
                                   "the observations are the logged prompts. "))
        if n_rescued:
            rs_txt += (f"In {n_rescued} decision{'s' if n_rescued > 1 else ''} the reasoning "
                       + (f"used up the {limits[0]:,}-token output limit" if limits and limits[0] and len(set(limits)) == 1
                          else "ran out")
                       + " before giving an answer; there the action is the model's answer to a follow-up question (marked on "
                       "those steps). ")
        rs_txt += run_note or ""
    else:
        rs_txt = "No model-call log was given, so reasoning is not shown. "
    foot_tail = (S_["foot_defs"] + " Source: Hou, Yuan, Leibo and Jaques, InvestESG: a multi-agent reinforcement learning "
                 "benchmark for studying climate investment as a social dilemma (ICLR 2025).")

    d = {"id": GID, "name": "InvestESG", "title": ("[Test build, partial log] " if test_build else "") + S_["title"],
         "lede": S_["lede"], "seatNames": seat_names, "tiles": S_["tiles"], "foot": "",
         "x": {"n": xn, "labels": labels}, "charts": charts, "kinds": kinds, "steps": steps, "feed": feed,
         "feedTitle": "Year by year",
         "feedSub": "Each decision as the seat made it (companies first, then investors, all from the same snapshot), then the "
                    "year's result: disasters, climate risk, margins and capital. The selected step is highlighted; later "
                    "entries are hidden."}
    mtxt = model_id or "model not recorded"
    d["eyebrow"] = (f"social_game_env {DOT} {GID} {DOT} {mtxt}" + (", reasoning on" if has_reason else "")
                    + f" {DOT} recorded {date}" + (f" {DOT} test build from a partial call log" if test_build else ""))
    # the page is one JSON file the host takes up to 16 MB: shorten the longest reasonings, visibly, if it would be larger
    n_short, gone = 0, 0
    full = [st["thinking"] for st in steps]
    for _ in range(8):
        d["foot"] = en(data_txt + chk + rs_txt + (
            f"To keep the page within its size limit, the {n_short} longest reasonings are shortened in the middle "
            f"({gone:,} characters left out, marked where they were cut; the call log has them whole). " if n_short else "")
            + foot_tail)
        size = len(json.dumps(d, ensure_ascii=False).encode("utf-8"))
        if size <= MAX_BYTES:
            break
        total = sum(len(st["thinking"]) for st in steps)
        n_short, gone = fit_thinking(steps, max(100_000, total - int((size - MAX_BYTES) * 1.02) - 20_000), full)
    else:
        warn(f"the page is still {size:,} bytes after shortening the reasoning")
    return d


# ------------------------------------------------------------------------------------------------ analysis
def analyse(P, rec, ref, no_mit_risk, rewards, cut_short):
    """The facts the charts and the text use, computed once from the year record."""
    M, N = int(P["companies"]), int(P["investors"])
    n = len(rec)
    cap_pct = 100 * P["action_cap"]
    alphas = P["alphas"]
    A = {"n": n}
    A["mit"] = [[r["share"][i][0] * 100 for r in rec] for i in range(M)]
    A["gw"] = [[r["share"][i][1] * 100 for r in rec] for i in range(M)]
    A["rs"] = [[r["share"][i][2] * 100 for r in rec] for i in range(M)]
    mit = A["mit"]
    A["mean_m"] = [sum(v) / n if n else 0.0 for v in mit]
    A["yrs_cap"] = [sum(1 for v in mit[i] if v >= cap_pct - 1e-9) for i in range(M)]
    A["yrs_zero"] = [sum(1 for v in mit[i] if v == 0) for i in range(M)]
    A["yrs_token"] = [sum(1 for v in mit[i] if 0 < v < 0.5) for i in range(M)]
    last = rec[-1] if rec else None
    A["tot_mit_i"] = [(last["cum"][i][0] if last and last["cum"][i][0] is not None else None) for i in range(M)]
    A["tot_mit"] = sum(A["tot_mit_i"]) if None not in A["tot_mit_i"] else None
    A["K_fin"] = [last["K_end"][i] for i in range(M)] if last else [None] * M
    A["G_fin"] = [last["G"][i] for i in range(M)] if last else [None] * M
    A["events"] = sum(r["X"] or 0 for r in rec)
    A["dis_years"] = [r["year"] for r in rec if r["X"]]
    # what the investors saw: last year's ESG scores and margins as printed (3 and 2 decimals). Ties are common (several
    # companies at the cap share the top score; every company that spends the same share earns the same margin), so a top set
    # often holds several companies.
    esg_seen = [None] + [[None if rec[k - 1]["bankrupt"][i] or rec[k - 1]["Q"][i] is None else 100 * rec[k - 1]["Q"][i]
                          for i in range(M)] for k in range(1, n)]
    mar_seen = [None] + [[None if rec[k - 1]["bankrupt"][i] or rec[k - 1]["margin"][i] is None else 100 * rec[k - 1]["margin"][i]
                          for i in range(M)] for k in range(1, n)]
    A["top_esg"] = [None] + [shown_top(esg_seen[k], 3) for k in range(1, n)]
    A["top_margin"] = [None] + [shown_top(mar_seen[k], 2) for k in range(1, n)]
    A["top_esg_v"] = [None] + [esg_seen[k][A["top_esg"][k][0]] if A["top_esg"][k] else None for k in range(1, n)]
    A["top_margin_v"] = [None] + [mar_seen[k][A["top_margin"][k][0]] if A["top_margin"][k] else None for k in range(1, n)]

    def cat(j, k):
        """cash / all / exactly the top set (esg, margin) / some of a top set that is not every company (esg_in, margin_in) /
        another selection (pick: it includes a company outside both top sets)."""
        S = tuple(rec[k]["choice"][j])
        live = tuple(i for i in range(M) if rec[k]["live"][i])
        if not S:
            return "cash"
        if S == live:
            return "all"
        te, tm = A["top_esg"][k], A["top_margin"][k]
        if te and S == te:
            return "esg"
        if tm and S == tm:
            return "margin"
        if te and te != live and set(S) < set(te):
            return "esg_in"
        if tm and tm != live and set(S) < set(tm):
            return "margin_in"
        return "pick"
    A["cats"] = [[cat(j, k) for k in range(n)] for j in range(N)]
    A["cat_n"] = [{c: A["cats"][j].count(c) for c in CATS} for j in range(N)]
    # money placed in each company, summed over the years (the rules' investment_matrix)
    A["inv_tot"] = [[sum(rec[k]["amt"][j] or 0.0 for k in range(n) if i in rec[k]["choice"][j]) for i in range(M)] for j in range(N)]
    A["inv_share"] = [[v / sum(row) if sum(row) else 0.0 for v in row] for row in A["inv_tot"]]
    # riding the lag: the ESG investors hold a company in a year it spends under 0.5%, on the score of a year it spent more
    A["lag"] = [[] for _ in range(M)]
    for k in range(1, n):
        for i in range(M):
            js = [j for j in range(N) if alphas[j] > 0 and i in rec[k]["choice"][j]]
            if js and mit[i][k] < 0.5 and mit[i][k - 1] >= 0.5:
                A["lag"][i].append((rec[k]["year"], sum(rec[k]["amt"][j] or 0.0 for j in js), js))
    A["mitigators"] = [i for i in range(M) if A["mean_m"][i] >= cap_pct / 2]
    A["riders"] = [i for i in range(M) if A["mean_m"][i] < 0.5]
    A["esg_inv"] = [j for j in range(N) if alphas[j] > 0]
    A["prof_inv"] = [j for j in range(N) if alphas[j] == 0]
    # the ESG investors' money: share of their money-years in the mitigators and in the free riders
    esg_tot = sum(sum(A["inv_tot"][j]) for j in A["esg_inv"])
    A["esg_in_mit"] = sum(A["inv_tot"][j][i] for j in A["esg_inv"] for i in A["mitigators"]) / esg_tot if esg_tot else None
    A["esg_in_rid"] = sum(A["inv_tot"][j][i] for j in A["esg_inv"] for i in A["riders"]) / esg_tot if esg_tot else None
    A["ev_same"] = sum(r["X"] for r in ref["nomit_same"]) if ref.get("nomit_same") else None
    return A


# ------------------------------------------------------------------------------------------------ the story: title, lede, tiles, subs
def story(P, A, rec, ref, no_mit_risk, rewards, cut_short, n_illegal):
    M, N, Y0, T = int(P["companies"]), int(P["investors"]), int(P["first_year"]), int(P["years"])
    n = A["n"]
    cap_pct = 100 * P["action_cap"]
    alphas = P["alphas"]
    out = {}
    if not n:
        out.update(title="InvestESG: no complete year yet", lede="Not enough of the game was recorded to show a year.",
                   tiles=[{"k": "Years played", "v": "0", "s": "No year has every seat's decision."},
                          {"k": "Decisions", "v": "0", "s": ""}], sub_mit="", sub_inv="", sub_cap="", sub_risk="", foot_defs="")
        return out
    last = rec[-1]
    yl = last["year"]
    mean_m, K_fin, G_fin = A["mean_m"], A["K_fin"], A["G_fin"]
    order = sorted(range(M), key=lambda i: -mean_m[i])
    mitigators, riders, esg_inv, prof_inv = A["mitigators"], A["riders"], A["esg_inv"], A["prof_inv"]
    middle = [i for i in range(M) if i not in mitigators and i not in riders]
    base, coop = ref.get("none"), ref.get("coop")
    risk_end, nm_end = last["risk"], no_mit_risk[n - 1]
    rich = max(range(M), key=lambda i: K_fin[i])
    poor = min(range(M), key=lambda i: K_fin[i])

    def m_avg(i):
        v = mean_m[i]
        return "0%" if v == 0 else "under 0.001%" if v < 0.0005 else mp(round(v, 3 if v < 0.01 else 2))

    # how the mitigators and the free riders came out. A group result is stated only when the groups do not overlap (every
    # member of one ahead of every member of the other); otherwise the companies are named one by one. Ranked by total gains G,
    # what a company is scored on, rather than by year-end capital, which holds the money the investors move every year.
    use_g = None not in G_fin
    val = G_fin if use_g else K_fin
    what_v = "in total gains" if use_g else "of capital at the end"

    def money_note(i):
        """'mostly on the ESG investors' money' when over the years more than half the capital it managed was theirs."""
        managed = [r["Kint"][i] for r in rec]
        if None in managed or not sum(managed):
            return ""
        inv = sum(r["inv_money"][i] or 0.0 for r in rec) / sum(managed)
        esg = sum(sum(r["amt"][j] or 0.0 for j in esg_inv if i in r["choice"][j]) for r in rec) / sum(managed)
        return ("mostly on the ESG investors' money" if esg > 0.5 else "mostly on investors' money" if inv > 0.5 else "")

    def rng_txt(xs):
        lo, hi = min(val[i] for i in xs), max(val[i] for i in xs)
        return tn0(lo) if len(xs) == 1 or tn0(lo) == tn0(hi) else f"{tn0(lo)} to {tn0(hi)}"

    def role_txt(i):
        return "mitigator" if i in mitigators else "free rider" if i in riders else "moderate"
    g_mit = "the mitigator" if len(mitigators) == 1 else "the mitigators"
    g_rid = "the free rider" if len(riders) == 1 else "the free riders"

    def split_of(v):
        """((ahead companies, name), (behind companies, name)) when every mitigator is ahead of every free rider by v, or the
        reverse; None when the groups overlap or one is empty."""
        if not mitigators or not riders:
            return None
        if min(v[i] for i in mitigators) > max(v[i] for i in riders):
            return (mitigators, g_mit), (riders, g_rid)
        if min(v[i] for i in riders) > max(v[i] for i in mitigators):
            return (riders, g_rid), (mitigators, g_mit)
        return None
    split = split_of(val)

    # ---- title
    esg_pull = ""
    many = len(mitigators) > 1
    if esg_inv and mitigators and A["esg_in_mit"] is not None:
        esg_pull = f" and draw{'' if many else 's'} {share_txt(A['esg_in_mit'])} of the ESG investors' money"
    if mitigators and all(A["yrs_cap"][i] == n for i in mitigators):
        how_much = f"spend{'' if many else 's'} the {mp(cap_pct)} cap on mitigation every year"
    elif mitigators:
        lo, hi = min(mean_m[i] for i in mitigators), max(mean_m[i] for i in mitigators)
        rng = mp(round(lo, 1)) if abs(hi - lo) < 0.05 else f"{mp(round(lo, 1))[:-1]} to {mp(round(hi, 1))}"
        how_much = f"carr{'y' if many else 'ies'} the mitigation ({rng} of capital a year on average)"
    if mitigators and riders:
        p1 = f"{companies_txt(mitigators)} {how_much}{esg_pull}"
        p2 = f"{companies_txt(riders)} free-ride{'s' if len(riders) == 1 else ''}"
    elif len(mitigators) == M:
        p1, p2 = f"All {nw(M)} companies " + how_much.replace("spends", "spend").replace("carries", "carry"), ""
    elif len(riders) == M:
        p1, p2 = f"All {nw(M)} companies free-ride on mitigation, spending under 0.5% a year on average", ""
    elif mitigators:
        p1, p2 = f"{companies_txt(mitigators)} {how_much}{esg_pull}", ""
    elif riders:
        p1, p2 = f"{companies_txt(riders)} free-ride{'s' if len(riders) == 1 else ''} while the others mitigate moderately", ""
    else:
        p1, p2 = "Every company mitigates moderately, between 0.5% and half the cap a year on average", ""
    top = max(range(M), key=lambda i: val[i])
    if split:
        (a_, an), (b_, bn) = split
        p3 = (f"{an} gain{'s' if len(a_) == 1 else ''} more than {bn} ({rng_txt(a_)} against {rng_txt(b_)} trillion "
              f"{what_v})" if use_g else
              f"{an} end{'s' if len(a_) == 1 else ''} richer than {bn} ({rng_txt(a_)} against {rng_txt(b_)} trillion)")
    elif mitigators and riders:                              # the groups overlap: name the companies, best first
        named = sorted(mitigators + riders, key=lambda i: -val[i])
        bits = []
        for i in named:
            note = money_note(i)
            bits.append(f"Company {i} ({role_txt(i)}{', ' + note if note else ''}) {tn0(val[i])}")
        p3 = ("total gains " if use_g else "capital at the end ") + ", ".join(bits) + " trillion"
    else:
        note = money_note(top)
        p3 = (f"Company {top} {'gains the most' if use_g else 'ends richest'} ({tn0(val[top])} trillion"
              + (" in all" if use_g else "")
              + (f", {note}" if note else "") + ")")
    # the no-mitigation probability is not capped (the paper's formula), so it can pass 100%: said so where it does
    nm_txt = (f"over 100% with no mitigation: the paper's formula is uncapped" if nm_end > 1 else
              f"{pc(nm_end, 0)} with no mitigation")
    p4 = ((f"climate risk ends at {pc(risk_end, 0)}" if not cut_short else f"climate risk {pc(risk_end, 0)} in {yl}")
          + f" ({nm_txt})")
    out["title"] = "; ".join(x for x in (p1, p2, p3, p4) if x)

    # ---- lede
    if esg_inv:
        e_txt = (f"{'Investor' if len(esg_inv) == 1 else 'Investors'} {and_list([str(j) for j in esg_inv])} "
                 f"{'is' if len(esg_inv) == 1 else 'are'} ESG-minded ({ALPHA} = "
                 + ", ".join(sorted({f'{alphas[j]:g}' for j in esg_inv})) + "): paid for "
                 + ("its" if len(esg_inv) == 1 else "their") + f" return plus {ALPHA} times the ESG score of the holdings"
                 + (f"; Investor {and_list([str(j) for j in prof_inv])} cares only about return" if prof_inv else "") + ". ")
    else:
        e_txt = "Every investor cares only about its return. "
    out["lede"] = (f"{nw(M).capitalize()} companies and {nw(N)} investors, every one played by the model, share one climate for "
                   f"{T} years ({Y0} to {Y0 + T - 1}). Each year every company decides what share of its capital, up to "
                   f"{mp(cap_pct)}, to spend on mitigation; that share is its public ESG score. Mitigation slows the rise of three "
                   f"disaster probabilities for everyone, and each disaster destroys {pc(P['damage'], 0)} of every company's "
                   f"capital, so mitigation is a public good paid for privately. At the same time each investor chooses which "
                   f"companies to fund, seeing last year's scores. " + e_txt
                   + "A company scores by its capital gains on all the money it manages, investors' money included.")

    # ---- tiles
    tiles = []
    spark = [round(100 * r["risk"], 2) for r in rec if r["risk"] is not None]
    ev = A["events"]
    tiles.append({"k": f"Climate risk in {yl}", "v": pc(risk_end), "spark": spark,
                  "s": (f"From {pc(rec[0]['risk'])} in {Y0}. If nobody ever mitigated: {pc(nm_end)}"
                        + (" (the paper's formula is not capped at 100%)" if nm_end > 1 else "")
                        + (f"; if every company spent 0.5% a year: {pc(coop[n - 1]['risk'])}" if coop else "")
                        + f". {ev if ev else 'No'} disaster{'' if ev == 1 else 's'} in {n} year{'s' if n > 1 else ''}"
                        + (f"; the same draws without this mitigation give {A['ev_same']}" if A["ev_same"] is not None
                           and A["ev_same"] != ev else "; the mitigation has not yet avoided one" if A["ev_same"] is not None
                           else "") + ".")})
    tot = A["tot_mit"]
    if tot:
        sh = sorted(range(M), key=lambda i: -(A["tot_mit_i"][i] or 0))
        rid_bits = [f"Company {i}" + (f" (nothing in {A['yrs_zero'][i]} of {n} years)" if A["yrs_zero"][i] else "")
                    for i in riders]
        tiles.append({"k": "Who paid for mitigation", "v": f"{tn0(tot)} tn",
                      "s": ("All companies' mitigation spending, by share: " + ", ".join(
                          f"Company {i} {share_txt((A['tot_mit_i'][i] or 0) / tot)}" for i in sh) + "."
                            + (f" Free rider{'s' if len(riders) > 1 else ''} (under 0.5% of capital a year on average): "
                               f"{and_list(rid_bits)}." if riders else ""))})
    else:
        tiles.append({"k": "Who paid for mitigation", "v": "nobody", "s": f"No company spent anything on mitigation in {n} years."})
    rw_ok = not cut_short and None not in rewards["co"] + rewards["inv"]
    tiles.append({"k": f"Company capital, end of {yl}", "v": f"{tn0(min(K_fin))} to {tn0(max(K_fin))} tn",
                  "s": ("; ".join(f"Company {i} {tn0(K_fin[i])} ({m_avg(i)} a year on mitigation)" for i in order) + ". "
                       + (f"Nobody mitigating, with the investors in every company: {tn0(base[n - 1]['K_end'][0])} each; "
                          f"everyone at 0.5%: {tn0(coop[n - 1]['K_end'][0])}. " if base and coop else "")
                       + ("Rewards " + " / ".join(f"{v:.3f}" for v in rewards["co"]) + "." if rw_ok else "")).strip()})
    inv_bits = []
    for j in range(N):
        fav = max(range(M), key=lambda i: A["inv_share"][j][i])
        inv_bits.append(f"Investor {j} ({ALPHA} {alphas[j]:g}) {tn0(last['cap_end'][j])}"
                        + (f", episode return {last['U'][j]:.2f}" if last["U"][j] is not None else "")
                        + (f", most money in Company {fav} ({share_txt(A['inv_share'][j][fav])})" if sum(A["inv_tot"][j]) else ""))
    tiles.append({"k": f"Investors' capital, end of {yl}", "v": " / ".join(tn0(last["cap_end"][j]) for j in range(N)) + " tn",
                  "s": ("; ".join(inv_bits) + f". Each started with {tn(P['investor_capital'])}; the episode return is the sum "
                       f"of yearly portfolio returns plus {ALPHA} times the portfolio's ESG score. "
                       + ("Rewards " + " / ".join(f"{v:.3f}" for v in rewards["inv"]) + "." if rw_ok else "")).strip()})
    out["tiles"] = tiles

    # ---- chart subtitles
    sub = [f"The share of its capital each company spent on mitigation each year (cap {mp(cap_pct)}), which is also its public "
           f"ESG score; zoom in to read the numbers."]
    for i in order:
        bits = []
        if A["yrs_cap"][i]:
            bits.append(f"the cap in {A['yrs_cap'][i]}")
        if A["yrs_zero"][i]:
            bits.append(f"nothing in {A['yrs_zero'][i]}")
        if A["yrs_token"][i]:
            bits.append(f"a token under 0.5% in {A['yrs_token'][i]}")
        rest = n - A["yrs_cap"][i] - A["yrs_zero"][i] - A["yrs_token"][i]
        if rest and bits:
            bits.append(f"0.5% to under the cap in {rest}")
        tm = A["tot_mit_i"][i]
        sub.append(f"Company {i}: {m_avg(i)} a year on average" + (f" ({', '.join(bits)} of {n} years)" if bits else "")
                   + (f", {amount(tm)} trillion in all" if tm is not None else "") + ".")
    lagged = [i for i in range(M) if A["lag"][i]]
    if lagged and esg_inv:
        parts = []
        for i in lagged:
            ys = [y for y, _, _ in A["lag"][i]]
            parts.append(f"Company {i} in {years_text(ys)}" if len(ys) <= 4 else f"Company {i} in {len(ys)} years (first {ys[0]})")
        sub.append("Investors see only last year's score, so a company can mitigate one year and stop the next while the ESG "
                   "investors still hold it: " + and_list(parts) + ".")
    out["sub_mit"] = " ".join(sub)

    sub = ["Each investor's holdings each year (its capital is split equally among the chosen companies), against what it "
           "saw: last year's ESG scores and margins, where ties are common (companies at the cap share the top score, and "
           "companies that spend the same share earn the same margin). Green = exactly the companies with last year's top ESG "
           "score, faded green = only some of them; yellow = exactly the companies with last year's best margin, faded yellow "
           "= only some of them; dark = another selection (it includes a company outside those groups); gray = every "
           "company; pale = cash. Zoom in to read the company numbers."]

    def grp(a, b, lab):
        if a and b:
            return f"{lab} in {a + b} (all of them in {a}, only some in {b})"
        return f"exactly {lab} in {a}" if a else f"only some of {lab} in {b}" if b else None
    for j in range(N):
        c = A["cat_n"][j]
        bits = [grp(c["esg"], c["esg_in"], "last year's top ESG scorers"),
                grp(c["margin"], c["margin_in"], "last year's best margins")]
        bits = [x for x in bits if x] + [f"{lab} in {c[key]}" for key, lab in (("pick", "another selection"),
                                                                             ("all", "every company"), ("cash", "cash")) if c[key]]
        fav = sorted(range(M), key=lambda i: -A["inv_share"][j][i])
        sub.append(f"Investor {j} ({ALPHA} {alphas[j]:g}): {', '.join(bits)} of {n} years"
                   + (f"; {share_txt(A['inv_share'][j][fav[0]])} of its money over the years in Company {fav[0]}"
                      if sum(A["inv_tot"][j]) else "") + ".")
    out["sub_inv"] = " ".join(sub)

    k_all0, k_all1 = sum(rec[0]["K_end"]), sum(K_fin)
    sub = [f"Each company's capital at the end of the year, investors' money included, as a share of all {nw(M)} companies' "
           f"capital, which grows from {tn0(float(P['company_capital']) * M)} trillion at the start to {tn0(k_all1)} by {yl} "
           f"(the trillions are in the feed and the grid tips). The investors take their stakes back at the start of each year "
           f"and put them where they choose, so a company they favour manages more and earns more for the same margin."]
    imo = last.get("inv_money_end") or [None] * M
    sub.append(f"Company {rich} ends richest ({tn0(K_fin[rich])}"
               + (f", {share_txt(imo[rich] / K_fin[rich])} of it investors' money" if imo[rich] is not None and K_fin[rich] else "")
               + f"), Company {poor} poorest ({tn0(K_fin[poor])}"
               + (f", {share_txt(imo[poor] / K_fin[poor])} investors' money" if imo[poor] is not None and K_fin[poor] else "") + ").")
    if mitigators and riders:
        kv, ks = K_fin, split_of(K_fin)

        def lo_hi(xs):
            lo, hi = min(kv[i] for i in xs), max(kv[i] for i in xs)
            return tn0(lo) if len(xs) == 1 or tn0(lo) == tn0(hi) else f"{tn0(lo)} to {tn0(hi)}"
        if ks:
            (a_, an), (b_, bn) = ks
            sub.append(f"{cap1(an)} ({companies_txt(a_)}) end{'s' if len(a_) == 1 else ''} with {lo_hi(a_)}, {bn} "
                       f"({companies_txt(b_)}) with {lo_hi(b_)}.")
        else:
            sub.append("The mitigators and the free riders do not split into a richer and a poorer group: "
                       + ", ".join(f"Company {i} ({role_txt(i)}) {tn0(kv[i])}"
                                   for i in sorted(mitigators + riders, key=lambda i: -kv[i])) + ".")
    if use_g:
        sub.append(f"Total gains G by {yl}, which a company's reward is computed from: "
                   + ", ".join(f"Company {i} {sg(G_fin[i], 1) if abs(G_fin[i]) < 100 else sg(G_fin[i], 0)}"
                               for i in sorted(range(M), key=lambda i: -G_fin[i])) + " trillion.")
    if base and coop:
        sub.append(f"For scale: nobody ever mitigating, with the investors in every company, gives {tn0(base[n - 1]['K_end'][0])} "
                   f"per company by {yl}; every company at 0.5% a year (the paper's cooperative benchmark), "
                   f"{tn0(coop[n - 1]['K_end'][0])}.")
    out["sub_cap"] = " ".join(sub)

    nd = len(A["dis_years"])
    over100 = next((k for k, v in enumerate(no_mit_risk) if v > 1), None)   # over the whole axis the dashed line spans
    sub = [f"1 {MINUS} the product of (1 {MINUS} p) over the three hazards ({', '.join(HAZARDS)}), whose probabilities start at "
           + " / ".join(pc(p, 0) for p in P["hazard_p0"]) + f" and rise every year by an amount divided by (1 + a coefficient "
           f"{TIMES} all companies' mitigation so far, in trillions). Dashed: nobody ever mitigates"
           + (f" (it passes 100% in {Y0 + over100}: the formula is not capped, as in the paper)" if over100 is not None else "")
           + ("; every company spends 0.5% a year with the investors in every company" if coop else "") + "."
           + f" The strip below the plot: years with at least one disaster ({nd} of {n}, {ev} disaster{'' if ev == 1 else 's'} "
             f"in all)."]
    if A["ev_same"] is not None:
        es = A["ev_same"]
        sub.append("The draws are the official code's fixed numbers, so the choices decide only which thresholds the "
                   "probabilities cross: with the same investor choices and no mitigation at all, the same draws give "
                   + (f"{es} disaster{'' if es == 1 else 's'} instead of {ev}: the mitigation avoided {es - ev}."
                      if es != ev else f"the same {ev if ev else 'zero'} disaster{'' if ev == 1 else 's'}: so far the "
                      "mitigation has avoided none."))
    out["sub_risk"] = " ".join(sub)

    out["foot_defs"] = (f"Definitions: a company's margin is its capital gain over the capital it managed that year; its reward "
                        f"maps its total gain G (trillions) to log(1 + G/{P['company_capital']:g}) {DIV} log(1 + G_max/"
                        f"{P['company_capital']:g}), G_max being the whole market compounding at {pc(P['market_growth'], 0)} for "
                        f"{T} years; an investor's reward is its episode return over {T} {TIMES} ({P['market_growth']:g} + "
                        f"{ALPHA} {TIMES} {P['action_cap']:g}). Mitigators: companies that spent at least half the cap a year on "
                        f"average; free riders: under 0.5%. Climate risk is the probability of at least one disaster in the year. "
                        f"Market wealth is the companies' capital plus the investors' cash.")
    return out


# ------------------------------------------------------------------------------------------------ TEST HELPER (not for real pages)
def bundle_from_calls(path):
    """TEST ONLY. A stand-in bundle assembled in memory from a model-call log, for running the whole pipeline before the real
    bundle exists: one move per decision (the last call of each run of calls with the same seat and year; its answer, or the
    follow-up answer when rescued, is the move, its prompt the view), no summary and no game_over. run.json is marked
    test_bundle_from_calls, which the page shows in its eyebrow, title and foot. It is never written to disk."""
    calls = load_calls(path, lenient=True)
    runs = split_runs(calls)
    if len(runs) > 1:                                        # a restarted recording: the live run is the last one
        warn(f"the call log holds {len(runs)} recording runs; the test build uses the last (calls {runs[-1][0] + 1} to "
             f"{runs[-1][1]}) and leaves out {runs[-1][0]} earlier calls")
        calls = calls[runs[-1][0]:runs[-1][1]]
    events = [{"event": "setup", "_visibility": "hidden"}]
    k = 0
    while k < len(calls):
        c = calls[k]
        y = RX_YEAR.search(c["prompt"] or "")
        j = k
        while j + 1 < len(calls) and calls[j + 1]["seat"] == c["seat"] and RX_YEAR.search(calls[j + 1]["prompt"] or "") \
                and y and RX_YEAR.search(calls[j + 1]["prompt"]).group(1) == y.group(1):
            j += 1
        last = calls[j]
        events.append({"event": "move", "actor": last["seat"], "turn": len(events), "move": last["answer"], "view": last["prompt"]})
        k = j + 1
    t0 = next((c["t_start"] for c in calls if c.get("t_start")), None)
    run = {"model": next((c.get("model") for c in calls if c.get("model")), None), "test_bundle_from_calls": True,
           "recorded_at": datetime.datetime.fromtimestamp(t0, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t0 else None}
    return {"events": events, "summary": {}, "run": run, "config": {}, "transcript": "", "meta": {}}, calls


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--test-from-calls":
        tb, tcalls = bundle_from_calls(sys.argv[2])
        common.write(build(tb, tcalls), sys.argv[3], date=recorded_date(tb, tcalls))
    else:
        if len(sys.argv) < 3:
            raise SystemExit(__doc__)
        src, out = sys.argv[1], sys.argv[2]
        cl_ = load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
        common.write(build(src, cl_), out, date=recorded_date(load_source(src), cl_))
