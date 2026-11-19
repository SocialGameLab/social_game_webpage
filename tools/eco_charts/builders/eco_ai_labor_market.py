"""Replay data for eco_ai_labor_market (AI-Work: a gig labour market where worker agents bid for jobs or train, under a public
reputation system; Chiu, Zhang and van der Schaar, "When AI Agents Compete for Jobs", ICML 2026).

usage: python3 -I eco_ai_labor_market.py BUNDLE OUT_DIR [CALLS.jsonl]
  BUNDLE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json, transcript.md), or a file
          inside one.
  OUT_DIR where <OUT_DIR>/eco_ai_labor_market.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: call number, t_start, messages or prompt, reasoning, answer, followup_answer
          when the answer came from a follow-up question, usage; see common.load_calls). The log is appended to by every
          recording process and its call counter restarts at 1 in each, so only this run's calls are used: those of the
          process that started at the run's start (meta.json/run.json source_run or recorded_at); calls of an earlier
          (failed) attempt or a later process are left out with a warning. Calls are tied to decisions in time order: a
          decision's calls are the consecutive calls whose prompt is exactly the move's view (the recorder sends the view
          itself as the prompt and resends it unchanged after an answer the game rejects); only when no call has that prompt
          are calls tied by seat and round read from the prompt. The call whose answer the game accepts (the env's own
          check: strict JSON, then its canonical action) is that decision's call; earlier calls in the window were rejected
          attempts, shown above it with the reason. Its reasoning becomes the step's thinking and its prompt the observation.

test only:  python3 -I eco_ai_labor_market.py --test-from-calls CALLS.jsonl OUT_DIR [RUN_LOG]
          builds the page from a call log alone (for example the growing log of a recording still running; a half-written
          last line is left out). The stand-in bundle is assembled in memory (TEST HELPER at the end of this file) from the
          log's last recording process only: one move per decision, taken from RUN_LOG (the recorder's run.log of the same
          run, numbered moves; a line ending ' (fallback)' marks the game's fallback) when given, else the last answer of
          each run of calls with the same prompt; no summary. The page says in its eyebrow, title and foot that it is a test
          build from a partial call log. Nothing is written but OUT_DIR/eco_ai_labor_market.json.

One round is one market round. Every seat sends one 'bid' or 'train' in seat order and the round is buffered (rules: Round
Procedure). There is no simulation to replay here (listings, Gumbel noise, matching and skill draws come from the engine), so
the page's record is read from what the seats were shown, which is the whole public record plus each seat's own outcomes:
  - the listings of round t (job, posted budget in cents) from any prompt of round t;
  - the leaderboard (every agent's total earnings) at the start of round t, so every agent's income in round t - 1;
  - the public market record of the last `history` rounds (each filled job with its posted budget, winner and the winner's
    reputation on that skill when it bid);
  - each seat's own last rounds (every bid's price and outcome, its income, feedback training, reputation before and after)
    and its reputation on every skill at the start of the round.
Every round is reported by several prompts; all reports must agree (checked). The seat's logged move is checked against its
own history line (the action the game applied), wins against the public record, incomes against the leaderboard and against
the winning prices (flat fee), reputations against the next prompt, the record summary line against the rebuilt totals, and
the scripted agents' incomes against their fixed price share of the won budgets (which also identifies their policy when the
summary does not name it). The last round appears in no prompt: its incomes come from the summary's final earnings, the
model workers' wins from the summary's jobs-won totals by skill and the one subset of their bids whose prices sum to that
income; when two subsets fit, the summary's per-agent mean winning price (mean of price / budget over all its wins) decides
between them: the last round's sum of price / budget is that mean times the jobs won minus the exact earlier ratios. When it is
still not unique the seat's wins are shown as a count only (no summary: the round's outcome is not recorded). A seat that bid
and won nothing in the last round gets its feedback training on its first bid's skill (rules: Round Procedure 6, certain when
fallback_train_prob is 1); each seat's feedback trainings are checked against the summary's fallback_trainings. The summary's
per-round series (listed, matched, payments, mean winning bid) and the rewards are checked against the rebuilt record.
An answer the game could not use is replaced by its fallback (train the highest-reputation skill; rules: Actions). The
recorder flags such a move ("fallback": true, with the rejected "reply"); the flag decides what the page shows and the reply,
read by the env's own check, gives the reason. Beyond the flag, only proof that the game did not apply the logged move makes
a fallback: a history line that shows training instead of it, or (a bundle without flags: a test build without RUN_LOG,
whose moves are raw answers) a move the game's check rejects; with flags present either one is also printed as a
disagreement. A training that none of the seat's calls answered is only a warning, never a fallback on the page.
Each round's market activity, leaderboard and listings are the same for every seat; a step's observation shows them in full
for the round's first decision and refers to them in the others (the page size limit leaves more room for reasoning).
Every disagreement is printed as a warning; the footnote states only the checks that passed."""
import datetime, itertools, json, math, os, re, statistics, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_ai_labor_market"
MINUS, DOT, TIMES, DIV, ARROW, NDASH = "−", "·", "×", "÷", "→", "–"
MAX_BYTES = 15_000_000                 # the host takes one JSON file of up to 16 MB
REPLY_CHARS = 2000                     # demo_training keeps a fallback's rejected reply up to this length
# The rules' Configuration defaults; used only for what neither the summary nor the prompts state.
DEFAULT_PARAMS = {"rounds": 100, "agents": 10, "seats": 8, "scripted": "specialist,greedy", "tasks": 4, "jobs_per_task": 4,
                  "budgets": [10, 8, 6, 4], "task_budget_scale": [1.0], "task_noise_scale": [1.0], "listing_prob": 0.8,
                  "budget_noise": 0.5, "capacity": 3, "max_bids": 5, "job_truncation": 5, "history": 10, "price_weight": 0.4,
                  "rho": 0, "gumbel_t": 0.01, "learning_prob": 0.1, "skill_decay": 0.9, "initial_skill": 0.4,
                  "task_noise": 0.05, "base_rate": 0.5, "prior_strength": 1, "rep_window": 5, "forgetting": 0.5,
                  "performance_pay": False, "open_bids": False, "fallback_train_prob": 1.0, "initial_calibration": False,
                  "code_reputation": False, "demand_shift_round": 0, "recession_block": 0, "recession_cycle": 3,
                  "recession_phase": 1, "recession_budget": 1.0, "recession_listing_prob": 0.1, "note_chars": 500,
                  "greedy_alpha": 0.8, "specialist_alpha": 0.9, "random_alpha": 0.9, "lowball_alpha": 0.5}
POLICY_NAME = {"specialist": "Specialist", "greedy": "Greedy", "random": "Random", "budget": "Budget", "lowball": "Lowball"}
POLICY_TXT = {"specialist": "prices its fixed list of jobs (skill A from the highest budget down, then B, ...) at {a} of the "
                            "budget and trains in about one round in five",
              "greedy": "prices the highest-budget listings at {a} of the budget and trains in about one round in ten",
              "random": "prices the jobs of a random skill order at {a} of the budget",
              "budget": "prices the highest-budget listings at the full budget", "lowball": "prices the highest-budget listings at {a}"}
POLICY_SHORT = {"specialist": ("a {n}", "bids {a} of the budget on a fixed list of jobs starting with skill A"),
                "greedy": ("a {n} agent", "bids {a} of the budget on the highest budgets"),
                "random": ("a {n} agent", "bids {a} of the budget on the jobs of a random skill order"),
                "budget": ("a {n} agent", "bids the full budget on the highest budgets"),
                "lowball": ("a {n} agent", "bids {a} of the budget on the highest budgets")}
KINDS = {"won": ["Bid, won jobs", "c3"], "lost": ["Bid, won nothing", "hold"], "train": ["Trained a skill", "c4"],
         "fallback": ["Unusable answer, game trained", "c2"], "pending": ["Bid, outcome not recorded", "muted"]}
NUM_WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
             10: "ten", 11: "eleven", 12: "twelve"}


def warn(msg):
    print("warning: " + msg, file=sys.stderr)


# ------------------------------------------------------------------------------------------------ English-only text
_CJK_RUN = re.compile(r"[぀-ヿ㐀-鿿가-힯]+")
_PUNCT = {"。": ". ", "、": ", ", "「": '"', "」": '"', "『": '"', "』": '"', "【": "[",
          "】": "]", "《": '"', "》": '"', "〈": "<", "〉": ">", "〜": "~"}


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
    """A bundle: a directory, a file inside one, or a bundle already in memory (the test helper's)."""
    if isinstance(src, dict):
        return src
    if os.path.isdir(src):
        b = common.load_bundle(src)
        meta = os.path.join(src, "meta.json")
        b["meta"] = json.load(open(meta, encoding="utf-8")) if os.path.exists(meta) else {}
        return b
    d = os.path.dirname(os.path.abspath(src))
    if os.path.exists(os.path.join(d, "game_log.jsonl")):
        return load_source(d)
    raise SystemExit("not a demo bundle: " + str(src))


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
    """common.load_calls plus what it drops from each line: why the call stopped ('finish'), the output tokens it used, its start
    time and model. Same lines, same order. With `lenient` (test builds from a live log) a half-written last line is left out;
    each call is then read exactly as common.load_calls reads it."""
    if not path or not os.path.exists(path):
        return []
    if lenient:
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
    return float(str(s).replace(",", ""))


def money(v, signed=False, dp=2):
    """$1,234.56 with a true minus sign; '+' only when signed."""
    s = f"${abs(v):,.{dp}f}"
    if v < 0 and s.strip("$0.,") != "":
        return MINUS + s
    return ("+" + s) if signed and v > 0 and s.strip("$0.,") != "" else s


def pc(r, d=0):
    """A share (0.54) as a percentage ('54%')."""
    return f"{100 * r:.{d}f}%"


def st(v):
    return f"{v:.1f}*"


def r3(v):
    """A reward to 3 decimals with a true minus sign; no trailing zeros (-1 stays -1)."""
    s = f"{abs(v):.3f}".rstrip("0").rstrip(".")
    return (MINUS if v < 0 and s != "0" else ("+" if v > 0 and s != "0" else "")) + s


def and_list(xs):
    xs = list(xs)
    if len(xs) <= 1:
        return "".join(xs)
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def nw(n):
    return NUM_WORDS.get(n, str(n))


def cap1(s):
    return s[:1].upper() + s[1:] if s else s


def plural(n, one, many=None):
    return f"{n} {one if n == 1 else (many or one + 's')}"


def rounds_txt(a, b):
    return f"round {a}" if a == b else f"rounds {a}{NDASH}{b}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# ------------------------------------------------------------------------------------------------ actions (rules: Actions)
def _json_obj(s):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (s or "").strip())
    try:
        return json.loads(s)
    except ValueError:
        i, j = s.find("{"), s.rfind("}")
        try:
            return json.loads(s[i:j + 1]) if 0 <= i < j else None
        except ValueError:
            return None


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_action(move):
    """The action as {"type": "bid", "bids": [(job, price rounded to cents)], "note"} or {"type": "train", "skill", "note"};
    None when it does not have the action's shape (rules: Actions). As in the env, a note must be text when present, and an
    empty note is no note (the env keeps a note only when it is non-empty). Lenient about code fences; what the game itself
    accepts (listings, prices, strict JSON) is env_check()."""
    a = move if isinstance(move, dict) else _json_obj(move if isinstance(move, str) else "")
    if not isinstance(a, dict) or not isinstance(a.get("payload"), dict) or set(a) - {"action_id", "payload"}:
        return None
    p, kind = a["payload"], a.get("action_id")
    if "note" in p and not isinstance(p["note"], str):
        return None
    note = p.get("note") or None
    if kind == "bid":
        bids = p.get("bids")
        if set(p) - {"bids", "note"} or not isinstance(bids, list) or not bids:
            return None
        out = []
        for x in bids:
            if not isinstance(x, dict) or set(x) != {"job", "price"} or not isinstance(x["job"], str) or not _is_num(x["price"]):
                return None
            out.append((x["job"], round(float(x["price"]), 2)))
        return {"type": "bid", "bids": out, "note": note}
    if kind == "train":
        if set(p) - {"skill", "note"} or not isinstance(p.get("skill"), str):
            return None
        return {"type": "train", "skill": p["skill"], "note": note}
    return None


def canonical_of(a):
    """A parsed action (parse_action) in the env's canonical form: {"action_id", "payload"}, an empty note dropped."""
    if a is None:
        return None
    note = {"note": a["note"]} if a["note"] else {}
    if a["type"] == "train":
        return {"action_id": "train", "payload": {"skill": a["skill"], **note}}
    return {"action_id": "bid", "payload": {"bids": [{"job": j, "price": p} for j, p in a["bids"]], **note}}


def env_check(text, listed, P, skills):
    """What the game makes of an answer: the env's parse_action, rule by rule (the client hands it the stripped reply, read
    as strict JSON, so a code fence is rejected; then ecosim's _check and the market's _canonical). `listed` is the round's
    listings {job: posted budget} as the prompt shows them (cents, as the env rounds them). Returns (canonical action, None)
    when the game accepts it (prices rounded to cents, an empty note dropped) or (None, why it is rejected)."""
    if isinstance(text, dict):
        data = text
    else:
        if not isinstance(text, str) or not text.strip():
            return None, "no answer"
        try:
            data = json.loads(text.strip())
        except ValueError:
            return None, "not plain JSON" + (": wrapped in a code fence" if text.strip().startswith("```") else "")
    if not isinstance(data, dict):
        return None, "not a JSON object"
    if set(data) - {"action_id", "payload"} or not isinstance(data.get("action_id"), str):
        return None, "not an action object (only action_id and payload)"
    payload = data.get("payload", {})
    if not isinstance(payload, dict):
        return None, "payload is not an object"
    nc = P.get("note_chars", 500) or 0
    note = payload.get("note", "")
    if "note" in payload and (nc <= 0 or not isinstance(note, str) or len(note) > nc):
        return None, ("a note is not allowed" if nc <= 0 else "the note is not text" if not isinstance(note, str)
                      else f"note of {len(note):,} characters (at most {nc})")
    extra = {"note": note} if ("note" in payload and note) else {}
    kind = data["action_id"]
    if kind == "train":
        if set(payload) - {"skill", "note"} or "skill" not in payload:
            return None, "a train payload takes skill (and note) only"
        if payload["skill"] not in skills:
            return None, f"unknown skill {str(payload['skill'])[:20]}"
        return {"action_id": "train", "payload": {"skill": payload["skill"], **extra}}, None
    if kind != "bid":
        return None, f"unknown action {kind[:20]}"
    if set(payload) - {"bids", "note"} or "bids" not in payload:
        return None, "a bid payload takes bids (and note) only"
    bids = payload["bids"]
    if not isinstance(bids, list):
        return None, "bids is not a list"
    if not 1 <= len(bids) <= P["max_bids"]:
        return None, f"{len(bids)} bids (1 to {P['max_bids']} allowed)"
    catalog = {f"{k}{j + 1}" for k in skills for j in range(int(P["jobs_per_task"]))}
    seen, out = set(), []
    for item in bids:
        if not isinstance(item, dict) or set(item) != {"job", "price"}:
            return None, "a bid is not {job, price}"
        job, price = item["job"], item["price"]
        if not isinstance(job, str) or job in seen or job not in catalog:
            return None, (f"{job} bid twice" if isinstance(job, str) and job in seen else f"unknown job {str(job)[:20]}")
        if job not in listed:
            return None, f"{job} is not listed this round"
        try:
            pf = float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else None
        except OverflowError:
            pf = None
        if pf is None or not math.isfinite(pf) or pf < 0.01 or pf > 10.0 * listed[job]:
            return None, (f"{job} priced {money(pf) if pf is not None and math.isfinite(pf) else 'not as a number'}, outside "
                          f"$0.01 to ten times its {money(listed[job])} budget")
        seen.add(job)
        out.append({"job": job, "price": round(pf, 2)})
    return {"action_id": "bid", "payload": {"bids": out, **extra}}, None


# ------------------------------------------------------------------------------------------------ what a seat was shown
RX_SEAT = re.compile(r"\bYou are seat P(\d+)\b")
RX_LABEL = re.compile(r"\bIn the market you are ([A-Za-z]+_\d+)")
RX_ROUND = re.compile(r"(?m)^Round (\d+)(?: of (\d+))?([^\n]*)$")
RX_SETTINGS = re.compile(r"Market settings: (\d+) agents, (\d+) skills?, a catalog of (\d+) jobs.*?capacity (\d+) jobs? per agent "
                         r"per round, up to (\d+) bids? per round")
RX_HIST_LINE = re.compile(r"(?m)^- R(\d+): ([^\n]*)$")
RX_ALLOC = re.compile(r"([A-Z]+\d+) \$(\d[\d,]*(?:\.\d+)?) -> ([A-Za-z]+_\d+)(?: (\d+(?:\.\d+)?)\*)?(?: at \$(\d[\d,]*(?:\.\d+)?))?")
RX_LB = re.compile(r"([A-Za-z]+_\d+)( \(you\))? \$(\d[\d,]*\.\d\d)")
RX_BID_ITEM = re.compile(r"([A-Z]+\d+) \$(\d[\d,]*\.\d+)/\$(\d[\d,]*(?:\.\d+)?) (won|lost)(?: \(([^)]*)\))?")
RX_REP_ITEM = re.compile(r"\b([A-Z]+) (\d+(?:\.\d+)?)\*(?: -> (\d+(?:\.\d+)?)\*)?")
RX_LIST_LINE = re.compile(r"(?m)^- skill ([A-Z]+): ([^\n]*)$")
RX_LIST_ITEM = re.compile(r"([A-Z]+\d+) \$(\d[\d,]*\.\d+)")
RX_RECORD = re.compile(r"Your record over rounds (\d+)-(\d+): earnings \$(\d[\d,]*\.\d\d), rank (\d+) of (\d+); (\d+) jobs? won"
                       r"(?: \(([^)]*)\))? from (\d+) bids? in (\d+) bidding rounds?; (\d+) rounds? training; (\d+) feedback "
                       r"trainings?")
HEADERS = [("market", r"(?m)^Market activity"), ("lb", r"(?m)^Leaderboard"), ("own", r"(?m)^Your (?:last \d+ rounds?|recent rounds)"),
           ("record", r"(?m)^Your record over"), ("rep", r"(?m)^Your reputation:"), ("note", r"(?m)^Your note from last round:"),
           ("listings", r"(?m)^Listings this round"), ("end", r"(?m)^Legal actions:")]


def round_of(text):
    m = RX_ROUND.search(text or "")
    return int(m.group(1)) if m else None


def _sections(text):
    pos = []
    for k, rx in HEADERS:
        m = re.search(rx, text)
        if m:
            pos.append((m.start(), k))
    pos.sort()
    out = {}
    for i, (p, k) in enumerate(pos):
        out[k] = text[p:pos[i + 1][0] if i + 1 < len(pos) else len(text)]
    return out


def parse_own_line(body):
    """One line of the seat's own history: BID items (job, price, budget to one decimal, won or lost), income, feedback training
    and its reputation on each skill involved (when it acted, and after the round if it changed); or TRAIN skill."""
    h = {"type": None, "skill": None, "bids": [], "income": None, "feedback": None, "rep": {}, "raw": body}
    m = re.match(r"(BID|TRAIN)\b", body)
    if not m:
        return h
    h["type"] = m.group(1)
    if h["type"] == "TRAIN":
        m = re.match(r"TRAIN ([A-Z]+)", body)
        h["skill"] = m.group(1) if m else None
    else:
        cut = body.find(". Income")
        part = body[:cut] if cut >= 0 else body
        h["bids"] = [{"job": x.group(1), "price": num(x.group(2)), "budget1": x.group(3), "won": x.group(4) == "won",
                      "extra": x.group(5)} for x in RX_BID_ITEM.finditer(part)]
    m = re.search(r"Income \$(\d[\d,]*\.\d\d)", body)
    if m:
        h["income"] = num(m.group(1))
    elif "no income" in body:
        h["income"] = 0.0
    m = re.search(r"Feedback training on ([A-Z]+)", body)
    if m:
        h["feedback"] = m.group(1)
    i = body.find("Reputation ")
    if i >= 0:
        for x in RX_REP_ITEM.finditer(body[i + len("Reputation "):]):
            h["rep"][x.group(1)] = (float(x.group(2)), float(x.group(3)) if x.group(3) else None)
    return h


def view_facts(view):
    """Everything a prompt states that the page uses (rules: Round Procedure, 'What a seat sees on its turn')."""
    f = {"seat": None, "label": None, "round": None, "of": None, "round_line": "", "settings": None, "market": {}, "lb": {},
         "own": {}, "record": None, "rep": {}, "note": None, "listings": None, "skills": []}
    view = view or ""
    m = RX_SEAT.search(view)
    f["seat"] = int(m.group(1)) if m else None
    m = RX_LABEL.search(view)
    f["label"] = m.group(1) if m else None
    m = RX_ROUND.search(view)
    if m:
        f["round"], f["of"], f["round_line"] = int(m.group(1)), int(m.group(2)) if m.group(2) else None, m.group(3)
    m = RX_SETTINGS.search(view)
    if m:
        f["settings"] = {"agents": int(m.group(1)), "tasks": int(m.group(2)), "catalog": int(m.group(3)),
                         "capacity": int(m.group(4)), "max_bids": int(m.group(5))}
    S = _sections(view)
    for m in RX_HIST_LINE.finditer(S.get("market", "")):
        f["market"][int(m.group(1))] = [{"job": a.group(1), "budget1": a.group(2), "winner": a.group(3),
                                         "rep": float(a.group(4)) if a.group(4) else None,
                                         "price": num(a.group(5)) if a.group(5) else None} for a in RX_ALLOC.finditer(m.group(2))]
    lbs = S.get("lb", "")
    if lbs:
        f["lb"] = {a.group(1): num(a.group(3)) for a in RX_LB.finditer(lbs[lbs.find(":") + 1:])}
        you = [a.group(1) for a in RX_LB.finditer(lbs) if a.group(2)]
        f["lb_you"] = you[0] if you else None
    for m in RX_HIST_LINE.finditer(S.get("own", "")):
        f["own"][int(m.group(1))] = parse_own_line(m.group(2))
    m = RX_RECORD.search(S.get("record", ""))
    if m:
        f["record"] = {"from": int(m.group(1)), "to": int(m.group(2)), "earnings": num(m.group(3)), "rank": int(m.group(4)),
                       "of": int(m.group(5)), "jobs": int(m.group(6)),
                       "by_skill": {k: int(v) for k, v in re.findall(r"([A-Z]+) (\d+)", m.group(7) or "")},
                       "bids": int(m.group(8)), "bid_rounds": int(m.group(9)), "train": int(m.group(10)),
                       "feedback": int(m.group(11))}
    reps = S.get("rep", "")
    if reps:
        line = reps.splitlines()[0]
        f["rep"] = {x.group(1): float(x.group(2)) for x in RX_REP_ITEM.finditer(line[line.find(":") + 1:])}
        f["skills"] = list(f["rep"])
    note = S.get("note", "")
    if note:
        t = note[note.find(":") + 1:].strip()
        f["note"] = None if t in ("none", "none.", "") else t
    lst = S.get("listings")
    if lst is not None:
        f["listings"] = {}
        for m in RX_LIST_LINE.finditer(lst):
            for x in RX_LIST_ITEM.finditer(m.group(2)):
                f["listings"][x.group(1)] = num(x.group(2))
    return f


# ------------------------------------------------------------------------------------------------ parameters
def read_params(info, F):
    P = dict(DEFAULT_PARAMS)
    params = info.get("params") or {}
    P.update({k: v for k, v in params.items() if v is not None})
    if not params:
        warn("the summary carries no parameters; using the rules' defaults and the prompts")
    s = next((f["settings"] for f in F if f["settings"]), None)
    if s:
        for k in ("agents", "tasks", "capacity", "max_bids"):
            if params.get(k) is not None and params[k] != s[k]:
                warn(f"parameter {k}: summary {params[k]}, prompt {s[k]}")
            elif params.get(k) is None:
                P[k] = s[k]
        if params.get("jobs_per_task") is None and s["tasks"]:
            P["jobs_per_task"] = s["catalog"] // s["tasks"]
    if isinstance(P.get("scripted"), str):
        P["scripted_list"] = [x.strip() for x in P["scripted"].split(",") if x.strip()]
    elif isinstance(P.get("scripted"), list):
        P["scripted_list"] = [str(x) for x in P["scripted"]]
    else:
        P["scripted_list"] = []
    return P


def alpha_of(policy, P):
    return {"specialist": P.get("specialist_alpha", 0.9), "greedy": P.get("greedy_alpha", 0.8),
            "random": P.get("random_alpha", 0.9), "budget": 1.0, "lowball": P.get("lowball_alpha", 0.5)}.get(policy)


def skill_names(P, F):
    sk = next((f["skills"] for f in F if f["skills"]), None)
    if sk:
        return sorted(sk)
    return [chr(65 + k) for k in range(int(P["tasks"]))]


def _expect_floor(b, sigma, floor=0.1):
    """E[max(floor, X)] for X ~ N(b, sigma^2): the expected posted budget of a job with base budget b (rules: Round Procedure 1)."""
    if sigma <= 0:
        return max(floor, b)
    z = (floor - b) / sigma
    Phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    return floor * Phi + b * (1 - Phi) + sigma * phi


def expected_posted(P, skills):
    """Expected total posted budget per round (0-based list over the horizon), with the recession and demand-shift schedules."""
    K, J = len(skills), int(P["jobs_per_task"])
    budgets = list(P["budgets"]) or [10]
    scale = list(P.get("task_budget_scale") or [1.0])
    nscale = list(P.get("task_noise_scale") or [1.0])
    out = []
    for t in range(int(P["rounds"])):
        rec = P.get("recession_block") and ((t // P["recession_block"]) % P["recession_cycle"]) == P["recession_phase"]
        tot = 0.0
        for k in range(K):
            kk = k
            if P.get("demand_shift_round") and t >= P["demand_shift_round"] and k in (0, 1):
                kk = 1 - k                                    # skills A and B swap budget and noise scales
            for j in range(J):
                b = budgets[j % len(budgets)] * scale[kk % len(scale)]
                sig = P["budget_noise"] * nscale[kk % len(nscale)]
                pi = P["listing_prob"]
                if rec:
                    b, pi = P["recession_budget"], P["recession_listing_prob"]
                tot += pi * _expect_floor(b, sig)
        out.append(tot)
    return out


# ------------------------------------------------------------------------------------------------ checks
class Tally:
    def __init__(self):
        self.n, self.ok = {}, {}

    def eq(self, group, ok, msg):
        self.n[group] = self.n.get(group, 0) + 1
        if ok:
            self.ok[group] = self.ok.get(group, 0) + 1
        else:
            warn(msg)
        return ok

    def full(self, g):
        return self.n.get(g, 0) > 0 and self.ok.get(g, 0) == self.n[g]

    def txt(self):
        return ", ".join(f"{g} {self.ok.get(g, 0)}/{n}" for g, n in self.n.items())


def per_skill(v, skills):
    if isinstance(v, dict):
        return {str(k): float(x) for k, x in v.items() if _is_num(x)}
    if isinstance(v, list):
        return {skills[i]: float(x) for i, x in enumerate(v) if i < len(skills) and _is_num(x)}
    return {}


def series_get(info, key, n):
    se = info.get("series")
    v = None
    if isinstance(se, dict) and isinstance(se.get(key), list):
        v = list(se[key])
    elif isinstance(se, list) and se and isinstance(se[0], dict):
        v = [r.get(key) for r in se]
    if v is None:
        return None
    return (v + [None] * n)[:n]


# ------------------------------------------------------------------------------------------------ model calls
def _utc(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run_start_of(b):
    """When the recorded run started, in epoch seconds: the stamp that opens the run directory's name (meta.json or run.json
    source_run; the recorder takes it before the run's first model call), else recorded_at with a time of day (the same
    stamp); None when neither is recorded."""
    meta, run = b.get("meta") or {}, b.get("run") or {}
    for v in (meta.get("source_run"), run.get("source_run")):
        m = re.match(r"(\d{8}T\d{6}Z)", str(v or ""))
        if m:
            return datetime.datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
    for v in (meta.get("recorded_at"), run.get("recorded_at")):
        if not v or "T" not in str(v):
            continue
        try:
            d = datetime.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except ValueError:
            continue
        return (d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)).timestamp()
    return None


def call_segments(calls):
    """The call log split into recording processes: [(first index, end index)]. Every process appends to the same file and
    numbers its calls from 1 (sitecustomize's per-process counter), so a counter that does not grow starts a new process."""
    segs, prev = [], None
    for k, c in enumerate(calls):
        n = c.get("call")
        if not segs or (n is not None and prev is not None and n <= prev):
            segs.append([k, k + 1])
        segs[-1][1] = k + 1
        prev = n if n is not None else prev
    return [tuple(s) for s in segs]


def select_calls(calls, b, views):
    """This run's calls only. With the run's start time known: the recording process whose first call came at or after it
    (calls of an earlier, failed attempt and of any later process are left out, and so is any call that started more than a
    minute before the run). Without it: the process whose prompts are most often the moves' views (ties: the latest)."""
    if not calls:
        return calls
    segs = call_segments(calls)
    first_t = lambda seg: next((calls[k]["t_start"] for k in range(*seg) if calls[k].get("t_start") is not None), None)
    t0 = run_start_of(b)
    if t0 is not None and any(first_t(s) is not None for s in segs):
        late = [s for s in segs if first_t(s) is not None and first_t(s) >= t0 - 60]
        pick = next((s for s in late if first_t(s) >= t0), late[0] if late else None)
        if pick is None:
            warn(f"call log: no recording process started at or after this run's start ({_utc(t0)}); no call is used")
            return []
        if len(segs) > 1:
            before = sum(e - a for a, e in segs if a < pick[0])
            after = sum(e - a for a, e in segs if a > pick[0])
            if before:
                warn(f"call log: {before} calls of an earlier recording process (before this run started at {_utc(t0)}) left out")
            if after:
                warn(f"call log: {after} calls of a later recording process left out")
        keep = [k for k in range(*pick) if calls[k].get("t_start") is None or calls[k]["t_start"] >= t0 - 60]
        if len(keep) < pick[1] - pick[0]:
            warn(f"call log: {pick[1] - pick[0] - len(keep)} calls started before this run ({_utc(t0)}); left out")
        return [calls[k] for k in keep]
    if len(segs) > 1:
        vs = set(views)
        score = [sum(1 for k in range(*s) if calls[k].get("prompt") in vs) for s in segs]
        best = max(range(len(segs)), key=lambda i: (score[i], i))
        warn(f"call log: {len(segs)} recording processes and no run start time; using the one whose prompts match the moves "
             f"({score[best]} calls), {len(calls) - (segs[best][1] - segs[best][0])} calls left out")
        return [calls[k] for k in range(*segs[best])]
    return calls


def attach_calls(steps, calls, views, judge):
    """Tie calls to steps in time order. A step's window is the run of consecutive calls, from the pointer, whose prompt is
    exactly the step's view (`views`, the prompt the recorder sent; it resends it unchanged after a rejected answer); only
    when no call has that prompt is the window the run of calls of the same seat and round (read from the prompt; unknown
    values agree with anything). `judge(answer, step)` is the env's own check of an answer, (canonical action or None, why).
    For a fallback (step['fb'] = (why, skill)) no answer was usable: the window's last call is shown, with the reason. Else the
    call whose answer the game accepts as the logged action (step['canon']) is the step's call; earlier calls in the window
    were rejected attempts, kept above it with the reason. Without such an answer the window's last call is used and the step
    says that the game applied another action. A call whose reasoning hit the output limit (finish 'length') ends with a
    visible cut mark; when its action came from a follow-up question (rescued), a note saying so leads its reasoning.
    Returns (tied, rescued, retried, limits, tied by seat and round instead of prompt)."""
    ptr, tied, rescued, retried, limits, loose = 0, 0, 0, 0, [], 0
    index = {}
    for k, c in enumerate(calls):
        index.setdefault(c.get("prompt"), []).append(k)

    def fits(k, s, strict=False):
        seat, r = calls[k].get("seat"), round_of(calls[k].get("prompt"))
        if strict:
            return seat == s["seat"] and r == s["round"]
        return seat in (None, s["seat"]) and r in (None, s["round"])
    for s, v in zip(steps, views):
        ks = [k for k in index.get(v, []) if k >= ptr] if v else []
        if ks:
            if ks[0] > ptr:
                warn(f"calls {ptr + 1}..{ks[0]} of the log match no decision; skipped")
            ptr = j = ks[0]
            while j < len(calls) and calls[j].get("prompt") == v:
                j += 1
            if any(k >= j for k in ks):
                warn(f"P{s['seat']} round {s['round']}: its prompt was sent again later in the log; only the first calls are used")
        else:
            if ptr < len(calls) and not fits(ptr, s):
                ahead = next((k for k in range(ptr, min(len(calls), ptr + 40)) if fits(k, s, strict=True)), None)
                if ahead is None:
                    warn(f"no call for P{s['seat']} round {s['round']}")
                    continue
                warn(f"calls {ptr + 1}..{ahead} of the log match no decision; skipped")
                ptr = ahead
            j = ptr
            while j < len(calls) and fits(j, s):
                j += 1
            if j > ptr:
                loose += 1
                warn(f"P{s['seat']} round {s['round']}: no call has its prompt; tied by seat and round")
        window = list(range(ptr, j))
        if not window:
            continue
        verdict = {k: judge(calls[k].get("answer"), s) for k in window}
        if s.get("fb"):
            hit, use = None, window[-1]
            ok = [k for k in window if verdict[k][0] is not None]
            if ok:
                warn(f"P{s['seat']} round {s['round']}: the recorder flags a fallback, yet the game's check accepts the answer of "
                     f"call {ok[0] + 1} of the log")
            if s.get("reply") is not None and (calls[use].get("answer") or "")[:REPLY_CHARS] != s["reply"]:
                warn(f"P{s['seat']} round {s['round']}: the last call's answer is not the reply the recorder logged")
        else:
            want = s.get("canon")
            hit = next((k for k in window if want is not None and verdict[k][0] == want), None)
            if hit is None and want is None:               # an unreadable move: the call whose answer is the logged text
                hit = next((k for k in window if (calls[k].get("answer") or "").strip() == (s["act"] or "").strip()), None)
            use = hit if hit is not None else window[-1]
            if use < window[-1]:
                warn(f"P{s['seat']} round {s['round']}: {window[-1] - use} more calls after the answer the game took")

        def rejected(k):
            why = verdict[k][1]
            return f"rejected by the game: {why}; " if why else ""
        parts = [f"[Earlier attempt, {rejected(k)}answer: {(calls[k].get('answer') or '(none)')[:300]}]\n"
                 f"{calls[k].get('reasoning') or '(no reasoning returned)'}" for k in range(window[0], use)]
        if use > window[0]:
            retried += 1
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
        if s.get("fb"):
            why = verdict[use][1] or s["fb"][0]
            parts.append(f"[The game could not use this answer ({why}) and applied its fallback: train {s['fb'][1]}, the skill "
                         f"with the highest reputation. The answer was: {(c.get('answer') or '(none)')[:300]}]")
        elif hit is None:
            parts.append(f"[The game applied another action than this answer: {(c.get('answer') or '(none)')[:300]}]")
            warn(f"P{s['seat']} round {s['round']}: no call answered the logged action {s['act'][:120]}")
        s["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        s["unmatched"] = hit is None and not s.get("fb")
        if c.get("prompt"):
            s["state"] = en(common.state_part(c["prompt"]))
            s["from_prompt"] = True
        ptr = j if ks else use + 1
        tied += 1
    if ptr < len(calls):
        warn(f"{len(calls) - ptr} calls after the last decision were not tied to any step")
    return tied, rescued, retried, limits, loose


def fit_thinking(steps, budget, full):
    """Keep the page within the size its host takes: when all reasoning together is longer than `budget` characters, shorten the
    longest ones (one common cap, the largest that fits) by removing their middle, with a visible mark. Always cuts from each
    step's full reasoning (`full`), so repeated calls do not cut a cut text. Returns (steps shortened, characters left out)."""
    lens = [len(t) for t in full]
    if sum(lens) <= budget:
        for s, t in zip(steps, full):
            s["thinking"] = t
        return 0, 0
    lo, hi = 1500, max(lens)
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


SHARED_BLOCKS = ("market", "lb", "listings")


def share_round_blocks(steps):
    """Each round's market activity, leaderboard and listings are the same in every seat's prompt (the leaderboard up to its
    '(you)' mark). Keep them in full in the observation of the round's first decision and replace them in the others by a
    one-line pointer (the leaderboard's keeps the seat's own place), only where they are identical. Returns characters saved."""
    first, saved = {}, 0
    for s in steps:
        text = s.get("state") or ""
        pos = sorted((m.start(), k) for k, rx in HEADERS for m in [re.search(rx, text)] if m)
        blocks = {k: (p, pos[i + 1][0] if i + 1 < len(pos) else len(text)) for i, (p, k) in enumerate(pos) if k in SHARED_BLOCKS}
        if s["x"] not in first:
            first[s["x"]] = (s["seat"], {k: text[a:b] for k, (a, b) in blocks.items()})
            continue
        ref_seat, ref = first[s["x"]]
        where = f"the same as in P{ref_seat}'s prompt this round, shown in full at the round's first decision"
        edits = []
        for k, (a, b) in blocks.items():
            blk = text[a:b]
            if k not in ref or (blk.replace(" (you)", "") != ref[k].replace(" (you)", "") if k == "lb" else blk != ref[k]):
                continue
            head = re.sub(r" \([^)]*\)$", "", blk.split("\n", 1)[0].split(":", 1)[0])
            line = f"{head}: [{where}"
            if k == "lb":
                ents = list(RX_LB.finditer(blk[blk.find(":") + 1:]))
                me = next((i for i, x in enumerate(ents) if x.group(2)), None)
                if me is not None:
                    line += f"; you are {ents[me].group(1)}, rank {me + 1} of {len(ents)}, ${ents[me].group(3)}"
            line += ".]"
            if len(line) < len(blk.rstrip()):
                edits.append((a, b, line + blk[len(blk.rstrip()):]))
        for a, b, new in sorted(edits, reverse=True):
            saved += (b - a) - len(new)
            text = text[:a] + new + text[b:]
        s["state"] = text
    return saved


# ------------------------------------------------------------------------------------------------ build
def build(src, calls=None):
    b = load_source(src)
    if isinstance(calls, str):
        calls = load_calls(calls)
    calls = calls or []
    events, summary, run = b.get("events") or [], b.get("summary") or {}, b.get("run") or {}
    info = summary.get("info") or {}
    test_build = bool(run.get("test_bundle_from_calls"))
    moves = [e for e in events if e.get("event") == "move"]
    if not moves:
        raise SystemExit("no move events in " + str(src if not isinstance(src, dict) else "the bundle"))
    F = [view_facts(e.get("view") or "") for e in moves]
    P = read_params(info, F)
    skills = skill_names(P, F)
    TL = Tally()
    calls = select_calls(calls, b, [e.get("view") or "" for e in moves])
    # Does the bundle say which moves were the game's fallback? demo_training flags each one ("fallback": true, "reply") and
    # counts them in summary.json and run.json; the test helper's stand-in says so when it read the recorder's run.log.
    has_flags = (any("fallback" in e for e in moves) or _is_num(summary.get("fallbacks"))
                 or _is_num((run.get("extras") or {}).get("fallbacks")) or bool(run.get("fallback_flags")))

    # ---- the decisions: seat and round of every move
    mv, last_seat, cur = [], None, 0
    for e, f in zip(moves, F):
        s = e.get("actor") if e.get("actor") is not None else f["seat"]
        if f["seat"] is not None and s != f["seat"]:
            warn(f"move actor P{s} but its view is seat P{f['seat']}")
        t = f["round"]
        if t is None:                                      # by seat order: a round ends when the seat number falls
            t = cur + 1 if last_seat is None or s <= last_seat else cur
        cur, last_seat = t, s
        mv.append((s, t))
    T = max(t for _, t in mv)
    seats = sorted({s for s, _ in mv})
    label_of = {}
    for (s, _), f in zip(mv, F):
        if f["label"]:
            if s in label_of and label_of[s] != f["label"]:
                warn(f"P{s} is called {label_of[s]} and {f['label']}")
            label_of.setdefault(s, f["label"])
    for s in seats:
        label_of.setdefault(s, f"P{s}")
    seat_of_label = {v: k for k, v in label_of.items()}
    per_round = {}
    for s, t in mv:
        per_round.setdefault(t, []).append(s)
    for t in range(1, T + 1):
        got = per_round.get(t, [])
        if t < T and sorted(got) != seats:
            warn(f"round {t}: decisions by {got}, expected one by each of {seats}")
    partial_last = sorted(per_round.get(T, [])) != seats

    # ---- summary: agents and per-round series
    ag_info = {}
    for a in info.get("agents") or []:
        if isinstance(a, dict) and a.get("label"):
            ag_info[a["label"]] = a
    for L, a in ag_info.items():
        if a.get("seat") is not None:
            TL.eq("names", seat_of_label.get(L) == a["seat"], f"summary: {L} is seat {a['seat']}, prompts say {seat_of_label.get(L)}")
    played = info.get("rounds_played")
    if played is not None and played not in (T, T - 1):
        warn(f"summary rounds_played {played}, the moves reach round {T}")
    final_from_summary = bool(ag_info) and played == T and not partial_last
    if played == T - 1 and ag_info:
        warn(f"the summary stops after round {T - 1}; round {T} has decisions but no outcome")

    # ---- what the prompts report, round by round; every report of the same thing must agree
    listings, lb, alloc, own, rep0, notes_seen, record = {}, {}, {}, {}, {}, {}, {}
    round_lines = {}
    for (s, t), f in zip(mv, F):
        if f["listings"] is not None:
            if t in listings:
                TL.eq("listings", listings[t] == f["listings"], f"round {t}: listings differ between prompts")
            else:
                listings[t] = f["listings"]
        if f["lb"]:
            if t in lb:
                TL.eq("leaderboard", lb[t] == f["lb"], f"round {t}: leaderboards differ between prompts")
            else:
                lb[t] = f["lb"]
            if f.get("lb_you") is not None:
                TL.eq("names", f["lb_you"] == label_of[s], f"P{s} round {t}: '(you)' marks {f['lb_you']}, not {label_of[s]}")
        for r, ents in f["market"].items():
            if r in alloc:
                TL.eq("market", alloc[r] == ents, f"round {r}: the public record differs between prompts")
            else:
                alloc[r] = ents
        for r, h in f["own"].items():
            if (s, r) in own:
                TL.eq("own history", own[(s, r)]["raw"] == h["raw"], f"P{s} round {r}: its history line differs between prompts")
            else:
                own[(s, r)] = h
        rep0[(s, t)] = f["rep"]
        notes_seen[(s, t)] = f["note"]
        record[(s, t)] = f["record"]
        round_lines[t] = f["round_line"]
    all_labels = []
    for t in sorted(lb):
        for L in lb[t]:
            if L not in all_labels:
                all_labels.append(L)
    for L in ag_info:
        if L not in all_labels:
            all_labels.append(L)
    for s in seats:
        if label_of[s] not in all_labels:
            all_labels.append(label_of[s])
    scripted = sorted(L for L in all_labels if L not in seat_of_label)
    N = len(all_labels)
    if N != P["agents"]:
        warn(f"{N} agents named in the record, {P['agents']} in the settings")
    recession = {t for t, line in round_lines.items() if "recession" in (line or "").lower()}

    # ---- each seat's decisions: the move, the action the game applied (its history line) and the outcome
    REC = {}
    skill_of = lambda job: job.rstrip("0123456789")
    certain_feedback = (P.get("fallback_train_prob") if _is_num(P.get("fallback_train_prob")) else 1.0) >= 1.0
    for (s, t), e, f in zip(mv, moves, F):
        raw = e.get("move")
        raw_s = raw if isinstance(raw, str) else json.dumps(raw)
        a = parse_action(raw)
        listed = listings.get(t) or {}
        h = own.get((s, t))
        if has_flags:
            # the recorder logs the action the game applied, in its canonical form: read it, do not judge it again (the
            # env checks the raw price and then rounds it, so a logged price can sit just past ten times the budget)
            canon, why = canonical_of(a), (None if a is not None else "unreadable logged move")
        elif t in listings or a is None or a["type"] == "train":
            canon, why = env_check(raw, listed, P, skills)    # a stand-in's move is a raw answer: as the game reads it
        else:
            canon, why = None, None
        flagged = bool(e.get("fallback"))
        fb_why = None
        if flagged:                                         # the recorder's flag; the reason from the reply it logged
            reply = e.get("reply")
            _, fb_why = env_check(reply, listed, P, skills) if isinstance(reply, str) else (None, "no usable answer")
            if fb_why is None:
                warn(f"P{s} round {t}: flagged as the game's fallback, yet its logged reply passes the game's check")
                fb_why = "rejected by the game"
            elif isinstance(reply, str) and len(reply) >= REPLY_CHARS and fb_why.startswith("not plain JSON"):
                fb_why = f"not a usable action (the reply is logged only to its first {REPLY_CHARS:,} characters)"
        r = {"seat": s, "t": t, "raw": raw_s or "", "move": a, "canon": canon, "listed": listed, "rep0": f["rep"],
             "note": a["note"] if a else None, "type": None, "bids": [], "skill": None, "income": None, "feedback": None,
             "rep_change": {}, "known": False, "fallback": flagged, "why": fb_why, "flagged": flagged,
             "reply": e.get("reply") if flagged and isinstance(e.get("reply"), str) else None, "source": None, "won_n": None}
        if (s, t) in REC:
            warn(f"P{s} round {t}: a second decision; the later one is used")
        if h and h["type"]:
            r["known"], r["source"] = True, "prompt"
            if h["type"] == "TRAIN":
                r["type"], r["skill"] = "train", h["skill"]
                same = a is not None and a["type"] == "train" and a["skill"] == h["skill"]
                if not same and not flagged:                # the game trained instead of the logged move: its fallback
                    r["fallback"], r["why"] = True, why or "rejected by the game"
                    if has_flags:
                        warn(f"P{s} round {t}: the game trained {h['skill']} instead of the logged move, which is not flagged "
                             "as a fallback")
                TL.eq("moves", same or (why is not None and not has_flags),
                      f"P{s} round {t}: move {raw_s[:100]} but the game trained {h['skill']}")
            else:
                r["type"] = "bid"
                if flagged:                                 # the game's own history line shows the bids it applied
                    warn(f"P{s} round {t}: flagged as a fallback, yet its history shows bids; shown as bids")
                    r["fallback"], r["why"] = False, None
                hb = h["bids"]
                same = a is not None and a["type"] == "bid" and a["bids"] == [(x["job"], round(x["price"], 2)) for x in hb]
                TL.eq("moves", same, f"P{s} round {t}: move {raw_s[:100]} but its history shows {h['raw'][:100]}")
                for x in hb:
                    bud = listed.get(x["job"])
                    if bud is not None:
                        TL.eq("budgets", f"{bud:.1f}" == x["budget1"] or abs(bud - num(x["budget1"])) < 0.05 + 1e-9,
                              f"P{s} round {t} {x['job']}: listed {bud}, history {x['budget1']}")
                    r["bids"].append({"job": x["job"], "price": round(x["price"], 2), "budget": bud if bud is not None else num(x["budget1"]),
                                      "won": x["won"]})
                r["won_n"] = sum(1 for x in hb if x["won"])
            r["income"], r["feedback"], r["rep_change"] = h["income"], h["feedback"], h["rep"]
        elif t in alloc and t + 1 in lb and label_of[s] in lb[t + 1] and a is not None and why is None:
            # the seat's own next prompt is missing (a round cut short) but another seat's reports the public record and the
            # leaderboard after this round: the outcome follows from them
            r["known"], r["source"] = True, "public"
            r["income"] = round(lb[t + 1][label_of[s]] - lb[t].get(label_of[s], 0.0), 2) if t in lb else None
            pub = {x["job"] for x in alloc[t] if x["winner"] == label_of[s]}
            if a["type"] == "train":
                r.update(type="train", skill=a["skill"])
                TL.eq("wins", not pub, f"P{s} round {t}: trained, yet the public record shows wins {sorted(pub)}")
            else:
                r["type"] = "bid"
                r["bids"] = [{"job": j, "price": p, "budget": listed.get(j), "won": j in pub} for j, p in a["bids"]]
                r["won_n"] = len(pub)
                if not pub and certain_feedback:            # rules: Round Procedure 6, the first bid's skill
                    r["feedback"] = skill_of(a["bids"][0][0])
                TL.eq("wins", pub <= {j for j, _ in a["bids"]}, f"P{s} round {t}: public wins {sorted(pub)} not among its bids")
        else:                                              # no later prompt reports this round (the last one)
            if (why is not None or a is None) and not flagged:  # the game cannot have applied it: its fallback
                best = max(skills, key=lambda k: (f["rep"].get(k, -1), -skills.index(k))) if f["rep"] else skills[0]
                r.update(type="train", skill=best, fallback=True, why=why or "unreadable answer")
                if has_flags:
                    warn(f"P{s} round {t}: the logged move is not one the game accepts ({why}), yet it is not flagged as a fallback")
            elif a is not None and a["type"] == "train":
                r.update(type="train", skill=a["skill"])
            elif a is not None:
                r["type"] = "bid"
                r["bids"] = [{"job": j, "price": p, "budget": listed.get(j), "won": None} for j, p in a["bids"]]
                if flagged:
                    warn(f"P{s} round {t}: flagged as a fallback, yet the logged move is a bid; shown as the bid")
                    r["fallback"], r["why"] = False, None
            else:                                          # flagged, but the logged move is unreadable
                best = max(skills, key=lambda k: (f["rep"].get(k, -1), -skills.index(k))) if f["rep"] else skills[0]
                r.update(type="train", skill=best)
                warn(f"P{s} round {t}: flagged as a fallback with an unreadable logged move; shown as training {best}")
        REC[(s, t)] = r
    # the recorder's fallbacks: as many as it counted, each one training the skill the seat was shown with the top reputation
    n_flag = sum(1 for r in REC.values() if r["flagged"])
    for src_, v in (("summary.json", summary.get("fallbacks")), ("run.json", (run.get("extras") or {}).get("fallbacks"))):
        if _is_num(v):
            TL.eq("fallbacks", int(v) == n_flag, f"{src_} counts {v} fallbacks, {n_flag} moves are flagged")
    for r in REC.values():
        if r["fallback"] and r["type"] == "train" and r["rep0"] and r["skill"] in r["rep0"]:
            top = max(r["rep0"].values())
            TL.eq("fallbacks", r["rep0"][r["skill"]] >= top - 1e-9,
                  f"P{r['seat']} round {r['t']}: fallback trained {r['skill']} at {r['rep0'][r['skill']]}*, but the seat was "
                  f"shown {top}* on another skill")

    # ---- scripted agents: wins from the public record, income from the leaderboard; policy from the summary or the price
    def lb_inc(L, t):
        if t in lb and t + 1 in lb and L in lb[t] and L in lb[t + 1]:
            return lb[t + 1][L] - lb[t][L]
        return None
    pol, pol_src = {}, {}
    measured = {}
    for L in scripted:
        won_b, got = 0.0, 0.0
        for t in range(1, T):
            ents = [x for x in alloc.get(t, []) if x["winner"] == L]
            inc = lb_inc(L, t)
            if ents and inc is not None and all(x["job"] in (listings.get(t) or {}) for x in ents):
                won_b += sum(listings[t][x["job"]] for x in ents)
                got += inc
        measured[L] = got / won_b if won_b > 0 else None
        p = (ag_info.get(L) or {}).get("policy")
        if p:
            pol[L], pol_src[L] = p, "summary"
    cands = list(dict.fromkeys(P["scripted_list"])) or ["specialist", "greedy"]
    for L in scripted:
        if L in pol:
            continue
        m = measured[L]
        fit = [p for p in cands if m is not None and alpha_of(p, P) is not None and abs(alpha_of(p, P) - m) < 0.006]
        if len(fit) == 1:
            pol[L], pol_src[L] = fit[0], "price"
        else:
            pol[L], pol_src[L] = None, None
            warn(f"{L}: policy unknown (summary silent; its wins were paid {m} of their budgets)")
    for L in scripted:
        a_ = alpha_of(pol[L], P) if pol[L] else None
        if a_ is not None and measured[L] is not None and pol_src[L] == "summary":
            TL.eq("scripted prices", abs(measured[L] - a_) < 0.006, f"{L} ({pol[L]}): paid {measured[L]:.4f} of its won budgets, rules {a_}")
    cnt = {}
    sname = {}
    for L in scripted:
        base = POLICY_NAME.get(pol[L], (pol[L] or "").capitalize()) if pol[L] else None
        cnt[base] = cnt.get(base, 0) + 1
    seen = {}
    for L in scripted:
        base = POLICY_NAME.get(pol[L], (pol[L] or "").capitalize()) if pol[L] else None
        if base is None:
            sname[L] = f"Scripted {L}"
        else:
            seen[base] = seen.get(base, 0) + 1
            sname[L] = base + (f" {seen[base]}" if cnt[base] > 1 else "")
    seat_name = {s: f"Worker P{s}" for s in seats}
    name_of = {label_of[s]: seat_name[s] for s in seats}
    name_of.update(sname)

    def scripted_price(L, t, job):
        a_ = alpha_of(pol.get(L), P) if pol.get(L) else None
        bud = (listings.get(t) or {}).get(job)
        return round(a_ * bud, 2) if a_ is not None and bud is not None else None

    # ---- checks across sources, rounds 1 .. T-1 (and the last round where the summary covers it)
    for (s, t), r in sorted(REC.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        L = label_of[s]
        if not r["known"]:
            continue
        own_line = r["source"] == "prompt"                  # the seat's own history line (else derived from the public record)
        if own_line and t in alloc and r["type"] == "bid":
            pub = sorted(x["job"] for x in alloc[t] if x["winner"] == L)
            mine = sorted(b_["job"] for b_ in r["bids"] if b_["won"])
            TL.eq("wins", pub == mine, f"P{s} round {t}: history wins {mine}, public record {pub}")
            for x in alloc[t]:
                if x["winner"] == L and x["rep"] is not None and r["rep0"].get(x["job"].rstrip("0123456789")) is not None:
                    TL.eq("reputation", abs(x["rep"] - r["rep0"][x["job"].rstrip("0123456789")]) < 1e-9,
                          f"P{s} round {t} {x['job']}: public reputation {x['rep']}, its prompt {r['rep0']}")
        elif own_line and t in alloc:
            TL.eq("wins", not any(x["winner"] == L for x in alloc[t]), f"P{s} round {t}: trained but the public record shows a win")
        inc = lb_inc(L, t)
        if inc is not None and r["income"] is not None:
            TL.eq("income", abs(inc - r["income"]) < 0.0051, f"P{s} round {t}: history income {r['income']}, leaderboard {inc:.2f}")
        if r["type"] == "bid" and r["income"] is not None and not P.get("performance_pay"):
            TL.eq("income", abs(sum(b_["price"] for b_ in r["bids"] if b_["won"]) - r["income"]) < 0.0051,
                  f"P{s} round {t}: income {r['income']} but winning prices sum to {sum(b_['price'] for b_ in r['bids'] if b_['won'])}")
        for k, (bef, aft) in (r["rep_change"] or {}).items():
            if k in r["rep0"]:
                TL.eq("reputation", abs(r["rep0"][k] - bef) < 1e-9, f"P{s} round {t}: reputation {k} {r['rep0'][k]} shown, {bef} in history")
            nxt = rep0.get((s, t + 1))
            if nxt and k in nxt:
                TL.eq("reputation", abs(nxt[k] - (aft if aft is not None else bef)) < 1e-9,
                      f"P{s} round {t}: reputation {k} after the round {aft or bef}, next prompt {nxt[k]}")
        if own_line and r["type"] == "bid" and r["won_n"] == 0 and P.get("fallback_train_prob", 1.0) >= 1.0 and r["bids"]:
            TL.eq("feedback", r["feedback"] == r["bids"][0]["job"].rstrip("0123456789"),
                  f"P{s} round {t}: won nothing; feedback training {r['feedback']}, first bid {r['bids'][0]['job']}")
    for L in scripted:
        for t in range(1, T):
            inc = lb_inc(L, t)
            ents = [x for x in alloc.get(t, []) if x["winner"] == L]
            if inc is None or t not in alloc:
                continue
            ps = [scripted_price(L, t, x["job"]) for x in ents]
            if all(p is not None for p in ps) and not P.get("performance_pay"):
                TL.eq("scripted prices", abs(sum(ps) - inc) < 0.0051 + 0.0051 * len(ps),
                      f"{L} round {t}: income {inc:.2f}, its fixed price on {[x['job'] for x in ents]} gives {sum(ps):.2f}")
    for t in sorted(lb):                                   # total earnings of everyone add up from the incomes
        if t + 1 in lb and t in alloc and not P.get("performance_pay"):
            pay = sum(lb[t + 1][L] - lb[t][L] for L in lb[t] if L in lb[t + 1])
            parts = []
            for x in alloc[t]:
                if x["winner"] in seat_of_label:
                    rr = REC.get((seat_of_label[x["winner"]], t))
                    parts.append(next((b_["price"] for b_ in (rr or {}).get("bids", []) if b_["job"] == x["job"]), None))
                else:
                    parts.append(scripted_price(x["winner"], t, x["job"]))
            if all(p is not None for p in parts):
                TL.eq("payments", abs(sum(parts) - pay) < 0.006 * max(1, len(parts)),
                      f"round {t}: the leaderboard grew by {pay:.2f}, the winning prices sum to {sum(parts):.2f}")

    # ---- the last round from the summary's final totals (it appears in no prompt)
    final_E, final_note, final_bad = {}, [], []
    last_n, last_unique, narrowed = 0, 0, []
    if final_from_summary:
        for L in all_labels:
            a = ag_info.get(L) or {}
            if _is_num(a.get("earnings")):
                final_E[L] = float(a["earnings"])
        lbT = lb.get(T, {})

        def ratio_sum_before(s):
            """The seat's sum of price / posted budget over its wins before round T, with the exact (cent) budgets."""
            tot = 0.0
            for (ss, t), rr in REC.items():
                if ss == s and t < T:
                    for b_ in rr["bids"]:
                        if b_["won"]:
                            bud = (listings.get(t) or {}).get(b_["job"])
                            if not bud:
                                return None
                            tot += b_["price"] / bud
            return tot
        for s in seats:
            L, r = label_of[s], REC.get((s, T))
            a = ag_info.get(L) or {}
            if r is None or L not in final_E or L not in lbT:
                continue
            last_n += 1
            inc = round(final_E[L] - lbT[L], 2)
            before = sum(1 for (ss, t), rr in REC.items() if ss == s and t < T for b_ in rr["bids"] if b_["won"])
            by_before = {}
            for (ss, t), rr in REC.items():
                if ss == s and t < T:
                    for b_ in rr["bids"]:
                        if b_["won"]:
                            k = skill_of(b_["job"])
                            by_before[k] = by_before.get(k, 0) + 1
            jw = a.get("jobs_won")
            k_T = int(jw) - before if _is_num(jw) else None
            wbs = per_skill(a.get("wins_by_skill"), skills)
            need = {k: int(round(v - by_before.get(k, 0))) for k, v in wbs.items()} if wbs else None
            # the round's sum of price / budget over the seat's wins: the summary's mean winning price (the mean of price /
            # posted budget over all its wins) times its jobs won, minus the exact ratios of its earlier wins
            rb, mwp = ratio_sum_before(s), a.get("mean_winning_price")
            target = None
            if _is_num(jw) and rb is not None:
                target = 0.0 - rb if int(jw) == 0 else (mwp * int(jw) - rb if _is_num(mwp) else None)
            if r["type"] == "train":                       # trained (or the game's fallback training): nothing won
                TL.eq("last round", abs(inc) < 0.0051 and k_T in (None, 0) and (target is None or abs(target) < 1e-6),
                      f"P{s} round {T}: trained, yet income {inc} / jobs {k_T} / winning-price sum {target}")
                r.update(known=True, source="summary", income=inc)
                last_unique += 1
                continue
            bids = r["bids"]
            fits = []
            for n_ in range(0, min(len(bids), int(P["capacity"])) + 1):
                if k_T is not None and n_ != k_T:
                    continue
                for comb in itertools.combinations(range(len(bids)), n_):
                    if abs(sum(bids[i]["price"] for i in comb) - inc) > 0.0051 and not P.get("performance_pay"):
                        continue
                    if need is not None:
                        got = {}
                        for i in comb:
                            k = skill_of(bids[i]["job"])
                            got[k] = got.get(k, 0) + 1
                        if any(got.get(k, 0) != v for k, v in need.items() if v) or any(v and not need.get(k) for k, v in got.items()):
                            continue
                    fits.append(comb)
            by_ratio = target is not None and all(b_["budget"] for b_ in bids)
            if by_ratio:
                fits_r = [c for c in fits if abs(sum(bids[i]["price"] / bids[i]["budget"] for i in c) - target) < 1e-8]
                if len(fits) > 1 and len(fits_r) == 1:
                    narrowed.append(f"P{s}")
                if fits and not fits_r:
                    warn(f"P{s} round {T}: the subsets that fit its income and jobs by skill give no winning-price sum of {target:.6f}")
                fits = fits_r
            r.update(known=True, source="summary", income=inc, won_n=k_T)
            TL.eq("last round", len(fits) >= 1, f"P{s} round {T}: no subset of its bids gives income {inc} with {k_T} jobs {need}"
                  + (f" and winning-price sum {target:.6f}" if by_ratio else ""))
            if len(fits) == 1 and (by_ratio or not P.get("performance_pay")):
                for i, b_ in enumerate(bids):
                    b_["won"] = i in fits[0]
                r["won_n"] = len(fits[0])
                last_unique += 1
            else:
                if k_T is None and fits and len({len(c) for c in fits}) == 1:
                    r["won_n"] = len(fits[0])
                if target is not None and r["won_n"]:
                    r["ratio_sum"] = target                 # the round's mean winning price is known even so
                (final_note if fits else final_bad).append(f"P{s}")
            # rules: Round Procedure 6 -- a seat that bid and won nothing is trained on its first bid's skill (certain when
            # fallback_train_prob is 1; else the summary's count says whether it happened)
            if r["won_n"] == 0 and bids:
                fb_before = sum(1 for (ss, t), rr in REC.items() if ss == s and t < T and rr["feedback"])
                if certain_feedback or (_is_num(a.get("fallback_trainings")) and int(a["fallback_trainings"]) - fb_before == 1):
                    r["feedback"] = skill_of(bids[0]["job"])
        for s in seats:                                    # feedback trainings: history lines plus the last round = summary
            ft = (ag_info.get(label_of[s]) or {}).get("fallback_trainings")
            if _is_num(ft) and (s, T) in REC and REC[(s, T)]["known"]:
                got = sum(1 for (ss, t), rr in REC.items() if ss == s and rr["feedback"])
                TL.eq("feedback totals", got == int(ft),
                      f"P{s}: {got} feedback trainings rebuilt (history lines and round {T}), summary fallback_trainings {ft}")
        scripted_T = {}
        for L in scripted:
            a = ag_info.get(L) or {}
            if L in final_E and L in lbT:
                before = sum(1 for t in range(1, T) for x in alloc.get(t, []) if x["winner"] == L)
                k_T = int(a["jobs_won"]) - before if _is_num(a.get("jobs_won")) else None
                scripted_T[L] = (k_T, round(final_E[L] - lbT[L], 2))
        lb_final = final_E if len(final_E) == len(all_labels) else None
    else:
        scripted_T, lb_final = {}, None

    # ---- per round, per agent
    R_known = [t for t in range(1, T + 1) if (t + 1 in lb) or (t == T and lb_final is not None)]
    known_set = set(R_known)

    def cum_after(L, t):
        if t + 1 in lb and L in lb[t + 1]:
            return lb[t + 1][L]
        if t == T and lb_final is not None:
            return lb_final.get(L)
        return None

    def income_of(L, t):
        a_ = cum_after(L, t)
        b_ = lb.get(t, {}).get(L)
        return None if a_ is None or b_ is None else round(a_ - b_, 2)

    def wins_of(L, t):
        """[(job, price or None, budget)] of agent L in round t; None when unknown."""
        if t in alloc:
            out = []
            for x in alloc[t]:
                if x["winner"] != L:
                    continue
                bud = (listings.get(t) or {}).get(x["job"])
                if L in seat_of_label:
                    rr = REC.get((seat_of_label[L], t)) or {}
                    p = next((b_["price"] for b_ in rr.get("bids", []) if b_["job"] == x["job"]), None)
                else:
                    p = x["price"] if x["price"] is not None else scripted_price(L, t, x["job"])
                out.append((x["job"], p, bud))
            return out
        if t == T and final_from_summary:
            if L in seat_of_label:
                rr = REC.get((seat_of_label[L], T)) or {}
                if rr.get("type") == "bid" and all(b_["won"] is not None for b_ in rr["bids"]):
                    return [(b_["job"], b_["price"], b_["budget"]) for b_ in rr["bids"] if b_["won"]]
                if rr.get("type") == "train":
                    return []
            return None
        return None

    def n_wins(L, t):
        """How many jobs agent L won in round t: its wins, else in the last round the count the summary gives (a model worker
        whose winning bids are not unique, a scripted agent); None when unknown."""
        w = wins_of(L, t)
        if w is not None:
            return len(w)
        if t == T and final_from_summary:
            if L in seat_of_label:
                return (REC.get((seat_of_label[L], T)) or {}).get("won_n")
            return scripted_T.get(L, (None, None))[0]
        return None

    seat_bids = {t: [b_ for s in seats for b_ in (REC.get((s, t)) or {}).get("bids", []) if b_["budget"]] for t in range(1, T + 1)}
    med_bid = [100 * statistics.median([b_["price"] / b_["budget"] for b_ in seat_bids[t]]) if seat_bids[t] else None for t in range(1, T + 1)]
    low_bid = [100 * min(b_["price"] / b_["budget"] for b_ in seat_bids[t]) if seat_bids[t] else None for t in range(1, T + 1)]
    win_all, win_seat, win_all_n = [], [], []
    for t in range(1, T + 1):
        rs, ws, complete = [], [], True
        for L in all_labels:
            w = wins_of(L, t)
            if w is None:
                if t == T and final_from_summary and L in scripted_T:
                    # a scripted agent's last-round wins: a count from the summary, each at its fixed share of the budget
                    k_T, a_ = scripted_T[L][0], (alpha_of(pol.get(L), P) if pol.get(L) else None)
                    if k_T is None or (k_T and a_ is None):
                        complete = False
                    elif k_T:
                        rs += [a_] * k_T
                    continue
                rr = REC.get((seat_of_label[L], t)) if t == T and L in seat_of_label else None
                if rr and rr.get("ratio_sum") is not None and rr.get("won_n"):
                    # which bids won is not unique, but their sum of price / budget is known: its mean stands for each
                    # (the round's mean and every pooled mean are then exact)
                    rs += [rr["ratio_sum"] / rr["won_n"]] * rr["won_n"]
                    ws += [rr["ratio_sum"] / rr["won_n"]] * rr["won_n"]
                    continue
                complete = False
                continue
            for job, p, bud in w:
                if p is None or not bud:
                    complete = False
                    continue
                rs.append(p / bud)
                if L in seat_of_label:
                    ws.append(p / bud)
        win_all.append(100 * mean(rs) if rs and complete else None)
        win_all_n.append(rs if complete else None)
        win_seat.append(100 * mean(ws) if ws else None)
    pooled = lambda ts, src: [x for t in ts for x in (src[t - 1] or [])]

    # summary series checks
    ser = {k: series_get(info, k, T) for k in ("listed", "matched", "payments", "mean_winning_price", "trainers", "mean_performance",
                                                  "client_utility", "bidders", "recession")}
    for t in range(1, T + 1):
        if ser["listed"] and ser["listed"][t - 1] is not None and t in listings:
            TL.eq("summary series", int(ser["listed"][t - 1]) == len(listings[t]), f"round {t}: summary listed {ser['listed'][t - 1]}, prompts {len(listings[t])}")
        if ser["matched"] and ser["matched"][t - 1] is not None and t in alloc:
            TL.eq("summary series", int(ser["matched"][t - 1]) == len(alloc[t]), f"round {t}: summary matched {ser['matched'][t - 1]}, record {len(alloc[t])}")
        if ser["payments"] and ser["payments"][t - 1] is not None:
            pay = [income_of(L, t) for L in all_labels]
            if all(x is not None for x in pay):
                TL.eq("summary series", abs(sum(pay) - ser["payments"][t - 1]) < 0.006 * N,
                      f"round {t}: summary payments {ser['payments'][t - 1]}, leaderboard {sum(pay):.2f}")
        if ser["mean_winning_price"] and ser["mean_winning_price"][t - 1] is not None and win_all[t - 1] is not None:
            TL.eq("summary series", abs(win_all[t - 1] / 100 - ser["mean_winning_price"][t - 1]) < 0.0015,
                  f"round {t}: summary mean winning bid {ser['mean_winning_price'][t - 1]:.4f}, rebuilt {win_all[t - 1] / 100:.4f}")
    if final_from_summary and ser["matched"] and ser["matched"][T - 1] is not None:
        k_each = [n_wins(L, T) for L in all_labels]
        if all(k is not None for k in k_each):
            TL.eq("summary series", sum(k_each) == int(ser["matched"][T - 1]),
                  f"round {T}: {sum(k_each)} jobs rebuilt, summary matched {ser['matched'][T - 1]}")
        else:
            warn(f"round {T}: the jobs won by " + ", ".join(L for L, k in zip(all_labels, k_each) if k is None)
                 + " are not known; the summary's matched count is not checked")

    # record summary lines in the prompts against the rebuilt totals
    for (s, t), rc in record.items():
        if not rc:
            continue
        prev = [REC[(s, u)] for u in range(rc["from"], rc["to"] + 1) if (s, u) in REC]
        if len(prev) != rc["to"] - rc["from"] + 1 or any(not x["known"] for x in prev):
            continue
        jobs = sum(1 for x in prev for b_ in x["bids"] if b_["won"])
        nb = sum(len(x["bids"]) for x in prev if x["type"] == "bid")
        nbr = sum(1 for x in prev if x["type"] == "bid")
        ntr = sum(1 for x in prev if x["type"] == "train")
        nfb = sum(1 for x in prev if x["feedback"])
        TL.eq("record lines", (jobs, nb, nbr, ntr, nfb) == (rc["jobs"], rc["bids"], rc["bid_rounds"], rc["train"], rc["feedback"])
              and abs(rc["earnings"] - lb.get(t, {}).get(label_of[s], -1)) < 0.0051,
              f"P{s} round {t}: record line {rc}, rebuilt jobs {jobs} bids {nb} bid rounds {nbr} train {ntr} feedback {nfb}")

    # rewards
    over = next((e for e in events if e.get("event") == "game_over"), {})
    raw_rw = summary.get("rewards") or (run.get("result") or {}).get("rewards") or over.get("rewards") or {}
    rewards = {int(k): float(v) for k, v in raw_rw.items()}
    exp_round = expected_posted(P, skills)
    S_eq = info.get("equal_share") if _is_num(info.get("equal_share")) else sum(exp_round) / max(1, int(P["agents"]))
    if _is_num(info.get("equal_share")):
        mine = sum(exp_round) / max(1, int(P["agents"]))
        if abs(mine - S_eq) > 0.01:
            warn(f"equal share: summary {S_eq}, rules' formula {mine:.4f}")
    if lb_final:
        for s in seats:
            if s in rewards and label_of[s] in lb_final:
                want = max(-1.0, min(1.0, lb_final[label_of[s]] / S_eq - 1))
                TL.eq("rewards", abs(want - rewards[s]) < 1e-6, f"P{s}: reward {rewards[s]}, from earnings {want}")
        for L in all_labels:
            a = ag_info.get(L) or {}
            if L in seat_of_label and _is_num(a.get("reward")) and seat_of_label[L] in rewards:
                TL.eq("rewards", abs(a["reward"] - rewards[seat_of_label[L]]) < 1e-9, f"{L}: summary agent reward {a['reward']}")
    if lb_final is None and T >= int(P["rounds"]) and not test_build:
        warn("no final earnings in the summary: the last round's outcome is not shown")

    # ---- names and texts used throughout
    seat_names = [seat_name[s] for s in seats]
    pol_txt = {}
    for L in scripted:
        a_ = alpha_of(pol[L], P) if pol[L] else None
        pol_txt[L] = (POLICY_TXT.get(pol[L], "follows a fixed rule").format(a=pc(a_)) if pol[L] else "follows a fixed rule (its policy is not recorded)")
    def scr_short(L):
        a_ = alpha_of(pol[L], P) if pol[L] else None
        if not pol[L] or pol[L] not in POLICY_SHORT:
            return f"{L}, a scripted agent with a fixed rule"
        art, what = POLICY_SHORT[pol[L]]
        return art.format(n=name_of[L]) + " that " + what.format(a=pc(a_) if a_ is not None else "")
    who_txt = (", ".join(f"{seat_name[s]} is {label_of[s]}" for s in seats) + "; "
               + and_list(f"{L} is the scripted {name_of[L]}" for L in scripted) + ".") if scripted else \
        ", ".join(f"{seat_name[s]} is {label_of[s]}" for s in seats) + "."

    def short(L):
        return f"P{seat_of_label[L]}" if L in seat_of_label else name_of.get(L, L)

    # ---- steps and feed
    steps, feed = [], []
    rule_txt = (f"{nw(N).capitalize()} agents: {nw(len(seats))} model workers" + ("; scripted: " + and_list(f"{scr_short(L)} ({L})" for L in scripted) if scripted else ", no scripted agents")
                + f". {len(skills)} skills ({', '.join(skills)}), a catalog of {len(skills) * int(P['jobs_per_task'])} jobs with base budgets "
                + ", ".join(money(x, dp=0) for x in P["budgets"]) + f" (each listed with probability {P['listing_prob']:g} and some noise). "
                f"Each round a worker bids on up to {P['max_bids']} listed jobs or trains one skill; it can hold at most {P['capacity']} jobs, "
                "a won job pays its bid in full, and the client of each job prefers a higher reputation and a lower price relative to the "
                f"budget. Bids are sealed. {who_txt}")
    feed.append({"x": 0, "step": 0, "who": "Rules", "text": rule_txt})
    for (s, t), e in zip(mv, moves):
        r = REC[(s, t)]
        L = label_of[s]
        bids = r["bids"]
        rats = [b_["price"] / b_["budget"] for b_ in bids if b_["budget"]]
        rng = (f"{pc(min(rats))}" if len(rats) == 1 or round(100 * min(rats)) == round(100 * max(rats))
               else f"{100 * min(rats):.0f}{NDASH}{pc(max(rats))}") if rats else ""
        if r["fallback"]:
            kind = "fallback"
            head = f"Round {t} {DOT} answer not usable ({r['why']}); the game trained {r['skill']}"
        elif r["type"] == "train":
            kind = "train"
            ch = (r["rep_change"] or {}).get(r["skill"])
            head = f"Round {t} {DOT} trains {r['skill']}" + (f" {DOT} reputation {r['skill']} {st(ch[0])} {ARROW} {st(ch[1])}" if ch and ch[1] is not None
                                                            else (f" {DOT} reputation {r['skill']} stays {st(ch[0])}" if ch else ""))
        elif not r["known"] or any(b_["won"] is None for b_ in bids) and r["won_n"] is None:
            kind = "pending"
            head = f"Round {t} {DOT} bids on {plural(len(bids), 'job')} at {rng} of budget {DOT} outcome not recorded"
        else:
            k = r["won_n"] if r["won_n"] is not None else sum(1 for b_ in bids if b_["won"])
            kind = "won" if k else "lost"
            head = (f"Round {t} {DOT} bids on {plural(len(bids), 'job')} at {rng} of budget {DOT} "
                    + (f"wins {k} ({money(r['income'])})" if k else "wins none" + (f", feedback training on {r['feedback']}" if r["feedback"] else "")))
        steps.append({"seat": s, "x": t - 1, "head": head, "kind": kind, "act": en(r["raw"]),
                      "state": en(common.state_part(e.get("view") or "")) or "Not recorded.", "round": t, "canon": r["canon"],
                      "fb": (r["why"], r["skill"]) if r["fallback"] else None, "reply": r["reply"]})
        # feed: the decision
        if r["type"] == "train":
            ch = (r["rep_change"] or {}).get(r["skill"])
            txt = (f"Trains {r['skill']} (no income)" + (f" because its answer was unusable ({r['why']})" if r["fallback"] else "") + "."
                   + (f" Reputation {r['skill']} {st(ch[0])} {ARROW} {st(ch[1])}." if ch and ch[1] is not None else (f" Reputation {r['skill']} {st(ch[0])}, unchanged on the one-decimal display." if ch else "")))
        else:
            items = []
            for b_ in bids:
                o = "" if b_["won"] is None else (", won" if b_["won"] else ", lost")
                items.append(f"{b_['job']} {money(b_['price'])}" + (f" of {money(b_['budget'])} ({pc(b_['price'] / b_['budget'])})" if b_["budget"] else "") + o)
            txt = "Bids, most preferred first: " + "; ".join(items) + "."
            if r["known"] and r["income"] is not None:
                txt += f" Income {money(r['income'])}."
                if r["source"] == "summary" and any(b_["won"] is None for b_ in bids):
                    txt += f" It won {r['won_n']} of them (summary totals; which ones is not recorded)."
            elif not r["known"]:
                txt += " Outcome not recorded."
            if r["feedback"]:
                txt += f" No job won: feedback training on {r['feedback']}."
            ch = [(k, v) for k, v in (r["rep_change"] or {}).items() if v[1] is not None]
            if ch:
                txt += " Reputation " + ", ".join(f"{k} {st(a)} {ARROW} {st(b2)}" for k, (a, b2) in ch) + "."
        if r["note"]:
            txt += f' Note to itself: "{r["note"]}"'
        feed.append({"x": t - 1, "step": len(steps) - 1, "who": seat_name[s], "kind": kind, "text": en(txt)})
        # the round's result after its last decision
        nxt_t = mv[len(steps)][1] if len(steps) < len(mv) else None
        if nxt_t != t:
            parts = []
            lt = listings.get(t) or {}
            if t in alloc:
                ents = alloc[t]
                unf = sorted(j for j in lt if j not in {x["job"] for x in ents})
                parts.append(f"{len(ents)} of {len(lt)} listed jobs filled" + (f"; unfilled: {', '.join(unf)}." if unf else "."))
                wl = []
                for x in ents:
                    bud = lt.get(x["job"])
                    p = None
                    for job, pp, bb in (wins_of(x["winner"], t) or []):
                        if job == x["job"]:
                            p = pp
                    wl.append(f"{x['job']} {short(x['winner'])}" + (f" {pc(p / bud)}" if p is not None and bud else ""))
                parts.append("Winners (share of budget): " + ", ".join(wl) + ".")
            elif t == T and final_from_summary:
                parts.append("This round appears in no prompt; from the summary's final totals:")
                wl = [f"{job} P{s2} {pc(pp / bb)}" for s2 in seats for job, pp, bb in (wins_of(label_of[s2], t) or []) if pp is not None and bb]
                wl += [f"P{s2} {plural(n_wins(label_of[s2], t), 'job')} (which of its bids is not unique)" for s2 in seats
                       if wins_of(label_of[s2], t) is None and n_wins(label_of[s2], t)]
                if wl:
                    parts.append("model workers' wins (share of budget): " + ", ".join(wl) + ";")
            if t in known_set:
                k_each = [n_wins(label_of[s2], t) for s2 in seats]
                k_s = sum(k_each) if all(k is not None for k in k_each) else None
                inc_s = sum(income_of(label_of[s2], t) or 0 for s2 in seats)
                sp = []
                for L in scripted:
                    kk = n_wins(L, t)
                    sp.append(f"{name_of[L]} {plural(kk or 0, 'job')}, {money(income_of(L, t) or 0)}")
                parts.append(f"Model workers {plural(k_s, 'job') if k_s is not None else 'some jobs'}, {money(inc_s)}" + ("; " + "; ".join(sp) if sp else "") + ".")
                tops = sorted(all_labels, key=lambda L: -(cum_after(L, t) or 0))[:3]
                parts.append("Leaders: " + ", ".join(f"{name_of[L]} {money(cum_after(L, t) or 0)}" for L in tops) + ".")
            else:
                parts.append("Outcome not recorded" + (" (the recording stops here)." if t == T else "."))
            feed.append({"x": t - 1, "step": len(steps) - 1, "who": f"Round {t}", "text": " ".join(parts)})

    judge = lambda answer, st_: env_check(answer, listings.get(st_["round"]) or {}, P, skills)
    tied, n_rescued, n_retried, _limits, n_loose = attach_calls(steps, calls, [e.get("view") or "" for e in moves], judge)
    has_reason = any(s_.get("thinking") for s_ in steps)
    from_prompt = sum(1 for s_ in steps if s_.get("from_prompt"))
    # a training no call answered may be a fallback the recorder did not flag: a cross-check only, the page follows the flag
    for s_ in steps:
        r = REC.get((s_["seat"], s_["x"] + 1))
        if s_.pop("unmatched", False) and calls and r and r["type"] == "train" and not r["fallback"]:
            warn(f"P{s_['seat']} round {s_['x'] + 1}: none of its calls answered the logged training {r['skill']}, which the "
                 "recorder does not flag as a fallback; shown as an ordinary training")
    for s_ in steps:
        s_.setdefault("thinking", "")
        for k in ("round", "canon", "fb", "reply", "from_prompt"):
            s_.pop(k, None)
    reason_chars = sum(len(s_["thinking"]) for s_ in steps)
    shared_saved = share_round_blocks(steps)

    # ---- the story numbers
    n_rounds_known = len(R_known)
    last_k = R_known[-1] if R_known else 0
    cum = {L: [cum_after(L, t) for t in range(1, T + 1)] for L in all_labels}
    final_cum = {L: next((v for v in reversed(cum[L]) if v is not None), 0.0) for L in all_labels}
    total_all = sum(final_cum.values())
    seat_total = sum(final_cum[label_of[s]] for s in seats)
    best_s = max(seats, key=lambda s: final_cum[label_of[s]])
    worst_s = min(seats, key=lambda s: final_cum[label_of[s]])
    complete_game = lb_final is not None and T >= int(P["rounds"])
    jobs_won = {L: 0 for L in all_labels}
    for t in range(1, T + 1):
        for L in all_labels:
            jobs_won[L] += n_wins(L, t) or 0
    listed_tot = sum(len(listings[t]) for t in R_known if t in listings)
    posted_tot = sum(sum(listings[t].values()) for t in R_known if t in listings)
    filled_tot = sum(len(alloc[t]) for t in R_known if t in alloc)
    if final_from_summary and T in known_set and T not in alloc:     # the engine's count; checked against the rebuilt one
        filled_tot += (int(ser["matched"][T - 1]) if ser["matched"] and ser["matched"][T - 1] is not None
                       else sum(n_wins(L, T) or 0 for L in all_labels))
    dec = [REC[k] for k in sorted(REC, key=lambda k: (k[1], k[0]))]
    n_dec = len(dec)
    n_train = sum(1 for r in dec if r["type"] == "train" and not r["fallback"])
    n_fb_act = sum(1 for r in dec if r["fallback"])
    n_feedback = sum(1 for r in dec if r["feedback"])
    train_by = {s: sum(1 for r in dec if r["seat"] == s and r["type"] == "train" and not r["fallback"]) for s in seats}
    train_by_skill = {}
    for r in dec:
        if r["type"] == "train" and not r["fallback"]:
            train_by_skill[r["skill"]] = train_by_skill.get(r["skill"], 0) + 1
    win_ratio_by = {}
    for s in seats:
        xs = [b_["price"] / b_["budget"] for r in dec if r["seat"] == s for b_ in r["bids"] if b_["won"] and b_["budget"]]
        win_ratio_by[s] = mean(xs)
    all_bid_ratios = [(b_["price"] / b_["budget"], r["seat"], r["t"], b_["job"]) for r in dec for b_ in r["bids"] if b_["budget"]]
    won_bid_ratios = [(b_["price"] / b_["budget"], r["seat"], r["t"], b_["job"]) for r in dec for b_ in r["bids"] if b_["budget"] and b_["won"]]
    above = [x for x in all_bid_ratios if x[0] > 1.0 + 1e-9]
    rep_mean = {s: [mean(list((REC.get((s, t)) or {}).get("rep0", {}).values())) if (s, t) in REC and REC[(s, t)]["rep0"] else None
                    for t in range(1, T + 1)] for s in seats}
    win_rep = [mean([x["rep"] for x in alloc[t] if x["rep"] is not None]) if t in alloc and alloc[t] else None for t in range(1, T + 1)]
    w = min(10, max(1, n_rounds_known // 3)) if n_rounds_known >= 3 else 1
    first_ts = R_known[:w]
    last_ts = R_known[-w:]
    fa, la = mean(pooled(first_ts, win_all_n)), mean(pooled(last_ts, win_all_n))
    fs = mean([x[0] for x in won_bid_ratios if x[2] in first_ts])
    ls = mean([x[0] for x in won_bid_ratios if x[2] in last_ts])
    pace = []
    acc = 0.0
    for t in range(T):
        acc += (exp_round[t] if t < len(exp_round) else 0.0) / max(1, int(P["agents"]))
        pace.append(round(acc, 4))
    th = P["price_weight"]
    trade = (2.5 / 2.0) ** ((1 - th) / th) - 1 if P.get("rho", 0) == 0 and 0 < th < 1 else None
    hid = {L: per_skill((ag_info.get(L) or {}).get("skills"), skills) for L in all_labels}
    rep_end = {L: per_skill((ag_info.get(L) or {}).get("reputation"), skills) for L in all_labels}
    mean_perf = ser["mean_performance"]


    # ---- charts
    labels = [f"R{t}" for t in range(1, T + 1)]
    # 1. who won jobs: one row per agent
    max_inc = max([REC[k]["income"] or 0 for k in REC] + [0.01])
    rows, cells = [], []
    for s in seats:
        L = label_of[s]
        row = []
        for t in range(1, T + 1):
            r = REC.get((s, t))
            if r is None:
                row.append(None)
                continue
            bids = r["bids"]
            rats = ", ".join(f"{b_['job']} {pc(b_['price'] / b_['budget'])}" + (" won" if b_["won"] else "") for b_ in bids if b_["budget"])
            if r["fallback"]:
                row.append({"c": "c2", "t": r["skill"], "o": .85,
                            "tip": ("no usable answer" if r["why"] == "no usable answer" else f"unusable answer ({r['why']})") + f"; the game trained {r['skill']}"})
            elif r["type"] == "train":
                ch = (r["rep_change"] or {}).get(r["skill"])
                row.append({"c": "c4", "t": r["skill"], "o": .9, "tip": f"trained {r['skill']}" + (f", reputation {st(ch[0])} {ARROW} {st(ch[1])}" if ch and ch[1] is not None else "")})
            elif not r["known"]:
                row.append({"c": "muted", "t": "?", "o": .6, "tip": f"bid {rats}; outcome not recorded"})
            else:
                k = r["won_n"] if r["won_n"] is not None else sum(1 for b_ in bids if b_["won"])
                if k:
                    tip = f"won {k} of {len(bids)}, {money(r['income'])}: {rats}" if all(b_["won"] is not None for b_ in bids) else \
                        f"won {k} of {len(bids)} ({money(r['income'])}; which ones is not recorded): {rats}"
                    row.append({"c": "c3", "t": str(k), "o": round(.35 + .65 * min(1.0, (r["income"] or 0) / max_inc), 3), "tip": tip})
                else:
                    row.append({"c": "hold", "t": "0", "o": .6, "tip": f"won none: {rats}" + (f"; feedback training on {r['feedback']}" if r["feedback"] else "")})
        rows.append(seat_name[s])
        cells.append(row)
    for L in scripted:
        row = []
        for t in range(1, T + 1):
            w_ = wins_of(L, t)
            inc = income_of(L, t)
            if t in alloc and w_ is not None:
                if w_:
                    row.append({"c": "c3", "t": str(len(w_)), "o": round(.35 + .65 * min(1.0, (inc or 0) / max_inc), 3),
                                "tip": f"won {len(w_)}: " + ", ".join(f"{j} {pc(p / bb)}" if p is not None and bb else j for j, p, bb in w_)
                                       + (f" ({money(inc)})" if inc is not None else "")})
                else:
                    row.append({"c": "muted", "o": .35, "tip": "no job won (its bids and training are not public)"})
            elif t == T and L in scripted_T:
                k_T, inc_T = scripted_T[L]
                row.append({"c": "c3", "t": str(k_T), "o": round(.35 + .65 * min(1.0, inc_T / max_inc), 3),
                            "tip": f"won {k_T} ({money(inc_T)}; which jobs is not recorded)"} if k_T else
                           {"c": "muted", "o": .35, "tip": "no job won (its bids and training are not public)"})
            else:
                row.append(None)
        rows.append(name_of[L])
        cells.append(row)
    used_kinds = {s_["kind"] for s_ in steps}
    legend = [["Won jobs (number in the cell; darker = more earned)", "c3"], ["Bid, won nothing", "hold"], ["Trained (skill in the cell)", "c4"]]
    if "fallback" in used_kinds:
        legend.append(["Unusable answer, game trained", "c2"])
    if "pending" in used_kinds:
        legend.append(["Bid, outcome not recorded (?)", "muted"])
    if scripted:
        legend.append(["Scripted agent, no job won", "muted"])
    jw_txt = ", ".join(f"P{s} {jobs_won[label_of[s]]}" for s in seats) + ("; " + ", ".join(f"{name_of[L]} {jobs_won[L]}" for L in scripted) if scripted else "")
    c_grid = {"title": "Jobs won, round by round",
              "sub": (f"One row per agent, one column per round. Green: jobs won that round (the number in the cell; darker = more "
                      f"earned); gray: bid and won nothing (the platform then trains the skill of the first bid); yellow: trained (the "
                      f"skill in the cell). A worker can hold at most {P['capacity']} jobs a round. The scripted agents' bids and "
                      f"training are not public, only their wins. Jobs won: {jw_txt}"
                      + (f" ({rounds_txt(1, last_k)})" if last_k < T else "") + "."),
              "grid": {"rows": rows, "cells": cells, "legend": legend, "rowHeight": 24}}

    # 2. price competition
    alphas = sorted({(alpha_of(pol[L], P), name_of[L]) for L in scripted if pol[L] and alpha_of(pol[L], P) is not None}, reverse=True)
    first_below = None
    if alphas:
        lo_a = min(a_ for a_, _ in alphas)
        first_below = next((t for t in range(1, T + 1) if med_bid[t - 1] is not None and med_bid[t - 1] < 100 * lo_a - 1e-9), None)
    lines = [{"name": "Workers' median bid", "color": "c1", "v": [None if v is None else round(v, 3) for v in med_bid]},
             {"name": "Workers' lowest bid", "color": "c2", "thin": True, "v": [None if v is None else round(v, 3) for v in low_bid]}]
    if any(v is not None for v in win_all):
        lines.append({"name": "Mean winning bid, all filled jobs", "color": "c3", "dots": T <= 30, "v": [None if v is None else round(v, 3) for v in win_all]})
    else:
        lines.append({"name": "Workers' mean winning bid", "color": "c3", "dots": T <= 30, "v": [None if v is None else round(v, 3) for v in win_seat]})
    for a_, nm in alphas:
        lines.append({"name": f"{nm}'s price {pc(a_)}", "color": "hold", "dash": True, "thin": True, "v": [round(100 * a_, 3)] * T})
    lines.append({"name": "Posted budget 100%", "color": "muted", "dash": True, "thin": True, "v": [100.0] * T})
    if not any(v is not None for v in lines[0]["v"]):
        lines = lines[2:]
    md_known = [(t, v) for t, v in zip(range(1, T + 1), med_bid) if v is not None]
    low_all = min(all_bid_ratios) if all_bid_ratios else None
    low_win = min(won_bid_ratios) if won_bid_ratios else None
    sub2 = ("Each bid's price divided by the job's posted budget. Median and lowest: over all bids the model workers placed in the "
            "round; mean winning bid: over every filled job" + (", the scripted agents' at their fixed share" if alphas else "") + ". ")
    if md_known:
        sub2 += f"The workers' median bid went from {md_known[0][1]:.0f}% in round {md_known[0][0]} to {md_known[-1][1]:.0f}% in round {md_known[-1][0]}"
        sub2 += (f"; it first fell below the {alphas[-1][1]}'s {pc(alphas[-1][0])} in round {first_below}." if first_below and alphas else ".")
    if low_win:
        dd = 1 if low_all and pc(low_all[0]) == pc(low_win[0]) else 0
        sub2 += (f" Lowest winning bid: {pc(low_win[0], dd)} (P{low_win[1]}, {low_win[3]} in round {low_win[2]})"
                 + (f"; lowest bid of all: {pc(low_all[0], dd)} (P{low_all[1]}, {low_all[3]} in round {low_all[2]})." if low_all and low_all[0] < low_win[0] - 1e-9 else "."))
    if above:
        top = max(above)
        sub2 += f" {plural(len(above), 'bid')} asked more than the budget (up to {pc(top[0])}, P{top[1]} in round {top[2]})."
    if recession:
        rr_ = sorted(recession)
        sub2 += (f" Recession rounds (budgets cut to {money(P['recession_budget'])} and few listings): {len(rr_)}, from round {rr_[0]} to "
                 f"round {rr_[-1]}.")
    if trade is not None:
        sub2 += (f" The clients' rule, which the workers are not told, scores reputation^{1 - th:g} {TIMES} (budget {DIV} price)^{th:g}: "
                 f"at equal reputation the lower price wins, and a 2.5-star worker can ask {pc(trade)} more than a 2.0-star one and still score the same.")
    c_price = {"title": "Price competition: bids as a share of the posted budget", "sub": sub2, "height": 250, "fmt": "pct0", "lines": lines}

    # 3. earnings
    order = sorted(all_labels, key=lambda L: -final_cum[L])
    e_lines = []
    for L in order:
        if L in seat_of_label:
            s = seat_of_label[L]
            e_lines.append({"name": seat_name[s], "color": "c1", "v": [None if v is None else round(v, 2) for v in cum[L]],
                            **({"end": True} if s in (best_s, worst_s) else {})})
    for L in scripted:
        e_lines.append({"name": name_of[L], "color": "c2" if pol.get(L) == "specialist" else ("c4" if pol.get(L) == "greedy" else "ink"),
                        "dash": True, "end": True, "v": [None if v is None else round(v, 2) for v in cum[L]]})
    e_lines.append({"name": "Equal share (reward 0)", "color": "hold", "dash": True, "thin": True, "end": True, "v": pace})
    fin_txt = ", ".join(f"{name_of[L]} {money(final_cum[L])}" for L in order)
    sub3 = (f"Each agent's total earnings after every round, from the leaderboard every worker sees"
            + (" (the last round from the summary's final earnings)" if lb_final else "") + ". Dashed gray: an equal share of the "
            f"money the market is expected to post, {money(S_eq, dp=0)} over {P['rounds']} rounds ({money(S_eq / max(1, int(P['rounds'])))} a round); "
            f"a worker that ends on it scores 0, twice it +1. " + (f"After round {last_k}: " if last_k < int(P["rounds"]) else "Final: ") + fin_txt + ".")
    c_earn = {"title": "Total earnings so far ($)", "sub": sub3, "height": 260, "fmt": "price", "lines": e_lines}

    # 4. reputation
    r_lines = [{"name": seat_name[s], "color": "c1", "thin": True, "v": [None if v is None else round(v, 3) for v in rep_mean[s]]} for s in seats]
    if any(v is not None for v in win_rep):
        r_lines.append({"name": "Winners' reputation on the job's skill", "color": "c3", "v": [None if v is None else round(v, 3) for v in win_rep]})
    if mean_perf and any(_is_num(v) for v in mean_perf):
        r_lines.append({"name": f"Delivered quality, all jobs ({TIMES}5)", "color": "c2", "v": [round(5 * v, 3) if _is_num(v) else None for v in mean_perf]})
    r_lines.append({"name": f"Start {st(5 * P['base_rate'])} (base rate)", "color": "hold", "dash": True, "thin": True, "v": [5 * P["base_rate"]] * T})
    r_lines.append({"name": f"Hidden skill at the start, {P['initial_skill']:.2f} ({st(5 * P['initial_skill'])})", "color": "muted", "dash": True,
                    "thin": True, "v": [5 * P["initial_skill"]] * T})
    rep_last = {s: next((v for v in reversed(rep_mean[s]) if v is not None), None) for s in seats}
    sub4 = (f"Each model worker's reputation averaged over the {len(skills)} skills, as its prompt showed it at the start of each round "
            f"(stars = 5 {TIMES} the platform's score). Everyone starts at {st(5 * P['base_rate'])}, but delivers quality near its hidden "
            f"skill ({P['initial_skill']:.2f} at the start, {st(5 * P['initial_skill'])}), so early jobs and benchmarks pull reputations "
            f"down; each training step closes {pc(1 - P['skill_decay'])} of the gap between the skill and 1 and publishes a benchmark. "
            "Green: the mean reputation of the round's winners on the skill of the job they won.")
    if any(v is not None for v in rep_last.values()):
        hi_s = max((s for s in seats if rep_last[s] is not None), key=lambda s: rep_last[s])
        lo_s = min((s for s in seats if rep_last[s] is not None), key=lambda s: rep_last[s])
        sub4 += f" Last shown: {st(rep_last[lo_s])} (P{lo_s}) to {st(rep_last[hi_s])} (P{hi_s})."
    if any(hid[label_of[s]] for s in seats):
        hs = [(s, mean(list(hid[label_of[s]].values()))) for s in seats if hid[label_of[s]]]
        mx = max(hs, key=lambda x: x[1])
        mn = min(hs, key=lambda x: x[1])
        sub4 += (f" Hidden skill at the end (summary, mean over skills; never shown to anyone): {mn[1]:.2f} (P{mn[0]}) to {mx[1]:.2f} "
                 f"(P{mx[0]}), against {P['initial_skill']:.2f} at the start.")
    fin_rep = [(s, mean(list(rep_end[label_of[s]].values()))) for s in seats if rep_end[label_of[s]]]
    if fin_rep:
        sub4 += (f" After the last round (summary): {st(min(v for _, v in fin_rep))} to {st(max(v for _, v in fin_rep))} for the model "
                 "workers" + ("; " + ", ".join(f"{name_of[L]} {st(mean(list(rep_end[L].values())))}" for L in scripted if rep_end[L]) if scripted else "") + ".")
    c_rep = {"title": "Reputation (stars out of 5)", "sub": sub4, "height": 230, "fmt": "price", "lines": r_lines}
    charts = []
    for c in (c_grid, c_price, c_earn, c_rep):            # a series with no value at all (nothing known yet) is left out
        if not c.get("grid"):
            c["lines"] = [l for l in c["lines"] if any(v is not None for v in l["v"])]
            if not any(l for l in c["lines"] if not l.get("dash")):
                continue
        charts.append(c)

    # ---- tiles
    win_label = "all filled jobs" if any(v is not None for v in win_all) else "the model workers' wins"
    if fa is None:
        fa, la = fs, ls
    tiles = []
    win_spark = [round(v, 2) for v in (win_all if any(v is not None for v in win_all) else win_seat) if v is not None]
    if fa is not None and la is not None:
        tiles.append({"k": "Winning bid, share of budget", "v": f"{pc(fa)} {ARROW} {pc(la)}", "spark": win_spark,
                      "s": (f"Mean over {win_label} in {rounds_txt(first_ts[0], first_ts[-1])} and in {rounds_txt(last_ts[0], last_ts[-1])}"
                            + (f"; the model workers' own wins: {pc(fs)} {ARROW} {pc(ls)}" if fs is not None and ls is not None and win_label == "all filled jobs" else "")
                            + (f". Lowest winning bid {pc(low_win[0])} (P{low_win[1]}, round {low_win[2]})" if low_win else "")
                            + ". Paper: winning bids in sealed-bid language-model markets end near 71% of the budget; the top model of "
                              "its Table 1 won at 55%.")})
    if total_all > 0:
        sp = "; ".join(f"{name_of[L]} {money(final_cum[L])}" for L in scripted)
        tiles.append({"k": "Earned by the model workers", "v": f"{money(seat_total, dp=0)} of {money(total_all, dp=0)}",
                      "s": (f"{pc(seat_total / total_all)} of all earnings" + (f"; the scripted agents: {sp}" if scripted else "")
                            + (f". All payments came to {pc(total_all / posted_tot)} of the {money(posted_tot, dp=0)} of posted budgets "
                               f"(paper: about 54%); {filled_tot} of {listed_tot} listed jobs were filled" if posted_tot else "")
                            + (f" ({rounds_txt(1, last_k)})" if last_k < int(P["rounds"]) else "") + ".")})
    bL, wL = label_of[best_s], label_of[worst_s]
    rw_b = rewards.get(best_s) if complete_game else None
    rw_w = rewards.get(worst_s) if complete_game else None
    best_s_txt = (f"{bL}: {pc(final_cum[bL] / total_all) if total_all else '0%'} of all earnings, {plural(jobs_won[bL], 'job')}"
                  + (f" at a mean {pc(win_ratio_by[best_s])} of budget" if win_ratio_by.get(best_s) is not None else "")
                  + (f", reward {r3(rw_b)}" if rw_b is not None else "") + ". "
                  + f"Lowest: {seat_name[worst_s]} {money(final_cum[wL])}" + (f" (reward {r3(rw_w)})" if rw_w is not None else "") + ". "
                  + f"An equal share, {money(S_eq, dp=0)} over {P['rounds']} rounds, scores 0. Paper's Table 1: best model $727, Specialist $375, Greedy $284.")
    tiles.append({"k": "Top model worker" + ("" if complete_game else f" after {rounds_txt(1, last_k)}"),
                  "v": f"{seat_name[best_s]} {money(final_cum[bL])}", "s": best_s_txt})
    tr_txt = ", ".join(f"P{s} {train_by[s]}" for s in seats)
    rep_all_last = mean([v for v in rep_last.values() if v is not None])
    tiles.append({"k": "Rounds spent training", "v": f"{n_train} of {n_dec}",
                  "s": (f"{pc(n_train / n_dec if n_dec else 0, 1)} of the model workers' decisions (paper: one model trained in 13% of "
                        f"rounds; Specialist 20.8%). By worker: {tr_txt}"
                        + (f"; skills trained: " + ", ".join(f"{k} {v}" for k, v in sorted(train_by_skill.items())) if train_by_skill else "")
                        + f". {plural(n_feedback, 'feedback training')} after a round without a win"
                        + (f"; {plural(n_fb_act, 'unusable answer')} replaced by training" if n_fb_act else "")
                        + (f". Mean reputation last shown: {st(rep_all_last)} (paper: 2.7{NDASH}3.4*)." if rep_all_last is not None else "."))})
    tiles = tiles[:4]

    # ---- title, lede
    n_seat = len(seats)
    if fa is not None and la is not None:
        d_ = 100 * (la - fa)
        trend = (f"winning bids fall from {pc(fa)} to {pc(la)} of the posted budget" if d_ <= -5 else
                 f"winning bids rise from {pc(fa)} to {pc(la)} of the posted budget" if d_ >= 5 else
                 f"winning bids stay near {pc((fa + la) / 2)} of the posted budget")
    else:
        trend = "no job has been filled yet"
    top_scr = max(scripted, key=lambda L: final_cum[L]) if scripted else None
    beat = top_scr is not None and final_cum[top_scr] > final_cum[bL]
    title = (f"{nw(n_seat).capitalize()} model workers {'undercut each other' if fa is not None and la is not None and la < fa - 0.05 else 'compete for gig jobs'}: "
             f"{trend}; " + (f"the scripted {name_of[top_scr]} out-earns them all ({money(final_cum[top_scr])})" if beat
                            else f"{seat_name[best_s]} earns most ({money(final_cum[bL])})"))
    if not complete_game:
        title = (f"Test build, {rounds_txt(1, last_k)} of a recording in progress. " if test_build else f"Cut short after {rounds_txt(1, last_k)}. ") + title
    scr_lede = (f": {nw(n_seat)} played by the model and {nw(len(scripted))} scripted ("
                + and_list(scr_short(L) for L in scripted) + ")") if scripted else f", all played by the model"
    lede = (f"{nw(N).capitalize()} workers compete for jobs on a gig platform for {P['rounds']} rounds{scr_lede}. Each round a random set "
            f"of the {len(skills) * int(P['jobs_per_task'])} catalog jobs is listed with a posted budget; every worker either bids a price on up to "
            f"{P['max_bids']} of them or trains one skill, the client of each job prefers a higher public reputation (stars) and a lower price "
            f"relative to the budget, and a worker wins at most {P['capacity']} jobs and is paid its bid. A worker's score is its total "
            f"earnings against an equal share of the money the market is expected to post, {money(S_eq, dp=0)}: (earnings {DIV} "
            f"{S_eq:,.0f}) {MINUS} 1, capped at {MINUS}1 and +1.")

    # ---- foot
    model = recorded_model(b, calls) or "the model"
    date = recorded_date(b, calls)
    seed = run.get("seed", (b.get("config") or {}).get("seed"))
    variant = os.path.splitext(os.path.basename((b.get("config") or {}).get("variant") or ""))[0]
    cfg = ", ".join(x for x in [f"config {variant}" if variant else "", f"seed {seed}" if seed is not None else "",
                                f"{N} agents: " + and_list([f"{n_seat} model seats"] + [f"a scripted {name_of[L]}" for L in scripted]),
                                f"{len(skills)} skills, capacity {P['capacity']}, up to {P['max_bids']} bids",
                                "open bids" if P.get("open_bids") else "sealed bids",
                                "performance pay" if P.get("performance_pay") else "flat fee",
                                f"history window {P['history']} rounds",
                                f"{T} of {P['rounds']} rounds" if T < int(P["rounds"]) else f"{T} rounds"] if x)
    if test_build:
        data_txt = (f"TEST BUILD, not a finished recording: this page was assembled from the call log of the {date} recording of {GID} "
                    f"while it was still running ({cfg}); there is no summary yet, so the last round shown has no outcome and no reward "
                    f"is final. ")
    else:
        data_txt = f"Data: the {date} recording of {GID} ({cfg}), the model seats played by {model}" + (" with reasoning on. " if has_reason else ". ")
    if has_reason:
        rs_txt = ((f"Reasoning is shown for all {tied} decisions" if tied == len(steps) else f"Reasoning is shown for {tied} of {len(steps)} decisions")
                  + f" ({reason_chars:,} characters)"
                  + (f"; {plural(n_retried, 'decision')} needed a second call after an answer the game did not take (the earlier "
                     "attempt is shown first, with the game's reason)" if n_retried else "")
                  + (f"; {plural(n_rescued, 'call')} reached the output limit and took the answer to a follow-up question" if n_rescued else "")
                  + (f"; {plural(n_loose, 'decision')} had no call with exactly its prompt and {'is' if n_loose == 1 else 'are'} tied "
                     "to calls by seat and round" if n_loose else "")
                  + ("; the observations are the prompts of those calls. " if from_prompt == len(steps) else ". "))
    else:
        rs_txt = "No reasoning was saved with this recording. "
    if shared_saved:
        rs_txt += ("Each round's market activity, leaderboard and listings are the same in every seat's prompt: the observations "
                   "show them in full at the round's first decision and point there in the others. ")
    chk = []
    if TL.full("listings") and TL.full("leaderboard") and TL.full("market"):
        chk.append(f"every round's listings, leaderboard and public record agree across all prompts that report them "
                   f"({TL.n['listings'] + TL.n['leaderboard'] + TL.n['market']:,} repeated reports)")
    if TL.full("moves"):
        chk.append(f"all {TL.n['moves']} logged moves are the actions the seats' own history lines show")
    if TL.full("wins") and TL.full("income"):
        chk.append(f"the seats' wins match the public record ({TL.n['wins']} rounds) and their incomes the leaderboard and the winning prices ({TL.n['income']} checks)")
    if TL.full("reputation"):
        chk.append(f"all {TL.n['reputation']} reputation values agree between the history lines, the next prompt and the public record")
    if TL.full("scripted prices"):
        chk.append(f"the scripted agents' incomes equal their fixed price share of the budgets they won in all {TL.n['scripted prices']} rounds")
    if TL.full("payments"):
        chk.append(f"the leaderboard's growth equals the sum of winning prices in all {TL.n['payments']} rounds")
    if TL.full("record lines"):
        chk.append(f"all {TL.n['record lines']} record summary lines match the rebuilt totals")
    if TL.full("summary series"):
        chk.append(f"the summary's per-round series agree ({TL.n['summary series']} values)")
    if TL.full("feedback"):
        chk.append(f"every round a seat bid and won nothing was followed by feedback training on its first bid's skill "
                   f"({TL.n['feedback']} cases)" + (", and each seat's feedback trainings, the last round's included, add up to "
                                                    "the summary's count" if TL.full("feedback totals") else ""))
    if TL.full("fallbacks") and has_flags and n_flag:
        chk.append(f"the {plural(n_flag, 'fallback')} the recorder flagged match its count, each training the skill with the top "
                   "reputation the seat was shown")
    if TL.full("last round"):
        chk.append(f"the summary's final totals fit the last round for all {last_n} seats, and its wins follow uniquely for "
                   f"{last_unique} of {last_n} seats" + (f" ({and_list(narrowed)} only through the mean winning price)" if narrowed else ""))
    if TL.full("rewards"):
        chk.append(f"the rewards follow from the earnings ({TL.n['rewards']} values)")
    bad = [g for g in TL.n if not TL.full(g)]
    chk_txt = ("Checks: " + "; ".join(chk) + ". ") if chk else ""
    if bad:
        chk_txt += ("Some cross-checks disagree (" + ", ".join(f"{g} {TL.ok.get(g, 0)} of {TL.n[g]}" for g in bad)
                    + "); the builder prints each disagreement. ")
    pol_src_txt = ""
    if scripted:
        srcs = {pol_src[L] for L in scripted}
        if srcs == {"price"}:
            pol_src_txt = ("The scripted agents' policies are identified from their prices: each one's income equals a fixed share of the "
                           "budgets it won (" + and_list(f"{L} {pc(measured[L], 1)}" for L in scripted if measured[L] is not None) + "). ")
        elif None in srcs:
            pol_src_txt = "The policy of a scripted agent could not be identified (no summary and no win yet). "
    scr_rule_txt = (" ".join(f"The scripted {name_of[L]} ({L}) {pol_txt[L]}." for L in scripted) + " ") if scripted else ""
    last_txt = ""
    if final_from_summary:
        fbT = [f"P{s}" for s in seats if (REC.get((s, T)) or {}).get("feedback")]
        last_txt = (f"Round {T} appears in no prompt: its incomes are the summary's final earnings minus the leaderboard before it, and "
                    f"the model workers' wins the one subset of their bids that matches their jobs won by skill and their income"
                    + (f"; where two subsets fit, the summary's mean winning price (the mean share of the budget over all of a "
                       f"worker's wins) decides ({and_list(narrowed)})" if narrowed else "")
                    + (f" (not unique for {and_list(final_note)}, shown as a count only)" if final_note else "")
                    + (f" (no subset fits for {and_list(final_bad)}, shown as a count only)" if final_bad else "") + ". "
                    + (f"{and_list(fbT)} bid and won nothing in it; {'its' if len(fbT) == 1 else 'their'} feedback training on the "
                       "first bid's skill is shown " + ("as the rules make it certain here. " if certain_feedback else
                                                       "where the summary's count of feedback trainings shows it. ") if fbT else ""))
    defs = (f"Share of budget: a bid's price {DIV} the job's posted budget. Reputation: shown in stars (5 {TIMES} a Beta-evidence score "
            f"that moves towards the delivered quality of paid jobs and towards training benchmarks, starting from a community base rate). "
            f"Feedback training: a worker that bids and wins nothing is trained on the skill of its first bid. ")
    paper = ("Paper (Chiu, Zhang and van der Schaar, ICML 2026, main market of ten agents for 100 rounds with Specialist and Greedy): the "
             "top two models earn $727 and $703 (about 15% of all earnings each) against $375 for Specialist and $284 for Greedy; the "
             "agents together earn about 54% of the posted budget; the top model wins through underbidding (mean winning bid 55% of the "
             "budget); average reputations 2.7 to 3.4 stars; open bidding pushes late winning bids from 71% to 61% of the budget. ")
    names_txt = (f"Market names (agent_01 ...) are a random permutation each episode and do not reveal which agents are scripted; "
                 f"{who_txt} Source: Chiu, Zhang and van der Schaar, When AI Agents Compete for Jobs (ICML 2026), and the official AI-Work code.")
    d = {"id": GID, "name": "AI labour market", "title": en(title), "lede": en(lede), "seatNames": seat_names, "tiles": tiles,
         "x": {"n": T, "labels": labels}, "charts": charts, "kinds": {k: v for k, v in KINDS.items() if k in used_kinds or k in ("won", "lost", "train")},
         "steps": steps, "feed": feed, "feedTitle": "Round by round",
         "feedSub": ("Each model worker's decision as the game applied it (bids with their share of the budget and outcome, or training), "
                     "with the note it left itself, then the round's result: filled jobs and winners, earnings, leaders. The workers "
                     "decide at the same time and never see each other's prices. The selected step is highlighted; later entries are hidden.")}
    mtxt = model if model != "the model" else "model not recorded"
    d["eyebrow"] = (f"social_game_env {DOT} {GID} {DOT} {mtxt}" + (", reasoning on" if has_reason else "") + f" {DOT} recorded {date}"
                    + (f" {DOT} test build from a partial call log" if test_build else ""))
    # the page is one JSON file the host takes up to 16 MB: shorten the longest reasonings, visibly, if it would be larger
    full = [s_["thinking"] for s_ in steps]
    n_short, gone = 0, 0
    for _ in range(8):
        d["foot"] = en(data_txt + rs_txt + chk_txt + scr_rule_txt + pol_src_txt + last_txt + (
            f"To keep the page within its size limit, the {n_short} longest reasonings are shortened in the middle "
            f"({gone:,} characters left out, marked where they were cut; the call log has them whole). " if n_short else "")
            + defs + paper + names_txt)
        size = len(json.dumps(d, ensure_ascii=False).encode("utf-8"))
        if size <= MAX_BYTES:
            break
        total = sum(len(s_["thinking"]) for s_ in steps)
        n_short, gone = fit_thinking(steps, max(100_000, total - int((size - MAX_BYTES) * 1.02) - 20_000), full)
    else:
        warn(f"the page is still {size:,} bytes after shortening the reasoning")
    print(f"checks: {TL.txt()}; calls tied {tied}/{len(steps)}, retried {n_retried}, rescued {n_rescued}; "
          f"rounds {T} (outcomes known for {n_rounds_known}); size {size:,} bytes", file=sys.stderr)
    return d


# ------------------------------------------------------------------------------------------------ TEST HELPER (not for real pages)
def bundle_from_calls(path, run_log=None):
    """TEST ONLY. A stand-in bundle assembled in memory from a model-call log, for running the whole pipeline before the real
    bundle exists. Only the log's last recording process is used (each process numbers its calls from 1; earlier ones are
    failed attempts). One move per decision: each run of consecutive calls with the same prompt is one decision (the recorder
    resends the prompt unchanged after an answer the game rejects); its move is the recorder's logged move from RUN_LOG when
    given (a line ending ' (fallback)' is the game's fallback: flagged, with the last call's answer as the rejected reply),
    else the last call's answer; its view is that prompt. No summary and no game_over. Decisions RUN_LOG does not have yet (a
    call still being retried) are left out. run.json is marked test_bundle_from_calls, which the page shows in its eyebrow,
    title and foot. It is never written to disk."""
    calls = load_calls(path, lenient=True)
    segs = call_segments(calls)
    if len(segs) > 1:
        warn(f"test bundle: the call log holds {len(segs)} recording processes; only the last one ({segs[-1][1] - segs[-1][0]} "
             f"calls) is used, {segs[-1][0]} earlier calls left out")
        calls = calls[segs[-1][0]:]
    groups = []
    for k, c in enumerate(calls):
        if groups and calls[groups[-1][-1]]["prompt"] == c["prompt"]:
            groups[-1].append(k)
        else:
            groups.append([k])
    logged = None
    if run_log and os.path.exists(run_log):
        logged = [(int(m.group(1)), m.group(2), bool(m.group(3))) for m in
                  re.finditer(r"(?m)^\d+\. P(\d+): (.*?)( \(fallback\))?$", open(run_log, encoding="utf-8").read())]
    events = [{"event": "setup", "_visibility": "hidden"}]
    for gi, ks in enumerate(groups):
        last = calls[ks[-1]]
        f = view_facts(last["prompt"])
        key = (f["seat"] if f["seat"] is not None else last["seat"], f["round"])
        move, ev = last["answer"], {}
        if logged is not None:
            if gi >= len(logged):
                warn(f"test bundle: the run log stops before decision {gi + 1} (P{key[0]} round {key[1]}); left out with the rest")
                break
            if logged[gi][0] != key[0]:
                warn(f"test bundle: run log decision {gi + 1} is P{logged[gi][0]}, the calls say P{key[0]}; run log ignored from here")
                logged = None
            else:
                move = logged[gi][1]
                P_ = dict(DEFAULT_PARAMS)
                if f["settings"]:
                    P_.update(max_bids=f["settings"]["max_bids"], jobs_per_task=f["settings"]["catalog"] // max(1, f["settings"]["tasks"]))
                sk = sorted(f["skills"]) or [chr(65 + i) for i in range(int(P_["tasks"]))]
                judge = lambda text: env_check(text, f["listings"] or {}, P_, sk)[0]
                if logged[gi][2]:
                    ev = {"fallback": True, "reply": (last["answer"] or "")[:REPLY_CHARS]}
                    if any(judge(calls[k]["answer"]) is not None for k in ks):
                        warn(f"test bundle: run log decision {gi + 1} is a fallback, yet one of its calls' answers is legal")
                else:
                    want = canonical_of(parse_action(move))
                    if want is None or all(judge(calls[k]["answer"]) != want for k in ks):
                        warn(f"test bundle: run log decision {gi + 1} is none of its calls' answers")
        events.append({"event": "move", "actor": key[0], "turn": len(events), "move": move, "view": last["prompt"], **ev})
    t0 = next((c["t_start"] for c in calls if c.get("t_start")), None)
    run = {"model": next((c.get("model") for c in calls if c.get("model")), None), "test_bundle_from_calls": True,
           "fallback_flags": logged is not None,
           "recorded_at": datetime.datetime.fromtimestamp(t0, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t0 else None}
    return {"events": events, "summary": {}, "run": run, "config": {}, "transcript": "", "meta": {}}, calls


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--test-from-calls":
        tb, tcalls = bundle_from_calls(sys.argv[2], sys.argv[4] if len(sys.argv) > 4 else None)
        common.write(build(tb, tcalls), sys.argv[3], date=recorded_date(tb, tcalls))
    else:
        if len(sys.argv) < 3:
            raise SystemExit(__doc__)
        src, out = sys.argv[1], sys.argv[2]
        cl_ = load_calls(sys.argv[3], lenient=True) if len(sys.argv) > 3 else []
        common.write(build(src, cl_), out, date=recorded_date(load_source(src), cl_))
