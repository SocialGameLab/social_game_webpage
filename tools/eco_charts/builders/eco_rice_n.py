"""Replay data for eco_rice_n (RICE-N: climate negotiations between regions in a calibrated climate-economy; Zhang et al.,
"AI for Global Climate Cooperation", ICML 2025, and the official climate-cooperation-competition code).

usage: python3 -I eco_rice_n.py BUNDLE OUT_DIR [CALLS.jsonl]
       python3 -I eco_rice_n.py --partial CALLS.jsonl RUN_LOG OUT_DIR
  BUNDLE  a demo bundle directory (game_log.jsonl, summary.json, run.json, config.json, meta.json), or a file inside one.
  OUT_DIR where <OUT_DIR>/eco_rice_n.json is written (via common.write).
  CALLS   optional model-call log (JSON lines: messages or prompt, reasoning, answer, followup_answer; see common.load_calls).
          A call belongs to the decision of the seat, step and stage its prompt names; of the calls for one decision, the one
          whose answer is the logged move at the action grid (the game snaps every rate to a multiple of 0.1) is the decision's
          call, and earlier ones were retries (kept above it). Its reasoning becomes the step's thinking.
  --partial  test helper (last section of this file): builds the page from the call log and the run log of a recording that
          stopped before the end. The bundle is assembled in memory (moves exactly as the run log has them, each move's view the
          prompt of its call); nothing is invented and no bundle file is written.

One game step is 5 years and has three stages that all seats play at the same time, in seat order, buffered (rules: Round
Procedure): proposals (bilateral: a promised and a requested minimum mitigation rate for every other region), evaluation (accept
or reject each proposal received) and activity (savings, mitigation, export limit, import bids, tariffs). Every move's view is the
seat's prompt, which reports the climate, the seat's own last step and score so far, a public table of every region's last plan,
and in the evaluation and activity stages the proposals received and the seat's minimum mitigation rate. The negotiation is
rebuilt from the logged moves (minimum_i = the largest promise of i that was accepted and the largest request to i that i
accepted) and checked against what every prompt reports; the plans are checked against the public table and the seats' own last
steps; the summary's series (when present) against the prompts. The last step's results come from the summary only."""
import datetime, json, math, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GID = "eco_rice_n"
DEFAULT_DATE = "2026-11-12"
MINUS, DOT, ARROW = "−", "·", "→"
SEAT_COLORS = ("c1", "c2", "c3", "c4", "ink")
STAGES = ("proposal", "evaluation", "activity")
STAGE_OF = {"propose": "proposal", "evaluate": "evaluation", "act": "activity"}
KINDS = {"promise": ["Proposal with a promise", "c3"], "ask": ["Asks others, promises nothing", "c2"],
         "silent": ["Sends no proposal", "muted"], "accept": ["Accepts a proposal", "c1"], "reject": ["Accepts none", "hold"],
         "mitigate": ["Plan with mitigation", "c4"], "nomit": ["Plan without mitigation", "ink"]}
# The rules' fixed-policy checks (Reward, "Scale": all 27 regions, 20 steps, default parameters, no trade) and the paper's Table 1
# (27 trained regions, temperature anomaly in 2115), quoted from rules/eco_rice_n_rules.md.
BAU_T2115, FULL_T2115 = 6.523, 2.101
REF_REGIONS = 27
PAPER_T2115 = {"none": 5.121, "bilateral": 3.34, "basic_club": 3.422}
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
N = r"[-+]?\d[\d,]*(?:\.\d+)?"


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
    """A bundle dict (from the --partial helper), a bundle directory, or a file inside one."""
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


def recorded_date(b, calls):
    v = (b.get("run") or {}).get("recorded_at") or (b.get("meta") or {}).get("recorded_at")
    if v:
        return str(v)[:10]
    t = next((c.get("t_start") for c in calls if c.get("t_start")), None)
    if t:
        try:
            return datetime.datetime.fromtimestamp(float(t), datetime.timezone.utc).strftime("%Y-%m-%d")
        except (TypeError, ValueError, OverflowError):
            pass
    return DEFAULT_DATE


# ------------------------------------------------------------------------------------------------ small helpers
def fnum(s):
    v = float(str(s).replace(",", "").replace("+", "")) + 0.0
    return 0.0 if v == 0 else v


def pc(v):
    """A rate as a whole percentage (rates are multiples of 0.1): 0.3 -> '30%'."""
    if v is None:
        return "?"
    x = 100 * v
    return f"{x:.0f}%" if abs(x - round(x)) < 1e-6 else f"{x:.1f}%"


def pcn(v):
    """A rate as a bare percentage number for grid cells: 0.3 -> '30'."""
    return f"{100 * v:.0f}"


def tr(v, d=3):
    """A quantity with a true minus sign."""
    s = f"{abs(v):,.{d}f}"
    return (MINUS if v < 0 and float(s.replace(",", "")) != 0 else "") + s


def cap(s):
    """First letter upper case, the rest unchanged (str.capitalize would lower-case region names)."""
    return s[:1].upper() + s[1:] if s else s


def and_list(xs):
    xs = list(xs)
    if len(xs) <= 1:
        return "".join(xs)
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def rnum(lab):
    m = re.match(r"R(\d+)$", lab or "")
    return int(m.group(1)) if m else 10 ** 6


def regions_text(labels):
    """R2, R3, R4, R5 -> 'R2 to R5'; R1, R3, R4, R5 -> 'R1, R3 to R5' (runs of three or more are joined)."""
    ls = sorted(set(labels), key=rnum)
    out, i = [], 0
    while i < len(ls):
        j = i
        while j + 1 < len(ls) and rnum(ls[j + 1]) == rnum(ls[j]) + 1:
            j += 1
        if j - i >= 2:
            out.append(f"{ls[i]} to {ls[j]}")
        else:
            out.extend(ls[i:j + 1])
        i = j + 1
    return and_list(out)


def years_span(ys):
    """Step years as runs of consecutive steps: 2015..2040 -> 'from 2015 to 2040'; 2015, 2020, 2060 -> 'in 2015, 2020 and 2060';
    2015..2040, 2060 -> 'in 2015 to 2040 and 2060'."""
    ys = sorted(set(ys))
    if not ys:
        return ""
    runs = [[ys[0]]]
    for y in ys[1:]:
        if y - runs[-1][-1] == 5:
            runs[-1].append(y)
        else:
            runs.append([y])
    if len(runs) == 1 and len(ys) > 2:
        return f"from {ys[0]} to {ys[-1]}"
    parts = []
    for r in runs:
        parts.extend([f"{r[0]} to {r[-1]}"] if len(r) > 2 else map(str, r))
    return "in " + and_list(parts)


def list_txt(items):
    """and_list, or '; ' between items when an item already has an 'and' in it ('R1 in 2020 and 2030; R3 in 2025')."""
    items = list(items)
    return "; ".join(items) if any(" and " in x for x in items) and len(items) > 1 else and_list(items)


def years_text(ys):
    ys = sorted(set(ys))
    if not ys:
        return "no step"
    if len(ys) == 1:
        return str(ys[0])
    if len(ys) > 2 and all(b - a == 5 for a, b in zip(ys, ys[1:])):
        return f"{ys[0]} to {ys[-1]}"
    return and_list(map(str, ys))


# ------------------------------------------------------------------------------------------------ actions
def load_json(s):
    if isinstance(s, dict):
        return s
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


def isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def read_action(a):
    """A move or an answer, normalized: {'id': 'propose', 'proposals': {label: (promise, request)}} (bilateral) or
    {'id': 'propose', 'rate': x} (club); {'id': 'evaluate', 'accept': [labels]}; {'id': 'act', 'savings', 'mitigation',
    'export_limit', 'imports': {}, 'tariffs': {}}; 'bad': True when the payload is not readable; None when not JSON."""
    a = load_json(a)
    if not isinstance(a, dict):
        return None
    aid = a.get("action_id")
    p = a.get("payload") if isinstance(a.get("payload"), dict) else {}
    if aid == "propose":
        if isinstance(p.get("proposals"), dict):
            out = {}
            for k, v in p["proposals"].items():
                if not (isinstance(v, dict) and isnum(v.get("promise", 0)) and isnum(v.get("request", 0))):
                    return {"id": aid, "bad": True}
                out[str(k)] = (float(v.get("promise", 0)) + 0.0, float(v.get("request", 0)) + 0.0)
            return {"id": aid, "proposals": out}
        if isnum(p.get("rate")):
            return {"id": aid, "rate": float(p["rate"])}
        return {"id": aid, "bad": True}
    if aid == "evaluate":
        acc = p.get("accept")
        if isinstance(acc, list) and all(isinstance(x, str) for x in acc):
            return {"id": aid, "accept": list(acc)}
        return {"id": aid, "bad": True}
    if aid == "act":
        out = {"id": aid}
        for k in ("savings", "mitigation", "export_limit"):
            if k in p:
                if not isnum(p[k]):
                    return {"id": aid, "bad": True}
                out[k] = float(p[k]) + 0.0
        for k in ("imports", "tariffs"):
            m = p.get(k) or {}
            if not isinstance(m, dict) or not all(isnum(v) for v in m.values()):
                return {"id": aid, "bad": True}
            out[k] = {str(r): float(v) + 0.0 for r, v in m.items()}
        return out
    return {"id": aid, "bad": True}


def same_action(x, y, tol):
    """True when answer x is the logged move y at the action grid: every rate within half a level (the game snaps rates to the
    nearest level), regions left out of a map counting as 0 (a proposal left out as 0/0), acceptance lists as sets."""
    if not x or not y or x.get("bad") or y.get("bad") or x["id"] != y["id"]:
        return False
    close = lambda a, b: abs(a - b) <= tol + 1e-9
    if x["id"] == "propose":
        if ("rate" in x) != ("rate" in y):
            return False
        if "rate" in x:
            return close(x["rate"], y["rate"])
        for k in set(x["proposals"]) | set(y["proposals"]):
            a, b = x["proposals"].get(k, (0.0, 0.0)), y["proposals"].get(k, (0.0, 0.0))
            if not (close(a[0], b[0]) and close(a[1], b[1])):
                return False
        return True
    if x["id"] == "evaluate":
        return sorted(set(x["accept"])) == sorted(set(y["accept"]))
    for k in ("savings", "mitigation", "export_limit"):
        if k in x and k in y and not close(x[k], y[k]):
            return False
    for k in ("imports", "tariffs"):
        a, b = x.get(k, {}), y.get(k, {})
        if any(not close(a.get(r, 0.0), b.get(r, 0.0)) for r in set(a) | set(b)):
            return False
    return True


def nonnull(props):
    return {k: v for k, v in (props or {}).items() if v[0] > 1e-9 or v[1] > 1e-9}


# ------------------------------------------------------------------------------------------------ what a seat was shown
_SEAT = re.compile(r"You are seat P(\d+)\. Your role: ([A-Z_]+)")
_REGION = re.compile(r"You govern region (R\d+) of (\d+)")
_HDR = re.compile(r"Year (\d{4}): step (\d+) of (\d+)\b.*?Stage: (\w+) stage")
_CLIM = re.compile(rf"Climate: atmospheric temperature ({N}) C above pre-industrial \(it was ({N}) in (\d{{4}})\); carbon in the "
                   rf"atmosphere ({N}) GtC\.(?: Last step the world emitted ({N}) GtCO2 per year and produced ({N}) trillion of output)?")
_NOW = re.compile(rf"Your region now: population ({N}) million; capital ({N}) trillion; technology factor ({N}); carbon intensity "
                  rf"({N}) GtCO2 per trillion of production; trade balance ({N})")
_LAST = re.compile(r"^Your last step \((\d{4})\): (.*)$", re.M)
_LAST_FIELDS = {"savings": rf"\bsavings ({N})", "mitigation": rf"\bmitigation ({N})", "production": rf"\bproduction ({N})",
                "kept": rf"\bkept ({N}) after climate damage", "lost": rf"\blost ({N})% to mitigation",
                "gross": rf"\bgross output ({N})", "invested": rf"\binvested ({N})", "exported": rf"\bexported ({N})",
                "imported": rf"\bimported ({N})", "consumption": rf"\bconsumption ({N})", "utility": rf"\butility ({N})",
                "welfare": rf"\bwelfare factor ({N})", "emissions": rf"\byour emissions ({N}) GtCO2",
                "score": rf"Your score so far: ({N})"}
_ROW = re.compile(r"^- (R\d+): (.*)$", re.M)
_RECV = re.compile(rf"^- (R\d+) promises to mitigate at least ({N}) and asks you to mitigate at least ({N})\.?\s*$", re.M)
_OWN = re.compile(r"^Your own proposals \(promise/request\): (.*)$", re.M)
_MIN = re.compile(rf"your minimum mitigation rate is ({N})")
_LASTMIN = re.compile(rf"Last step your minimum mitigation rate from agreements was ({N}); (\d+) proposals? (?:was|were) accepted in total")
_TARIFF = re.compile(r"Tariffs in force this step on your exports: (.*?); your tariffs in force: (.*?)\.?\s*$", re.M)
_SCRIPTED = re.compile(rf"Regions (R\d+) to (R\d+) are not players: they always save ({N}) of output, mitigate ({N})")
_LEVELS = re.compile(rf"All rates are multiples of ({N}) from 0 to ({N})")


def tariff_map(txt):
    t = (txt or "").strip()
    if not t or t.startswith("none"):
        return {}
    return {r: fnum(v) for r, v in re.findall(rf"(R\d+)\D{{0,4}}?({N})", t)}


def view_facts(view):
    """Everything a prompt reports, as numbers (None when the prompt does not say)."""
    v = view or ""
    f = {"seat": None, "role": None, "region": None, "n_regions": None, "year": None, "step": None, "steps": None, "stage": None,
         "temp": None, "temp0": None, "year0": None, "carbon": None, "world_em": None, "world_out": None, "pop": None, "cap": None,
         "tech": None, "intensity": None, "balance": None, "last": None, "table": {}, "received": None, "own": None,
         "no_received": False, "minimum": None, "last_min": None, "accepted_total": None, "tariffs_on_me": None,
         "my_tariffs": None, "scripted": None, "levels": None, "top_rate": None, "protocol": None, "has_view": bool(v.strip())}
    m = _SEAT.search(v)
    if m:
        f["seat"], f["role"] = int(m.group(1)), m.group(2)
    m = _REGION.search(v)
    if m:
        f["region"], f["n_regions"] = m.group(1), int(m.group(2))
    m = _HDR.search(v)
    if m:
        f["year"], f["step"], f["steps"], f["stage"] = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4).lower()
    m = _CLIM.search(v)
    if m:
        f["temp"], f["temp0"], f["year0"], f["carbon"] = fnum(m.group(1)), fnum(m.group(2)), int(m.group(3)), fnum(m.group(4))
        if m.group(5):
            f["world_em"], f["world_out"] = fnum(m.group(5)), fnum(m.group(6))
    m = _NOW.search(v)
    if m:
        f["pop"], f["cap"], f["tech"], f["intensity"], f["balance"] = (fnum(m.group(k)) for k in range(1, 6))
    m = _LAST.search(v)
    if m:
        line = m.group(2)
        last = {"year": int(m.group(1))}
        for k, rx in _LAST_FIELDS.items():
            mm = re.search(rx, line)
            last[k] = fnum(mm.group(1)) if mm else None
        f["last"] = last
    for m in _ROW.finditer(v):
        row = {k: fnum(x) for k, x in re.findall(rf"\b(pop|cap|out|con|sav|mit|exp|bal) ({N})", m.group(2))}
        row["you"], row["fixed"] = "(you)" in m.group(2), "(fixed plan)" in m.group(2)
        f["table"][m.group(1)] = row
    rec = _RECV.findall(v)
    if rec or "Proposals you received" in v:
        f["received"] = {r: (fnum(p), fnum(q)) for r, p, q in rec}
    if "You received no proposal" in v:
        f["received"], f["no_received"] = {}, True
    m = _OWN.search(v)
    if m:
        f["own"] = {r: (fnum(p), fnum(q)) for r, p, q in re.findall(rf"(R\d+) ({N})/({N})", m.group(1))}
    m = _MIN.search(v)
    if m:
        f["minimum"] = fnum(m.group(1))
    m = _LASTMIN.search(v)
    if m:
        f["last_min"], f["accepted_total"] = fnum(m.group(1)), int(m.group(2))
    m = _TARIFF.search(v)
    if m:
        f["tariffs_on_me"], f["my_tariffs"] = tariff_map(m.group(1)), tariff_map(m.group(2))
    rules = v[:v.find("You are seat")] if "You are seat" in v else v
    m = _SCRIPTED.search(rules)
    if m:
        f["scripted"] = (m.group(1), m.group(2), fnum(m.group(3)), fnum(m.group(4)))
    m = _LEVELS.search(rules)
    if m:
        f["levels"], f["top_rate"] = fnum(m.group(1)), fnum(m.group(2))
    if re.search(r"Negotiation \(bilateral", rules):
        f["protocol"] = "bilateral"
    elif re.search(r"basic club|club rate", rules, re.I):
        f["protocol"] = "basic_club"
    elif rules.strip():
        f["protocol"] = "none"
    return f


# ------------------------------------------------------------------------------------------------ the record
def read_record(moves, params):
    """One entry per logged move with its step, stage, view and action; the seats' region labels; the rules' settings."""
    entries, seen, cur, last_ord = [], set(), 1, -1
    label, rules = {}, {"levels": None, "top_rate": None, "protocol": None, "scripted": None, "n_regions": None, "steps": None}
    for i, e in enumerate(moves):
        s = e.get("actor")
        f = view_facts(e.get("view") or "")
        a = read_action(e.get("move"))
        stage = f["stage"] if f["stage"] in STAGES else STAGE_OF.get((a or {}).get("id"))
        t = f["step"]
        if t is None:                                   # no header: a new step when a stage repeats or the order goes back
            o = STAGES.index(stage) if stage in STAGES else 0
            if (s, stage) in seen or o < last_ord:
                cur, seen = cur + 1, set()
            t = cur
        else:
            if t != cur:
                seen = set()
            cur = t
        seen.add((s, stage))
        last_ord = STAGES.index(stage) if stage in STAGES else 0
        if f["region"]:
            if label.get(s, f["region"]) != f["region"]:
                warn(f"P{s} governs {label[s]} and {f['region']}")
            label.setdefault(s, f["region"])
        for k in ("levels", "top_rate", "protocol", "scripted", "n_regions", "steps"):
            if rules[k] is None and f.get(k) is not None:
                rules[k] = f[k]
        raw = e.get("move")
        raw = raw if isinstance(raw, str) else json.dumps(raw) if raw is not None else ""
        entries.append({"i": i, "seat": s, "t": t, "stage": stage, "f": f, "a": a, "raw": raw,
                        "view": e.get("view") or ""})
    seats = sorted({en_["seat"] for en_ in entries})
    for s in seats:
        label.setdefault(s, f"R{s + 1}")
    if rules["levels"] is None:
        al = params.get("action_levels")
        rules["levels"] = 1.0 / al if isinstance(al, (int, float)) and al > 0 else 0.1
    if rules["protocol"] is None:
        rules["protocol"] = params.get("protocol") or ("bilateral" if any(x["stage"] == "proposal" for x in entries) else "none")
    if rules["n_regions"] is None:
        rules["n_regions"] = params.get("regions") or 27
    if rules["steps"] is None:
        rules["steps"] = params.get("steps") or max(x["t"] for x in entries)
    return entries, seats, label, rules


def negotiation(t, R):
    """Step t's proposals, acceptances and the minimum mitigation rates they imply (rules: Round Procedure, step 2)."""
    seats, lab, by_lab = R["seats"], R["label"], R["by_label"]
    sent = {s: R["props"].get((t, s)) for s in seats}
    rate = {s: R["rates"].get((t, s)) for s in seats}
    received = {s: {lab[j]: sent[j][lab[s]] for j in seats if j != s and sent[j] and lab[s] in nonnull(sent[j])} for s in seats}
    evaluated = {s for s in seats if (t, s) in R["accs"]}
    deals = []
    for s in sorted(evaluated):
        for L in R["accs"][(t, s)]:
            j = by_lab.get(L)
            if R["protocol"] == "basic_club":
                rj = rate.get(j) if j is not None else None
                deals.append({"by": j, "to": s, "label": L, "p": rj or 0.0, "r": rj or 0.0, "null": not rj})
                continue
            pr = (sent.get(j) or {}).get(lab[s], (0.0, 0.0)) if j is not None else (0.0, 0.0)
            deals.append({"by": j, "to": s, "label": L, "p": pr[0], "r": pr[1], "null": pr[0] <= 1e-9 and pr[1] <= 1e-9})
    mins = {}
    for s in seats:
        if R["protocol"] == "basic_club":           # the paper's rule: the largest club rate among the proposals s accepted
            mins[s] = max([0.0] + [d["r"] for d in deals if d["to"] == s])
        else:
            mins[s] = max([0.0] + [d["p"] for d in deals if d["by"] == s] + [d["r"] for d in deals if d["to"] == s])
    return {"sent": sent, "rate": rate, "received": received, "evaluated": evaluated, "deals": deals,
            "real": [d for d in deals if not d["null"]], "min_calc": mins}


# ------------------------------------------------------------------------------------------------ model calls
def call_key(c):
    m = _HDR.search(c.get("prompt") or "")
    return (c.get("seat"), int(m.group(2)), m.group(4).lower()) if m else None


def attach_calls(steps, calls, tol):
    """Tie calls to decisions by (seat, step, stage), read from each call's prompt; a call without a readable header (a retry
    whose last message is not the prompt) takes the key of the previous call of the same seat. Within a decision's calls, the
    one whose answer is the logged move is the decision's call; earlier ones were retries, kept above it."""
    if not calls:
        return 0, 0
    keys, last_by_seat = [], {}
    for c in calls:
        k = call_key(c)
        if k is None:
            k = last_by_seat.get(c.get("seat"))
        keys.append(k)
        if k is not None:
            last_by_seat[c.get("seat")] = k
    groups = {}
    for k, key in enumerate(keys):
        if key is not None:
            groups.setdefault(key, []).append(k)
    used, tied = set(), 0
    for st in steps:
        g = groups.get((st["seat"], st["t"], st["stage"]), [])
        if not g:
            warn(f"no call for P{st['seat']} step {st['t']} {st['stage']}")
            continue
        hit = next((k for k in g if same_action(read_action(calls[k].get("answer")), st["a"], tol)), None)
        use = hit if hit is not None else g[-1]
        before = [k for k in g if k < use]
        parts = [f"[Earlier attempt, answer: {calls[k]['answer'][:300] or '(none)'}]\n{calls[k].get('reasoning') or '(no reasoning returned)'}"
                 for k in before]
        c = calls[use]
        parts.append(("[Final attempt]\n" if parts else "") + (c.get("reasoning") or ("(no reasoning returned)" if parts else "")))
        if hit is None:
            parts.append(f"[The game applied another action than this answer: {c['answer'][:300] or '(none)'}]")
            warn(f"P{st['seat']} step {st['t']} {st['stage']}: no call answered the logged move")
        if c.get("rescued"):
            parts.append("[The reasoning ran out before an answer; the action is the answer to a follow-up question.]")
        st["thinking"] = en("\n\n".join(p for p in parts if p).strip())
        st["rescued"], st["retries"] = bool(c.get("rescued")), len(before)
        st["chars"] = sum(len(calls[k].get("reasoning") or "") for k in before + [use])
        used.update(before + [use])
        tied += 1
    left = [k for k in range(len(calls)) if k not in used]
    if left:
        warn(f"{len(left)} calls tied to no decision (calls {left[:6]}{'...' if len(left) > 6 else ''})")
    return tied, len(left)


# ------------------------------------------------------------------------------------------------ build
def build(src, calls=None):
    b = load_source(src)
    if isinstance(calls, str):
        calls = common.load_calls(calls)
    calls = calls or []
    events, summary, run = b["events"], b.get("summary") or {}, b.get("run") or {}
    info = summary.get("info") if isinstance(summary.get("info"), dict) else {}
    params = info.get("params") if isinstance(info.get("params"), dict) else {}
    moves = [e for e in events if e.get("event") == "move"]
    if not moves:
        raise SystemExit("no move events in " + str(src if isinstance(src, str) else "the bundle"))
    over = next((e for e in events if e.get("event") == "game_over"), None)
    raw_rw = summary.get("rewards") or (run.get("result") or {}).get("rewards") or (over or {}).get("rewards") or {}
    rewards = {int(k): float(v) for k, v in raw_rw.items() if isnum(v)}

    entries, seats, label, rules = read_record(moves, params)
    tol = 0.5 * rules["levels"]
    R = {"seats": seats, "label": label, "by_label": {v: k for k, v in label.items()}, "protocol": rules["protocol"],
         "props": {}, "rates": {}, "accs": {}, "plans": {}}
    views, first_view = {}, {}
    for x in entries:
        a, key = x["a"], (x["t"], x["seat"])
        views[(x["t"], x["seat"], x["stage"])] = x["f"]
        first_view.setdefault(key, x["f"])
        if a is None or a.get("bad"):
            warn(f"P{x['seat']} step {x['t']} {x['stage']}: unreadable move {x['raw'][:120]!r}")
            continue
        if x["stage"] == "proposal":
            if "proposals" in a:
                R["props"][key] = a["proposals"]
            elif "rate" in a:
                R["rates"][key] = a["rate"]
        elif x["stage"] == "evaluation":
            R["accs"][key] = a["accept"]
        elif x["stage"] == "activity":
            R["plans"][key] = a
    T_total, T_seen = int(rules["steps"]), max(x["t"] for x in entries)
    last_done = all((T_seen, s) in R["plans"] for s in seats)
    complete = T_seen >= T_total and last_done
    if T_seen > T_total:
        warn(f"moves reach step {T_seen} of {T_total}")
    year0 = next((x["f"]["year"] - 5 * (x["t"] - 1) for x in entries if x["f"]["year"]), 2015)
    year = lambda t: year0 + 5 * (t - 1)
    T = T_seen
    n_x = T + (1 if complete else 0)
    labels = [str(year(t)) for t in range(1, n_x + 1)]
    NEG = {t: negotiation(t, R) for t in range(1, T + 1)}
    N_s = len(seats)
    nw = NUM_WORDS.get(N_s, str(N_s))
    lab = label
    seat_names = [f"Region {lab[s]}" for s in seats]
    sidx = {s: k for k, s in enumerate(seats)}
    scol = {s: SEAT_COLORS[k % len(SEAT_COLORS)] for k, s in enumerate(seats)}
    n_regions = int(rules["n_regions"])
    scripted = rules["scripted"]                       # (first, last, savings, mitigation) or None
    n_fixed = (rnum(scripted[1]) - rnum(scripted[0]) + 1) if scripted else 0
    fixed_sav, fixed_mit = (scripted[2], scripted[3]) if scripted else (params.get("baseline_savings", 0.3), 0.0)
    fixed_labels = {f"R{k}" for k in range(rnum(scripted[0]), rnum(scripted[1]) + 1)} if scripted else set()
    protocol = rules["protocol"]
    negot = protocol != "none"

    # ---- checks: every number a prompt reports against the record ----------------------------------------------------
    CK = {k: [0, 0] for k in ("own", "recv", "min", "lastmin", "acc_total", "table", "fixed", "self", "global", "series", "rw")}
    BAD = {k: [] for k in CK}                          # each disagreement: {t, s, shown, want} as far as known

    def ok(k, good, msg, **ex):
        CK[k][0] += 1
        if good:
            CK[k][1] += 1
        else:
            BAD[k].append(ex)
            warn(msg)
    close = lambda a, b, d: a is not None and b is not None and abs(a - b) <= 0.5 * 10 ** (-d) + 1e-9
    min_shown = {}
    for (t, s, stage), f in views.items():
        ng = NEG[t]
        if not f["has_view"]:
            continue
        if stage == "evaluation" and protocol == "bilateral":
            if f["own"] is not None or ng["sent"].get(s) is not None:
                want = nonnull(ng["sent"].get(s))
                got = f["own"] or {}
                ok("own", set(got) == set(want) and all(close(got[r][0], want[r][0], 1) and close(got[r][1], want[r][1], 1) for r in got),
                   f"P{s} step {t}: prompt lists own proposals {got}, moves {want}", t=t, s=s)
            if f["received"] is not None:
                want = ng["received"][s]
                got = f["received"]
                ok("recv", set(got) == set(want) and all(close(got[r][0], want[r][0], 1) and close(got[r][1], want[r][1], 1) for r in got),
                   f"P{s} step {t}: prompt lists received proposals {got}, moves {want}", t=t, s=s)
        if stage == "activity" and f["minimum"] is not None:
            min_shown[(t, s)] = f["minimum"]
            if ng["evaluated"] == set(seats):
                ok("min", close(f["minimum"], ng["min_calc"][s], 2),
                   f"P{s} step {t}: prompt minimum {f['minimum']}, rebuilt {ng['min_calc'][s]}", t=t, s=s, shown=f["minimum"],
                   want=ng["min_calc"][s])
        # The game (rice_n.py, 2026-11) resets every minimum to 0 at the end of the activity step and only then writes this recap,
        # so a region bound in the last step reads 'was 0'; fail_sentences() says so on the page in plain words.
        if stage == "proposal" and f["last_min"] is not None and t > 1 and NEG[t - 1]["evaluated"] == set(seats):
            pg = NEG[t - 1]
            mshow = min_shown.get((t - 1, s), pg["min_calc"][s])
            ok("lastmin", close(f["last_min"], mshow, 2), f"P{s} step {t}: prompt says last minimum {f['last_min']}, record {mshow}",
               t=t, s=s, shown=f["last_min"], want=mshow)
            ok("acc_total", f["accepted_total"] in (len(pg["real"]), len(pg["deals"])),
               f"step {t - 1}: prompt says {f['accepted_total']} proposals accepted, record {len(pg['real'])} (all {len(pg['deals'])})",
               t=t, s=s, shown=f["accepted_total"], want=len(pg["real"]))
    # the public table and the seat's own last step against the plans; the fixed-plan regions against their rule
    for (t, s), f in first_view.items():
        if t < 2 or not f["has_view"]:
            continue
        for j in seats:
            row = f["table"].get(lab[j])
            p = R["plans"].get((t - 1, j))
            if row is None or p is None or "sav" not in row:
                continue
            good = close(row.get("sav"), p.get("savings"), 1) and close(row.get("mit"), p.get("mitigation", row.get("mit")), 1) \
                and close(row.get("exp"), p.get("export_limit", row.get("exp")), 1)
            ok("table", good, f"step {t} P{s}'s table: {lab[j]} sav/mit/exp {row.get('sav')}/{row.get('mit')}/{row.get('exp')}, plan {p}",
               t=t, s=j)
        for L, row in f["table"].items():
            if row["fixed"] and "sav" in row:
                ok("fixed", close(row["sav"], fixed_sav, 1) and close(row["mit"], fixed_mit, 1) and row.get("exp", 0) == 0,
                   f"step {t}: fixed-plan {L} shows sav {row['sav']} mit {row['mit']} exp {row.get('exp')}", t=t, label=L)
        p, la = R["plans"].get((t - 1, s)), f["last"]
        if p and la:
            ok("self", close(la["savings"], p.get("savings"), 1) and close(la["mitigation"], p.get("mitigation", la["mitigation"]), 1),
               f"P{s} step {t}: own last step savings/mitigation {la['savings']}/{la['mitigation']}, plan {p}", t=t, s=s)
    # world numbers: every prompt of a step reports the same
    G = {}
    for t in range(1, T + 1):
        vs = [f for (tt, s, st), f in views.items() if tt == t]
        g = {}
        for k in ("temp", "carbon", "world_em", "world_out"):
            vals = sorted({f[k] for f in vs if f[k] is not None})
            g[k] = vals[0] if vals else None
            if len(vals) > 1:
                ok("global", False, f"step {t}: prompts disagree on {k}: {vals}", t=t, what=k)
            elif vals:
                ok("global", True, "")
        kept = sorted({f["last"]["kept"] for f in vs if f["last"] and f["last"]["kept"] is not None})
        g["kept_prev"] = kept[0] if kept else None
        G[t] = g
    temp0 = next((f["temp0"] for f in first_view.values() if f["temp0"] is not None), G[1]["temp"])
    temps = [G[t]["temp"] for t in range(1, T + 1)]
    carb = [G[t]["carbon"] for t in range(1, T + 1)]
    em = [G[t + 1]["world_em"] if t < T else None for t in range(1, T + 1)]          # flows during step t (reported at t + 1)
    out = [G[t + 1]["world_out"] if t < T else None for t in range(1, T + 1)]
    kept = [G[t + 1]["kept_prev"] if t < T else None for t in range(1, T + 1)]
    # the share of production kept after climate damage, against the temperature at the start of the step: the paper's form
    # 1 - a2 T^2 or the code's 1 / (1 + a2 T^2), a2 = 0.00236 (rules: Round Procedure, step 3; Differences, damage function)
    dmg = {"paper": [0, 0], "code": [0, 0]}
    for t in range(1, T + 1):
        if kept[t - 1] is not None and temps[t - 1] is not None:
            T_ = temps[t - 1]
            tol_k = 2 * 0.00236 * abs(T_) * 0.005 + 0.00006         # the prompt rounds T to 0.01 and the share kept to 0.0001
            for form, val in (("paper", 1 - 0.00236 * T_ ** 2), ("code", 1 / (1 + 0.00236 * T_ ** 2))):
                dmg[form][0] += 1
                dmg[form][1] += abs(val - kept[t - 1]) <= tol_k
    fits = [f_ for f_ in ("paper", "code") if dmg[f_][0] and dmg[f_][1] == dmg[f_][0]]
    damage_form = (params.get("damage_form") if params.get("damage_form") in fits else fits[0]) if fits else None
    if dmg["paper"][0] and not fits:
        warn(f"the damage factors fit neither form: paper {dmg['paper']}, code {dmg['code']}")
    # the summary's series (when present) against the prompts; they also give the last step
    ser = lambda k: info.get(k) if isinstance(info.get(k), list) and all(v is None or isnum(v) for v in info.get(k)) else None
    ts, es, os_, ag = ser("temperature_series"), ser("emissions_series"), ser("gross_output_series"), ser("agreements_series")
    if ts:
        for t in range(1, min(T, len(ts)) + 1):
            if temps[t - 1] is not None and ts[t - 1] is not None:
                ok("series", close(ts[t - 1], temps[t - 1], 2), f"{year(t)}: summary temperature {ts[t - 1]}, prompts {temps[t - 1]}", t=t, what="temperature")
    for arr, mine, nm in ((es, em, "emissions"), (os_, out, "output")):
        if arr:
            for t in range(1, min(T, len(arr)) + 1):
                if mine[t - 1] is not None and arr[t - 1] is not None:
                    ok("series", close(arr[t - 1], mine[t - 1], 1), f"step {t}: summary {nm} {arr[t - 1]}, prompts {mine[t - 1]}", t=t, what="world " + nm)
            if complete and len(arr) >= T and arr[T - 1] is not None:
                mine[T - 1] = arr[T - 1]
    if ag:
        for t in range(1, min(T, len(ag)) + 1):
            if NEG[t]["evaluated"] == set(seats) and ag[t - 1] is not None:
                ok("series", ag[t - 1] in (len(NEG[t]["real"]), len(NEG[t]["deals"])),
                   f"step {t}: summary agreements {ag[t - 1]}, record {len(NEG[t]['real'])}", t=t, what="agreements")
    t_end = None
    if complete:
        if ts and len(ts) > T and ts[T] is not None:
            t_end = ts[T]
        elif isnum(info.get("temperature_at")):
            t_end = info["temperature_at"]
    # per region: the summary's region records (score, series) against the record
    info_reg = {}
    for r in info.get("regions") or []:
        if isinstance(r, dict) and r.get("label"):
            info_reg[r["label"]] = r
    for s in seats:
        r = info_reg.get(lab[s]) or {}
        if isnum(r.get("score")) and s in rewards and complete:
            ok("rw", abs(r["score"] - rewards[s]) < 1e-6, f"{lab[s]}: summary score {r['score']}, reward {rewards[s]}", s=s)
        for k, pk in (("mitigation_series", "mitigation"), ("savings_series", "savings")):
            arr = r.get(k)
            if isinstance(arr, list):
                for t in range(1, min(T, len(arr)) + 1):
                    p = R["plans"].get((t, s))
                    if p and pk in p and isnum(arr[t - 1]):
                        ok("series", close(arr[t - 1], p[pk], 6), f"{lab[s]} step {t}: summary {pk} {arr[t - 1]}, plan {p[pk]}", t=t, s=s, what=pk)

    # ---- per seat and step: what the step brought (reported at the next step; the last step from the summary) ---------------
    res = {}
    for s in seats:
        for t in range(1, T + 1):
            f = first_view.get((t + 1, s))
            if f and f["last"]:
                res[(t, s)] = dict(f["last"])
        if complete:
            r = info_reg.get(lab[s]) or {}
            last = {}
            for k, sk in (("gross", "gross_output_series"), ("consumption", "consumption_series"), ("emissions", "emissions_series")):
                arr = r.get(sk)
                if isinstance(arr, list) and len(arr) >= T and isnum(arr[T - 1]):
                    last[k] = arr[T - 1]
            if s in rewards:
                last["score"] = rewards[s]
            if last:
                res[(T, s)] = last

    # ---- the model regions' share of world emissions (each prompt's 'your emissions' against 'the world emitted') ------------
    share = {}
    for t in range(1, T + 1):
        es_ = [(res.get((t, s)) or {}).get("emissions") for s in seats]
        if em[t - 1] and all(v is not None for v in es_):
            share[t] = sum(es_) / em[t - 1]
    sh = lambda x: f"{100 * x:.1f}%"
    share_t = sorted(share)
    share_max = max(share.values()) if share else None

    # ---- who was bound, when, and by what (a region is bound when its minimum mitigation rate is above 0) --------------------
    def min_at(t, s):
        if (t, s) in min_shown:
            return min_shown[(t, s)]
        return NEG[t]["min_calc"][s] if NEG[t]["evaluated"] == set(seats) else None
    bound_at = {(t, s): min_at(t, s) for t in range(1, T + 1) for s in seats if (min_at(t, s) or 0) > 1e-9}
    bound_regs = sorted({s for _, s in bound_at})

    # ---- steps: one per decision, in log order --------------------------------------------------------------------------
    def plan_txt(p, short=False):
        if not p:
            return "no plan"
        bits = [f"saves {pc(p.get('savings'))}", f"mitigates {pc(p.get('mitigation', 0.0))}"]
        trade = []
        if p.get("export_limit", 0) > 1e-9:
            trade.append(f"exports up to {pc(p['export_limit'])}")
        imp = {k: v for k, v in (p.get("imports") or {}).items() if v > 1e-9}
        tar = {k: v for k, v in (p.get("tariffs") or {}).items() if v > 1e-9}
        if imp:
            trade.append(f"bids on {len(imp)}" if short else "import bids " + ", ".join(f"{k} {pc(v)}" for k, v in sorted(imp.items(), key=lambda kv: rnum(kv[0]))))
        if tar:
            trade.append(f"tariffs on {len(tar)}" if short else "tariffs " + ", ".join(f"{k} {pc(v)}" for k, v in sorted(tar.items(), key=lambda kv: rnum(kv[0]))))
        return ", ".join(bits + trade)

    def prop_summary(sent):
        """A short head for one proposal stage: the proposals grouped by their promise/request pair, e.g. 'asks 20% of R4 and
        10% of R1 to R3 and R5, promises 0%' or 'offers 10% for 10% to R2 and R3, asks R4 to mitigate 20%'."""
        nn = nonnull(sent)
        if not nn:
            return None, "sends no proposal"
        groups = {}
        for r, v in nn.items():
            groups.setdefault(v, []).append(r)
        order = sorted(groups.items(), key=lambda kv: (-kv[0][0], -kv[0][1]))
        kind = "promise" if any(v[0] > 1e-9 for v in groups) else "ask"
        if len(groups) > 3:                             # too many pairs for a head: give the ranges
            ps, rs = sorted({v[0] for v in nn.values()}), sorted({v[1] for v in nn.values()})
            rng = lambda xs: pc(xs[0]) if len(xs) == 1 else f"{pc(xs[0])} to {pc(xs[-1])}"
            if kind == "ask":
                return kind, f"asks {regions_text(nn)} to mitigate {rng(rs)}, promises 0%"
            return kind, f"proposes to {regions_text(nn)}: promises {rng(ps)}, asks {rng(rs)}"
        offers = [f"offers {pc(v[0])} for {pc(v[1])} to {regions_text(rs)}" for v, rs in order if v[0] > 1e-9]
        asks = [(regions_text(rs), pc(v[1])) for v, rs in order if v[0] <= 1e-9]
        ask_txt = ((f"asks {asks[0][0]} to mitigate {asks[0][1]}" if len(asks) == 1 else
                    "asks " + and_list(f"{r} of {tg}" for tg, r in asks)) if asks else "")
        if kind == "ask":
            return kind, ask_txt + ", promises 0%"
        return kind, ", ".join(offers + ([ask_txt] if ask_txt else []))

    # An accepted proposal binds its sender to the promise and its receiver to the request (rules: Round Procedure, step 2), so
    # a request that promises nothing binds only the region that accepted it.
    def deals_sentences(ds):
        """One step's accepted proposals, grouped by receiver and promise/request pair, saying whom each binds."""
        g = {}
        for d in ds:
            g.setdefault((d["to"], d["p"], d["r"], d["null"]), []).append(d["label"])
        out = []
        for (to, p, r, null), bys in sorted(g.items(), key=lambda kv: (rnum(lab[kv[0][0]]), -kv[0][2], -kv[0][1])):
            A, B, one = lab[to], regions_text(bys), len(bys) == 1
            if null:
                out.append(f"{A} accepted {B}'s empty proposal{'' if one else 's'}, which bind{'s' if one else ''} nobody")
            elif protocol == "basic_club":
                out.append(f"{A} accepted the club rate of {pc(r)} from {B}")
            elif p <= 1e-9:
                out.append((f"{A} accepted {B}'s request to mitigate at least {pc(r)}; {B} promised nothing, so it binds {A} only" if one else
                            f"{A} accepted the requests of {B} to mitigate at least {pc(r)}; they promised nothing, so they bind {A} only"))
            elif r <= 1e-9:
                out.append((f"{A} accepted {B}'s offer to mitigate at least {pc(p)} with nothing asked of {A}, so it binds {B} only" if one else
                            f"{A} accepted offers from {B} to mitigate at least {pc(p)} with nothing asked of {A}, so they bind only {B}"))
            else:
                out.append((f"{A} accepted {B}'s proposal ({B} at least {pc(p)}, {A} at least {pc(r)}), which binds both" if one else
                            f"{A} accepted the proposals of {B} (each of them at least {pc(p)}, {A} at least {pc(r)}), which bind both sides"))
        return out

    def deals_of(t, s):
        """For region s in step t: the accepted proposals it is a party to, from its side ('its request', 'it accepted ...')."""
        ng_, me = NEG[t], lab[s]
        out, g_by, g_to = [], {}, {}
        for d in ng_["real"]:
            if d["by"] == s:
                g_by.setdefault((d["p"], d["r"]), []).append(lab[d["to"]])
            if d["to"] == s:
                g_to.setdefault((d["p"], d["r"]), []).append(d["label"])
        for (p, r), xs in sorted(g_by.items(), key=lambda kv: (-kv[0][0], -kv[0][1])):
            X, many = regions_text(xs), len(xs) > 1
            if protocol == "basic_club":
                out.append(f"{X} accepted its club rate of {pc(r)} (binds {X})")
            elif p <= 1e-9:
                out.append(f"{X} accepted its request{'s' if many else ''} for {pc(r)} (" + (f"each binds only the region that accepted it)" if many else f"binds {X} only)"))
            elif r <= 1e-9:
                out.append(f"{X} accepted its offer{'s' if many else ''} of {pc(p)} ({'they bind' if many else 'binds'} {me} only)")
            else:
                out.append(f"{X} accepted its proposal{'s' if many else ''} of {pc(p)} for {pc(r)} (binds both sides)")
        for (p, r), xs in sorted(g_to.items(), key=lambda kv: (-kv[0][1], -kv[0][0])):
            X, many = regions_text(xs), len(xs) > 1
            if protocol == "basic_club":
                out.append(f"it accepted the club rate of {pc(r)} from {X} (binds {me})")
            elif p <= 1e-9:
                out.append(f"it accepted {'the requests of ' + X if many else X + chr(39) + 's request'} for {pc(r)} ({'they bind' if many else 'binds'} {me} only)")
            elif r <= 1e-9:
                out.append(f"it accepted {'the offers of ' + X if many else X + chr(39) + 's offer'} of {pc(p)} ({'each binds only its sender' if many else 'binds ' + X + ' only'})")
            else:
                out.append(f"it accepted {'the proposals of ' + X if many else X + chr(39) + 's proposal'} of {pc(p)} for {pc(r)} (binds both sides)")
        return out

    steps, feed = [], []
    rule_bits = (f"{n_regions} regions share one climate and economy; {N_s} are played by the model"
                 + (f" and {n_fixed} ({scripted[0]} to {scripted[1]}) follow a fixed plan: save {pc(fixed_sav)} of output, mitigate "
                    f"{pc(fixed_mit)}, no trade, no proposals, decline every proposal" if scripted else "") + ". "
                 + {"bilateral": "Each step: every region proposes to others a mitigation rate it promises and one it requests; each "
                                 "region accepts or rejects what it received (an accepted proposal binds its sender to the promise and the receiver to "
                                 "the request); then every region "
                                 "sets savings, mitigation (at least its minimum), export limit, import bids and tariffs. ",
                    "basic_club": "Each step: every region proposes one club mitigation rate, accepts or rejects the others' proposals, "
                                  "then sets its plan. ", "none": "Each step every region sets its plan. "}.get(protocol, "")
                 + f"{T_total} steps of 5 years from {year(1)}.")
    feed.append({"x": 0, "step": 0, "who": "Rules", "text": rule_bits})
    for k, x in enumerate(entries):
        s, t, stage, a, f = x["seat"], x["t"], x["stage"], x["a"] or {}, x["f"]
        ng = NEG[t]
        y = year(t)
        if stage == "proposal":
            if "rate" in a:
                kind = "promise" if a["rate"] > 1e-9 else "silent"
                head = f"{y} {DOT} proposes a club rate of {pc(a['rate'])}" if a["rate"] > 1e-9 else f"{y} {DOT} proposes no club"
                ftxt = cap(head.split(DOT, 1)[1].strip()) + "."
            elif "proposals" in a:
                kind, txt = prop_summary(a["proposals"])
                kind = kind or "silent"
                head = f"{y} {DOT} {txt}"
                nn = nonnull(a["proposals"])
                if not nn:
                    ftxt = "Sends no proposal."
                else:
                    groups = {}
                    for r, v in nn.items():
                        groups.setdefault(v, []).append(r)
                    parts = [(f"offers {regions_text(rs)} a deal: it mitigates at least {pc(v[0])} and each region that accepts at "
                              f"least {pc(v[1])}" if v[0] > 1e-9 else f"asks {regions_text(rs)} to mitigate at least {pc(v[1])}")
                             + ((" (fixed-plan regions, which decline)" if len(rs) > 1 else " (a fixed-plan region, which declines)")
                                if set(rs) <= fixed_labels else "")
                             for v, rs in sorted(groups.items(), key=lambda kv: (-kv[0][1], -kv[0][0]))]
                    ftxt = cap("; ".join(parts)) + ("; promises nothing in return." if kind == "ask" else ".")
            else:
                kind, head, ftxt = "silent", f"{y} {DOT} unreadable proposal", "Unreadable proposal."
        elif stage == "evaluation":
            acc = a.get("accept") or []
            got = ng["received"].get(s) or {}
            if protocol == "basic_club":
                got = {lab[j]: (ng["rate"][j], ng["rate"][j]) for j in seats if j != s and ng["rate"].get(j)}
            kind = "accept" if acc else "reject"
            if acc:
                rest = len([r for r in got if r not in acc])
                head = f"{y} {DOT} accepts {regions_text(acc)}" + (f", rejects {rest}" if rest else "")
            elif got:
                head = f"{y} {DOT} rejects all {len(got)} proposals received" if len(got) > 1 else f"{y} {DOT} rejects the one proposal received"
            else:
                head = f"{y} {DOT} received no proposal"
            rtxt = ", ".join(f"{r} {pc(v[0])}/{pc(v[1])}" for r, v in sorted(got.items(), key=lambda kv: rnum(kv[0])))
            ftxt = (f"Received {len(got)} (promise/request): {rtxt}. " if got else "Received no proposal. ") + \
                   (f"Accepts {regions_text(acc)}." if acc else ("Accepts none." if got else ""))
            ftxt = ftxt.strip()
        else:
            p = a if a.get("id") == "act" else None
            mit = (p or {}).get("mitigation", 0.0)
            kind = "mitigate" if mit > 1e-9 else "nomit"
            mn = min_shown.get((t, s))
            head = f"{y} {DOT} plan: {plan_txt(p, short=True)}" + (f" (minimum {pc(mn)})" if mn and mn > 1e-9 else "")
            ftxt = cap(plan_txt(p)) + (f"; agreed minimum {pc(mn)}" if mn and mn > 1e-9 else "") + "."
        st = {"seat": sidx[s], "x": t - 1, "head": head, "kind": kind, "act": en(x["raw"]),
              "state": en(common.state_part(x["view"])) or "Not recorded.", "t": t, "stage": stage, "a": x["a"], "s": s}
        steps.append(st)
        feed.append({"x": t - 1, "step": k, "who": seat_names[sidx[s]], "kind": kind, "text": ftxt})
        nxt = entries[k + 1] if k + 1 < len(entries) else None
        stage_end = nxt is None or nxt["t"] != t or nxt["stage"] != stage
        if stage_end and stage == "evaluation":
            if ng["evaluated"] != set(seats):
                txt = (f"Only {len(ng['evaluated'])} of {N_s} regions answered before the recording stopped; the step's agreements "
                       f"are not known.")
            elif not ng["deals"]:
                txt = "No proposal was accepted: no region is bound to mitigate this step."
            else:
                mins = {s2: min_shown.get((t, s2), ng["min_calc"][s2]) for s2 in seats}
                on = [s2 for s2 in seats if mins[s2] > 1e-9]
                off = [lab[s2] for s2 in seats if mins[s2] <= 1e-9]
                txt = (f"{len(ng['deals'])} proposal{'s' if len(ng['deals']) != 1 else ''} accepted. " + ". ".join(cap(x) for x in deals_sentences(ng["deals"]))
                       + ". " + ((f"Minimum mitigation this step: " + ", ".join(f"{lab[s2]} {pc(mins[s2])}" for s2 in on)
                                  + (f"; {regions_text(off)} 0% (not bound)." if off else "."))
                                 if on else "No region is bound this step."))
            feed.append({"x": t - 1, "step": k, "who": f"{y} agreements", "text": txt})
        if stage_end and stage == "activity" and all((t, s2) in R["plans"] for s2 in seats):
            bits = []
            if em[t - 1] is not None:
                bits.append(f"The world emitted {em[t - 1]:,.1f} GtCO2 a year and produced {out[t - 1]:,.1f} trillion of output"
                            if out[t - 1] is not None else f"The world emitted {em[t - 1]:,.1f} GtCO2 a year")
            tn = temps[t] if t < T else t_end
            if tn is not None and temps[t - 1] is not None:
                bits.append(f"temperature {temps[t - 1]:.2f} {ARROW} {tn:.2f} °C")
            sent_txt = "; ".join(bits)
            rr = [(s2, res.get((t, s2))) for s2 in seats]
            if any(r for _, r in rr):
                cons = ", ".join(f"{lab[s2]} {tr(r['consumption'])}" for s2, r in rr if r and r.get("consumption") is not None)
                scs = ", ".join(f"{lab[s2]} {r['score']:.4f}" for s2, r in rr if r and r.get("score") is not None)
                txt = (sent_txt + ". " if sent_txt else "") + (f"Consumption (trillions): {cons}. " if cons else "") + \
                      ((f"Final score: {scs}." if t == T and complete else f"Score so far: {scs}.") if scs else "")
            else:
                txt = (sent_txt + ". " if sent_txt else "") + ("The game ended; the results of this step are not in the record." if t == T else "")
            if txt.strip():
                feed.append({"x": t - 1, "step": k, "who": f"{y} to {y + 5}", "text": txt.strip()})
    stop_note = (b.get("partial") or {}).get("note")
    if not complete:
        last = entries[-1]
        feed.append({"x": last["t"] - 1, "step": len(steps) - 1, "who": "End of recording",
                     "text": f"The recording ends after {len(entries)} decisions, in the {year(last['t'])} {last['stage']} stage"
                             + (f" ({stop_note})" if stop_note else "") + ". There are no final scores."})
    tied, untied = attach_calls(steps, calls, tol)
    has_reason = any(st.get("thinking") for st in steps)
    reason_chars = sum(st.get("chars", 0) for st in steps)
    rescued = [st for st in steps if st.get("rescued")]
    retried = [st for st in steps if st.get("retries")]

    # ---- the game in numbers -------------------------------------------------------------------------------------------
    plans = [(t, s, R["plans"][(t, s)]) for t in range(1, T + 1) for s in seats if (t, s) in R["plans"]]
    mit_all = [p.get("mitigation", 0.0) for _, _, p in plans]
    any_mit = any(m > 1e-9 for m in mit_all) or any(v > 1e-9 for v in min_shown.values())
    sav_zero = [(t, s) for t, s, p in plans if p.get("savings", 0) <= 1e-9]
    sav_high = [(t, s, p["savings"]) for t, s, p in plans if p.get("savings", 0) >= 0.9 - 1e-9]
    sav_mean = {s: (sum(p.get("savings", 0) for t, s2, p in plans if s2 == s) / max(1, sum(1 for t, s2, p in plans if s2 == s))) for s in seats}
    trade_plans = [(t, s, p) for t, s, p in plans if p.get("export_limit", 0) > 1e-9 or any(v > 1e-9 for v in (p.get("imports") or {}).values())
                   or any(v > 1e-9 for v in (p.get("tariffs") or {}).values())]
    sent_all = [(t, s, r, v) for t in range(1, T + 1) for s in seats for r, v in nonnull(NEG[t]["sent"].get(s)).items()]
    to_seats = [z for z in sent_all if z[2] in R["by_label"]]
    to_fixed = [z for z in sent_all if z[2] not in R["by_label"]]
    with_promise = [z for z in to_seats if z[3][0] > 1e-9]
    asks_only = [z for z in to_seats if z[3][0] <= 1e-9]
    req_counts = {}
    for z in asks_only:
        req_counts[z[3][1]] = req_counts.get(z[3][1], 0) + 1
    deals_all = [(t, d) for t in range(1, T + 1) for d in NEG[t]["deals"]]
    real_all = [(t, d) for t, d in deals_all if not d["null"]]
    evals = [(t, s) for t in range(1, T + 1) for s in seats if (t, s) in R["accs"]]
    evals_with = [(t, s) for t, s in evals if NEG[t]["received"].get(s)]
    evals_rejected = [(t, s) for t, s in evals_with if not R["accs"][(t, s)]]
    silent_props = [(t, s) for t in range(1, T + 1) for s in seats if (t, s) in R["props"] and not nonnull(R["props"][(t, s)])]
    n_props_moves = sum(1 for t in range(1, T + 1) for s in seats if (t, s) in R["props"] or (t, s) in R["rates"])
    tv = [temps[xi] if xi < T else t_end for xi in range(n_x)]      # temperature at each x (start of the step; the end)
    known_t = [(xi, v) for xi, v in enumerate(tv) if v is not None]
    if known_t and temp0 is None:
        temp0 = known_t[0][1]
    temp_known = bool(known_t) and temp0 is not None
    temp_last = known_t[-1][1] if known_t else None
    y_last = int(labels[known_t[-1][0]]) if known_t else year(T)
    score_last = {s: (rewards[s] if complete and s in rewards else None) for s in seats}
    if not any(v is not None for v in score_last.values()):           # the last score any prompt reported
        for s in seats:
            tt = max((t for t in range(1, T + 1) if (res.get((t, s)) or {}).get("score") is not None), default=None)
            score_last[s] = res[(tt, s)]["score"] if tt else None
        score_when = max((t for t in range(1, T + 1) for s in seats if (res.get((t, s)) or {}).get("score") is not None), default=None)
    else:
        score_when = T

    # ---- agreements: who accepted what and whom it bound; who mitigated ------------------------------------------------------
    acceptors = sorted({d["to"] for _, d in real_all})
    req_acc = [(t, d) for t, d in real_all if protocol == "bilateral" and d["p"] <= 1e-9]   # accepted requests promising nothing
    prom_acc = [(t, d) for t, d in real_all if protocol == "bilateral" and d["p"] > 1e-9]
    offers_acc = [z for z in with_promise if any(d["by"] == z[1] and lab[d["to"]] == z[2] for d in NEG[z[0]]["real"])]
    mit_at = {(t, s): p["mitigation"] for t, s, p in plans if p.get("mitigation", 0) > 1e-9}
    mit_regs = sorted({s for _, s in mit_at})

    def rate_span(dct, s):
        """A region's rates in dct {(t, s): rate} with their years: '10% from 2020 to 2040', '10% to 30% in 2015 and 2030'."""
        ts = sorted(t for (t, s2) in dct if s2 == s)
        rs = sorted({round(dct[(t, s)], 6) for t in ts})
        return (pc(rs[0]) if len(rs) == 1 else f"{pc(rs[0])} to {pc(rs[-1])}") + " " + years_span([year(t) for t in ts])

    def offers_txt(zs, single=False):
        pairs = sorted({z[3] for z in zs})
        w = "offer" if single else "offers"
        return f"{w} of {pc(pairs[0][0])} for {pc(pairs[0][1])}" if len(pairs) == 1 else f"{w} with a promise"

    def mit_story(subject=None):
        """Who mitigated, e.g. 'it mitigates 10% from 2015 to 2040, the others never do'."""
        if not mit_regs:
            return "no region mitigates"
        if len(mit_regs) <= 3:
            same = {}
            for s in mit_regs:
                same.setdefault(rate_span(mit_at, s), []).append(s)
            out = [(f"it mitigates {rs}" if ss == [subject] else f"{regions_text([lab[s] for s in ss])} mitigate{'s' if len(ss) == 1 else ''} {rs}")
                   for rs, ss in same.items()]
            return list_txt(out) + (", the others never do" if len(mit_regs) < N_s else "")
        return f"the mean mitigation rate is {pc(round(sum(mit_all) / max(1, len(mit_all)), 3))} over {len(plans)} plans"

    def mit_detail(s1, only=False):
        """'R1 mitigated 10% from 2015 to 2040 (bound by agreements from 2020 to 2040; on its own in 2015)'; with only=True
        'Only R1 ever mitigated: 10% from 2015 to 2040 (...)'."""
        ts_b = sorted(t for (t, s2) in bound_at if s2 == s1 and (t, s2) in mit_at)    # bound steps whose plan is recorded
        own = sorted(t for (t, s2) in mit_at if s2 == s1 and (t, s2) not in bound_at)
        return ((f"Only {lab[s1]} ever mitigated: " if only else f"{lab[s1]} mitigated ") + rate_span(mit_at, s1)
                + ((f" (bound by agreements {years_span([year(t) for t in ts_b])}"
                    + (f"; on its own {years_span([year(t) for t in own])}" if own else "") + ")") if ts_b else " on its own"))

    bound_txt = list_txt(f"{lab[s]} at {rate_span(bound_at, s)}" for s in bound_regs)       # 'R1 at 10% from 2020 to 2040'
    agree_txt = agree_tile = ""
    if real_all:
        n_ra, k_ra = len(real_all), len({t for t, _ in real_all})
        all_req = bool(req_acc) and len(req_acc) == n_ra
        agree_txt = (f"{n_ra} proposal{'s were' if n_ra != 1 else ' was'} accepted, in {k_ra} step{'s' if k_ra != 1 else ''}"
                     + ((": it was a request that promised nothing, so it bound only the region that accepted it" if n_ra == 1 else
                         f": {'both' if n_ra == 2 else 'all'} were requests that promised nothing, so each bound only the region that accepted it")
                        + (f" ({bound_txt})" if bound_txt else "") if all_req else
                        f"; {len(prom_acc)} of them carried a promise" + (f". Bound: {bound_txt}" if bound_txt else "") if prom_acc else
                        (f". Bound: {bound_txt}" if bound_txt else "")) + ".")
        agree_tile = ((("The one accepted was a request that promised nothing, so it bound only the region that accepted it" if n_ra == 1 else
                        f"{'Both' if n_ra == 2 else 'All ' + str(n_ra)} accepted were requests that promised nothing, so each bound only "
                        "the region that accepted it")
                       + (f": {bound_txt}" if bound_txt else "") + ".") if all_req else
                      (f"Agreements bound {bound_txt}." if bound_txt else f"Agreements in {k_ra} step{'s' if k_ra != 1 else ''}."))
        if with_promise:
            n_wp = len(with_promise)
            o_txt = (f" None of the {n_wp} proposals with a promise was accepted." if not offers_acc and n_wp > 1 else
                     " The one proposal with a promise was not accepted." if not offers_acc else
                     (f" {'Both' if n_wp == 2 else 'All ' + str(n_wp)} proposals with a promise were accepted." if n_wp > 1 else
                      " The one proposal with a promise was accepted.")
                     if len(offers_acc) == n_wp else
                     f" {len(offers_acc)} of the {n_wp} proposals with a promise {'were' if len(offers_acc) != 1 else 'was'} accepted.")
            agree_txt += o_txt
            agree_tile += o_txt

    # ---- charts ---------------------------------------------------------------------------------------------------------
    charts = []
    # 1. negotiation grid
    def fate(t, s, r):
        """What became of s's proposal to region r in step t."""
        j = R["by_label"].get(r)
        if j is None:
            return "(fixed plan, declined)"
        if (t, j) not in R["accs"]:
            return "(no answer recorded)"
        return "(accepted)" if lab[s] in R["accs"][(t, j)] else "(rejected)"
    if negot:
        cells, c4_cases = [], set()
        for s in seats:
            row = []
            for xi in range(n_x):
                t = xi + 1
                if t > T or not any((t, s, st_) in views for st_ in STAGES):
                    row.append(None)
                    continue
                ng = NEG[t]
                sent = nonnull(ng["sent"].get(s)) if protocol == "bilateral" else {}
                dl = deals_of(t, s)
                mn = min_at(t, s)
                bound = mn is not None and mn > 1e-9
                lines_ = []
                if protocol == "basic_club":
                    rt = ng["rate"].get(s)
                    lines_.append(f"proposed a club rate of {pc(rt)}" if rt else "proposed no club")
                elif sent:
                    lines_.append("sent (promise/request): " + ", ".join(
                        f"{r} {pc(v[0])}/{pc(v[1])} {fate(t, s, r)}" for r, v in sorted(sent.items(), key=lambda kv: rnum(kv[0]))))
                else:
                    lines_.append("sent no proposal")
                got = ng["received"].get(s) or {}
                if (t, s) in R["accs"]:
                    acc = R["accs"][(t, s)]
                    lines_.append(("received: " + ", ".join(f"{r} {pc(v[0])}/{pc(v[1])}" for r, v in sorted(got.items(), key=lambda kv: rnum(kv[0])))
                                   + (f"; accepted {regions_text(acc)}" if acc else "; accepted none")) if got else
                                  ("received none" + (f"; accepted {regions_text(acc)}" if acc else "")))
                elif got:
                    lines_.append("received: " + ", ".join(f"{r} {pc(v[0])}/{pc(v[1])}" for r, v in got.items()) + "; answer not recorded")
                if (t, s) not in R["accs"] and not complete and t == T:
                    lines_.append("the recording stopped before its answer")
                if mn is not None:
                    lines_.append(f"minimum {pc(mn)}" + ("" if bound else " (not bound)"))
                p = R["plans"].get((t, s))
                if p:
                    lines_.append(f"then {plan_txt(p, short=True)}")
                tip = cap((f"bound at {pc(mn)}: " if bound and dl else f"bound at {pc(mn)}; " if bound else "") + "; ".join(dl + lines_))
                if bound:                                   # only a minimum above 0 binds (an accepted 0% promise binds nothing)
                    row.append({"c": "c1", "o": .9, "t": pcn(mn), "tip": tip})
                elif protocol == "bilateral" and dl and mn is not None:
                    # a party to an accepted proposal that binds only the other side: its request accepted (it promised 0), or
                    # it accepted an offer that asked nothing of it; the number is what the other side is bound to
                    other = [d["r"] for d in ng["real"] if d["by"] == s] + [d["p"] for d in ng["real"] if d["to"] == s]
                    c4_cases.update(["req"] if any(d["by"] == s for d in ng["real"]) else [])
                    c4_cases.update(["offer"] if any(d["to"] == s for d in ng["real"]) else [])
                    row.append({"c": "c4", "o": .85, "t": pcn(max(other)), "tip": tip})
                elif protocol == "basic_club":
                    rt = ng["rate"].get(s)
                    row.append({"c": "c3", "o": .8, "t": pcn(rt), "tip": tip} if rt else {"c": "muted", "o": .5, "tip": tip})
                elif sent and max(v[0] for v in sent.values()) > 1e-9:
                    row.append({"c": "c3", "o": .85, "t": pcn(max(v[0] for v in sent.values())), "tip": tip})
                elif sent:
                    row.append({"c": "c2", "o": round(.3 + .6 * max(v[1] for v in sent.values()), 3), "t": pcn(max(v[1] for v in sent.values())), "tip": tip})
                else:
                    row.append({"c": "muted", "o": .5, "tip": tip})
            cells.append(row)
        used = {c["c"] for row in cells for c in row if c}
        c4_label = {frozenset(["req"]): ("Its request was accepted, but it promised nothing: only the region that accepted is bound "
                                         "(number: the request, %)"),
                    frozenset(["offer"]): ("It accepted an offer that asked nothing of it: only the offering region is bound "
                                           "(number: the offer, %)")}.get(frozenset(c4_cases),
                                                                          "In an accepted proposal that binds only the other region "
                                                                          "(number: the rate the other region is bound to, %)")
        c4_sub = {frozenset(["req"]): ("yellow: a request of its was accepted, but it promised nothing, so only the region that "
                                       "accepted is bound (number: the request)"),
                  frozenset(["offer"]): "yellow: it accepted an offer that asked nothing of it, so only the offering region is bound "
                                        "(number: the offer)"}.get(frozenset(c4_cases), "yellow: it was a party to an accepted proposal "
                                                                   "that binds only the other region (number: the rate that region is bound to)")
        if protocol == "bilateral":
            rq = sorted(req_counts.items(), key=lambda kv: -kv[0])
            rq_txt = and_list(f"{n} for {pc(r)}" for r, n in rq)
            prom_groups = {}
            for t, s, r, v in with_promise:
                prom_groups.setdefault((t, s, v), []).append(r)
            prom_txt = "; ".join(f"{lab[s]} in {year(t)}, {pc(v[0])} for {pc(v[1])} to {regions_text(rs)}" for (t, s, v), rs in sorted(prom_groups.items()))
            colors_txt = "; ".join(x for c, x in (
                ("c2", "orange: it asked others to mitigate and promised nothing (number: its highest request, in %; darker for a larger request)"),
                ("c3", "green: it promised a rate (number: its highest promise)"),
                ("c1", "blue: an accepted proposal bound it, i.e. its minimum mitigation rate is above 0 (number: that minimum)"),
                ("c4", c4_sub), ("muted", "pale: it sent no proposal")) if c in used)
            sub = (f"One row per region, one cell per step: what the region proposed to the other {NUM_WORDS.get(N_s - 1, N_s - 1)} model-played regions"
                   + (" (or to fixed-plan regions)" if to_fixed else "") + f" and whether an accepted proposal bound it. {cap(colors_txt)}. "
                   f"Hover for every proposal sent and received and the answers. Over the game the {nw} regions sent each other {len(to_seats)} proposals"
                   + (f" (and {len(to_fixed)} to fixed-plan regions, which decline everything)" if to_fixed else "")
                   + (f": {len(asks_only)} asked for mitigation without promising any ({rq_txt})" if asks_only else "")
                   + (f", {len(with_promise)} promised a rate ({prom_txt})" if with_promise else ", none promised a rate")
                   + f"; in {len(silent_props)} of {n_props_moves} proposal stages a region sent nothing. "
                   + (f"Every region rejected every proposal it received ({len(evals_rejected)} evaluations with proposals to answer), so no "
                      f"region was ever bound." if not deals_all and evals_with else
                      (agree_txt if real_all else "No proposal binding anyone was accepted.")))
        else:
            sub = (f"One row per region, one cell per step. Green: it proposed a club rate (number: the rate, in %); blue: it was bound by "
                   f"an accepted proposal (number: its minimum); pale: no proposal. Hover for details. {len(real_all)} proposals accepted.")
        legend = [x for x in (["Bound by an accepted proposal (number: its minimum, %)", "c1"], [c4_label, "c4"],
                              ["Promised a rate (number: highest promise, %)", "c3"],
                              ["Asked others only, promised nothing (number: highest request, %)", "c2"], ["Sent no proposal", "muted"])
                  if x[1] in used]
        charts.append({"title": "Proposals and agreements, step by step", "sub": sub,
                       "grid": {"rows": seat_names, "cells": cells, "legend": legend, "rowHeight": 26}})

    # 2. plans: mitigation and savings
    def series_plan(s, k):
        return [round(100 * R["plans"][(xi + 1, s)].get(k, 0.0), 6) if (xi + 1, s) in R["plans"] else None for xi in range(n_x)]
    played = [xi for xi in range(n_x) if any((xi + 1, s) in R["plans"] for s in seats)]
    zero_txt = (f"{len(sav_zero)} of {len(plans)} plans saved nothing" + (f", the first in {year(min(t for t, _ in sav_zero))}" if sav_zero else "")) if plans else ""
    high_txt = ""
    if sav_high:
        hs = []
        for t, s, v in sav_high:
            r = res.get((t, s)) or {}
            stp = next((st for st in steps if st["s"] == s and st["t"] == t and st["stage"] == "activity"), None)
            hs.append(f"{lab[s]} invested {pc(v)} of its {year(t)} output"
                      + ((" and consumed nothing that step" if r["consumption"] <= 1e-9 else f" and consumed {tr(r['consumption'])} trillion that step")
                         if r.get("consumption") is not None else "")
                      + (" (an answer to a follow-up question after its reasoning ran out)" if stp and stp.get("rescued") else ""))
        high_txt = " " + "; ".join(hs) + "."
    mean_txt = ", ".join(f"{lab[s]} {pc(round(sav_mean[s], 3))}" for s in seats)
    trade_txt = (f" Trade: {len(trade_plans)} plans set an export limit, import bids or tariffs." if trade_plans else
                 " No plan ever allowed exports, bid on imports or set a tariff.") if plans else ""
    if plans and not any_mit:
        lines = [dict(name=seat_names[sidx[s]], color=scol[s], dots=True, v=series_plan(s, "savings")) for s in seats]
        lines.append({"name": f"Mitigation, all {nw} regions", "color": "hold", "dash": True, "v": [0.0 if xi in played else None for xi in range(n_x)]})
        charts.append({"title": "Savings and mitigation in each plan (%)",
                       "sub": (f"Solid lines: the share of its output each region invested (savings rate), as set in its plan for each "
                               f"5-year step. Dashed: the mitigation rate, 0% in all {len(plans)} plans of the {nw} regions: no region "
                               f"ever avoided any of its emissions" + (f", and the {n_fixed} fixed-plan regions save {pc(fixed_sav)} and "
                                                                       f"mitigate {pc(fixed_mit)} throughout" if scripted else "")
                               + f". Average savings: {mean_txt}; {zero_txt}." + high_txt + trade_txt),
                       "height": 240, "fmt": "pct0", "lines": [l for l in lines if any(v is not None for v in l["v"])]})
    elif plans:
        lines = [dict(name=seat_names[sidx[s]], color=scol[s], dots=True, v=series_plan(s, "mitigation")) for s in seats]
        lines += [dict(name=f"{lab[s]} savings", color=scol[s], dash=True, thin=True, v=series_plan(s, "savings")) for s in seats]
        for s in seats:
            v = [round(100 * min_shown[(xi + 1, s)], 6) if min_shown.get((xi + 1, s), 0) > 1e-9 else None for xi in range(n_x)]
            if any(x is not None for x in v):
                lines.append({"name": f"{lab[s]} agreed minimum", "color": scol[s], "noline": True, "dots": True, "v": v})
        mm = sum(mit_all) / len(mit_all)
        top = max(plans, key=lambda z: z[2].get("mitigation", 0))
        has_min = any(l.get("noline") for l in lines)
        charts.append({"title": "Mitigation and savings in each plan (%)",
                       "sub": (f"Solid lines: each region's mitigation rate (share of its emissions avoided); dashed: its savings rate "
                               f"(share of output invested)" + ("; dots: the minimum its accepted agreements required" if has_min else "")
                               + (f". {mit_detail(mit_regs[0], only=True)}, in {len(mit_at)} of the {len(plans)} plans." if len(mit_regs) == 1 and N_s > 1 else
                                  f". Mean mitigation {pc(round(mm, 3))} over {len(plans)} plans; the highest, {pc(top[2].get('mitigation', 0))}, by "
                                  f"{lab[top[1]]} in {year(top[0])}.")
                               + (f" The {n_fixed} fixed-plan regions mitigate {pc(fixed_mit)}." if scripted else "")
                               + f" Average savings: {mean_txt}; {zero_txt}." + high_txt + trade_txt),
                       "height": 260, "fmt": "pct0", "lines": [l for l in lines if any(v is not None for v in l["v"])]})

    # 3. climate
    if temp_known:
        e_known = [(t, em[t - 1]) for t in range(1, T + 1) if em[t - 1] is not None]
        o_known = [(t, out[t - 1]) for t in range(1, T + 1) if out[t - 1] is not None]
        k_known = [(t, kept[t - 1]) for t in range(1, T + 1) if kept[t - 1] is not None]
        csub = (f"Atmospheric temperature above pre-industrial at the start of each 5-year step"
                + (f" and at the end of the game ({year(T + 1)})" if t_end is not None else "")
                + f": {temp0:.2f} °C in {year(1)}, {temp_last:.2f} °C in {y_last}. Every region's emissions are its output times its "
                  f"carbon intensity times (1 {MINUS} its mitigation rate), and all of them warm the one shared atmosphere.")
        if scripted and share:
            a_t, b_t = share_t[0], share_t[-1]
            csub += (f" The {nw} model regions emitted {sh(share[a_t])} of the world's CO2 in the {year(a_t)} step"
                     + (f" and {sh(share[b_t])} in the {year(b_t)} step" if b_t != a_t else "")
                     + f" (their own emissions in their prompts against the world total); the other {n_fixed} regions follow a fixed plan "
                     + ("without mitigation" if fixed_mit <= 1e-9 else f"that mitigates {pc(fixed_mit)}")
                     + (" and emit the rest, so they set most of the warming." if share_max < 0.5 else " and emit the rest."))
        if e_known:
            csub += (f" World emissions rose from {e_known[0][1]:,.1f} GtCO2 a year in the {year(e_known[0][0])} step to "
                     f"{e_known[-1][1]:,.1f} in the {year(e_known[-1][0])} step")
            if o_known:
                csub += f", world output from {o_known[0][1]:,.1f} to {o_known[-1][1]:,.1f} trillion"
            csub += "."
        cs = ser("carbon_series")
        c_end = (cs[T], year(T + 1)) if complete and cs and len(cs) > T and cs[T] is not None else (carb[-1], year(T))
        if carb[0] is not None and c_end[0] is not None:
            csub += f" Carbon in the atmosphere: {carb[0]:,.0f} GtC in {year(1)}, {c_end[0]:,.0f} in {c_end[1]}."
        if k_known:
            csub += (f" Climate damage took {100 * (1 - k_known[0][1]):.1f}% of every region's production in the {year(k_known[0][0])} step and "
                     f"{100 * (1 - k_known[-1][1]):.1f}% in the {year(k_known[-1][0])} step.")
        if scripted:
            same_plan = abs(fixed_sav - 0.3) < 1e-9 and fixed_mit <= 1e-9
            csub += (f" For scale, the rules' fixed-policy runs (all {REF_REGIONS} regions alike, no trade) end in {year(1) + 100} at "
                     f"{BAU_T2115:.2f} °C when every region saves 30% and never mitigates"
                     + (f" (the plan the {n_fixed} fixed-plan regions follow)" if same_plan else "")
                     + (f"; the {FULL_T2115:.2f} °C of full mitigation needs all {REF_REGIONS} regions and is out of reach here: the {nw} "
                        f"model regions can avoid at most their own emissions, never more than {sh(share_max)} of the world's." if share else
                        f"; the {FULL_T2115:.2f} °C of full mitigation needs all {REF_REGIONS} regions, not only the {nw} played here."))
        else:
            csub += (f" For scale, the rules' fixed-policy runs (all {REF_REGIONS} regions alike, no trade) end in {year(1) + 100} at "
                     f"{BAU_T2115:.2f} °C when every region saves 30% and never mitigates and at {FULL_T2115:.2f} °C with full mitigation.")
        charts.append({"title": "Global temperature (°C above pre-industrial)", "sub": csub, "height": 220, "fmt": "price",
                       "lines": [{"name": "Atmospheric temperature", "color": "c2", "area": True, "dots": True, "v": tv}]})

    # 4. score so far
    sc_lines = []
    for s in seats:
        v = []
        for xi in range(n_x):
            t = xi + 1
            if xi == T:                                 # the end point: the final score
                v.append(rewards.get(s) if complete else None)
            elif t >= 2:
                r = res.get((t - 1, s)) or {}
                v.append(r.get("score"))
            else:
                v.append(None)
        sc_lines.append(dict(name=seat_names[sidx[s]], color=scol[s], dots=True, v=v))
    if any(v is not None for l in sc_lines for v in l["v"]):
        drops = []
        for s, l in zip(seats, sc_lines):
            for xi in range(1, n_x):
                if l["v"][xi] is not None and l["v"][xi - 1] is not None:
                    drops.append((l["v"][xi] - l["v"][xi - 1], s, xi))
        dtxt = ""
        if drops:
            d, s, xi = min(drops)
            if d < -0.005:
                p = R["plans"].get((xi, s)) or {}
                r = res.get((xi, s)) or {}
                why = ""
                if r.get("consumption") is not None and r["consumption"] <= 1e-9:
                    why = f" after it invested {pc(p.get('savings'))} of its {year(xi)} output and consumed nothing"
                elif p:
                    why = f" after its {year(xi)} plan ({plan_txt(p, short=True)})"
                dtxt = (f" The score is a weighted average of these step ratios, so it falls after a step whose ratio is below the "
                        f"average so far: the largest fall was "
                        f"{lab[s]}'s, from {sc_lines[sidx[s]]['v'][xi - 1]:.4f} to {sc_lines[sidx[s]]['v'][xi]:.4f}{why}.")
        cap_txt = ""                                    # what a step earns depends on consumption per person
        r1 = {s: res.get((1, s)) or {} for s in seats}
        p1 = {s: (first_view.get((1, s)) or {}).get("pop") for s in seats}
        ok1 = [s for s in seats if r1[s].get("score") is not None and r1[s].get("consumption") is not None and p1[s]]
        if len(ok1) >= 2:
            hi = max(ok1, key=lambda s: r1[s]["score"])
            lo = min(ok1, key=lambda s: r1[s]["score"])
            cap_txt = (f" A step's utility relative to its maximum depends only on the region's consumption per person (and on "
                       f"tariffs others set on its exports), so regions start far apart: in the {year(1)} step {lab[hi]} consumed {tr(r1[hi]['consumption'])} trillion for "
                       f"{p1[hi]:,.1f} million people (score {r1[hi]['score']:.4f}), {lab[lo]} {tr(r1[lo]['consumption'])} trillion for "
                       f"{p1[lo]:,.1f} million ({r1[lo]['score']:.4f}).")
        charts.append({"title": "Score so far, by region",
                       "sub": (f"Each region's score as its prompt showed it at the start of each step: the discounted sum of its "
                               f"utilities from consumption so far, divided by the largest sum possible so far (0 to 1)"
                               + (f"; {year(T + 1)}: the final score, the same ratio over all {T} steps." if complete else
                                  f". The recording stopped before the end, so the last values are the scores after the {year(score_when)} step." if score_when else ".")
                               + cap_txt + dtxt),
                       "height": 220, "fmt": "price4", "lines": [l for l in sc_lines if any(v is not None for v in l["v"])]})

    # ---- tiles ----------------------------------------------------------------------------------------------------------
    tiles = []
    fixed_plan_txt = "a fixed plan without mitigation" if fixed_mit <= 1e-9 else f"a fixed plan that mitigates {pc(fixed_mit)}"
    if share:
        a_t, b_t = share_t[0], share_t[-1]
        share_tile = (f"The {nw} model regions emitted {sh(share[a_t])} of the world's CO2 in the {year(a_t)} step"
                      + (f" and {sh(share[b_t])} in the {year(b_t)} step" if b_t != a_t else "")
                      + f"; the other {n_fixed} follow {fixed_plan_txt}" + (", so the warming is set mostly by them." if share_max < 0.5 else "."))
    else:
        share_tile = f"The other {n_fixed} regions follow {fixed_plan_txt}."
    if temp_known:
        tiles.append({"k": "Global temperature", "v": f"{temp0:.2f} {ARROW} {temp_last:.2f} °C",
                      "spark": [round(v, 4) for v in tv if v is not None],
                      "s": (f"Above pre-industrial, {year(1)} {ARROW} {y_last}"
                            + (" (the recording stopped before the last step ended)" if not complete else "")
                            + (". " + share_tile + f" The rules' business-as-usual run (all {REF_REGIONS} regions save 30%, never mitigate) "
                               f"ends at {BAU_T2115:.2f} °C in {year(1) + 100}." if scripted else
                               f". The rules' fixed-policy runs end at {BAU_T2115:.2f} °C in {year(1) + 100} if every region saves 30% and never "
                               f"mitigates, {FULL_T2115:.2f} °C with full mitigation; the paper's trained regions reach "
                               f"{PAPER_T2115['bilateral']:.2f} °C in {year(1) + 100} with bilateral negotiation, {PAPER_T2115['none']:.2f} °C without."))})
    if negot and protocol == "basic_club":
        n_rates = sum(1 for (t, s), v in R["rates"].items() if v > 1e-9)
        tiles.append({"k": "Club rates accepted", "v": f"{len(real_all)}",
                      "s": (f"Acceptances of a club rate over {T} steps; the {nw} regions proposed a club rate in {n_rates} of "
                            f"{n_props_moves} proposal stages. " + (agree_tile or "No region was ever bound to mitigate."))})
    elif negot:
        tiles.append({"k": "Proposals accepted", "v": f"{len(real_all)} of {len(to_seats)}",
                      "s": (f"Proposals the {nw} regions sent each other over {T} steps"
                            + (f"; {len(asks_only)} asked others to mitigate (up to {pc(max(v[1] for *_, v in asks_only))}) while promising nothing"
                               if asks_only else "")
                            + (f", {len(with_promise)} promised a rate" if with_promise else ", none promised a rate")
                            + (". No region was ever bound to mitigate." if not real_all else ". " + agree_tile))})
    if plans:
        if not any_mit:
            tiles.append({"k": "Mitigation chosen", "v": "0% in every plan",
                          "s": (f"The mitigation rate was 0 in all {len(plans)} plans of the {nw} regions"
                                + (f", as for the {n_fixed} fixed-plan regions" if scripted and fixed_mit <= 1e-9 else "")
                                + f". Savings averaged {pc(round(sum(p.get('savings', 0) for *_, p in plans) / len(plans), 3))}; "
                                  f"{len(sav_zero)} plans invested nothing.")})
        elif len(mit_regs) == 1 and N_s > 1:                # one region carries all the mitigation: its figure, not the all-plan mean
            mm = sum(mit_all) / len(mit_all)
            s1 = mit_regs[0]
            r1 = sorted({round(v, 6) for (t, s2), v in mit_at.items() if s2 == s1})
            tiles.append({"k": "Mitigation chosen", "v": f"{lab[s1]} only, {pc(r1[0])}" if len(r1) == 1 else f"{lab[s1]} only",
                          "s": (f"{mit_detail(s1)}; {regions_text([lab[s] for s in seats if s != s1])} never mitigated. "
                                f"{len(mit_at)} of {len(plans)} plans mitigated anything; the mean over all plans is {pc(round(mm, 3))}.")})
        else:
            mm = sum(mit_all) / len(mit_all)
            tiles.append({"k": "Mitigation chosen", "v": f"{pc(round(mm, 3))} on average",
                          "s": ("By region: " + ", ".join(
                              f"{lab[s]} {pc(round(sum(p.get('mitigation', 0) for t, s2, p in plans if s2 == s) / max(1, sum(1 for t, s2, p in plans if s2 == s)), 3))}"
                              for s in seats) + f". {sum(1 for m in mit_all if m > 1e-9)} of {len(plans)} plans mitigated anything.")})
    sv = {s: v for s, v in score_last.items() if v is not None}
    if sv:
        hi, lo = max(sv, key=sv.get), min(sv, key=sv.get)
        tiles.append({"k": "Final score" if complete and rewards else f"Score after the {year(score_when)} step",
                      "v": f"{sv[lo]:.3f} to {sv[hi]:.3f}",
                      "s": (", ".join(f"{lab[s]} {sv[s]:.3f}" for s in seats if s in sv) + ". Discounted utility from consumption "
                            "divided by its largest possible value (0 to 1)"
                            + ("." if complete and rewards else "; the recording stopped before the final scores.")
                            + (f" Highest: {lab[hi]}." if complete and rewards else ""))})
    if len(tiles) < 2:
        tiles.append({"k": "Decisions recorded", "v": f"{len(steps)}",
                      "s": f"Model decisions in {T} step{'s' if T != 1 else ''}" + ("" if complete else "; the recording stopped before the end") + "."})
    tiles = tiles[:4]

    # ---- title, lede, foot ----------------------------------------------------------------------------------------------
    # With fixed-plan regions the model regions emit a small share of the CO2, so the title gives that share, not the warming
    # (the warming would read as their doing); with every region played, the warming is theirs.
    if scripted:
        warm = ""
        if share:
            a_t, b_t = share_t[0], share_t[-1]
            warm = (f". The {nw} model regions emit {sh(share[a_t])} of world CO2 in {year(a_t)}"
                    + (f", {sh(share[b_t])} in {year(b_t)}" if b_t != a_t else "")
                    + (f"; the {n_fixed} fixed-plan regions drive the warming" if share_max < 0.5 else ""))
    else:
        warm = f", and the world warms from {temp0:.2f} to {temp_last:.2f} °C by {y_last}" if temp_known and y_last > year(1) else ""
    offers = {}
    for t, s, r, v in with_promise:
        offers.setdefault((t, s), set()).add(v)
    if not plans:
        last_e = entries[-1]
        title = (f"Cut short: the recording stops after {len(steps)} decisions, in the {year(last_e['t'])} {last_e['stage']} stage"
                 + (f"; {len(real_all)} of the {len(to_seats)} proposals sent so far were accepted" if negot and to_seats else ""))
    elif negot and not real_all and not any_mit:
        if not with_promise:
            what = "every proposal asks the others to cut emissions and promises nothing"
        elif len(offers) == 1 and len(next(iter(offers.values()))) == 1:
            (t1, s1), vs = next(iter(offers.items()))
            v1 = next(iter(vs))
            what = (f"apart from {lab[s1]}'s {pc(v1[0])}-for-{pc(v1[1])} offer in {year(t1)}, every proposal asks the others to cut "
                    f"emissions and promises nothing")
        else:
            what = (f"{len(with_promise)} of the {len(to_seats)} proposals promise any mitigation, the rest only ask the others to "
                    f"cut emissions")
        title = f"No deal and no mitigation: {what}; none is accepted" + warm
    elif negot and real_all:
        n_ra = len(real_all)
        if bound_regs and len(bound_regs) < N_s:        # only some regions are ever bound: name them and say how
            one = len(bound_regs) == 1
            subj = bound_regs[0] if one else None
            title = f"Only {regions_text([lab[s] for s in bound_regs])} {'is' if one else 'are'} ever bound"
            if req_acc and len(req_acc) == n_ra and set(acceptors) == set(bound_regs):
                title += (f": {'it accepts' if one else 'they accept'} {n_ra} request{'s' if n_ra != 1 else ''} from regions that "
                          f"promise nothing")
            else:
                title += f" by the {n_ra} accepted proposal{'s' if n_ra != 1 else ''}"
            own = [z for z in with_promise if z[1] in bound_regs]
            if own:
                acc_own = [z for z in offers_acc if z[1] in bound_regs]
                owners = sorted({z[1] for z in own})
                pron = "its" if one else f"{lab[owners[0]]}'s" if len(owners) == 1 else "their"
                title += ((f", and none of {pron} {len(own)} {offers_txt(own)} is accepted" if len(own) > 1 else
                           f", and {pron} one {offers_txt(own, single=True)} is not accepted") if not acc_own else
                          (f", and {'both' if len(own) == 2 else 'all ' + str(len(own))} of {pron} {offers_txt(own)} are accepted" if len(own) > 1 else
                           f", and {pron} one {offers_txt(own, single=True)} is accepted") if len(acc_own) == len(own) else
                          f", and {len(acc_own)} of {pron} {len(own)} {offers_txt(own)} {'are' if len(acc_own) != 1 else 'is'} accepted")
            title += "; " + mit_story(subj)
        elif len(bound_regs) == N_s:
            title = (f"{n_ra} accepted proposal{'s' if n_ra != 1 else ''} in {len({t for t, _ in real_all})} of {T} steps bind every "
                     f"region at least once; " + mit_story())
        else:                                           # accepted proposals, but no step's minimum is known yet (cut short)
            title = (f"{n_ra} accepted proposal{'s' if n_ra != 1 else ''} in {len({t for t, _ in real_all})} of {T} steps; "
                     + mit_story())
        title += warm
    elif negot:
        title = "No proposal is accepted; " + mit_story() + warm
    else:
        title = cap(mit_story()) + warm
    lede = (f"{nw.capitalize()} regions, each played by the model, share one climate-economy"
            + (f" with {n_fixed} more that follow a fixed plan (save {pc(fixed_sav)} of output, "
               + ("never mitigate" if fixed_mit <= 1e-9 else f"mitigate {pc(fixed_mit)}") + ")"
               + (f" and emit {sh(1 - share_max)} or more of the world's CO2 in every step" if share else "") if scripted else "")
            + f". Every 5-year step from {year(1)} to {year(T_total)} "
            + ({"bilateral": "each region proposes to the others a mitigation rate it promises and one it requests, accepts or rejects "
                             "the proposals it received (an accepted proposal binds its sender to the promise and the receiver to the "
                             "request), then sets how much of its output to "
                             "invest and how much of its emissions to avoid, and its trade. ",
                "basic_club": "each region proposes one club mitigation rate, accepts or rejects the others' proposals (the largest "
                              "accepted rate binds it), then sets its savings, mitigation and trade. "}.get(protocol, "each region sets its "
                                                                                                                   "savings, mitigation and trade. "))
            + "Mitigation costs the region that pays for it, while every region's emissions warm the shared climate and damage "
              "everyone's output later. A region's score is its discounted utility from consumption divided by the largest possible "
              "value (0 to 1).")
    model = (run.get("model") or (b.get("meta") or {}).get("model") or (calls[0].get("model") if calls and calls[0].get("model") else None)
             or "the model").split("/")[-1]
    date = recorded_date(b, calls)
    seed = run.get("seed", (b.get("config") or {}).get("seed"))
    variant = os.path.splitext(os.path.basename((b.get("config") or {}).get("variant") or ""))[0]
    cfg = ", ".join(x for x in [f"config {variant}" if variant else "", f"seed {seed}" if seed is not None else "",
                                f"{N_s} of {n_regions} regions played by the model",
                                {"bilateral": "bilateral negotiation", "basic_club": "Basic Club", "none": "no negotiation"}.get(protocol, protocol),
                                f"{T_total} steps of 5 years", f"rates in steps of {rules['levels']:g}"] if x)
    def who_when(exs):
        """'R1 from 2025 to 2045 and R3 in 2030': the regions and step years of a list of disagreements."""
        by = {}
        for e in exs:
            if e.get("s") is not None and e.get("t") is not None:
                by.setdefault(e["s"], set()).add(year(e["t"]))
        same = {}                                       # regions with the same years together: 'R1 to R3 in 2020 and 2030'
        for k, ys in by.items():
            same.setdefault(tuple(sorted(ys)), []).append(lab[k])
        return list_txt(f"{regions_text(ls)} {years_span(ys)}" for ys, ls in sorted(same.items(), key=lambda kv: min(rnum(x) for x in kv[1])))

    def fail_sentences():
        """Every check that found a disagreement, in plain words (what the prompts said, against what the record says)."""
        what = {"own": "the own proposals listed in {n} of {N} evaluation prompts differ from the region's logged proposals",
                "recv": "the received proposals listed in {n} of {N} evaluation prompts differ from the logged proposals",
                "min": "the minimum mitigation rate stated in {n} of {N} activity prompts differs from the one rebuilt from the "
                       "proposals and acceptances",
                "lastmin": "the last step's minimum mitigation rate recapped in {n} of {N} proposal-stage prompts differs from the record",
                "acc_total": "the number of proposals accepted in the last step, as {n} of {N} proposal-stage prompts state it, "
                             "differs from the record",
                "table": "{n} of {N} rows of the public table differ from the logged plans",
                "fixed": "{n} of {N} table rows of fixed-plan regions differ from the fixed plan",
                "self": "the region's own last step in {n} of {N} prompts differs from its logged plan",
                "global": "the prompts of one step disagree on a world number (temperature, carbon, emissions or output) in {n} cases",
                "series": "{n} of {N} values in the summary's series differ from the prompts or the plans",
                "rw": "{n} of {N} region scores in the summary differ from the rewards"}
        out, plain = [], []
        for k, (n_all, n_good) in CK.items():
            ex = BAD[k]
            if not n_all or n_good == n_all:
                continue
            if k == "lastmin" and all((e.get("shown") or 0) <= 1e-9 < (e.get("want") or 0) for e in ex):
                rates = sorted({round(e["want"], 6) for e in ex})
                ww = who_when(ex)
                out.append(f"In {len(ex)} proposal-stage prompt{'s' if len(ex) != 1 else ''}" + (f" ({ww})" if ww else "")
                           + " the game told the region that its minimum mitigation rate from agreements in the last step was 0, "
                           f"although an agreement had bound it at {pc(rates[0]) if len(rates) == 1 else pc(rates[0]) + ' to ' + pc(rates[-1])} "
                           "(the game clears the minimum before it writes that line); the model saw that wrong value.")
                continue
            ww = who_when(ex)
            plain.append(what[k].format(n=f"{n_all - n_good:,}", N=f"{n_all:,}") + (f" ({ww})" if ww and len(ww) < 120 else ""))
        if plain:
            out.append("The checks found these disagreements: " + "; ".join(plain) + ".")
        return out

    chk = []
    if CK["own"][0] and CK["own"][1] == CK["own"][0] and CK["recv"][1] == CK["recv"][0]:
        chk.append(f"all {CK['own'][0] + CK['recv'][0]} lists of proposals sent and received in the evaluation prompts match the logged proposals")
    if CK["min"][0] and CK["min"][1] == CK["min"][0]:
        chk.append(f"all {CK['min'][0]} minimum mitigation rates the activity prompts state match the rates rebuilt from the proposals and acceptances")
    if CK["lastmin"][0] and CK["lastmin"][1] == CK["lastmin"][0] and CK["acc_total"][1] == CK["acc_total"][0]:
        chk.append(f"so do the {CK['lastmin'][0]:,} recaps of the last step's minimum and the number of accepted proposals")
    elif CK["acc_total"][0] and CK["acc_total"][1] == CK["acc_total"][0]:
        chk.append(f"all {CK['acc_total'][0]:,} recaps of the number of proposals accepted in the last step match the record")
    if CK["table"][0] and CK["table"][1] == CK["table"][0] and CK["self"][1] == CK["self"][0]:
        chk.append(f"the public table's savings, mitigation and export limit of the model-played regions ({CK['table'][0]:,} rows) and the "
                   f"regions' own last steps ({CK['self'][0]:,}) match the logged plans")
    if CK["fixed"][0] and CK["fixed"][1] == CK["fixed"][0]:
        chk.append(f"all {CK['fixed'][0]:,} table rows of fixed-plan regions show their fixed plan")
    if CK["global"][0] and CK["global"][1] == CK["global"][0]:
        chk.append("every prompt of a step reports the same temperature, carbon, world emissions and output")
    if CK["series"][0] and CK["series"][1] == CK["series"][0]:
        chk.append(f"the summary's series agree with the prompts and the plans ({CK['series'][0]:,} values)")
    if CK["rw"][0] and CK["rw"][1] == CK["rw"][0]:
        chk.append("the summary's region scores equal the rewards")
    chk_txt = ("Checks: " + "; ".join(chk) + ". ") if chk else ""
    chk_txt += "".join(x + " " for x in fail_sentences())
    if has_reason:
        rs_txt = ((f"Reasoning was saved for all {tied} decisions" if tied == len(steps) else f"Reasoning was saved for {tied} of {len(steps)} decisions")
                  + f" ({reason_chars:,} characters) and is shown with each step"
                  + (f"; {len(rescued)} answer{'s' if len(rescued) != 1 else ''} came from a follow-up question after the reasoning ran out of tokens" if rescued else "")
                  + (f"; {len(retried)} decision{'s' if len(retried) != 1 else ''} needed another call after an illegal answer (all calls are shown)" if retried else "")
                  + ". ")
    else:
        rs_txt = "No reasoning was saved for this recording. "
    partial = b.get("partial") or {}
    if partial:
        src_txt = (f"Data: the {date} recording of {GID} ({cfg}), played by {model} with reasoning on. "
                   f"{cap(partial.get('note', 'the recording stopped before the end').rstrip('.'))}: it holds {len(entries)} of the "
                   f"{T_total * N_s * (3 if negot else 1)} decisions of a full game and stops in the {year(entries[-1]['t'])} "
                   f"{entries[-1]['stage']} stage, so there are no final scores and no {year(T_total + 1)} values. This page is assembled from "
                   f"that run's log of moves (exactly as the game logged them) and its call log; each decision's observation is the "
                   f"prompt of its call. ")
    else:
        src_txt = (f"Data: the {date} recording of {GID} ({cfg}), played by {model} with reasoning on"
                   + ("" if complete else f"; the recording ends after {len(entries)} decisions, in the {year(entries[-1]['t'])} "
                                          f"{entries[-1]['stage']} stage") + ". ")
    fb = summary.get("fallbacks", (run.get("extras") or {}).get("fallbacks"))
    fb_txt = f"The game applied its fallback action {fb} times. " if isinstance(fb, int) and fb > 0 else ""
    idx_txt = ""
    if complete and isnum(info.get("climate_index")) and isnum(info.get("economic_index")):
        idx_txt = (f"Evaluation indices (0 at the worse of the two extreme policies, 1 at the better): climate {info['climate_index']:.3f}, "
                   f"economy {info['economic_index']:.3f}. ")
    foot = (src_txt + rs_txt + fb_txt + chk_txt
            + "Rates are shown in percent (the game's actions use 0 to 1). Each 5-year step is labeled by the year it starts; once the "
              "receiver accepts a proposal, its sender must mitigate at least what it promised and the receiver at least what was "
              "requested of it (so a request that promises 0% binds only the receiver), and a region's minimum mitigation rate for the "
              "step is the largest promise of its own that was accepted and the largest request to it that it accepted. Temperature: the atmosphere at the "
              "start of each step"
            + ({"paper": f"; the share of production a region keeps after climate damage is 1 {MINUS} 0.00236 T², T that temperature "
                         f"(the paper's form; all {dmg['paper'][0]} damage factors in the prompts match it)",
                "code": f"; the share of production a region keeps after climate damage is 1 / (1 + 0.00236 T²), T that temperature "
                        f"(the official code's form; all {dmg['code'][0]} damage factors in the prompts match it)"}.get(damage_form, ""))
            + ". Score so far: the discounted (1.5% a year) sum of the region's utilities divided by the largest possible sum so far; "
              "the final score is the same ratio over the whole game. " + idx_txt
            + f"Paper (27 regions trained with A2C, 2115): temperature anomaly {PAPER_T2115['none']:.2f} °C without negotiation, "
              f"{PAPER_T2115['bilateral']:.2f} °C with bilateral negotiation, {PAPER_T2115['basic_club']:.2f} °C with Basic Club"
            + (f" (all {REF_REGIONS} regions trained there, so these values are not reachable by {nw} model regions among {n_fixed} fixed-plan ones). "
               if scripted else ". ")
            + "Source: Zhang et al., AI for Global Climate Cooperation: Modeling Global Climate Negotiations, Agreements, and "
              "Long-Term Cooperation in RICE-N (ICML 2025), and the official climate-cooperation-competition code.")

    for st in steps:
        st.setdefault("thinking", "")
        for k in ("t", "stage", "a", "s", "rescued", "retries", "chars"):
            st.pop(k, None)
    d = {"id": GID, "name": "RICE-N climate negotiations", "title": title, "lede": lede, "seatNames": seat_names, "tiles": tiles,
         "foot": foot, "x": {"n": n_x, "labels": labels}, "charts": charts,
         "kinds": {k: v for k, v in KINDS.items() if any(st["kind"] == k for st in steps)}, "steps": steps, "feed": feed,
         "feedTitle": "Step by step",
         "feedSub": ("Each region's proposals, its answers to the proposals it received and its plan, in the order they were made, "
                     "with each step's agreements and results. Regions decide at the same time within a stage and see the others' "
                     "choices only afterwards. Rates in %; promise/request pairs are the rate a region promised for itself and the "
                     "rate it asked of the receiver.")}
    d["eyebrow"] = (f"social_game_env {DOT} {GID} {DOT} {model}" + (", reasoning on" if has_reason else "") + f" {DOT} recorded {date}"
                    + ("" if has_reason else f" {DOT} reasoning not saved")
                    + ("" if complete else f" {DOT} stopped in {year(entries[-1]['t'])}"))
    print("checks: " + ", ".join(f"{k} {g}/{n}" for k, (n, g) in CK.items() if n)
          + f"; calls tied {tied}/{len(steps)}, untied {untied}; complete {complete}")
    return d, date


# ================================================================================ test helper: a bundle from a partial recording
# Used only with --partial, to run the full pipeline on the call log of a recording that stopped before the end (the 2026-11-11
# run of this game stopped at the spending cap). The moves are the run log's lines, exactly as the game logged them; the view
# of each move is the prompt of its call. There is no summary, no game_over event and no reward: nothing is made up.
_RUNLOG = re.compile(r"^(\d+)\. P(\d+): (\{.*\})\s*$")


def bundle_from_partial(calls_path, runlog_path, note="the run's spending cap stopped the recording"):
    raw = [json.loads(l) for l in open(calls_path, encoding="utf-8") if l.strip()]
    calls = common.load_calls(calls_path)
    lines = open(runlog_path, encoding="utf-8").read().splitlines()
    moves = []
    for l in lines:
        m = _RUNLOG.match(l.strip())
        if m:
            moves.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    head = next((l for l in lines if l.startswith("Game start")), "")
    seed = re.search(r"seed (\d+)", head)
    keys = [call_key(c) for c in calls]
    events, ptr = [{"_visibility": "hidden", "event": "setup", "roles": {str(s): "region" for s in sorted({m[1] for m in moves})}}], 0
    for n, s, mv in moves:
        while ptr < len(calls) and calls[ptr]["seat"] != s:
            warn(f"call {ptr} (P{calls[ptr]['seat']}) before move {n} (P{s}) is skipped")
            ptr += 1
        if ptr >= len(calls):
            raise SystemExit(f"no call left for move {n}")
        j = ptr
        while j < len(calls) and calls[j]["seat"] == s and keys[j] == keys[ptr]:
            j += 1
        tol = 0.05
        if not any(same_action(read_action(calls[k]["answer"]), read_action(mv), tol) for k in range(ptr, j)):
            warn(f"move {n} (P{s}) matches no answer of calls {ptr}..{j - 1}")
        events.append({"_audience": [s], "_visibility": "private", "actor": s, "event": "move", "move": mv, "turn": n,
                       "view": calls[j - 1]["prompt"]})
        ptr = j
    if ptr < len(calls):
        print(f"partial: {len(calls) - ptr} calls after the last logged move (made before the stop, never applied)")
    t0 = next((c.get("t_start") for c in raw if c.get("t_start")), None)
    when = datetime.datetime.fromtimestamp(float(t0), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t0 else None
    run = {"game": GID, "model": next((c.get("model") for c in raw if c.get("model")), None), "recorded_at": when,
           "seed": int(seed.group(1)) if seed else None, "n_players": len({m[1] for m in moves})}
    return {"events": events, "summary": {}, "run": run, "config": {}, "meta": {}, "transcript": "",
            "partial": {"note": note, "calls": len(calls), "moves": len(moves)}}, calls


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--partial":
        bundle, calls = bundle_from_partial(sys.argv[2], sys.argv[3])
        data, date = build(bundle, calls)
        common.write(data, sys.argv[4], date=date)
    elif len(sys.argv) >= 3 and not sys.argv[1].startswith("--"):
        src, out = sys.argv[1], sys.argv[2]
        calls = common.load_calls(sys.argv[3]) if len(sys.argv) > 3 else []
        data, date = build(src, calls)
        common.write(data, out, date=date)
    else:
        raise SystemExit(__doc__)
