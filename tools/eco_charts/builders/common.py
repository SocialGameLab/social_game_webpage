"""Shared helpers for the per-game replay builders (one builder per game: games/<game_id>.py).
A builder turns one recorded game (a demo bundle) plus, when available, the model-call log with reasoning into the JSON the
replay page reads. See SCHEMA.md for the format. Every string must be English: check() rejects any CJK character."""
import json, math, os, re

EYEBROW = "social_game_env · {gid} · deepseek-flash, reasoning on · recorded {date}"
COLORS = {"c1", "c2", "c3", "c4", "ink", "hold", "muted"}
FMTS = {"price", "price2", "price4", "pct", "pct0", "pct3", "int", "money", "money0", "bps"}


def load_bundle(d):
    """A demo bundle directory: game_log.jsonl (events), summary.json, run.json, config.json, transcript.md."""
    rd = lambda n: open(os.path.join(d, n), encoding="utf-8").read() if os.path.exists(os.path.join(d, n)) else ""
    events = [json.loads(l) for l in rd("game_log.jsonl").splitlines() if l.strip()]
    js = lambda n: json.loads(rd(n)) if rd(n).strip() else {}
    return {"events": events, "summary": js("summary.json"), "run": js("run.json"), "config": js("config.json"), "transcript": rd("transcript.md")}


def seat_of(prompt):
    m = re.search(r"\bYou are (?:seat |player )?P(\d+)\b", prompt or "")
    return int(m.group(1)) if m else None


def load_calls(path):
    """The model-call log written while recording (one JSON object per call, in time order): prompt sent, reasoning, answer.
    Returns [] when the file is missing (recordings made before reasoning was saved)."""
    if not path or not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        c = json.loads(line)
        prompt = c.get("prompt")
        if prompt is None:
            msgs = c.get("messages") or []
            prompt = msgs[-1]["content"] if msgs else ""
        out.append({"seat": c.get("seat", seat_of(prompt)), "prompt": prompt, "answer": (c.get("followup_answer") or c.get("answer") or "").strip(),
                    "reasoning": c.get("reasoning") or "", "rescued": bool(c.get("followup_answer"))})
    return out


def state_part(view):
    """The per-turn part of a prompt: from 'You are seat' up to the action-format section."""
    i = view.find("You are seat"); s = view[i:] if i >= 0 else view
    m = re.search(r"(Legal actions:|sample actions:|Action schema|Respond with exactly one JSON)", s)
    return (s[:m.start()] if m else s).strip()


def _walk(o, path, problems):
    if isinstance(o, str):
        if re.search(r"[　-〿㐀-鿿＀-￯]", o): problems.append(f"non-English text at {path}: {o[:40]!r}")
    elif isinstance(o, float):
        if math.isnan(o) or math.isinf(o): problems.append(f"NaN/inf at {path}")
    elif isinstance(o, dict):
        for k, v in o.items(): _walk(v, f"{path}.{k}", problems)
    elif isinstance(o, list):
        for i, v in enumerate(o): _walk(v, f"{path}[{i}]", problems)


def check(d):
    p = []
    for k in ("id", "name", "title", "lede", "eyebrow", "tiles", "x", "charts", "kinds", "steps"):
        if k not in d: p.append(f"missing key {k}")
    if p: return p
    n = d["x"].get("n", 0)
    if n < 1 or len(d["x"].get("labels", [])) != n: p.append("x.labels must have x.n entries")
    if not 2 <= len(d["tiles"]) <= 4: p.append("2-4 tiles expected")
    if not d["steps"]: p.append("no steps")
    seats = d.get("seatNames", ["Model"])
    for i, s in enumerate(d["steps"]):
        for k in ("x", "head", "kind", "act"):
            if k not in s: p.append(f"steps[{i}] missing {k}")
        if not 0 <= s.get("x", -1) < n: p.append(f"steps[{i}].x out of range")
        if s.get("kind") not in d["kinds"]: p.append(f"steps[{i}].kind {s.get('kind')!r} not in kinds")
        if not 0 <= s.get("seat", 0) < len(seats): p.append(f"steps[{i}].seat out of range")
    for k, v in d["kinds"].items():
        if not (isinstance(v, list) and len(v) == 2 and v[1] in COLORS): p.append(f"kinds[{k}] must be [label, color]")
    for ci, c in enumerate(d["charts"]):
        if c.get("grid"):
            g = c["grid"]
            if len(g.get("cells", [])) != len(g.get("rows", [])): p.append(f"charts[{ci}].grid.cells must have one row per name")
            for r in g.get("cells", []):
                if len(r) != n: p.append(f"charts[{ci}].grid row length {len(r)} != x.n {n}")
                for cell in r:
                    if cell and cell.get("c", "muted") not in COLORS: p.append(f"charts[{ci}] grid color {cell.get('c')}")
            continue
        if c.get("fmt") not in FMTS: p.append(f"charts[{ci}].fmt {c.get('fmt')!r}")
        series = list(c.get("lines", [])) + list(c.get("bars", []))
        if not series and not c.get("candles"): p.append(f"charts[{ci}] has nothing to draw")
        for s in series:
            if len(s["v"]) != n: p.append(f"charts[{ci}] series {s.get('name')} length {len(s['v'])} != x.n {n}")
            if s.get("color") not in COLORS: p.append(f"charts[{ci}] series color {s.get('color')}")
            if not any(v is not None for v in s["v"]): p.append(f"charts[{ci}] series {s.get('name')} is empty")
        if c.get("candles") is not None and len(c["candles"]) != n: p.append(f"charts[{ci}] candles length")
        for mk in c.get("markers", []):
            if not 0 <= mk["x"] < n: p.append(f"charts[{ci}] marker x out of range")
    for e in d.get("feed", []):
        if not 0 <= e.get("x", -1) < n: p.append("feed entry x out of range")
    _walk(d, "", p)
    return p


def write(d, out_dir, date="2026-11-09"):
    d.setdefault("eyebrow", EYEBROW.format(gid=d["id"], date=date))
    d.setdefault("seatNames", ["Model"])
    probs = check(d)
    if probs:
        raise SystemExit("REJECTED " + d["id"] + ":\n  " + "\n  ".join(probs[:30]))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, d["id"] + ".json")
    json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print("wrote", path, "steps", len(d["steps"]), "x", d["x"]["n"], "charts", len(d["charts"]))
    return path
