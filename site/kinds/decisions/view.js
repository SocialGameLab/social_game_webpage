/* Kind "decisions": games recorded through social-game-env's `training` engine whose
 * moves are JSON actions and whose prompts are long text, such as the eco market and
 * trading games. Each move reads as one decision: who, what it did in words, its
 * parameters and anything it wrote. A run with an eco chart replay also carries the
 * replay's one-line summary of each decision (`head`, `label`; tools/eco_charts/labels.py).
 * The board pairs the current decision with what that seat saw before it, cut to the
 * game state: the rules and the answer format are on the "How to play" page.
 * Options (view_options):
 *   chapter_every  decisions per "Jump to" entry (default: about 20 entries per run) */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;
  const state = MatchRoom.state;

  // Where the game state starts in a prompt, and where the instructions for answering begin.
  const STATE_START = "You are seat";
  const STATE_END = /(Legal actions:|sample actions:|Action schema|Respond with exactly one JSON)/;
  // Payload fields that hold text the seat wrote, shown as its words rather than as a value.
  const TEXT_KEYS = new Set(["message", "talk", "text", "post", "description", "note", "say", "comment"]);

  function parseMove(e) {
    try {
      const m = JSON.parse(e.move);
      return { id: m.action_id, payload: m.payload || {} };
    } catch (_) {
      return { id: String(e.move ?? ""), payload: {} };
    }
  }

  function stateOf(view) {
    const s = String(view ?? "");
    const at = s.indexOf(STATE_START);
    const rest = at >= 0 ? s.slice(at) : s;
    const end = STATE_END.exec(rest);
    return (end ? rest.slice(0, end.index) : rest).trim();
  }

  const short = (v) => (typeof v === "number" ? String(+v.toFixed(4)) : typeof v === "object" ? JSON.stringify(v) : String(v));
  const chip = (k, v) => `<span class="dc-chip">${k ? `<span class="kv__k">${esc(title(k))}</span> ` : ""}${esc(v)}</span>`;

  /** A payload as labelled values, plus any text the seat wrote. */
  function params(payload, limit) {
    const chips = [];
    const words = [];
    Object.entries(payload).forEach(([k, v]) => {
      if (typeof v === "string" && (TEXT_KEYS.has(k) || v.length > 60)) {
        if (v.trim()) words.push(v);
      } else if (Array.isArray(v) && v.length && v.every((x) => x && typeof x === "object" && !Array.isArray(x))) {
        v.slice(0, limit).forEach((x) => chips.push(chip("", Object.entries(x).map(([a, b]) => `${a} ${short(b)}`).join(" · "))));
        if (v.length > limit) chips.push(chip("", `+${v.length - limit} more`));
      } else if (Array.isArray(v)) {
        chips.push(chip(k, v.length ? v.map(short).join(", ") : "none"));
      } else if (v && typeof v === "object") {
        const kv = Object.entries(v);
        chips.push(chip(k, kv.length ? kv.slice(0, limit).map(([a, b]) => `${a} ${short(b)}`).join(" · ") + (kv.length > limit ? ` · +${kv.length - limit} more` : "") : "none"));
      } else {
        chips.push(chip(k, short(v)));
      }
    });
    return { chips, words };
  }

  // The log keeps the last reply, so an empty one means no reply ever arrived.
  const fallbackTag = (e) => (e.fallback ? ` <span class="dc-fb" title="${e.reply ? "The reply was rejected" : "No reply arrived"}, so the env's fallback move was played">fallback</span>` : "");

  /** One decision: who, what it did, the replay's summary, its parameters and words. `full` keeps long text whole. */
  function decision(r, e, full) {
    const mv = parseMove(e);
    const { chips, words } = params(mv.payload, full ? 12 : 6);
    const said = words.map((w) => `<div class="dc-words">“${esc(full || w.length <= 280 ? w : `${w.slice(0, 277)}…`)}”</div>`).join("");
    return `
      <div class="dc">
        <div class="dc-top">${ui.dot(e.actor)}<strong>${esc(ui.shortName(r, e.actor))}</strong>
          <span class="dc-label">${esc(e.label || title(mv.id))}</span>${fallbackTag(e)} ${ui.visBadge(r, e)}</div>
        ${e.head ? `<div class="dc-head">${esc(e.head)}</div>` : ""}
        ${chips.length ? `<div class="dc-chips">${chips.join("")}</div>` : ""}
        ${said}
      </div>`;
  }

  // A public move can still carry a private view: the table saw the order, not the account.
  const canReadView = (e) => typeof e.view === "string" && (!e.view_audience || ui.canSee({ _visibility: "private", _audience: e.view_audience }, state.lens));
  const signed = (v) => `${v > 0 ? "+" : ""}${v}`;

  MatchRoom.defineKind("decisions", (opt) => ({
    chapter(e, r) {
      if (e.event === "final") return "End";
      if (e.event !== "move") return null;
      const every = opt.chapter_every || Math.max(10, Math.ceil(((r && r.turns) || 0) / 20));
      if (e.turn !== 1 && (e.turn - 1) % every) return null;
      return e.head ? e.head.split(" · ")[0] : `Decision ${e.turn}`;
    },

    renderEvent(r, e) {
      if (e.event === "setup") return ui.sys("setup", `${r.players.length} player${r.players.length > 1 ? "s" : ""} · ${esc(r.env_id)}`, r, e);
      if (e.event === "move") return `<div class="dc-row"><span class="ev-tag">#${esc(e.turn)}</span>${decision(r, e, false)}</div>`;
      if (e.event === "final") return ui.card(`<strong>${r.truncated ? `Stopped after ${esc(r.turns)} decisions.` : `Game over after ${esc(r.turns)} decision${r.turns === 1 ? "" : "s"}.`}</strong>`, r, e, true);
      if (e.event === "game_over") return ui.sys("game over", ui.fields(r, e), r, e);
      return ui.autoEvent(r, e);
    },

    result(r) {
      if (r.truncated) return { head: `Stopped after ${r.turns} decisions`, sub: "The safety limit ended the game early." };
      // A top score is no win when players don't compete for a fixed total.
      const utility = (r.about || {}).utility;
      if (utility && !["zero_sum", "constant_sum"].includes(utility) && r.result && r.result.rewards) {
        return { head: "Final returns", sub: Object.entries(r.result.rewards).map(([s, v]) => `${ui.shortName(r, Number(s))}: ${signed(v)}`).join(" · ") };
      }
      return null;
    },

    board(r, evs) {
      const moves = evs.filter((e) => e.event === "move");
      const last = moves[moves.length - 1];
      const facts = [["Env", r.env_id], ["Decisions", r.turns], r.fallbacks ? ["Fallback moves", r.fallbacks] : null].filter(Boolean);
      // The game's chart replay (content `page_link`), opened at this decision: #<game>[.nohist].s<index of the move>.
      const link = (ui.game() || {}).page_link;
      const all = r._dcMoves || (r._dcMoves = r.events.filter((e) => e.event === "move"));
      const cta = link && last ? `<a class="dc-cta" href="${esc(`${link.href}${r.id.endsWith("_no_history") ? ".nohist" : ""}.s${all.indexOf(last)}`)}">See decision ${esc(last.turn)} in the ${esc(link.label)} →</a>` : "";
      const seen = !last ? "" : canReadView(last)
        ? `<pre class="dc-view">${esc(stateOf(last.view)) || "(empty)"}</pre>`
        : `<p class="muted small">This lens cannot see what ${esc(ui.shortName(r, last.actor))} saw.</p>`;
      return [
        ui.endBanner(r),
        cta,
        last ? ui.block(`Decision ${esc(last.turn)} of ${esc(r.turns)}`, decision(r, last, true)) : ui.block("Decisions", `<p class="muted small">No decision yet.</p>`),
        last ? ui.block(`What ${esc(ui.shortName(r, last.actor))} saw before it`, seen) : "",
        ui.block("Match", `<div class="kv-list small">${facts.map(([k, v]) => `<span class="kv"><span class="kv__k">${esc(k)}</span> ${esc(v)}</span>`).join("")}</div>`),
      ].join("");
    },
  }));
})();
