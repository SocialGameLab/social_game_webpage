/* Kind "rounds": repeated matrix and economics games. Each round's result becomes
 * a row of a round-by-seat table; talk shows as bubbles. Options (view_options):
 *   round_events  event types that close a round (default below)
 *   round_label   word for a round in the feed, e.g. "Month" (default "Round") */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;

  const ROUND_EVENTS = ["round_result", "month_result", "push"];

  MatchRoom.defineKind("rounds", (opt) => {
    const types = opt.round_events || ROUND_EVENTS;
    const word = opt.round_label || "Round";
    const num = (e) => e.round ?? e.month;

    return {
      chapter: (e) => (types.includes(e.event) && num(e) != null ? `${word} ${num(e)}` : null),

      renderEvent(r, e) {
        if (e.event === "rules") return ui.sys("rules", ui.fields(r, e), r, e);
        if (e.event === "talk") return ui.bubble(r, e.actor, e.text, { event: e });
        if (types.includes(e.event)) return ui.card(`<strong>${num(e) != null ? `${word} ${esc(num(e))}` : esc(title(e.event))}.</strong> ${ui.fields(r, e, ["round", "month"])}`, r, e);
        if (e.event === "game_end") return ui.card(`<strong>Game ends.</strong> ${ui.fields(r, e)}`, r, e, true);
        return ui.autoEvent(r, e);
      },

      knowledge: (r, p) => (Object.keys(p.knowledge || {}).length ? `<div class="brief">${ui.fields(r, p.knowledge)}</div>` : ""),

      board(r, evs) {
        const scores = ui.latestScores(r, evs);
        return [
          ui.endBanner(r),
          scores ? ui.block("Running score", `<div class="kv-list small">${r.players.map((p) => `<span class="kv">${ui.dot(p.seat)}${esc(p.name)} <strong>${esc(scores[p.seat] ?? "—")}</strong></span>`).join("")}</div>`) : "",
          ui.block(`${word}s`, ui.roundTable(r, evs, types)),
          ui.rulesBlock(r, evs),
        ].join("");
      },
    };
  });
})();
