/* Kind "deals": negotiation and contract games. Offers, counters and replies show
 * as bubbles with their terms; the board holds the latest offer, what this lens
 * knows privately, and a table of settled rounds. Options (view_options):
 *   offer_events  event types that put terms on the table (default below)
 *   round_events  event types that settle a round (default ["round_result", "round_end", "decide_result"]) */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;

  const OFFER_EVENTS = ["offer", "wage_offer", "loan_offer", "contract_proposed", "quote", "buyer_counter", "bribe", "trade_proposal"];
  const ROUND_EVENTS = ["round_result", "round_end", "decide_result"];
  const isPrivate = (e) => e._visibility && e._visibility !== "public";

  MatchRoom.defineKind("deals", (opt) => {
    const offers = opt.offer_events || OFFER_EVENTS;
    const rounds = opt.round_events || ROUND_EVENTS;

    return {
      chapter: (e) => (e.event === "leg_start" ? `Leg ${e.leg}` : e.event === "round_announce" ? `Round ${e.round}` : null),

      renderEvent(r, e) {
        if (e.event === "rules") return ui.sys("rules", ui.fields(r, e), r, e);
        if (e.event === "leg_start") return ui.chapterHead(`Leg ${esc(e.leg)}`, ui.fields(r, e, ["leg"]));
        if (e.event === "round_announce") return ui.chapterHead(`Round ${esc(e.round)}`, ui.fields(r, e, ["round"]));
        if (offers.includes(e.event) && ui.speakerOf(r, e) === undefined && typeof e.say !== "string") {
          return ui.card(`<strong>${esc(title(e.event))}.</strong> ${ui.fields(r, e)}`, r, e);
        }
        if (rounds.includes(e.event)) return ui.card(`<strong>${esc(title(e.event))}.</strong> ${ui.fields(r, e)}`, r, e);
        if (e.event === "game_end") return ui.card(`<strong>Game ends.</strong> ${ui.fields(r, e)}`, r, e, true);
        return ui.autoEvent(r, e);
      },

      knowledge: (r, p) => (Object.keys(p.knowledge || {}).length ? `<div class="brief">${ui.fields(r, p.knowledge)}</div>` : ""),

      board(r, evs) {
        const last = [...evs].reverse().find((e) => offers.includes(e.event));
        const secrets = evs.filter((e) => isPrivate(e) && !offers.includes(e.event)).slice(-3);
        const settled = evs.some((e) => rounds.includes(e.event));
        const scores = ui.latestScores(r, evs);
        return [
          ui.endBanner(r),
          ui.block("On the table", last ? `<div class="small"><strong>${esc(title(last.event))}</strong><div class="kv-list">${ui.fields(r, last, ["say"])}</div></div>` : `<p class="muted small">No offer yet.</p>`),
          secrets.length ? ui.block("Known only to this lens", secrets.map((e) => `<div class="small"><span class="kv__k">${esc(title(e.event))}</span> ${ui.fields(r, e)}</div>`).join("")) : "",
          scores ? ui.block("Running score", `<div class="kv-list small">${r.players.map((p) => `<span class="kv">${ui.dot(p.seat)}${esc(p.name)} <strong>${esc(scores[p.seat] ?? "—")}</strong></span>`).join("")}</div>`) : "",
          settled ? ui.block("Settled rounds", ui.roundTable(r, evs, rounds)) : "",
          ui.rulesBlock(r, evs),
        ].join("");
      },
    };
  });
})();
