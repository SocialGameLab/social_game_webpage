/* Kind "table": card and dice games around a table, with bids, claims and plays.
 * The board shows the latest move on the table, then each seat: whether it is
 * still in, its last public move, and what this lens privately knows about it.
 * Options (view_options):
 *   move_events  event types that put something on the table (default below)
 *   out_events   event types that knock a seat out (default ["out", "eliminated"])
 *   round_event  event type that starts a round (default "round_start")
 *   round_resets true when every seat is back in at each new round (Love Letter) */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;

  const MOVE_EVENTS = ["bid", "offer", "pass_claim", "claim", "play", "cut", "swap_attempt", "call", "dudo", "calza"];
  const OUT_EVENTS = ["out", "eliminated"];
  const isPrivate = (e) => e._visibility && e._visibility !== "public";
  // The seat an event is about: who acted, or whose cards it concerns.
  const seatOf = (r, e) => [e.actor, e.seat, e.target, e.cutter].find((s) => ui.isSeat(r, s));

  MatchRoom.defineKind("table", (opt) => {
    const moves = opt.move_events || MOVE_EVENTS;
    const outs = opt.out_events || OUT_EVENTS;
    const roundEvent = opt.round_event || "round_start";

    return {
      chapter: (e) => (e.event === roundEvent || e.event === "cutter_start" ? `Round ${e.round ?? ""}`.trim() : null),

      renderEvent(r, e) {
        if (e.event === roundEvent || e.event === "cutter_start") return ui.chapterHead(`Round ${esc(e.round ?? "")}`, ui.fields(r, e, ["round"]));
        if (outs.includes(e.event)) return ui.card(`<strong>${esc(ui.shortName(r, seatOf(r, e) ?? e.target))} is out.</strong> ${ui.fields(r, e, ["target", "seat"])}`, r, e, true);
        if (moves.includes(e.event) && !isPrivate(e)) return ui.sys(e.event.replace(/_/g, " "), ui.fields(r, e), r, e);
        return ui.autoEvent(r, e);
      },

      knowledge: (r, p) => (Object.keys(p.knowledge || {}).length ? `<div class="brief">${ui.fields(r, p.knowledge)}</div>` : ""),

      board(r, evs) {
        const out = new Set(), last = {}, secret = {};
        let move = null;
        evs.forEach((e) => {
          if (opt.round_resets && e.event === roundEvent) out.clear();
          const s = seatOf(r, e);
          if (outs.includes(e.event) && s !== undefined) out.add(s);
          if (moves.includes(e.event) && !isPrivate(e)) move = e;
          if (s === undefined) return;
          if (isPrivate(e)) secret[s] = e; else if (!outs.includes(e.event)) last[s] = e;
        });
        const rows = r.players.map((p) => `
          <li class="tb-seat${out.has(p.seat) ? " tb-seat--out" : ""}">
            <div class="tb-who">${ui.dot(p.seat)}<strong>${esc(p.name)}</strong>${out.has(p.seat) ? ` <span class="small muted">out</span>` : ""}</div>
            ${last[p.seat] ? `<div class="small"><span class="kv__k">${esc(title(last[p.seat].event))}</span> ${ui.fields(r, last[p.seat], ["actor", "seat"])}</div>` : `<div class="small muted">No move yet</div>`}
            ${secret[p.seat] ? `<div class="small tb-secret"><span class="kv__k">${esc(title(secret[p.seat].event))}</span> ${ui.fields(r, secret[p.seat], ["actor", "seat"])}</div>` : ""}
          </li>`).join("");
        return [
          ui.endBanner(r),
          ui.block("On the table", move ? `<div class="small"><strong>${esc(title(move.event))}</strong> ${ui.fields(r, move)}</div>` : `<p class="muted small">Nothing played yet.</p>`),
          ui.block("Seats", `<ul class="tb-list">${rows}</ul>`),
          ui.rulesBlock(r, evs),
        ].join("");
      },
    };
  });
})();
