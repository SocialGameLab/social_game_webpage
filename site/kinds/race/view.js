/* Kind "race": dice, push-your-luck and small cooperative games scored per seat.
 * The board shows each seat's score so far, the latest roll or hand, and a round
 * table when the game logs rounds. Options (view_options):
 *   round_events  event types to tabulate per round (default ["push"])
 *   target_key    rules field holding the winning score (default "winscore") */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;

  const ROLL_KEYS = ["dice", "roll", "player_hand", "card", "action", "message"];

  MatchRoom.defineKind("race", (opt) => {
    const rounds = opt.round_events || ["push"];
    const targetKey = opt.target_key || "winscore";

    return {
      chapter: (e) => (rounds.includes(e.event) && e.round != null ? `Round ${e.round}` : null),

      renderEvent(r, e) {
        if (e.event === "rules") return ui.sys("rules", ui.fields(r, e) || "Standard rules.", r, e);
        if (e.event === "game_end") return ui.card(`<strong>Game ends.</strong> ${ui.fields(r, e)}`, r, e, true);
        if (/bust/.test(e.event)) return ui.card(`<strong>${esc(ui.shortName(r, e.actor))} busts.</strong> ${ui.fields(r, e, ["actor"])}`, r, e);
        const who = ui.speakerOf(r, e) ?? (ui.isSeat(r, e.seat) ? e.seat : undefined);
        if (who !== undefined) return ui.sys(e.event.replace(/_/g, " "), `<strong>${esc(ui.shortName(r, who))}</strong> ${ui.fields(r, e, ["actor", "seat"])}`, r, e);
        return ui.autoEvent(r, e);
      },

      knowledge: (r, p) => (Object.keys(p.knowledge || {}).length ? `<div class="brief">${ui.fields(r, p.knowledge)}</div>` : ""),

      board(r, evs) {
        const score = {};
        let roll = null;
        evs.forEach((e) => {
          const s = ui.isSeat(r, e.actor) ? e.actor : ui.isSeat(r, e.seat) ? e.seat : undefined;
          if (s !== undefined && typeof e.score === "number") score[s] = e.score;
          if (ROLL_KEYS.some((k) => k in e)) roll = e;
        });
        const rules = evs.find((e) => e.event === "rules") || {};
        const target = rules[targetKey];
        const max = Math.max(target || 0, ...Object.values(score), 1);
        const bars = r.players.map((p) => `
          <li class="rc-row">${ui.dot(p.seat)}<span>${esc(p.name)}</span>
            <span class="rc-bar"><span style="width:${Math.max(0, Math.min(100, ((score[p.seat] || 0) / max) * 100))}%;background:${ui.seatColor(p.seat)}"></span></span>
            <strong>${esc(score[p.seat] ?? "—")}</strong></li>`).join("");
        return [
          ui.endBanner(r),
          Object.keys(score).length ? ui.block(target ? `Score · first to ${esc(target)}` : "Score", `<ul class="rc-list">${bars}</ul>`) : "",
          roll ? ui.block("Latest", `<div class="small"><strong>${esc(title(roll.event))}</strong> ${ui.fields(r, roll)}</div>`) : "",
          evs.some((e) => rounds.includes(e.event)) ? ui.block("Rounds", ui.roundTable(r, evs, rounds)) : "",
          ui.rulesBlock(r, evs),
        ].join("");
      },
    };
  });
})();
