/* Werewolf: night kills, the Seer's checks, day speeches and lynches. */
(function () {
  "use strict";

  const { esc } = MatchRoom.util;
  const ui = MatchRoom.ui;
  const state = MatchRoom.state;

  // Night events carry the day that just ended, so night 1 is day 0.
  const nightHead = (e) => `Night ${e.day + 1}`;

  MatchRoom.defineGame("werewolf", {
    rolesArePublic: false,

    chapter: (e) => (e.event === "night_kill" || e.event === "no_night_kill" ? nightHead(e) : e.event === "lynch" || e.event === "no_lynch" ? `Day ${e.day} vote` : null),

    renderEvent(r, e) {
      switch (e.event) {
        case "seer_result": return ui.sys("seer", `The Seer checks <strong>${esc(ui.shortName(r, e.target))}</strong>: ${e.is_wolf ? "<strong>a wolf</strong>" : "not a wolf"}.`, r, e);
        case "night_kill": return ui.chapterHead(nightHead(e), `<strong>${esc(ui.shortName(r, e.target))}</strong> is killed in the night.`);
        case "no_night_kill": return ui.chapterHead(nightHead(e), "The wolves could not agree; nobody dies.");
        case "lynch": return ui.card(`<strong>Day ${e.day}: the village lynches ${esc(ui.shortName(r, e.target))}.</strong>`, r, e, true);
        case "no_lynch": return ui.card(`<strong>Day ${e.day}: the vote is tied.</strong> Nobody is lynched.`, r, e);
        default: return undefined;
      }
    },

    seatRole(r, p) {
      if (ui.roleVisibleTo(p.seat)) return undefined;
      // A wolf knows the other wolves even when their roles are hidden.
      const k = typeof state.lens === "number" ? ui.player(r, state.lens).knowledge || {} : {};
      if ((k.known_wolves || []).includes(p.seat)) return `<span class="team team--WOLVES">Wolf (you know this)</span>`;
      return undefined;
    },

    knowledge(r, p) {
      const k = p.knowledge || {};
      return (k.known_wolves || []).length ? `<div class="brief">Fellow wolf: ${ui.seatList(r, k.known_wolves)}</div>` : "";
    },

    board(r, evs) {
      const fate = {};
      const checks = [];
      let day = 0;
      evs.forEach((e) => {
        if (e.event === "night_kill") fate[e.target] = `killed, night ${e.day + 1}`;
        if (e.event === "lynch") fate[e.target] = `lynched, day ${e.day}`;
        if (e.event === "discussion" || e.event === "lynch" || e.event === "no_lynch") day = e.day;
        if (e.event === "seer_result") checks.push(e);
      });
      const alive = r.players.filter((p) => !(p.seat in fate)).length;
      const rows = r.players.map((p) => `
        <li class="ww-row${p.seat in fate ? " ww-row--dead" : ""}">${ui.dot(p.seat)}<span>${esc(p.name)}</span><span class="small muted">${esc(fate[p.seat] || "alive")}</span></li>`).join("");
      return [
        ui.endBanner(r),
        ui.block(day ? `Day ${day} · ${alive} alive` : `${alive} alive`, `<ul class="ww-list">${rows}</ul><p class="small muted">Wolves win once they match the villagers in number.</p>`),
        checks.length ? ui.block("Seer's checks", `<ul class="ww-list">${checks.map((c) => `<li class="ww-row">${ui.dot(c.target)}<span>${esc(ui.shortName(r, c.target))}</span><span class="small team team--${c.is_wolf ? "WOLVES" : "VILLAGE"}">${c.is_wolf ? "wolf" : "not a wolf"}</span></li>`).join("")}</ul>`) : "",
      ].join("");
    },
  });
})();
