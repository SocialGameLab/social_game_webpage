/* Wavelength: a psychic's clue on a spectrum, the team's dial, the rival's side guess. */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;

  const name = (r, s) => `<strong>${esc(ui.shortName(r, s))}</strong>`;
  const team = (t) => `<span class="team team--${esc(t)}">${esc(title(t))}</span>`;
  const STOPS = 11; // positions 0–10

  MatchRoom.defineGame("wavelength", {
    rolesArePublic: true,

    chapter: (e) => (e.event === "clue" ? `Round ${e.round}: ${title(e.team)}` : null),

    renderEvent(r, e) {
      switch (e.event) {
        case "target_reveal": return ui.sys("target", `The psychic sees the target at <strong>${esc(e.answer)}</strong> on ${esc(e.spectrum[0])} ↔ ${esc(e.spectrum[1])}.`, r, e);
        case "setup_target": return ui.sys("target", `Round ${esc(e.round)}: ${name(r, e.psychic)} is the ${team(e.team)} psychic; the target is ${esc(e.target)}.`, r, e);
        case "clue": return ui.chapterHead(`Round ${esc(e.round)} · ${team(e.team)}`, `${esc(ui.shortName(r, e.psychic))} gives “${esc(e.text)}” for ${esc(e.spectrum[0])} ↔ ${esc(e.spectrum[1])}.`);
        case "dial": return ui.sys("dial", `${name(r, e.actor)} sets the dial to <strong>${esc(e.position)}</strong>.`, r, e);
        case "side_guess": return ui.sys("guess", `${name(r, e.actor)} bets the target is to the <strong>${esc(title(e.side))}</strong>.`, r, e);
        case "reveal": return ui.card(`<strong>Target ${esc(e.target)}, dial ${esc(e.dial)}:</strong> ${team(e.active_team)} scores ${esc(e.active_points)}${e.side_correct ? "; the side guess was right" : ""}. Red ${esc(e.red_score)} – Blue ${esc(e.blue_score)}.`, r, e, true);
        case "sudden_death": return ui.card(`<strong>Sudden death</strong> at Red ${esc(e.red_score)} – Blue ${esc(e.blue_score)}.`, r, e, true);
        case "game_end": return ui.card(`<strong>Game ends</strong> (${esc(title(e.outcome))}).${e.red_score != null ? ` Red ${esc(e.red_score)} – Blue ${esc(e.blue_score)}.` : ""}`, r, e, true);
        default: return undefined;
      }
    },

    board(r, evs) {
      let clue = null, dial = null, target = null, side = null, red = 0, blue = 0;
      evs.forEach((e) => {
        if (e.event === "clue") { clue = e; dial = null; target = null; side = null; }
        if (e.event === "dial") dial = e.position;
        if (e.event === "side_guess") side = e.side;
        // Only lenses that were shown the target (the psychic, or everyone after the reveal) know it.
        if (e.event === "target_reveal") target = e.answer;
        if (e.event === "setup_target") target = e.target;
        if (e.event === "reveal") { target = e.target; red = e.red_score; blue = e.blue_score; }
      });
      const scale = Array.from({ length: STOPS }, (_, i) => `<span class="wl-stop${i === target ? " wl-stop--target" : ""}${i === dial ? " wl-stop--dial" : ""}">${i}</span>`).join("");
      return [
        ui.endBanner(r),
        ui.block("Score · first to 10", `<div class="wl-score">${team("RED")} <strong>${esc(red)}</strong> · ${team("BLUE")} <strong>${esc(blue)}</strong></div>`),
        clue ? ui.block(`${esc(clue.spectrum[0])} ↔ ${esc(clue.spectrum[1])}`, `
          <div class="small" style="margin-bottom:6px">Clue “<strong>${esc(clue.text)}</strong>” from ${name(r, clue.psychic)}</div>
          <div class="wl-scale">${scale}</div>
          <div class="small muted" style="margin-top:6px">${dial != null ? `Dial at ${esc(dial)}` : "Dial not set yet"}${target != null ? ` · target ${esc(target)}` : " · target hidden from this lens"}${side ? ` · side bet ${esc(title(side))}` : ""}</div>`) : "",
      ].join("");
    },
  });
})();
