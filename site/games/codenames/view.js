/* Codenames: clues, guesses, and the 5×5 board with or without the key. */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;

  const TOTAL = { RED: 9, BLUE: 8 };
  const colorName = (c) => ({ RED: "red agent", BLUE: "blue agent", NEUTRAL: "bystander", ASSASSIN: "the assassin" }[c] || c);

  MatchRoom.defineGame("codenames", {
    chapter: (e) => (e.event === "clue" ? `${title(e.team)} clue: ${e.word}` : null),

    renderEvent(r, e) {
      switch (e.event) {
        case "board": return ui.sys("board", "25 words dealt face up.", r, e);
        case "key_reveal": return ui.sys("key", `Key sent to the spymasters ${ui.seatList(r, e._audience)}.`, r, e);
        case "clue": {
          const sm = r.players.find((p) => p.role === `${e.team}_SPYMASTER`);
          return ui.chapterHead(`<span class="team team--${esc(e.team)}">${esc(title(e.team))}</span> clue: “${esc(e.word)}” · ${e.count}`, `Given by spymaster ${esc(sm ? sm.name : "")}.`);
        }
        case "reveal": return ui.sys("guess", `<strong>${esc(ui.shortName(r, e.actor))}</strong> picks <strong>${esc(e.word)}</strong> → <span class="team team--${esc(e.color)}">${esc(colorName(e.color))}</span>`, r, e);
        default: return undefined;
      }
    },

    knowledge: (r, p) => (/SPYMASTER/.test(p.role) ? `<div class="brief muted">Sees the key.</div>` : ""),

    board(r, evs) {
      let words = null, key = null, clue = null, last = null;
      const revealed = {};
      evs.forEach((e) => {
        if (e.event === "board") words = e.board;
        if (e.event === "key_reveal") key = e.colors;
        if (e.event === "clue") clue = e;
        if (e.event === "reveal") { revealed[e.index] = e.color; last = e.index; }
      });
      if (!words) return ui.block("Board", `<p class="muted small">The board is dealt in the next step.</p>`);
      const left = { ...TOTAL };
      Object.values(revealed).forEach((c) => { if (c in left) left[c] -= 1; });
      const grid = words.map((w, i) => {
        const cls = [i in revealed ? `cn-card--${revealed[i]}` : key ? `cn-card--key-${key[i]}` : "", i === last ? "cn-card--last" : ""].join(" ");
        return `<div class="cn-card ${cls}">${esc(w)}</div>`;
      }).join("");
      return [
        ui.endBanner(r),
        ui.block("Current clue", `
          ${clue ? `<div class="clue-chip team team--${esc(clue.team)}">${esc(clue.word)} · ${clue.count}</div>` : `<p class="muted small">No clue yet.</p>`}
          <div class="score-row"><span class="team team--RED">Red left ${left.RED}</span><span class="team team--BLUE">Blue left ${left.BLUE}</span></div>`),
        ui.block(`Board ${key ? "· key visible to this lens (outlines)" : "· operative view"}`, `<div class="cn-grid">${grid}</div>`),
      ].join("");
    },
  });
})();
