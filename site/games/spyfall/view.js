/* Spyfall: five rounds, a re-dealt Spy each round, questions, accusations and guesses. */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;
  const state = MatchRoom.state;

  const name = (r, s) => `<strong>${esc(ui.shortName(r, s))}</strong>`;
  const OUTCOME = {
    correct_spy_guess: "The Spy named the location",
    wrong_spy_guess: "The Spy guessed wrong",
    spy_indicted: "The Spy was indicted",
    innocent_indicted: "An innocent was indicted",
    timeout: "The questions ran out; the Spy escapes",
  };

  // The round number is public, but the events that open a round are not, so
  // count rounds over every event so far, not just the ones this lens has seen.
  const currentRound = (r) => r.events.slice(0, ui.steps(r)[state.cursor] + 1).reduce((n, e) => Math.max(n, e.round || 0), 1);

  // Roles are re-dealt every round, so players[] (round 1) cannot say who is the
  // Spy now. Work it out from the events this lens has seen.
  function roundState(r) {
    const s = { round: currentRound(r), spy: null, reveal: null };
    ui.visibleEvents(r, state.cursor).forEach((e) => {
      if (e.event === "round_roles" && e.round === s.round) s.spy = e.spy;
      if (e.event === "location_reveal" && e.round === s.round) s.reveal = e;
    });
    return s;
  }

  MatchRoom.defineGame("spyfall", {
    // Roles change per round; seatRole draws them, and bubbles must not show round 1's.
    rolesArePublic: true,

    chapter: (e) => (e.event === "location_reveal" ? `Round ${e.round}` : null),

    speaker: (e) => (e.event === "question_asked" ? e.asker : e.event === "question_answered" ? e.respondent : e.event === "accusation" ? e.accuser : e.actor),

    renderEvent(r, e) {
      switch (e.event) {
        case "location_reveal": return ui.chapterHead(`Round ${e.round}`, `The location, <strong>${esc(e.location)}</strong>, goes to everyone but the Spy.`);
        case "round_roles": return ui.sys("spy", `${name(r, e.spy)} is the Spy this round.`, r, e);
        case "question_asked": return ui.bubble(r, e.asker, e.text, { event: e, meta: `asks ${esc(ui.shortName(r, e.respondent))}` });
        case "question_answered": return ui.bubble(r, e.respondent, e.text, { event: e, cls: "bubble--answer", meta: "answers" });
        case "accusation": return ui.card(`${name(r, e.accuser)} accuses ${name(r, e.suspect)} of being the Spy.`, r, e);
        case "indictment_vote": return ui.sys("vote", `Vote on ${name(r, e.suspect)}: ${e.approvals} approve, ${e.rejections} reject. ${e.rejections ? "Not unanimous; questions resume." : "Unanimous."}`, r, e);
        case "spy_guess": return ui.card(`<strong>${esc(ui.shortName(r, e.actor))} reveals as the Spy</strong> and guesses <strong>${esc(e.location)}</strong>.`, r, e, true);
        case "round_score": return ui.card(`<strong>Round ${e.round}:</strong> ${esc(OUTCOME[e.outcome] || title(e.outcome))}.`, r, e);
        default: return undefined;
      }
    },

    seatRole(r, p) {
      const s = roundState(r);
      if (state.lens === "all" && s.spy != null) return p.seat === s.spy ? `<span class="team team--SPY">Spy</span>` : `<span class="team team--PLAYERS">Knows the location</span>`;
      if (state.lens === p.seat) return s.reveal && (s.reveal._audience || []).includes(p.seat) ? `<span class="team team--PLAYERS">Knows: ${esc(s.reveal.location)}</span>` : `<span class="team team--SPY">Spy (only you know)</span>`;
      return `<span class="seat__role--hidden">▇▇▇▇ hidden</span>`;
    },

    result: (r) => (r.result && r.result.winners.length ? { head: `${r.result.winners.map((s) => ui.shortName(r, s)).join(" and ")} win${r.result.winners.length > 1 ? "" : "s"}`, sub: "Highest score after five rounds" } : null),

    board(r, evs) {
      const round = currentRound(r);
      const outcomes = {}, known = {};
      evs.forEach((e) => {
        if (e.event === "location_reveal") known[e.round] = e.location;
        if (e.event === "round_score") outcomes[e.round] = e.outcome;
      });
      const location = known[round] || null;
      const places = ((r.result || {}).variables || {}).locations || [];
      const total = ((r.result || {}).variables || {}).total_rounds || 5;
      const rounds = Array.from({ length: total }, (_, i) => i + 1).map((n) => `
        <li class="sf-round${n === round && !outcomes[n] ? " sf-round--active" : ""}"><span>Round ${n}</span><span class="small muted">${outcomes[n] ? esc(OUTCOME[outcomes[n]] || title(outcomes[n])) : n === round ? "in play" : ""}</span></li>`).join("");
      return [
        ui.endBanner(r),
        ui.block(`Possible locations${location ? " · this lens knows the answer" : ""}`, `<div class="sf-places">${places.map((pl) => `<span class="sf-place${pl === location ? " sf-place--known" : ""}">${esc(pl)}</span>`).join("")}</div>`),
        ui.block("Rounds", `<ul class="sf-rounds">${rounds}</ul>`),
      ].join("");
    },
  });
})();
