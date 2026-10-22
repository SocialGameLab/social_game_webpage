/* Avalon: proposals, signals, votes, missions, and the final assassination. */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;
  const state = MatchRoom.state;

  const TEAM_SIZE = { 5: [2, 3, 2, 3, 3], 6: [2, 3, 4, 3, 4], 7: [2, 3, 3, 4, 4], 8: [3, 4, 4, 5, 5], 9: [3, 4, 4, 5, 5], 10: [3, 4, 4, 5, 5] };

  function signalText(r, e) {
    const who = `<strong>${esc(ui.shortName(r, e.actor))}</strong>`;
    const tgt = e.target != null ? `<strong>${esc(ui.shortName(r, e.target))}</strong>` : "";
    switch (e.type) {
      case "VOTE_INTENT": return `${who} intends to <strong>${esc(title(e.intent))}</strong>.`;
      case "OPPOSE": return `${who} opposes ${tgt}.`;
      case "TRUST": return `${who} trusts ${tgt}.`;
      case "ACCUSE": return `${who} accuses ${tgt}.`;
      case "NOMINATE": return `${who} nominates ${tgt}.`;
      case "NOMINATE_SELF": return `${who} nominates themself.`;
      case "CLAIM_ROLE": return `${who} claims to be <strong>${esc(title(e.role))}</strong>.`;
      case "PASS": return `${who} passes.`;
      default: return `${who} ${esc(title(e.type))} ${tgt}`;
    }
  }

  MatchRoom.defineGame("avalon", {
    rolesArePublic: false,

    chapter: (e) => (e.event === "proposal" ? `Mission ${e.mission} · proposal ${e.proposal_attempt}` : null),

    renderEvent(r, e) {
      switch (e.event) {
        case "proposal": return ui.chapterHead(`Mission ${e.mission} · proposal ${e.proposal_attempt}`, `Leader ${esc(ui.shortName(r, e.leader))} proposes ${ui.seatList(r, e.team)}.`);
        case "signal": return ui.sys("signal", signalText(r, e), r, e);
        case "vote_result": return ui.card(`<strong>Vote ${e.approve}–${e.reject}.</strong> ${e.passed ? "Team approved." : "Team rejected; leadership passes on."}`, r, e);
        case "mission_cards": return ui.card(`Mission ${e.mission} cards: <strong>${e.success_count} Success</strong>, <strong>${e.fail_count} Fail</strong>.`, r, e);
        case "mission_result": return ui.card(`<strong>Mission ${e.mission} ${e.succeeded ? "succeeded" : "failed"}.</strong> Good ${e.good_points} – Evil ${e.evil_points}.`, r, e, true);
        case "assassination": return ui.card(`<strong>${esc(ui.shortName(r, e.actor))} names ${esc(ui.shortName(r, e.target))} as Merlin.</strong> ${e.success ? "Correct: evil steals the game." : "Wrong: good holds on."}`, r, e, true);
        default: return undefined;
      }
    },

    seatRole(r, p) {
      if (ui.roleVisibleTo(p.seat)) {
        return `<span class="team team--${esc(p.team)}">${esc(title(p.role))}</span>${p.team && p.team !== "PLAYERS" ? ` · ${esc(title(p.team))}` : ""}`;
      }
      // Merlin and evil players know who is evil even when the role is hidden.
      const k = typeof state.lens === "number" ? ui.player(r, state.lens).knowledge || {} : {};
      if ((k.known_evil || []).includes(p.seat)) return `<span class="team team--EVIL">Evil (you know this)</span>`;
      return undefined;
    },

    knowledge(r, p) {
      const k = p.knowledge || {};
      return (k.known_evil || []).length ? `<div class="brief">Knows evil: ${ui.seatList(r, k.known_evil)}</div>` : "";
    },

    board(r, evs) {
      const sizes = TEAM_SIZE[r.players.length] || [];
      const results = {};
      const cards = {};
      let proposal = null, vote = null, rejections = 0, assassination = null;
      evs.forEach((e) => {
        if (e.event === "proposal") { proposal = e; vote = null; }
        if (e.event === "vote_result") { vote = e; rejections = e.passed ? 0 : rejections + 1; }
        if (e.event === "mission_cards") cards[e.mission] = e;
        if (e.event === "mission_result") results[e.mission] = e.succeeded;
        if (e.event === "assassination") assassination = e;
      });
      const active = proposal ? proposal.mission : 1;
      const missions = sizes.map((n, i) => {
        const m = i + 1;
        const cls = m in results ? (results[m] ? "mission--ok" : "mission--fail") : m === active ? "mission--active" : "";
        return `<div class="mission ${cls}" title="Mission ${m}: team of ${n}">${m in results ? (results[m] ? "✓" : "✗") : n}</div>`;
      }).join("");
      const caps = sizes.map((_, i) => { const c = cards[i + 1]; return `<span>${c ? `${c.fail_count} fail` : "&nbsp;"}</span>`; }).join("");
      return [
        ui.endBanner(r),
        ui.block("Missions · team size", `<div class="missions">${missions}</div><div class="mission-cap">${caps}</div>`),
        proposal ? ui.block(`Mission ${proposal.mission} · proposal ${proposal.proposal_attempt}`, `
          <div class="small" style="margin-bottom:6px">Leader <strong>${esc(ui.shortName(r, proposal.leader))}</strong></div>
          <div class="team-chips">${proposal.team.map((s) => `<span class="team-chip">${ui.dot(s)}${esc(ui.shortName(r, s))}</span>`).join("")}</div>
          ${vote ? `<div class="small" style="margin-top:8px">Vote ${vote.approve}–${vote.reject}: <strong>${vote.passed ? "approved" : "rejected"}</strong></div>` : ""}`) : "",
        ui.block("Rejected in a row", `<div class="reject-track">${[1, 2, 3, 4, 5].map((i) => `<span class="reject-pip${i <= rejections ? " reject-pip--on" : ""}"></span>`).join("")}<span>5 hands evil the game</span></div>`),
        assassination ? ui.block("Assassination", `<div class="small"><strong>${esc(ui.shortName(r, assassination.actor))}</strong> → <strong>${esc(ui.shortName(r, assassination.target))}</strong>: ${assassination.success ? "Merlin found" : "missed"}</div>`) : "",
      ].join("");
    },
  });
})();
