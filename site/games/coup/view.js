/* Coup: claimed actions, challenges, blocks and lost influence. */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;

  const name = (r, s) => `<strong>${esc(ui.shortName(r, s))}</strong>`;
  const ACTION = { income: "takes Income", foreign_aid: "takes Foreign Aid", coup: "launches a Coup on", tax: "takes Tax", steal: "steals from", assassinate: "sends the Assassin after", exchange: "exchanges cards" };
  const act = (r, e) => `${name(r, e.actor)} ${esc(ACTION[e.kind] || title(e.kind))}${e.target != null ? ` ${name(r, e.target)}` : ""}`;

  MatchRoom.defineGame("coup", {
    chapter: (e) => (e.event === "claim" || e.event === "action" ? `P${e.actor}: ${title(e.kind)}` : null),

    speaker: (e) => ({ claim: e.actor, action: e.actor, challenge: e.challenger, block: e.blocker, influence_lost: e.seat, out: e.target })[e.event],

    renderEvent(r, e) {
      switch (e.event) {
        case "action": return ui.chapterHead(act(r, e), "No claim needed.");
        case "claim": return ui.chapterHead(act(r, e), `Claims <strong>${esc(e.claim)}</strong>.`);
        case "challenge": return ui.sys("challenge", `${name(r, e.challenger)} challenges ${name(r, e.claimer)}'s <strong>${esc(e.claim)}</strong>.`, r, e);
        case "challenge_result": return ui.card(e.outcome === "bluff_caught" ? `<strong>Bluff caught.</strong> ${name(r, e.loser)} was lying and loses an influence.` : `<strong>The claim was true.</strong> ${name(r, e.claimer)} shows the ${esc(e.claim)}; ${name(r, e.loser)} loses an influence.`, r, e);
        case "block": return ui.sys("block", `${name(r, e.blocker)} blocks the ${esc(title(e.action))} with <strong>${esc(e.claim)}</strong>.`, r, e);
        case "block_result": return ui.sys("block", `The block by ${name(r, e.blocker)} stands.`, r, e);
        case "influence_lost": return ui.card(`${name(r, e.seat)} turns over <strong>${esc(e.card)}</strong>.`, r, e);
        case "out": return ui.card(`<strong>${esc(ui.shortName(r, e.target))} is out of the game.</strong>`, r, e, true);
        case "game_end": return ui.card(`<strong>${(e.winner || []).map((s) => esc(ui.shortName(r, s))).join(" and ")} is the last one standing.</strong>`, r, e, true);
        default: return undefined;
      }
    },

    seatRole: () => "",

    result: (r) => (r.result && r.result.winners.length ? { head: `${r.result.winners.map((s) => ui.shortName(r, s)).join(" and ")} wins`, sub: "Last player with influence" } : null),

    board(r, evs) {
      const lost = {}, claims = {}, out = new Set();
      r.players.forEach((p) => { lost[p.seat] = []; claims[p.seat] = []; });
      evs.forEach((e) => {
        if (e.event === "influence_lost") lost[e.seat].push(e.card);
        if (e.event === "out") out.add(e.target);
        if (e.event === "claim") claims[e.actor].push(e.claim);
        if (e.event === "block") claims[e.blocker].push(e.claim);
      });
      const rows = r.players.map((p) => {
        const cards = [0, 1].map((i) => (lost[p.seat][i] ? `<span class="cp-card cp-card--lost">${esc(lost[p.seat][i])}</span>` : `<span class="cp-card">hidden</span>`)).join("");
        const said = [...new Set(claims[p.seat])];
        return `
          <li class="cp-row${out.has(p.seat) ? " cp-row--out" : ""}">
            <div class="cp-who">${ui.dot(p.seat)}<strong>${esc(p.name)}</strong>${out.has(p.seat) ? `<span class="small muted">out</span>` : ""}</div>
            <div class="cp-cards">${cards}</div>
            <div class="small muted">${said.length ? `Claimed ${said.map(esc).join(", ")}` : "No claims yet"}</div>
          </li>`;
      }).join("");
      return [
        ui.endBanner(r),
        ui.block("Influence · face-up cards are lost", `<ul class="cp-list">${rows}</ul>`),
      ].join("");
    },
  });
})();
