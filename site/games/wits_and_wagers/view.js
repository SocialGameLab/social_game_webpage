/* Wits & Wagers: a trivia question, secret estimates, then bets on the closest one. */
(function () {
  "use strict";

  const { esc } = MatchRoom.util;
  const ui = MatchRoom.ui;

  const name = (r, s) => `<strong>${esc(ui.shortName(r, s))}</strong>`;

  MatchRoom.defineGame("wits_and_wagers", {
    rolesArePublic: true,

    chapter: (e) => (e.event === "question" ? `Q${e.round + 1}: ${e.text}` : null),

    renderEvent(r, e) {
      switch (e.event) {
        case "question": return ui.chapterHead(`Question ${esc(e.round + 1)}`, esc(e.text));
        case "estimate_private": return ui.sys("estimate", `${name(r, e.actor)} writes down <strong>${esc(e.value)}</strong>.`, r, e);
        case "estimate": return ui.sys("estimate", `${name(r, e.seat)}'s estimate is turned up: <strong>${esc(e.value)}</strong>.`, r, e);
        case "answer": return ui.card(`<strong>The answer is ${esc(e.answer)}.</strong> ${ui.isSeat(r, e.best_seat) ? `Closest without going over: ${name(r, e.best_seat)} with ${esc(e.best_value)}.` : "Every estimate was too high."}`, r, e, true);
        case "wager": return ui.sys("wager", `${name(r, e.actor)} bets on ${name(r, e.backed)}'s estimate.`, r, e);
        default: return undefined;
      }
    },

    board(r, evs) {
      let q = null, answer = null;
      let estimates = {}, wagers = {};
      evs.forEach((e) => {
        if (e.event === "question") { q = e; answer = null; estimates = {}; wagers = {}; }
        if (e.event === "estimate") estimates[e.seat] = e.value;
        if (e.event === "answer") answer = e;
        if (e.event === "wager") wagers[e.actor] = e.backed;
      });
      if (!q) return ui.endBanner(r);
      const rows = Object.entries(estimates).sort((a, b) => a[1] - b[1]).map(([s, v]) => {
        const backers = Object.keys(wagers).filter((w) => wagers[w] === Number(s)).map(Number);
        return `<li class="wg-est${answer && answer.best_seat === Number(s) ? " wg-est--best" : ""}"><span>${ui.dot(Number(s))}${esc(ui.shortName(r, Number(s)))}</span><strong>${esc(v)}</strong><span class="small muted">${backers.length ? `backed by ${backers.map((b) => esc(ui.shortName(r, b))).join(", ")}` : ""}</span></li>`;
      }).join("");
      return [
        ui.endBanner(r),
        ui.block(`Question ${esc(q.round + 1)}`, `<p class="small">${esc(q.text)}</p>${answer ? `<p class="small">Answer <strong>${esc(answer.answer)}</strong></p>` : ""}`),
        rows ? ui.block("Estimates, low to high", `<ul class="wg-list">${rows}</ul>`) : "",
      ].join("");
    },
  });
})();
