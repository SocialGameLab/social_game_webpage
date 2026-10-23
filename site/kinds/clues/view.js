/* Kind "clues": secret-word and deduction party games. Somebody holds a secret,
 * the table trades clues, questions and guesses, and a verdict closes each round.
 * The board shows what this lens knows, the clues of the current round, and the
 * results so far. Options (view_options):
 *   roles_public  true when every seat has the same role (default false)
 *   redeal        roles re-dealt each round: {event, seat_key, role, other, secret_event}
 *                 e.g. {event: round_roles, seat_key: chameleon, role: Chameleon,
 *                       other: Knows the topic, secret_event: topic_reveal} */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;
  const state = MatchRoom.state;

  const SPEECH = ["text", "say", "word", "answer"];
  const RESULT = /verdict|round_score|elimin|award|solved|escape|accusation|guesser_vote|reveal$|decode|intercept/;
  const isPrivate = (e) => e._visibility && e._visibility !== "public";
  const speechKey = (e) => SPEECH.find((k) => typeof e[k] === "string");

  // The round number is public even when the events that open a round are not.
  const currentRound = (r) => r.events.slice(0, ui.steps(r)[state.cursor] + 1).reduce((n, e) => Math.max(n, typeof e.round === "number" ? e.round : 0), 0);

  MatchRoom.defineKind("clues", (opt) => {
    const redeal = opt.redeal;

    function roleNow(r, p) {
      const round = currentRound(r);
      const evs = ui.visibleEvents(r, state.cursor);
      if (state.lens === "all") {
        const deal = [...evs].reverse().find((e) => e.event === redeal.event && e.round === round);
        if (deal) return deal[redeal.seat_key] === p.seat ? `<span class="team team--EVIL">${esc(redeal.role)}</span>` : `<span class="team team--GOOD">${esc(redeal.other)}</span>`;
      }
      if (state.lens === p.seat) {
        const told = evs.some((e) => e.event === redeal.secret_event && e.round === round);
        return told ? `<span class="team team--GOOD">${esc(redeal.other)}</span>` : `<span class="team team--EVIL">${esc(redeal.role)} (only you know)</span>`;
      }
      return `<span class="seat__role--hidden">▇▇▇▇ hidden</span>`;
    }

    return {
      rolesArePublic: Boolean(opt.roles_public || redeal),

      chapter: (e) => (isPrivate(e) && typeof e.round === "number" && e._visibility === "private" ? `Round ${e.round}` : e.event === "question" && e.number === 1 ? "Questions" : null),

      renderEvent(r, e) {
        const said = speechKey(e);
        const who = ui.speakerOf(r, e);
        if (said && who !== undefined) {
          return ui.bubble(r, who, e[said], { event: e, meta: esc(title(e.event)), after: (() => { const rest = ui.fields(r, e, [said, "actor", "round"]); return rest ? `<div class="kv-line small">${rest}</div>` : ""; })() });
        }
        if (isPrivate(e)) return ui.sys(e.event.replace(/_/g, " "), ui.fields(r, e), r, e);
        if (RESULT.test(e.event)) return ui.card(`<strong>${esc(title(e.event))}.</strong> ${ui.fields(r, e)}`, r, e, /verdict|game_end/.test(e.event));
        return ui.autoEvent(r, e);
      },

      seatRole: redeal ? (r, p) => roleNow(r, p) : undefined,

      knowledge: (r, p) => (Object.keys(p.knowledge || {}).length ? `<div class="brief">${ui.fields(r, p.knowledge)}</div>` : ""),

      board(r, evs) {
        const round = currentRound(r);
        const now = evs.filter((e) => !round || e.round === undefined || e.round === round);
        const secrets = now.filter((e) => isPrivate(e) && !speechKey(e)).slice(-3);
        const clues = now.filter((e) => speechKey(e) && ui.speakerOf(r, e) !== undefined && !/discuss/.test(e.event)).slice(-10);
        const results = evs.filter((e) => RESULT.test(e.event) && !isPrivate(e)).slice(-6);
        return [
          ui.endBanner(r),
          ui.block(round ? `Round ${round} · what this lens knows` : "What this lens knows", secrets.length ? secrets.map((e) => `<div class="small"><span class="kv__k">${esc(title(e.event))}</span> ${ui.fields(r, e, ["round"])}</div>`).join("") : `<p class="muted small">Nothing secret is visible to this lens.</p>`),
          clues.length ? ui.block("Clues and guesses", `<ul class="cl-list">${clues.map((e) => `<li>${ui.dot(ui.speakerOf(r, e))}<strong>${esc(ui.shortName(r, ui.speakerOf(r, e)))}</strong> <span class="small muted">${esc(title(e.event))}</span> ${esc(e[speechKey(e)]) || `<span class="muted">(blank)</span>`}</li>`).join("")}</ul>`) : "",
          results.length ? ui.block("Results", results.map((e) => `<div class="small"><span class="kv__k">${esc(title(e.event))}</span> ${ui.fields(r, e)}</div>`).join("")) : "",
        ].join("");
      },
    };
  });
})();
