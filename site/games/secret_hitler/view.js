/* Secret Hitler: nominations, elections, the secret legislative hands, and powers. */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;
  const state = MatchRoom.state;

  const name = (r, s) => `<strong>${esc(ui.shortName(r, s))}</strong>`;
  const policy = (c) => `<span class="team team--${esc(c)}">${esc(title(c))}</span>`;
  const hand = (e) => `${e.lib} Liberal, ${e.fasc} Fascist`;

  MatchRoom.defineGame("secret_hitler", {
    rolesArePublic: false,

    chapter: (e) => (e.event === "nomination" ? `P${e.president} nominates P${e.chancellor}` : null),

    speaker: (e) => (e.actor ?? e.president),

    renderEvent(r, e) {
      switch (e.event) {
        case "nomination": return ui.chapterHead(`President ${esc(ui.shortName(r, e.president))} nominates ${esc(ui.shortName(r, e.chancellor))}`, "Every living player votes Ja or Nein.");
        case "election": return ui.card(e.passed ? `<strong>Ja!</strong> ${name(r, e.president)} and ${name(r, e.chancellor)} form a government.` : `<strong>Nein.</strong> The election fails and the tracker moves on.`, r, e);
        case "president_draw": return ui.sys("hand", `The President draws ${hand(e)}.`, r, e);
        case "president_discard": return ui.sys("discard", `The President discards a ${policy(e.card)} policy.`, r, e);
        case "chancellor_hand": return ui.sys("hand", `The Chancellor receives ${hand(e)}.`, r, e);
        case "policy_enacted": return ui.card(`<strong>${policy(e.card)} policy enacted.</strong> Liberal ${e.lib_policies} – Fascist ${e.fasc_policies}.`, r, e, true);
        case "chaos_policy": return ui.card(`<strong>Three failed elections:</strong> the top card, ${policy(e.card)}, is enacted. Liberal ${e.lib_policies} – Fascist ${e.fasc_policies}.`, r, e, true);
        case "policy_peek": return ui.sys("peek", `${name(r, e.president)} peeks at the next three cards: ${e.policies.map(policy).join(", ")}.`, r, e);
        case "investigation": return ui.sys("power", `${name(r, e.president)} investigates ${name(r, e.target)}.`, r, e);
        case "investigation_result": return ui.sys("party", `${name(r, e.target)} belongs to the ${policy(e.party)} party.`, r, e);
        case "special_election": return ui.card(`${name(r, e.actor)} calls a special election: ${name(r, e.target)} is the next President.`, r, e);
        case "execution": return ui.card(`<strong>${esc(ui.shortName(r, e.actor))} executes ${esc(ui.shortName(r, e.target))}.</strong>`, r, e, true);
        case "veto_decision": return ui.card(`President ${name(r, e.president)} ${e.accepted ? "<strong>accepts the veto</strong>: both policies are discarded" : "<strong>rejects the veto</strong>: the Chancellor must enact"}.`, r, e);
        case "hitler_elected": return ui.card(`<strong>Hitler, ${esc(ui.shortName(r, e.chancellor))}, is elected Chancellor.</strong>`, r, e, true);
        default: return undefined;
      }
    },

    seatRole(r, p) {
      if (ui.roleVisibleTo(p.seat)) return undefined;
      // Fascists (and Hitler at 5–6 players) know their teammates.
      const k = typeof state.lens === "number" ? ui.player(r, state.lens).knowledge || {} : {};
      if (k.hitler_seat === p.seat) return `<span class="team team--FASCIST">Hitler (you know this)</span>`;
      if ((k.known_fascists || []).includes(p.seat)) return `<span class="team team--FASCIST">Fascist (you know this)</span>`;
      return undefined;
    },

    knowledge(r, p) {
      const k = p.knowledge || {};
      const bits = [];
      if ((k.known_fascists || []).length) bits.push(`Fascists: ${ui.seatList(r, k.known_fascists)}`);
      if (k.hitler_seat != null && k.hitler_seat !== p.seat) bits.push(`Hitler: ${ui.seatList(r, [k.hitler_seat])}`);
      return bits.length ? `<div class="brief">${bits.join(" · ")}</div>` : "";
    },

    board(r, evs) {
      let lib = 0, fasc = 0, tracker = 0, gov = null, nominee = null;
      const dead = [];
      evs.forEach((e) => {
        if (e.event === "nomination") nominee = e;
        if (e.event === "election") { if (e.passed) { gov = e; tracker = 0; } else tracker += 1; }
        if (e.event === "veto_decision" && e.accepted) tracker += 1;
        if (e.event === "policy_enacted" || e.event === "chaos_policy") { lib = e.lib_policies; fasc = e.fasc_policies; }
        if (e.event === "chaos_policy") tracker = 0;
        if (e.event === "execution") dead.push(e.target);
      });
      const track = (n, total, cls) => `<div class="sh-track">${Array.from({ length: total }, (_, i) => `<span class="sh-slot${i < n ? ` sh-slot--${cls}` : ""}"></span>`).join("")}</div>`;
      return [
        ui.endBanner(r),
        ui.block(`Liberal policies · ${lib} of 5`, track(lib, 5, "lib")),
        ui.block(`Fascist policies · ${fasc} of 6`, track(fasc, 6, "fasc")),
        ui.block("Failed elections in a row", `<div class="sh-pips">${[1, 2, 3].map((i) => `<span class="sh-pip${i <= tracker ? " sh-pip--on" : ""}"></span>`).join("")}<span>3 enacts the top card</span></div>`),
        gov ? ui.block("Last government", `<div class="small">President ${name(r, gov.president)} · Chancellor ${name(r, gov.chancellor)}</div>`) : nominee ? ui.block("Nominated", `<div class="small">President ${name(r, nominee.president)} · Chancellor ${name(r, nominee.chancellor)}</div>`) : "",
        dead.length ? ui.block("Executed", `<div class="small">${ui.seatList(r, dead)}</div>`) : "",
      ].join("");
    },
  });
})();
