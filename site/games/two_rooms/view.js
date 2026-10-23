/* Two Rooms and a Boom: two rooms, card shares, leaders and hostage swaps. */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;
  const state = MatchRoom.state;

  const name = (r, s) => `<strong>${esc(ui.shortName(r, s))}</strong>`;
  const role = (x) => `<span class="team team--${/RED|BOMBER/.test(x) ? "RED" : "BLUE"}">${esc(title(x))}</span>`;
  // The round is public even though the hostage swaps that end each round are not.
  const currentRound = (r) => 1 + r.events.slice(0, ui.steps(r)[state.cursor] + 1).filter((e) => e.event === "hostage_exchange").length;

  MatchRoom.defineGame("two_rooms", {
    rolesArePublic: false,

    chapter: (e) => (e.event === "room_assignment" ? "Rooms dealt" : e.event === "hostage_exchange" ? `Swap after round ${e.round}` : null),

    renderEvent(r, e) {
      switch (e.event) {
        case "room_assignment": return ui.sys("room", `${name(r, e._audience[0])} starts in <strong>room ${e.room}</strong> with ${ui.seatList(r, e.roommates.filter((s) => s !== e._audience[0]))}.`, r, e);
        case "card_reveal": return ui.sys("reveal", `${name(r, e.actor)} shows their card to the room: ${role(e.role)}.`, r, e);
        case "card_share": return ui.sys("share", `${name(r, e.actor)} shows ${name(r, e.target)} their card: ${role(e.role)}.`, r, e);
        case "leader_appointed": return ui.sys("leader", `${name(r, e.leader)} leads room ${e.room}.`, r, e);
        case "leader_vote": return ui.sys("vote", `Room ${e.room} votes ${name(r, e.leader)} in as leader.`, r, e);
        case "hostage_exchange": return ui.chapterHead(`Hostages swap after round ${esc(e.round)}`, "The leaders send hostages to the other room.");
        case "reveal": return ui.card(`<strong>Final reveal.</strong> The President, ${name(r, e.president)}, is in room ${e.president_room}; the Bomber, ${name(r, e.bomber)}, is in room ${e.bomber_room}. ${e.bomber_room === e.president_room ? "Same room: Red wins." : "Different rooms: Blue wins."}`, r, e, true);
        default: return undefined;
      }
    },

    seatRole(r, p) {
      if (ui.roleVisibleTo(p.seat)) return undefined;
      // Cards shown to this lens reveal roles.
      const seen = [...ui.visibleEvents(r, state.cursor)].reverse().find((e) => (e.event === "card_share" || e.event === "card_reveal") && e.actor === p.seat);
      return seen ? `${role(seen.role)} <span class="small muted">(card seen)</span>` : undefined;
    },

    board(r, evs) {
      // Swaps are hidden from the seats, but the round each event falls in is public.
      if (!r._trRound) { let n = 1; r._trRound = new Map(r.events.map((e) => { const at = n; if (e.event === "hostage_exchange") n += 1; return [e, at]; })); }
      let rooms = null, round = 0;
      evs.forEach((e) => {
        // A new round: nobody knows a room again until its next leader event.
        if (r._trRound.get(e) !== round) { round = r._trRound.get(e); rooms = { 0: { members: null, leader: null }, 1: { members: null, leader: null } }; }
        if (e.event === "room_assignment") rooms[e.room].members = e.roommates;
        // Each room's leader events go to exactly that room's members.
        if (e.event === "leader_appointed" || e.event === "leader_vote") { rooms[e.room].members = e._audience || rooms[e.room].members; rooms[e.room].leader = e.leader; }
      });
      rooms = rooms || { 0: { members: null, leader: null }, 1: { members: null, leader: null } };
      const total = ((r.result || {}).variables || {}).total_rounds || 3;
      const room = (i) => ui.block(`Room ${i}${rooms[i].leader != null ? ` · leader ${esc(ui.shortName(r, rooms[i].leader))}` : ""}`,
        rooms[i].members ? `<div class="tr-chips">${rooms[i].members.map((s) => `<span class="tr-chip">${ui.dot(s)}${esc(ui.shortName(r, s))}</span>`).join("")}</div>` : `<p class="muted small">This lens does not know who is here.</p>`);
      return [
        ui.endBanner(r),
        ui.block(`Round ${Math.min(currentRound(r), total)} of ${total}`, `<p class="small muted">Red wins if the Bomber ends in the President's room.</p>`),
        room(0),
        room(1),
      ].join("");
    },
  });
})();
