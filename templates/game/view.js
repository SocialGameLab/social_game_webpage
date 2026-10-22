/* __TITLE__: what this file draws, in one line. */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;

  // Every hook is optional; see MatchRoom.defineGame in assets/js/core.js.
  // Shared events (setup, roster, discussion, game_end, game_over) already
  // render. Anything you do not handle shows as raw JSON, which is a good way
  // to see which events this game logs.
  MatchRoom.defineGame("__ID__", {
    rolesArePublic: true,

    // Feed entries that start a new section; they fill the "Jump to" menu.
    chapter: (e) => (e.event === "round_start" ? `Round ${e.round}` : null),

    renderEvent(r, e) {
      switch (e.event) {
        // Pick the closest shared piece so the feed reads the same across games:
        //   ui.chapterHead(head, sub)          section heading
        //   ui.sys(tag, html, r, e)            one-line system note
        //   ui.card(html, r, e, big)           outcome box; big = decisive moment
        //   ui.bubble(r, seat, text, {event})  a seat speaking
        case "round_start": return ui.chapterHead(`Round ${e.round}`, "");
        default: return undefined; // fall back to the shared renderer
      }
    },

    // Private info under a seat, shown only to "Everything" and that seat's lens.
    knowledge(r, p) {
      return "";
    },

    // Right-hand panel. `evs` are the events the current lens has seen so far.
    board(r, evs) {
      return [
        ui.endBanner(r),
        ui.block("State", `<p class="muted small">${evs.length} events so far.</p>`),
      ].join("");
    },
  });
})();
