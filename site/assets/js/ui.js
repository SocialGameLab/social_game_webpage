/* Match Room UI kit: the building blocks every game view draws with, so new
 * games look like the existing ones. Exposed as MatchRoom.ui.
 *
 * Event visibility comes from the engine log: `_visibility` is "public",
 * "private" (seen by the seats in `_audience`) or "hidden" (engine only). The
 * lens is "all", "public", or a seat number. */
(function () {
  "use strict";

  const MR = window.MatchRoom;
  const { esc, title } = MR.util;
  const state = MR.state;

  const SEAT_COLORS = ["#c2553a", "#2f7fb5", "#8a5bb8", "#3e9a55", "#c99a1e", "#c24d86", "#2d9c9c", "#7a6e3a", "#5f6fd0", "#a0522d"];

  // ---------------------------------------------------------------- current selection
  const game = () => MR.data[state.gameId];
  const view = () => MR.views[state.gameId] || {};
  const run = () => { const g = game(); return g.runs.find((r) => r.id === state.runId) || g.runs[0]; };

  // ---------------------------------------------------------------- seats
  const seatColor = (s) => SEAT_COLORS[s % SEAT_COLORS.length];
  const player = (r, s) => r.players[s] || { seat: s, name: `P${s}` };
  const isGenericName = (p) => p.name === `P${p.seat}`;
  const seatLabel = (r, s) => { const p = player(r, s); return isGenericName(p) ? p.name : `${p.name} (P${s})`; };
  const shortName = (r, s) => player(r, s).name;
  const seatList = (r, seats) => (seats || []).map((s) => `<strong>${esc(shortName(r, s))}</strong>`).join(", ");
  const dot = (s, extra) => `<span class="lens-dot" style="${extra || ""}background:${seatColor(s)}"></span>`;

  /** Whether the current lens may see this seat's role. */
  const roleVisibleTo = (seat) => state.lens === "all" || state.lens === seat || view().rolesArePublic;
  /** Whether the current lens may see this seat's private knowledge. */
  const privateVisibleTo = (seat) => state.lens === "all" || state.lens === seat;

  // ---------------------------------------------------------------- lens and steps
  function canSee(e, lens) {
    const v = e._visibility;
    if (!v || v === "public") return true;
    if (lens === "all") return true;
    if (v === "private" && typeof lens === "number") return (e._audience || []).includes(lens);
    return false;
  }

  function audienceText(r, e) {
    if (e._visibility === "hidden") return "engine only";
    return "seen by " + (e._audience || []).map((s) => shortName(r, s)).join(", ");
  }

  // Steps are the events the feed renders; extra roster lines fold into the first.
  function steps(r) {
    if (!r._steps) {
      let rosterSeen = false;
      r._steps = r.events.reduce((acc, e, i) => {
        if (e.event === "roster") { if (rosterSeen) return acc; rosterSeen = true; }
        acc.push(i);
        return acc;
      }, []);
    }
    return r._steps;
  }

  const visibleEvents = (r, uptoStep) => steps(r).slice(0, uptoStep + 1).map((i) => r.events[i]).filter((e) => canSee(e, state.lens));
  const currentEvent = (r) => r.events[steps(r)[state.cursor]] || {};
  const atEnd = (r) => state.cursor >= steps(r).length - 1;

  // ---------------------------------------------------------------- feed pieces
  function visBadge(r, e) {
    if (state.lens !== "all" || !e._visibility || e._visibility === "public") return "";
    return `<span class="vis-badge vis-badge--${e._visibility}">${esc(audienceText(r, e))}</span>`;
  }

  /** One-line system note: a mono tag, then text. */
  const sys = (tag, html, r, e) => `<div class="ev-sys"><span class="ev-tag">${esc(tag)}</span><span>${html} ${visBadge(r, e)}</span></div>`;

  /** Boxed outcome; `big` adds the brass highlight for decisive moments. */
  const card = (html, r, e, big) => `<div class="ev-card${big ? " ev-card--big" : ""}">${html} ${visBadge(r, e)}</div>`;

  /** Section heading in the feed; these are also good "Jump to" targets. */
  const chapterHead = (head, sub) => `<div class="ev-chapter">${head}${sub ? `<small>${sub}</small>` : ""}</div>`;

  /** A seat speaking. opts: cls (bubble--whisper | bubble--answer), meta, after. */
  function bubble(r, seat, body, opts) {
    const o = opts || {};
    const p = player(r, seat);
    const e = o.event;
    const roleBit = roleVisibleTo(seat) && p.role && !view().rolesArePublic ? `<span class="team team--${esc(p.team)}">${esc(title(p.role))}</span>` : "";
    const text = body && String(body).trim() ? esc(body) : `<span class="bubble__empty">(no message)</span>`;
    return `
      <div class="bubble ${o.cls || ""}">
        <div class="avatar" style="background:${seatColor(seat)}" aria-hidden="true">${esc(isGenericName(p) ? p.name : p.name.slice(0, 2))}</div>
        <div>
          <div class="bubble__meta"><span class="bubble__who">${esc(p.name)}</span>${isGenericName(p) ? "" : `<span class="seat__seat">P${seat}</span>`}${roleBit}${o.meta ? `<span class="small">${o.meta}</span>` : ""}${o.metaHtml || ""}${e ? visBadge(r, e) : ""}</div>
          <div class="bubble__body">${text}${o.after || ""}</div>
        </div>
      </div>`;
  }

  // ---------------------------------------------------------------- board pieces
  /** Labelled section of the right-hand board panel. */
  const block = (label, html) => `<div class="board-block">${label ? `<div class="board-label">${label}</div>` : ""}${html}</div>`;

  const banner = (head, sub) => block("", `<div class="banner">${esc(head)}<small>${esc(sub || "")}</small></div>`);

  /** Final-result banner from run.result; a view's result(r) can override it. */
  function endBanner(r) {
    if (!atEnd(r)) return "";
    const custom = view().result && view().result(r);
    if (custom) return banner(custom.head, custom.sub);
    if (!r.result) return "";
    const res = r.result;
    const head = res.winner ? `${title(res.winner)} wins` : "Game over";
    const sub = res.rewards ? Object.entries(res.rewards).map(([s, v]) => `${shortName(r, Number(s))}: ${v > 0 ? "+" : ""}${v}`).join(" · ") : "";
    return banner(head, sub);
  }

  // ---------------------------------------------------------------- dialog
  /** Fill and show the shared <dialog>. Any [data-close] inside closes it. */
  function openDialog(head, body) {
    const dlg = document.getElementById("dialog");
    dlg.innerHTML = `
      <div class="dialog__head"><h2>${esc(head)}</h2><button class="chip-btn" type="button" data-close>Close</button></div>
      <div class="dialog__body">${body}</div>`;
    if (!dlg.open) dlg.showModal();
  }

  MR.ui = {
    game, view, run,
    seatColor, player, isGenericName, seatLabel, shortName, seatList, dot, roleVisibleTo, privateVisibleTo,
    canSee, steps, visibleEvents, currentEvent, atEnd,
    visBadge, sys, card, chapterHead, bubble,
    block, banner, endBanner, openDialog,
  };
})();
