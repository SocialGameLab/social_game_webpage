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
    const seats = (res.winners || []).map((s) => shortName(r, s));
    const head = res.winner ? `${title(res.winner)} wins` : seats.length && seats.length < r.players.length ? `${seats.join(" and ")} win${seats.length > 1 ? "" : "s"}` : "Game over";
    const sub = res.rewards ? Object.entries(res.rewards).map(([s, v]) => `${shortName(r, Number(s))}: ${v > 0 ? "+" : ""}${v}`).join(" · ") : "";
    return banner(head, sub);
  }

  // ---------------------------------------------------------------- readable events
  // Shared views (kinds) draw most events from their fields alone. These helpers
  // turn a raw log event into short labelled text with seat names in place of seat
  // numbers. Which keys name seats is a convention of the engine's specs.
  const SEAT_KEYS = new Set(["actor", "target", "seat", "winner", "loser", "leader", "giver", "guesser", "asker", "respondent", "answerer", "psychic", "proposer", "responder", "offerer", "employer", "worker", "lender", "borrower", "buyer", "seller", "host", "applicant", "caller", "doubter", "bidder", "blocker", "challenger", "claimer", "cutter", "accused", "suspect", "accuser", "chameleon", "undercover", "insider", "master", "president", "chancellor", "encryptor", "bomber", "best_seat", "dealer", "spy", "sender", "receiver", "inspector", "inspectee", "starting_seat", "first", "backed"]);
  const SEAT_LIST_KEYS = new Set(["winners", "team", "roommates", "tied", "seen_by", "spy_bonus", "known_evil", "known_wolves", "known_mafia", "known_moriarty", "survivors", "eliminated"]);
  // Seats that "say" an event, in order of preference, for the bubble's speaker.
  const SPEAKER_KEYS = ["actor", "asker", "respondent", "offerer", "proposer", "responder", "giver", "psychic", "encryptor", "seller", "buyer", "worker", "employer", "lender", "borrower", "cutter", "caller", "doubter", "bidder", "host", "sender"];
  const HIDDEN_KEYS = new Set(["event", "_visibility", "_audience"]);

  const isSeat = (r, v) => Number.isInteger(v) && v >= 0 && v < r.players.length;
  const label = (k) => title(k);

  /** One value as HTML: seats become names, per-seat arrays become "name value" pairs. */
  function fmt(r, key, v) {
    if (v === null || v === undefined) return "—";
    if (typeof v === "boolean") return v ? "yes" : "no";
    if (SEAT_KEYS.has(key) && isSeat(r, v)) return `<strong>${esc(shortName(r, v))}</strong>`;
    if (Array.isArray(v)) {
      if (SEAT_LIST_KEYS.has(key) && v.every((x) => isSeat(r, x))) return v.length ? seatList(r, v) : "nobody";
      if (v.length === r.players.length && r.players.length > 1 && !/range/.test(key) && v.every((x) => typeof x !== "object" || Array.isArray(x))) {
        return v.map((x, s) => `${esc(shortName(r, s))} <strong>${fmt(r, "", x)}</strong>`).join(" · ");
      }
      return v.map((x) => (typeof x === "object" ? esc(JSON.stringify(x)) : esc(String(x)))).join(", ");
    }
    if (typeof v === "object") return Object.entries(v).map(([k, x]) => `${esc(label(k))} ${fmt(r, k, x)}`).join(", ");
    return esc(String(v));
  }

  /** Fold `score0`, `score1`, … into one per-seat array under `score`. */
  function groupSeatKeys(r, e) {
    const out = {};
    Object.entries(e).forEach(([k, v]) => {
      if (HIDDEN_KEYS.has(k)) return;
      const m = /^([a-z_]*[a-z])_?(\d+)$/.exec(k);
      if (m && Number(m[2]) < r.players.length && r.players.length > 1) {
        (out[m[1]] = Array.isArray(out[m[1]]) ? out[m[1]] : Array(r.players.length).fill(null))[Number(m[2])] = v;
      } else out[k] = v;
    });
    return out;
  }

  /** "Label value · Label value" for an event's fields, minus `skip`. */
  const fields = (r, e, skip) => Object.entries(groupSeatKeys(r, e))
    .filter(([k]) => !(skip || []).includes(k))
    .map(([k, v]) => `<span class="kv"><span class="kv__k">${esc(label(k))}</span> ${fmt(r, k, v)}</span>`).join(" ");

  const speakerOf = (r, e) => { const k = SPEAKER_KEYS.find((key) => isSeat(r, e[key])); return k === undefined ? undefined : e[k]; };
  const speechOf = (e) => (typeof e.text === "string" ? "text" : typeof e.say === "string" ? "say" : null);
  const OUTCOME = /result|verdict|score|award|end|win|out$|elimin|lynch|kill|solved|escape|showdown|caught|accus|call|dudo|calza|reveal$/;

  /** A readable rendering for any event: a bubble when a seat speaks, a card for
   *  outcomes, a system line otherwise. `opts.big(e)` marks decisive moments. */
  function autoEvent(r, e, opts) {
    const o = opts || {};
    const speech = speechOf(e);
    const who = speakerOf(r, e);
    if (speech && who !== undefined) {
      const rest = fields(r, e, [speech, SPEAKER_KEYS.find((k) => e[k] === who)]);
      return bubble(r, who, e[speech], { event: e, meta: esc(title(e.event)), after: rest ? `<div class="kv-line small">${rest}</div>` : "" });
    }
    const body = fields(r, e);
    if (OUTCOME.test(e.event) && (!e._visibility || e._visibility === "public")) {
      return card(`<strong>${esc(title(e.event))}.</strong> ${body}`, r, e, o.big && o.big(e));
    }
    return sys(e.event.replace(/_/g, " "), body || esc(title(e.event)), r, e);
  }

  /** Round-by-seat table from events of the given types. Each cell stacks that
   *  seat's values (choice, payoff, score…); shared values go in a last column. */
  function roundTable(r, evs, types, opts) {
    const o = opts || {};
    const rows = evs.filter((e) => types.includes(e.event));
    if (!rows.length) return `<p class="muted small">No rounds played yet.</p>`;
    const n = r.players.length;
    const per = (g) => Object.keys(g).filter((k) => Array.isArray(g[k]) && g[k].length === n && !SEAT_LIST_KEYS.has(k) && !/range/.test(k));
    const roundKey = o.roundKey || ["round", "month", "leg_round"].find((k) => k in rows[0]);
    const groups = rows.map((e) => groupSeatKeys(r, e));
    const seatCols = [...new Set(groups.flatMap(per))];
    const shared = [...new Set(groups.flatMap((g) => Object.keys(g).filter((k) => !seatCols.includes(k) && k !== roundKey && !(o.skip || []).includes(k))))];
    const head = `<tr><th>${esc(label(roundKey || "#"))}</th>${r.players.map((p) => `<th>${dot(p.seat)}${esc(p.name)}</th>`).join("")}${shared.length ? "<th></th>" : ""}</tr>`;
    const body = groups.slice(-(o.limit || 12)).map((g, i, all) => `
      <tr${i === all.length - 1 ? ` class="rt-last"` : ""}><td>${esc(g[roundKey] ?? rows.length - all.length + i + 1)}</td>
        ${r.players.map((p) => `<td>${seatCols.filter((k) => Array.isArray(g[k])).map((k) => `<div title="${esc(label(k))}">${fmt(r, "", g[k][p.seat])}</div>`).join("")}</td>`).join("")}
        ${shared.length ? `<td class="small">${shared.map((k) => `<div><span class="kv__k">${esc(label(k))}</span> ${fmt(r, k, g[k])}</div>`).join("")}</td>` : ""}</tr>`).join("");
    const legend = seatCols.length ? `<p class="small muted rt-legend">Each cell: ${seatCols.map((k) => esc(label(k))).join(" · ")}</p>` : "";
    return `${legend}<div class="rt-wrap"><table class="rt">${head}${body}</table></div>`;
  }

  /** The game's opening `rules` event (if any) as a compact list. */
  function rulesBlock(r, evs) {
    const rules = evs.find((e) => e.event === "rules");
    const body = rules && fields(r, rules);
    return body ? block("Rules of this match", `<div class="kv-list small">${body}</div>`) : "";
  }

  /** Each seat's latest value of a running score, from per-seat fields of the events seen so far. */
  function latestScores(r, evs, key) {
    const k = key || "score";
    let last = null;
    evs.forEach((e) => { const g = groupSeatKeys(r, e); if (Array.isArray(g[k]) && g[k].length === r.players.length) last = g[k]; });
    return last;
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
    isSeat, fmt, fields, groupSeatKeys, speakerOf, autoEvent, roundTable, rulesBlock, latestScores,
  };
})();
