/* Match Room app: routing, the game menu, the two game views ("How to play"
 * and "Watch the match"), playback and input. The home page is home.js. Game-specific drawing lives in
 * games/<id>/view.js; shared pieces in ui.js. Every view is a string template
 * re-rendered from MatchRoom.state. */
(function () {
  "use strict";

  const MR = window.MatchRoom;
  const { esc, inline, title } = MR.util;
  const ui = MR.ui;
  const { game, view, run, steps } = ui;
  const state = MR.state;

  const app = document.getElementById("app");
  const gameSelect = document.getElementById("game-select");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const BUILD_HINT = "uv run --project ../social_game_env python tools/build_data.py";

  const site = () => MR.manifest.site || {};

  // ---------------------------------------------------------------- routing
  // Hash is a plain token (game.view.run) so links can carry it. An empty or
  // unknown hash is the home page.
  function readHash() {
    const [gameId, viewName, runId] = location.hash.replace(/^#/, "").split(".");
    const known = MR.manifest.games.find((x) => x.id === gameId);
    if (!known) return { gameId: null };
    return { gameId: known.id, view: viewName === "watch" ? "watch" : "rules", runId };
  }

  function writeHash() {
    const parts = [state.gameId, state.view];
    if (state.gameId && game().runs.length > 1) parts.push(state.runId);
    const url = state.gameId ? "#" + parts.join(".") : location.pathname + location.search;
    try { history.replaceState(null, "", url); } catch (_) { /* sandboxed frame */ }
  }

  /** Switch page, game, view or run. Loads the game first if it has not arrived yet. */
  async function go(patch) {
    stop();
    const target = { ...state, ...patch };
    if (!target.gameId) {
      state.gameId = null;
      writeHash();
      renderHome();
      return;
    }
    if (!MR.data[target.gameId] || !MR.views[target.gameId]) {
      renderNav(target.gameId);
      app.innerHTML = `<p class="feed__empty">Loading…</p>`;
      try { await MR.loadGame(target.gameId); } catch (err) {
        app.innerHTML = `<p class="feed__empty">${esc(err.message)}. Build the data with <code>${esc(BUILD_HINT)}</code>.</p>`;
        return;
      }
    }
    const changedRun = target.gameId !== state.gameId || target.runId !== state.runId;
    Object.assign(state, patch);
    const runs = game().runs;
    if (!runs.some((r) => r.id === state.runId)) state.runId = (runs[0] || {}).id;
    if (changedRun) { state.cursor = 0; state.lens = "all"; }
    writeHash();
    render();
  }

  // ---------------------------------------------------------------- top level
  // The masthead holds one menu, grouped by kind, so it stays the same size
  // however many games there are. The home page is the full gallery.
  function renderNav(current) {
    const groups = {};
    MR.manifest.games.forEach((g) => (groups[g.kind] = groups[g.kind] || []).push(g));
    gameSelect.innerHTML = `<option value="" ${current ? "" : "selected"}>${current ? "All games" : "Choose a game…"}</option>` +
      Object.entries(groups).map(([kind, gs]) => `<optgroup label="${esc(kind)}">${gs.map((g) => `<option value="${esc(g.id)}" ${g.id === current ? "selected" : ""}>${esc(g.title)}</option>`).join("")}</optgroup>`).join("");
    document.querySelectorAll(".nav-link[data-home]").forEach((a) => a.setAttribute("aria-current", current ? "false" : "page"));
  }

  function renderHome() {
    renderNav(null);
    document.title = site().title || "Match Room";
    MR.home.render(app);
  }

  function render() {
    renderNav(state.gameId);
    const g = game();
    document.title = `${g.title} · ${site().title || "Match Room"}`;
    app.innerHTML = `
      <section class="game-head">
        <div>
          <div class="eyebrow">${esc(g.kind)} · ${esc(g.players)}</div>
          <h1 class="game-title">${esc(g.title)}</h1>
          <p class="tagline">${esc(g.tagline)}</p>
        </div>
        <div class="view-switch" role="group" aria-label="View">
          <button type="button" data-view="rules" aria-pressed="${state.view === "rules"}">How to play</button>
          <button type="button" data-view="watch" aria-pressed="${state.view === "watch"}">Watch the match</button>
        </div>
      </section>
      <hr class="rule">
      ${state.view === "rules" ? rulesView(g) : watchView(g)}`;
    if (state.view === "watch" && g.runs.length) updateRoom(false);
  }

  // ---------------------------------------------------------------- how to play
  function list(items, ordered) {
    const tag = ordered ? "ol" : "ul";
    return `<${tag}>${(items || []).map((t) => `<li><span>${inline(t)}</span></li>`).join("")}</${tag}>`;
  }

  function sourceRef(path) {
    if (!path) return "";
    const repo = site().source_repo;
    const code = `<code>${esc(path)}</code>`;
    return repo ? `<a href="${esc(`${repo.replace(/\/$/, "")}/blob/${site().source_branch || "main"}/${path}`)}">${code}</a>` : code;
  }

  // Only the sources a game has: GameSpec games name a rule doc, spec and config;
  // games recorded through a training env name the env id instead.
  function sources(g) {
    const refs = [["", g.rules_doc], ["spec ", g.spec], ["config ", g.config]].filter(([, p]) => p).map(([k, p]) => `${k}${sourceRef(p)}`);
    if (g.env_id) refs.push(`env <code>${esc(g.env_id)}</code>`);
    return refs.length ? `<p class="sources">Source of truth: ${refs.join(" · ")}</p>` : "";
  }

  function rulesView(g) {
    const r = g.runs.length ? run() : null;
    const prompt = r && r.prompt;
    const promptHead = prompt
      ? `<strong>First prompt sent to ${esc(ui.seatLabel(r, prompt.seat))}</strong>, ${esc(title(prompt.phase))} phase${r.label ? ` in ${esc(r.label)}` : ""}. From the recorded run, seed ${esc(r.seed)}.`
      : "No recorded run yet.";
    return `
      <div class="rules">
        <div>
          <section><h2>Setup</h2>${list(g.setup)}</section>
          <section><h2>Each round</h2>${list(g.round, true)}</section>
          <section><h2>How it ends</h2>${list(g.win)}</section>
          ${sources(g)}
        </div>
        <div>
          <section><h2>What an agent sees</h2>
            ${list(g.agents)}
            <div class="prompt-card" style="margin-top:14px">
              <div class="prompt-card__head">${promptHead}</div>
              ${prompt ? `<pre class="prompt-text">${esc(prompt.text)}</pre>` : ""}
            </div>
          </section>
          <section><h2>Watch for</h2>${list(g.watch_for)}</section>
          ${g.runs.length ? `<button class="cta" type="button" data-view="watch">Watch the recorded match →</button>` : ""}
        </div>
      </div>`;
  }

  // ---------------------------------------------------------------- watch
  function watchView(g) {
    if (!g.runs.length) return `<p class="feed__empty">No completed run for this game yet. Record one, then rebuild the data with <code>${esc(BUILD_HINT)}</code>.</p>`;
    const r = run();
    const runPicker = g.runs.length > 1 ? `
      <div class="run-picker" role="group" aria-label="Recorded run">
        <span class="toolbar__label">${esc(g.run_label || "Run")}</span>
        ${g.runs.map((x) => `<button class="chip-btn" type="button" data-run="${esc(x.id)}" aria-pressed="${x.id === r.id}">${esc(x.label || x.id)}</button>`).join("")}
      </div>` : "";
    const lensBtns = [
      `<button class="chip-btn" type="button" data-lens="all" aria-pressed="${state.lens === "all"}">Everything</button>`,
      `<button class="chip-btn" type="button" data-lens="public" aria-pressed="${state.lens === "public"}">Public table</button>`,
      ...r.players.map((p) => `<button class="chip-btn" type="button" data-lens="${p.seat}" aria-pressed="${state.lens === p.seat}">${ui.dot(p.seat)}${esc(p.name)}</button>`),
    ].join("");
    return `
      <div class="toolbar">
        ${runPicker}
        <div class="lens" role="group" aria-label="Watch through the eyes of">
          <span class="toolbar__label">Lens</span>${lensBtns}
        </div>
      </div>
      ${r.note ? `<p class="run-note">${esc(r.note)}</p>` : ""}
      <div class="room">
        <aside class="panel room__seats sticky" aria-label="Players"><h2>At the table</h2><div class="seat-list" id="seats"></div></aside>
        <section class="feed" id="feed" aria-live="polite" aria-label="Match log"></section>
        <aside class="panel room__board sticky" id="board" aria-label="Game state"></aside>
      </div>
      ${transport(r)}
      <dialog class="dialog" id="dialog"></dialog>`;
  }

  function chapters(r) {
    const label = view().chapter;
    if (!label) return [];
    const out = [];
    steps(r).forEach((i, stepIdx) => {
      const text = label(r.events[i], r);
      // Repeats in a row (one label per introduction, say) collapse into the first.
      if (text && (!out.length || out[out.length - 1][1] !== text)) out.push([stepIdx, text]);
    });
    return out;
  }

  function transport(r) {
    const n = steps(r).length;
    const ch = chapters(r);
    return `
      <div class="transport" role="group" aria-label="Playback">
        <button class="transport__btn" type="button" data-step="start" aria-label="Go to start">⏮</button>
        <button class="transport__btn" type="button" data-step="-1" aria-label="Previous event">◀</button>
        <button class="transport__btn transport__btn--play" type="button" data-play aria-label="Play">Play</button>
        <button class="transport__btn" type="button" data-step="1" aria-label="Next event">▶</button>
        <button class="transport__btn" type="button" data-step="end" aria-label="Go to end">⏭</button>
        <input type="range" min="0" max="${n - 1}" value="${state.cursor}" id="scrub" aria-label="Event position">
        <span class="transport__count" id="count"></span>
        <select id="speed" aria-label="Playback speed">
          ${[0.5, 1, 2, 4].map((s) => `<option value="${s}" ${s === state.speed ? "selected" : ""}>${s}×</option>`).join("")}
        </select>
        ${ch.length ? `<select id="chapter" aria-label="Jump to"><option value="">Jump to…</option>${ch.map(([s, label]) => `<option value="${s}">${esc(label.length > 60 ? label.slice(0, 58) + "…" : label)}</option>`).join("")}</select>` : ""}
      </div>`;
  }

  function updateRoom(scroll) {
    const r = run();
    const n = steps(r).length;
    state.cursor = Math.max(0, Math.min(state.cursor, n - 1));
    document.getElementById("feed").innerHTML = feedHtml(r);
    document.getElementById("seats").innerHTML = seatsHtml(r);
    document.getElementById("board").innerHTML = view().board ? view().board(r, ui.visibleEvents(r, state.cursor)) : ui.endBanner(r);
    document.getElementById("scrub").value = state.cursor;
    document.getElementById("count").textContent = `${state.cursor + 1} / ${n}`;
    const play = app.querySelector("[data-play]");
    play.textContent = state.playing ? "Pause" : "Play";
    play.setAttribute("aria-label", state.playing ? "Pause" : "Play");
    if (scroll) {
      const cur = document.querySelector(".ev--current");
      if (cur) cur.scrollIntoView({ block: "nearest", behavior: reduceMotion ? "auto" : "smooth" });
    }
  }

  // ---------------------------------------------------------------- feed
  function feedHtml(r) {
    const idx = steps(r).slice(0, state.cursor + 1);
    const out = [];
    let hidden = [];
    const flush = () => {
      if (!hidden.length) return;
      const who = state.lens === "public" ? "the public table" : ui.shortName(r, state.lens);
      const kinds = [...new Set(hidden.map((e) => title(e.event)))].join(", ");
      out.push(`<div class="redacted" title="${esc(kinds)}"><span class="redacted__bar" aria-hidden="true"></span>${hidden.length} event${hidden.length > 1 ? "s" : ""} ${esc(who)} cannot see</div>`);
      hidden = [];
    };
    idx.forEach((i, k) => {
      const e = r.events[i];
      const current = k === idx.length - 1;
      if (!ui.canSee(e, state.lens)) { hidden.push(e); if (current) flush(); return; }
      flush();
      const html = renderEvent(r, e);
      if (html) out.push(`<div class="ev${current ? " ev--current" : ""}">${html}</div>`);
    });
    flush();
    if (state.cursor === 0 && out.length <= 1) out.push(`<p class="feed__empty">Press Play, or use → to step through the match. Space toggles playback.</p>`);
    return out.join("");
  }

  /** The game's view gets first pick; these cover events shared by all games. */
  function renderEvent(r, e) {
    const custom = view().renderEvent && view().renderEvent(r, e);
    if (custom !== undefined) return custom;
    switch (e.event) {
      case "setup": return ui.sys("setup", "Roles and private information dealt in secret.", r, e);
      case "game_over": return ui.sys("end", "Game over.", r, e);
      case "roster": return ui.sys("table", r.players.map((p) => `<strong>${esc(p.name)}</strong>`).join(" · ") + " sit down.", r, e);
      case "discussion": return ui.bubble(r, e.actor, e.text, { event: e });
      case "game_end": return ui.card(`<strong>Game ends:</strong> ${esc(title(e.outcome))}.`, r, e, true);
      default: {
        const { event, _visibility, _audience, ...rest } = e;
        return ui.sys(event, `<code>${esc(JSON.stringify(rest))}</code>`, r, e);
      }
    }
  }

  // ---------------------------------------------------------------- seats
  function seatsHtml(r) {
    const v = view();
    const cur = ui.currentEvent(r);
    const speaking = v.speaker ? v.speaker(cur) : cur.actor;
    return r.players.map((p) => {
      let roleLine = v.seatRole && v.seatRole(r, p);
      if (roleLine === undefined) {
        roleLine = ui.roleVisibleTo(p.seat)
          ? `<span class="team team--${esc(p.team)}">${esc(title(p.role))}</span>`
          : `<span class="seat__role--hidden">▇▇▇▇ hidden</span>`;
      }
      return `
        <div class="seat${speaking === p.seat ? " seat--speaking" : ""}">
          <span class="seat__dot" style="background:${ui.seatColor(p.seat)}"></span>
          <div>
            <span class="seat__name">${esc(p.name)}</span>${ui.isGenericName(p) ? "" : `<span class="seat__seat">P${p.seat}</span>`}
            <div class="seat__role">${roleLine}</div>
          </div>
          ${ui.privateVisibleTo(p.seat) && v.knowledge ? v.knowledge(r, p) : ""}
        </div>`;
    }).join("");
  }

  // ---------------------------------------------------------------- playback
  function stepDelay(r) {
    const e = ui.currentEvent(r);
    const text = e.text || e.say || "";
    const base = text ? Math.min(4200, 900 + text.length * 18) : 650;
    return base / state.speed;
  }

  function tick() {
    const r = run();
    if (ui.atEnd(r)) { stop(); updateRoom(false); return; }
    state.cursor += 1;
    updateRoom(true);
    state.timer = setTimeout(tick, stepDelay(r));
  }

  function play() {
    const r = run();
    if (ui.atEnd(r)) state.cursor = 0;
    state.playing = true;
    updateRoom(true);
    state.timer = setTimeout(tick, stepDelay(r));
  }

  function stop() {
    state.playing = false;
    clearTimeout(state.timer);
    state.timer = null;
  }

  function seek(to) {
    stop();
    state.cursor = to;
    updateRoom(true);
  }

  const togglePlay = () => (state.playing ? (stop(), updateRoom(false)) : play());

  // ---------------------------------------------------------------- input
  gameSelect.addEventListener("change", () => { go({ gameId: gameSelect.value || null }); window.scrollTo({ top: 0 }); });

  document.addEventListener("click", (ev) => {
    if (!ev.target.closest("[data-home]")) return;
    ev.preventDefault();
    go({ gameId: null });
    window.scrollTo({ top: 0 });
  });

  app.addEventListener("click", (ev) => {
    const t = ev.target;
    const viewBtn = t.closest("[data-view]");
    if (viewBtn) { go({ view: viewBtn.dataset.view }); window.scrollTo({ top: 0 }); return; }
    const runBtn = t.closest("[data-run]");
    if (runBtn) { go({ runId: runBtn.dataset.run }); return; }
    const lens = t.closest("[data-lens]");
    if (lens) {
      const v = lens.dataset.lens;
      state.lens = v === "all" || v === "public" ? v : Number(v);
      app.querySelectorAll("[data-lens]").forEach((b) => b.setAttribute("aria-pressed", String(b === lens)));
      updateRoom(false);
      return;
    }
    const step = t.closest("[data-step]");
    if (step) {
      const n = steps(run()).length;
      const s = step.dataset.step;
      seek(s === "start" ? 0 : s === "end" ? n - 1 : Math.max(0, Math.min(n - 1, state.cursor + Number(s))));
      return;
    }
    if (t.closest("[data-play]")) { togglePlay(); return; }
    if (t.closest("[data-close]")) { document.getElementById("dialog").close(); return; }
    if (view().onClick) view().onClick(t, run());
  });

  app.addEventListener("input", (ev) => {
    if (ev.target.id === "scrub") seek(Number(ev.target.value));
  });

  app.addEventListener("change", (ev) => {
    if (ev.target.id === "speed") state.speed = Number(ev.target.value);
    if (ev.target.id === "chapter" && ev.target.value !== "") { seek(Number(ev.target.value)); ev.target.value = ""; }
  });

  document.addEventListener("keydown", (ev) => {
    if (state.view !== "watch" || !game() || !game().runs.length) return;
    if (document.querySelector("dialog[open]")) return;
    const tag = (ev.target.tagName || "").toLowerCase();
    if (tag === "select" || tag === "input" || tag === "textarea") return;
    const n = steps(run()).length;
    if (ev.key === "ArrowRight") { ev.preventDefault(); seek(Math.min(n - 1, state.cursor + 1)); }
    else if (ev.key === "ArrowLeft") { ev.preventDefault(); seek(Math.max(0, state.cursor - 1)); }
    else if (ev.key === " " && tag !== "button") { ev.preventDefault(); togglePlay(); }
  });

  // ---------------------------------------------------------------- theme
  const themeBtn = document.getElementById("theme-toggle");
  const savedTheme = MR.util.store.get("match-room-theme");
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  themeBtn.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    MR.util.store.set("match-room-theme", next);
  });

  // ---------------------------------------------------------------- boot
  if (!MR.manifest || !MR.manifest.games.length) {
    app.innerHTML = `<p class="feed__empty">No data found. Build it with <code>${esc(BUILD_HINT)}</code>, then reload.</p>`;
    return;
  }
  const s = site();
  if (s.title) document.querySelector(".wordmark").textContent = s.title;
  MR.home.bind(app);
  if (s.subtitle) document.querySelector(".masthead__sub").textContent = s.subtitle;
  window.addEventListener("hashchange", () => { const h = readHash(); state.cursor = 0; go(h); });
  go(readHash());
})();
