/* Match Room home page: what the site is, how to read a match, and a
 * filterable gallery with one card per game. Everything here comes from
 * data/manifest.js, so no game has to load before someone picks it. */
(function () {
  "use strict";

  const MR = window.MatchRoom;
  const { esc, inline } = MR.util;

  const filter = { kind: "all", q: "" }; // survives leaving and returning to the home page

  // Search and kind chips earn their space only once the gallery is long.
  const SEARCH_FROM = 7;

  function matches(g) {
    if (filter.kind !== "all" && g.kind !== filter.kind) return false;
    const q = filter.q.trim().toLowerCase();
    return !q || [g.title, g.kind, g.tagline, g.players].join(" ").toLowerCase().includes(q);
  }

  function card(g) {
    const runs = g.runs || 0;
    return `
      <article class="game-card">
        <div class="eyebrow">${esc(g.kind)}</div>
        <h3 class="game-card__title"><a href="#${esc(g.id)}.rules">${esc(g.title)}</a></h3>
        <p class="game-card__tagline">${esc(g.tagline)}</p>
        <div class="game-card__foot">
          <span class="muted small">${esc(g.players)}</span>
          ${runs ? `<a class="game-card__watch" href="#${esc(g.id)}.watch">Watch ${runs > 1 ? `${runs} matches` : "the match"} →</a>` : `<span class="muted small">No recording yet</span>`}
        </div>
      </article>`;
  }

  function gridHtml(games) {
    const shown = games.filter(matches);
    if (!shown.length) return `<p class="feed__empty">No game matches “${esc(filter.q)}”.</p>`;
    return shown.map(card).join("");
  }

  function render(app) {
    const { site, games } = MR.manifest;
    const home = site.home || {};
    const kinds = [...new Set(games.map((g) => g.kind))];
    const tools = games.length >= SEARCH_FROM ? `
      <div class="toolbar gallery__tools">
        <input type="search" id="game-search" class="search" placeholder="Search games" aria-label="Search games" value="${esc(filter.q)}">
        <div class="lens" role="group" aria-label="Filter by kind">
          ${["all", ...kinds].map((k) => `<button class="chip-btn" type="button" data-kind="${esc(k)}" aria-pressed="${filter.kind === k}">${k === "all" ? "All" : esc(k)}</button>`).join("")}
        </div>
      </div>` : "";
    app.innerHTML = `
      <section class="hero">
        <div class="eyebrow">${esc(site.subtitle || "")}</div>
        <h1 class="hero__title">${esc(home.headline || site.title)}</h1>
        ${home.lead ? `<p class="tagline">${inline(home.lead)}</p>` : ""}
      </section>
      ${(home.steps || []).length ? `<ol class="how">${home.steps.map((s) => `<li><h2>${esc(s.title)}</h2><p>${inline(s.text)}</p></li>`).join("")}</ol>` : ""}
      <hr class="rule">
      <section class="gallery" aria-labelledby="gallery-title">
        <div class="gallery__head">
          <h2 id="gallery-title">Games <span class="muted">${games.length}</span></h2>
        </div>
        ${tools}
        <div class="gallery__grid" id="gallery-grid">${gridHtml(games)}</div>
        ${home.note ? `<p class="sources">${inline(home.note)}</p>` : ""}
      </section>`;
  }

  // Filtering redraws only the grid, so the search box keeps focus.
  function bind(app) {
    app.addEventListener("input", (ev) => {
      if (ev.target.id !== "game-search") return;
      filter.q = ev.target.value;
      document.getElementById("gallery-grid").innerHTML = gridHtml(MR.manifest.games);
    });
    app.addEventListener("click", (ev) => {
      const chip = ev.target.closest("[data-kind]");
      if (!chip) return;
      filter.kind = chip.dataset.kind;
      app.querySelectorAll("[data-kind]").forEach((b) => b.setAttribute("aria-pressed", String(b === chip)));
      document.getElementById("gallery-grid").innerHTML = gridHtml(MR.manifest.games);
    });
  }

  MR.home = { render, bind };
})();
