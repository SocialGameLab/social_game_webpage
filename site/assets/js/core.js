/* Match Room core: the global namespace, string helpers, the game registry and
 * the loader. Plain scripts, no modules, so the page also opens from file://.
 *
 * Load order (see index.html): core.js → ui.js → data/manifest.js → app.js.
 * Each game then arrives on demand as games/<id>/view.{css,js} + data/<id>.js. */
(function () {
  "use strict";

  const MR = (window.MatchRoom = window.MatchRoom || {});

  // ---------------------------------------------------------------- utils
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  MR.util = {
    esc,
    /** Escape, then turn `backticks` into <code>. Used for hand-written copy. */
    inline: (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>"),
    /** RED_TEAM → "Red team". */
    title: (s) => String(s ?? "").toLowerCase().replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase()),
    pct: (x) => `${Math.round(x * 100)}%`,
    store: {
      get(k) { try { return localStorage.getItem(k); } catch (_) { return null; } },
      set(k, v) { try { localStorage.setItem(k, v); } catch (_) { /* private mode */ } },
    },
  };

  // ---------------------------------------------------------------- state
  // One shared, mutable record of what is on screen. app.js changes it; the
  // helpers in ui.js and the game views only read it.
  MR.state = { gameId: null, runId: null, view: "rules", lens: "all", cursor: 0, playing: false, speed: 1, timer: null };

  // ---------------------------------------------------------------- registry
  // Three things arrive per game, each through its own call:
  //   setManifest(...)       data/manifest.js  site settings + placard list
  //   addData({...})         data/<id>.js      copy + recorded runs (generated)
  //   defineGame(id, view)   games/<id>/view.js how to draw that game's events
  MR.manifest = null;
  MR.data = {};
  MR.views = {};

  MR.setManifest = (m) => { MR.manifest = m; };
  MR.addData = (game) => { MR.data[game.id] = game; };

  /* A view may implement any of these; everything is optional.
   *   rolesArePublic      true unless roles are a secret (default true)
   *   renderEvent(r, e)   HTML for one log event, or undefined to use the default
   *   chapter(e)          label for the "Jump to" menu, or null
   *   speaker(e)          seat to highlight at the table (default e.actor)
   *   seatRole(r, p)      line under a seat's name, or undefined for the default
   *   knowledge(r, p)     private info under a seat, shown to its own lens
   *   board(r, evs)       right-hand panel, given the events this lens has seen
   *   result(r)           {head, sub} for the end banner, overriding the default
   *   onClick(t, r)       handle a click inside the room; return true if handled */
  MR.defineGame = (id, view) => { MR.views[id] = { rolesArePublic: true, ...view }; };

  // ---------------------------------------------------------------- loader
  function addScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = () => reject(new Error(`Could not load ${src}`));
      document.head.appendChild(s);
    });
  }

  function addStyle(href) {
    if (document.querySelector(`link[data-game-css="${href}"]`)) return;
    const l = document.createElement("link");
    l.rel = "stylesheet";
    l.href = href;
    l.dataset.gameCss = href;
    document.head.appendChild(l);
  }

  const pending = {};
  /** Resolve once both the view and the data for `id` are registered. A
   *  single-file build inlines everything, so nothing is fetched there. */
  MR.loadGame = (id) => {
    if (MR.views[id] && MR.data[id]) return Promise.resolve();
    if (!pending[id]) {
      addStyle(`games/${id}/view.css`);
      pending[id] = Promise.all([
        MR.views[id] ? null : addScript(`games/${id}/view.js`),
        MR.data[id] ? null : addScript(`data/${id}.js`),
      ]).catch((err) => { delete pending[id]; throw err; });
    }
    return pending[id];
  };
})();
