/* Kind "moves": games recorded through social-game-env's `training` engine
 * (OpenSpiel games and puzzles). Each turn is one `move` event carrying the
 * mover's view of the state; a puzzle ends with a `verdict`, a game with a
 * `final` position. The board shows the latest view the lens may see (a chess
 * FEN is drawn as a board), the recent moves and the match facts.
 * Options (view_options):
 *   chapter_every  moves per "Jump to" entry (default 20)
 *   recent         moves listed on the board (default 12) */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;

  const FEN = /^((?:[prnbqkPRNBQK1-8]+\/){7}[prnbqkPRNBQK1-8]+)(?:\s|$)/;
  const PIECES = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };

  function fenBoard(fen) {
    const rows = fen.split("/").map((row) => row.replace(/\d/g, (n) => " ".repeat(Number(n))).split(""));
    const cells = rows.map((row, i) => row.map((c, j) => {
      const piece = c === " " ? "" : `<span class="mv-pc mv-pc--${c === c.toUpperCase() ? "w" : "b"}">${PIECES[c.toLowerCase()]}</span>`;
      return `<span class="mv-sq${(i + j) % 2 ? " mv-sq--dark" : ""}">${piece}</span>`;
    }).join("")).join("");
    return `<div class="mv-chess" role="img" aria-label="Chess position ${esc(fen)}">${cells}</div>`;
  }

  function drawView(view) {
    const fen = FEN.exec(String(view || "").trim());
    const text = `<pre class="mv-view">${esc(String(view ?? "").replace(/\n+$/, "")) || "(empty)"}</pre>`;
    return fen ? `${fenBoard(fen[1])}<details class="mv-raw small"><summary>Raw state</summary>${text}</details>` : text;
  }

  const isPuzzle = (r) => r.prompt && r.prompt.kind === "answer";
  const verdictOf = (r) => r.events.find((e) => e.event === "verdict");
  const fallbackTag = (e) => (e.fallback ? ` <span class="mv-fb" title="The reply did not parse, so the env's fallback move was played">fallback</span>` : "");

  MatchRoom.defineKind("moves", (opt) => {
    const every = opt.chapter_every || 20;
    const recent = opt.recent || 12;

    return {
      chapter(e) {
        if (e.event === "move") return e.turn === 1 ? "Start" : e.turn % every === 1 ? `Move ${e.turn}` : null;
        if (e.event === "verdict") return "Verdict";
        if (e.event === "final") return "End";
        return null;
      },

      renderEvent(r, e) {
        if (e.event === "setup") return ui.sys("setup", `${r.players.length} player${r.players.length > 1 ? "s" : ""} · ${esc(r.env_id)}`, r, e);
        if (e.event === "move" && isPuzzle(r)) return ui.bubble(r, e.actor, e.move, { event: e, meta: "Answer", metaHtml: fallbackTag(e) });
        if (e.event === "move") return ui.sys(`move ${e.turn}`, `<strong>${esc(ui.shortName(r, e.actor))}</strong> <code class="mv-move">${esc(e.move)}</code>${fallbackTag(e)}`, r, e);
        if (e.event === "verdict") return ui.card(`<strong>${e.correct ? "Solved." : "Not solved."}</strong> ${esc(e.detail)} <span class="kv"><span class="kv__k">Reward</span> ${esc(e.reward)}</span>`, r, e, true);
        if (e.event === "final") return ui.card(`<strong>${r.truncated ? `Stopped after ${esc(r.turns)} moves.` : `Game over after ${esc(r.turns)} move${r.turns === 1 ? "" : "s"}.`}</strong>`, r, e, true);
        if (e.event === "game_over") return ui.sys("game over", ui.fields(r, e), r, e);
        return ui.autoEvent(r, e);
      },

      result(r) {
        const v = verdictOf(r);
        if (v) return { head: v.correct ? "Solved" : "Not solved", sub: v.detail };
        if (r.truncated) return { head: `Stopped after ${r.turns} moves`, sub: "The safety limit ended the game early." };
        return null;
      },

      board(r, evs) {
        const views = evs.filter((e) => typeof e.view === "string");
        const last = views[views.length - 1];
        const moves = evs.filter((e) => e.event === "move");
        const about = r.about || {};
        const facts = [
          about.long_name && ["Game", about.long_name],
          ["Env", r.env_id],
          about.information && ["Information", title(about.information)],
          about.dynamics && ["Moves", title(about.dynamics)],
          about.utility && ["Payoffs", title(about.utility)],
          ["Turns", r.turns],
          r.fallbacks ? ["Fallback moves", r.fallbacks] : null,
        ].filter(Boolean);
        const head = !last ? "" : last.event === "final" ? "Final position" : isPuzzle(r) ? "Puzzle" : `What ${esc(ui.shortName(r, last.actor))} saw before move ${esc(last.turn)}`;
        return [
          ui.endBanner(r),
          last ? ui.block(head, isPuzzle(r) ? `<pre class="mv-view">${esc(r.prompt.text)}</pre>` : drawView(last.view)) : "",
          moves.length && !isPuzzle(r) ? ui.block(`Moves · ${moves.length} seen`, `<ol class="mv-list small">${moves.slice(-recent).map((m) => `
            <li value="${esc(m.turn)}">${ui.dot(m.actor)}${esc(ui.shortName(r, m.actor))} <code class="mv-move">${esc(m.move)}</code>${fallbackTag(m)}</li>`).join("")}</ol>`) : "",
          ui.block("Match", `<div class="kv-list small">${facts.map(([k, v]) => `<span class="kv"><span class="kv__k">${esc(k)}</span> ${esc(v)}</span>`).join("")}</div>`),
        ].join("");
      },
    };
  });
})();
