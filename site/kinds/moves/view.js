/* Kind "moves": games recorded through social-game-env's `training` engine
 * (OpenSpiel games and puzzles). Each turn is one `move` event carrying the
 * mover's view of the state; a public move names who may read that view in
 * `view_audience`. A puzzle ends with a `verdict`, a game with a `final`
 * position. The board shows the latest view the lens may see (a chess
 * FEN is drawn as a board), the recent moves and the match facts.
 * Options (view_options):
 *   chapter_every  moves per "Jump to" entry (default 20)
 *   recent         moves listed on the board (default 12)
 *   wrap           columns for a view that arrives as one flat line */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;
  const state = MatchRoom.state;

  const PIECES = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };

  /* The eight rows of a FEN, or null. Hidden-board views add two marks: "?" is a
   * square the player cannot see (Dark Chess), and a space is one sensed empty
   * (Reconnaissance Blind Chess), so a row can't be split on whitespace. */
  function parseFen(text) {
    const parts = text.split("/");
    if (parts.length !== 8) return null;
    const rows = [];
    for (let i = 0; i < 8; i++) {
      const row = [];
      let at = 0;
      for (; at < parts[i].length && row.length < 8; at++) {
        const c = parts[i][at];
        if (/[1-8]/.test(c)) row.push(...Array(Number(c)).fill(""));
        else if (/[prnbqkPRNBQK? ]/.test(c)) row.push(c);
        else return null;
      }
      const rest = parts[i].slice(at);
      if (row.length !== 8 || (i < 7 ? rest : rest && !/^\s/.test(rest))) return null;
      rows.push(row);
    }
    return rows;
  }

  function fenBoard(rows) {
    const cells = rows.map((row, i) => row.map((c, j) => {
      const mark = c === "?" ? " mv-sq--fog" : c === " " ? " mv-sq--seen" : "";
      const piece = PIECES[c.toLowerCase()] ? `<span class="mv-pc mv-pc--${c === c.toUpperCase() ? "w" : "b"}">${PIECES[c.toLowerCase()]}</span>` : "";
      return `<span class="mv-sq${(i + j) % 2 ? " mv-sq--dark" : ""}${mark}">${piece}</span>`;
    }).join("")).join("");
    const marks = rows.flat();
    const key = [marks.includes("?") && "Grey squares are hidden.", marks.includes(" ") && "Outlined squares were sensed empty."].filter(Boolean).join(" ");
    return `<div class="mv-chess" role="img" aria-label="Chess position">${cells}</div>${key ? `<p class="mv-key small">${key}</p>` : ""}`;
  }

  function drawView(view, wrap) {
    let shown = String(view ?? "").replace(/\n+$/, "");
    if (wrap && !shown.includes("\n")) shown = shown.match(new RegExp(`.{1,${wrap}}`, "g")).join("\n");
    const rows = parseFen(shown.trim());
    const text = `<pre class="mv-view">${esc(shown) || "(empty)"}</pre>`;
    return rows ? `${fenBoard(rows)}<details class="mv-raw small"><summary>Raw state</summary>${text}</details>` : text;
  }

  // A public move can still carry a private view: the table saw the bid, not the hand.
  const canReadView = (e) => typeof e.view === "string" && (!e.view_audience || ui.canSee({ _visibility: "private", _audience: e.view_audience }, state.lens));
  const isPuzzle = (r) => r.prompt && r.prompt.kind === "answer";
  const verdictOf = (r) => r.events.find((e) => e.event === "verdict");
  // The log keeps the last reply, so an empty one means no reply ever arrived.
  const fallbackTag = (e) => (e.fallback ? ` <span class="mv-fb" title="${e.reply ? "The reply was rejected" : "No reply arrived"}, so the env's fallback move was played">fallback</span>` : "");
  const signed = (v) => `${v > 0 ? "+" : ""}${v}`;
  const simultaneous = (r) => (r.about || {}).dynamics === "simultaneous" && r.players.length > 1;

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
        // A top score is no win when players don't compete for a fixed total.
        const utility = (r.about || {}).utility;
        if (utility && !["zero_sum", "constant_sum"].includes(utility) && r.result && r.result.rewards) {
          return { head: "Final returns", sub: Object.entries(r.result.rewards).map(([s, v]) => `${ui.shortName(r, Number(s))}: ${signed(v)}`).join(" · ") };
        }
        return null;
      },

      board(r, evs) {
        const views = evs.filter(canReadView);
        const last = views[views.length - 1];
        const moves = evs.filter((e) => e.event === "move");
        const about = r.about || {};
        const facts = [
          about.long_name && ["Game", about.long_name],
          ["Env", r.env_id],
          about.information && ["Information", title(about.information)],
          about.dynamics && ["Moves", title(about.dynamics)],
          about.utility && ["Payoffs", title(about.utility)],
          simultaneous(r) ? ["Rounds", Math.floor(r.turns / r.players.length)] : ["Turns", r.turns],
          r.fallbacks ? ["Fallback moves", r.fallbacks] : null,
        ].filter(Boolean);
        const head = !last ? "" : last.event === "final" ? "Final position" : isPuzzle(r) ? "Puzzle" : `What ${esc(ui.shortName(r, last.actor))} saw before move ${esc(last.turn)}`;
        return [
          ui.endBanner(r),
          last ? ui.block(head, isPuzzle(r) ? `<pre class="mv-view">${esc(r.prompt.text)}</pre>` : drawView(last.view, opt.wrap)) : "",
          moves.length && !isPuzzle(r) ? ui.block(`Moves · ${moves.length} seen`, `<ol class="mv-list small">${moves.slice(-recent).map((m) => `
            <li value="${esc(m.turn)}">${ui.dot(m.actor)}${esc(ui.shortName(r, m.actor))} <code class="mv-move">${esc(m.move)}</code>${fallbackTag(m)}</li>`).join("")}</ol>`) : "",
          ui.block("Match", `<div class="kv-list small">${facts.map(([k, v]) => `<span class="kv"><span class="kv__k">${esc(k)}</span> ${esc(v)}</span>`).join("")}</div>`),
        ].join("");
      },
    };
  });
})();
