/* Legal Court: opening statements, argument rounds with evidence reveals,
 * rebuttals and citations, the jury tally after each round, the verdict, and
 * a sealed-outcome panel. */
(function () {
  "use strict";

  const { esc, title, pct } = MatchRoom.util;
  const ui = MatchRoom.ui;
  const state = MatchRoom.state;

  const KIND = { statement: "statement", rebuttal: "rebuts", citation: "cites law" };
  const side = (r, seat) => (ui.player(r, seat).team === "PLAINTIFF" ? "plaintiff" : "defense");
  const sideName = (r, team) => { const p = r.players.find((x) => x.team === team); return p ? p.name : title(team); };

  /** Statements come in two blocks: the first two are openings, the last two closings. */
  function statementIndex(r, e) {
    const all = r.events.filter((x) => x.event === "argument" && x.kind === "statement");
    return all.indexOf(e);
  }
  const statementPhase = (r, e) => (statementIndex(r, e) < 2 ? "opening" : "closing");

  /** Events so far, folded into what the board shows. */
  function tally(evs) {
    const polls = [], revealed = {};
    evs.forEach((e) => {
      if (e.event === "jury_poll") polls.push(e);
      if (e.event === "evidence") (revealed[e.actor] = revealed[e.actor] || []).push(e.index);
    });
    const verdict = evs.find((e) => e.event === "verdict");
    return { polls, revealed, verdict };
  }

  function verdictHtml(r) {
    const v = r.verdict;
    const real = v.ground_truth, got = v.verdict;
    const align = v.citation_alignment || {};
    const rewards = v.advocate_rewards || {};
    const sideRow = (team) => {
      const key = team.toLowerCase();
      const rw = rewards[team] || {};
      const cites = (align[`${key}_aligned`] || []);
      return `<tr><td><span class="team team--${team}">${esc(sideName(r, team))}</span></td>
        <td>${cites.length ? cites.map((c) => `<code>${esc(c)}</code>`).join(", ") : "<span class=\"muted\">none</span>"}</td>
        <td style="text-align:center">${rw.aligned != null ? `${rw.aligned} / ${rw.governing_total}` : "–"}</td>
        <td style="text-align:center">${rw.verdict != null ? (rw.verdict > 0 ? "+" : "") + rw.verdict : "–"}</td></tr>`;
    };
    const jurorRows = Object.entries(v.jurors || {}).map(([seat, j]) => {
      const last = (j.votes || []).slice(-1)[0] || {};
      const score = (s) => s ? ["legal_grounding", "evidence", "responsiveness", "factual_consistency", "logical_rigor"].map((k) => s[k] ?? "–").join(" · ") : "–";
      return `<div class="lc-juror"><div class="board-label">${esc(ui.shortName(r, Number(seat)))}${j.model ? ` · <span class="muted">${esc(j.model)}</span>` : ""}</div>
        <div class="small">Final vote: <strong>${last.for_plaintiff == null ? "–" : last.for_plaintiff ? esc(sideName(r, "PLAINTIFF")) : esc(sideName(r, "DEFENSE"))}</strong>.
        Rubric (grounding · evidence · responsiveness · consistency · logic): ${esc(sideName(r, "PLAINTIFF"))} ${score(last.plaintiff)}; ${esc(sideName(r, "DEFENSE"))} ${score(last.defense)}.</div>
        ${last.reason ? `<div class="callout">${esc(last.reason)}</div>` : ""}</div>`;
    }).join("");
    return `
      <p class="small">The jury found for <strong>${esc(sideName(r, got))}</strong>. The real court found for <strong>${esc(sideName(r, real))}</strong>:
        <strong style="color:var(--${v.correct ? "ok" : "bad"})">${v.correct ? "the jury matched the real outcome" : "the jury got it wrong"}</strong>.</p>
      ${v.court_opinion ? `<details class="drawer"><summary>What the real court said (sealed during play)</summary><p class="small lc-pre">${esc(v.court_opinion)}</p></details>` : ""}
      <div class="board-label" style="margin-top:14px">Governing law (shown to the jury only at the verdict)</div>
      <p class="small">${(v.governing_law || []).length ? v.governing_law.map((c) => `<code>${esc(c)}</code>`).join(", ") : "<span class=\"muted\">none recorded</span>"}</p>
      <div class="board-label" style="margin-top:14px">Citations that matched the governing law</div>
      <div style="overflow-x:auto"><table class="lc-table">
        <thead><tr><th scope="col">Side</th><th scope="col">Matched citations</th><th scope="col">Aligned</th><th scope="col">Verdict reward</th></tr></thead>
        <tbody>${sideRow("PLAINTIFF")}${sideRow("DEFENSE")}</tbody></table></div>
      <div class="board-label" style="margin-top:14px">Jurors</div>
      ${jurorRows || '<p class="small muted">No juror reasoning was recorded.</p>'}`;
  }

  const openVerdict = (r) => ui.openDialog(`Verdict · ${r.label || r.id}`, verdictHtml(r));

  MatchRoom.defineGame("legal_court", {
    rolesArePublic: true,

    chapter(e) {
      if (e.event === "jury_poll") return `Round ${e.round + 1} · jury poll`;
      if (e.event === "verdict") return "Verdict";
      return null;
    },

    renderEvent(r, e) {
      const who = (s) => esc(ui.shortName(r, s));
      switch (e.event) {
        case "argument": {
          if (e.kind === "statement") {
            const phase = statementPhase(r, e);
            const first = statementIndex(r, e) % 2 === 0;
            return (first ? ui.chapterHead(`${title(phase)} statements`, "") : "") + ui.bubble(r, e.actor, e.text, { event: e, meta: `${phase} statement` });
          }
          return ui.bubble(r, e.actor, e.text, { event: e, cls: e.kind === "citation" ? "bubble--answer" : "", meta: KIND[e.kind] || title(e.kind) });
        }
        case "evidence": return ui.card(`<span class="lc-ev-tag team team--${esc(ui.player(r, e.actor).team)}">${who(e.actor)} presents evidence</span><div class="lc-ev-text">${esc(e.text)}</div>`, r, e);
        case "jury_poll": {
          const p = e.for_plaintiff, d = e.for_defense;
          const lean = p === d ? "The jury is split." : `The jury leans <strong>${esc(sideName(r, p > d ? "PLAINTIFF" : "DEFENSE"))}</strong>.`;
          return ui.card(`<strong>Round ${e.round + 1} of ${r.case.rounds} · jury poll ${p}–${d}.</strong> ${lean}`, r, e);
        }
        case "verdict": return ui.card(`<strong>Verdict: ${esc(sideName(r, e.verdict))}</strong>, ${e.votes_for_plaintiff}–${e.votes_for_defense}. The real outcome is in the verdict panel on the right.`, r, e, true);
        default: return undefined;
      }
    },

    seatRole(r, p) {
      if (p.team === "COURT") return `<span class="team team--COURT">juror</span>`;
      return `<span class="team team--${esc(p.team)}">${esc(title(p.role))}</span>`;
    },

    knowledge(r, p) {
      const items = (p.knowledge || {}).evidence || [];
      if (!items.length) return "";
      const { revealed } = tally(ui.visibleEvents(r, state.cursor));
      const mine = revealed[p.seat] || [];
      return `<details class="brief"><summary>Evidence held (${items.length - mine.length} unrevealed)</summary>
        <ol class="lc-held">${items.map((t, i) => `<li class="${mine.includes(i) ? "lc-held--played" : ""}">${esc(t)}${mine.includes(i) ? ' <span class="muted small">(revealed)</span>' : ""}</li>`).join("")}</ol></details>`;
    },

    result(r) {
      if (!r.verdict) return null;
      return { head: `Jury verdict: ${sideName(r, r.verdict.verdict)}`, sub: "Open the verdict panel to compare with the real court." };
    },

    board(r, evs) {
      const { polls, revealed, verdict } = tally(evs);
      const c = r.case || {};
      const done = ui.atEnd(r);
      const pips = Array.from({ length: c.rounds || 0 }, (_, i) => {
        const poll = polls[i];
        if (!poll) return `<span class="lc-pip${i === polls.length && !verdict ? " lc-pip--active" : ""}" title="Round ${i + 1}">${i + 1}</span>`;
        const cls = poll.for_plaintiff > poll.for_defense ? "lc-pip--p" : poll.for_plaintiff < poll.for_defense ? "lc-pip--d" : "lc-pip--tie";
        return `<span class="lc-pip ${cls}" title="Round ${i + 1}: ${poll.for_plaintiff}–${poll.for_defense}">${poll.for_plaintiff}–${poll.for_defense}</span>`;
      }).join("");
      const count = (team) => { const p = r.players.find((x) => x.team === team); return p ? (revealed[p.seat] || []).length : 0; };
      const held = (team) => { const p = r.players.find((x) => x.team === team); return p ? ((p.knowledge || {}).evidence || []).length : 0; };
      return [
        ui.endBanner(r),
        ui.block("Case", `<div class="lc-case-name">${esc(c.name || "")}</div><div class="small muted">${esc(c.cause || "")}</div>`),
        ui.block("Jury tally by round", `<div class="lc-track">${pips}</div>
          <div class="small muted" style="margin-top:6px"><span class="team team--PLAINTIFF">■</span> ${esc(sideName(r, "PLAINTIFF"))} ahead · <span class="team team--DEFENSE">■</span> ${esc(sideName(r, "DEFENSE"))} ahead</div>`),
        ui.block("Evidence on the record", `<div class="small"><span class="team team--PLAINTIFF">${esc(sideName(r, "PLAINTIFF"))}</span> ${count("PLAINTIFF")} of ${held("PLAINTIFF")} · <span class="team team--DEFENSE">${esc(sideName(r, "DEFENSE"))}</span> ${count("DEFENSE")} of ${held("DEFENSE")}</div>`),
        ui.block("", `<details class="drawer"><summary>Case file · the claim</summary><p class="small lc-pre">${esc(c.claim || "")}</p></details>`),
        ui.block("", `<details class="drawer"><summary>Case file · the facts</summary><p class="small lc-pre">${esc(c.facts || "")}</p></details>`),
        `<button class="cta verdict-btn" type="button" data-verdict>${done ? "Open the verdict" : "Open the verdict (spoilers)"}</button>`,
      ].join("");
    },

    onClick(t, r) {
      if (t.closest("[data-verdict]")) { openVerdict(r); return true; }
      return false;
    },
  });
})();
