/* Jubensha QA: introductions, per-question discussion and whispers, private
 * answers, and the judge's verdict dialog. */
(function () {
  "use strict";

  const { esc, pct } = MatchRoom.util;
  const ui = MatchRoom.ui;
  const state = MatchRoom.state;

  let pick = null; // [question index, seat] selected in the verdict table

  function judgeMark(r, e) {
    if (state.lens !== "all" || !r.verdict) return "";
    const grade = ((r.verdict.grades || {})[String(e.actor)] || {})[String(e.q_index)];
    if (!grade) return "";
    return `<span class="team" style="color:var(--${grade.correct ? "ok" : "bad"})">judge: ${grade.correct ? "correct" : "wrong"}</span>`;
  }

  function verdictHtml(r) {
    const v = r.verdict;
    const answers = {};
    r.events.forEach((e) => { if (e.event === "answer") (answers[e.actor] = answers[e.actor] || {})[e.q_index] = e.text; });
    const head = r.players.map((p) => `<th scope="col">${ui.dot(p.seat, "display:inline-block;")} ${esc(p.name)}</th>`).join("");
    const rows = r.questions.map((qt, qi) => {
      const owner = v.owners[String(qi)];
      const cells = r.players.map((p) => {
        const g = (v.grades[String(p.seat)] || {})[String(qi)] || {};
        const pressed = pick && pick[0] === qi && pick[1] === p.seat;
        return `<td style="text-align:center"><button type="button" class="mark ${g.correct ? "mark--ok" : "mark--bad"}${owner === p.seat ? " mark--owner" : ""}" data-pick="${qi},${p.seat}" aria-pressed="${!!pressed}" aria-label="${esc(p.name)} on Q${qi + 1}: ${g.correct ? "correct" : "wrong"}">${g.correct ? "✓" : "✗"}</button></td>`;
      }).join("");
      return `<tr><td class="q"><strong>Q${qi + 1}.</strong> ${esc(qt)}<div class="muted small">${pct(v.per_question_accuracy[String(qi)])} correct</div></td>${cells}</tr>`;
    }).join("");
    const accRow = r.players.map((p) => `<td style="text-align:center"><strong>${pct(v.per_agent_accuracy[String(p.seat)])}</strong></td>`).join("");
    const defRow = r.players.map((p) => { const d = v.per_agent_defense[String(p.seat)]; return `<td style="text-align:center" class="muted">${d == null ? "–" : pct(d)}</td>`; }).join("");
    let detail = `<p class="muted small" style="margin-top:14px">Select a mark to compare that answer with the key.</p>`;
    if (pick) {
      const [qi, seat] = pick;
      const g = (v.grades[String(seat)] || {})[String(qi)] || {};
      detail = `
        <div class="verdict-detail">
          <div class="panel"><div class="board-label">Answer key · Q${qi + 1}</div>${esc(v.answer_key[qi])}</div>
          <div class="panel"><div class="board-label">${esc(ui.shortName(r, seat))}'s answer · judged ${g.correct ? "correct" : "wrong"}</div>${esc((answers[seat] || {})[qi] || "(no answer)")}<div class="callout">Judge: ${esc(g.reason || "")}</div></div>
        </div>`;
    }
    const fallbacks = Object.keys(v.judge_fallbacks || {}).length;
    return `
      <p class="small">Overall accuracy <strong>${pct(v.overall_accuracy)}</strong>. ${fallbacks ? `${fallbacks} question(s) fell back to zero grades after judge failures.` : "Every question was graded by the judge."}</p>
      <div style="overflow-x:auto"><table class="verdict-table">
        <thead><tr><th scope="col" style="text-align:left">Question</th>${head}</tr></thead>
        <tbody>${rows}
          <tr><td class="q"><strong>Accuracy</strong></td>${accRow}</tr>
          <tr><td class="q"><strong>Defense</strong><div class="muted small">Share of others who missed questions about this seat</div></td>${defRow}</tr>
        </tbody></table></div>
      <div class="callout">A gold outline marks the seat the engine told to defend that question. Ownership comes from matching character names inside the answer key, so it can point at the wrong seat when the key mentions a bystander.</div>
      ${detail}`;
  }

  const openVerdict = (r) => ui.openDialog(`Verdict · ${r.label || r.id}`, verdictHtml(r));

  MatchRoom.defineGame("jubensha_qa", {
    chapter(e) {
      if (e.event === "question") return `Q${e.q_index + 1}: ${e.question}`;
      if (e.event === "intro") return "Introductions";
      return null;
    },

    renderEvent(r, e) {
      const who = (s) => esc(ui.shortName(r, s));
      switch (e.event) {
        case "briefing": return ui.card(`Background story and <strong>${r.briefing ? r.briefing.clues.length : "all"} clue cards</strong> handed to everyone. Open the case file on the right to read them.`, r, e);
        case "intro": return ui.bubble(r, e.actor, e.text, { event: e, meta: "introduction" });
        case "question": return ui.chapterHead(`Question ${e.q_index + 1} of ${r.questions.length}`, esc(e.question));
        case "partner_choice": return ui.sys("pairing", `<strong>${who(e.actor)}</strong> asks <strong>${who(e.target)}</strong> for a private chat.`, r, e);
        case "whisper": return ui.bubble(r, e.actor, e.text, { event: e, cls: "bubble--whisper", meta: `whispers to ${who(e.partner)}` });
        case "answer": return ui.bubble(r, e.actor, e.text, { event: e, cls: "bubble--answer", meta: `answers Q${e.q_index + 1}`, metaHtml: judgeMark(r, e) });
        default: return undefined;
      }
    },

    seatRole(r, p) {
      if (!r.verdict || !ui.atEnd(r)) return `<span class="muted">character</span>`;
      return `scored <strong>${pct(r.verdict.per_agent_accuracy[String(p.seat)])}</strong>`;
    },

    knowledge(r, p) {
      const k = p.knowledge || {};
      const items = (xs, f) => (xs || []).map((x) => `<li>${f(x)}</li>`).join("");
      const sec = items(k.key_secrets, esc);
      const mot = items(k.motives, esc);
      const prot = items(k.protected, (q) => `Q${q.index}: ${esc(q.question)}`);
      return `
        <details class="brief">
          <summary>Private brief</summary>
          ${sec ? `<h4>Secrets</h4><ul>${sec}</ul>` : ""}
          ${mot ? `<h4>Motives</h4><ul>${mot}</ul>` : ""}
          <h4>Told to defend</h4>${prot ? `<ul>${prot}</ul>` : `<p class="muted" style="margin:0">No questions.</p>`}
        </details>`;
    },

    board(r, evs) {
      let q = -1;
      evs.forEach((e) => { if (e.event === "question") q = e.q_index; });
      const done = ui.atEnd(r);
      const pips = r.questions.map((_, i) => `<span class="q-pip${done || i < q ? " q-pip--done" : i === q ? " q-pip--active" : ""}" title="${esc(r.questions[i])}">${i + 1}</span>`).join("");
      const b = r.briefing || { background: [], clues: [] };
      return [
        done ? ui.banner("Judged", `Overall accuracy ${pct(r.verdict.overall_accuracy)}`) : "",
        ui.block("Questions", `
          <div class="q-track">${pips}</div>
          <div class="q-now">${q >= 0 ? `Q${q + 1}. ${esc(r.questions[q])}` : '<span class="muted">Introductions first.</span>'}</div>`),
        ui.block("", `<details class="drawer"><summary>Case file · background</summary>
          ${b.background.length ? b.background.map((t) => `<p class="small">${esc(t)}</p>`).join("") : '<p class="small muted">This script has no background text.</p>'}
        </details>`),
        ui.block("", `<details class="drawer"><summary>Case file · ${b.clues.length} clue cards</summary>
          <ul class="clue-list">${b.clues.map((c) => `<li><span class="clue-loc">${esc(c.location || "Location unknown")}</span>${esc(c.content)}</li>`).join("")}</ul>
        </details>`),
        `<button class="cta verdict-btn" type="button" data-verdict>${done ? "Open the verdict" : "Open the verdict (spoilers)"}</button>`,
      ].join("");
    },

    onClick(t, r) {
      if (t.closest("[data-verdict]")) { pick = null; openVerdict(r); return true; }
      const cell = t.closest("[data-pick]");
      if (cell) { pick = cell.dataset.pick.split(",").map(Number); openVerdict(r); return true; }
      return false;
    },
  });
})();
