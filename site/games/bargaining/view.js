/* Bargaining: alternating offers over a pool of items with private values. */
(function () {
  "use strict";

  const { esc, title } = MatchRoom.util;
  const ui = MatchRoom.ui;

  function splitRow(label, items) {
    const tokens = Object.entries(items || {}).filter(([, n]) => n > 0).map(([k, n]) => `<span class="token">${n} ${esc(k)}${n > 1 ? "s" : ""}</span>`).join("");
    return `<div class="split__row"><span class="muted">${label}</span><span class="split__items">${tokens || '<span class="muted">nothing</span>'}</span></div>`;
  }

  const split = (o, style) => `<div class="split"${style ? ` style="${style}"` : ""}>${splitRow("keeps", o.take)}${splitRow("gives", o.leaves)}</div>`;

  MatchRoom.defineGame("bargaining", {
    speaker: (e) => e.offerer ?? e.actor,

    renderEvent(r, e) {
      switch (e.event) {
        case "offer": return ui.bubble(r, e.offerer, e.say, { event: e, meta: "offer", after: split(e, "margin-top:8px") });
        case "game_end": return ui.card(`<strong>${e.outcome === "AGREED" ? "Deal accepted." : title(e.outcome) + "."}</strong> Joint welfare ${esc(e.joint_welfare)} of a possible 20.`, r, e, true);
        default: return undefined;
      }
    },

    knowledge(r, p) {
      const v = (p.knowledge || {}).values;
      return v ? `<div class="brief">Private values: ${Object.entries(v).map(([i, n]) => `${esc(i)} ${n}`).join(" · ")}</div>` : "";
    },

    result(r) {
      const sub = Object.entries(r.result.rewards || {}).map(([s, v]) => `${ui.shortName(r, Number(s))}: ${v > 0 ? "+" : ""}${v}`).join(" · ");
      return { head: r.result.variables.outcome === "AGREED" ? "Deal" : "No deal", sub };
    },

    board(r, evs) {
      const pool = r.result.variables.pool || {};
      const vals = r.players.map((p) => (ui.privateVisibleTo(p.seat) ? (p.knowledge || {}).values : null));
      const rows = Object.entries(pool).map(([item, n]) => `<tr><td>${esc(item)}</td><td>${n}</td>${vals.map((v) => `<td>${v ? v[item] : "?"}</td>`).join("")}</tr>`).join("");
      const offers = evs.filter((e) => e.event === "offer");
      const o = offers[offers.length - 1];
      return [
        ui.endBanner(r),
        ui.block("Pool and private values", `
          <table class="pool-table"><thead><tr><th>Item</th><th>Pool</th>${r.players.map((p) => `<th>${esc(p.name)} value</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>
          <p class="small muted">Each player's values make the whole pool worth 10 to them.</p>`),
        ui.block("Standing offer", o
          ? `<div class="small" style="margin-bottom:4px"><strong>${esc(ui.shortName(r, o.offerer))}</strong> offers</div>${split(o)}`
          : `<p class="muted small">No offer yet.</p>`),
      ].join("");
    },
  });
})();
