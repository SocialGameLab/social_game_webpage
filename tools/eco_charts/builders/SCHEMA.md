# Replay page data format (one JSON per game)

Top level: `id` (game id), `name` (short English name), `title` (one-line headline of this game), `lede` (2-3 plain sentences:
what the players do and how they score), `eyebrow` (set by common.write), `seatNames` (one label per player seat, e.g.
["Buyer", "Seller"]; default ["Model"]), `tiles` (2-4 summary tiles `{k, v, s}`: label, big value, small note; optional
`spark`: list of numbers for a sparkline in the first tile), `foot` (one paragraph: data provenance, definitions),
`x` (`{n, labels}`: the shared horizontal axis, e.g. rounds; every chart and step uses these indices 0..n-1),
`charts` (list, below), `kinds` (`{kind_id: [label, color]}` for action types), `steps` (one per model decision, in time order),
optional `feed` (a transcript panel) with `feedTitle`, `feedSub`, optional `book` (order books, trading games only).

Colors are tokens: c1 blue, c2 orange, c3 green, c4 yellow, ink (text color), hold (gray), muted (light gray).
Number formats (`fmt`): price, price2, price4 (2/2/4 decimals), pct (signed %, 2 dp), pct0 (%, 1 dp), pct3, int, money ($, signed),
money0, bps.

Chart types (each chart: `title`, `sub`, `height` in px ~170-280):
- Line/area: `lines: [{name, color, v: [n values or null], dash?, area?, step?, thin?, dots?, noline?, end? (label at the right end)}]`.
- Bars (grouped per x): `bars: [{name, color, v}]` plus `lines: []`.
- Candles: `candles: [[o,h,l,c] or null per x]`.
- `markers: [{x, y, kind: "buy"|"sell"}]` (triangles), `flags: [{x, text}]` (dashed yellow event lines with hover text),
  `band: {v: [+1/-1/0 per x], pos, neg}` (a colored strip under the plot), `zero: true` (zero line).
- Grid (a player x step matrix, for choices, votes, contributions): `grid: {rows: [names], cells: [[{c, t, tip, o} or null per x] per row],
  legend: [[label, color]], rowHeight?}`; `t` is a short label drawn in the cell when it fits, `tip` the hover text.

Steps: `{seat, x, head, kind, act, state, thinking}` — `head` a short English title ("Round 3 · offers $420"), `act` the raw action
JSON string, `state` what the model saw (prompt state part), `thinking` the model's reasoning ("" if not recorded).
Feed entries: `{x, step?, who, kind?, text}` (e.g. each offer with its message); `kind` must be a key of `kinds` if given.

Rules: English only (common.check rejects CJK), no NaN, every series has exactly x.n values, at least one value per series.
