# CLAUDE.md

Match Room: a static GitHub Pages site that explains each social-game-env game
and replays recorded LLM-agent matches. README.md covers usage; this file is
the contract for keeping the structure consistent as it grows.

## Ground rules

- **No build step, no framework, no modules.** Plain `<script>` files sharing
  the `window.MatchRoom` namespace, so `site/index.html` also opens from
  `file://`. Don't introduce npm, bundlers, ES modules or `fetch()` for data.
- **Only `site/` is published.** Everything else (content, recordings, tools)
  is source.
- **`site/data/` is generated.** Never hand-edit it. Change `content/` or
  `recordings/`, then run `tools/build_data.py`. Commit the regenerated data
  with the change that caused it, because Pages serves the committed files.
- **Python tools run in social-game-env's environment**
  (`uv run --project ../social_game_env python tools/...`). They import its
  engine; don't vendor or copy engine code into this repo.

## Where things go

| Change | Edit | Not |
|---|---|---|
| Rules copy, taglines, "watch for" notes | `content/games/<id>.yaml` | JS strings |
| Site title, home-page copy, game order, source-repo link | `content/site.yaml` | `index.html` / `home.js` |
| How one game's events, seats and board look | `site/games/<id>/view.js` + `view.css` | `app.js` / `ui.js` |
| Something every game needs | `site/assets/js/ui.js` (a helper) or `app.js` (page behavior) | copy-paste across views |
| Colors | tokens in `site/assets/css/theme.css`, light **and** both dark blocks | hex values in game CSS |
| A game that `runners.play` can't drive | a new entry in `ENGINES` (`tools/build_data.py`) and `_command` (`tools/record_runs.py`) | `if game_id == ...` branches |

Navigation must scale with the catalog. The masthead holds one grouped game
menu, never a tab per game. The home gallery is the place to browse, and it
gains search and kind filters at 7 or more games. Don't add per-game UI to
the masthead or the home page; a game's card comes from the `title`, `kind`,
`players` and `tagline` in its content file.

The core (`core.js`, `ui.js`, `home.js`, `app.js`) must stay game-agnostic: no game ids,
role names or event names specific to one game. If a view needs something the
core doesn't offer, add an optional hook to `MatchRoom.defineGame`, document
it in the hook list in `core.js` and the README table, and give it a default
that leaves the existing games unchanged.

## Adding a game

1. `python3 tools/new_game.py <id> "<Title>"`. The id matches the game's name in
   social-game-env.
2. Fill in `content/games/<id>.yaml`. The field reference is
   `templates/game/content.yaml`.
3. Record and build: `tools/record_runs.py <id>`, then `tools/build_data.py`.
   Recording calls the LLM and costs money, so ask before running it.
4. Draw the game in `view.js`. Unhandled events render as raw JSON, so work
   through them until none are left.

Keep the new game consistent with the others:
- Build the feed from `ui.chapterHead` (section starts, which are also
  `chapter()` targets), `ui.sys` (bookkeeping), `ui.card` (outcomes, with
  `big` for decisive moments) and `ui.bubble` (a seat speaking).
- Build the board from `ui.block(label, html)`, with `ui.endBanner(r)` first.
- Respect the lens. Private info goes through `knowledge()` (core shows it only
  to "Everything" and the owning seat), and `board()` is given only the events
  the lens has seen. Never show hidden information on the board or in seats.
- Prefix game-specific CSS classes with a short tag (`cn-`, `q-`, …).

If you change the templates or the hook API, update `templates/game/` in the
same change so new games start from the current pattern.

## Checking a change

- Serve with `python3 -m http.server 8765 -d site` and step through every game
  you touched, across the "Everything", "Public table" and one seat lens, start
  to end.
- After refactors of shared code, the rendered `#feed`, `#seats` and `#board`
  should match before and after for every game, run, lens and position.
  Compare them headlessly (for example with jsdom) rather than by eye.
- `node --check` every JS file you edited.
- `python3 tools/check_site.py` must pass. CI runs it before every deploy and
  will block a game that is missing its content, view or data file.

## Style

Match the surrounding code: compact string-template rendering, `esc()` on
every interpolated value from data, short comments that say why. Page copy is
plain and short. Each list item in content files is one sentence.
