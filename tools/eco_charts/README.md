# Eco chart replays

`site/eco/index.html` replays the recorded runs of the 50 economics games (`eco_*`) with charts: summary tiles,
game-specific charts on a shared step axis (prices, offers, contributions, choices against equilibrium and human
benchmarks, order books), a decision timeline with one lane per player, and for each decision the action and the
state the model saw. Each eco game's Match Room page links to it with its **Chart replay** button (`page_link` in
`content/games/<id>.yaml`).

The page draws precomputed data. The builders here compute it in Python from social-game-env's demo runs, because
each game's analysis (benchmarks, rationing, order books, re-derived payoffs) is easier to keep and check in Python.
The page itself has no dependencies and opens from `file://`.

## Files

- `builders/`: one `eco_<game>.py` per game; `matrix2x2.py` for the 13 repeated 2x2 games (`eco_sc1` to `eco_sc12`,
  `eco_ghp4`); `build_trading_en.py` for seven trading games (InvestorBench, two-trader liquidation, ABIDES, MarS,
  TwinMarket, MarketSim, QFinZero); `common.py` (loading and checks); `SCHEMA.md` (the data format).
- `to_site.py`: writes a builder's JSON as `site/eco/data/<game>.js`, without reasoning text.
- `check/smoke_one.js`: loads the page headless (Node.js with jsdom) with one game's data and works every control. It
  fails on script errors, empty fields, NaN or non-English text.

## Which runs

All runs were played by deepseek-flash with seed 42. For the 30 games of the second eco batch the page draws the
game's own run (`demo/<game>/<game>`); for the 20 earlier games it draws `demo/<game>/<game>_deepseek`, the second
run next to the site's gpt-6-luna recording. Nine games also have a run without chat history
(`demo/<game>/<game>_no_history`), shown as `<game>__no_history`.

## Rebuild

With social-game-env checked out next to this repo (each builder's docstring lists its arguments):

```bash
python3 tools/eco_charts/builders/eco_pirate_game.py ../social_game_env/demo/eco_pirate_game/eco_pirate_game /tmp/eco
python3 tools/eco_charts/builders/matrix2x2.py eco_sc3 ../social_game_env/demo/eco_sc3/eco_sc3_deepseek /tmp/eco
python3 tools/eco_charts/builders/build_trading_en.py ../social_game_env/demo /tmp/eco
python3 tools/eco_charts/builders/build_trading_en.py ../social_game_env/demo /tmp/eco --no-history
python3 tools/eco_charts/to_site.py /tmp/eco/*.json
```

For another game's run without chat history, run its builder on the `<game>_no_history` bundle into a separate
folder and save the result as `<game>__no_history.json`.

The published data was built from the runs together with the model-call logs saved at recording time (the optional
`CALLS` argument of the builders); those logs hold the reasoning, which this site leaves out, so they are not in
either repo. The seven trading games do not use them: rebuilt from the runs alone, their files match
`site/eco/data/` byte for byte. For the other games the charts come out the same, except a few details that only the
model's raw replies hold (a request that was rationed in the common-pool game, a first reply that was rejected in
public goods, the bidders' private values in the sealed-bid auction). Each step's observation is then rebuilt from
the event log, and the header line and footnote say that no reasoning was saved.

## Check

```bash
NODE_PATH=/path/to/node_modules node tools/eco_charts/check/smoke_one.js site/eco/index.html site/eco/data/eco_cournot.js
```
