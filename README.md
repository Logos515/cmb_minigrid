# Common Memory Bridge x MiniGrid

This repository is the research scaffold for testing whether a compact Common
Memory Bridge (CMB) can transfer history-dependent control information from a
DreamerV3-style RSSM-A to an independently trained RSSM-B.

The current implementation covers **Steps 0-4**: package/runtime contracts,
the official MiniGrid-Memory adapter, strict HiddenHistoryNav counterfactual
pairs, a deployment-safe trajectory dataset pipeline, and a lightweight PyTorch
categorical RSSM with CPU acceptance tests. Formal RSSM/policy/bridge training
and experiment evaluation remain future steps.

## Requirements

- Python 3.11 or newer
- PyTorch (CPU is sufficient for the acceptance tests through Step 4)

Install the package and development tools in an isolated environment:

```bash
python -m pip install -e '.[dev]'
```

## Reproduction commands

```bash
pytest -q
python -m cmb_minigrid.cli --help
python -m cmb_minigrid.cli --config configs/default.yaml --seed 42 --device cpu
python -m cmb_minigrid.scripts.smoke_env --seed 42
python -m cmb_minigrid.scripts.smoke_hidden_history --count 100 --seed 42
python -m cmb_minigrid.scripts.collect_data --config configs/data.yaml --force
python -m cmb_minigrid.scripts.validate_data
python -m cmb_minigrid.scripts.stats_data
python -m cmb_minigrid.scripts.smoke_rssm --config configs/rssm_a.yaml --device cpu --sequence-length 64
python -m cmb_minigrid.scripts.smoke_rssm --config configs/rssm_b.yaml --device cpu --sequence-length 64
```

The first command resolves a reproducible experiment ID and creates its log
directory. The environment and pair commands run seeded smoke checks. The
Step 3 commands collect JSONL trajectories, write a grouped split manifest,
validate every public transition and privileged snapshot, restore the handoff
snapshot in a fresh environment, and print dataset statistics.

The default data configuration collects five real map families, ten pairs per
family, and both expert and exploration episodes (200 records total). Override
the scale with `--pairs-per-family 20` or choose a subset with
`--map-families hidden_history_nav_v1 hidden_history_nav_v2`. The legacy
`--pairs` flag remains an alias for the per-family count. With five families,
whole-family train/val/test allocation is 3/1/1; snapshots record the family
for exact restoration. These are five fixed geometry templates with seeded
exploration, not fifty independently randomized layouts.

For this repository's conda environment, use
`/data2/xuyihao/envs/cmb_minigrid/bin/python` in place of `python`, including
`python -m pytest -q`. The Step 4 CLI reads only the train split, checks sampled
losses and gradients, then runs 40 diagnostic updates on one fixed batch using
deterministic categorical modes. `--overfit-steps 0` disables these updates.
This checks that the model can memorize a batch; it does not train a usable
world model or establish history retention. Configuration, seed, dimensions,
initial/final loss and gradient norm are saved under `runs/*/smoke_rssm.json`.

`cmb_minigrid.models.RSSM` provides `initial(batch)`,
`observe_step(prev_state, prev_action, obs)`, `imagine_step(state, action)`,
`features(state)`, plus `observe`, `decode`, and masked `loss` helpers. Inputs
are the public tensor observation dictionary from `collate_trajectories`;
expert actions and privileged fields are excluded. See `docs/spec.md` for
initialization, tensor layout, prediction targets and KL conventions.

## Layout

```text
cmb_minigrid/       Runtime, environments, data, RSSM models, and smoke CLIs
configs/            YAML configuration presets
environments/       Reserved for MiniGrid environment adapters (Step 1+)
models/             Reserved for RSSM and policy code (Step 4+)
bridge/             Reserved for CMB and baseline bridges (Step 7+)
data/               Trajectory JSONL, split manifests, and smoke samples
trainers/           Reserved for model/bridge training (Step 5+)
evaluation/         Reserved for closed-loop evaluation (Step 6+)
scripts/            Reserved for later experiment CLIs
tests/              Step 0 contract and runtime tests
```

The binding data and time-index rules are documented in
[`docs/spec.md`](docs/spec.md). The running implementation record is
[`docs/implementation_log.md`](docs/implementation_log.md).
