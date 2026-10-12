# Common Memory Bridge x MiniGrid

This repository is the research scaffold for testing whether a compact Common
Memory Bridge (CMB) can transfer history-dependent control information from a
DreamerV3-style RSSM-A to an independently trained RSSM-B.

The current implementation covers **Steps 0-1**: package/runtime contracts and
an official MiniGrid-Memory environment adapter. HiddenHistoryNav, RSSM
networks, bridge training, and experiment execution are intentionally not
implemented yet.

## Requirements

- Python 3.11 or newer
- PyTorch (CPU is sufficient for the Step 0 smoke tests)

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
```

The first command resolves a reproducible experiment ID and creates its log
directory. The second runs a 20-transition seeded MiniGrid-Memory smoke test
and writes `data/samples/minigrid_memory_seed42.json`. The third generates and
validates 100 strict HiddenHistoryNav counterfactual pairs.

## Layout

```text
cmb_minigrid/       Runtime utilities and shared contracts
configs/            YAML configuration presets
environments/       Reserved for MiniGrid environment adapters (Step 1+)
models/             Reserved for RSSM and policy code (Step 4+)
bridge/             Reserved for CMB and baseline bridges (Step 7+)
data/               Reserved for trajectory and handoff datasets (Step 3+)
trainers/           Reserved for model/bridge training (Step 5+)
evaluation/         Reserved for closed-loop evaluation (Step 6+)
scripts/            Reserved for later experiment CLIs
tests/              Step 0 contract and runtime tests
```

The binding data and time-index rules are documented in
[`docs/spec.md`](docs/spec.md). The running implementation record is
[`docs/implementation_log.md`](docs/implementation_log.md).
