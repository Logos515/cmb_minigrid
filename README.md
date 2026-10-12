# Common Memory Bridge x MiniGrid

This repository is the research scaffold for testing whether a compact Common
Memory Bridge (CMB) can transfer history-dependent control information from a
DreamerV3-style RSSM-A to an independently trained RSSM-B.

The current implementation is **Step 0 only**. It defines the package layout,
configuration/runtime helpers, and immutable data contracts. Environment
generation, RSSM networks, bridge training, and experiment execution are
intentionally not implemented yet.

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
```

The last command resolves a reproducible experiment ID and creates its log
directory. It does not train a model or run an environment.

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
