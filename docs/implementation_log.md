# Implementation Log

## Step 0 — project scaffold and contracts

### Implemented

- Added an editable Python package with runtime/configuration helpers.
- Added YAML presets for the shared runtime, RSSM-A/B dimensions, and CMB
  capacity.
- Added `AgentObs`, `Trajectory`, and `RSSMState` validation contracts.
- Added seed management, deterministic PyTorch settings, device selection,
  stable experiment IDs, and run-directory creation.
- Added package directories for environments, models, bridge, data, trainers,
  evaluation, scripts, and tests.
- Added `docs/spec.md` with the frozen time-index, observation, state, split,
  and no-leakage rules.

### Verified

- Environment used: conda environment `cmb_minigrid`, Python 3.11.17,
  PyTorch 2.14.1+cu130, NumPy 2.4.6, PyYAML 6.0.3, Gymnasium 1.4.0,
  MiniGrid 3.1.0, pytest 9.1.1.
- Editable installation succeeded with
  `/data2/xuyihao/envs/cmb_minigrid/bin/python -m pip install -e '.[dev]'`
  (project installed as `cmb-minigrid 0.1.0`).
- `/data2/xuyihao/envs/cmb_minigrid/bin/python -m pytest -q` — **7 passed**.
- `/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.cli --help` —
  exit code 0; help includes config, seed, device, output directory, and
  experiment name options.
- `/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.cli --config
  configs/default.yaml --seed 42 --device cpu --output-dir
  /tmp/cmb_minigrid_runs` — exit code 0; produced experiment ID
  `cmb_minigrid_step0-aa46e782ff` and created its log directory.
- Invalid config path smoke — exit code 2 with readable message
  `configuration file does not exist: /does/not/exist.yaml`.
- `ruff check cmb_minigrid tests` — clean after applying its four typing
  modernization suggestions.

### Not implemented / not tested yet

- No MiniGrid environment factory or rollout (Step 1).
- No HiddenHistoryNav/counterfactual pair generator (Step 2).
- No trajectory dataset, RSSM, policy, bridge, or training code (Steps 3–8).
- No closed-loop evaluation or experiment results (Steps 9–10).
- No training, environment rollout, or data generation commands were run,
  because those belong to later steps.

## Step 1 — official MiniGrid-Memory adapter

### Implemented

- Added `MiniGridMemoryEnv` and `make_memory_env()` around
  `MiniGrid-MemoryS13-v0`, with public actions `left/right/forward` only.
- Converted the local symbolic `image` to copied `uint8[7,7,3]`, retained
  `direction`, and added fixed public task ID `minigrid_memory_s13` in the
  existing `AgentObs` contract.
- Suppressed raw MiniGrid `info`; neither global map, internal environment
  state, matching target, hidden label, nor expert action is part of `AgentObs`
  or the wrapper's returned info.
- Added seeded random and scripted rollouts, stable per-observation SHA-256
  hashes, optional `human`/`rgb_array` rendering, and JSON rollout export.
- Added `cmb_minigrid.scripts.smoke_env` and committed a 20-transition seed-42
  sample at `data/samples/minigrid_memory_seed42.json`.
- Added Gymnasium-compatible value equality to `AgentObs` so deterministic
  environment checking can compare observations containing NumPy arrays.

### Action-set decision

The installed MiniGrid 3.1.0 native action enum has seven values. Inspection of
`MemoryEnv.step()` shows success/failure is determined by moving onto the
matching/non-matching target cells; pickup, drop, toggle, and done are not
needed for navigation or task completion. Therefore the requested 3-action
public interface preserves the task without adding actions.

### Verified

- Environment: `cmb_minigrid`, Python 3.11.17, Gymnasium 1.4.0, MiniGrid 3.1.0.
- `/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.smoke_env
  --seed 42` — exit code 0; `transitions=20`, observation shown as
  `image:(7, 7, 3):uint8,direction:int,task_id:str`; sample JSON written to
  `data/samples/minigrid_memory_seed42.json`.
- Scripted rollout command with seed 43 and 20 steps — exit code 0; wrote
  `/tmp/cmb_minigrid_scripted.json`.
- `gymnasium.utils.env_checker.check_env(..., skip_render_check=True)` —
  passed after adding array-aware `AgentObs` equality.
- `/data2/xuyihao/envs/cmb_minigrid/bin/python -m pytest -q` — **12 passed**.
- `/data2/xuyihao/envs/cmb_minigrid/bin/ruff check cmb_minigrid tests` —
  `All checks passed!`.
- Tests cover 3-action exposure, observation shape/dtype and field allowlist,
  empty info, exact seed/action replay hashes, `rgb_array` output, and 20-step
  sample serialization without privileged fields.

### Not run / remaining

- Human-window rendering was not run in the headless session; `rgb_array`
  rendering is covered by a test.
- No expert path or completion-rate claim was produced; this step only verifies
  the official environment interface and reproducible rollout mechanics.
- No HiddenHistoryNav, RSSM, policy, bridge, or training code was implemented.

### Step 2 dependency

Use `AgentObs` and the 3-action map as fixed. The official Memory task is only a
smoke environment; implement the strict paired-history and leakage checks in
`HiddenHistoryNav` without adding hidden labels or privileged simulator fields
to deployment observations.

## Step 2 — HiddenHistoryNav and counterfactual pairs

### Implemented

- Added `HiddenHistoryNavEnv` (`HiddenHistoryNav-v0`) with a deterministic
  MiniGrid map: an early red/blue clue, shared corridor, handoff at `t=12`, and
  hidden upper/lower branch goals whose open side differs by hidden label.
- Preserved the public `AgentObs` shape/dtype/action contract and suppressed
  hidden label, map, goal side, and expert action from observations and `info`.
- Added deterministic expert routes, `EnvSnapshot` JSON serialization,
  restore-and-replay, `generate_counterfactual_pair(seed)`, and
  `validate_pair(pair,k=8)`.
- Added `smoke_hidden_history` CLI and committed a 100-pair report/sample at
  `data/samples/hidden_history_pairs.json`.

### Verified

- `/data2/xuyihao/envs/cmb_minigrid/bin/python -m
  cmb_minigrid.scripts.smoke_hidden_history --count 100 --seed 42` — exit code
  0; `pairs=100 validated=100 reachable=100`, `handoff_t=12 recent_k=8`, and
  random no-history accuracy `0.520 n=100`.
- `/data2/xuyihao/envs/cmb_minigrid/bin/python -m pytest -q` — **17 passed**.
- `/data2/xuyihao/envs/cmb_minigrid/bin/ruff check cmb_minigrid tests` —
  `All checks passed!`.
- Gymnasium `check_env(HiddenHistoryNavEnv(...), skip_render_check=True)` —
  passed.
- Tests cover exact current/recent observation and action equality, different
  early clue and branch action, both reachable goals, deployment-field
  allowlist, snapshot JSON round-trip, restored future replay, and CLI batch
  reporting.

### Not run / remaining

- Human-window rendering was not run in the headless session.
- The random baseline is a diagnostic sample statistic only; no confidence
  interval or trained model result is claimed.
- No trajectory dataset, RSSM, policy, bridge, or training code was added.

### Step 3 dependency

Keep pair grouping by `map_family_id` and `pair_id`; store privileged snapshots
and expert actions separately from deployment observations. Dataset collection
must preserve `obs[0:T+1]` / `action[0:T]`, handoff snapshots, and the strict
counterfactual split rules without frame-level leakage.

## Step 3 — trajectory collection, grouped dataset, and offline supervision

### Implemented

- Added `TrajectoryRecord` with aligned `agent_obs[0:T+1]`,
  `actions/reward/terminated/truncated/expert_actions[0:T]`, map-family and
  pair identifiers, `handoff_t`, serialized handoff `env_snapshot`, and a
  separate per-time-step `privileged_state` sequence.
- Added seeded expert and exploratory collection. Exploration begins at the
  handoff, and the smoke collector includes a reproducible time-limit failure
  while retaining expert labels planned from the actual privileged state.
- Added JSONL persistence, strict record validation, grouped train/val/test
  manifests, `TrajectoryDataset`, deployment-safe CPU collation, observation
  and transition padding masks, and metadata without privileged fields.
- Added full replay validation: every observation, reward, done flag, and
  privileged snapshot is compared step by step; a fresh environment is restored
  from `env_snapshot` at `handoff_t` and its suffix is replayed; expert labels
  are independently recomputed from the privileged simulator state.
- Added configurable `collect_data`, `validate_data`, and `stats_data` CLIs.
  `configs/data.yaml` now supplies the smoke defaults and split ratios, while
  explicit CLI arguments take precedence.

### Initial single-family verification (superseded dataset)

- `/data2/xuyihao/envs/cmb_minigrid/bin/python -m pytest -q` — **25 passed**.
- `/data2/xuyihao/envs/cmb_minigrid/bin/ruff check cmb_minigrid tests` —
  clean.
- Configured smoke collection with one pair — exit code 0; generated
  `records=4`, `successes=3`, `truncated=1`.
- `validate_data` on that fresh output — exit code 0;
  `records=4 replayed=4 labels=4 handoff_restore=validated split_errors=0`.
- Existing eight-record sample validation — exit code 0;
  `records=8 replayed=8 labels=8 handoff_restore=validated split_errors=0`.
- Statistics report eight records, two pairs, four expert and four
  exploration episodes, horizons 18–64, six terminated and two truncated.
- CPU collation tests verify variable-length `T+1` observation masks,
  `T` transition masks, `-1` action padding, and no `privileged_state` or
  `env_snapshot` in the deployment batch.

### Step 3 map-family expansion — implementation and interfaces

Files changed for this expansion:

- `cmb_minigrid/environments/hidden_history.py` and `environments/__init__.py`:
  `MapLayout`, five real geometry templates, family-aware construction/reset,
  `generate_counterfactual_pair(seed, map_family_id=...)`, shared privileged BFS
  planning, and family-aware snapshots/restoration. Default v1 retains its
  19×15 geometry, routes, three actions, observation fields, and task ID.
- `cmb_minigrid/data/trajectory.py`, `dataset.py`, and `data/__init__.py`:
  `collect_records(..., map_family_ids=...)` interprets `pair_count` per family;
  pair IDs include family and seed. Old v1 snapshot JSON is normalized when
  loaded. Added `validate_counterfactual_records()` for the saved endpoints,
  family consistency checks, and allocation of whole families to non-empty
  positive-ratio splits when enough families exist. Zero-ratio splits stay
  empty; assignments are independent of record order.
- `cmb_minigrid/scripts/collect_data.py`, `validate_data.py`, `stats_data.py`:
  `--map-families`, `--pairs-per-family` (`--pairs` remains an alias), saved
  pair validation, and per-family record/pair/split statistics.
- `configs/data.yaml`, `tests/test_hidden_history.py`, `tests/test_data.py`:
  five-family/ten-pair defaults and regression tests for physical layout
  differences, absence of shortcuts around the hub, shortest expert routes,
  family-aware JSON snapshot restoration, legacy compatibility, leakage,
  seeded reproducibility, and real CPU DataLoader batches for all splits.
- `data/samples/step3_trajectories.jsonl`, `step3_manifest.json`: regenerated
  with seed 42. `README.md`, `docs/spec.md`, `docs/step3.md`, and this log
  document the compatible expansion and its measured acceptance results.

| Family | Geometry | Pairs | Records | Split |
| --- | --- | ---: | ---: | --- |
| hidden_history_nav_v1 | Original horizontal corridor / vertical fork | 10 | 40 | test |
| hidden_history_nav_v2 | Bent corridor / vertical fork | 10 | 40 | val |
| hidden_history_nav_v3 | Vertical corridor / horizontal fork | 10 | 40 | train |
| hidden_history_nav_v4 | Horizontal corridor / branches with one bend | 10 | 40 | train |
| hidden_history_nav_v5 | Horizontal corridor / branches with two bends | 10 | 40 | train |

### Step 3 map-family expansion verification

- `/data2/xuyihao/envs/cmb_minigrid/bin/python -m pytest -q` — exit code 0;
  **44 passed in 17.05s**.
- `/data2/xuyihao/envs/cmb_minigrid/bin/ruff check cmb_minigrid tests` — exit
  code 0; `All checks passed!`.
- `/data2/xuyihao/envs/cmb_minigrid/bin/python -m
  cmb_minigrid.scripts.collect_data --config configs/data.yaml
  --pairs-per-family 10 --output data/samples/step3_trajectories.jsonl
  --manifest data/samples/step3_manifest.json --force` — exit code 0;
  `records=200 pairs_per_family=10 families=5 pairs=50 successes=150
  truncated=50`, split counts `train=120,val=40,test=40`.
- `/data2/xuyihao/envs/cmb_minigrid/bin/python -m
  cmb_minigrid.scripts.validate_data` — exit code 0;
  `records=200 replayed=200 labels=200 pairs=50 handoff_restore=validated
  split_errors=0 pair_errors=0`.
- `/data2/xuyihao/envs/cmb_minigrid/bin/python -m
  cmb_minigrid.scripts.stats_data` — exit code 0; five map families, 50
  pairs, 100 expert and 100 exploration records, horizons 18–64, 150
  terminated and 50 truncated.
- An additional CPU DataLoader check on the final sample — exit code 0;
  each split produced an image tensor `(4,65,7,7,3)` on CPU with correct
  `T+1`/`T` masks and no privileged simulator fields in the batch. The same
  check is automated as `test_each_split_loads_cpu_batch` for all splits.

### Limitations and not run

- Five fixed geometry templates are implemented. Ten seeded pair collections
  per family provide repeated expert trajectories and differing exploratory
  suffixes, not fifty independent maps. This is a pilot dataset, not evidence
  of broad spatial generalization or model performance.
- Configured ratios remain `(0.8,0.1,0.1)`; indivisible families and non-empty
  splits yield 3/1/1 families (120/40/40 records, realized 60/20/20 percent).
  Family and pair leakage checks both pass. More families are needed to
  approximate the requested ratios closely and assess varied held-out maps.
- Larger-scale collection was not run beyond the requested 10 pairs/family.
  Human-window rendering was not run in this headless session. RSSM, policy,
  bridge training and closed-loop evaluation remain outside Step 3.
- No GPU-specific test was run because Step 3 acceptance requires CPU
  execution and the data pipeline has no GPU-only code path.

### Step 4 dependency

Implement and test independent RSSM-A/B interfaces in Step 4; the validated
dataset provides CPU batches for later training in Step 5. Keep
`privileged_state`, hidden labels, maps, and expert labels out of deployed
B/CMB inputs; use expert actions only as offline supervision.

## Step 4 — lightweight PyTorch categorical RSSM (2026-10-13)

### Implementation and changed files

- Added `cmb_minigrid/models/__init__.py` and `models/rssm.py`:
  `RSSMConfig`, `SymbolicEncoder`, `RSSM`, `RSSMSequence`, and
  `categorical_kl`. Separate object/color/state embeddings `(11,6,3)` feed a
  small MLP producing 64 visual features; direction contributes four one-hot
  values. Ordinary GRUCell, prior/posterior MLPs and `16x16` categorical
  stochastic state are used. Heads reconstruct symbolic channels/direction
  and predict scalar reward/Bernoulli continue. This is a small PyTorch
  research implementation, not a line-for-line official JAX reproduction.
- Added `tests/test_rssm.py` before writing the model implementation: 12 CPU
  checks (including A/B parametrizations) cover dimensions, categorical
  probabilities/one-hot states, finite losses/gradients, initial zero action,
  step/sequence equivalence, observation-free imagination, prefix causality,
  seeded sampling, deterministic evaluation, straight-through gradients,
  unimix numerical stability, KL stop-gradients/free-nats, next-state targets,
  truncation semantics, padding, contract failures and A/B toy batch overfit.
- Added `cmb_minigrid/scripts/smoke_rssm.py`: train-split-only DataLoader
  diagnostic with seed/device/config, sampled backward pass and bounded
  deterministic single-batch updates. JSON reports are stored under `runs/`;
  no model checkpoint, optimizer resume, validation loop or formal trainer
  was added. It returns nonzero if losses/gradients are invalid or requested
  overfit updates fail to reduce the total below 70% of its initial value.
- Extended `configs/rssm_a.yaml` and `configs/rssm_b.yaml` with encoder/MLP,
  unimix and loss settings; A remains deter=128/seed42, B deter=256/seed43,
  both remain stoch=16x16 and visual embedding=64.
- Updated `README.md`, `docs/spec.md` and this log with scope, interfaces,
  reproduction and measured acceptance. No data, split, task, public
  observation/action interface, state contract or experimental baseline changed.

### Interfaces and alignment

`initial(batch)` creates a model-device/dtype sentinel; consume `obs[0]` once
using `observe_step(initial, None, obs0)` (or an all-zero previous action vector).
Integer action 0 still means left. `observe_step(state[t], action[t], obs[t+1])`
returns posterior state[t+1]. `imagine_step(state, action)` has no observation
input and never calls the encoder; it returns a prior state. `features(state)`
is `[B,384]` for A / `[B,512]` for B. Tensor observations contain exactly the
three public fields; the constant task identifier has no language embedding.

`observe` accepts `[B,T+1,7,7,3]` images and `[B,T]` actions and returns a
dedicated sequence result with a tuple of batch-first `RSSMState`s and stacked
features/prior/posterior logits. This preserves the frozen single-state
contract. `decode` supports batch or batch/time features; `loss` reads only
public observations, actions, reward, terminal flags and masks from Step 3.
Expert labels and privileged state/metadata never enter the model.

Reconstruction includes valid obs[0:T+1], averaging image cross-entropy over
pixels/channels and adding direction cross-entropy. Reward MSE and continue
BCE for action[t] are predicted from state[t+1]; continue is `~terminated`,
not `~(terminated|truncated)`. Valid terminal/final-observation targets are
included; padded states freeze and all padded targets are sanitized/excluded.
Masks must be prefixes and valid transitions after terminal/truncated steps
are rejected.

Dynamics KL `KL(sg(q)||p)` and representation KL `KL(q||sg(p))` sum categories
and groups before per-sample free-nats clamping, then average valid samples.
Defaults: unimix=.01, free_nats=1, dynamics scale=.5, representation scale=.1,
reward/continue scale=1. Categorical training uses straight-through samples;
eval uses argmax, with an explicit sampling override. A KL below free-nats
correctly yields zero prior gradient. The initial test run was exit 1 (2 failed,
8 passed) because the connectivity test incorrectly required nonzero prior
gradients below this floor. That test now disables the floor solely to check
connectivity; a separate test asserts zero gradients below the default floor.
Defaults were preserved.

### Actual verification in cmb_minigrid (Python 3.11.17)

Commands below were executed from the repository root. All final commands
returned **exit code 0**:

```bash
/data2/xuyihao/envs/cmb_minigrid/bin/python -m pytest -q
/data2/xuyihao/envs/cmb_minigrid/bin/ruff check cmb_minigrid tests
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.smoke_rssm --config configs/rssm_a.yaml --device cpu --batch-size 4 --sequence-length 64 --overfit-steps 40
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.smoke_rssm --config configs/rssm_b.yaml --device cpu --batch-size 4 --sequence-length 64 --overfit-steps 40
```

- Pytest: `56 passed in 12.07s` (44 existing + 12 new cases).
- Ruff: `All checks passed!`.
- Both CLIs read only the train split (120 episodes), use one fixed batch of
  four records, and verify finite sampled losses and nonzero finite total
  gradients. The full padded horizon is 64 transitions / 65 observations.
- Both A and B also passed an earlier 8-transition/40-update diagnostic with
  exit code 0; these shorter diagnostics do not assess long-history retention.

| Full-horizon diagnostic | RSSM-A | RSSM-B |
| --- | ---: | ---: |
| Seed | 42 | 43 |
| Parameters | 665322 | 1011306 |
| Feature tensor | [4,65,384] | [4,65,512] |
| Prior/posterior tensor | [4,65,16,16] | [4,65,16,16] |
| Sampled total loss | 4.474426 | 4.456645 |
| Sampled total gradient norm | 0.797845 | 0.781619 |
| Initial deterministic total | 4.471661 | 4.418174 |
| Final deterministic total (40 updates) | 0.996768 | 0.916985 |
| Final / initial total | 0.222908 | 0.207548 |
| Final reconstruction | 0.277419 | 0.249884 |
| Final reward MSE | 0.017090 | 0.011376 |
| Final continue BCE | 0.102258 | 0.055725 |
| Final dynamics / representation KL | 1 / 1 | 1 / 1 |

Reports (local, ignored runtime artifacts):
`runs/cmb_minigrid_rssm_a-e0682b695c/smoke_rssm.json` and
`runs/cmb_minigrid_rssm_b-a202e3d01f/smoke_rssm.json`. They record full config,
device, seed, experiment ID, dimensions, sampled gradient/loss and each
initial/final loss component. Toy repeated-symbol batches separately pass a
40-update overfit assertion for both sizes (total <30% of initial,
reconstruction <0.1 and reward MSE <0.02).

### Limitations, unrun checks and next dependencies

- No unresolved Step 4 acceptance failure remains. The total KL floor is
  0.6; a nonzero final loss here is expected. Memorizing a batch does not prove
  predictive generalization, retention of the hidden cue or closed-loop control.
- GPU, mixed-precision, throughput and cross-device reproducibility checks
  were not run: this step requires CPU smoke acceptance and uses float32.
  Deterministic CPU sampling/model initialization are tested; cross-platform
  bitwise equality is not claimed.
- No formal A/B training, validation/early stopping, checkpoint resume,
  multi-seed research experiment, hidden-history probe, policy, bridge or
  closed-loop evaluation was run or implemented: those are subsequent steps.
- Step 5 can consume the unchanged train/val/test dataset and this masked
  loss/interface. It still needs its own sequence-training, logging,
  optimizer/checkpoint lifecycle, validation and history-retention gates.

## Step 5 — independent RSSM-A/B training and acceptance (2026-10-13)

### Changed files and scope

- Added `cmb_minigrid/training/__init__.py`, `training/rssm.py` and
  `training/probe.py`: independent full-episode training, checkpoint lifecycle,
  valid-target-weighted validation, early stopping and frozen linear cue probes.
- Added `cmb_minigrid/scripts/train_rssm.py` and `scripts/probe_rssm.py`.
- Added `configs/train_rssm.yaml` and `tests/test_training.py` (8 CPU tests).
- Updated `README.md`, `docs/spec.md` and this implementation log.
- Read the existing spec/log/configurations first; the worktree was initially
  clean. Reused the Step 4 categorical reconstruction, reward/continue and
  split KL losses without modifying their implementation or scales. No task,
  observation, state/action interface, baseline, dataset or split was changed.
  No policy, bridge or subsequent-step training was implemented or run.

### Implementation and interfaces

`train_rssm(config, output_dir, resume=...)` trains one independent RSSM. A
keeps deter=128/seed42, B keeps deter=256/seed43; each has separate initialization,
Adam optimizer, logs and checkpoints. Configured defaults: 1000 maximum updates,
batch size 8, learning rate .001, gradient clip 100, validation every 50 updates,
patience 5, min_delta .001 and one CPU thread. Formal runs initialize afresh
after smoke; the first 100 sampled updates are reproducible with the same seed.
Smoke defaults to exactly 100 updates and disables early stopping.

Episodes are shuffled as complete sequences, with full BPTT starting at obs[0].
There is no cropped, zero-initialized history suffix and no frame-level split.
An explicit batch allowlist excludes expert actions, metadata and privileged
state from model losses. Existing prefix masks sanitize padding, include the
terminal transition/final observation and reject any valid post-done transition;
continue still bootstraps time-limit truncation. Validation uses all 40 val
episodes and deterministic categorical modes; components are weighted across
batches by valid observation/transition counts before recomputing total loss.
No test episodes are trained or evaluated.

`metrics.jsonl` records every sampled train component, total gradient norm
before clipping, per-module gradient norms after clipping, valid target counts,
and each validation result. Non-finite loss/gradient or zero total gradient
fails explicitly; in-loop errors create `failure.json` and retain the previous
safe checkpoint. Zero prior gradient below the free-nats floor is legitimate.
Run `report.json` distinguishes decreasing loss from lack of improvement; it
does not claim memory sufficiency. `config.json` records resolved configuration,
seed/device, paths and SHA-256 fingerprints of both data and split manifest.

`best.pt` is selected by validation; `last.pt` is the latest safe resume point.
Checkpoints store schema, full config, model, Adam, stopping counters, step,
initial losses, episode permutation/cursor and Python/NumPy/CPU/CUDA RNG state.
Temporary-file replacement makes checkpoint writes atomic. Existing run
directories are rejected unless explicitly resumed. Resume checks configuration
and file fingerprints; only maximum step budget/device may change. Resume must
use the run's latest `last.pt`; early-stopped runs remain stopped. Exact CPU
continuation is tested at scheduled validation boundaries. A budget ending
between scheduled boundaries introduces an extra endpoint validation, so its
stopping decisions need not match an uninterrupted budget.

`load_rssm(path, device)` loads either model independently into eval mode using
only that checkpoint, returning `(RSSM, checkpoint)`. No external model YAML or
peer is required. The initial sentinel, state[t] convention and full-history
filtering APIs remain unchanged.

`run_probes(...)`/`probe_rssm` validate saved strict expert pairs and freeze both
RSSMs (`eval`, `requires_grad=False`, `no_grad` extraction). Separate linear
logistic probes use train-only normalization, fixed L2=.01 and 100 LBFGS
iterations, with no validation tuning. Label targets are offline hidden labels
only. Each pair is filtered through handoff t=12 using obs[:13]/action[:12];
future branch observations/actions are excluded. Features include full-history
`features(state)` (deter + one-hot stoch), reset-current state, reset recent
eight-transition state, early state and raw current/early categorical observation.
Reports include train/val accuracy, both-endpoint accuracy and pair feature
distance. Probe weights/normalizers are separate files; they never feed back
into RSSM training, filtering or loss. No test-set probe is run.

### Actual commands and exit codes

All commands below were executed in the repository root using the existing
`cmb_minigrid` environment. Every listed command returned **exit code 0**:

```bash
/data2/xuyihao/envs/cmb_minigrid/bin/python -m pytest -q tests/test_training.py
/data2/xuyihao/envs/cmb_minigrid/bin/python -m pytest -q
/data2/xuyihao/envs/cmb_minigrid/bin/ruff check cmb_minigrid tests
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.validate_data
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.train_rssm --help
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.probe_rssm --help
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.train_rssm --config configs/rssm_a.yaml --device cpu --mode smoke --output-dir runs/step5/smoke
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.train_rssm --config configs/rssm_b.yaml --device cpu --mode smoke --output-dir runs/step5/smoke
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.train_rssm --config configs/rssm_a.yaml --device cpu --output-dir runs/step5/formal
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.train_rssm --config configs/rssm_b.yaml --device cpu --output-dir runs/step5/formal
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.probe_rssm --checkpoint-a runs/step5/formal/cmb_minigrid_rssm_a-14bec33435/best.pt --checkpoint-b runs/step5/formal/cmb_minigrid_rssm_b-9f7a571497/best.pt --output-dir runs/step5/probes --device cpu --seed 42
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.train_rssm --config configs/rssm_a.yaml --device cpu --max-steps 2 --output-dir runs/step5/resume_check
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.train_rssm --config configs/rssm_a.yaml --device cpu --max-steps 4 --resume runs/step5/resume_check/cmb_minigrid_rssm_a-653729eb84/last.pt
git diff --check
```

- Environment inspection: Python 3.11.17, PyTorch 2.14.1+cu130, Gymnasium 1.4.0,
  MiniGrid 3.1.0; `cuda_available False`, `cuda_devices 0`, with an NVML
  initialization warning. All actual training/probing used CPU float32.
- New tests: `8 passed in 12.37s`; full suite: `64 passed in 27.17s`.
  Ruff: `All checks passed!`; diff check: no whitespace errors.
- Data CLI: `records=200 replayed=200 labels=200 pairs=50
  handoff_restore=validated split_errors=0 pair_errors=0`.
- Tests cover exact sampled resume (weights, Adam moments, RNG, sampler and
  stopping state), independent A/B loading and all-time filtering equivalence,
  validation weighting with mixed horizons, patience and best/last separation,
  explicit non-finite failure, changed config/data and existing-run rejection,
  empty validation failure, frozen probe parameter preservation, chance-level
  paired-current controls, visible-cue positive control and future poisoning.
  Existing Step 4 tests still cover loss masks, terminal/truncation semantics
  and KL stop-gradients/free-nats.
- Actual resume CLI: restored step 2 and executed steps 3–4; validation total
  decreased from 4.219810 to 3.969110. This separate lifecycle check does not
  alter the 100-step smoke or formal checkpoints.

### Measured training results (pilot)

Both smoke CLIs finished all 100 updates, with finite losses/gradients,
`status=loss_decreased`, train=120 episodes and val=40 episodes.

| CPU smoke | A (seed42) | B (seed43) |
| --- | ---: | ---: |
| Initial deterministic train total | 4.411941 | 4.430413 |
| Final deterministic train total | 1.789683 | 1.826862 |
| Initial deterministic val total | 4.420314 | 4.419200 |
| Final deterministic val total | 1.880305 | 1.772794 |

Formal runs used the unchanged default 1000-update ceiling and both actually
early-stopped after five validation checks without a sufficient improvement.

| Formal CPU training | A | B |
| --- | ---: | ---: |
| Executed updates | 400 | 350 |
| Selected best step | 150 | 100 |
| Best deterministic val total | 1.492647 | 1.772794 |
| Last deterministic train total | 1.388558 | 1.058326 |
| Last deterministic val total | 2.606383 | 2.045344 |
| First 25 sampled updates mean total | 3.183479 | 3.010983 |
| Last 25 sampled updates mean total | 1.313743 | 1.092375 |
| Minimum pre-clip total gradient norm | 0.267121 | 0.302103 |
| Maximum pre-clip total gradient norm | 9.336225 | 16.183205 |
| Updates with zero prior gradient | 93 | 77 |

Losses decrease overall, but are not monotone. Validation rebounds markedly:
A's step-350 val loss is 5.181147, and B's step-200 value is 3.231568. This is
evidence of poor generalization/overfitting on the pilot, not a usable-memory
claim. Best checkpoints, rather than the last states, were used for probes.

Formal artifacts (local and ignored by git):

- `runs/step5/formal/cmb_minigrid_rssm_a-14bec33435/{best.pt,last.pt,report.json,metrics.jsonl,config.json}`.
- `runs/step5/formal/cmb_minigrid_rssm_b-9f7a571497/{best.pt,last.pt,report.json,metrics.jsonl,config.json}`.
- Smoke artifacts: `runs/step5/smoke/cmb_minigrid_rssm_a-f747fde0fa/` and
  `runs/step5/smoke/cmb_minigrid_rssm_b-3199aaf0d5/`.
- Probe artifacts: `runs/step5/probes/probe_report.json` and 12 independent
  `probe_{A,B}_{feature}.pt` files.

An additional actual CPU check independently loaded the two formal best
checkpoints and filtered a longest train episode (64 transitions / 65
observations). A features were `[1,65,384]`, B `[1,65,512]`; every time step
matched repeated `observe_step` exactly (`rtol=atol=0`), and all features were
finite. The command returned exit 0, selected_step=150/100 and seed=42/43.

### Cue-memory diagnostic failure

The probe CLI completed successfully as a diagnostic tool (exit 0); its model
memory result is **negative**, not a successful history-retention gate. It fit
30 train pairs / 60 endpoints from three families and evaluated 10 val pairs /
20 endpoints from the held-out validation family. Both A and B produced:

| Frozen feature | Train cue accuracy | Val cue accuracy | Val both-endpoint accuracy |
| --- | ---: | ---: | ---: |
| Full-history handoff state | 0.5 | 0.5 | 0 |
| Current-only reset state | 0.5 | 0.5 | 0 |
| Recent-eight-transition reset state | 0.5 | 0.5 | 0 |
| Early state (obs[0]) | 0.5 | 0.5 | 0 |
| Raw current observation | 0.5 | 0.5 | 0 |
| Raw early observation | 1.0 | 1.0 | 1 |

Paired full-history, early, current and recent **feature distances were exactly
zero** for both models in train and val. The early observation positive control
has mean pair distance 1.414214 and perfectly decodes the cue; thus the diagnostic
is not limited by an invisible label. Current/recent chance controls confirm
that this cue is absent from the permitted handoff window.

An additional actual diagnostic on one val expert pair (exit 0) compared state
components at t=0 and t=12. Both models' deter and hard one-hot stoch pair
distances were zero at both times. At t=0 posterior-logit distances were
0.021823682 (A) and 0.010492683 (B), but at t=12 these were also zero. These
measurements localize a deterministic-filtering problem: the early encoder/
posterior can respond slightly to the cue, but argmax selects the same categories,
so the recurrent state receives no differing stochastic input along the shared
action/observation prefix. This is measured evidence for these checkpoints, not
a conclusion about all hyperparameters or sampled filtering modes.

### Unresolved issues, unrun checks and next dependency

- Engineering acceptance is verified: independent checkpoints, full-history
  filtering, decreasing overall losses, masks, validation stopping, logs and
  optimizer/config/RNG resume. **The trained pilot does not preserve the early
  cue in deterministic `features(state)` at handoff. Memory sufficiency is not
  accepted.** No hidden-label auxiliary loss, probe feedback or experiment
  definition change was introduced to force a positive result.
- GPU training, GPU resume, mixed precision and cross-device reproducibility
  were not run because this session reports no available CUDA device. CPU smoke
  and exact CPU continuation are verified; no cross-device equality is claimed.
- Test-split probes, large datasets, multiple independent A/B seed runs and
  confidence intervals were not run: this step is a small existing-data pilot,
  and test results were not consumed for training or model selection. Repeated
  seeded pairs use five fixed geometry templates, not independent random maps.
- Policy/Actor, bridge, Oracle/Reset control comparisons and closed-loop success
  were not run because the user authorized only Step 5. Current-observation
  decoding and world-model loss alone are not grounds for moving on.
- Next dependency: first resolve cue retention in an authorized Step 5 follow-up
  (categorical collapse/deterministic filtering and long-history learning),
  preserving task/splits and keeping probes offline. Only after adequate memory
  evidence should later policy/control gates be considered. No later step was
  started automatically.

## Step 5 follow-up: standalone categorical response diagnosis (2026-10-13)

### Files, scope and interface

- Completed the existing untracked draft
  `cmb_minigrid/scripts/diagnose_categorical_response.py`, added
  `tests/test_categorical_response.py`, and updated this log. Existing model,
  trainer, probe, configs, data, splits and task interfaces were not edited.
- CLI accepts `--checkpoint`, `--input`, `--manifest`, `--split` (default val),
  `--pair-id` (default first expert pair in that split), `--device`, `--seed`
  and optional `--output`. It validates the manifest and selected strict pair,
  checks checkpoint input/manifest fingerprints, and sorts worlds by hidden
  label for reporting only. Labels never enter filtering.
- Calls actual `observe_step(..., sample=False)` from the initial sentinel at
  t=0 and uses action[0] at t=1. Temporary read-only forward hooks capture
  encoder and raw posterior outputs; hooks are removed in a finally block.
  Independently computed transition, normalized logits and hard one-hot state
  must match the returned state exactly. No optimization or retraining occurs.
- Prints mean absolute, maximum absolute and L2 differences for encoder, raw
  posterior, stored posterior logits (normalized log probabilities), posterior
  probabilities, stochastic state and deterministic state. Also prints the
  two argmax vectors, changed-group count, public observation/action equality,
  per-group probability perturbations and top-1/top-2 probability margins.
  A margin greater than twice the maximum category probability perturbation
  is a sufficient bound for the winner to remain unchanged in that group.

### Actual commands and verification

Final commands below all returned exit code 0 from the repository root:

```bash
/data2/xuyihao/envs/cmb_minigrid/bin/python -m pytest -q tests/test_categorical_response.py
/data2/xuyihao/envs/cmb_minigrid/bin/python -m pytest -q
/data2/xuyihao/envs/cmb_minigrid/bin/ruff check cmb_minigrid tests
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.diagnose_categorical_response --checkpoint runs/step5/formal/cmb_minigrid_rssm_a-14bec33435/best.pt --device cpu --seed 42 --split val --output runs/step5/probes/categorical_response_a.json
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.diagnose_categorical_response --checkpoint runs/step5/formal/cmb_minigrid_rssm_b-9f7a571497/best.pt --device cpu --seed 42 --split val --output runs/step5/probes/categorical_response_b.json
git diff --check
```

New tests: `5 passed in 4.72s`; full suite: `69 passed in 28.51s`;
Ruff: `All checks passed!`. Tests check controlled nonzero responses with and
without category flips, zero posterior response, exact agreement with sequence
filtering, unchanged weights/RNG/gradients, hook removal, missing/incomplete
pairs, mismatched initial actions and non-finite differences. First test attempt
returned exit 1 (`1 failed, 4 passed`) because the test tried to mutate an action
tuple; corrected to create a new tuple. Initial focused Ruff returned exit 1 for
loop-closure binding and `sorted(...)[0]`; both were corrected before final runs.

### Measured result and evidence boundary

Both CLI runs selected validation pair `pair-hidden_history_nav_v2-0052`, worlds
`left_blocked`/`right_blocked`. A checkpoint step=150/seed42;
B checkpoint step=100/seed43. Both t=0 and t=1 images differ in exactly one
color-channel entry and no object/state entries; directions/task IDs match.
t=0 previous action is None; t=1 previous actions are `[2,2]`.

| Model/time | Encoder L2 | Stored logits L2 | Probabilities L2 | Argmax changes | Stoch L2 | Deter L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A, t=0 | 0.032048747 | 0.021823680 | 0.001617347 | 0/16 | 0 | 0 |
| A, t=1 | 0.119881116 | 0.096732296 | 0.007295698 | 0/16 | 0 | 0 |
| B, t=0 | 0.045795418 | 0.010492683 | 0.000660211 | 0/16 | 0 | 0 |
| B, t=1 | 0.060542379 | 0.011704215 | 0.000755153 | 0/16 | 0 | 0 |

For all four rows, all 16 world-0 probability margins exceed twice their
group's maximum probability perturbation. The largest absolute probability
differences are A: 0.000404082 at t=0, 0.001881883 at t=1;
B: 0.000138953 at t=0, 0.000152647 at t=1. Raw posterior responses are also
nonzero. Every reconstructed state matches `observe_step` exactly.

These checkpoints respond to color before hard selection, but the response is
too small relative to category decision margins to change the hard stochastic
features. Equal hard stochastic inputs and actions leave the recurrence equal
at t=1. Equal deter at t=0 is expected because its transition precedes obs[0].
Posterior logits remain different within the returned RSSMState, but neither
the GRU transition nor `features(state)` uses those logits. This localizes the
loss of cue distinction in deterministic recurrent/probe features to argmax;
it does not explain why training learned such a weak cue response.

Reports are new ignored runtime artifacts:
`runs/step5/probes/categorical_response_a.json` and
`runs/step5/probes/categorical_response_b.json`. No previous probe report or
checkpoint was overwritten.

### Unrun checks and next dependency

GPU diagnostics, sampled filtering, all-pair sweeps, test-split evaluation,
other checkpoints/seeds and later time-step diagnoses were not run: this
request targets t=0/1 deterministic filtering for the existing A/B checkpoints.
The finding applies to the selected validation pair, not every possible input.
No model fix, retraining, policy or bridge work was attempted. A subsequent
authorized investigation must distinguish weak cue learning from the hard
filtering bottleneck before claiming that cue memory is adequate.

### Repeat verification of the requested t=0/1 diagnosis

Reused the standalone script without editing original code or overwriting any
reports/checkpoints. Only this log was additionally edited. Actual commands:

```bash
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.diagnose_categorical_response --checkpoint runs/step5/formal/cmb_minigrid_rssm_a-14bec33435/best.pt --device cpu --seed 42 --split val
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.diagnose_categorical_response --checkpoint runs/step5/formal/cmb_minigrid_rssm_b-9f7a571497/best.pt --device cpu --seed 42 --split val
/data2/xuyihao/envs/cmb_minigrid/bin/python -m pytest -q tests/test_categorical_response.py
/data2/xuyihao/envs/cmb_minigrid/bin/ruff check cmb_minigrid/scripts/diagnose_categorical_response.py tests/test_categorical_response.py
git diff --check
```

All returned exit code 0. Focused tests: `5 passed in 4.38s`; Ruff:
`All checks passed!`. Both model runs reproduced the table above for validation
pair `pair-hidden_history_nav_v2-0052`, with 0/16 argmax changes and zero hard
stochastic/deterministic differences at both times. All four margin checks
returned 16/16 groups satisfying the sufficient no-flip bound, and all state
checks exactly matched `observe_step`. The cue is still visibly different at
t=1; equal hard states do not imply equal encoder outputs or equal distributions.
The full suite was not repeated in this verification because no script/test
code changed since the previously recorded 69-test pass. GPU, sampled filtering,
all-pair sweeps and retraining remain unrun for the scope reasons above; the
underlying cause of weak learned cue response remains a separate dependency.
