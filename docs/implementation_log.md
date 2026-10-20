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

## Step 5 — final RSSM-A/B training and memory verification (2026-10-14)

### Final implementation

- `RSSMConfig.encoder_type` supports `symbolic_mlp` (default, backward
  compatible) and `cnn`. The final models use `cnn`: each MiniGrid categorical
  channel is embedded separately, followed by spatial convolutions and global
  pooling to produce the unchanged 64-dimensional observation embedding.
- The objective includes the masked RSSM reconstruction, reward/continue and
  split KL losses, plus `recall_scale=5.0` at `recall_t=12` and
  `visual_recall_scale=5.0` at observation times 0 and 1. Targets use only the
  visible public Key color in `obs[0]`; auxiliary outputs never enter
  filtering, the GRU or categorical sampling.
- Training remains full-history sequence BPTT with padding/terminal masks,
  deterministic validation, gradient monitoring, validation-selected
  `best.pt`, safe `last.pt`, and saved Adam/RNG/config/sampler state.
- Frozen linear probes consume only the prefix through handoff t=12 and never
  feed back into training or filtering. Current-observation and recent-history
  controls remain part of the diagnostic.

Relevant interfaces are in `cmb_minigrid/models/rssm.py`,
`cmb_minigrid/training/rssm.py`, `cmb_minigrid/training/probe.py`,
`cmb_minigrid/scripts/probe_rssm.py`, and
`cmb_minigrid/scripts/diagnose_categorical_response.py`. CNN training is
configured by `configs/train_rssm_cnn.yaml`.

### Final artifacts and measured acceptance

The selected user-provided checkpoints are `models/best_a.pt` (RSSM-A,
seed 42) and `models/best_b.pt` (RSSM-B, seed 45). Both independently load
from the checkpoint alone and were tested on 10 validation counterfactual
pairs, 20 endpoints, using public history through t=12.

| Metric at t=12 | RSSM-A | RSSM-B |
| --- | ---: | ---: |
| Full-history frozen probe accuracy | 100% | 100% |
| Trained recall-head accuracy | 100% | 100% |
| Both endpoints correct | 100% | 100% |
| Mean deterministic-state pair L2 | 13.1766 | 16.7183 |
| Mean stochastic-state pair L2 | 3.1623 | 4.8990 |
| Mean full-feature pair L2 | 13.5507 | 17.4213 |
| Changed categorical groups | 5/16 | 12/16 |
| Current-state probe accuracy | 50% | 50% |
| Recent-state probe accuracy | 50% | 50% |

The direct visual classification head reaches 100% at both t=0 and t=1 for
both models. The standalone categorical diagnosis on a validation pair is:

| Model/time | Encoder L2 | Posterior-logit L2 | Probability L2 | Argmax changes | Stochastic L2 | Deterministic L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A, t=0 | 14.5876 | 26.0298 | 3.5283 | 11/16 | 4.6904 | 0 |
| A, t=1 | 14.0465 | 27.4548 | 3.5494 | 10/16 | 4.4721 | 4.1939 |
| B, t=0 | 15.8761 | 60.2976 | 4.2754 | 16/16 | 5.6569 | 0 |
| B, t=1 | 15.4173 | 61.7733 | 4.2569 | 16/16 | 5.6569 | 8.2202 |

The zero deterministic difference at t=0 is expected because the recurrent
transition precedes conditioning on observation 0. At t=1 the deterministic
states differ, showing that the early color entered the recurrent state. At
t=12 both models preserve color information in full-history states; the
current/recent reset controls remain at chance.

### Verification and boundaries

Final checks used Python 3.11.17 and CPU float32:

```bash
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.probe_rssm \
  --checkpoint-a models/best_a.pt --checkpoint-b models/best_b.pt \
  --output-dir runs/step5/user_models_probe_20261014 --device cpu --seed 42
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.diagnose_categorical_response \
  --checkpoint models/best_a.pt --device cpu --split val \
  --output runs/step5/user_models_diag/categorical_a.json
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.diagnose_categorical_response \
  --checkpoint models/best_b.pt --device cpu --split val \
  --output runs/step5/user_models_diag/categorical_b.json
```

All three commands returned exit code 0. Reports are in
`runs/step5/user_models_probe_20261014/` and
`runs/step5/user_models_diag/`. The full suite after the CNN implementation
passed (`84 passed`), Ruff was clean, and `git diff --check` was clean.

This establishes cue decoding and retention in the RSSM state, not downstream
policy, bridge or closed-loop control success. GPU/mixed-precision comparison,
test-split evaluation, additional seed/weight sweeps and control evaluation
remain unrun.

## Step 6 — frozen-RSSM Actor training and Oracle/Reset closed-loop gate (2026-10-16)

### Final implementation

- `cmb_minigrid/models/policy.py` defines `ActorPolicy`, a three-action
  (`left=0`, `right=1`, `forward=2`) behavior-cloning head consuming an RSSM
  feature and a fixed public goal token.
- `cmb_minigrid/training/policy.py` trains the Actor on detached posterior
  features of a frozen RSSM using train/val expert records and offline
  `expert_actions` labels. Class weighting is selected by `class_weighting`
  (`inverse`, `sqrt` or `none`).
- `run_closed_loop` implements three independent closed loops:
  `a_full_history`, `oracle_b_full_history` and `reset_b`. Full-history
  filtering follows `obs[t] -> state[t] -> action[t]`; Reset-B reconstructs
  each state from `initial(1)` and the current observation only. Expert
  actions never advance the environment.
- `run_offline_replay` / `evaluate_offline_replays` replay `record.actions[t]`
  and record Actor predictions without passing them to `env.step`
  (`actor_actions_replayed=false`).
- Closed-loop and offline trajectory records include `first_error_t`,
  predicted/expert actions, `first_error_phase`, `first_error_before_handoff`
  and `steps_before_first_error`; each mode has a `first_error_summary`.
  Validation logging also reports full training-set loss/accuracy
  (`train_metrics` in `metrics.jsonl`, `final_train` in `report.json`).

CLIs are `cmb_minigrid.scripts.train_policy`, `evaluate_policy`,
`evaluate_policy_offline` and `diagnose_policy_errors`. Configs are
`configs/train_policy.yaml`, `configs/train_policy_unweighted.yaml` and
`configs/train_policy_sqrt.yaml`. Tests are `tests/test_policy.py` and
`tests/test_policy_offline.py`. Step 5 checkpoints, trajectory data, split,
observation contract, environment and Bridge code were not modified.

The public `evaluate_policy` CLI is fixed to the test split. Training-split
evaluation was run through an inline Python heredoc that selects
`manifest.assignment == "train"` and calls `run_closed_loop`; a `--split`
option is needed before repeated training-split evaluation in later steps.

### Final artifacts and measured acceptance

The selected frozen components are:

| Role | Path | Notes |
| --- | --- | --- |
| RSSM-A | `models/best_a.pt` | Step 5 checkpoint, unchanged |
| RSSM-B | `models/best_b.pt` | Step 5 checkpoint, unchanged |
| Actor-A | `models/policy_a.pt` | user-provided |
| Actor-B | `models/policy_b.pt` | user-provided; square-root inverse-frequency weighting, feature 512, hidden 128, `best_step=50`, `best_val=0.7915889` |

The Actor-B SHA-256 is
`6a8294a722cf1b34073f8dedefb078dc0369020083f3a09d30e7909cf3950d63`.
Data are `data/samples/step3_trajectories.jsonl` with
`data/samples/step3_manifest.json`: map family v1 is test, v2 validation and
v3/v4/v5 train, with 20 expert episodes per family. The branch decision is at
`handoff_t=12`. Train action counts are `forward=1120`, `left=60`,
`right=60`.

Training-split closed-loop results on 60 expert episodes:

| Mode | Episodes | Branch accuracy | Closed-loop success | First-error summary |
| --- | ---: | ---: | ---: | --- |
| A full-history | 60 | 1.000 | 1.000 (60/60) | no errors |
| Oracle-B full-history | 60 | 1.000 | 1.000 (60/60) | no errors |
| Reset-B | 60 | 0.000 | 0.000 (0/60) | all first errors at t=12 |

Reports are
`runs/step6/evaluation_models_train_20261016/closed_loop_report.json` (A mode;
its B entries use a superseded Actor-B) and
`runs/step6/evaluation_policy_b_retest_train_20261016/closed_loop_report.json`
(current Actor-B, Oracle-B and Reset-B). The Oracle-B minus Reset-B gap is
1.0 for both branch accuracy and success on the training split. Reset-B is
correct through the shared prefix and fails only at the handoff, so the
remaining failure is the history-dependent branch choice that a Bridge must
transmit.

### Verification and boundaries

The Oracle/Reset gate holds only on the training split. On the 20 test
episodes the earlier weighted Actors gave 0% closed-loop success for A and
Oracle-B, no Oracle-B versus Reset-B gap, and `bridge_gate=hold_diagnosis`
(`runs/step6/evaluation_weighted/`). First-error and t=8..11 diagnoses
(`runs/step6/evaluation_first_error/`,
`runs/step6/policy_error_diagnosis_t8_t11/`) show that full-history Actors
make premature `left/right` turns while every expert label at t=8..11 is
`forward`, and that their own early errors cause closed-loop distribution
shift; teacher-forced replay still reaches the handoff action correctly. This
is an Actor generalization failure, not missing color information in the
Step 5 B state. Training-split Bridge results therefore establish in-
distribution cue transmission only, and evaluating on the training split is a
change from the test-split protocol in `AI_TASKS.md` that should be recorded
when Bridge results are reported.

Actor training overfits quickly: all formal variants selected step 50 as
best validation checkpoint and validation loss then increased. High offline
teacher-forced accuracy was not sufficient evidence of a usable closed-loop
Actor. A square-root Actor-B test run reported Oracle-B success `1.0` with
branch accuracy `0.0`; this success/branch metric-contract inconsistency is
undiagnosed and should be checked before reusing these metrics for Bridge
evaluation.

The last full checks were `88 passed`, Ruff clean and `git diff --check`
clean; after the weighting ablation only the focused weighting tests were run
(`5 passed`). The full suite, Ruff and `git diff --check` were not rerun.
GPU/mixed-precision training, Actor resume, multi-seed policy sweeps and
validation/test evaluation of the current `models/policy_a.pt` and
`models/policy_b.pt` remain unrun.

## Step 7 — Handoff dataset and baseline initializers (2026-10-20)

### Files

New: `cmb_minigrid/handoff/{__init__,dataset,initializers}.py`,
`cmb_minigrid/scripts/build_handoff.py`, `tests/test_handoff.py`. No existing
file was modified; the Step 3 data, split, Step 5 RSSMs, Step 6 Actors and the
observation contract are unchanged. Environment: conda `cmb_minigrid`
(Python 3.11.17, torch 2.14.1, minigrid 3.2.0), run with
`env -u PYTHONPATH` because the shell exports a ROS `PYTHONPATH`.

### Implementation and interfaces

- `HandoffSample` (one per record with `env_snapshot`, all roles/splits):
  `history` (`HandoffHistory`: `obs[0:t+1]`, `action[0:t]`, `.obs_t`),
  `future` (`HandoffFuture`: `obs[t+1:]`, `action[t:]`, expert actions,
  rewards, flags), `env_snapshot`, `pair_id`, `split`, privileged
  `hidden_label` (evaluation only), `state_a`, `state_b_star` (batch-1
  `RSSMState`, both already fused with `obs[t]`). `obs_t` is also stored
  separately and checked against the history tail on load.
- `filter_history(model, history, sample=False, seed=None)` is the single
  filtering path: `initial(1)`, first frame with no previous action, then one
  `observe_step` per frame. Both A and B are filtered from the public prefix
  only (B never reads the simulator map or hidden label). Each record's
  snapshot is checked with `replay_from_handoff` at build time.
- Initializers take only `HandoffInput(history, state_a[, oracle_state_b])`;
  the future is never passed. `ReplayInit(B, N)` filters the last `N` frames
  (clipped at `obs[0]`) from `initial(1)`; the action before the window is not
  read. `ResetInit` is `N=1`. `OracleInit` returns the stored `state_b_star`
  (the only method given it). `MapInit` wraps `StateMap('linear'|'mlp')`:
  `features(state_A)` -> tanh deter + raw logits -> B's own `_logits`
  (unimix + log-softmax) -> hard one-hot via `RSSM.sample`, so the discrete
  state is always a legal one-hot with normalized logits.
  `validate_b_state` enforces shape, finiteness, deter in [-1,1], exact
  one-hot and normalized logits.
- `initialize_all(methods, sample)` runs every method from the same sample and
  returns `HandoffInit(method, state_b, t, env_snapshot, b_fused_frames)`.
  `continue_from_handoff` does `observe_step(state[t], action[t], obs[t+1])`;
  `restore_env(sample)` rebuilds the simulator from the snapshot and checks it
  reproduces `obs_t`.
- Linear/MLP are fitted on train-split states only: every step `t'` of every
  train episode gives an aligned `(features(state_A[t']), state_B*[t'])` pair
  (`collect_aligned_states`); loss = MSE(deter) + KL(B* posterior || mapped
  categorical); full-batch Adam, seeded. The MLP optionally has a bottleneck
  (default 64 = 4x16 payload) and `--mlp-target-params` picks the hidden width
  (`matched_hidden_dim`).
- CLI: `python -m cmb_minigrid.scripts.build_handoff --checkpoint-a
  models/best_a.pt --checkpoint-b models/best_b.pt --output-dir <dir>`; writes
  `handoff_dataset.pt`, `linear_map.pt`, `mlp_map.pt`, `handoff_report.json`.

### Verified

- `pytest tests/test_handoff.py`: 13 passed (exit 0). Covers: required fields
  and dataset round-trip; Replay-1/4/8/16 call `observe_step` exactly on the
  last N frames with the expected first-frame/no-previous-action and
  `action[i-1]` usage (spied); poisoning the stored future leaves every
  method's state bit-identical; mutating post-handoff observations in the
  source record leaves `state_A/state_B*` unchanged; full Replay and Replay-16
  equal the stored Oracle-B (deterministic) and seeded sampling is reproducible
  without touching the global RNG; `obs_t` is fused once (last frame) and
  `continue_from_handoff` equals filtering `obs[0:t+2]`, while re-fusing
  `obs_t` visibly changes the state (guard is sensitive); all methods keep the
  same snapshot/`t`, repeat exactly, and the restored env reproduces the true
  future observations; A->B maps emit legal one-hot/logits even with
  x50-scaled weights, and a raw-float state is rejected; map fitting lowers
  loss; MLP width matching.
- CLI on the real checkpoints (`runs/step7/handoff_20261020`, seed 42, exit
  0): 200 samples (train 120 / val 40 / test 40, all `handoff_t=12`, none
  skipped, all snapshots rebuilt). All methods repeatable and same-snapshot;
  max |deter| difference between replay_full and stored Oracle-B is 0.0; 50
  expert pairs have identical `obs_t` but differing states (min A-state
  distance 12.1, min B*-state distance 16.4).
  Mean deter L2 to Oracle-B (val / test): reset 11.08/11.08, replay-4
  9.56/9.58, replay-8 9.26/9.09, replay-16 0/0, linear 0.89/0.43, MLP
  1.15/0.75. Categorical-mode exact match with Oracle (val/test): linear
  0.5/1.0, MLP 0.0/1.0. These are state-distance diagnostics only; no action or
  closed-loop result is claimed. `--mlp-target-params 100000` smoke run: hidden
  97, 100,098 parameters, exit 0.
- Full suite: 102 passed, 3 failed. Ruff clean on the new files; `git diff
  --check` clean.

### Unresolved / not run

- The 3 failures predate this step (`test_categorical_response` x2 expect
  exactly 0 deterministic difference, got 2.98e-8; `test_training` frozen-probe
  test requires `atol=0` equality across batch rows). They look like batch-size
  float noise under torch 2.14.1 and were not changed. This also means
  Oracle/Replay equality is only claimed for the same batch-1 code path.
- `handoff_t` is 12 everywhere, so Replay-16 sees the whole prefix and is
  identical to Oracle by construction; Replay-4/8 are the informative windows.
- "Capacity-matched" is undefined until the CMB exists (Step 8): the MLP
  default (hidden 128, bottleneck 64, 131,904 params) vs Linear (197,120) are
  not matched to anything yet; use `--mlp-target-params` once the CMB size is
  fixed. Linear/MLP see only `state_A` (no B-side `obs_t` encoding).
- Linear/MLP test numbers use 40 samples from one test family and a single
  seed; no CIs. Not run: GPU, multi-seed, stochastic-mode dataset build, Actor-B
  action evaluation from these states (Step 9).
- Untracked `~`, `probe_report.json`, `categorical_response_*.json` in the repo
  root predate this step and were left alone.

### Next-step dependencies

Step 8 needs `handoff_dataset.pt` (`state_a`, `state_b_star`, `obs_t`,
`future`) and `initialize_all` baselines; Step 9 needs `restore_env`,
`continue_from_handoff` and the same-snapshot contract.


## Step 8 — CMB Bridge training and Reset comparison (2026-10-20)

### Files

New: `cmb_minigrid/bridge/{__init__,cmb,training}.py`,
`cmb_minigrid/scripts/train_cmb.py`, `tests/test_cmb.py`. Modified:
`configs/cmb.yaml` (added `sender_hidden`/`receiver_hidden`; existing
`tokens: 4`, `token_classes: 16`, `encoder_dim: 64` unchanged). RSSM-A/B,
Actor-B, the Step 3 data/split, observation contract and Step 7 code are
unchanged. The top-level stub `bridge/__init__.py` is unused.

### Implementation and interfaces

- `CMBBridge(CMBConfig)`:
  - `send(features_a)` -> `CMBMessage(logits, onehot)`; Sender is an MLP from
    `features(state_A[t])` (384) to 4 tokens x 16 classes, a **discrete**
    message (straight-through one-hot; argmax in eval, sampling optional).
    Payload is 16 bits (2 bytes packed).
  - `receive(message, obs_t, rssm_b)` concatenates the flattened message, frozen
    B's 64-dim `encoder(obs_t.image)` and the public direction one-hot (+4;
    B's own posterior also gets direction), then an MLP outputs 256 deter
    (tanh, B's GRU range) and 16x16 logits (via B's `_logits`: unimix +
    log-softmax) and a hard one-hot sampled with `rssm_b.sample`. This is
    `state_B_hat[t]`, already conditioned on `obs[t]`; B's `observe_step` is not
    called on `obs[t]` (a test fails if it is). Frozen B is passed as an argument
    and is not in the bridge checkpoint. `save_cmb`/`load_cmb`, 140,608
    parameters (Sender 57,536, Receiver 83,072).
- `train_cmb` (CLI `python -m cmb_minigrid.scripts.train_cmb`): A, B and Actor-B
  are `requires_grad=False`. Student path CMB -> `state_B_hat` -> Actor-B /
  `imagine_step` is differentiable through the frozen modules; the Oracle
  teacher (`state_B_star`, its Actor-B logits, its `imagine_step` rollout) is
  detached / `no_grad`.
  - Stage 1 loss: MSE(deter) + KL(B* posterior || hat posterior) + KL(Actor-B
    action distribution from B* || from hat).
  - Stage 2 adds multi-step distillation: same recorded `action[t:t+K]` through
    `imagine_step` for teacher and student, per-step deter MSE + prior KL, masked
    for episodes shorter than K, averaged for K=5 and K=10 then averaged.
  - Real future observations are never a training signal; they are used only to
    evaluate decoded imagined states (image CE) against `obs[t+k]`.
  - Defaults: Adam lr 1e-3, batch 32 of 120 train samples, 300+300 updates,
    validation every 25, patience 8, best-val selection per stage (stage 2 starts
    from the selected stage-1 weights). Each update checks finite loss and
    nonzero Sender and Receiver gradient norms and logs them in
    `metrics.jsonl`; checkpoints `stage1_best.pt`, `stage1_selected.pt`,
    `best.pt`; `report.json` holds val/test metrics for Reset, Replay-4/8,
    Linear, MLP, CMB and Oracle-B on identical samples, a per-metric
    `cmb_not_worse` verdict vs Reset, and the overhead table.
- Equal-capacity MLP: Step 7 MLP maps re-fitted with
  `build_handoff --mlp-target-params 140608` (hidden 136, 64-float bottleneck,
  140,112 parameters; `runs/step7/handoff_matched_20261020`, dataset bitwise
  identical to the Step 7 one). This MLP maps `state_A` only (no `obs_t` encoding),
  is trained with the Step 7 all-timestep MSE+KL objective, not the Step 8 action
  / multi-step losses, so "equal capacity" means parameter count and bottleneck
  width, not an identical training recipe.

### Overhead (from `report.json`)

| Method | Parameters | Payload |
| --- | ---: | --- |
| CMB (4x16 discrete) | 140,608 | 16 bits (2 bytes packed; 256 bytes as float one-hot) |
| MLP (matched) | 140,112 | 64 floats = 256 bytes |
| Linear | 197,120 | 384 floats = 1,536 bytes |

### Verified

- `pytest tests/test_cmb.py`: 10 passed. Covers: `state_B_hat` legal and
  consumed by Actor-B, `imagine_step` and `observe_step(obs[t+1])`; Receiver
  never calls `observe_step`; nonzero finite Sender/Receiver gradients in both
  stages, and from the multi-step term alone, with no gradient on A/B/Actor and
  none on a teacher tensor that requires grad; changing real future observations
  leaves the training loss identical; the message affects `state_B_hat` for equal
  `obs_t`; checkpoint round trip; single-batch overfit and seed reproducibility;
  overhead counts; `evaluate_methods` and a 3+3-step end-to-end training run.
- Smoke CLI (`--smoke`, 20+20 updates, exit 0, `runs/step8/smoke_20261020`) and
  the formal run
  `python -m cmb_minigrid.scripts.train_cmb --handoff-dataset
  runs/step7/handoff_matched_20261020/handoff_dataset.pt --checkpoint-a
  models/best_a.pt --checkpoint-b models/best_b.pt --policy-b models/policy_b.pt
  --linear-map .../linear_map.pt --mlp-map .../mlp_map.pt --output-dir
  runs/step8/formal_20261020 --seed 42` (exit 0; seeds 43 and 44 also exit 0,
  `runs/step8/formal_seed43_20261020`, `formal_seed44_20261020`). Actor-B
  SHA-256 `6a8294a7...` (unchanged).
- Full suite: 112 passed, 3 failed (same 3 pre-existing failures as Step 7); Ruff
  and `git diff --check` clean.

### Results (val = family v2, test = family v1, 40 samples each, handoff t=12)

Seed 42 (best stage-2 step 525), test split:

| Metric | Reset | Linear | MLP | CMB | Oracle-B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Handoff action acc vs expert | 0.00 | 0.50 | 0.00 | 1.00 | 1.00 |
| Action KL to Oracle | 14.46 | 0.68 | 2.66 | 0.013 | 0 |
| Future image CE, K=10 | 0.282 | 0.343 | 0.349 | 0.311 | 0.351 |
| Imagined deter drift to Oracle, K=10 | 9.76 | - | - | 1.06 | 0 |

CMB seeds on test (acc / KL / drift K=10): 42: 1.00 / 0.013 / 1.06; 43: 1.00 /
0.005 / 1.44; **44: 0.50 / 0.944 / 1.50**. Seed variance is therefore large and
one of three CMB training seeds recovered only half of the handoff actions.

### Acceptance and honest assessment

- Passed: `state_B_hat` is consumed by Actor-B and the RSSM; nonzero
  backpropagating gradients; the student branch is differentiable through frozen
  B and only the Oracle teacher is `no_grad`.
- "Not worse than Reset" is **mixed, not a clean pass**:
  - Test, seeds 42/43: action accuracy, action KL and imagined-state drift beat
    Reset by a wide margin, but **future image CE is worse than Reset for every
    seed** (0.31 vs 0.28 at K=10; 0.39 vs 0.29 at K=5), so that criterion fails.
  - The Oracle-B state itself has worse future image CE (0.35-0.47) than Reset
    (0.27-0.29), so image CE does not measure memory here and a student that
    imitates Oracle inherits this; Replay-4 has the lowest CE (0.24).
  - Val: Actor-B with the Oracle state picks the wrong handoff action on all val
    samples (accuracy 0.0), so Actor-B does not generalize to family v2 and
    "action KL to Oracle" on val distills a wrong teacher. CMB's val KL (4.44)
    is worse than Reset's (1.53) while its accuracy vs expert is 1.0 (it
    disagrees with the faulty Oracle). Val loss was also very noisy (about 3 to
    28 between checks), so best-checkpoint selection is unreliable.
- Handoff action accuracy is binary-ish: the expert handoff action is a turn
  while Reset predicts forward, hence Reset's 0.0 on both splits.
- Nothing here shows a shared, model-independent space: the bridge is a learned
  bottleneck mapping for one specific A/B pair on one task with a one-bit cue.
  History Swap and closed-loop evidence belong to Step 9.

### Unresolved / not run

- Pre-existing failing tests (see Step 7) not touched.
- Only 120 train handoff states (t=12 only) -> overfitting/noise; no
  augmentation across other `t`, no learning-rate or loss-weight sweeps, no
  early-stopping variant that avoids the noisy val criterion.
- Three bridge seeds with the same RSSM/Actor checkpoints; no CIs, no
  RSSM-seed variation (Step 10). The MLP baseline is not trained with the Step 8
  losses; a recipe-matched MLP bridge is not run.
- Not run: GPU, resume-from-checkpoint of `train_cmb`, closed-loop control
  (Step 9), stochastic message sampling (`sample_message`), capacity ablations.

### Next-step dependencies

Step 9 can load `models`-style checkpoints with `load_cmb`
(`runs/step8/formal_20261020/best.pt`; note seed 44 shows instability), use
`CMBBridge.send`/`receive` separately for History Swap (messages are `[B,4]`
integer indices via `CMBMessage.indices`), and the Step 7
`restore_env`/`continue_from_handoff` helpers.


## Step 9 — closed-loop takeover, History Swap and result tables (2026-10-20)

### Files

New: `cmb_minigrid/handoff/runner.py`, `cmb_minigrid/scripts/run_handoff.py`,
`tests/test_handoff_runner.py`. No existing code, data, split, observation
contract, RSSM, Actor or bridge checkpoint was modified.

### Implementation and interfaces

- `HandoffRunner(rssm_a, rssm_b, policy, methods, stochastic=False, seed=0)`:
  - `run_sample(sample, method)` restores a fresh `HiddenHistoryNavEnv` from
    the sample's handoff snapshot, checks `env.snapshot() == sample.env_snapshot`,
    builds `state_B[t]` with the method (via `initialize_all`, so it is
    validated as a legal B state), then loops: Actor-B picks `action[t]` from
    `features(state)`, the real env steps, `obs[t+1]` is fused into B with
    `observe_step(state, action[t], obs[t+1])`. `obs[t]` is never fused again.
  - `run(samples)` runs every sample x method. `stochastic=True` samples B's
    discrete states under a per-run seeded RNG (`seed*1_000_003 + sample index`)
    scope; default is deterministic argmax.
- `CMBInit` wraps a trained `CMBBridge`: Sender on `features(state_A[t])`,
  Receiver on B's encoding of `obs_t`. Methods: `reset`, `replay_4/8/16`
  (`replay_16` = full history because `t=12`), `linear`, `mlp`, `cmb`, `oracle_b`.
- Per-row record: ids, split, family, method, `snapshot_sha`, `obs_t_sha`,
  `snapshot_restored_equal`, first action vs expert handoff action
  (`first_branch_correct`), `success` (Goal reached before the 64-step limit),
  `truncated`, steps after handoff, `wrong_actions` (action differs from the
  privileged BFS planner's action at the current pose), `collisions`
  (forward that did not move the agent), `optimal_steps` (planner steps from the
  snapshot), `path_efficiency` (`optimal/steps` if successful else 0), reward,
  `payload_bytes`, `b_observe_steps_at_takeover` (frames conceptually fused to
  build the state; the cached Oracle state makes no real B call), `takeover_ms`
  and `closed_loop_ms` (wall-clock, single CPU thread, noisy). The planner and
  hidden label are used only for scoring.
- Payload accounting (what A's side hands over, assumed byte widths): Reset 0;
  Replay-N `w*148 + (w-1)` bytes (uint8 7x7x3 image + direction per frame plus
  actions, `w=min(N,t+1)`); Linear 384 float32 = 1,536; MLP 64-float bottleneck
  = 256; CMB 16 bits = 2 bytes packed; Oracle-B full history = 1,936 (unconstrained).
- `run_history_swap(runner, cmb, samples)`: for each expert pair (equal `obs_t`,
  different early cue) and each ordered (own, donor), three conditions are run in
  the *own* endpoint's freshly restored snapshot with the *own* `obs_t` passed to
  the Receiver; only the message differs: `own`, `swap` (donor's message) and
  `same_cue` (message from another pair with the same hidden label in the same
  family). Rows record snapshot/obs hashes, messages, first action, outcome,
  `flipped_vs_own` and `follows_message_cue`.
- Outputs of `python -m cmb_minigrid.scripts.run_handoff`: `handoff_runs.jsonl/.csv`,
  `main_table.csv` (rows for scope `family`, `split`, `all`), `history_swap_rows.csv`,
  `history_swap_summary.csv`, `report.json` (args, hashes, versions, tables,
  integrity flags). The script checks that the dataset and each CMB checkpoint
  were built from the given A/B (and Actor-B for the CMB) hashes. Default roles
  `expert` only: exploration records share prefix and snapshot with their expert,
  so their closed loop would just duplicate rows.

### Verified

- `pytest tests/test_handoff_runner.py`: 8 passed. Covers: identical snapshot
  hash/`obs_t` hash/restored-equal across all 8 methods; spied `observe_step`
  calls (takeover frames are exactly the last N prefix frames with the first
  carrying no action; the first fusion after takeover is `obs[t+1]` reached by
  `action[t]`; call counts equal takeover + steps - 1, so `obs_t` is not re-fused);
  exact reproduction of rows (excluding timing) for deterministic and seeded
  stochastic runs; a forward-only stub times out at 52 steps with 52 collisions
  and zero efficiency, an expert-action stub gets success, 0 wrong actions,
  efficiency 1.0; payload bytes; table/CSV consistency; History Swap with an
  index-tagged stub bridge (swap flips to the donor's action, same-cue control
  does not, snapshot and `obs_t` hashes identical in every condition, Receiver
  receives only the shared `obs_t`) and with a real `CMBBridge`.
- CLI smoke (6 test samples, exit 0, `runs/step9/smoke_20261020`) and formal run:
  `python -m cmb_minigrid.scripts.run_handoff --handoff-dataset
  runs/step7/handoff_matched_20261020/handoff_dataset.pt --checkpoint-a
  models/best_a.pt --checkpoint-b models/best_b.pt --policy-b models/policy_b.pt
  --cmb-checkpoint runs/step8/formal_20261020/best.pt
  runs/step8/formal_seed43_20261020/best.pt runs/step8/formal_seed44_20261020/best.pt
  --linear-map .../linear_map.pt --mlp-map .../mlp_map.pt --output-dir
  runs/step9/formal_20261020_a` (exit 0; 100 expert samples x 10 methods = 1000
  rows). A second identical run (`_b`, exit 0) gave identical rows once the two
  timing columns are dropped. `all_snapshots_restored_equal=True`,
  `history_swap_integrity=True`.
- Full suite: 120 passed, 3 failed (the same 3 pre-existing failures); Ruff and
  `git diff --check` clean.

### Results (expert samples; test = family v1, val = v2, train = v3-v5)

Test split, n=20 per method:

| Method | First-branch acc | Success | Mean wrong actions | Mean collisions | Path eff. (all) | Payload B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reset | 0.00 | 0.50 | 29.0 | 24.0 | 0.18 | 0 |
| Replay-4 | 0.00 | 0.50 | 27.0 | 22.0 | 0.21 | 595 |
| Replay-8 | 0.00 | 0.50 | 25.0 | 19.5 | 0.30 | 1,191 |
| Replay-16 (= full) | 1.00 | 1.00 | 0.0 | 0.0 | 1.00 | 1,936 |
| Linear | 0.50 | 1.00 | 4.0 | 3.0 | 0.55 | 1,536 |
| MLP (matched) | 0.00 | 1.00 | 3.0 | 2.5 | 0.68 | 256 |
| CMB seed 42 | 1.00 | 1.00 | 0.0 | 0.0 | 1.00 | 2 |
| CMB seed 43 | 1.00 | 1.00 | 1.5 | 1.0 | 0.80 | 2 |
| CMB seed 44 | 0.50 | 1.00 | 3.5 | 3.0 | 0.71 | 2 |
| Oracle-B | 1.00 | 1.00 | 0.0 | 0.0 | 1.00 | 1,936 |

Per-family and per-split rows for all three splits are in `main_table.csv`.
Train split (in distribution): CMB seeds 42/43/44 first-branch 1.00/1.00/0.67,
success 1.00/0.83/0.67; Oracle 1.00/1.00.

History Swap (all conditions use the own snapshot and own `obs_t`):

| CMB seed | Split | `own` first-branch | `swap` flips first action | `swap` follows donor cue | `swap` success | `same_cue` flips |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | test/val/train | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| 43 | test/val/train | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| 44 | test/val | 0.50 | 1.00 | 0.50 | 0.00 | 0.00 |
| 44 | train | 0.67 | 1.00 | 0.67 | 0.00 | 0.00 |

### Interpretation and limits

- Closed-loop **success is not a memory measure here**: the wrong branch is a
  visible dead end and Actor-B can sometimes recover, so Reset reaches the goal
  in 50% of test episodes while its first handoff action is never the correct
  turn. First-branch accuracy, wrong actions and path efficiency are the more
  diagnostic columns. Replay-4/8 do not help (their window never contains
  `obs[0]`/`obs[1]`, which hold the cue); Replay-16 equals Oracle by construction.
- On val the Oracle-B first-branch accuracy is 0.00 (Actor-B does not
  generalize to family v2, as in Step 8), although its success is 1.00 through
  recovery; the val Oracle is therefore not a valid ceiling, and CMB's val
  first-branch 1.00 (seeds 42/43) disagrees with that faulty Oracle.
- History Swap shows that the message is **causally used**: with the physical
  state and `obs_t` fixed, exchanging only the message flips Actor-B's handoff
  action in 100% of cases for all three bridge seeds, same-cue messages from other
  histories do not, and swapped runs never succeed. It does not show *what*
  the message encodes beyond the binary early cue, nor a model-independent
  shared space (one A/B pair, one task, one-bit cue). Seed 44 flips actions
  but often to the wrong one, i.e. its own message is correct only half the time.
- Timing columns are wall-clock on one CPU thread and are noisy; no confidence
  intervals or multi-RSSM-seed statistics yet (Step 10).

### Unresolved / not run

- 3 pre-existing test failures untouched (see Step 7).
- Not run: GPU; the stochastic (`--stochastic`) mode on the real CLI (only
  unit-tested); null/zero-message and other ablations, capacity and history-gap
  sweeps, different RSSM seeds (Step 10); exploration-role handoffs; closed loop
  with Actor-A; History Swap for Linear/MLP baselines.
- Payload byte widths are stated assumptions, not measured transfers.

### Next-step dependencies

Step 10 can reuse `HandoffRunner`, `run_history_swap` and the CSV writers; the
multi-seed requirement means training several RSSM-A/B pairs (new Step 5-8
artifacts), which has not been done.


## Step 10 — statistics, ablations and report (pilot) (2026-10-20)

### Files

New: `cmb_minigrid/experiments/{__init__,stats,svg}.py`,
`cmb_minigrid/scripts/{run_pilot,make_report}.py`, `tests/test_experiment_stats.py`,
`tests/test_experiment_svg.py`, `tests/test_make_report.py`,
`reports/{experiment_report.md,analysis.md,config_manifest.json,repro_commands.sh,tables/,figures/}`.
Modified (backward compatible): `cmb_minigrid/bridge/training.py` (`train_cmb`
accepts `state_weight/action_weight/multistep_weight`; a stage with 0 steps is
skipped; `best.pt` is written after every executed stage), `scripts/train_cmb.py`
(`--tokens --token-classes --state-weight --action-weight --multistep-weight`),
`handoff/runner.py` (History Swap gains a `zero` condition and `path_efficiency`),
`scripts/run_handoff.py` (`--cmb-names`), `tests/test_cmb.py`,
`tests/test_handoff_runner.py`. Data, split, observation contract, Step 5/6 recipes
and configs were not changed.

### Implementation

- `run_pilot`: per RSSM seed group runs existing CLIs as subprocesses (resumable,
  commands/exit codes/seconds in `runs/step10/manifest.json`): `train_rssm` A
  (`rssm_a.yaml` + `train_rssm_cnn.yaml`, seed 1000+g) and B (`rssm_b.yaml`, seed
  2000+g) with 1000 updates and no practical early stopping for both (the pre-existing
  A used patience 5), `probe_rssm`, `train_policy` (`train_policy_sqrt.yaml`, seed
  2000+g), `build_handoff` (MLP matched to the 4x16 CMB's 140,608 parameters),
  six `train_cmb` variants (full 4x16, capacity 2x16 / 8x16 / 8x32, loss: state-only,
  state+action), and `run_handoff` with Replay windows {2,4,6,8,10,12,13,16}, Linear,
  MLP, all CMB variants, Oracle and History Swap (own / swap / zero / same-cue).
  Group `g0` = the pre-existing checkpoints (`models/best_a.pt`, `best_b.pt`,
  `policy_b.pt`). One bridge seed (42) per group.
- `stats.py`: clustered bootstrap, unit = pair (both endpoints together) nested in
  map family nested in RSSM seed group; groups resampled with replacement, then
  families (if >1) and pairs; statistic = mean over groups of the pair mean;
  percentile 95% CI, 4000 replicates, seed 12345; paired differences on aligned
  units. `svg.py`: dependency-free forest plots (matplotlib is not installed), palette
  slots from the validated reference palette (`validate_palette.py`: pass; the aqua
  slot is unused), per-seed dots, tooltips and numeric labels.
- `make_report`: reads only run artifacts; writes `tables/*.csv` (methods per scope and
  split, paired differences, gates, message conditions, overhead), eight SVG figures,
  `config_manifest.json` (versions, seeds, data/config hashes, per-run configs),
  `repro_commands.sh` and `experiment_report.md` (auto tables + hand-written
  `analysis.md`). Operational gates (mine): probe full-history val accuracy >= 0.9 for
  A and B; Oracle-B minus Reset-B first-branch accuracy >= 0.5 on test.

### Verified

- New tests: stats 6, SVG 1, make_report 1 (synthetic tree with known values: estimates,
  CIs, paired difference, gates, swap/zero rows, hand-written analysis inclusion,
  manifest/commands), plus loss-ablation (`train_cmb` with `action_weight=0`,
  `stage2_steps=0`) and `zero` History Swap tests. Full suite: **130 passed, 3 failed**
  (the same 3 pre-existing failures); Ruff and `git diff --check` clean.
- Smoke pipeline (`run_pilot --root runs/step10_smoke --groups 1 2 --smoke`, exit 0,
  3 min) and formal pilot: `python -m cmb_minigrid.scripts.run_pilot --root runs/step10
  --groups 1 2 ... 15 --workers 8` (first invocation groups 1-5 plus g0: 69 jobs ok in
  948 s; the extension to groups 6-15 added jobs, in total 189 jobs, 0 failed groups,
  1,603 s for the second invocation). Report:
  `python -m cmb_minigrid.scripts.make_report --root runs/step10 --output reports
  --analysis reports/analysis.md` (exit 0). Figures were rendered with a headless
  browser and inspected for label overlap.

### Results (details and CIs in `reports/experiment_report.md`)

- Gates over 15 new groups: probe 6/15, actor 10/15, both 6/15 (g4, g8, g10, g12, g14,
  g15); Oracle-B first-branch accuracy is 1.0 in only 6/15 groups. The Step 5/6 recipes
  are not seed-robust, which caps every bridge distilled from Oracle-B.
- First 5 groups (pre-planned): CMB first-branch 0.40 [0.10, 0.70]; CMB - Reset +0.40
  [+0.10, +0.70]; vs Replay-8 +0.10 [-0.30, +0.60], MLP +0.00 [-0.30, +0.30], Linear +0.10
  [0.00, +0.30], Oracle/Replay-13 0.00 [0.00, 0.00]. All 15 groups: CMB 0.47 [0.27, 0.70],
  Oracle 0.53, Replay-13 0.53, Linear 0.33, MLP 0.33. Gate-passed 6 groups (sensitivity):
  CMB 0.92 = Oracle = Replay-13, Linear 0.67, MLP 0.50; CMB - MLP +0.42 [+0.17, +0.67].
- The CMB never exceeded Replay with a window that reaches `obs[0]` or Oracle-B; it matches
  them with 2 bytes vs 1,936 bytes. Replay-4/8 do not contain the cue. Linear/MLP
  have higher closed-loop success (0.83) than CMB/Oracle (0.75) in the gate-passed
  cohort because success includes recovery from the wrong-branch dead end.
- Ablations: no detectable capacity effect beyond 8 bits (all-15 first-branch 0.47 /
  0.47 / 0.47 / 0.50); first-branch identical across loss variants, small inconsistent
  path-efficiency differences; swapped/zeroed message drop first-branch to 0.10 / 0.13
  (all 15) and 0.00 / 0.00 (gate-passed) while a same-cue message from another history
  changes nothing.
- Evidence boundaries (also in the report): single cue transmitted — supported where A
  carries the cue; sustained closed-loop recovery — not established (even Oracle-B has
  success 0.75 and path efficiency 0.53, Actor-B errs after the branch); shared cross-model
  space — not tested.

### Not done / unresolved

- **History-gap ablation {8,16,32,64} not run**: the gap is fixed at 12 by the environment
  (`HANDOFF_T`); other gaps need new map families (task-definition change requiring your
  approval), new data, retrained RSSMs and a larger `max_steps` (currently 64).
- The cohort was extended from 5 to 15 groups after seeing 1/5 pass the gates (post-hoc,
  disclosed in the report); the gate-passed analysis is selected on Oracle-B performance.
  Dozens of paired comparisons, no multiplicity correction. Single test map family (10 pairs
  per group), one bridge seed per group. The Step 5/6 recipe was not tuned to fix the seed
  fragility.
- Reproducibility caveat: `git_commit` in the manifest is HEAD (the Step 7 commit) because
  the Step 8-10 code was uncommitted when the pilot ran; commit before relying on the hash.
  Bitwise cross-machine equality is not claimed.
- Not run: GPU, bridge-seed variance within the new groups, val/train-based ablation figures
  (tables only), Step 11. The 3 pre-existing test failures remain.


## Research report (2026-10-20)

Wrote `reports/research_report.md`, a hand-written synthesis of Steps 3-10 (Chinese), on
top of the auto-generated `reports/experiment_report.md`. No new experiments were run for
it; new numbers in its "phenomena" section (first-action modes, probe-vs-Oracle
correspondence, per-family Oracle accuracy, Actor-B behavior on family v3) were computed
by re-reading `runs/step10/*/closed_loop/handoff_runs.jsonl` and
`runs/step10/*/probe/probe_report.json`. Hypotheses that were not tested are labeled as such
in the report (e.g. why imagined rollouts from full-history states decode worse than from
the Reset state; why the cue representation fails to transfer to the held-out family).
Report-level caveats repeated there: post-hoc cohort extension, CMB vs Linear/MLP recipe
mismatch, Step 6 vs Step 9 meaning of "Reset", history-gap ablation not run.
