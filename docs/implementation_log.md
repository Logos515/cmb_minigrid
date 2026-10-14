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
