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
