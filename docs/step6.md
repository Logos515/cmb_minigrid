## Step 6: frozen-RSSM Actor-B and Oracle/Reset evaluation (2026-10-14)

### Changed files and interfaces

- Added `cmb_minigrid/models/policy.py` with `ActorPolicy`, a three-action
  behavior-cloning head consuming an RSSM feature and a fixed public goal token.
- Added `cmb_minigrid/training/policy.py` with detached posterior feature
  extraction, frozen-RSSM Actor training, checkpoint loading, and the three
  closed loops `a_full_history`, `oracle_b_full_history`, and `reset_b`.
  Full-history filtering follows `obs[t] -> state[t] -> action[t]`; Reset-B
  reconstructs each state from `initial(1)` and the current observation only.
- Added `cmb_minigrid/scripts/train_policy.py`,
  `cmb_minigrid/scripts/evaluate_policy.py`, and
  `configs/train_policy.yaml`.
- Added `tests/test_policy.py`. Step 5 checkpoints, trajectory data, split,
  observation contract, environment, and Bridge code were not modified.

Training uses train/val expert records and offline `expert_actions` labels.
RSSM parameters are frozen and features are detached before Actor optimization.
Inverse-frequency train-label weighting is enabled by default because the
actual train distribution is `{forward: 1120, left: 60, right: 60}`. Test data
is used only by closed-loop evaluation.

### Actual commands and results

Commands ran from the repository root with Python 3.11.17 in `cmb_minigrid`,
CPU float32 and one thread. These commands returned exit code 0:

```bash
/data2/xuyihao/envs/cmb_minigrid/bin/python -m pytest -q tests/test_policy.py
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.train_policy --rssm-checkpoint models/best_a.pt --config configs/train_policy.yaml --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/policy_a_weighted_smoke --device cpu --mode smoke --seed 42
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.train_policy --rssm-checkpoint models/best_b.pt --config configs/train_policy.yaml --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/policy_b_weighted_smoke --device cpu --mode smoke --seed 45
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.train_policy --rssm-checkpoint models/best_a.pt --config configs/train_policy.yaml --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/policy_a_weighted_formal --device cpu --mode formal --seed 42
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.train_policy --rssm-checkpoint models/best_b.pt --config configs/train_policy.yaml --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/policy_b_weighted_formal --device cpu --mode formal --seed 45
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.evaluate_policy --checkpoint-a models/best_a.pt --checkpoint-b models/best_b.pt --policy-a runs/step6/policy_a_weighted_formal/best.pt --policy-b runs/step6/policy_b_weighted_formal/best.pt --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/evaluation_weighted --device cpu
/data2/xuyihao/envs/cmb_minigrid/bin/python -m pytest -q
/data2/xuyihao/envs/cmb_minigrid/bin/ruff check cmb_minigrid tests
git diff --check
```

Focused tests: `4 passed in 3.66s`. Full suite: `88 passed in 22.57s`.
Ruff reported `All checks passed!`; `git diff --check` returned exit code 0.
Both weighted smoke runs executed exactly 100 updates. A reached best
validation loss `0.842617` at step 50 and B `0.992652` at step 50; final
unweighted validation accuracies were `0.7778` and `0.6944`. Weighted formal
runs stopped at step 300 after validation patience was exhausted, with best step 50.
Their final ordinary validation metrics were A loss `1.785382`, accuracy
`0.8611`, and B loss `2.565033`, accuracy `0.7778`. These are action metrics,
not task-success guarantees.

### Test-split closed-loop result

The evaluator processed 20 grouped test expert episodes and wrote
`runs/step6/evaluation_weighted/closed_loop_report.json` plus per-trajectory
records:

| Mode | Branch accuracy | Success rate | Episodes |
| --- | ---: | ---: | ---: |
| A full-history | 50% (10/20) | 0% (0/20) | 20 |
| Oracle-B full-history | 0% (0/20) | 0% (0/20) | 20 |
| Reset-B | 0% (0/20) | 0% (0/20) | 20 |

Oracle-B minus Reset-B was `0.0` for branch accuracy and success rate. The
evaluator reported `bridge_gate=hold_diagnosis`. Step 6 therefore does not
meet the closed-loop success or Oracle-B gap acceptance. This does not
contradict Step 5's finding that B's t=12 full-history state decodes color:
the current Actor-B failed to turn that representation into the branch action.
The prior unweighted policy selected all-forward actions; inverse-frequency
weighting improved A's handoff branch accuracy to 10/20, but did not produce
successful trajectories or an Oracle-B advantage.

### Unresolved issues and next dependency

- No successful closed-loop task completion or Oracle-B versus Reset-B gap was
  observed. Bridge training is explicitly stopped.
- The test split contains only 10 examples of each handoff branch action.
  More policy seeds, checkpoint selection, or a policy design review may be
  needed, but were not run in Step 6.
- GPU, mixed-precision, Actor resume, multi-seed policy sweep, alternative
  goals, and any privileged-label policy control were not run and are not
  claimed by this log.

### Offline behavior-cloning replay follow-up (2026-10-14)

Added `run_offline_replay` and `evaluate_offline_replays` in
`cmb_minigrid/training/policy.py`, plus
`cmb_minigrid.scripts.evaluate_policy_offline`. For each selected expert
trajectory the evaluator replays `record.actions[t]` into the environment,
checks the resulting observation against the saved trajectory, filters the
frozen RSSM with the recorded history, and records the Actor prediction and
expert label at each `t`. The Actor prediction is never passed to `env.step`.
The JSON explicitly stores `actor_actions_replayed=false`, per-step rows,
episode accuracy, and handoff accuracy. The default split is test; no training
or closed-loop behavior was changed.

Changed files for this follow-up: `cmb_minigrid/training/policy.py`,
`cmb_minigrid/scripts/evaluate_policy_offline.py`,
`tests/test_policy_offline.py`, `docs/spec.md`, and this log.

Actual evaluation commands, both exit code 0:

```bash
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.evaluate_policy_offline --rssm-checkpoint models/best_a.pt --policy-checkpoint runs/step6/policy_a_weighted_formal/best.pt --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/offline_replay_a --split test --device cpu
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.evaluate_policy_offline --rssm-checkpoint models/best_b.pt --policy-checkpoint runs/step6/policy_b_weighted_formal/best.pt --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/offline_replay_b --split test --device cpu
```

Both evaluated 20 test episodes / 360 action steps and reported
`actor_actions_replayed=false`. A action accuracy was `0.666667` and handoff
accuracy `1.0`; B action accuracy was `0.722222` and handoff accuracy `1.0`.
These are teacher-forced offline metrics, not task success or closed-loop
results. They explain why an Actor can select the correct branch when supplied
the expert prefix yet still fail closed-loop after its own earlier action
deviates.

Focused verification:
`/data2/xuyihao/envs/cmb_minigrid/bin/python -m pytest -q tests/test_policy.py tests/test_policy_offline.py`
returned exit code 0, `6 passed in 5.30s`.
`/data2/xuyihao/envs/cmb_minigrid/bin/ruff check cmb_minigrid/training/policy.py cmb_minigrid/scripts/evaluate_policy_offline.py tests/test_policy_offline.py`
returned exit code 0, `All checks passed!`.

### Training-set accuracy logging follow-up

Updated `cmb_minigrid/training/policy.py` so every validation interval prints
the full training-set loss and accuracy in addition to the minibatch loss and
validation metrics:

```text
step=50 batch_loss=0.333655 train_loss=0.505935 train_acc=0.718 val_loss=0.842617 val_acc=0.611
step=100 batch_loss=0.155933 train_loss=0.322348 train_acc=0.847 val_loss=0.889019 val_acc=0.778
```

Validation events in `metrics.jsonl` now include `train_metrics`, and
`report.json` includes `final_train`. The current A CPU smoke CLI completed
100 updates with exit code 0; final training accuracy was `0.846774` on 1240
training action steps.

### First-error closed-loop diagnosis (2026-10-15)

Added first-error fields to closed-loop and offline replay trajectory records:
`first_error_t`, predicted/expert actions, `first_error_phase`,
`first_error_before_handoff`, and `steps_before_first_error`. Each mode also
has a `first_error_summary` with counts and min/mean/max first-error time.
The closed-loop report was regenerated at
`runs/step6/evaluation_first_error/`; offline reports were regenerated at
`runs/step6/offline_replay_a_first_error/` and
`runs/step6/offline_replay_b_first_error/`.

The actual evaluation commands returned exit code 0:

```bash
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.evaluate_policy --checkpoint-a models/best_a.pt --checkpoint-b models/best_b.pt --policy-a runs/step6/policy_a_weighted_formal/best.pt --policy-b runs/step6/policy_b_weighted_formal/best.pt --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/evaluation_first_error --device cpu
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.evaluate_policy_offline --rssm-checkpoint models/best_a.pt --policy-checkpoint runs/step6/policy_a_weighted_formal/best.pt --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/offline_replay_a_first_error --split test --device cpu
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.evaluate_policy_offline --rssm-checkpoint models/best_b.pt --policy-checkpoint runs/step6/policy_b_weighted_formal/best.pt --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/offline_replay_b_first_error --split test --device cpu
```

Closed-loop first-error summary on 20 test episodes:

| Mode | First error range | Mean first error | Before handoff |
| --- | ---: | ---: | ---: |
| A full-history | t=9..9 | t=9.0 | 20/20 |
| Oracle-B full-history | t=8..11 | t=9.5 | 20/20 |
| Reset-B | t=12..12 | t=12.0 | 0/20 |

This identifies the failure location. A and Oracle-B begin deviating during
the shared prefix, before the branch decision at `handoff_t=12`; Reset-B first
fails exactly at the handoff because its current-observation-only state cannot
select the hidden branch. The offline teacher-forced diagnostics reached the
handoff correctly for both models, with first errors before handoff in all
20 episodes: A range `t=9..9`, mean `9.0`; B range `t=8..11`, mean `9.5`.
Thus the offline Actor can use the expert history to predict the handoff
action, while the closed-loop Actor's own earlier errors change its subsequent
state distribution. This is a policy closed-loop distribution-shift failure,
not evidence that the Step 5 B state lacks the color cue.

### t=8..11 action/probability and geometry diagnosis (2026-10-15)

Added `cmb_minigrid.scripts.diagnose_policy_errors`, which evaluates all 100
expert episodes under teacher-forced public history at `t=(8,9,10,11)`.
For every family/time it records A full-history, B full-history, and Reset-B
logits, probabilities, predicted actions, expert action, the complete `7x7x3`
local observation plus direction, and the Full-history-vs-Reset comparison.
The report is:
`runs/step6/policy_error_diagnosis_t8_t11/policy_error_diagnosis.json`.

The command returned exit code 0:

```bash
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.diagnose_policy_errors --checkpoint-a models/best_a.pt --checkpoint-b models/best_b.pt --policy-a runs/step6/policy_a_weighted_formal/best.pt --policy-b runs/step6/policy_b_weighted_formal/best.pt --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/policy_error_diagnosis_t8_t11 --device cpu --times 8 9 10 11
```

All four expert labels are `forward` for every family and episode at these
times. Therefore the observed error is not “Left predicted as Forward”:
Full-history policies make premature `left/right` turns while the correct
action is still `forward`. The Reset-B policy predicts `forward` on all 20
episodes at every family/time, with mean probabilities approximately
`[left=.055, right=.035, forward=.910]`.

| Family/split | B full-history accuracy t8/t9/t10/t11 | Reset-B accuracy |
| --- | --- | --- |
| v1 / test | 50% / 50% / 50% / 0% | 100% / 100% / 100% / 100% |
| v2 / val | 100% / 50% / 0% / 0% | 100% / 100% / 100% / 100% |
| v3 / train | 100% / 100% / 100% / 50% | 100% / 100% / 100% / 100% |
| v4 / train | 50% / 50% / 50% / 0% | 100% / 100% / 100% / 100% |
| v5 / train | 50% / 50% / 50% / 0% | 100% / 100% / 100% / 100% |

At each family/time there is exactly one local observation pattern across the
20 counterfactual expert endpoints, and direction is unchanged. The local
patterns differ between map families, but Reset-B remains correct everywhere.
This argues against “the CNN cannot recognize the turn geometry” as the direct
cause at t=8..11: the current local observation is sufficient to choose
forward, while the history-conditioned policy is the one introducing a
premature turn. The failure is consistent with the recurrent cue/state being
used too early by the Actor, or with a policy representation that conflates
hidden-branch information with immediate action timing.

The family pattern is not test-v1-only. The same premature-turn behavior is
present in train v4/v5 and validation v2; train v3 is substantially better
through t=10 but degrades at t=11. At t=8..11, B full-history is wrong while
Reset-B is correct in 10/20 episodes for v1 at t8/t9/t10, all 20 for v1 at
t11, 10/20 for v2 at t9, all 20 for v2 at t10/t11, 10/20 for v3 at t11,
10/20 for v4 at t8/t9/t10, all 20 for v4/v5 at t11, and the analogous
10/20 pattern for v5 at t8/t9/t10. The complete probabilities and local
observations are in the JSON artifact rather than only aggregate accuracy.

### Actor-B weighting ablation (2026-10-16)

Trained two additional Actor-B policies with the unchanged RSSM-B checkpoint
`models/best_b.pt`, the same Step 3 data and train/validation split, seed 45,
CPU, and the same 500-update formal training settings as the existing
inverse-frequency weighted policy. The existing weighted policy was not
overwritten. The new modes are implemented by `class_weighting` in
`cmb_minigrid/training/policy.py` and selected by the corresponding configs:

- `none`: unweighted cross-entropy, output
  `runs/step6/policy_b_unweighted_formal/`.
- `sqrt`: normalized inverse-square-root class-frequency weighting, output
  `runs/step6/policy_b_sqrt_formal/`.

The train action counts were `left=60`, `right=60`, `forward=1120`. The
resulting weights were inverse `[6.888889, 6.888889, 0.369048]`, square-root
`[1.344414, 1.344414, 0.311171]`, and none `[1, 1, 1]`. Both runs selected
step 50 as the best validation checkpoint and then overfit:

| Actor-B variant | Best val loss | Final train accuracy | Final val loss | Final val accuracy |
| --- | ---: | ---: | ---: | ---: |
| Unweighted | 0.738923 | 0.991935 | 3.498652 | 0.777778 |
| Sqrt inverse-frequency | 0.791589 | 0.991935 | 3.541445 | 0.750000 |

The actual training commands returned exit code 0:

```bash
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.train_policy --config configs/train_policy_unweighted.yaml --rssm-checkpoint models/best_b.pt --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/policy_b_unweighted_formal --device cpu --seed 45
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.train_policy --config configs/train_policy_sqrt.yaml --rssm-checkpoint models/best_b.pt --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/policy_b_sqrt_formal --device cpu --seed 45
```

Offline teacher-forced test replay also returned exit code 0 for both models.
The unweighted policy reached action accuracy `0.944444`, episode mean
accuracy `0.944444`, but handoff accuracy `0.0`; every episode's first error
was at `t=12`. The square-root policy reached action accuracy `0.916667`,
episode mean accuracy `0.916667`, and handoff accuracy `0.5`; its first error
was before handoff in 10/20 episodes, with mean first-error time `11.5`.
Reports:
`runs/step6/offline_replay_b_unweighted/offline_replay_report.json` and
`runs/step6/offline_replay_b_sqrt/offline_replay_report.json`.

```bash
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.evaluate_policy_offline --rssm-checkpoint models/best_b.pt --policy-checkpoint runs/step6/policy_b_unweighted_formal/best.pt --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/offline_replay_b_unweighted --split test --device cpu
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.evaluate_policy_offline --rssm-checkpoint models/best_b.pt --policy-checkpoint runs/step6/policy_b_sqrt_formal/best.pt --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/offline_replay_b_sqrt --split test --device cpu
```

Closed-loop evaluation used the unchanged weighted Actor-A for the A mode and
each new policy for B. Both commands returned exit code 0. On the 20 test
episodes, neither variant produced a positive Oracle-B versus Reset-B branch
accuracy gap: both reported branch accuracy `0.0` for Oracle-B and Reset-B.
The unweighted model had success rate `0.0` in both modes and first errors at
`t=12`. The square-root model reported Oracle-B success rate `1.0` versus
Reset-B `0.0`; however its branch-accuracy field remained `0.0`, so this is
not treated as a complete task-success result and requires metric-contract
diagnosis before claiming an Oracle/Reset improvement. A mode remained
unchanged with first error `t=9` and success rate `0.0`.

```bash
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.evaluate_policy --checkpoint-a models/best_a.pt --checkpoint-b models/best_b.pt --policy-a runs/step6/policy_a_weighted_formal/best.pt --policy-b runs/step6/policy_b_unweighted_formal/best.pt --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/evaluation_b_unweighted --device cpu
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.evaluate_policy --checkpoint-a models/best_a.pt --checkpoint-b models/best_b.pt --policy-a runs/step6/policy_a_weighted_formal/best.pt --policy-b runs/step6/policy_b_sqrt_formal/best.pt --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/evaluation_b_sqrt --device cpu
```

The t=8..11 diagnosis commands also returned exit code 0. Under teacher-forced
history, the unweighted B policy was correct at all four times for all five
map families (`v1` test, `v2` validation, `v3/v4/v5` train), matching Reset-B.
The square-root policy was also correct at t=8..10 everywhere, but at t=11 it
was correct on only 10/20 episodes for v1, v4 and v5; the other 10/20 made a
premature right prediction while Reset-B remained 20/20 correct. Thus removing
class weighting improves the measured prefix behavior, but neither ablation
solves the closed-loop handoff problem. High offline accuracy alone is not
sufficient evidence of a usable history-conditioned Actor.

```bash
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.diagnose_policy_errors --checkpoint-a models/best_a.pt --checkpoint-b models/best_b.pt --policy-a runs/step6/policy_a_weighted_formal/best.pt --policy-b runs/step6/policy_b_unweighted_formal/best.pt --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/policy_error_diagnosis_b_unweighted --device cpu --times 8 9 10 11
/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.diagnose_policy_errors --checkpoint-a models/best_a.pt --checkpoint-b models/best_b.pt --policy-a runs/step6/policy_a_weighted_formal/best.pt --policy-b runs/step6/policy_b_sqrt_formal/best.pt --input data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --output-dir runs/step6/policy_error_diagnosis_b_sqrt --device cpu --times 8 9 10 11
```

Focused weighting tests passed (`5 passed`). The full regression suite,
ruff, and `git diff --check` were not rerun after this log-only update; they
remain the next verification step. GPU training, additional seeds, test-set
closed-loop metric-contract diagnosis, and further policy architecture changes
were not run.

### Training-split closed-loop validation of saved A/B policies (2026-10-16)

Evaluated the user-provided policy checkpoints
`models/policy_a.pt` and `models/policy_b.pt` with their corresponding frozen
RSSM checkpoints `models/best_a.pt` and `models/best_b.pt`. The evaluation used
the existing `run_closed_loop` implementation, selected only expert records
assigned to the manifest `train` split, and did not use expert actions to
advance the environment. The report is:
`runs/step6/evaluation_models_train_20261016/closed_loop_report.json`.

The actual command returned exit code 0. It was an inline Python heredoc run
from the repository root with
`/data2/xuyihao/envs/cmb_minigrid/bin/python`; the body loaded the four
checkpoints above, selected `manifest.assignment == "train"`, called
`run_closed_loop` for the three modes, and wrote the report above. This was
used because the existing public `evaluate_policy` CLI is intentionally fixed
to the test split.

There were 60 training expert episodes. Results:

| Mode | Episodes | Branch accuracy | Closed-loop success | First-error summary |
| --- | ---: | ---: | ---: | --- |
| A full-history | 60 | 1.000 | 1.000 | no errors |
| Oracle-B full-history | 60 | 1.000 | 0.833 | 10 errors, all at t=16 |
| Reset-B | 60 | 0.000 | 0.000 | 60 errors, all at t=12 |

The Oracle-B failures were all in `hidden_history_nav_v4` left-branch
episodes. At `t=16`, B predicted `left` (action 0) while the expert action
was `forward` (action 2). B still selected the correct hidden branch at
`handoff_t=12` on all 60 training episodes. A had no first action error on any
training trajectory and completed all 60.

Thus the saved A policy can execute the training split closed-loop under the
current evaluator. The saved B policy can infer the branch at handoff on the
training split, but is not fully closed-loop reliable: it succeeds on 50/60
episodes and fails later on the v4 left branch. Reset-B confirms the expected
history dependence by failing at the handoff on all 60 episodes. No code tests
were rerun because this validation changed no source code; the generated
report and the checkpoint-loading/evaluation command were directly verified.

### Retest after replacing Actor-B checkpoint (2026-10-16)

`models/policy_b.pt` was replaced and retested with the same frozen RSSM-B
checkpoint `models/best_b.pt` and the same 60-episode train expert split.
The new policy checkpoint metadata is `step=1000`, `best_step=50`,
`best_val=0.7915889024734497`, feature dimension 512, hidden dimension 128,
and square-root inverse-frequency weighting. Its SHA-256 is
`6a8294a722cf1b34073f8dedefb078dc0369020083f3a09d30e7909cf3950d63`.

The actual evaluation command returned exit code 0. It was an inline Python
heredoc run with
`/data2/xuyihao/envs/cmb_minigrid/bin/python` from the repository root. The
body loaded `models/best_b.pt` and `models/policy_b.pt`, selected expert
records with `manifest.assignment == "train"`, evaluated B full-history and
Reset-B through `run_closed_loop`, and wrote:
`runs/step6/evaluation_policy_b_retest_train_20261016/closed_loop_report.json`.

| Mode | Episodes | Branch accuracy | Closed-loop success | First-error summary |
| --- | ---: | ---: | ---: | --- |
| B full-history | 60 | 1.000 | 1.000 (60/60) | no errors |
| Reset-B | 60 | 0.000 | 0.000 | all first errors at t=12 |

Compared with the previous B checkpoint, the new model removed the 10
`hidden_history_nav_v4` left-branch failures at `t=16`. On this training-split
retest, B full-history can both select the correct branch and complete every
trajectory. This is a training-set result and does not establish
generalization to validation or test episodes. No source code tests were
rerun because only the model artifact changed; the new report was written and
its checkpoint loading and JSON output were verified.
