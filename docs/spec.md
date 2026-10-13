# CMB MiniGrid Interface Specification

This document freezes the Step 0 contracts for the Common Memory Bridge
experiment. Later steps may add implementations, but must not silently change
these meanings or introduce privileged inputs at deployment time.

## Scope and baselines

The experiment tests whether a history-bearing RSSM-A can transfer information
to an independently trained RSSM-B through a compact CMB. The first models use
the same observation/action interface but different hidden dimensions and
seeds. The six required baselines are Reset-B, Replay-N (N=4/8/16 as
configured), Linear Bridge, capacity-matched MLP Bridge, CMB, and Full-history
Oracle-B.

Step 0 created the contracts and runtime tooling. Step 1 adds the official
MiniGrid-Memory environment adapter and Step 2 adds a strict HiddenHistoryNav
counterfactual generator. Step 3 adds grouped trajectory data and Step 4 adds
a lightweight PyTorch categorical RSSM and bounded CPU acceptance diagnostics.
Step 5 adds independent A/B sequence training and frozen cue diagnostics on the
unchanged Step 3 splits. No policy/bridge training or control-performance result
is claimed.

## Time and state indexing

For horizon `T`, a trajectory stores `obs[0:T+1]` and `action[0:T]`.
`action[t]` is selected after observing `obs[t]` and produces `obs[t+1]`.
`state[t]` is the posterior RSSM state after consuming `obs[t]` (and the
history/actions needed to reach it). A handoff at `t` therefore gives B a
state already conditioned on `obs[t]`; B must not fuse `obs[t]` a second time.
An imagined transition starts at `state[t]`, applies `action[t]`, and predicts
the next state/observation.

The `Trajectory` contract enforces `len(observations) = len(actions) + 1`.
The `RSSMState` contract uses batch-first shapes `deter=[B,D]` and
`stoch=[B,G,C]`, where the first round is `G=16`, `C=16`, `D_A=128`, and
`D_B=256`.

## Step 4 PyTorch model interface

`RSSM(RSSMConfig(...))` is an `nn.Module`. The symbolic encoder separately
embeds object/color/state IDs with vocabularies `(11,6,3)`, then uses an MLP
to produce a 64-dimensional visual encoding. Direction is appended as four
one-hot values in the posterior input. The fixed public task identifier is
retained in the input dictionary but is not language-encoded. Ordinary
`GRUCell`, prior/posterior MLPs, categorical image reconstruction, direction
reconstruction, scalar reward and Bernoulli continue heads are used. This is
a small DreamerV3-style prototype, not an official JAX reproduction.

- `initial(batch)` returns a sentinel with zero deterministic state,
  category-zero one-hot stochastic state and uniform logits on model
  device/dtype. This sentinel has not consumed any observation.
- `observe_step(prev_state, prev_action, obs, sample=None)` returns a posterior
  `RSSMState`. `obs` is a dictionary containing exactly `image` (integer
  `[B,7,7,3]`), `direction` (integer `[B]` in 0..3), and public `task_id`.
  Filter `obs[0]` once with `prev_action=None` or a zero `[B,3]` vector.
  Subsequent calls use `state[t]`, `action[t]`, `obs[t+1]`.
- Actions accept integer IDs `[B]` (0=left, 1=right, 2=forward) or `[B,3]`
  one-hot vectors. Zero vectors represent no previous action, not action 0.
- `imagine_step(state, action, sample=None)` returns a prior `RSSMState` with
  no observation/encoder access. `features(state)` concatenates deterministic
  and flattened stochastic state to `[B,384]` for A or `[B,512]` for B.
- `observe(obs, actions, observation_mask=None, sample=None)` consumes
  `[B,T+1,7,7,3]` images, `[B,T+1]` directions and `[B,T]` actions. It returns
  `RSSMSequence`: a tuple of batch-first posterior states, `features`
  `[B,T+1,F]`, and prior/posterior logits `[B,T+1,16,16]`. Single-state
  `RSSMState` shapes remain unchanged. Padding freezes the previous state.
- `decode(features)` supports arbitrary leading axes; image logits are a
  three-element tuple ending in `[7,7,11]`, `[7,7,6]`, `[7,7,3]`; direction
  logits end in `[4]`, and reward/continue logits retain leading axes.

Training mode uses categorical sampling with straight-through gradients;
`eval()` uses deterministic argmax, while `sample=True/False` overrides the
mode. Unimix defaults to 0.01. Stored logits are normalized log probabilities.

`loss(batch, sample=None)` reads public observations, actions, reward,
terminated/truncated and masks from the Step 3 batch. It excludes expert labels
and metadata. Reconstruction covers all valid posterior states, including
`obs[0]`; it averages pixel/channel cross-entropies and adds direction
cross-entropy. Reward MSE and continue BCE for transition `t` use `state[t+1]`.
The continue target is `~terminated`: time-limit truncation bootstraps rather
than asserting an absorbing terminal state. Terminal transitions and final
observations contribute, while padding does not. Masks must be contiguous
prefixes and no valid transition may follow termination or truncation.

Dynamics KL is `KL(sg(q)||p)` and representation KL is `KL(q||sg(p))`.
Categories and groups are summed per batch/time sample before clamping at
`free_nats=1`; valid samples are then averaged. The total is reconstruction
plus reward plus continue plus `0.5*dynamics_kl + 0.1*representation_kl`.
Low KL can therefore yield zero prior gradient and a total KL floor of 0.6;
this is intentional. The diagnostic uses bounded single-batch updates only;
optimizer schedules, checkpoints and validation belong to Step 5.

## Step 5 training and checkpoint interface

`cmb_minigrid.scripts.train_rssm` trains one A/B preset per invocation with its
own seed, initialization, Adam optimizer and run directory. Shared settings are
in `configs/train_rssm.yaml`. Entire happened episodes are filtered from obs[0]
on each minibatch; no shuffled frame or zero-initialized cropped suffix replaces
the full history. Only the existing public observations, actions, rewards,
terminal flags and masks enter `RSSM.loss`; Step 4 targets/KL remain unchanged.

Validation uses deterministic categorical modes on the whole grouped val split.
Each loss component is weighted by valid observation or transition count across
batches, then the total is recomputed with the model's loss scales. Formal
training stops after configured patience intervals without an improvement larger
than min_delta. Smoke executes 100 sampled updates by default with stopping
disabled. Both modes report failure if valid train loss and validation selection
do not improve, or losses/gradients become non-finite. A zero prior gradient at
the free-nats floor remains legitimate and is logged separately.

Each checkpoint stores schema/config, model and optimizer state, step, best
validation/counters, shuffled episode order/cursor, and RNG states. `best.pt`
is validation-selected; `last.pt` is the latest safe resume point. Checkpoint
writes use temporary-file replacement. Dataset/manifest SHA-256 fingerprints
and the resolved configuration are checked on resume; only max_steps/device may
change. `load_rssm(path, device)` loads either model independently into eval mode
with no peer checkpoint or external YAML required. `state[t]` indexing and the
full-history filtering interface are exactly those of Step 4.

`cmb_minigrid.scripts.probe_rssm` freezes both loaded RSSMs, validates strict
counterfactual pairs and fits detached linear hidden-cue probes using only train
expert endpoints and train-fitted normalization. Evaluation uses val expert
endpoints; test remains untouched. Feature extraction stops at handoff_t with
obs[0:t+1]/action[0:t], excluding the future branch. Controls include current
categorical observations, reset-current states and reset recent-k=8 states;
early states/observations check cue visibility. Paired feature distances and
both-endpoint prediction accuracy accompany single-endpoint accuracy. Hidden
labels are offline probe targets only; fitted weights are separate artifacts
with no connection to main-model forward/loss. Neither cue decoding nor a
world-model loss decrease is a sufficient memory/control acceptance criterion.

## Observation contract

`AgentObs` contains only a local symbolic image `(H,W,3)`, non-negative
direction, and a public fixed task identifier. It must not contain a global
map, `env_state`, hidden-world label, expert action, future observation, or
history tag. Privileged state may be retained separately for data generation
and evaluation, never as a deployed B/CMB input.

The Step 2 `HiddenHistoryNav-v0` adapter uses task ID `hidden_history_nav`,
mission text `reach the open branch indicated by the earlier channel clue`, and
map family `hidden_history_nav_v1`. It places an early red/blue channel clue,
then a shared horizontal corridor ending at handoff `t=12`, followed by a
vertical branch. In the `left_blocked` world the lower goal is open and in the
`right_blocked` world the upper goal is open. Paired worlds have equal public
observations, mission, and actions through the handoff window, but different
hidden labels, early clue observations, correct branch, and expert branch
action. Snapshots store seed, hidden label, pose, direction, step count, and
terminal flags; the deterministic map is regenerated from seed and label.

The authorized Step 3 expansion adds `hidden_history_nav_v2` through `v5`
without changing v1's 19×15 map or its default call. The added families use a
bent public corridor, a vertical corridor with a horizontal fork, and branches
with one or two bends. All retain `handoff_t=12`, the same task/mission, the
three public actions, and `uint8[7,7,3]` local observations. Each goal requires
passing the handoff hub; both counterfactual endpoints have equal route length.
Snapshots now include `map_family_id`; a missing field means legacy v1. A
family ID is privileged metadata and is excluded from `AgentObs`.

The Step 1 `MiniGrid-MemoryS13-v0` adapter uses fixed task ID
`minigrid_memory_s13`; its observation image is `uint8[7,7,3]`. Its public
action IDs are `0=left`, `1=right`, and `2=forward`, mapped to the corresponding
MiniGrid actions. The native task has additional pickup/drop/toggle/done
actions, but its success and failure conditions are reached by navigating onto
the corresponding target squares; no interaction action is needed to execute
the task. Raw MiniGrid `info` is not forwarded through the adapter.

## Counterfactual and split rules

At a counterfactual handoff, both paired episodes must have exactly equal
current local observation, direction, task instruction, and recent `k=8`
observation/action history, while differing in earlier hidden evidence and
correct branch. Pixel similarity alone is insufficient. Pair endpoints belong
to the same split; train/validation/test assignment is grouped by
`map_family_id` and `pair_id`, never by random frame.

With enough families, each split with a positive configured ratio receives at
least one whole family; zero-ratio splits remain empty. With five families and
ratios `(0.8,0.1,0.1)`, whole-family allocation is `(3,1,1)`, so the realized
episode proportions are `(0.6,0.2,0.2)` when families have equal sizes. The
ratios are approximate targets, not permission to split a family. The initial
five-family dataset uses fixed topology templates and ten seeded pair
collections per family; these repetitions are not fifty independent maps.

Oracle-B filters the same happened trajectory with B and is not allowed to read
the simulator's privileged map. Hidden labels, maps, expert actions, and
future data are allowed for supervision/evaluation only.

## Reproducibility

Every entry point takes a non-negative integer seed and records the resolved
device, configuration, and experiment ID. Python, NumPy, and PyTorch RNGs are
seeded by `cmb_minigrid.reproducibility.set_seed`; environment implementations
must additionally pass the seed to Gymnasium `reset` and preserve action
replay. Deterministic PyTorch algorithms are requested by default.

## Required evidence boundary

The target ordering `Oracle-B > CMB-B > Reset-B` is a hypothesis, not a forced
assertion. A lower hidden-state MSE alone does not establish successful memory
handoff, and equal tensor shapes do not prove a model-independent shared space.
No later step may feed labels, maps, correct actions, or complete history to a
deployed B/CMB shortcut.
