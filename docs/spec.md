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
counterfactual generator. No model, bridge, training, or performance result is
implemented or claimed.

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
