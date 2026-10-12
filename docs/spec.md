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

Step 0 creates only contracts and runtime tooling. It does not claim any
environment, model, bridge, or performance result.

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

## Counterfactual and split rules

At a counterfactual handoff, both paired episodes must have exactly equal
current local observation, direction, task instruction, and recent `k=8`
observation/action history, while differing in earlier hidden evidence and
correct branch. Pixel similarity alone is insufficient. Pair endpoints belong
to the same split; train/validation/test assignment is grouped by
`map_family_id` and `pair_id`, never by random frame.

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
