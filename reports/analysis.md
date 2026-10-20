## 0. Findings and how far they go

*Written by hand after reading the tables below; every number is copied from them.
CIs are 95% clustered-bootstrap intervals (RSSM seed group > pair). Dozens of paired
comparisons were computed with no multiplicity correction, so isolated "CI excludes 0"
cases are exploratory.*

**Study-design note (post-hoc).** The pre-planned cohort was 5 RSSM seed groups (g1-g5).
Only 1 of them (g4) passed both operational gates, so the cohort was extended to 15 groups
(g6-g15) to estimate the pass rate. The extension was decided after seeing the first
results; all three cohorts (first 5, all 15, gate-passed 6) are reported and none was
dropped. The "gate-passed" cohort is selected on Oracle-B's own performance and is a
sensitivity analysis, not a headline.

**1. The RSSM/Actor-B pipeline itself is not seed-robust.** Over 15 new seed groups the
frozen-probe gate passes 6/15, the Oracle-vs-Reset actor gate 10/15, both 6/15; Oracle-B
reaches first-branch accuracy 1.0 in only 6/15 groups (g4, g8, g9, g10, g14, g15), and in
5 groups (g1, g3, g6, g7, g13) it is 0.0. A bridge distilled from Oracle-B cannot do better
than its teacher, so group-level results are dominated by whether the two world models and
Actor-B work at all. This is a finding about the Step 5/6 recipes, not about the bridge.

**2. Where a working Oracle exists, the CMB reproduces its first handoff decision.**
CMB first-branch accuracy equals Oracle-B's in 14 of 15 new groups plus g0; the exception is
g9, where RSSM-A fails its own cue probe (val 0.5), i.e. A has nothing to send.
- First 5 groups (all outcomes): CMB 0.40 [0.10, 0.70] vs Reset 0.00; CMB - Reset = +0.40
  [+0.10, +0.70]. CMB - Replay-8 = +0.10 [-0.30, +0.60], CMB - MLP = +0.00 [-0.30, +0.30],
  CMB - Linear = +0.10 [0.00, +0.30]: **not distinguishable from Replay-8, MLP or Linear in
  this cohort.**
- All 15 groups: CMB 0.47 [0.27, 0.70], Oracle 0.53 [0.33, 0.73], Replay-13 0.53, Linear
  0.33 [0.13, 0.53], MLP 0.33 [0.17, 0.50], Replay-8 0.13 [0.03, 0.23], Reset 0.00.
- Gate-passed 6 groups (sensitivity): CMB 0.92 [0.75, 1.00] = Oracle = Replay-13;
  Linear 0.67 [0.33, 0.92]; MLP 0.50 [0.17, 0.83]; Replay-8 0.00. CMB - Linear = +0.25
  [+0.08, +0.42] and CMB - MLP = +0.42 [+0.17, +0.67] exclude 0 there.

**3. The CMB did not exceed full replay or Oracle-B in any cohort.** CMB - Oracle and
CMB - Replay-13 are 0.00 [0.00, 0.00] on first-branch accuracy and -0.09 [-0.21, +0.00] on
path efficiency (first 5). Replay with a window that reaches `obs[0]` (13 frames, 1,936
bytes) matches the CMB (2 bytes) when A's state holds the cue; the CMB's demonstrated
advantage over it is payload size only, at the cost of training a bridge, and Replay needs
no A model. Replay windows of 4 or 8 frames (595 / 1,191 bytes) never contain the cue: Replay-4 stays at
0.00 in every cohort and Replay-8 is 0.00 in the gate-passed cohort; its 0.30 / 0.13 in the
first-5 / all-15 cohorts comes from groups (e.g. g2, g3, g5) whose Actor-B emits the same
branch for both cues (Oracle-B is also 0.5 there), not from recovered memory. Replay-12
(which includes `obs[1]`) is already partial (0.33 in gate-passed groups).
Linear and MLP A->B did **not** beat the CMB on first-branch accuracy
in the gate-passed cohort, but their closed-loop *success* (0.83) is higher than CMB/Oracle
(0.75): success also counts recoveries from the visible wrong-branch dead end, which is why
success is not used as the memory measure.

**4. Ablations show little that is detectable at this sample size.**
- Capacity 2x16 / 4x16 / 8x16 / 8x32: first-branch 0.47 / 0.47 / 0.47 / 0.50 over all 15
  groups, paired differences vs 4x16 within about +/-0.13; in the gate-passed cohort 2x16 is
  -0.17 [-0.33, 0.00] and 8x32 -0.08 [-0.25, 0.00] (CIs touch 0). No capacity benefit beyond
  8 bits is demonstrated for a one-bit cue.
- Loss: first-branch accuracy is the same for state-only, state+action and full recipes.
  Path efficiency differences are small and inconsistent (state+action vs full: -0.07
  [-0.16, -0.00] over all 15; state-only vs full: -0.04 [-0.08, -0.01] gate-passed); I do
  not claim the Action or Future loss helps.
- Message condition (4x16): own message 0.47 [0.27, 0.70]; same cue from another history
  0.47 (no change); swapped 0.10 [0.00, 0.20]; zeroed 0.13 [0.03, 0.23] over all 15; in
  the gate-passed cohort own 0.92 vs swapped 0.00 and zeroed 0.00. The decision follows the
  message when environment and `obs_t` are fixed.
- **History-gap ablation {8,16,32,64} was not run** (see Section 6).

**5. Evidence boundaries.**
- *Single cue transmitted* — **supported** for groups whose RSSM-A carries the cue: swap /
  zero collapse the decision, same-cue messages do not, and the CMB matches Oracle-B. Still a
  one-bit cue at one fixed handoff time.
- *Sustained closed-loop recovery* — **not established.** Even Oracle-B reaches only success
  0.75 and path efficiency 0.53 in the gate-passed groups (12 wrong actions per episode on
  average), because Actor-B itself errs after the branch. The CMB equals Oracle-B within the
  CI, so the ceiling is the Actor, not the bridge; this experiment cannot show sustained
  prediction recovery.
- *Truly shared cross-model space* — **not tested.** One direction (A to B), same data and
  recipe, one bridge per pair, no third model or unseen pair.

