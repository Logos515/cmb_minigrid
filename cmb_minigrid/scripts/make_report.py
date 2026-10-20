"""Step 10: aggregate pilot runs into CSV tables, SVG figures, a manifest and the report.

Reads only artifacts written by ``run_pilot`` (per-group closed-loop rows,
History Swap rows, probe reports, CMB reports, ``manifest.json``). Nothing here
is simulated: every number in the tables and figures comes from those files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from cmb_minigrid.experiments.stats import (
    cluster_bootstrap,
    difference,
    pair_units,
    per_group_means,
)
from cmb_minigrid.experiments.svg import forest_plot

BOOT = 4000
BOOT_SEED = 12345
METRICS = ["first_branch_correct", "success", "path_efficiency", "wrong_actions", "collisions"]
METRIC_LABEL = {"first_branch_correct": "first-branch acc.", "success": "success",
                "path_efficiency": "path efficiency", "wrong_actions": "wrong actions",
                "collisions": "collisions"}
MAIN_METHODS = ["reset", "replay_4", "replay_8", "replay_13", "replay_16", "linear", "mlp",
                "cmb", "oracle_b"]
WINDOW_METHODS = ["reset", "replay_2", "replay_4", "replay_6", "replay_8", "replay_10",
                  "replay_12", "replay_13", "replay_16", "oracle_b"]
CAP_METHODS = [("cap_2x16", "2x16 (8 bit)"), ("cmb", "4x16 (16 bit)"),
               ("cap_8x16", "8x16 (32 bit)"), ("cap_8x32", "8x32 (40 bit)")]
LOSS_METHODS = [("loss_state_only", "state loss only"),
                ("loss_state_action", "state + action KL"), ("cmb", "state + action + future (full)")]
COMPARISONS = [("cmb", "reset"), ("cmb", "replay_8"), ("cmb", "replay_13"), ("cmb", "linear"),
               ("cmb", "mlp"), ("cmb", "oracle_b"), ("cap_2x16", "cmb"), ("cap_8x16", "cmb"),
               ("cap_8x32", "cmb"), ("loss_state_only", "cmb"), ("loss_state_action", "cmb")]


def label(method: str) -> str:
    return {"reset": "Reset (Replay-1)", "oracle_b": "Oracle-B (full history)",
            "cmb": "CMB 4x16 (full recipe)", "linear": "Linear A->B",
            "mlp": "MLP A->B (matched params)"}.get(
        method, method.replace("replay_", "Replay-").replace("cap_", "CMB ").replace("loss_", "CMB "))


def role(method: str) -> str:
    return ("cmb" if method == "cmb" else "oracle" if method == "oracle_b"
            else "ablation" if method.startswith(("cap_", "loss_")) else "baseline")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- loading


def load(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    rows, swap, gates, overhead, cmb_meta = [], [], [], {}, {}
    for group in manifest["groups"]:
        d = root / group
        loop = d / "closed_loop" / "handoff_runs.jsonl"
        if group in manifest["failed_groups"] or not loop.exists():
            gates.append({"group": group, "status": "missing_or_failed"})
            continue
        for line in loop.read_text().splitlines():
            r = json.loads(line)
            r["group"] = group
            rows.append(r)
        with (d / "closed_loop" / "history_swap_rows.csv").open() as handle:
            for r in csv.DictReader(handle):
                swap.append({
                    "group": group, "cmb": r["cmb"], "condition": r["condition"],
                    "split": r["split"], "map_family_id": r["map_family_id"],
                    "pair_id": r["pair_id"],
                    "first_branch_correct": float(r["first_action"] == r["own_expert_action"]),
                    "success": float(r["success"] == "True"),
                    "path_efficiency": float(r["path_efficiency"]),
                    "flipped_vs_own": float(r["flipped_vs_own"] == "True")})
        probe = json.loads((d / "probe" / "probe_report.json").read_text())["models"]
        gate = {"group": group, "status": "ok"}
        for name in ("A", "B"):
            p = probe[name]["probes"]
            gate[f"probe_{name}_full_val_acc"] = p["full_history"]["val"]["accuracy"]
            gate[f"probe_{name}_current_val_acc"] = p["current_state"]["val"]["accuracy"]
            gate[f"probe_{name}_status"] = probe[name]["cue_retention_status"]
        gates.append(gate)
        for variant in ("cmb", "cap_2x16", "cap_8x16", "cap_8x32"):
            rep = json.loads((d / "cmb" / variant / "report.json").read_text())
            overhead[variant] = rep["overhead"]
            cmb_meta.setdefault(group, {})[variant] = {
                k: rep["stages"].get(k) for k in ("stage1", "stage2")}
    return {"manifest": manifest, "rows": rows, "swap": swap, "gates": gates,
            "overhead": overhead, "cmb_meta": cmb_meta}


def add_gate_columns(data: dict[str, Any], threshold: float = 0.5) -> None:
    """Step 6 gate: Oracle-B first-branch minus Reset first-branch on test (operational cut-off)."""
    for gate in data["gates"]:
        if gate["status"] != "ok":
            continue
        rows = [r for r in data["rows"] if r["group"] == gate["group"] and r["split"] == "test"]

        def mean(method, rows=rows):
            return float(np.mean([r["first_branch_correct"] for r in rows if r["method"] == method]))

        gate["oracle_first_branch_test"] = mean("oracle_b")
        gate["reset_first_branch_test"] = mean("reset")
        gate["actor_gap"] = gate["oracle_first_branch_test"] - gate["reset_first_branch_test"]
        gate["gate_probe_pass"] = bool(min(gate["probe_A_full_val_acc"],
                                           gate["probe_B_full_val_acc"]) >= 0.9)
        gate["gate_actor_pass"] = bool(gate["actor_gap"] >= threshold)
        gate["gate_pass"] = gate["gate_probe_pass"] and gate["gate_actor_pass"]


# --------------------------------------------------------------------------- statistics


def ci(rows, method, metric, split, *, key="method"):
    units = pair_units(rows, method=method, metric=metric, split=split, method_key=key)
    if not units:
        return None
    return cluster_bootstrap(units, n_boot=BOOT, seed=BOOT_SEED), per_group_means(units)


def fmt(stat) -> str:
    return f"{stat['estimate']:.2f} [{stat['lo']:.2f}, {stat['hi']:.2f}]"


def method_table(rows, methods, split, scope) -> list[dict[str, Any]]:
    out = []
    for m in methods:
        for metric in METRICS:
            got = ci(rows, m, metric, split)
            if got is None:
                continue
            stat, groups = got
            out.append({"scope": scope, "split": split, "method": m, "metric": metric,
                        "estimate": stat["estimate"], "ci_lo": stat["lo"], "ci_hi": stat["hi"],
                        "n_groups": stat["n_groups"], "n_pairs": stat["n_pairs"],
                        "per_group": json.dumps({g: round(v, 4) for g, v in groups.items()})})
    return out


def paired_table(rows, split, scope) -> list[dict[str, Any]]:
    out = []
    for a, b in COMPARISONS:
        for metric in ("first_branch_correct", "path_efficiency", "wrong_actions"):
            ua = pair_units(rows, method=a, metric=metric, split=split)
            ub = pair_units(rows, method=b, metric=metric, split=split)
            if not ua or not ub:
                continue
            stat = cluster_bootstrap(difference(ua, ub), n_boot=BOOT, seed=BOOT_SEED)
            out.append({"scope": scope, "split": split, "a": a, "b": b, "metric": metric,
                        "diff_a_minus_b": stat["estimate"], "ci_lo": stat["lo"],
                        "ci_hi": stat["hi"], "excludes_zero": bool(stat["lo"] > 0 or stat["hi"] < 0),
                        "n_groups": stat["n_groups"], "n_pairs": stat["n_pairs"]})
    return out


def message_table(swap, variant, scope) -> list[dict[str, Any]]:
    out = []
    rows = [r for r in swap if r["cmb"] == variant]
    for condition in ("own", "same_cue", "swap", "zero"):
        for metric in ("first_branch_correct", "success", "flipped_vs_own"):
            sub = [{**r, "method": r["condition"]} for r in rows]
            got = ci(sub, condition, metric, "test")
            if got is None:
                continue
            stat, groups = got
            out.append({"scope": scope, "cmb": variant, "condition": condition, "metric": metric,
                        "estimate": stat["estimate"], "ci_lo": stat["lo"], "ci_hi": stat["hi"],
                        "n_groups": stat["n_groups"], "n_pairs": stat["n_pairs"],
                        "per_group": json.dumps({g: round(v, 4) for g, v in groups.items()})})
    return out


# --------------------------------------------------------------------------- output helpers


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_table(header: list[str], body: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines)


def forest(path: Path, entries, metric, split, rows, title, subtitle, xlim=(0.0, 1.0)):
    plot_rows = []
    for method, text in entries:
        got = ci(rows, method, metric, split)
        if got is None:
            continue
        stat, groups = got
        plot_rows.append({"label": text, "est": stat["estimate"], "lo": stat["lo"],
                          "hi": stat["hi"], "role": role(method), "points": list(groups.values())})
    forest_plot(path, plot_rows, title=title, subtitle=subtitle, xlabel=METRIC_LABEL[metric],
                xlim=xlim)


# --------------------------------------------------------------------------- report


def build(root: Path, out: Path, analysis: Path | None,
          primary_groups: list[str] | None = None) -> dict[str, Any]:
    data = load(root)
    add_gate_columns(data)
    rows, swap, gates = data["rows"], data["swap"], data["gates"]
    extended = sorted({r["group"] for r in rows if r["group"] != "g0"},
                      key=lambda g: int(g[1:]))
    primary = [g for g in (primary_groups or extended) if g in extended]
    legacy = [g for g in ("g0",) if any(r["group"] == g for r in rows)]
    passed = [g["group"] for g in gates if g.get("gate_pass") and g["group"] in extended]
    sets = {"primary": primary, "extended": extended, "primary_plus_g0": primary + legacy,
            "gate_passed_only": passed}
    scopes = {k: [r for r in rows if r["group"] in v] for k, v in sets.items()}
    swap_scopes = {k: [r for r in swap if r["group"] in v] for k, v in sets.items()}
    methods = sorted({r["method"] for r in rows})
    tables = out / "tables"
    figures = out / "figures"
    produced: dict[str, Any] = {"tables": [], "figures": []}

    for scope, subset in scopes.items():
        if not subset:
            continue
        for split in ("test", "val", "train"):
            write_csv(method_table(subset, methods, split, scope),
                      tables / f"methods_{scope}_{split}.csv")
        write_csv(paired_table(subset, "test", scope), tables / f"paired_{scope}_test.csv")
    write_csv([{k: v for k, v in g.items()} for g in gates], tables / "gates.csv")
    msg_rows = []
    for scope, subset in swap_scopes.items():
        if subset:
            for variant in sorted({r["cmb"] for r in subset}):
                msg_rows += message_table(subset, variant, scope)
    write_csv(msg_rows, tables / "message_conditions.csv")
    over = [{"variant": v, **{f"cmb_{k}": val for k, val in o["cmb"].items() if k != "note"},
             "note": o["cmb"]["note"], "mlp_total": o.get("mlp", {}).get("total"),
             "mlp_payload_bytes": o.get("mlp", {}).get("payload_bytes_float32"),
             "linear_total": o.get("linear", {}).get("total"),
             "linear_payload_bytes": o.get("linear", {}).get("payload_bytes_float32")}
            for v, o in sorted(data["overhead"].items())]
    # overhead is identical across groups (it depends on shapes only); keep one row per variant
    write_csv(over, tables / "overhead.csv")

    prim = scopes["primary"]
    if prim:
        def sub_for(n):
            return (f"test split, family v1; cohort = {n} RSSM seed groups (dots); "
                    "whiskers = 95% clustered bootstrap CI")

        sub = sub_for(f"first {len(primary)}")
        ext_sub = sub_for(f"all {len(extended)} new")
        ext = scopes["extended"]
        forest(figures / "main_first_branch.svg", [(m, label(m)) for m in MAIN_METHODS],
               "first_branch_correct", "test", prim,
               "First handoff action correct, by initialization method", sub)
        forest(figures / "main_path_efficiency.svg", [(m, label(m)) for m in MAIN_METHODS],
               "path_efficiency", "test", prim, "Path efficiency after takeover", sub)
        forest(figures / "replay_window.svg", [(m, label(m)) for m in WINDOW_METHODS],
               "first_branch_correct", "test", ext,
               "Replay window sweep (frames 0-12 hold the cue; Replay-13/16 = full history)", ext_sub)
        forest(figures / "cmb_capacity.svg", CAP_METHODS, "first_branch_correct", "test", ext,
               "CMB capacity ablation (tokens x classes)", ext_sub)
        forest(figures / "loss_ablation.svg", LOSS_METHODS, "first_branch_correct", "test", ext,
               "Loss ablation (4x16 CMB)", ext_sub)
        cond = []
        for c, text in (("own", "own message (correct)"), ("same_cue", "same-cue other history"),
                        ("swap", "swapped message"), ("zero", "zeroed message")):
            got = ci([{**r, "method": r["condition"]} for r in swap_scopes["extended"]
                      if r["cmb"] == "cmb"], c, "first_branch_correct", "test")
            if got:
                cond.append({"label": text, "est": got[0]["estimate"], "lo": got[0]["lo"],
                             "hi": got[0]["hi"], "role": "cmb" if c == "own" else "ablation",
                             "points": list(got[1].values())})
        forest_plot(figures / "message_conditions.svg", cond, title="Correct / swapped / zeroed CMB",
                    subtitle=(f"4x16 CMB, test split, all {len(extended)} new groups; environment and obs_t "
                              "identical in every condition"),
                    xlabel=METRIC_LABEL["first_branch_correct"])
        for scope, tag in (("extended", "extended"), ("gate_passed_only", "gate_passed")):
            if scopes[scope]:
                forest(figures / f"main_first_branch_{tag}.svg", [(m, label(m)) for m in MAIN_METHODS],
                       "first_branch_correct", "test", scopes[scope],
                       f"First handoff action correct ({scope.replace('_', ' ')}: "
                       f"{len(sets[scope])} groups)", sub_for(f"{len(sets[scope])} selected"
                                                            if scope != "extended" else
                                                            f"all {len(extended)} new"))
        produced["figures"] = sorted(p.name for p in figures.glob("*.svg"))
    produced["tables"] = sorted(p.name for p in tables.glob("*.csv"))

    config_manifest(root, out, data, analysis)
    report = render(data, out, primary, extended, legacy, passed, analysis, scopes,
                    swap_scopes)
    (out / "experiment_report.md").write_text(report, encoding="utf-8")
    return produced


def config_manifest(root: Path, out: Path, data: dict[str, Any],
                    analysis: Path | None = None) -> None:
    import gymnasium
    import minigrid
    import torch

    manifest = data["manifest"]
    repo = Path(__file__).resolve().parents[2]
    configs = {str(p.relative_to(repo)): sha(p) for p in sorted((repo / "configs").glob("*.yaml"))}
    groups = {}
    for group in manifest["groups"]:
        d = root / group
        if not d.exists():
            continue
        entry = {"paths": manifest["paths"].get(group, {})}
        for path in sorted(d.glob("rssm_*/*/config.json")) + sorted(d.glob("policy_b/config.json")) \
                + sorted(d.glob("cmb/*/config.json")):
            entry[str(path.relative_to(root))] = json.loads(path.read_text())
        groups[group] = entry
    payload = {
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "gymnasium": gymnasium.__version__, "minigrid": minigrid.__version__,
                     "numpy": np.__version__},
        "git_commit": manifest.get("git_commit"),
        "git_note": "HEAD at run time; Step 8-10 code was uncommitted in the working tree",
        "seeds": manifest.get("seeds"),
        "bootstrap": {"n_boot": BOOT, "seed": BOOT_SEED, "level": 0.95,
                      "clusters": "RSSM seed group > map family > pair (both endpoints together)"},
        "data": {"trajectories_sha256": sha(repo / "data/samples/step3_trajectories.jsonl"),
                 "split_manifest_sha256": sha(repo / "data/samples/step3_manifest.json")},
        "configs_sha256": configs, "wall_seconds": manifest.get("wall_seconds"),
        "failed_groups": manifest.get("failed_groups"), "groups": groups,
    }
    (out / "config_manifest.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    commands = ["#!/usr/bin/env bash", "# Generated from runs manifest; run from the repo root in the",
                "# cmb_minigrid conda environment (unset PYTHONPATH).", "set -e"]
    commands += [j["command"] for j in manifest["jobs"]]
    commands.append(f"python -m cmb_minigrid.scripts.make_report --root {root} --output {out}"
                    + (f" --analysis {analysis}" if analysis else ""))
    (out / "repro_commands.sh").write_text("\n".join(commands) + "\n")


def table_by_method(rows, methods, split, metrics) -> str:
    body = []
    for m in methods:
        cells = [label(m)]
        for metric in metrics:
            got = ci(rows, m, metric, split)
            cells.append(fmt(got[0]) if got else "-")
        body.append(cells)
    return md_table(["Method"] + [METRIC_LABEL[x] for x in metrics], body)


def render(data, out, primary, extended, legacy, passed, analysis_path, scopes,
           swap_scopes) -> str:
    rows, gates = data["rows"], data["gates"]
    manifest = data["manifest"]
    prim = [r for r in rows if r["group"] in primary]
    parts = ["# CMB MiniGrid experiment report (Step 10 pilot)\n"]
    parts.append(
        "All numbers below are computed from the artifacts in "
        f"`{manifest['root']}` by `cmb_minigrid.scripts.make_report`; nothing is simulated.\n"
        f"**Scope: pilot-scale.** {len(primary)} pre-planned independent RSSM training-seed groups "
        f"({', '.join(primary)}; A seed 1000+g, B seed 2000+g), one bridge-training seed (42) per "
        "group, one task (HiddenHistoryNav, handoff at t=12, one-bit early cue), held-out test "
        "family v1 (10 counterfactual pairs per group). "
        + ("Group `g0` is the pre-existing Step 5-9 checkpoint set and is shown separately. "
           if legacy else "") + "\n")
    if analysis_path and analysis_path.exists():
        parts.append(analysis_path.read_text(encoding="utf-8").strip() + "\n")
    parts.append("## 1. Gates per group\n")
    gh = ["group", "probe full-history val acc A / B", "Oracle first-branch (test)",
          "Reset first-branch (test)", "gap", "probe gate", "actor gate"]
    body = []
    for g in gates:
        if g["status"] != "ok":
            body.append([g["group"], "missing/failed", "", "", "", "", ""])
            continue
        body.append([g["group"], f"{g['probe_A_full_val_acc']:.2f} / {g['probe_B_full_val_acc']:.2f}",
                     f"{g['oracle_first_branch_test']:.2f}", f"{g['reset_first_branch_test']:.2f}",
                     f"{g['actor_gap']:.2f}", "pass" if g["gate_probe_pass"] else "FAIL",
                     "pass" if g["gate_actor_pass"] else "FAIL"])
    parts.append(md_table(gh, body) + "\n")
    parts.append("Operational gate definitions (mine, not from the spec): probe gate = frozen "
                 "linear-probe val accuracy >= 0.9 on the full-history state for both A and B; "
                 "actor gate = Oracle-B minus Reset-B first-branch accuracy >= 0.5 on test. "
                 f"Groups passing both: {', '.join(passed) or 'none'}. **All groups are included in "
                 "the headline statistics regardless of gate status**; `gate_passed_only` is a "
                 "labeled sensitivity analysis in `tables/*_gate_passed_only_*.csv`.\n")

    new = [g for g in gates if g["group"] in extended and g["status"] == "ok"]
    n_probe = sum(g["gate_probe_pass"] for g in new)
    n_actor = sum(g["gate_actor_pass"] for g in new)
    n_oracle = sum(g["oracle_first_branch_test"] == 1.0 for g in new)
    parts.append(f"Gate pass rates over the {len(new)} new groups: probe gate {n_probe}/{len(new)}, "
                 f"actor gate {n_actor}/{len(new)}, both {len(passed)}/{len(new)}. The Step 5/6 "
                 "training recipes are therefore **not seed-robust** in this repository: Oracle-B "
                 f"reaches first-branch accuracy 1.0 in only {n_oracle}/{len(new)} groups, which "
                 "caps what any bridge distilled from it can reach.\n")
    parts.append("## 2. Main comparison (test split, clustered bootstrap 95% CI)\n")
    parts.append(f"Headline cohort: pre-planned groups {', '.join(primary)} (all gate outcomes included).\n")
    parts.append(table_by_method(prim, MAIN_METHODS, "test", [
        "first_branch_correct", "success", "path_efficiency", "wrong_actions"]) + "\n")
    parts.append("![first branch](figures/main_first_branch.svg)\n\n"
                 "![path efficiency](figures/main_path_efficiency.svg)\n")
    pt = [r for r in paired_table(prim, "test", "primary")
          if r["metric"] in ("first_branch_correct", "path_efficiency")]
    body = [[f"{label(r['a'])} - {label(r['b'])}", METRIC_LABEL[r["metric"]],
             f"{r['diff_a_minus_b']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]",
             "yes" if r["excludes_zero"] else "no"] for r in pt]
    parts.append("### Paired differences (same pairs, same groups)\n")
    parts.append(md_table(["A - B", "metric", "difference [95% CI]", "CI excludes 0"], body) + "\n")
    if extended != primary:
        parts.append("### Extended cohort and gate-passed sensitivity\n")
        parts.append(f"The cohort was extended from {len(primary)} to {len(extended)} groups **after** "
                     f"seeing that only part of the first {len(primary)} passed the gates, to estimate "
                     "the pass rate; this is a post-hoc decision. Both cohorts are reported.\n")
        parts.append(f"**All {len(extended)} new groups** (headline-style, no gate filtering):\n")
        parts.append(table_by_method(scopes["extended"], MAIN_METHODS, "test", [
            "first_branch_correct", "success", "path_efficiency", "wrong_actions"]) + "\n")
        parts.append("![first branch extended](figures/main_first_branch_extended.svg)\n")
        if passed:
            parts.append(f"**Gate-passed groups only** ({', '.join(passed)}; sensitivity analysis, "
                         "selected on Oracle-B performance, **not** a headline):\n")
            parts.append(table_by_method(scopes["gate_passed_only"], MAIN_METHODS, "test", [
                "first_branch_correct", "success", "path_efficiency", "wrong_actions"]) + "\n")
            parts.append("![first branch gate passed](figures/main_first_branch_gate_passed.svg)\n")
            pe = [r for r in paired_table(scopes["gate_passed_only"], "test", "gate_passed_only")
                  if r["metric"] == "first_branch_correct" and r["a"] == "cmb"]
            parts.append(md_table(
                ["A - B (gate-passed)", "first-branch diff [95% CI]", "CI excludes 0"],
                [[f"{label(r['a'])} - {label(r['b'])}",
                  f"{r['diff_a_minus_b']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]",
                  "yes" if r["excludes_zero"] else "no"] for r in pe]) + "\n")
    comm = [["Reset", 0], ["Replay-4", 595], ["Replay-8", 1191], ["Replay-13/16 / Oracle", 1936],
            ["Linear", 1536], ["MLP", 256], ["CMB 4x16", 2]]
    parts.append("Payload at handoff (bytes, stated assumptions in the Step 9 log): "
                 + ", ".join(f"{a} {b}" for a, b in comm) + ".\n")

    parts.append("## 3. Ablations (test split, first-branch accuracy)\n")
    parts.append("Each ablation shows three cohorts side by side: the pre-planned "
                 f"{len(primary)} groups, all {len(extended)} new groups, and the gate-passed groups "
                 f"({len(passed)}; sensitivity, selected on Oracle-B performance). Path-efficiency "
                 "versions are in `tables/methods_*_test.csv`.\n")
    cohorts = [(f"first {len(primary)} groups", scopes["primary"]),
               (f"all {len(extended)} new groups", scopes["extended"]),
               (f"gate-passed ({len(passed)})", scopes["gate_passed_only"])]

    def scoped(entries, metric="first_branch_correct", extra=None):
        body = []
        for key, text in entries:
            cells = [text] + (extra(key) if extra else [])
            for _, subset in cohorts:
                got = ci(subset, key, metric, "test") if subset else None
                cells.append(fmt(got[0]) if got else "-")
            body.append(cells)
        return body

    head = [name for name, _ in cohorts]
    parts.append("### Replay window\n\n![replay window](figures/replay_window.svg)\n")
    parts.append(md_table(["Method"] + head,
                          scoped([(m, label(m)) for m in WINDOW_METHODS])) + "\n")
    parts.append("### CMB capacity\n\n![capacity](figures/cmb_capacity.svg)\n")
    over = data["overhead"]
    parts.append(md_table(["variant", "params", "payload bits"] + head, scoped(
        CAP_METHODS, extra=lambda v: [over.get(v, {}).get("cmb", {}).get("total"),
                                      over.get(v, {}).get("cmb", {}).get("payload_bits")])) + "\n")
    parts.append("### Action / future loss\n\n![loss](figures/loss_ablation.svg)\n")
    parts.append(md_table(["variant"] + head, scoped(LOSS_METHODS)) + "\n")
    parts.append("### Correct / swapped / zeroed CMB (4x16)\n\n![message](figures/message_conditions.svg)\n")
    body = []
    for c, text in (("own", "own message (correct)"), ("same_cue", "same cue, other history"),
                    ("swap", "swapped message"), ("zero", "zeroed message")):
        cells = [text]
        for name, _ in cohorts:
            key = {"first": "primary", "all": "extended"}.get(name.split()[0], "gate_passed_only")
            sub = [{**r, "method": r["condition"]} for r in swap_scopes[key] if r["cmb"] == "cmb"]
            got = ci(sub, c, "first_branch_correct", "test") if sub else None
            cells.append(fmt(got[0]) if got else "-")
        body.append(cells)
    parts.append(md_table(["condition"] + head, body) + "\n")

    parts.append("## 4. Per RSSM-seed-group first-branch accuracy (test)\n")
    ghdr = ["group", "reset", "replay_8", "linear", "mlp", "cmb", "oracle_b"]
    body = []
    for g in extended + legacy:
        sub = [r for r in rows if r["group"] == g and r["split"] == "test"]
        cells = []
        for m in ghdr[1:]:
            values = [r["first_branch_correct"] for r in sub if r["method"] == m]
            cells.append(f"{np.mean(values):.2f}" if values else "-")
        body.append([g] + cells)
    parts.append(md_table(ghdr, body) + "\n")

    parts.append("## 5. Evidence boundaries\n")
    parts.append(
        "1. **Single cue transmitted.** Supported by: CMB first-branch accuracy above Reset/Replay-4/8 "
        "on the held-out test family, and the History Swap flip (Section 3). Limit: the cue is one "
        "bit, the handoff is at one fixed time, and first-branch accuracy is measured at the single "
        "decision step.\n"
        "2. **Sustained closed-loop prediction recovery.** Needs state/imagination fidelity over "
        "many steps, not just the first action. Evidence available here: path efficiency, wrong "
        "actions and collisions after takeover (Section 2) and the Step 8 imagined-state drift "
        "(`runs/step8`); the future-image-CE metric does **not** measure it (Oracle-B itself is worse "
        "than Reset there). Treat this claim as weaker than claim 1.\n"
        "3. **Truly shared cross-model space.** Not tested. All results are one A-to-B direction, "
        "RSSMs trained on the same data and recipe, one adapter pair per group. No A-to-C, C-to-B, "
        "unseen-pair or different-architecture test was run (Step 11). Equal tensor shapes or a "
        "working bottleneck do not establish this.\n")

    parts.append("## 6. Not done / limitations\n")
    parts.append(
        "- **History-gap ablation {8,16,32,64} not run.** The environment fixes the cue-to-decision "
        "gap at 12 steps (`HANDOFF_T`); other gaps need new map families (a task-definition change), new "
        "data, retrained RSSMs and a larger `max_steps` (the current limit is 64). Replay windows "
        "are the only history-length ablation here.\n"
        f"- {len(primary)} pre-planned (+{len(extended) - len(primary)} extended) RSSM seed groups, one bridge seed each; bridge-seed variance is "
        "documented only in Steps 8-9 (g0, seeds 42-44). Test is a single map family, so family-level "
        "clustering is degenerate on test (pairs and groups are resampled).\n"
        "- Validation family v2 is not a usable ceiling when Actor-B fails to generalize there; the "
        "headline uses test. Train/val tables are in `tables/`.\n"
        "- Operational gates and cut-offs are mine; payload byte widths are stated assumptions; "
        "wall-clock takeover times are not reported in this document.\n"
        "- Pre-existing failing tests (3) are unrelated to these results.\n")

    parts.append("## 7. Reproduction\n")
    parts.append(f"Versions, seeds, data/config hashes and per-run training configs: "
                 f"`config_manifest.json`. Exact commands with exit codes: `{manifest['root']}"
                 "/manifest.json`; the runnable list is `repro_commands.sh`. Bootstrap: "
                 f"{BOOT} replicates, seed {BOOT_SEED}, percentile 95% CI, clusters = RSSM seed group "
                 "> map family > pair (endpoints stay together).\n")
    return "\n".join(parts)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Step 10 report generator")
    p.add_argument("--root", type=Path, default=Path("runs/step10"))
    p.add_argument("--output", type=Path, default=Path("reports"))
    p.add_argument("--analysis", type=Path, default=None,
                   help="hand-written interpretation (markdown) inserted after the scope note")
    p.add_argument("--primary-groups", nargs="+", default=["g1", "g2", "g3", "g4", "g5"])
    args = p.parse_args(argv)
    try:
        produced = build(args.root, args.output, args.analysis, args.primary_groups)
    except (OSError, KeyError, ValueError) as exc:
        print(f"make_report_failed: {exc}")
        return 1
    print(json.dumps(produced))
    print(f"report={args.output / 'experiment_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
