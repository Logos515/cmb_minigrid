"""make_report on a synthetic runs tree with known values."""

import csv
import json
import xml.etree.ElementTree as ET

from cmb_minigrid.scripts.make_report import build


def write_group(root, group, cmb_correct):
    d = root / group
    (d / "closed_loop").mkdir(parents=True)
    (d / "probe").mkdir()
    rows = []
    for pair in ("p1", "p2"):
        for end, label in (("L", "left"), ("R", "right")):
            for method, value in (("reset", 0.0), ("oracle_b", 1.0), ("cmb", cmb_correct)):
                rows.append({"split": "test", "map_family_id": "fam1", "pair_id": f"{pair}",
                             "method": method, "hidden_label": label, "episode_id": f"{pair}{end}",
                             "first_branch_correct": bool(value), "success": bool(value),
                             "path_efficiency": value, "wrong_actions": 1 - value,
                             "collisions": 0.0})
    (d / "closed_loop" / "handoff_runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    fields = ["cmb", "condition", "split", "map_family_id", "pair_id", "first_action",
              "own_expert_action", "success", "path_efficiency", "flipped_vs_own"]
    with (d / "closed_loop" / "history_swap_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in ("p1", "p2"):
            for condition, action in (("own", 1), ("swap", 0), ("zero", 2), ("same_cue", 1)):
                writer.writerow({"cmb": "cmb", "condition": condition, "split": "test",
                                 "map_family_id": "fam1", "pair_id": pair, "first_action": action,
                                 "own_expert_action": 1, "success": action == 1,
                                 "path_efficiency": float(action == 1),
                                 "flipped_vs_own": action != 1})
    probe = {"probes": {"full_history": {"val": {"accuracy": 1.0}},
                        "current_state": {"val": {"accuracy": 0.5}}},
             "cue_retention_status": "established"}
    (d / "probe" / "probe_report.json").write_text(json.dumps({"models": {"A": probe, "B": probe}}))
    for variant in ("cmb", "cap_2x16", "cap_8x16", "cap_8x32"):
        (d / "cmb" / variant).mkdir(parents=True)
        (d / "cmb" / variant / "report.json").write_text(json.dumps({
            "stages": {"stage1": {}, "stage2": {}},
            "overhead": {"cmb": {"total": 10, "payload_bits": 16.0, "note": "x"},
                         "mlp": {"total": 11, "payload_bytes_float32": 256}}}))


def test_report_numbers_ci_gates_and_figures(tmp_path):
    root = tmp_path / "runs"
    for g, c in (("g1", 1.0), ("g2", 1.0)):
        write_group(root, g, c)
    manifest = {"root": str(root), "groups": ["g1", "g2"], "failed_groups": [], "wall_seconds": 1.0,
                "git_commit": "abc", "seeds": {"g1": {"rssm_a": 1001}}, "paths": {"g1": {}},
                "jobs": [{"group": "g1", "name": "x", "command": "python -m foo", "status": "ok"}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "reports"
    analysis = tmp_path / "analysis.md"
    analysis.write_text("## 0. Interpretation\nHAND WRITTEN\n")
    produced = build(root, out, analysis)
    assert "main_first_branch.svg" in produced["figures"] and "gates.csv" in produced["tables"]
    for svg in (out / "figures").glob("*.svg"):
        ET.parse(svg)
    methods = {(r["method"], r["metric"]): r for r in csv.DictReader(
        (out / "tables" / "methods_primary_test.csv").open())}
    cmb = methods[("cmb", "first_branch_correct")]
    assert float(cmb["estimate"]) == 1.0 and float(cmb["ci_lo"]) == 1.0 and cmb["n_groups"] == "2"
    assert float(methods[("reset", "first_branch_correct")]["estimate"]) == 0.0
    paired = next(r for r in csv.DictReader((out / "tables" / "paired_primary_test.csv").open())
              if r["a"] == "cmb" and r["b"] == "reset" and r["metric"] == "first_branch_correct")
    assert float(paired["diff_a_minus_b"]) == 1.0 and paired["excludes_zero"] == "True"
    gates = list(csv.DictReader((out / "tables" / "gates.csv").open()))
    assert all(g["gate_pass"] == "True" for g in gates)
    msg = {(r["condition"], r["metric"]): float(r["estimate"]) for r in csv.DictReader(
        (out / "tables" / "message_conditions.csv").open()) if r["scope"] == "primary"}
    assert msg[("own", "first_branch_correct")] == 1.0 and msg[("swap", "first_branch_correct")] == 0.0
    assert msg[("swap", "flipped_vs_own")] == 1.0
    text = (out / "experiment_report.md").read_text()
    assert "HAND WRITTEN" in text and "Evidence boundaries" in text and "{" not in text.split("## 1.")[0]
    assert json.loads((out / "config_manifest.json").read_text())["bootstrap"]["n_boot"] > 0
    assert "python -m foo" in (out / "repro_commands.sh").read_text()
