"""Clustered bootstrap contract tests with synthetic data of known truth."""

import numpy as np
import pytest

from cmb_minigrid.experiments.stats import (
    cluster_bootstrap,
    difference,
    pair_units,
    per_group_means,
    point_estimate,
)


def rows(values):  # values: {group: {family: [(pair, [endpoint values])]}}
    out = []
    for g, fams in values.items():
        for f, pairs in fams.items():
            for pair, endpoints in pairs:
                for v in endpoints:
                    out.append({"group": g, "map_family_id": f, "pair_id": pair, "split": "test",
                                "method": "m", "x": v, "y": 2 * v})
    return out


def test_pair_units_average_endpoints_and_align_by_sorted_pair():
    data = rows({"g1": {"f": [("b", [1, 3]), ("a", [0, 0])]}, "g2": {"f": [("a", [1, 1])]}})
    units = pair_units(data, method="m", metric="x", split="test")
    assert units["g1"]["f"].tolist() == [0.0, 2.0]  # pairs a, b
    assert point_estimate(units) == pytest.approx((1.0 + 1.0) / 2)
    assert per_group_means(units) == {"g1": 1.0, "g2": 1.0}
    other = pair_units(data, method="m", metric="y", split="test")
    assert difference(other, units)["g1"]["f"].tolist() == [0.0, 2.0]


def test_bootstrap_is_deterministic_and_brackets_truth():
    rng = np.random.default_rng(0)
    data = rows({f"g{i}": {"f": [(f"p{j}", [rng.normal(0.7, 0.1)] * 2) for j in range(10)]}
                 for i in range(5)})
    units = pair_units(data, method="m", metric="x", split="test")
    a = cluster_bootstrap(units, n_boot=500, seed=3)
    assert a == cluster_bootstrap(units, n_boot=500, seed=3)
    assert a["lo"] < a["estimate"] < a["hi"] and a["lo"] < 0.7 < a["hi"] + 0.05
    assert a["n_groups"] == 5 and a["n_pairs"] == 50


def test_constant_data_gives_degenerate_interval_and_clusters_stay_whole():
    data = rows({"g1": {"f": [("p1", [1.0, 1.0]), ("p2", [0.0, 0.0])]}})
    result = cluster_bootstrap(pair_units(data, method="m", metric="x", split="test"),
                               n_boot=300, seed=0)
    assert 0.0 <= result["lo"] <= 0.5 <= result["hi"] <= 1.0
    flat = rows({"g1": {"f": [("p1", [0.5, 0.5]), ("p2", [0.5, 0.5])]}})
    degenerate = cluster_bootstrap(pair_units(flat, method="m", metric="x"), n_boot=100)
    assert degenerate["lo"] == degenerate["hi"] == 0.5


def test_group_resampling_widens_interval_when_groups_disagree():
    same = rows({f"g{i}": {"f": [("p1", [0.5] * 2), ("p2", [0.5] * 2)]} for i in range(5)})
    differ = rows({f"g{i}": {"f": [("p1", [float(i % 2)] * 2), ("p2", [float(i % 2)] * 2)]}
                   for i in range(5)})
    def width(d):
        r = cluster_bootstrap(pair_units(d, method="m", metric="x"), n_boot=400)
        return r["hi"] - r["lo"]

    assert width(differ) > width(same) == 0


def test_misaligned_units_rejected():
    a = pair_units(rows({"g1": {"f": [("p1", [1, 1])]}}), method="m", metric="x")
    b = pair_units(rows({"g1": {"f": [("p1", [1, 1]), ("p2", [1, 1])]}}), method="m", metric="x")
    with pytest.raises(ValueError, match="aligned"):
        difference(a, b)


def test_multi_family_groups_resample_families_then_pairs():
    data = rows({"g1": {"fa": [("a1", [0.0] * 2), ("a2", [0.0] * 2)],
                        "fb": [("b1", [1.0] * 2), ("b2", [1.0] * 2)]}})
    result = cluster_bootstrap(pair_units(data, method="m", metric="x"), n_boot=2000, seed=1)
    assert result["estimate"] == 0.5 and result["lo"] == 0.0 and result["hi"] == 1.0
    assert result["n_pairs"] == 4
