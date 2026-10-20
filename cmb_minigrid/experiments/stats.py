"""Clustered bootstrap over RSSM-training groups, map families and pairs.

The unit of analysis is a *pair* (both endpoints share ``obs_t`` and are scored
together), nested in a map family, nested in an RSSM training-seed group. A
bootstrap replicate resamples groups with replacement; within each chosen group
it resamples families (only when the split has more than one) and then pairs
within each chosen family, so a pair's two endpoints are never separated and
whole groups are dropped/duplicated together. The statistic is the mean over
groups of the within-group pair mean. Percentile intervals, fixed RNG seed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

Units = dict[str, dict[str, np.ndarray]]  # group -> family -> pair-level values


def pair_units(rows: Iterable[Mapping[str, Any]], *, method: str, metric: str,
               split: str | None = None, method_key: str = "method") -> Units:
    """Pair-level means of ``metric`` for one method; pairs are sorted so methods align."""
    cells: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r[method_key] != method or (split is not None and r["split"] != split):
            continue
        cells[(r["group"], r["map_family_id"], r["pair_id"])].append(float(r[metric]))
    units: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for (group, family, pair), values in sorted(cells.items()):
        units[group][family].append((pair, float(np.mean(values))))
    return {g: {f: np.array([v for _, v in pairs]) for f, pairs in fams.items()}
            for g, fams in units.items()}


def _same_shape(a: Units, b: Units) -> None:
    if {g: {f: len(v) for f, v in fams.items()} for g, fams in a.items()} != {
            g: {f: len(v) for f, v in fams.items()} for g, fams in b.items()}:
        raise ValueError("units are not aligned (different groups/families/pairs)")


def difference(a: Units, b: Units) -> Units:
    """Pair-wise a - b (both from ``pair_units`` on the same rows)."""
    _same_shape(a, b)
    return {g: {f: fams[f] - b[g][f] for f in fams} for g, fams in a.items()}


def point_estimate(units: Units) -> float:
    return float(np.mean([np.concatenate(list(fams.values())).mean()
                          for fams in units.values()]))


def per_group_means(units: Units) -> dict[str, float]:
    return {g: float(np.concatenate(list(fams.values())).mean()) for g, fams in sorted(units.items())}


def _group_means(by_family: dict[str, np.ndarray], n_boot: int, rng) -> np.ndarray:
    """Bootstrap means of one group: resample families (if several), then pairs, vectorized."""
    families = sorted(by_family)
    if len(families) == 1:
        values = by_family[families[0]]
        return values[rng.integers(len(values), size=(n_boot, len(values)))].mean(1)
    picks = rng.integers(len(families), size=(n_boot, len(families)))
    total, count = np.zeros(n_boot), np.zeros(n_boot)
    for slot in range(len(families)):
        for index, family in enumerate(families):
            values = by_family[family]
            chosen = picks[:, slot] == index
            if chosen.any():
                draws = rng.integers(len(values), size=(int(chosen.sum()), len(values)))
                total[chosen] += values[draws].sum(1)
                count[chosen] += len(values)
    return total / count


def cluster_bootstrap(units: Units, *, n_boot: int = 4000, seed: int = 0,
                      level: float = 0.95) -> dict[str, Any]:
    if not units:
        raise ValueError("no units to bootstrap")
    rng = np.random.default_rng(seed)
    groups = sorted(units)
    means = np.stack([_group_means(units[g], n_boot, rng) for g in groups])  # [G, B]
    chosen = rng.integers(len(groups), size=(len(groups), n_boot))
    estimates = means[chosen, np.arange(n_boot)].mean(0)
    alpha = (1 - level) / 2
    return {"estimate": point_estimate(units),
            "lo": float(np.quantile(estimates, alpha)),
            "hi": float(np.quantile(estimates, 1 - alpha)),
            "n_groups": len(groups),
            "n_pairs": int(sum(len(v) for fams in units.values() for v in fams.values())),
            "n_boot": n_boot, "level": level}
