"""Dependency-free SVG forest plots (estimate, 95% CI whisker, per-seed dots).

Palette slots follow the validated reference palette (blue, orange; neutral gray
for baselines, ink for the Oracle). Every row carries its label and numeric value
as text, so identity never depends on color alone; each mark has a native SVG
``<title>`` tooltip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ROLE_COLORS = {"cmb": "#2a78d6", "ablation": "#eb6834", "baseline": "#8a8984", "oracle": INK}
FONT = "font-family=\"system-ui, -apple-system, 'Segoe UI', sans-serif\""


def forest_plot(path: str | Path, rows: list[dict[str, Any]], *, title: str, xlabel: str,
                subtitle: str = "", xlim: tuple[float, float] = (0.0, 1.0),
                width: int = 820) -> Path:
    """rows: ``label``, ``est``, optional ``lo``/``hi``, ``role``, ``points`` (per-seed values)."""
    left, right, top, row_h = 210, 110, 70, 26
    height = top + row_h * len(rows) + 52
    plot_w = width - left - right
    lo_x, hi_x = xlim

    def x(v: float) -> float:
        return left + (min(max(v, lo_x), hi_x) - lo_x) / (hi_x - lo_x) * plot_w

    out = [(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'width="{width}" height="{height}" role="img" aria-label="{escape(title)}">'),
           f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
           (f'<text x="16" y="24" {FONT} font-size="15" font-weight="600" fill="{INK}">'
           f'{escape(title)}</text>')]
    if subtitle:
        out.append(f'<text x="16" y="44" {FONT} font-size="11.5" fill="{MUTED}">'
                   f'{escape(subtitle)}</text>')
    ticks = [lo_x + (hi_x - lo_x) * i / 4 for i in range(5)]
    for t in ticks:
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" '
                   f'y2="{top + row_h * len(rows)}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x(t):.1f}" y="{top + row_h * len(rows) + 16}" text-anchor="middle" '
                   f'{FONT} font-size="11" fill="{MUTED}">{t:g}</text>')
    out.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 10}" text-anchor="middle" {FONT} '
               f'font-size="12" fill="{MUTED}">{escape(xlabel)}</text>')
    for i, row in enumerate(rows):
        cy = top + row_h * i + row_h / 2
        color = ROLE_COLORS.get(row.get("role", "baseline"), ROLE_COLORS["baseline"])
        est, lo, hi = row["est"], row.get("lo"), row.get("hi")
        interval = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else "n/a"
        tip = f'{row["label"]}: {est:.3f} (95% CI {interval})'
        out.append(f'<g><title>{escape(tip)}</title>')
        out.append(f'<text x="{left - 10}" y="{cy + 4:.1f}" text-anchor="end" {FONT} '
                   f'font-size="12" fill="{INK}">{escape(row["label"])}</text>')
        if lo is not None and hi is not None:
            out.append(f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" '
                       f'stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
        for k, p in enumerate(row.get("points", [])):
            jitter = ((k % 5) - 2) * 2.2
            out.append(f'<circle cx="{x(p):.1f}" cy="{cy + jitter:.1f}" r="2.6" fill="none" '
                       f'stroke="{MUTED}" stroke-width="1"><title>seed group {k + 1}: '
                       f'{p:.3f}</title></circle>')
        out.append(f'<circle cx="{x(est):.1f}" cy="{cy:.1f}" r="4.5" fill="{color}" '
                   f'stroke="{SURFACE}" stroke-width="2"/>')
        out.append(f'<text x="{left + plot_w + 10}" y="{cy + 4:.1f}" {FONT} font-size="11.5" '
                   f'fill="{INK}">{est:.2f}</text>')
        out.append("</g>")
    out.append("</svg>")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(out), encoding="utf-8")
    return target
