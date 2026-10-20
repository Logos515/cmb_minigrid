import xml.etree.ElementTree as ET

from cmb_minigrid.experiments.svg import forest_plot


def test_forest_plot_is_valid_svg_with_labels_values_and_tooltips(tmp_path):
    rows = [{"label": "CMB <4x16>", "est": 0.8, "lo": 0.5, "hi": 1.0, "role": "cmb",
             "points": [0.7, 0.9]},
            {"label": "Reset", "est": 0.0, "lo": 0.0, "hi": 0.0, "role": "baseline"},
            {"label": "no CI", "est": 0.4, "role": "ablation"}]
    path = forest_plot(tmp_path / "f.svg", rows, title="T & title", xlabel="x")
    root = ET.parse(path).getroot()  # raises if not well-formed (escaping)
    text = " ".join(t.text or "" for t in root.iter("{http://www.w3.org/2000/svg}text"))
    assert "CMB <4x16>" in text and "0.80" in text and "Reset" in text
    titles = [t.text for t in root.iter("{http://www.w3.org/2000/svg}title")]
    assert any("0.800 (95% CI [0.50, 1.00])" in (t or "") for t in titles)
    assert any("n/a" in (t or "") for t in titles)
