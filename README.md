# LayerIt!

**LayerIt!** is a modular, composable Python framework for the synchronized visualization of music performances and their corresponding musical scores. It integrates audio analysis, automatic beat tracking, and score alignment into a unified layer-based rendering pipeline, producing publication-quality raster (PNG) and vector (SVG) outputs.

Presented as a Late-Breaking Demo at **ISMIR 2026**.

---

## Overview

LayerIt! addresses a core challenge in Music Information Retrieval (MIR) research: producing synchronized, multi-modal visualizations that overlay acoustic features (spectrograms, beat events) with symbolic score representations in a shared time axis. The system is built around an extensible **layer architecture** — each visualization element is an independent, reusable component that can be freely composed at runtime.

---

## Requirements

- Python 3.12.3+

---

## Installation

### 1. Create and Activate a Virtual Environment

```bash
python3.12 -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install the Package

LayerIt is pip-installable directly from the repository root:

```bash
pip install --upgrade pip
pip install .

# Or, for an editable/development install:
pip install -e .
```

This installs the `layerit` package along with all runtime dependencies declared in [pyproject.toml](pyproject.toml).

Alternatively, to install dependencies only (e.g. when developing against `src/functions` directly without installing the package):

```bash
pip install -r requirements.txt
```

**Core dependencies:** `librosa`, `matplotlib`, `numpy`, `soundfile`, `scikit-learn`, `Pillow`

---

## Architecture

### Layer-Based Composition

All visualization components inherit from the abstract base class `Layer` ([src/functions/visualization_system.py](src/functions/visualization_system.py)) and must implement two methods:

| Method | Signature | Description |
|---|---|---|
| `load_data` | `(**kwargs) → bool` | Load and validate data; return `True` on success |
| `draw` | `(ax, shared_data) → (lines, labels)` | Draw onto a Matplotlib axis |

An optional `to_svg_group(shared_data)` method can be implemented to support vector SVG export.

Layers are assembled and rendered by a `Visualizer` instance, which manages the shared rendering context (`shared_data`) and coordinates data loading and drawing across all layers in the stack.

### Pipeline

```
Audio (.wav)
  └─ Spectrogram() / Chromagram() / Waveform() ─────────────────────┐
                                                                     ▼
Beat .npz (external) ────────────────────────────────────────────── ┤
                                                                     ▼
MAPS JSON + Warped Score SVG  →  Onset / Warp_Score  ──────────── Visualizer
                                                                     │
                                                         ┌───────────┴───────────┐
                                                         ▼                       ▼
                                                   PNG export               SVG export
                                                                       (per-layer groups)
                                                                               │
                                                                               ▼
                                                                         compose()
                                                                  → Composite multi-panel SVG
```

---

## Modules

### `src/functions/visualization_system.py`
- **`Layer`** — Abstract base class for all visualization layers.
- **`Visualizer`** — Orchestrates layer composition, data loading, drawing, and export. Key methods:
  - `add_panel(*layers, height_scale, show_axes)` — Add a panel (one or more layers stacked on the same axis)
  - `compose(output_file, gap, score_position, print_output)` — Render all panels and the score into a single composite SVG

### `src/functions/Beat_Layers.py`
Beat visualization layers that consume a pre-computed beat `.npz` file.

| Class / Function | Description |
|---|---|
| `BeatLogits` | Plots the raw beat logit curve as a line overlay |
| `DownbeatLogits` | Plots the raw downbeat logit curve as a line overlay |
| `BeatsLayer` | Renders detected beat positions (excluding downbeats) as vertical markers |
| `DownbeatsLayer` | Renders detected downbeat positions as vertical markers |
| `BeatAccurateLayer` | Legacy combined beats + downbeats marker layer |
| `BeatWindowLayer` | Highlights beat confidence windows; opacity gradient from threshold to peak |
| `DownbeatWindowLayer` | Highlights downbeat confidence windows; opacity gradient from threshold to peak |
| `NPZ_to_BeatTXT` | Writes detected beats (excluding downbeats) from a `.npz` to a tab-separated TXT file |
| `NPZ_to_DownbeatTXT` | Writes detected downbeats from a `.npz` to a tab-separated TXT file |

All beat layers share a secondary logit y-axis (`ax2`) via the `BeatLayer` base class. A logit of `0` corresponds to a sigmoid probability of 50%. The expected `.npz` keys are `beat_times`, `beat_activation`, `downbeat_activation`, `detected_beats`, and `detected_downbeats`.

### `src/functions/Audio_Layers.py`
| Class | Description |
|---|---|
| `Spectrogram` | Mel-scale spectrogram. Configurable frequency window (`freq_window`) and colormap (`color_map`) |
| `Chromagram` | Chroma (pitch class) representation via CQT, with pitch class labels |
| `Waveform` | Amplitude waveform, optionally normalized (`normalize`) |

### `src/functions/warp_score.py`
Utilities for score–audio alignment and SVG compositing.

| Class / Function | Description |
|---|---|
| `Onsets_Layer` | Renders note onset times (from MAPS JSON) as vertical dashed lines |
| `Warp_Score` | Parses warped score SVGs; extracts time axis bounds, element translations, and viewBox geometry for pixel-accurate alignment |
| `TXT_to_Maps(txt_maps_file, output_file)` | Converts tab-separated onset annotation files to `.maps.json` format |

---

## Available Layers at a Glance

```python
from layerit import (
    MelSpec,
    Chromagram,
    Waveform,
    Onset,
    BeatLogits,
    DownbeatLogits,
    BeatAccurateLayer,
    BeatsLayer,
    DownbeatsLayer,
    BeatWindowLayer,
    DownbeatWindowLayer,
)
```

If `layerit` is not installed (e.g. running scripts directly from the repository without `pip install .`), import from `src.functions` instead — see [examples/](examples/) for this pattern.

---

## Data Requirements

| File | Format | Description |
|---|---|---|
| Audio recording | `.wav` | The performance to analyse |
| Beat data | `.npz` | Pre-computed beat/downbeat logits and events (see keys below) |
| Warped score | `.svg` | Score image with an embedded `timeAxis` group (output of ScoreWarp) |
| Alignment maps | `.maps.json` | Array of `{ obs_mean_onset, xml_id }` entries mapping score elements to performance time |

The `.npz` must contain the keys `beat_times`, `beat_activation`, `downbeat_activation`, `detected_beats`, and `detected_downbeats`. Generate it with any beat tracker and save with `numpy.savez`.

### Obtaining Prerequisite Files

- **Score alignment (MAPS)** — Use [trompa-align](https://github.com/trompamusic/trompa-align)
- **Score warping (SVG)** — Use [ScoreWarp](https://iwk-digital.github.io/scorewarp/) to generate a time-axis-annotated SVG from MEI or MusicXML
  - [Verovio Online Editor](https://editor.verovio.org/) — preview and edit MEI files
- **Beat data (.npz)** — Run any beat tracker and save results with `numpy.savez` using the keys listed in the Data Requirements section.

---

## Usage

### Minimal Example

```python
from layerit import Visualizer, MelSpec, BeatLogits, Onset

fig = Visualizer(audio="recording.wav", score="score.svg", maps="alignment.maps.json", beats="beats.npz")
fig.add_panel(MelSpec(freq_window=(20, 2000), color_map="magma"),
              BeatLogits(),
              Onset(onset_color='white', line_width=0.3))
fig.compose("output.svg")
```

### Full Example

See [examples/LBD_Figure.py](examples/LBD_Figure.py) for the complete case study figure from the paper: a three-panel layout with beat logit curves, a waveform with beat markers, and a mel spectrogram with onsets.

### Converting Annotation Files

`TXT_to_Maps` is not re-exported at the package level; import it directly from `warp_score`:

```python
from layerit.warp_score import TXT_to_Maps

TXT_to_Maps("annotations/performance.txt", output_file="performance.maps.json")
```

Input format (tab-separated):
```
0.034829932    a13d7g5m
0.470204082    v2xdb2q
```

`NPZ_to_BeatTXT` and `NPZ_to_DownbeatTXT` extract detected events from a `.npz` to plain text:

```python
from layerit import NPZ_to_BeatTXT, NPZ_to_DownbeatTXT

NPZ_to_BeatTXT("beats.npz", output_file="beats.txt")
NPZ_to_DownbeatTXT("beats.npz", output_file="downbeats.txt")
```

---

## Output

All outputs are directed to the `output/` directory by default.

| Format | Method | Notes |
|---|---|---|
| SVG (multi-panel) | `compose()` | Panels + warped score stacked vertically; each layer is a named `<g class="layer ...">` group |

---

## Project Structure

```
LayerIt/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── functions/
│   │   ├── __init__.py
│   │   ├── visualization_system.py   # Layer ABC + Visualizer
│   │   ├── shapes.py                 # Curve, Events, Intervals, Field base classes
│   │   ├── Beat_Layers.py            # Beat logit/marker/window layers
│   │   ├── Audio_Layers.py           # MelSpec, Chromagram, Waveform
│   │   └── warp_score.py             # Onset, Warp_Score, TXT_to_Maps
│   └── input_files/
│       └── ClairDeLune/              # Case study: Clair de Lune, bars 1–6
│           ├── clair-de-lune M6-basic.mei
│           ├── ClairDeLune_MariaJoaoPires_untilM6.maps.json
│           ├── ClairDeLune_MariaJoaoPires_untilM6.svg
│           ├── ClairDeLune_MariaJoaoPires_untilM6.wav  # 6-bar excerpt
│           └── Clair_Beat.npz
├── examples/
│   ├── LBD_Figure.py                 # Reproduces the paper figure
│   ├── Turn_txt_into_MAPS.py
│   └── BEAT_to_TXT.py
└── output/
    └── LDB_FIG.svg                   # Rendered figure
```

---

## Resources

- [ScoreWarp](https://iwk-digital.github.io/scorewarp/) — Generate time-axis-annotated SVGs from MEI or MusicXML
- [trompa-align](https://github.com/trompamusic/trompa-align) — Score-to-performance alignment, produces MAPS JSON
- [Verovio Online Editor](https://editor.verovio.org/) — Visualise and edit MEI score files
- [MusicXML Converter](https://musicxml.tools/converter) — Convert `.mxl` to `.musicxml`

