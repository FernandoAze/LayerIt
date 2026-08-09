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

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Core dependencies:** `librosa`, `matplotlib`, `numpy`, `soundfile`, `torch`, `beat_this`, `modusa`

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
  ├─ Run_BeatThis()  →  beat_data.npz   ──────────────────────┐
  └─ Spectrogram() / Chromagram() / Waveform() ────────────────┤
                                                               ▼
MAPS JSON + Warped Score SVG  →  Onsets_Layer / Warp_Score  → Visualizer
                                                               │
                                                    ┌──────────┴──────────┐
                                                    ▼                     ▼
                                               PNG export           SVG export
                                                                  (per-layer groups)
                                                                         │
                                                                         ▼
                                                               combine_layers_with_score()
                                                               → Composite score + layers SVG
```

---

## Modules

### `src/functions/visualization_system.py`
- **`Layer`** — Abstract base class for all visualization layers.
- **`Visualizer`** — Orchestrates layer composition, data loading, drawing, and export. Key methods:
  - `add_layer(layer)` — Register a layer
  - `load_all_layers(audio_path, **kwargs)` — Load all layers; computes audio duration into `shared_data`
  - `draw()` — Render all layers onto a single Matplotlib figure
  - `turn_to_PNG(filename, svg_warped_score, dpi, print_output)` — Export rasterized PNG with exact pixel dimensions
  - `turn_to_SVG(filename, svg_warped_score, show_axes, print_output)` — Export vector SVG with each layer as a named `<g>` group
  - `combine_layers_with_score(filename, original_score, layers_svg, maps_file, PNG_layer, show_score, print_output)` — Composites the layer SVG (and optional PNG) onto the time-aligned score SVG
  - `create_final_SVG(width, height, svg_layers, output_file, background_color, print_output)` — Stacks multiple SVG/PNG visualizations at given y-offsets into a single composite SVG
  - `get_SVG_Root_Dimensions(svg_warped_score)` — Reads width/height from an SVG's root element
  - `get_timeAxis_attributes(svg_warped_score)` — Reads the total timeline duration and pixel length from a warped score's `timeAxis` group

### `src/functions/Beat_Layers.py`
Beat visualization layers built on top of the [**BeatThis!**](https://github.com/CPJKU/beat_this) beat tracking algorithm.

| Class / Function | Description |
|---|---|
| `Run_BeatThis(audio_path, output_path)` | Runs BeatThis! inference on a WAV file; saves beat/downbeat logits and detected events to `.npz` |
| `BeatProbabilityLayer` | Plots the raw beat probability curve as a line overlay |
| `DownbeatProbabilityLayer` | Plots the raw downbeat probability curve as a line overlay |
| `BeatAccurateLayer` | Renders detected beat and downbeat positions as vertical marker lines |
| `BeatWindowLayer` | Highlights beat confidence windows; opacity gradient from threshold to peak |
| `DownbeatWindowLayer` | Highlights downbeat confidence windows; opacity gradient from threshold to peak |

All beat layers share a secondary y-axis (`ax2`, 0–100%) via the `BeatLayer` base class.

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
from src.functions import (
    Spectrogram,
    Chromagram,
    Waveform,
    Onsets_Layer,
    BeatProbabilityLayer,
    DownbeatProbabilityLayer,
    BeatAccurateLayer,
    BeatWindowLayer,
    DownbeatWindowLayer,
)
```

---

## Data Requirements

| File | Format | Description |
|---|---|---|
| Audio recording | `.wav` | The performance to analyse |
| Beat analysis | `.npz` | Output of `Run_BeatThis()`; contains logits and detected events |
| Warped score | `.svg` | Score image with an embedded `timeAxis` group (output of ScoreWarp) |
| Alignment maps | `.maps.json` | Array of `{ obs_mean_onset, xml_id }` entries mapping score elements to performance time |

### Obtaining Prerequisite Files

- **Score alignment (MAPS)** — Use [trompa-align](https://github.com/trompamusic/trompa-align)
  > Weigl, D. (2020). *Multimodal Music Information Alignment*. TROMPA Deliverable TR-D3.5.
- **Score warping (SVG)** — Use [ScoreWarp](https://github.com/) to generate a time-axis-annotated SVG from MEI or MusicXML
  - [Verovio Online Editor](https://editor.verovio.org/) — preview and edit MEI files
  - [MusicXML Converter](https://musicxml.tools/converter) — convert `.mxl` to `.musicxml`

---

## Usage

### Minimal Example

```python
from src.functions import Visualizer, Spectrogram, BeatProbabilityLayer, Onsets_Layer

viz = Visualizer()
viz.add_layer(Spectrogram(freq_window=(20, 2000), color_map="magma"))
viz.add_layer(BeatProbabilityLayer(color='red'))
viz.add_layer(Onsets_Layer(onset_color='white'))

viz.load_all_layers(
    audio_path="path/to/recording.wav",
    beat_file="path/to/beat_data.npz",
    maps_file="path/to/alignment.maps.json"
)

fig, ax = viz.draw()
viz.turn_to_PNG("output/visualization.png", svg_warped_score="path/to/score.svg", dpi=300)
```

### Full Pipeline Example

See [examples/BWV856_EXAMPLE2.py](examples/BWV856_EXAMPLE2.py) for a complete demonstration including:
1. Beat detection with `Run_BeatThis()`
2. Spectrogram PNG export
3. Layer SVG export
4. Score compositing with `combine_layers_with_score()`

See [examples/BWV856_COMBINE.py](examples/BWV856_COMBINE.py) and [examples/BWV856_EXAMPLE3 SelfComparison copy.py](examples/BWV856_EXAMPLE3%20SelfComparison%20copy.py) for stacking multiple performance visualizations into a single comparative SVG using `create_final_SVG()`.

See [examples/Figure521.py](examples/Figure521.py), [examples/Figure522.py](examples/Figure522.py), [examples/Figure523.py](examples/Figure523.py) and [examples/Figure524.py](examples/Figure524.py) for smaller, focused examples combining `Spectrogram`, `Chromagram`, `Waveform`, `Onsets_Layer` and the beat layers.

### Converting Annotation Files

`TXT_to_Maps` is not re-exported at the package level; import it directly from `warp_score`:

```python
from src.functions.warp_score import TXT_to_Maps

TXT_to_Maps("annotations/performance.txt", output_file="performance.maps.json")
```

Input format (tab-separated):
```
0.034829932    a13d7g5m
0.470204082    v2xdb2q
```

---

## Output

All outputs are directed to the `output/` directory by default.

| Format | Method | Notes |
|---|---|---|
| PNG | `turn_to_PNG()` | Rasterized at configurable DPI; dimensions matched to warped score |
| SVG (layers) | `turn_to_SVG()` | Vector; each layer rendered as a named `<g class="layer ...">` group |
| SVG (composite) | `combine_layers_with_score()` | Layers overlaid on the time-aligned score SVG |
| SVG (multi-performance) | `create_final_SVG()` | Multiple visualizations stacked vertically |

---

## Project Structure

```
LayerIt!/
├── README.md
├── requirements.txt
├── agents.md
├── src/
│   ├── functions/
│   │   ├── __init__.py
│   │   ├── visualization_system.py   # Layer ABC + Visualizer
│   │   ├── Beat_Layers.py            # Run_BeatThis + BeatThis! layers
│   │   ├── Audio_Layers.py           # Spectrogram, Chromagram, Waveform layers
│   │   └── warp_score.py             # Onsets_Layer, Warp_Score, TXT_to_Maps
│   └── input_files/
│       ├── BWV856/
│       │   ├── BWV856.mei
│       │   ├── bwv856 LouJ01 asap.maps
│       │   ├── Performance1/         # Andras Schiff
│       │   ├── Performance2/         # Glenn Gould
│       │   ├── Performance3/         # Marta Argherich
│       │   └── TXTS/                 # Raw onset annotations (.txt)
│       ├── Chopin_op10_ScoreWarpDemo/
│       ├── ClairDeLune/
│       └── PreludeN2/
├── examples/
│   ├── BWV856_EXAMPLE1.py
│   ├── BWV856_EXAMPLE2.py
│   ├── BWV856_EXAMPLE2_SelfComparison.py
│   ├── BWV856_EXAMPLE3 SelfComparison copy.py
│   ├── BWV856_COMBINE.py
│   ├── BWV856_FIGURE5_2_1.py
│   ├── Figure521.py
│   ├── Figure522.py
│   ├── Figure523.py
│   ├── Figure524.py
│   ├── LBD_Figure.py
│   └── Turn_txt_into_MAPS.py
└── output/
```

---

## Resources

- [BeatThis!](https://github.com/CPJKU/beat_this) — Beat tracking model used for beat/downbeat analysis
- [trompa-align](https://github.com/trompamusic/trompa-align) — Score-to-performance alignment, produces MAPS JSON
- [Verovio Online Editor](https://editor.verovio.org/) — Visualise and edit MEI score files
- [MusicXML Converter](https://musicxml.tools/converter) — Convert `.mxl` to `.musicxml`

