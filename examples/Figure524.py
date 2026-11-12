import sys
from pathlib import Path

# Add parent directory to path so we can import src
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.insert(0, str(root_dir))
input_parent_dir = str("src/input_files/BWV856/Performance1")

from src.functions import *

audio_file = str(root_dir / input_parent_dir / "BWV856_AndrasSchiff.wav")
svg_score = str(root_dir / input_parent_dir / "SW andras.svg")
maps_file = str(root_dir / input_parent_dir / "andras.maps.json")
beat_file = str(root_dir / input_parent_dir / "beat_example1.npz")

fig = Visualizer(audio=audio_file, score=svg_score, maps=maps_file, beats=beat_file)

fig.add_panel(Waveform(color=(0, 0, 1), normalize=True),
              BeatsLayer(line_width=0.7))

SVG_fig = fig.compose("FIG524.svg", print_output=True)