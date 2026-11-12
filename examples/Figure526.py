import sys
from pathlib import Path

# Add parent directory to path so we can import src
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.insert(0, str(root_dir))
input_parent_dir = str("src/input_files/Mozart_12-1_WuuE02_ASAP")

from src.functions import *

audio_file = str(root_dir / input_parent_dir / "WuuE02M.wav")
maps_file = str(root_dir / input_parent_dir / "Mozart_PP.maps.json")
svg_score = str(root_dir / input_parent_dir / "Mozart_PianoSonata_12-1.svg")


fig = Visualizer(audio=audio_file, score=svg_score, maps=maps_file)

fig.add_panel(Waveform(color="#FFD500", normalize=True),
              Onset(onset_color="#000000", line_width=0.5),
              height_scale=0.5)

fig.add_panel(MelSpec(freq_window=(40, 1600), color_map="cool"),
              Onset(onset_color="#000000", line_width=0.5),
              height_scale=0.5)

fig.add_panel(Chromagram(color_map="cool"),
              height_scale=0.5)

SVG_fig = fig.compose("FIG526.svg", print_output=True)