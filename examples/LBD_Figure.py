import sys
from pathlib import Path

# Add parent directory to path so we can import src
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.insert(0, str(root_dir))
input_parent_dir = str("src/input_files/ClairDeLune/")

from src.functions import *

audio_file = str(root_dir / input_parent_dir / "ClairDeLune_MariaJoaoPires_untilM6.wav")
svg_score = str(root_dir / input_parent_dir / "ClairDeLune_MariaJoaoPires_untilM6.svg")
maps_file = str(root_dir / input_parent_dir / "ClairDeLune_MariaJoaoPires_untilM6.maps.json")
beat_file = str(root_dir / input_parent_dir / "Clair_Beat.npz")

fig = Visualizer(audio=audio_file, score=svg_score, maps=maps_file, beats=beat_file)

fig.add_panel(Onset(onset_color=(0, 0, 0), line_width=0.5, line_type="dashed"),
              BeatLogits(line_width=1.5, color=('#E69F00'), line_type="dotted"),
              DownbeatLogits(line_width=1.5, color=('#0072B2')),
              height_scale=0.5)

fig.add_panel(Waveform(color=(1, 0, 1), normalize=True),
              BeatsLayer(line_width=1.5, color=('#E69F00'),line_type="dotted"),
              DownbeatsLayer(line_width=1, color=('#0072B2')),
              height_scale=0.5)

fig.add_panel(MelSpec(freq_window=(100, 1500), color_map="summer"))

fig.compose("LDB_FIG.svg", print_output=True)