import sys
from pathlib import Path
import json

# Add parent directory to path so we can import src
script_dir = Path(__file__).parent
root_dir = script_dir.parent
output_dir = root_dir / "output"
sys.path.insert(0, str(root_dir))
input_parent_dir = str("src/input_files/BWV856/Performance1")

from src.functions import *

audio_file = str(root_dir / input_parent_dir / "BWV856_AndrasSchiff.wav")
svg_score = str(root_dir / input_parent_dir / "SW andras.svg")
maps_file = str(root_dir / input_parent_dir / "andras.maps.json")
beat_file  = str(root_dir / input_parent_dir / "beat_example1.npz")

import tempfile
tmp = tempfile.TemporaryDirectory()
tmp_dir = Path(tmp.name)

#================================ Spectrogram Layer ===============================
spectrogramConfig = {
    "freq_window": (20, 2000),
    "color_map": "cool"
}
viz_spec = Visualizer()
viz_spec.add_layer(Spectrogram(**spectrogramConfig))
viz_spec.load_all_layers(audio_path=audio_file)
fig, ax = viz_spec.draw()
Spectogram_Layer=viz_spec.turn_to_SVG(filename=str(tmp_dir /"SPECTROGRAM.svg"),svg_warped_score=svg_score, show_axes=True)
#================================ Beat/Downbeat Layer ===============================
data_layer = Visualizer()
data_layer.add_layer(BeatAccurateLayer())
data_layer.add_layer(Onsets_Layer(onset_color=(0, 0, 0), line_width=0.5))
data_layer.load_all_layers(audio_path=audio_file, maps_file=maps_file, beat_file=beat_file)
fig, ax = data_layer.draw()
Data_Layer=data_layer.turn_to_SVG(filename=str(tmp_dir/"Data_Layer.svg"),svg_warped_score=svg_score)

Layer_Width = data_layer.get_SVG_Root_Dimensions(svg_score)[0]
Layer_Height = data_layer.get_SVG_Root_Dimensions(svg_score)[1]

svg_layers_to_stack = [
    (Spectogram_Layer, 5),
    (Data_Layer, 5),

    (svg_score, Layer_Height + (2*5)),
    (Data_Layer, Layer_Height + (2*5))]

EndVisualization = Visualizer()
EndVisualization.create_final_SVG(width=Layer_Width,height=2*(Layer_Height+10), svg_layers=svg_layers_to_stack, background_color="#ffffff", output_file=str(output_dir/"FIG522.svg"), print_output=True)
