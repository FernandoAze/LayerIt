import sys
from pathlib import Path
import json

# Add parent directory to path so we can import src
script_dir = Path(__file__).parent
root_dir = script_dir.parent
output_dir = root_dir/"output"
sys.path.insert(0, str(root_dir))
input_parent_dir = str("src/input_files/BWV856/Performance1")

from src.functions import *

audio_file = str(root_dir/input_parent_dir/"BWV856_AndrasSchiff.wav")
svg_score = str(root_dir/input_parent_dir/"SW andras.svg")
maps_file = str(root_dir/input_parent_dir/"andras.maps.json")
beat_file  = str(root_dir/input_parent_dir/"beat_example1.npz")

# tmp_dir: makes intermediate not shown
import tempfile 
tmp = tempfile.TemporaryDirectory()
tmp_dir = Path(tmp.name)

layer = Visualizer()
layer.add_layer(BeatAccurateLayer(line_width=0.7))
layer.add_layer(Waveform(color=(1, 0, 1), normalize=True))
layer.load_all_layers(audio_path=audio_file, beat_file=beat_file)
fig, ax = layer.draw()
Waveform_wBeatThis=layer.turn_to_SVG(str(tmp_dir/"Waveform_wBeatThis.svg"), svg_score, show_axes=True)

#EndVisualization: Combines the three layers with the score.
EndVisualization = Visualizer()
Layer_Width = EndVisualization.get_SVG_Root_Dimensions(svg_score)[0]
Layer_Height = EndVisualization.get_SVG_Root_Dimensions(svg_score)[1]

combination_of_layers = [(Waveform_wBeatThis, 25),(svg_score, Layer_Height + (2*25))]

EndVisualization.create_final_SVG(  width            = Layer_Width,
                                    height           = Layer_Height+50,
                                    svg_layers       = combination_of_layers,
                                    background_color = "#ffffff",
                                    output_file      = str(output_dir / "FIG524.svg"),
                                    print_output     = True)  
