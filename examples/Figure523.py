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

# tmp_dir: makes intermediate not shown
import tempfile 
tmp = tempfile.TemporaryDirectory()
tmp_dir = Path(tmp.name)

#Creation of layers starts here. By default, Beat and Downbeat representations are set to Red and Blue respectively. 
#These may be altered by passing parameters to layer constructors (e.g., .add_layer(BeatAccurateLayer(beat_color=..., downbeat_color=...))..

#Layer 1: shows output of the BT algorithm, in this case BeatThis output
data_layer_1 = Visualizer()
data_layer_1.add_layer(BeatAccurateLayer(line_width=0.7))
data_layer_1.load_all_layers(audio_path=audio_file, beat_file=beat_file)
fig, ax = data_layer_1.draw()
AccurateBeat_layer=data_layer_1.turn_to_SVG(str(tmp_dir / "Data_Layer.svg"), svg_score)

#Layer 2: shows output of the BeatThis, Beat and Downbeat Probabilities. 
data_layer_2 = Visualizer()
data_layer_2.add_layer(BeatProbabilityLayer())
data_layer_2.add_layer(DownbeatProbabilityLayer())
data_layer_2.load_all_layers(audio_path=audio_file, beat_file=beat_file)
fig2, ax2 = data_layer_2.draw()
beatProbs_layer=data_layer_2.turn_to_SVG(str(tmp_dir / "Second_Data_Layer.svg"), svg_score)

#Layer 3: Shows a configurable window for each beat, beat_window makes a percentage threshold where a faded line starts to be displayed. 
data_layer_3 = Visualizer()
data_layer_3.add_layer(BeatWindowLayer(beat_window=0.2, alpha_max=0.5))
data_layer_3.add_layer(DownbeatWindowLayer(beat_window=0.5, alpha_max=0.7))
data_layer_3.load_all_layers(audio_path=audio_file, beat_file=beat_file)
fig3, ax3 = data_layer_3.draw()
BeatWindow_layer=data_layer_3.turn_to_SVG(str(tmp_dir / "Third_Data_Layer.svg"), svg_score)

#EndVisualization: Combines the three layers with the score.
EndVisualization = Visualizer()
Layer_Width = EndVisualization.get_SVG_Root_Dimensions(svg_score)[0]
Layer_Height = EndVisualization.get_SVG_Root_Dimensions(svg_score)[1]
combination_of_layers = [
    #Top Visual
    (AccurateBeat_layer, 10),
    (beatProbs_layer, 10),
    (svg_score, 10),
    
    #Bottom Visual
    (BeatWindow_layer,Layer_Height+(10*2)),
    (beatProbs_layer, Layer_Height+(10*2)),
    (svg_score, Layer_Height+(10*2))
]
EndVisualization.create_final_SVG(  
                                    width            = Layer_Width,
                                    height           = Layer_Height*3+(10*3),
                                    svg_layers       = combination_of_layers,
                                    background_color = "#ffffff",
                                    output_file      = str(output_dir / "FIG523.svg"),
                                    print_output     = True)