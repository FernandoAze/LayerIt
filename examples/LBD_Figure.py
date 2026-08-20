import sys
from pathlib import Path
import json

# Add parent directory to path so we can import src
script_dir = Path(__file__).parent
root_dir = script_dir.parent
output_dir = root_dir / "output"
sys.path.insert(0, str(root_dir))
input_parent_dir = str("src/input_files/PreludeN2")

from src.functions import *

audio_file = str(root_dir / input_parent_dir / "Prelude_n2_Cm.wav")
svg_score = str(root_dir / input_parent_dir / "warped_0826.svg")
maps_file = str(root_dir / input_parent_dir / "OnsetsMei_PreludioN2.maps.json")
beat_file  = str(root_dir / input_parent_dir / "beat_Bach.npz")

#================================ Spectrogram Layer ===============================
viz_spec = Visualizer()
spectrogramConfig = {"freq_window": (100, 1500),"color_map": "summer"}
viz_spec.add_layer(Spectrogram(**spectrogramConfig))
viz_spec.add_layer(Onsets_Layer(onset_color=(0, 0, 0), line_width=0.3))
viz_spec.load_all_layers(audio_path=audio_file, maps_file=maps_file)
fig, ax = viz_spec.draw()
Spectogram_Layer=viz_spec.turn_to_SVG(filename=str("SPECTROGRAM.svg"),
                     svg_warped_score =svg_score,
                     show_axes        =True)
#================================ Waveform Layer ===============================
waveform_layer=Visualizer()
waveform_layer.add_layer(Waveform(color=(1, 0, 1), normalize=True))
waveform_layer.add_layer(BeatAccurateLayer(line_width=1))
waveform_layer.add_layer(Onsets_Layer(onset_color=(0, 0, 0), line_width=0.3))
waveform_layer.load_all_layers(audio_path=audio_file, maps_file=maps_file, beat_file=beat_file)
fig, ax = waveform_layer.draw()
Waveform_Layer=waveform_layer.turn_to_SVG(filename=str("Waveform_Layer.svg"), svg_warped_score=svg_score, show_axes=True)
#================================ Beat/Downbeat Activation Layer ===============================
beatActivation_layer = Visualizer()
beatActivation_layer.add_layer(BeatProbabilityLayer(line_width=0.7))
beatActivation_layer.add_layer(DownbeatProbabilityLayer(line_width=0.7))
beatActivation_layer.add_layer(Onsets_Layer(onset_color=(0, 0, 0), line_width=0.3))
beatActivation_layer.load_all_layers(audio_path=audio_file, beat_file=beat_file, maps_file=maps_file)
fig, ax = beatActivation_layer.draw()
beatActivation_layer=beatActivation_layer.turn_to_SVG(filename=str("beatActivation_layer.svg"), svg_warped_score=svg_score, show_axes=True)

EndVisualization = Visualizer()
Layer_Width = EndVisualization.get_SVG_Root_Dimensions(Waveform_Layer)[0]
Layer_Height = EndVisualization.get_SVG_Root_Dimensions(Waveform_Layer)[1]
svg_layers_to_stack = [
    (svg_score,0),
    (Spectogram_Layer, Layer_Height-5),
    (beatActivation_layer, Layer_Height*2+15),
    (Waveform_Layer, Layer_Height*3+35)]
EndVisualization.create_final_SVG(  width              =Layer_Width,
                                    height             =Layer_Height*4+60,
                                    background_color   = "#ffffff",
                                    svg_layers         = svg_layers_to_stack,
                                    output_file        = str(output_dir / "LDB_FIG.svg"),
                                    print_output       = True)