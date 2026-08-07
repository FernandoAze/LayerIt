import sys
from pathlib import Path
import json

# Add parent directory to path so we can import src
script_dir = Path(__file__).parent
root_dir = script_dir.parent
output_dir = root_dir / "output"
sys.path.insert(0, str(root_dir))
input_parent_dir = str("src/input_files/Chopin_op10_ScoreWarpDemo")

from src.functions import *

audio_file = str(root_dir / input_parent_dir / "Chopin_op10_no3_p11.wav")
svg_score = str(root_dir / input_parent_dir / "Chopin_op10_no3_p11-mei.maps.json.svg")
maps_file = str(root_dir / input_parent_dir / "My Chopin_op10_no3_p11-mei.maps.json")
beat_file  = str(root_dir / input_parent_dir / "CHOPIN_BEAT.npz")

Run_BeatThis(audio_path=audio_file, output_path=str(output_dir / "CHOPIN_BEAT.npz"))
#================================ Spectrogram Layer ===============================
viz_spec = Visualizer()
spectrogramConfig = {"freq_window": (100, 1500),"color_map": "summer"}
viz_spec.add_layer(Spectrogram(**spectrogramConfig))
viz_spec.load_all_layers(audio_path=audio_file)
fig, ax = viz_spec.draw()
Spectogram_Layer=viz_spec.turn_to_SVG(filename=str("SPECTROGRAM.svg"),
                     svg_warped_score =svg_score,
                     show_axes        =True)
#================================ Waveform Layer ===============================
waveform_layer=Visualizer()
waveform_layer.add_layer(Waveform(color=(1, 0, 1), normalize=True))
waveform_layer.add_layer(BeatAccurateLayer(line_width=1))
waveform_layer.load_all_layers(audio_path=audio_file, maps_file=maps_file, beat_file=beat_file)
fig, ax = waveform_layer.draw()
Waveform_Layer=waveform_layer.turn_to_SVG(filename=str("Waveform_Layer.svg"), svg_warped_score=svg_score, show_axes=True)
#================================ Onset Layer ===============================
onset_layer = Visualizer()
onset_layer.add_layer(Onsets_Layer(onset_color=(0, 0, 0), line_width=0.3))
onset_layer.load_all_layers(audio_path=audio_file, maps_file=maps_file)
fig, ax = onset_layer.draw()
Onset_Layer=onset_layer.turn_to_SVG(filename=str("Onset_Layer.svg"),
                                    svg_warped_score    =svg_score,
                                    print_output        =False)
#================================ BeatProb Layer ===============================
beatProbs_layer = Visualizer()
beatProbs_layer.add_layer(BeatProbabilityLayer(line_width=0.7))
beatProbs_layer.add_layer(DownbeatProbabilityLayer(line_width=0.7))
beatProbs_layer.load_all_layers(audio_path=audio_file, beat_file=beat_file)
fig, ax = beatProbs_layer.draw()
beatProb=beatProbs_layer.turn_to_SVG(filename=str("beatProbs_Layer.svg"),svg_warped_score=svg_score, show_axes=True)
#================================ BeatProb Layer ===============================
EndVisualization = Visualizer()
Layer_Width = onset_layer.get_SVG_Root_Dimensions(Waveform_Layer)[0]
Layer_Height = onset_layer.get_SVG_Root_Dimensions(Waveform_Layer)[1]
svg_layers_to_stack = [
    (svg_score,0),

    (Spectogram_Layer, Layer_Height-5),
    (Onset_Layer, Layer_Height-5),

    (Onset_Layer, Layer_Height*2+15),
    (beatProb, Layer_Height*2+15),

    (Onset_Layer, Layer_Height*3+35),
    (Waveform_Layer, Layer_Height*3+35)]
EndVisualization.create_final_SVG(  width              =Layer_Width,
                                    height             =Layer_Height*4+60,
                                    background_color   = "#ffffff",
                                    svg_layers         = svg_layers_to_stack,
                                    output_file        = str(output_dir / "LDB_FIG.svg"),
                                    print_output       = True)