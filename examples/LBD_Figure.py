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

#tmp_dir: is used to hide intermediate files
import tempfile
tmp = tempfile.TemporaryDirectory()
tmp_dir = Path(tmp.name)
#================================ Spectrogram Layer ===============================
viz_spec = Visualizer()
spectrogramConfig = {"freq_window": (100, 1500),"color_map": "summer"}
viz_spec.add_layer(Spectrogram(**spectrogramConfig))
viz_spec.load_all_layers(audio_path=audio_file)
fig, ax = viz_spec.draw()
Spectogram_Layer=viz_spec.turn_to_SVG(filename=str(  tmp_dir / "SPECTROGRAM.svg"),
                     svg_warped_score =svg_score,
                     show_axes        =True)
#================================ Chromagram Layer ===============================
chroma_layer=Visualizer()
chroma_layer.add_layer(Chromagram(color_map="summer"))
chroma_layer.load_all_layers(audio_path=audio_file)
Chroma_Layer=chroma_layer.turn_to_PNG(filename=str(tmp_dir / "Chroma_Layer.png"), svg_warped_score=svg_score, dpi=300)
#================================ Waveform Layer ===============================
waveform_layer=Visualizer()
waveform_layer.add_layer(Waveform(color=(1, 0, 1), normalize=True))
waveform_layer.add_layer(BeatAccurateLayer(line_width=1.2))
waveform_layer.load_all_layers(audio_path=audio_file, maps_file=maps_file, beat_file=beat_file)
fig, ax = waveform_layer.draw()
Waveform_Layer=waveform_layer.turn_to_SVG(filename=str(tmp_dir/"Waveform_Layer.svg"), svg_warped_score=svg_score, show_axes=True)
#================================ Onset Layer ===============================
onset_layer = Visualizer()
onset_layer.add_layer(Onsets_Layer(onset_color=(0, 0, 0), line_width=0.5))
onset_layer.load_all_layers(audio_path=audio_file, maps_file=maps_file)
fig, ax = onset_layer.draw()
Onset_Layer=onset_layer.turn_to_SVG(filename=str(tmp_dir / "Onset_Layer.svg"),
                                    svg_warped_score    =svg_score,
                                    print_output        =False)
#================================ BeatProb Layer ===============================
beatProbs_layer = Visualizer()
beatProbs_layer.add_layer(BeatProbabilityLayer(line_width=1.2))
beatProbs_layer.load_all_layers(audio_path=audio_file, beat_file=beat_file)
fig, ax = beatProbs_layer.draw()
beatProb=beatProbs_layer.turn_to_SVG(filename=str(tmp_dir / "beatProbs_Layer.svg"),svg_warped_score  =svg_score)
#================================ BeatProb Layer ===============================
beatProbs_layer = Visualizer()
beatProbs_layer.add_layer(DownbeatProbabilityLayer(line_width=1.2))
beatProbs_layer.load_all_layers(audio_path=audio_file, beat_file=beat_file)
fig, ax = beatProbs_layer.draw()
downbeatProb=beatProbs_layer.turn_to_SVG(filename=str(tmp_dir/"downbeatProbs_Layer.svg"), svg_warped_score  =svg_score)
#================================ Combine Layers ===============================
EndVisualization = Visualizer()
Layer_Width = onset_layer.get_SVG_Root_Dimensions(Waveform_Layer)[0]
Layer_Height = onset_layer.get_SVG_Root_Dimensions(Waveform_Layer)[1]
svg_layers_to_stack = [
    (svg_score,0),
    (Onset_Layer, 0),

    (Spectogram_Layer, Layer_Height-5),
    (Onset_Layer, Layer_Height-5),

    (Chroma_Layer, Layer_Height*2+10),
    (Onset_Layer, Layer_Height*2+10),
    (beatProb, Layer_Height*2+10),
    (downbeatProb, Layer_Height*2+10),

    (Waveform_Layer, Layer_Height*3+15)]
EndVisualization.create_final_SVG(  width              =Layer_Width,
                                    height             =Layer_Height*4+40,
                                    background_color   = "#ffffff",
                                    svg_layers         = svg_layers_to_stack,
                                    output_file        = str(output_dir / "LDB_FIG.svg"),
                                    print_output       = True)