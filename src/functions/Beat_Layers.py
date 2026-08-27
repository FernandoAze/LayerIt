"""
Beat visualization layers: logit curves, detected event markers, and confidence windows.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Dict, Any, List, Tuple, Optional

''' Import Layer base class and shape primitives '''
from .visualization_system import Layer
from .shapes import Curve, Events, Intervals

def _load_beat_npz(beat_file: str, required_keys: List[str], layer_name: str) -> Optional[Dict]:
    ''' Load and validate a beat .npz file with the required keys '''
    try:
        if beat_file is None:
            return None
        beat_data = np.load(beat_file, allow_pickle=True)
        beat_data_dict = {key: value for key, value in beat_data.items()}

        missing = [k for k in required_keys if k not in beat_data_dict]
        if missing:
            print(f"✗ {layer_name}: Missing keys {missing}")
            return None
        return beat_data_dict
    except Exception as e:
        print(f"✗ {layer_name} error: {e}")
        return None


def NPZ_to_BeatTXT(beat_file: str, output_file: str = None, print_output: bool = False) -> Optional[str]:
    '''
    Extract detected beats (excluding downbeats) from a beat .npz and
    write a tab-separated TXT file with beat index and time.

    Output format:
        1    0.523100
        2    0.981400
        ...
    '''
    from pathlib import Path
    try:
        data = _load_beat_npz(beat_file, ['detected_beats', 'detected_downbeats'], 'NPZ_to_BeatTXT')
        if data is None:
            return None

        downbeat_set = set(np.round(data['detected_downbeats'], 6))
        beats = sorted(t for t in data['detected_beats'] if round(t, 6) not in downbeat_set)

        if output_file is None:
            output_file = str(Path(beat_file).with_suffix('')) + '_beats.txt'

        with open(output_file, 'w') as f:
            for idx, t in enumerate(beats, start=1):
                f.write(f"{idx}\t{t:.6f}\n")

        if print_output:
            print(f"✓ NPZ_to_BeatTXT: {len(beats)} beats written to {output_file}")
        return output_file

    except Exception as e:
        print(f"✗ NPZ_to_BeatTXT error: {e}")
        return None


def NPZ_to_DownbeatTXT(beat_file: str, output_file: str = None, print_output: bool = False) -> Optional[str]:
    '''
    Extract detected downbeats from a beat .npz and write a
    tab-separated TXT file with downbeat index and time.

    Output format:
        1    0.523100
        2    1.899800
        ...
    '''
    from pathlib import Path
    try:
        data = _load_beat_npz(beat_file, ['detected_downbeats'], 'NPZ_to_DownbeatTXT')
        if data is None:
            return None

        downbeats = sorted(data['detected_downbeats'])

        if output_file is None:
            output_file = str(Path(beat_file).with_suffix('')) + '_downbeats.txt'

        with open(output_file, 'w') as f:
            for idx, t in enumerate(downbeats, start=1):
                f.write(f"{idx}\t{t:.6f}\n")

        if print_output:
            print(f"✓ NPZ_to_DownbeatTXT: {len(downbeats)} downbeats written to {output_file}")
        return output_file

    except Exception as e:
        print(f"✗ NPZ_to_DownbeatTXT error: {e}")
        return None


class BeatLogits(Curve):
    """Visualizes raw beat logits as a line overlay."""

    def __init__(self, name: str = "Beat Probability", color='r', line_width: float = 0.5, line_type: str = "solid"):
        super().__init__(name, color=color, line_width=line_width, label='Beat Logit',
                          secondary_axis=True, axis_label='Beat activation (logit)',
                          svg_class='beat-probability', line_type=line_type)

    def load_data(self, beat_file: str = None, print_output: bool = False, **kwargs) -> bool:
        data = _load_beat_npz(beat_file, ['beat_times', 'beat_activation'], self.name)
        if data is None:
            return False
        self._data = data
        if print_output==True:
            print(f"✓ {self.name}: Loaded beat data")
        return True

    def _get_xy(self, shared_data: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._data is None:
            return None
        return self._data["beat_times"], self._data["beat_activation"]


class DownbeatLogits(Curve):
    """Visualizes raw downbeat logits as a line overlay."""

    def __init__(self, name: str = "Downbeat Probability", color='blue', line_width: float = 0.5, line_type: str = "solid"):
        super().__init__(name, color=color, line_width=line_width, alpha=0.9, label='Downbeat Logit',
                          secondary_axis=True, axis_label='Beat activation (logit)',
                          svg_class='downbeat-probability', line_type=line_type)

    def load_data(self, beat_file: str = None, print_output: bool = False, **kwargs) -> bool:
        data = _load_beat_npz(beat_file, ['beat_times', 'downbeat_activation'], self.name)
        if data is None:
            return False
        self._data = data
        if print_output==True:    
            print(f"✓ {self.name}: Loaded downbeat data")
        return True

    def _get_xy(self, shared_data: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._data is None:
            return None
        return self._data["beat_times"], self._data["downbeat_activation"]


class BeatsLayer(Events):
    """Visualizes detected beat times (excluding downbeats) as vertical markers."""

    def __init__(self, name: str = "Beat", color='red', line_width: float = 1, line_type: str = "solid"):
        super().__init__(name, color=color, line_width=line_width, line_type=line_type, secondary_axis=True, svg_class='beat-marker')

    def load_data(self, beat_file: str = None, print_output: bool = False, **kwargs) -> bool:
        data = _load_beat_npz(beat_file, ['detected_beats', 'detected_downbeats'], self.name)
        if data is None:
            return False
        self._data = data
        if print_output==True:
            print(f"✓ {self.name}: Loaded {len(self._data['detected_beats'])} beats")
        return True

    def _get_times(self, shared_data: Dict[str, Any]) -> Optional[np.ndarray]:
        if self._data is None:
            return None
        downbeat_set = set(np.round(self._data["detected_downbeats"], 6))
        return np.array([t for t in self._data["detected_beats"] if round(t, 6) not in downbeat_set])


class DownbeatsLayer(Events):
    """Visualizes detected downbeat times as vertical markers."""

    def __init__(self, name: str = "Downbeat", color='blue', line_width: float = 1, line_type: str = "solid"):
        super().__init__(name, color=color, line_width=line_width, line_type=line_type, secondary_axis=True, svg_class='downbeat-marker')

    def load_data(self, beat_file: str = None, print_output: bool = False, **kwargs) -> bool:
        data = _load_beat_npz(beat_file, ['detected_beats', 'detected_downbeats'], self.name)
        if data is None:
            return False
        self._data = data
        if print_output==True:
            print(f"✓ {self.name}: Loaded {len(self._data['detected_downbeats'])} downbeats")
        return True

    def _get_times(self, shared_data: Dict[str, Any]) -> Optional[np.ndarray]:
        if self._data is None:
            return None
        return np.array(self._data["detected_downbeats"])


class BeatAccurateLayer(Layer):
    """Legacy combined view of detected beats and downbeats as vertical lines.

    Kept for backward compatibility; new code can use BeatsLayer and
    DownbeatsLayer directly.
    """

    def __init__(self, name: str = "Beat Accurate", beat_color='red', downbeat_color='blue', line_width: float = 1):
        super().__init__(name)
        self._beats = BeatsLayer(name="Beat", color=beat_color, line_width=line_width)
        self._downbeats = DownbeatsLayer(name="Downbeat", color=downbeat_color, line_width=line_width)

    def load_data(self, beat_file: str = None, print_output: bool = False, **kwargs) -> bool:
        loaded_beats = self._beats.load_data(beat_file=beat_file, print_output=print_output, **kwargs)
        loaded_downbeats = self._downbeats.load_data(beat_file=beat_file, print_output=print_output, **kwargs)
        return loaded_beats and loaded_downbeats

    def draw(self, ax: Axes, shared_data: Dict[str, Any]) -> Tuple[List, List]:
        beat_lines, beat_labels = self._beats.draw(ax, shared_data)
        downbeat_lines, downbeat_labels = self._downbeats.draw(ax, shared_data)
        return beat_lines + downbeat_lines, beat_labels + downbeat_labels

    def to_svg_group(self, shared_data: Dict[str, Any]) -> Optional[str]:
        lines = self._beats._lines_svg(shared_data) + self._downbeats._lines_svg(shared_data)
        if not lines:
            return None
        svg_group = f'''  <g id="{self.name}" class="layer beat-accurate">
{chr(10).join(lines)}
  </g>'''
        return svg_group


class BeatWindowLayer(Intervals):
    """Visualizes beat confidence windows with gradient transparency.

    Shows regions where beat probability exceeds a threshold, with transparency
    gradient: opaque at peak confidence, transparent at threshold boundaries.
    """

    def __init__(self, name: str = "Beat Window", beat_window: float = 70, color='red', alpha_max: float = 0.3):
        super().__init__(name, color=color, threshold=beat_window, alpha_max=alpha_max, svg_class="beat-window")

    def load_data(self, beat_file: str = None, print_output: bool = False, **kwargs) -> bool:
        data = _load_beat_npz(beat_file, ['beat_times', 'beat_activation'], self.name)
        if data is None:
            return False
        self._data = data
        if print_output==True:
            windows = self._find_windows(self._data['beat_activation'])
            print(f"✓ {self.name}: Loaded beat data with threshold {self.threshold:.1f}%, found {len(windows)} windows")
        return True

    def _get_activation(self, shared_data: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._data is None:
            return None
        return self._data['beat_times'], self._data['beat_activation']


class DownbeatWindowLayer(Intervals):
    """Visualizes downbeat confidence windows with gradient transparency.

    Shows regions where downbeat probability exceeds a threshold, with transparency
    gradient: opaque at peak confidence, transparent at threshold boundaries.
    """

    def __init__(self, name: str = "Downbeat Window", beat_window: float = 70, color='blue', alpha_max: float = 0.3):
        super().__init__(name, color=color, threshold=beat_window, alpha_max=alpha_max, svg_class="downbeat-window")

    def load_data(self, beat_file: str = None, print_output: bool = False, **kwargs) -> bool:
        data = _load_beat_npz(beat_file, ['beat_times', 'downbeat_activation'], self.name)
        if data is None:
            return False
        self._data = data
        if print_output:
            windows = self._find_windows(self._data['downbeat_activation'])
            print(f"✓ {self.name}: Loaded downbeat data with threshold {self.threshold:.1f}%, found {len(windows)} windows")
        return True

    def _get_activation(self, shared_data: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._data is None:
            return None
        return self._data['beat_times'], self._data['downbeat_activation']
