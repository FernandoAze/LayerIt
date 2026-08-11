"""
BeatThis! algorithm visualization layers.
Contains all visualization layers specific to the BeatThis! beat tracking algorithm.
"""

from abc import ABC, abstractmethod
from matplotlib import lines
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Dict, Any, List, Tuple, Optional

''' Import Layer base class '''
from .visualization_system import Layer
"""
Utility class to run BeatThis! beat detection and save results.
Wraps the run_beat_detection function from beat_this_analysis_gen.py.
"""

def Run_BeatThis(audio_path, output_path: str = None, print_output: bool = False) -> str:
    #This functions makes a beat prediction using BeatThis! Algorithm. 
    #It saves the output in a .npz that will be loaded into the beat visualization layers.
    from beat_this.inference import Audio2Frames, Audio2Beats
    from beat_this.preprocessing import load_audio
    from pathlib import Path
    import numpy as np

    if print_output:
        print("\033[92m\n" + "="*60)
        print("RUNNING BEATTHIS!")
        print("="*60 + "\033[0m")
    
    waveform, sample_rate = load_audio(audio_path)
    
    if print_output:
        print(f"✓ Audio loaded. Sample rate: {sample_rate}, Duration: {len(waveform) / sample_rate:.2f}s")
    
    if print_output:
        print("Initializing model (downloading checkpoint if needed)...")
    detector = Audio2Frames(checkpoint_path="final0", device="cpu")
    if print_output:
        print("✓ Model initialized. Processing audio...")

    beat_logits, downbeat_logits = detector(waveform, sample_rate)

    hop_length = 441
    target_sr = 22050
    beat_times = np.arange(len(beat_logits)) * (hop_length / target_sr)
    
    if print_output:
        print("Detecting beat positions...")
    beat_detector = Audio2Beats(checkpoint_path="final0", device="cpu")
    detected_beats, detected_downbeats = beat_detector(waveform, sample_rate)
    
    if print_output:
        print(f"✓ Detected {len(detected_beats)} beats and {len(detected_downbeats)} downbeats")
    
    # Create absolute path for output
    module_dir = Path(__file__).parent
    output_dir = module_dir.parent / "input_files" / "beat_this_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "beat_probs.npz"
    
    if print_output:
        print("✓ Saving output files...")
    if output_path is None:
        output_path = str(output_file)

    np.savez(output_path,
            beat_times=beat_times,
            beat_probs=beat_logits.numpy(),
            downbeat_probs=downbeat_logits.numpy(),
            detected_beats=detected_beats,
            detected_downbeats=detected_downbeats)
    if print_output:
        print(f"✓ File saved: {output_path}")
    return output_path

class BeatLayer(Layer):
    """Base class for BeatThis! algorithm visualization layers.
    
    Manages shared audio/beat data loading and probability visualization.
    All BeatThis! output layers inherit from this class to share common parameters.
    """
    
    def __init__(self, name: str = "BeatThis Layer"):
        super().__init__(name)
    
    def _load_npz_data(self, beat_file: str, required_keys: List[str]) -> Optional[Dict]:
        ''' Load and validate .npz file with required keys '''
        try:
            if beat_file is None:
                return None
            beat_data = np.load(beat_file, allow_pickle=True)
            beat_data_dict = {key: value for key, value in beat_data.items()}
            
            missing = [k for k in required_keys if k not in beat_data_dict]
            if missing:
                print(f"✗ {self.name}: Missing keys {missing}")
                return None
            return beat_data_dict
        except Exception as e:
            print(f"✗ {self.name} error: {e}")
            return None
    
    def _logit_axis_limits(self, logits: np.ndarray) -> Tuple[float, float]:
        '''Return symmetric logit limits that include the zero decision boundary.'''
        max_abs_logit = np.max(np.abs(logits))
        limit = max(1.0, float(np.ceil(max_abs_logit)))
        return -limit, limit

    def _setup_logit_axis(self, ax: Axes, shared_data: Dict[str, Any], logits: np.ndarray) -> Axes:
        '''Get or create the secondary axis used for raw BeatThis logits.'''
        if "ax2" not in shared_data:
            ax2 = ax.twinx()
            shared_data["ax2"] = ax2
        else:
            ax2 = shared_data["ax2"]
        
        ''' Match the primary axis x-limits to avoid extra whitespace '''
        if "times" in shared_data:
            ax2.set_xlim(shared_data["times"][0], shared_data["times"][-1])

        logit_y_min, logit_y_max = self._logit_axis_limits(logits)
        current_y_min, current_y_max = ax2.get_ylim()
        ax2.set_ylim(min(current_y_min, logit_y_min), max(current_y_max, logit_y_max))
        ax2.set_ylabel('Beat activation (logit)', fontweight='bold', fontsize=11)
        return ax2
    
    def _rgb_to_hex(self, rgb):
        '''Convert RGB tuple (0-1) or matplotlib color to hex'''
        if isinstance(rgb, tuple) and len(rgb) >= 3:
            r, g, b = [int(c * 255) if c <= 1 else int(c) for c in rgb[:3]]
            return f'#{r:02x}{g:02x}{b:02x}'
        ''' Handle matplotlib color names '''
        try:
            from matplotlib.colors import to_hex
            return to_hex(rgb)
        except:
            return '#000000'
    
    def _time_to_pixel_x(self, t: float, ctx: Dict) -> float:
        '''Convert time coordinate to pixel X coordinate'''
        if ctx["x_max"] == ctx["x_min"]:
            return 0
        return ((t - ctx["x_min"]) / (ctx["x_max"] - ctx["x_min"])) * ctx["width_px"]
    
    def _logit_to_pixel_y(self, logit: float, ctx: Dict) -> float:
        '''Convert a logit to an inverted SVG Y coordinate.'''
        logit_y_min = ctx["logit_y_min"]
        logit_y_max = ctx["logit_y_max"]
        if logit_y_max == logit_y_min:
            return ctx["height_px"] / 2
        return (1 - (logit - logit_y_min) / (logit_y_max - logit_y_min)) * ctx["height_px"]

    def _logit_axes_to_svg(self, ctx: Dict) -> List[str]:
        '''Build the shared logit and timeline axes for SVG beat layers.'''
        if not ctx.get("show_axes", False) or ctx.get("logit_axes_added", False):
            return []

        width_px = ctx["width_px"]
        height_px = ctx["height_px"]
        x_min = ctx["x_min"]
        x_max = ctx["x_max"]
        logit_y_min = ctx["logit_y_min"]
        logit_y_max = ctx["logit_y_max"]

        if x_max == x_min or logit_y_min is None or logit_y_max is None:
            return []

        ctx["logit_axes_added"] = True
        parts = [
            f'    <line x1="0" y1="0" x2="0" y2="{height_px}" stroke="#111" stroke-width="1"/>',
            f'    <line x1="0" y1="{height_px}" x2="{width_px}" y2="{height_px}" stroke="#111" stroke-width="1"/>',
        ]

        for logit in np.linspace(logit_y_min, logit_y_max, 5):
            y = self._logit_to_pixel_y(logit, ctx)
            parts.append(f'    <line x1="-4" y1="{y:.1f}" x2="0" y2="{y:.1f}" stroke="#111" stroke-width="1"/>')
            parts.append(f'    <text x="-6" y="{y + 3:.1f}" text-anchor="end" font-size="8" font-family="Arial,sans-serif" fill="#111">{logit:.0f}</text>')

        zero_y = self._logit_to_pixel_y(0, ctx)
        parts.append(f'    <line x1="0" y1="{zero_y:.1f}" x2="{width_px}" y2="{zero_y:.1f}" stroke="#999" stroke-width="0.5" stroke-dasharray="4,3"/>')

        t_start = int(np.ceil(x_min))
        t_end = int(np.floor(x_max))
        for time in range(t_start, t_end + 1):
            x = self._time_to_pixel_x(time, ctx)
            if time % 5 == 0:
                parts.append(f'    <line x1="{x:.1f}" y1="{height_px}" x2="{x:.1f}" y2="{height_px + 6}" stroke="#111" stroke-width="1"/>')
                parts.append(f'    <text x="{x:.1f}" y="{height_px + 14:.1f}" text-anchor="middle" font-size="8" font-family="Arial,sans-serif" fill="#111">{time}s</text>')
            else:
                parts.append(f'    <line x1="{x:.1f}" y1="{height_px}" x2="{x:.1f}" y2="{height_px + 4}" stroke="#111" stroke-width="0.7"/>')

        return parts
    
    def _probability_to_svg_group(self, shared_data: Dict[str, Any], prob_key: str, svg_class: str, opacity: float = 1.0, line_width: float = 0.5) -> Optional[str]:
        '''
        Generic method to convert probability curve to SVG polyline.
        
        Args:
            shared_data: Shared visualization data
            prob_key: Key for probability data in self._data (e.g., 'beat_probs', 'downbeat_probs')
            svg_class: CSS class for the SVG group (e.g., 'beat-probability')
            opacity: Optional opacity for the polyline (default 1.0)
        '''
        if self._data is None or "svg_context" not in shared_data:
            return None
        
        ctx = shared_data["svg_context"]
        beat_times = self._data.get("beat_times", [])
        logits = self._data.get(prob_key, [])
        
        if len(beat_times) == 0:
            return None
        
        ''' Convert data points to SVG coordinates '''
        points = []
        for t, logit in zip(beat_times, logits):
            x = self._time_to_pixel_x(t, ctx)
            y = self._logit_to_pixel_y(logit, ctx)
            points.append(f"{x:.2f},{y:.2f}")
        
        points_str = " ".join(points)
        color_hex = self._rgb_to_hex(self.color)
        opacity_attr = f' opacity="{opacity}"' if opacity < 1.0 else ''

        parts = [f'  <g id="{self.name}" class="layer {svg_class}">']
        parts.extend(self._logit_axes_to_svg(ctx))
        parts.append(f'    <polyline points="{points_str}" stroke="{color_hex}" stroke-width="{line_width}" fill="none"{opacity_attr}/>')
        parts.append('  </g>')
        return '\n'.join(parts)

class BeatProbabilityLayer(BeatLayer):
    """Visualizes raw beat logits from the BeatThis! algorithm."""
    
    def __init__(self, name: str = "Beat Probability", color='r', line_width: float = 0.5):
        super().__init__(name)
        self.color = color
        self.line_width = line_width
    
    def load_data(self, beat_file: str = None, print_output: bool = False, **kwargs) -> bool:
        data = self._load_npz_data(beat_file, ['beat_times', 'beat_probs'])
        if data is None:
            return False
        self._data = data
        if print_output==True:
            print(f"✓ {self.name}: Loaded beat data")
        return True
    
    def draw(self, ax: Axes, shared_data: Dict[str, Any]) -> Tuple[List, List]:
        if self._data is None:
            print(f"✗ {self.name}: No data loaded")
            return [], []
        
        ax2 = self._setup_logit_axis(ax, shared_data, self._data["beat_probs"])
        
        line, = ax2.plot(self._data["beat_times"], self._data["beat_probs"], '-', 
                color=self.color, linewidth=self.line_width, label='Beat Logit')
        return [line], ['Beat Logit']
    
    def to_svg_group(self, shared_data: Dict[str, Any]) -> Optional[str]:
        '''Convert beat probability curve to SVG polyline'''
        return self._probability_to_svg_group(shared_data, 'beat_probs', 'beat-probability', line_width=self.line_width)


class DownbeatProbabilityLayer(BeatLayer):
    """Visualizes raw downbeat logits from the BeatThis! algorithm."""
    
    def __init__(self, name: str = "Downbeat Probability", color='blue', line_width: float = 0.5):
        super().__init__(name)
        self.color = color
        self.line_width = line_width
    
    def load_data(self, beat_file: str = None, print_output: bool = False, **kwargs) -> bool:
        data = self._load_npz_data(beat_file, ['beat_times', 'downbeat_probs'])
        if data is None:
            return False
        self._data = data
        if print_output==True:    
            print(f"✓ {self.name}: Loaded downbeat data")
        return True
    
    def draw(self, ax: Axes, shared_data: Dict[str, Any]) -> Tuple[List, List]:
        if self._data is None:
            print(f"✗ {self.name}: No data loaded")
            return [], []
        
        ax2 = self._setup_logit_axis(ax, shared_data, self._data["downbeat_probs"])
        
        line, = ax2.plot(self._data["beat_times"], self._data["downbeat_probs"], '-',
                color=self.color, linewidth=self.line_width, label='Downbeat Logit', alpha=0.9)
        return [line], ['Downbeat Logit']
    
    def to_svg_group(self, shared_data: Dict[str, Any]) -> Optional[str]:
        '''Convert downbeat probability curve to SVG polyline'''
        return self._probability_to_svg_group(shared_data, 'downbeat_probs', 'downbeat-probability', opacity=0.9, line_width=self.line_width)

class BeatAccurateLayer(BeatLayer):
    """Visualizes detected beat times as vertical lines."""
    
    def __init__(self, name: str = "Beat Accurate", beat_color='red', downbeat_color='blue', line_width: float = 1):
        super().__init__(name)
        self.beat_color = beat_color
        self.downbeat_color = downbeat_color
        self.line_width = line_width
    
    def load_data(self, beat_file: str = None, print_output: bool = False, **kwargs) -> bool:
        data = self._load_npz_data(beat_file, ['detected_beats', 'detected_downbeats'])
        if data is None:
            return False
        self._data = data
        if print_output==True:  
            print(f"✓ {self.name}: Loaded {len(self._data['detected_beats'])} beats, {len(self._data['detected_downbeats'])} downbeats")
        return True
    
    def draw(self, ax: Axes, shared_data: Dict[str, Any]) -> Tuple[List, List]:
        if self._data is None:
            print(f"✗ {self.name}: No data loaded")
            return [], []
        
        ax2 = self._setup_logit_axis(ax, shared_data, np.array([-1.0, 1.0]))
        downbeat_set = set(np.round(self._data["detected_downbeats"], 6))
        
        ''' Draw regular beats (exclude downbeats) '''
        beat_lines = [ax2.axvline(x=t, color=self.beat_color, linewidth=self.line_width) 
                     for t in self._data["detected_beats"]
                     if round(t, 6) not in downbeat_set]
        
        ''' Draw downbeats '''
        downbeat_lines = [ax2.axvline(x=t, color=self.downbeat_color, linewidth=self.line_width) 
                         for t in self._data["detected_downbeats"]]
        
        labels = []
        if beat_lines:
            labels.append('Beat')
        if downbeat_lines:
            labels.append('Downbeat')
        
        return beat_lines + downbeat_lines, labels
    
    def to_svg_group(self, shared_data: Dict[str, Any]) -> Optional[str]:
        '''Convert detected beats/downbeats to SVG vertical lines'''
        if self._data is None or "svg_context" not in shared_data:
            return None
        
        ctx = shared_data["svg_context"]
        beat_times = self._data.get("detected_beats", [])
        downbeat_times = self._data.get("detected_downbeats", [])
        
        if len(beat_times) == 0 and len(downbeat_times) == 0:
            return None
        
        lines = []
        downbeat_set = set(np.round(downbeat_times, 6))
        
        ''' Draw regular beats '''
        beat_color_hex = self._rgb_to_hex(self.beat_color)
        for t in beat_times:
            if round(t, 6) not in downbeat_set:
                x = self._time_to_pixel_x(t, ctx)
                lines.append(f'    <line x1="{x:.2f}" y1="0" x2="{x:.2f}" y2="{ctx["height_px"]}" stroke="{beat_color_hex}" stroke-width="{self.line_width}"/>')
        
        ''' Draw downbeats '''
        downbeat_color_hex = self._rgb_to_hex(self.downbeat_color)
        for t in downbeat_times:
            x = self._time_to_pixel_x(t, ctx)
            lines.append(f'    <line x1="{x:.2f}" y1="0" x2="{x:.2f}" y2="{ctx["height_px"]}" stroke="{downbeat_color_hex}" stroke-width="{self.line_width}"/>')
        
        svg_group = f'''  <g id="{self.name}" class="layer beat-accurate">
{chr(10).join(lines)}
  </g>'''
        
        return svg_group


class BeatWindowLayer(BeatLayer):
    """Visualizes beat confidence windows with gradient transparency.
    
    Shows regions where beat probability exceeds a threshold, with transparency 
    gradient: opaque at peak confidence, transparent at threshold boundaries.
    """
    
    def __init__(self, name: str = "Beat Window", beat_window: float = 70, color='red', alpha_max: float = 0.3):
        """
        Args:
            name: Layer name
            beat_window: Probability threshold (0-100 or 0-1) converted to a logit threshold
            color: Rectangle fill color
            alpha_max: Maximum opacity at peak (0-1)
        """
        super().__init__(name)
        self.beat_window = self._normalize_threshold(beat_window)
        self.logit_threshold = self._probability_threshold_to_logit(self.beat_window)
        self.color = color
        self.alpha_max = alpha_max
    
    def _normalize_threshold(self, threshold: float) -> float:
        '''Convert threshold to 0-100 scale if needed'''
        return threshold * 100 if threshold <= 1.0 else threshold

    def _probability_threshold_to_logit(self, threshold: float) -> float:
        '''Convert a percentage threshold to the equivalent raw logit threshold.'''
        probability = np.clip(threshold / 100, np.finfo(float).eps, 1 - np.finfo(float).eps)
        return float(np.log(probability / (1 - probability)))
    
    def _find_windows(self, probs: np.ndarray) -> List[Tuple[int, int, float]]:
        '''
        Find contiguous regions where probability exceeds threshold.
        Returns list of (start_idx, end_idx, peak_prob) tuples.
        '''
        above_threshold = probs >= self.logit_threshold
        
        windows = []
        in_window = False
        window_start = 0
        window_probs = []
        
        for i, is_above in enumerate(above_threshold):
            if is_above:
                if not in_window:
                    window_start = i
                    in_window = True
                window_probs.append(probs[i])
            else:
                if in_window:
                    peak_prob = np.max(window_probs)
                    windows.append((window_start, i - 1, peak_prob))
                    in_window = False
                    window_probs = []
        
        ''' Handle case where window extends to end of data '''
        if in_window:
            peak_prob = np.max(window_probs)
            windows.append((window_start, len(probs) - 1, peak_prob))
        
        return windows
    
    def _calculate_opacity(self, idx: int, start_idx: int, end_idx: int, peak_prob: float, 
                          probs: np.ndarray) -> float:
        '''
        Calculate opacity for a point in the window.
        Opacity = 1.0 at peak, 0.0 at threshold boundaries.
        '''
        current_logit = probs[idx]
        
        ''' Distance from threshold (normalized 0-1, where 1 = at peak) '''
        if peak_prob == self.logit_threshold:
            return self.alpha_max
        distance_from_threshold = (current_logit - self.logit_threshold) / (peak_prob - self.logit_threshold)
        distance_from_threshold = np.clip(distance_from_threshold, 0, 1)
        
        return distance_from_threshold * self.alpha_max
    
    def load_data(self, beat_file: str = None, print_output: bool = False, **kwargs) -> bool:
        data = self._load_npz_data(beat_file, ['beat_times', 'beat_probs'])
        if data is None:
            return False
        self._data = data
        windows = self._find_windows(self._data['beat_probs'])
        if print_output==True:    
            print(f"✓ {self.name}: Loaded beat data with threshold {self.beat_window:.1f}%, found {len(windows)} windows")
        return True
    
    def draw(self, ax: Axes, shared_data: Dict[str, Any]) -> Tuple[List, List]:
        if self._data is None:
            print(f"✗ {self.name}: No data loaded")
            return [], []
        
        ax2 = self._setup_logit_axis(ax, shared_data, self._data['beat_probs'])
        beat_times = self._data['beat_times']
        beat_probs = self._data['beat_probs']
        windows = self._find_windows(beat_probs)
        
        rectangles = []
        for start_idx, end_idx, peak_prob in windows:
            t_start = beat_times[start_idx]
            t_end = beat_times[end_idx]
            
            ''' Draw with full opacity in matplotlib '''
            logit_y_min, logit_y_max = ax2.get_ylim()
            rect = plt.Rectangle((t_start, logit_y_min), t_end - t_start, logit_y_max - logit_y_min,
                                alpha=1.0, color=self.color, label='Beat Window')
            ax2.add_patch(rect)
            rectangles.append(rect)
        
        return rectangles, ['Beat Window'] if rectangles else []
    
    def to_svg_group(self, shared_data: Dict[str, Any]) -> Optional[str]:
        '''Convert beat windows to SVG rectangles with opacity gradient'''
        if self._data is None or "svg_context" not in shared_data:
            return None
        
        ctx = shared_data["svg_context"]
        beat_times = self._data.get("beat_times", [])
        beat_probs = self._data.get("beat_probs", [])
        
        if len(beat_times) == 0:
            return None
        
        windows = self._find_windows(beat_probs)
        if len(windows) == 0:
            return None
        
        gradients = []
        rectangles = []
        color_hex = self._rgb_to_hex(self.color)
        
        for idx, (start_idx, end_idx, peak_prob) in enumerate(windows):
            t_start = beat_times[start_idx]
            t_end = beat_times[end_idx]
            
            x1 = self._time_to_pixel_x(t_start, ctx)
            x2 = self._time_to_pixel_x(t_end, ctx)
            
            ''' Create unique gradient ID for this window '''
            gradient_id = f"{self.name.replace(' ', '_')}_gradient_{idx}"
            
            ''' Define linear gradient: transparent at edges, opaque at peak '''
            gradient_svg = f'''    <linearGradient id="{gradient_id}" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:{color_hex};stop-opacity:0"/>
      <stop offset="50%" style="stop-color:{color_hex};stop-opacity:{self.alpha_max:.2f}"/>
      <stop offset="100%" style="stop-color:{color_hex};stop-opacity:0"/>
    </linearGradient>'''
            gradients.append(gradient_svg)
            
            ''' Draw rectangle using gradient '''
            rect_svg = f'    <rect x="{x1:.2f}" y="0" width="{x2-x1:.2f}" height="{ctx["height_px"]}" fill="url(#{gradient_id})"/>'
            rectangles.append(rect_svg)
        
        svg_group = f'''  <g id="{self.name}" class="layer beat-window">
    <defs>
{chr(10).join(gradients)}
    </defs>
{chr(10).join(rectangles)}
  </g>'''
        
        return svg_group


class DownbeatWindowLayer(BeatLayer):
    """Visualizes downbeat confidence windows with gradient transparency.
    
    Shows regions where downbeat probability exceeds a threshold, with transparency 
    gradient: opaque at peak confidence, transparent at threshold boundaries.
    """
    
    def __init__(self, name: str = "Downbeat Window", beat_window: float = 70, color='blue', alpha_max: float = 0.3):
        """
        Args:
            name: Layer name
            beat_window: Probability threshold (0-100 or 0-1) converted to a logit threshold
            color: Rectangle fill color
            alpha_max: Maximum opacity at peak (0-1)
        """
        super().__init__(name)
        self.beat_window = self._normalize_threshold(beat_window)
        self.logit_threshold = self._probability_threshold_to_logit(self.beat_window)
        self.color = color
        self.alpha_max = alpha_max
    
    def _normalize_threshold(self, threshold: float) -> float:
        '''Convert threshold to 0-100 scale if needed'''
        return threshold * 100 if threshold <= 1.0 else threshold

    def _probability_threshold_to_logit(self, threshold: float) -> float:
        '''Convert a percentage threshold to the equivalent raw logit threshold.'''
        probability = np.clip(threshold / 100, np.finfo(float).eps, 1 - np.finfo(float).eps)
        return float(np.log(probability / (1 - probability)))
    
    def _find_windows(self, probs: np.ndarray) -> List[Tuple[int, int, float]]:
        '''
        Find contiguous regions where probability exceeds threshold.
        Returns list of (start_idx, end_idx, peak_prob) tuples.
        '''
        above_threshold = probs >= self.logit_threshold
        
        windows = []
        in_window = False
        window_start = 0
        window_probs = []
        
        for i, is_above in enumerate(above_threshold):
            if is_above:
                if not in_window:
                    window_start = i
                    in_window = True
                window_probs.append(probs[i])
            else:
                if in_window:
                    peak_prob = np.max(window_probs)
                    windows.append((window_start, i - 1, peak_prob))
                    in_window = False
                    window_probs = []
        
        ''' Handle case where window extends to end of data '''
        if in_window:
            peak_prob = np.max(window_probs)
            windows.append((window_start, len(probs) - 1, peak_prob))
        
        return windows
    
    def _calculate_opacity(self, idx: int, start_idx: int, end_idx: int, peak_prob: float, 
                          probs: np.ndarray) -> float:
        '''
        Calculate opacity for a point in the window.
        Opacity = 1.0 at peak, 0.0 at threshold boundaries.
        '''
        current_logit = probs[idx]
        
        ''' Distance from threshold (normalized 0-1, where 1 = at peak) '''
        if peak_prob == self.logit_threshold:
            return self.alpha_max
        distance_from_threshold = (current_logit - self.logit_threshold) / (peak_prob - self.logit_threshold)
        distance_from_threshold = np.clip(distance_from_threshold, 0, 1)
        
        return distance_from_threshold * self.alpha_max
    
    def load_data(self, beat_file: str = None, print_output: bool = False, **kwargs) -> bool:
        data = self._load_npz_data(beat_file, ['beat_times', 'downbeat_probs'])
        if data is None:
            return False
        self._data = data
        windows = self._find_windows(self._data['downbeat_probs'])
        if print_output:
            print(f"✓ {self.name}: Loaded downbeat data with threshold {self.beat_window:.1f}%, found {len(windows)} windows")
        return True
    
    def draw(self, ax: Axes, shared_data: Dict[str, Any]) -> Tuple[List, List]:
        if self._data is None:
            print(f"✗ {self.name}: No data loaded")
            return [], []
        
        ax2 = self._setup_logit_axis(ax, shared_data, self._data['downbeat_probs'])
        beat_times = self._data['beat_times']
        downbeat_probs = self._data['downbeat_probs']
        windows = self._find_windows(downbeat_probs)
        
        rectangles = []
        for start_idx, end_idx, peak_prob in windows:
            t_start = beat_times[start_idx]
            t_end = beat_times[end_idx]
            
            ''' Draw with full opacity in matplotlib '''
            logit_y_min, logit_y_max = ax2.get_ylim()
            rect = plt.Rectangle((t_start, logit_y_min), t_end - t_start, logit_y_max - logit_y_min,
                                alpha=1.0, color=self.color, label='Downbeat Window')
            ax2.add_patch(rect)
            rectangles.append(rect)
        
        return rectangles, ['Downbeat Window'] if rectangles else []
    
    def to_svg_group(self, shared_data: Dict[str, Any]) -> Optional[str]:
        '''Convert downbeat windows to SVG rectangles with opacity gradient'''
        if self._data is None or "svg_context" not in shared_data:
            return None
        
        ctx = shared_data["svg_context"]
        beat_times = self._data.get("beat_times", [])
        downbeat_probs = self._data.get("downbeat_probs", [])
        
        if len(beat_times) == 0:
            return None
        
        windows = self._find_windows(downbeat_probs)
        if len(windows) == 0:
            return None
        
        gradients = []
        rectangles = []
        color_hex = self._rgb_to_hex(self.color)
        
        for idx, (start_idx, end_idx, peak_prob) in enumerate(windows):
            t_start = beat_times[start_idx]
            t_end = beat_times[end_idx]
            
            x1 = self._time_to_pixel_x(t_start, ctx)
            x2 = self._time_to_pixel_x(t_end, ctx)
            
            ''' Create unique gradient ID for this window '''
            gradient_id = f"{self.name.replace(' ', '_')}_gradient_{idx}"
            
            ''' Define linear gradient: transparent at edges, opaque at peak '''
            gradient_svg = f'''    <linearGradient id="{gradient_id}" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:{color_hex};stop-opacity:0"/>
      <stop offset="50%" style="stop-color:{color_hex};stop-opacity:{self.alpha_max:.2f}"/>
      <stop offset="100%" style="stop-color:{color_hex};stop-opacity:0"/>
    </linearGradient>'''
            gradients.append(gradient_svg)
            
            ''' Draw rectangle using gradient '''
            rect_svg = f'    <rect x="{x1:.2f}" y="0" width="{x2-x1:.2f}" height="{ctx["height_px"]}" fill="url(#{gradient_id})"/>'
            rectangles.append(rect_svg)
        
        svg_group = f'''  <g id="{self.name}" class="layer downbeat-window">
    <defs>
{chr(10).join(gradients)}
    </defs>
{chr(10).join(rectangles)}
  </g>'''
        
        return svg_group

