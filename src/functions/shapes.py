"""
Shape-based rendering primitives shared across LayerIt layers.

Each shape owns one draw() and one to_svg_group() implementation. Concrete
layers subclass a shape and supply only the data (via a small hook method)
and style (constructor kwargs), instead of re-implementing rendering.
"""

from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Dict, Any, List, Tuple, Optional
import base64
import io
from PIL import Image as PILImage

from .visualization_system import Layer


def rgb_to_hex(color) -> str:
    ''' Convert an RGB tuple (0-1 or 0-255 scale) or matplotlib color to hex '''
    if isinstance(color, tuple) and len(color) >= 3:
        r, g, b = [int(c * 255) if c <= 1 else int(c) for c in color[:3]]
        return f'#{r:02x}{g:02x}{b:02x}'
    try:
        from matplotlib.colors import to_hex
        return to_hex(color)
    except Exception:
        return '#000000'


def time_to_pixel_x(t: float, ctx: Dict) -> float:
    ''' Convert a time coordinate to pixel X using the shared SVG context '''
    if ctx["x_max"] == ctx["x_min"]:
        return 0.0
    return ((t - ctx["x_min"]) / (ctx["x_max"] - ctx["x_min"])) * ctx["width_px"]


def value_to_pixel_y(value: float, y_min: float, y_max: float, height_px: float) -> float:
    ''' Convert a value to an inverted (SVG-space) pixel Y coordinate '''
    if y_max == y_min:
        return height_px / 2
    return (1 - (value - y_min) / (y_max - y_min)) * height_px


class Curve(Layer):
    """A single 2D line: a raw signal (e.g. a waveform) or a per-frame
    activation curve (e.g. a beat logit), drawn on the panel's primary
    axis or on a shared secondary axis.
    """

    def __init__(self, name: str, color, line_width: float = 0.5,
                 alpha: float = 1.0, label: Optional[str] = None,
                 secondary_axis: bool = False, axis_label: Optional[str] = None,
                 decimate_per_pixel: Optional[int] = None, svg_class: str = "curve",
                 line_type: str = "solid"):
        super().__init__(name)
        self.color = color
        self.line_width = line_width
        self.alpha = alpha
        self.label = label or name
        self.secondary_axis = secondary_axis
        self.axis_label = axis_label
        self.decimate_per_pixel = decimate_per_pixel
        self.svg_class = svg_class
        self.line_type = line_type

    @abstractmethod
    def _get_xy(self, shared_data: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        ''' Return (x, y) sample arrays for the curve, or None if not loaded '''
        pass

    def _secondary_y_limits(self, y: np.ndarray) -> Tuple[float, float]:
        ''' Symmetric y-limits used only when secondary_axis is True '''
        limit = max(1.0, float(np.ceil(np.max(np.abs(y)))))
        return -limit, limit

    def _setup_secondary_axis(self, ax: Axes, shared_data: Dict[str, Any], y: np.ndarray) -> Axes:
        if "ax2" not in shared_data:
            ax2 = ax.twinx()
            shared_data["ax2"] = ax2
        else:
            ax2 = shared_data["ax2"]

        ''' Match the primary axis x-limits to avoid extra whitespace '''
        if "times" in shared_data:
            ax2.set_xlim(shared_data["times"][0], shared_data["times"][-1])

        y_min, y_max = self._secondary_y_limits(y)
        cur_min, cur_max = ax2.get_ylim()
        ax2.set_ylim(min(cur_min, y_min), max(cur_max, y_max))
        if self.axis_label:
            ax2.set_ylabel(self.axis_label, fontweight='bold', fontsize=11)
        return ax2

    def draw(self, ax: Axes, shared_data: Dict[str, Any]) -> Tuple[List, List]:
        xy = self._get_xy(shared_data)
        if xy is None:
            print(f"✗ {self.name}: No data loaded")
            return [], []
        x, y = xy

        target_ax = self._setup_secondary_axis(ax, shared_data, y) if self.secondary_axis else ax
        _mpl_styles = {"solid": "-", "dashed": "--", "dotted": ":", "dashdotted": "-."}
        style = _mpl_styles.get(self.line_type, "-")
        line, = target_ax.plot(x, y, style, color=self.color, linewidth=self.line_width,
                                alpha=self.alpha, label=self.label)
        return [line], [self.label]

    def _axis_svg(self, ctx: Dict, y_min: float, y_max: float, tick_format: str, flag_key: str) -> List[str]:
        ''' Shared axis (ticks + timeline) markup, drawn once per SVG context '''
        if not ctx.get("show_axes", False) or ctx.get(flag_key, False):
            return []

        width_px, height_px = ctx["width_px"], ctx["height_px"]
        x_min, x_max = ctx["x_min"], ctx["x_max"]
        if x_max == x_min or y_min is None or y_max is None or y_min == y_max:
            return []

        ctx[flag_key] = True
        parts = [
            f'    <line x1="0" y1="0" x2="0" y2="{height_px}" stroke="#111" stroke-width="1"/>',
            f'    <line x1="0" y1="{height_px}" x2="{width_px}" y2="{height_px}" stroke="#111" stroke-width="1"/>',
        ]

        for val in np.linspace(y_min, y_max, 5):
            y_s = value_to_pixel_y(val, y_min, y_max, height_px)
            parts.append(f'    <line x1="-4" y1="{y_s:.1f}" x2="0" y2="{y_s:.1f}" stroke="#111" stroke-width="1"/>')
            parts.append(f'    <text x="-6" y="{y_s + 3:.1f}" text-anchor="end" font-size="8" font-family="Arial,sans-serif" fill="#111">{tick_format.format(val)}</text>')

        if y_min < 0 < y_max:
            zero_y = value_to_pixel_y(0, y_min, y_max, height_px)
            parts.append(f'    <line x1="0" y1="{zero_y:.1f}" x2="{width_px}" y2="{zero_y:.1f}" stroke="#999" stroke-width="0.5" stroke-dasharray="4,3"/>')

        t_start = int(np.ceil(x_min))
        t_end = int(np.floor(x_max))
        for t in range(t_start, t_end + 1):
            x_s = time_to_pixel_x(t, ctx)
            if t % 5 == 0:
                parts.append(f'    <line x1="{x_s:.1f}" y1="{height_px}" x2="{x_s:.1f}" y2="{height_px + 6}" stroke="#111" stroke-width="1"/>')
                parts.append(f'    <text x="{x_s:.1f}" y="{height_px + 14:.1f}" text-anchor="middle" font-size="8" font-family="Arial,sans-serif" fill="#111">{t}s</text>')
            else:
                parts.append(f'    <line x1="{x_s:.1f}" y1="{height_px}" x2="{x_s:.1f}" y2="{height_px + 4}" stroke="#111" stroke-width="0.7"/>')

        return parts

    def to_svg_group(self, shared_data: Dict[str, Any]) -> Optional[str]:
        if "svg_context" not in shared_data:
            return None
        xy = self._get_xy(shared_data)
        if xy is None:
            return None
        x, y = xy

        ctx = shared_data["svg_context"]
        width_px, height_px = ctx["width_px"], ctx["height_px"]

        if self.secondary_axis:
            y_min, y_max = ctx["logit_y_min"], ctx["logit_y_max"]
            axis_flag, tick_format = "logit_axes_added", "{:.0f}"
        else:
            y_min, y_max = ctx["y_min"], ctx["y_max"]
            axis_flag, tick_format = "primary_axes_added", "{:.1f}"

        if ctx["x_max"] == ctx["x_min"] or y_min is None or y_max is None or y_min == y_max:
            return None

        if self.decimate_per_pixel and len(x) > width_px * self.decimate_per_pixel:
            indices = np.linspace(0, len(x) - 1, int(width_px * self.decimate_per_pixel), dtype=int)
            x, y = x[indices], y[indices]

        points = " ".join(
            f"{time_to_pixel_x(t, ctx):.2f},{value_to_pixel_y(val, y_min, y_max, height_px):.2f}"
            for t, val in zip(x, y)
        )

        color_hex = rgb_to_hex(self.color)
        opacity_attr = f' opacity="{self.alpha}"' if self.alpha < 1.0 else ''
        _svg_dash = {"dashed": ' stroke-dasharray="4,4"', "dotted": ' stroke-dasharray="1,3" stroke-linecap="round"', "dashdotted": ' stroke-dasharray="4,2,1,2"'}
        dash_attr = _svg_dash.get(self.line_type, '')

        parts = [f'  <g id="{self.name}" class="layer {self.svg_class}">']
        parts.extend(self._axis_svg(ctx, y_min, y_max, tick_format, axis_flag))
        parts.append(f'    <polyline points="{points}" stroke="{color_hex}" stroke-width="{self.line_width}" fill="none"{opacity_attr}{dash_attr}/>')
        parts.append('  </g>')
        return '\n'.join(parts)


class Events(Layer):
    """A set of instantaneous markers (vertical lines): onsets, beats, downbeats."""

    def __init__(self, name: str, color, line_width: float = 0.5,
                 line_type: str = "solid", label: Optional[str] = None,
                 secondary_axis: bool = False, svg_class: str = "events"):
        super().__init__(name)
        self.color = color
        self.line_width = line_width
        self.line_type = line_type
        self.label = label or name
        self.secondary_axis = secondary_axis
        self.svg_class = svg_class

    @abstractmethod
    def _get_times(self, shared_data: Dict[str, Any]) -> Optional[np.ndarray]:
        ''' Return the event times, or None if not loaded '''
        pass

    def _setup_secondary_axis(self, ax: Axes, shared_data: Dict[str, Any]) -> Axes:
        if "ax2" not in shared_data:
            ax2 = ax.twinx()
            shared_data["ax2"] = ax2
        else:
            ax2 = shared_data["ax2"]
        cur_min, cur_max = ax2.get_ylim()
        ax2.set_ylim(min(cur_min, -1.0), max(cur_max, 1.0))
        return ax2

    def draw(self, ax: Axes, shared_data: Dict[str, Any]) -> Tuple[List, List]:
        times = self._get_times(shared_data)
        if times is None or len(times) == 0:
            print(f"✗ {self.name}: No data loaded")
            return [], []

        target_ax = self._setup_secondary_axis(ax, shared_data) if self.secondary_axis else ax
        _mpl_styles = {"solid": "-", "dashed": "--", "dotted": ":", "dashdotted": "-."}
        style = _mpl_styles.get(self.line_type, "-")
        lines = [target_ax.axvline(x=t, color=self.color, linestyle=style, linewidth=self.line_width, label=self.label)
                 for t in times]
        return lines, [self.label] if lines else []

    def _lines_svg(self, shared_data: Dict[str, Any]) -> List[str]:
        ''' Return the raw <line> elements for this layer's events, without the wrapping <g> '''
        if "svg_context" not in shared_data:
            return []
        times = self._get_times(shared_data)
        if times is None or len(times) == 0:
            return []

        ctx = shared_data["svg_context"]
        color_hex = rgb_to_hex(self.color)
        _svg_dash = {"dashed": ' stroke-dasharray="4,4"', "dotted": ' stroke-dasharray="1,3" stroke-linecap="round"', "dashdotted": ' stroke-dasharray="4,2,1,2"'}
        dash_attr = _svg_dash.get(self.line_type, '')
        return [
            f'    <line x1="{time_to_pixel_x(t, ctx):.2f}" y1="0" x2="{time_to_pixel_x(t, ctx):.2f}" y2="{ctx["height_px"]}" stroke="{color_hex}" stroke-width="{self.line_width}"{dash_attr}/>'
            for t in times
        ]

    def to_svg_group(self, shared_data: Dict[str, Any]) -> Optional[str]:
        lines = self._lines_svg(shared_data)
        if not lines:
            return None
        svg_group = f'''  <g id="{self.name}" class="layer {self.svg_class}">
{chr(10).join(lines)}
  </g>'''
        return svg_group


class Intervals(Layer):
    """Confidence windows: contiguous regions where an activation curve
    exceeds a threshold, rendered as rectangles with an opacity gradient
    from threshold (transparent) to peak (opaque).
    """

    def __init__(self, name: str, color, threshold: float = 70,
                 alpha_max: float = 0.3, svg_class: str = "intervals"):
        super().__init__(name)
        self.threshold = self._normalize_threshold(threshold)
        self.beat_window = self.threshold  # backward-compatible alias
        self.logit_threshold = self._threshold_to_logit(self.threshold)
        self.color = color
        self.alpha_max = alpha_max
        self.svg_class = svg_class

    @staticmethod
    def _normalize_threshold(threshold: float) -> float:
        ''' Convert threshold to 0-100 scale if needed '''
        return threshold * 100 if threshold <= 1.0 else threshold

    @staticmethod
    def _threshold_to_logit(threshold: float) -> float:
        ''' Convert a percentage threshold to the equivalent raw logit threshold '''
        probability = np.clip(threshold / 100, np.finfo(float).eps, 1 - np.finfo(float).eps)
        return float(np.log(probability / (1 - probability)))

    def _find_windows(self, activation: np.ndarray) -> List[Tuple[int, int, float]]:
        ''' Find contiguous regions where activation exceeds logit_threshold '''
        above_threshold = activation >= self.logit_threshold

        windows = []
        in_window = False
        window_start = 0
        window_values = []

        for i, is_above in enumerate(above_threshold):
            if is_above:
                if not in_window:
                    window_start = i
                    in_window = True
                window_values.append(activation[i])
            else:
                if in_window:
                    windows.append((window_start, i - 1, np.max(window_values)))
                    in_window = False
                    window_values = []

        if in_window:
            windows.append((window_start, len(activation) - 1, np.max(window_values)))

        return windows

    @abstractmethod
    def _get_activation(self, shared_data: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        ''' Return (times, activation) arrays, or None if not loaded '''
        pass

    def _setup_secondary_axis(self, ax: Axes, shared_data: Dict[str, Any], activation: np.ndarray) -> Axes:
        if "ax2" not in shared_data:
            ax2 = ax.twinx()
            shared_data["ax2"] = ax2
        else:
            ax2 = shared_data["ax2"]
        limit = max(1.0, float(np.ceil(np.max(np.abs(activation)))))
        cur_min, cur_max = ax2.get_ylim()
        ax2.set_ylim(min(cur_min, -limit), max(cur_max, limit))
        return ax2

    def draw(self, ax: Axes, shared_data: Dict[str, Any]) -> Tuple[List, List]:
        ta = self._get_activation(shared_data)
        if ta is None:
            print(f"✗ {self.name}: No data loaded")
            return [], []
        times, activation = ta

        ax2 = self._setup_secondary_axis(ax, shared_data, activation)
        windows = self._find_windows(activation)

        rectangles = []
        for start_idx, end_idx, _ in windows:
            t_start, t_end = times[start_idx], times[end_idx]

            ''' Draw with full opacity in matplotlib '''
            logit_y_min, logit_y_max = ax2.get_ylim()
            rect = plt.Rectangle((t_start, logit_y_min), t_end - t_start, logit_y_max - logit_y_min,
                                  alpha=1.0, color=self.color, label=self.name)
            ax2.add_patch(rect)
            rectangles.append(rect)

        return rectangles, [self.name] if rectangles else []

    def to_svg_group(self, shared_data: Dict[str, Any]) -> Optional[str]:
        if "svg_context" not in shared_data:
            return None
        ta = self._get_activation(shared_data)
        if ta is None:
            return None
        times, activation = ta

        windows = self._find_windows(activation)
        if len(windows) == 0:
            return None

        ctx = shared_data["svg_context"]
        color_hex = rgb_to_hex(self.color)
        gradients, rectangles = [], []

        for idx, (start_idx, end_idx, _) in enumerate(windows):
            x1 = time_to_pixel_x(times[start_idx], ctx)
            x2 = time_to_pixel_x(times[end_idx], ctx)

            ''' Create unique gradient ID for this window '''
            gradient_id = f"{self.name.replace(' ', '_')}_gradient_{idx}"

            ''' Define linear gradient: transparent at edges, opaque at peak '''
            gradients.append(f'''    <linearGradient id="{gradient_id}" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:{color_hex};stop-opacity:0"/>
      <stop offset="50%" style="stop-color:{color_hex};stop-opacity:{self.alpha_max:.2f}"/>
      <stop offset="100%" style="stop-color:{color_hex};stop-opacity:0"/>
    </linearGradient>''')

            rectangles.append(f'    <rect x="{x1:.2f}" y="0" width="{x2 - x1:.2f}" height="{ctx["height_px"]}" fill="url(#{gradient_id})"/>')

        svg_group = f'''  <g id="{self.name}" class="layer {self.svg_class}">
    <defs>
{chr(10).join(gradients)}
    </defs>
{chr(10).join(rectangles)}
  </g>'''

        return svg_group


class Field(Layer):
    """A dense 2D time-frequency (or time-pitch) representation, embedded as a
    raster image inside its group, with its axes drawn over it as SVG.
    """

    def __init__(self, name: str, color_map: str = "magma",
                 resample_filter: Optional[int] = None, svg_class: str = "field"):
        super().__init__(name)
        self.color_map = color_map
        self.resample_filter = resample_filter if resample_filter is not None else PILImage.LANCZOS
        self.svg_class = svg_class

    @abstractmethod
    def _get_matrix(self, shared_data: Dict[str, Any]) -> Optional[np.ndarray]:
        ''' Return the 2D matrix to render, with row 0 at the bottom of the image '''
        pass

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        ''' Min-max normalise to [0, 1]. Override when data is already in range '''
        m_min, m_max = matrix.min(), matrix.max()
        return (matrix - m_min) / (m_max - m_min) if m_max > m_min else np.zeros_like(matrix)

    def _y_ticks(self, shared_data: Dict[str, Any]) -> List[Tuple[float, str, float]]:
        ''' Return (fraction_from_bottom, label, text_y_offset) triples for the y-axis overlay '''
        return []

    def to_svg_group(self, shared_data: Dict[str, Any]) -> Optional[str]:
        matrix = self._get_matrix(shared_data)
        if matrix is None:
            return None
        try:
            ctx = shared_data.get("svg_context")
            if ctx is None:
                return None

            width_px = int(round(ctx["width_px"]))
            height_px = int(round(ctx["height_px"]))
            x_min, x_max = ctx["x_min"], ctx["x_max"]
            show_axes = ctx.get("show_axes", False)

            ''' Bypass matplotlib entirely: normalize -> colormap -> PIL -> PNG bytes '''
            normalized = self._normalize(matrix)
            rgba = plt.get_cmap(self.color_map)(normalized)
            rgba = rgba[::-1, :, :]  # flip: row 0 -> bottom of image, origin='lower'
            img = PILImage.fromarray((rgba * 255).astype(np.uint8), "RGBA")
            img = img.resize((width_px, height_px), self.resample_filter)

            buf = io.BytesIO()
            img.save(buf, format="png")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            ''' Image fills the full SVG area — no margins, no whitespace '''
            parts = [f'  <g id="{self.name}" class="layer {self.svg_class}">']
            parts.append(f'    <image x="0" y="0" width="{width_px}" height="{height_px}" href="data:image/png;base64,{b64}" preserveAspectRatio="none"/>')

            if show_axes:
                ''' Y axis line along left edge '''
                parts.append(f'    <line x1="0" y1="0" x2="0" y2="{height_px}" stroke="#111" stroke-width="1"/>')
                ''' X axis line along bottom edge '''
                parts.append(f'    <line x1="0" y1="{height_px}" x2="{width_px}" y2="{height_px}" stroke="#111" stroke-width="1"/>')

                ''' Y ticks + labels drawn outwards (left of image) '''
                for frac, label, text_dy in self._y_ticks(shared_data):
                    y_s = height_px * (1 - frac)
                    parts.append(f'    <line x1="-4" y1="{y_s:.1f}" x2="0" y2="{y_s:.1f}" stroke="#111" stroke-width="1"/>')
                    parts.append(f'    <text x="-6" y="{y_s + 2 + text_dy:.1f}" text-anchor="end" font-size="8" font-family="Arial,sans-serif" fill="#111">{label}</text>')

                ''' X ticks: minor every 1s, major (labeled) every 5s '''
                t_start = int(np.ceil(x_min))
                t_end = int(np.floor(x_max))
                for t in range(t_start, t_end + 1):
                    x_s = time_to_pixel_x(t, ctx)
                    if t % 5 == 0:
                        parts.append(f'    <line x1="{x_s:.1f}" y1="{height_px}" x2="{x_s:.1f}" y2="{height_px + 6}" stroke="#111" stroke-width="1"/>')
                        parts.append(f'    <text x="{x_s:.1f}" y="{height_px + 14:.1f}" text-anchor="middle" font-size="8" font-family="Arial,sans-serif" fill="#111">{t}s</text>')
                    else:
                        parts.append(f'    <line x1="{x_s:.1f}" y1="{height_px}" x2="{x_s:.1f}" y2="{height_px + 4}" stroke="#111" stroke-width="0.7"/>')

            parts.append("  </g>")
            return "\n".join(parts)

        except Exception as e:
            print(f"✗ Error converting {self.name} to SVG: {e}")
            return None
