"""
Spectrogram visualization layer.
Visualizes audio spectrograms with mel-scale frequency binning.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Dict, Any, List, Tuple, Optional
import io
import base64
from PIL import Image as PILImage

# Import Layer base class
from .visualization_system import Layer

class Spectrogram(Layer):
    # =========
    # INITIALIZATION AND CONFIGURATION
    def __init__(self, name: str = "Spectrogram", 
                 freq_window: Tuple[int, int] = (20, 4000),
                 color_map: str = "magma"):
        super().__init__(name)

        self.freq_window = freq_window
        self.color_map = color_map
    # =========

    def load_data(self, audio_path: str, print_output: bool = False, **kwargs) -> bool:
        # ========================================================
        # LOAD AUDIO & COMPUTE SPECTROGRAM
        import librosa
        from pathlib import Path
        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=False)
            filename = Path(audio_path).stem
            
            if audio.ndim == 2:
                audio = np.mean(audio, axis=0)
            
            winlen = int(0.256 * sr)
            hoplen = winlen // 16
            S_mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=winlen,
                                                   hop_length=hoplen, n_mels=512,
                                                   fmin=self.freq_window[0], fmax=self.freq_window[1])
            S_db = librosa.power_to_db(S_mel, ref=np.max)
            mel_freqs = librosa.mel_frequencies(n_mels=512, fmin=self.freq_window[0], fmax=self.freq_window[1])
            times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hoplen)
            
            self._data = {
                "S_db": S_db,
                "freqs": mel_freqs,
                "times": times,
                "sr": sr,
                "filename": filename,
                "audio": audio
            }
            if print_output:
                print(f"✓ SpectrogramLayer: Loaded {filename}")
            return True
        # LOAD AUDIO & COMPUTE SPECTROGRAM
        # ========================================================

        except Exception as e:      # debug info in case of errors
            print(f"✗ SpectrogramLayer error: {e}")
            return False
        
    
    def draw(self, ax: Axes, shared_data: Dict[str, Any]) -> Tuple[List, List]:

        # ========================================================
        # PAINT SPECTROGRAM
        if self._data is None:
            print("✗ SpectrogramLayer: No data loaded")
            return [], []

        times = self._data["times"]
        freqs = self._data["freqs"]
        ''' Match modusa.paint.image's pixel-edge extent calculation for origin="lower" '''
        dx = times[1] - times[0] if len(times) > 1 else 1
        dy = freqs[1] - freqs[0] if len(freqs) > 1 else 1
        extent = [times[0] - dx / 2, times[-1] + dx / 2, freqs[0] - dy / 2, freqs[-1] + dy / 2]
        ax.imshow(self._data["S_db"], aspect="auto", origin="lower", cmap=self.color_map, extent=extent)
        
        shared_data.update(self._data)
        
        ax.set_ylim(self._data["freqs"][0], self._data["freqs"][-1])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Spectrogram: {self._data['filename']}")
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        # PAINT SPECTROGRAM
        # ========================================================

        return [], []

    def to_svg_group(self, shared_data: Dict[str, Any]) -> Optional[str]:
        '''Render spectrogram as full-size PNG then overlay axis labels as SVG elements directly on the image'''
        if self._data is None:
            return None
        try:
            ctx = shared_data.get("svg_context")
            if ctx is None:
                return None

            width_px  = int(round(ctx["width_px"]))
            height_px = int(round(ctx["height_px"]))
            x_min     = ctx["x_min"]
            x_max     = ctx["x_max"]
            show_axes = ctx.get("show_axes", False)

            ''' Bypass matplotlib entirely: normalize → colormap → PIL → PNG bytes '''
            S_db  = self._data["S_db"]
            S_min, S_max = S_db.min(), S_db.max()
            S_norm  = (S_db - S_min) / (S_max - S_min) if S_max > S_min else np.zeros_like(S_db)
            rgba    = plt.get_cmap(self.color_map)(S_norm)   # (n_mels, n_frames, 4)
            rgba    = rgba[::-1, :, :]                        # flip: origin='lower'
            img     = PILImage.fromarray((rgba * 255).astype(np.uint8), "RGBA")
            img     = img.resize((width_px, height_px), PILImage.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="png")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            ''' Image fills the full SVG area — no margins, no whitespace '''
            parts = [f'  <g id="{self.name}" class="layer spectrogram">']
            parts.append(f'    <image x="0" y="0" width="{width_px}" height="{height_px}" href="data:image/png;base64,{b64}" preserveAspectRatio="none"/>')

            if show_axes:
                f_min = self._data["freqs"][0]
                f_max = self._data["freqs"][-1]
                N_y = 6

                ''' Y axis line along left edge '''
                parts.append(f'    <line x1="0" y1="0" x2="0" y2="{height_px}" stroke="#111" stroke-width="1"/>')
                ''' X axis line along bottom edge '''
                parts.append(f'    <line x1="0" y1="{height_px}" x2="{width_px}" y2="{height_px}" stroke="#111" stroke-width="1"/>')

                ''' Y ticks + labels drawn outwards (left of image) '''
                for i in range(N_y):
                    f = f_min + (f_max - f_min) * i / (N_y - 1)
                    y_s = height_px - (f - f_min) / (f_max - f_min) * height_px
                    y_offset = -3 if i == 0 else 0
                    label = f"{int(f)}Hz"
                    parts.append(f'    <line x1="-4" y1="{y_s:.1f}" x2="0" y2="{y_s:.1f}" stroke="#111" stroke-width="1"/>')
                    parts.append(f'    <text x="-6" y="{y_s + 2 + y_offset:.1f}" text-anchor="end" font-size="8" font-family="Arial,sans-serif" fill="#111">{label}</text>')

                ''' X ticks: minor every 1s, major (labeled) every 5s '''
                t_start = int(np.ceil(x_min))
                t_end = int(np.floor(x_max))
                for t in range(t_start, t_end + 1):
                    x_s = (t - x_min) / (x_max - x_min) * width_px
                    if t % 5 == 0:
                        parts.append(f'    <line x1="{x_s:.1f}" y1="{height_px}" x2="{x_s:.1f}" y2="{height_px + 6}" stroke="#111" stroke-width="1"/>')
                        label = f"{t}s"
                        parts.append(f'    <text x="{x_s:.1f}" y="{height_px + 14:.1f}" text-anchor="middle" font-size="8" font-family="Arial,sans-serif" fill="#111">{label}</text>')
                    else:
                        parts.append(f'    <line x1="{x_s:.1f}" y1="{height_px}" x2="{x_s:.1f}" y2="{height_px + 4}" stroke="#111" stroke-width="0.7"/>')

            parts.append("  </g>")
            return "\n".join(parts)

        except Exception as e:
            print(f"✗ Error converting Spectrogram to SVG: {e}")
            return None


class Chromagram(Layer):
    # =========
    # INITIALIZATION AND CONFIGURATION
    def __init__(self, name: str = "Chromagram",
                 color_map: str = "coolwarm",
                 n_chroma: int = 12):
        super().__init__(name)
        self.color_map = color_map
        self.n_chroma = n_chroma
    # =========

    def load_data(self, audio_path: str, print_output: bool = False, **kwargs) -> bool:
        # ========================================================
        # LOAD AUDIO & COMPUTE CHROMAGRAM
        import librosa
        from pathlib import Path
        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=False)
            filename = Path(audio_path).stem

            if audio.ndim == 2:
                audio = np.mean(audio, axis=0)

            hoplen = 512
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr,
                                                 hop_length=hoplen,
                                                 n_chroma=self.n_chroma)
            times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hoplen)
            pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F',
                             'F#', 'G', 'G#', 'A', 'A#', 'B']

            self._data = {
                "chroma": chroma,
                "times": times,
                "sr": sr,
                "filename": filename,
                "audio": audio,
                "pitch_classes": pitch_classes
            }
            if print_output:
                print(f"✓ Chromagram: Loaded {filename}")
            return True
        # LOAD AUDIO & COMPUTE CHROMAGRAM
        # ========================================================

        except Exception as e:
            print(f"✗ Chromagram error: {e}")
            return False

    def draw(self, ax: Axes, shared_data: Dict[str, Any]) -> Tuple[List, List]:
        # ========================================================
        # PAINT CHROMAGRAM
        import librosa.display
        if self._data is None:
            print("✗ Chromagram: No data loaded")
            return [], []

        img = librosa.display.specshow(
            self._data["chroma"],
            x_axis='time',
            y_axis='chroma',
            sr=self._data["sr"],
            hop_length=512,
            cmap=self.color_map,
            ax=ax
        )

        ax.set_title(f"Chromagram: {self._data['filename']}")

        pitch_classes = self._data["pitch_classes"]
        import matplotlib.transforms as transforms
        for i, label in enumerate(pitch_classes):
            offset = transforms.ScaledTranslation(0, -5 / ax.get_figure().dpi, ax.get_figure().dpi_scale_trans)
            trans = ax.get_yaxis_transform() + offset
            ax.text(
                0, i + 0.35, f" {label}",
                transform=trans,
                ha='left', va='center',
                fontsize=3, fontweight='bold', color='white',
    
            )

        shared_data.update(self._data)
        # PAINT CHROMAGRAM
        # ========================================================

        return [], []


class Waveform(Layer):
    # =========
    # INITIALIZATION AND CONFIGURATION
    def __init__(self, name: str = "Waveform",
                 color: str = "steelblue",
                 alpha: float = 0.8,
                 normalize: bool = False):
        super().__init__(name)
        self.color = color
        self.alpha = alpha
        self.normalize = normalize
    # =========

    def load_data(self, audio_path: str, print_output: bool = False, **kwargs) -> bool:
        # ========================================================
        # LOAD AUDIO
        import librosa
        from pathlib import Path
        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=False)
            filename = Path(audio_path).stem

            if audio.ndim == 2:
                audio = np.mean(audio, axis=0)

            duration = len(audio) / sr
            times = np.linspace(0, duration, num=len(audio))

            self._data = {
                "audio": audio,
                "times": times,
                "sr": sr,
                "filename": filename,
                "duration": duration
            }
            if print_output:
                print(f"✓ Waveform: Loaded {filename}, duration={duration:.2f}s")
            return True
        # LOAD AUDIO
        # ========================================================

        except Exception as e:
            print(f"✗ Waveform error: {e}")
            return False

    def draw(self, ax: Axes, shared_data: Dict[str, Any]) -> Tuple[List, List]:
        # ========================================================
        # PAINT WAVEFORM
        if self._data is None:
            print("✗ Waveform: No data loaded")
            return [], []

        audio = self._data["audio"]
        if self.normalize:
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak

        line, = ax.plot(
            self._data["times"],
            audio,
            color=self.color,
            linewidth=0.4,
            alpha=self.alpha,
            label="Waveform"
        )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Waveform: {self._data['filename']}")
        ax.set_ylim(-1.0, 1.0)
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

        shared_data.update(self._data)
        # PAINT WAVEFORM
        # ========================================================

        return [line], ["Waveform"]

    def to_svg_group(self, shared_data: Dict[str, Any]) -> Optional[str]:
        '''Convert waveform to SVG polyline, downsampled to ~4 points per pixel, with optional axes'''
        if self._data is None or "svg_context" not in shared_data:
            return None

        ctx = shared_data["svg_context"]
        times = self._data["times"]
        audio = self._data["audio"]
        if self.normalize:
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak

        x_min, x_max = ctx["x_min"], ctx["x_max"]
        y_min, y_max = ctx["y_min"], ctx["y_max"]
        width_px, height_px = ctx["width_px"], ctx["height_px"]
        show_axes = ctx.get("show_axes", False)

        if x_max == x_min or y_max == y_min:
            return None

        target_points = int(width_px * 4)
        if len(times) > target_points:
            indices = np.linspace(0, len(times) - 1, target_points, dtype=int)
            times = times[indices]
            audio = audio[indices]

        points = []
        for t, amp in zip(times, audio):
            x = ((t - x_min) / (x_max - x_min)) * width_px
            y = (1.0 - (amp - y_min) / (y_max - y_min)) * height_px
            points.append(f"{x:.2f},{y:.2f}")

        points_str = " ".join(points)

        if isinstance(self.color, tuple) and len(self.color) >= 3:
            r, g, b = [int(c * 255) if c <= 1 else int(c) for c in self.color[:3]]
            color_hex = f'#{r:02x}{g:02x}{b:02x}'
        else:
            try:
                from matplotlib.colors import to_hex
                color_hex = to_hex(self.color)
            except Exception:
                color_hex = '#000000'

        opacity_attr = f' opacity="{self.alpha}"' if self.alpha < 1.0 else ''

        parts = [f'  <g id="{self.name}" class="layer waveform">']
        parts.append(f'    <polyline points="{points_str}" stroke="{color_hex}" stroke-width="0.4" fill="none"{opacity_attr}/>')

        if show_axes:
            N_y = 5

            ''' Y axis line along left edge '''
            parts.append(f'    <line x1="0" y1="0" x2="0" y2="{height_px}" stroke="#111" stroke-width="1"/>')
            ''' X axis line along bottom edge '''
            parts.append(f'    <line x1="0" y1="{height_px}" x2="{width_px}" y2="{height_px}" stroke="#111" stroke-width="1"/>')

            ''' Zero amplitude dashed reference line '''
            if y_min < 0 < y_max:
                y_zero = (1.0 - (0 - y_min) / (y_max - y_min)) * height_px
                parts.append(f'    <line x1="0" y1="{y_zero:.1f}" x2="{width_px}" y2="{y_zero:.1f}" stroke="#999" stroke-width="0.5" stroke-dasharray="4,3"/>')

            ''' Y ticks and labels drawn outwards (left of image) '''
            for i in range(N_y):
                amp = y_min + (y_max - y_min) * i / (N_y - 1)
                y_s = (1.0 - (amp - y_min) / (y_max - y_min)) * height_px
                label = f"{amp:.1f}"
                parts.append(f'    <line x1="-4" y1="{y_s:.1f}" x2="0" y2="{y_s:.1f}" stroke="#111" stroke-width="1"/>')
                parts.append(f'    <text x="-6" y="{y_s + 3:.1f}" text-anchor="end" font-size="8" font-family="Arial,sans-serif" fill="#111">{label}</text>')

            ''' X ticks: minor every 1s, major (labeled) every 5s '''
            t_start = int(np.ceil(x_min))
            t_end = int(np.floor(x_max))
            for t in range(t_start, t_end + 1):
                x_s = (t - x_min) / (x_max - x_min) * width_px
                if t % 5 == 0:
                    parts.append(f'    <line x1="{x_s:.1f}" y1="{height_px}" x2="{x_s:.1f}" y2="{height_px + 6}" stroke="#111" stroke-width="1"/>')
                    label = f"{t}s"
                    parts.append(f'    <text x="{x_s:.1f}" y="{height_px + 14:.1f}" text-anchor="middle" font-size="8" font-family="Arial,sans-serif" fill="#111">{label}</text>')
                else:
                    parts.append(f'    <line x1="{x_s:.1f}" y1="{height_px}" x2="{x_s:.1f}" y2="{height_px + 4}" stroke="#111" stroke-width="0.7"/>')

        parts.append('  </g>')
        return '\n'.join(parts)