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

# Import Layer base class and shape primitives
from .visualization_system import Layer
from .shapes import Field, Curve

class MelSpec(Field):
    # =========
    # INITIALIZATION AND CONFIGURATION
    def __init__(self, name: str = "Spectrogram", 
                 freq_window: Tuple[int, int] = (20, 4000),
                 color_map: str = "magma"):
        super().__init__(name, color_map=color_map, svg_class="spectrogram")

        self.freq_window = freq_window
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

    def _get_matrix(self, shared_data: Dict[str, Any]) -> Optional[np.ndarray]:
        if self._data is None:
            return None
        return self._data["S_db"]

    def _y_ticks(self, shared_data: Dict[str, Any]) -> List[Tuple[float, str, float]]:
        f_min = self._data["freqs"][0]
        f_max = self._data["freqs"][-1]
        N_y = 6
        ticks = []
        for i in range(N_y):
            f = f_min + (f_max - f_min) * i / (N_y - 1)
            frac = i / (N_y - 1)
            text_dy = -3 if i == 0 else 0
            ticks.append((frac, f"{int(f)}Hz", text_dy))
        return ticks


class Chromagram(Field):
    # =========
    # INITIALIZATION AND CONFIGURATION
    def __init__(self, name: str = "Chromagram",
                 color_map: str = "coolwarm",
                 n_chroma: int = 12):
        super().__init__(name, color_map=color_map, resample_filter=PILImage.NEAREST, svg_class="chromagram")
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

    def _get_matrix(self, shared_data: Dict[str, Any]) -> Optional[np.ndarray]:
        if self._data is None:
            return None
        return self._data["chroma"]

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        ''' chroma_cqt values are already in [0, 1]; clip rather than min-max rescale '''
        return np.clip(matrix, 0, 1)

    def _y_ticks(self, shared_data: Dict[str, Any]) -> List[Tuple[float, str, float]]:
        pitch_classes = self._data["pitch_classes"]
        N_y = len(pitch_classes)
        return [((i + 0.5) / N_y, pitch_classes[i], 0.0) for i in range(N_y)]


class Waveform(Curve):
    # =========
    # INITIALIZATION AND CONFIGURATION
    def __init__(self, name: str = "Waveform",
                 color: str = "steelblue",
                 alpha: float = 0.8,
                 normalize: bool = False):
        super().__init__(name, color=color, line_width=0.4, alpha=alpha,
                          label="Waveform", decimate_per_pixel=4, svg_class="waveform")
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

    def _amplitude(self) -> np.ndarray:
        audio = self._data["audio"]
        if self.normalize:
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak
        return audio

    def _get_xy(self, shared_data: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._data is None:
            return None
        return self._data["times"], self._amplitude()

    def draw(self, ax: Axes, shared_data: Dict[str, Any]) -> Tuple[List, List]:
        # ========================================================
        # PAINT WAVEFORM
        if self._data is None:
            print("✗ Waveform: No data loaded")
            return [], []

        lines, labels = super().draw(ax, shared_data)

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

        return lines, labels