"""
Beat Visualization Package

Main classes and functions for layered visualization system.
"""

''' Core visualization system '''
from .visualization_system import Layer, Visualizer

''' Score warping and onset layers '''
from .warp_score import Onset, Warp_Score

''' Spectrogram, Chromagram and Waveform layers '''
from .Audio_Layers import MelSpec, Chromagram, Waveform

''' Shape-based rendering primitives '''
from .shapes import Curve, Events, Intervals, Field

''' Beat visualization layers and utilities '''
from .Beat_Layers import (
    NPZ_to_BeatTXT,
    NPZ_to_DownbeatTXT,
    BeatLogits,
    DownbeatLogits,
    BeatsLayer,
    DownbeatsLayer,
    BeatWindowLayer,
    DownbeatWindowLayer
)

__all__ = [
    'Layer',
    'Visualizer',
    'Onset',
    'Warp_Score',
    'MelSpec',
    'Chromagram',
    'Waveform',
    'Curve',
    'Events',
    'Intervals',
    'Field',
    'NPZ_to_BeatTXT',
    'NPZ_to_DownbeatTXT',
    'BeatLogits',
    'DownbeatLogits',
    'BeatsLayer',
    'DownbeatsLayer',
    'BeatWindowLayer',
    'DownbeatWindowLayer',
]
