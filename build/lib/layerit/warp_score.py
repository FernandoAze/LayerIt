from abc import ABC, abstractmethod
import sys
from matplotlib import lines
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Dict, Any, List, Tuple, Optional
import json
import xml.etree.ElementTree as ET
from PIL import Image
import base64
import io
from pathlib import Path
from sklearn import tree
import soundfile
import re


from .visualization_system import Layer

class Onsets_Layer(Layer):
    def __init__(self, name: str = "obs_mean_onsets", onset_color: str = 'yellow', line_width: float = 0.2):
        super().__init__(name)
        self.onset_color = onset_color
        self.line_width = line_width
    def load_data(self, maps_file: str = None, **kwargs) -> bool:
        """Load onsets from MAPS JSON file"""
        try:
            if maps_file is None:
                print(f"✗ {self.name}: Maps file path must be provided")
                return False
            
            with open(str(maps_file), 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list) or len(data) == 0:
                print("✗ Maps file ERROR: Expected a non-empty array")
                return False
            
            ''' Check if obs_mean_onset exists in the first entry '''
            if 'obs_mean_onset' not in data[0]:
                print("✗ Maps file ERROR: 'obs_mean_onset' not found in data")
                return False
            
            onset_times = [entry['obs_mean_onset'] for entry in data]
            
            self._data = {
                "onset_times": onset_times,
            }
            # print(f"✓ {self.name}: Loaded {len(self._data['onset_times'])} onsets")
            return True
        except Exception as e:
            print(f"✗ {self.name} error: {e}")
            return False
    
    def draw(self, ax, shared_data) -> Tuple[List, List]:
        if self._data is None:
            print("✗ Onsets_Layer: No data loaded")
            return [], []
        
        lines = []

        for onset in self._data['onset_times']:
            line = ax.axvline(x=onset, color=self.onset_color,
            linestyle='--', linewidth=self.line_width, label='Onset')
            lines.append(line)
        
        if lines:
            labels = [self.name]
        
        return lines, labels
    
    #========================================================
    # Methods for Layers that are vector based for SVG output
    def to_svg_group(self, shared_data: Dict[str, Any]) -> Optional[str]:
        '''Convert onsets to SVG dashed vertical lines'''
        if self._data is None or "svg_context" not in shared_data:
            return None
        
        ctx = shared_data["svg_context"]
        onset_times = self._data.get("onset_times", [])
        
        if len(onset_times) == 0:
            return None
        
        lines = []
        color_hex = self._rgb_to_hex(self.onset_color)
        
        ''' Draw dashed vertical lines for each onset '''
        for onset_time in onset_times:
            if ctx["x_max"] == ctx["x_min"]:
                x = 0
            else:
                x = ((onset_time - ctx["x_min"]) / (ctx["x_max"] - ctx["x_min"])) * ctx["width_px"]
            
            lines.append(f'    <line x1="{x:.2f}" y1="0" x2="{x:.2f}" y2="{ctx["height_px"]}" stroke="{color_hex}" stroke-width="{self.line_width}" stroke-dasharray="2,2"/>')
        
        svg_group = f'''  <g id="{self.name}" class="layer onsets">
{chr(10).join(lines)}
  </g>'''
        
        return svg_group

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
    # Methods for Layers that are vector based for SVG output
    #========================================================

def TXT_to_Maps(txt_maps_file: str, output_file: str = None, print_output: bool = False) -> list:
    '''
    Convert txt annotation file to maps.json format.
    
    Input txt format (tab-separated):
        onset_time    xml_id1,xml_id2,...
        0.034829932   a13d7g5m
        0.470204082   v2xdb2q
    
    Output maps format:
        [{"obs_mean_onset": 0.034829932, "xml_id": ["a13d7g5m"], "obs_num": 1}, ...]
    '''
    try:
        if not Path(txt_maps_file).exists():
            print(f"✗ TXT_to_Maps: File not found: {txt_maps_file}")
            return None
        
        maps_data = []
        obs_num = 1
        
        with open(txt_maps_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                ''' Split by tab to get onset_time and xml_ids '''
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                try:
                    obs_mean_onset = float(parts[0])
                except ValueError:
                    continue
                
                ''' Parse xml_ids (can be comma-separated) '''
                xml_ids_str = parts[1].strip()
                xml_ids = [xid.strip() for xid in xml_ids_str.split(',')]
                
                ''' Create entry '''
                entry = {
                    "obs_mean_onset": obs_mean_onset,
                    "xml_id": xml_ids,
                    "obs_num": obs_num
                }
                maps_data.append(entry)
                obs_num += 1
        
        if not maps_data:
            print(f"✗ TXT_to_Maps: No valid data found in {txt_maps_file}")
            return None
        
        ''' Determine output file name if not provided '''
        if output_file is None:
            txt_path = Path(txt_maps_file)
            output_file = str(txt_path.parent / f"{txt_path.stem}.maps")
        
        ''' Save to JSON file '''
        with open(output_file, 'w') as f:
            json.dump(maps_data, f, indent=0)
        
        if print_output:
            print(f"✓ TXT_to_Maps: Converted {len(maps_data)} entries")
            print(f"✓ Output saved to: {output_file}")
        
        return maps_data
        
    except Exception as e:
        print(f"✗ TXT_to_Maps error: {e}")
        return None

class Warp_Score():

    def __init__(self):
        self._data = None

    def load_data(self, maps_file: str = None, **kwargs) -> bool:
        ''' Load maps file containing score alignment data '''
        try:
            if maps_file is None:
                print("✗ Warp_Score: Maps file path must be provided")
                return False
            
            with open(str(maps_file), 'r') as f:
                self._data = json.load(f)
            
            if not isinstance(self._data, list) or len(self._data) == 0:
                print("✗ Maps file ERROR: Expected a non-empty array")
                return False
            
            print(f"✓ Warp_Score: Loaded {len(self._data)} entries from maps file")
            return True
        except Exception as e:
            print(f"✗ Warp_Score error: {e}")
            return False
        
    def get_timeAxis_bounds(self, svg_file: str, print_output: bool = False):
        ''' Extract the x-coordinates of the first and last vertical lines in timeAxis '''
        try:
            tree = ET.parse(svg_file)
            root = tree.getroot()
            ns = {'svg': 'http://www.w3.org/2000/svg'}
            
            ''' Find timeAxis with namespace handling '''
            time_axis = root.find(".//svg:g[@class='timeAxis']", ns)
            if time_axis is None:
                time_axis = root.find(".//g[@class='timeAxis']")
            
            if time_axis is None:
                print("✗ Could not find timeAxis element")
                return None
            
            ''' Extract all line elements and find vertical lines (where x1 == x2) '''
            lines = time_axis.findall(".//svg:line", ns)
            if not lines:
                lines = time_axis.findall(".//line")
            
            if not lines:
                print("✗ No line elements found in timeAxis")
                return None
            
            ''' Filter vertical lines (x1 == x2) and get their x positions '''
            vertical_lines = []
            for line in lines:
                x1 = line.get('x1')
                x2 = line.get('x2')
                if x1 and x2:
                    try:
                        x1_val = float(x1)
                        x2_val = float(x2)
                        if abs(x1_val - x2_val) < 0.01:
                            vertical_lines.append(x1_val)
                    except ValueError:
                        continue
            
            if len(vertical_lines) < 2:
                print("✗ Could not find at least 2 vertical lines in timeAxis")
                return None
            
            vertical_lines.sort()
            first_x = vertical_lines[0]
            last_x = vertical_lines[-1]
            timeline_width = last_x - first_x
            
            if print_output==True:
                print(f"✓ TimeAxis bounds: first_x={first_x}, last_x={last_x}, width={timeline_width}")
            
            return first_x, last_x, timeline_width
            
        except Exception as e:
            print(f"✗ Error getting timeAxis bounds: {e}")
            return None
        

    def audio_duration(self, audio_file: str, print_output: bool = False):
        # Get the duration of the audio file in seconds
        audio_data, samplerate = soundfile.read(audio_file)
        duration = len(audio_data) / samplerate

        if print_output == True:
            print(f"✓ Audio duration: {duration:.2f} seconds")
        
        return duration

    def extract_viewBox_dimensions(self, svg_file: str, print_output: bool = False):
        svg_tree = ET.parse(svg_file)
        root = svg_tree.getroot()
        ns = {'svg': 'http://www.w3.org/2000/svg'}

        ''' Get viewBox from root or nested SVG '''
        viewbox = root.get('viewBox')
        if not viewbox:
            ''' Look for viewBox in nested SVG elements '''
            nested_svg = root.find(".//svg:svg", ns)
            if nested_svg is not None:
                viewbox = nested_svg.get('viewBox')

        if not viewbox:
            return None

        vb_parts = viewbox.split()
        vb_x, vb_y, vb_width, vb_height = map(float, vb_parts)

        ''' Get display dimensions from root SVG '''
        width_str = root.get('width')
        height_str = root.get('height')

        if width_str:
            display_width = float(width_str.replace('px', ''))
        else:
            display_width = vb_width

        if height_str:
            display_height = float(height_str.replace('px', ''))
        else:
            display_height = vb_height
        
        ''' Calculate scale factors '''
        scale_x = display_width / vb_width
        scale_y = display_height / vb_height

        if print_output==True:
            print(f"✓ SVG viewBox: x={vb_x}, y={vb_y}, width={vb_width}, height={vb_height}")
            print(f"✓ SVG display dimensions: width={display_width}, height={display_height}")
            print(f"✓ Scale factors: scale_x={scale_x}, scale_y={scale_y}")
        else:
            pass

        return {
            'display_width': display_width,
            'display_height': display_height,
            'viewbox_width': vb_width,
            'viewbox_height': vb_height,
            'scale_x': scale_x,
            'scale_y': scale_y
        }
    
    def get_FirstLast_NoteID(self, maps_file: str, print_output: bool = False):
        ''' Retrieve the xml_id of the first and last note from the maps file '''
        if maps_file is None:
            print("✗ valid maps_file must be provided")
            return None
        
        with open(str(maps_file), 'r') as f:
            maps = json.load(f)
        
        first_note_id = maps[0].get('xml_id')
        last_note_id = maps[-1].get('xml_id')

        if print_output == True:
            print(f"First note xml_id: {first_note_id}, Last note xml_id: {last_note_id}")

        return first_note_id, last_note_id
    
    def get_translate_value(self, svg_file: str, element_id: str, print_output: bool = False):
        '''Extract the translate x-value from an SVG element by its ID.
        Checks element and walks up parent chain to find translate.'''
        try:
            if isinstance(svg_file, str) and (
                svg_file.startswith('<?xml') or 
                svg_file.startswith('<svg')
            ):
                root = ET.fromstring(svg_file)
            else:
                tree = ET.parse(svg_file)
                root = tree.getroot()
        except ET.ParseError as e:
            print(f"✗ Error parsing SVG: {e}")
            return None
        
        ''' Build parent map '''
        parent_map = {c: p for p in root.iter() for c in p}
        
        ''' Find element by ID or data-id '''
        element = None
        for elem in root.iter():
            if elem.get('id') == element_id or elem.get('data-id') == element_id:
                element = elem
                break
        
        if element is None:
            print(f"✗ Element with ID '{element_id}' not found")
            return None
        
        ''' Walk up the parent chain looking for translate '''
        current = element
        while current is not None:
            transform = current.get('transform')
            if transform:
                match = re.search(r'translate\s*\(\s*([\d.\-]+)', transform)
                if match:
                    translate_value = float(match.group(1))
                    if print_output:
                        print(f"✓ Found translate value: {translate_value} for element '{element_id}'")
                    return translate_value
            current = parent_map.get(current)
        
        print(f"✗ No translate found for element '{element_id}' or its parents")
        return None

    def get_LastTranslation(self, svg_file: str, maps_file: str, print_output: bool = False):

        lastNoteID=self.get_FirstLast_NoteID(maps_file, print_output)[1]

        translate_value=self.get_translate_value(svg_file, lastNoteID, print_output)

        if print_output==True:
            print(f"✓ Last translation value: {translate_value}")

        return translate_value
    
    def extract_ScoreSVG_dimensions(self, svg_file: str, maps_file: str, print_output: bool = False):

        svg_tree = ET.parse(svg_file)
        root = svg_tree.getroot()
        ns = {'svg': 'http://www.w3.org/2000/svg'}

        ''' Get viewBox from root or nested SVG '''
        viewbox = root.get('viewBox')

        if not viewbox:
            ''' Look for viewBox in nested SVG elements '''
            nested_svg = root.find(".//svg:svg", ns)
            if nested_svg is not None:
                viewbox = nested_svg.get('viewBox')
        
        vb_parts = viewbox.split()
        dimensions = list(map(float, vb_parts))

        LastTranslation=self.get_LastTranslation(svg_file, maps_file, print_output)

        score_width = dimensions[2 ] + (LastTranslation*(-1)) 
        score_height = dimensions[3]

        if print_output==True:
            print(f"Score segment WIDTH: {score_width} ")
            print(f"Score segment HEIGHT: {score_height} ")

        return score_width, score_height



    def get_first_and_last_note_positions(self, svg_file: str, maps_file: str, print_output: bool = False):
        ''' Retrieve the x1 attribute (timeline begin) from the first note and last note in SVG using maps file for reference '''
        if svg_file is None or maps_file is None:
            print("✗ valid svg_file and maps_file must be provided")
            return None
    
        tree = ET.parse(svg_file)
        root = tree.getroot()
        
        # Load maps data
        with open(str(maps_file), 'r') as f:
            maps = json.load(f)
        
        first_note_id, last_note_id = self.get_FirstLast_NoteID(maps_file, print_output)
        
        ''' Look up the actual x position in the SVG using xml_id, finding use element inside note '''
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        first_element = root.find(f".//*[@data-id='{first_note_id}']//svg:use", ns)
        last_element = root.find(f".//*[@data-id='{last_note_id}']//svg:use", ns)
        
        if first_element is None or last_element is None:
            print(f"✗ Could not find use elements with ids: {first_note_id}, {last_note_id}")
            return None
        
        first_note_x = float(first_element.get('x', 0))
        last_note_x = float(last_element.get('x', 0))
        
        score_timeline_length = last_note_x - first_note_x

        if print_output == True:
            print(f"First note id: {first_note_id}, Last note id: {last_note_id}")
            print(f"First note x1: {first_note_x}, Last note x1: {last_note_x}")
            print(f"Score timeline length: {score_timeline_length}")

        return first_note_x, last_note_x, score_timeline_length
    
    def scale_Layers(self, score_svg_file: str = None, layers_svg_file: str = None, maps_file: str = None, print_output: bool = False):
        '''Calculate scale factor for layers SVG to fit within the timeAxis boundaries'''
        
        ''' Get timeAxis bounds '''
        timeAxis_bounds = self.get_timeAxis_bounds(score_svg_file, print_output)
        if timeAxis_bounds is None:
            print("✗ Could not get timeAxis bounds for scaling")
            return (1.0, 1.0)
        
        timeAxis_first_x, timeAxis_last_x, timeline_width = timeAxis_bounds
        
        ''' Parse the layers SVG file '''
        layers_tree = ET.parse(layers_svg_file)
        layers_root = layers_tree.getroot()
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        ''' Get width and height from layers SVG '''
        layers_width = layers_root.get('width')
        layers_height = layers_root.get('height')

        layers_width = float(layers_width.replace('px', ''))
        layers_height = float(layers_height.replace('px', ''))

        #MMMMMMMMMMMMMMMMMMMM
        #MMMMMM???????MMMMMMM
        #MMMMMMMMMMMMMMMMMMMM
        ''' Target height to match PNG display height '''
        target_height = 110
        #MMMMMMMMMMMMMMMMMMMM
        #MMMMMM???????MMMMMMM
        #MMMMMMMMMMMMMMMMMMMM

        ''' Calculate scale factors based on timeAxis width and target height '''
        scale_x = timeline_width / layers_width
        scale_y = target_height / layers_height
        scale_factor = (scale_x, scale_y)
        
        if print_output:
            print(f"Layers SVG: width={layers_width}, height={layers_height}")
            print(f"TimeAxis width={timeline_width}, Target height={target_height}")
            print(f"Scale factors: scale_x={scale_x}, scale_y={scale_y}")
    
        return scale_factor
