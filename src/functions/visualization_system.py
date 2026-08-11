"""
Lego-like visualization system using OOP.
Each visual element is a Layer that knows how to draw itself.
The Visualizer assembles layers together.
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Dict, Any, List, Tuple, Optional
import os
import xml.etree.ElementTree as ET
from PIL import Image
import base64
import io
from pathlib import Path
import sys

# class Layer(ABC), defines the template for all layer subclasses.
class Layer(ABC): 
    def __init__(self, name: str = "Layer"):
        self._data = None # Placeholder for layer-specific data
        self.name = name # Optional name for debugging and legend purposes

    @abstractmethod
    def load_data(self, **kwargs) -> bool:
        pass
    
    @abstractmethod
    def draw(self, ax: Axes, shared_data: Dict[str, Any]) -> Tuple[List, List]:
        pass
    
    def to_svg_group(self, shared_data: Dict[str, Any]) -> Optional[str]:
        '''
        Convert layer drawing to SVG group markup.
        Override in subclasses to support SVG export.
        
        Returns:
            SVG group markup string, or None if not supported
        '''
        return None



class Visualizer:
    def __init__(self, figsize: Optional[Tuple[float, float]] = None, 
                 plot_size_inPxl: Optional[Tuple[int, int]] = None, 
                 dpi: int = 300):
        """
        Initialize Visualizer with customizable figure size.
        
        Args:
            figsize: Figure size as (width, height) in inches. Default (14, 8) if neither figsize nor pixel_size specified.
            pixel_size: Figure size as (width, height) in pixels. Converts to inches using dpi parameter.
            dpi: Dots per inch for pixel-to-inch conversion. Default is 96 (standard screen DPI).
        """
        self.layers: List[Layer] = []
        self.shared_data: Dict[str, Any] = {}
        self.fig = None
        self.ax = None
        self.all_lines = []
        self.all_labels = []
        self.dpi = dpi
        
        # Convert pixel_size to inches if provided, otherwise use figsize or default
        if plot_size_inPxl is not None:
            self.figsize = (plot_size_inPxl[0] / dpi, plot_size_inPxl[1] / dpi)
        elif figsize is not None:
            self.figsize = figsize
        else:
            self.figsize = (14, 8)  # Default size in inches
    
    def add_layer(self, layer: Layer) -> 'Visualizer':
        self.layers.append(layer)
        # print(f"Added layer: {layer.name}")
        return self
    
    def load_all_layers(self, audio_path: str = None, **kwargs) -> bool:
        # Calculate audio duration, store in shared_data.
        # Used to force x-axis time limit to audio duration.
        if audio_path is not None:
            from .warp_score import Warp_Score
            audio_duration = Warp_Score().audio_duration(audio_path)
            self.shared_data['audio_duration'] = audio_duration
            self.shared_data['audio_path'] = audio_path
        for layer in self.layers:
            if not layer.load_data(audio_path=audio_path, **kwargs):
                print(f"⚠ Warning: Layer '{layer.name}' failed to load")
                return False
        return True
    
    def draw(self) -> Tuple[plt.Figure, plt.Axes]:
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
        for layer in self.layers:
            # print(f"Drawing layer: {layer.name}")
            lines, labels = layer.draw(self.ax, self.shared_data)
            self.all_lines.extend(lines)
            self.all_labels.extend(labels)
        
        ''' Enforce full audio duration on x-axis '''
        if 'audio_duration' in self.shared_data:
            audio_duration = self.shared_data['audio_duration']
            self.ax.set_xlim(0, audio_duration)
            
            ''' Also set secondary axis if it exists '''
            if 'ax2' in self.shared_data:
                self.shared_data['ax2'].set_xlim(0, audio_duration)
        
        if self.all_lines:
            self.ax.legend(self.all_lines, self.all_labels, loc='upper right')
        
        plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1)
        
        ''' Store axes for SVG conversion '''
        self.shared_data["ax"] = self.ax
        if "ax2" in self.shared_data:
            self.shared_data["ax2_exists"] = True
        
        return self.fig, self.ax
    
    def show(self):
        plt.show()

    def getSVG_Score_Height(self, svg_score: str, maps_json_file: str) -> Optional[float]:
        '''
        Extract the height of the score from the SVG file using the maps JSON for reference.
        
        Args:
            svg_score: Path to the SVG score file
            maps_json_file: Path to the maps JSON file containing note positions
        '''
    def get_SVG_Root_Dimensions(self, svg_warped_score: str, print_output: bool = False):

        svg_tree = ET.parse(svg_warped_score)
        root = svg_tree.getroot()

        width_str = root.get('width')
        height_str = root.get('height')

        if width_str and height_str:
            width = float(width_str.replace('px', ''))
            height = float(height_str.replace('px', ''))
            if print_output:
                print(f"✓ SVG root dimensions: width={width}, height={height}")
            return width, height
        else:
            if print_output:
                print("✗ SVG root does not have explicit width and height attributes")
            return None
    
    def get_timeAxis_attributes(self, svg_warped_score: str, print_output: bool = False):
        '''
        Extract the total timeline time from the warped score SVG.
        '''
        try:
            svg_tree = ET.parse(svg_warped_score)
            root = svg_tree.getroot()
            
            ''' Handle XML namespaces in SVG files '''
            namespace = {'svg': 'http://www.w3.org/2000/svg'}
            
            ''' Find all g elements and locate the one with class='timeAxis' '''
            time_axis_group = None
            ''' Try with namespace first '''
            for group in root.iter('{http://www.w3.org/2000/svg}g'):
                if group.get('class') == 'timeAxis':
                    time_axis_group = group
                    break
            
            ''' If not found, try without namespace '''
            if time_axis_group is None:
                for group in root.iter('g'):
                    if group.get('class') == 'timeAxis':
                        time_axis_group = group
                        break
            
            if time_axis_group is None:
                if print_output:
                    print("✗ Error: timeAxis group not found")
                return None
            
            ''' Get the last child element from the timeAxis group '''
            children = list(time_axis_group)
            
            if not children:
                if print_output:
                    print("✗ Error: No child elements found in timeAxis group")
                return None
            
            ''' Get the last child element '''
            last_element = children[-1]
            last_text_content = last_element.text
            
            if last_text_content is None:
                if print_output:
                    print("✗ Error: Last text element is empty")
                return None
            
            ''' Convert to numeric value '''
            timeline_time = float(last_text_content)
            
            ''' Convert to int if it's a whole number '''
            if timeline_time.is_integer():
                timeline_time = int(timeline_time)
            
            if print_output:
                print(f"✓ Total timeline time: {timeline_time}")
            
            ''' Look for timeline start and end in pixels '''
            ''' Get the SECOND element of the timeAxis group, it should be a <line> element with x1 and x2 values '''
            if len(children) < 2:
                if print_output:
                    print("✗ Error: Expected at least 2 elements in timeAxis group")
                return None
            
            second_element = children[1]
            x1_str = second_element.get('x1')
            x2_str = second_element.get('x2')
            
            if x1_str is None or x2_str is None:
                if print_output:
                    print("✗ Error: Could not find x1 and x2 attributes in second element")
                return None
            
            x1 = float(x1_str)
            x2 = float(x2_str)
            
            timeline_lengthPx= x2 - x1

            if print_output:
                print(f"✓ Timeline x1: {x1}, x2: {x2}")

            return timeline_time, timeline_lengthPx
            
        except Exception as e:
            if print_output:
                print(f"✗ Error extracting timeline time: {e}")
            return None

    def get_Layers_WidthHeight(self, svg_warped_score: str, print_output: bool = False):
        '''
        Get the width and height for the layers based on the warped score SVG dimensions.
        Returns:
            Tuple of (width, height) in pixels, or None if error
        '''

        layersHeight = self.get_SVG_Root_Dimensions(svg_warped_score, print_output)[1]
        totalWidth = self.get_timeAxis_attributes(svg_warped_score, print_output)[1]
        
        totalTimelineTime = self.get_timeAxis_attributes(svg_warped_score, print_output)[0]
        
        audioDuration = self.shared_data.get('audio_duration', None)

        layersWidth = (totalWidth / totalTimelineTime) * audioDuration

        if print_output==True:
            print(f"✓ Layers width: {layersWidth}, Layers height: {layersHeight}")
            
        return layersWidth, layersHeight       
    
    def turn_to_SVG(self, filename: str, svg_warped_score: str, plot_size: Optional[Tuple[int, int]] = None, show_axes: bool = False, print_output: bool = False):
        '''
        Convert all layers to a vector-based SVG with each layer as a separate group.
        
        Args:
            filename: Output SVG filename
            svg_warped_score: Path to the warped score SVG file
            plot_size: Tuple of (width, height) in pixels
            show_axes: If True, layers that support it will overlay axis labels as SVG elements on the image.
            print_output: Whether to print status messages
        
        Returns:
            filename if successful, False otherwise
        '''
        if "ax" not in self.shared_data:
            print("✗ Error: No axes found. Call draw() first.")
            return False
        
        if plot_size is None:
            width_px =self.get_Layers_WidthHeight(svg_warped_score, print_output)[0]
            height_px =self.get_Layers_WidthHeight(svg_warped_score, print_output)[1]
        else:
            width_px, height_px = plot_size

        ax = self.shared_data["ax"]

        ''' Get axis limits for coordinate conversion '''
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        logit_axis = self.shared_data.get("ax2")
        logit_y_min, logit_y_max = logit_axis.get_ylim() if logit_axis is not None else (None, None)
        
        ''' Store axis info for layer SVG conversion '''
        self.shared_data["svg_context"] = {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "logit_y_min": logit_y_min,
            "logit_y_max": logit_y_max,
            "width_px": width_px,
            "height_px": height_px,
            "show_axes": show_axes,
            "logit_axes_added": False,
        }
        
        svg_groups = []
        skipped_layers = []
        
        ''' Collect SVG groups from each layer '''
        for layer in self.layers:
            svg_content = layer.to_svg_group(self.shared_data)
            if svg_content is None:
                skipped_layers.append(layer.name)
            else:
                svg_groups.append(svg_content)
        
        ''' Print warnings for skipped layers '''
        if skipped_layers:
            for layer_name in skipped_layers:
                print(f"⚠ Warning: Layer '{layer_name}' doesn't support SVG export, skipping")
        
        ''' Build SVG markup '''
        svg_content = '\n'.join(svg_groups)
        svg_markup = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width_px}px"
     height="{height_px}px"
     viewBox="0 0 {width_px} {height_px}"
     overflow="visible">
  <!-- Beat Spec Visual Layers -->
{svg_content}
</svg>'''
        
        try:
            with open(filename, 'w') as f:
                f.write(svg_markup)
            
            if print_output:
                print(f"✅ SVG saved successfully: {filename} (~{width_px}x{height_px}px)")
            return filename
        except Exception as e:
            print(f"✗ Error saving SVG: {e}")
            return False
    
    def turn_to_PNG(self, filename: str, svg_warped_score: str, plot_size: Optional[Tuple[int, int]] = None, dpi: int = 150, print_output: bool = False) -> bool:
        """
        Save visualization as PNG with exact pixel dimensions and no padding/axis.
        
        Args:
            filename: Output PNG filename
            svg_warped_score: Path to the warped score SVG file
            plot_size: Tuple of (width, height) in pixels (exact)
            dpi: Dots per inch (default 150). PNG will be exactly width × height pixels.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if plot_size is None:
            width_px =self.get_Layers_WidthHeight(svg_warped_score, print_output)[0]
            height_px =self.get_Layers_WidthHeight(svg_warped_score, print_output)[1]
        else:
            width_px, height_px = plot_size
        
        # Convert pixels to inches using provided DPI for figure creation
        figsize_inches = (width_px / dpi, height_px / dpi)
        
        # Create new figure with specified DPI
        fig_export, ax_export = plt.subplots(figsize=figsize_inches, dpi=dpi)
        
        # Redraw all layers on the new figure without axis/padding
        for layer in self.layers:
            layer.draw(ax_export, self.shared_data)
        
        # Remove axis completely and remove title
        ax_export.axis('off')
        ax_export.set_title('')

        # Remove all margins and padding
        fig_export.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Define output directories (relative to parent folders from script location)
        output_dirs = [
            os.path.join(os.path.dirname(__file__), '..', 'input_files'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'output')
        ]
        
        # Create directories if they don't exist and save to both locations
        for output_dir in output_dirs:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            
            # Save figure as PNG with exact dimensions
            fig_export.savefig(
                output_path,
                format='png',
                dpi=dpi,
                pad_inches=0,
                facecolor='white'
            )
        
        # Clean up the temporary figure
        plt.close(fig_export)

        if print_output==True:
                print(f"✅ PNG saved successfully: {filename} ---> ({width_px}x{height_px}px @ {dpi}dpi)")
        return output_path
        
    def combine_layers_with_score(self, filename: str, original_score: str, layers_svg: str, maps_file: str, PNG_layer: Optional[str] = None, show_score: bool = True, print_output: bool = False):
        ''' Combine PNG (optional) and layers SVG with score SVG by creating Layers group with scaled PNG and layer groups '''
        try:
            ''' Get timeAxis bounds to size the PNG and Layers group properly '''
            from .warp_score import Warp_Score
            ws = Warp_Score()
            timeAxis_bounds = ws.get_timeAxis_bounds(original_score, print_output)
            if timeAxis_bounds is None:
                print("✗ Could not get timeAxis bounds")
                return False
            
            timeAxis_first_x, timeAxis_last_x, timeline_width = timeAxis_bounds

            ''' Parse original score SVG '''
            svg_tree = ET.parse(original_score)
            composite_svg = svg_tree.getroot()

            ''' Register namespace '''
            svg_ns = 'http://www.w3.org/2000/svg'
            ET.register_namespace('', svg_ns)

            ''' Hide all existing score children when show_score is False '''
            if not show_score:
                for child in list(composite_svg):
                    child.set('display', 'none')

            ns = {'svg': 'http://www.w3.org/2000/svg'}
            
            ''' Parse layers SVG to extract layer groups '''
            layers_tree = ET.parse(layers_svg)
            layers_root = layers_tree.getroot()
            
            ''' Extract only direct child layer groups from the layers SVG '''
            layer_groups = layers_root.findall("./svg:g", ns)
            
            ''' If not found with namespace, try without '''
            if len(layer_groups) == 0:
                layer_groups = layers_root.findall("./g")
            
            ''' Create new Layers group positioned at timeAxis '''
            layers_group = ET.Element('g', {
                'class': 'Layers',
                'transform': f'translate({timeAxis_first_x}, 0)',
                'width': str(timeline_width),
            })

            ''' Optionally embed PNG as base64 and add to Layers group '''
            if PNG_layer is not None:
                png_img = Image.open(PNG_layer)
                png_width, png_height = png_img.size

                target_png_height = 110
                scale_y = target_png_height / png_height

                layers_group.set('height', str(int(png_height * scale_y)))

                png_buffer = io.BytesIO()
                png_img.save(png_buffer, format='PNG')
                png_base64 = base64.b64encode(png_buffer.getvalue()).decode('utf-8')

                png_image_elem = ET.Element('image', {
                    'x': '0',
                    'y': '0',
                    # 'width': str(timeline_width),
                    # 'height': str(int(png_height * scale_y)),
                    'href': f'data:image/png;base64,{png_base64}'
                })
                layers_group.append(png_image_elem)
            
            ''' Add layer groups from layers SVG to the new Layers group '''
            if len(layer_groups) > 0:
                for layer_group in layer_groups:
                    ''' Deep copy the layer group to avoid modifying the original tree '''
                    layer_copy = ET.fromstring(ET.tostring(layer_group))
                    layers_group.append(layer_copy)
                if print_output:
                    print(f"✓ Added {len(layer_groups)} layer groups to the new Layers group")
            else:
                if print_output:
                    print("⚠ Warning: No layer groups found in layers SVG")
            
            ''' Find parent of timeAxis to insert Layers group before it '''
            parent_map = {c: p for p in composite_svg.iter() for c in p}
            time_axis = composite_svg.find(".//svg:g[@class='timeAxis']", ns)
            if time_axis is None:
                time_axis = composite_svg.find(".//g[@class='timeAxis']")
            
            if time_axis is not None:
                time_axis_parent = parent_map.get(time_axis, composite_svg)
                ''' Insert Layers group at the first position (before timeAxis) '''
                time_axis_parent.insert(0, layers_group)
            else:
                ''' If no timeAxis found, append to root '''
                composite_svg.append(layers_group)
            
            ''' Save modified SVG to output folder '''
            root_dir = Path(__file__).parent.parent.parent
            output_dir = root_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / filename)
            
            svg_tree.write(output_path, encoding='UTF-8', xml_declaration=True)
            
            if print_output:
                print(f"✓ Layers group with PNG and layer groups added and saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"✗ Error combining PNG and layers with score: {e}")
            return False

    def create_final_SVG(self, width: int, height: int, svg_layers: List[Tuple[str, float]], output_file: str, background_color: str = '#ffffff', print_output: bool = False):
        '''
        Combine multiple SVG files into a single final SVG with each as a separate nested SVG with y-offsets.
        Preserves each SVG's coordinate system and root element attributes (including ID).
        
        Args:
            width: Width in pixels for the final SVG
            height: Height in pixels for the final SVG
            svg_layers: List of tuples (svg_file_path, y_offset) for each layer to combine
            output_file: Output SVG filename (saves to /output directory)
            background_color: Color for the background rectangle
            print_output: Whether to print status messages
        
        Returns:
            str: Path to output file if successful, False otherwise
        '''
        try:
            ''' Build SVG content by combining nested SVGs with preserved coordinate systems '''
            visual_groups_markup = []

            '''
            Try to find a reference timeAxis (x start and timeline width) from any
            of the provided SVG layers. If found, use it to align SVGs that do
            not contain the score/timeAxis so their content shares the same
            horizontal position and length.
            '''
            ref_timeAxis_x = None
            ref_timeline_width = None
            for svg_path, _ in svg_layers:
                try:
                    svg_tree_ref = ET.parse(svg_path)
                    svg_root_ref = svg_tree_ref.getroot()
                    ns = {'svg': 'http://www.w3.org/2000/svg'}
                    time_axis = svg_root_ref.find(".//svg:g[@class='timeAxis']", ns)
                    if time_axis is None:
                        time_axis = svg_root_ref.find(".//g[@class='timeAxis']")
                    if time_axis is not None:
                        children = list(time_axis)
                        if len(children) >= 2:
                            second_element = children[1]
                            x1_str = second_element.get('x1')
                            x2_str = second_element.get('x2')
                            if x1_str is not None and x2_str is not None:
                                ref_timeAxis_x = float(x1_str)
                                ref_timeline_width = float(x2_str) - float(x1_str)
                                break
                except Exception:
                    continue
            
            ''' Process each visual layer '''
            for visual_index, (svg_path, y_offset) in enumerate(svg_layers):
                try:
                    visual_path = Path(svg_path)
                    is_png = visual_path.suffix.lower() == '.png'

                    root_id = ''
                    id_attr = ''
                    viewBox_attr = ''
                    width_attr = ''
                    height_attr = ''
                    nested_x = '0'
                    contains_timeAxis = False

                    if is_png:
                        ''' Load PNG metadata and embed the image as base64 inside a nested SVG '''
                        png_img = Image.open(svg_path)
                        png_width, png_height = png_img.size

                        png_buffer = io.BytesIO()
                        png_img.save(png_buffer, format='PNG')
                        png_base64 = base64.b64encode(png_buffer.getvalue()).decode('utf-8')

                        viewBox_attr = f' viewBox="0 0 {png_width} {png_height}"'
                        width_attr = f' width="{png_width}"'
                        height_attr = f' height="{png_height}"'

                        ''' PNGs created by turn_to_PNG already have the target layer size '''
                        inner_content = f'''<image x="0" y="0" width="{png_width}" height="{png_height}" href="data:image/png;base64,{png_base64}" />'''

                        if ref_timeAxis_x is not None:
                            nested_x = str(int(ref_timeAxis_x))
                    else:
                        ''' Parse SVG file to extract its properties and content '''
                        svg_tree = ET.parse(svg_path)
                        svg_root = svg_tree.getroot()

                        ''' Extract SVG root's attributes to preserve the coordinate system '''
                        root_id = svg_root.get('id', '')
                        svg_viewBox = svg_root.get('viewBox', '')
                        svg_width = svg_root.get('width', '')
                        svg_height = svg_root.get('height', '')

                        ''' Build attribute strings '''
                        id_attr = f' id="{root_id}"' if root_id else ''
                        viewBox_attr = f' viewBox="{svg_viewBox}"' if svg_viewBox else ''
                        width_attr = f' width="{svg_width}"' if svg_width else ''
                        height_attr = f' height="{svg_height}"' if svg_height else ''

                        ''' Extract all children from SVG root and convert to string '''
                        children_markup = []
                        for child in svg_root:
                            ''' Serialize each child element as string to preserve it exactly '''
                            child_str = ET.tostring(child, encoding='unicode')
                            children_markup.append(child_str)

                        inner_content = '\n'.join(children_markup)

                        ''' Determine if this SVG contains its own timeAxis (score) '''
                        ns = {'svg': 'http://www.w3.org/2000/svg'}
                        try:
                            ta = svg_root.find(".//svg:g[@class='timeAxis']", ns)
                            if ta is None:
                                ta = svg_root.find(".//g[@class='timeAxis']")
                            contains_timeAxis = ta is not None
                        except Exception:
                            contains_timeAxis = False

                        ''' If the nested SVG does not contain the score/timeAxis, position
                        it using the reference timeAxis x offset so it aligns with other
                        layers that do include the score. Also ensure it has a width
                        when missing by using the reference timeline width. '''
                        if not contains_timeAxis and ref_timeAxis_x is not None:
                            nested_x = str(int(ref_timeAxis_x))
                            if not svg_width and ref_timeline_width is not None:
                                width_attr = f' width="{int(ref_timeline_width)}"'

                    nested_svg = f'''  <svg class="visualization_{visual_index}"{id_attr}{viewBox_attr}{width_attr}{height_attr} x="{nested_x}" y="{y_offset}" overflow="visible" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
{inner_content}
  </svg>'''
                    
                    visual_groups_markup.append(nested_svg)
                    
                    if print_output:
                        print(f"✓ Added layer {visual_index}: {Path(svg_path).name} at y_offset={y_offset}" + (f" (id={root_id})" if root_id else ""))
                    
                except Exception as e:
                    print(f"✗ Error processing SVG layer {visual_index} ({svg_path}): {e}")
                    continue
            
            ''' Build final SVG markup '''
            svg_ns = 'http://www.w3.org/2000/svg'
            visual_groups_str = '\n'.join(visual_groups_markup)
            
            final_svg_markup = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="{svg_ns}"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width}px"
     height="{height}px"
     viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="100%" height="100%" fill="{background_color}" />
{visual_groups_str}
</svg>'''
            
            ''' Save to output directory '''
            root_dir = Path(__file__).parent.parent.parent
            output_dir = root_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / output_file)
            
            ''' Write the final SVG '''
            with open(output_path, 'w', encoding='UTF-8') as f:
                f.write(final_svg_markup)
            
            if print_output:
                print(f"✅ Final SVG created: {output_path} ({width}x{height}px)")
            
            return output_path
            
        except Exception as e:
            print(f"✗ Error creating final SVG: {e}")
            return False