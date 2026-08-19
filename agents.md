## Requirements
- Python 3.12.3+

## Architecture
LayerIt uses a modular Layer-based system for composable visualizations.
All visualization components inherit from the `visualization_system.Layer` class.
New visualizations should follow this pattern.

Each layer receives its inputs through `load_data(**kwargs)` — files are passed directly as kwargs to `load_all_layers()` (e.g., `audio_path`, `maps_file`, `beat_file`). There is no `Load_Files` helper class.

## Layer implementation:
- `load_data(**kwargs)` — Load and validate data, return bool
- `draw(ax, shared_data)` — Draw visualization, return (lines, labels)

## Other Notes
- Dont remove lines that are comments or commented out segments of the code with `#`.
- Comments that you add shall be always with ''' ''' (either inline or in block), those comments you can remove if you find it fit to do so.
- Never add comments with # only with add coments with '''
- Dont add Debbuging features (like unecessary print()'s ), unless when requested.
- Prioritize using depandancies already in requirements.txt, if you need new dependancies remeber to add them to requirements.txt and tell me before implementing.


