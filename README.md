# phallicphin

A novelty surf fin with a standard **Futures** base, traced from a photo of a printed original.
Print-ready for a Bambu Lab P2S in PETG.

![preview](preview.png)

| file | what |
|---|---|
| `phallicphin.stl` | the fin, 178 x 133 x 7.2 mm, watertight |
| `phallicphin_P2S_PETG.3mf` | Bambu Studio project: P2S, PETG, 0.16 mm, 5 walls, 40% gyroid, no supports (~2h14m, ~59 g) |

## Design

- **Base:** solid Futures tab, 113 x 13.2 x 7.2 mm, front V-notch and rear set-screw slot (`tab_profile_mm.csv`).
  If it's tight in your box, set `TAB_W = 7.0` in `fin.py` and rebuild.
- **Flat-backed:** the Z=0 face is flat and all the shaping is on top, so it prints lying down with no
  supports and the layers run up the fin (strong at the root).
- **Shape:** 7.2 mm plateau with a 12 mm rounded edge band, shaft tapering to 5 mm, a domed 7.2 mm head
  with a rim, 0.8 mm minimum edge thickness.

## Rebuild

```sh
python3 -m venv .venv
.venv/bin/pip install trimesh manifold3d numpy scipy shapely scikit-image opencv-python-headless networkx matplotlib

.venv/bin/python trace_outline.py ref/photo.png outline_mm.csv      # photo -> outline (scale: tab = 113 mm)
.venv/bin/python fin.py phallicphin.stl                              # params at the top of fin.py
.venv/bin/python make_3mf.py phallicphin.stl template_P2S_PETG.3mf phallicphin_P2S_PETG.3mf print_fin.json
.venv/bin/python render_preview.py phallicphin.stl ref/photo.png preview.png
```
