"""Package an STL into a Bambu P2S / PETG project, centered flat on the plate (Z=0 face down).
usage: make_3mf.py <stl> <template.3mf> <out.3mf> [print_overrides.json] [filament_overrides.json]"""
import json, math, re, sys, uuid, zipfile
import numpy as np
import trimesh

stl, tpl, out = sys.argv[1:4]
name = stl.rsplit("/", 1)[-1].rsplit(".", 1)[0]
m = trimesh.load(stl)
m.apply_translation(-m.bounding_box.centroid)  # object-local coords centered, like Bambu does
half_z = m.extents[2] / 2

verts = "".join(f'<vertex x="{x:.5f}" y="{y:.5f}" z="{z:.5f}"/>' for x, y, z in m.vertices)
tris = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in m.faces)
obj_model = f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p">
 <metadata name="BambuStudio:3mfVersion">1</metadata>
 <resources>
  <object id="1" p:UUID="{uuid.uuid4()}" type="model">
   <mesh><vertices>{verts}</vertices><triangles>{tris}</triangles></mesh>
  </object>
 </resources>
 <build/>
</model>'''

a = math.radians(0)
c, s = math.cos(a), math.sin(a)
rot = f"{c:.6f} {s:.6f} 0 {-s:.6f} {c:.6f} 0 0 0 1"
main_model = f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p">
 <metadata name="Application">BambuStudio-02.08.02.61</metadata>
 <metadata name="BambuStudio:3mfVersion">1</metadata>
 <metadata name="Title">{name}</metadata>
 <resources>
  <object id="2" p:UUID="{uuid.uuid4()}" type="model">
   <components>
    <component p:path="/3D/Objects/object_1.model" objectid="1" p:UUID="{uuid.uuid4()}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>
   </components>
  </object>
 </resources>
 <build p:UUID="{uuid.uuid4()}">
  <item objectid="2" p:UUID="{uuid.uuid4()}" transform="{rot} 128 128 {half_z:.4f}" printable="1"/>
 </build>
</model>'''

model_settings = f'''<?xml version="1.0" encoding="UTF-8"?>
<config>
  <object id="2">
    <metadata key="name" value="{name}"/>
    <metadata key="extruder" value="1"/>
    <metadata face_count="{len(m.faces)}"/>
    <part id="1" subtype="normal_part">
      <metadata key="name" value="{name}"/>
      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>
      <mesh_stat face_count="{len(m.faces)}" edges_fixed="0" degenerate_facets="0" facets_removed="0" facets_reversed="0" backwards_edges="0"/>
    </part>
  </object>
  <plate>
    <metadata key="plater_id" value="1"/>
    <metadata key="plater_name" value=""/>
    <metadata key="locked" value="false"/>
    <model_instance>
      <metadata key="object_id" value="2"/>
      <metadata key="instance_id" value="0"/>
      <metadata key="identify_id" value="47"/>
    </model_instance>
  </plate>
  <assemble>
  </assemble>
</config>'''

rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/Objects/object_1.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>'''

zin = zipfile.ZipFile(tpl)
ps = json.loads(zin.read("Metadata/project_settings.config"))
ps["printable_area"] = ["0x0", "256x0", "256x256", "0x256"]
changes = {"enable_support": "0"}
ps.update(changes)
extra_print = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else {}
extra_fil = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else {}
ps.update(extra_print)
ps.update(extra_fil)
dsts = ps["different_settings_to_system"]  # [print, filament, printer]
for i, keys in ((0, set(extra_print)), (1, set(extra_fil))):
    dsts[i] = ";".join(sorted(set(filter(None, dsts[i].split(";"))) | keys))

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("[Content_Types].xml", "_rels/.rels", "Metadata/slice_info.config"):
        z.writestr(name, zin.read(name))
    z.writestr("3D/3dmodel.model", main_model)
    z.writestr("3D/_rels/3dmodel.model.rels", rels)
    z.writestr("3D/Objects/object_1.model", obj_model)
    z.writestr("Metadata/model_settings.config", model_settings)
    z.writestr("Metadata/project_settings.config", json.dumps(ps, indent=4))
print("wrote", out)
