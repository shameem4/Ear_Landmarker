"""Render a head mesh with physically-based skin, headless, via Blender Cycles.

Run as:  blender -b -P scripts/blender_skin_render.py -- <args.json>

WHY A SECOND RENDERER. scripts/skin.py fakes skin in Open3D by baking ambient
occlusion and a reddening term into vertex colours. It cannot do subsurface
scattering (light entering the skin, scattering, and leaving somewhere else) or
specular highlights, both of which a real ear shows -- the helix rim is almost
always the brightest thing in an ear photograph. Cycles does both properly. This
exists to measure whether that costs or buys anything measurable; see
RESULTS.md. Open3D remains the default because it is ~100x faster.

The camera matches scripts/render3d_ears: positioned along +Y from the ear,
looking at it, +Z up, 50 degree vertical FOV, mid-grey background.
"""
import json
import math
import sys

import bpy


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def set_socket(node, names, value):
    """Set the first socket that exists. Principled BSDF renamed sockets in 4.x."""
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return True
    return False


def skin_material(base_rgb, roughness=0.45, sss_weight=0.35, sss_scale=0.012,
                  freckles=0.0, freckle_scale=140.0, curvature_spec=0.0):
    mat = bpy.data.materials.new("skin")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    set_socket(bsdf, ["Base Color"], (*base_rgb, 1.0))
    if freckles > 0:
        # High-frequency noise, thresholded hard so it reads as discrete spots
        # rather than mottling, then mixed toward a darker, redder tone.
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = freckle_scale
        tex.inputs["Detail"].default_value = 2.0
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.interpolation = "CONSTANT"
        ramp.color_ramp.elements[0].position = 0.60
        ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
        ramp.color_ramp.elements[1].position = 0.66
        ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
        mix = nt.nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MIX"
        mix.inputs["Color1"].default_value = (*base_rgb, 1.0)
        spot = tuple(c * 0.55 for c in base_rgb)
        mix.inputs["Color2"].default_value = (spot[0] * 1.15, spot[1], spot[2], 1.0)
        nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], mix.inputs["Fac"])
        nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
        mix.inputs["Fac"].default_value = freckles
    set_socket(bsdf, ["Roughness"], roughness)
    if curvature_spec > 0:
        # Measured on 1,500 training crops, the brightest 1% of ear pixels fall on
        # the antihelix (36.7%), concha (35.6%) and helix rim (31.1%) and avoid the
        # lobe (10.6%). Those are the taut, convex structures. Pointiness is
        # Blender's per-vertex convexity, so driving roughness DOWN where it is
        # high puts the sheen on the ridges and leaves the lobe matte -- which a
        # single global roughness value cannot do, and global roughness was the
        # flattest factor measured (spread 0.003) for exactly that reason.
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.42      # concave -> matte
        ramp.color_ramp.elements[0].color = (1, 1, 1, 1)
        ramp.color_ramp.elements[1].position = 0.58      # convex  -> glossy
        ramp.color_ramp.elements[1].color = (0, 0, 0, 1)
        mapr = nt.nodes.new("ShaderNodeMapRange")
        mapr.inputs["From Min"].default_value = 0.0
        mapr.inputs["From Max"].default_value = 1.0
        mapr.inputs["To Min"].default_value = max(0.05, roughness - curvature_spec * 0.35)
        mapr.inputs["To Max"].default_value = min(0.95, roughness + curvature_spec * 0.25)
        nt.links.new(geo.outputs["Pointiness"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], mapr.inputs["Value"])
        nt.links.new(mapr.outputs["Result"], bsdf.inputs["Roughness"])
    # Subsurface: light scatters furthest in red, which is what makes thin skin
    # (helix rim, lobe) glow warm and recesses redden rather than simply darken.
    set_socket(bsdf, ["Subsurface Weight", "Subsurface"], sss_weight)
    set_socket(bsdf, ["Subsurface Radius"], (1.0, 0.30, 0.17))
    set_socket(bsdf, ["Subsurface Scale"], sss_scale)
    set_socket(bsdf, ["Specular IOR Level", "Specular"], 0.5)
    set_socket(bsdf, ["IOR"], 1.4)
    return mat


def main():
    args = json.load(open(sys.argv[sys.argv.index("--") + 1]))
    clear()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.get("samples", 96)
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = args.get("size", 512)
    scene.render.film_transparent = False

    # Mid-grey world, matching the Open3D background so crops are comparable.
    world = bpy.data.worlds.new("w")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = args.get("ambient", 0.6)

    bpy.ops.wm.ply_import(filepath=args["mesh"])
    obj = bpy.context.selected_objects[0]
    obj.data.materials.clear()
    obj.data.materials.append(skin_material(args["tone"],
                                            roughness=args.get("roughness", 0.45),
                                            sss_weight=args.get("sss", 0.35),
                                            sss_scale=args.get("sss_scale", 0.012),
                                            freckles=args.get("freckles", 0.0),
                                            curvature_spec=args.get("curvature_spec", 0.0)))
    for p in obj.data.polygons:
        p.use_smooth = True

    ear = args["ear"]
    dist = args["dist"]
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens_unit = "FOV"
    # sensor_fit MUST be set before angle_y, and must be VERTICAL. Blender's
    # default AUTO fit measures the angle across the LARGER sensor dimension --
    # the 36mm width, not the 24mm height -- so on a square render setting
    # angle_y to 50 deg produced an effective 70 deg field. Measured: markers
    # landed at 0.665x their projected offset from centre, which is exactly
    # tan(25)/tan(35). That silently shrinks everything in frame relative to
    # what render3d_ears.project() predicts.
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle_y = math.radians(50.0)
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (ear[0], ear[1] + dist, ear[2])
    # look down -Y with +Z up
    cam.rotation_euler = (math.radians(90.0), 0.0, math.radians(180.0))
    scene.camera = cam

    # Key light offset from the camera so the ear's relief casts readable shadow,
    # plus a soft fill: a single frontal light flattens exactly the structure the
    # landmarker needs.
    key = bpy.data.lights.new("key", type="AREA")
    key.energy = args.get("key_energy", 60.0)
    # Light SIZE sets shadow softness, and at the default 1.2 the source is wider
    # than the head: shadows wash out and the ear's relief goes flat. It was never
    # part of the azimuth/energy sweep, so it is exposed rather than tuned here.
    key.size = dist * args.get("key_size", 1.2)
    ko = bpy.data.objects.new("key", key)
    # Azimuth 0 puts the key beside the camera; negative swings it toward the
    # front of the face, positive behind the head, changing which side of the
    # helix is lit and how deep the concha shadow runs.
    az = math.radians(args.get("key_azimuth", -40.0))
    el = math.radians(args.get("key_elevation", 35.0))
    ko.location = (ear[0] + dist * 1.1 * math.sin(az) * math.cos(el),
                   ear[1] + dist * 1.1 * math.cos(az) * math.cos(el),
                   ear[2] + dist * 1.1 * math.sin(el))
    scene.collection.objects.link(ko)
    con = ko.constraints.new("TRACK_TO")
    tgt = bpy.data.objects.new("tgt", None)
    tgt.location = ear
    scene.collection.objects.link(tgt)
    con.target = tgt

    fill = bpy.data.lights.new("fill", type="AREA")
    fill.energy = args.get("fill_energy", 18.0)
    fill.size = dist * 2.0
    fo = bpy.data.objects.new("fill", fill)
    fo.location = (ear[0] + dist * 0.9, ear[1] + dist * 0.8, ear[2] - dist * 0.3)
    scene.collection.objects.link(fo)
    fo.constraints.new("TRACK_TO").target = tgt

    scene.render.filepath = args["out"]
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)


main()
