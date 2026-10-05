"""Give a colourless scan mesh a plausible skin appearance.

WHY: HUTUBS and SONICOM ship geometry with no usable colour -- HUTUBS has a flat
0.588 grey placeholder, SONICOM is STL which carries no colour channel at all.
Rendered as clay they sit far outside the training distribution, and the cost is
not cosmetic: the ear DETECTOR boxes a whole head as an ear on a grey render, at
0.62 confidence. Anything measured on clay is measuring domain gap.

WHAT MAKES IT READ AS SKIN, in rough order of importance:
  1. ambient occlusion -- the concha is a bowl and the crus a trench. Without
     darkening in cavities an ear renders as a flat blob, which is exactly what
     the clay versions look like. This is the one that matters.
  2. subsurface reddening -- occluded skin goes red before it goes dark, because
     the light that survives there has been scattered through tissue. AO that
     only darkens looks like dirt.
  3. a base tone drawn per subject, spanning the range of human skin rather than
     one "default" colour.
  4. low-frequency blotching and fine grain, so the surface is not uniform.

The tones below are sRGB values spanning Fitzpatrick I-VI. They are a coarse
sample of human skin colour, not a calibrated reference.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

# Representative sRGB skin tones, light to deep: Fitzpatrick I-VI. This is the
# pool a RANDOM tone is drawn from, so it stays inside plausible human skin --
# these colours go into training images, where the appearance distribution has to
# match photographs.
SKIN_TONES = np.array([
    [0.96, 0.84, 0.76], [0.93, 0.79, 0.69], [0.88, 0.72, 0.60],
    [0.80, 0.63, 0.50], [0.71, 0.54, 0.42], [0.60, 0.44, 0.34],
    [0.48, 0.34, 0.26], [0.36, 0.25, 0.19], [0.27, 0.18, 0.14],
])

# Deeper than Fitzpatrick VI, i.e. past real skin. NOT in the random pool, and
# not for training images. They exist because the landmarker reads the ear more
# cleanly on them: judged by eye over tones 0-11 on identical geometry, 10 and 11
# gave the best landmark placement. The measured picture is consistent but
# weaker -- detector confidence plateaus from tone 6 (0.96-0.97 through 11) while
# the landmarks keep shifting, so confidence alone could not have chosen these.
# Index these explicitly, as LABEL_TONE does.
EXTRA_TONES = np.array([
    [0.20, 0.13, 0.10], [0.14, 0.09, 0.07], [0.09, 0.06, 0.045], [0.05, 0.035, 0.025],
])
ALL_TONES = np.vstack([SKIN_TONES, EXTRA_TONES])

# The tone to render at when the point of the render is to LABEL it.
LABEL_TONE = 10


def ambient_occlusion(mesh, n_rays: int = 24, radius_frac: float = 0.04,
                      seed: int = 0) -> np.ndarray:
    """Per-vertex accessibility in [0, 1]: 1 = open, 0 = deep in a cavity.

    Casts a cosine-weighted hemisphere of short rays about each vertex normal and
    measures how many escape. `radius_frac` is the ray length as a fraction of
    the mesh's largest extent -- long enough to feel the concha bowl, short
    enough that the whole head does not shadow itself.
    """
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    if len(N) == 0:
        mesh.compute_vertex_normals()
        N = np.asarray(mesh.vertex_normals)
    extent = np.ptp(V, axis=0).max()
    rlen = extent * radius_frac

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    rng = np.random.default_rng(seed)
    # Build an orthonormal frame per vertex to sample the hemisphere in.
    helper = np.tile(np.array([0.0, 0.0, 1.0]), (len(V), 1))
    flip = np.abs(N[:, 2]) > 0.9
    helper[flip] = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(N, helper); t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-9)
    t2 = np.cross(N, t1)

    open_count = np.zeros(len(V))
    origins = V + N * (extent * 1e-4)          # lift off the surface
    for _ in range(n_rays):
        u1, u2 = rng.random(len(V)), rng.random(len(V))
        r, theta = np.sqrt(u1), 2 * np.pi * u2          # cosine-weighted
        d = (t1 * (r * np.cos(theta))[:, None]
             + t2 * (r * np.sin(theta))[:, None]
             + N * np.sqrt(np.maximum(1 - u1, 0))[:, None])
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        hit = scene.cast_rays(o3d.core.Tensor(
            np.hstack([origins, d]).astype(np.float32)))["t_hit"].numpy()
        open_count += ~(np.isfinite(hit) & (hit < rlen))
    return open_count / n_rays


def apply_skin(mesh, seed: int = 0, tone: int | None = None,
               ao_strength: float = 0.75, n_rays: int = 24):
    """Colour a mesh in place with a synthetic skin appearance. Returns it."""
    rng = np.random.default_rng(seed)
    V = np.asarray(mesh.vertices)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # An explicit index may reach the deep tones; a random draw may not, so
    # generated training images stay inside plausible human skin.
    base = (SKIN_TONES[rng.integers(len(SKIN_TONES))] if tone is None
            else ALL_TONES[tone]).copy()
    base *= rng.uniform(0.94, 1.06)                       # per-subject exposure

    acc = ambient_occlusion(mesh, n_rays=n_rays, seed=seed)
    shade = (1.0 - ao_strength) + ao_strength * acc[:, None]

    # Occluded skin reddens before it darkens: hold red back from the shading.
    warm = np.ones((len(V), 3))
    warm[:, 0] = 1.0 + (1.0 - acc) * 0.22
    warm[:, 1] = 1.0 - (1.0 - acc) * 0.06
    warm[:, 2] = 1.0 - (1.0 - acc) * 0.10

    ext = np.ptp(V, axis=0).max()
    P = (V - V.mean(0)) / ext
    blotch = np.zeros(len(V))
    for f, a in ((7.0, 0.030), (17.0, 0.016)):            # low-frequency mottling
        ph = rng.random(3) * 2 * np.pi
        blotch += a * np.sin(P[:, 0] * f + ph[0]) * np.sin(P[:, 1] * f + ph[1]) \
                    * np.sin(P[:, 2] * f + ph[2])
    grain = rng.normal(0, 0.012, len(V))                  # fine surface noise

    col = base[None, :] * shade * warm + (blotch + grain)[:, None]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(col, 0, 1))
    return mesh
