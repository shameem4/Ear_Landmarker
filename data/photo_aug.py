"""Make a render look like a photograph, without re-rendering it.

Head 1 scores 5.1x the achievable floor on real photographs. Measured with
scripts/render_vs_photo.py, a logistic probe on seven plain image statistics
tells a render from a photo 93.4% of the time, and the gaps that carry it are,
in order: global contrast (d = 2.25), background sd (1.71), pixel noise
(1.56), sharpness (1.00).

TWO THINGS THAT MEASUREMENT CORRECTED, both of which I had backwards:

  Renders are SIX TIMES LESS SHARP than the photographs, not more. The
  intuition that a photo is soft and a render is crisp is wrong here: real
  ears come with hair, pores, shadows and scene edges, while a render is
  smooth skin with vertex-colour texture that is Nyquist-limited anyway.

  There is no backdrop to replace. Background covers 5.9% of the average crop
  and 0% of most of them -- the frame is skin. So "background sd" is not a
  grey backdrop at all, it is skin being too smooth and too evenly lit, and
  compositing scenery behind the ear would fix almost nothing.

Everything here is therefore photometric or occlusive, never geometric, so
landmark coordinates are untouched and this composes with the existing
geometric augmentation rather than competing with it.

Occluders deliberately do NOT mark the points they cover as invisible. A real
annotator looking at an ear half under hair still places the hidden points,
and that is the behaviour wanted at inference.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def _fbm(rng, size, octaves, base):
    """Fractal noise in [0, 1]. Cheap, and enough for shading and pores."""
    out = np.zeros((size, size), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = max(2, int(base * 2 ** o))
        g = rng.random((n, n)).astype(np.float32)
        g = np.asarray(Image.fromarray((g * 255).astype(np.uint8))
                       .resize((size, size), Image.BICUBIC), np.float32) / 255.
        out += amp * g
        tot += amp
        amp *= 0.5
    return out / tot


def photo_realism(img, rng, strength=1.0, shading=1.0, texture=1.0,
                  noise=1.0, occlude=1.0):
    """PIL RGB render -> PIL RGB that scores closer to a photograph.

    THE KNOBS ARE SEPARATE BECAUSE ONE KNOB WAS MEASURABLY WRONG. Scaling
    every effect together, the setting that matched the photographs' pixel
    noise and sharpness (0.5) left global contrast at 26 against 48, while the
    setting that fixed contrast (1.5) overshot sharpness by 11x -- laplacian
    variance 2132 against the photographs' 191. Skin needs far more shading
    and contrast than it needs grain, so the two are scaled independently and
    the defaults below are fitted to the measured photo statistics rather than
    chosen. `strength` still scales all of them, for sweeping.
    """
    a = np.asarray(img, np.float32) / 255.0
    H, W = a.shape[:2]
    s = float(strength)
    sh, tx = s * shading, s * texture
    nz, oc = s * noise, s * occlude

    # 1. LOCAL SHADING. The single biggest gap is global contrast, and a
    #    render is lit by one soft source so the skin carries almost no
    #    large-scale brightness variation. A smooth multiplicative field adds
    #    the shadowing a real face has from its own geometry and surroundings.
    if rng.random() < 0.9:
        f = _fbm(rng, max(H, W), 3, 2)[:H, :W]
        amp = rng.uniform(0.20, 0.85) * sh
        a *= (1.0 + amp * (f - f.mean())[:, :, None] * 2.0)

    # 2. CONTRAST AND GAMMA, widened well past the existing colour jitter.
    if rng.random() < 0.9:
        g = np.exp(rng.uniform(-0.45, 0.45) * sh)
        a = np.clip(a, 0, None) ** g
        m = a.mean()
        a = m + (a - m) * (1.0 + rng.uniform(-0.20, 1.50) * sh)

    # 3. MICRO-TEXTURE. Renders lack pores entirely; this is what most of the
    #    6.6x sharpness gap is. Multiplicative so it rides on the skin rather
    #    than sitting on top of it as a grey film.
    if rng.random() < 0.8:
        # PINK, NOT WHITE. White noise matched the photographs' laplacian
        # variance but put 24% too much energy above the mid frequencies
        # (0.304 against 0.245), because real skin and real optics roll off
        # as roughly 1/f and flat noise does not. Shaping the spectrum fixes
        # the excess without losing the sharpness the texture was added for.
        w = rng.normal(0, 1, (H, W)).astype(np.float32)
        F = np.fft.fftshift(np.fft.fft2(w))
        yy, xx = np.mgrid[:H, :W]
        r = np.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2)
        F *= 1.0 / np.maximum(r, 1.0) ** rng.uniform(0.5, 1.1)
        t = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
        t = (t - t.mean()) / (t.std() + 1e-8)
        a *= (1.0 + rng.uniform(0.010, 0.055) * tx * t[:, :, None])

    # 4. OCCLUDERS: hair. Soft dark strokes, drawn as a swept curve so they
    #    look like strands rather than scribbles. Points underneath stay
    #    supervised, on purpose -- see the module docstring.
    if rng.random() < 0.45 * min(oc, 1.0):
        for _ in range(rng.integers(1, 5)):
            y0 = rng.uniform(-0.2, 1.2) * H
            x0 = rng.uniform(-0.2, 1.2) * W
            ang = rng.uniform(0, np.pi)
            ln = rng.uniform(0.4, 1.4) * max(H, W)
            wd = rng.uniform(0.01, 0.06) * max(H, W)
            curve = rng.uniform(-0.4, 0.4)
            t = np.linspace(0, 1, 160)
            xs = x0 + np.cos(ang) * ln * t + curve * ln * np.sin(np.pi * t) * -np.sin(ang)
            ys = y0 + np.sin(ang) * ln * t + curve * ln * np.sin(np.pi * t) * np.cos(ang)
            yy, xx = np.mgrid[0:H, 0:W]
            d2 = np.full((H, W), 1e9, np.float32)
            for xc, yc in zip(xs[::4], ys[::4]):
                d2 = np.minimum(d2, (xx - xc) ** 2 + (yy - yc) ** 2)
            mask = np.exp(-d2 / (2 * wd ** 2)).astype(np.float32)
            dark = rng.uniform(0.25, 0.8) * min(oc, 1.0)
            tint = np.array([rng.uniform(0.25, 0.75)] * 3, np.float32)
            tint *= rng.uniform(0.6, 1.4, 3)
            a = a * (1 - mask[:, :, None] * dark) + \
                tint * (mask[:, :, None] * dark) * rng.uniform(0.0, 0.5)

    # 5. SENSOR NOISE, luma and chroma separately, because a camera's chroma
    #    noise is coarser and a render has neither.
    if rng.random() < 0.9:
        sd = rng.uniform(0.003, 0.020) * nz
        a += rng.normal(0, sd, (H, W, 1)).astype(np.float32)
        a += rng.normal(0, sd * 0.6, (H, W, 3)).astype(np.float32)

    # Some of the wild photographs are greyscale outright; a render never is.
    if rng.random() < 0.08:
        a = a @ np.array([0.299, 0.587, 0.114], np.float32)
        a = np.repeat(a[:, :, None], 3, axis=2)

    a = np.clip(a, 0, 1)
    out = Image.fromarray((a * 255).astype(np.uint8))

    # 6. DEFOCUS, then JPEG. Both ways round exist in the wild; this order
    #    matches a camera, which blurs optically before it compresses.
    if rng.random() < 0.35:
        from PIL import ImageFilter
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.3)))
    if rng.random() < 0.7:
        b = io.BytesIO()
        out.save(b, "JPEG", quality=int(rng.integers(30, 92)))
        b.seek(0)
        out = Image.open(b).convert("RGB")
    return out
