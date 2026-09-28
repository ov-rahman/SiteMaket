"""Pixel-art fog: layered drifting mist, quantised with ordered dithering.

Density is continuous noise; it is snapped to a few alpha steps with a Bayer
matrix, so the fog keeps a crisp pixel texture instead of a smooth blur.
Animation loops seamlessly by cross-fading two copies of each drifting layer
(one period apart), so no spatial repetition is needed.
"""
import numpy as np
from PIL import Image

from pixelkit import value_noise

FOG_RGB = (226, 232, 214)
_B4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0


def bayer(h, w):
    return np.tile(_B4, (h // 4 + 1, w // 4 + 1))[:h, :w]


def _stretched(w, h, cell_x, cell_y, seed, ox):
    """Anisotropic noise (long horizontal ribbons)."""
    # isotropic noise on a vertically stretched grid, then squashed back
    q = 4                                           # work at 1/4 width for speed
    ws = max(2, w // q)
    hs = max(2, int(h * cell_x / cell_y) // q)
    n = value_noise(ws, hs, cell_x / q, seed=seed, ox=ox / q)
    im = Image.fromarray(n.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=float)


def density(w, h, t=0.0, loop=1.0, water=None, seed=3, strength=1.0):
    """Fog density in [0,1] at time t (seconds); loops every `loop` seconds."""
    yy = np.arange(h)[:, None] / h

    def drift(t_):
        big = value_noise(w, h, 150, seed=seed, ox=t_ * 9.0, oy=t_ * 1.5)
        mid = value_noise(w, h, 56, seed=seed + 7, ox=t_ * 16.0)
        rib = _stretched(w, h, 90, 10, seed + 13, t_ * 24.0)
        return big, mid, rib

    s = (t % loop) / loop
    a = drift(t % loop)
    b = drift(t % loop - loop)
    big = a[0] * (1 - s) + b[0] * s
    mid = a[1] * (1 - s) + b[1] * s
    rib = a[2] * (1 - s) + b[2] * s
    d = 0.12 + (big - 0.5) * 0.55 + (mid - 0.5) * 0.25 + 0.30 * (1 - yy) ** 1.6
    if water is not None:
        d = d + water * (0.10 + np.clip(rib - 0.55, 0, 1) * 0.9)
    else:
        d = d + np.clip(rib - 0.55, 0, 1) * 0.5
    return np.clip(d * strength, 0, 1)


def overlay(d, levels=16, max_alpha=0.6, rgb=FOG_RGB):
    """Quantise density into dithered alpha steps -> RGBA PIL image."""
    h, w = d.shape
    scaled = d * levels
    lvl = np.floor(scaled + bayer(h, w)).clip(0, levels)
    alpha = (lvl / levels * max_alpha * 255).astype(np.uint8)
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = rgb
    out[..., 3] = alpha
    return Image.fromarray(out, "RGBA")
