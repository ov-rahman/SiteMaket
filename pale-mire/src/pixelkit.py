"""Tiny indexed-colour pixel-art toolkit.

Every sprite is a Canvas: a numpy array of palette indices where -1 means
transparent. Drawing happens with crisp, pixel-centre rules (no anti-aliasing),
so everything stays inside the palette.
"""
import numpy as np
from PIL import Image

from palette import RGBA, CHARMAP

T = -1  # transparent


class Canvas:
    def __init__(self, w, h, fill=T):
        self.w, self.h = w, h
        self.px = np.full((h, w), fill, dtype=np.int16)

    # ------------------------------------------------------------ basics
    def copy(self):
        c = Canvas(self.w, self.h)
        c.px = self.px.copy()
        return c

    def inside(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

    def set(self, x, y, c):
        x, y = int(x), int(y)
        if c is not None and self.inside(x, y):
            self.px[y, x] = c

    def get(self, x, y):
        x, y = int(x), int(y)
        return int(self.px[y, x]) if self.inside(x, y) else T

    def opaque(self):
        return self.px >= 0

    def rect(self, x, y, w, h, c):
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(self.w, int(x + w)), min(self.h, int(y + h))
        if x1 > x0 and y1 > y0:
            self.px[y0:y1, x0:x1] = c

    def hline(self, x0, x1, y, c):
        if x1 < x0:
            x0, x1 = x1, x0
        self.rect(x0, y, x1 - x0 + 1, 1, c)

    def vline(self, x, y0, y1, c):
        if y1 < y0:
            y0, y1 = y1, y0
        self.rect(x, y0, 1, y1 - y0 + 1, c)

    def line(self, x0, y0, x1, y1, c):
        for x, y in line_points(x0, y0, x1, y1):
            self.set(x, y, c)

    def poly(self, pts, c):
        m = poly_mask(self.w, self.h, pts)
        self.px[m] = c
        return m

    def ellipse(self, cx, cy, rx, ry, c):
        m = ellipse_mask(self.w, self.h, cx, cy, rx, ry)
        self.px[m] = c
        return m

    def fill_mask(self, m, c):
        self.px[m] = c

    def recolor(self, mapping, mask=None):
        """mapping: dict old->new (applied simultaneously)."""
        src = self.px.copy()
        for a, b in mapping.items():
            sel = src == a
            if mask is not None:
                sel &= mask
            self.px[sel] = b

    def blit(self, other, x, y, mask=None):
        """Paste another canvas (transparent pixels skipped)."""
        ox, oy = int(x), int(y)
        sx0, sy0 = max(0, -ox), max(0, -oy)
        dx0, dy0 = max(0, ox), max(0, oy)
        w = min(other.w - sx0, self.w - dx0)
        h = min(other.h - sy0, self.h - dy0)
        if w <= 0 or h <= 0:
            return
        src = other.px[sy0:sy0 + h, sx0:sx0 + w]
        sel = src >= 0
        if mask is not None:
            sel &= mask[sy0:sy0 + h, sx0:sx0 + w]
        dst = self.px[dy0:dy0 + h, dx0:dx0 + w]
        dst[sel] = src[sel]

    def crop(self, x, y, w, h):
        c = Canvas(w, h)
        c.blit(self, -x, -y)
        return c

    def flip_h(self):
        c = Canvas(self.w, self.h)
        c.px = self.px[:, ::-1].copy()
        return c

    # ------------------------------------------------------------ export
    def to_image(self):
        h, w = self.px.shape
        out = np.zeros((h, w, 4), dtype=np.uint8)
        idx = self.px
        sel = idx >= 0
        out[sel] = RGBA[idx[sel]]
        return Image.fromarray(out, "RGBA")

    def save(self, path, scale=1):
        im = self.to_image()
        if scale != 1:
            im = im.resize((self.w * scale, self.h * scale), Image.NEAREST)
        im.save(path)


# ---------------------------------------------------------------- helpers

def from_ascii(rows, charmap=None):
    """Build a canvas from a list of strings (see palette.CHARMAP)."""
    cm = charmap or CHARMAP
    rows = [r for r in rows]
    h = len(rows)
    w = max(len(r) for r in rows)
    c = Canvas(w, h)
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            if ch in (".", " "):
                continue
            c.px[y, x] = cm[ch]
    return c


def line_points(x0, y0, x1, y1):
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    pts = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        pts.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return pts


def poly_mask(w, h, pts):
    """Scanline fill using pixel centres."""
    m = np.zeros((h, w), dtype=bool)
    n = len(pts)
    ys = np.arange(h) + 0.5
    for yi, yc in enumerate(ys):
        xs = []
        for i in range(n):
            xa, ya = pts[i]
            xb, yb = pts[(i + 1) % n]
            if (ya <= yc < yb) or (yb <= yc < ya):
                t = (yc - ya) / (yb - ya)
                xs.append(xa + t * (xb - xa))
        xs.sort()
        for j in range(0, len(xs) - 1, 2):
            a = int(np.ceil(xs[j] - 0.5))
            b = int(np.floor(xs[j + 1] - 0.5))
            a, b = max(a, 0), min(b, w - 1)
            if b >= a:
                m[yi, a:b + 1] = True
    return m


def ellipse_mask(w, h, cx, cy, rx, ry):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2 <= 1.0


def shift(m, dx, dy, fill=False):
    out = np.full_like(m, fill)
    h, w = m.shape
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    out[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] = m[ys0:ys1, xs0:xs1]
    return out


def dilate(m, diag=False):
    o = m | shift(m, 1, 0) | shift(m, -1, 0) | shift(m, 0, 1) | shift(m, 0, -1)
    if diag:
        o |= shift(m, 1, 1) | shift(m, -1, 1) | shift(m, 1, -1) | shift(m, -1, -1)
    return o


def erode(m, diag=False):
    return ~dilate(~m, diag)


def edge_inner(m, diag=False):
    return m & ~erode(m, diag)


def edge_outer(m, diag=False):
    return dilate(m, diag) & ~m


def edge_dir(m, dx, dy):
    """Pixels of m whose neighbour in direction (dx,dy) is outside m."""
    return m & ~shift(m, -dx, -dy)


# ---------------------------------------------------------------- noise

def _hash(ix, iy, seed):
    ix = np.asarray(ix, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    h = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / float(0x1000000)


def hash01(x, y, seed=0):
    return _hash(x, y, seed)


def value_noise(w, h, cell, seed=0, period=None, ox=0, oy=0):
    """Smooth value noise. `period` (in pixels) makes it tile seamlessly."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    fx = (xx + ox) / cell
    fy = (yy + oy) / cell
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    tx = fx - x0
    ty = fy - y0
    tx = tx * tx * (3 - 2 * tx)
    ty = ty * ty * (3 - 2 * ty)

    def g(ix, iy):
        if period:
            p = max(1, int(round(period / cell)))
            ix, iy = ix % p, iy % p
        return _hash(ix, iy, seed)

    a = g(x0, y0)
    b = g(x0 + 1, y0)
    c = g(x0, y0 + 1)
    d = g(x0 + 1, y0 + 1)
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty


def fbm(w, h, cell, seed=0, octaves=3, period=None, ox=0, oy=0):
    tot = np.zeros((h, w))
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        tot += amp * value_noise(w, h, cell / (2 ** o), seed + o * 101, period, ox, oy)
        norm += amp
        amp *= 0.5
    return tot / norm


class Rng:
    """Deterministic RNG wrapper."""

    def __init__(self, seed):
        self.r = np.random.default_rng(seed)

    def rand(self):
        return float(self.r.random())

    def uniform(self, a, b):
        return float(self.r.uniform(a, b))

    def randint(self, a, b):
        """inclusive"""
        return int(self.r.integers(a, b + 1))

    def choice(self, seq):
        return seq[int(self.r.integers(0, len(seq)))]

    def chance(self, p):
        return self.r.random() < p


# ---------------------------------------------------------------- image utils

def upscale(img, s):
    return img.resize((img.width * s, img.height * s), Image.NEAREST)


def composite(layers, w, h):
    """Alpha-composite a list of PIL RGBA images."""
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for im in layers:
        out.alpha_composite(im)
    return out
