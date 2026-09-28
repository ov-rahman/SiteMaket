"""Reusable surface painters: stone, planks, shingles, holes, cracks, vines…

All painters take a boolean `mask` (area to paint) and write palette indices
into a Canvas, so shapes can be arbitrary polygons.
"""
import numpy as np

from palette import K, V
from pixelkit import (Canvas, poly_mask, ellipse_mask, line_points, dilate, erode,
                      edge_dir, edge_inner, shift, Rng, value_noise)


# ---------------------------------------------------------------- palettes
STONE = dict(mortar=K[5], dark=K[8], base=K[9], light=K[10], hi=K[11], lo=K[7])
STONE_LIT = dict(mortar=K[6], dark=K[10], base=K[11], light=K[12], hi=K[13], lo=K[9])
STONE_PALE = dict(mortar=K[7], dark=K[11], base=K[12], light=K[13], hi=K[14], lo=K[10])
STONE_SHADE = dict(mortar=K[4], dark=K[7], base=K[8], light=K[9], hi=K[10], lo=K[6])
WOOD = dict(gap=V[1], dark=V[3], base=V[4], light=V[5], hi=V[6], lo=V[2])
WOOD_LIT = dict(gap=V[2], dark=V[4], base=V[5], light=V[6], hi=V[7], lo=V[3])
ROOF = dict(gap=V[1], dark=V[4], base=V[5], light=V[6], hi=V[7], lo=V[3])
ROOF_LIT = dict(gap=V[2], dark=V[5], base=V[6], light=V[7], hi=V[8], lo=V[4])


def stone_blocks(c, mask, rng, pal=STONE, row_h=6, bw=(8, 14), x0=0, y0=0,
                 chip=0.25, stagger=True, rows_tone=None):
    """Ashlar masonry clipped to `mask`."""
    h, w = mask.shape
    tmp = np.full((h, w), -1, dtype=np.int16)
    y = y0 - row_h * 4
    row = 0
    while y < h:
        x = x0 - rng.randint(0, bw[1]) - (bw[0] // 2 if (stagger and row % 2) else 0)
        while x < w:
            bwid = rng.randint(*bw)
            r = rng.rand()
            tone = pal["base"] if r < 0.55 else (pal["light"] if r < 0.8 else pal["dark"])
            if rows_tone is not None:
                tone = rows_tone(row, tone)
            x1, y1 = x + bwid - 1, y + row_h - 1
            ys, xs = max(y, 0), max(x, 0)
            ye, xe = min(y1, h), min(x1, w)
            if ye > ys and xe > xs:
                tmp[ys:ye, xs:xe] = tone
                # mortar: bottom row + right column of the block
                if 0 <= y1 < h:
                    tmp[y1, xs:min(x1 + 1, w)] = pal["mortar"]
                if 0 <= x1 < w:
                    tmp[ys:min(y1 + 1, h), x1] = pal["mortar"]
                # top highlight + bottom shade inside the block
                if 0 <= y < h and tone != pal["dark"]:
                    tmp[y, xs:max(xs, xe - 1)] = pal["hi"] if tone == pal["light"] else pal["light"]
                if 0 <= y1 - 1 < h and y1 - 1 > y:
                    tmp[y1 - 1, xs + 1:xe] = pal["lo"] if tone == pal["dark"] else pal["dark"]
                # chips / weathering
                if rng.chance(chip):
                    cx, cy = rng.randint(xs, max(xs, xe - 1)), rng.randint(ys, max(ys, ye - 1))
                    if 0 <= cy < h and 0 <= cx < w:
                        tmp[cy, cx] = pal["lo"]
                        if cx + 1 < w and rng.chance(0.5):
                            tmp[cy, cx + 1] = pal["lo"]
            x += bwid
        y += row_h
        row += 1
    sel = mask & (tmp >= 0)
    c.px[sel] = tmp[sel]


def planks(c, mask, rng, pal=WOOD, width=5, vertical=True, x0=0, y0=0, grain=0.12,
           nails=True, gaps=True):
    h, w = mask.shape
    tmp = np.full((h, w), pal["base"], dtype=np.int16)
    n = (w if vertical else h)
    pos = (x0 if vertical else y0) - rng.randint(0, width)
    while pos < n:
        pw = width + rng.randint(-1, 1)
        tone = rng.choice([pal["base"], pal["base"], pal["light"], pal["dark"]])
        a, b = max(pos, 0), min(pos + pw, n)
        if b > a:
            if vertical:
                tmp[:, a:b] = tone
                tmp[:, a] = pal["light"] if tone != pal["light"] else pal["hi"]
                if gaps and b - 1 < n:
                    tmp[:, b - 1] = pal["gap"]
                # grain streaks
                for _ in range(int(h * grain)):
                    gx = min(rng.randint(a + 1, max(a + 1, b - 2)), w - 1)
                    gy = rng.randint(0, h - 1)
                    ln = rng.randint(2, 5)
                    tmp[gy:gy + ln, gx] = pal["dark"] if tone != pal["dark"] else pal["lo"]
            else:
                tmp[a:b, :] = tone
                tmp[a, :] = pal["light"] if tone != pal["light"] else pal["hi"]
                if gaps and b - 1 < n:
                    tmp[b - 1, :] = pal["gap"]
                for _ in range(int(w * grain)):
                    gy = min(rng.randint(a + 1, max(a + 1, b - 2)), h - 1)
                    gx = rng.randint(0, w - 1)
                    ln = rng.randint(2, 6)
                    tmp[gy, gx:gx + ln] = pal["dark"] if tone != pal["dark"] else pal["lo"]
        pos += pw
    sel = mask
    c.px[sel] = tmp[sel]


def shingles(c, mask, rng, pal=ROOF, row_h=5, sw=(5, 8), y_start=None, missing=0.0,
             hole_color=None):
    """Rows of slates/shingles. Rows are laid bottom-up (eave first)."""
    h, w = mask.shape
    ys = np.where(mask.any(axis=1))[0]
    if len(ys) == 0:
        return
    top, bot = ys.min(), ys.max()
    tmp = np.full((h, w), -1, dtype=np.int16)
    y = bot + 1 if y_start is None else y_start
    row = 0
    while y > top - row_h:
        ry0 = y - row_h
        x = -rng.randint(0, sw[1])
        while x < w:
            s = rng.randint(*sw)
            r = rng.rand()
            tone = pal["base"] if r < 0.5 else (pal["light"] if r < 0.75 else pal["dark"])
            xa, xb = max(x, 0), min(x + s, w)
            ya, yb = max(ry0, 0), min(y, h)
            if xb > xa and yb > ya:
                if missing and rng.chance(missing):
                    tmp[ya:yb, xa:xb] = hole_color if hole_color is not None else pal["gap"]
                else:
                    tmp[ya:yb, xa:xb] = tone
                    # lower edge: dark lip (shadow onto the row below)
                    if 0 <= y - 1 < h:
                        tmp[y - 1, xa:xb] = pal["gap"]
                    if 0 <= y - 2 < h:
                        tmp[y - 2, xa:xb] = pal["lo"] if tone == pal["dark"] else pal["dark"]
                    # left separator
                    tmp[ya:yb - 1, xa] = pal["gap"] if rng.chance(0.7) else pal["dark"]
                    # top light
                    if 0 <= ry0 < h:
                        tmp[ry0, xa + 1:xb] = pal["hi"] if tone == pal["light"] else pal["light"]
            x += s
        y -= row_h - 1   # rows overlap by one pixel
        row += 1
    sel = mask & (tmp >= 0)
    c.px[sel] = tmp[sel]


def jagged_blob(w, h, cx, cy, rx, ry, rng, rough=0.35, n=14):
    """Irregular polygon mask (for holes, breaks, rubble)."""
    pts = []
    for i in range(n):
        a = 2 * np.pi * i / n + rng.uniform(-0.15, 0.15)
        r = 1.0 - rng.uniform(0, rough)
        pts.append((cx + np.cos(a) * rx * r, cy + np.sin(a) * ry * r))
    return poly_mask(w, h, pts)


def roof_hole(c, roof_mask, hole_mask, rng, rafters=True, pal=ROOF, rafter_dx=9,
              rafter_col=None, rafter_slope=None):
    """Punch a ragged hole into a roof: dark interior, broken lips, rafters.

    Rafters run down the slope: vertical on screen for a roof facing the
    viewer, or along `rafter_slope` (dy/dx) for a gable plane seen sideways.
    """
    hole = hole_mask & roof_mask
    c.px[hole] = K[0]
    inner = erode(hole)
    yy, xx = np.nonzero(inner)
    for y, x in zip(yy, xx):
        if (x * 7 + y * 3) % 11 == 0:
            c.px[y, x] = K[1]
    rc = rafter_col if rafter_col is not None else (V[3], V[4])
    if rafters and hole.any():
        ys_all, xs_all = np.nonzero(hole)
        if rafter_slope is None:
            for rx in range(xs_all.min() + rng.randint(1, 4), xs_all.max(), rafter_dx):
                ys = np.where(hole[:, rx])[0]
                if len(ys) < 3:
                    continue
                c.px[ys, rx] = rc[0]
                left = ys[hole[ys, max(rx - 1, 0)]]
                c.px[left, max(rx - 1, 0)] = rc[1]
            # one purlin across
            if rng.chance(0.7):
                py = rng.randint(ys_all.min() + 2, max(ys_all.min() + 2, ys_all.max() - 2))
                c.px[py, np.where(hole[py])[0]] = rc[0]
        else:
            k = rafter_slope
            cs = ys_all - k * xs_all
            for c0 in np.arange(cs.min() + rng.randint(2, 5), cs.max(), rafter_dx):
                for x in range(xs_all.min(), xs_all.max() + 1):
                    y = int(round(c0 + k * x))
                    if 0 <= y < hole.shape[0] and hole[y, x]:
                        c.px[y, x] = rc[0]
                        if y - 1 >= 0 and hole[y - 1, x]:
                            c.px[y - 1, x] = rc[1]
            # a purlin parallel to the ridge (vertical on screen)
            px_ = int((xs_all.min() + xs_all.max()) / 2) + rng.randint(-2, 2)
            c.px[np.where(hole[:, px_])[0], px_] = rc[0]
    top_lip = hole & ~shift(hole, 0, 1)
    lip_above = shift(top_lip, 0, -1) & roof_mask & ~hole
    c.px[lip_above] = pal["hi"]
    left_lip = shift(hole & ~shift(hole, 1, 0), -1, 0) & roof_mask & ~hole
    c.px[left_lip & (np.random.default_rng(3).random(hole.shape) > 0.4)] = pal["light"]
    bot = shift(hole & ~shift(hole, 0, -1), 0, 1) & roof_mask & ~hole
    c.px[bot] = pal["gap"]


def crack(c, x, y, length, rng, col=K[4], hi=None, dirx=None, mask=None):
    """Wandering hairline crack going roughly downward."""
    dx = dirx if dirx is not None else rng.choice([-1, 1])
    for i in range(length):
        if mask is None or (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]):
            c.set(x, y, col)
            if hi is not None:
                c.set(x + 1, y, hi)
        r = rng.rand()
        if r < 0.45:
            y += 1
        elif r < 0.8:
            x += dx
            y += 1
        else:
            x -= dx
        if rng.chance(0.08):
            dx = -dx


def vine(c, x, y, length, rng, cols=(K[4], K[6], K[8]), up=False, mask=None, leaf=0.35):
    """Creeping ivy strand with small leaf clusters."""
    stem, leafc, leafhi = cols
    dy = -1 if up else 1
    pts = []
    for i in range(length):
        pts.append((x, y))
        y += dy
        if rng.chance(0.35):
            x += rng.choice([-1, 1])
    for (px, py) in pts:
        if mask is None or (0 <= py < mask.shape[0] and 0 <= px < mask.shape[1] and mask[py, px]):
            c.set(px, py, stem)
    for (px, py) in pts:
        if rng.chance(leaf):
            s = rng.choice([-1, 1])
            for (lx, ly, col) in [(px + s, py, leafc), (px + s, py - 1, leafc), (px + 2 * s, py, leafhi if rng.chance(0.4) else leafc)]:
                if mask is None or (0 <= ly < mask.shape[0] and 0 <= lx < mask.shape[1] and mask[ly, lx]):
                    c.set(lx, ly, col)


def outline(c, color=K[1], diag=False, inside=False, mask=None):
    """Add a 1px outline around all opaque pixels (outside by default)."""
    m = c.opaque() if mask is None else mask
    if inside:
        e = edge_inner(m, diag)
    else:
        e = dilate(m, diag) & ~m
    c.px[e] = color
    return e


def selective_outline(c, mask=None, light=K[3], dark=K[1], diag=False):
    """Outside outline: lighter on the top/left, darker on the bottom/right."""
    m = c.opaque() if mask is None else mask
    e = dilate(m, diag) & ~m
    top_left = e & (shift(m, 0, -1) | shift(m, -1, 0))
    c.px[e] = dark
    c.px[top_left & ~(shift(m, 0, 1) | shift(m, 1, 0))] = light
    return e


def shade_right(c, mask, cols_map, width=3):
    """Darken the rightmost `width` pixels of every row of `mask` via mapping."""
    h, w = mask.shape
    for y in range(h):
        xs = np.where(mask[y])[0]
        if len(xs) == 0:
            continue
        for x in xs[-width:]:
            v = int(c.px[y, x])
            if v in cols_map:
                c.px[y, x] = cols_map[v]


DARKEN = {K[i]: K[max(0, i - 1)] for i in range(1, len(K))}
DARKEN.update({V[i]: V[max(0, i - 1)] for i in range(1, len(V))})
LIGHTEN = {K[i]: K[min(len(K) - 1, i + 1)] for i in range(len(K))}
LIGHTEN.update({V[i]: V[min(len(V) - 1, i + 1)] for i in range(len(V))})


def darken(c, mask, steps=1):
    for _ in range(steps):
        c.recolor(DARKEN, mask)


def lighten(c, mask, steps=1):
    for _ in range(steps):
        c.recolor(LIGHTEN, mask)


def dither_mask(mask, level, seed=0):
    """Ordered (Bayer 4x4) dither selection: keeps `level` (0..1) of mask."""
    B = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0
    h, w = mask.shape
    t = np.tile(B, (h // 4 + 1, w // 4 + 1))[:h, :w]
    if np.ndim(level) == 0:
        return mask & (t < level)
    return mask & (t < level)


def rubble(c, x, y, n, rng, spread=(10, 4), pal=STONE, sizes=(1, 3)):
    """Scatter small broken stones around (x,y)."""
    for _ in range(n):
        rx = x + rng.randint(-spread[0], spread[0])
        ry = y + rng.randint(-spread[1], spread[1])
        s = rng.randint(*sizes)
        stone(c, rx, ry, s + 1, s, rng, pal)


def stone(c, x, y, w, h, rng, pal=STONE, outline_col=K[2]):
    """A small angular stone: lit top, dark underside, shadow to the lower right."""
    x, y, w, h = int(x), int(y), max(1, int(w)), max(1, int(h))
    trim = w >= 4 and h >= 3
    for yy in range(h):
        for xx in range(w):
            if trim and (xx, yy) in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
                continue
            if yy == 0 or (trim and yy == 1 and xx in (0, w - 1)):
                col = pal["hi"] if xx <= w // 3 else pal["light"]
            elif yy == h - 1 and h > 1:
                col = pal["dark"]
            else:
                col = pal["base"]
            c.set(x + xx, y + yy, col)
    # contact shadow: under and right of the stone, only on empty pixels
    for xx in range(1 if trim else 0, w + 1):
        if c.get(x + xx, y + h) < 0 or c.get(x + xx, y + h) in (K[6], K[7], K[8]):
            c.set(x + xx, y + h, outline_col)
    for yy in range(1, h):
        if c.get(x + w, y + yy) < 0:
            c.set(x + w, y + yy, outline_col)
