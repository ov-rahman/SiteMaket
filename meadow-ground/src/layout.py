"""A sample meadow room: 32 x 24 tiles of 20 px (640 x 480 = four Undertale screens).

A worn path comes in from the south, curls through a small clearing and
leaves east; a side track runs off to the west. Thick grass gathers along
the edges, and a bed of golden flowers grows in the north-west corner.
"""
import numpy as np

from pixelkit import value_noise
from terrain import PATH, GRASS, LUSH, FLOWERS

MW, MH = 32, 24
TS = 20


def _seg_dist(px, py, a, b):
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _smooth(pts, k=8):
    """Catmull-Rom spline through the control points."""
    p = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for j in range(k):
            t = j / k
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * ((2 * p1[c]) + (-p0[c] + p2[c]) * t + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * t2
                                    + (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * t3) for c in (0, 1)))
    out.append(pts[-1])
    return out


def _poly_dist(px, py, pts):
    pts = _smooth(pts)
    d = np.full(px.shape, 1e9)
    for a, b in zip(pts[:-1], pts[1:]):
        d = np.minimum(d, _seg_dist(px, py, a, b))
    return d


# paths in corner coordinates (x, y)
MAIN = [(15.5, 25), (15, 21), (13.5, 17.5), (13, 14.5), (15, 12), (19, 11), (23, 9.5), (27, 9), (33, 8.5)]
WEST = [(12.5, 14.5), (9, 15.5), (5, 15), (-1, 16)]
NORTH = [(19, 11), (19.5, 7), (18.5, 3), (19, -1)]
CLEARING = (13.2, 14.2, 3.2, 2.4)


def corner_grid():
    cy, cx = np.mgrid[0:MH + 1, 0:MW + 1].astype(float)
    n1 = value_noise(MW + 1, MH + 1, 3, seed=5)
    n2 = value_noise(MW + 1, MH + 1, 4, seed=8)
    g = np.full((MH + 1, MW + 1), GRASS, dtype=int)

    # thick grass hugging the room edges + a few patches inside
    edge = np.minimum.reduce([cx, cy, MW - cx, MH - cy])
    lush = (edge < 1.6 + (n1 - 0.5) * 3.6)
    for (x, y, rx, ry) in ((24.5, 16.5, 3.2, 2.0), (27, 3.5, 2.6, 1.7), (4.5, 20.5, 2.4, 1.6)):
        lush |= ((cx - x) / rx) ** 2 + ((cy - y) / ry) ** 2 + (n2 - 0.5) * 0.9 < 1
    g[lush] = LUSH
    # golden flower bed in the north-west
    fx, fy = 7.5, 5.5
    bed = ((cx - fx) / 3.6) ** 2 + ((cy - fy) / 2.4) ** 2 + (n2 - 0.5) * 0.6 < 1
    g[bed] = FLOWERS
    g[(((cx - fx) / 5.2) ** 2 + ((cy - fy) / 3.8) ** 2 < 1) & (g < LUSH)] = LUSH

    # paths (never under the flower bed)
    wob = (n1 - 0.5) * 0.6
    path = (_poly_dist(cx, cy, MAIN) < 1.15 + wob) | (_poly_dist(cx, cy, WEST) < 0.9 + wob * 0.8)
    path |= _poly_dist(cx, cy, NORTH) < 0.8 + wob * 0.7
    x0, y0, rx, ry = CLEARING
    path |= ((cx - x0) / rx) ** 2 + ((cy - y0) / ry) ** 2 + wob * 0.3 < 1
    g[path] = PATH

    # neighbouring corners may differ by one level only, so every tile is a
    # clean two-terrain transition: lower any corner that towers over a neighbour
    for _ in range(4):
        p = np.pad(g, 1, mode="edge")
        lo = np.full_like(g, 9)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    lo = np.minimum(lo, p[1 + dy:1 + dy + g.shape[0], 1 + dx:1 + dx + g.shape[1]])
        g = np.minimum(g, lo + 1)
    return g
