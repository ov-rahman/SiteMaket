"""Geometric mask helpers (arches, rings, gables)."""
import numpy as np

from pixelkit import poly_mask


def grid(w, h):
    yy, xx = np.mgrid[0:h, 0:w]
    return xx + 0.5, yy + 0.5


def pointed_arch(w, h, xl, xr, y_spring, y_bottom, radius_k=1.0):
    """Gothic arch opening between x in [xl, xr), spring line y_spring."""
    X, Y = grid(w, h)
    span = xr - xl
    r = span * radius_k
    rect = (X >= xl) & (X < xr) & (Y >= y_spring) & (Y < y_bottom)
    c1 = (X - (xr - r)) ** 2 + (Y - y_spring) ** 2 <= r * r   # left arc, centred right
    c2 = (X - (xl + r)) ** 2 + (Y - y_spring) ** 2 <= r * r
    top = (Y < y_spring) & c1 & c2 & (X >= xl) & (X < xr)
    return rect | top


def round_arch(w, h, xl, xr, y_spring, y_bottom):
    X, Y = grid(w, h)
    cx = (xl + xr) / 2
    r = (xr - xl) / 2
    rect = (X >= xl) & (X < xr) & (Y >= y_spring) & (Y < y_bottom)
    top = (Y < y_spring) & ((X - cx) ** 2 + (Y - y_spring) ** 2 <= r * r)
    return rect | top


def disc(w, h, cx, cy, r):
    X, Y = grid(w, h)
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r * r


def ring(w, h, cx, cy, r0, r1):
    X, Y = grid(w, h)
    d = (X - cx) ** 2 + (Y - cy) ** 2
    return (d <= r1 * r1) & (d > r0 * r0)


def rect(w, h, x, y, rw, rh):
    m = np.zeros((h, w), dtype=bool)
    m[max(0, y):max(0, y + rh), max(0, x):max(0, x + rw)] = True
    return m


def poly(w, h, pts):
    return poly_mask(w, h, pts)


def band_below_line(w, h, p0, p1, thickness):
    """Parallelogram between line p0-p1 and the same line moved down."""
    (x0, y0), (x1, y1) = p0, p1
    return poly_mask(w, h, [(x0, y0), (x1, y1), (x1, y1 + thickness), (x0, y0 + thickness)])
