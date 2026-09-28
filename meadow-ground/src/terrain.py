"""Corner-based Wang tiles for a meadow, drawn in a clean Undertale-like way.

Terrains are nested levels:  PATH(0) < GRASS(1) < LUSH(2) < FLOWERS(3).
Every level boundary is a *clean* curve: a bilinear corner field, a gentle
interior wobble (zero at the tile border) and tile-periodic scallops that
turn the edge into little rounded grass tufts. No dithering, no noise.
"""
import math
import numpy as np

from palette import G, L, P, F, S
from pixelkit import Canvas, Rng, value_noise, hash01, dilate, erode

TS = 20
PATH, GRASS, LUSH, FLOWERS = 0, 1, 2, 3
NAMES = {PATH: "Path", GRASS: "Grass", LUSH: "Lush grass", FLOWERS: "Golden flowers"}
NVAR_PURE = 4
NVAR_EDGE = 2
M = 3                                 # margin used for neighbour tests

_ys, _xs = np.mgrid[-M:TS + M, -M:TS + M].astype(float)
_U = (_xs + 0.5) / TS
_V = (_ys + 0.5) / TS
_WIN = (np.clip(np.sin(np.pi * np.clip(_U, 0, 1)), 0, 1) * np.clip(np.sin(np.pi * np.clip(_V, 0, 1)), 0, 1)) ** 2

# per-level edge style: scallop period (must divide TS), amplitude, wobble
EDGE = {
    GRASS: dict(period=10, amp=0.07, wob=0.75, seed=11),
    LUSH: dict(period=10, amp=0.10, wob=0.55, seed=23),
    FLOWERS: dict(period=10, amp=0.08, wob=0.50, seed=37),
}


def _bilinear(c, u, v):
    tl, tr, bl, br = c
    return (tl * (1 - u) + tr * u) * (1 - v) + (bl * (1 - u) + br * u) * v


def _scallop(period, phase):
    """Smooth, tile-periodic waves along both axes (no cusps at tile borders)."""
    bx = np.sin(2 * np.pi * (_xs + 0.5) / period + phase)
    by = np.sin(2 * np.pi * (_ys + 0.5) / period + phase + 1.3)
    return 0.5 * (bx + by)


def masks(corners, variant):
    """Level of every pixel on the extended (TS+2M)^2 grid."""
    n = TS + 2 * M
    level = np.zeros((n, n), dtype=np.int16)
    present = np.ones((n, n), dtype=bool)
    for lv in (GRASS, LUSH, FLOWERS):
        e = EDGE[lv]
        ind = [1.0 if c >= lv else 0.0 for c in corners]
        if min(ind) == max(ind):
            f = np.full((n, n), ind[0])
        else:
            f = _bilinear(ind, _U, _V)
            wob = (value_noise(n, n, 9, seed=e["seed"] + variant * 101) - 0.5
                   + (value_noise(n, n, 5, seed=e["seed"] + variant * 57 + 3) - 0.5) * 0.5)
            f = f + wob * e["wob"] * _WIN
            f = f + _scallop(e["period"], 0.0) * e["amp"]
        m = f > 0.5
        # tidy the edge: no one-pixel spurs or pin-holes (opening, then closing)
        m = dilate(erode(m, True), True)
        m = erode(dilate(m, True), True)
        present = present & m
        level[present] = lv
    return level


# ---------------------------------------------------------------- stamps
_C = {}
for _i, _ch in enumerate("abcde"):
    _C[_ch] = G[_i]
for _i, _ch in enumerate("fghij"):
    _C[_ch] = L[_i]
for _i, _ch in enumerate("klmno"):
    _C[_ch] = P[_i]
for _i, _ch in enumerate("pqrstuvwx"):
    _C[_ch] = F[_i]

GRASS_TUFTS = [["b.b", ".b."], ["b...b", ".b.b.", "..b.."], ["b.b.b", ".b.b."], ["d", "b"],
               ["d.d", "b.b"], [".d.", "b.b"]]
LUSH_TUFTS = [["g.g", ".g."], ["i.i.i", "g.g.g"], ["j", "i", "g"], ["g..g", ".gg."], ["i.i", "g.g"]]
PEBBLES = [["on", "ll"], ["o", "l"], ["n.", ".l"], ["onn", "lll"]]
FLOWER = [".srs.", "srrrq", "rrpqq", ".qqq.", "g.g.g"]       # buttercup, lit from the top-left
FLOWER_B = [".sr..", "srrq.", "rpqq.", ".qq.g", "..g.."]
BUD = [".r.", "rpq", ".g."]


def _fits(mask, x, y, rows):
    for dy, row in enumerate(rows):
        for dx, ch in enumerate(row):
            if ch == ".":
                continue
            yy, xx = y + dy + M, x + dx + M
            if not mask[yy, xx]:
                return False
    return True


def _stamp(px, x, y, rows):
    for dy, row in enumerate(rows):
        for dx, ch in enumerate(row):
            if ch != ".":
                px[y + dy + M, x + dx + M] = _C[ch]


def _scatter(rng, px, mask, stamps, count, margin=2):
    placed = []
    for _ in range(count * 6):
        if len(placed) >= count:
            break
        rows = rng.choice(stamps)
        w, h = max(len(r) for r in rows), len(rows)
        x = rng.randint(margin, TS - margin - w)
        y = rng.randint(margin, TS - margin - h)
        if any(abs(x - a) < 5 and abs(y - b) < 4 for (a, b) in placed):
            continue
        # keep one pixel of clearance from other terrains
        clear = all(mask[y + dy + M - 1:y + dy + M + 2, x + M - 1:x + w + M + 1].all() for dy in range(h))
        if clear and _fits(mask, x, y, rows):
            _stamp(px, x, y, rows)
            placed.append((x, y))
    return placed


# ---------------------------------------------------------------- tile
def make_tile(corners, variant=0):
    corners = tuple(int(c) for c in corners)
    key = sum((c + 1) * 5 ** i for i, c in enumerate(corners))
    rng = Rng(int(hash01(key, variant, 7) * 1e9))
    lv = masks(corners, variant)
    n = TS + 2 * M
    px = np.zeros((n, n), dtype=np.int16)
    mp, mg, ml, mf = (lv == PATH), (lv == GRASS), (lv == LUSH), (lv == FLOWERS)

    # ---- flat base colours
    px[mp] = P[2]
    px[mg] = G[2]
    px[ml | mf] = L[2]

    # ---- sparse, simple details
    dens = [1, 2, 2, 3][variant % 4]
    _scatter(rng, px, mp, PEBBLES, [0, 1, 1, 2][variant % 4])
    for _ in range(rng.randint(0, 2)):                        # dark specks on the dirt
        x, y = rng.randint(2, TS - 3), rng.randint(2, TS - 3)
        if mp[y + M, x + M]:
            px[y + M, x + M] = P[1]
    _scatter(rng, px, mg, GRASS_TUFTS, dens)
    for _ in range(rng.randint(0, 2)):                        # a light fleck or two
        x, y = rng.randint(2, TS - 3), rng.randint(2, TS - 3)
        if mg[y + M, x + M] and px[y + M, x + M] == G[2]:
            px[y + M, x + M] = G[3]
    _scatter(rng, px, ml, LUSH_TUFTS, dens + 2)
    # golden flowers on a jittered grid (2 x 2 per tile) + small buds between
    for gy in range(2):
        for gx in range(2):
            x = 1 + gx * 10 + rng.randint(0, 3)
            y = 1 + gy * 10 + rng.randint(0, 3)
            rows = FLOWER if rng.chance(0.6) else FLOWER_B
            if _fits(mf, x, y, rows) and rng.chance(0.85):
                _stamp(px, x, y, rows)
    for _ in range(3):
        x, y = rng.randint(1, TS - 4), rng.randint(1, TS - 4)
        if _fits(mf, x, y, BUD) and (px[y + M:y + M + 3, x + M:x + M + 3] == L[2]).all():
            _stamp(px, x, y, BUD)

    # ---- edges: one clean dark line + a soft shadow where a higher terrain overhangs
    def nb(mask, dx, dy):
        return np.roll(np.roll(mask, dy, 0), dx, 1)

    def touching(m, other):
        return m & (nb(other, 1, 0) | nb(other, -1, 0) | nb(other, 0, 1) | nb(other, 0, -1))

    above_p = lv > PATH
    # dirt: dark border all round, doubled under the grass (the grass sits higher)
    px[touching(mp, above_p)] = P[1]
    px[mp & nb(above_p, 0, 2)] = P[1]
    # grass: its lower lip is outlined
    px[mg & nb(mp, 0, -1)] = G[1]
    # lush patches and flower beds: dark rim, darker lower lip, shadow on the grass below
    lushy = ml | mf
    px[touching(lushy, mg | mp)] = L[1]
    px[lushy & nb(mg | mp, 0, -1)] = L[0]
    px[mg & nb(lushy, 0, 1)] = G[1]

    c = Canvas(TS, TS)
    c.px[:] = px[M:M + TS, M:M + TS]
    return c
