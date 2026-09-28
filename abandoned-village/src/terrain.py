"""Corner-based Wang terrain tiles (void / grass / moss / path).

Terrains are nested by 'level': VOID(0) < GRASS(1) < MOSS(2) < PATH(3).
For every level we build a continuous scalar field from the four corners
(bilinear, domain-warped) + noise. Everything that touches a tile border is
either tile-periodic or faded out by a window function, so any two tiles
that share a pair of corners meet seamlessly, while the interior of every
variant wanders freely (organic, non-repeating edges).
"""
import numpy as np

from palette import K
from pixelkit import Canvas, value_noise, hash01, Rng

TS = 16
VOID, GRASS, MOSS, PATH = 0, 1, 2, 3
NAMES = {VOID: "Void", GRASS: "Grass", MOSS: "Moss", PATH: "Path"}
NVAR_PURE = 3    # variants of a single-terrain tile
NVAR_EDGE = 3    # variants of a transition tile

_yy, _xx = np.mgrid[0:TS, 0:TS].astype(float)
_U = (_xx + 0.5) / TS
_V = (_yy + 0.5) / TS
_SU = np.sin(np.pi * _U)
_SV = np.sin(np.pi * _V)
_WIN = _SU * _SV  # 0 at tile borders, 1 in the centre


def _bilinear(c, u, v):
    tl, tr, bl, br = c
    return (tl * (1 - u) + tr * u) * (1 - v) + (bl * (1 - u) + br * u) * v


# per-level shape parameters
SHAPE = {
    #        warp  edge-noise interior-noise  jitter  jitter-cell   threshold
    1: dict(w=0.26, a1=0.10, a3=0.80, j=0.26, jy=2, th=0.50, seed=11),  # grass vs void: ragged
    2: dict(w=0.30, a1=0.10, a3=0.95, j=0.24, jy=1, th=0.50, seed=23),  # moss: cloudy, dithered
    3: dict(w=0.22, a1=0.08, a3=0.70, j=0.20, jy=1, th=0.53, seed=37),  # path
}


def fields(corners, variant):
    out = {}
    for L, p in SHAPE.items():
        ind = [1.0 if c >= L else 0.0 for c in corners]
        if min(ind) == max(ind):
            out[L] = np.full((TS, TS), ind[0])
            continue
        s = p["seed"]
        # domain warp that vanishes on the matching borders
        wx = (value_noise(TS, TS, 6, seed=s + 1 + variant * 17) - 0.5) * 2 * p["w"] * _SU
        wy = (value_noise(TS, TS, 6, seed=s + 2 + variant * 17) - 0.5) * 2 * p["w"] * _SV
        f = _bilinear(ind, np.clip(_U + wx, 0, 1), np.clip(_V + wy, 0, 1))
        # tiny tile-periodic noise (keeps borders consistent)
        f = f + (value_noise(TS, TS, 4, seed=s + 5, period=TS) - 0.5) * p["a1"]
        # free interior noise, different for each variant
        n3 = value_noise(TS, TS, 6, seed=s * 7 + variant * 131 + 3)
        n4 = value_noise(TS, TS, 3, seed=s * 5 + variant * 71 + 9)
        f = f + (n3 - 0.5) * p["a3"] * _WIN + (n4 - 0.5) * 0.3 * _WIN
        out[L] = f
    return out


def _speckle(rng, arr, mask, color, density):
    r = rng.r.random(arr.shape)
    sel = (r < density) & mask
    arr[sel] = color


def _stamp(arr, mask, x, y, rows):
    for dy, row in enumerate(rows):
        for dx, ch in enumerate(row):
            if ch == ".":
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < TS and 0 <= xx < TS and mask[yy, xx]:
                arr[yy, xx] = _HEX[ch]


_HEX = {c: K[i] for i, c in enumerate("0123456789abcdefgh")}

GRASS_STAMPS = [
    ["8.8", "686"],
    [".9.", "8.8", "6.6"],
    ["9..", ".8.", "6.6"],
    ["8", "6"],
    ["9.8", "868"],
    ["88", "66"],
    ["6.6"],
]
MOSS_STAMPS = [
    ["b.", "9b"],
    [".b", "a9"],
    ["bb", "99"],
    ["a.a"],
]
PATH_STAMPS = [
    ["cc", "99"],          # pebble
    ["c", "9"],
    ["ccb", "999"],
    ["a.a"],
    ["aa"],
    [".c.", "c9c", ".9."],
]


def _textures(rng, masks, variant):
    tex = np.zeros((TS, TS), dtype=np.int16)
    g, m, p = masks[GRASS], masks[MOSS], masks[PATH]
    # clustered speckles: only where a local noise is high
    clus = value_noise(TS, TS, 3, seed=rng.randint(0, 99999))

    # --- grass
    tex[g] = K[7]
    _speckle(rng, tex, g & (clus > 0.55), K[6], 0.10)
    _speckle(rng, tex, g & (clus < 0.25), K[8], 0.06)
    for _ in range(rng.randint(0, 1) + (variant % 2)):
        _stamp(tex, g, rng.randint(2, 12), rng.randint(2, 12), rng.choice(GRASS_STAMPS))

    # --- moss (lighter, rough)
    tex[m] = K[10]
    _speckle(rng, tex, m & (clus > 0.5), K[9], 0.16)
    _speckle(rng, tex, m & (clus < 0.3), K[11], 0.05)
    for _ in range(rng.randint(0, 1)):
        _stamp(tex, m, rng.randint(2, 12), rng.randint(2, 12), rng.choice(MOSS_STAMPS))

    # --- path (smooth, worn)
    tex[p] = K[11]
    _speckle(rng, tex, p & (clus > 0.6), K[10], 0.10)
    _speckle(rng, tex, p & (clus < 0.25), K[12], 0.05)
    for _ in range(rng.randint(0, 1)):
        _stamp(tex, p, rng.randint(2, 12), rng.randint(2, 12), rng.choice(PATH_STAMPS))
    return tex


def make_tile(corners, variant=0):
    key = sum((c + 1) * 5 ** i for i, c in enumerate(corners))
    rng = Rng(int(hash01(key, variant, 99) * 1e9))
    F = fields(corners, variant)
    jit = rng.r.random((TS, TS))
    # vertically streaked jitter for ragged grass blades at the void edge
    jb = hash01(_xx.astype(int) + key * 0, (_yy.astype(int) // SHAPE[1]["jy"]) + variant * 7, 5)

    level = np.zeros((TS, TS), dtype=np.int16)
    present = np.ones((TS, TS), dtype=bool)
    for L in (1, 2, 3):
        p = SHAPE[L]
        j = jb if L == 1 else jit
        present = present & (F[L] + (j - 0.5) * p["j"] > p["th"])
        level[present] = L
    masks = {L: level == L for L in (0, 1, 2, 3)}
    tex = _textures(rng, masks, variant)
    f1, f2, f3 = F[1], F[2], F[3]

    # ---- void: flat black; faint soil rim under the grass edge
    v = masks[VOID]
    tex[v] = K[0]
    tex[v & (f1 > 0.36)] = K[1]
    tex[v & (f1 > 0.44) & (jit > 0.5)] = K[2]

    # ---- grass darkens gradually towards the void edge (depth)
    g = masks[GRASS]
    tex[g & (f1 < 0.80) & (tex == K[8])] = K[7]
    tex[g & (f1 < 0.74) & (jit > 0.5) & (tex == K[7])] = K[6]
    tex[g & (f1 < 0.66) & (tex == K[7])] = K[6]
    tex[g & (f1 < 0.60) & (jit > 0.45)] = K[5]
    # pale blade tips on the very rim
    tex[g & (f1 < 0.56) & (jb > 0.8)] = K[8]

    # ---- soft light around moss (grass side)
    tex[g & (f2 > 0.38) & (jit > 0.72) & (tex == K[7])] = K[8]
    # ---- moss: lighter towards the path
    m = masks[MOSS]
    tex[m & (f3 > 0.44) & (jit > 0.55)] = K[11]
    # ---- path: dithered, slightly darker border
    pm = masks[PATH]
    tex[pm & (f3 < 0.60) & (jit > 0.6)] = K[10]

    c = Canvas(TS, TS)
    c.px[:] = tex
    return c
