"""Corner-based Wang terrain for the swamp: deep water / shallows / mud / grass.

Levels are nested: DEEP(0) < SHALLOW(1) < MUD(2) < GRASS(3). Each level has a
continuous field built from the four corners (bilinear, domain-warped, noisy
in the interior only), so tiles sharing corners meet seamlessly.

Water tiles are animated: every frame has identical content except the
'glints' - short reflections that brighten and fade in turn.
"""
import numpy as np

from palette import G, W, B
from pixelkit import Canvas, value_noise, hash01, Rng

TS = 16
DEEP, SHALLOW, MUD, GRASS = 0, 1, 2, 3
NAMES = {DEEP: "Deep water", SHALLOW: "Shallows", MUD: "Mud", GRASS: "Grass"}
NVAR_PURE = 3
NVAR_EDGE = 3
FRAMES = 4

_yy, _xx = np.mgrid[0:TS, 0:TS].astype(float)
_U = (_xx + 0.5) / TS
_V = (_yy + 0.5) / TS
_SU = np.sin(np.pi * _U)
_SV = np.sin(np.pi * _V)
_WIN = _SU * _SV


def _bilinear(c, u, v):
    tl, tr, bl, br = c
    return (tl * (1 - u) + tr * u) * (1 - v) + (bl * (1 - u) + br * u) * v


SHAPE = {
    #        warp  edge  interior jitter  threshold
    1: dict(w=0.30, a1=0.10, a3=0.90, j=0.30, th=0.40, seed=11),   # shallows: soft, wide dither
    2: dict(w=0.26, a1=0.08, a3=0.80, j=0.12, th=0.50, seed=23),   # shoreline: crisp-ish
    3: dict(w=0.24, a1=0.08, a3=0.75, j=0.22, th=0.60, seed=37),   # grass: ragged blades
}


def fields(corners, variant):
    out = {}
    for L, p in SHAPE.items():
        ind = [1.0 if c >= L else 0.0 for c in corners]
        if min(ind) == max(ind):
            out[L] = np.full((TS, TS), ind[0])
            continue
        s = p["seed"]
        wx = (value_noise(TS, TS, 6, seed=s + 1 + variant * 17) - 0.5) * 2 * p["w"] * _SU
        wy = (value_noise(TS, TS, 6, seed=s + 2 + variant * 17) - 0.5) * 2 * p["w"] * _SV
        f = _bilinear(ind, np.clip(_U + wx, 0, 1), np.clip(_V + wy, 0, 1))
        f = f + (value_noise(TS, TS, 4, seed=s + 5, period=TS) - 0.5) * p["a1"]
        n3 = value_noise(TS, TS, 6, seed=s * 7 + variant * 131 + 3)
        n4 = value_noise(TS, TS, 3, seed=s * 5 + variant * 71 + 9)
        f = f + (n3 - 0.5) * p["a3"] * _WIN + (n4 - 0.5) * 0.3 * _WIN
        out[L] = f
    return out


def _speckle(rng, arr, mask, color, density):
    r = rng.r.random(arr.shape)
    arr[(r < density) & mask] = color


_HEX = {}
for _i, _c in enumerate("0123456789abcdef"):
    _HEX[_c] = G[_i]
for _i, _c in enumerate("ABCDEFGHIJKL"):
    _HEX[_c] = W[_i]
for _i, _c in enumerate("mnopqrstuv"):
    _HEX[_c] = B[_i]


def _stamp(arr, mask, x, y, rows):
    for dy, row in enumerate(rows):
        for dx, ch in enumerate(row):
            if ch == ".":
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < TS and 0 <= xx < TS and mask[yy, xx]:
                arr[yy, xx] = _HEX[ch]


GRASS_STAMPS = [
    ["a.a", "989"],
    [".b.", "a.a", "8.8"],
    ["b..", ".a.", "8.9"],
    ["a", "8"],
    ["a.b", "9a9"],
    ["aa", "88"],
]
MUD_STAMPS = [
    ["tu", "rs"],          # pebble
    ["HI"],                 # puddle glint
    ["IH", ".G"],
    ["r.r"],                # tracks
]


def _textures(rng, masks, variant):
    """Static texture + list of glint pixels [(y, x, phase, kind)]."""
    tex = np.zeros((TS, TS), dtype=np.int16)
    glints = []
    d, s, m, g = masks[DEEP], masks[SHALLOW], masks[MUD], masks[GRASS]
    clus = value_noise(TS, TS, 3, seed=rng.randint(0, 99999))
    clus2 = value_noise(TS, TS, 5, seed=rng.randint(0, 99999))

    # --- deep water: calm dark jade, barely textured
    tex[d] = W[5]
    _speckle(rng, tex, d & (clus2 > 0.6), W[4], 0.10)
    _speckle(rng, tex, d & (clus2 < 0.3), W[6], 0.04)
    # --- shallows: lighter, silty
    tex[s] = W[7]
    _speckle(rng, tex, s & (clus > 0.55), W[6], 0.22)
    _speckle(rng, tex, s & (clus < 0.2), W[8], 0.06)
    # --- mud: greenish, wet, calm
    tex[m] = B[6]
    _speckle(rng, tex, m & (clus > 0.55), B[5], 0.18)
    _speckle(rng, tex, m & (clus2 > 0.62), G[6], 0.16)
    _speckle(rng, tex, m & (clus < 0.2), B[7], 0.06)
    for _ in range(rng.randint(0, 1)):
        rows = rng.choice(MUD_STAMPS)
        _stamp(tex, m, rng.randint(2, 12), rng.randint(2, 12), rows)
    # --- grass: pale sage
    tex[g] = G[9]
    _speckle(rng, tex, g & (clus > 0.58), G[8], 0.16)
    _speckle(rng, tex, g & (clus < 0.22), G[10], 0.10)
    for _ in range(rng.randint(0, 2) + (variant % 2)):
        _stamp(tex, g, rng.randint(2, 12), rng.randint(2, 12), rng.choice(GRASS_STAMPS))

    # --- glints: reflections on the water that twinkle in turn
    water = d | s
    n = rng.choice([0, 1, 1, 2])
    for _ in range(n):
        y, x = rng.randint(1, 14), rng.randint(1, 11)
        ln = rng.randint(1, 3)
        phase = rng.randint(0, FRAMES - 1)
        seg = [(y, x + k) for k in range(ln) if water[y, x + k]]
        if len(seg) == ln:
            for (yy, xx) in seg:
                glints.append((yy, xx, phase, "s" if s[yy, xx] else "d"))
    return tex, glints


GLINT = {  # colour per frame offset (0 = brightest)
    "d": [W[8], W[7], W[6], None],
    "s": [W[10], W[9], W[8], None],
}


def make_tile(corners, variant=0, frame=0):
    key = sum((c + 1) * 5 ** i for i, c in enumerate(corners))
    rng = Rng(int(hash01(key, variant, 99) * 1e9))
    F = fields(corners, variant)
    jit = rng.r.random((TS, TS))
    level = np.zeros((TS, TS), dtype=np.int16)
    present = np.ones((TS, TS), dtype=bool)
    for L in (1, 2, 3):
        p = SHAPE[L]
        present = present & (F[L] + (jit - 0.5) * p["j"] > p["th"])
        level[present] = L
    masks = {L: level == L for L in (0, 1, 2, 3)}
    tex, glints = _textures(rng, masks, variant)
    f1, f2, f3 = F[1], F[2], F[3]
    d, s, m, g = masks[DEEP], masks[SHALLOW], masks[MUD], masks[GRASS]

    # ---- depth: shallows darken as they deepen, deep water lightens near the shelf
    tex[s & (f1 < 0.52) & (jit > 0.45)] = W[6]
    tex[d & (f1 > 0.28) & (jit > 0.55)] = W[6]
    # ---- the waterline: pale scum and floating bits hugging the bank
    scum = (d | s) & (f2 > SHAPE[2]["th"] - 0.08)
    tex[scum & (jit > 0.35)] = W[9]
    tex[scum & (jit > 0.8)] = W[10]
    tex[(d | s) & (f2 > SHAPE[2]["th"] - 0.14) & (f2 <= SHAPE[2]["th"] - 0.08) & (jit > 0.7)] = W[8]
    # ---- wet, dark mud at the water's edge
    wet = m & (f2 < SHAPE[2]["th"] + 0.09)
    tex[wet] = np.where(jit[wet] > 0.5, B[5], B[4])
    # ---- grass edge: a darker rim, blades hanging over the mud
    rim = g & (f3 < SHAPE[3]["th"] + 0.07)
    tex[rim & (jit > 0.4)] = G[8]
    tex[rim & (jit > 0.85)] = G[7]
    blades = m & (f3 > SHAPE[3]["th"] - 0.06) & (hash01(_xx.astype(int), (_yy.astype(int)) // 2, key + variant) > 0.7)
    tex[blades] = G[8]

    # ---- glints for this frame
    for (y, x, phase, kind) in glints:
        col = GLINT[kind][(frame - phase) % FRAMES]
        if col is not None and level[y, x] <= SHALLOW:
            tex[y, x] = col

    c = Canvas(TS, TS)
    c.px[:] = tex
    return c
