"""Swamp flora: reeds & cattails, lily pads, duckweed, cypress knees, ferns,
fungi, fallen logs and dead snags."""
import math
import numpy as np

from palette import G, W, B, F, SH1, SH2, SH3
from pixelkit import Canvas, Rng, dilate, erode, shift, value_noise, hash01, from_ascii, poly_mask
import trees as TR


# ================================================================ reeds / cattails
def reeds(seed=0, w=32, h=48, n=None, in_water=True, frame=0, cattails=True, dry=False):
    """A clump of reeds; `frame` (0..3) sways the tips for animation.

    Stems enter the water at slightly different depths; below each stem a
    short broken reflection, and a faint ripple where the clump stands."""
    r = Rng(seed)
    c = Canvas(w, h)
    base_y = h - 6
    n = n or r.randint(9, 15)
    blades = []
    for i in range(n):
        x0 = w / 2 + r.uniform(-w * 0.3, w * 0.3)
        hgt = r.uniform(h * 0.32, h * 0.86)
        lean = r.uniform(-0.3, 0.3) + (x0 - w / 2) / w * 0.6
        by = base_y + r.randint(-2, 2) + (1 if abs(x0 - w / 2) < w * 0.12 else 0)
        blades.append((x0, hgt, lean, r.rand(), by))
    blades.sort(key=lambda b: (b[4], -b[1]))           # back rows first
    sway = [0, 1, 0, -1][frame % 4]
    lit = [G[10], G[11]] if not dry else [F[4], F[5]]
    mid = [G[8], G[9]] if not dry else [B[7], F[4]]
    dark = [G[6], G[7]] if not dry else [B[5], B[6]]
    feet = []
    for (x0, hgt, lean, tone, by) in blades:
        pts = []
        for k in range(int(hgt)):
            t = k / hgt
            x = x0 + lean * k * 0.35 + sway * t * t * 1.5
            pts.append((int(round(x)), int(by - k), t))
        col_set = lit if tone > 0.66 else (mid if tone > 0.3 else dark)
        for (x, y, t) in pts:
            if 0 <= x < w and 0 <= y < h:
                c.px[y, x] = col_set[0] if t < 0.7 else col_set[1]
        if tone > 0.5:                                   # a broader leaf near the base
            for (x, y, t) in pts[: int(len(pts) * 0.35)]:
                if 0 <= x + 1 < w and c.px[y, x + 1] < 0:
                    c.px[y, x + 1] = dark[0]
        if cattails and tone < 0.2 and hgt > h * 0.6 and not dry:
            (x, y, t) = pts[-1]
            for dy in range(1, 7):
                for dx in (0, 1):
                    yy, xx = y + dy + 1, x + dx
                    if 0 <= yy < h and 0 <= xx < w:
                        c.px[yy, xx] = F[1] if dx == 0 else F[0]
        # darker, wet lower stem
        for (x, y, t) in pts[:2]:
            if 0 <= x < w and 0 <= y < h:
                c.px[y, x] = G[5] if not dry else B[4]
        feet.append((int(round(x0)), by))
    # thin dark outline only on the right side of stems (keeps them readable on pale water)
    solid = c.px >= 0
    right = shift(solid, 1, 0) & ~solid
    c.px[right & (hash01(np.arange(w)[None, :], np.arange(h)[:, None], seed) > 0.4)] = G[4]
    if in_water:
        for (x, by) in feet:
            # broken reflection under the stem
            for k in range(1, r.randint(3, 5)):
                y = by + k
                if 0 <= x < w and 0 <= y < h and c.px[y, x] < 0 and k % 2:
                    c.px[y, x] = W[5]
            # ripple glint beside the stem, drifting with the frame
            y = by + 1
            xx = x + (1 if (frame + x) % 2 else -1)
            if 0 <= xx < w and 0 <= y < h and c.px[y, xx] < 0:
                c.px[y, xx] = W[9]
    else:
        for (x, by) in feet:
            if 0 <= x < w and 0 <= by + 1 < h and c.px[by + 1, x] < 0:
                c.px[by + 1, x] = SH1
    return c


# ================================================================ lily pads
LILY = [
    # a single pad with a notch
    ["..7777..",
     ".79998.7",
     "7999987.",
     "7998887.",
     "7988877.",
     ".78777..",
     "..6666..."[:8]],
    # small pad
    [".777.",
     "79987",
     "7..87",
     ".776."],
    # pad with a flower
    ["...ii....",
     "..ijji...",
     ".7ijjji7.",
     "79ijjii87",
     "7998i8887",
     ".7888877.",
     "..66666.."],
]


def lily_pads(seed=0, flower=False):
    r = Rng(seed)
    c = Canvas(16, 16)
    k = r.randint(1, 3)
    for i in range(k):
        rad = r.uniform(2.6, 4.6) if i else r.uniform(3.8, 5.2)
        x, y = r.uniform(rad + 1, 15 - rad), r.uniform(rad + 1, 15 - rad * 0.8)
        X, Y = np.meshgrid(np.arange(16) + 0.5, np.arange(16) + 0.5)
        m = ((X - x) / rad) ** 2 + ((Y - y) / (rad * 0.72)) ** 2 <= 1
        ang = r.uniform(0, 2 * math.pi)
        a = np.arctan2(Y - y, X - x)
        notch = (np.abs(((a - ang + math.pi) % (2 * math.pi)) - math.pi) < 0.28) & (((X - x) ** 2 + (Y - y) ** 2) > 1.5)
        m &= ~notch
        m &= c.px < 0
        c.px[m] = G[8]
        c.px[m & ((X - x) + (Y - y) * 1.3 < -rad * 0.5)] = G[10]
        c.px[m & ((X - x) + (Y - y) * 1.3 > rad * 0.6)] = G[7]
        rim = m & ~shift(m, 0, -1)
        c.px[rim] = G[6]
        # vein
        c.set(int(x), int(y), G[7])
        # shadow on the water below the pad
        sh = shift(m, 1, 1) & ~m & (c.px < 0)
        c.px[sh] = SH1
    if flower:
        x, y = r.randint(4, 11), r.randint(4, 10)
        fl = from_ascii([".j.", "jlj", "ij."])
        c.blit(fl, x - 1, y - 1)
        c.set(x, y - 2, F[3])
    return c


def duckweed(seed=0, density=0.5):
    """Speckled floating film; stays off the tile border so patches tile."""
    r = Rng(seed)
    c = Canvas(16, 16)
    n = value_noise(16, 16, 5, seed=seed * 13 + 1)
    X, Y = np.meshgrid(np.arange(16), np.arange(16))
    border = (X >= 1) & (X <= 14) & (Y >= 1) & (Y <= 14)
    rnd = np.random.default_rng(seed).random((16, 16))
    m = border & (n > 1 - density) & (rnd < 0.55)
    c.px[m] = G[9]
    c.px[m & (rnd < 0.18)] = G[11]
    c.px[m & (rnd > 0.45)] = G[8]
    return c


# ================================================================ cypress knees
KNEES = [
    ["..u..",
     ".tus.",
     ".tsr.",
     "tusrq",
     "JIIIJ"],
    [".u.",
     "tur",
     "tsr",
     "JIJ"],
    ["...u....",
     "..tus...",
     "..tsr.u.",
     ".tusrqtr",
     ".tsrqqsr",
     "JIIIJJIJ"],
]


def knees(seed=0):
    r = Rng(seed)
    c = Canvas(16, 16)
    for i in range(r.randint(1, 3)):
        k = from_ascii(KNEES[r.randint(0, len(KNEES) - 1)])
        c.blit(k, r.randint(1, 16 - k.w), r.randint(4, 16 - k.h))
    solid = (c.px >= 0) & ~np.isin(c.px, [W[8], W[9]])
    c.px[dilate(solid) & ~solid & (c.px < 0)] = G[1]
    return c


# ================================================================ ferns / fungi / grass
FERNS = [
    ["..a.......a..",
     "...a..a..a...",
     "a...a.a.a...a",
     ".a..9aaa9..a.",
     "..9..999..9..",
     "...99989999..",
     "..8.88888.8..",
     ".8..78887..8.",
     "....67776....",
     ".....555....."],
    [".a.....a.",
     "..a...a..",
     "a..a.a..a",
     ".9.999.9.",
     "..98989..",
     ".8.787.8.",
     "...666..."],
]
FUNGI = [
    [".ll..",
     "lkkl.",
     ".22.l",
     ".22lk",
     "..22."],
    ["..lll..",
     ".lkkkl.",
     "lkkkkkl",
     "...2...",
     "...2.ll",
     "..22lkk",
     "...2.2."],
]
GRASS = [
    [".a...a.",
     ".9.a.9.",
     "a9.9.9a",
     ".989.8.",
     "..787..",
     "..666.."],
    ["..b..",
     "a.a.a",
     "9a9a9",
     "89898",
     "67676"],
    ["a......a",
     ".9..a.9.",
     ".9.a9.9.",
     "..9998..",
     ".78887..",
     "..6666.."],
]


def sprite16(rows, ox=None, oy=None, h=16):
    c = Canvas(16, h)
    t = from_ascii(rows)
    c.blit(t, (16 - t.w) // 2 if ox is None else ox, h - 1 - t.h if oy is None else oy)
    return c


# ================================================================ fallen log (bridge)
def fallen_log(length=64, seed=0):
    """Mossy trunk lying east-west; walkable as a bridge."""
    r = Rng(seed)
    W_, H_ = length, 32
    c = Canvas(W_, H_)
    y0, th = 10, 13
    ramp = [B[2], B[3], B[4], B[5], B[6], B[7]]
    for x in range(4, W_ - 5):
        wob = int(round(math.sin(x * 0.11 + seed) * 0.8))
        for k in range(th):
            y = y0 + k + wob
            t = k / (th - 1)
            idx = 5 if t < 0.12 else 4 if t < 0.3 else 3 if t < 0.55 else 2 if t < 0.8 else 1
            c.px[y, x] = ramp[idx]
    # bark cracks along the log
    for _ in range(length // 3):
        x, k = r.randint(6, W_ - 8), r.randint(3, th - 3)
        for d in range(r.randint(2, 6)):
            if x + d < W_ - 5:
                c.px[y0 + k, x + d] = B[2]
    # moss blanket on top
    mn = value_noise(W_, H_, 5, seed=seed + 3)
    for x in range(4, W_ - 5):
        col = np.where(c.px[:, x] >= 0)[0]
        if not len(col):
            continue
        top = col.min()
        depth = int(1 + mn[top, x] * 4)
        for k in range(depth):
            c.px[top + k, x] = G[8] if k == 0 else G[6]
        if mn[top, x] > 0.6:
            c.px[top, x] = G[9]
    # cut end (left) shows rings; broken end (right) jagged
    X, Y = np.meshgrid(np.arange(W_) + 0.5, np.arange(H_) + 0.5)
    end = ((X - 5) / 3.4) ** 2 + ((Y - (y0 + th / 2)) / (th / 2)) ** 2 <= 1
    c.px[end] = B[6]
    c.px[end & ((((X - 5) / 1.8) ** 2 + ((Y - (y0 + th / 2)) / (th / 3.5)) ** 2) <= 1)] = B[5]
    c.px[end & ((((X - 5) / 0.8) ** 2 + ((Y - (y0 + th / 2)) / 1.4) ** 2) <= 1)] = B[4]
    for k in range(th + 1):
        d = r.randint(0, 4)
        for x in range(W_ - 5 - d, W_ - 4):
            c.px[y0 + k, x] = -1
    # shelf fungi
    for _ in range(r.randint(2, 4)):
        x = r.randint(10, W_ - 12)
        f = from_ascii(["lkk", "kk2"] if r.chance(0.5) else ["llk", ".k2"])
        c.blit(f, x, y0 + th - 3)
    solid = c.px >= 0
    c.px[dilate(solid) & ~solid] = G[0]
    # shadow + ripple on the water under the log
    sh = shift(solid, 0, 3) & ~solid & (c.px < 0)
    c.px[sh] = SH1
    rip = shift(solid, 0, 5) & ~dilate(solid, True) & (c.px < 0)
    c.px[rip & (X.astype(int) % 4 == 0)] = W[9]
    return c


# ================================================================ dead snag
def snag(seed=3, w=64, h=128):
    """Bleached dead cypress: tall thin trunk, snapped top, a few bare limbs."""
    r = Rng(seed)
    p = TR.TreeParts(w, h)
    base_y = h - 8
    top_y = base_y - int(h * 0.62)
    lean = r.uniform(-0.06, 0.06)
    c = p.trunk
    # limbs first (behind the trunk), starting at the trunk edge
    for i in range(r.randint(2, 3)):
        side = -1 if i % 2 == 0 else 1
        sy = top_y + r.uniform(10, (base_y - top_y) * 0.5)
        sx = w / 2 + lean * (base_y - sy) + side * 3
        ex = sx + side * r.uniform(10, 22)
        ey = sy - r.uniform(12, 26)
        m, nx, ny = TR._tube(w, h, TR._bezier((sx, sy), (sx + side * 8, sy - 3), (ex, ey), 30), 2.2, 0.8)
        tmp = Canvas(w, h)
        TR._shade_normals(tmp, m, nx, ny, TR.BARK_RAMP[2:])
        c.blit(tmp, 0, 0)
        for d in (-1, 1):                       # a twig fork at the end
            c.line(int(ex), int(ey), int(ex + d * 3), int(ey - 4), B[6])
    TR._trunk(p, w / 2, base_y, top_y, 9, 10, lean, r, n_roots=3, hollow=False, moss=0.5)
    # snapped, splintered top
    tx = int(w / 2 + lean * (base_y - top_y))
    for dx in range(-7, 8):
        d = r.randint(0, 6) if abs(dx) < 5 else 12
        for y in range(top_y - 3, top_y + d):
            if 0 <= y < h and 0 <= tx + dx < w:
                c.px[y, tx + dx] = -1
    solid = c.px >= 0
    c.px[dilate(solid) & ~solid] = G[0]
    p.foot = (int(w / 2 - 7), base_y - 8, 14, 10)
    p.base_y = base_y
    return p
