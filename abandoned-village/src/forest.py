"""Dark pine forest backdrop - horizontally tileable band with variants and caps.

The band is PERIOD px wide and BAND_H px tall. Everything that crosses the
left/right border is shared by all variants, so variants can be mixed in any
order and still join seamlessly.
"""
import math
import numpy as np

from palette import K
from pixelkit import Canvas, Rng, shift, hash01, value_noise
import materials as M

PERIOD = 128
BAND_H = 176
EDGE = 20
_B4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0


def _bayer(h, w):
    return np.tile(_B4, (h // 4 + 1, w // 4 + 1))[:h, :w]


def _pine(w, h, x, tip, base, width, seed, tier=None):
    """Asymmetric fir: drooping tiers, ragged branch ends. Wraps horizontally."""
    r = Rng(seed)
    tier = tier or r.randint(6, 9)
    m = np.zeros((h, w), dtype=bool)
    slope = width / max(1, (base - tip))
    lean = r.uniform(-0.04, 0.04)
    for y in range(max(0, int(tip)), min(h, int(base))):
        d = y - tip
        f = (d % tier) / tier
        # a tier widens towards its drooping lower edge
        k = 0.55 + 0.45 * f ** 1.6
        hwL = slope * d * k * (0.85 + 0.3 * hash01(int(d // tier), seed, 1)) + 0.4
        hwR = slope * d * k * (0.85 + 0.3 * hash01(int(d // tier), seed, 2)) + 0.4
        hwL += (hash01(y, seed, 3) - 0.5) * 1.4
        hwR += (hash01(y, seed, 4) - 0.5) * 1.4
        cx = x + lean * d
        for xx in range(int(math.floor(cx - hwL)), int(math.ceil(cx + hwR)) + 1):
            m[y, xx % w] = True
    return m


def _profile(n_shared_seed, variant, amp, cell, base):
    """Periodic height profile; identical in the border zone for all variants."""
    xs = np.arange(PERIOD)
    shared = value_noise(PERIOD, 1, cell, seed=n_shared_seed, period=PERIOD)[0]
    own = value_noise(PERIOD, 1, cell, seed=n_shared_seed + 1000 + variant * 7)[0]
    win = np.clip(np.minimum(xs, PERIOD - 1 - xs) / EDGE - 0.4, 0, 1)
    win = win * win * (3 - 2 * win)
    return base + (shared * (1 - win) + own * win - 0.5) * amp


def _trees_for(variant, layer, n, tip_rng, base_rng, width_rng, seed):
    shared_r = Rng(seed)
    trees = []
    # one shared tree straddling the border
    trees.append((shared_r.uniform(-4, 4) % PERIOD, shared_r.uniform(*tip_rng), shared_r.uniform(*base_rng),
                  shared_r.uniform(*width_rng), shared_r.randint(0, 99999)))
    r = Rng(seed * 3 + variant * 101 + 7)
    slots = np.linspace(EDGE, PERIOD - EDGE, n)
    for i in range(n):
        x = slots[i] + r.uniform(-6, 6)
        width = r.uniform(*width_rng)
        width = min(width, x - 3, PERIOD - x - 3)
        trees.append((x, r.uniform(*tip_rng), r.uniform(*base_rng), width, r.randint(0, 99999)))
    return trees


def band(variant=0):
    W, H = PERIOD, BAND_H
    c = Canvas(W, H)
    yy, xx = np.mgrid[0:H, 0:W]
    bay = _bayer(H, W)

    # ---- the sky that shows between the crowns: near-black, faintly blue low down
    c.px[:, :] = K[1]
    c.px[(yy > 96) & (bay < (yy - 96) / 8.0)] = K[2]
    c.px[yy > 104] = K[2]

    # ---- far layer: huge black spires, crowns go off the top
    far = _trees_for(variant, 0, 4, (-30, 10), (150, 170), (12, 17), 11)
    far_m = np.zeros((H, W), dtype=bool)
    for t in far:
        far_m |= _pine(W, H, *t)
    c.px[far_m] = K[0]
    c.px[far_m & (yy > 116) & (bay < (yy - 116) / 8.0)] = K[1]
    c.px[far_m & (yy > 124)] = K[1]

    # ---- mid layer
    mid = _trees_for(variant, 1, 4, (24, 56), (150, 170), (12, 17), 23)
    mid_m = np.zeros((H, W), dtype=bool)
    for t in mid:
        mid_m |= _pine(W, H, *t)
    c.px[mid_m] = K[2]
    rim = mid_m & ~shift(mid_m, 1, 0)
    c.px[rim & (yy > 40)] = K[3]
    c.px[mid_m & (yy > 136) & (bay < 0.5)] = K[3]

    # ---- near layer: moonlit rims on the left of every tier
    near = _trees_for(variant, 2, 3, (58, 96), (160, 176), (13, 18), 37)
    near_m = np.zeros((H, W), dtype=bool)
    for t in near:
        near_m |= _pine(W, H, *t)
    c.px[near_m] = K[3]
    rim = near_m & ~shift(near_m, 1, 0)
    c.px[rim] = K[5]
    c.px[shift(rim, 1, 0) & near_m & (bay < 0.5)] = K[4]
    under = near_m & ~shift(near_m, 0, -1)          # tier undersides in deep shadow
    c.px[shift(under, 0, -1) & near_m] = K[2]

    # ---- undergrowth: pale scrub clumps where the forest meets the meadow
    rs = Rng(4040)
    clumps = []
    for i in range(3):                                    # shared, across the border
        clumps.append(((rs.uniform(-8, 8)) % W, rs.uniform(152, 162), rs.uniform(9, 12), rs.uniform(6, 8), rs.randint(0, 9999)))
    rv = Rng(4041 + variant * 31)
    for x in np.linspace(EDGE, W - EDGE, 9):
        clumps.append((x + rv.uniform(-4, 4), rv.uniform(150, 164), rv.uniform(7, 11), rv.uniform(5, 8), rv.randint(0, 9999)))
    clumps.sort(key=lambda t: t[1])                        # back to front
    solid = yy >= 160
    c.px[solid] = K[5]
    for (cx, cy, rx, ry, sd) in clumps:
        bm = np.zeros((H, W), dtype=bool)
        for off in (-W, 0, W):
            ang = np.arctan2(yy + 0.5 - cy, xx + 0.5 - cx - off)
            wob = 1 + 0.18 * np.sin(ang * 5 + sd) + 0.1 * np.sin(ang * 9 + sd * 2)
            bm |= ((xx + 0.5 - cx - off) / (rx * wob)) ** 2 + ((yy + 0.5 - cy) / (ry * wob)) ** 2 <= 1
        bm &= yy < 174
        # shading of a rounded clump: lit upper-left, dark lower-right
        dx = np.minimum(np.abs(xx + 0.5 - cx), np.minimum(np.abs(xx + 0.5 - cx - W), np.abs(xx + 0.5 - cx + W)))
        sx = np.where(np.abs(xx + 0.5 - cx) <= dx + 1e-6, xx + 0.5 - cx,
                      np.where(np.abs(xx + 0.5 - cx - W) <= dx + 1e-6, xx + 0.5 - cx - W, xx + 0.5 - cx + W))
        lit = (sx / rx * 0.7 + (yy + 0.5 - cy) / ry) < -0.35
        dark = (sx / rx * 0.6 + (yy + 0.5 - cy) / ry) > 0.55
        c.px[bm] = K[6]
        c.px[bm & lit] = K[7]
        c.px[bm & dark] = K[5]
        rim = bm & ~shift(bm, 0, 1)
        c.px[rim & (sx < rx * 0.4)] = K[8]
        c.px[rim & (sx >= rx * 0.4)] = K[7]
        # a few leaf flecks
        c.px[bm & lit & (hash01(xx, yy, sd) < 0.12)] = K[8]
        c.px[bm & ~lit & (hash01(xx, yy, sd + 1) < 0.08)] = K[5]

    # ---- the bottom edge: ragged grass/scrub line over the meadow
    bottom = _profile(901, variant, 10, 14, 168)
    for x in range(W):
        b = int(bottom[x] + (hash01(x, 7, 99) - 0.5) * 3)
        c.px[b:, x] = -1
        c.set(x, b - 1, K[4])
        if hash01(x, 11, 42) > 0.7:
            hgt = 2 + int(hash01(x, 12, 42) * 4)
            c.vline(x, b - hgt, b - 1, K[6])
            c.set(x, b - hgt, K[8])
    return c


def cap(src, side, width=48):
    """Darken the band towards the map edge so it melts into the void."""
    c = src.copy()
    W, H = c.w, c.h
    xs = np.arange(W)
    t = (width - xs) / width if side == "left" else (xs - (W - width)) / width
    t = np.clip(t, 0, 1)
    thr = _bayer(H, W)
    opaque = c.px >= 0
    for steps in range(1, 6):
        sel = opaque & (thr < (t[None, :] * 5 - steps + 1))
        M.darken(c, sel)
    c.px[opaque & (t[None, :] > 0.9)] = K[0]
    return c


if __name__ == "__main__":
    import sys
    a, b = band(0), band(1)
    out = Canvas(PERIOD * 5, BAND_H)
    for i, v in enumerate([a, b, a, a, b]):
        out.blit(v, i * PERIOD, 0)
    out.save(sys.argv[1], 2)
