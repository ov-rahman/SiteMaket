"""Man-made remains in the mire: plank boardwalks on stilts and a sunken boat."""
import math
import numpy as np

from palette import G, W, B, SH1, SH2, SH3
from pixelkit import Canvas, Rng, dilate, erode, shift, hash01, poly_mask

PLANK = [B[4], B[5], B[6], B[7], B[8]]     # dark .. light


def _plank_tone(r):
    x = r.rand()
    return PLANK[2] if x < 0.5 else PLANK[3] if x < 0.78 else PLANK[1]


def walk_h(seed=0, broken=False, post_l=True, post_r=True, cap_l=False, cap_r=False):
    """East-west boardwalk tile (16x16): planks run north-south across the deck."""
    r = Rng(700 + seed)
    c = Canvas(16, 16)
    top, bot = 2, 11                           # deck rows
    x = 0
    while x < 16:
        pw = r.randint(3, 4)
        tone = _plank_tone(r)
        miss = broken and 4 <= x <= 10 and r.chance(0.6)
        for xx in range(x, min(16, x + pw)):
            if miss:
                continue
            jag = r.randint(0, 1) if xx in (x, x + pw - 1) else 0
            for y in range(top + jag, bot):
                c.px[y, xx] = tone
            c.px[top + jag, xx] = PLANK[4] if tone != PLANK[1] else PLANK[2]
        if not miss:
            gap = min(15, x + pw - 1)
            for y in range(top, bot):
                c.px[y, gap] = B[3]
            if r.chance(0.6):
                c.px[top + 2, x + 1 if x + 1 < 16 else x] = B[2]          # nail
                c.px[bot - 3, x + 1 if x + 1 < 16 else x] = B[2]
        x += pw
    # front beam (the deck's edge) and its shadow on the water
    for xx in range(16):
        c.px[bot, xx] = B[3]
        c.px[bot + 1, xx] = B[2]
        c.px[bot + 2, xx] = SH2
        c.px[bot + 3, xx] = SH1
    # stilts
    for px_, on in ((1, post_l), (13, post_r)):
        if not on:
            continue
        for y in range(bot, 16):
            c.px[y, px_] = B[5]
            c.px[y, px_ + 1] = B[3]
        c.px[15, px_ - 1] = W[9]
        c.px[15, px_ + 2] = W[9]
    if broken:
        # a snapped plank dangling into the water
        for k in range(6):
            yy, xx = 6 + k, 7 + (k // 2)
            if yy < 16:
                c.px[yy, xx] = PLANK[2]
                c.px[yy, xx + 1] = B[3]
    # back edge dark line
    for xx in range(16):
        if c.px[top, xx] >= 0 and c.px[top - 1, xx] < 0:
            c.px[top - 1, xx] = G[1]
    for side, on in ((0, cap_l), (15, cap_r)):
        if on:
            for y in range(top - 1, bot + 2):
                c.px[y, side] = G[1]
    return c


def walk_v(seed=0, broken=False, cap_t=False, cap_b=False):
    """North-south boardwalk tile: planks run east-west, rails on both sides."""
    r = Rng(800 + seed)
    c = Canvas(16, 16)
    x0, x1 = 3, 12
    y = 0
    while y < 16:
        ph = r.randint(3, 4)
        tone = _plank_tone(r)
        miss = broken and 4 <= y <= 10 and r.chance(0.6)
        for yy in range(y, min(16, y + ph)):
            if miss:
                continue
            jag = r.randint(0, 1) if yy in (y, y + ph - 1) else 0
            for xx in range(x0 + jag, x1 + 1 - jag):
                c.px[yy, xx] = tone
        if not miss:
            c.px[y, x0:x1 + 1] = np.where(c.px[y, x0:x1 + 1] >= 0, PLANK[4] if tone != PLANK[1] else PLANK[2], -1)
            e = min(15, y + ph - 1)
            c.px[e, x0:x1 + 1] = np.where(c.px[e, x0:x1 + 1] >= 0, B[3], -1)
        y += ph
    # side beams, right side in shade + shadow on the water to the right
    for yy in range(16):
        c.px[yy, x0 - 1] = B[6]
        c.px[yy, x0 - 2] = G[1]
        c.px[yy, x1 + 1] = B[2]
        c.px[yy, x1 + 2] = SH2
        c.px[yy, x1 + 3] = SH1
    # stilts poking out beside the deck
    for py_ in (3, 11):
        c.px[py_, x1 + 2] = B[4]
        c.px[py_ + 1, x1 + 2] = B[3]
        c.px[py_ + 2, x1 + 2] = W[9]
    if cap_t:
        c.px[0, x0 - 2:x1 + 2] = G[1]
    if cap_b:
        c.px[15, x0 - 2:x1 + 2] = B[2]
    return c


def platform(w=48, h=48, seed=0):
    """A square landing on stilts (wider deck)."""
    r = Rng(900 + seed)
    c = Canvas(w, h)
    top, bot = 3, h - 8
    x = 1
    while x < w - 1:
        pw = r.randint(3, 4)
        tone = _plank_tone(r)
        for xx in range(x, min(w - 1, x + pw)):
            c.px[top:bot, xx] = tone
            c.px[top, xx] = PLANK[4]
        c.px[top:bot, min(w - 2, x + pw - 1)] = B[3]
        x += pw
    c.px[bot, 1:w - 1] = B[3]
    c.px[bot + 1, 1:w - 1] = B[2]
    c.px[bot + 2, 1:w - 1] = SH2
    c.px[bot + 3, 1:w - 1] = SH1
    for px_ in (2, w // 2 - 1, w - 5):
        for y in range(bot, h - 1):
            c.px[y, px_] = B[5]
            c.px[y, px_ + 1] = B[3]
        c.px[h - 1, px_ - 1] = W[9]
        c.px[h - 1, px_ + 2] = W[9]
    # mooring post with rope
    c.px[top - 3:top + 5, w - 4] = B[6]
    c.px[top - 3:top + 5, w - 3] = B[3]
    c.px[top + 1, w - 5:w - 2] = G[10]
    solid = c.px >= 0
    edge = dilate(solid) & ~solid & (np.arange(h)[:, None] < bot)
    c.px[edge] = G[1]
    return c


def boat(seed=0):
    """A half-sunk rowboat, bow sticking out of the water."""
    W_, H_ = 48, 32
    c = Canvas(W_, H_)
    X, Y = np.meshgrid(np.arange(W_) + 0.5, np.arange(H_) + 0.5)
    hull = poly_mask(W_, H_, [(4, 18), (10, 11), (30, 8), (42, 10), (44, 14), (38, 20), (12, 23)])
    inner = poly_mask(W_, H_, [(8, 17), (12, 13), (29, 11), (39, 12), (40, 14), (35, 18), (13, 20)])
    c.px[hull] = B[5]
    c.px[hull & (Y > 16)] = B[4]
    c.px[hull & ~shift(hull, 0, 1)] = B[7]
    c.px[inner] = B[3]
    # water filling the stern
    water = inner & (X < 24)
    c.px[water] = W[5]
    c.px[water & ~shift(water, 1, 0)] = W[8]
    c.px[inner & ~water & ~shift(inner, 0, 1)] = B[2]
    # thwart (seat) and ribs
    c.px[inner & (np.abs(X - 30) < 1.5)] = B[6]
    for rx in (18, 24, 35):
        c.px[inner & (np.abs(X - rx) < 0.6) & ~water] = B[4]
    # the stern sinks: fade into the water
    sink = hull & (X < 9)
    c.px[sink] = W[6]
    c.px[sink & (hash01(X.astype(int), Y.astype(int), 3) > 0.5)] = B[4]
    solid = c.px >= 0
    c.px[dilate(solid) & ~solid & (X > 7)] = G[1]
    # ripple around the hull
    rip = dilate(dilate(solid)) & ~dilate(solid) & (Y > 14)
    c.px[rip & (X.astype(int) % 3 != 0)] = W[9]
    return c
