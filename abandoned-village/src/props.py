"""Village props: well, statue, signpost, fences, barrels, rocks, tufts..."""
import numpy as np

from palette import K, V, SH1, SH2
from pixelkit import Canvas, Rng, dilate, erode, shift, poly_mask, ellipse_mask, from_ascii
import materials as M
import shapes as S


def _cyl_shade(x, x0, x1, cols):
    """Pick a colour for a cylinder column: lit left, shaded right."""
    t = (x - x0) / max(1, (x1 - x0))
    if t < 0.12:
        return cols[3]
    if t < 0.45:
        return cols[2]
    if t < 0.8:
        return cols[1]
    return cols[0]


# ================================================================ well
def well():
    W, H = 48, 64
    c = Canvas(W, H)
    cx, top_y, rx, ry, hgt = 24, 42, 16, 6, 13
    # ---- stone drum (front face)
    X, Y = S.grid(W, H)
    top_e = ((X - cx) / rx) ** 2 + ((Y - top_y) / ry) ** 2 <= 1
    bot_e = ((X - cx) / rx) ** 2 + ((Y - top_y - hgt) / ry) ** 2 <= 1
    band = (np.abs(X - cx) <= rx) & (Y >= top_y) & (Y <= top_y + hgt)
    drum = (band | bot_e) & ~(top_e & (Y < top_y))
    drum &= ~top_e | (Y >= top_y)
    # courses follow the curvature
    for y in range(H):
        for x in range(W):
            if not drum[y, x]:
                continue
            dx = (x + 0.5 - cx) / rx
            curve = ry * np.sqrt(max(0.0, 1 - dx * dx))
            v = (y + 0.5 - top_y - curve)          # distance below the rim line
            row = int(np.floor(v / 4.0))
            ang = np.arcsin(np.clip(dx, -1, 1))
            u = ang * 5.0 + (row % 2) * 0.9
            base = _cyl_shade(x, cx - rx, cx + rx, (K[6], K[8], K[9], K[11]))
            col = base
            if v % 4.0 < 1.0 or abs(u - np.round(u)) < 0.13:
                col = K[5] if base in (K[6], K[8]) else K[6]
            c.px[y, x] = col
    # ---- rim (top ring) and dark water
    inner = ((X - cx) / (rx - 4)) ** 2 + ((Y - top_y) / (ry - 2)) ** 2 <= 1
    rim = top_e & ~inner
    c.px[rim] = K[11]
    for y, x in zip(*np.nonzero(rim)):
        a = np.arctan2((y + 0.5 - top_y) / ry, (x + 0.5 - cx) / rx)
        if int((a + np.pi) / (2 * np.pi) * 14) % 2 == 0:
            c.px[y, x] = K[12]
        if y + 0.5 < top_y - 1 and x < cx:
            c.px[y, x] = K[13]
    c.px[rim & ~shift(rim, 0, 1) & (X < cx + 6)] = K[14]
    c.px[inner] = K[0]
    c.px[inner & (Y < top_y - 0.5)] = K[1]
    c.hline(cx - 3, cx + 1, top_y + 1, K[4])     # moon on the water
    c.set(cx + 3, top_y + 1, K[3])
    # ---- posts
    for px, lit in ((8, True), (37, False)):
        c.rect(px, 13, 3, top_y - 12, V[5] if lit else V[4])
        c.vline(px, 13, top_y, V[7] if lit else V[5])
        c.vline(px + 2, 13, top_y, V[2])
        c.hline(px, px + 2, top_y + 1, V[2])
    # ---- beam, rope and bucket
    c.rect(6, 17, 37, 3, V[4])
    c.hline(6, 42, 17, V[6])
    c.hline(6, 42, 19, V[1])
    c.rect(41, 16, 3, 5, V[3])            # crank hub
    c.line(44, 18, 46, 23, V[5])          # crank handle
    c.set(46, 24, V[6])
    c.vline(cx, 20, 30, K[10])            # rope
    c.vline(cx + 1, 21, 29, K[6])
    bk = poly_mask(W, H, [(cx - 4, 30), (cx + 5, 30), (cx + 4, 37), (cx - 3, 37)])
    c.px[bk] = V[4]
    c.px[bk & ~shift(bk, 1, 0)] = V[6]
    c.hline(cx - 4, cx + 4, 30, V[7])
    c.hline(cx - 3, cx + 4, 33, V[2])
    c.px[bk & ~shift(bk, 0, -1)] = V[2]
    # ---- little roof (side gable, planks), one corner broken
    roof = poly_mask(W, H, [(1, 16), (47, 16), (41, 4), (7, 4)])
    M.planks(c, roof, Rng(4), M.WOOD_LIT, width=4, vertical=True, grain=0.25)
    c.hline(7, 40, 4, V[8])
    c.hline(7, 40, 5, V[7])
    c.hline(1, 46, 16, V[1])
    c.hline(1, 46, 15, V[3])
    brk = poly_mask(W, H, [(34, 3), (48, 3), (48, 17), (44, 17), (42, 12), (38, 11), (36, 7)])
    c.px[brk & roof] = -1
    c.line(36, 7, 41, 16, V[8])
    c.px[S.rect(W, H, 37, 13, 3, 3) & brk] = -1
    # a loose plank hanging from the gap
    c.line(40, 12, 44, 22, V[5])
    c.line(41, 12, 45, 22, V[2])
    M.selective_outline(c, light=K[3], dark=K[1])
    return c


# ================================================================ statue
def shade_runs(c, mask, ramp):
    """Cylindrical shading per horizontal run: lit left edge, dark right edge."""
    h, w = mask.shape
    for y in range(h):
        x = 0
        while x < w:
            if not mask[y, x]:
                x += 1
                continue
            x0 = x
            while x < w and mask[y, x]:
                x += 1
            x1 = x - 1
            n = x1 - x0 + 1
            for i in range(n):
                t = (i + 0.5) / n
                if n <= 2:
                    col = ramp[3] if i == 0 else ramp[1]
                elif t < 0.2:
                    col = ramp[4]
                elif t < 0.45:
                    col = ramp[3]
                elif t < 0.72:
                    col = ramp[2]
                elif t < 0.9:
                    col = ramp[1]
                else:
                    col = ramp[0]
                c.px[y, x0 + i] = col


def statue():
    """'The Hollow Saint': a hunched figure with a tall pierced head."""
    W, H = 48, 80
    c = Canvas(W, H)
    # ---- pedestal (3/4 box): top face, cornice, front face, plinth
    x0, x1 = 9, 38
    c.rect(x0, 50, x1 - x0 + 1, 7, K[11])
    c.hline(x0, x1, 50, K[13])
    c.vline(x0, 50, 56, K[12])
    c.vline(x1, 51, 56, K[9])
    c.rect(x0 - 2, 57, x1 - x0 + 5, 3, K[12])
    c.hline(x0 - 2, x1 + 2, 57, K[14])
    c.hline(x0 - 2, x1 + 2, 59, K[7])
    front = S.rect(W, H, x0, 60, x1 - x0 + 1, 13)
    M.stone_blocks(c, front, Rng(9), M.STONE_LIT, row_h=7, bw=(10, 16), y0=60 % 7)
    c.vline(x0, 60, 72, K[12])
    c.vline(x1, 60, 72, K[8])
    c.vline(x1 - 1, 60, 72, K[9])
    c.rect(16, 62, 16, 7, K[8])          # plaque
    c.rect(17, 63, 14, 5, K[6])
    for (yy, a, b) in ((64, 19, 29), (66, 20, 27)):
        for xx in range(a, b, 2):
            c.set(xx, yy, K[9])
    c.hline(16, 31, 62, K[5])
    c.hline(16, 31, 69, K[12])
    c.rect(x0 - 3, 73, x1 - x0 + 7, 4, K[10])
    c.hline(x0 - 3, x1 + 3, 73, K[12])
    c.hline(x0 - 3, x1 + 3, 76, K[6])
    c.vline(x1 + 3, 73, 76, K[7])
    # ---- figure (built from shapes, shaded like rounded stone)
    X, Y = S.grid(W, H)
    cx = 24
    head = ((X - cx) / 6.2) ** 2 + ((Y - 13) / 9.5) ** 2 <= 1
    hole = ((X - cx) / 2.4) ** 2 + ((Y - 11.5) / 4.6) ** 2 <= 1
    neck = S.rect(W, H, cx - 2, 20, 5, 6)
    torso = poly_mask(W, H, [(cx - 5, 24), (cx + 5, 24), (cx + 9, 32), (cx - 9, 32)])
    limbL = poly_mask(W, H, [(cx - 9, 29), (cx - 3, 29), (cx - 5, 52), (cx - 13, 52)])
    limbR = poly_mask(W, H, [(cx + 3, 29), (cx + 9, 29), (cx + 13, 52), (cx + 5, 52)])
    knee = ((X - cx) / 5) ** 2 + ((Y - 44) / 6.5) ** 2 <= 1
    gap = poly_mask(W, H, [(cx - 3, 33), (cx + 3, 33), (cx + 4, 38), (cx - 4, 38)])
    body = (head | neck | torso | limbL | limbR | knee) & ~hole
    ramp = [K[8], K[10], K[11], K[12], K[13]]
    shade_runs(c, body & ~knee, ramp)
    shade_runs(c, knee & ~limbL & ~limbR, [K[7], K[9], K[10], K[11], K[11]])
    c.px[gap] = K[2]
    c.px[gap & (Y > 36)] = K[4]
    # top-lit highlights
    top = body & ~shift(body, 0, 1)
    c.px[top & (X < cx + 3)] = K[14]
    # inner rim of the pierced head (lower lip lit, upper lip dark)
    hr = dilate(hole) & body
    c.px[hr & (Y < 11.5)] = K[7]
    c.px[hr & (Y >= 11.5)] = K[12]
    # separation lines between limbs and knee
    sep = (dilate(limbL) | dilate(limbR)) & knee & ~limbL & ~limbR
    c.px[sep & (Y > 38)] = K[6]
    # moss on the lower robe + pedestal
    r = Rng(5)
    for _ in range(30):
        x = r.randint(12, 36)
        y = r.randint(44, 52) if r.chance(0.6) else r.randint(60, 76)
        if c.get(x, y) >= 0 and not gap[y, x]:
            c.set(x, y, K[8] if r.chance(0.6) else K[7])
    # a crack down the head
    for (x, y) in ((cx + 4, 6), (cx + 4, 7), (cx + 5, 8), (cx + 5, 9), (cx + 4, 10)):
        c.set(x, y, K[8])
    M.selective_outline(c, light=K[3], dark=K[1])
    # hole is see-through: keep it transparent (outline inside it = dark ring)
    return c


# ================================================================ signpost
def signpost():
    W, H = 32, 48
    c = Canvas(W, H)
    # post
    c.rect(14, 8, 4, 38, V[5])
    c.vline(14, 8, 45, V[7])
    c.vline(17, 8, 45, V[2])
    c.vline(16, 12, 44, V[4])
    c.hline(14, 17, 8, V[8])
    # main board: arrow pointing right
    b = poly_mask(W, H, [(3, 11), (25, 11), (30, 16), (25, 21), (3, 21)])
    M.planks(c, b, Rng(3), M.WOOD_LIT, width=5, vertical=False, grain=0.3, gaps=True)
    c.px[b & ~shift(b, 0, 1)] = V[8]
    c.px[b & ~shift(b, 1, 0)] = V[7]
    c.px[b & ~shift(b, 0, -1)] = V[2]
    # carved (illegible) letters
    for x in (7, 9, 12, 13, 16, 19, 21):
        c.vline(x, 14, 15 + (x % 3 == 0), V[2])
    c.hline(7, 10, 17, V[2])
    c.set(15, 13, V[6])
    c.set(15, 18, V[3])            # nail
    # lower broken board hanging askew, pointing left
    b2 = poly_mask(W, H, [(4, 26), (9, 23), (21, 27), (21, 32), (9, 29), (6, 30)])
    M.planks(c, b2, Rng(5), M.WOOD, width=5, vertical=False, grain=0.3)
    c.px[b2 & ~shift(b2, 0, 1)] = V[6]
    c.px[b2 & ~shift(b2, 0, -1)] = V[1]
    c.set(16, 29, V[1])
    M.selective_outline(c, light=K[3], dark=K[1])
    return c


# ================================================================ fence
def _picket(c, x, top, bottom, w=4, lean=0, broken=False):
    m = S.rect(c.w, c.h, x, top + 2, w, bottom - top - 2)
    cap = S.rect(c.w, c.h, x + 1, top, w - 2, 2) | S.rect(c.w, c.h, x, top + 1, w, 1)
    m |= cap
    if broken:
        m &= ~poly_mask(c.w, c.h, [(x - 1, top - 1), (x + w + 1, top - 1), (x + w + 1, top + 5), (x, top + 3)])
    if lean:
        out = np.zeros_like(m)
        for y in range(c.h):
            s = int(round((bottom - y) * lean / 10.0))
            if s:
                out[y] = np.roll(m[y], s)
            else:
                out[y] = m[y]
        m = out
    c.px[m] = V[5]
    c.px[m & ~shift(m, 1, 0)] = V[7]
    c.px[m & ~shift(m, -1, 0)] = V[3]
    c.px[m & ~shift(m, 0, 1)] = V[8]
    # grain / knot
    c.set(x + 2, top + 8, V[4])
    c.set(x + 2, top + 9, V[4])
    return m


def fence(kind="mid", seed=0):
    """16x32 fence tile; the fence stands on the bottom of the tile."""
    W, H = 16, 32
    c = Canvas(W, H)
    r = Rng(700 + seed)
    bottom = 30
    rails = (18, 25)
    # rails first (pickets overlap them)
    xa = {"mid": 0, "left": 3, "right": 0, "broken": 0, "post": 6}[kind]
    xb = {"mid": 15, "left": 15, "right": 12, "broken": 15, "post": 9}[kind]
    for ry in rails:
        if kind == "broken" and ry == rails[0]:
            # snapped top rail hanging down
            c.hline(0, 6, ry, V[4])
            c.line(7, ry, 14, ry + 5, V[4])
            c.line(7, ry - 1, 14, ry + 4, V[6])
            c.hline(0, 6, ry - 1, V[6])
            continue
        if kind == "post":
            continue
        c.rect(xa, ry, xb - xa + 1, 2, V[4])
        c.hline(xa, xb, ry, V[6])
        c.hline(xa, xb, ry + 2, V[1])
    tops = [12 + r.randint(0, 2), 12 + r.randint(0, 2)]
    if kind == "post":
        _picket(c, 6, 10, bottom, w=5)
    elif kind == "broken":
        _picket(c, 2, tops[0] + 4, bottom, broken=True)
        # second picket missing: stump only
        _picket(c, 10, 24, bottom)
    else:
        _picket(c, 2, tops[0], bottom, lean=(1 if seed % 3 == 1 else 0))
        _picket(c, 10, tops[1], bottom, lean=(-1 if seed % 3 == 2 else 0))
    # outline (only sides/top: fences tile horizontally, so no outline on the
    # left/right tile border where the rails continue)
    m = c.opaque()
    e = dilate(m) & ~m
    if kind in ("mid", "broken"):
        e[:, 0] = False
        e[:, 15] = False
    if kind == "left":
        e[:, 15] = False
    if kind == "right":
        e[:, 0] = False
    c.px[e] = V[0]
    # little grass at the foot
    for x in (1, 5, 9, 13):
        if r.chance(0.6):
            c.set(x, bottom, K[8])
            c.set(x + 1, bottom - 1, K[9])
    return c


def fence_v(kind="mid", seed=0):
    """Vertical run of fence (seen from the front: posts in a column)."""
    W, H = 16, 16
    c = Canvas(W, H)
    # side rail running north-south
    c.rect(7, 0, 2, 16, V[4])
    c.vline(7, 0, 15, V[6])
    c.vline(9, 0, 15, V[1])
    if kind in ("mid", "bottom"):
        m = S.rect(W, H, 6, 2, 4, 12)
        c.px[m] = V[5]
        c.vline(6, 2, 13, V[7])
        c.vline(9, 2, 13, V[3])
        c.hline(6, 9, 2, V[8])
        c.hline(6, 9, 13, V[1])
    return c


# ================================================================ barrels & co.
def barrel(kind="open"):
    W, H = 16, 32
    c = Canvas(W, H)
    x0, x1, top, bot = 1, 14, 12, 30
    rx = (x1 - x0 + 1) / 2
    cx = x0 + rx
    X, Y = S.grid(W, H)
    body = (X >= x0) & (X <= x1 + 1) & (Y >= top + 2) & (Y <= bot - 1)
    body |= ((X - cx) / rx) ** 2 + ((Y - (bot - 2)) / 2.5) ** 2 <= 1
    for y, x in zip(*np.nonzero(body)):
        c.px[y, x] = _cyl_shade(x, x0, x1 + 1, (V[2], V[3], V[4], V[6]))
    # staves
    for sx in (4, 7, 10, 13):
        for y in range(top + 3, bot - 1):
            if body[y, sx]:
                c.px[y, sx] = V[2] if sx > 8 else V[3]
    # hoops
    for hy in (top + 5, bot - 5):
        for x in range(x0, x1 + 2):
            if body[hy, x]:
                c.px[hy, x] = K[5] if x > 9 else K[7]
                if body[hy + 1, x]:
                    c.px[hy + 1, x] = K[3]
    # top
    topm = ((X - cx) / rx) ** 2 + ((Y - (top + 2)) / 3) ** 2 <= 1
    c.px[topm] = V[6]
    inner = ((X - cx) / (rx - 1.5)) ** 2 + ((Y - (top + 2.2)) / 1.8) ** 2 <= 1
    if kind == "open":
        c.px[inner] = K[1]
        c.px[inner & (Y > top + 2.5)] = K[3]         # dark water
        c.set(int(cx) - 1, top + 3, K[6])
        c.px[topm & ~inner & (Y < top + 2)] = V[8]
    elif kind == "closed":
        c.px[inner] = V[5]
        for x in (5, 8, 11):
            for y in range(top, top + 5):
                if inner[y, x]:
                    c.px[y, x] = V[3]
        c.px[topm & ~shift(topm, 0, 1)] = V[8]
    M.selective_outline(c, light=K[3], dark=K[1])
    return c


def barrel_broken():
    """A barrel lying on its side, lid gone, a couple of staves sprung loose."""
    W, H = 16, 32
    c = Canvas(W, H)
    x0, x1, y0 = 1, 11, 20
    bands = [V[6], V[5], V[5], V[4], V[4], V[3], V[2], V[2]]
    for i, col in enumerate(bands):
        w0 = 1 if i in (0, 7) else 0
        c.hline(x0 + w0, x1, y0 + i, col)
    for x in (4, 9):                       # hoops
        c.vline(x, y0, y0 + 7, K[5])
        c.set(x, y0, K[9])
    for x in range(x0 + 1, x1, 2):
        c.set(x, y0 + 3, V[3])
    # open end (the staves' rim + the dark inside)
    end = ((S.grid(W, H)[0] - (x1 + 1)) / 3.2) ** 2 + ((S.grid(W, H)[1] - (y0 + 4)) / 4.4) ** 2 <= 1
    inner = ((S.grid(W, H)[0] - (x1 + 1.4)) / 2) ** 2 + ((S.grid(W, H)[1] - (y0 + 4)) / 3.2) ** 2 <= 1
    c.px[end] = V[6]
    c.px[end & (S.grid(W, H)[1] > y0 + 5)] = V[4]
    c.px[inner] = K[1]
    c.px[inner & (S.grid(W, H)[1] > y0 + 5)] = K[3]
    # a loose stave on the ground
    c.line(2, 30, 9, 29, V[5])
    c.line(2, 31, 9, 30, V[2])
    M.selective_outline(c, light=K[3], dark=K[1])
    return c


def stump():
    W, H = 16, 32
    c = Canvas(W, H)
    X, Y = S.grid(W, H)
    body = (X >= 2) & (X <= 14) & (Y >= 21) & (Y <= 28)
    body |= ((X - 8) / 6.5) ** 2 + ((Y - 28) / 2.5) ** 2 <= 1
    # roots
    body |= poly_mask(W, H, [(0, 31), (3, 26), (5, 30)]) | poly_mask(W, H, [(16, 31), (12, 26), (11, 30)])
    for y, x in zip(*np.nonzero(body)):
        c.px[y, x] = _cyl_shade(x, 1, 15, (V[2], V[3], V[4], V[5]))
    for x in (4, 7, 11):
        c.vline(x, 23, 29, V[2])
    top = ((X - 8) / 6.5) ** 2 + ((Y - 21) / 3) ** 2 <= 1
    c.px[top] = K[11]
    c.px[S.ring(W, H, 8, 21, 2, 3.2) & top] = K[9]
    c.set(8, 21, K[8])
    c.px[top & ~shift(top, 0, -1)] = K[8]
    c.px[top & ~shift(top, 0, 1)] = K[12]
    # jagged splinters
    for (x, h) in ((4, 3), (6, 2), (11, 4), (12, 2)):
        c.vline(x, 21 - h, 20, K[12])
        c.set(x + 1, 20, V[3])
    M.selective_outline(c, light=K[3], dark=K[1])
    return c


CRATE = [
    "..............",
    ".GGHGGGHGGGHG.",
    ".FGGFGGGFGGGF.",
    ".CCCCCCCCCCCC.",
    ".EFFFFFFFFFFD.",
    ".EFGGGGGGGGFD.",
    ".E6FFFFFFFF6D.",
    ".ED6FFFFFF6CD.",
    ".EFFD6FFF6DFD.",
    ".EFFFD66DFFFD.",
    ".EFFF6DD6FFFD.",
    ".EFF6FFFFD6FD.",
    ".E6DFFFFFFD6D.",
    ".CCCCCCCCCCCC.",
]


def crate():
    W, H = 16, 32
    c = Canvas(W, H)
    c.blit(from_ascii(CRATE), 1, 17)
    M.selective_outline(c, light=K[3], dark=K[1])
    return c


# ================================================================ rocks
ROCK = [K[5], K[7], K[9], K[11], K[13]]     # shadow .. highlight


def rock(w, h, seed, ramp=ROCK):
    """A faceted boulder: Voronoi facets shaded as planes of an ellipsoid."""
    W, H = 16 * ((w + 17) // 16), 16 * ((h + 17) // 16)
    c = Canvas(W, H)
    r = Rng(seed)
    cx, cy = W / 2, H - h / 2 - 2.5
    m = M.jagged_blob(W, H, cx, cy, w / 2, h / 2, r, rough=0.16, n=9)
    m &= (np.arange(H)[:, None] < H - 3)
    X, Y = S.grid(W, H)
    u = np.clip((X - cx) / (w / 2), -1, 1)
    v = np.clip((Y - cy) / (h / 2), -1, 1)
    nz = np.sqrt(np.clip(1 - u * u - v * v, 0.05, 1))
    L = np.array([-0.55, -0.75, 0.55])
    L = L / np.linalg.norm(L)
    # facets
    ys, xs = np.nonzero(m)
    k = max(3, (w * h) // 40)
    seeds = [(xs[i], ys[i]) for i in (r.randint(0, len(xs) - 1) for _ in range(k))]
    lab = np.full((H, W), -1)
    for y, x in zip(ys, xs):
        lab[y, x] = int(np.argmin([(x - sx) ** 2 + (y - sy) ** 2 for (sx, sy) in seeds]))
    lits = []
    for i in range(k):
        sel = lab == i
        if not sel.any():
            continue
        n = np.array([u[sel].mean(), v[sel].mean(), nz[sel].mean()])
        n = n / np.linalg.norm(n)
        lits.append((float(n @ L), i))
    lits.sort(reverse=True)
    # rank-based tones: always one bright facet, one dark, the rest in between
    for rank, (lit, i) in enumerate(lits):
        t = rank / max(1, len(lits) - 1)
        tone = ramp[3] if t < 0.2 else ramp[2] if t < 0.6 else ramp[1]
        c.px[lab == i] = tone
    # specular spot on the brightest facet
    sel = lab == lits[0][1]
    yb, xb = np.nonzero(sel)
    j = int(np.argmin(xb + yb))
    c.px[yb[j]:yb[j] + 1, xb[j]:xb[j] + 2] = ramp[4]
    # crisp bottom: the underside is always dark
    c.px[m & ~shift(m, 0, -1)] = ramp[0]
    c.px[m & ~shift(m, 0, -2) & (v > 0.3)] = ramp[0] if w < 14 else ramp[1]
    # top rim catches the moon
    top = m & ~shift(m, 0, 1) & (u < 0.35)
    c.px[top] = ramp[4] if w >= 10 else ramp[3]
    M.selective_outline(c, light=K[3], dark=K[1])
    from palette import SH1, SH3
    base_y = int(np.nonzero(m.any(axis=1))[0].max()) + 2
    sh = S.rect(W, H, int(cx - w / 2) + 2, base_y, int(w), 1) & (c.px < 0)
    c.px[sh] = SH1
    sh2 = S.rect(W, H, int(cx - w / 2) + 4, base_y - 1, int(w), 1) & (c.px < 0) & (X > cx)
    c.px[sh2] = SH3
    return c


PEBBLES = [
    ["dc.", "b9a", ".66"],
    ["dc", "b9", "66"],
    [".dcb", "dbb9", "b99a", ".666"],
    ["c", "9", "6"],
    ["dcb", "99a", "666"],
]


def pebbles(seed):
    c = Canvas(16, 16)
    r = Rng(seed)
    for i in range(r.randint(2, 3)):
        p = from_ascii(PEBBLES[r.randint(0, len(PEBBLES) - 1)])
        c.blit(p, r.randint(1, 12), r.randint(3, 12))
    return c


# ================================================================ grass tufts
# tufts: dark spiky blades (read well on the grass) with pale moonlit tips
TUFTS = [
    [".a.....a.",
     ".6..a..6.",
     "..5.6.5..",
     "..4.5.4..",
     "...454...",
     "..34543..",
     ".2345432.",
     "..11111.."],
    ["...a.....",
     "...6..a..",
     ".a.5..6..",
     ".6.4.5...",
     "..44.4...",
     "..4554...",
     ".234432..",
     "..1111..."],
    ["a..a...a..a",
     "6..6.a.6..6",
     ".5.5.6.5.5.",
     ".454.5.454.",
     "..455554...",
     ".34554543..",
     "..2343432..",
     "...11111..."],
    ["..bb....",
     "..ba....",
     "...8....",
     "...5..9.",
     ".9.5..5.",
     ".5.4.4..",
     "..454...",
     ".34543..",
     "..1111.."],
    [".9.a.",
     ".5.6.",
     "9565.",
     "45454",
     "23432",
     ".111."],
    [".a...a...",
     "..6.6..a.",
     "9..5...6.",
     ".5.4.5.5.",
     "..4444...",
     ".8.454.8.",
     "..34543..",
     "...1111.."],
    [".d...d..",
     ".5.c.5..",
     "..565...",
     ".9.5.9..",
     "..454...",
     ".23432..",
     "..111..."],
    ["a.......",
     ".6...a..",
     ".5..6...",
     "..4.5...",
     "..454.a.",
     "...446..",
     "..3454..",
     ".234432.",
     "..1111.."],
]


def tuft(i, ox=None, oy=None):
    c = Canvas(16, 16)
    t = from_ascii(TUFTS[i % len(TUFTS)])
    x = (16 - t.w) // 2 if ox is None else ox
    y = 15 - t.h if oy is None else oy
    c.blit(t, x, y)
    return c


# ================================================================ graves & ruins
def gravestone(kind=0):
    W, H = 16, 32
    c = Canvas(W, H)
    if kind == 0:     # rounded headstone
        m = S.round_arch(W, H, 3, 13, 17, 30)
    elif kind == 1:   # cross
        m = S.rect(W, H, 7, 12, 3, 18) | S.rect(W, H, 3, 16, 11, 3)
    else:             # broken slab, tilted
        m = poly_mask(W, H, [(3, 19), (9, 16), (13, 19), (13, 30), (3, 30)])
    c.px[m] = K[11]
    c.px[m & ~shift(m, 1, 0)] = K[13]
    c.px[m & ~shift(m, 0, 1)] = K[14]
    c.px[m & ~shift(m, -1, 0)] = K[8]
    if kind == 0:
        c.hline(6, 10, 21, K[8])
        c.hline(6, 9, 23, K[8])
        c.hline(6, 10, 25, K[8])
    # mound
    c.rect(2, 29, 12, 2, K[8])
    c.hline(2, 13, 29, K[9])
    M.selective_outline(c, light=K[3], dark=K[1])
    return c


def pillar_ruin(h=40, seed=0):
    W, H = 16, 48
    c = Canvas(W, H)
    top = H - 3 - h
    col = S.rect(W, H, 4, top, 8, h)
    M.stone_blocks(c, col, Rng(seed), M.STONE_LIT, row_h=6, bw=(8, 8), y0=(H - 3) % 6, stagger=False)
    c.vline(4, top, H - 4, K[13])
    c.vline(11, top, H - 4, K[7])
    r = Rng(seed + 1)
    for dx in range(8):
        d = r.randint(0, 5)
        c.vline(4 + dx, top, top + d, -1)
        c.set(4 + dx, top + d + 1, K[14] if dx < 6 else K[10])
    c.rect(3, H - 4, 10, 3, K[10])
    c.hline(3, 12, H - 4, K[12])
    M.selective_outline(c, light=K[3], dark=K[1])
    return c


def fallen_column():
    """A toppled column drum lying on its side (horizontal cylinder in 3/4 view)."""
    W, H = 32, 16
    c = Canvas(W, H)
    x0, x1, y0 = 3, 26, 5
    bands = [K[13], K[12], K[12], K[11], K[10], K[9], K[8]]
    for i, col in enumerate(bands):
        c.hline(x0, x1, y0 + i, col)
    # drum joints and fluting
    for x in (10, 18):
        c.vline(x, y0, y0 + 6, K[8])
        c.vline(x + 1, y0, y0 + 6, K[13] if False else K[12])
    for x in range(x0 + 2, x1, 3):
        c.set(x, y0 + 3, K[10])
    # the intact end: circular cross-section
    end = S.disc(W, H, x1 + 1.5, y0 + 3.5, 3.6)
    c.px[end] = K[12]
    c.px[S.ring(W, H, x1 + 1.5, y0 + 3.5, 2.4, 3.6) & (np.arange(H)[:, None] < y0 + 3)] = K[14]
    c.px[S.ring(W, H, x1 + 1.5, y0 + 3.5, 2.4, 3.6) & (np.arange(H)[:, None] >= y0 + 4)] = K[9]
    c.set(x1 + 1, y0 + 3, K[10])
    # the snapped end: jagged
    for (y, d) in zip(range(y0, y0 + 7), (2, 0, 1, 3, 1, 0, 2)):
        for x in range(x0, x0 + d):
            c.set(x, y, -1)
        c.set(x0 + d, y, K[14] if y < y0 + 3 else K[10])
    M.selective_outline(c, light=K[3], dark=K[1])
    from palette import SH1
    c.px[S.rect(W, H, x0 + 1, y0 + 8, x1 - x0 + 4, 2) & (c.px < 0)] = SH1
    return c


if __name__ == "__main__":
    import sys
    well().save(sys.argv[1] + "/well.png", 6)
