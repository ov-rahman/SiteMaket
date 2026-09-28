"""Swamp trees: giant buttressed cypresses, smaller cypresses and dead snags.

A tree is returned as separate layers in the same coordinate frame:
    trunk   - roots + trunk + limbs (object layer, y-sorted with the player)
    canopy  - foliage pads + hanging moss (drawn above the player)
    shadow  - soft dappled canopy shadow on the ground (semi-transparent)
"""
import math
import numpy as np

from palette import G, W, B, SH1, SH2, SH3
from pixelkit import Canvas, Rng, dilate, erode, shift, value_noise, hash01, poly_mask

BARK_RAMP = [B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8]]     # dark .. light
LEAF_RAMP = [G[2], G[3], G[4], G[5], G[6], G[7], G[8], G[9]]
_B4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0


def bayer(h, w):
    return np.tile(_B4, (h // 4 + 1, w // 4 + 1))[:h, :w]


class TreeParts:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.trunk = Canvas(w, h)
        self.canopy = Canvas(w, h)
        self.shadow = Canvas(w, h)
        self.base_y = h
        self.foot = None            # collision (x, y, w, h)
        self.canopy_box = None      # (x0, y0, x1, y1) of the crown


# ---------------------------------------------------------------- helpers

def _bezier(p0, p1, p2, n):
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
        pts.append((x, y, t))
    return pts


def _tube(w, h, pts, r0, r1):
    """Rasterise a tapered tube; returns mask + per-pixel normal (nx, ny)."""
    mask = np.zeros((h, w), dtype=bool)
    nx = np.zeros((h, w))
    ny = np.zeros((h, w))
    best = np.full((h, w), 1e9)
    for (x, y, t) in pts:
        r = r0 + (r1 - r0) * t
        ri = int(math.ceil(r)) + 1
        y0, y1 = max(0, int(y) - ri), min(h, int(y) + ri + 1)
        x0, x1 = max(0, int(x) - ri), min(w, int(x) + ri + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx + 0.5 - x, yy + 0.5 - y
        q = (dx * dx + dy * dy) / (r * r)
        sel = (q <= 1) & (q < best[y0:y1, x0:x1])
        best[y0:y1, x0:x1][sel] = q[sel]
        mask[y0:y1, x0:x1][sel] = True
        nx[y0:y1, x0:x1][sel] = (dx / max(r, 0.5))[sel]
        ny[y0:y1, x0:x1][sel] = (dy / max(r, 0.5))[sel]
    return mask, nx, ny


def _shade_normals(c, mask, nx, ny, ramp, light=(-0.62, -0.78), bias=0.0, contrast=1.0):
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    lx, ly = light
    lz = 0.55
    ln = math.sqrt(lx * lx + ly * ly + lz * lz)
    lit = ((nx * lx + ny * ly + nz * lz) / ln) * contrast + bias
    idx = np.clip(((lit + 0.35) / 1.35) * (len(ramp) - 1), 0, len(ramp) - 1).astype(int)
    rv = np.array(ramp)
    c.px[mask] = rv[idx[mask]]


# ---------------------------------------------------------------- trunk

def _trunk(parts, cx, base_y, top_y, w_trunk, flare, lean, rng, n_roots=6, hollow=True, moss=1.0):
    W_, H_ = parts.w, parts.h
    c = parts.trunk
    height = base_y - top_y
    rows = {}
    body = np.zeros((H_, W_), dtype=bool)
    lobes = rng.uniform(0, 6.28)
    for y in range(top_y, base_y + 1):
        t = base_y - y
        k = t / max(1, height)
        w = w_trunk * (1.08 - 0.16 * k) + flare * math.exp(-t / 15.0)
        off = lean * t
        # buttress lobes bulge the silhouette near the ground
        lob = math.exp(-t / 10.0) * 7
        xl = cx - w / 2 + off - lob * (0.5 + 0.5 * math.sin(lobes))
        xr = cx + w / 2 + off + lob * (0.5 + 0.5 * math.cos(lobes))
        rows[y] = (xl, xr)
        a, b = max(0, int(math.floor(xl))), min(W_, int(math.ceil(xr)))
        body[y, a:b] = True
    # cylindrical shading
    for y, (xl, xr) in rows.items():
        a, b = max(0, int(math.floor(xl))), min(W_, int(math.ceil(xr)))
        xs = np.arange(a, b) + 0.5
        u = (xs - (xl + xr) / 2) / max(1, (xr - xl) / 2)
        lit = -0.8 * u + 0.2 * np.sqrt(np.clip(1 - u * u, 0, 1))
        i = np.clip(((lit + 0.8) / 1.6) * (len(BARK_RAMP) - 1) + 0.5, 0, len(BARK_RAMP) - 1).astype(int)
        c.px[y, a:b] = np.array(BARK_RAMP)[i]
    # flutes: grooves that fan out with the flare (constant relative position)
    flutes = sorted(rng.uniform(-0.85, 0.85) for _ in range(9))
    for fu in flutes:
        ph = rng.uniform(0, 6.28)
        for y, (xl, xr) in rows.items():
            x = int((xl + xr) / 2 + fu * (xr - xl) / 2 + math.sin(y * 0.05 + ph) * 1.5)
            if 0 <= x < W_ and body[y, x]:
                c.px[y, x] = B[2] if fu > -0.5 else B[4]
                if fu < 0.4 and 0 <= x - 1 < W_ and body[y, x - 1]:
                    c.px[y, x - 1] = B[7] if fu < -0.2 else B[6]
    # fibrous bark: short vertical strands
    for _ in range(int(height * w_trunk / 22)):
        y = rng.randint(top_y, base_y - 5)
        xl, xr = rows[y]
        x = int(rng.uniform(xl + 2, xr - 2))
        u = (x - (xl + xr) / 2) / max(1, (xr - xl) / 2)
        col = B[3] if u > 0.1 else (B[5] if u > -0.5 else B[6])
        for d in range(rng.randint(3, 8)):
            if y + d < H_ and 0 <= x < W_ and body[y + d, x]:
                c.px[y + d, x] = col
    # ---- buttress roots: fins that leave the trunk wide and taper into the water
    roots = []
    for i in range(n_roots):
        side = -1 if i % 2 == 0 else 1
        k = i // 2
        sy = base_y - rng.uniform(10, 26)
        xl, xr = rows[int(sy)]
        sx = (xl + xr) / 2 + side * (xr - xl) / 2 * rng.uniform(0.2, 0.7)
        reach = rng.uniform(26, 52) * (1 + 0.12 * k)
        ex = sx + side * reach
        ey = base_y + rng.uniform(2, 10) + k * 2
        mx, my = sx + side * reach * 0.35, ey - rng.uniform(2, 6)
        roots.append(((sx, sy), (mx, my), (ex, ey), rng.uniform(11, 16) * (w_trunk / 60)))
    for _ in range(2):  # roots running towards the viewer
        sx = cx + rng.uniform(-0.28, 0.28) * w_trunk
        sy = base_y - rng.uniform(6, 12)
        ex, ey = sx + rng.uniform(-20, 20), base_y + rng.uniform(12, 20)
        roots.append(((sx, sy), ((sx + ex) / 2, ey - 3), (ex, ey), rng.uniform(9, 12) * (w_trunk / 60)))
    roots.sort(key=lambda r: r[2][1])
    rmask = np.zeros((H_, W_), dtype=bool)
    for (p0, p1, p2, r0) in roots:
        n = int(math.hypot(p2[0] - p0[0], p2[1] - p0[1]) * 1.5) + 4
        pts = _bezier(p0, p1, p2, n)
        # radius falls off fast: fin-like
        pts2 = [(x, y, t ** 0.6) for (x, y, t) in pts]
        m, nx, ny = _tube(W_, H_, pts2, r0, 1.8)
        tmp = Canvas(W_, H_)
        _shade_normals(tmp, m, nx * 0.8, ny * 0.8, BARK_RAMP[1:7], contrast=0.9)
        # bark streaks running along the root
        for (x, y, t) in pts[2::3]:
            for off in (-0.4, 0.3):
                px_, py_ = int(x + off * 3), int(y - r0 * (1 - t ** 0.6) * 0.3)
                if 0 <= px_ < W_ and 0 <= py_ < H_ and m[py_, px_]:
                    tmp.px[py_, px_] = B[3]
        low = m & ~shift(m, 0, -1)
        tmp.px[low] = B[1]
        tmp.px[m & ~shift(m, 0, 1) & ~body] = B[7]
        c.blit(tmp, 0, 0)
        rmask |= m
    # hollow at the foot of the trunk
    if hollow:
        X, Y = np.meshgrid(np.arange(W_) + 0.5, np.arange(H_) + 0.5)
        hx = cx + rng.uniform(-0.2, 0.2) * w_trunk
        hw, hh = w_trunk * rng.uniform(0.16, 0.22), rng.uniform(16, 24)
        hm = (((X - hx) / hw) ** 2 + ((Y - base_y + 2) / hh) ** 2 <= 1) & (Y < base_y + 1) & body & ~rmask
        c.px[hm] = G[0]
        c.px[hm & (Y > base_y - hh * 0.45)] = B[0]
        rim = dilate(hm, True) & ~hm & (body | rmask)
        c.px[rim & (X < hx)] = B[7]
        c.px[rim & (X >= hx)] = B[2]
    # moss: coherent patches on the lit side and on top of roots
    mn = value_noise(W_, H_, 4, seed=rng.randint(0, 9999)) * 0.7 + value_noise(W_, H_, 1.5, seed=7) * 0.3
    solid = c.px >= 0
    X = np.arange(W_)[None, :]
    Yg = np.arange(H_)[:, None]
    lower = Yg > top_y + height * 0.4
    tops = solid & ~shift(solid, 0, 4)
    patches = solid & lower & (mn > 0.66 - 0.08 * moss) & ((X < cx + lean * (base_y - Yg)) | tops)
    c.px[patches] = G[5]
    c.px[patches & (mn > 0.74)] = G[6]
    c.px[patches & ~shift(patches, 0, 1)] = G[7]
    # silhouette outline
    solid = c.px >= 0
    c.px[dilate(solid) & ~solid] = G[0]
    parts.base_y = base_y
    parts.foot = (int(cx - w_trunk * 0.7), base_y - 20, int(w_trunk * 1.4), 24)
    return rows


def _limbs(parts, sx, sy, spec, rng):
    """Thick limbs from the trunk top into the crown (on the trunk layer)."""
    W_, H_ = parts.w, parts.h
    for (ex, ey, r0) in spec:
        mx = (sx + ex) / 2 + rng.uniform(-10, 10)
        my = min(sy, ey) + (max(sy, ey) - min(sy, ey)) * 0.45
        pts = _bezier((sx, sy), (mx, my), (ex, ey), 50)
        pts = [(x + math.sin(t * 9 + r0) * 1.2, y, t ** 0.8) for (x, y, t) in pts]
        m, nx, ny = _tube(W_, H_, pts, r0, 1.4)
        tmp = Canvas(W_, H_)
        _shade_normals(tmp, m, nx * 0.85, ny * 0.85, BARK_RAMP[:6], contrast=0.9)
        # bark: broken dark streaks along the limb
        for (x, y, t) in pts[3::4]:
            px_, py_ = int(x + rng.uniform(-1.5, 1.5)), int(y + rng.uniform(-1.5, 1.5))
            if 0 <= px_ < W_ and 0 <= py_ < H_ and m[py_, px_]:
                tmp.px[py_, px_] = B[1]
        tmp.px[m & ~erode(m)] = G[0]
        parts.trunk.blit(tmp, 0, 0)


# ---------------------------------------------------------------- crown

def _crown(parts, cx, cy, cw, ch, rng, moss=1.0, tiers=3):
    """Irregular, feathery crown: pads made of many small leaf clusters."""
    W_, H_ = parts.w, parts.h
    c = parts.canopy
    X, Y = np.meshgrid(np.arange(W_) + 0.5, np.arange(H_) + 0.5)
    # pad centres: an asymmetric spray of flat masses
    pads = []
    for t in range(tiers):
        f = t / max(1, tiers - 1)
        n = 2 + t
        yb = cy - ch * 0.38 + f * ch * 0.66
        for i in range(n):
            u = (i + rng.uniform(0.2, 0.8)) / n
            x = cx + (u - 0.5) * cw * (0.5 + 0.45 * f)
            pads.append((x, yb + rng.uniform(-10, 10), cw * rng.uniform(0.13, 0.2) * (0.8 + 0.4 * f),
                         ch * rng.uniform(0.12, 0.17)))
    # leaf clusters sampled inside every pad
    clusters = []
    for (px, py, prx, pry) in pads:
        k = int(prx * pry / 34) + 6
        for _ in range(k):
            a = rng.uniform(0, 2 * math.pi)
            r = math.sqrt(rng.rand())
            x = px + math.cos(a) * r * prx
            y = py + math.sin(a) * r * pry * 0.9
            rad = rng.uniform(4.5, 9.5) * (1.1 - 0.3 * r)
            clusters.append((x, y, rad, (y - py) / pry))
    clusters.sort(key=lambda q: q[1])
    crown = np.zeros((H_, W_), dtype=bool)
    ramp_lit = [G[3], G[4], G[5], G[6], G[7]]
    for (x, y, rad, vpos) in clusters:
        x0, x1 = max(0, int(x - rad - 2)), min(W_, int(x + rad + 3))
        y0, y1 = max(0, int(y - rad - 2)), min(H_, int(y + rad + 3))
        if x0 >= x1 or y0 >= y1:
            continue
        XX, YY = X[y0:y1, x0:x1], Y[y0:y1, x0:x1]
        ang = np.arctan2(YY - y, XX - x)
        wob = 1 + 0.18 * np.sin(ang * 5 + x) + 0.1 * np.sin(ang * 9 + y)
        m = ((XX - x) ** 2 + (YY - y) ** 2) <= (rad * wob) ** 2
        nx, ny = (XX - x) / rad, (YY - y) / rad
        gx, gy = (x - cx) / (cw * 0.5), (y - cy) / (ch * 0.5)
        lit = -(0.55 * nx + 0.85 * ny) - 0.35 * max(0.0, vpos) - 0.35 * gx - 0.3 * gy
        tone = np.where(lit > 0.55, 4, np.where(lit > 0.1, 3, np.where(lit > -0.4, 2, 1)))
        tone = np.where(vpos > 0.55, np.minimum(tone, 2), tone)
        reg = c.px[y0:y1, x0:x1]
        reg[m] = np.array(ramp_lit)[tone[m]]
        # dark rim under each cluster separates it from the one below
        low = m & ~np.roll(m, -1, axis=0)
        low[-1, :] = False
        reg[low] = G[2]
        crown[y0:y1, x0:x1] |= m
    # feathery edge: little needle tufts poking out of the silhouette
    edge = dilate(crown) & ~crown
    tuft = edge & (hash01(X.astype(int), Y.astype(int), 5) > 0.62)
    c.px[tuft & (Y < cy)] = G[5]
    c.px[tuft & (Y >= cy)] = G[3]
    crown |= tuft
    # sparse flecks: a few bright needles on the lit tops, dark holes in the mass
    fl = hash01(X.astype(int), Y.astype(int), 91)
    c.px[crown & (c.px == G[7]) & (fl > 0.8)] = G[8]
    c.px[crown & (c.px == G[5]) & (fl < 0.05)] = G[7]
    c.px[crown & (c.px == G[4]) & (fl < 0.04)] = G[1]
    # hanging moss beards
    bottoms = {}
    for x in range(W_):
        col = np.where(crown[:, x])[0]
        if len(col):
            bottoms[x] = col.max()
    xs = sorted(bottoms)
    nbeards = max(2, int(len(xs) / 34 * moss))
    placed = []
    for _ in range(nbeards * 4):
        if len(placed) >= nbeards:
            break
        x0 = rng.choice(xs)
        if any(abs(x0 - p) < 14 for p in placed):
            continue
        placed.append(x0)
        _beard(c, x0, bottoms, rng, W_, H_)
    solid = c.px >= 0
    out = dilate(solid) & ~solid
    c.px[out] = G[1]
    parts.canopy_box = (int(X[crown].min()), int(Y[crown].min()), int(X[crown].max()), int(Y[crown].max()))
    return crown


def _beard(c, x0, bottoms, rng, W_, H_):
    """A curtain of Spanish moss: soft grey-green strands, longest in the middle."""
    n = rng.randint(6, 11)
    ln_max = rng.randint(22, 44)
    for i in range(n):
        x = x0 + i - n // 2
        if x not in bottoms:
            continue
        mid = 1 - abs(i - (n - 1) / 2) / ((n + 1) / 2)
        ln = int(ln_max * (0.3 + 0.7 * mid ** 0.8) * rng.uniform(0.7, 1.1))
        y = bottoms[x] - rng.randint(2, 5)
        xx = x
        for k in range(ln):
            if not (0 <= y < H_ and 0 <= xx < W_):
                break
            t = k / max(1, ln)
            if t > 0.7 and ((k + i) % 2 == 1):
                y += 1
                continue
            shade = i / max(1, n - 1)                  # left strands lit, right in shade
            col = G[10] if shade < 0.45 else (G[9] if shade < 0.8 else G[8])
            if k % 6 == 3 and shade < 0.5:
                col = G[11]
            if t > 0.85:
                col = G[9]
            c.px[y, xx] = col
            y += 1
            if hash01(x, k, 17) > 0.88:
                xx += 1 if hash01(x, k, 19) > 0.5 else -1


def _crown_shadow(parts, cx, base_y, cw, rng, depth=0.2):
    W_, H_ = parts.w, parts.h
    X, Y = np.meshgrid(np.arange(W_) + 0.5, np.arange(H_) + 0.5)
    rx, ry = cw * 0.46, cw * depth
    d = ((X - cx - 6) / rx) ** 2 + ((Y - base_y + ry * 0.2) / ry) ** 2
    d = d + (value_noise(W_, H_, 10, seed=rng.randint(0, 9999)) - 0.5) * 0.45
    bay = bayer(H_, W_)
    s = parts.shadow
    s.px[(d < 1.0) & (bay < np.clip((1.0 - d) * 3.0, 0, 1))] = SH3
    s.px[(d < 0.6) & (bay < np.clip((0.6 - d) * 4, 0, 1))] = SH1
    holes = (value_noise(W_, H_, 3.5, seed=rng.randint(0, 9999)) > 0.74) & (d < 0.95)
    s.px[holes] = -1


# ---------------------------------------------------------------- presets

def giant_cypress(seed=1, w=352, h=432, crown_w=290, crown_h=180, trunk_w=64, flare=70,
                  trunk_h=165, lean=0.0, moss=1.0, roots=6):
    rng = Rng(seed)
    p = TreeParts(w, h)
    cx = w / 2
    base_y = h - 34
    top_y = base_y - trunk_h
    _trunk(p, cx, base_y, top_y, trunk_w, flare, lean, rng, n_roots=roots)
    tx = cx + lean * trunk_h
    ccy = top_y - crown_h * 0.22
    _limbs(p, tx, top_y + 18, [(tx - crown_w * 0.28, ccy + crown_h * 0.05, trunk_w * 0.2),
                               (tx + crown_w * 0.3, ccy + crown_h * 0.1, trunk_w * 0.2),
                               (tx - crown_w * 0.08, ccy - crown_h * 0.2, trunk_w * 0.18),
                               (tx + crown_w * 0.12, ccy - crown_h * 0.25, trunk_w * 0.15)], rng)
    _crown(p, tx, ccy, crown_w, crown_h, rng, moss=moss)
    _crown_shadow(p, cx, base_y, crown_w, rng)
    return p


def cypress(seed=5, w=176, h=224, crown_w=150, crown_h=90, trunk_w=22, flare=26, trunk_h=78, lean=0.0,
            moss=0.8):
    rng = Rng(seed)
    p = TreeParts(w, h)
    cx = w / 2
    base_y = h - 18
    top_y = base_y - trunk_h
    _trunk(p, cx, base_y, top_y, trunk_w, flare, lean, rng, n_roots=4, hollow=False)
    tx = cx + lean * trunk_h
    ccy = top_y - crown_h * 0.2
    _limbs(p, tx, top_y + 10, [(tx - crown_w * 0.25, ccy + 6, trunk_w * 0.22),
                               (tx + crown_w * 0.25, ccy + 4, trunk_w * 0.22)], rng)
    _crown(p, tx, ccy, crown_w, crown_h, rng, moss=moss, tiers=2)
    _crown_shadow(p, cx, base_y, crown_w, rng)
    return p
