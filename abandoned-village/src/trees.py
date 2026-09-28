"""Dead, leafless trees grown with a small recursive branching model."""
import math
import numpy as np

from palette import K, V
from pixelkit import Canvas, Rng, dilate, shift
import materials as M

BARK = [V[4], V[6], K[11], K[12], K[13]]    # violet shadows .. cool moonlit highlights


class Tree:
    def __init__(self, w, h, seed, trunk_w=6.5, trunk_len=0.42, spread=0.9, depth=5,
                 lean=0.0, branchiness=1.0, tip_len=0.62):
        self.w, self.h = w, h
        self.r = Rng(seed)
        self.mask = np.zeros((h, w), dtype=bool)
        self.rad = np.zeros((h, w))             # branch radius at each pixel
        self.tips = []
        self.trunk_w = trunk_w
        self.trunk_len = trunk_len
        self.spread = spread
        self.depth = depth
        self.lean = lean
        self.branchiness = branchiness
        self.tip_len = tip_len

    def _dab(self, x, y, r):
        ri = int(math.ceil(r))
        for dy in range(-ri, ri + 1):
            for dx in range(-ri, ri + 1):
                px, py = int(math.floor(x + dx)), int(math.floor(y + dy))
                if 0 <= px < self.w and 0 <= py < self.h:
                    if (px + 0.5 - x) ** 2 + (py + 0.5 - y) ** 2 <= r * r + 0.25:
                        self.mask[py, px] = True
                        self.rad[py, px] = max(self.rad[py, px], r)

    def _walk(self, x, y, ang, length, w0, w1):
        r = self.r
        steps = max(2, int(length))
        pts = []
        for i in range(steps):
            t = i / steps
            wr = w0 + (w1 - w0) * t
            if wr < 1.2:
                ix, iy = int(math.floor(x)), int(math.floor(y))
                if 0 <= ix < self.w and 0 <= iy < self.h:
                    self.mask[iy, ix] = True
                    self.rad[iy, ix] = max(self.rad[iy, ix], 0.5)
            else:
                self._dab(x, y, wr / 2)
            pts.append((x, y, ang, wr))
            ang += r.uniform(-0.07, 0.07)
            ang += (-math.pi / 2 - ang) * 0.02          # gently curve upward
            if x < 4:
                ang += 0.08
            if x > self.w - 5:
                ang -= 0.08
            x += math.cos(ang)
            y += math.sin(ang)
        return x, y, ang, pts

    def branch(self, x, y, ang, length, width, depth):
        r = self.r
        if depth == 0:                                   # trunk
            x, y, ang, pts = self._walk(x, y, ang, length, width, width * 0.7)
            # an early side limb sometimes
            if r.chance(0.6 * self.branchiness):
                px, py, pa, pw = pts[int(len(pts) * r.uniform(0.55, 0.8))]
                side = r.choice([-1, 1])
                self.branch(px, py, pa + side * r.uniform(0.7, 1.0) * self.spread,
                            length * r.uniform(0.45, 0.6), pw * 0.5, 1)
            n = r.randint(2, 3)
            offs = [-0.55, 0.45] if n == 2 else [-0.7, 0.05, 0.6]
            for o in offs:
                self.branch(x, y, ang + (o + r.uniform(-0.12, 0.12)) * self.spread,
                            length * r.uniform(0.75, 1.05) * self.tip_len / 0.62, width * 0.55, 1)
        elif depth == 1:                                 # limbs
            x, y, ang, pts = self._walk(x, y, ang, length, width, max(1.0, width * 0.45))
            k = r.randint(1, 2) if self.branchiness >= 1 else r.randint(0, 1)
            for _ in range(k):
                px, py, pa, pw = pts[int(len(pts) * r.uniform(0.35, 0.8))]
                side = r.choice([-1, 1])
                self.branch(px, py, pa + side * r.uniform(0.6, 0.95) * self.spread,
                            length * r.uniform(0.3, 0.45), max(1.0, pw * 0.55), 2)
            self._fork(x, y, ang, length * 0.22)
        else:                                            # twigs
            x, y, ang, pts = self._walk(x, y, ang, length, width, 1.0)
            self._fork(x, y, ang, length * 0.35)

    def _fork(self, x, y, ang, ln):
        r = self.r
        for side in (-1, 1):
            if r.chance(0.2):
                continue
            ex, ey, _, _ = self._walk(x, y, ang + side * r.uniform(0.3, 0.6), max(2, ln * r.uniform(0.7, 1.2)), 1.0, 1.0)
            self.tips.append((int(ex), int(ey)))

    def grow(self):
        bx, by = self.w / 2 + self.lean * 3, self.h - 3
        # root flare
        for (dx, dy) in ((-5, 1.5), (5, 1.5), (-3, 2.5), (3, 2.5)):
            for i in range(5):
                t = i / 5
                self._dab(bx + dx * t, by + dy * t - 1, (self.trunk_w / 2) * (1 - t) + 0.5)
        ang = -math.pi / 2 + self.lean * 0.12
        self.branch(bx, by, ang, self.h * self.trunk_len, self.trunk_w, 0)
        return self

    def render(self):
        c = Canvas(self.w, self.h)
        m = self.mask
        thick = self.rad >= 1.4
        # base mid tone
        c.px[m] = BARK[2]
        # thick parts: cylinder shading per row-run (lit left, shaded right)
        for y in range(self.h):
            xs = np.where(thick[y])[0]
            if len(xs) == 0:
                continue
            runs = np.split(xs, np.where(np.diff(xs) != 1)[0] + 1)
            for run in runs:
                n = len(run)
                for i, x in enumerate(run):
                    t = (i + 0.5) / n
                    if t < 0.22:
                        col = BARK[4]
                    elif t < 0.45:
                        col = BARK[3]
                    elif t < 0.78:
                        col = BARK[2]
                    else:
                        col = BARK[1]
                    c.px[y, x] = col
        # thin parts: light on the top-left side
        thin = m & ~thick
        tl = thin & (~shift(m, 1, 0) | ~shift(m, 0, 1))
        c.px[thin] = BARK[2]
        c.px[tl] = BARK[3]
        br = thin & (~shift(m, -1, 0) | ~shift(m, 0, -1)) & ~tl
        c.px[br] = BARK[1]
        # bark streaks on the trunk
        r = Rng(99)
        ys, xs = np.nonzero(self.rad >= 2.4)
        for _ in range(len(ys) // 14):
            i = r.randint(0, len(ys) - 1)
            y, x = ys[i], xs[i]
            for d in range(r.randint(2, 5)):
                if 0 <= y + d < self.h and self.rad[y + d, x] >= 2.4:
                    c.px[y + d, x] = BARK[1]
        # knot hole
        if len(ys):
            i = r.randint(0, len(ys) - 1)
            y, x = ys[i], xs[i]
            if self.rad[y, x] >= 3:
                c.set(x, y, K[1])
                c.set(x, y + 1, K[2])
                c.set(x - 1, y, BARK[1])
        # pale tips
        for (x, y) in self.tips:
            if 0 <= x < self.w and 0 <= y < self.h and m[y, x]:
                c.px[y, x] = BARK[4]
        M.selective_outline(c, light=K[3], dark=K[2])
        return c


PRESETS = {
    #            w    h   seed trunk_w trunk_len spread depth lean branchiness
    "tall":    (64, 144, 11, 7.5, 0.42, 0.85, 3, -0.3, 1.0),
    "crooked": (64, 128, 23, 7.0, 0.40, 0.95, 3, 0.5, 1.0),
    "medium":  (48, 96, 37, 6.0, 0.42, 0.85, 3, 0.2, 1.0),
    "small":   (48, 80, 41, 5.0, 0.42, 0.85, 3, -0.3, 0.8),
    "thin":    (32, 96, 53, 4.5, 0.50, 0.55, 3, 0.0, 0.6),
    "wide":    (64, 112, 67, 7.0, 0.38, 1.05, 3, 0.3, 1.1),
}


def tree(name):
    w, h, seed, tw, tl, sp, d, lean, br = PRESETS[name]
    return Tree(w, h, seed, tw, tl, sp, d, lean, br).grow().render()


def tree_by(w, h, seed, **kw):
    return Tree(w, h, seed, **kw).grow().render()
