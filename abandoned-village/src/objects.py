"""Registry of every multi-tile object: sprite + contact shadow + metadata.

Each object canvas is a multiple of 16 px on both axes so it can be sliced
straight into tiles. `foot` is the collision rectangle (local px) and
`base` the y used for depth sorting (the line where it touches the ground).
"""
import numpy as np

from palette import K, SH1, SH2, SH3
from pixelkit import Canvas, dilate, shift
import shapes as S
import materials as M
import church as CH
import houses as HS
import props as P
import trees as TR


class Obj:
    def __init__(self, name, canvas, foot, base=None, shadow=None, group="props", overhead_from=None):
        self.name = name
        self.c = canvas
        self.foot = foot                     # (x, y, w, h) local collision box, or None
        self.base = base if base is not None else (foot[1] + foot[3] if foot else canvas.h)
        self.group = group
        self.shadow = shadow
        assert canvas.w % 16 == 0 and canvas.h % 16 == 0, (name, canvas.w, canvas.h)

    @property
    def tw(self):
        return self.c.w // 16

    @property
    def th(self):
        return self.c.h // 16


def add_shadow(c, mask, strong=None, soft_ring=True):
    """Paint a semi-transparent shadow under the sprite (only on empty pixels)."""
    empty = c.px < 0
    if soft_ring:
        ring = dilate(mask, True) & ~mask
        c.px[ring & empty & M.dither_mask(ring, 0.5)] = SH3
    c.px[mask & empty] = SH1
    if strong is not None:
        c.px[strong & empty] = SH2
    return c


def _foot_shadow(c, x, y, w, h, dx=3, dy=2, grow=1):
    m = S.rect(c.w, c.h, x + dx, y + dy, w, h)
    for _ in range(grow):
        m = dilate(m)
    return m


def grow_canvas(c, w, h, ox=0, oy=0):
    n = Canvas(w, h)
    n.blit(c, ox, oy)
    return n


def build_all():
    objs = {}

    # ---------------------------------------------------------------- church
    c = CH.church()
    sh = S.poly(c.w, c.h, [(20, 192), (158, 192), (164, 200), (26, 202)])
    sh |= S.poly(c.w, c.h, [(CH.CX - 20, 212), (CH.CX + 28, 212), (CH.CX + 30, 216), (CH.CX - 18, 216)])
    for px0 in (1, 166):
        sh |= S.rect(c.w, c.h, px0 + 2, 199, 11, 4)
    add_shadow(c, sh)
    objs["church"] = Obj("church", c, foot=(14, 150, 150, 48), base=198, group="buildings")

    # ---------------------------------------------------------------- houses
    for name, fn, foot, sh_rect in (
            ("house_a", HS.house_a, (8, 70, 96, 36), (12, 102, 94, 6)),
            ("house_b", HS.house_b, (8, 70, 96, 36), (12, 102, 94, 6)),
            ("house_c", HS.house_c, (12, 76, 94, 30), (16, 102, 90, 6)),
            ("house_d", HS.house_d, (8, 86, 126, 36), (12, 118, 124, 6))):
        c = fn()
        x, y, w, h = sh_rect
        add_shadow(c, S.rect(c.w, c.h, x, y, w, h) | S.rect(c.w, c.h, x + w - 2, y - 30, 4, 34))
        objs[name] = Obj(name, c, foot=foot, base=foot[1] + foot[3], group="buildings")

    # ---------------------------------------------------------------- props
    c = P.well()
    X, Y = S.grid(c.w, c.h)
    add_shadow(c, ((X - 27) / 17) ** 2 + ((Y - 58) / 5) ** 2 <= 1)
    objs["well"] = Obj("well", c, foot=(6, 40, 36, 20), base=60)

    c = grow_canvas(P.statue(), 48, 80)
    X, Y = S.grid(c.w, c.h)
    add_shadow(c, S.rect(c.w, c.h, 9, 76, 34, 3) | S.rect(c.w, c.h, 42, 60, 3, 18))
    objs["statue"] = Obj("statue", c, foot=(6, 58, 36, 20), base=77)

    c = P.signpost()
    X, Y = S.grid(c.w, c.h)
    add_shadow(c, ((X - 18) / 6) ** 2 + ((Y - 46) / 2) ** 2 <= 1)
    objs["signpost"] = Obj("signpost", c, foot=(12, 40, 8, 6), base=46)

    # ---------------------------------------------------------------- trees
    for name in TR.PRESETS:
        c = TR.tree(name)
        X, Y = S.grid(c.w, c.h)
        cx = c.w / 2 + TR.PRESETS[name][7] * 3
        add_shadow(c, ((X - cx - 3) / 9) ** 2 + ((Y - c.h + 2) / 2.6) ** 2 <= 1,
                   strong=((X - cx - 1) / 5) ** 2 + ((Y - c.h + 2) / 1.8) ** 2 <= 1)
        objs["tree_" + name] = Obj("tree_" + name, c, foot=(int(cx) - 4, c.h - 8, 8, 6), base=c.h - 2, group="trees")

    # ---------------------------------------------------------------- small props (1 tile wide, 2 tall)
    def small(name, c, foot, shadow_rx=6, shadow_cy=30):
        X, Y = S.grid(c.w, c.h)
        add_shadow(c, ((X - 10) / shadow_rx) ** 2 + ((Y - shadow_cy) / 2) ** 2 <= 1)
        objs[name] = Obj(name, c, foot=foot, base=foot[1] + foot[3])

    small("barrel_open", P.barrel("open"), (1, 22, 14, 9))
    small("barrel_closed", P.barrel("closed"), (1, 22, 14, 9))
    small("barrel_broken", P.barrel_broken(), (0, 22, 15, 9))
    small("stump", P.stump(), (2, 22, 12, 8))
    small("crate", P.crate(), (1, 20, 14, 11))
    for i in range(3):
        small(f"grave_{i}", P.gravestone(i), (3, 22, 10, 8))
    c = P.pillar_ruin(40, 3)
    small("pillar_ruin", c, (4, 38, 8, 7), shadow_rx=6, shadow_cy=46)
    return objs


FENCE_KINDS = [("fence_left", "left", 0), ("fence_mid_a", "mid", 1), ("fence_mid_b", "mid", 2),
               ("fence_mid_c", "mid", 4), ("fence_broken", "broken", 3), ("fence_right", "right", 5),
               ("fence_post", "post", 6)]


def fences():
    out = {}
    for name, kind, seed in FENCE_KINDS:
        c = P.fence(kind, seed)
        sh = S.rect(16, 32, 0, 30, 16, 2) & (c.px < 0)
        c.px[sh] = SH3
        c.px[S.rect(16, 32, 0, 31, 16, 1) & (c.px == SH3)] = SH1
        out[name] = c
    return out
