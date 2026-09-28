"""Build everything: tileset atlas, Tiled files, renders, sprites, palette.

    python3 build.py            # writes into ../ (tileset/, maps/, render/, ...)
"""
import itertools
import json
import os
import sys
import xml.sax.saxutils as sx

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from palette import HEX, K, SH1, SH3, RGBA
from pixelkit import Canvas, hash01, upscale
import terrain as TR
import forest as FO
import objects as OB
import props as P
import layout as L
import materials as M
import character as CHR

TS = 16
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NAME = "abandoned_village"
ATLAS_COLS = 32


# ======================================================================= atlas
class Atlas:
    def __init__(self, cols):
        self.cols = cols
        self.cells = {}          # (col,row) -> Canvas 16x16
        self.row = 0             # current shelf top
        self.col = 0
        self.shelf_h = 0
        self.blocks = {}         # name -> (col,row,w,h)

    def newline(self):
        if self.col:
            self.row += self.shelf_h
        self.col, self.shelf_h = 0, 0

    def add(self, name, canvas, gap=0):
        tw, th = canvas.w // TS, canvas.h // TS
        if self.col + tw > self.cols:
            self.newline()
        c0, r0 = self.col, self.row
        for ty in range(th):
            for tx in range(tw):
                t = canvas.crop(tx * TS, ty * TS, TS, TS)
                self.cells[(c0 + tx, r0 + ty)] = t
        self.blocks[name] = (c0, r0, tw, th)
        self.col += tw + gap
        self.shelf_h = max(self.shelf_h, th)
        return c0, r0

    def tid(self, col, row):
        return row * self.cols + col

    def block_tile(self, name, tx, ty):
        c0, r0, tw, th = self.blocks[name]
        return self.tid(c0 + tx, r0 + ty)

    def rows(self):
        return max(r for (_, r) in self.cells) + 1

    def image(self):
        h = self.rows()
        c = Canvas(self.cols * TS, h * TS)
        for (col, row), t in self.cells.items():
            c.blit(t, col * TS, row * TS)
        return c


# ======================================================================= terrain tiles
TORUS = [[1, 1, 1, 0], [1, 1, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]   # every 2x2 window unique
PAIRS = [(TR.VOID, TR.GRASS), (TR.GRASS, TR.MOSS), (TR.GRASS, TR.PATH), (TR.MOSS, TR.PATH)]


def build_terrain(atlas):
    """Terrain section: every 2-terrain pair as three seamless 4x4 blocks
    (variants 0,1,2 - each block tiles with itself and reads like a picture),
    then a block with every grass/moss/path three-way corner."""
    lookup = {}        # (corners tuple, variant) -> tile id
    atlas.newline()
    for (a, b) in PAIRS:
        for v in range(TR.NVAR_EDGE):
            blk = Canvas(64, 64)
            ids = []
            for r in range(4):
                for c in range(4):
                    bits = (TORUS[r][c], TORUS[r][(c + 1) % 4], TORUS[(r + 1) % 4][c], TORUS[(r + 1) % 4][(c + 1) % 4])
                    corners = tuple(b if bit else a for bit in bits)
                    vv = 0 if (len(set(corners)) == 1 and corners[0] == TR.VOID) else v
                    blk.blit(TR.make_tile(corners, vv), c * TS, r * TS)
                    ids.append((corners, c, r, vv))
            c0, r0 = atlas.add(f"wang_{a}_{b}_v{v}", blk)
            for corners, c, r, vv in ids:
                lookup.setdefault((corners, vv), atlas.tid(c0 + c, r0 + r))
    # three-terrain corners (grass/moss/path in one tile), 9 x 4 block
    tri = [cs for cs in itertools.product((TR.GRASS, TR.MOSS, TR.PATH), repeat=4) if len(set(cs)) == 3]
    blk = Canvas(9 * TS, 4 * TS)
    for i, cs in enumerate(tri):
        blk.blit(TR.make_tile(cs, 0), (i % 9) * TS, (i // 9) * TS)
    c0, r0 = atlas.add("wang_three_way", blk)
    for i, cs in enumerate(tri):
        lookup[(cs, 0)] = atlas.tid(c0 + i % 9, r0 + i // 9)
    return lookup


def terrain_id(lookup, corners, tx, ty):
    corners = tuple(int(c) for c in corners)
    pure = len(set(corners)) == 1
    nv = 1 if (pure and corners[0] == TR.VOID) else (TR.NVAR_PURE if pure else TR.NVAR_EDGE)
    v = int(hash01(tx, ty, 777) * nv)
    if (corners, v) in lookup:
        return lookup[(corners, v)]
    if (corners, 0) in lookup:
        return lookup[(corners, 0)]
    # a combination we never generate (void touching moss/path): fall back to grass edge
    fixed = tuple(TR.GRASS if c > TR.GRASS and TR.VOID in corners else c for c in corners)
    return lookup[(fixed, 0)]


# ======================================================================= decor
def build_decor():
    d = {}
    for i in range(len(P.TUFTS)):
        t = P.tuft(i)
        d[f"tuft_{i}"] = t
        # the same tuft shifted, so rows of tufts never look grid-aligned
        s = P.tuft(i, ox=1 if i % 2 else 7 - (i % 3), oy=None)
        d[f"tuft_{i}b"] = s
    for i in range(6):
        d[f"pebbles_{i}"] = P.pebbles(40 + i)
    d["rock_small"] = P.rock(11, 8, 1)
    d["rock_small_b"] = P.rock(9, 7, 5)
    d["rock_big"] = P.rock(22, 13, 2)
    d["fallen_column"] = P.fallen_column()
    d["debris_planks"] = debris_planks()
    d["debris_slates"] = debris_slates()
    return d


DEBRIS_PLANKS = [
    "..........1111..",
    "......1111GHH1..",
    "...111GGHHFF1...",
    "..1HHGFFFEE1....",
    "..1EEEDD111.....",
    "...1111.........",
    "................",
    ".....11111......",
    "....1GGHHG1.....",
    "....1EEDDE1.....",
    ".....11111......",
]
DEBRIS_SLATES = [
    "........111.....",
    ".......1HG1.....",
    "..111..1FE1.111.",
    ".1HGG1..11.1HG1.",
    ".1EEE1.....1ED1.",
    "..111.......11..",
    ".......1111.....",
    "......1HHG1.....",
    "......1EED1.....",
    ".......111......",
]


def debris_planks():
    from pixelkit import from_ascii
    c = Canvas(16, 16)
    c.blit(from_ascii(DEBRIS_PLANKS), 0, 4)
    return c


def debris_slates():
    from pixelkit import from_ascii
    c = Canvas(16, 16)
    c.blit(from_ascii(DEBRIS_SLATES), 0, 5)
    return c


# ======================================================================= map assembly
class Placed:
    def __init__(self, name, canvas, x, y, base, foot=None, overhead_rows=0, kind="object"):
        self.name, self.c, self.x, self.y = name, canvas, x, y      # x,y in px (tile aligned)
        self.base = base                                            # map-space sort y
        self.foot = foot                                            # map-space rect
        self.overhead_rows = overhead_rows
        self.kind = kind


def place_everything(objs, fences, decor, g):
    items = []
    for (name, tx, ty) in L.PLACEMENTS:
        o = objs[name]
        foot = None
        if o.foot:
            fx, fy, fw, fh = o.foot
            foot = (tx * TS + fx, ty * TS + fy, fw, fh)
        # rows fully above the collision box go to the overhead layer
        orows = 0
        if o.foot and o.th > 2:
            orows = max(0, o.foot[1] // TS)
        items.append(Placed(name, o.c, tx * TS, ty * TS, ty * TS + o.base, foot, orows))
    for (tx, ty, kinds) in L.FENCES:
        for i, k in enumerate(kinds):
            x, y = (tx + i) * TS, ty * TS
            foot = (x, y + 24, 16, 7) if k != "fence_post" else (x + 5, y + 24, 6, 7)
            items.append(Placed(k, fences[k], x, y, y + 30, foot, 0, "fence"))
    # tufts growing in front of walls/fences overlap them -> sorted like objects
    rng = np.random.default_rng(12)
    front = [(3, 12, 1), (5, 12, 0), (8, 12, 3), (36, 12, 5), (33, 12, 1), (4, 26, 2), (8, 26, 6), (29, 27, 0),
             (33, 27, 7), (35, 27, 3), (16, 13, 5), (24, 13, 2), (11, 18, 1), (13, 18, 4), (25, 19, 6), (26, 19, 0),
             (1, 20, 3), (37, 26, 5), (12, 25, 2), (35, 17, 1), (22, 13, 7)]
    for (tx, ty, i) in front:
        n = f"tuft_{i}b" if (tx + ty) % 2 else f"tuft_{i}"
        items.append(Placed(n, decor[n], tx * TS, ty * TS, ty * TS + 15, None, 0, "tuft"))
    items.sort(key=lambda p: (p.base, p.x))
    return items


def scatter_decor(g, items, decor):
    """Ground decals: tufts on grass, pebbles on the path, a few rocks."""
    occ = np.zeros((L.MH, L.MW), dtype=bool)
    for p in items:
        if p.kind == "tuft":
            continue
        tw, th = p.c.w // TS, p.c.h // TS
        tx0, ty0 = p.x // TS, p.y // TS
        # only the lower part of an object really occupies ground
        for ty in range(max(ty0, ty0 + th - 3), ty0 + th):
            for tx in range(tx0, tx0 + tw):
                if 0 <= ty < L.MH and 0 <= tx < L.MW:
                    occ[ty, tx] = True
    lay = {}
    for ty in range(10, L.MH):
        for tx in range(L.MW):
            if occ[ty, tx]:
                continue
            cs = [g[ty, tx], g[ty, tx + 1], g[ty + 1, tx], g[ty + 1, tx + 1]]
            if TR.VOID in cs:
                continue
            r = hash01(tx, ty, 4242)
            r2 = hash01(tx, ty, 4343)
            if all(c == TR.PATH for c in cs):
                if r < 0.05:
                    lay[(tx, ty)] = f"pebbles_{int(r2 * 6)}"
            elif all(c == TR.GRASS for c in cs):
                if r < 0.16:
                    i = int(r2 * len(P.TUFTS))
                    lay[(tx, ty)] = f"tuft_{i}" + ("b" if r2 * 10 % 2 > 1 else "")
                elif r < 0.19:
                    lay[(tx, ty)] = f"pebbles_{int(r2 * 6)}"
            elif TR.MOSS in cs and TR.PATH not in cs:
                if r < 0.07:
                    lay[(tx, ty)] = f"tuft_{int(r2 * 5)}b"
    # hand-placed rocks & debris
    for (tx, ty, n) in ((14, 16, "rock_small"), (15, 17, "pebbles_1"), (10, 18, "rock_small_b"),
                        (23, 17, "rock_small"), (27, 18, "pebbles_3"), (22, 14, "pebbles_2"),
                        (17, 14, "rock_small_b"), (6, 14, "debris_slates"),
                        (33, 14, "debris_planks"), (30, 28, "debris_slates"), (19, 22, "pebbles_4"),
                        (9, 27, "rock_small"), (2, 18, "rock_small_b"), (34, 19, "rock_big"),
                        (6, 19, "rock_big"), (18, 28, "pebbles_0")):
        lay[(tx, ty)] = n
        if n in ("fallen_column", "rock_big"):
            lay.pop((tx + 1, ty), None)
    return lay


# ======================================================================= layers
def allocate_layers(items, below=True):
    """Tile layers preserving draw order: an item goes above anything it overlaps."""
    layers = []            # list of dict (tx,ty)->(item, lx, ly)
    for p in items:
        tw, th = p.c.w // TS, p.c.h // TS
        cells = []
        for ly in range(th):
            is_over = ly < p.overhead_rows
            if is_over == below:
                continue
            for lx in range(tw):
                t = p.c.crop(lx * TS, ly * TS, TS, TS)
                if (t.px >= 0).any():
                    cells.append((p.x // TS + lx, p.y // TS + ly, lx, ly))
        if not cells:
            continue
        lvl = 0
        for i, layer in enumerate(layers):
            if any((cx, cy) in layer for (cx, cy, _, _) in cells):
                lvl = i + 1
        while len(layers) <= lvl:
            layers.append({})
        for (cx, cy, lx, ly) in cells:
            if 0 <= cx < L.MW and 0 <= cy < L.MH:
                layers[lvl][(cx, cy)] = (p, lx, ly)
    return layers


# ======================================================================= collision
def collision_rects(g, items):
    rects = []
    for p in items:
        if p.foot:
            rects.append(("solid", p.name) + tuple(int(v) for v in p.foot))
    # void: merge void tiles row by row
    for ty in range(L.MH):
        tx = 0
        while tx < L.MW:
            cs = [g[ty, tx], g[ty, tx + 1], g[ty + 1, tx], g[ty + 1, tx + 1]]
            if sum(c == TR.VOID for c in cs) >= 2:
                x0 = tx
                while tx < L.MW and sum(c == TR.VOID for c in [g[ty, tx], g[ty, tx + 1], g[ty + 1, tx], g[ty + 1, tx + 1]]) >= 2:
                    tx += 1
                rects.append(("void", "void", x0 * TS, ty * TS, (tx - x0) * TS, TS))
            else:
                tx += 1
    # the forest wall
    rects.append(("solid", "forest", 0, 0, L.MW * TS, 162))
    return rects


# ======================================================================= main
def main():
    out = {k: os.path.join(ROOT, k) for k in ("tileset", "maps", "render", "sprites", "palette")}
    for p in out.values():
        os.makedirs(p, exist_ok=True)

    print("building objects…")
    objs = OB.build_all()
    fences = OB.fences()
    decor = build_decor()
    band_a, band_b = FO.band(0), FO.band(1)
    cap_l, cap_r = FO.cap(band_a, "left"), FO.cap(band_b, "right")

    print("packing atlas…")
    atlas = Atlas(ATLAS_COLS)
    lookup = build_terrain(atlas)
    atlas.newline()
    for n in decor:
        atlas.add(n, decor[n])
    atlas.newline()
    for n, c in fences.items():
        atlas.add(n, c)
    for n in ("barrel_open", "barrel_closed", "barrel_broken", "stump", "crate", "grave_0", "grave_1", "grave_2"):
        atlas.add(n, objs[n].c)
    atlas.add("pillar_ruin", objs["pillar_ruin"].c)
    atlas.add("signpost", objs["signpost"].c)
    atlas.add("well", objs["well"].c)
    atlas.add("statue", objs["statue"].c)
    atlas.newline()
    for n in ("tree_tall", "tree_crooked", "tree_wide", "tree_medium", "tree_small", "tree_thin"):
        atlas.add(n, objs[n].c)
    atlas.newline()
    atlas.add("church", objs["church"].c)
    atlas.add("house_d", objs["house_d"].c)
    atlas.add("house_c", objs["house_c"].c)
    atlas.newline()
    atlas.add("house_a", objs["house_a"].c)
    atlas.add("house_b", objs["house_b"].c)
    atlas.add("forest_a", band_a)
    atlas.newline()
    atlas.add("forest_b", band_b)
    atlas.add("forest_cap_l", cap_l.crop(0, 0, 48, FO.BAND_H))
    atlas.add("forest_cap_r", cap_r.crop(FO.PERIOD - 48, 0, 48, FO.BAND_H))
    atlas.newline()

    print("laying out the map…")
    g = L.corner_grid()
    items = place_everything(objs, fences, decor, g)
    ground = np.zeros((L.MH, L.MW), dtype=int)
    for ty in range(L.MH):
        for tx in range(L.MW):
            ground[ty, tx] = terrain_id(lookup, (g[ty, tx], g[ty, tx + 1], g[ty + 1, tx], g[ty + 1, tx + 1]), tx, ty)
    decor_lay = scatter_decor(g, items, decor)
    decor_ids = np.full((L.MH, L.MW), -1, dtype=int)
    for (tx, ty), n in decor_lay.items():
        c0, r0, tw, th = atlas.blocks[n]
        for dx in range(tw):
            if tx + dx < L.MW:
                decor_ids[ty, tx + dx] = atlas.tid(c0 + dx, r0)
    # forest band across the top: cap_l | a b a a b ... | cap_r   (period 8 tiles)
    forest_ids = np.full((L.MH, L.MW), -1, dtype=int)
    seq = ["forest_a", "forest_b", "forest_a", "forest_a", "forest_b"]
    for tx in range(L.MW):
        k = tx // (FO.PERIOD // TS)
        name = seq[k % len(seq)]
        lx = tx % (FO.PERIOD // TS)
        if tx < 3:
            name, lx = "forest_cap_l", tx
        elif tx >= L.MW - 3:
            name, lx = "forest_cap_r", tx - (L.MW - 3)
        for ty in range(FO.BAND_H // TS):
            t = atlas.cells[(atlas.blocks[name][0] + lx, atlas.blocks[name][1] + ty)]
            if (t.px >= 0).any():
                forest_ids[ty, tx] = atlas.block_tile(name, lx, ty)
    below = allocate_layers(items, below=True)
    above = allocate_layers(items, below=False)

    def ids_of(layer):
        a = np.full((L.MH, L.MW), -1, dtype=int)
        for (cx, cy), (p, lx, ly) in layer.items():
            a[cy, cx] = atlas.block_tile(p.name, lx, ly)
        return a

    layers = [("Ground", ground), ("Decor", decor_ids), ("Forest", forest_ids)]
    for i, lay in enumerate(below):
        layers.append((f"Objects {i + 1}", ids_of(lay)))
    for i, lay in enumerate(above):
        layers.append((f"Overhead {i + 1}", ids_of(lay)))

    # ---- write the tileset image + preview
    img = atlas.image()
    tileset_png = os.path.join(out["tileset"], f"{NAME}_tileset.png")
    img.save(tileset_png)
    save_tileset_preview(img, os.path.join(out["tileset"], f"{NAME}_tileset_preview.png"), atlas)

    # ---- render the map straight from the tile layers
    tiles = {}
    atlas_img = img.to_image()

    def tile_img(tid):
        if tid not in tiles:
            c, r = tid % ATLAS_COLS, tid // ATLAS_COLS
            tiles[tid] = atlas_img.crop((c * TS, r * TS, c * TS + TS, r * TS + TS))
        return tiles[tid]

    def render_layers(layer_list):
        im = Image.new("RGBA", (L.MW * TS, L.MH * TS), (0, 0, 0, 255))
        for _, a in layer_list:
            for ty in range(L.MH):
                for tx in range(L.MW):
                    if a[ty, tx] >= 0:
                        im.alpha_composite(tile_img(a[ty, tx]), (tx * TS, ty * TS))
        return im

    base_layers = [l for l in layers if not l[0].startswith("Overhead")]
    over_layers = [l for l in layers if l[0].startswith("Overhead")]
    no_player = render_layers(base_layers + over_layers)
    no_player.save(os.path.join(out["render"], f"{NAME}_noplayer.png"))

    # with the wanderer standing on the path (y-sorted against objects)
    hero = CHR.sheet()
    frame = CHR.frame(hero, "up", 0)
    px, py = L.PLAYER_START
    with_player = render_ysorted(img, items, ground, decor_ids, forest_ids, frame, px, py, atlas, layers)
    with_player.save(os.path.join(out["render"], f"{NAME}.png"))
    for s in (2, 3, 4):
        upscale(with_player, s).save(os.path.join(out["render"], f"{NAME}@{s}x.png"))

    # ---- Tiled files
    write_tsx(os.path.join(out["tileset"], f"{NAME}.tsx"), img, atlas, lookup)
    rects = collision_rects(g, items)
    write_tmx(os.path.join(out["maps"], f"{NAME}.tmx"), layers, rects)
    write_json(os.path.join(out["maps"], f"{NAME}.json"), layers, rects, img, atlas, lookup)

    # ---- individual sprites + placement list for y-sorting engines
    placed = []
    for n, o in objs.items():
        o.c.save(os.path.join(out["sprites"], f"{n}.png"))
    for n, c in fences.items():
        c.save(os.path.join(out["sprites"], f"{n}.png"))
    for p in items:
        if p.kind == "tuft":
            continue
        placed.append(dict(name=p.name, sprite=f"sprites/{p.name}.png", x=p.x, y=p.y,
                           w=p.c.w, h=p.c.h, sort_y=p.base,
                           collision=list(p.foot) if p.foot else None))
    with open(os.path.join(out["maps"], f"{NAME}_objects.json"), "w") as f:
        json.dump(dict(map=f"{NAME}.tmx", tile_size=TS, width=L.MW, height=L.MH,
                       player_start=dict(x=px, y=py), objects=placed), f, indent=1)
    hero.save(os.path.join(out["sprites"], "wanderer.png"))
    upscale(hero.to_image(), 4).save(os.path.join(out["sprites"], "wanderer@4x.png"))

    # ---- palette + interactive preview
    write_palette(out["palette"])
    write_preview(os.path.join(ROOT, "preview"), atlas, layers, items, rects, objs, fences, decor,
                  tileset_png, hero)
    print("tiles:", len(atlas.cells), "atlas rows:", atlas.rows(), "layers:", [l[0] for l in layers])
    return dict(items=items, layers=layers, atlas=atlas, g=g, rects=rects)


def render_ysorted(img, items, ground, decor_ids, forest_ids, hero_frame, px, py, atlas, layers):
    """Ground → decor → forest → objects+player sorted by their base line."""
    atlas_img = img.to_image()
    im = Image.new("RGBA", (L.MW * TS, L.MH * TS), (0, 0, 0, 255))
    for a in (ground, decor_ids, forest_ids):
        for ty in range(L.MH):
            for tx in range(L.MW):
                if a[ty, tx] >= 0:
                    c, r = a[ty, tx] % ATLAS_COLS, a[ty, tx] // ATLAS_COLS
                    im.alpha_composite(atlas_img.crop((c * TS, r * TS, c * TS + TS, r * TS + TS)), (tx * TS, ty * TS))
    drawables = [(p.base, 0, p) for p in items]
    drawables.append((py, 1, "hero"))
    drawables.sort(key=lambda d: (d[0], d[1]))
    for _, _, p in drawables:
        if p == "hero":
            hi = hero_frame.to_image()
            im.alpha_composite(hi, (px - hi.width // 2, py - hi.height))
        else:
            im.alpha_composite(p.c.to_image(), (p.x, p.y))
    return im


def save_tileset_preview(img, path, atlas, s=3):
    im = upscale(img.to_image(), s)
    bg = Image.new("RGBA", im.size, (18, 20, 34, 255))
    # checker so transparent tiles are visible
    d = ImageDraw.Draw(bg)
    for y in range(0, im.height, 8 * s):
        for x in range(0, im.width, 8 * s):
            if (x // (8 * s) + y // (8 * s)) % 2:
                d.rectangle([x, y, x + 8 * s - 1, y + 8 * s - 1], fill=(24, 27, 44, 255))
    bg.alpha_composite(im)
    d = ImageDraw.Draw(bg)
    for x in range(0, bg.width, TS * s):
        d.line([(x, 0), (x, bg.height)], fill=(60, 66, 110, 90))
    for y in range(0, bg.height, TS * s):
        d.line([(0, y), (bg.width, y)], fill=(60, 66, 110, 90))
    bg.save(path)


# ======================================================================= Tiled writers
def write_tsx(path, img, atlas, lookup):
    colors = [("Void", "#030409"), ("Grass", "#27305b"), ("Moss", "#3b4778"), ("Path", "#434f82")]
    rep = {TR.VOID: lookup[((TR.VOID,) * 4, 0)], TR.GRASS: lookup[((TR.GRASS,) * 4, 0)],
           TR.MOSS: lookup[((TR.MOSS,) * 4, 0)], TR.PATH: lookup[((TR.PATH,) * 4, 0)]}
    seen = {}
    for (corners, v), tid in lookup.items():
        seen[tid] = corners
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<tileset version="1.10" tiledversion="1.10.2" name="{NAME}" tilewidth="{TS}" tileheight="{TS}" '
             f'tilecount="{atlas.cols * atlas.rows()}" columns="{atlas.cols}">',
             f' <image source="{NAME}_tileset.png" width="{img.w}" height="{img.h}"/>',
             ' <wangsets>',
             f'  <wangset name="Ground" type="corner" tile="{rep[TR.GRASS]}">']
    for i, (n, col) in enumerate(colors):
        lines.append(f'   <wangcolor name="{n}" color="{col}" tile="{rep[i]}" probability="1"/>')
    for tid in sorted(seen):
        tl, tr, bl, br = (c + 1 for c in seen[tid])
        lines.append(f'   <wangtile tileid="{tid}" wangid="0,{tr},0,{br},0,{bl},0,{tl}"/>')
    lines += ['  </wangset>', ' </wangsets>', '</tileset>']
    open(path, "w").write("\n".join(lines) + "\n")


def write_tmx(path, layers, rects):
    lid = 1
    oid = 1
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           f'<map version="1.10" tiledversion="1.10.2" orientation="orthogonal" renderorder="right-down" '
           f'width="{L.MW}" height="{L.MH}" tilewidth="{TS}" tileheight="{TS}" infinite="0" '
           f'backgroundcolor="#030409" nextlayerid="{len(layers) + 3}" nextobjectid="{len(rects) + 10}">',
           f' <tileset firstgid="1" source="../tileset/{NAME}.tsx"/>']
    for name, a in layers:
        csv = ",\n".join(",".join(str(int(v) + 1) for v in row) for row in a)
        out.append(f' <layer id="{lid}" name="{name}" width="{L.MW}" height="{L.MH}">')
        out.append('  <data encoding="csv">')
        out.append(csv)
        out.append('  </data>')
        out.append(' </layer>')
        lid += 1
    out.append(f' <objectgroup id="{lid}" name="Collision" visible="0">')
    for (kind, name, x, y, w, h) in rects:
        out.append(f'  <object id="{oid}" name="{sx.escape(name)}" type="{kind}" x="{x}" y="{y}" width="{w}" height="{h}"/>')
        oid += 1
    out.append(' </objectgroup>')
    lid += 1
    px, py = L.PLAYER_START
    out.append(f' <objectgroup id="{lid}" name="Markers">')
    out.append(f'  <object id="{oid}" name="PlayerStart" type="spawn" x="{px}" y="{py}"><point/></object>')
    out.append(f'  <object id="{oid + 1}" name="ExitSouth" type="exit" x="{17 * TS}" y="{L.MH * TS - 8}" width="{7 * TS}" height="8"/>')
    out.append(f'  <object id="{oid + 2}" name="ChapelDoor" type="door" x="{15 * TS + 77}" y="{150}" width="22" height="46"/>')
    out.append(' </objectgroup>')
    out.append('</map>')
    open(path, "w").write("\n".join(out) + "\n")


def write_json(path, layers, rects, img, atlas, lookup):
    """Tiled-JSON (works with Phaser, LDtk importers, etc.) with an embedded tileset."""
    data = dict(type="map", version="1.10", tiledversion="1.10.2", orientation="orthogonal",
                renderorder="right-down", width=L.MW, height=L.MH, tilewidth=TS, tileheight=TS,
                infinite=False, backgroundcolor="#030409", nextlayerid=len(layers) + 3,
                nextobjectid=len(rects) + 10, layers=[], tilesets=[])
    lid = 1
    for name, a in layers:
        data["layers"].append(dict(id=lid, name=name, type="tilelayer", width=L.MW, height=L.MH, x=0, y=0,
                                   opacity=1, visible=True, data=[int(v) + 1 for v in a.flatten()]))
        lid += 1
    objs = [dict(id=i + 1, name=n, type=k, x=x, y=y, width=w, height=h, rotation=0, visible=True)
            for i, (k, n, x, y, w, h) in enumerate(rects)]
    data["layers"].append(dict(id=lid, name="Collision", type="objectgroup", draworder="topdown",
                               opacity=1, visible=False, x=0, y=0, objects=objs))
    px, py = L.PLAYER_START
    data["layers"].append(dict(id=lid + 1, name="Markers", type="objectgroup", draworder="topdown", opacity=1,
                               visible=True, x=0, y=0,
                               objects=[dict(id=len(objs) + 1, name="PlayerStart", type="spawn", x=px, y=py,
                                             point=True, width=0, height=0, rotation=0, visible=True)]))
    data["tilesets"].append(dict(firstgid=1, name=NAME, image=f"../tileset/{NAME}_tileset.png",
                                 imagewidth=img.w, imageheight=img.h, tilewidth=TS, tileheight=TS,
                                 columns=atlas.cols, tilecount=atlas.cols * atlas.rows(), margin=0, spacing=0))
    with open(path, "w") as f:
        json.dump(data, f)


BLOCK_NAMES = {
    "church": "Часовня (11×14)", "house_a": "Дом с дырявой вальмовой крышей", "house_b": "Дом с провалившейся крышей",
    "house_c": "Фронтонный дом с высокой трубой", "house_d": "Широкий дом со сгоревшей трубой",
    "well": "Колодец", "statue": "Безликая статуя", "signpost": "Указатель", "stump": "Пень", "crate": "Ящик",
    "barrel_open": "Открытая бочка", "barrel_closed": "Закрытая бочка", "barrel_broken": "Опрокинутая бочка",
    "grave_0": "Надгробие", "grave_1": "Крест", "grave_2": "Расколотая плита", "pillar_ruin": "Обломок колонны",
    "fence_left": "Забор · левый край", "fence_mid_a": "Забор · середина", "fence_mid_b": "Забор · середина",
    "fence_mid_c": "Забор · середина", "fence_broken": "Забор · сломанный", "fence_right": "Забор · правый край",
    "fence_post": "Забор · столб", "tree_tall": "Мёртвое дерево · высокое", "tree_crooked": "Мёртвое дерево · кривое",
    "tree_wide": "Мёртвое дерево · раскидистое", "tree_medium": "Мёртвое дерево · среднее",
    "tree_small": "Мёртвое дерево · малое", "tree_thin": "Мёртвое дерево · тонкое",
    "forest_a": "Лес · вариант A (тайлится по X)", "forest_b": "Лес · вариант B (тайлится по X)",
    "forest_cap_l": "Лес · левый торец", "forest_cap_r": "Лес · правый торец",
    "rock_small": "Камень", "rock_small_b": "Камень", "rock_big": "Валун", "fallen_column": "Упавшая колонна",
    "debris_planks": "Обломки досок", "debris_slates": "Осколки черепицы", "wang_three_way": "Террейн · стыки трава/мох/тропа",
}
_TNAME = {0: "пустота", 1: "трава", 2: "мох", 3: "тропа"}

INTERACT = {
    "church": "Дверь часовни приоткрыта. Изнутри тянет холодом и старым воском.",
    "well": "Колодец очень глубокий. Внизу что-то поблёскивает — вода или чьи-то глаза.",
    "statue": "Безликая статуя. На табличке у подножия: «…мы ждём тебя». Остальное стёрто.",
    "signpost": "Указатель на восток. Надпись смыло дождями, нижняя доска висит на одном гвозде.",
    "house_a": "Дверь заперта изнутри. Хотя крыша вся в дырах — кто её запер?",
    "house_b": "Потолка нет: над головой только балки и ночное небо.",
    "house_c": "Дверь лежит на пороге. Внутри пусто, гуляет сквозняк.",
    "house_d": "В темноте на столе стоят две чашки. Их давно никто не трогал.",
    "grave": "Имя на камне не прочесть.",
    "barrel_open": "В бочке дождевая вода. В ней дрожит луна.",
    "barrel_closed": "Крышка забита гвоздями.",
    "stump": "На пне вырезаны чьи-то инициалы.",
    "crate": "Пустой ящик. На дне — сухие листья.",
}
DOORS = {"house_a": (20, 80, 14, 26), "house_b": (44, 80, 14, 26), "house_c": (24, 80, 14, 26),
         "house_d": (34, 94, 14, 28), "church": (77, 150, 22, 48)}


def write_preview(d, atlas, layers, items, rects, objs, fences, decor, tileset_png, hero):
    import base64
    import io
    os.makedirs(d, exist_ok=True)

    def uri_img(im):
        b = io.BytesIO()
        im.save(b, "PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

    lay = {n: [int(v) for v in a.flatten()] for n, a in layers if n in ("Ground", "Decor", "Forest")}
    objects, sprites, interact = [], {}, []
    for p in items:
        objects.append(dict(s=p.name, x=p.x, y=p.y, sy=p.base))
        if p.name not in sprites:
            sprites[p.name] = uri_img(p.c.to_image())
        key = "grave" if p.name.startswith("grave") else p.name
        if key in INTERACT:
            if p.name in DOORS:
                x, y, w, h = DOORS[p.name]
                r = [p.x + x, p.y + y, w, h]
            else:
                r = list(p.foot) if p.foot else [p.x, p.y, p.c.w, p.c.h]
            interact.append(dict(rect=[int(v) for v in r], text=INTERACT[key]))
    names = dict(BLOCK_NAMES)
    for n in atlas.blocks:
        if n.startswith("wang_") and n != "wang_three_way":
            _, a_, b_, v_ = n.split("_")
            names[n] = f"Террейн · {_TNAME[int(a_)]} / {_TNAME[int(b_)]} · вариант {int(v_[1:]) + 1}"
        elif n.startswith("tuft_"):
            names[n] = "Пучок травы"
        elif n.startswith("pebbles_"):
            names[n] = "Камешки"
    data = dict(tile=TS, w=L.MW, h=L.MH, atlasCols=ATLAS_COLS, layers=lay, objects=objects,
                solids=[[x, y, w, h] for (_, _, x, y, w, h) in rects], start=list(L.PLAYER_START),
                interact=interact, blocks={k: list(v) for k, v in atlas.blocks.items()}, names=names,
                palette=list(HEX))
    imgs = dict(atlas=uri_img(Image.open(tileset_png)), hero=uri_img(hero.to_image()), sprites=sprites)
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview_template.html"), encoding="utf-8").read()
    body = tpl.replace("/*__MAP_DATA__*/", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    body = body.replace("/*__IMG_DATA__*/", json.dumps(imgs, separators=(",", ":")))
    head, rest = body.split("</style>", 1)
    doc = ('<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           + head + "</style>\n</head>\n<body>\n" + rest + "</body>\n</html>\n")
    open(os.path.join(d, "index.html"), "w", encoding="utf-8").write(doc)
    if os.environ.get("ARTIFACT_OUT"):
        # the artifact host adds its own document skeleton
        open(os.environ["ARTIFACT_OUT"], "w", encoding="utf-8").write(body)
    return body


def write_palette(d):
    # GIMP / Aseprite palette
    lines = ["GIMP Palette", "Name: Moonlit Indigo", "Columns: 8", "#"]
    for h in HEX:
        r, g_, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
        lines.append(f"{r:3d} {g_:3d} {b:3d}\t{h}")
    open(os.path.join(d, "moonlit_indigo.gpl"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(d, "moonlit_indigo.hex"), "w").write("\n".join(h.lstrip("#") for h in HEX) + "\n")
    sw = 24
    im = Image.new("RGBA", (sw * 10, sw * 4), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(im)
    rows = [HEX[:10], HEX[10:18], HEX[18:28], HEX[28:38]]
    for r, row in enumerate(rows):
        for i, h in enumerate(row):
            d2.rectangle([i * sw, r * sw, i * sw + sw - 1, r * sw + sw - 1], fill=h)
    im.save(os.path.join(d, "moonlit_indigo.png"))


if __name__ == "__main__":
    main()
