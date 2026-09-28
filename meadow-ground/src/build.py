"""Build the meadow ground tileset, a sample room, Tiled files and renders.

    python3 build.py        -> ../tileset, ../maps, ../render, ../palette
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from palette import HEX
from pixelkit import Canvas, hash01, upscale
import terrain as TR
import decals as DC
import layout as LY

TS = TR.TS
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NAME = "meadow_ground"
COLS = 12

# (patch terrain, surrounding terrain) - each drawn as a classic autotile block
PAIRS = [(TR.PATH, TR.GRASS), (TR.LUSH, TR.GRASS), (TR.FLOWERS, TR.LUSH)]


def classic_block(x, y, variant):
    """6x3 block: 3x3 patch, 2x2 inner corners, 2 diagonals, 3 plain tiles."""
    X, Y = x, y
    spec = {
        (0, 0): (Y, Y, Y, X), (1, 0): (Y, Y, X, X), (2, 0): (Y, Y, X, Y),
        (0, 1): (Y, X, Y, X), (1, 1): (X, X, X, X), (2, 1): (X, Y, X, Y),
        (0, 2): (Y, X, Y, Y), (1, 2): (X, X, Y, Y), (2, 2): (X, Y, Y, Y),
        (3, 0): (X, X, X, Y), (4, 0): (X, X, Y, X),
        (3, 1): (X, Y, X, X), (4, 1): (Y, X, X, X),
        (3, 2): (X, Y, Y, X), (4, 2): (Y, X, X, Y),
        (5, 0): (Y, Y, Y, Y), (5, 1): (X, X, X, X), (5, 2): (Y, Y, Y, Y),
    }
    pure_var = {(5, 0): 1, (5, 1): 2, (5, 2): 3}
    return spec, pure_var


class Atlas:
    def __init__(self):
        self.cells = {}
        self.lookup = {}          # (corners, variant) -> tile id
        self.names = {}           # tile id -> label

    def put(self, col, row, canvas, corners=None, variant=0, label=""):
        self.cells[(col, row)] = canvas
        tid = row * COLS + col
        if corners is not None:
            self.lookup.setdefault((tuple(corners), variant), tid)
        self.names[tid] = label
        return tid

    def rows(self):
        return max(r for (_, r) in self.cells) + 1

    def image(self):
        c = Canvas(COLS * TS, self.rows() * TS)
        for (col, row), t in self.cells.items():
            c.blit(t, col * TS, row * TS)
        return c


def build_atlas():
    a = Atlas()
    row = 0
    for (x, y) in PAIRS:
        for v in range(TR.NVAR_EDGE):
            spec, pure_var = classic_block(x, y, v)
            for (cx, cy), corners in spec.items():
                pure = len(set(corners)) == 1
                vv = pure_var.get((cx, cy), v if not pure else 0)
                label = f"{TR.NAMES[x]} in {TR.NAMES[y]}" + (f" · v{v + 1}" if not pure else "")
                a.put(v * 6 + cx, row + cy, TR.make_tile(corners, vv), corners, vv, label)
        row += 3
    # plain variants of every terrain
    col = 0
    for t in (TR.GRASS, TR.PATH, TR.LUSH, TR.FLOWERS):
        for v in range(TR.NVAR_PURE):
            if col >= COLS:
                col, row = 0, row + 1
            a.put(col, row, TR.make_tile((t,) * 4, v), (t,) * 4, v, f"{TR.NAMES[t]} · plain v{v + 1}")
            col += 1
    row += 1
    decal_ids = {}
    col = 0
    for n, c in DC.build().items():
        if col >= COLS:
            col, row = 0, row + 1
        decal_ids[n] = a.put(col, row, c, label=n.replace("_", " "))
        col += 1
    return a, decal_ids


def terrain_id(a, corners, tx, ty):
    corners = tuple(int(c) for c in corners)
    pure = len(set(corners)) == 1
    nv = TR.NVAR_PURE if pure else TR.NVAR_EDGE
    v = int(hash01(tx, ty, 99) * nv)
    if (corners, v) in a.lookup:
        return a.lookup[(corners, v)]
    if (corners, 0) in a.lookup:
        return a.lookup[(corners, 0)]
    # any combination outside the classic blocks is generated on demand
    raise KeyError(corners)


def place_decals(g, decal_ids):
    lay = np.full((LY.MH, LY.MW), -1, dtype=int)
    on_grass = ["daisies_a", "daisies_b", "daisy_big", "daisies_mix", "clover", "clover_pair", "tufts",
                "tuft_tall", "tuft_wide", "dandelion", "forget_me_nots", "pink_flowers", "buttercup", "stone"]
    on_lush = ["tuft_tall", "tuft_wide", "buttercups", "tufts", "mushrooms", "clover"]
    on_path = ["pebbles_a", "pebbles_b", "leaves", "stepping_stone"]
    for ty in range(LY.MH):
        for tx in range(LY.MW):
            cs = {int(g[ty, tx]), int(g[ty, tx + 1]), int(g[ty + 1, tx]), int(g[ty + 1, tx + 1])}
            if len(cs) != 1:
                continue
            t = cs.pop()
            r, r2 = hash01(tx, ty, 11), hash01(tx, ty, 13)
            pool, p = {TR.GRASS: (on_grass, 0.28), TR.LUSH: (on_lush, 0.2), TR.PATH: (on_path, 0.12)}.get(t, (None, 0))
            if pool and r < p:
                lay[ty, tx] = decal_ids[pool[int(r2 * len(pool))]]
    return lay


def main():
    for d in ("tileset", "maps", "render", "palette"):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    a, decal_ids = build_atlas()
    img = a.image()
    ts_png = os.path.join(ROOT, "tileset", f"{NAME}_tileset.png")
    img.save(ts_png)
    preview(img, os.path.join(ROOT, "tileset", f"{NAME}_tileset@3x.png"))

    g = LY.corner_grid()
    ground = np.zeros((LY.MH, LY.MW), dtype=int)
    missing = set()
    for ty in range(LY.MH):
        for tx in range(LY.MW):
            cs = (g[ty, tx], g[ty, tx + 1], g[ty + 1, tx], g[ty + 1, tx + 1])
            try:
                ground[ty, tx] = terrain_id(a, cs, tx, ty)
            except KeyError:
                missing.add(tuple(int(c) for c in cs))
    if missing:
        print("combinations not in the atlas:", sorted(missing))
        # extra row for anything unusual the layout produced
        row = a.rows()
        for i, cs in enumerate(sorted(missing)):
            a.put(i % COLS, row + i // COLS, TR.make_tile(cs, 0), cs, 0, "extra corner combo")
        img = a.image()
        img.save(ts_png)
        preview(img, os.path.join(ROOT, "tileset", f"{NAME}_tileset@3x.png"))
        for ty in range(LY.MH):
            for tx in range(LY.MW):
                cs = (g[ty, tx], g[ty, tx + 1], g[ty + 1, tx], g[ty + 1, tx + 1])
                ground[ty, tx] = terrain_id(a, cs, tx, ty)
    details = place_decals(g, decal_ids)

    # ---- render straight from the tile layers
    atlas_im = img.to_image()
    out = Image.new("RGBA", (LY.MW * TS, LY.MH * TS), (0, 0, 0, 255))
    for lay in (ground, details):
        for ty in range(LY.MH):
            for tx in range(LY.MW):
                t = lay[ty, tx]
                if t >= 0:
                    c, r = t % COLS, t // COLS
                    out.alpha_composite(atlas_im.crop((c * TS, r * TS, c * TS + TS, r * TS + TS)), (tx * TS, ty * TS))
    out.save(os.path.join(ROOT, "render", f"{NAME}_room.png"))
    for s in (2, 3):
        upscale(out, s).save(os.path.join(ROOT, "render", f"{NAME}_room@{s}x.png"))

    write_tsx(img, a)
    write_maps(ground, details, img, a)
    write_palette()
    print("atlas", img.w, "x", img.h, "tiles", len(a.cells))


def preview(img, path, s=3):
    im = upscale(img.to_image(), s)
    bg = Image.new("RGBA", im.size, (32, 32, 40, 255))
    d = ImageDraw.Draw(bg)
    for y in range(0, im.height, 10 * s):
        for x in range(0, im.width, 10 * s):
            if (x // (10 * s) + y // (10 * s)) % 2:
                d.rectangle([x, y, x + 10 * s - 1, y + 10 * s - 1], fill=(44, 44, 54, 255))
    bg.alpha_composite(im)
    d = ImageDraw.Draw(bg)
    for x in range(0, bg.width, TS * s):
        d.line([(x, 0), (x, bg.height)], fill=(0, 0, 0, 70))
    for y in range(0, bg.height, TS * s):
        d.line([(0, y), (bg.width, y)], fill=(0, 0, 0, 70))
    bg.save(path)


def write_tsx(img, a):
    colors = [("Path", "#c28f57"), ("Grass", "#46a043"), ("Lush grass", "#348a39"), ("Golden flowers", "#ffd83d")]
    rep = {t: a.lookup[((t,) * 4, 0)] for t in range(4)}
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<tileset version="1.10" tiledversion="1.10.2" name="{NAME}" tilewidth="{TS}" tileheight="{TS}" '
             f'tilecount="{COLS * a.rows()}" columns="{COLS}">',
             f' <image source="{NAME}_tileset.png" width="{img.w}" height="{img.h}"/>',
             ' <wangsets>',
             f'  <wangset name="Meadow" type="corner" tile="{rep[1]}">']
    for i, (n, col) in enumerate(colors):
        lines.append(f'   <wangcolor name="{n}" color="{col}" tile="{rep[i]}" probability="1"/>')
    seen = {}
    for (corners, v), tid in a.lookup.items():
        seen[tid] = corners
    for tid in sorted(seen):
        tl, tr, bl, br = (c + 1 for c in seen[tid])
        lines.append(f'   <wangtile tileid="{tid}" wangid="0,{tr},0,{br},0,{bl},0,{tl}"/>')
    lines += ['  </wangset>', ' </wangsets>', '</tileset>']
    open(os.path.join(ROOT, "tileset", f"{NAME}.tsx"), "w").write("\n".join(lines) + "\n")


def write_maps(ground, details, img, a):
    layers = [("Ground", ground), ("Details", details)]
    tmx = ['<?xml version="1.0" encoding="UTF-8"?>',
           f'<map version="1.10" tiledversion="1.10.2" orientation="orthogonal" renderorder="right-down" '
           f'width="{LY.MW}" height="{LY.MH}" tilewidth="{TS}" tileheight="{TS}" infinite="0" '
           f'nextlayerid="3" nextobjectid="1">',
           f' <tileset firstgid="1" source="../tileset/{NAME}.tsx"/>']
    for i, (n, arr) in enumerate(layers):
        csv = ",\n".join(",".join(str(int(v) + 1) for v in row) for row in arr)
        tmx += [f' <layer id="{i + 1}" name="{n}" width="{LY.MW}" height="{LY.MH}">', '  <data encoding="csv">', csv,
                '  </data>', ' </layer>']
    tmx.append('</map>')
    open(os.path.join(ROOT, "maps", f"{NAME}_room.tmx"), "w").write("\n".join(tmx) + "\n")
    data = dict(type="map", version="1.10", orientation="orthogonal", renderorder="right-down", width=LY.MW,
                height=LY.MH, tilewidth=TS, tileheight=TS, infinite=False, nextlayerid=3, nextobjectid=1,
                layers=[dict(id=i + 1, name=n, type="tilelayer", width=LY.MW, height=LY.MH, x=0, y=0, opacity=1,
                             visible=True, data=[int(v) + 1 for v in arr.flatten()]) for i, (n, arr) in enumerate(layers)],
                tilesets=[dict(firstgid=1, name=NAME, image=f"../tileset/{NAME}_tileset.png", imagewidth=img.w,
                               imageheight=img.h, tilewidth=TS, tileheight=TS, columns=COLS,
                               tilecount=COLS * a.rows(), margin=0, spacing=0)])
    json.dump(data, open(os.path.join(ROOT, "maps", f"{NAME}_room.json"), "w"))


def write_palette():
    d = os.path.join(ROOT, "palette")
    lines = ["GIMP Palette", "Name: Meadow", "Columns: 5", "#"]
    for h in HEX:
        r, g_, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
        lines.append(f"{r:3d} {g_:3d} {b:3d}\t{h}")
    open(os.path.join(d, "meadow.gpl"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(d, "meadow.hex"), "w").write("\n".join(h.lstrip("#") for h in HEX) + "\n")


if __name__ == "__main__":
    main()
