"""Render the composition sketch of 'The Pale Mire' (+ a short fog test GIF).

    python3 sketch.py            -> ../sketch/*.png, *.gif
"""
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pixelkit import Canvas, hash01, upscale
from palette import G, W
import terrain as TR
import layout as L
import trees as TRE
import flora as FL
import props as PR
import fog as FOG
import character as CHR

TS = 16
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "sketch")
MWpx, MHpx = L.MW * TS, L.MH * TS


def corners(g, tx, ty):
    return (int(g[ty, tx]), int(g[ty, tx + 1]), int(g[ty + 1, tx]), int(g[ty + 1, tx + 1]))


def terrain_image(g, frame, cache={}):
    c = Canvas(MWpx, MHpx)
    for ty in range(L.MH):
        for tx in range(L.MW):
            cor = corners(g, tx, ty)
            pure = len(set(cor)) == 1
            nv = TR.NVAR_PURE if pure else TR.NVAR_EDGE
            v = int(hash01(tx, ty, 777) * nv)
            key = (cor, v, frame)
            if key not in cache:
                cache[key] = TR.make_tile(cor, v, frame)
            c.blit(cache[key], tx * TS, ty * TS)
    return c.to_image()


class Sprite:
    def __init__(self, img, x, y, sort_y, kind="obj", anim=None):
        self.img, self.x, self.y, self.sort_y, self.kind = img, int(round(x)), int(round(y)), sort_y, kind
        self.anim = anim            # optional list of frames (images)

    def at(self, f):
        if self.anim:
            return self.anim[f % len(self.anim)]
        return self.img


def build_scene(frame=0):
    g = L.corner_grid()
    water = (g <= TR.SHALLOW)
    land_c = g >= TR.MUD
    # distance (in corners) to land, for placing pads away from the banks
    dist = np.full(g.shape, 99)
    dist[land_c] = 0
    for _ in range(6):
        d2 = dist.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                d2 = np.minimum(d2, np.roll(np.roll(dist, dy, 0), dx, 1) + 1)
        dist = d2

    occupied = set(L.WALK_V) | set(L.WALK_H)
    px0, py0 = L.PLATFORM
    for dy in range(3):
        for dx in range(3):
            occupied.add((px0 + dx, py0 + dy))

    decals, shadows, floor, objs, canopies = [], [], [], [], []
    from pixelkit import value_noise
    reed_field = value_noise(L.MW + 1, L.MH + 1, 3.5, seed=77)

    # ---- trees
    giant_params = {
        7: dict(), 1: dict(crown_w=300, trunk_w=60, trunk_h=158),
        11: dict(crown_w=280, trunk_w=58, trunk_h=150, moss=1.2), 19: dict(crown_w=290, trunk_w=62),
        23: dict(crown_w=320, trunk_w=66, trunk_h=175),
    }
    giant_bases = []
    for (seed, bx, by, flip) in L.GIANTS:
        p = TRE.giant_cypress(seed=seed, **giant_params.get(seed, {}))
        layers = [p.shadow, p.trunk, p.canopy]
        if flip:
            layers = [l.flip_h() for l in layers]
        x0, y0 = bx - p.w / 2, by - p.base_y
        shadows.append(Sprite(layers[0].to_image(), x0, y0, by))
        objs.append(Sprite(layers[1].to_image(), x0, y0, by))
        canopies.append(Sprite(layers[2].to_image(), x0, y0, by))
        giant_bases.append((bx, by))
    for (seed, bx, by) in L.CYPRESSES:
        p = TRE.cypress(seed=seed)
        x0, y0 = bx - p.w / 2, by - p.base_y
        shadows.append(Sprite(p.shadow.to_image(), x0, y0, by))
        objs.append(Sprite(p.trunk.to_image(), x0, y0, by))
        canopies.append(Sprite(p.canopy.to_image(), x0, y0, by))
    for (seed, bx, by) in L.SNAGS:
        p = FL.snag(seed)
        objs.append(Sprite(p.trunk.to_image(), bx - p.w / 2, by - p.base_y, by))

    # ---- boardwalk (floor level)
    for (tx, ty) in L.WALK_V:
        t = PR.walk_v(tx * 7 + ty, broken=(tx, ty) in L.BROKEN,
                      cap_t=(tx, ty - 1) not in occupied, cap_b=(tx, ty + 1) not in occupied)
        floor.append(Sprite(t.to_image(), tx * TS, ty * TS, ty * TS))
    for (tx, ty) in L.WALK_H:
        t = PR.walk_h(tx * 7 + ty, broken=(tx, ty) in L.BROKEN, post_l=(tx % 2 == 0), post_r=(tx % 2 == 1))
        floor.append(Sprite(t.to_image(), tx * TS, ty * TS, ty * TS))
    floor.append(Sprite(PR.platform().to_image(), px0 * TS, py0 * TS, py0 * TS))
    bx, by = L.BOAT
    objs.append(Sprite(PR.boat().to_image(), bx - 24, by - 20, by))
    for (x, y, ln, sd) in L.LOGS:
        objs.append(Sprite(FL.fallen_log(ln, sd).to_image(), x, y - 16, y + 8))

    # ---- scattered flora by terrain
    for ty in range(L.MH):
        for tx in range(L.MW):
            if (tx, ty) in occupied or (tx, ty - 1) in occupied or (tx, ty + 1) in occupied:
                continue
            cor = corners(g, tx, ty)
            r = hash01(tx, ty, 4242)
            r2 = hash01(tx, ty, 4343)
            allw = all(c <= TR.SHALLOW for c in cor)
            shore = any(c <= TR.SHALLOW for c in cor) and any(c >= TR.MUD for c in cor)
            d = min(dist[ty, tx], dist[ty + 1, tx + 1], dist[ty, tx + 1], dist[ty + 1, tx])
            x, y = tx * TS, ty * TS
            near_giant = min(math.hypot(x + 8 - gx, y + 8 - gy) for (gx, gy) in giant_bases)
            clump = reed_field[ty, tx] > 0.52
            if shore and (r < (0.6 if clump else 0.08)):
                sd = int(r2 * 1000)
                fr = [FL.reeds(sd, in_water=True, frame=k).to_image() for k in range(4)]
                objs.append(Sprite(fr[0], x - 8, y + TS - 44, y + TS, anim=fr))
            elif allw and clump and d <= 1 and r < 0.3:
                sd = int(r2 * 1000)
                fr = [FL.reeds(sd, in_water=True, frame=k, n=7).to_image() for k in range(4)]
                objs.append(Sprite(fr[0], x - 8, y + TS - 44, y + TS, anim=fr))
            elif allw and near_giant < 90 and r < 0.5:
                decals.append(Sprite(FL.knees(int(r2 * 100)).to_image(), x, y, y))
            elif allw and d >= 2 and r < 0.2:
                decals.append(Sprite(FL.lily_pads(int(r2 * 1000), flower=r2 < 0.25).to_image(), x, y, y))
            elif allw and d <= 2 and r < 0.34:
                decals.append(Sprite(FL.duckweed(int(r2 * 1000), 0.55).to_image(), x, y, y))
            elif all(c == TR.GRASS for c in cor) and r < 0.2:
                kind = int(r2 * 10)
                if kind < 5:
                    s = FL.sprite16(FL.GRASS[kind % 3])
                elif kind < 8:
                    s = FL.sprite16(FL.FERNS[kind % 2], h=16)
                else:
                    s = FL.sprite16(FL.FUNGI[kind % 2])
                objs.append(Sprite(s.to_image(), x, y, y + TS - 1))
            elif all(c >= TR.MUD for c in cor) and clump and r < 0.12:
                sd = int(r2 * 1000)
                fr = [FL.reeds(sd, in_water=False, dry=True, n=6, frame=k).to_image() for k in range(4)]
                objs.append(Sprite(fr[0], x - 8, y + TS - 44, y + TS, anim=fr))
    return g, decals, shadows, floor, objs, canopies


def compose(g, decals, shadows, floor, objs, canopies, frame=0, player=True, fog_t=None, fog_loop=6.0,
            sway=0, player_pos=None, player_dir="down"):
    im = terrain_image(g, frame)
    for s in decals + shadows + floor:
        _paste(im, s)
    items = list(objs)
    if player:
        hero = CHR.frame(CHR.sheet(), player_dir, 0).to_image()
        px, py = player_pos or L.PLAYER
        items.append(Sprite(hero, px - 8, py - 24, py))
    for s in sorted(items, key=lambda s: s.sort_y):
        _paste(im, s, sway)
    for s in sorted(canopies, key=lambda s: s.sort_y):
        _paste(im, s)
    if fog_t is not None:
        wmask = np.asarray(Image.fromarray((g <= TR.SHALLOW).astype(np.float32), "F")
                           .resize((MWpx, MHpx), Image.BILINEAR))
        d = FOG.density(MWpx, MHpx, fog_t, fog_loop, water=wmask)
        im.alpha_composite(FOG.overlay(d))
    return im


def _paste(im, s, f=0):
    """alpha_composite that tolerates sprites hanging off the map."""
    x, y = s.x, s.y
    src = s.at(f)
    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    w = min(src.width - sx0, im.width - dx0)
    h = min(src.height - sy0, im.height - dy0)
    if w <= 0 or h <= 0:
        return
    im.alpha_composite(src.crop((sx0, sy0, sx0 + w, sy0 + h)), (dx0, dy0))


def main():
    os.makedirs(OUT, exist_ok=True)
    scene = build_scene(0)
    full = compose(*scene, frame=0, fog_t=0.0)
    full.save(os.path.join(OUT, "pale_mire_sketch.png"))
    upscale(full, 2).save(os.path.join(OUT, "pale_mire_sketch@2x.png"))
    clean = compose(*scene, frame=0, fog_t=None)
    clean.save(os.path.join(OUT, "pale_mire_sketch_nofog.png"))
    # a close-up the size of a game screen, to judge the scale
    box = KEY_BOX
    ppos = (500, 150)
    key = compose(*scene, frame=0, fog_t=0.0, player_pos=ppos, player_dir="up").crop(box)
    upscale(key, 3).save(os.path.join(OUT, "pale_mire_keyframe@3x.png"))
    print("sketch written")
    if "--gif" in sys.argv:
        gif(scene, box, ppos)


KEY_BOX = (170, 0, 650, 270)          # 480 x 270 game pixels


def gif(scene, box, ppos, n=40, ms=110, scale=2, path=None):
    frames = []
    loop = n * ms / 1000.0
    for f in range(n):
        im = compose(*scene, frame=(f // 2) % 4, fog_t=f * ms / 1000.0, fog_loop=loop, sway=(f // 5) % 4,
                     player_pos=ppos, player_dir="up").crop(box)
        frames.append(upscale(im, scale).convert("RGB"))
    # one shared palette for every frame (no flicker)
    strip = Image.new("RGB", (frames[0].width, frames[0].height * 4))
    for i, k in enumerate(range(0, n, n // 4)):
        strip.paste(frames[k], (0, i * frames[0].height))
    pal = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    path = path or os.path.join(OUT, "pale_mire_fog_test.gif")
    q[0].save(path, save_all=True, append_images=q[1:], duration=ms, loop=0, optimize=False, disposal=1)
    print("gif", path, os.path.getsize(path) // 1024, "KB")


if __name__ == "__main__":
    main()
