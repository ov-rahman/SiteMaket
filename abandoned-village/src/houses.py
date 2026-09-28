"""Four abandoned cottages, each broken in its own way."""
import numpy as np

from palette import K, V
from pixelkit import Canvas, Rng, dilate, erode, shift, poly_mask
import materials as M
import shapes as S


# ================================================================ shared parts

def stone_wall(c, x0, y0, x1, y1, rng, pal=M.STONE, row_h=6, bw=(8, 13)):
    m = S.rect(c.w, c.h, x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    M.stone_blocks(c, m, rng, pal, row_h=row_h, bw=bw, y0=(y1 + 1) % row_h)
    # lit left edge, shaded right edge
    c.vline(x0, y0, y1, pal["light"])
    c.vline(x1, y0, y1, pal["lo"])
    c.vline(x1 - 1, y0, y1, pal["dark"])
    return m


def foundation(c, x0, x1, y, pal=M.STONE):
    ao = S.rect(c.w, c.h, x0, y - 9, x1 - x0 + 1, 6) & (c.px >= 0)
    M.darken(c, M.dither_mask(ao & (np.arange(c.h)[:, None] < y - 6), 0.5))
    M.darken(c, ao & (np.arange(c.h)[:, None] >= y - 6))
    c.rect(x0, y - 3, x1 - x0 + 1, 3, pal["dark"])
    c.hline(x0, x1, y - 3, pal["base"])
    for x in range(x0 + 2, x1, 7):
        c.vline(x, y - 2, y - 1, pal["mortar"])


def eave_shadow(c, x0, x1, y, h=3):
    c.px[y:y + h, x0:x1 + 1] = np.where(c.px[y:y + h, x0:x1 + 1] >= 0, K[4], c.px[y:y + h, x0:x1 + 1])
    c.hline(x0, x1, y, K[2])
    if h > 2:
        m = S.rect(c.w, c.h, x0, y + h - 1, x1 - x0 + 1, 1)
        c.px[M.dither_mask(m, 0.5)] = K[5]


def doorway(c, x, y, w, h, rng, leaf="left", arch=False):
    """Open dark doorway with a wooden frame; optional broken leaf."""
    m = S.rect(c.w, c.h, x, y, w, h)
    if arch:
        m = S.round_arch(c.w, c.h, x, x + w, y + w // 2, y + h)
    frame = dilate(m) & ~m & (np.arange(c.h)[:, None] < y + h)
    frame2 = dilate(frame) & ~m & ~frame & (np.arange(c.h)[:, None] < y + h)
    c.px[frame2] = V[2]
    c.px[frame] = V[5]
    c.px[frame & ~shift(frame, 1, 0)] = V[6]
    c.px[m] = K[0]
    # a bit of the dark interior floor
    c.px[m & (np.arange(c.h)[:, None] >= y + h - 3)] = K[2]
    c.px[m & ~shift(m, 0, 1)] = K[1]
    # lintel
    c.rect(x - 2, y - 3, w + 4, 2, K[11])
    c.hline(x - 2, x + w + 1, y - 3, K[13])
    if leaf == "left":
        lm = poly_mask(c.w, c.h, [(x, y), (x + 5, y + 2), (x + 5, y + h), (x, y + h)])
        M.planks(c, lm, rng, M.WOOD, width=3, vertical=True)
        c.px[lm & ~shift(lm, -1, 0)] = V[1]
        c.hline(x, x + 4, y + 5, V[2])
        c.hline(x, x + 4, y + h - 6, V[2])
    elif leaf == "right":
        lm = poly_mask(c.w, c.h, [(x + w - 5, y + 2), (x + w, y), (x + w, y + h), (x + w - 5, y + h)])
        M.planks(c, lm, rng, M.WOOD, width=3, vertical=True)
        c.px[lm & ~shift(lm, 1, 0)] = V[1]
        c.hline(x + w - 5, x + w - 1, y + 5, V[2])
        c.hline(x + w - 5, x + w - 1, y + h - 6, V[2])
    elif leaf == "fallen":
        # door lying on the threshold, seen at an angle
        lm = poly_mask(c.w, c.h, [(x - 1, y + h - 5), (x + w, y + h - 7), (x + w + 2, y + h - 2), (x + 1, y + h)])
        M.planks(c, lm, rng, M.WOOD_LIT, width=3, vertical=True)
        c.px[lm & ~shift(lm, 0, -1)] = V[1]
    return m


def window(c, x, y, w, h, broken=0, boarded=False, rng=None, pane_col=K[2]):
    """Four-pane window with a pale frame (catches the moonlight)."""
    rng = rng or Rng(1)
    # outer frame
    c.rect(x - 1, y - 1, w + 2, h + 2, K[5])
    c.rect(x, y, w, h, K[12])
    c.hline(x, x + w - 1, y, K[14])
    c.vline(x, y, y + h - 1, K[13])
    c.vline(x + w - 1, y + 1, y + h - 1, K[10])
    # panes
    mx, my = x + w // 2, y + h // 2
    panes = [(x + 2, y + 2, mx - x - 2, my - y - 2), (mx + 1, y + 2, x + w - 3 - mx, my - y - 2),
             (x + 2, my + 1, mx - x - 2, y + h - 3 - my), (mx + 1, my + 1, x + w - 3 - mx, y + h - 3 - my)]
    for i, (px, py, pw, ph) in enumerate(panes):
        c.rect(px, py, pw, ph, pane_col)
        c.hline(px, px + pw - 1, py, K[1])     # shadow of the frame
        if i < 2 or rng.chance(0.5):
            c.set(px + pw - 2, py + 1, K[6])       # moon glint
            c.set(px + pw - 3, py + 2, K[5])
    for i in range(broken):
        px, py, pw, ph = panes[(i * 3 + 1) % 4]
        c.rect(px, py, pw, ph, K[0])
        # shards on the edges
        c.set(px, py + ph - 1, K[7])
        c.set(px + pw - 1, py, K[7])
        c.set(px + 1, py + ph - 1, K[5])
    if boarded:
        for (yy, d) in ((y + 3, 2), (y + h - 5, -2)):
            bm = poly_mask(c.w, c.h, [(x - 2, yy), (x + w + 1, yy - d), (x + w + 1, yy - d + 3), (x - 2, yy + 3)])
            c.px[bm] = V[5]
            c.px[bm & ~shift(bm, 0, 1)] = V[7]
            c.px[bm & ~shift(bm, 0, -1)] = V[2]
    # sill
    c.rect(x - 2, y + h, w + 4, 2, K[12])
    c.hline(x - 2, x + w + 1, y + h, K[14])
    c.hline(x - 1, x + w, y + h + 2, K[5])


def chimney(c, x, y_top, w, h, rng, broken=False, side=3, pal=M.STONE, cast=True):
    """Stone chimney: lit front face + shaded right face + dark flue."""
    if cast:
        # moonlight comes from the upper left: shadow falls onto the roof to the right
        sh = poly_mask(c.w, c.h, [(x + w + side, y_top + 6), (x + w + side + 6, y_top + 12),
                                  (x + w + side + 6, y_top + h + 3), (x + 2, y_top + h + 3), (x + 2, y_top + h)])
        sh &= (c.px >= 0) & ~S.rect(c.w, c.h, x, y_top, w + side, h)
        M.darken(c, sh)
    m = S.rect(c.w, c.h, x, y_top, w, h)
    M.stone_blocks(c, m, rng, pal, row_h=4, bw=(4, 7), y0=y_top % 4)
    c.vline(x, y_top, y_top + h - 1, pal["hi"])
    sm = S.rect(c.w, c.h, x + w, y_top + 1, side, h - 1)
    M.stone_blocks(c, sm, rng, M.STONE_SHADE, row_h=4, bw=(3, 3), y0=y_top % 4)
    c.vline(x + w + side - 1, y_top + 1, y_top + h - 1, K[5])
    # cap
    c.rect(x - 1, y_top - 2, w + side + 2, 3, K[12])
    c.hline(x - 1, x + w + side, y_top - 2, K[14])
    c.hline(x - 1, x + w + side, y_top, K[7])
    c.rect(x + 2, y_top - 2, w - 3, 2, K[0])
    c.hline(x + 2, x + w - 2, y_top - 2, K[1])
    if broken:
        for dx, d in enumerate([0, 2, 1, 4, 3, 5, 2, 1, 0, 3]):
            if dx < w + side + 2:
                c.vline(x - 1 + dx, y_top - 2, y_top - 2 + d, -1)
                if d < h:
                    c.set(x - 1 + dx, y_top - 1 + d, K[13] if dx < w else K[9])


def hip_roof(c, eave_y, ridge_y, ex0, ex1, rx0, rx1, back_y, rng, pal=M.ROOF, pal_lit=M.ROOF_LIT):
    """Hip roof in 3/4 view: front trapezoid + lit left hip + shaded right hip."""
    W, H = c.w, c.h
    front = poly_mask(W, H, [(ex0, eave_y), (rx0, ridge_y), (rx1, ridge_y), (ex1, eave_y)])
    left = poly_mask(W, H, [(ex0, eave_y), (ex0, back_y), (rx0, ridge_y)])
    right = poly_mask(W, H, [(ex1, eave_y), (rx1, ridge_y), (ex1, back_y)])
    M.shingles(c, front, rng, pal, row_h=5, sw=(5, 8))
    M.shingles(c, left & ~front, rng, pal_lit, row_h=5, sw=(4, 6))
    M.shingles(c, right & ~front, rng, M.ROOF | dict(base=V[4], light=V[5], dark=V[3], hi=V[5], lo=V[2]),
               row_h=5, sw=(4, 6))
    roof = front | left | right
    # hip lines (ridge-to-corner) catch the light
    for (a, b, col) in (((ex0, eave_y), (rx0, ridge_y), V[8]), ((ex1, eave_y), (rx1, ridge_y), V[3])):
        c.line(a[0], a[1], b[0], b[1], col)
    # ridge cap
    c.rect(rx0, ridge_y - 1, rx1 - rx0 + 1, 3, V[6])
    c.hline(rx0, rx1, ridge_y - 1, V[8])
    c.hline(rx0, rx1, ridge_y + 1, V[3])
    for x in range(rx0 + 2, rx1, 4):
        c.set(x, ridge_y, V[4])
    # eave edge
    c.hline(ex0, ex1, eave_y - 1, V[2])
    c.hline(ex0, ex1, eave_y - 2, V[6])
    return roof | S.rect(W, H, rx0, ridge_y - 1, rx1 - rx0 + 1, 3)


def gable_front_roof(c, apex, eL, eR, depth, rng, pal_l=M.ROOF_LIT, pal_r=M.ROOF, verge_t=4):
    """Front-gable roof: two receding planes above the verge lines."""
    W, H = c.w, c.h
    (ax, ay), (lx, ly), (rx, ry) = apex, eL, eR
    left = poly_mask(W, H, [(ax, ay - depth), (lx, ly - depth), (lx, ly), (ax, ay)])
    right = poly_mask(W, H, [(ax, ay - depth), (ax, ay), (rx, ry), (rx, ry - depth)])
    for (m, pal, side) in ((left, pal_l, -1), (right, pal_r, 1)):
        tmp = Canvas(W, H)
        xs = np.where(m.any(axis=0))[0]
        for x in xs:
            ci = abs(x - ax) // 4
            k = abs(x - ax) % 4
            ys = np.where(m[:, x])[0]
            yv = ys.max()
            off = (ci * 3) % 6
            for y in ys:
                u = int(yv - y) + off
                seg = u // 6
                r = ((ci * 31 + seg * 17) * 2654435761 >> 7) % 100
                tone = pal["base"] if r < 55 else (pal["light"] if r < 78 else pal["dark"])
                v = tone
                if u % 6 == 0:
                    v = pal["gap"]
                elif k == (3 if side < 0 else 0):
                    v = pal["dark"] if tone != pal["dark"] else pal["lo"]
                elif k == (0 if side < 0 else 3):
                    v = pal["hi"] if tone == pal["light"] else pal["light"]
                tmp.px[y, x] = v
        c.blit(tmp, 0, 0)
    # ridge
    c.vline(ax, ay - depth, ay, V[8])
    c.vline(ax + 1, ay - depth, ay, V[3])
    # verges (bargeboards)
    vl = S.band_below_line(W, H, (ax, ay), (lx, ly), verge_t)
    vr = S.band_below_line(W, H, (ax, ay), (rx, ry), verge_t)
    for vm, lit in ((vl, True), (vr, False)):
        c.px[vm] = V[5] if lit else V[4]
        c.px[vm & ~shift(vm, 0, 1)] = V[8] if lit else V[6]
        c.px[vm & ~shift(vm, 0, -1)] = V[1]
    return left | right | vl | vr


# ================================================================ house A
def house_a():
    """Stone cottage, hip roof full of holes, chimney on the right."""
    W, H = 112, 112
    c = Canvas(W, H)
    rng = Rng(2101)
    wl, wr, wt, wb = 10, 101, 62, 105
    stone_wall(c, wl, wt, wr, wb, rng)
    foundation(c, wl, wr, wb + 1)
    roof = hip_roof(c, eave_y=68, ridge_y=18, ex0=4, ex1=107, rx0=28, rx1=84, back_y=30, rng=rng)
    # chimney (drawn over the roof, base hidden in the slope)
    chimney(c, 72, 8, 9, 26, Rng(7), broken=False)
    # patch the roof back over the chimney base (it sits behind the front slope)
    # holes
    M.roof_hole(c, roof, M.jagged_blob(W, H, 44, 44, 11, 7, Rng(3), rough=0.45), Rng(4))
    M.roof_hole(c, roof, M.jagged_blob(W, H, 64, 56, 6, 4, Rng(5), rough=0.5), Rng(6), rafters=False)
    M.roof_hole(c, roof, M.jagged_blob(W, H, 22, 56, 5, 4, Rng(9), rough=0.5), Rng(10), rafters=False)
    eave_shadow(c, wl, wr, 68)
    # door + window
    doorway(c, 20, 80, 14, 26, Rng(11), leaf="left")
    window(c, 60, 76, 18, 15, broken=1, rng=Rng(12))
    # weathering
    M.crack(c, 50, 72, 16, Rng(13), K[6], dirx=1)
    M.crack(c, 90, 80, 14, Rng(14), K[6], dirx=-1)
    M.vine(c, 12, 104, 24, Rng(15), up=True)
    M.vine(c, 99, 104, 18, Rng(16), up=True)
    M.selective_outline(c, light=K[3], dark=K[1])
    return c


# ================================================================ house B
def house_b():
    """Cottage whose roof caved in: bare rafters over a black interior."""
    W, H = 112, 112
    c = Canvas(W, H)
    rng = Rng(2201)
    wl, wr, wt, wb = 10, 101, 62, 105
    stone_wall(c, wl, wt, wr, wb, rng)
    foundation(c, wl, wr, wb + 1)
    roof = hip_roof(c, eave_y=68, ridge_y=18, ex0=4, ex1=107, rx0=28, rx1=84, back_y=30, rng=rng)
    # ---- the collapse
    hole = poly_mask(W, H, [(38, 17), (60, 16), (72, 21), (86, 17), (98, 26), (102, 34), (96, 40),
                            (100, 50), (94, 58), (84, 62), (76, 60), (66, 64), (54, 58), (46, 60),
                            (40, 50), (44, 42), (36, 34), (42, 26)]) & roof
    c.px[hole] = K[0]
    # far side of the attic: the back slope's underside, faint
    back = hole & (np.arange(H)[:, None] < 34)
    c.px[back] = K[1]
    for yy in range(20, 34, 4):
        c.px[yy, :][back[yy]] = K[2]
    # sagging ridge beam
    beam = [(36, 19), (52, 20), (62, 23), (70, 27)]
    for (p, q) in zip(beam[:-1], beam[1:]):
        c.line(p[0], p[1], q[0], q[1], V[5])
        c.line(p[0], p[1] + 1, q[0], q[1] + 1, V[2])
    c.set(70, 27, V[7])
    c.set(71, 28, V[6])
    # rafters: some whole, some snapped, one fallen
    for (x0, y0, x1, y1) in ((46, 20, 44, 58), (56, 21, 55, 62), (88, 20, 90, 44), (96, 30, 98, 40)):
        c.line(x0, y0, x1, y1, V[4])
        c.line(x0 + 1, y0, x1 + 1, y1, V[2])
        c.set(x0, y0, V[6])
    for (x0, y0, x1, y1) in ((66, 22, 68, 34), (78, 20, 79, 30)):     # snapped stubs
        c.line(x0, y0, x1, y1, V[4])
        c.set(x1, y1, V[7])
    c.line(62, 58, 86, 36, V[5])                                       # fallen rafter
    c.line(62, 59, 86, 37, V[2])
    c.set(86, 36, V[7])
    # broken shingle lips around the collapse
    top_lip = shift(hole & ~shift(hole, 0, 1), 0, -1) & roof & ~hole
    c.px[top_lip] = V[8]
    bot_lip = shift(hole & ~shift(hole, 0, -1), 0, 1) & roof & ~hole
    c.px[bot_lip] = V[2]
    side = (shift(hole, 1, 0) | shift(hole, -1, 0)) & roof & ~hole & ~top_lip & ~bot_lip
    c.px[side] = V[3]
    # loose slates slipping down the slope
    for (x, y) in ((30, 50), (70, 64), (24, 60)):
        c.rect(x, y, 5, 3, V[6])
        c.hline(x, x + 4, y, V[8])
        c.hline(x, x + 4, y + 3, V[1])
    # ---- front wall top partly fallen on the right
    for (x, y, w, h) in ((78, wt - 2, 10, 5), (88, wt - 2, 8, 3)):
        c.rect(x, y, w, h, K[1])
        c.hline(x, x + w - 1, y + h, K[12])
    eave_shadow(c, wl, 77, 68)
    doorway(c, 44, 80, 14, 26, Rng(11), leaf="right")
    window(c, 72, 78, 18, 14, broken=2, rng=Rng(21))
    M.crack(c, 30, 72, 20, Rng(22), K[6], dirx=1)
    M.crack(c, 84, 70, 14, Rng(25), K[6], dirx=-1)
    M.vine(c, 12, 104, 30, Rng(23), up=True)
    M.vine(c, 99, 104, 14, Rng(24), up=True)
    # rubble fallen from the roof
    r = Rng(26)
    for _ in range(5):
        M.stone(c, r.randint(66, 98), r.randint(100, 104), r.randint(2, 3), 2, r, M.STONE)
    M.selective_outline(c, light=K[3], dark=K[1])
    return c


# ================================================================ house C
def house_c():
    """Front-gable cottage with a tall chimney and a caved-in slope."""
    W, H = 128, 112
    c = Canvas(W, H)
    rng = Rng(2301)
    apex = (58, 40)
    eL, eR = (8, 74), (108, 74)
    depth = 26
    roof = gable_front_roof(c, apex, eL, eR, depth, rng)
    # big hole in the right slope
    hole = M.jagged_blob(W, H, 84, 44, 13, 11, Rng(3), rough=0.5, n=17)
    M.roof_hole(c, roof, hole, Rng(4), rafter_dx=7, rafter_slope=(eR[1] - apex[1]) / (eR[0] - apex[0]))
    M.roof_hole(c, roof, M.jagged_blob(W, H, 30, 46, 4, 4, Rng(5)), Rng(6), rafters=False)
    # front gable wall (pentagon)
    wl, wr, wb = 14, 102, 105
    front = poly_mask(W, H, [(wl, wb + 1), (wl, 74), (apex[0], apex[1] + 4), (wr + 1, 74), (wr + 1, wb + 1)])
    M.stone_blocks(c, front, rng, M.STONE, row_h=6, bw=(8, 13), y0=(wb + 1) % 6)
    c.vline(wl, 74, wb, K[10])
    c.vline(wr, 74, wb, K[7])
    # verge redraw over the wall top
    for (p1, lit) in ((eL, True), (eR, False)):
        vm = S.band_below_line(W, H, apex, p1, 4)
        c.px[vm] = V[5] if lit else V[4]
        c.px[vm & ~shift(vm, 0, 1)] = V[8] if lit else V[6]
        c.px[vm & ~shift(vm, 0, -1)] = V[1]
        c.px[shift(vm & ~shift(vm, 0, -1), 0, 1) & front] = K[4]
    # small attic vent in the gable
    c.rect(54, 56, 8, 6, K[1])
    c.rect(55, 57, 6, 4, K[0])
    for x in (55, 57, 59):
        c.vline(x, 57, 60, V[4])
    c.hline(53, 62, 55, K[12])
    c.hline(53, 62, 62, K[12])
    foundation(c, wl, wr, wb + 1)
    doorway(c, 24, 80, 14, 26, Rng(11), leaf="fallen")
    window(c, 66, 78, 18, 14, broken=4, boarded=True, rng=Rng(12))
    # tall chimney on the left slope
    chimney(c, 22, 6, 10, 42, Rng(13), broken=True)
    M.crack(c, 90, 84, 16, Rng(14), K[6], dirx=-1)
    M.vine(c, 16, 104, 30, Rng(15), up=True)
    M.vine(c, 100, 104, 22, Rng(16), up=True)
    M.selective_outline(c, light=K[3], dark=K[1])
    return c


# ================================================================ house D
def house_d():
    """Wide cottage, shingled roof, a tall burnt-out chimney in front."""
    W, H = 144, 128
    c = Canvas(W, H)
    rng = Rng(2401)
    wl, wr, wt, wb = 10, 131, 78, 121
    stone_wall(c, wl, wt, wr, wb, rng, pal=M.STONE)
    foundation(c, wl, wr, wb + 1)
    roof = hip_roof(c, eave_y=84, ridge_y=34, ex0=4, ex1=138, rx0=26, rx1=116, back_y=46, rng=rng)
    # ragged, broken-off right corner of the roof: dark attic + snapped rafters
    bite = poly_mask(W, H, [(144, 38), (124, 42), (128, 50), (120, 56), (126, 62), (121, 70),
                            (128, 76), (126, 84), (144, 84)])
    c.px[bite & ~roof] = -1
    att = bite & roof
    c.px[att] = K[1]
    c.px[att & (np.arange(H)[:, None] > 78)] = K[2]
    c.hline(118, 138, 82, V[3])            # wall plate
    c.hline(118, 138, 81, V[5])
    for (x0, y0, x1, y1) in ((120, 50, 140, 44), (122, 60, 141, 56), (124, 70, 140, 68)):
        c.line(x0, y0, x1, y1, V[4])
        c.line(x0, y0 - 1, x1, y1 - 1, V[6])
    lip = shift(att, -1, 0) & roof & ~att
    c.px[lip] = V[8]
    M.roof_hole(c, roof, M.jagged_blob(W, H, 100, 62, 9, 6, Rng(3)), Rng(4))
    # burnt chimney rising out of the front slope, its face split open
    cx0 = 56
    chimney(c, cx0, 20, 16, 50, Rng(5), broken=True, side=4)
    gash = poly_mask(W, H, [(cx0 + 4, 19), (cx0 + 12, 19), (cx0 + 11, 30), (cx0 + 12, 40),
                            (cx0 + 9, 50), (cx0 + 8, 58), (cx0 + 6, 48), (cx0 + 4, 38), (cx0 + 5, 28)])
    c.px[gash] = K[0]
    c.px[gash & (np.arange(W)[None, :] > cx0 + 8)] = K[1]
    c.px[shift(gash, -1, 0) & ~gash & S.rect(W, H, cx0, 18, 16, 44)] = K[13]
    c.px[shift(gash, 1, 0) & ~gash & S.rect(W, H, cx0, 18, 16, 44)] = K[6]

    eave_shadow(c, wl, wr, 84)
    doorway(c, 34, 94, 14, 28, Rng(11), leaf="left")
    window(c, 84, 90, 22, 16, broken=0, rng=Rng(12))
    # long crack with ivy growing out of it
    M.crack(c, 64, 88, 30, Rng(13), K[4], dirx=1)
    M.vine(c, 66, 88, 30, Rng(14))
    M.vine(c, 12, 120, 20, Rng(15), up=True)
    M.vine(c, 128, 120, 26, Rng(16), up=True)
    M.selective_outline(c, light=K[3], dark=K[1])
    return c


if __name__ == "__main__":
    import sys
    for f in (house_a, house_b, house_c, house_d):
        f().save(f"{sys.argv[1]}/{f.__name__}.png", 4)
