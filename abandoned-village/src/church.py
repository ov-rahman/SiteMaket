"""The ruined chapel - centrepiece of the village (11 x 14 tiles)."""
import numpy as np

from palette import K, V
from pixelkit import Canvas, Rng, dilate, erode, shift, poly_mask, line_points
import materials as M
import shapes as S

W, H = 176, 224
CX = 88
BASE = 196


def _line_y(p0, p1, x):
    (x0, y0), (x1, y1) = p0, p1
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def roof_plane(c, rng, side, apex, eave, depth, lit):
    """Receding roof plane above a verge line (front-gable roof in 3/4 view).

    Courses run north-south (vertical on screen); shingles are short
    vertical slabs staggered from course to course.
    """
    (ax, ay), (ex, ey) = apex, eave
    pts = [(ax, ay - depth), (ex, ey - depth), (ex, ey), (ax, ay)]
    m = poly_mask(W, H, pts)
    pal = M.ROOF_LIT if lit else M.ROOF
    tmp = Canvas(W, H)
    xs = np.where(m.any(axis=0))[0]
    course_w = 4
    for x in xs:
        col = np.where(m[:, x])[0]
        if len(col) == 0:
            continue
        ci = abs(x - ax) // course_w
        k = abs(x - ax) % course_w
        yv = _line_y((ax, ay), (ex, ey), x + 0.5)
        off = (ci * 3) % 5
        for y in col:
            u = int(yv - y) + off
            seg = u // 5
            r = ((ci * 31 + seg * 17) * 2654435761 >> 7) % 100
            tone = pal["base"] if r < 55 else (pal["light"] if r < 78 else pal["dark"])
            v = tone
            if (u % 5) == 0:
                v = pal["gap"]
            elif k == (course_w - 1 if side == "L" else 0):
                v = pal["dark"] if tone != pal["dark"] else pal["lo"]
            elif k == (0 if side == "L" else course_w - 1) and lit:
                v = pal["hi"] if tone == pal["light"] else pal["light"]
            tmp.px[y, x] = v
    c.blit(tmp, 0, 0)
    return m


def verge(c, apex, eave, t, lit=True, end_cut=True):
    (ax, ay), (ex, ey) = apex, eave
    m = S.band_below_line(W, H, (ax, ay), (ex, ey), t)
    c.px[m] = V[3] if not lit else V[4]
    top = m & ~shift(m, 0, 1)
    c.px[top] = V[7] if lit else V[5]
    under = shift(top, 0, 1) & m
    c.px[under] = V[6] if lit else V[4]
    bot = m & ~shift(m, 0, -1)
    c.px[bot] = V[1]
    # rafter ends dotted along the underside
    ys, xs = np.nonzero(bot)
    for y, x in zip(ys, xs):
        if x % 6 == 0 and y - 1 >= 0:
            c.px[y - 1, x] = V[2]
    return m


def church():
    rng = Rng(1901)
    c = Canvas(W, H)

    apex_top = (CX, 50)
    eL, eR = (10, 134), (166, 134)
    depth = 14
    band = 7

    # ---------------------------------------------------------------- roof planes
    roofL = roof_plane(c, rng, "L", apex_top, eL, depth, lit=True)
    roofR = roof_plane(c, rng, "R", apex_top, eR, depth, lit=False)
    # ridge line
    c.vline(CX, apex_top[1] - depth, apex_top[1], V[8])
    c.vline(CX + 1, apex_top[1] - depth, apex_top[1], V[3])

    # ---------------------------------------------------------------- gable wall
    wall_top_L = (CX, apex_top[1] + band)
    wall = poly_mask(W, H, [(22, BASE), (22, _line_y(apex_top, eL, 22) + band - 1),
                            (CX, apex_top[1] + band - 1),
                            (154, _line_y(apex_top, eR, 154) + band - 1), (154, BASE)])
    M.stone_blocks(c, wall, rng, M.STONE, row_h=6, bw=(9, 15), y0=BASE % 6)
    rightwall = wall & (np.arange(W)[None, :] > CX + 30)
    M.stone_blocks(c, rightwall, Rng(1902), M.STONE_SHADE, row_h=6, bw=(9, 15), y0=BASE % 6)

    # ---------------------------------------------------------------- verge bands
    vL = verge(c, apex_top, eL, band, lit=True)
    vR = verge(c, apex_top, eR, band, lit=False)

    # ---------------------------------------------------------------- side lancet windows
    for (xl, xr) in ((32, 42), (134, 144)):
        opening = S.pointed_arch(W, H, xl, xr, 152, 176)
        frame = dilate(opening, True) & ~opening
        c.px[frame] = K[12]
        c.px[frame & ~shift(frame, 0, 1)] = K[13]
        c.px[opening] = K[1]
        c.px[opening & ~shift(opening, 0, 1)] = K[0]
        # mullion stub + broken glass glints
        c.vline((xl + xr) // 2, 162, 175, K[8])
        c.set(xl + 1, 170, K[5])
        c.set(xr - 2, 158, K[5])
        c.hline(xl - 1, xr, 176, K[13])   # sill
        c.hline(xl - 1, xr, 177, K[8])

    # ---------------------------------------------------------------- frontispiece
    f_l, f_r, f_sh, f_ap = 60, 116, 78, 38
    front = poly_mask(W, H, [(f_l, BASE), (f_l, f_sh), (CX, f_ap), (f_r, f_sh), (f_r, BASE)])
    M.stone_blocks(c, front, Rng(77), M.STONE_LIT, row_h=6, bw=(8, 14), y0=BASE % 6)
    # cast shadow of the projecting frontispiece onto the right wall
    shadow_strip = wall & ~front & (np.arange(W)[None, :] >= f_r) & (np.arange(W)[None, :] < f_r + 12) \
        & (np.arange(H)[:, None] > f_sh - 4)
    M.darken(c, shadow_strip)
    # water stains under the rose window
    for sx in (CX - 7, CX - 2, CX + 5):
        st = S.rect(W, H, sx, 118, 2, 18 + (sx % 7)) & front
        M.darken(c, M.dither_mask(st, 0.5))
    # coping on the gable
    for (p0, p1) in (((CX, f_ap - 3), (f_l - 3, f_sh + 1)), ((CX, f_ap - 3), (f_r + 3, f_sh + 1))):
        cm = S.band_below_line(W, H, p0, p1, 4)
        c.px[cm] = K[13]
        c.px[cm & ~shift(cm, 0, 1)] = K[15]
        c.px[cm & ~shift(cm, 0, -1)] = K[5]
        c.px[shift(cm & ~shift(cm, 0, -1), 0, 1) & front & ~cm] = K[8]
    # kneelers (little stone blocks at the gable feet)
    for kx in (f_l - 4, f_r - 3):
        c.rect(kx, f_sh - 2, 8, 7, K[12])
        c.hline(kx, kx + 7, f_sh - 2, K[14])
        c.vline(kx + 7, f_sh - 1, f_sh + 4, K[8])
        c.hline(kx, kx + 7, f_sh + 5, K[5])

    # ---------------------------------------------------------------- buttresses
    for bx, lit in ((f_l - 5, True), (f_r - 2, False)):
        # upper narrow part
        c.rect(bx, f_sh + 6, 7, BASE - f_sh - 6, K[11] if lit else K[10])
        c.vline(bx, f_sh + 6, BASE - 1, K[13] if lit else K[11])
        c.vline(bx + 6, f_sh + 6, BASE - 1, K[8] if lit else K[7])
        # stepped lower part
        c.rect(bx - 2, 150, 11, BASE - 150, K[11] if lit else K[10])
        c.vline(bx - 2, 150, BASE - 1, K[13] if lit else K[11])
        c.vline(bx + 8, 150, BASE - 1, K[8] if lit else K[7])
        # sloped set-off
        for i in range(3):
            c.hline(bx - 2 + i, bx + 8 - i, 147 + i, K[14] if lit else K[12])
        c.hline(bx - 2, bx + 8, 150, K[6])
        # courses
        for y in range(f_sh + 12, BASE, 7):
            c.hline(bx + 1, bx + 5, y, K[8] if lit else K[7])
        for y in range(156, BASE, 7):
            c.hline(bx - 1, bx + 7, y, K[8] if lit else K[7])
        # pinnacle
        px0 = bx + 3
        for i in range(6):
            c.hline(px0 - (i // 2), px0 + (i // 2), f_sh - 2 + i - 6, K[13] if lit else K[11])
        c.set(px0, f_sh - 9, K[15] if lit else K[12])

    # ---------------------------------------------------------------- rose window
    rx, ry = CX, 104
    outer = S.disc(W, H, rx, ry, 13)
    c.px[outer] = K[12]
    c.px[S.ring(W, H, rx, ry, 11.5, 13)] = K[13]
    c.px[S.ring(W, H, rx - 0.7, ry - 0.7, 11.5, 13) & ~S.ring(W, H, rx + 0.8, ry + 0.8, 11.5, 13.5)] = K[15]
    c.px[S.ring(W, H, rx + 0.9, ry + 0.9, 11.8, 13.2) & ~S.disc(W, H, rx - 0.6, ry - 0.6, 13)] = K[8]
    glass = S.disc(W, H, rx, ry, 10)
    c.px[glass] = K[2]
    # faint glass panes catching moonlight
    for (gx, gy) in ((-5, -5), (4, -6), (-6, 3), (5, 4), (-3, -7)):
        c.set(rx + gx, ry + gy, K[5])
        c.set(rx + gx + 1, ry + gy, K[4])
    # tracery: cross + inner ring
    c.rect(rx - 1, ry - 10, 2, 20, K[11])
    c.rect(rx - 10, ry - 1, 20, 2, K[11])
    c.vline(rx - 1, ry - 10, ry + 9, K[13])
    c.hline(rx - 10, rx + 9, ry - 1, K[13])
    c.px[S.ring(W, H, rx, ry, 3.2, 4.6)] = K[11]
    c.px[S.ring(W, H, rx - 0.5, ry - 0.5, 3.2, 4.6) & (np.arange(H)[:, None] < ry)] = K[13]
    c.px[S.disc(W, H, rx, ry, 3.2)] = K[3]
    c.set(rx - 1, ry - 1, K[12])
    # a broken shard: missing glass wedge lower-right
    shard = poly_mask(W, H, [(rx + 2, ry + 3), (rx + 9, ry + 2), (rx + 5, ry + 8)]) & glass
    c.px[shard] = K[0]

    # ---------------------------------------------------------------- door
    dl, dr, ds, db = 77, 99, 164, 194
    op = S.pointed_arch(W, H, dl, dr, ds, db + 2)
    r1 = S.pointed_arch(W, H, dl - 3, dr + 3, ds, db + 2) & ~op
    r2 = S.pointed_arch(W, H, dl - 6, dr + 6, ds, db + 2) & ~op & ~r1
    c.px[r2] = K[11]
    c.px[r2 & ~shift(r2, 1, 0)] = K[13]
    c.px[r2 & ~shift(r2, 0, 1)] = K[13]
    c.px[r1] = K[13]
    c.px[r1 & ~shift(r1, 1, 0)] = K[15]
    c.px[r1 & ~shift(r1, 0, 1)] = K[15]
    c.px[r1 & ~shift(r1, -1, 0)] = K[10]
    c.px[r2 & ~shift(r2, -1, 0)] = K[8]
    # voussoir joints on the rings
    ys, xs = np.nonzero(r1 | r2)
    for y, x in zip(ys, xs):
        if y < ds and (x + y) % 5 == 0:
            c.px[y, x] = K[9]
    c.px[op] = K[0]
    # inner depth: the interior is not totally black
    inner = op & (np.arange(W)[None, :] < dl + 3)
    c.px[inner] = K[2]
    c.px[op & ~shift(op, 0, 1)] = K[1]
    # broken door leaf hanging on the right
    leaf = poly_mask(W, H, [(dr - 5, ds - 6), (dr, ds - 2), (dr, db + 1), (dr - 7, db + 1)]) & op
    M.planks(c, leaf, Rng(8), M.WOOD, width=3, vertical=True)
    c.px[leaf & ~shift(leaf, 1, 0)] = V[2]
    c.hline(dr - 6, dr - 1, ds + 8, V[1])
    c.hline(dr - 7, dr - 1, db - 8, V[1])
    # threshold
    c.hline(dl - 6, dr + 5, db + 1, K[12])

    # ---------------------------------------------------------------- bellcote + cross
    bl, br, bt, bb = 79, 97, 24, 42
    c.rect(bl, bt, br - bl, bb - bt, K[12])
    c.vline(bl, bt, bb - 1, K[14])
    c.vline(br - 1, bt, bb - 1, K[9])
    bell_op = S.round_arch(W, H, bl + 3, br - 3, bt + 8, bb - 3)
    c.px[bell_op] = K[1]
    # the bell
    bm = poly_mask(W, H, [(CX - 2, bt + 7), (CX + 2, bt + 7), (CX + 4, bt + 14), (CX - 4, bt + 14)])
    c.px[bm] = V[4]
    c.px[bm & ~shift(bm, 1, 0)] = V[7]
    c.hline(CX - 4, CX + 3, bt + 14, V[6])
    c.set(CX, bt + 15, V[3])
    c.hline(bl, br - 1, bb - 3, K[13])   # sill
    c.hline(bl, br - 1, bb - 2, K[9])
    # cap (little pointed roof)
    cap = poly_mask(W, H, [(bl - 3, bt + 1), (CX, 13), (br + 3, bt + 1)])
    c.px[cap] = V[5]
    c.px[cap & (np.arange(W)[None, :] < CX)] = V[6]
    c.px[cap & ~shift(cap, 0, 1)] = V[8]
    c.px[cap & ~shift(cap, 0, -1)] = V[2]
    # cross
    cr = Canvas(W, H)
    cr.rect(CX - 1, 0, 3, 16, K[14])
    cr.rect(CX - 6, 4, 13, 3, K[14])
    cr.vline(CX - 1, 0, 15, K[16])
    cr.hline(CX - 6, CX + 6, 4, K[16])
    cr.vline(CX + 1, 7, 15, K[11])
    cr.hline(CX + 2, CX + 6, 6, K[11])
    cr.set(CX - 1, 0, K[17])
    c.blit(cr, 0, 0)
    M.outline(c, K[2], mask=cr.opaque())

    # ---------------------------------------------------------------- ruin: left side
    orig_wall = wall.copy()
    # collapsed eave + wall corner. Inside the old wall outline we see the
    # dark interior; outside it (the eave overhang) is simply gone.
    biteL = poly_mask(W, H, [(4, 100), (26, 96), (34, 104), (41, 118), (39, 130), (35, 138), (37, 148),
                             (31, 156), (33, 166), (26, 172), (21, 176), (8, 176)])
    c.px[biteL & ~orig_wall] = -1
    interior = biteL & orig_wall
    c.px[interior] = K[2]
    c.px[interior & (np.arange(H)[:, None] > 150)] = K[3]
    # far wall of the nave faintly visible through the hole
    for yy in range(112, 170, 6):
        c.px[yy, :][interior[yy]] = K[3]
    # heap of fallen masonry inside: dark mound with loose ashlar blocks on it
    heap = poly_mask(W, H, [(21, 177), (23, 168), (28, 163), (34, 166), (38, 172), (39, 177)]) & interior
    c.px[heap] = K[4]
    c.px[heap & ~shift(heap, 0, 1)] = K[6]
    for (bx, by, bw, bh) in ((24, 168, 5, 3), (30, 165, 4, 3), (27, 172, 6, 3), (33, 170, 4, 3), (22, 174, 4, 2)):
        M.stone(c, bx, by, bw, bh, None, M.STONE_SHADE)
    # jagged masonry lip around the hole
    lip = dilate(interior, True) & ~interior & orig_wall & ~biteL
    c.px[lip] = K[11]
    c.px[lip & ~shift(interior, 0, 1) & shift(interior, 1, 0)] = K[7]
    # exposed rafters sticking out of the broken verge
    for (x0, y0, x1, y1) in ((40, 106, 24, 118), (37, 114, 20, 128), (30, 104, 18, 112)):
        c.line(x0, y0, x1, y1, V[3])
        c.line(x0, y0 - 1, x1, y1 - 1, V[6])
    # the ruined column that survived the collapse
    tx0, tw, ttop = 42, 10, 60
    col = S.rect(W, H, tx0, ttop, tw, BASE - ttop)
    M.stone_blocks(c, col, Rng(31), M.STONE_LIT, row_h=6, bw=(10, 10), y0=BASE % 6, stagger=False)
    c.vline(tx0, ttop, BASE - 1, K[13])
    c.vline(tx0 + 1, ttop, BASE - 1, K[12])
    c.vline(tx0 + tw - 1, ttop, BASE - 1, K[7])
    c.vline(tx0 + tw - 2, ttop, BASE - 1, K[9])
    for yy in (ttop + 22, 148):           # banded rings
        c.hline(tx0 - 1, tx0 + tw, yy, K[13])
        c.hline(tx0 - 1, tx0 + tw, yy + 1, K[7])
    jag = [0, 3, 1, 5, 4, 2, 0, 1, 3, 2]
    for dx, d in enumerate(jag):
        c.vline(tx0 + dx, ttop, ttop + d, -1)
        c.set(tx0 + dx, ttop + d, K[14] if dx < 7 else K[11])
    # vines
    M.vine(c, 47, 72, 70, Rng(41), cols=(K[4], K[6], K[8]))
    M.vine(c, 30, 150, 34, Rng(42), cols=(K[4], K[6], K[8]))
    M.vine(c, 64, 118, 40, Rng(43), cols=(K[5], K[7], K[9]))
    M.vine(c, 26, 176, 16, Rng(46), cols=(K[4], K[6], K[8]), up=True)

    # ---------------------------------------------------------------- ruin: right side
    biteR = poly_mask(W, H, [(176, 100), (152, 104), (142, 116), (145, 126), (141, 134), (146, 142),
                             (150, 150), (148, 160), (155, 168), (176, 168)])
    c.px[biteR & ~orig_wall] = -1
    interiorR = biteR & orig_wall
    c.px[interiorR] = K[1]
    c.px[interiorR & (np.arange(H)[:, None] > 146)] = K[2]
    lipR = dilate(interiorR, True) & ~interiorR & orig_wall & ~biteR
    c.px[lipR] = K[9]
    # the broken end of the verge sags down like a snapped plank
    sag = poly_mask(W, H, [(143, 115), (150, 113), (163, 138), (158, 142)])
    c.px[sag] = V[4]
    c.px[sag & ~shift(sag, 1, 0)] = V[6]
    c.px[sag & ~shift(sag, -1, 0)] = V[2]
    c.px[sag & ~shift(sag, 0, -1)] = V[1]
    for t in range(4, 26, 5):
        c.set(146 + t // 2, 116 + t, V[2])
    # dangling rafters
    for (x0, y0, x1, y1) in ((142, 112, 156, 104), (146, 122, 160, 118), (150, 130, 158, 132)):
        c.line(x0, y0, x1, y1, V[3])
        c.line(x0, y0 - 1, x1, y1 - 1, V[5])
    # cracks + ivy
    M.crack(c, 128, 124, 30, Rng(51), K[5], dirx=1, mask=wall)
    M.crack(c, 70, 128, 22, Rng(52), K[8], dirx=-1, mask=front)
    M.crack(c, 106, 176, 14, Rng(53), K[8], dirx=1, mask=front)
    M.vine(c, 118, 90, 54, Rng(44), cols=(K[4], K[6], K[8]))
    M.vine(c, 140, 150, 30, Rng(45), cols=(K[4], K[5], K[7]))

    # ---------------------------------------------------------------- foundation / plinth
    c.rect(20, BASE - 4, 136, 4, K[8])
    c.hline(20, 155, BASE - 4, K[10])
    for x in range(22, 156, 9):
        c.vline(x, BASE - 3, BASE - 1, K[5])

    # ---------------------------------------------------------------- steps
    steps = [(BASE - 2, 34), (BASE + 3, 40), (BASE + 8, 46), (BASE + 13, 52)]
    for i, (sy, sw) in enumerate(steps):
        x0 = CX - sw // 2
        c.rect(x0, sy, sw, 5, K[11])
        c.hline(x0, x0 + sw - 1, sy, K[14])
        c.hline(x0, x0 + sw - 1, sy + 1, K[13])
        c.hline(x0, x0 + sw - 1, sy + 4, K[7])
        c.vline(x0, sy, sy + 4, K[12])
        c.vline(x0 + sw - 1, sy + 1, sy + 4, K[8])
        # worn centre + cracks
        c.hline(CX - 5, CX + 4, sy + 2, K[12])
        r = Rng(90 + i)
        for _ in range(2):
            cx = r.randint(x0 + 3, x0 + sw - 4)
            c.vline(cx, sy + 2, sy + 3, K[8])
    # side cheeks of the steps
    for side in (-1, 1):
        for i, (sy, sw) in enumerate(steps):
            x = CX + side * (sw // 2) - (1 if side > 0 else 0)
    # a broken step corner
    c.rect(CX + 20, BASE + 13, 6, 5, -1)
    c.rect(CX + 20, BASE + 14, 5, 4, K[8])
    c.hline(CX + 20, CX + 24, BASE + 14, K[12])

    # ---------------------------------------------------------------- free-standing broken pillars
    for (px0, ptop, pw, seed) in ((1, 140, 9, 61), (166, 128, 9, 62)):
        col = S.rect(W, H, px0, ptop, pw, BASE + 4 - ptop)
        M.stone_blocks(c, col, Rng(seed), M.STONE_LIT, row_h=6, bw=(9, 9), y0=(BASE + 4) % 6)
        c.vline(px0, ptop, BASE + 3, K[13])
        c.vline(px0 + pw - 1, ptop, BASE + 3, K[7])
        # capital ring
        c.hline(px0 - 1, px0 + pw, ptop + 10, K[13])
        c.hline(px0 - 1, px0 + pw, ptop + 11, K[8])
        # jagged broken top
        r = Rng(seed + 5)
        for dx in range(pw):
            d = r.randint(0, 5)
            c.vline(px0 + dx, ptop, ptop + d, -1)
            c.set(px0 + dx, ptop + d + 1, K[14])
        # base block
        c.rect(px0 - 1, BASE, pw + 2, 5, K[10])
        c.hline(px0 - 1, px0 + pw, BASE, K[12])

    # ---------------------------------------------------------------- outline
    body = c.opaque()
    M.selective_outline(c, body, light=K[3], dark=K[1])

    # ---------------------------------------------------------------- rubble at the base
    r = Rng(71)
    for (x, y, n) in ((30, BASE + 4, 7), (140, BASE + 4, 8), (8, BASE + 8, 4), (60, BASE + 10, 3),
                      (118, BASE + 12, 4), (168, BASE + 8, 4)):
        M.rubble(c, x, y, n, r, spread=(9, 3), pal=M.STONE_LIT, sizes=(1, 3))
    return c


def church_shadow():
    """Soft shadow cast on the ground (to the lower right)."""
    from palette import SH1, SH3
    c = Canvas(W, H)
    m = poly_mask(W, H, [(20, BASE - 6), (160, BASE - 6), (176, BASE + 6), (176, BASE + 10), (28, BASE + 10)])
    c.px[m] = SH1
    return c


if __name__ == "__main__":
    import sys
    c = church()
    c.save(sys.argv[1], 4)
