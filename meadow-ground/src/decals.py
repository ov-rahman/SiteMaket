"""Flat ground details (20x20, transparent): flowers, tufts, pebbles, stones…
Drawn by hand as tiny ASCII sprites - simple shapes, flat colours, one
dark outline tone, the way Undertale draws small things."""
from palette import G, L, P, F, S, SH1
from pixelkit import Canvas, from_ascii, Rng

TS = 20

# chars: grass 'abcde' lush 'fghij' path 'klmno'
#        flowers 'p'(dark gold) 'q'(gold) 'r'(yellow) 's'(pale) 't'(white) 'u'/'v'(pink) 'w'/'x'(blue)
#        stone 'y'(outline) 'z' '1' '2'(light)   '+' soft shadow
DAISY = [".t.",
         "tst",
         ".t."]
DAISY_BIG = [".tt.",
             "tqst",
             "tsst",
             ".tt."]
BUTTERCUP = [".sr.",
             "srrq",
             "rpqq",
             ".qq."]
FORGET_ME_NOT = [".x.",
                 "xsx",
                 ".w."]
CLOVER = [".bb.b",
          "bccbc",
          "bcbcb",
          ".b.b."]
PINK = [".v.",
        "vuv",
        ".u."]
DANDELION = [".t.t.",
             "t.t.t",
             ".ttt.",
             "..b..",
             "..b.."]
TUFT_TALL = ["d...d",
             "c.d.c",
             "bcbcb",
             ".bbb."]
TUFT = ["d.d",
        "bcb"]
TUFT_WIDE = ["d..d..d",
             "c.dcd.c",
             "bcbbbcb"]
PEBBLES = [["on.", "ll.", "..n"], ["on", "ll"], [".on", "onl", "ll."]]
STONE = ["..yyyy..",
         ".y2221y.",
         "y2211z1y",
         "y1111zzy",
         ".yzzzzy.",
         "..yyyy.."]
STEP = [".yyyyy.",
        "y22211y",
        "y2111zy",
        ".yzzzy."]
LEAF = ["..nn",
        ".nml",
        "nml.",
        "l..."]
MUSHROOMS = [".uu...",
             "uvvu..",
             ".ot..u",
             ".ot.uvu",
             "......o"]


def _place(rows, spots):
    c = Canvas(TS, TS)
    t = from_ascii(rows)
    for (x, y) in spots:
        c.blit(t, x, y)
    return c


def build():
    d = {}
    d["daisies_a"] = _place(DAISY, [(3, 4), (12, 9), (6, 14)])
    d["daisies_b"] = _place(DAISY, [(10, 3), (4, 11)])
    d["daisy_big"] = _place(DAISY_BIG, [(8, 8)])
    d["daisies_mix"] = _place(DAISY, [(2, 13), (14, 5)])
    d["daisies_mix"].blit(from_ascii(DAISY_BIG), 8, 10)
    d["buttercups"] = _place(BUTTERCUP, [(3, 3), (12, 11)])
    d["buttercup"] = _place(BUTTERCUP, [(8, 8)])
    d["forget_me_nots"] = _place(FORGET_ME_NOT, [(4, 5), (11, 3), (8, 12), (14, 13)])
    d["pink_flowers"] = _place(PINK, [(5, 8), (12, 12)])
    d["clover"] = _place(CLOVER, [(7, 8)])
    d["clover_pair"] = _place(CLOVER, [(2, 3), (12, 12)])
    d["dandelion"] = _place(DANDELION, [(7, 6)])
    d["tuft_tall"] = _place(TUFT_TALL, [(7, 8)])
    d["tufts"] = _place(TUFT, [(3, 5), (13, 12)])
    d["tuft_wide"] = _place(TUFT_WIDE, [(6, 9)])
    d["pebbles_a"] = _place(PEBBLES[0], [(4, 6)])
    d["pebbles_b"] = _place(PEBBLES[1], [(12, 4)])
    d["pebbles_b"].blit(from_ascii(PEBBLES[2]), 5, 12)
    d["stone"] = _place(STONE, [(6, 7)])
    d["stepping_stone"] = _place(STEP, [(6, 8)])
    d["leaves"] = _place(LEAF, [(4, 4), (12, 11)])
    d["mushrooms"] = _place(MUSHROOMS, [(7, 7)])
    # soft shadow under raised things
    for k in ("stone", "stepping_stone", "mushrooms"):
        c = d[k]
        solid = c.px >= 0
        import numpy as np
        sh = np.roll(np.roll(solid, 1, 0), 1, 1) & ~solid
        c.px[sh] = SH1
    return d
