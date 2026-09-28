"""'The Wanderer' - an original player character, 16x24, 4 directions x 3 frames.

Sheet layout (48 x 96):   columns = idle, step A, step B
                          rows    = down, left, right, up
Walk cycle: idle, A, idle, B.
"""
from pixelkit import Canvas, from_ascii

FW, FH = 16, 24

# k l m n = hair (dark→light)   o p q = skin   r s t = scarf
# B C = trousers   D E F G H I = sweater (violet ramp)   0 1 = outlines/shoes
HEAD_DOWN = [
    "................",
    ".....111111.....",
    "....1mmnnmm1....",
    "...1mnnnmmmm1...",
    "..1mmnmmmmmml1..",
    "..1lmmmmmmmml1..",
    "..1lmlmmlmmll1..",
    "..1llpqqqqpll1..",
    "..1lpqqqqqqpl1..",
    "...1q1qqqq1q1...",
    "...1pqqqqqqp1...",
    "....1oppppo1....",
]
BODY_DOWN = [
    "....1rsttsr1....",
    "...1Ersssssr1...",
    "..1EFGGGsrGFE1..",
    "..1EFIIIsIIFE1..",
    ".1pEFGGGrGGFEp1.",
    ".1oEFGGGGGGFEo1.",
    "..1EEFFFFFFEE1..",
]
HEAD_UP = [
    "................",
    ".....111111.....",
    "....1mmnnmm1....",
    "...1mnnnnmmm1...",
    "..1mnnmmmmmml1..",
    "..1mmnmmmmmml1..",
    "..1lmmmmmmmll1..",
    "..1lmmmmmmmll1..",
    "..1llmmmmmlll1..",
    "...1llmmmlll1...",
    "...1lllllllk1...",
    "....1llllll1....",
]
BODY_UP = [
    "....1rssssr1....",
    "...1Errrsssr1...",
    "..1EFGGGGGsrE1..",
    "..1EFIIIIIsIE1..",
    ".1oEFGGGGGrGEo1.",
    ".1oEFGGGGGGFEo1.",
    "..1EEFFFFFFEE1..",
]
HEAD_LEFT = [
    "................",
    "......11111.....",
    ".....1mmnnm1....",
    "....1mnnnmmm1...",
    "...1mnnmmmmmm1..",
    "..1mmmmmmmmmml1.",
    "..1lmlmmmmmmml1.",
    ".1lqqpmmmmmmll1.",
    ".1qqqqqplmmmll1.",
    ".1q1qqqplmmll1..",
    "..1pqqqpllll1...",
    "...1oppo1ll1....",
]
BODY_LEFT = [
    "....1rsstr1.....",
    "....1Essssr1....",
    "...1EFGGGsGE1...",
    "...1EFIIIsIE1...",
    "...1EpGGGrFE1...",
    "...1EoGGGGFE1...",
    "....1EFFFFE1....",
]
LEGS_FRONT = {
    "idle": ["...1CCCCCCCC1...",
             "...1CB1..1BC1...",
             "...1CB1..1BC1...",
             "...1001..1001...",
             "....11....11...."],
    "a":    ["...1CCCCCCCC1...",
             "...1CB1..1BC1...",
             "...1CB1..1001...",
             "...1001...11....",
             "....11.........."],
    "b":    ["...1CCCCCCCC1...",
             "...1CB1..1BC1...",
             "...1001..1BC1...",
             "....11...1001...",
             "..........11...."],
}
LEGS_SIDE = {
    "idle": ["....1CCCCCC1....",
             ".....1CBBC1.....",
             ".....1CBBC1.....",
             "....1000001.....",
             ".....11111......"],
    "a":    ["....1CCCCCC1....",
             "....1CB11BC1....",
             "...1CB1..1BC1...",
             "..1001....1001..",
             "...11......11..."],
    "b":    ["....1CCCCCC1....",
             ".....1CBBC1.....",
             ".....1CB1C1.....",
             ".....10001......",
             "......111......."],
}


def _frame(head, body, legs, bob=0):
    rows = head + body + legs
    assert all(len(r) == FW for r in rows), [len(r) for r in rows]
    c = Canvas(FW, FH)
    f = from_ascii(rows)
    c.blit(f, 0, FH - f.h + bob)
    return c


def sheet():
    s = Canvas(FW * 3, FH * 4)
    specs = [
        ("down", HEAD_DOWN, BODY_DOWN, LEGS_FRONT),
        ("left", HEAD_LEFT, BODY_LEFT, LEGS_SIDE),
        ("right", None, None, None),
        ("up", HEAD_UP, BODY_UP, LEGS_FRONT),
    ]
    for row, (name, head, body, legs) in enumerate(specs):
        for col, step in enumerate(("idle", "a", "b")):
            if name == "right":
                f = _frame(HEAD_LEFT, BODY_LEFT, LEGS_SIDE[step], bob=0).flip_h()
            else:
                f = _frame(head, body, legs[step], bob=0)
            s.blit(f, col * FW, row * FH)
    return s


DIRS = {"down": 0, "left": 1, "right": 2, "up": 3}


def frame(s, direction, i):
    return s.crop(i * FW, DIRS[direction] * FH, FW, FH)
