"""Layout of 'The Pale Mire' (64 x 40 tiles of 16 px = 1024 x 640).

The path from the abandoned village comes down onto the northern bank.
A boardwalk on stilts crosses the dark central pool to a landing, then
splits: east to the reed island, south to the far bank where the road
continues. Four giant cypresses stand at the corners, their crowns spilling
over the edges of the map; a fifth one, rooted beyond the southern edge,
hangs its crown into view.
"""
import numpy as np

from pixelkit import value_noise
from terrain import DEEP, SHALLOW, MUD, GRASS

MW, MH = 64, 40
TS = 16


def _blob(cx, cy, x0, y0, rx, ry, w=1.0):
    return w * np.exp(-(((cx - x0) / rx) ** 2 + ((cy - y0) / ry) ** 2))


LAND = [  # x, y, rx, ry, weight
    (31, 1.5, 15, 7.5, 1.3),   # northern bank (entry)
    (5, 4, 10, 10, 1.2),       # top-left, giant #1
    (58, 3, 10, 8, 1.2),       # top-right, giant #2
    (50, 20, 7, 5, 1.05),      # reed island (east)
    (21, 23, 3.4, 2.4, 0.95),  # little islet
    (33, 40.5, 14, 6.5, 1.25), # southern bank (exit)
    (4, 36, 9, 7.5, 1.2),      # south-west, giant #3
    (62, 30, 6, 6.5, 1.15),    # east edge, giant #4
    (39, 29, 2.2, 1.8, 0.85),  # tussock
    (14, 14, 4.5, 2.6, 0.75),  # mud flat
]
DEPTH = [
    (27.5, 20, 9, 4.8, 1.0),   # the central pool
    (11, 24.5, 5.5, 3, 0.9),   # west pool
    (47, 32, 8, 3, 0.9),       # south-east channel
    (41, 12.5, 4.5, 2.6, 0.8),
    (56, 25, 3, 2, 0.7),
]


def corner_grid():
    cy, cx = np.mgrid[0:MH + 1, 0:MW + 1].astype(float)
    n1 = value_noise(MW + 1, MH + 1, 3, seed=41)
    n2 = value_noise(MW + 1, MH + 1, 2, seed=43)
    land = sum(_blob(cx, cy, *b) for b in LAND) + (n1 - 0.5) * 0.55
    deep = sum(_blob(cx, cy, *b) for b in DEPTH) + (n2 - 0.5) * 0.5
    g = np.full((MH + 1, MW + 1), SHALLOW, dtype=int)
    g[deep > 0.55] = DEEP
    g[land > 0.42] = MUD
    g[land > 0.6] = GRASS
    # deep water never touches mud or grass directly: keep a shallow shelf
    near_land = np.zeros_like(g, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            near_land |= np.roll(np.roll(g >= MUD, dy, 0), dx, 1)
    g[(g == DEEP) & near_land] = SHALLOW
    return g


# ------------------------------------------------------------------ objects
GIANTS = [  # (seed, base_x_px, base_y_px, flip)
    (7, 7.5 * TS, 12.2 * TS, False),
    (1, 56.5 * TS, 10.6 * TS, True),
    (11, 5.5 * TS, 33.6 * TS, False),
    (19, 60.5 * TS, 31.2 * TS, True),
    (23, 47 * TS, 46.5 * TS, False),      # rooted beyond the southern edge
]
CYPRESSES = [(5, 49.5 * TS, 21 * TS), (9, 24.5 * TS, 37.5 * TS), (13, 21.5 * TS, 7.2 * TS),
             (17, 41.5 * TS, 6.6 * TS)]
SNAGS = [(3, 14.5 * TS, 18.5 * TS), (8, 44.5 * TS, 28.2 * TS), (12, 36.2 * TS, 13.2 * TS), (14, 5 * TS, 24 * TS)]

# boardwalk: ('h'|'v'|'hb'|'vb', tx, ty) + platform top-left tile
WALK_V = [(31, y) for y in range(9, 18)] + [(31, y) for y in range(21, 34)]
WALK_H = [(x, 19) for x in range(33, 43)]
BROKEN = {(31, 26), (37, 19)}
PLATFORM = (30, 18)
BOAT = (23.5 * TS, 15.5 * TS)
LOGS = [(41 * TS, 23 * TS, 64, 2)]
PLAYER = (31.5 * TS, 20.2 * TS)
