"""Layout of the 'Abandoned Village' location (40 x 30 tiles of 16 px).

Composition follows the reference: chapel at the top centre in front of a
dark pine forest, four ruined cottages in the corners, a well on the left,
the hollow statue on the right, a signpost by the east exit and a worn path
leading south out of the village into the darkness.
"""
import numpy as np

from pixelkit import value_noise, hash01
from terrain import VOID, GRASS, MOSS, PATH

MW, MH = 40, 30          # map size in tiles
TS = 16


def corner_grid():
    """Terrain level at each tile corner (MH+1 x MW+1)."""
    cy, cx = np.mgrid[0:MH + 1, 0:MW + 1].astype(float)
    n1 = value_noise(MW + 1, MH + 1, 3, seed=5)
    n2 = value_noise(MW + 1, MH + 1, 2, seed=9)
    g = np.full((MH + 1, MW + 1), GRASS, dtype=int)

    # ------------------------------------------------ void around the meadow
    left = np.interp(cy, [0, 10, 14, 16, 17, 20, 22, 26, 28, 30], [1.6, 1.6, 1.4, 2.0, 2.6, 2.4, 1.2, 1.4, 1.6, 2])
    right = np.interp(cy, [0, 12, 15, 17, 18, 20, 21, 26, 28, 30], [38.6, 38.6, 38.4, 38.2, 36.6, 36.8, 38.8, 39.4, 38.6, 38])
    bottom_l = np.interp(cx, [0, 4, 8, 12, 14, 16, 17], [27.4, 28.2, 28.6, 28.4, 28.2, 29.0, 31])
    bottom_r = np.interp(cx, [23, 24, 26, 28, 32, 36, 40], [31, 29.2, 28.8, 29.0, 28.8, 28.2, 27.6])
    edge_noise = (n1 - 0.5) * 0.9
    void = (cx < left + edge_noise) | (cx > right - edge_noise)
    void |= (cx <= 17) & (cy > bottom_l + edge_noise)
    void |= (cx >= 23) & (cy > bottom_r + edge_noise)
    g[void] = VOID

    # ------------------------------------------------ light moss patches (cloudy)
    def blob(x0, y0, rx, ry, w=1.0):
        return w * np.exp(-(((cx - x0) / rx) ** 2 + ((cy - y0) / ry) ** 2))

    moss = (blob(8, 16, 6.5, 2.4) + blob(5, 13, 3.2, 1.6) + blob(12, 13.2, 4, 1.4)
            + blob(29, 16, 6, 1.5) + blob(34, 14, 4.5, 1.4) + blob(27, 22.5, 5, 1.4)
            + blob(8, 26.8, 4, 1.0, 0.9) + blob(33, 18.5, 2.6, 1.2, 0.8) + blob(15, 19.5, 2.2, 1.6, 0.9)
            + blob(24, 26, 2.4, 2, 0.9))
    moss = moss + (n2 - 0.5) * 0.7
    g[(moss > 0.55) & (g == GRASS)] = MOSS

    # ------------------------------------------------ the path
    # centre line runs from the chapel steps (x=20.5) straight south, wobbling
    centre = 20.5 + (value_noise(1, MH + 1, 3, seed=21)[:, 0] - 0.5) * 1.2
    half = np.interp(cy[:, 0], [12, 13, 14, 15, 16, 18, 24, 27, 30], [4.2, 5.2, 4.6, 2.6, 1.8, 1.6, 1.8, 2.4, 3.0])
    d = np.abs(cx - centre[:, None]) - half[:, None] - (n1 - 0.5) * 1.1
    path = (d < 0) & (cy >= 12.6)
    g[path & (g != VOID)] = PATH
    # moss fringe hugging the path
    fringe = (d < 1.1) & (d >= 0) & (cy >= 13) & (g == GRASS) & (n2 > 0.42)
    g[fringe] = MOSS

    # ------------------------------------------------ keep grass between void and moss/path
    v = g == VOID
    near_void = np.zeros_like(v)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            near_void |= np.roll(np.roll(v, dy, 0), dx, 1)
    g[near_void & (g > GRASS)] = GRASS
    return g


# ---------------------------------------------------------------- objects
# (name, tile_x, tile_y) - drawn in list order (back to front)
PLACEMENTS = [
    # trees standing behind the buildings
    ("tree_tall", 1, 3),
    ("tree_crooked", 27, 4),
    ("tree_small", 10, 6),
    ("tree_thin", 36, 2),
    # buildings
    ("church", 15, 0),
    ("house_a", 3, 6),
    ("house_b", 31, 6),
    ("tree_thin", 1, 15),
    ("tree_small", 11, 21),
    ("house_c", 3, 20),
    ("house_d", 28, 20),
    ("tree_medium", 37, 21),
    # props
    ("well", 11, 15),
    ("statue", 24, 15),
    ("signpost", 35, 15),
    ("stump", 12, 11),
    ("barrel_open", 32, 11),
    ("barrel_closed", 2, 25),
    ("barrel_open", 27, 26),
    ("barrel_broken", 33, 12),
    ("crate", 37, 25),
    ("grave_0", 13, 11),
    ("grave_2", 14, 12),
]

# fences: (tile_x, tile_y, [kinds...]) - consecutive tiles to the right
FENCES = [
    (7, 11, ["fence_left", "fence_mid_a", "fence_mid_b", "fence_broken", "fence_right"]),
    (35, 12, ["fence_left", "fence_mid_c", "fence_mid_a", "fence_right"]),
    (10, 25, ["fence_left", "fence_mid_b", "fence_broken", "fence_right"]),
    (24, 26, ["fence_left", "fence_mid_a", "fence_right"]),
    (30, 11, ["fence_post"]),
    (2, 23, ["fence_post"]),
]

PLAYER_START = (320, 420)    # px (feet position), on the path facing north


def path_like(g, tx, ty):
    """True if tile (tx,ty) is mostly path."""
    c = [g[ty, tx], g[ty, tx + 1], g[ty + 1, tx], g[ty + 1, tx + 1]]
    return sum(1 for v in c if v == PATH) >= 2
