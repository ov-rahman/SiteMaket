"""'Moonlit Indigo' palette for the abandoned village.

Ramps are hue-shifted: shadows lean towards deep blue/violet, lights towards
a cold desaturated periwinkle, as if everything is lit only by the moon.
"""
import numpy as np

# Night ramp: ground, stone, foliage, void  (K0 = darkest)
NIGHT = [
    "#030409",  # K0  void
    "#06081a",  # K1
    "#0a0e23",  # K2
    "#0f142f",  # K3
    "#151b3b",  # K4
    "#1b2246",  # K5
    "#212951",  # K6
    "#27305b",  # K7  grass
    "#2d3764",  # K8
    "#343f6e",  # K9
    "#3b4778",  # K10 moss / light patches
    "#434f82",  # K11 path
    "#4c598c",  # K12
    "#576497",  # K13
    "#6370a3",  # K14
    "#7280b1",  # K15
    "#8793c3",  # K16
    "#a2acd6",  # K17 rare glints
]

# Violet ramp: wood, roofs, cloth
VIOLET = [
    "#0c0a1f",  # V0
    "#13102c",  # V1
    "#1b1739",  # V2
    "#241f48",  # V3
    "#2d2757",  # V4
    "#373065",  # V5
    "#423a74",  # V6
    "#4f4783",  # V7
    "#5e5693",  # V8
    "#6f68a4",  # V9
]

# Character accents (the only warm-ish colours in the scene)
ACCENT = [
    "#1a0a14",  # A0 hair darkest
    "#33141f",  # A1 hair dark
    "#52232b",  # A2 hair
    "#733834",  # A3 hair light
    "#5e4760",  # A4 skin shadow
    "#83677e",  # A5 skin
    "#a3879a",  # A6 skin light
    "#4a1f37",  # A7 scarf dark
    "#6f2c45",  # A8 scarf
    "#94405a",  # A9 scarf light
]

HEX = NIGHT + VIOLET + ACCENT
K = list(range(0, len(NIGHT)))
V = list(range(len(NIGHT), len(NIGHT) + len(VIOLET)))
A = list(range(len(NIGHT) + len(VIOLET), len(HEX)))

# semi-transparent helpers (not part of the art palette proper)
SH1 = len(HEX)       # soft shadow
SH2 = len(HEX) + 1   # contact shadow
SH3 = len(HEX) + 2   # very light shadow
FOG = len(HEX) + 3   # moonlit haze


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


RGBA = np.array([_rgb(h) + (255,) for h in HEX]
                + [(3, 4, 14, 96), (3, 4, 12, 150), (3, 4, 14, 56), (120, 130, 190, 26)],
                dtype=np.uint8)

# ASCII map for hand-pixelled sprites
CHARMAP = {}
for i, ch in enumerate("0123456789abcdefgh"):
    CHARMAP[ch] = K[i]
for i, ch in enumerate("ABCDEFGHIJ"):
    CHARMAP[ch] = V[i]
for i, ch in enumerate("klmnopqrst"):
    CHARMAP[ch] = A[i]
CHARMAP["x"] = SH2
CHARMAP["y"] = SH1
CHARMAP["z"] = SH3
