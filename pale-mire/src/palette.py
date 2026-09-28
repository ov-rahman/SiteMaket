"""'Pale Mire' palette - a foggy, washed-out swamp.

Ramps are hue-shifted: shadows lean to cold blue-green, lights to a
yellowish, milky sage, so the fog reads as damp daylight rather than grey.
"""
import numpy as np

# Land: moss, grass, foliage, fog (G0 = darkest)
LAND = [
    "#131b16", "#1a241d", "#212d25", "#29372d", "#324136", "#3b4d3e",
    "#475a47", "#546852", "#62765d", "#718469", "#829377", "#94a287",
    "#a7b299", "#bbc2ad", "#cfd4c2", "#e3e7d8",
]
# Water: murky jade (W0 = deepest)
WATER = [
    "#101a19", "#152221", "#1b2b29", "#223532", "#2a403c", "#344b46",
    "#3f5751", "#4c645c", "#5a7168", "#6b8176", "#809386", "#9aab9e",
]
# Bark, mud, wood (B0 = darkest)
BARK = [
    "#161713", "#1e1f1a", "#282922", "#32332a", "#3e3f34", "#4b4c3f",
    "#5a5b4b", "#6a6b59", "#7f806c", "#979882",
]
# Flora accents: cattail, lily flower, fungus
FLORA = [
    "#4a4234", "#655a45", "#b89f98", "#ddd0c8", "#a4966f", "#c7bb95",
]
# Character (same wanderer as the village): violet sweater + warm accents
VIOLET = [
    "#0c0a1f", "#13102c", "#1b1739", "#241f48", "#2d2757", "#373065",
    "#423a74", "#4f4783", "#5e5693", "#6f68a4",
]
ACCENT = [
    "#1a0a14", "#33141f", "#52232b", "#733834", "#5e4760", "#83677e",
    "#a3879a", "#4a1f37", "#6f2c45", "#94405a",
]

HEX = LAND + WATER + BARK + FLORA + VIOLET + ACCENT


def _idx(start, n):
    return list(range(start, start + n))


G = _idx(0, len(LAND))
W = _idx(len(LAND), len(WATER))
B = _idx(G[-1] + len(WATER) + 1, len(BARK))
F = _idx(B[-1] + 1, len(FLORA))
V = _idx(F[-1] + 1, len(VIOLET))
A = _idx(V[-1] + 1, len(ACCENT))
assert A[-1] == len(HEX) - 1

# semi-transparent helpers
SH1 = len(HEX)        # soft shadow
SH2 = len(HEX) + 1    # contact shadow
SH3 = len(HEX) + 2    # faint shadow
MIST1 = len(HEX) + 3  # thin mist
MIST2 = len(HEX) + 4  # thick mist


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


RGBA = np.array([_rgb(h) + (255,) for h in HEX]
                + [(12, 26, 20, 92), (10, 22, 17, 150), (12, 26, 20, 52),
                   (227, 231, 216, 70), (227, 231, 216, 140)], dtype=np.uint8)

# ASCII maps for hand-pixelled sprites
CHARMAP = {}
for i, ch in enumerate("0123456789abcdef"):
    CHARMAP[ch] = G[i]
for i, ch in enumerate("ABCDEFGHIJKL"):
    CHARMAP[ch] = W[i]
for i, ch in enumerate("mnopqrstuv"):
    CHARMAP[ch] = B[i]
for i, ch in enumerate("gh"):
    CHARMAP[ch] = F[i]          # cattail
CHARMAP["i"] = F[2]             # lily shadow
CHARMAP["j"] = F[3]             # lily flower
CHARMAP["k"] = F[4]             # fungus
CHARMAP["l"] = F[5]             # fungus light
CHARMAP["#"] = SH2
CHARMAP["+"] = SH1
CHARMAP["-"] = SH3

# the wanderer keeps the village character's letters
CHAR_CHARMAP = {"0": G[0], "1": G[1]}
for i, ch in enumerate("ABCDEFGHIJ"):
    CHAR_CHARMAP[ch] = V[i]
for i, ch in enumerate("klmnopqrst"):
    CHAR_CHARMAP[ch] = A[i]
