"""'Meadow' palette in the spirit of Undertale: flat colours, few tones per
material, one outline shade each. No gradients, no noise."""
import numpy as np

GRASS = ["#1d4a28", "#2c7536", "#46a043", "#67bd51", "#9edb6c"]    # outline, dark, base, light, tip
LUSH = ["#17391f", "#225e2d", "#348a39", "#4fa844", "#86cc5e"]     # denser, darker grass
PATH = ["#5e3b20", "#93643a", "#c28f57", "#dcb277", "#f0d6a2"]      # outline, dark, base, light, pebble
FLOWER = ["#b8720f", "#f2a81d", "#ffd83d", "#fff3a8", "#ffffff",    # gold ramp + white
          "#e0487a", "#ff90b8", "#4f8fe0", "#8cc6ff"]              # pink, blue
STONE = ["#3c3b48", "#6a6a7a", "#9d9dac", "#cdcdd8"]

HEX = GRASS + LUSH + PATH + FLOWER + STONE


def _r(start, n):
    return list(range(start, start + n))


G = _r(0, 5)
L = _r(5, 5)
P = _r(10, 5)
F = _r(15, 9)
S = _r(24, 4)
SH1 = len(HEX)   # soft shadow (semi-transparent)


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


RGBA = np.array([_rgb(h) + (255,) for h in HEX] + [(20, 40, 20, 70)], dtype=np.uint8)

# ASCII: g0-g4 -> 'abcde' (grass), lush 'fghij', path 'klmno', flowers 'pqrstuvwx', stone 'yz12'
CHARMAP = {}
for i, ch in enumerate("abcde"):
    CHARMAP[ch] = G[i]
for i, ch in enumerate("fghij"):
    CHARMAP[ch] = L[i]
for i, ch in enumerate("klmno"):
    CHARMAP[ch] = P[i]
for i, ch in enumerate("pqrstuvwx"):
    CHARMAP[ch] = F[i]
for i, ch in enumerate("yz12"):
    CHARMAP[ch] = S[i]
CHARMAP["+"] = SH1
