"""LEGO colour palette (subset of LDraw colours, names as in BrickNet's color_names.json) and CIELAB quantisation."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

import bricknet

# approximate sRGB of common solid colours (LDConfig values)
_PALETTE_HEX = {
    "black": "1B2A34", "white": "F4F4F4", "light bluish gray": "A0A5A9", "dark bluish gray": "6C6E68",
    "red": "B40000", "dark red": "720012", "blue": "1E5AA8", "dark blue": "19325A", "medium azure": "36AEBF",
    "green": "00852B", "dark green": "00451A", "bright green": "58AB41", "lime": "A5CA18", "yellow": "FAC80A",
    "bright light orange": "F8BB3D", "orange": "D67923", "tan": "E4CD9E", "medium nougat": "AA7D55",
    "reddish brown": "5F3109", "pearl gold": "AA7F2E", "sand green": "708E7C", "magenta": "923978",
}


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


@lru_cache(maxsize=1)
def palette() -> tuple[tuple[str, ...], np.ndarray]:
    """(names, Lab array) restricted to colours BrickNet knows."""
    known = bricknet.load_catalog().color_to_code
    names = tuple(n for n in _PALETTE_HEX if n in known)
    rgb = np.array([[int(_PALETTE_HEX[n][i : i + 2], 16) for i in (0, 2, 4)] for n in names])
    return names, _srgb_to_lab(rgb)


def quantize(rgb: np.ndarray) -> np.ndarray:
    """Index into ``palette()[0]`` of the nearest LEGO colour (CIELAB distance) for each RGB row."""
    names, lab = palette()
    q = _srgb_to_lab(np.atleast_2d(rgb))
    return np.argmin(((q[:, None, :] - lab[None]) ** 2).sum(-1), axis=1)


def dominant(rgb: np.ndarray, k: int = 2, min_share: float = 0.08) -> tuple[str, ...]:
    """The k most common LEGO colours among RGB samples (each later colour needs ``min_share`` of the samples)."""
    names, _ = palette()
    if not len(rgb):
        return ()
    counts = np.bincount(quantize(rgb), minlength=len(names))
    order = np.argsort(counts)[::-1]
    out = [names[order[0]]]
    for i in order[1:k]:
        if counts[i] >= min_share * counts.sum():
            out.append(names[i])
    return tuple(out)


def code(name: str) -> int:
    return bricknet.load_catalog().color_to_code[name]


# ---- colour classes of the descriptor -----------------------------------------------------------------------------
# One letter per coarse class, shared by both halves: LEGO parts are classed by colour name, photo / mesh samples by
# RGB (nearest prototype in CIELAB). ``t`` (glass) and ``l`` (lights) come from transparent parts / label masks.
CLASSES = ("k", "g", "s", "w", "r", "m", "o", "y", "n", "b", "a", "e", "p", "t", "l")
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
GLASS, LIGHT = CLASS_INDEX["t"], CLASS_INDEX["l"]
CLASS_NAMES = {  # class -> a LEGO colour (rendering / recolouring)
    "k": "black", "g": "dark bluish grey", "s": "light bluish grey", "w": "white", "r": "red", "m": "dark red",
    "o": "orange", "y": "yellow", "n": "tan", "b": "blue", "a": "medium azure", "e": "green", "p": "magenta",
    "t": "trans clear", "l": "trans red",
}
_CLASS_PROTOTYPES = {
    "k": ("1B2A34", "05131D", "2A2A2A"), "g": ("6C6E68", "595D60"), "s": ("A0A5A9", "8A928D", "C0C0C0"),
    "w": ("F4F4F4", "E6E3DA"), "r": ("B40000", "C91A09", "D01012"), "m": ("720012", "6B0F1A"),
    "o": ("D67923", "FE8A18", "F8BB3D"), "y": ("FAC80A", "F2CD37", "FFF03A"),
    "n": ("E4CD9E", "AA7D55", "5F3109", "AA7F2E", "958A73"), "b": ("1E5AA8", "19325A", "0055BF"),
    "a": ("36AEBF", "5A93DB", "9FC3E9", "078BC9"), "e": ("00852B", "00451A", "58AB41", "A5CA18", "708E7C"),
    "p": ("923978", "C870A0", "81007B", "E4ADC8"),
}
_NAME_RULES = (  # first matching substring wins
    ("trans", "t"), ("black", "k"), ("white", "w"), ("very light", "w"), ("milky", "w"), ("glow in dark", "w"),
    ("luminous", "w"), ("dark bluish grey", "g"), ("dark grey", "g"), ("dbgray", "g"), ("magnet", "g"),
    ("grey", "s"), ("silver", "s"), ("alloy", "s"), ("dark red", "m"), ("sand red", "m"), ("rust", "m"),
    ("brown", "n"), ("gold", "n"), ("copper", "n"), ("brass", "n"), ("amber", "n"), ("tan", "n"), ("nougat", "n"),
    ("blue violet", "b"), ("pink", "p"), ("magenta", "p"), ("purple", "p"), ("lavender", "p"), ("lilac", "p"),
    ("violet", "p"), ("orange", "o"), ("red", "r"), ("coral", "r"), ("salmon", "r"), ("yellowish green", "e"),
    ("lime", "e"), ("green", "e"), ("olive", "e"), ("yellow", "y"), ("lemmon", "y"), ("azure", "a"), ("aqua", "a"),
    ("turquoise", "a"), ("sky", "a"), ("light blue", "a"), ("maersk", "a"), ("sand blue", "a"), ("blue", "b"),
)


def class_of_name(name: str) -> int:
    """Colour class of a LEGO colour name (transparent colours -> glass)."""
    name = name.lower()
    return CLASS_INDEX[next((c for key, c in _NAME_RULES if key in name), "s")]


@lru_cache(maxsize=1)
def _prototypes() -> tuple[np.ndarray, np.ndarray]:
    cls, rgb = [], []
    for c, hexes in _CLASS_PROTOTYPES.items():
        for h in hexes:
            cls.append(CLASS_INDEX[c])
            rgb.append([int(h[i : i + 2], 16) for i in (0, 2, 4)])
    return np.array(cls), _srgb_to_lab(np.array(rgb))


def class_of_rgb(rgb: np.ndarray) -> np.ndarray:
    """Colour class of RGB (0-255) samples; NaN rows -> -1."""
    rgb = np.atleast_2d(np.asarray(rgb, dtype=np.float64))
    out = np.full(len(rgb), -1, dtype=np.int64)
    ok = ~np.isnan(rgb).any(1)
    if ok.any():
        cls, lab = _prototypes()
        q = _srgb_to_lab(rgb[ok])
        out[ok] = cls[np.argmin(((q[:, None, :] - lab[None]) ** 2).sum(-1), axis=1)]
    return out
