"""A LEGO model as a list of placed parts: LDraw I/O, real-mesh rendering, bill of materials, connectivity."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .parts import mesh_cache, part_name

# LDraw colour codes for the palette names used here
LDRAW_CODE = {"black": 0, "blue": 1, "green": 2, "red": 4, "bright green": 10, "yellow": 14, "white": 15,
              "tan": 19, "orange": 25, "magenta": 26, "lime": 27, "reddish brown": 70, "light bluish gray": 71,
              "dark bluish gray": 72, "medium nougat": 84, "bright light orange": 191, "dark blue": 272,
              "dark green": 288, "pearl gold": 297, "dark red": 320, "medium azure": 322, "sand green": 378,
              "trans black": 40, "trans clear": 47, "trans red": 36, "trans yellow": 46, "flat silver": 179}
LD_TO_CAR = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # LDraw (X, Y, Z) -> render frame (x, y, z up)


@dataclass
class Placed:
    stem: str
    color: int
    m: np.ndarray  # 4x4 LDraw
    role: str = ""  # fill / cap / wheel / axle / holder / ...
    cells: tuple = ()  # grid cells it occupies (i, j, k), for bookkeeping


@dataclass
class Model:
    parts: list[Placed] = field(default_factory=list)

    def add(self, p: Placed):
        self.parts.append(p)

    def to_ldr(self, title: str = "brickcar") -> str:
        lines = [f"0 {title}", "0 Name: brickcar.ldr", "0 Author: brickcar"]
        for p in self.parts:
            r, t = p.m[:3, :3], p.m[:3, 3]
            nums = " ".join(f"{x:g}" for x in (*np.round(t, 3), *np.round(r, 6).ravel()))
            lines.append(f"1 {p.color} {nums} {p.stem}.dat")
        return "\n".join(lines) + "\n"

    def bom(self) -> list[tuple[str, str, int, int]]:
        c = Counter((p.stem, p.color) for p in self.parts)
        return sorted(((s, part_name(s), col, n) for (s, col), n in c.items()), key=lambda r: (-r[3], r[0]))

    def components(self) -> list[tuple[int, ...]]:
        """Connected groups by BrickNet's connection inference (studs, axles, pins...)."""
        import bricknet

        g = bricknet.parse_ldr(self.to_ldr())
        return sorted(g.components, key=len, reverse=True)

    def triangles(self, colors: dict | None = None):
        """-> (T, 3, 3) triangles in the render frame (LDU, z up) and (T, 3) rgb."""
        table = ldraw_rgb()
        tris, rgbs = [], []
        for p in self.parts:
            mesh = mesh_cache().get(p.stem)
            if mesh is None:
                continue
            v, f = np.asarray(mesh[0], float), np.asarray(mesh[1])
            w = (v @ p.m[:3, :3].T + p.m[:3, 3]) @ LD_TO_CAR.T
            tris.append(w[f])
            c = np.repeat(np.asarray(table.get(p.color, (0.6, 0.6, 0.6)))[None], len(f), 0)
            if p.role == "wheel":  # rim + tyre are one part: the outer ring is the black tyre
                ctr = (v.min(0) + v.max(0)) / 2
                rad = np.linalg.norm((v[f].mean(1) - ctr)[:, :2], axis=1)
                c[rad > 0.7 * rad.max()] = table.get(0, (0.1, 0.1, 0.1))
            rgbs.append(c)
        return np.concatenate(tris).astype(np.float32), np.concatenate(rgbs).astype(np.float32)


@lru_cache(maxsize=1)
def ldraw_rgb() -> dict[int, tuple[float, float, float]]:
    """LDraw colour code -> rgb in [0, 1] from LDConfig.ldr (trans colours slightly lightened to read as glass)."""
    path = Path(__file__).resolve().parents[3] / "shape2brick" / "data" / "raw" / "ldraw" / "LDConfig.ldr"
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        t = line.split()
        if len(t) > 6 and t[0] == "0" and t[1] == "!COLOUR" and "CODE" in t and "VALUE" in t:
            code = int(t[t.index("CODE") + 1])
            h = t[t.index("VALUE") + 1].lstrip("#")
            rgb = tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
            if "ALPHA" in t:  # glass: show it dark and a bit glossy rather than see-through
                rgb = tuple(0.55 * c + 0.08 for c in rgb)
            out[code] = rgb
    return out


def save(model: Model, out: Path, title: str):
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.ldr").write_text(model.to_ldr(title))
    rows = model.bom()
    names = {v: k for k, v in LDRAW_CODE.items()}
    lines = ["part,name,color,ldraw_color,count"]
    lines += [f"{s},\"{n}\",{names.get(c, c)},{c},{k}" for s, n, c, k in rows]
    (out / "bom.csv").write_text("\n".join(lines) + "\n")
