"""Part vocabulary with geometry measured from the part meshes (BrickNet GLBs, LDraw frame, LDU, -Y up).

For every part and each of the four 90-degree turns about the vertical axis we store
  * the footprint in studs along the car's x (LDraw X) and y (LDraw Z),
  * the translation that puts the turned part's bounding box on a grid corner with its bottom on a given plate
    level (so differing part origins never matter),
  * a *top height map*: the highest point of the part over ``SUB`` x ``SUB`` samples per stud, in plates above
    the part's bottom. Fitting a cap part to the car's surface compares this map with the target heights.
  * which turn is its mirror image across the car's centre plane (for symmetric building).

Car building frame: cell (i, j, k) = stud i along x (forward), stud j along y (left), plate k up from the ground.
LDraw position of a cell corner: X = 20 i, Z = 20 j, Y = -8 k.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

STUD, PLATE = 20.0, 8.0
SUB = 4  # height-map samples per stud

# kind: how the designer may use a part. "fill" parts build the body, "cap" parts finish its top surface.
FILL_BRICKS = {"3005": (1, 1), "3004": (1, 2), "3622": (1, 3), "3010": (1, 4), "3009": (1, 6), "3008": (1, 8),
               "3003": (2, 2), "3002": (2, 3), "3001": (2, 4), "2456": (2, 6), "3007": (2, 8)}
FILL_PLATES = {"3024": (1, 1), "3023": (1, 2), "3623": (1, 3), "3710": (1, 4), "3666": (1, 6), "3460": (1, 8),
               "4477": (1, 10), "60479": (1, 12), "3022": (2, 2), "3021": (2, 3), "3020": (2, 4), "3795": (2, 6),
               "3034": (2, 8), "3832": (2, 10), "2445": (2, 12), "4282": (2, 16), "3031": (4, 4), "3032": (4, 6),
               "3035": (4, 8), "3030": (4, 10), "3029": (4, 12), "3958": (6, 6), "3036": (6, 8), "3033": (6, 10),
               "3028": (6, 12), "41539": (8, 8)}
CAPS = {
    # studless flat finish
    "3070b": "tile", "3069b": "tile", "63864": "tile", "2431": "tile", "6636": "tile", "4162": "tile",
    "3068b": "tile", "26603": "tile", "87079": "tile",
    # curved slopes (studless)
    "11477": "curve", "15068": "curve", "50950": "curve", "61678": "curve", "93606": "curve", "42022": "curve",
    "24309": "curve", "93273": "curve", "88930": "curve_studs",
    # cheese / small slopes (studless)
    "54200": "cheese", "85984": "cheese",
    # classic studded slopes
    "3040": "slope", "3039": "slope", "3038": "slope", "3037": "slope", "4286": "slope", "3298": "slope",
    "4161": "slope", "60477": "slope", "30363": "slope",
    # car windscreens (used only where the car's surface is glass)
    "3571": "windscreen", "45705": "windscreen", "41751": "windscreen", "30497": "windscreen", "2826": "windscreen",
    "62576": "windscreen", "61191": "windscreen", "2694": "windscreen", "24607": "windscreen", "1874": "windscreen",
    "80222": "windscreen", "65632": "windscreen", "87606": "windscreen", "3383": "windscreen", "3384": "windscreen",
    "18973": "windscreen", "47506": "windscreen", "92832": "windscreen",
    # studded flats (fallback finish)
    "3024": "plate", "3023": "plate", "3623": "plate", "3710": "plate", "3022": "plate", "3021": "plate",
    "3020": "plate", "3005": "brick", "3004": "brick", "3003": "brick",
}
STUDLESS = {"tile", "curve", "cheese", "windscreen"}
AXLES = {2: "32062", 3: "4519", 4: "3705", 5: "32073", 6: "3706", 7: "44294", 8: "3707"}
WHEEL_HOLDER = "3701"  # technic brick 1x4 with holes: holes 14 LDU above its bottom, at cell boundaries
# complete wheels (rim + tyre in one part) with an axle hole, that look like car wheels
WHEELS = ["18976c01", "56902c01", "3482c04", "55981c01", "55981c03", "55981c02", "56904c01", "2695c01",
          "56145c02", "56145c01", "42716c01", "41896c01"]


def rot_y(k: int) -> np.ndarray:
    a = np.radians(90 * k)
    c, s = round(np.cos(a)), round(np.sin(a))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


@dataclass
class Turn:
    k: int
    R: np.ndarray
    nx: int  # studs along car x
    ny: int  # studs along car y
    lo: np.ndarray  # turned bbox min (X, Y, Z), LDU
    hi: np.ndarray
    hmap: np.ndarray  # (nx * SUB, ny * SUB) plates above bottom
    mirror: int | None = None

    def matrix(self, i: int, j: int, k: int) -> np.ndarray:
        """LDraw 4x4 placing the footprint's low corner at cell (i, j) with the part's bottom on plate level k."""
        m = np.eye(4)
        m[:3, :3] = self.R
        m[:3, 3] = [STUD * i - self.lo[0], -PLATE * k - self.hi[1], STUD * j - self.lo[2]]
        return m


@dataclass
class Part:
    stem: str
    name: str
    kind: str
    turns: list[Turn] = field(default_factory=list)
    height: float = 0.0  # plates, body (without studs)

    @property
    def studless(self) -> bool:
        return self.kind in STUDLESS


@lru_cache(maxsize=1)
def mesh_cache():
    from shape2brick.ldraw.shape import PartMeshCache

    return PartMeshCache()


@lru_cache(maxsize=1)
def catalog():
    import bricknet

    return bricknet.load_catalog()


def part_name(stem: str) -> str:
    cat = catalog()
    i = cat.stem_to_id.get(stem)
    return cat.id_to_name[i] if i is not None else stem


def _surface_points(v: np.ndarray, f: np.ndarray, n: int = 40000) -> np.ndarray:
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(0)
    i = rng.choice(len(tri), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n))[:, None], rng.random(n)[:, None]
    t = tri[i]
    return np.concatenate([(1 - r1) * t[:, 0] + r1 * (1 - r2) * t[:, 1] + r1 * r2 * t[:, 2], v])


@lru_cache(maxsize=None)
def load_part(stem: str, kind: str = "") -> Part:
    mesh = mesh_cache().get(stem)
    if mesh is None:
        raise KeyError(f"no mesh for part {stem}")
    v, f = mesh
    pts = _surface_points(np.asarray(v, float), np.asarray(f))
    p = Part(stem, part_name(stem), kind)
    for k in range(4):
        q = pts @ rot_y(k).T
        lo, hi = q.min(0), q.max(0)
        nx, ny = max(1, round((hi[0] - lo[0]) / STUD)), max(1, round((hi[2] - lo[2]) / STUD))
        hm = np.zeros((nx * SUB, ny * SUB))
        ix = np.clip(((q[:, 0] - lo[0]) / (STUD / SUB)).astype(int), 0, nx * SUB - 1)
        iz = np.clip(((q[:, 2] - lo[2]) / (STUD / SUB)).astype(int), 0, ny * SUB - 1)
        np.maximum.at(hm, (ix, iz), (hi[1] - q[:, 1]) / PLATE)
        p.turns.append(Turn(k, rot_y(k), nx, ny, lo, hi, hm))
    for t in p.turns:  # mirror across the car's centre plane = flip along car y
        for u in p.turns:
            if (u.nx, u.ny) == (t.nx, t.ny) and np.abs(u.hmap - t.hmap[:, ::-1]).max() < 0.35:
                t.mirror = u.k
                break
    body = pts[:, 1] >= pts[:, 1].min() + (4.0 if not p.studless else 0.0)  # drop stud tops
    p.height = (pts[:, 1].max() - pts[body, 1].min()) / PLATE
    return p


def caps() -> list[Part]:
    out = []
    for stem, kind in CAPS.items():
        try:
            out.append(load_part(stem, kind))
        except KeyError:
            pass
    return out


def fills() -> tuple[dict[tuple[int, int], str], dict[tuple[int, int], str]]:
    """(footprint (a, b) with a <= b) -> stem, for bricks and plates."""
    return dict((v, k) for k, v in FILL_BRICKS.items()), dict((v, k) for k, v in FILL_PLATES.items())


def wheel_options() -> list[dict]:
    out = []
    for stem in WHEELS:
        m = mesh_cache().get(stem)
        if m is None:
            continue
        v = np.asarray(m[0])
        lo, hi = v.min(0), v.max(0)
        out.append({"stem": stem, "name": part_name(stem), "diameter": float(max(hi[0] - lo[0], hi[1] - lo[1])),
                    "width": float(hi[2] - lo[2]), "centre": (lo + hi) / 2, "zlo": float(lo[2]), "zhi": float(hi[2])})
    return out
