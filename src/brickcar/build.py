"""The designer: target -> LEGO model with real wheels, a stacked body and a fitted, studless top surface.

Order (each step leaves the grid in a state the next one relies on):

1. wheels   for every axle pick the complete wheel (rim + tyre) whose diameter / width match the car's tyres;
            mount it on a Technic axle through a Technic brick 1x4 with holes (the hole height fixes the holder's
            plate level, the holes sit on stud boundaries), mirrored left / right; cut stepped wheel arches into
            the columns outboard of the holder.
2. surface  "caps" finish the top of every column: tiles, curved slopes, cheese slopes, slopes (and studded
            plates as a fallback). Every (part, turn, position) is scored by how well its top height map
            matches the car's surface heights there, after choosing its best plate level; the cheapest
            candidates are taken greedily, each together with its mirror image. A cap's level is the top of the
            stack below it.
3. body     each column is filled from the car's underside (or the wheel arch) up to its cap with plates and
            bricks, layer by layer: bricks only on shared courses, the largest part that fits wins, with a bonus
            for bridging two parts of the layer below (running bond) and for alternating directions.
4. colour   visible parts take the car's colour where they are (nearest point of the hybrid mesh, LEGO palette);
            dark parts above the belt line become trans-black glass; hidden parts are light bluish grey.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from car2brick.colors import palette, quantize

from .model import LDRAW_CODE, Model, Placed
from .parts import AXLES, PLATE, STUD, SUB, WHEEL_HOLDER, caps, fills, load_part, rot_y, wheel_options
from .target import Target


@dataclass
class Config:
    cap_part_cost: float = 0.45  # plates of error a cap may "spend" to save one part (per covered cell)
    glass_bonus: float = 0.6  # windscreen parts: cost bonus on glass, penalty x 4 off glass (plates)
    stud_cost: float = 0.35  # extra cost per cell of a studded cap (we prefer a finished, studless look)
    min_fill: int = 1  # plates of body under every cap (so it has studs to sit on)
    arch_clearance: float = 6.0  # LDU around the tyre
    interior_color: str = "light bluish gray"
    flip_wheels: bool = False


@dataclass
class Design:
    model: Model
    base: np.ndarray  # (L, W) plate level of the caps
    bottom: np.ndarray  # (L, W) lowest body level
    occ: np.ndarray  # (L, W) columns used
    info: dict = field(default_factory=dict)


class Designer:
    def __init__(self, t: Target, cfg: Config = Config()):
        self.t, self.cfg = t, cfg
        self.L, self.W = t.L, t.W
        self.levels = t.H + 6
        self.occ = t.occ.copy()
        cb = t.col_bottom()
        mid = self._between_axles()
        self.floor = int(max(1, np.round(np.nanmedian(cb[mid][self.occ[mid]]))))
        self.B = np.where(self.occ, np.maximum(self.floor, np.round(np.nan_to_num(cb, nan=self.floor))), 0).astype(int)
        self.T = self._fill_nan_tops()
        self.grid = np.full((self.L, self.W, self.levels), -1, dtype=np.int32)  # part index per cell
        self.glass = self._glass_columns()
        self.model = Model()
        self.info: dict = {"floor": self.floor}

    # -- helpers ---------------------------------------------------------------------------------------------------
    def _between_axles(self) -> np.ndarray:
        m = np.zeros((self.L, self.W), bool)
        xs = sorted(w.x for w in self.t.wheels)
        if len(xs) >= 2:
            i = (self.t.cell_x(np.arange(self.L)) > xs[0] + 0.4) & (self.t.cell_x(np.arange(self.L)) < xs[-1] - 0.4)
            m[i] = True
        else:
            m[:] = True
        return m

    def _glass_columns(self) -> np.ndarray:
        """Columns whose top surface is dark and above the belt line (windows / windscreen)."""
        ct = self.t.col_top()
        ii, jj = np.meshgrid(np.arange(self.L), np.arange(self.W), indexing="ij")
        pts = np.stack([self.t.cell_x(ii), self.t.cell_y(jj), np.nan_to_num(ct) * self.t.plate], -1).reshape(-1, 3)
        dark = (self.t.color_at(pts).max(1) < 0.25).reshape(self.L, self.W)
        return (dark & self.occ & (np.nan_to_num(ct) > 0.5 * np.nanmax(ct))).astype(float)

    def _fill_nan_tops(self) -> np.ndarray:
        T = self.t.top.copy()
        ct = self.t.col_top()
        rep = np.repeat(np.repeat(np.nan_to_num(ct, nan=0.0), SUB, 0), SUB, 1)
        return np.where(np.isnan(T), rep, T)

    def _add(self, stem, color, m, role, cells=()):
        idx = len(self.model.parts)
        self.model.add(Placed(stem, color, m, role, tuple(cells)))
        for c in cells:
            self.grid[c] = idx
        return idx

    # -- 1. wheels ---------------------------------------------------------------------------------------------------
    def wheels(self):
        opts = wheel_options()
        k = STUD / self.t.stud  # LDU per metre
        half = self.W * STUD / 2
        self.info["wheels"] = []
        for wh in self.t.wheels:
            D, Wd = 2 * wh.radius * k, wh.width * k
            style = lambda o: 25.0 if any(w in o["name"] for w in ("balloon", "off road", "off-road")) else 0.0
            o = min(opts, key=lambda o: abs(o["diameter"] - D) + 0.4 * abs(o["width"] - Wd) + style(o))
            R, w = o["diameter"] / 2, o["width"]
            a = max(0, int(np.floor((R - 14) / PLATE)))  # holder level: hole (14 LDU up) at or below the axle
            zc = PLATE * a + 14
            ia = int(np.clip(np.round((wh.x - self.t.x0) / self.t.stud), 2, self.L - 2))  # cell boundary
            y_in = half - 2 - w  # tyre inner face, LDU from the centre line (outer face 2 LDU inside the body)
            r_h = max(0, int(np.floor((y_in - 2) / STUD)) - 1)
            yc = (r_h + 1) * STUD + 2 + w / 2
            n_ax = next((n for n in sorted(AXLES) if n * STUD >= yc + w / 2 - r_h * STUD - 6), 8)
            holder = load_part(WHEEL_HOLDER)
            for side in (+1, -1):
                j = self.W // 2 + r_h if side > 0 else self.W // 2 - 1 - r_h
                cells = [(i, j, z) for i in range(ia - 2, ia + 2) for z in range(a, a + 3)]
                self._add(WHEEL_HOLDER, LDRAW_CODE["black"], holder.turns[0].matrix(ia - 2, j, a), "holder", cells)
                # axle: local x -> car y; centred between the holder's inner face and the wheel's outer face
                y_ax = side * (r_h * STUD + n_ax * STUD / 2)
                m = np.eye(4)
                m[:3, :3] = rot_y(1)
                m[:3, 3] = [STUD * ia, -zc, half + y_ax]
                self.model.add(Placed(AXLES[n_ax], LDRAW_CODE["black"], m, "axle"))
                # wheel: its axis is local z; outer face outwards
                flip = (side < 0) != self.cfg.flip_wheels
                Rw = rot_y(2) if flip else np.eye(3)
                c = Rw @ o["centre"]
                m = np.eye(4)
                m[:3, :3] = Rw
                m[:3, 3] = np.array([STUD * ia, -zc, half + side * yc]) - c
                self.model.add(Placed(o["stem"], LDRAW_CODE["light bluish gray"], m, "wheel"))
                # wheel arch: columns outboard of the holder, near the axle
                rows = range(j + 1, self.W) if side > 0 else range(0, j)
                for i in range(self.L):
                    dx = max(0.0, abs(STUD * (i + 0.5) - STUD * ia) - STUD / 2)
                    rr = R + self.cfg.arch_clearance
                    if dx >= rr:
                        continue
                    lvl = int(np.ceil((zc + np.sqrt(rr * rr - dx * dx)) / PLATE))
                    for jj in rows:
                        self.B[i, jj] = max(self.B[i, jj], lvl)
                for i in range(ia - 2, ia + 2):  # the holder hangs from the column above it
                    self.occ[i, j] = True
                    self.B[i, j] = min(self.B[i, j], a + 3)
            self.info["wheels"].append({"x_m": wh.x, "target_diameter_ldu": round(D, 1), "part": o["stem"],
                                        "name": o["name"], "diameter_ldu": o["diameter"], "holder_level": a,
                                        "axle": AXLES[n_ax]})
        # columns whose arch cut reaches their top are gone
        top = self.t.col_top()
        gone = self.occ & (self.B >= np.nan_to_num(top, nan=0) - 0.5) & (self.grid.max(2) < 0)
        self.occ &= ~gone

    # -- 2. surface ----------------------------------------------------------------------------------------------
    def surface(self):
        cfg = self.cfg
        cands = []  # (cost, part, turn, i, j, base)
        occ = self.occ
        for p in caps():
            for t in p.turns:
                if t.nx > self.L or t.ny > self.W or (t.mirror is None):
                    continue
                win = sliding_window_view(self.T, (t.nx * SUB, t.ny * SUB))[::SUB, ::SUB]  # (I, J, nx*S, ny*S)
                ok = sliding_window_view(occ, (t.nx, t.ny)).all((-1, -2))
                bmax = sliding_window_view(np.where(occ, self.B, -99), (t.nx, t.ny)).max((-1, -2))
                diff = win - t.hmap
                base = np.round(np.median(diff.reshape(*diff.shape[:2], -1), axis=2))
                base = np.maximum(base, bmax + cfg.min_fill)
                err = np.abs(base[..., None, None] + t.hmap - win).mean((-1, -2))
                n = t.nx * t.ny
                cost = err + cfg.cap_part_cost / n + (0.0 if p.studless else cfg.stud_cost)
                if p.kind == "windscreen":
                    g = sliding_window_view(self.glass, (t.nx, t.ny)).mean((-1, -2))
                    cost = cost - cfg.glass_bonus * g + 4 * cfg.glass_bonus * (1 - g)
                for i, j in zip(*np.nonzero(ok)):
                    cands.append((float(cost[i, j]), p, t, int(i), int(j), int(base[i, j])))
        cands.sort(key=lambda c: (c[0], -c[2].nx * c[2].ny))
        covered = np.zeros_like(occ)
        self.base = np.zeros((self.L, self.W), int)
        n_caps = Counter()
        for cost, p, t, i, j, b in cands:
            jm = self.W - j - t.ny
            mt = p.turns[t.mirror]
            same = jm == j and mt.k == t.k
            if covered[i : i + t.nx, j : j + t.ny].any():
                continue
            if not same:
                if (jm < j + t.ny and jm + t.ny > j) or covered[i : i + t.nx, jm : jm + t.ny].any():
                    continue  # overlapping mirror (straddles the centre but is not symmetric itself)
            for jj, tt in ((j, t),) + (() if same else ((jm, mt),)):
                covered[i : i + t.nx, jj : jj + t.ny] = True
                self.base[i : i + t.nx, jj : jj + t.ny] = b
                h = int(np.ceil(p.height - 1e-6))
                cells = [(a, c, z) for a in range(i, i + t.nx) for c in range(jj, jj + t.ny)
                         for z in range(b, min(b + h, self.levels))]
                self._add(p.stem, -1, tt.matrix(i, jj, b), "cap:" + p.kind, cells)
                n_caps[p.kind] += 1
            if covered[occ].all():
                break
        self.info["caps"] = dict(n_caps)

    # -- 2b. details (SNOT) -----------------------------------------------------------------------------------------
    def _snot(self, i, j, z, face, tile, tile_color, body_color=-1, pair=False):
        """Brick 1x1 with a stud on its side at cell (i, j, z), side stud facing ``face`` (+x front, -x rear), and a
        tile on that stud (a 1x2 tile spans this brick and the one at j + 1 when ``pair``)."""
        k = 3 if face > 0 else 1  # turn that points the side stud (local -Z) along +x / -x
        R = rot_y(k)
        brick = np.eye(4)
        brick[:3, :3] = R
        brick[:3, 3] = [STUD * i + STUD / 2, -PLATE * z - 24, STUD * j + STUD / 2]
        bricks = [(i, j)] + ([(i, j + 1)] if pair else [])
        for (a, c) in bricks:
            m = brick.copy()
            m[:3, 3] = [STUD * a + STUD / 2, -PLATE * z - 24, STUD * c + STUD / 2]
            self._add("87087", body_color, m, "snot", [(a, c, zz) for zz in range(z, z + 3)])
        local = np.eye(4)  # tile bottom on the side stud at (0, 10, -10), its top facing -Z
        local[:3, :3] = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
        local[:3, 3] = [0, 10, -18]
        if pair:  # centre between the two bricks: +y in the car is the brick's local x for the front, -x for the rear
            local[0, 3] = 10 if face > 0 else -10
        self.model.add(Placed(tile, tile_color, brick @ local, "detail"))

    def _face_slot(self, j, face):
        """Front-most (face > 0) or rear-most column of row j with room for a brick 1x1 on a course below its cap,
        nothing in front of it at those levels; the highest such course. -> (i, z) or None."""
        cols = range(self.L - 1, -1, -1) if face > 0 else range(self.L)
        for i in cols:
            if not self.occ[i, j]:
                continue
            courses = [z for z in range(self.B[i, j], self.base[i, j] - 2) if (z - self.floor) % 3 == 0]
            for z in sorted(courses, reverse=True):
                if (self.grid[i, j, z : z + 3] >= 0).any():
                    continue
                n = i + face
                if 0 <= n < self.L and self.occ[n, j] and self.B[n, j] < z + 3 and self.base[n, j] > z:
                    continue
                return i, z
        return None

    def details(self):
        n = Counter()
        W = self.W
        light_rows = [1, 2] if W >= 10 else [1]
        for face, tile, col, key in ((+1, "98138", LDRAW_CODE["trans clear"], "headlight"),
                                     (-1, "98138", LDRAW_CODE["trans red"], "taillight")):
            for r in light_rows:
                for j in (r, W - 1 - r):
                    s = self._face_slot(j, face)
                    if s:
                        self._snot(s[0], j, s[1], face, tile, col)
                        n[key] += 1
        # grille: 1x2 grille tiles on pairs of SNOT bricks across the middle of the nose
        for j in range(light_rows[-1] + 2, W // 2, 2):
            for jj in (j, W - 2 - j):
                s1, s2 = self._face_slot(jj, +1), self._face_slot(jj + 1, +1)
                if s1 and s2 and s1 == s2:
                    self._snot(s1[0], jj, s1[1], +1, "2412b", LDRAW_CODE["black"], pair=True)
                    n["grille"] += 1
        self.info["details"] = dict(n)

    # -- 3. body ---------------------------------------------------------------------------------------------------
    def body(self):
        bricks, plates = fills()
        sizes = []
        for (a, b), stem in bricks.items():
            sizes.append((3, a, b, stem))
        for (a, b), stem in plates.items():
            sizes.append((1, a, b, stem))
        need = np.zeros((self.L, self.W, self.levels), bool)
        for i, j in zip(*np.nonzero(self.occ)):
            need[i, j, self.B[i, j] : self.base[i, j]] = True
        need &= self.grid < 0
        n_fill = 0
        for z in range(self.levels):
            if not need[:, :, z].any():
                continue
            course = (z - self.floor) % 3 == 0
            along_x = (z % 2) == 0
            order = [(i, j) for i in range(self.L) for j in range(self.W)]
            if not along_x:
                order = [(i, j) for j in range(self.W) for i in range(self.L)]
            below = self.grid[:, :, z - 1] if z > 0 else np.full((self.L, self.W), -1)
            for i, j in order:
                if not need[i, j, z] or self.grid[i, j, z] >= 0:
                    continue
                best = None
                for h, a, b, stem in sizes:
                    if h == 3 and not course:
                        continue
                    for nx, ny in ((a, b), (b, a)) if a != b else ((a, b),):
                        if i + nx > self.L or j + ny > self.W or z + h > self.levels:
                            continue
                        blk = need[i : i + nx, j : j + ny, z : z + h]
                        if not blk.all() or (self.grid[i : i + nx, j : j + ny, z : z + h] >= 0).any():
                            continue
                        under = below[i : i + nx, j : j + ny]
                        bridges = len(set(under[under >= 0].ravel().tolist()))
                        score = nx * ny * h + 2.0 * max(0, bridges - 1) + (0.5 if (nx >= ny) == along_x else 0)
                        if best is None or score > best[0]:
                            best = (score, h, nx, ny, stem)
                if best is None:
                    continue
                _, h, nx, ny, stem = best
                part = load_part(stem)
                turn = next(t for t in part.turns if (t.nx, t.ny) == (nx, ny))
                cells = [(a, c, zz) for a in range(i, i + nx) for c in range(j, j + ny) for zz in range(z, z + h)]
                self._add(stem, -1, turn.matrix(i, j, z), "fill", cells)
                n_fill += 1
        self.info["fill_parts"] = n_fill

    # -- 4. colour -------------------------------------------------------------------------------------------------
    def colour(self):
        names, _ = palette()
        t = self.t
        filled = self.grid >= 0
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
        belt = 0.45 * np.nanmax(self.T)
        for idx, p in enumerate(self.model.parts):
            if p.color >= 0:
                continue
            pts = []
            for i, j, z in p.cells:
                exposed_top = z + 1 >= self.levels or not filled[i, j, z + 1]
                if exposed_top:
                    pts.append((t.cell_x(i), t.cell_y(j), (z + 1.2) * t.plate))
                for di, dj in dirs:
                    a, b = i + di, j + dj
                    if a < 0 or b < 0 or a >= self.L or b >= self.W or not filled[a, b, z]:
                        pts.append((t.cell_x(i + 0.6 * di), t.cell_y(j + 0.6 * dj), (z + 0.5) * t.plate))
            if not pts:
                p.color = LDRAW_CODE[self.cfg.interior_color]
                continue
            rgb = t.color_at(np.array(pts))
            q = quantize(rgb * 255)
            name = names[Counter(q.tolist()).most_common(1)[0][0]]
            zmean = np.mean([c[2] for c in p.cells]) if p.cells else 0
            if name in ("black", "dark bluish gray") and zmean > belt and p.role != "holder":
                name = "trans black"
            p.color = LDRAW_CODE.get(name, LDRAW_CODE["light bluish gray"])

    def prune(self, min_parts: int = 5):
        """Drop small groups that are not connected to the main body (thin mirrors, single-column slivers)."""
        comps = self.model.components()
        drop = {i for c in comps[1:] if len(c) < min_parts for i in c}
        if drop:
            self.model.parts = [p for i, p in enumerate(self.model.parts) if i not in drop]
        self.info["dropped_parts"] = len(drop)

    def run(self) -> Design:
        self.wheels()
        self.surface()
        self.details()
        self.body()
        self.colour()
        self.prune()
        comps = self.model.components()
        self.info.update({"parts": len(self.model.parts), "components": len(comps),
                          "largest_component": len(comps[0]) if comps else 0,
                          "size_studs": [self.L, self.W, int(self.base.max())]})
        return Design(self.model, self.base, self.B, self.occ, self.info)
