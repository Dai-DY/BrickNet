"""Program-chosen connections ("snap"): given a part and a target position, enumerate every way the part can be
connected to the parts already placed and keep those that land near the target.

Ported from the sibling shape2brick project (``shape2brick/snap.py``), where it was the step that made shape
following work: with target-first sequences the model's *targets* follow the shape, but the connection lines it
writes do not reach them (median 11-19 cells off on a 64 grid) - inverting the kinematics is too hard for the
model. So the model writes the target and the part; this module lists the connections that get there, and the
model only ranks them.

Kinematics as in ``bricknet.graph`` (``_edge_matrix`` / ``_graph_edge_rotation`` / ``decode_graph``):
    child pose = parent pose @ parent connector frame @ J(params) @ inv(child connector frame)
    stud: J = Ry(yaw); hinge: Ry(yaw) (then Rx(pi) when flipped); axle: as hinge plus a translation of -slide
    along the connector's y axis; fixed: J = I.
Mating rules as in ``graph._find_pairs``. Parameter grid: stud yaw in {0, 90, 180, 270} (99 % of BrickNet's
connections); axle flip x yaw in 90 degree steps with the slide solved for analytically; hinge every 15 degrees;
ball joints are not enumerated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

STUD_YAWS = (0, 90, 180, 270)
AXLE_YAWS = (0, 90, 180, 270)
HINGE_YAWS = tuple(range(0, 360, 15))
AXLE_DEFAULT_RAD, AXLE_MIN_OVERLAP = 4.0, 4.0  # as bricknet.graph._AXLE_DEFAULT_RAD / _AXLE_MIN_OVERLAP
FAMILIES = ("stud", "axle", "hinge", "fixed")


def _ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]])


RX_PI = np.diag([1.0, -1.0, -1.0, 1.0])


def joint(fam: str, params: tuple) -> np.ndarray:
    """Connection parameters -> 4x4 joint transform (same as ``bricknet.graph._graph_edge_rotation``).
    params: stud (yaw,); hinge (flip, yaw); axle (flip, yaw, slide); fixed ()."""
    if fam == "stud":
        return _ry(params[0])
    if fam == "hinge":
        R = _ry(params[1])
        return R @ RX_PI if params[0] else R
    if fam == "axle":
        R = _ry(params[1])
        if params[0]:
            R = R @ RX_PI
        R[1, 3] = -params[2]
        return R
    if fam == "fixed":
        return np.eye(4)
    raise ValueError(fam)


def _param_grid(fam: str) -> list[tuple]:
    if fam == "stud":
        return [(y,) for y in STUD_YAWS]
    if fam == "hinge":
        return [(f, y) for f in (False, True) for y in HINGE_YAWS]
    if fam == "axle":
        return [(f, y, 0) for f in (False, True) for y in AXLE_YAWS]
    return [()]


@dataclass
class Cand:
    parent: int  # index of the placed parent part
    fam: str
    ia: int  # parent connector (flat index)
    ib: int  # child connector (flat index)
    params: tuple
    point: np.ndarray  # where the child's reference point lands (frame of the placed poses)
    cheb: float  # max per-axis distance to the target
    dist: float  # Euclidean distance to the target


class ConnIndex:
    """Per-part connectors (flat order) and frames, and which connectors of two parts can mate (cached)."""

    def __init__(self):
        self._info: dict[int, tuple] = {}
        self._compat: dict[tuple[int, int], dict] = {}

    def info(self, pid: int):
        """(conns, frames (N, 4, 4), inverse frames); each conn is (kind, sub, polarity, index in group, length)."""
        if pid not in self._info:
            from bricknet.graph import _part_info

            pi = _part_info(pid)
            frames = np.asarray(pi.frames, float).reshape(-1, 4, 4)
            self._info[pid] = (pi.conns, frames, np.linalg.inv(frames) if len(frames) else frames)
        return self._info[pid]

    def compat(self, pid_a: int, pid_b: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Mating connectors of parent part ``pid_a`` and child part ``pid_b``: {family: (parent idx, child idx)}."""
        key = (pid_a, pid_b)
        if key in self._compat:
            return self._compat[key]
        from bricknet.core import _AXLE_MATES, _STUD_MATES

        ca, cb = self.info(pid_a)[0], self.info(pid_b)[0]

        def groups(conns):
            g: dict[tuple, list[int]] = {}
            for i, c in enumerate(conns):
                g.setdefault((c[0], c[1], c[2]), []).append(i)
            return g

        out: dict[str, list] = {}
        for (ka, sa, pa), ia in groups(ca).items():
            for (kb, sb, pb), ib in groups(cb).items():
                fam = None
                if {ka, kb} == {"stud", "hole"} and (min(sa, sb), max(sa, sb)) in _STUD_MATES:
                    fam = "stud"
                elif ka == kb == "axle" and (min(sa, sb), max(sa, sb)) in _AXLE_MATES:
                    fam = "axle"
                elif ka == kb and ka in ("hinge", "fixed") and sa == sb and pa is not None and pb is not None \
                        and pa != pb:
                    fam = ka
                if fam is None:
                    continue
                aa, bb = np.meshgrid(ia, ib, indexing="ij")
                out.setdefault(fam, []).append((aa.ravel(), bb.ravel()))
        res = {f: (np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])) for f, v in out.items()}
        self._compat[key] = res
        return res


def enumerate_candidates(index: ConnIndex, placed: list[tuple[int, np.ndarray]], pid: int, ref_local: np.ndarray,
                         target: np.ndarray, radius: float, families=FAMILIES) -> list[Cand]:
    """All connections of part ``pid`` to ``placed = [(part_id, pose), ...]`` that put the part's reference point
    (``ref_local``, in its own frame; car2brick uses the origin, which is what anchors record) within ``radius``
    of ``target`` on every axis. Sorted by (per-axis distance, Euclidean distance)."""
    conns_b, frames_b, inv_b = index.info(pid)
    if not len(frames_b):
        return []
    v = (inv_b @ np.append(np.asarray(ref_local, float), 1.0))[:, :3]  # reference point in each child connector frame
    dv = np.linalg.norm(v, axis=1)
    target = np.asarray(target, float)
    reach = float(dv.max()) + radius * math.sqrt(3) + 1e-6
    out: list[Cand] = []
    for j, (ppid, T) in enumerate(placed):
        comp = index.compat(ppid, pid)
        if not comp:
            continue
        conns_a, frames_a, _ = index.info(ppid)
        Fw = T @ frames_a
        pa = Fw[:, :3, 3]
        da = np.linalg.norm(pa - target, axis=1)
        if da.min() > reach and "axle" not in comp:
            continue
        for fam, (ia, ib) in comp.items():
            if fam not in families:
                continue
            if fam != "axle":  # |point - parent connector| = |v| for any rigid joint: triangle-inequality prune
                keep = np.abs(da[ia] - dv[ib]) <= radius * math.sqrt(3) + 1e-6
                ia, ib = ia[keep], ib[keep]
            if not len(ia):
                continue
            Rw = Fw[ia, :3, :3]
            for prm in _param_grid(fam):
                J = joint(fam, prm)
                local = v[ib] @ J[:3, :3].T + J[:3, 3]
                c = np.einsum("nij,nj->ni", Rw, local) + pa[ia]
                if fam != "axle":
                    cheb = np.abs(c - target).max(1)
                    for h in np.where(cheb <= radius + 1e-6)[0]:
                        out.append(Cand(j, fam, int(ia[h]), int(ib[h]), prm, c[h], float(cheb[h]),
                                        float(np.linalg.norm(c[h] - target))))
                    continue
                # axle: point = c0 - slide * (parent connector y axis); every integer slide (within the overlap
                # limit) that lands inside the target box, since the model may have learnt any of them
                ax = Rw[:, :, 1]
                la = np.array([max(float(conns_a[i][4]), AXLE_DEFAULT_RAD) for i in ia])
                lb = np.array([max(float(conns_b[i][4]), AXLE_DEFAULT_RAD) for i in ib])
                smax = np.floor(la + lb - AXLE_MIN_OVERLAP)
                for h in range(len(ia)):
                    sm = int(smax[h])
                    s = np.arange(-sm, sm + 1)
                    pts = c[h] - s[:, None] * ax[h]
                    cheb = np.abs(pts - target).max(1)
                    for q in np.where(cheb <= radius + 1e-6)[0]:
                        out.append(Cand(j, fam, int(ia[h]), int(ib[h]), (prm[0], prm[1], int(s[q])), pts[q],
                                        float(cheb[q]), float(np.linalg.norm(pts[q] - target))))
    out.sort(key=lambda c: (c.cheb, c.dist))
    return out


def make_edge(index: ConnIndex, cand: Cand, child: int, ppid: int, pid: int):
    """Cand -> bricknet Edge (``child`` is the new part's index)."""
    from bricknet.core import AxleEdge, AxleSub, FixedEdge, HingeEdge, StudEdge, StudSub

    pc, cc = index.info(ppid)[0][cand.ia], index.info(pid)[0][cand.ib]
    base = (cand.parent, child)
    if cand.fam == "stud":
        return StudEdge(*base, StudSub[pc[1]], StudSub[cc[1]], pc[3], cc[3], yaw=int(cand.params[0]) % 360)
    if cand.fam == "axle":
        f, y, s = cand.params
        return AxleEdge(*base, AxleSub[pc[1]], AxleSub[cc[1]], pc[3], cc[3], flip=bool(f), yaw=int(y) % 360,
                        slide=int(s))
    sub = int(pc[1])
    ps, cs = sub * 2 + (pc[2] == "on"), sub * 2 + (cc[2] == "on")
    if cand.fam == "hinge":
        return HingeEdge(*base, ps, cs, pc[3], cc[3], flip=bool(cand.params[0]), yaw=int(cand.params[1]) % 360)
    return FixedEdge(*base, ps, cs, pc[3], cc[3])


def edge_line(e) -> str:
    """Path text of one edge (exactly as ``bricknet.tree.serialize_tree`` writes it, without the newline)."""
    from bricknet.core import AxleEdge, AxleSub, BallEdge, HingeEdge, StudEdge, StudSub
    from bricknet.tree import i2id

    p, pc, cc = i2id(e.parent), i2id(e.parent_conn), i2id(e.child_conn)
    if isinstance(e, StudEdge):
        return f"{p} stud {StudSub(e.parent_sub).name} {pc} {StudSub(e.child_sub).name} {cc} {e.yaw}"
    if isinstance(e, AxleEdge):
        f = "flip" if e.flip else "regular"
        return f"{p} axle {AxleSub(e.parent_sub).name} {pc} {AxleSub(e.child_sub).name} {cc} {f} {e.yaw} {e.slide}"
    sub, pol = i2id(e.parent_sub // 2), "in" if e.parent_sub % 2 == 0 else "on"
    if isinstance(e, HingeEdge):
        return f"{p} hinge {sub} {pol} {pc} {cc} {'flip' if e.flip else 'regular'} {e.yaw}"
    if isinstance(e, BallEdge):
        return f"{p} ball {sub} {pol} {pc} {cc} {e.rx} {e.ry} {e.rz}"
    return f"{p} fixed {sub} {pol} {pc} {cc}"
