"""LDraw file helpers: flatten multi-part documents (.mpd / submodel .ldr) and keep instruction step numbers.

``flatten(text)`` returns plain part lines (``1 colour x y z a..i stem.dat``) with all submodel references expanded,
plus the build step of every part line (``0 STEP`` / ``0 ROTSTEP`` in the main model; parts of a submodel get the
step of the line that places the submodel). Parts unknown to BrickNet's catalog are dropped (reported in
``skipped``), so the result is always accepted by ``bricknet.parse_ldr``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import bricknet
from bricknet.data import load_aliases


@dataclass
class Flat:
    lines: list[str] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _split_files(text: str) -> tuple[str, dict[str, list[str]]]:
    files: dict[str, list[str]] = {}
    main, cur = None, None
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("0 file "):
            cur = s[7:].strip().lower()
            files[cur] = []
            main = main or cur
        elif s.lower().startswith("0 nofile"):
            cur = None
        elif cur is not None:
            files[cur].append(s)
    if not files:
        files["__main__"] = [ln.strip() for ln in text.splitlines()]
        main = "__main__"
    return main, files


def flatten(text: str) -> Flat:
    catalog = bricknet.load_catalog()
    aliases = load_aliases()
    main, files = _split_files(text)
    out = Flat()

    def known(stem: str) -> bool:
        return stem in catalog.stem_to_id or stem in aliases

    def walk(name: str, mat: np.ndarray, color: int, step: int | None, depth: int):
        step_no = 0
        for s in files.get(name, []):
            tok = s.split()
            if not tok:
                continue
            if tok[0] == "0" and len(tok) > 1 and tok[1].upper() in ("STEP", "ROTSTEP") and step is None:
                step_no += 1
                continue
            if tok[0] != "1" or len(tok) < 15:
                continue
            ref = " ".join(tok[14:]).lower()
            v = np.array(tok[2:14], dtype=np.float64)
            local = np.eye(4)
            local[:3, :3] = v[3:].reshape(3, 3)
            local[:3, 3] = v[:3]
            world = mat @ local
            c = int(tok[1])
            c = color if c in (16, 24) else c
            cur_step = step if step is not None else step_no
            if ref in files and depth < 32:
                walk(ref, world, c, cur_step, depth + 1)
                continue
            stem = ref.removesuffix(".dat").replace("\\", "/").split("/")[-1]
            if not known(stem):
                out.skipped[stem] = out.skipped.get(stem, 0) + 1
                continue
            nums = " ".join(f"{x:.6g}" for x in [*world[:3, 3], *world[:3, :3].flat])
            out.lines.append(f"1 {c} {nums} {stem}.dat")
            out.steps.append(cur_step)

    walk(main, np.eye(4), 71, None, 0)  # top-level 'main colour' (16) -> light bluish gray
    return out
