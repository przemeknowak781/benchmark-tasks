"""Rdzeń punktacji zadania „Clean the oil off the piston” — czysty numpy, bez Isaac Sim.

Ten moduł zostanie wklejony do jednoplikowego pliku zadania `piston_clean.py` (yam_arena czyta jeden plik).
Trzymam go osobno, żeby dało się go przetestować na maszynie bez symulatora.

Model (spec v0.3, wariant 1 — wycieranie szmatką przed oględzinami):
- Powierzchnia tłoka podzielona na komórki ok. 1 cm: płaszcz (20 wokół x 5 w pionie, bez pasa pierścieni
  u góry) + denko (komórki siatki 1 cm wewnątrz koła). Każda komórka: brudna / czysta.
- Szmatka to złożona poduszka (pudełko). Komórka czyści się, gdy jej punkt leży w poduszce powiększonej
  o CONTACT_GAP. Tylko szmatka czyści — chwytaki nie.
- Szmatka ma pojemność CAPACITY komórek. Nasycona szmatka zamiast czyścić brudzi z powrotem komórki,
  których dotyka (rozmazuje olej). Zanurzenie poduszki w wanience z rozpuszczalnikiem zeruje ładunek.
- Zero za cały epizod: upadek tłoka (prędkość pionowa szybsza niż przy swobodnym spadku z 3 cm),
  tłok w wanience (zanurzanie zamiast wycierania), tłok poza matą na końcu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- geometria (m, kg) ----------------------------------------------------------------------------------
PISTON_D = 0.075  # Honda D16 (1,6 l), aluminium; wymiary i źródła w piston_geometry.py
PISTON_H = 0.050
RING_BELT = 0.0175  # górne 17,5 mm z rowkami pierścieni — poza punktacją (osobne narzędzie)
DISH_R, DISH_DEPTH = 0.026, 0.003  # wgłębienie w denku: tam komórki leżą 3 mm niżej
INNER_R = 0.034  # wewnętrzna ścianka płaszcza (75/2 - 3,5 mm)
BELT_Z = 0.0055  # do tej wysokości sięga cienki płaszcz od środka; wyżej gruby pierścień pod rowkami
UNDERCROWN_Z, UNDERCROWN_R = 0.016, 0.031  # spód denka od środka
PIN_Z, BOSS_HALF_DEG, BOSS_HALF_Z = -0.005, 20.0, 0.013  # piasty zasłaniają ściankę przy osi sworznia (x)
N_AROUND = 20
N_HIGH = 5
CELL = 0.01

PAD_SIZE = (0.04, 0.04, 0.04)  # gąbka 4 cm na końcu kijka (jak szczotka do mycia części, bez metalu)
CONTACT_GAP = 0.003  # 3 mm: komórka w zasięgu poduszki
SOFT_LAYER = 0.005  # gąbka: twardszy rdzeń 3 cm + 5 mm miękkiej warstwy (zderza się rdzeń)
CORE_TOUCH = 0.0015  # rdzeń musi dotykać powierzchni (komórka w rdzeniu + 1,5 mm)
WRAP = 0.006  # ściśnięta gąbka obejmuje krzywiznę: komórki do 6 mm za miękkim licem
CAPACITY = 60  # tyle komórek zbiera czysta gąbka, zanim zacznie rozmazywać

# wanienka celowo węższa niż średnica tłoka: tłoka nie da się w niej zanurzyć
TRAY_INNER = (0.10, 0.06, 0.025)  # x, y, głębokość
SOLVENT_DEPTH = 0.010

DROP_HEIGHT = 0.03
DROP_SPEED = math.sqrt(2 * 9.81 * DROP_HEIGHT)  # 0,77 m/s

CLEAN_STEPS = ((0.95, 1.0), (0.75, 0.75), (0.50, 0.5), (0.25, 0.25))


# --- siatka komórek w układzie tłoka (oś z = oś tłoka, denko na +z, środek bryły w 0) --------------------
def _skirt_cells() -> tuple[np.ndarray, np.ndarray]:
    r = PISTON_D / 2
    z0, z1 = -PISTON_H / 2, PISTON_H / 2 - RING_BELT
    th = (np.arange(N_AROUND) + 0.5) * 2 * math.pi / N_AROUND
    zs = z0 + (np.arange(N_HIGH) + 0.5) * (z1 - z0) / N_HIGH
    t, z = np.meshgrid(th, zs, indexing="ij")
    pts = np.stack([r * np.cos(t), r * np.sin(t), z], axis=-1).reshape(-1, 3)
    normals = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=-1).reshape(-1, 3)
    return pts, normals


def _crown_cells() -> tuple[np.ndarray, np.ndarray]:
    r = PISTON_D / 2
    g = (np.arange(-4, 5) + 0.0) * CELL
    gx, gy = np.meshgrid(g, g, indexing="ij")
    keep = np.hypot(gx, gy) <= r  # środki komórek wewnątrz denka
    z = np.where(np.hypot(gx[keep], gy[keep]) < DISH_R, PISTON_H / 2 - DISH_DEPTH, PISTON_H / 2)
    pts = np.stack([gx[keep], gy[keep], z], axis=1)
    normals = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    return pts, normals


def _inner_cells() -> tuple[np.ndarray, np.ndarray]:
    """Wnętrze płaszcza: ścianka od środka (bez miejsc zasłoniętych piastami) i spód denka."""
    z0, z1 = -PISTON_H / 2, BELT_Z
    rows = 3
    th = (np.arange(N_AROUND) + 0.5) * 2 * math.pi / N_AROUND
    zs = z0 + (np.arange(rows) + 0.5) * (z1 - z0) / rows
    pts, nrm = [], []
    for t in th:
        near_x = min(abs(math.degrees(t)) % 180, 180 - abs(math.degrees(t)) % 180) <= BOSS_HALF_DEG
        for z in zs:
            if near_x and abs(z - PIN_Z) <= BOSS_HALF_Z:
                continue
            pts.append((INNER_R * math.cos(t), INNER_R * math.sin(t), z))
            nrm.append((-math.cos(t), -math.sin(t), 0.0))
    g = np.arange(-3, 4) * CELL
    for x in g:
        for y in g:
            if math.hypot(x, y) <= UNDERCROWN_R - CELL / 2:
                pts.append((x, y, UNDERCROWN_Z))
                nrm.append((0.0, 0.0, -1.0))
    return np.asarray(pts), np.asarray(nrm)


_SKIRT, _SKIRT_N = _skirt_cells()
_CROWN, _CROWN_N = _crown_cells()
_INNER, _INNER_N = _inner_cells()
CELLS_LOCAL = np.vstack([_SKIRT, _CROWN, _INNER])
NORMALS_LOCAL = np.vstack([_SKIRT_N, _CROWN_N, _INNER_N])
N_CELLS = len(CELLS_LOCAL)
N_SKIRT = len(_SKIRT)
N_CROWN = len(_CROWN)
N_INNER = len(_INNER)


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def cells_world(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    return CELLS_LOCAL @ quat_xyzw_to_matrix(quat_xyzw).T + np.asarray(pos)


def pad_contacts(cells_w: np.ndarray, pad_pos: np.ndarray, pad_quat: np.ndarray, half=None) -> np.ndarray:
    """Maska komórek wewnątrz pudełka gąbki o połowie rozmiaru `half` (domyślnie gąbka + CONTACT_GAP)."""
    rot = quat_xyzw_to_matrix(pad_quat)
    local = (cells_w - np.asarray(pad_pos)) @ rot  # do układu gąbki
    if half is None:
        half = np.asarray(PAD_SIZE) / 2 + CONTACT_GAP
    return np.all(np.abs(local) <= half, axis=1)


def sponge_wipes(cells_w, facing, pad_pos, pad_quat, size=None) -> np.ndarray:
    """Czyści tylko dociśnięta gąbka: rdzeń dotyka powierzchni, a wtedy liczą się komórki pod miękką
    warstwą objętą z zapasem WRAP (gąbka ugina się na krzywiźnie). Bez docisku — nic."""
    size = np.asarray(PAD_SIZE if size is None else size)
    core = size / 2 - SOFT_LAYER + CORE_TOUCH
    if not (pad_contacts(cells_w, pad_pos, pad_quat, core) & facing).any():
        return np.zeros(len(cells_w), dtype=bool)
    return pad_contacts(cells_w, pad_pos, pad_quat, size / 2 + WRAP) & facing


FLAT_PAD_SIZE = (0.08, 0.08, 0.02)  # druga, płaska gąbka na macie: do denka (dociskanego od góry)


def pad_in_solvent(pad_pos: np.ndarray, tray_center: np.ndarray, tray_floor_z: float) -> bool:
    dx, dy = np.abs(np.asarray(pad_pos[:2]) - np.asarray(tray_center[:2]))
    low = pad_pos[2] - PAD_SIZE[2] / 2  # spód gąbki w rozpuszczalniku
    return bool(dx <= TRAY_INNER[0] / 2 and dy <= TRAY_INNER[1] / 2 and low <= tray_floor_z + SOLVENT_DEPTH)


def bbox_overlaps_tray(low: np.ndarray, high: np.ndarray, tray_center: np.ndarray, tray_floor_z: float) -> bool:
    tlo = np.array([tray_center[0] - TRAY_INNER[0] / 2, tray_center[1] - TRAY_INNER[1] / 2, tray_floor_z])
    thi = np.array([tray_center[0] + TRAY_INNER[0] / 2, tray_center[1] + TRAY_INNER[1] / 2, tray_floor_z + SOLVENT_DEPTH])
    return bool(np.all(np.asarray(low) < thi) and np.all(np.asarray(high) > tlo))


def ladder(fraction: float) -> float:
    for threshold, value in CLEAN_STEPS:
        if fraction >= threshold - 1e-12:
            return value
    return 0.0


@dataclass
class Ledger:
    """Stan epizodu, aktualizowany po każdym kroku sterowania."""

    dirty: np.ndarray = field(default_factory=lambda: np.ones(N_CELLS, dtype=bool))
    load: int = 0
    load_flat: int = 0  # ładunek płaskiej gąbki (wanienka jest dla niej za mała: denko ma 45 komórek < CAPACITY)
    rinses: int = 0
    zeroed: str | None = None  # powód zera za cały epizod

    def step(
        self,
        piston_pos, piston_quat, piston_vel, piston_low, piston_high,
        pad_pos, pad_quat, tray_center, tray_floor_z,
        flat_pos=None, flat_quat=None,
    ) -> None:
        if self.zeroed is None and piston_vel[2] < -DROP_SPEED:
            self.zeroed = "upuszczony tłok"
        if self.zeroed is None and bbox_overlaps_tray(piston_low, piston_high, tray_center, tray_floor_z):
            self.zeroed = "tłok w rozpuszczalniku"
        rot = quat_xyzw_to_matrix(piston_quat)
        cw = CELLS_LOCAL @ rot.T + np.asarray(piston_pos)
        nw = NORMALS_LOCAL @ rot.T
        if pad_in_solvent(pad_pos, tray_center, tray_floor_z):
            if self.load:
                self.rinses += 1
            self.load = 0
        else:
            self.load = self._wipe(cw, nw, pad_pos, pad_quat, PAD_SIZE, self.load)
        if flat_pos is not None:
            self.load_flat = self._wipe(cw, nw, flat_pos, flat_quat, FLAT_PAD_SIZE, self.load_flat)

    def _wipe(self, cw, nw, pos, quat, size, load) -> int:
        facing = np.einsum("ij,ij->i", np.asarray(pos) - cw, nw) > 0  # gąbka po stronie powierzchni
        touch = sponge_wipes(cw, facing, pos, quat, size)
        if not touch.any():
            return load
        if load >= CAPACITY:  # nasycona: rozmazuje
            self.dirty |= touch
            return load
        take = np.flatnonzero(touch & self.dirty)[: CAPACITY - load]
        self.dirty[take] = False
        return load + len(take)

    @property
    def clean_fraction(self) -> float:
        return float((~self.dirty).mean())

    def score(self, on_mat: bool, resting: bool) -> float:
        if self.zeroed is not None or not on_mat:
            return 0.0
        value = ladder(self.clean_fraction)
        if value == 1.0 and not resting:
            return 0.75  # 95% czyste, ale zespół nie leży spokojnie na macie (np. wciąż w chwytaku)
        return value
