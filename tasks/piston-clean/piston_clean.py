"""On the station table, in front of the arms: a dark rubber bench mat, 30 x 30 cm. On it lies a piston taken out
of a single-cylinder engine (cast aluminium, 65 mm across, 55 mm long, its crown facing left) still on its
connecting rod (forged steel, 130 mm long), which hangs from the wrist pin through the piston's skirt and swings
freely on it. Brown used engine oil coats the piston's skirt and crown. To the right of the mat, in front of the
right arm: a folded lint-free cotton cloth (5 x 5 cm, 1.2 cm thick) and a small steel tray of parts-washer
solvent, 10 x 6 cm inside and 2.5 cm deep.

Only the cloth wipes oil off; the grippers do not. A cloth that has picked up a lot of oil smears it back
instead of cleaning, until it is rinsed in the solvent tray.
"""

# No `from __future__ import annotations`: the task loader imports this file without registering it in
# sys.modules (see finish-painting-panel).
import importlib
import importlib.abc
import importlib.util
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box, Cylinder

# rdzeń punktacji leży obok (wersja robocza; przed zgłoszeniem wklejony do tego pliku)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import piston_scoring as ps  # noqa: E402

SCENE = "station"
TOP: float = TABLE_TOP[SCENE]

# --- układ (świat: x od ramion, y w lewo, z w górę; metry) ---------------------------------------------------
MAT_CENTER = (0.30, 0.03)
MAT_SIZE = (0.30, 0.30, 0.004)
MAT_TOP = TOP + MAT_SIZE[2]

PISTON = "piston"
ROD = "conrod"
PAD = "cloth"
PISTON_R = ps.PISTON_D / 2
PISTON_POS = (0.30, 0.09, MAT_TOP + PISTON_R)  # oś wzdłuż y, denko w stronę +y (w lewo)
PISTON_MASS = 0.18
ROD_SIZE = (0.022, 0.13, 0.014)  # x (szerokość), y (długość), z (grubość)
ROD_MASS = 0.15
PIN = PISTON_POS  # sworzeń w środku tłoka (uproszczenie: realnie ~40% wysokości od denka)
ROD_POS = (PIN[0], PIN[1] - ROD_SIZE[1] / 2, PIN[2])  # mała główka na sworzniu; opadnie na matę
PIN_JOINT = "wrist_pin"
PIN_LIMIT_DEG = 60.0

PAD_POS = (0.24, -0.20, TOP + ps.PAD_SIZE[2] / 2 + 0.001)
PAD_MASS = 0.005

TRAY_CENTER = (0.36, -0.22)
TRAY_WALL = 0.004
TRAY_FLOOR_Z = TOP + 0.002  # dno wanienki (2 mm blachy)

STEEL = (0.62, 0.64, 0.66)
ALU = (0.78, 0.78, 0.76)
OILY = (0.45, 0.33, 0.16)  # olej na tłoku pokazuje kolor; czystość liczy punktacja
MAT_RGB = (0.12, 0.12, 0.12)
CLOTH_RGB = (0.93, 0.93, 0.90)


def _tray_parts() -> list:
    cx, cy = TRAY_CENTER
    ix, iy, depth = ps.TRAY_INNER
    w = TRAY_WALL
    z = TOP + depth / 2
    return [
        Box("tray_floor", (cx, cy, TOP + 0.001), STEEL, size=(ix + 2 * w, iy + 2 * w, 0.002)),
        Box("tray_wall_n", (cx + ix / 2 + w / 2, cy, z), STEEL, size=(w, iy + 2 * w, depth)),
        Box("tray_wall_s", (cx - ix / 2 - w / 2, cy, z), STEEL, size=(w, iy + 2 * w, depth)),
        Box("tray_wall_w", (cx, cy + iy / 2 + w / 2, z), STEEL, size=(ix, w, depth)),
        Box("tray_wall_e", (cx, cy - iy / 2 - w / 2, z), STEEL, size=(ix, w, depth)),
    ]


OBJECTS = [
    Box("bench_mat", (*MAT_CENTER, TOP + MAT_SIZE[2] / 2), MAT_RGB, size=MAT_SIZE),
    Cylinder(PISTON, PISTON_POS, OILY, mass=PISTON_MASS, radius=PISTON_R, height=ps.PISTON_H),
    Box(ROD, ROD_POS, STEEL, mass=ROD_MASS, size=ROD_SIZE),
    Box(PAD, PAD_POS, CLOTH_RGB, mass=PAD_MASS, size=ps.PAD_SIZE),
    *_tray_parts(),
]


# --- przegub sworznia: po zbudowaniu korbowodu dokładamy RevoluteJoint (wzorzec: finish-painting-panel) ------
_SCENE_MODULE = "yam_arena.scene"
_WARNED: set[str] = set()


def _warn_once(key: str, text: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(f"piston_clean: {text}", stacklevel=2)


def author_pin(stage: Any, env_root: str) -> None:
    gf = importlib.import_module("pxr.Gf")
    physics = importlib.import_module("pxr.UsdPhysics")
    joint = physics.RevoluteJoint.Define(stage, f"{env_root}/{PIN_JOINT}")
    joint.CreateAxisAttr("X")  # oś sworznia poprzeczna do osi tłoka; korbowód wahnie się w płaszczyźnie y-z
    joint.CreateBody0Rel().SetTargets([f"{env_root}/{PISTON}"])
    joint.CreateBody1Rel().SetTargets([f"{env_root}/{ROD}"])
    joint.CreateLocalPos0Attr(gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(gf.Vec3f(0.0, ROD_SIZE[1] / 2, 0.0))  # mała główka korbowodu
    joint.CreateLocalRot1Attr(gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLowerLimitAttr(-PIN_LIMIT_DEG)
    joint.CreateUpperLimitAttr(PIN_LIMIT_DEG)


def _env_roots(prim: Any) -> list[str]:
    stage = prim.GetStage()
    envs = prim.GetPath().GetParentPath().GetParentPath()
    return [str(e.GetPath()) for e in stage.GetPrimAtPath(envs).GetChildren() if e.GetChild(PISTON).IsValid()]


def _with_pin(spawn):
    def spawn_and_pin(prim_path: str, cfg: Any, *args: Any, **kwargs: Any) -> Any:
        prim = spawn(prim_path, cfg, *args, **kwargs)
        try:
            for root in _env_roots(prim):
                author_pin(prim.GetStage(), root)
        except Exception as error:  # bez sworznia korbowód leży luzem; zadanie byłoby inne
            _warn_once("pin", f"could not build the wrist pin ({type(error).__name__}: {error})")
        return prim

    spawn_and_pin.piston_pin = True
    return spawn_and_pin


def _spawner_cfgs(item: object):
    seen: set[int] = set()
    stack = [item, *vars(item).values()] if hasattr(item, "__dict__") else [item]
    for candidate in stack:
        for value in (candidate, getattr(candidate, "spawn", None), getattr(candidate, "spawner_cfg", None)):
            if value is None or id(value) in seen:
                continue
            seen.add(id(value))
            if callable(getattr(value, "func", None)) and (hasattr(value, "size") or hasattr(value, "radius")):
                yield value


def build_pin(scene: Any) -> bool:
    original = getattr(scene, "task_object", None)
    if original is None or getattr(original, "piston_pin", False):
        return False

    def task_object(spec: Any) -> Any:
        made = original(spec)
        name = getattr(spec, "name", None)
        if name in (PISTON, ROD):
            for cfg in _spawner_cfgs(made):
                if name == PISTON:
                    if hasattr(cfg, "axis"):
                        cfg.axis = "Y"  # tłok leży na boku
                    else:
                        _warn_once("axis", "the piston's spawner has no axis setting")
                elif not getattr(cfg.func, "piston_pin", False):
                    cfg.func = _with_pin(cfg.func)  # korbowód budowany po tłoku: obie bryły już są
        return made

    task_object.piston_pin = True
    scene.task_object = task_object
    return True


class _PinWatcher(importlib.abc.MetaPathFinder):
    """Czeka na import yam_arena.scene (dopiero po starcie Isaac Sim), potem nakłada build_pin."""

    marker = "piston_clean.pin_watcher"

    def __init__(self) -> None:
        self._looking = False

    def find_spec(self, fullname, path, target=None):
        if fullname != _SCENE_MODULE or self._looking:
            return None
        self._looking = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._looking = False
        if spec is not None and spec.loader is not None:
            spec.loader = _PinAfterLoad(spec.loader, self)
        return spec

    def disarm(self) -> None:
        if self in sys.meta_path:
            sys.meta_path.remove(self)


class _PinAfterLoad(importlib.abc.Loader):
    def __init__(self, inner, watcher) -> None:
        self._inner = inner
        self._watcher = watcher

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module) -> None:
        self._inner.exec_module(module)
        build_pin(module)
        self._watcher.disarm()

    def __getattr__(self, name):
        if name.startswith("__") or name in ("_inner", "_watcher"):
            raise AttributeError(name)
        return getattr(self._inner, name)


def install_pin() -> None:
    """Nie importuje sceny przed startem symulatora (to psuje omni.syntheticdata)."""
    scene = sys.modules.get(_SCENE_MODULE)
    if scene is not None:
        build_pin(scene)
        return
    if not any(getattr(f, "marker", None) == _PinWatcher.marker for f in sys.meta_path):
        sys.meta_path.insert(0, _PinWatcher())


install_pin()


# --- punktacja ------------------------------------------------------------------------------------------------
_FIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # oś z siatki -> oś y tłoka (denko +y)
_LEDGER = ps.Ledger()
_START = np.array(PISTON_POS)


def _piston_quat_for_grid(q_xyzw: np.ndarray) -> np.ndarray:
    """Kwaternion bryły tłoka złożony z obrotem siatki (oś z) na oś cylindra (y)."""
    r = ps.quat_xyzw_to_matrix(q_xyzw) @ _FIX
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    if w < 1e-6:  # obrót o ~180°: wystarczy dowolna dokładna reprezentacja
        x = math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2
        return np.array([x, (r[0, 1] + r[1, 0]) / (4 * x), (r[0, 2] + r[2, 0]) / (4 * x), (r[2, 1] - r[1, 2]) / (4 * x)])
    return np.array([(r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w), w])


def _observe(objects) -> None:
    global _LEDGER
    p, pad = objects[PISTON], objects[PAD]
    at_start = np.linalg.norm(np.asarray(p.pos) - _START) < 0.01 and np.linalg.norm(np.asarray(pad.pos) - np.array(PAD_POS)) < 0.01
    if at_start and (_LEDGER.zeroed or _LEDGER.clean_fraction > 0) and p.resting():
        _LEDGER = ps.Ledger()  # nowa próba (stan startowy); w wersji do zgłoszenia: hak na reset jak w malowaniu
    _LEDGER.step(
        p.pos, _piston_quat_for_grid(p.quat_xyzw), p.lin_vel, p.low, p.high,
        pad.pos, pad.quat_xyzw, np.array([*TRAY_CENTER, 0.0]), TRAY_FLOOR_Z,
    )


def score(objects) -> float:
    _observe(objects)
    p, rod = objects[PISTON], objects[ROD]
    cx, cy = MAT_CENTER
    on_mat = abs(p.pos[0] - cx) <= MAT_SIZE[0] / 2 and abs(p.pos[1] - cy) <= MAT_SIZE[1] / 2 and p.low[2] <= MAT_TOP + 0.01
    return _LEDGER.score(on_mat=bool(on_mat), resting=p.resting() and rod.resting())


# --- demo: na razie tylko postój ramion nad stołem (podgląd sceny); pełne demo w następnym kroku -----------------
def demo() -> dict:
    hold = lambda y: [Waypoint(3.0, (0.22, y, TOP + 0.25), FINGER_OPEN, "hold above the table")]  # noqa: E731
    return {"left": hold(0.25), "right": hold(-0.25)}
