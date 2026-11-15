"""On the station table, in front of the arms: a dark rubber bench mat, 30 x 30 cm. On it lies a piston taken out
of a single-cylinder engine (cast aluminium, 65 mm across, 55 mm long, its crown facing left) still on its
connecting rod (forged steel, 130 mm long), which hangs from the wrist pin through the piston's skirt and swings
freely on it. Brown used engine oil coats the piston's skirt and crown. To the right of the mat, in front of the
right arm: a parts-cleaning sponge on a stick (a 20 cm wooden handle, its far end resting on a 4 cm cube of
sponge, the near end on a small block) and a small steel tray of parts-washer solvent, 10 x 6 cm inside and
2.5 cm deep.

Only the sponge wipes oil off; the grippers and the handle do not. A sponge that has picked up a lot of oil
smears it back instead of cleaning, until it is rinsed in the solvent tray.
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
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box, Cylinder

# rdzeń punktacji leży obok (wersja robocza; przed zgłoszeniem wklejony do tego pliku)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import piston_geometry as pg  # noqa: E402
import piston_scoring as ps  # noqa: E402

SCENE = "station"
TOP: float = TABLE_TOP[SCENE]

# --- układ (świat: x od ramion, y w lewo, z w górę; metry) ---------------------------------------------------
MAT_CENTER = (0.30, 0.03)
MAT_SIZE = (0.30, 0.30, 0.004)
MAT_TOP = TOP + MAT_SIZE[2]

PISTON = "piston"
ROD = "conrod"
STICK = "sponge_stick"
PAD = "sponge"
PISTON_R = ps.PISTON_D / 2
PISTON_POS = (0.30, 0.06, MAT_TOP + ps.PISTON_H / 2)  # tłok stoi denkiem na macie, płaszcz otwarty do góry
PISTON_MASS = 0.30  # tłok ~245 g (oferty CP dla D16) + sworzeń rurowy ~55 g
ROD_SIZE = (pg.ROD_T, 0.035, pg.CTC + 2 * 0.022)  # zderzenia: x grubość (oś sworznia), y szerokość belki, z długość
ROD_MASS = 0.47  # ~470 g z pokrywą i śrubami
PIN_OFFSET = -pg.PIN_Z  # oś sworznia 30 mm nad denkiem = 5 mm nad środkiem bryły (denko na dole)
PIN = (PISTON_POS[0], PISTON_POS[1], PISTON_POS[2] + PIN_OFFSET)
ROD_POS = (PIN[0], PIN[1], PIN[2] + pg.CTC / 2)  # stopa na sworzniu, korbowód w górę; opadnie na krawędź płaszcza
PIN_JOINT = "wrist_pin"
PIN_LIMIT_DEG = pg.rod_swing_limit_deg()  # ~44,8°: bok belki opiera się o dolną krawędź płaszcza

STICK_SIZE = (0.20, 0.016, 0.016)  # drewniany trzonek wzdłuż x
STICK_MASS = 0.025  # drewno ~0,6 g/cm3
PAD_POS = (0.33, -0.25, TOP + ps.PAD_SIZE[2] / 2 + 0.001)
PAD_MASS = 0.003  # gąbka PU ~0,05 g/cm3
STICK_TO_PAD = STICK_SIZE[0] / 2 + ps.PAD_SIZE[0] / 2  # środek kijka -> środek gąbki, wzdłuż x
STICK_POS = (PAD_POS[0] - STICK_TO_PAD, PAD_POS[1], PAD_POS[2])
REST_SIZE = (0.03, 0.03, STICK_POS[2] - STICK_SIZE[2] / 2 - TOP)  # klocek pod bliższym końcem trzonka
REST_POS = (STICK_POS[0] - STICK_SIZE[0] / 2 + 0.02, STICK_POS[1], TOP + REST_SIZE[2] / 2)
STICK_JOINT = "sponge_glue"

TRAY_CENTER = (0.36, -0.15)
TRAY_WALL = 0.004
TRAY_FLOOR_Z = TOP + 0.002  # dno wanienki (2 mm blachy)

STEEL = (0.62, 0.64, 0.66)
ALU = (0.78, 0.78, 0.76)
OILY = (0.52, 0.40, 0.22)  # olej na tłoku pokazuje kolor; czystość liczy punktacja
MAT_RGB = (0.12, 0.12, 0.12)
SPONGE_RGB = (0.95, 0.80, 0.25)
WOOD_RGB = (0.62, 0.45, 0.28)


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
    Box("stick_rest", REST_POS, (0.35, 0.36, 0.38), size=REST_SIZE),
    Box(STICK, STICK_POS, WOOD_RGB, mass=STICK_MASS, size=STICK_SIZE),
    Box(PAD, PAD_POS, SPONGE_RGB, mass=PAD_MASS, size=ps.PAD_SIZE),
    *_tray_parts(),
]


# --- przegub sworznia: po zbudowaniu korbowodu dokładamy RevoluteJoint (wzorzec: finish-painting-panel) ------
_SCENE_MODULE = "yam_arena.scene"
_WARNED: set[str] = set()


def _warn_once(key: str, text: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(f"piston_clean: {text}", stacklevel=2)


def _material(stage, path, rgb, metallic, roughness):
    gf = importlib.import_module("pxr.Gf")
    sdf = importlib.import_module("pxr.Sdf")
    shade = importlib.import_module("pxr.UsdShade")
    mat = shade.Material.Define(stage, path)
    sh = shade.Shader.Define(stage, f"{path}/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    sh.CreateInput("diffuseColor", sdf.ValueTypeNames.Color3f).Set(gf.Vec3f(*lin))
    sh.CreateInput("metallic", sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("roughness", sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _mesh(stage, path, data, mat, rot=None, translate=(0.0, 0.0, 0.0)):
    """Siatka z punktami przeliczonymi do układu bryły (rot 3x3, translate)."""
    gf = importlib.import_module("pxr.Gf")
    geom = importlib.import_module("pxr.UsdGeom")
    shade = importlib.import_module("pxr.UsdShade")
    pts, counts, idx = data
    pts = pts if rot is None else pts @ np.asarray(rot).T
    pts = pts + np.asarray(translate)
    m = geom.Mesh.Define(stage, path)
    m.CreatePointsAttr([gf.Vec3f(*map(float, p)) for p in pts])
    m.CreateFaceVertexCountsAttr([int(c) for c in counts])
    m.CreateFaceVertexIndicesAttr([int(i) for i in idx])
    m.CreateDoubleSidedAttr(True)
    m.CreateSubdivisionSchemeAttr("none")
    shade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)


def _cube(stage, path, centre, size, mat=None, yaw=0.0, collide=False):
    gf = importlib.import_module("pxr.Gf")
    geom = importlib.import_module("pxr.UsdGeom")
    shade = importlib.import_module("pxr.UsdShade")
    physics = importlib.import_module("pxr.UsdPhysics")
    c = geom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    x = geom.Xformable(c.GetPrim())
    x.AddTranslateOp().Set(gf.Vec3d(*map(float, centre)))
    if yaw:
        x.AddRotateZOp().Set(math.degrees(yaw))
    x.AddScaleOp().Set(gf.Vec3f(*map(float, size)))
    if mat is not None:
        shade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    if collide:
        physics.CollisionAPI.Apply(c.GetPrim())
        geom.Imageable(c.GetPrim()).CreatePurposeAttr("guide")


def _cylinder_collider(stage, path, centre, radius, height):
    gf = importlib.import_module("pxr.Gf")
    geom = importlib.import_module("pxr.UsdGeom")
    physics = importlib.import_module("pxr.UsdPhysics")
    c = geom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(float(radius))
    c.CreateHeightAttr(float(height))
    c.CreateAxisAttr("Z")
    geom.Xformable(c.GetPrim()).AddTranslateOp().Set(gf.Vec3d(*map(float, centre)))
    physics.CollisionAPI.Apply(c.GetPrim())
    geom.Imageable(c.GetPrim()).CreatePurposeAttr("guide")


def _retire_primitive(stage, body_path):
    """Prosta bryła ze spawnera: niewidoczna (guide); w tłoku także bez zderzeń (zastępuje ją pusty płaszcz)."""
    geom = importlib.import_module("pxr.UsdGeom")
    usd = importlib.import_module("pxr.Usd")
    physics = importlib.import_module("pxr.UsdPhysics")
    for prim in usd.PrimRange(stage.GetPrimAtPath(body_path)):
        path = str(prim.GetPath())
        if "/visual" in path or "/collision" in path or not prim.IsA(geom.Gprim):
            continue
        geom.Imageable(prim).CreatePurposeAttr("guide")
        if body_path.endswith(PISTON) and prim.HasAPI(physics.CollisionAPI):
            physics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)


P_BODY = np.diag([1.0, -1.0, -1.0])  # układ tłoka z piston_geometry (denko +z) -> bryła (denko w dół)
R_ROD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # korbowód: długość y -> z, stopa na dole
Z_TO_X = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # oś z pierścienia -> oś sworznia x


def _perm(rot, size):
    return tuple(float(abs(v)) for v in np.abs(np.asarray(rot)) @ np.asarray(size))


def author_visuals(stage: Any, env_root: str) -> None:
    piston, rod = f"{env_root}/{PISTON}", f"{env_root}/{ROD}"
    _retire_primitive(stage, piston)
    _retire_primitive(stage, rod)
    alu = _material(stage, f"{piston}/visual/alu", OILY, 0.6, 0.35)  # aluminium pod warstwą oleju
    steel = _material(stage, f"{rod}/visual/steel", (0.36, 0.37, 0.39), 0.85, 0.4)
    _mesh(stage, f"{piston}/visual/body", pg.lathe(pg.piston_profile(), 96), alu, P_BODY)
    inner = pg.R - pg.WALL
    for side in (-1, 1):
        _cube(stage, f"{piston}/visual/boss_{'ab'[side > 0]}", (side * (inner - 0.008), 0.0, PIN_OFFSET), (0.016, 0.022, 0.026), alu)
    _mesh(stage, f"{piston}/visual/pin", pg.annulus(0.0055, pg.PIN_D / 2, 2 * inner - 0.002), steel, Z_TO_X, (0.0, 0.0, PIN_OFFSET))
    for i, (kind, c, dims) in enumerate(pg.piston_colliders()):
        cb = P_BODY @ np.asarray(c)
        if kind == "cyl":
            _cylinder_collider(stage, f"{piston}/collision/c{i}", cb, dims[0], dims[1])
        else:
            _cube(stage, f"{piston}/collision/c{i}", cb, dims[:3], yaw=-dims[3], collide=True)
    _mesh(stage, f"{rod}/visual/small_end", pg.annulus(pg.SMALL_IN, pg.SMALL_OUT, pg.ROD_T), steel, R_ROD @ Z_TO_X, R_ROD @ (0.0, pg.CTC / 2, 0.0))
    _mesh(stage, f"{rod}/visual/big_end", pg.annulus(pg.BIG_IN, pg.BIG_OUT, pg.ROD_T), steel, R_ROD @ Z_TO_X, R_ROD @ (0.0, -pg.CTC / 2, 0.0))
    for i, (c, sz) in enumerate(pg.rod_beam_boxes()):
        _cube(stage, f"{rod}/visual/beam_{i}", R_ROD @ np.asarray(c), _perm(R_ROD, sz), steel)
    for side in (-1, 1):  # śruby pokrywy łba
        c = (0.0, -pg.CTC / 2, side * (pg.BIG_IN + 0.0035))
        _cube(stage, f"{rod}/visual/bolt_{'ab'[side > 0]}", R_ROD @ np.asarray(c), _perm(R_ROD, (pg.ROD_T + 0.004, 0.012, 0.008)), steel)


def author_pin_and_visuals(stage: Any, env_root: str) -> None:
    author_pin(stage, env_root)
    try:
        author_visuals(stage, env_root)
    except Exception as error:  # bez siatek zostają proste bryły; zadanie działa, tylko wygląda gorzej
        _warn_once("visuals", f"could not build the piston visuals ({type(error).__name__}: {error})")


def author_pin(stage: Any, env_root: str) -> None:
    gf = importlib.import_module("pxr.Gf")
    physics = importlib.import_module("pxr.UsdPhysics")
    joint = physics.RevoluteJoint.Define(stage, f"{env_root}/{PIN_JOINT}")
    joint.CreateAxisAttr("X")  # oś sworznia poprzeczna do osi tłoka; korbowód wahnie się w płaszczyźnie y-z
    joint.CreateBody0Rel().SetTargets([f"{env_root}/{PISTON}"])
    joint.CreateBody1Rel().SetTargets([f"{env_root}/{ROD}"])
    joint.CreateLocalPos0Attr(gf.Vec3f(0.0, 0.0, PIN_OFFSET))
    joint.CreateLocalRot0Attr(gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(gf.Vec3f(0.0, 0.0, -pg.CTC / 2))  # oś otworu stopy korbowodu
    joint.CreateLocalRot1Attr(gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLowerLimitAttr(-PIN_LIMIT_DEG)
    joint.CreateUpperLimitAttr(PIN_LIMIT_DEG)


def author_glue(stage: Any, env_root: str) -> None:
    gf = importlib.import_module("pxr.Gf")
    physics = importlib.import_module("pxr.UsdPhysics")
    joint = physics.FixedJoint.Define(stage, f"{env_root}/{STICK_JOINT}")
    joint.CreateBody0Rel().SetTargets([f"{env_root}/{STICK}"])
    joint.CreateBody1Rel().SetTargets([f"{env_root}/{PAD}"])
    joint.CreateLocalPos0Attr(gf.Vec3f(STICK_TO_PAD, 0.0, 0.0))
    joint.CreateLocalRot0Attr(gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(gf.Quatf(1.0, 0.0, 0.0, 0.0))


def _env_roots(prim: Any) -> list[str]:
    stage = prim.GetStage()
    envs = prim.GetPath().GetParentPath().GetParentPath()
    return [str(e.GetPath()) for e in stage.GetPrimAtPath(envs).GetChildren() if e.GetChild(PISTON).IsValid()]


def _with_pin(spawn, author=None):
    author = author or author_pin_and_visuals

    def spawn_and_pin(prim_path: str, cfg: Any, *args: Any, **kwargs: Any) -> Any:
        prim = spawn(prim_path, cfg, *args, **kwargs)
        try:
            for root in _env_roots(prim):
                author(prim.GetStage(), root)
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
        if name == PAD:  # gąbka budowana po kijku: przyklejamy ją stałym przegubem
            for cfg in _spawner_cfgs(made):
                if not getattr(cfg.func, "piston_pin", False):
                    cfg.func = _with_pin(cfg.func, author_glue)
        if name in (PISTON, ROD):
            for cfg in _spawner_cfgs(made):
                if name == PISTON:
                    pass  # walec stoi pionowo (domyślnie); jego zderzenia wyłącza author_visuals
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
_FIX = np.diag([1.0, -1.0, -1.0])  # układ siatki (denko +z) -> bryła tłoka (denko na dole, -z)
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


_DEBUG = __import__("os").environ.get("PISTON_DEBUG")
_N = [0]


def _debug(objects) -> None:
    """Tylko do strojenia (PISTON_DEBUG=plik): pozy co 10 kroków. Demo ich nie czyta."""
    _N[0] += 1
    if _N[0] % 10:
        return
    with open(_DEBUG, "a") as f:
        row = [f"{_N[0]:4d}"]
        for name in (PISTON, ROD, STICK, PAD):
            o = objects[name]
            row.append(f"{name}=({o.pos[0]:.3f},{o.pos[1]:.3f},{o.pos[2]:.3f}) q=({o.quat_xyzw[0]:.2f},{o.quat_xyzw[1]:.2f},{o.quat_xyzw[2]:.2f},{o.quat_xyzw[3]:.2f})")
        row.append(f"clean={_LEDGER.clean_fraction:.2f} load={_LEDGER.load} zero={_LEDGER.zeroed}")
        f.write(" ".join(row) + chr(10))


def score(objects) -> float:
    _observe(objects)
    if _DEBUG:
        _debug(objects)
    p, rod = objects[PISTON], objects[ROD]
    cx, cy = MAT_CENTER
    on_mat = abs(p.pos[0] - cx) <= MAT_SIZE[0] / 2 and abs(p.pos[1] - cy) <= MAT_SIZE[1] / 2 and p.low[2] <= MAT_TOP + 0.01
    return _LEDGER.score(on_mat=bool(on_mat), resting=p.resting() and rod.resting())


# --- demo (iteracja 1): chwyt korbowodu, podniesienie, obrót; prawa ręka bierze kijek -----------------------------
# Otwarta pętla: pozycje policzone z ustalonego układu startowego, bez odczytu póz z symulatora.
_ROD_TILT = math.asin((PIN[2] - (MAT_TOP + ROD_SIZE[2] / 2)) / pg.CTC)  # korbowód opadnięty na matę
GRIP_D = 0.075  # chwyt 7,5 cm od sworznia: zwisający tłok mija palce
ROD_GRIP = (PIN[0], PIN[1] - GRIP_D * math.cos(_ROD_TILT), PIN[2] - GRIP_D * math.sin(_ROD_TILT))
STICK_GRIP = (STICK_POS[0] - 0.03, STICK_POS[1], STICK_POS[2])


def _above(p, dz):
    return (p[0], p[1], p[2] + dz)


def _turned_grip(pin_xy, z, phi):
    """Gdzie trzymać korbowód, żeby po obrocie o phi sworzeń został nad tym samym punktem."""
    d = GRIP_D * math.cos(_ROD_TILT)
    return (pin_xy[0] + d * math.sin(phi), pin_xy[1] - d * math.cos(phi), z)


def demo() -> dict:
    """Na razie postój: obserwacja, jak zespół ustala się po starcie (korbowód opada na krawędź płaszcza)."""
    hold = lambda y: [Waypoint(4.0, (0.22, y, TOP + 0.30), FINGER_OPEN, "hold above the table")]  # noqa: E731
    return {"left": hold(0.25), "right": hold(-0.25)}
