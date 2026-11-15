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
MAT_CENTER = (0.30, -0.02)
MAT_SIZE = (0.30, 0.30, 0.004)
MAT_TOP = TOP + MAT_SIZE[2]

PISTON = "piston"
ROD = "conrod"
STICK = "sponge_stick"
PAD = "sponge"
FLAT = "flat_sponge"
PISTON_R = ps.PISTON_D / 2
PISTON_POS = (0.30, 0.02, MAT_TOP + ps.PISTON_H / 2)  # tłok stoi denkiem na macie, płaszcz otwarty do góry
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
SOFT = 0.005  # miękka warstwa gąbki (mm-y ugięcia): zderza się rdzeń, punktacja liczy całą gąbkę 4 cm
PAD_CORE = tuple(s - 2 * SOFT for s in ps.PAD_SIZE)
PAD_POS = (0.33, 0.30, TOP + PAD_CORE[2] / 2 + 0.001)
PAD_MASS = 0.003  # gąbka PU ~0,05 g/cm3
STICK_TO_PAD = STICK_SIZE[0] / 2 + ps.PAD_SIZE[0] / 2  # środek kijka -> środek gąbki, wzdłuż x
STICK_DZ = ps.PAD_SIZE[2] / 2 - STICK_SIZE[2] / 2  # kijek przy górze gąbki: gąbka wchodzi w płaszcz, kijek nad brzegiem
STICK_POS = (PAD_POS[0] - STICK_TO_PAD, PAD_POS[1], PAD_POS[2] + STICK_DZ)
REST_SIZE = (0.03, 0.03, STICK_POS[2] - STICK_SIZE[2] / 2 - TOP)  # klocek pod bliższym końcem trzonka
REST_POS = (STICK_POS[0] - STICK_SIZE[0] / 2 + 0.02, STICK_POS[1], TOP + REST_SIZE[2] / 2)
STICK_JOINT = "sponge_glue"

TRAY_CENTER = (0.36, 0.20)
FLAT_CORE = tuple(v - 2 * SOFT for v in ps.FLAT_PAD_SIZE)
FLAT_POS = (0.24, -0.26, TOP + FLAT_CORE[2] / 2 + 0.001)  # płaska gąbka przed prawą ręką
FLAT_MASS = 0.006
TRAY_WALL = 0.004
TRAY_FLOOR_Z = TOP + 0.002  # dno wanienki (2 mm blachy)

STEEL = (0.62, 0.64, 0.66)
ALU = (0.78, 0.78, 0.76)
OILY = (0.30, 0.20, 0.08)  # ciemnobrązowy przepracowany olej: łatki na brudnych komórkach
ALU = (0.74, 0.74, 0.71)  # czyste aluminium odlewu
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
    Box(PAD, PAD_POS, SPONGE_RGB, mass=PAD_MASS, size=PAD_CORE),
    Box(FLAT, FLAT_POS, SPONGE_RGB, mass=FLAT_MASS, size=FLAT_CORE),
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
    alu = _material(stage, f"{piston}/visual/alu", ALU, 0.6, 0.35)
    oil = _material(stage, f"{piston}/visual/oil", OILY, 0.1, 0.15)  # połyskliwy olej
    _author_dirt(stage, piston, oil)
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


_DIRT_PATHS: dict[str, list[str]] = {}  # korzeń środowiska -> ścieżki łatek oleju (kolejność komórek)
_SHOWN: dict[str, np.ndarray] = {}


def _author_dirt(stage, piston, mat) -> None:
    """Na każdej komórce kwadratowa łatka oleju 0,3 mm nad powierzchnią; znika, gdy komórka jest czysta."""
    half = ps.CELL * 0.48
    cells = ps.CELLS_LOCAL @ P_BODY.T
    normals = ps.NORMALS_LOCAL @ P_BODY.T
    paths = []
    for i, (c, n) in enumerate(zip(cells, normals)):
        n = n / np.linalg.norm(n)
        a = np.cross(n, [0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        o = c + n * 0.0003
        quad = np.array([o - half * a - half * b, o + half * a - half * b, o + half * a + half * b, o - half * a + half * b])
        path = f"{piston}/visual/dirt/c{i:03d}"
        _mesh(stage, path, (quad, np.array([4]), np.arange(4)), mat)
        paths.append(path)
    root = piston.rsplit("/", 1)[0]
    _DIRT_PATHS[root] = paths
    _SHOWN[root] = np.ones(len(paths), dtype=bool)


def _show_dirt(dirty: np.ndarray) -> None:
    """Pokazuje stan oleju na tłoku (tylko zmienione łatki). Bez symulatora — nic."""
    if not _DIRT_PATHS:
        return
    try:
        stage = importlib.import_module("omni.usd").get_context().get_stage()
        geom = importlib.import_module("pxr.UsdGeom")
    except Exception:
        return
    for root, paths in _DIRT_PATHS.items():
        changed = np.flatnonzero(_SHOWN[root] != dirty)
        for i in changed:
            prim = stage.GetPrimAtPath(paths[i])
            if prim.IsValid():
                img = geom.Imageable(prim)
                img.MakeVisible() if dirty[i] else img.MakeInvisible()
        _SHOWN[root] = dirty.copy()


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
    pad = f"{env_root}/{PAD}"
    _retire_primitive(stage, pad)  # rdzeń niewidoczny; widać pełną gąbkę
    foam = _material(stage, f"{pad}/visual/foam", SPONGE_RGB, 0.0, 0.9)
    _cube(stage, f"{pad}/visual/foam_block", (0.0, 0.0, 0.0), ps.PAD_SIZE, foam)
    _author_glue_joint(stage, env_root)


def author_flat(stage: Any, env_root: str) -> None:
    flat = f"{env_root}/{FLAT}"
    _retire_primitive(stage, flat)
    foam = _material(stage, f"{flat}/visual/foam", SPONGE_RGB, 0.0, 0.9)
    _cube(stage, f"{flat}/visual/foam_block", (0.0, 0.0, 0.0), ps.FLAT_PAD_SIZE, foam)


def _author_glue_joint(stage: Any, env_root: str) -> None:
    gf = importlib.import_module("pxr.Gf")
    physics = importlib.import_module("pxr.UsdPhysics")
    joint = physics.FixedJoint.Define(stage, f"{env_root}/{STICK_JOINT}")
    joint.CreateBody0Rel().SetTargets([f"{env_root}/{STICK}"])
    joint.CreateBody1Rel().SetTargets([f"{env_root}/{PAD}"])
    joint.CreateLocalPos0Attr(gf.Vec3f(STICK_TO_PAD, 0.0, -STICK_DZ))
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
        if name == FLAT:
            for cfg in _spawner_cfgs(made):
                if not getattr(cfg.func, "piston_pin", False):
                    cfg.func = _with_pin(cfg.func, author_flat)
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
        objects[FLAT].pos, objects[FLAT].quat_xyzw,
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
        for name in (PISTON, ROD, PAD, FLAT):
            o = objects[name]
            row.append(f"{name}=({o.pos[0]:.3f},{o.pos[1]:.3f},{o.pos[2]:.3f}) q=({o.quat_xyzw[0]:.2f},{o.quat_xyzw[1]:.2f},{o.quat_xyzw[2]:.2f},{o.quat_xyzw[3]:.2f})")
        d = _LEDGER.dirty
        g = (ps.N_SKIRT, ps.N_SKIRT + ps.N_CROWN)
        row.append(f"clean={_LEDGER.clean_fraction:.2f} dirty skirt/crown/inner={d[:g[0]].sum()}/{d[g[0]:g[1]].sum()}/{d[g[1]:].sum()} load={_LEDGER.load} rinses={_LEDGER.rinses} zero={_LEDGER.zeroed}")
        f.write(" ".join(row) + chr(10))


def score(objects) -> float:
    _observe(objects)
    _show_dirt(_LEDGER.dirty)
    if _DEBUG:
        _debug(objects)
    p, rod = objects[PISTON], objects[ROD]
    cx, cy = MAT_CENTER
    on_mat = abs(p.pos[0] - cx) <= MAT_SIZE[0] / 2 and abs(p.pos[1] - cy) <= MAT_SIZE[1] / 2 and p.low[2] <= MAT_TOP + 0.01
    return _LEDGER.score(on_mat=bool(on_mat), resting=p.resting() and rod.resting())


# --- demo (iteracja 1): chwyt korbowodu, podniesienie, obrót; prawa ręka bierze kijek -----------------------------
# Otwarta pętla: pozycje policzone z ustalonego układu startowego, bez odczytu póz z symulatora.
# Położenie po przewróceniu się zespołu (ustalone w symulacji przy object_jitter=0, kalibracja z zapisu póz;
# skrypt ich nie czyta): korbowód leży na wąskiej krawędzi wzdłuż -y, łeb w stronę prawej ręki.
ROD_REST = (0.297, PISTON_POS[1] - 0.114, MAT_TOP + 0.0435)  # środek bryły korbowodu
ROD_GRIP = (ROD_REST[0], ROD_REST[1] - 0.04, ROD_REST[2] + 0.005)  # 4 cm od środka w stronę łba
STICK_GRIP = (STICK_POS[0] - 0.03, STICK_POS[1], STICK_POS[2])
YAW_ACROSS_X = math.pi / 2  # palce zamykają się wzdłuż x (w poprzek grubości korbowodu)


def _above(p, dz):
    return (p[0], p[1], p[2] + dz)


# Pozy zwisającego tłoka po podniesieniu (kalibracja przy object_jitter=0): środek i obrót wokół pionu.
HANG = {  # yaw nadgarstka prawej ręki (wzgl. chwytu) -> (środek tłoka, obrót tłoka wokół pionu)
    0: ((0.289, -0.070, 0.832), math.radians(3.4)),
    -90: ((0.358, -0.132, 0.830), math.radians(-87.7)),
    90: ((0.237, -0.099, 0.836), math.radians(58.7)),
}
L_PAD = STICK_TO_PAD + 0.03  # TCP -> środek gąbki wzdłuż kijka
GAP = -0.006  # rdzeń planowo 6 mm w płaszczu: ramię z kijkiem ugina się ~7 mm (zmierzone), realnie styk z dociskiem
SAG_Z = 0.006  # kijek z gąbką opada ~6 mm względem planu (zmierzone)
SKIRT_BAND_Z = (ps.PISTON_H / 2 - ps.RING_BELT) / 2 + 0.0  # środek pasa płaszcza nad środkiem bryły (denko w dół)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _tcp_for_pad(pad, stick_dir):
    """TCP lewej ręki dla środka gąbki `pad`, gdy kijek wskazuje (od rączki do gąbki) kierunek stick_dir [rad]."""
    return (pad[0] - L_PAD * math.cos(stick_dir), pad[1] - L_PAD * math.sin(stick_dir), pad[2] + STICK_DZ)


def _side_wipe(centre, psi, t=1.0):
    """Gąbka dochodzi do płaszcza od azymutu psi (świat), czołem do osi tłoka."""
    stick_dir = _wrap(psi + math.pi)
    r_in = ps.PISTON_D / 2 + PAD_CORE[0] / 2 + GAP  # rdzeń 0,5 mm od płaszcza, miękka warstwa wciśnięta
    z = centre[2] + SKIRT_BAND_Z + SAG_Z
    near = (centre[0] + r_in * math.cos(psi), centre[1] + r_in * math.sin(psi), z)
    far = (centre[0] + (r_in + 0.05) * math.cos(psi), centre[1] + (r_in + 0.05) * math.sin(psi), z)
    return [
        Waypoint(1.5 * t, _tcp_for_pad(far, stick_dir), FINGER_CLOSED, f"approach {math.degrees(psi):.0f}", stick_dir),
        Waypoint(1.0 * t, _tcp_for_pad(near, stick_dir), FINGER_CLOSED, "wipe", stick_dir),
        Waypoint(0.5 * t, _tcp_for_pad(near, stick_dir), FINGER_CLOSED, "hold", stick_dir),
        Waypoint(0.8 * t, _tcp_for_pad(far, stick_dir), FINGER_CLOSED, "back off", stick_dir),
    ]


def _rinse():
    pad = (TRAY_CENTER[0], TRAY_CENTER[1], TRAY_FLOOR_Z + ps.PAD_SIZE[2] / 2 + 0.003)
    above = (pad[0], pad[1], pad[2] + 0.08)
    return [
        Waypoint(2.0, _tcp_for_pad(above, 0.0), FINGER_CLOSED, "over the tray", 0.0),
        Waypoint(1.0, _tcp_for_pad(pad, 0.0), FINGER_CLOSED, "dip", 0.0),
        Waypoint(0.6, _tcp_for_pad(pad, 0.0), FINGER_CLOSED, "rinse", 0.0),
        Waypoint(1.0, _tcp_for_pad(above, 0.0), FINGER_CLOSED, "lift out", 0.0),
    ]


PHASES = ((0, (50, 90, 130, 170, 190)), (-90, (130, 170, 190)), (90, (20, 60, 100)))  # yaw prawej -> azymuty gąbki
SHIFT = {0: (0.0, 0.0), -90: (0.0, 0.0), 90: (-0.07, 0.0)}  # przesunięcie punktu uniesienia w fazie (bliżej lewej ręki)


def hang(yaw_deg):
    (cx, cy, cz), rot = HANG[yaw_deg]
    dx, dy = SHIFT[yaw_deg]
    return (cx + dx, cy + dy, cz), rot


def _dur(ws):
    return sum(w.duration for w in ws)


def demo() -> dict:
    lift = (ROD_GRIP[0], ROD_GRIP[1], TOP + 0.20)
    right = [
        Waypoint(2.0, _above(ROD_GRIP, 0.08), FINGER_OPEN, "above the rod", YAW_ACROSS_X),
        Waypoint(1.5, ROD_GRIP, FINGER_OPEN, "descend to the rod", YAW_ACROSS_X),
        Waypoint(0.8, ROD_GRIP, FINGER_CLOSED, "grip the rod", YAW_ACROSS_X),
        Waypoint(2.5, lift, FINGER_CLOSED, "lift", YAW_ACROSS_X),
    ]
    left = [
        Waypoint(2.0, _above(STICK_GRIP, 0.08), FINGER_OPEN, "above the handle", 0.0),
        Waypoint(1.5, STICK_GRIP, FINGER_OPEN, "descend to the handle", 0.0),
        Waypoint(0.8, STICK_GRIP, FINGER_CLOSED, "grip the handle", 0.0),
        Waypoint(1.5, _above(STICK_GRIP, 0.12), FINGER_CLOSED, "lift the sponge", 0.0),
        Waypoint(1.5, _above(STICK_GRIP, 0.12), FINGER_CLOSED, "wait for the piston", 0.0),
    ]
    park = _above(STICK_GRIP, 0.12)
    touches = 0
    for k, (yaw_deg, psis) in enumerate(PHASES):
        if k:  # obrót tłoka: lewa ręka odsuwa gąbkę, prawa obraca nadgarstek
            left.append(Waypoint(1.5, park, FINGER_CLOSED, "park the sponge", 0.0))
            hold = _dur(left) - _dur(right)
            right.append(Waypoint(max(hold, 0.1), right[-1].pos, FINGER_CLOSED, "hold", right[-1].yaw))
            dx, dy = SHIFT[yaw_deg]
            moved = (lift[0] + dx, lift[1] + dy, lift[2])
            right.append(Waypoint(2.5, moved, FINGER_CLOSED, f"turn {yaw_deg}", YAW_ACROSS_X + math.radians(yaw_deg)))
            left.append(Waypoint(_dur(right) - _dur(left) + 1.0, park, FINGER_CLOSED, "wait for the turn", 0.0))
        centre, _rot = hang(yaw_deg)
        for deg in psis:
            if touches and touches % 2 == 0:
                left += _rinse()
            left += _side_wipe(centre, math.radians(deg))
            touches += 1
    left.append(Waypoint(1.5, park, FINGER_CLOSED, "park the sponge", 0.0))
    right.append(Waypoint(max(_dur(left) - _dur(right), 0.1), right[-1].pos, FINGER_CLOSED, "hold", right[-1].yaw))
    # denko: z powrotem do yaw 0, nad płaską gąbkę, docisk, krótkie przetarcie
    off = np.asarray(HANG[0][0]) - np.asarray(lift)  # TCP -> środek tłoka przy yaw 0
    crown_z = FLAT_POS[2] + FLAT_CORE[2] / 2 + ps.PISTON_H / 2 - 0.002  # rdzeń gąbki wciśnięty 2 mm
    over = np.array([FLAT_POS[0], FLAT_POS[1], crown_z + 0.06]) - off
    press = np.array([FLAT_POS[0], FLAT_POS[1], crown_z]) - off
    right += [
        Waypoint(2.5, lift, FINGER_CLOSED, "turn 0", YAW_ACROSS_X),
        Waypoint(2.5, tuple(over), FINGER_CLOSED, "over the flat sponge", YAW_ACROSS_X),
        Waypoint(1.5, tuple(press), FINGER_CLOSED, "press crown", YAW_ACROSS_X),
        Waypoint(1.0, tuple(press + (0.01, 0.0, 0.0)), FINGER_CLOSED, "press crown", YAW_ACROSS_X),
        Waypoint(1.0, tuple(press - (0.01, 0.0, 0.0)), FINGER_CLOSED, "press crown", YAW_ACROSS_X),
        Waypoint(1.5, tuple(over), FINGER_CLOSED, "lift off the sponge", YAW_ACROSS_X),
    ]
    # odłożenie na matę denkiem w dół i puszczenie (zespół sam się położy, jak na starcie)
    rest = np.array([MAT_CENTER[0], MAT_CENTER[1], MAT_TOP + ps.PISTON_H / 2 + 0.004]) - off
    right += [
        Waypoint(2.5, tuple(rest + (0, 0, 0.05)), FINGER_CLOSED, "over the mat", YAW_ACROSS_X),
        Waypoint(1.5, tuple(rest), FINGER_CLOSED, "set down", YAW_ACROSS_X),
        Waypoint(0.8, tuple(rest), FINGER_OPEN, "release", YAW_ACROSS_X),
        Waypoint(1.5, tuple(rest + (0, 0, 0.04)), FINGER_OPEN, "clear", YAW_ACROSS_X),
        Waypoint(3.0, tuple(rest + (0, 0, 0.04)), FINGER_OPEN, "wait still", YAW_ACROSS_X),
    ]
    left.append(Waypoint(max(_dur(right) - _dur(left), 0.1), park, FINGER_CLOSED, "wait", 0.0))
    return {"left": left, "right": right}
