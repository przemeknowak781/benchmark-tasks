"""Testy rdzenia punktacji: pełne czyszczenie daje 1.0, exploity ze specyfikacji dają mniej.

Uruchomienie: python -m pytest -q tasks/piston-clean/test_piston_scoring.py
"""

import math

import numpy as np
import pytest

import piston_scoring as ps

TRAY = np.array([0.30, 0.20, 0.0])
TRAY_FLOOR = 0.0
STILL = np.zeros(3)
PAD_SIDE_Q = np.array([0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)])  # cienką osią wzdłuż x
FLAT_Q = np.array([0.0, 0.0, 0.0, 1.0])
PISTON_POS = np.array([0.0, 0.0, 0.10])


def yaw_q(a):
    return np.array([0.0, 0.0, math.sin(a / 2), math.cos(a / 2)])


def bbox(pos):
    h = np.array([ps.PISTON_D / 2, ps.PISTON_D / 2, ps.PISTON_H / 2])
    return pos - h, pos + h


def step(ledger, pad_pos, pad_q, piston_q=FLAT_Q, piston_pos=PISTON_POS, vel=STILL):
    lo, hi = bbox(piston_pos)
    ledger.step(piston_pos, piston_q, vel, lo, hi, pad_pos, pad_q, TRAY, TRAY_FLOOR)


def rinse(ledger):
    step(ledger, TRAY + np.array([0, 0, 0.005]), FLAT_Q)


def side_pad_pos(dz=0.0):
    return PISTON_POS + np.array([ps.PISTON_D / 2 + (ps.PAD_SIZE[2] / 2 - ps.SOFT_LAYER), 0.0, dz])


def wipe_everything(ledger, rinse_when_loaded=True):
    for k in range(ps.N_AROUND):
        if rinse_when_loaded and ledger.load > ps.CAPACITY - 20:
            rinse(ledger)
        for dz in (-0.012, 0.006):  # gąbka 4 cm: dwa pasy wzdłuż płaszcza
            if rinse_when_loaded and ledger.load > ps.CAPACITY - 12:
                rinse(ledger)
            step(ledger, side_pad_pos(dz), PAD_SIDE_Q, piston_q=yaw_q(k * 2 * math.pi / ps.N_AROUND))
    for dx, dy in [(a, b) for a in (-0.02, 0.0, 0.02) for b in (-0.02, 0.0, 0.02)]:
        if rinse_when_loaded and ledger.load > ps.CAPACITY - 16:
            rinse(ledger)
        top = PISTON_POS + np.array([dx, dy, ps.PISTON_H / 2 + (ps.PAD_SIZE[2] / 2 - ps.SOFT_LAYER) - 0.001])  # gąbka wciśnięta 1 mm
        step(ledger, top, FLAT_Q)


def wipe_inside(ledger):
    """Gąbka wewnątrz płaszcza: ścianka dookoła i spód denka."""
    for k in range(ps.N_AROUND):
        a = 2 * math.pi * k / ps.N_AROUND
        for dz in (-0.014, -0.004):
            if ledger.load > ps.CAPACITY - 12:
                rinse(ledger)
            c = PISTON_POS + np.array([(ps.INNER_R - (ps.PAD_SIZE[0] / 2 - ps.SOFT_LAYER) + 0.001) * math.cos(a), (ps.INNER_R - (ps.PAD_SIZE[0] / 2 - ps.SOFT_LAYER) + 0.001) * math.sin(a), dz])
            step(ledger, c, yaw_q(a))
    for dx, dy in [(a, b) for a in (-0.012, 0.012) for b in (-0.012, 0.012)]:
        if ledger.load > ps.CAPACITY - 16:
            rinse(ledger)
        step(ledger, PISTON_POS + np.array([dx, dy, ps.UNDERCROWN_Z - (ps.PAD_SIZE[2] / 2 - ps.SOFT_LAYER) + 0.001]), FLAT_Q)


def test_outside_wipe_does_not_clean_inside():
    led = ps.Ledger()
    wipe_everything(led)
    inner = led.dirty[ps.N_SKIRT + ps.N_CROWN :]
    assert inner.all()  # przez ściankę się nie czyści


def test_grid_size_matches_spec():
    assert ps.N_SKIRT == 100
    assert 38 <= ps.N_CROWN <= 50  # denko 75 mm: ok. 44 cm2
    assert ps.CAPACITY < ps.N_CELLS * 0.95 / 3  # pełny wynik wymaga co najmniej 3 płukań
    assert 60 <= ps.N_INNER <= 90


def test_full_clean_with_rinses_scores_one():
    led = ps.Ledger()
    wipe_everything(led)
    wipe_inside(led)
    assert led.clean_fraction >= 0.95
    assert led.rinses >= 3
    assert led.score(on_mat=True, resting=True) == 1.0


def test_still_in_gripper_caps_below_one():
    led = ps.Ledger()
    wipe_everything(led)
    wipe_inside(led)
    assert led.score(on_mat=True, resting=False) == 0.75


def test_never_rinsing_stalls_and_smears():
    led = ps.Ledger()
    wipe_everything(led, rinse_when_loaded=False)
    assert led.rinses == 0
    assert led.clean_fraction < 0.25  # szmatka nasycona rozmazała z powrotem
    assert led.score(True, True) == 0.0


def test_camera_side_only_is_partial():
    led = ps.Ledger()
    for k in range(ps.N_AROUND // 2):
        if led.load > ps.CAPACITY - 20:
            rinse(led)
        step(led, side_pad_pos(), PAD_SIDE_Q, piston_q=yaw_q(k * 2 * math.pi / ps.N_AROUND))
    assert led.score(True, True) <= 0.25


def test_hovering_sponge_cleans_nothing():
    led = ps.Ledger()
    hover = PISTON_POS + np.array([ps.PISTON_D / 2 + ps.PAD_SIZE[2] / 2 + 0.002, 0.0, 0.0])  # 2 mm przed płaszczem
    step(led, hover, PAD_SIDE_Q)
    assert led.clean_fraction == 0.0


def test_no_pad_contact_cleans_nothing():
    led = ps.Ledger()
    far = PISTON_POS + np.array([0.2, 0.0, 0.0])  # chwytak przy tłoku, szmatka daleko
    step(led, far, FLAT_Q)
    assert led.clean_fraction == 0.0


def test_drop_zeroes_whole_episode():
    led = ps.Ledger()
    step(led, side_pad_pos(), PAD_SIDE_Q, vel=np.array([0, 0, -0.9]))
    wipe_everything(led)
    assert led.zeroed == "upuszczony tłok"
    assert led.score(True, True) == 0.0


def test_gentle_set_down_is_not_a_drop():
    led = ps.Ledger()
    step(led, side_pad_pos(), PAD_SIDE_Q, vel=np.array([0, 0, -0.3]))
    assert led.zeroed is None


def test_dunking_in_tray_zeroes():
    led = ps.Ledger()
    step(led, side_pad_pos(), PAD_SIDE_Q, piston_pos=TRAY + np.array([0, 0, 0.02]))
    assert led.zeroed == "tłok w rozpuszczalniku"


def test_tray_too_narrow_to_immerse_piston():
    assert ps.TRAY_INNER[1] < ps.PISTON_D


def test_off_mat_zeroes():
    led = ps.Ledger()
    wipe_everything(led)
    assert led.score(on_mat=False, resting=True) == 0.0


@pytest.mark.parametrize("frac,expected", [(0.24, 0.0), (0.25, 0.25), (0.5, 0.5), (0.8, 0.75), (0.95, 1.0)])
def test_ladder(frac, expected):
    assert ps.ladder(frac) == expected
