"""Unit tests for the two separated coin-delivery certificates."""
from __future__ import annotations

import pytest

from hymeko_rl.coin_delivery.delivery_certificate import (
    COIN_DELIVERY_STRICT, COIN_GRASP_DELIVERY_STRICT, CertStep, DeliveryCertifier, DeliveryThresholds)


def _push_coast_step(**kw) -> CertStep:
    """A centered+settled step delivered by a ONE-finger coast (right fingertip only, no bilateral force)."""
    base = dict(disk_to_zone=0.019, disk_speed=0.02, left_fingertip=False, right_fingertip=True,
                arm_body_contact=False, arm_body_impulse=0.0, force_left=0.0, force_right=0.05)
    base.update(kw)
    return CertStep(**base)


def test_push_coast_delivers_but_is_not_a_grasp() -> None:
    """A robot-attributed push/coast with positive clearance passes DELIVERY, never GRASP (the E0 case)."""
    c = DeliveryCertifier(initial_clearance=0.0386)
    for _ in range(6):
        c.update(_push_coast_step())
    assert c.satisfies(COIN_DELIVERY_STRICT)
    assert not c.satisfies(COIN_GRASP_DELIVERY_STRICT)
    assert c.best_delivery_dwell == 6 and c.best_grasp_dwell == 0


def test_negative_clearance_is_not_a_delivery() -> None:
    """Overlapping initial footprints (coin already in/under the zone) can never certify DELIVERY."""
    c = DeliveryCertifier(initial_clearance=-0.03)
    for _ in range(8):
        c.update(_push_coast_step())
    assert not c.satisfies(COIN_DELIVERY_STRICT)


def test_no_robot_contact_is_not_attributed() -> None:
    """A coin that settles in the zone with the robot never touching it fails robot-attribution."""
    c = DeliveryCertifier(initial_clearance=0.04)
    for _ in range(8):
        c.update(_push_coast_step(left_fingertip=False, right_fingertip=False))
    assert not c.satisfies(COIN_DELIVERY_STRICT)
    assert not c.robot_touched


def test_body_shove_breaks_clean_mechanism() -> None:
    """A forbidden arm-body contact during the dwell resets the delivery counter (dirty mechanism)."""
    c = DeliveryCertifier(initial_clearance=0.04)
    for i in range(6):
        c.update(_push_coast_step(arm_body_contact=(i == 3)))
    assert not c.satisfies(COIN_DELIVERY_STRICT)
    assert c.best_delivery_dwell == 3        # counter reset at the shove


def test_too_fast_does_not_settle() -> None:
    c = DeliveryCertifier(initial_clearance=0.04)
    for _ in range(8):
        c.update(_push_coast_step(disk_speed=0.09))
    assert not c.satisfies(COIN_DELIVERY_STRICT)
    assert c.best_delivery_dwell == 0


def test_bilateral_force_certifies_grasp_and_implies_delivery() -> None:
    """Sustained bilateral force closure certifies BOTH grasp and (a fortiori) delivery."""
    c = DeliveryCertifier(initial_clearance=0.05)
    grasp = dict(disk_to_zone=0.015, disk_speed=0.01, left_fingertip=True, right_fingertip=True,
                 arm_body_contact=False, arm_body_impulse=0.0, force_left=1.0, force_right=1.0)
    for _ in range(6):
        c.update(CertStep(**grasp))
    assert c.satisfies(COIN_GRASP_DELIVERY_STRICT)
    assert c.satisfies(COIN_DELIVERY_STRICT)


def test_unknown_certificate_raises() -> None:
    c = DeliveryCertifier(initial_clearance=0.04)
    with pytest.raises(ValueError):
        c.satisfies("NONSENSE")


def test_thresholds_are_configurable() -> None:
    th = DeliveryThresholds(dwell_req=3)
    c = DeliveryCertifier(initial_clearance=0.04, th=th)
    for _ in range(3):
        c.update(_push_coast_step())
    assert c.satisfies(COIN_DELIVERY_STRICT)
