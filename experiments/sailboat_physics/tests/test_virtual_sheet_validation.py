from __future__ import annotations

import math

import pytest

from experiments.sailboat_physics.virtual_sheet_validation import (
    BOUNDED_TRANSIENT,
    DEGRADED,
    FRAMEWORK_V1,
    FRAMEWORK_V2,
    INWARD_REVERSAL,
    OPEN,
    RIGHT_CENSORED,
    VALID_SHEET_OUT_RELEASE,
    VirtualSheetSample,
    evaluate_compliance,
)


REFERENCE = math.radians(15.0)


def sample(
    time_s: float,
    penetration_deg: float,
    *,
    raw_deg: float = 16.605,
    effective_deg: float = 16.605,
    boom_deg: float = -31.605,
    boom_velocity_rad_s: float = -0.60,
    sheet_out_release: bool = False,
    tightening: bool = False,
    torque_capped: bool = False,
) -> VirtualSheetSample:
    penetration_rad = math.radians(penetration_deg)
    raw_rad = math.radians(raw_deg)
    effective_rad = math.radians(effective_deg)
    side = -1.0 if boom_deg < 0.0 else 1.0
    applied_torque = -side * 135.0 if penetration_rad > 0.0 else 0.0

    return VirtualSheetSample(
        sim_time_s=time_s,
        raw_allowance_rad=raw_rad,
        effective_allowance_rad=effective_rad,
        allowance_lag_rad=effective_rad - raw_rad,
        boom_angle_rad=math.radians(boom_deg),
        boom_velocity_rad_s=boom_velocity_rad_s,
        penetration_rad=penetration_rad,
        requested_torque_nm=abs(applied_torque),
        applied_torque_nm=applied_torque,
        sheet_out_release=sheet_out_release,
        tightening=tightening,
        torque_capped=torque_capped,
    )


def five_millisecond_sheet_out_recovery() -> list[VirtualSheetSample]:
    """Reproduce the 54.045--54.050 s valid sheet-out recovery shape."""

    return [
        sample(54.044, 14.980, boom_deg=-31.585),
        sample(54.045, 15.014, boom_deg=-31.619),
        sample(54.046, 15.049, boom_deg=-31.654),
        sample(54.047, 15.083, boom_deg=-31.688),
        sample(54.048, 15.116, boom_deg=-31.721),
        sample(
            54.049,
            15.149,
            boom_deg=-31.754,
            boom_velocity_rad_s=-0.584,
        ),
        sample(
            54.050,
            14.733,
            raw_deg=17.055,
            effective_deg=17.055,
            boom_deg=-31.788,
            boom_velocity_rad_s=-0.580,
            sheet_out_release=True,
        ),
    ]


def test_v1_is_preserved_and_rejects_recovery_without_inward_reversal():
    result = evaluate_compliance(
        five_millisecond_sheet_out_recovery(),
        framework_version=FRAMEWORK_V1,
    )

    assert result.classification == DEGRADED
    assert result.inward_reversal_count == 0
    assert result.sheet_out_recovery_count == 0


def test_v2_accepts_valid_five_millisecond_sheet_out_recovery():
    result = evaluate_compliance(
        five_millisecond_sheet_out_recovery(),
        framework_version=FRAMEWORK_V2,
    )

    assert result.classification == BOUNDED_TRANSIENT
    assert result.episode_count == 1
    assert result.inward_reversal_count == 0
    assert result.sheet_out_recovery_count == 1
    assert result.episodes[0].recovery_mechanism == VALID_SHEET_OUT_RELEASE
    assert result.episodes[0].duration_s == pytest.approx(0.005)
    assert math.degrees(result.episodes[0].peak_penetration_rad) == pytest.approx(
        15.149
    )


def test_v2_keeps_inward_reversal_as_a_valid_recovery_mechanism():
    samples = [
        sample(1.000, 14.9, boom_velocity_rad_s=0.20),
        sample(1.001, 15.1, boom_velocity_rad_s=-0.20),
        sample(1.002, 15.2, boom_velocity_rad_s=-0.05),
        sample(1.003, 15.0, boom_velocity_rad_s=0.10),
    ]

    result = evaluate_compliance(samples, framework_version=FRAMEWORK_V2)

    assert result.classification == BOUNDED_TRANSIENT
    assert result.episodes[0].recovery_mechanism == INWARD_REVERSAL


@pytest.mark.parametrize(
    "release,raw_deg,effective_deg",
    [
        (False, 17.055, 17.055),
        (True, 17.055, 16.605),
        (True, 17.055, 17.200),
    ],
)
def test_v2_rejects_invalid_or_noncontractual_boundary_recovery(
    release: bool,
    raw_deg: float,
    effective_deg: float,
):
    samples = five_millisecond_sheet_out_recovery()
    samples[-1] = sample(
        54.050,
        14.733,
        raw_deg=raw_deg,
        effective_deg=effective_deg,
        boom_deg=-31.788,
        boom_velocity_rad_s=-0.580,
        sheet_out_release=release,
    )

    result = evaluate_compliance(samples, framework_version=FRAMEWORK_V2)

    assert result.classification == DEGRADED
    assert result.sheet_out_recovery_count == 0


def test_v2_keeps_right_censored_episode_open():
    samples = [
        sample(2.000, 14.9),
        sample(2.001, 15.1),
        sample(2.002, 15.2),
    ]

    result = evaluate_compliance(samples, framework_version=FRAMEWORK_V2)

    assert result.classification == OPEN
    assert result.unrecovered_episode_count == 1
    assert result.episodes[0].recovery_mechanism == RIGHT_CENSORED


def test_framework_version_must_be_selected_explicitly():
    with pytest.raises(ValueError, match="framework_version"):
        evaluate_compliance(
            five_millisecond_sheet_out_recovery(),
            framework_version="latest",
        )
