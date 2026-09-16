"""Pure helpers for the fixed-heading ArduPilot GUIDED contract."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


# ArduPilot output functions that can command propulsion motors.  The polar
# experiment is deliberately wind-only, so none may be assigned.
MOTOR_SERVO_FUNCTIONS = frozenset({33, 34, 35, 36, 37, 38, 39, 40, 70, 73, 74})


def local_ned_velocity(
    heading_deg: float,
    desired_speed_mps: float,
) -> tuple[float, float]:
    """Return north/east velocity components for a compass heading."""

    speed = float(desired_speed_mps)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("desired_speed_mps must be finite and positive")
    heading = float(heading_deg)
    if not math.isfinite(heading):
        raise ValueError("heading_deg must be finite")
    radians = math.radians(heading)
    return speed * math.cos(radians), speed * math.sin(radians)


def guided_velocity_yaw_type_mask(mavlink: Any) -> int:
    """Use only vx, vy and yaw in SET_POSITION_TARGET_LOCAL_NED."""

    ignored = (
        "POSITION_TARGET_TYPEMASK_X_IGNORE",
        "POSITION_TARGET_TYPEMASK_Y_IGNORE",
        "POSITION_TARGET_TYPEMASK_Z_IGNORE",
        "POSITION_TARGET_TYPEMASK_VZ_IGNORE",
        "POSITION_TARGET_TYPEMASK_AX_IGNORE",
        "POSITION_TARGET_TYPEMASK_AY_IGNORE",
        "POSITION_TARGET_TYPEMASK_AZ_IGNORE",
        "POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE",
    )
    try:
        return sum(int(getattr(mavlink, name)) for name in ignored)
    except AttributeError as exc:
        raise ValueError(f"MAVLink module lacks required type-mask constant: {exc}") from exc


@dataclass
class GuidedHeadingTarget:
    """Periodically refresh one fixed velocity-heading target."""

    master: Any
    mavlink: Any
    heading_deg: float
    desired_speed_mps: float
    refresh_hz: float = 2.0

    def __post_init__(self) -> None:
        self.velocity_north_mps, self.velocity_east_mps = local_ned_velocity(
            self.heading_deg,
            self.desired_speed_mps,
        )
        if not math.isfinite(float(self.refresh_hz)) or self.refresh_hz <= 0.0:
            raise ValueError("refresh_hz must be finite and positive")
        self.type_mask = guided_velocity_yaw_type_mask(self.mavlink)
        self._last_send_monotonic_s: float | None = None
        self.send_count = 0

    @property
    def refresh_period_s(self) -> float:
        return 1.0 / float(self.refresh_hz)

    def send_if_due(self, monotonic_s: float, *, force: bool = False) -> bool:
        now = float(monotonic_s)
        if (
            not force
            and self._last_send_monotonic_s is not None
            and now - self._last_send_monotonic_s + 1e-12 < self.refresh_period_s
        ):
            return False

        self.master.mav.set_position_target_local_ned_send(
            max(0, int(now * 1000.0)) & 0xFFFFFFFF,
            self.master.target_system,
            self.master.target_component,
            int(self.mavlink.MAV_FRAME_LOCAL_NED),
            self.type_mask,
            0.0,
            0.0,
            0.0,
            self.velocity_north_mps,
            self.velocity_east_mps,
            0.0,
            0.0,
            0.0,
            0.0,
            math.radians(float(self.heading_deg) % 360.0),
            0.0,
        )
        self._last_send_monotonic_s = now
        self.send_count += 1
        return True


def parse_ardupilot_parameter_file(path: str | Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                values[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return values


def validate_wind_only_servo_functions(values: Mapping[str, float]) -> None:
    """Reject a parameter file that assigns any propulsion output."""

    assigned = {
        name: int(round(float(value)))
        for name, value in values.items()
        if name.startswith("SERVO") and name.endswith("_FUNCTION")
    }
    if assigned.get("SERVO1_FUNCTION") != 26:
        raise ValueError("SERVO1_FUNCTION must remain steering (26)")
    if assigned.get("SERVO2_FUNCTION") != 89:
        raise ValueError("SERVO2_FUNCTION must remain mainsail (89)")
    motors = {name: value for name, value in assigned.items() if value in MOTOR_SERVO_FUNCTIONS}
    if motors:
        raise ValueError(f"wind-only experiment forbids motor servo functions: {motors}")


def validate_wind_only_sdf(path: str | Path) -> None:
    """Verify ArduPilotPlugin maps only the rudder and mainsail controls."""

    # The model contains sdformat extension elements such as ``gz:...`` whose
    # prefix is intentionally resolved by sdformat rather than an XML namespace
    # declaration.  ElementTree therefore cannot parse the source in isolation.
    # Restrict this static guard to the ArduPilot control blocks we need.
    text = Path(path).read_text(encoding="utf-8")
    joint_names = [
        match.group(1).strip()
        for block in re.findall(r"<control\b[^>]*>(.*?)</control>", text, re.DOTALL)
        if (match := re.search(r"<jointName>\s*([^<]+?)\s*</jointName>", block))
    ]
    if sorted(joint_names) != ["rudder_joint", "sail_joint"]:
        raise ValueError(
            "wind-only ArduPilotPlugin must expose exactly rudder_joint and sail_joint; "
            f"found {joint_names}"
        )
