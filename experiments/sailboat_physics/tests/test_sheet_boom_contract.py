from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from experiments.sailboat_BO.run_trial import (
    pwm_to_sheet_allowance_rad,
    read_param_map,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PARAM_FILE = REPO_ROOT / "models/dave_robot_models/config/sailboat/ardurover.parm"
MODEL_FILE = REPO_ROOT / "models/dave_robot_models/description/sailboat/model.sdf"


def _parse_sdf(path: Path) -> ET.Element:
    """Parse SDF while preserving vendor-prefixed extension elements.

    Gazebo SDF files may use extension QNames such as ``gz:type`` without an
    XML namespace declaration.  sdformat accepts those extensions, whereas
    Python's strict ElementTree parser requires every prefix to be bound.  Add
    in-memory namespace declarations for all referenced, undeclared prefixes;
    the source model is left untouched and all contract assertions still read
    their values from the model itself.
    """
    text = path.read_text(encoding="utf-8")
    declared_prefixes = set(
        re.findall(r"\sxmlns:([A-Za-z_][\w.-]*)\s*=", text)
    )
    element_prefixes = set(
        re.findall(r"</?([A-Za-z_][\w.-]*):[A-Za-z_][\w.-]*(?=[\s>/])", text)
    )
    attribute_prefixes = set(
        re.findall(r"\s([A-Za-z_][\w.-]*):[A-Za-z_][\w.-]*\s*=", text)
    )
    referenced_prefixes = (element_prefixes | attribute_prefixes) - {"xml", "xmlns"}
    missing_prefixes = sorted(referenced_prefixes - declared_prefixes)

    if missing_prefixes:
        declarations = "".join(
            f' xmlns:{prefix}="urn:sdf-extension:{prefix}"'
            for prefix in missing_prefixes
        )
        text, replacement_count = re.subn(
            r"<sdf(?=[\s>])",
            f"<sdf{declarations}",
            text,
            count=1,
        )
        if replacement_count != 1:
            raise ValueError(f"SDF root element not found in {path}")

    return ET.fromstring(text)


def _plugins(root: ET.Element, name: str) -> list[ET.Element]:
    return [plugin for plugin in root.findall(".//plugin") if plugin.get("name") == name]


def test_sheet_range_matches_sail_joint_hard_limit():
    params = read_param_map(PARAM_FILE)
    root = _parse_sdf(MODEL_FILE)
    sail_joint = root.find(".//joint[@name='sail_joint']")

    assert sail_joint is not None
    assert params["SAIL_ANGLE_MIN"] == pytest.approx(0.0)
    assert params["SAIL_ANGLE_MAX"] == pytest.approx(45.0)
    assert float(sail_joint.findtext("axis/limit/lower")) == pytest.approx(-math.pi / 4, abs=1e-5)
    assert float(sail_joint.findtext("axis/limit/upper")) == pytest.approx(math.pi / 4, abs=1e-5)
    assert float(sail_joint.findtext("axis/dynamics/damping")) == pytest.approx(1.0)
    assert float(sail_joint.findtext("axis/dynamics/friction")) == pytest.approx(0.05)


def test_ardupilot_mainsail_channel_outputs_unsigned_allowance():
    root = _parse_sdf(MODEL_FILE)
    control = root.find(".//plugin[@name='ArduPilotPlugin']/control[@channel='1']")

    assert control is not None
    assert control.findtext("cmd_topic") == "/model/sailboat/joint/sail_joint/base_cmd_pos"
    assert float(control.findtext("offset")) == pytest.approx(0.0)
    assert float(control.findtext("multiplier")) == pytest.approx(math.pi / 4, abs=1e-5)


def test_sail_uses_unilateral_asv_controller_not_generic_position_controller():
    root = _parse_sdf(MODEL_FILE)
    sail_controllers = _plugins(root, "gz::sim::systems::SailPositionController")

    assert len(sail_controllers) == 1
    controller = sail_controllers[0]
    assert controller.get("filename") == "asv_sim2-sail-position-controller-system"
    assert controller.findtext("joint_name") == "sail_joint"
    assert controller.findtext("topic") == "/model/sailboat/joint/sail_joint/cmd_pos"
    assert float(controller.findtext("p_gain")) == pytest.approx(30.0)
    assert float(controller.findtext("i_gain")) == pytest.approx(0.0)
    assert float(controller.findtext("d_gain")) == pytest.approx(0.0)
    assert float(controller.findtext("cmd_max")) == pytest.approx(100.0)
    assert float(controller.findtext("cmd_min")) == pytest.approx(-100.0)
    assert float(controller.findtext("initial_position")) == pytest.approx(math.pi / 4, abs=1e-5)

    generic_sail_controllers = [
        plugin
        for plugin in _plugins(root, "gz::sim::systems::JointPositionController")
        if plugin.findtext("joint_name") == "sail_joint"
    ]
    assert not generic_sail_controllers


def test_sail_wrench_diagnostics_use_hull_roll_axis():
    root = _parse_sdf(MODEL_FILE)
    plugin = root.find(".//plugin[@name='ioes::sim::SailLiftDragSystem']")

    assert plugin is not None
    assert plugin.findtext("sail_link") == "sail_link"
    assert plugin.findtext("base_link") == "base_link"
    assert plugin.findtext("roll_axis") == "0 1 0"
    assert plugin.findtext("debug") == "true"
    assert float(plugin.findtext("debug_period")) == pytest.approx(1.0)


def test_mainsheet_pwm_logging_uses_allowance_magnitude():
    assert pwm_to_sheet_allowance_rad(1000, 1000, 2000, 0, 45) == pytest.approx(0.0)
    assert pwm_to_sheet_allowance_rad(1500, 1000, 2000, 0, 45) == pytest.approx(math.pi / 8)
    assert pwm_to_sheet_allowance_rad(2000, 1000, 2000, 0, 45) == pytest.approx(math.pi / 4)
