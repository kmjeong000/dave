from __future__ import annotations

import pytest

from experiments.sailboat_RL.ros_backend import Ros2AttachBackend


class _FakeParameter:
    class Type:
        BOOL = "bool"

    def __init__(self, name, parameter_type, value):
        self.name = name
        self.type_ = parameter_type
        self.value = value


class _FakeResult:
    def __init__(self, successful=True, reason=""):
        self.successful = successful
        self.reason = reason


class _FakeResponse:
    def __init__(self, results):
        self.results = results


class _FakeFuture:
    def __init__(self, response=None):
        self.response = response or _FakeResponse([_FakeResult()])

    def done(self):
        return True

    def result(self):
        return self.response


class _FakeParameterClient:
    def __init__(self, response=None, services_ready=True):
        self.wait_timeout_s = None
        self.parameters = None
        self.response = response
        self.services_ready = services_ready

    def wait_for_services(self, timeout_sec=None):
        self.wait_timeout_s = timeout_sec
        return self.services_ready

    def set_parameters(self, parameters):
        self.parameters = parameters
        return _FakeFuture(self.response)


class _FakeRclpy:
    def __init__(self):
        self.spin_timeout_s = None
        self.spin_once_timeout_s = None
        self.shutdown_called = False

    def spin_until_future_complete(self, node, future, timeout_sec=None):
        del node, future
        self.spin_timeout_s = timeout_sec

    def spin_once(self, node, timeout_sec=None):
        del node
        self.spin_once_timeout_s = timeout_sec

    def ok(self):
        return True

    def shutdown(self):
        self.shutdown_called = True


class _FakeNode:
    def __init__(self):
        self.destroyed = False

    def destroy_node(self):
        self.destroyed = True


def test_set_adapter_enabled_uses_jazzy_parameter_service_api():
    backend = object.__new__(Ros2AttachBackend)
    backend.manage_adapter_enabled = True
    backend.telemetry_timeout_s = 3.5
    backend._adapter_parameters = _FakeParameterClient()
    backend._Parameter = _FakeParameter
    backend._rclpy = _FakeRclpy()
    backend._node = object()

    backend._set_adapter_enabled(True)

    assert backend._adapter_parameters.wait_timeout_s == 3.5
    assert backend._rclpy.spin_timeout_s == 3.5
    assert len(backend._adapter_parameters.parameters) == 1
    parameter = backend._adapter_parameters.parameters[0]
    assert parameter.name == "residual_enabled"
    assert parameter.type_ == _FakeParameter.Type.BOOL
    assert parameter.value is True


def test_set_adapter_enabled_reports_parameter_rejection():
    response = _FakeResponse([_FakeResult(False, "adapter rejected test value")])
    backend = object.__new__(Ros2AttachBackend)
    backend.manage_adapter_enabled = True
    backend.telemetry_timeout_s = 3.5
    backend._adapter_parameters = _FakeParameterClient(response)
    backend._Parameter = _FakeParameter
    backend._rclpy = _FakeRclpy()
    backend._node = object()

    with pytest.raises(RuntimeError, match="adapter rejected test value"):
        backend._set_adapter_enabled(False)


def test_close_accepts_adapter_service_already_gone_and_releases_ros_resources():
    backend = object.__new__(Ros2AttachBackend)
    backend.manage_adapter_enabled = True
    backend.telemetry_timeout_s = 0.1
    backend._adapter_parameters = _FakeParameterClient(services_ready=False)
    backend._Parameter = _FakeParameter
    backend._rclpy = _FakeRclpy()
    backend._node = _FakeNode()
    backend._owns_rclpy = True
    backend._closed = False
    published = []
    backend._publish_residuals = lambda rudder, sail: published.append(
        (rudder, sail)
    )

    backend.close()

    assert published == [(0.0, 0.0)]
    assert backend._adapter_parameters.parameters is None
    assert backend._node.destroyed
    assert backend._rclpy.shutdown_called
    assert backend._closed


def test_close_releases_ros_resources_before_reporting_parameter_rejection():
    response = _FakeResponse([_FakeResult(False, "adapter rejected disable")])
    backend = object.__new__(Ros2AttachBackend)
    backend.manage_adapter_enabled = True
    backend.telemetry_timeout_s = 0.1
    backend._adapter_parameters = _FakeParameterClient(response)
    backend._Parameter = _FakeParameter
    backend._rclpy = _FakeRclpy()
    backend._node = _FakeNode()
    backend._owns_rclpy = True
    backend._closed = False
    backend._publish_residuals = lambda rudder, sail: None

    with pytest.raises(RuntimeError, match="adapter rejected disable"):
        backend.close()

    assert backend._node.destroyed
    assert backend._rclpy.shutdown_called
    assert backend._closed


def test_waypoint_capture_requires_configured_hold_time():
    backend = object.__new__(Ros2AttachBackend)
    backend.waypoints = [(0.0, 0.0), (10.0, 0.0)]
    backend.capture_radius_m = 5.0
    backend.capture_hold_s = 1.0
    backend._waypoint_index = 0
    backend._within_capture_radius_since_s = None
    backend._mission_complete = False
    backend._pose = {"sim_time_s": 10.0, "x_m": 0.0, "y_m": 0.0}

    backend._update_waypoint()
    assert backend._waypoint_index == 0

    backend._pose["sim_time_s"] = 10.9
    backend._update_waypoint()
    assert backend._waypoint_index == 0

    backend._pose["sim_time_s"] = 11.0
    backend._update_waypoint()
    assert backend._waypoint_index == 1
    assert not backend._mission_complete
