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
    def __init__(self, response=None):
        self.wait_timeout_s = None
        self.parameters = None
        self.response = response

    def wait_for_services(self, timeout_sec=None):
        self.wait_timeout_s = timeout_sec
        return True

    def set_parameters(self, parameters):
        self.parameters = parameters
        return _FakeFuture(self.response)


class _FakeRclpy:
    def __init__(self):
        self.spin_timeout_s = None

    def spin_until_future_complete(self, node, future, timeout_sec=None):
        del node, future
        self.spin_timeout_s = timeout_sec


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
