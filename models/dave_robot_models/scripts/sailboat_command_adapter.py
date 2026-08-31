#!/usr/bin/env python3

"""Own the final sailboat actuator topics and apply bounded RL residuals."""

import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64

from command_adapter_core import (
    clamp,
    mix_command,
    rate_limit,
    select_diagnostic_override,
)


class SailboatCommandAdapter(Node):
    def __init__(self):
        super().__init__("sailboat_command_adapter")

        namespace = str(self.declare_parameter("namespace", "sailboat").value).strip("/")
        residual_enabled = bool(self.declare_parameter("residual_enabled", False).value)
        self.declare_parameter("diagnostic_rudder_override_enabled", False)
        self.declare_parameter("diagnostic_rudder_command_rad", 0.0)
        self.base_timeout_s = float(
            self.declare_parameter("base_command_timeout_s", 0.5).value
        )
        self.residual_timeout_s = float(
            self.declare_parameter("residual_command_timeout_s", 0.5).value
        )

        self.limits = {
            "rudder": {
                "min": float(self.declare_parameter("rudder_min_rad", -0.7854).value),
                "max": float(self.declare_parameter("rudder_max_rad", 0.7854).value),
                "residual": float(
                    self.declare_parameter(
                        "rudder_residual_limit_rad", 0.0872664626
                    ).value
                ),
                "residual_rate": float(
                    self.declare_parameter(
                        "rudder_residual_rate_limit_rad_s", 0.1745329252
                    ).value
                ),
            },
            "sail": {
                "min": float(self.declare_parameter("sail_min_rad", -0.7854).value),
                "max": float(self.declare_parameter("sail_max_rad", 0.7854).value),
                "residual": float(
                    self.declare_parameter(
                        "sail_residual_limit_rad", 0.0872664626
                    ).value
                ),
                "residual_rate": float(
                    self.declare_parameter(
                        "sail_residual_rate_limit_rad_s", 0.1745329252
                    ).value
                ),
            },
        }

        joint_prefix = f"/model/{namespace}/joint"
        residual_prefix = f"/{namespace}/rl/residual"
        self.command_publishers = {
            "rudder": self.create_publisher(
                Float64, f"{joint_prefix}/rudder_joint/cmd_pos", 10
            ),
            "sail": self.create_publisher(
                Float64, f"{joint_prefix}/sail_joint/cmd_pos", 10
            ),
        }

        self.base_commands = {"rudder": None, "sail": None}
        self.base_times = {"rudder": None, "sail": None}
        self.residual_commands = {"rudder": 0.0, "sail": 0.0}
        self.residual_times = {"rudder": None, "sail": None}
        self.applied_residuals = {"rudder": 0.0, "sail": 0.0}
        self.mix_times = {"rudder": time.monotonic(), "sail": time.monotonic()}
        self.command_subscriptions = []

        for surface in ("rudder", "sail"):
            joint = f"{surface}_joint"
            self.command_subscriptions.extend(
                [
                    self.create_subscription(
                        Float64,
                        f"{joint_prefix}/{joint}/base_cmd_pos",
                        lambda msg, name=surface: self._on_base(name, msg),
                        10,
                    ),
                    self.create_subscription(
                        Float64,
                        f"{residual_prefix}/{surface}",
                        lambda msg, name=surface: self._on_residual(name, msg),
                        10,
                    ),
                ]
            )

        self.get_logger().info(
            "sailboat command adapter ready: "
            f"namespace={namespace}, residual_enabled={residual_enabled}"
        )

    def _on_base(self, surface, msg):
        self.base_commands[surface] = float(msg.data)
        self.base_times[surface] = time.monotonic()
        self._publish(surface)

    def _on_residual(self, surface, msg):
        self.residual_commands[surface] = float(msg.data)
        self.residual_times[surface] = time.monotonic()
        self._publish(surface)

    def _publish(self, surface):
        now = time.monotonic()
        base = self.base_commands[surface]
        base_time = self.base_times[surface]
        if base is None or base_time is None:
            return
        if self.base_timeout_s > 0.0 and now - base_time > self.base_timeout_s:
            self.get_logger().warning(
                f"not publishing {surface}: base command is stale",
                throttle_duration_sec=2.0,
            )
            return

        residual = self.residual_commands[surface]
        residual_time = self.residual_times[surface]
        residual_is_fresh = (
            residual_time is not None
            and (
                self.residual_timeout_s <= 0.0
                or now - residual_time <= self.residual_timeout_s
            )
        )
        limits = self.limits[surface]
        residual_enabled = bool(self.get_parameter("residual_enabled").value)
        try:
            if not residual_enabled:
                applied_residual = 0.0
            else:
                target_residual = residual if residual_is_fresh else 0.0
                target_residual = clamp(
                    target_residual,
                    -limits["residual"],
                    limits["residual"],
                )
                applied_residual = rate_limit(
                    target_residual,
                    self.applied_residuals[surface],
                    max_rate=limits["residual_rate"],
                    elapsed_s=max(0.0, now - self.mix_times[surface]),
                )

            diagnostic_override_enabled = bool(
                surface == "rudder"
                and self.get_parameter(
                    "diagnostic_rudder_override_enabled"
                ).value
            )
            if diagnostic_override_enabled:
                command = select_diagnostic_override(
                    base,
                    self.get_parameter("diagnostic_rudder_command_rad").value,
                    override_enabled=True,
                    command_min=limits["min"],
                    command_max=limits["max"],
                )
                applied_residual = 0.0
            else:
                command = mix_command(
                    base,
                    applied_residual,
                    residual_enabled=residual_enabled,
                    command_min=limits["min"],
                    command_max=limits["max"],
                    residual_limit=limits["residual"],
                )
        except ValueError as exc:
            self.get_logger().error(f"rejected invalid {surface} command: {exc}")
            return

        self.applied_residuals[surface] = applied_residual
        self.mix_times[surface] = now
        output = Float64()
        output.data = command
        self.command_publishers[surface].publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = SailboatCommandAdapter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
