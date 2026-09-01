from __future__ import annotations

import re
import unittest
from pathlib import Path

from experiments.sailboat_physics.tests.sdf_test_utils import (
    parse_sdf_with_extensions,
)


def method_body(source: str, signature: str) -> str:
    signature_index = source.index(signature)
    opening_brace = source.index("{", signature_index)
    depth = 0
    for index in range(opening_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening_brace + 1 : index]
    raise AssertionError(f"unterminated method body: {signature}")


class StartupStateHoldContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[3]
        cls.model_path = (
            cls.repo_root
            / "models"
            / "dave_robot_models"
            / "description"
            / "sailboat"
            / "model.sdf"
        )
        cls.source_path = (
            cls.repo_root
            / "gazebo"
            / "dave_gz_model_plugins"
            / "src"
            / "StartupStateHoldSystem.cc"
        )
        cls.cmake_path = (
            cls.repo_root
            / "gazebo"
            / "dave_gz_model_plugins"
            / "CMakeLists.txt"
        )

    def test_model_starts_held_with_a_finite_fail_safe(self):
        root = parse_sdf_with_extensions(self.model_path)
        plugin = next(
            item
            for item in root.findall(".//plugin")
            if item.get("name") == "ioes::sim::StartupStateHoldSystem"
        )

        self.assertEqual(plugin.get("filename"), "libStartupStateHoldSystem.so")
        self.assertEqual(
            plugin.findtext("hold_topic"),
            "/model/sailboat/startup_hold",
        )
        self.assertEqual(plugin.findtext("initially_held"), "true")
        self.assertGreater(
            float(plugin.findtext("auto_release_timeout_s", "0")),
            0.0,
        )

    def test_plugin_holds_pose_and_all_link_velocities(self):
        source = self.source_path.read_text(encoding="utf-8")

        self.assertIn("SetWorldPoseCmd", source)
        self.assertIn("SetLinearVelocity", source)
        self.assertIn("SetAngularVelocity", source)
        self.assertIn("gz::msgs::Boolean", source)
        self.assertIn("autoReleaseTimeoutS", source)

    def test_release_removes_only_persistent_velocity_commands_in_preupdate(self):
        source = self.source_path.read_text(encoding="utf-8")
        pre_update = method_body(source, "void PreUpdate(")
        callback = method_body(source, "void OnHoldMsg(")
        cleanup = method_body(source, "void CleanupVelocityCommands(")

        self.assertIn("this->requestedHeld = _msg.data();", callback)
        self.assertNotIn("RemoveComponent", callback)
        self.assertIn("requestedHeld != this->appliedHeld", pre_update)
        self.assertEqual(
            pre_update.count("this->appliedHeld = requestedHeld;"),
            1,
        )
        self.assertIn("this->CleanupVelocityCommands(_ecm);", pre_update)

        removed_types = re.findall(r"RemoveComponent<\s*([^>]+)>", cleanup)
        self.assertCountEqual(
            removed_types,
            [
                "gz::sim::components::LinearVelocityCmd",
                "gz::sim::components::AngularVelocityCmd",
            ],
        )
        self.assertNotIn("WorldPoseCmd", cleanup)

    def test_release_lifecycle_has_observable_runtime_logs(self):
        source = self.source_path.read_text(encoding="utf-8")

        self.assertIn("hold command received", source)
        self.assertIn("internal held-state transition", source)
        self.assertIn("velocity command cleanup", source)

    def test_plugin_is_built_and_installed(self):
        cmake = self.cmake_path.read_text(encoding="utf-8")

        self.assertIn("add_library(StartupStateHoldSystem SHARED", cmake)
        self.assertRegex(
            cmake,
            r"install\([\s\S]*TARGETS[\s\S]*StartupStateHoldSystem",
        )


if __name__ == "__main__":
    unittest.main()
