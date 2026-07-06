#!/usr/bin/env python3
from __future__ import annotations

import codecs
import os
import shutil
from pathlib import Path


TEXT_PATTERNS = (
    "CMakeLists.txt",
    "package.xml",
    "*.cmake",
    "*.dsv.in",
    "*.world",
    "*.sdf",
    "*.xml",
)

CANONICAL_TEXT = {
    Path("models/dave_worlds/CMakeLists.txt"): """cmake_minimum_required(VERSION 3.5)
project(dave_worlds)

# Find dependencies
find_package(ament_cmake REQUIRED)

install(
  DIRECTORY worlds media
  DESTINATION share/${PROJECT_NAME}
)

ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.dsv.in")

ament_package()
""",
    Path("models/dave_worlds/package.xml"): """<?xml version="1.0" encoding="UTF-8"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>dave_worlds</name>
  <version>0.0.0</version>
  <description>Demo World files</description>
  <maintainer email="helenamoyen@usp.br">lena</maintainer>
  <maintainer email="gaurav.og.9920@gmail.com">Gaurav Kumar</maintainer>
  <license>Apache-2.0</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>
  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
""",
}


def decode_text(data: bytes) -> str:
    data = data.replace(codecs.BOM_UTF8, b"", 1)
    if data.startswith(codecs.BOM_UTF16_LE) or data.startswith(codecs.BOM_UTF16_BE):
        return data.decode("utf-16")

    if b"\x00" in data:
        for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        data = data.replace(b"\x00", b"")

    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="ignore")


def sanitize_text_file(path: Path) -> bool:
    original = path.read_bytes()
    text = decode_text(original)
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    text = "".join(ch for ch in text if ch in ("\n", "\t") or ord(ch) >= 32)
    if not text.endswith("\n"):
        text += "\n"

    normalized = text.encode("utf-8")
    changed = normalized != original
    if changed:
        path.write_bytes(normalized)
    return changed


def repair_canonical_files(repo_root: Path) -> list[str]:
    repaired: list[str] = []
    for relative_path, canonical_text in CANONICAL_TEXT.items():
        target = repo_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = canonical_text.encode("utf-8")
        current = target.read_bytes() if target.exists() else b""
        if current != expected:
            target.write_bytes(expected)
            repaired.append(str(relative_path))
    return repaired


def main() -> int:
    username = os.environ.get("USERNAME", "docker")
    staging = Path(
        os.environ.get("SAILBOAT_SRC_STAGING", f"/home/{username}/sailboat_src_staging")
    )
    workspace = Path(os.environ.get("SAILBOAT_WS", f"/home/{username}/sailboat_ws"))
    repo_root = workspace / "src" / "dave"

    if repo_root.exists():
        shutil.rmtree(repo_root)
    repo_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staging, repo_root)

    sanitized_count = 0
    for pattern in TEXT_PATTERNS:
        for path in repo_root.rglob(pattern):
            if sanitize_text_file(path):
                sanitized_count += 1

    repaired = repair_canonical_files(repo_root)

    for relative in (
        "build",
        "install",
        "log",
        "src/dave/build",
        "src/dave/install",
        "src/dave/log",
    ):
        target = workspace / relative
        if target.exists():
            shutil.rmtree(target)

    print(
        f"[prepare_sailboat_workspace] repo={repo_root} sanitized={sanitized_count} repaired={len(repaired)}"
    )
    for item in repaired:
        print(f"[prepare_sailboat_workspace] repaired {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
