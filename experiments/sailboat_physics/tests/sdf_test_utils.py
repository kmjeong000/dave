from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_sdf_with_extensions(path: Path) -> ET.Element:
    """Parse SDF while preserving undeclared vendor-prefixed extensions.

    sdformat accepts extension QNames such as ``gz:type`` without requiring an
    XML namespace declaration. Python's strict ElementTree parser does not, so
    bind only the referenced, undeclared prefixes in memory. The source SDF is
    never rewritten and tests continue to inspect its actual element values.
    """
    text = path.read_text(encoding="utf-8")
    declared_prefixes = set(
        re.findall(r"\sxmlns:([A-Za-z_][\w.-]*)\s*=", text)
    )
    element_prefixes = set(
        re.findall(
            r"</?([A-Za-z_][\w.-]*):[A-Za-z_][\w.-]*(?=[\s>/])",
            text,
        )
    )
    attribute_prefixes = set(
        re.findall(r"\s([A-Za-z_][\w.-]*):[A-Za-z_][\w.-]*\s*=", text)
    )
    referenced_prefixes = (element_prefixes | attribute_prefixes) - {
        "xml",
        "xmlns",
    }
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
