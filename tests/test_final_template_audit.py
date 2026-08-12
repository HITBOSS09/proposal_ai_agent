"""Read-only invariants discovered from the sole final publishing template."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_final_template.py"


def _audit_module():
    spec = importlib.util.spec_from_file_location("audit_final_template", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exactly_one_final_template_and_read_only_inventory() -> None:
    module = _audit_module()
    template = module.discover_final_template()
    before = template.read_bytes()

    result = module.inspect_template(template)

    assert result == {
        "path": str(template.resolve()),
        "sha256": "df518b4df02bfcd4e31cfa03e6120c598a4f92a7a3b2f5ddfbdc97b45f80a31f",
        "package_valid": True,
        "package_parts": 49,
        "body_children": 161,
        "paragraphs": 140,
        "sections": 18,
        "section_types": ["nextPage", "continuous"] + ["nextPage"] * 16,
        "headers": 18,
        "footers": 18,
        "page_fields": 36,
        "numpages_fields": 36,
        "tables": 20,
        "table_rows": 181,
        "table_cells": 722,
        "merged_cell_markers": 10,
        "media": 1,
        "drawingml": 70,
        "vml_pict": 69,
        "text_boxes": 94,
        "styles": 19,
        "abstract_numbering": 8,
        "numbering_instances": 8,
        "relationships": 46,
        "has_theme": True,
        "has_content_types": True,
    }
    assert template.read_bytes() == before


def test_template_discovery_fails_closed_for_zero_or_multiple_candidates(tmp_path: Path) -> None:
    module = _audit_module()
    with pytest.raises(module.TemplateCandidateError, match="found 0"):
        module.discover_final_template(tmp_path)
    (tmp_path / "one.docx").write_bytes(b"one")
    (tmp_path / "two.docx").write_bytes(b"two")
    with pytest.raises(module.TemplateCandidateError, match="found 2"):
        module.discover_final_template(tmp_path)
