"""Create the deterministic Phase 4C.1.5 PRU population acceptance artifact."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

from proposal_ai_agent.proposal_generation.publishing import (
    CertifiedContracts,
    CertifiedTemplateBinding,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "documents/PRU_T72_Module_Breakdown.docx"
CONTRACTS = ROOT / "templates/bdil_pru_v1"
OUTPUT = ROOT / "output/phase4c_population_acceptance.docx"
EXPECTED_SHA = "e81d6c34362fb0e7d17b8a82e5e756ad696f98d61e90100e3ca5b7bec7182d55"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def _cell_id(table: int, owner: str, purpose: str, row: int, column: int) -> str:
    return f"bdil_pru_v1.table.{table:02d}.{owner}.{purpose}.r{row}.c{column}"


def _populate_existing_row(
    binding: CertifiedTemplateBinding,
    table: int,
    owner: str,
    purpose: str,
    row: int,
    values: tuple[str, ...],
) -> None:
    for column, value in enumerate(values):
        binding.replace_dynamic_cell(
            _cell_id(table, owner, purpose, row, column), value
        )


def _package_facts(path: Path) -> dict[str, object]:
    with ZipFile(path) as package:
        names = package.namelist()
        document = etree.fromstring(package.read("word/document.xml"))
        media = {
            name: sha256(package.read(name)).hexdigest()
            for name in names if name.startswith("word/media/")
        }
        footer_fields = []
        for index in range(1, 6):
            footer = etree.fromstring(package.read(f"word/footer{index}.xml"))
            footer_fields.extend(
                value.strip()
                for value in footer.xpath(".//w:instrText/text()", namespaces=NS)
            )
        tables = document.xpath("./w:body/w:tbl", namespaces=NS)
        return {
            "sections": len(document.xpath(".//w:sectPr", namespaces=NS)),
            "tables": len(tables),
            "table_rows": [len(table.xpath("./w:tr", namespaces=NS)) for table in tables],
            "table_properties": [
                etree.tostring(table.find(f"{{{W}}}tblPr"), method="c14n")
                for table in tables
            ],
            "table_grids": [
                etree.tostring(table.find(f"{{{W}}}tblGrid"), method="c14n")
                for table in tables
            ],
            "body_drawings": len(document.xpath(".//w:body//w:drawing", namespaces=NS)),
            "headers": len([n for n in names if n.startswith("word/header") and n.endswith(".xml")]),
            "footers": len([n for n in names if n.startswith("word/footer") and n.endswith(".xml")]),
            "media": media,
            "footer_fields": footer_fields,
        }


def _validate_output(source: Path, output: Path) -> None:
    before = _package_facts(source)
    after = _package_facts(output)
    assert after["sections"] == before["sections"] == 17
    assert after["tables"] == before["tables"] == 20
    assert after["headers"] == before["headers"] == 5
    assert after["footers"] == before["footers"] == 5
    assert after["body_drawings"] == before["body_drawings"]
    assert after["media"] == before["media"]
    assert after["footer_fields"] == before["footer_fields"]
    assert after["table_properties"] == before["table_properties"]
    assert after["table_grids"] == before["table_grids"]
    expected_rows = list(before["table_rows"])
    expected_rows[13] += 1
    assert after["table_rows"] == expected_rows
    with ZipFile(output) as package:
        combined = "".join(
            etree.fromstring(package.read(name)).xpath("string()")
            for name in package.namelist()
            if name == "word/document.xml" or name.startswith("word/header")
        )
    for marker in (
        "BDIL SKYSHIELD-X Acceptance Demonstrator",
        "Autonomous Multi-Sensor Defence Platform",
        "CONTROLLED ACCEPTANCE TEST",
        "Synthetic Quantum Radar Controller",
        "Duplicate-Table Isolation Beacon",
        "Certified Cloned Sensor Node",
    ):
        assert marker in combined
    assert "**" not in combined


def run_acceptance(output: Path = OUTPUT) -> Path:
    source_before = file_sha256(SOURCE)
    if source_before != EXPECTED_SHA:
        raise RuntimeError("source template does not match the certified SHA-256")
    contracts = CertifiedContracts.load(CONTRACTS)
    binding = CertifiedTemplateBinding(SOURCE, contracts, working_copy=output)

    # Certified textual fields: cover, ordinary body, module-specific body, and headers.
    binding.replace_dynamic_text(
        "cover_document_title", "BDIL SKYSHIELD-X Acceptance Demonstrator"
    )
    binding.replace_dynamic_text(
        "m1_integration.text_slot_01",
        "This deterministic acceptance paragraph verifies certified body-text population while retaining all template-owned typography and paragraph geometry.",
    )
    binding.replace_dynamic_text(
        "m5_integration.text_slot_18",
        "The synthetic SKYSHIELD-X module combines artificial multi-sensor inputs solely to demonstrate module-specific certified text binding.",
    )
    for index in range(1, 6):
        binding.replace_dynamic_text(
            f"header_project_text_{index}",
            "BDIL SKYSHIELD-X — CONTROLLED ACCEPTANCE TEST",
        )

    # Certified cover metadata cells.
    binding.replace_dynamic_cell(
        _cell_id(0, "cover", "document_control", 1, 1),
        "Autonomous Multi-Sensor Defence Platform",
    )
    binding.replace_dynamic_cell(
        _cell_id(0, "cover", "document_control", 5, 1),
        "CONTROLLED ACCEPTANCE TEST",
    )

    # A simple fixed-size power table.
    _populate_existing_row(
        binding, 1, "m1_power", "power_consumption", 0,
        ("1", "Synthetic Sensor Control Unit", "111W"),
    )

    # An orange table whose existing rows are sufficient.
    _populate_existing_row(
        binding, 12, "m5_power", "power_placeholder", 1,
        ("1", "Artificial Fusion Processor", "2", "48", "222 W", "Acceptance-only data"),
    )
    _populate_existing_row(
        binding, 12, "m5_power", "power_placeholder", 2,
        ("2", "Synthetic Optical Array", "4", "48", "144 W", "Deterministic fixture"),
    )

    # Clone before changing the source prototype row; populate through clone provenance.
    clone = binding.insert_certified_row(
        "bdil_pru_v1.table.13.m5_bom.bom_placeholder.row_prototype.2",
        insert_after_row_index=5,
    )
    binding.populate_certified_cloned_row(
        clone,
        ("6", "Certified Cloned Sensor Node", "Synthetic Type CX-6", "1", "Cloned prototype acceptance"),
    )

    # BOM population in the intended M5 table.
    _populate_existing_row(
        binding, 13, "m5_bom", "bom_placeholder", 1,
        ("1", "Synthetic Quantum Radar Controller", "Acceptance Model QR-1", "1", "Artificial fixture"),
    )
    _populate_existing_row(
        binding, 13, "m5_bom", "bom_placeholder", 2,
        ("2", "Acceptance Telemetry Emulator", "Synthetic Type TE-2", "2", "No operational use"),
    )

    # A visually similar BOM in M6 proves semantic-parent/table identity isolation.
    _populate_existing_row(
        binding, 15, "m6_bom", "bom_placeholder", 1,
        ("1", "Duplicate-Table Isolation Beacon", "Synthetic Type IB-1", "1", "M6 binding only"),
    )

    result = binding.save(output)
    if file_sha256(SOURCE) != source_before:
        raise RuntimeError("source template changed during acceptance population")
    _validate_output(SOURCE, result)
    return result


if __name__ == "__main__":
    generated = run_acceptance()
    print(f"Created: {generated}")
    print(f"Source SHA-256 before/after: {EXPECTED_SHA}")
