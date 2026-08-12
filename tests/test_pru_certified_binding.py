"""Phase 4C.1 certification and fail-closed binding tests."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from lxml import etree
import pytest

from proposal_ai_agent.proposal_generation.publishing import (
    BindingAuthorizationError,
    CertifiedContracts,
    CertifiedTemplateBinding,
    LocatorMismatch,
    StructuralOperation,
    TemplateIntegrityError,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "documents/PRU_T72_Module_Breakdown.docx"
CONTRACT_ROOT = ROOT / "templates/bdil_pru_v1"
EXPECTED_SHA = "e81d6c34362fb0e7d17b8a82e5e756ad696f98d61e90100e3ca5b7bec7182d55"


@pytest.fixture()
def contracts() -> CertifiedContracts:
    return CertifiedContracts.load(CONTRACT_ROOT)


def test_source_is_byte_identical_and_contracts_are_version_pinned(contracts) -> None:
    before = file_sha256(TEMPLATE)
    CertifiedTemplateBinding(TEMPLATE, contracts)
    assert before == file_sha256(TEMPLATE) == EXPECTED_SHA == contracts.template_sha256


def test_all_tables_have_unique_contextual_identities_and_bom_tables_cannot_collide(contracts) -> None:
    tables = contracts.template_map["tables"]
    identities = {
        (t["table_id"], t["semantic_parent"], t["locator"]["body_node_index"],
         t["locator"]["table_signature"])
        for t in tables
    }
    assert len(tables) == len(identities) == 20
    bom = [t for t in tables if "bom" in t["purpose"]]
    assert len({t["table_id"] for t in bom}) == len(bom)
    assert len({(t["semantic_parent"], t["purpose"], t["locator"]["body_node_index"]) for t in bom}) == len(bom)


def test_every_dynamic_cell_has_exact_table_row_and_column_binding(contracts) -> None:
    cells = contracts.template_map["dynamic_table_cell_bindings"]
    assert len(cells) == 615
    assert len({cell["binding_id"] for cell in cells}) == 615
    for cell in cells:
        locator = cell["locator"]
        assert cell["table_id"] in contracts.schema["table_ids"]
        assert all(key in locator for key in (
            "template_sha256", "table_index", "body_node_index", "table_signature",
            "row_index", "physical_cell_index", "grid_column_index", "tcPr_sha256",
        ))
        assert cell["column_semantic"]


def test_all_repeatable_rows_resolve_to_existing_format_complete_prototypes(contracts) -> None:
    binding = CertifiedTemplateBinding(TEMPLATE, contracts)
    prototypes = contracts.template_map["repeatable_row_prototypes"]
    assert len(prototypes) == 26
    for prototype in prototypes:
        clone = binding.clone_certified_row(prototype["prototype_id"])
        assert clone.tag.endswith("tr")
        assert prototype["clone_preserves"] == ["trPr", "tcPr", "gridSpan", "pPr", "rPr"]


def test_static_cells_are_rejected_and_dynamic_cell_edit_preserves_formatting(contracts) -> None:
    binding = CertifiedTemplateBinding(TEMPLATE, contracts)
    static_id = contracts.template_map["static_table_cells"][0]["binding_id"]
    with pytest.raises(BindingAuthorizationError):
        binding.replace_dynamic_cell(static_id, "forbidden")

    dynamic_id = "bdil_pru_v1.table.00.cover.document_control.r0.c1"
    cell = binding.resolve_dynamic_cell(dynamic_id)
    tcpr = etree.tostring(cell.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tcPr"))
    ppr = [etree.tostring(node) for node in cell.xpath("./w:p/w:pPr")]
    rpr = [etree.tostring(node) for node in cell.xpath(".//w:r/w:rPr")]
    binding.replace_dynamic_cell(dynamic_id, "BDIL-CERT-001")
    assert "BDIL-CERT-001" == "".join(cell.xpath(".//w:t/text()"))
    assert etree.tostring(cell.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tcPr")) == tcpr
    assert [etree.tostring(node) for node in cell.xpath("./w:p/w:pPr")] == ppr
    assert [etree.tostring(node) for node in cell.xpath(".//w:r/w:rPr")] == rpr


def test_row_clone_preserves_table_and_row_openxml_contract(contracts) -> None:
    binding = CertifiedTemplateBinding(TEMPLATE, contracts)
    table = binding.document.tables[13]._tbl
    tblpr = etree.tostring(table.tblPr)
    tblgrid = etree.tostring(table.tblGrid)
    original = table.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr")[1]
    clone = binding.clone_certified_row(
        "bdil_pru_v1.table.13.m5_bom.bom_placeholder.row_prototype.1"
    )
    assert etree.tostring(table.tblPr) == tblpr
    assert etree.tostring(table.tblGrid) == tblgrid
    for path in ("./w:trPr", "./w:tc/w:tcPr", ".//w:p/w:pPr", ".//w:r/w:rPr", ".//w:gridSpan"):
        serialized_original = [etree.tostring(n, method="c14n", exclusive=True) for n in original.xpath(path)]
        serialized_clone = [etree.tostring(n, method="c14n", exclusive=True) for n in clone.xpath(path)]
        assert serialized_original == serialized_clone


def test_dynamic_text_fields_are_unique_and_exactly_located(contracts) -> None:
    fields = contracts.schema["dynamic_text_fields"]
    assert len(fields) == 38
    assert len({field["field_id"] for field in fields}) == 38
    assert all(field["locator"]["template_sha256"] == EXPECTED_SHA for field in fields)
    assert all(field["locator"].get("verified_text") for field in fields)


def test_visual_assets_unknowns_and_word_fields_are_protected(contracts) -> None:
    assets = contracts.template_map["visual_assets"]
    assert len(assets) == 8
    assert [asset["asset_id"] for asset in assets if asset["classification"] == "STATIC_ASSET"] == ["cover_logo"]
    assert len([asset for asset in assets if asset["classification"] == "PROJECT_SPECIFIC_TECHNICAL_VISUAL"]) == 7
    assert all(not asset["mutable"] for asset in assets)
    assert contracts.policy["image_policy"]["DYNAMIC_IMAGE"] == []
    assert contracts.schema["remaining_unknown_nodes"] == []
    assert len(contracts.schema["unknown_resolution"]) == 8
    fields = contracts.schema["computed_fields"]
    assert len(fields) == 10
    assert {field["field"] for field in fields} == {"PAGE", "NUMPAGES"}
    assert all(not field["mutable"] and field["preserve_compatibility_branches"] for field in fields)


def test_user_insert_requires_explicit_override_and_certified_boundary(contracts) -> None:
    boundary = "between_module_2_3"
    with pytest.raises(BindingAuthorizationError):
        contracts.validate_structural_operation(
            StructuralOperation.INSERT_AFTER, boundary_id=boundary
        )
    with pytest.raises(BindingAuthorizationError):
        contracts.validate_structural_operation(
            StructuralOperation.INSERT_AFTER,
            boundary_id="inside_split_table",
            explicit_user_override=True,
        )
    contracts.validate_structural_operation(
        StructuralOperation.INSERT_AFTER,
        boundary_id=boundary,
        explicit_user_override=True,
    )


def test_remove_reorder_and_unknown_or_static_mutation_fail_closed(contracts) -> None:
    for operation in (StructuralOperation.REMOVE, StructuralOperation.REORDER):
        with pytest.raises(BindingAuthorizationError):
            contracts.validate_structural_operation(
                operation, target_id="module_1", explicit_user_override=True
            )
    for target in ("association_line", "image_7", "module_1"):
        with pytest.raises(BindingAuthorizationError):
            contracts.validate_structural_operation(
                StructuralOperation.REPLACE_CONTENT, target_id=target
            )


def test_locator_and_template_sha_mismatch_fail_closed(tmp_path: Path, contracts) -> None:
    broken = deepcopy(contracts)
    broken.template_map["tables"][0]["locator"]["table_signature"] = "0" * 64
    with pytest.raises(LocatorMismatch):
        CertifiedTemplateBinding(TEMPLATE, broken)

    tampered = tmp_path / "tampered.docx"
    tampered.write_bytes(TEMPLATE.read_bytes() + b"tamper")
    with pytest.raises(TemplateIntegrityError):
        CertifiedTemplateBinding(tampered, contracts)


def test_contract_does_not_introduce_markdown_or_generic_table_reconstruction(contracts) -> None:
    policy_text = json.dumps(contracts.policy)
    assert "markdown_parsing" in policy_text
    assert "generic_table_reconstruction" in policy_text
    assert "unsupported_operations" in contracts.policy
    assert all(table["locator"]["table_index"] >= 0 for table in contracts.template_map["tables"])


def test_source_package_has_all_protected_footer_fields() -> None:
    with ZipFile(TEMPLATE) as package:
        instructions = []
        for index in range(1, 6):
            root = etree.fromstring(package.read(f"word/footer{index}.xml"))
            instructions.extend("".join(root.xpath(
                ".//w:instrText/text()",
                namespaces={"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"},
            )).split())
    assert instructions.count("PAGE") == 10
    assert instructions.count("NUMPAGES") == 10
