"""Versioned semantic locations and reusable prototypes for DOCX templates."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


PRU_TEMPLATE_SHA256 = "e81d6c34362fb0e7d17b8a82e5e756ad696f98d61e90100e3ca5b7bec7182d55"


class Story(str, Enum):
    BODY = "body"
    HEADER = "header"
    FOOTER = "footer"


class ElementKind(str, Enum):
    PARAGRAPH = "paragraph"
    TABLE_CELL = "table_cell"
    DRAWING_TEXT = "drawing_text"
    BODY_REGION = "body_region"


class SemanticField(str, Enum):
    PROGRAMME_LABEL = "PROGRAMME_LABEL"
    DOCUMENT_TITLE = "DOCUMENT_TITLE"
    PROJECT_SUBTITLE = "PROJECT_SUBTITLE"
    DOCUMENT_NUMBER = "DOCUMENT_NUMBER"
    PROGRAMME_NAME = "PROGRAMME_NAME"
    REFERENCE_NUMBER = "REFERENCE_NUMBER"
    DOCUMENT_DATE = "DOCUMENT_DATE"
    PREPARED_BY = "PREPARED_BY"
    STATUS = "STATUS"
    HEADER_PROJECT_TEXT = "HEADER_PROJECT_TEXT"


class ReusableComponent(str, Enum):
    MODULE_BANNER = "MODULE_BANNER"
    LEGACY_BOM_TABLE = "LEGACY_BOM_TABLE"
    ORANGE_BOM_TABLE = "ORANGE_BOM_TABLE"
    LEGACY_POWER_TABLE = "LEGACY_POWER_TABLE"
    ORANGE_POWER_TABLE = "ORANGE_POWER_TABLE"
    BODY_PARAGRAPH = "BODY_PARAGRAPH"
    SECTION_HEADING = "SECTION_HEADING"
    BULLET_LIST = "BULLET_LIST"
    DIAGRAM_PLACEHOLDER = "DIAGRAM_PLACEHOLDER"


class StructuralLocator(BaseModel):
    """A fail-closed structural signature for one existing template target."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    story: Story
    element_kind: ElementKind
    part_name: str | None = None
    section_index: int | None = Field(default=None, ge=0)
    body_node_index: int | None = Field(default=None, ge=0)
    table_index: int | None = Field(default=None, ge=0)
    row_index: int | None = Field(default=None, ge=0)
    cell_index: int | None = Field(default=None, ge=0)
    drawing_name: str | None = None
    surrounding_label: str | None = None
    expected_text: str | None = None

    @model_validator(mode="after")
    def validate_coordinates(self) -> "StructuralLocator":
        if self.story is Story.BODY and self.part_name is not None:
            raise ValueError("body locators must not specify part_name")
        if self.story is not Story.BODY and not self.part_name:
            raise ValueError("header/footer locators require part_name")
        if self.element_kind is ElementKind.TABLE_CELL:
            if None in (self.table_index, self.row_index, self.cell_index):
                raise ValueError("table-cell locators require table, row, and cell indexes")
        if self.element_kind is ElementKind.DRAWING_TEXT and not self.drawing_name:
            raise ValueError("drawing-text locators require drawing_name")
        return self


class SemanticTarget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    field: SemanticField
    locators: tuple[StructuralLocator, ...]


class ComponentPrototype(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    component: ReusableComponent
    locator: StructuralLocator
    column_count: int | None = Field(default=None, ge=1)
    header_row_index: int | None = Field(default=None, ge=0)
    data_row_indexes: tuple[int, ...] = ()
    numbering_id: int | None = Field(default=None, ge=0)
    numbering_level: int | None = Field(default=None, ge=0)


class DocumentRegion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    start_body_node: int = Field(ge=0)
    end_body_node: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "DocumentRegion":
        if self.end_body_node < self.start_body_node:
            raise ValueError("document region end precedes start")
        return self


class TemplateSemanticMap(BaseModel):
    """Minimal metadata contract for one audited template version."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    template_name: str
    template_path: Path
    template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    map_version: str
    footer_branding_text: str
    dynamic_fields: tuple[SemanticTarget, ...]
    reusable_components: tuple[ComponentPrototype, ...]
    document_regions: tuple[DocumentRegion, ...]

    def target(self, field: SemanticField) -> SemanticTarget:
        matches = tuple(target for target in self.dynamic_fields if target.field is field)
        if len(matches) != 1:
            raise KeyError(f"semantic field is not uniquely mapped: {field.value}")
        return matches[0]

    def prototype(self, component: ReusableComponent) -> ComponentPrototype:
        matches = tuple(item for item in self.reusable_components if item.component is component)
        if len(matches) != 1:
            raise KeyError(f"component is not uniquely mapped: {component.value}")
        return matches[0]

    def region(self, name: str) -> DocumentRegion:
        matches = tuple(region for region in self.document_regions if region.name == name)
        if len(matches) != 1:
            raise KeyError(f"document region is not uniquely mapped: {name}")
        return matches[0]


def pru_template_semantic_map(
    template_path: str | Path = "documents/PRU_T72_Module_Breakdown.docx",
) -> TemplateSemanticMap:
    """Return the Phase 3-audited semantic map for the frozen PRU template."""

    cell_fields = (
        (SemanticField.DOCUMENT_NUMBER, 0, "Document No.", None),
        (SemanticField.PROGRAMME_NAME, 1, "Programme", "SENTINEL ACUS — Autonomous Combat Upgrade System"),
        (SemanticField.REFERENCE_NUMBER, 2, "Reference", None),
        (SemanticField.DOCUMENT_DATE, 3, "Date", "17 July 2026"),
        (SemanticField.PREPARED_BY, 4, "Prepared by", "Bharati Defence and Infrastructure Limited (BDIL)"),
        (SemanticField.STATUS, 5, "Status", "Draft — For Internal Review"),
    )
    targets = [
        SemanticTarget(
            field=field,
            locators=(StructuralLocator(
                story=Story.BODY, element_kind=ElementKind.TABLE_CELL,
                table_index=0, row_index=row, cell_index=1,
                surrounding_label=label, expected_text=expected,
            ),),
        )
        for field, row, label, expected in cell_fields
    ]
    cover_lines = (
        (SemanticField.PROGRAMME_LABEL, "ADITI 4.0 · AUTONOMOUS T-72 PROJECT"),
        (SemanticField.DOCUMENT_TITLE, "PRODUCT REQUIREMENT UNIT (PRU)"),
        (
            SemanticField.PROJECT_SUBTITLE,
            "Conversion of In-Service T-72 into an Optionally-Manned, AI-Enabled Autonomous AFV Platform",
        ),
    )
    targets.extend(
        SemanticTarget(
            field=field,
            locators=(StructuralLocator(
                story=Story.BODY, element_kind=ElementKind.DRAWING_TEXT,
                body_node_index=4, drawing_name="Textbox 5", expected_text=text,
            ),),
        )
        for field, text in cover_lines
    )
    header_text = "ADITI 4.0 · Autonomous T-72 Project — PRU Module Breakdown"
    targets.append(SemanticTarget(
        field=SemanticField.HEADER_PROJECT_TEXT,
        locators=tuple(
            StructuralLocator(
                story=Story.HEADER, element_kind=ElementKind.DRAWING_TEXT,
                part_name=f"word/header{index}.xml", drawing_name=f"Textbox {drawing_id}",
                expected_text=header_text,
            )
            for index, drawing_id in ((1, 1), (2, 9), (3, 22), (4, 33), (5, 38))
        ),
    ))
    prototypes = (
        ComponentPrototype(component=ReusableComponent.MODULE_BANNER, locator=StructuralLocator(
            story=Story.BODY, element_kind=ElementKind.DRAWING_TEXT,
            body_node_index=10, drawing_name="Textbox 6",
            expected_text="MODULE 1 | DRIVE-BY-WIRE (DBW) SYSTEM",
        )),
        ComponentPrototype(component=ReusableComponent.LEGACY_BOM_TABLE, locator=StructuralLocator(
            story=Story.BODY, element_kind=ElementKind.BODY_REGION, body_node_index=31, table_index=2,
        ), column_count=4, header_row_index=1, data_row_indexes=(2,)),
        ComponentPrototype(component=ReusableComponent.ORANGE_BOM_TABLE, locator=StructuralLocator(
            story=Story.BODY, element_kind=ElementKind.BODY_REGION, body_node_index=140, table_index=13,
        ), column_count=5, header_row_index=0, data_row_indexes=(1, 2)),
        ComponentPrototype(component=ReusableComponent.LEGACY_POWER_TABLE, locator=StructuralLocator(
            story=Story.BODY, element_kind=ElementKind.BODY_REGION, body_node_index=183, table_index=19,
        ), column_count=3, header_row_index=0, data_row_indexes=(1,)),
        ComponentPrototype(component=ReusableComponent.ORANGE_POWER_TABLE, locator=StructuralLocator(
            story=Story.BODY, element_kind=ElementKind.BODY_REGION, body_node_index=136, table_index=12,
        ), column_count=6, header_row_index=0, data_row_indexes=(1, 2)),
        ComponentPrototype(component=ReusableComponent.BODY_PARAGRAPH, locator=StructuralLocator(
            story=Story.BODY, element_kind=ElementKind.PARAGRAPH, body_node_index=13,
            expected_text="The Drive-by-Wire system is built around a Vehicle Control Interface Unit",
        )),
        ComponentPrototype(component=ReusableComponent.SECTION_HEADING, locator=StructuralLocator(
            story=Story.BODY, element_kind=ElementKind.PARAGRAPH, body_node_index=12,
            expected_text="Integration",
        )),
        ComponentPrototype(component=ReusableComponent.BULLET_LIST, locator=StructuralLocator(
            story=Story.BODY, element_kind=ElementKind.PARAGRAPH, body_node_index=13,
            expected_text="The Drive-by-Wire system is built around a Vehicle Control Interface Unit",
        ), numbering_id=1, numbering_level=3),
        ComponentPrototype(component=ReusableComponent.DIAGRAM_PLACEHOLDER, locator=StructuralLocator(
            story=Story.BODY, element_kind=ElementKind.DRAWING_TEXT,
            body_node_index=132, drawing_name="Textbox 32",
            expected_text="[ SPACE RESERVED FOR LINE / INTEGRATION DIAGRAM ]",
        )),
    )
    return TemplateSemanticMap(
        template_name="PRU T-72 Module Breakdown",
        template_path=Path(template_path),
        template_sha256=PRU_TEMPLATE_SHA256,
        map_version="1.0",
        footer_branding_text="Bharati Defence and Infrastructure Limited",
        dynamic_fields=tuple(targets),
        reusable_components=prototypes,
        document_regions=(
            DocumentRegion(name="cover", start_body_node=0, end_body_node=8),
            DocumentRegion(name="module_1", start_body_node=9, end_body_node=41),
            DocumentRegion(name="module_5_prototype", start_body_node=123, end_body_node=141),
        ),
    )


__all__ = [
    "ComponentPrototype", "DocumentRegion", "ElementKind", "PRU_TEMPLATE_SHA256",
    "ReusableComponent", "SemanticField", "SemanticTarget", "Story",
    "StructuralLocator", "TemplateSemanticMap", "pru_template_semantic_map",
]
