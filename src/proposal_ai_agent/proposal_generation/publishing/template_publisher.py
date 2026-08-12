"""Bind semantic composition content into verified native DOCX prototypes."""

from __future__ import annotations

from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Iterable

from docx import Document

from ..composition import ComponentInstance, CompositionDocument
from ..proposal_ir import (
    BulletList,
    Callout,
    Heading,
    Paragraph,
    RequirementMatrix,
    Table,
    VisualPlaceholder,
)
from .openxml_editor import OpenXmlEditor, assert_footer_contract, file_sha256
from .template_map import (
    ReusableComponent,
    SemanticField,
    TemplateSemanticMap,
)


class UnsupportedTemplateComponent(ValueError):
    """A semantic node has no verified visual representation in the template."""


class MarkdownContentError(ValueError):
    """Semantic content contains presentation syntax instead of typed IR."""


class TemplatePublisher:
    """Publish one composition by cloning and populating audited template nodes."""

    _SUPPORTED_COMPONENTS = frozenset({
        "cover_page", "module_banner", "heading", "appendix", "references",
    })

    def publish(
        self,
        composition: CompositionDocument,
        template_map: TemplateSemanticMap,
        source_template: str | Path,
        output_path: str | Path,
    ) -> Path:
        if not isinstance(composition, CompositionDocument):
            raise TypeError("composition must be a CompositionDocument")
        source = Path(source_template).resolve()
        destination = Path(output_path).resolve()
        if destination.suffix.lower() != ".docx":
            raise ValueError("output_path must have a .docx suffix")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_hash = file_sha256(source)
        self._preflight(composition, template_map)

        with TemporaryDirectory(prefix="phase4b-", dir=destination.parent) as temporary:
            editor = OpenXmlEditor(source, Path(temporary) / "working.docx", template_map)
            prototypes = self._capture_prototypes(editor, template_map)
            self._bind_cover(editor, composition, template_map)
            self._bind_headers(editor, composition, template_map)
            section_slots = editor.clear_dynamic_body(
                start_body_node=template_map.region("cover").end_body_node + 1
            )
            self._render_composition(editor, composition, template_map, prototypes, section_slots)
            editor.assert_footers_preserved(template_map.footer_branding_text)
            result = editor.save(destination)

        reopened = Document(result)
        assert_footer_contract(reopened, template_map.footer_branding_text)
        if file_sha256(source) != source_hash:
            raise RuntimeError("source template changed during publishing")
        return result

    def _capture_prototypes(self, editor: OpenXmlEditor, template_map: TemplateSemanticMap) -> dict:
        components = (
            ReusableComponent.MODULE_BANNER,
            ReusableComponent.BODY_PARAGRAPH,
            ReusableComponent.SECTION_HEADING,
            ReusableComponent.DIAGRAM_PLACEHOLDER,
            ReusableComponent.LEGACY_BOM_TABLE,
            ReusableComponent.ORANGE_BOM_TABLE,
            ReusableComponent.LEGACY_POWER_TABLE,
            ReusableComponent.ORANGE_POWER_TABLE,
        )
        return {
            component: (
                editor.clone_banner(template_map.prototype(component).locator)
                if component is ReusableComponent.MODULE_BANNER
                else editor.clone_body_node(template_map.prototype(component).locator)
            )
            for component in components
        }

    def _bind_cover(
        self, editor: OpenXmlEditor, composition: CompositionDocument, template_map: TemplateSemanticMap,
    ) -> None:
        title = self._plain(composition.title, "composition title")
        values = {
            SemanticField.PROGRAMME_LABEL: title.upper(),
            SemanticField.DOCUMENT_TITLE: title,
            SemanticField.PROJECT_SUBTITLE: self._cover_subtitle(composition) or title,
            SemanticField.DOCUMENT_NUMBER: composition.proposal_id,
            SemanticField.PROGRAMME_NAME: title,
            SemanticField.REFERENCE_NUMBER: "",
            SemanticField.DOCUMENT_DATE: "",
            SemanticField.PREPARED_BY: "",
            SemanticField.STATUS: "",
        }
        for field, value in values.items():
            for locator in template_map.target(field).locators:
                editor.replace_text(locator, value)

    def _bind_headers(
        self, editor: OpenXmlEditor, composition: CompositionDocument, template_map: TemplateSemanticMap,
    ) -> None:
        title = self._plain(composition.title, "header title")
        for locator in template_map.target(SemanticField.HEADER_PROJECT_TEXT).locators:
            editor.replace_drawing_text(locator, title)

    def _render_composition(
        self,
        editor: OpenXmlEditor,
        composition: CompositionDocument,
        template_map: TemplateSemanticMap,
        prototypes: dict,
        section_slots: tuple[object, ...],
    ) -> None:
        body_components = tuple(
            component for component in composition.components
            if component.component_name != "cover_page"
        )
        if not section_slots:
            raise RuntimeError("template provides no preserved section insertion slots")
        slot_indexes = self._spread_indexes(len(body_components), len(section_slots))
        for component, slot_index in zip(body_components, slot_indexes):
            self._render_component(
                editor, component, composition, template_map,
                prototypes, section_slots[slot_index], top_level=True,
            )

    def _render_component(
        self,
        editor: OpenXmlEditor,
        component: ComponentInstance,
        composition: CompositionDocument,
        template_map: TemplateSemanticMap,
        prototypes: dict,
        insertion_point,
        *,
        top_level: bool,
    ) -> None:
        heading = self._component_heading(component)
        if component.component_name == "module_banner":
            banner = editor.clone_detached_node(prototypes[ReusableComponent.MODULE_BANNER])
            locator = template_map.prototype(ReusableComponent.MODULE_BANNER).locator
            editor.replace_cloned_drawing_text(
                banner, locator.drawing_name or "", locator.expected_text,
                self._plain(heading.text, f"heading in {component.section_id}"),
            )
            editor.insert_body_node(insertion_point, banner, "before")
        elif component.component_name in {"heading", "appendix", "references"}:
            heading_node = editor.clone_detached_node(prototypes[ReusableComponent.SECTION_HEADING])
            editor.replace_all_text_in_node(
                heading_node, self._plain(heading.text, f"heading in {component.section_id}")
            )
            editor.insert_body_node(insertion_point, heading_node, "before")
        elif component.component_name == "cover_page":
            return
        else:
            self._unsupported(component, component.component_name, template_map, "component has no binding")

        for slot in component.slots:
            if slot.slot_name == "heading":
                continue
            for content in slot.contents:
                self._render_content(
                    editor, content.content, component, template_map, prototypes, insertion_point
                )

        if component.component_name == "references":
            for reference in composition.references:
                values = tuple(
                    value for value in (
                        reference.reference_id, reference.title, reference.source, reference.locator,
                    ) if value
                )
                self._insert_body_paragraph(
                    editor, prototypes, insertion_point,
                    self._plain(" — ".join(values), f"reference {reference.reference_id}"),
                )
        for child in component.children:
            self._render_component(
                editor, child, composition, template_map, prototypes, insertion_point, top_level=False,
            )

    def _render_content(
        self,
        editor: OpenXmlEditor,
        node,
        component: ComponentInstance,
        template_map: TemplateSemanticMap,
        prototypes: dict,
        insertion_point,
    ) -> None:
        if isinstance(node, Heading):
            heading = editor.clone_detached_node(prototypes[ReusableComponent.SECTION_HEADING])
            editor.replace_all_text_in_node(
                heading, self._plain(node.text, f"heading in {component.section_id}")
            )
            editor.insert_body_node(insertion_point, heading, "before")
        elif isinstance(node, Paragraph):
            self._insert_body_paragraph(
                editor, prototypes, insertion_point,
                self._plain(node.text, f"paragraph in {component.section_id}"),
            )
        elif isinstance(node, BulletList):
            for item in node.items:
                paragraph = editor.clone_detached_node(prototypes[ReusableComponent.BODY_PARAGRAPH])
                editor.replace_all_text_in_node(
                    paragraph, self._plain(item, f"list item in {component.section_id}")
                )
                list_prototype = template_map.prototype(ReusableComponent.BULLET_LIST)
                assert list_prototype.numbering_id is not None
                assert list_prototype.numbering_level is not None
                editor.apply_existing_numbering(
                    paragraph,
                    num_id=list_prototype.numbering_id,
                    level=list_prototype.numbering_level,
                )
                editor.insert_body_node(insertion_point, paragraph, "before")
        elif isinstance(node, Table):
            self._render_table(editor, node, component, template_map, prototypes, insertion_point)
        elif isinstance(node, VisualPlaceholder):
            placeholder = editor.clone_detached_node(prototypes[ReusableComponent.DIAGRAM_PLACEHOLDER])
            locator = template_map.prototype(ReusableComponent.DIAGRAM_PLACEHOLDER).locator
            visual_text = node.description if node.caption is None else f"{node.description}\n{node.caption}"
            editor.replace_cloned_drawing_text(
                placeholder, locator.drawing_name or "", None,
                self._plain(visual_text, f"visual placeholder in {component.section_id}"),
            )
            editor.insert_body_node(insertion_point, placeholder, "before")
        elif isinstance(node, RequirementMatrix):
            self._unsupported(
                component, node.kind, template_map,
                "no verified compliance/requirement-matrix prototype exists",
            )
        elif isinstance(node, Callout):
            self._unsupported(
                component, node.kind, template_map, "no verified callout prototype exists",
            )
        else:
            self._unsupported(component, type(node).__name__, template_map, "unknown semantic node")

    def _render_table(
        self,
        editor: OpenXmlEditor,
        node: Table,
        component: ComponentInstance,
        template_map: TemplateSemanticMap,
        prototypes: dict,
        insertion_point,
    ) -> None:
        matches = tuple(
            prototype for prototype in template_map.reusable_components
            if prototype.column_count == len(node.headers)
        )
        if len(matches) != 1:
            self._unsupported(
                component, node.kind, template_map,
                f"no verified table prototype has {len(node.headers)} columns",
            )
        layout = matches[0]
        if layout.header_row_index is None or not layout.data_row_indexes:
            self._unsupported(component, node.kind, template_map, "table prototype row map is incomplete")
        table = editor.clone_detached_node(prototypes[layout.component])
        editor.populate_table(
            table,
            tuple(self._plain(value, f"table header in {component.section_id}") for value in node.headers),
            tuple(
                tuple(self._plain(value, f"table cell in {component.section_id}") for value in row)
                for row in node.rows
            ),
            header_row_index=layout.header_row_index,
            data_row_index=layout.data_row_indexes[0],
            alternate_data_row_index=(
                layout.data_row_indexes[1] if len(layout.data_row_indexes) > 1 else None
            ),
        )
        editor.insert_body_node(insertion_point, table, "before")

    def _insert_body_paragraph(self, editor, prototypes, insertion_point, text: str) -> None:
        paragraph = editor.clone_detached_node(prototypes[ReusableComponent.BODY_PARAGRAPH])
        editor.replace_all_text_in_node(paragraph, text)
        editor.insert_body_node(insertion_point, paragraph, "before")

    def _preflight(self, composition: CompositionDocument, template_map: TemplateSemanticMap) -> None:
        components = tuple(self._walk_components(composition.components))
        for component in components:
            if component.component_name not in self._SUPPORTED_COMPONENTS:
                self._unsupported(component, component.component_name, template_map, "component has no binding")
            heading = self._component_heading(component)
            self._plain(heading.text, f"heading in {component.section_id}")
            for slot in component.slots:
                for content in slot.contents:
                    node = content.content
                    if slot.slot_name == "heading":
                        if not isinstance(node, Heading):
                            self._unsupported(component, type(node).__name__, template_map, "heading slot is invalid")
                        continue
                    if component.component_name == "cover_page" and not isinstance(node, Paragraph):
                        self._unsupported(
                            component, node.kind, template_map,
                            "cover supports prose only through its verified subtitle field",
                        )
                    if isinstance(node, (RequirementMatrix, Callout)):
                        reason = (
                            "no verified compliance/requirement-matrix prototype exists"
                            if isinstance(node, RequirementMatrix)
                            else "no verified callout prototype exists"
                        )
                        self._unsupported(component, node.kind, template_map, reason)
                    if isinstance(node, Table) and not any(
                        prototype.column_count == len(node.headers)
                        for prototype in template_map.reusable_components
                    ):
                        self._unsupported(
                            component, node.kind, template_map,
                            f"no verified table prototype has {len(node.headers)} columns",
                        )
                    for value in self._node_strings(node):
                        self._plain(value, f"{node.kind} in {component.section_id}")
        has_references = any(component.component_name == "references" for component in components)
        if composition.references and not has_references:
            raise UnsupportedTemplateComponent(
                f"component_type=references section_id=<missing> template={template_map.template_name}: "
                "composition references have no references component"
            )

    @staticmethod
    def _component_heading(component: ComponentInstance) -> Heading:
        matches = tuple(
            content.content
            for slot in component.slots if slot.slot_name == "heading"
            for content in slot.contents if isinstance(content.content, Heading)
        )
        if len(matches) != 1:
            raise UnsupportedTemplateComponent(
                f"component_type={component.component_name} section_id={component.section_id}: "
                "component requires exactly one typed heading"
            )
        return matches[0]

    def _cover_subtitle(self, composition: CompositionDocument) -> str | None:
        paragraphs = tuple(
            content.content.text
            for component in composition.components if component.component_name == "cover_page"
            for slot in component.slots if slot.slot_name != "heading"
            for content in slot.contents if isinstance(content.content, Paragraph)
        )
        if not paragraphs:
            return None
        return "\n".join(self._plain(text, "cover subtitle") for text in paragraphs)

    @staticmethod
    def _node_strings(node) -> tuple[str, ...]:
        if isinstance(node, Heading):
            return (node.text,)
        if isinstance(node, Paragraph):
            return (node.text,)
        if isinstance(node, BulletList):
            return node.items
        if isinstance(node, Table):
            return node.headers + tuple(value for row in node.rows for value in row)
        if isinstance(node, VisualPlaceholder):
            return (node.description,) + ((node.caption,) if node.caption else ())
        if isinstance(node, Callout):
            return (node.label, node.text)
        if isinstance(node, RequirementMatrix):
            return tuple(
                value for entry in node.entries
                for value in (entry.requirement_id, entry.requirement, entry.response)
            )
        return ()

    @staticmethod
    def _plain(value: str, location: str) -> str:
        markdown = re.compile(r"(?m)^\s{0,3}(?:#{1,6}|[-*])\s+")
        if markdown.search(value):
            raise MarkdownContentError(f"raw Markdown is not allowed in {location}")
        return value

    @staticmethod
    def _walk_components(components: Iterable[ComponentInstance]):
        for component in components:
            yield component
            yield from TemplatePublisher._walk_components(component.children)

    @staticmethod
    def _spread_indexes(item_count: int, slot_count: int) -> tuple[int, ...]:
        if item_count == 0:
            return ()
        if item_count == 1:
            return (slot_count - 1,)
        return tuple(
            round(index * (slot_count - 1) / (item_count - 1))
            for index in range(item_count)
        )

    @staticmethod
    def _unsupported(
        component: ComponentInstance,
        node_type: str,
        template_map: TemplateSemanticMap,
        reason: str,
    ) -> None:
        raise UnsupportedTemplateComponent(
            f"component_type={node_type} section_id={component.section_id} "
            f"component_id={component.component_id} template={template_map.template_name}: {reason}"
        )


__all__ = ["MarkdownContentError", "TemplatePublisher", "UnsupportedTemplateComponent"]
