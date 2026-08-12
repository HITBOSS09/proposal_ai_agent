"""Theme- and renderer-independent semantic component registry."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from .document_plan import SectionRole


class BaseComponentDefinition(BaseModel):
    """Immutable semantic capability declaration for one reusable component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_name: str
    supported_section_roles: tuple[SectionRole, ...]
    required_block_types: tuple[str, ...]
    optional_block_types: tuple[str, ...]
    allows_children: bool
    supports_numbering: bool
    supports_page_break: bool


class CoverPageComponent(BaseComponentDefinition):
    """Semantic cover-page component definition."""

    component_name: str = "cover_page"
    supported_section_roles: tuple[SectionRole, ...] = (SectionRole.COVER,)
    required_block_types: tuple[str, ...] = ()
    optional_block_types: tuple[str, ...] = ()
    allows_children: bool = False
    supports_numbering: bool = False
    supports_page_break: bool = False


class TOCComponent(BaseComponentDefinition):
    """Semantic table-of-contents component definition."""

    component_name: str = "table_of_contents"
    supported_section_roles: tuple[SectionRole, ...] = (SectionRole.TABLE_OF_CONTENTS,)
    required_block_types: tuple[str, ...] = ()
    optional_block_types: tuple[str, ...] = ()
    allows_children: bool = False
    supports_numbering: bool = False
    supports_page_break: bool = True


class ModuleBannerComponent(BaseComponentDefinition):
    """Semantic module-boundary component definition."""

    component_name: str = "module_banner"
    supported_section_roles: tuple[SectionRole, ...] = (SectionRole.MODULE,)
    required_block_types: tuple[str, ...] = ()
    optional_block_types: tuple[str, ...] = ()
    allows_children: bool = True
    supports_numbering: bool = False
    supports_page_break: bool = True


class HeadingComponent(BaseComponentDefinition):
    """Semantic body-section container and heading component definition."""

    component_name: str = "heading"
    supported_section_roles: tuple[SectionRole, ...] = (SectionRole.BODY,)
    required_block_types: tuple[str, ...] = ()
    optional_block_types: tuple[str, ...] = (
        "paragraph",
        "bullet_list",
        "table",
        "requirement_matrix",
        "visual_placeholder",
        "callout",
    )
    allows_children: bool = True
    supports_numbering: bool = True
    supports_page_break: bool = True


class ParagraphComponent(BaseComponentDefinition):
    """Reusable semantic paragraph component definition."""

    component_name: str = "paragraph"
    supported_section_roles: tuple[SectionRole, ...] = ()
    required_block_types: tuple[str, ...] = ("paragraph",)
    optional_block_types: tuple[str, ...] = ()
    allows_children: bool = False
    supports_numbering: bool = False
    supports_page_break: bool = False


class BulletListComponent(BaseComponentDefinition):
    """Reusable semantic bullet-list component definition."""

    component_name: str = "bullet_list"
    supported_section_roles: tuple[SectionRole, ...] = ()
    required_block_types: tuple[str, ...] = ("bullet_list",)
    optional_block_types: tuple[str, ...] = ()
    allows_children: bool = False
    supports_numbering: bool = False
    supports_page_break: bool = False


class TableComponent(BaseComponentDefinition):
    """Reusable semantic table component definition."""

    component_name: str = "table"
    supported_section_roles: tuple[SectionRole, ...] = ()
    required_block_types: tuple[str, ...] = ("table",)
    optional_block_types: tuple[str, ...] = ()
    allows_children: bool = False
    supports_numbering: bool = False
    supports_page_break: bool = False


class RequirementMatrixComponent(BaseComponentDefinition):
    """Reusable semantic requirement-matrix component definition."""

    component_name: str = "requirement_matrix"
    supported_section_roles: tuple[SectionRole, ...] = ()
    required_block_types: tuple[str, ...] = ("requirement_matrix",)
    optional_block_types: tuple[str, ...] = ()
    allows_children: bool = False
    supports_numbering: bool = False
    supports_page_break: bool = False


class VisualPlaceholderComponent(BaseComponentDefinition):
    """Reusable semantic visual-placeholder component definition."""

    component_name: str = "visual_placeholder"
    supported_section_roles: tuple[SectionRole, ...] = ()
    required_block_types: tuple[str, ...] = ("visual_placeholder",)
    optional_block_types: tuple[str, ...] = ()
    allows_children: bool = False
    supports_numbering: bool = False
    supports_page_break: bool = False


class CalloutComponent(BaseComponentDefinition):
    """Reusable semantic callout component definition."""

    component_name: str = "callout"
    supported_section_roles: tuple[SectionRole, ...] = ()
    required_block_types: tuple[str, ...] = ("callout",)
    optional_block_types: tuple[str, ...] = ()
    allows_children: bool = False
    supports_numbering: bool = False
    supports_page_break: bool = False


class AppendixComponent(BaseComponentDefinition):
    """Semantic appendix and annex component definition."""

    component_name: str = "appendix"
    supported_section_roles: tuple[SectionRole, ...] = (SectionRole.APPENDIX, SectionRole.ANNEX)
    required_block_types: tuple[str, ...] = ()
    optional_block_types: tuple[str, ...] = (
        "paragraph",
        "bullet_list",
        "table",
        "requirement_matrix",
        "visual_placeholder",
        "callout",
    )
    allows_children: bool = True
    supports_numbering: bool = True
    supports_page_break: bool = True


class ReferenceComponent(BaseComponentDefinition):
    """Semantic reference-section component definition."""

    component_name: str = "references"
    supported_section_roles: tuple[SectionRole, ...] = (SectionRole.REFERENCES,)
    required_block_types: tuple[str, ...] = ()
    optional_block_types: tuple[str, ...] = ()
    allows_children: bool = False
    supports_numbering: bool = False
    supports_page_break: bool = True


class DuplicateComponentRegistration(ValueError):
    """A semantic role was registered to more than one component definition."""


class ComponentRoleNotRegistered(LookupError):
    """No component definition is registered for the requested section role."""


class ComponentRegistry:
    """Resolve semantic section roles to reusable component definitions."""

    def __init__(self, definitions: Iterable[BaseComponentDefinition] = ()) -> None:
        self._definitions_by_role: dict[SectionRole, BaseComponentDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: BaseComponentDefinition) -> None:
        """Register one definition for each of its supported semantic roles."""
        if not isinstance(definition, BaseComponentDefinition):
            raise TypeError("definition must be a BaseComponentDefinition")
        for role in definition.supported_section_roles:
            if role in self._definitions_by_role:
                raise DuplicateComponentRegistration(f"component already registered for role: {role.value}")
        for role in definition.supported_section_roles:
            self._definitions_by_role[role] = definition

    def resolve(self, role: SectionRole) -> BaseComponentDefinition:
        """Return the definition registered for one semantic section role."""
        try:
            return self._definitions_by_role[role]
        except KeyError as error:
            role_name = role.value if isinstance(role, SectionRole) else str(role)
            raise ComponentRoleNotRegistered(f"no component registered for role: {role_name}") from error

    @classmethod
    def standard(cls) -> "ComponentRegistry":
        """Build the registry containing the compiler's standard role mappings."""
        return cls(
            (
                CoverPageComponent(),
                TOCComponent(),
                ModuleBannerComponent(),
                HeadingComponent(),
                AppendixComponent(),
                ReferenceComponent(),
            )
        )
