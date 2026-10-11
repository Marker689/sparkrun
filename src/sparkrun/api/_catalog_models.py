"""Public catalog payload types, re-exported by :mod:`sparkrun.api`.

These describe ordinary dictionaries, not runtime validators or constructors
with defaults. Unknown observations use None; NotRequired keys may be absent.
Consumers should tolerate additional response fields. See docs/CATALOG_API.md.
"""

from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

from sparkrun.core.recipe import Recipe

CatalogFacet: TypeAlias = Literal["min_nodes", "tp", "pp", "quantization", "context_length", "parameters_b"]
"""Filterable declared metadata; filter values use their string representation."""

ResolvedCatalogRecipe: TypeAlias = tuple[Recipe, dict[str, Any]]
"""Recipe and normalized launch overrides; not a JSON payload or a launch plan."""


class CatalogBenchmarkContext(TypedDict, total=False):
    """Optional recipe-declared context, not measurements made by browsing."""

    output_tokens_per_second: int | float
    time_to_first_token_ms: int | float
    input_tokens: int | float
    output_tokens: int | float
    concurrency: int | float
    hardware: str
    runtime: str
    date: str
    description: str


class CatalogRecipeMetadata(TypedDict, total=False):
    """Declared facets; an unreadable/oversize metadata document yields {}."""

    min_nodes: int | float | None
    tp: int | float | None
    pp: int | float | None
    quantization: str | None
    context_length: int | float | None
    parameters_b: int | float | None
    benchmarks: list[CatalogBenchmarkContext]


class CatalogRecipe(TypedDict):
    """One browse row. Metadata facets are flat; details nest them instead."""

    reference: str
    source_path: str
    name: str | None
    model: str | None
    runtime: str | None
    description: str | None
    registry: str | None
    # Lightweight summaries retain raw values if bounded metadata cannot be
    # read. Consumers must narrow these two fields before numeric use.
    min_nodes: object
    tp: object
    pp: NotRequired[int | float | None]
    quantization: NotRequired[str | None]
    context_length: NotRequired[int | float | None]
    parameters_b: NotRequired[int | float | None]
    benchmarks: NotRequired[list[CatalogBenchmarkContext]]


class CatalogPage(TypedDict):
    """Filtered result page; facets describe candidates before facet filtering."""

    recipes: list[CatalogRecipe]
    facets: dict[CatalogFacet, list[str]]
    total: int
    next_offset: int | None
    unavailable_registries: list[str]


class CatalogIssue(TypedDict):
    """Validation or trust diagnosis. Short catalog issues omit remediation fields."""

    severity: str
    code: str
    message: str
    summary: NotRequired[str]
    fix: NotRequired[str]
    deprecation: NotRequired[bool]


class CatalogRecipeDetails(TypedDict):
    """Preview of an exact selection; revision and trust are observations."""

    reference: str
    source_path: str
    name: str
    registry: str | None
    model: str
    hf_model: str
    runtime: str
    description: str
    min_nodes: int
    # Selected raw recipe defaults; validate/narrow before treating as numbers.
    defaults: dict[str, object]
    metadata: CatalogRecipeMetadata
    recipe_revision: str
    plugin_items: dict[str, Any]
    native_api_options: list[str]
    native_protocols: list[str]
    capabilities: list[str]
    required_plugins: list[str]
    available_plugins: list[str]
    trusted: bool
    issues: list[CatalogIssue]


class CatalogRegistry(TypedDict):
    """Configured registry inventory; cached does not imply current or trusted."""

    name: str
    enabled: bool
    visible: bool
    trusted: bool
    cached: bool


class CatalogCluster(TypedDict):
    """Configured cluster inventory, without a live probe."""

    name: str
    description: str
    host_count: int
    hosts: NotRequired[list[str]]
    default: bool


class CatalogHostCapacity(TypedDict):
    """Advisory host observation; missing occupancy is unknown, not zero."""

    host: str
    reachable: bool
    free_slots: int | None
    used_slots: int | None
    workloads: int | None


class CatalogCapacity(TypedDict):
    """Live advisory snapshot; does not reserve capacity or ensure full coverage."""

    cluster: str
    observed_at: float
    hosts: list[CatalogHostCapacity]
    advisory: Literal[True]


class CatalogRefreshResult(TypedDict):
    """Refresh outcomes for attempted registries, retaining individual failures."""

    updated: dict[str, bool]
    failed: list[str]


class CatalogRegistryResult(TypedDict):
    """Inventory after an explicit registry configuration operation."""

    registries: list[CatalogRegistry]
