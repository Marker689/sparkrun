"""Per-machine container image resolution.

A recipe normally names one ``container:`` and every node runs it.  A recipe
that serves pre-optimized, machine-tuned images instead declares a
``containers:`` block binding an image to a **hostname**:

.. code-block:: yaml

    container: nvcr.io/nvidia/vllm:25.09        # fallback for unlisted machines
    containers:
      - image: myorg/vllm-spark:node-01
        host: spark-01
      - image: myorg/vllm-spark:node-02
        host: spark-02

Keying by hostname rather than by rank or node index is deliberate.  The image
is a property of the *machine*, so a rank-indexed map would be silently wrong
the moment the scheduler ordered hosts differently — and "silently wrong image"
is the entire failure mode this feature has to avoid.  Hostnames are also
declarative, which is what lets :func:`generate_intent_id` hash the map without
becoming placement-dependent (see :attr:`ImagePlan.declared`).

Note this is *not* spelled inside ``layout:``.  ``RecipeLayout.placements`` is
honored **verbatim** by every scheduler, so putting the image there would pin
placement as a side effect of naming an image — wrong when the tuned images
exist on every cluster machine and a ``--tp 2`` launch should still be free to
pick the two idlest.  A recipe that wants both declares ``layout:`` as well.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from collections.abc import Mapping, Sequence

if TYPE_CHECKING:
    from sparkrun.core.recipe import DistributionContainerEntry, Recipe
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.runtimes.base import RuntimePlugin

logger = logging.getLogger(__name__)


class ImagePlanError(ValueError):
    """A ``containers:`` block that cannot be resolved to a runnable plan.

    Always a recipe/cluster mismatch the user must fix — a typo'd hostname, a
    duplicate entry, or a machine with neither an entry nor a ``container:``
    fallback. Also raised when a runtime has no default image.
    """


def parse_container_entries(raw: Any) -> list[dict[str, str]]:
    """Normalize a raw recipe ``containers:`` value to ``[{host, image}, …]``.

    Permissive by design (the recipe loader parses without a cluster in hand);
    every real validation happens in :func:`resolve_image_plan`, which is the
    only place the cluster's host list is known.  Entries missing either key are
    dropped here rather than raised on, so a malformed block surfaces as the
    actionable "machine X has no image" from the resolver instead of a parse
    error that cannot name the machine.
    """
    if not isinstance(raw, list):
        return []
    entries: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        host = str(item.get("host") or "").strip()
        image = str(item.get("image") or "").strip()
        if not host or not image:
            logger.warning("Ignoring containers: entry missing host or image: %r", item)
            continue
        entries.append({"host": host, "image": image})
    return entries


@dataclass(frozen=True)
class ImagePlan:
    """Which container image each node of a launch runs.

    Two maps are kept deliberately, because they answer different questions and
    conflating them breaks workload identity:

    * :attr:`declared` is what the *recipe* said — sorted ``(host, image)``
      pairs, independent of which hosts this launch happens to use.  This is
      what :func:`~sparkrun.orchestration.job_metadata.generate_intent_id`
      hashes; hashing the resolved map instead would make the intent depend on
      placement, so ``stop`` / ``logs`` / ``--ensure`` would stop matching a
      workload whenever the scheduler picked a different host subset.
    * :attr:`images_by_node` is what this launch actually runs, positionally
      aligned with the resolved host list.  This drives container launch and
      the distribution derivation.
    """

    default_image: str
    """Common fallback used during resolution, or empty when none applies.

    The selected images are in ``images_by_node``; use ``head_image()`` for
    the first host's actual image, which can differ from this fallback.
    """

    declared: tuple[tuple[str, str], ...] = ()
    """Sorted ``(host, image)`` pairs exactly as declared.  Empty ⇒ no
    ``containers:`` block; hardware defaults can still differ per host."""

    images_by_node: tuple[str, ...] = ()
    """Image per node, aligned with the resolved host list."""

    @property
    def heterogeneous(self) -> bool:
        """True when this launch runs more than one distinct image."""
        return len(set(self.images_by_node)) > 1

    @property
    def distinct(self) -> tuple[str, ...]:
        """Distinct images in use, in first-node order (stable for logging)."""
        seen: dict[str, None] = {}
        for img in self.images_by_node:
            seen.setdefault(img, None)
        return tuple(seen)

    def image_for_node(self, index: int) -> str:
        """Image for a resolved node; invalid indices are programming errors."""
        if not 0 <= index < len(self.images_by_node):
            raise IndexError("image plan has no node at index %s" % index)
        return self.images_by_node[index]

    def head_image(self) -> str:
        """Image the head node runs, for summaries and single-host operations."""
        return self.images_by_node[0] if self.images_by_node else self.default_image


def resolve_image_plan(
    recipe: "Recipe",
    default_image: str,
    host_list: list[str],
    cluster_hosts: list[str] | None = None,
    *,
    fallback_images: Mapping[str, str] | None = None,
) -> ImagePlan:
    """Resolve a recipe's images against the hosts this launch will use.

    Args:
        recipe: Loaded recipe; ``recipe.containers`` supplies the declarations.
        default_image: Resolved ``container:`` — *after* the builder phase, so a
            builder that rewrote the image ref is reflected.
        host_list: Hosts this launch was placed on, in rank order.
        cluster_hosts: The cluster's full host list, used to validate declared
            hostnames.  ``None`` skips that check (explicit ``--hosts`` runs,
            where there is no cluster definition to check against).
        fallback_images: Hardware-resolved fallback per host, before any builder.
            An explicit per-host empty value means no default exists there.

    Raises:
        ImagePlanError: A declared host is not in the cluster, a host is
            declared twice, a selected host has neither an entry nor a
            ``container:`` fallback, or an image is not a valid reference.
    """
    declared_raw = getattr(recipe, "containers", None) or []

    by_host: dict[str, str] = {}
    for entry in declared_raw:
        host = entry["host"]
        if host in by_host:
            raise ImagePlanError(
                "Recipe declares container image for host '%s' more than once. "
                "Machine-specific images must be unambiguous; remove the duplicate." % host
            )
        by_host[host] = entry["image"]

    # A typo'd hostname would otherwise fall through to the generic `container:`
    # and silently run an untuned image on a machine the user believed covered.
    # Checked against the *cluster's* hosts, not the selected subset, so
    # declaring more machines than a given launch uses stays legal (and is the
    # expected case — the block usually covers the whole cluster).
    if cluster_hosts:
        unknown = sorted(set(by_host) - set(cluster_hosts))
        if unknown:
            raise ImagePlanError(
                "Recipe declares container images for host(s) not in this cluster: %s. "
                "Known hosts: %s." % (", ".join(unknown), ", ".join(cluster_hosts))
            )

    images: list[str] = []
    missing: list[str] = []
    fell_back: list[str] = []
    for host in host_list:
        img = by_host.get(host)
        if img is None:
            fallback = fallback_images.get(host, default_image) if fallback_images is not None else default_image
            if not fallback:
                missing.append(host)
                continue
            fell_back.append(host)
            img = fallback
        images.append(img)

    if missing:
        raise ImagePlanError(
            "No container image for host(s) %s: they have no `containers:` entry and the "
            "recipe/runtime supplies no fallback. Set `container:` or a per-host image." % ", ".join(missing)
        )

    from sparkrun.utils.shell import validate_image_ref

    for img in dict.fromkeys(images):
        try:
            validate_image_ref(img)
        except ValueError as exc:
            raise ImagePlanError(str(exc)) from exc

    if fell_back and declared_raw:
        # Never silent: on a machine-tuned cluster, running the generic image is
        # a material difference the user should see without --verbose.
        logger.info(
            "Host(s) %s have no machine-specific image; using their resolved defaults",
            ", ".join(fell_back),
        )

    return ImagePlan(
        default_image=default_image,
        declared=tuple(sorted((h, i) for h, i in by_host.items())),
        images_by_node=tuple(images),
    )


def derive_container_entries(plan: ImagePlan, host_list: list[str]) -> list["DistributionContainerEntry"]:
    """Derive distribution entries from a resolved plan.

    One entry per *distinct* image, targeting the node indices that run it.
    This is what keeps "what to ship" and "what to run" from ever disagreeing:
    the distribution view is derived from the launch view rather than being
    declared alongside it.  A hand-written ``distribution_config.containers``
    still wins — the caller checks its ``explicit`` flag before applying this.
    """
    from sparkrun.core.recipe import DistributionContainerEntry

    if len(plan.images_by_node) != len(host_list):
        raise ImagePlanError("image distribution requires one image per host")
    targets: dict[str, list[int]] = {}
    for index, image in enumerate(plan.images_by_node):
        targets.setdefault(image, []).append(index)

    return [DistributionContainerEntry(name=image, target=indices) for image, indices in targets.items()]


def resolve_runtime_image_plan(
    recipe: Recipe,
    runtime: RuntimePlugin,
    hosts: list[str],
    *,
    cluster: ClusterDefinition | None = None,
    host_hardware=None,
    images_by_node: Sequence[str] | None = None,
) -> ImagePlan:
    """Resolve explicit images and each selected host's runtime/platform default.

    This is a read-only plan: builders and distribution consume its images
    without changing the recipe declarations used for workload identity.
    Prepared references replace declarations/defaults entirely. Their final
    plan obeys the same host alignment and runtime constraints.
    """
    if images_by_node is not None:
        if isinstance(images_by_node, (str, bytes)):
            raise ImagePlanError("prepared images must be a sequence of image references")
        plan = ImagePlan(default_image="", images_by_node=tuple(images_by_node))
        return validate_runtime_image_plan(plan, runtime, hosts)
    from sparkrun.core.hardware import resolve_host_hardware

    declared_hosts = {entry["host"] for entry in (getattr(recipe, "containers", None) or [])}
    fallbacks = {}
    for host in hosts:
        if recipe.container or host in declared_hosts:
            fallbacks[host] = recipe.container
        else:
            fallbacks[host] = runtime.resolve_container(
                recipe, host_hardware=(host_hardware[host] if host_hardware is not None else resolve_host_hardware([host], cluster)[host])
            )
    plan = resolve_image_plan(
        recipe,
        next(iter(fallbacks.values()), "") if len(set(fallbacks.values())) == 1 else "",
        hosts,
        cluster_hosts=list(cluster.hosts) if cluster is not None else None,
        fallback_images=fallbacks,
    )
    return validate_runtime_image_plan(plan, runtime, hosts)


def validate_runtime_image_plan(plan: ImagePlan, runtime: RuntimePlugin, hosts: Sequence[str]) -> ImagePlan:
    """Normalize the selected references and enforce the final launch contract."""
    from dataclasses import replace

    if not hosts or len(plan.images_by_node) != len(hosts):
        raise ImagePlanError("image plan has %d image(s) for %d host(s); expected one per host" % (len(plan.images_by_node), len(hosts)))
    if any(not isinstance(image, str) or not image.strip() for image in plan.images_by_node):
        raise ImagePlanError("image references must be non-empty strings")
    plan = replace(plan, images_by_node=tuple(image.strip() for image in plan.images_by_node))
    if plan.heterogeneous and not runtime.supports_heterogeneous_images:
        raise ImagePlanError(
            "Runtime '%s' requires one image across all hosts; set an explicit `container:` image compatible with every host"
            % runtime.runtime_name
        )
    return plan
