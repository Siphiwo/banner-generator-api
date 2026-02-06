from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

from app.api.v1.schemas import JobStatus, OutputSize


def utcnow() -> datetime:
    """Return an explicit, timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class Region:
    """
    Simple rectangular region on the banner with an optional semantic label.

    This is used for faces, text blocks, logo/product regions, and any other
    rectangular areas that we want to reason about during layout planning.
    """

    x: int
    y: int
    width: int
    height: int
    # Confidence score in the range [0, 1]. For detectors that do not expose a
    # natural score (e.g. classical CV), this will be set to 1.0.
    score: float = 1.0
    # Optional semantic label such as "face", "text", "logo", or "product".
    label: str | None = None


@dataclass(slots=True)
class BannerAnalysis:
    """
    Content-aware analysis of the master banner image.

    This is the internal "banner model" that later stages (layout planning,
    risk scoring, generation) will consume. It is intentionally decoupled from
    API schemas so we can evolve it freely.
    """

    width: int
    height: int
    # Detected semantic regions.
    faces: List[Region] = field(default_factory=list)
    text_regions: List[Region] = field(default_factory=list)
    logo_regions: List[Region] = field(default_factory=list)
    # Paths to derived masks/maps stored alongside the master banner:
    # - foreground_mask_path: binary foreground/background mask
    # - saliency_map_path: grayscale saliency heatmap
    # - protection_mask_path: binary mask of "protected" content (faces/text/logos)
    foreground_mask_path: str | None = None
    saliency_map_path: str | None = None
    protection_mask_path: str | None = None


@dataclass(slots=True)
class AssetAlignment:
    """
    Alignment decision for a single optional user-uploaded asset.

    This captures where and how an asset should be placed relative to the
    analyzed banner content. It is internal-only and can later be influenced
    by explicit user intent flags coming from the API.
    """

    asset_path: str
    # High-level semantic role, e.g. "logo", "product", "text-overlay".
    role: str = "logo"
    # Recommended placement region on the banner.
    target_region: Region | None = None
    # Whether this placement should be treated as protected in later stages.
    locked: bool = True
    # Reserved flag to indicate that this placement was explicitly requested
    # by the user rather than inferred automatically.
    user_override: bool = False


@dataclass(slots=True)
class Job:
    """
    Internal representation of a banner resize job.

    This is intentionally separate from API schemas so we can evolve internal
    fields (e.g. storage details, analysis metadata) without breaking the API.
    """

    id: str
    status: JobStatus
    outputs: List[OutputSize]
    master_banner_path: str
    additional_asset_paths: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    # Analysis and planning metadata (internal only).
    # Content-aware analysis of the master banner.
    banner_analysis: BannerAnalysis | None = None
    # Alignment decisions for optional user assets.
    asset_alignment: List[AssetAlignment] = field(default_factory=list)
    # Aspect-ratio and layout planning metadata.
    aspect_ratio_buckets: Dict[str, List[OutputSize]] = field(default_factory=dict)
    layout_plan: Dict[str, str] = field(default_factory=dict)

