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
    # Detected border/frame regions (for responsive layout).
    border_regions: List[Region] = field(default_factory=list)
    # Paths to derived masks/maps stored alongside the master banner:
    # - foreground_mask_path: binary foreground/background mask
    # - saliency_map_path: grayscale saliency heatmap
    # - protection_mask_path: binary mask of "protected" content (faces/text/logos)
    # - background_mask_path: binary mask of pure background regions (for extension)
    foreground_mask_path: str | None = None
    saliency_map_path: str | None = None
    protection_mask_path: str | None = None
    background_mask_path: str | None = None


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
class AspectRatioRisk:
    """
    Risk assessment for resizing to a specific aspect ratio.

    This captures model-assisted predictions about how problematic a given
    resize will be based on the detected banner content.
    """

    output_size: OutputSize
    # Overall risk score in [0, 1] where 0 = safe, 1 = high risk of distortion.
    risk_score: float
    # Individual risk factors contributing to the overall score.
    content_clipping_risk: float  # Risk of cropping faces/text/logos
    layout_stress_risk: float  # Risk of extreme stretching or compression
    saliency_loss_risk: float  # Risk of losing important visual content
    # Human-readable explanation of the risk assessment.
    reasoning: str
    # Deterministic strategy class assigned based on risk profile.
    strategy_class: str


@dataclass(slots=True)
class LayoutPlan:
    """
    Concrete layout plan for a specific output size.

    This captures the deterministic decisions about how to transform the master
    banner into the target size while preserving important content.
    """

    output_size: OutputSize
    strategy_class: str
    # Anchor point: which content region to preserve and center on.
    # If None, use geometric center.
    anchor_region: Region | None = None
    # Crop region in source banner coordinates (x, y, width, height).
    # If None, no crop is needed (pure resize or padding).
    crop_region: Region | None = None
    # Scaling decisions: "fit" (letterbox), "fill" (crop), "stretch" (distort).
    scaling_mode: str = "fill"
    # Background expansion zones for adaptive padding (list of regions where
    # background can be extended/regenerated).
    expansion_zones: List[Region] = field(default_factory=list)
    # Protected regions that must not be cropped or distorted.
    protected_regions: List[Region] = field(default_factory=list)
    # Human-readable explanation of the layout decisions.
    reasoning: str = ""


@dataclass(slots=True)
class QualityCheck:
    """
    Quality validation result for a generated output.

    This captures automated quality checks performed on the generated banner
    to ensure it meets minimum quality standards and flag issues for review.
    """

    output_size: OutputSize
    # Overall quality score in [0, 1] where 1 = perfect, 0 = failed.
    quality_score: float
    # Individual quality metrics.
    content_preservation_score: float  # How well protected content was preserved
    aspect_ratio_accuracy: float  # How close to target aspect ratio
    visual_quality_score: float  # Checks for artifacts, color shifts, etc.
    # Confidence in the output quality [0, 1].
    confidence: float
    # List of warnings or issues detected.
    warnings: List[str] = field(default_factory=list)
    # Whether this output needs manual review.
    needs_manual_review: bool = False
    # Human-readable explanation of quality assessment.
    reasoning: str = ""
    # Path to the generated output file (for reference).
    output_path: str | None = None


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
    # Risk assessments per output size (AI Step C).
    aspect_ratio_risks: List[AspectRatioRisk] = field(default_factory=list)
    # Concrete layout plans per output size (AI Step D).
    layout_plans: List[LayoutPlan] = field(default_factory=list)
    # Quality validation results per output (AI Step F).
    quality_checks: List[QualityCheck] = field(default_factory=list)
    # Legacy layout plan (deprecated, kept for backward compatibility).
    layout_plan: Dict[str, str] = field(default_factory=dict)

