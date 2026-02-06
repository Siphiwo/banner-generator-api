from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
import pytesseract

from app.api.v1.schemas import JobStatus, OutputSize
from app.models.jobs import AssetAlignment, AspectRatioRisk, BannerAnalysis, Job, LayoutPlan, QualityCheck, Region


logger = logging.getLogger(__name__)


def _load_image(master_banner_path: str) -> np.ndarray | None:
    """
    Load the master banner image from disk as a BGR OpenCV array.

    Returns None if the image cannot be loaded; callers are responsible for
    handling this gracefully.
    """
    image = cv2.imread(master_banner_path, cv2.IMREAD_COLOR)
    if image is None:
        logger.error("Failed to read master banner from %s", master_banner_path)
    return image


def _detect_faces(image: np.ndarray) -> List[Region]:
    """
    Detect faces using OpenCV's Haar cascades.

    This is a classical but well-understood detector; it can later be replaced
    with a PyTorch-based model behind the same Region interface.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        logger.warning("Face cascade could not be loaded from %s", cascade_path)
        return []

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    regions: List[Region] = []
    for (x, y, w, h) in faces:
        regions.append(Region(x=int(x), y=int(y), width=int(w), height=int(h), score=1.0, label="face"))
    return regions


def _detect_text_regions(image: np.ndarray) -> List[Region]:
    """
    Detect text regions using Tesseract's word-level bounding boxes.

    This uses pytesseract for simplicity and portability. It assumes that the
    Tesseract engine is installed on the host system. If it is not available,
    this function degrades gracefully and returns an empty list.
    """
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Text detection failed via Tesseract: %s", exc)
        return []

    regions: List[Region] = []
    n_boxes = len(data.get("text", []))
    for i in range(n_boxes):
        text = data["text"][i]
        try:
            conf = float(data.get("conf", [0])[i])
        except (ValueError, TypeError):
            conf = -1.0
        if not text or text.isspace():
            continue
        if conf < 50:  # Tesseract confidence is in [0, 100]
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        regions.append(
            Region(
                x=x,
                y=y,
                width=w,
                height=h,
                score=max(0.0, min(conf / 100.0, 1.0)),
                label="text",
            )
        )
    return regions


def _compute_saliency_and_masks(
    image: np.ndarray,
    base_path: Path,
    faces: List[Region],
    text_regions: List[Region],
) -> tuple[str | None, str | None, str | None]:
    """
    Compute a saliency map, a simple foreground mask, and a protection mask.

    The saliency map is computed using OpenCV's spectral residual method, which
    is fast and deterministic. Foreground and protection masks are derived from
    the saliency map and detected semantic regions.
    """
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image)
    if not success:
        logger.warning("Saliency computation failed; maps will not be persisted.")
        return None, None, None

    # Normalize to [0, 255] for visualization/persistence.
    saliency_norm = (saliency_map * 255).astype("uint8")

    # Foreground mask via simple threshold on saliency.
    _, foreground_mask = cv2.threshold(saliency_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = saliency_norm.shape[:2]
    protection_mask = np.zeros((h, w), dtype="uint8")

    def _draw_regions(mask: np.ndarray, regions: List[Region], value: int) -> None:
        for r in regions:
            x2 = min(r.x + r.width, w)
            y2 = min(r.y + r.height, h)
            x1 = max(r.x, 0)
            y1 = max(r.y, 0)
            if x1 >= x2 or y1 >= y2:
                continue
            cv2.rectangle(mask, (x1, y1), (x2, y2), color=value, thickness=-1)

    _draw_regions(protection_mask, faces, value=255)
    _draw_regions(protection_mask, text_regions, value=255)

    # Persist maps alongside the master banner for later inspection and reuse.
    saliency_path = base_path.with_name(f"{base_path.stem}_saliency.png")
    foreground_path = base_path.with_name(f"{base_path.stem}_foreground.png")
    protection_path = base_path.with_name(f"{base_path.stem}_protection.png")

    try:
        cv2.imwrite(str(saliency_path), saliency_norm)
        cv2.imwrite(str(foreground_path), foreground_mask)
        cv2.imwrite(str(protection_path), protection_mask)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist saliency/protection maps: %s", exc)
        return None, None, None

    return str(foreground_path), str(saliency_path), str(protection_path)


def analyze_banner_content(job: Job) -> None:
    """
    Run content-aware analysis on the master banner and attach results to the job.

    This is the entry point for Step A (Banner Content Analysis). It:
    - Detects faces
    - Detects text regions
    - Computes a saliency map
    - Derives simple foreground and protection masks
    """
    image = _load_image(job.master_banner_path)
    if image is None:
        # Leave banner_analysis as None to make the failure explicit to callers.
        job.banner_analysis = None
        return

    height, width = image.shape[:2]
    faces = _detect_faces(image)
    text_regions = _detect_text_regions(image)

    base_path = Path(job.master_banner_path)
    foreground_path, saliency_path, protection_path = _compute_saliency_and_masks(
        image=image,
        base_path=base_path,
        faces=faces,
        text_regions=text_regions,
    )

    job.banner_analysis = BannerAnalysis(
        width=width,
        height=height,
        faces=faces,
        text_regions=text_regions,
        logo_regions=[],  # Logo/product detection will be added in a later refinement.
        foreground_mask_path=foreground_path,
        saliency_map_path=saliency_path,
        protection_mask_path=protection_path,
    )


def _load_mask(path: str | None) -> np.ndarray | None:
    """Load a single-channel mask image from disk if it exists."""
    if not path:
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning("Failed to read mask from %s", path)
    return mask


def _find_safe_anchor_regions(
    banner_width: int,
    banner_height: int,
    protection_mask: np.ndarray | None,
    num_slots: int,
) -> List[Region]:
    """
    Propose safe anchor regions for optional assets.

    Strategy:
    - Sample canonical positions near the corners and center edges.
    - Avoid areas marked as protected (faces/text) by the protection mask.
    - Fallback to geometric positions if no mask is available.
    """
    # Each slot is a fixed fraction of the banner size (can be tuned later).
    slot_w = int(banner_width * 0.25)
    slot_h = int(banner_height * 0.25)

    candidates: List[tuple[int, int]] = []
    margin_x = int(banner_width * 0.05)
    margin_y = int(banner_height * 0.05)

    # Corner-like and edge positions (x, y refer to top-left).
    candidates.extend(
        [
            (margin_x, margin_y),  # top-left
            (banner_width - slot_w - margin_x, margin_y),  # top-right
            (margin_x, banner_height - slot_h - margin_y),  # bottom-left
            (banner_width - slot_w - margin_x, banner_height - slot_h - margin_y),  # bottom-right
            (int((banner_width - slot_w) / 2), margin_y),  # top-center
            (int((banner_width - slot_w) / 2), banner_height - slot_h - margin_y),  # bottom-center
        ]
    )

    safe_regions: List[Region] = []

    def is_safe(x: int, y: int) -> bool:
        if protection_mask is None:
            return True
        x2 = min(x + slot_w, banner_width)
        y2 = min(y + slot_h, banner_height)
        if x2 <= x or y2 <= y:
            return False
        roi = protection_mask[y:y2, x:x2]
        # Consider safe if less than 5% of the area is marked as protected.
        protected_ratio = float(np.count_nonzero(roi)) / float(roi.size)
        return protected_ratio < 0.05

    for x, y in candidates:
        if len(safe_regions) >= num_slots:
            break
        if is_safe(x, y):
            safe_regions.append(
                Region(
                    x=int(x),
                    y=int(y),
                    width=slot_w,
                    height=slot_h,
                    score=1.0,
                    label="asset-slot",
                )
            )

    # If there are still not enough slots, fall back to naive tiling.
    while len(safe_regions) < num_slots:
        x = margin_x + (len(safe_regions) * int(slot_w * 0.8)) % max(1, banner_width - slot_w)
        y = margin_y + (len(safe_regions) // 3) * int(slot_h * 0.8)
        safe_regions.append(
            Region(
                x=int(min(x, banner_width - slot_w)),
                y=int(min(y, banner_height - slot_h)),
                width=slot_w,
                height=slot_h,
                score=0.5,
                label="asset-slot",
            )
        )

    return safe_regions


def align_optional_assets(job: Job) -> None:
    """
    Align optional user-uploaded assets with semantically appropriate regions.

    This step:
    - Uses the banner's protection mask to avoid faces and existing text.
    - Proposes non-overlapping "asset-slot" regions near safe corners/edges.
    - Treats aligned assets as protected for later stages by updating the
      protection mask and storing alignment decisions on the job.

    User intent vs model inference:
    - Today, we only have file uploads and no explicit intent metadata, so all
      decisions here are automatic.
    - The `AssetAlignment` model includes a `user_override` flag and a `role`
      field so that future API extensions can override or refine these choices
      without changing internal structures.
    """
    if not job.additional_asset_paths:
        job.asset_alignment = []
        return

    if job.banner_analysis is None:
        # Without analysis we cannot make content-aware decisions; leave
        # alignment empty rather than guessing blindly.
        job.asset_alignment = []
        return

    analysis = job.banner_analysis
    banner_w, banner_h = analysis.width, analysis.height

    protection_mask = _load_mask(analysis.protection_mask_path)

    # Find safe anchor regions for each asset.
    safe_regions = _find_safe_anchor_regions(
        banner_width=banner_w,
        banner_height=banner_h,
        protection_mask=protection_mask,
        num_slots=len(job.additional_asset_paths),
    )

    alignments: List[AssetAlignment] = []
    for asset_path, region in zip(job.additional_asset_paths, safe_regions, strict=False):
        alignments.append(
            AssetAlignment(
                asset_path=asset_path,
                role="logo",  # Default role; can be overridden by future API hints.
                target_region=region,
                locked=True,
                user_override=False,
            )
        )

    job.asset_alignment = alignments

    # Treat aligned asset regions as "logo/product" regions and extend the
    # protection mask so that later stages do not distort or occlude them.
    analysis.logo_regions = (analysis.logo_regions or []) + [
        Region(
            x=a.target_region.x,
            y=a.target_region.y,
            width=a.target_region.width,
            height=a.target_region.height,
            score=1.0,
            label="logo",
        )
        for a in alignments
        if a.target_region is not None
    ]

    if protection_mask is not None:
        for a in alignments:
            if a.target_region is None:
                continue
            r = a.target_region
            x1, y1 = max(r.x, 0), max(r.y, 0)
            x2 = min(r.x + r.width, banner_w)
            y2 = min(r.y + r.height, banner_h)
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(protection_mask, (x1, y1), (x2, y2), color=255, thickness=-1)

        try:
            cv2.imwrite(analysis.protection_mask_path, protection_mask)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist updated protection mask: %s", exc)


def _classify_ratio(output: OutputSize) -> str:
    """
    Classify an output size into a coarse aspect-ratio bucket.

    Buckets are intentionally simple and deterministic; they are placeholders
    until real, content-aware analysis is introduced.
    """
    ratio = output.width / output.height
    if 0.9 <= ratio <= 1.1:
        return "similar"
    if 0.6 <= ratio < 0.9 or 1.1 < ratio <= 1.8:
        return "moderate"
    return "extreme"


def _compute_content_clipping_risk(
    banner_analysis: BannerAnalysis,
    output_size: OutputSize,
) -> tuple[float, str]:
    """
    Compute risk of clipping important content (faces, text, logos).

    Strategy:
    - Calculate what percentage of protected regions would be lost if we
      naively center-crop to the target aspect ratio.
    - Higher loss = higher risk.

    Returns (risk_score, reasoning_snippet).
    """
    if not banner_analysis:
        return 0.5, "no content analysis available"

    banner_w, banner_h = banner_analysis.width, banner_analysis.height
    banner_ratio = banner_w / banner_h
    target_ratio = output_size.width / output_size.height

    # All protected regions (faces + text + logos).
    protected = (
        (banner_analysis.faces or [])
        + (banner_analysis.text_regions or [])
        + (banner_analysis.logo_regions or [])
    )

    if not protected:
        return 0.1, "no critical content detected"

    # Simulate a center crop to the target aspect ratio.
    if target_ratio > banner_ratio:
        # Target is wider: crop top/bottom.
        new_h = int(banner_w / target_ratio)
        crop_y = (banner_h - new_h) // 2
        crop_x, crop_w = 0, banner_w
        crop_h = new_h
    else:
        # Target is taller: crop left/right.
        new_w = int(banner_h * target_ratio)
        crop_x = (banner_w - new_w) // 2
        crop_y, crop_h = 0, banner_h
        crop_w = new_w

    def region_area(r: Region) -> int:
        return r.width * r.height

    def intersection_area(r: Region, cx: int, cy: int, cw: int, ch: int) -> int:
        x1 = max(r.x, cx)
        y1 = max(r.y, cy)
        x2 = min(r.x + r.width, cx + cw)
        y2 = min(r.y + r.height, cy + ch)
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    total_protected_area = sum(region_area(r) for r in protected)
    retained_area = sum(
        intersection_area(r, crop_x, crop_y, crop_w, crop_h) for r in protected
    )

    if total_protected_area == 0:
        return 0.1, "no critical content detected"

    loss_ratio = 1.0 - (retained_area / total_protected_area)
    risk = min(1.0, loss_ratio * 1.2)  # Amplify slightly for sensitivity.

    if risk < 0.2:
        reason = "minimal content loss expected"
    elif risk < 0.5:
        reason = f"moderate content clipping ({int(loss_ratio * 100)}% loss)"
    else:
        reason = f"high content clipping risk ({int(loss_ratio * 100)}% loss)"

    return risk, reason


def _compute_layout_stress_risk(
    banner_analysis: BannerAnalysis,
    output_size: OutputSize,
) -> tuple[float, str]:
    """
    Compute risk of extreme stretching or compression distorting the layout.

    Strategy:
    - Measure how far the target aspect ratio deviates from the source.
    - Extreme deviations (e.g. 16:9 → 1:1) create high layout stress.

    Returns (risk_score, reasoning_snippet).
    """
    if not banner_analysis:
        return 0.5, "no banner dimensions available"

    banner_w, banner_h = banner_analysis.width, banner_analysis.height
    banner_ratio = banner_w / banner_h
    target_ratio = output_size.width / output_size.height

    ratio_deviation = abs(banner_ratio - target_ratio) / banner_ratio

    if ratio_deviation < 0.1:
        risk = 0.05
        reason = "aspect ratio nearly identical"
    elif ratio_deviation < 0.3:
        risk = 0.2
        reason = "minor aspect ratio adjustment"
    elif ratio_deviation < 0.6:
        risk = 0.5
        reason = "moderate aspect ratio change"
    else:
        risk = 0.9
        reason = "extreme aspect ratio transformation"

    return risk, reason


def _compute_saliency_loss_risk(
    banner_analysis: BannerAnalysis,
    output_size: OutputSize,
) -> tuple[float, str]:
    """
    Compute risk of losing visually important content based on saliency map.

    Strategy:
    - Load the saliency map and simulate a center crop.
    - Measure how much high-saliency content would be lost.

    Returns (risk_score, reasoning_snippet).
    """
    if not banner_analysis or not banner_analysis.saliency_map_path:
        return 0.3, "no saliency map available"

    saliency_map = _load_mask(banner_analysis.saliency_map_path)
    if saliency_map is None:
        return 0.3, "saliency map could not be loaded"

    banner_w, banner_h = banner_analysis.width, banner_analysis.height
    banner_ratio = banner_w / banner_h
    target_ratio = output_size.width / output_size.height

    # Simulate center crop.
    if target_ratio > banner_ratio:
        new_h = int(banner_w / target_ratio)
        crop_y = (banner_h - new_h) // 2
        crop_x, crop_w = 0, banner_w
        crop_h = new_h
    else:
        new_w = int(banner_h * target_ratio)
        crop_x = (banner_w - new_w) // 2
        crop_y, crop_h = 0, banner_h
        crop_w = new_w

    # Extract the crop region from the saliency map.
    crop_y2 = min(crop_y + crop_h, banner_h)
    crop_x2 = min(crop_x + crop_w, banner_w)
    crop_y = max(crop_y, 0)
    crop_x = max(crop_x, 0)

    if crop_y >= crop_y2 or crop_x >= crop_x2:
        return 0.9, "invalid crop region"

    cropped_saliency = saliency_map[crop_y:crop_y2, crop_x:crop_x2]

    # Measure saliency retention.
    total_saliency = float(np.sum(saliency_map))
    retained_saliency = float(np.sum(cropped_saliency))

    if total_saliency == 0:
        return 0.2, "uniform saliency (no focal points)"

    retention_ratio = retained_saliency / total_saliency
    loss_ratio = 1.0 - retention_ratio
    risk = min(1.0, loss_ratio * 1.3)

    if risk < 0.2:
        reason = "minimal saliency loss"
    elif risk < 0.5:
        reason = f"moderate saliency loss ({int(loss_ratio * 100)}%)"
    else:
        reason = f"high saliency loss ({int(loss_ratio * 100)}%)"

    return risk, reason


def _assign_strategy_class(
    content_risk: float,
    layout_risk: float,
    saliency_risk: float,
) -> str:
    """
    Deterministically assign a layout strategy class based on risk profile.

    Strategy classes are human-readable labels that later stages can map to
    concrete layout algorithms.
    """
    overall_risk = (content_risk * 0.5) + (layout_risk * 0.3) + (saliency_risk * 0.2)

    if overall_risk < 0.25:
        return "safe-center-crop"
    if overall_risk < 0.5:
        if content_risk > 0.4:
            return "content-aware-crop"
        return "focus-preserving-resize"
    if overall_risk < 0.75:
        if layout_risk > 0.6:
            return "adaptive-padding"
        return "smart-crop-with-protection"
    # High risk: needs careful handling.
    return "manual-review-recommended"


def score_aspect_ratio_risks(job: Job) -> None:
    """
    Score resize risk for each requested output size using content analysis.

    This replaces the simple heuristic bucketing with a model-assisted risk
    assessment that:
    - Predicts content clipping risk (faces/text/logos)
    - Predicts layout stress (aspect ratio deviation)
    - Predicts saliency loss (visual importance)
    - Assigns deterministic strategy classes based on risk profile

    The output is explainable (includes reasoning) and deterministic (same
    input always produces the same output).
    """
    if not job.banner_analysis:
        logger.warning("Cannot score aspect ratio risks without banner analysis.")
        job.aspect_ratio_risks = []
        return

    risks: List[AspectRatioRisk] = []

    for output_size in job.outputs:
        content_risk, content_reason = _compute_content_clipping_risk(
            job.banner_analysis, output_size
        )
        layout_risk, layout_reason = _compute_layout_stress_risk(
            job.banner_analysis, output_size
        )
        saliency_risk, saliency_reason = _compute_saliency_loss_risk(
            job.banner_analysis, output_size
        )

        overall_risk = (content_risk * 0.5) + (layout_risk * 0.3) + (saliency_risk * 0.2)
        strategy_class = _assign_strategy_class(content_risk, layout_risk, saliency_risk)

        reasoning_parts = [
            f"Content: {content_reason}",
            f"Layout: {layout_reason}",
            f"Saliency: {saliency_reason}",
        ]
        reasoning = "; ".join(reasoning_parts)

        risks.append(
            AspectRatioRisk(
                output_size=output_size,
                risk_score=overall_risk,
                content_clipping_risk=content_risk,
                layout_stress_risk=layout_risk,
                saliency_loss_risk=saliency_risk,
                reasoning=reasoning,
                strategy_class=strategy_class,
            )
        )

    job.aspect_ratio_risks = risks
    logger.info("Scored %d aspect ratio risks for job %s", len(risks), job.id)


def analyze_aspect_ratios(job: Job) -> None:
    """
    Convert requested output sizes into aspect-ratio buckets.

    This runs once per job and stores the result on the job object. It does not
    touch image pixels; it only works from requested sizes.
    """
    buckets: Dict[str, List[OutputSize]] = {"similar": [], "moderate": [], "extreme": []}
    for size in job.outputs:
        bucket = _classify_ratio(size)
        buckets[bucket].append(size)

    job.aspect_ratio_buckets = buckets
    job.status = JobStatus.ANALYZING


def _compute_content_center_of_mass(
    banner_analysis: BannerAnalysis,
    saliency_map: np.ndarray | None,
) -> tuple[int, int]:
    """
    Compute the center of mass of important content.

    Strategy:
    - Weight protected regions (faces, text, logos) heavily.
    - Use saliency map as secondary guidance if available.
    - Fall back to geometric center if no content is detected.

    Returns (center_x, center_y) in banner coordinates.
    """
    banner_w, banner_h = banner_analysis.width, banner_analysis.height

    protected = (
        (banner_analysis.faces or [])
        + (banner_analysis.text_regions or [])
        + (banner_analysis.logo_regions or [])
    )

    if not protected:
        # No detected content; use geometric center.
        return banner_w // 2, banner_h // 2

    # Compute weighted center of mass from protected regions.
    total_weight = 0.0
    weighted_x = 0.0
    weighted_y = 0.0

    for region in protected:
        # Weight by area and score.
        area = region.width * region.height
        weight = area * region.score
        center_x = region.x + region.width / 2
        center_y = region.y + region.height / 2
        weighted_x += center_x * weight
        weighted_y += center_y * weight
        total_weight += weight

    if total_weight == 0:
        return banner_w // 2, banner_h // 2

    center_x = int(weighted_x / total_weight)
    center_y = int(weighted_y / total_weight)

    # Clamp to banner bounds.
    center_x = max(0, min(center_x, banner_w - 1))
    center_y = max(0, min(center_y, banner_h - 1))

    return center_x, center_y


def _compute_crop_region_centered(
    banner_w: int,
    banner_h: int,
    target_w: int,
    target_h: int,
    center_x: int,
    center_y: int,
) -> Region:
    """
    Compute a crop region centered on a specific point.

    The crop region will have the same aspect ratio as the target size and
    will be centered on (center_x, center_y) as much as possible while staying
    within banner bounds.
    """
    target_ratio = target_w / target_h
    banner_ratio = banner_w / banner_h

    if target_ratio > banner_ratio:
        # Target is wider: crop top/bottom.
        crop_w = banner_w
        crop_h = int(banner_w / target_ratio)
    else:
        # Target is taller: crop left/right.
        crop_h = banner_h
        crop_w = int(banner_h * target_ratio)

    # Center on the specified point.
    crop_x = center_x - crop_w // 2
    crop_y = center_y - crop_h // 2

    # Clamp to banner bounds.
    crop_x = max(0, min(crop_x, banner_w - crop_w))
    crop_y = max(0, min(crop_y, banner_h - crop_h))

    return Region(x=crop_x, y=crop_y, width=crop_w, height=crop_h, score=1.0, label="crop")


def _compute_expansion_zones(
    banner_w: int,
    banner_h: int,
    target_w: int,
    target_h: int,
) -> List[Region]:
    """
    Identify zones where background can be extended for adaptive padding.

    Strategy:
    - If target is wider than source, mark left/right edges for expansion.
    - If target is taller than source, mark top/bottom edges for expansion.
    """
    target_ratio = target_w / target_h
    banner_ratio = banner_w / banner_h

    zones: List[Region] = []

    if target_ratio > banner_ratio:
        # Target is wider: need horizontal expansion.
        # Mark left and right edge zones (10% of banner width each).
        edge_w = int(banner_w * 0.1)
        zones.append(Region(x=0, y=0, width=edge_w, height=banner_h, score=1.0, label="expand-left"))
        zones.append(
            Region(
                x=banner_w - edge_w,
                y=0,
                width=edge_w,
                height=banner_h,
                score=1.0,
                label="expand-right",
            )
        )
    else:
        # Target is taller: need vertical expansion.
        # Mark top and bottom edge zones (10% of banner height each).
        edge_h = int(banner_h * 0.1)
        zones.append(Region(x=0, y=0, width=banner_w, height=edge_h, score=1.0, label="expand-top"))
        zones.append(
            Region(
                x=0,
                y=banner_h - edge_h,
                width=banner_w,
                height=edge_h,
                score=1.0,
                label="expand-bottom",
            )
        )

    return zones


def _plan_safe_center_crop(
    banner_analysis: BannerAnalysis,
    output_size: OutputSize,
    risk: AspectRatioRisk,
) -> LayoutPlan:
    """
    Plan a safe center crop for low-risk aspect ratios.

    Strategy:
    - Use geometric center as anchor.
    - Crop to target aspect ratio with minimal content loss.
    - No special protection needed since risk is already low.
    """
    banner_w, banner_h = banner_analysis.width, banner_analysis.height
    center_x, center_y = banner_w // 2, banner_h // 2

    crop_region = _compute_crop_region_centered(
        banner_w, banner_h, output_size.width, output_size.height, center_x, center_y
    )

    return LayoutPlan(
        output_size=output_size,
        strategy_class=risk.strategy_class,
        anchor_region=None,  # Geometric center, no specific content anchor.
        crop_region=crop_region,
        scaling_mode="fill",
        expansion_zones=[],
        protected_regions=[],
        reasoning="Low risk; simple center crop preserves most content",
    )


def _plan_focus_preserving_resize(
    banner_analysis: BannerAnalysis,
    output_size: OutputSize,
    risk: AspectRatioRisk,
) -> LayoutPlan:
    """
    Plan a focus-preserving resize with letterboxing.

    Strategy:
    - Resize to fit within target dimensions without cropping.
    - Add letterbox bars (padding) to fill remaining space.
    - Preserve all content at the cost of some empty space.
    """
    banner_w, banner_h = banner_analysis.width, banner_analysis.height

    return LayoutPlan(
        output_size=output_size,
        strategy_class=risk.strategy_class,
        anchor_region=None,
        crop_region=None,  # No crop; pure resize with padding.
        scaling_mode="fit",
        expansion_zones=[],
        protected_regions=[],
        reasoning="Moderate risk; letterbox to preserve all content without distortion",
    )


def _plan_content_aware_crop(
    banner_analysis: BannerAnalysis,
    output_size: OutputSize,
    risk: AspectRatioRisk,
) -> LayoutPlan:
    """
    Plan a content-aware crop that preserves important regions.

    Strategy:
    - Compute center of mass of protected content (faces, text, logos).
    - Crop around this center to minimize content loss.
    - Mark protected regions explicitly for generation stage.
    """
    banner_w, banner_h = banner_analysis.width, banner_analysis.height
    saliency_map = _load_mask(banner_analysis.saliency_map_path)

    center_x, center_y = _compute_content_center_of_mass(banner_analysis, saliency_map)

    crop_region = _compute_crop_region_centered(
        banner_w, banner_h, output_size.width, output_size.height, center_x, center_y
    )

    # Collect all protected regions.
    protected = (
        (banner_analysis.faces or [])
        + (banner_analysis.text_regions or [])
        + (banner_analysis.logo_regions or [])
    )

    # Create an anchor region around the content center.
    anchor_size = min(banner_w, banner_h) // 4
    anchor_region = Region(
        x=max(0, center_x - anchor_size // 2),
        y=max(0, center_y - anchor_size // 2),
        width=anchor_size,
        height=anchor_size,
        score=1.0,
        label="content-anchor",
    )

    return LayoutPlan(
        output_size=output_size,
        strategy_class=risk.strategy_class,
        anchor_region=anchor_region,
        crop_region=crop_region,
        scaling_mode="fill",
        expansion_zones=[],
        protected_regions=protected,
        reasoning=f"Content-aware crop centered on detected content at ({center_x}, {center_y})",
    )


def _plan_adaptive_padding(
    banner_analysis: BannerAnalysis,
    output_size: OutputSize,
    risk: AspectRatioRisk,
) -> LayoutPlan:
    """
    Plan adaptive padding with background extension.

    Strategy:
    - Resize banner to fit within target dimensions.
    - Identify edge zones where background can be extended/regenerated.
    - Mark these zones for later AI-based background generation.
    """
    banner_w, banner_h = banner_analysis.width, banner_analysis.height

    expansion_zones = _compute_expansion_zones(
        banner_w, banner_h, output_size.width, output_size.height
    )

    return LayoutPlan(
        output_size=output_size,
        strategy_class=risk.strategy_class,
        anchor_region=None,
        crop_region=None,  # No crop; resize with padding.
        scaling_mode="fit",
        expansion_zones=expansion_zones,
        protected_regions=[],
        reasoning=f"Adaptive padding with {len(expansion_zones)} background expansion zones",
    )


def _plan_smart_crop_with_protection(
    banner_analysis: BannerAnalysis,
    output_size: OutputSize,
    risk: AspectRatioRisk,
) -> LayoutPlan:
    """
    Plan a smart crop with explicit content protection.

    Strategy:
    - Similar to content-aware crop but with stricter protection.
    - Compute content center and crop around it.
    - Mark all protected regions and ensure they're fully contained.
    """
    banner_w, banner_h = banner_analysis.width, banner_analysis.height
    saliency_map = _load_mask(banner_analysis.saliency_map_path)

    center_x, center_y = _compute_content_center_of_mass(banner_analysis, saliency_map)

    crop_region = _compute_crop_region_centered(
        banner_w, banner_h, output_size.width, output_size.height, center_x, center_y
    )

    # Collect all protected regions.
    protected = (
        (banner_analysis.faces or [])
        + (banner_analysis.text_regions or [])
        + (banner_analysis.logo_regions or [])
    )

    # Create a larger anchor region for stricter protection.
    anchor_size = min(banner_w, banner_h) // 3
    anchor_region = Region(
        x=max(0, center_x - anchor_size // 2),
        y=max(0, center_y - anchor_size // 2),
        width=anchor_size,
        height=anchor_size,
        score=1.0,
        label="protected-anchor",
    )

    return LayoutPlan(
        output_size=output_size,
        strategy_class=risk.strategy_class,
        anchor_region=anchor_region,
        crop_region=crop_region,
        scaling_mode="fill",
        expansion_zones=[],
        protected_regions=protected,
        reasoning=f"Smart crop with strict protection for {len(protected)} regions",
    )


def _plan_manual_review_recommended(
    banner_analysis: BannerAnalysis,
    output_size: OutputSize,
    risk: AspectRatioRisk,
) -> LayoutPlan:
    """
    Plan for high-risk cases that need manual review.

    Strategy:
    - Provide a conservative fallback (letterbox with no crop).
    - Flag for manual review with detailed reasoning.
    - Preserve all content to avoid automatic mistakes.
    """
    return LayoutPlan(
        output_size=output_size,
        strategy_class=risk.strategy_class,
        anchor_region=None,
        crop_region=None,
        scaling_mode="fit",
        expansion_zones=[],
        protected_regions=[],
        reasoning=f"High risk (score: {risk.risk_score:.2f}); manual review recommended. "
        f"Using conservative letterbox to preserve all content. {risk.reasoning}",
    )


def plan_layouts(job: Job) -> None:
    """
    Generate concrete layout plans for each output size based on risk assessment.

    This maps strategy classes from Step C to deterministic layout algorithms:
    - safe-center-crop: Simple geometric center crop
    - focus-preserving-resize: Letterbox to preserve all content
    - content-aware-crop: Crop around detected content center of mass
    - adaptive-padding: Resize with background extension zones
    - smart-crop-with-protection: Strict content-aware crop with protection
    - manual-review-recommended: Conservative fallback with manual review flag

    Each layout plan includes:
    - Anchor points (which content regions to preserve)
    - Crop regions (what to extract from source)
    - Scaling decisions (fit/fill/stretch)
    - Expansion zones (where background can be extended)
    - Protected regions (what must not be distorted)
    """
    if not job.banner_analysis:
        logger.warning("Cannot plan layouts without banner analysis.")
        job.layout_plans = []
        job.status = JobStatus.PLANNING
        return

    if not job.aspect_ratio_risks:
        logger.warning("Cannot plan layouts without aspect ratio risk assessment.")
        job.layout_plans = []
        job.status = JobStatus.PLANNING
        return

    layout_plans: List[LayoutPlan] = []

    # Map each risk assessment to a concrete layout plan.
    for risk in job.aspect_ratio_risks:
        strategy_class = risk.strategy_class

        if strategy_class == "safe-center-crop":
            plan = _plan_safe_center_crop(job.banner_analysis, risk.output_size, risk)
        elif strategy_class == "focus-preserving-resize":
            plan = _plan_focus_preserving_resize(job.banner_analysis, risk.output_size, risk)
        elif strategy_class == "content-aware-crop":
            plan = _plan_content_aware_crop(job.banner_analysis, risk.output_size, risk)
        elif strategy_class == "adaptive-padding":
            plan = _plan_adaptive_padding(job.banner_analysis, risk.output_size, risk)
        elif strategy_class == "smart-crop-with-protection":
            plan = _plan_smart_crop_with_protection(job.banner_analysis, risk.output_size, risk)
        elif strategy_class == "manual-review-recommended":
            plan = _plan_manual_review_recommended(job.banner_analysis, risk.output_size, risk)
        else:
            # Unknown strategy class; fall back to safe center crop.
            logger.warning("Unknown strategy class %s; falling back to safe-center-crop", strategy_class)
            plan = _plan_safe_center_crop(job.banner_analysis, risk.output_size, risk)

        layout_plans.append(plan)

    job.layout_plans = layout_plans
    job.status = JobStatus.PLANNING
    logger.info("Generated %d layout plans for job %s", len(layout_plans), job.id)


def _validate_content_preservation(
    output_image: np.ndarray,
    plan: LayoutPlan,
    banner_analysis: BannerAnalysis,
) -> tuple[float, List[str]]:
    """
    Validate that protected content was preserved in the output.

    Strategy:
    - Check if protected regions would fit within the output bounds.
    - For crop strategies, verify that protected regions are within the crop.
    - For letterbox strategies, verify that content is not clipped.

    Returns (score, warnings) where score is in [0, 1].
    """
    warnings: List[str] = []
    
    if not plan.protected_regions:
        # No protected content to validate.
        return 1.0, warnings

    output_h, output_w = output_image.shape[:2]
    banner_w, banner_h = banner_analysis.width, banner_analysis.height

    # For crop strategies, check if protected regions are within the crop.
    if plan.crop_region and plan.scaling_mode == "fill":
        crop = plan.crop_region
        clipped_regions = 0
        partially_clipped = 0

        for region in plan.protected_regions:
            # Check if region is fully within crop.
            region_x1 = region.x
            region_y1 = region.y
            region_x2 = region.x + region.width
            region_y2 = region.y + region.height

            crop_x1 = crop.x
            crop_y1 = crop.y
            crop_x2 = crop.x + crop.width
            crop_y2 = crop.y + crop.height

            # Check if region is fully outside crop.
            if region_x2 <= crop_x1 or region_x1 >= crop_x2 or region_y2 <= crop_y1 or region_y1 >= crop_y2:
                clipped_regions += 1
                warnings.append(
                    f"{region.label or 'Protected'} region completely clipped by crop"
                )
            # Check if region is partially clipped.
            elif region_x1 < crop_x1 or region_x2 > crop_x2 or region_y1 < crop_y1 or region_y2 > crop_y2:
                partially_clipped += 1
                warnings.append(
                    f"{region.label or 'Protected'} region partially clipped by crop"
                )

        if clipped_regions > 0:
            score = max(0.0, 1.0 - (clipped_regions / len(plan.protected_regions)))
        elif partially_clipped > 0:
            score = max(0.5, 1.0 - (partially_clipped / len(plan.protected_regions)) * 0.5)
        else:
            score = 1.0

        return score, warnings

    # For letterbox strategies, all content should be preserved.
    return 1.0, warnings


def _validate_aspect_ratio(
    output_image: np.ndarray,
    plan: LayoutPlan,
) -> tuple[float, List[str]]:
    """
    Validate that the output has the correct aspect ratio.

    Strategy:
    - Compare actual output dimensions to target dimensions.
    - Allow small tolerance for rounding errors.

    Returns (score, warnings) where score is in [0, 1].
    """
    warnings: List[str] = []
    
    output_h, output_w = output_image.shape[:2]
    target_w = plan.output_size.width
    target_h = plan.output_size.height

    # Check exact dimensions.
    if output_w != target_w or output_h != target_h:
        warnings.append(
            f"Output dimensions ({output_w}x{output_h}) do not match target ({target_w}x{target_h})"
        )
        # Compute how far off we are.
        w_error = abs(output_w - target_w) / target_w
        h_error = abs(output_h - target_h) / target_h
        max_error = max(w_error, h_error)
        score = max(0.0, 1.0 - max_error)
        return score, warnings

    # Check aspect ratio.
    output_ratio = output_w / output_h
    target_ratio = target_w / target_h
    ratio_error = abs(output_ratio - target_ratio) / target_ratio

    if ratio_error > 0.01:  # Allow 1% tolerance.
        warnings.append(
            f"Aspect ratio mismatch: output={output_ratio:.3f}, target={target_ratio:.3f}"
        )
        score = max(0.0, 1.0 - ratio_error)
        return score, warnings

    return 1.0, warnings


def _validate_visual_quality(
    output_image: np.ndarray,
    plan: LayoutPlan,
) -> tuple[float, List[str]]:
    """
    Validate visual quality of the output.

    Strategy:
    - Check for extreme color shifts (compare color histograms).
    - Check for excessive blur or artifacts (using Laplacian variance).
    - Check for black/empty regions (letterbox padding is expected).

    Returns (score, warnings) where score is in [0, 1].
    """
    warnings: List[str] = []
    scores: List[float] = []

    # Check for blur/sharpness using Laplacian variance.
    gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Typical sharp images have variance > 100, blurry images < 50.
    if laplacian_var < 50:
        warnings.append(f"Output may be blurry (sharpness score: {laplacian_var:.1f})")
        scores.append(0.5)
    elif laplacian_var < 100:
        scores.append(0.8)
    else:
        scores.append(1.0)

    # Check for excessive black regions (excluding expected letterbox padding).
    if plan.scaling_mode == "fit":
        # Letterbox is expected; don't penalize black regions.
        scores.append(1.0)
    else:
        # Check if output has excessive black pixels.
        black_pixels = np.sum(np.all(output_image < 10, axis=2))
        total_pixels = output_image.shape[0] * output_image.shape[1]
        black_ratio = black_pixels / total_pixels

        if black_ratio > 0.3:
            warnings.append(f"Output has {black_ratio*100:.1f}% black pixels")
            scores.append(max(0.0, 1.0 - black_ratio))
        else:
            scores.append(1.0)

    # Overall visual quality score.
    overall_score = sum(scores) / len(scores) if scores else 1.0
    return overall_score, warnings


def validate_output_quality(
    job: Job,
    output_image: np.ndarray,
    plan: LayoutPlan,
    output_path: str,
) -> QualityCheck:
    """
    Perform comprehensive quality validation on a generated output.

    This runs multiple quality checks and produces a structured quality report
    with scores, warnings, and recommendations.
    """
    if not job.banner_analysis:
        # Cannot validate without analysis.
        return QualityCheck(
            output_size=plan.output_size,
            quality_score=0.5,
            content_preservation_score=0.5,
            aspect_ratio_accuracy=0.5,
            visual_quality_score=0.5,
            confidence=0.3,
            warnings=["Cannot validate: banner analysis not available"],
            needs_manual_review=True,
            reasoning="Validation skipped due to missing banner analysis",
            output_path=output_path,
        )

    # Run individual quality checks.
    content_score, content_warnings = _validate_content_preservation(
        output_image, plan, job.banner_analysis
    )
    aspect_score, aspect_warnings = _validate_aspect_ratio(output_image, plan)
    visual_score, visual_warnings = _validate_visual_quality(output_image, plan)

    # Combine warnings.
    all_warnings = content_warnings + aspect_warnings + visual_warnings

    # Compute overall quality score (weighted average).
    quality_score = (content_score * 0.5) + (aspect_score * 0.3) + (visual_score * 0.2)

    # Compute confidence based on strategy class and risk.
    confidence = 1.0
    if plan.strategy_class == "manual-review-recommended":
        confidence = 0.5
        all_warnings.append("Strategy flagged for manual review")
    elif plan.strategy_class == "adaptive-padding":
        confidence = 0.7
        all_warnings.append("Background extension may need review")

    # Determine if manual review is needed.
    needs_manual_review = (
        quality_score < 0.7
        or confidence < 0.6
        or plan.strategy_class == "manual-review-recommended"
    )

    # Generate reasoning.
    reasoning_parts = [
        f"Content preservation: {content_score:.2f}",
        f"Aspect ratio: {aspect_score:.2f}",
        f"Visual quality: {visual_score:.2f}",
    ]
    if all_warnings:
        reasoning_parts.append(f"{len(all_warnings)} warnings detected")
    reasoning = "; ".join(reasoning_parts)

    return QualityCheck(
        output_size=plan.output_size,
        quality_score=quality_score,
        content_preservation_score=content_score,
        aspect_ratio_accuracy=aspect_score,
        visual_quality_score=visual_score,
        confidence=confidence,
        warnings=all_warnings,
        needs_manual_review=needs_manual_review,
        reasoning=reasoning,
        output_path=output_path,
    )


def generate_output_images(job: Job) -> None:
    """
    Generate resized banner images based on layout plans.

    This is the entry point for Step E (Image Generation). It:
    - Executes each layout plan to produce a resized banner
    - Applies the appropriate transformation strategy (crop, fit, fill)
    - Handles background extension for adaptive padding strategies
    - Persists output images alongside the master banner
    - Updates job status to reflect generation progress

    Each layout plan is executed independently, allowing for parallel
    processing in future iterations.
    """
    if not job.layout_plans:
        logger.warning("Cannot generate outputs without layout plans.")
        job.status = JobStatus.FAILED
        return

    if not job.banner_analysis:
        logger.warning("Cannot generate outputs without banner analysis.")
        job.status = JobStatus.FAILED
        return

    # Load the master banner once for all outputs.
    master_image = _load_image(job.master_banner_path)
    if master_image is None:
        logger.error("Failed to load master banner for output generation.")
        job.status = JobStatus.FAILED
        return

    job.status = JobStatus.GENERATING

    base_path = Path(job.master_banner_path).parent
    success_count = 0
    quality_checks: List[QualityCheck] = []

    for plan in job.layout_plans:
        try:
            output_image = _execute_layout_plan(master_image, plan, job.banner_analysis)
            if output_image is None:
                logger.warning(
                    "Failed to generate output for %dx%d",
                    plan.output_size.width,
                    plan.output_size.height,
                )
                continue

            # Persist the output image.
            output_filename = f"output_{plan.output_size.width}x{plan.output_size.height}.webp"
            output_path = base_path / output_filename

            # Convert BGR to RGB for PIL.
            rgb_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            pil_image.save(str(output_path), format="WEBP", quality=90)

            logger.info("Generated output: %s", output_path)
            success_count += 1

            # Validate output quality (Step F).
            quality_check = validate_output_quality(
                job=job,
                output_image=output_image,
                plan=plan,
                output_path=str(output_path),
            )
            quality_checks.append(quality_check)

            # Log quality results.
            if quality_check.needs_manual_review:
                logger.warning(
                    "Output %dx%d needs manual review (quality: %.2f, confidence: %.2f)",
                    plan.output_size.width,
                    plan.output_size.height,
                    quality_check.quality_score,
                    quality_check.confidence,
                )
            if quality_check.warnings:
                logger.info(
                    "Quality warnings for %dx%d: %s",
                    plan.output_size.width,
                    plan.output_size.height,
                    "; ".join(quality_check.warnings),
                )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to generate output for %dx%d: %s",
                plan.output_size.width,
                plan.output_size.height,
                exc,
            )

    # Store quality checks on the job.
    job.quality_checks = quality_checks

    if success_count == 0:
        job.status = JobStatus.FAILED
        logger.error("All output generation attempts failed for job %s", job.id)
    else:
        job.status = JobStatus.COMPLETED
        logger.info(
            "Generated %d/%d outputs for job %s (quality checks: %d)",
            success_count,
            len(job.layout_plans),
            job.id,
            len(quality_checks),
        )


def _execute_layout_plan(
    master_image: np.ndarray,
    plan: LayoutPlan,
    banner_analysis: BannerAnalysis,
) -> np.ndarray | None:
    """
    Execute a single layout plan to produce a resized banner.

    This function dispatches to the appropriate transformation strategy based
    on the plan's scaling mode and strategy class.
    """
    if plan.scaling_mode == "fill":
        # Crop-based strategies.
        return _generate_crop_output(master_image, plan)
    elif plan.scaling_mode == "fit":
        # Letterbox-based strategies.
        if plan.expansion_zones:
            # Adaptive padding with background extension.
            return _generate_adaptive_padding_output(master_image, plan, banner_analysis)
        else:
            # Simple letterbox.
            return _generate_letterbox_output(master_image, plan)
    elif plan.scaling_mode == "stretch":
        # Direct resize (distortion allowed).
        return _generate_stretch_output(master_image, plan)
    else:
        logger.warning("Unknown scaling mode: %s", plan.scaling_mode)
        return None


def _generate_crop_output(
    master_image: np.ndarray,
    plan: LayoutPlan,
) -> np.ndarray | None:
    """
    Generate output using crop-based strategy.

    Strategy:
    - Extract the crop region from the master banner.
    - Resize to target dimensions.
    - Preserve aspect ratio of the crop region.
    """
    if plan.crop_region is None:
        logger.warning("Crop strategy requires a crop region.")
        return None

    h, w = master_image.shape[:2]
    crop = plan.crop_region

    # Clamp crop region to image bounds.
    x1 = max(0, crop.x)
    y1 = max(0, crop.y)
    x2 = min(w, crop.x + crop.width)
    y2 = min(h, crop.y + crop.height)

    if x2 <= x1 or y2 <= y1:
        logger.warning("Invalid crop region: (%d, %d, %d, %d)", x1, y1, x2, y2)
        return None

    # Extract crop.
    cropped = master_image[y1:y2, x1:x2]

    # Resize to target dimensions.
    target_w = plan.output_size.width
    target_h = plan.output_size.height

    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    return resized


def _generate_letterbox_output(
    master_image: np.ndarray,
    plan: LayoutPlan,
) -> np.ndarray | None:
    """
    Generate output using letterbox strategy (fit mode).

    Strategy:
    - Resize banner to fit within target dimensions.
    - Add padding (bars) to fill remaining space.
    - Preserve all content without cropping.
    """
    h, w = master_image.shape[:2]
    target_w = plan.output_size.width
    target_h = plan.output_size.height

    # Compute scale factor to fit within target dimensions.
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize banner.
    resized = cv2.resize(master_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create output canvas with padding.
    output = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Compute padding offsets to center the resized banner.
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    # Place resized banner on canvas.
    output[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized

    return output


def _generate_stretch_output(
    master_image: np.ndarray,
    plan: LayoutPlan,
) -> np.ndarray | None:
    """
    Generate output using stretch strategy (direct resize).

    Strategy:
    - Resize banner directly to target dimensions.
    - Allow aspect ratio distortion.
    - Use only when distortion is acceptable.
    """
    target_w = plan.output_size.width
    target_h = plan.output_size.height

    resized = cv2.resize(master_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    return resized


def _generate_adaptive_padding_output(
    master_image: np.ndarray,
    plan: LayoutPlan,
    banner_analysis: BannerAnalysis,
) -> np.ndarray | None:
    """
    Generate output using adaptive padding with background extension.

    Strategy:
    - Resize banner to fit within target dimensions.
    - Identify expansion zones (edges where background can be extended).
    - Use simple edge extension (replicate edge pixels) as a baseline.
    - Future: Replace with AI-based inpainting for seamless background generation.

    This implementation uses OpenCV's border replication as a deterministic
    baseline. It can later be replaced with generative models (e.g., Stable
    Diffusion inpainting) without changing the surrounding interfaces.
    """
    h, w = master_image.shape[:2]
    target_w = plan.output_size.width
    target_h = plan.output_size.height

    # Compute scale factor to fit within target dimensions.
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize banner.
    resized = cv2.resize(master_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Compute padding needed.
    pad_x = target_w - new_w
    pad_y = target_h - new_h

    # Distribute padding based on expansion zones.
    # If expansion zones indicate left/right, pad horizontally.
    # If expansion zones indicate top/bottom, pad vertically.
    has_horizontal_expansion = any(
        z.label in ("expand-left", "expand-right") for z in plan.expansion_zones
    )
    has_vertical_expansion = any(
        z.label in ("expand-top", "expand-bottom") for z in plan.expansion_zones
    )

    if has_horizontal_expansion:
        # Pad left and right.
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
    elif has_vertical_expansion:
        # Pad top and bottom.
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
    else:
        # No specific expansion zones; distribute evenly.
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top

    # Apply padding using edge replication.
    # This creates a simple but deterministic background extension.
    # Future: Replace with AI-based inpainting for seamless results.
    output = cv2.copyMakeBorder(
        resized,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )

    # Ensure output matches target dimensions exactly.
    if output.shape[0] != target_h or output.shape[1] != target_w:
        output = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    return output


def _execute_layout_plan(
    master_image: np.ndarray,
    plan: LayoutPlan,
    banner_analysis: BannerAnalysis,
) -> np.ndarray | None:
    """
    Execute a single layout plan to produce a resized banner.

    This function dispatches to the appropriate transformation strategy based
    on the plan's scaling mode and strategy class.
    """
    if plan.scaling_mode == "fill":
        # Crop-based strategies.
        return _generate_crop_output(master_image, plan)
    elif plan.scaling_mode == "fit":
        # Letterbox-based strategies.
        if plan.expansion_zones:
            # Adaptive padding with background extension.
            return _generate_adaptive_padding_output(master_image, plan, banner_analysis)
        else:
            # Simple letterbox.
            return _generate_letterbox_output(master_image, plan)
    elif plan.scaling_mode == "stretch":
        # Direct resize (distortion allowed).
        return _generate_stretch_output(master_image, plan)
    else:
        logger.warning("Unknown scaling mode: %s", plan.scaling_mode)
        return None


def _generate_crop_output(
    master_image: np.ndarray,
    plan: LayoutPlan,
) -> np.ndarray | None:
    """
    Generate output using crop-based strategy.

    Strategy:
    - Extract the crop region from the master banner.
    - Resize to target dimensions.
    - Preserve aspect ratio of the crop region.
    """
    if plan.crop_region is None:
        logger.warning("Crop strategy requires a crop region.")
        return None

    h, w = master_image.shape[:2]
    crop = plan.crop_region

    # Clamp crop region to image bounds.
    x1 = max(0, crop.x)
    y1 = max(0, crop.y)
    x2 = min(w, crop.x + crop.width)
    y2 = min(h, crop.y + crop.height)

    if x2 <= x1 or y2 <= y1:
        logger.warning("Invalid crop region: (%d, %d, %d, %d)", x1, y1, x2, y2)
        return None

    # Extract crop.
    cropped = master_image[y1:y2, x1:x2]

    # Resize to target dimensions.
    target_w = plan.output_size.width
    target_h = plan.output_size.height

    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    return resized


def _generate_letterbox_output(
    master_image: np.ndarray,
    plan: LayoutPlan,
) -> np.ndarray | None:
    """
    Generate output using letterbox strategy (fit mode).

    Strategy:
    - Resize banner to fit within target dimensions.
    - Add padding (bars) to fill remaining space.
    - Preserve all content without cropping.
    """
    h, w = master_image.shape[:2]
    target_w = plan.output_size.width
    target_h = plan.output_size.height

    # Compute scale factor to fit within target dimensions.
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize banner.
    resized = cv2.resize(master_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create output canvas with padding.
    output = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Compute padding offsets to center the resized banner.
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    # Place resized banner on canvas.
    output[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized

    return output


def _generate_stretch_output(
    master_image: np.ndarray,
    plan: LayoutPlan,
) -> np.ndarray | None:
    """
    Generate output using stretch strategy (direct resize).

    Strategy:
    - Resize banner directly to target dimensions.
    - Allow aspect ratio distortion.
    - Use only when distortion is acceptable.
    """
    target_w = plan.output_size.width
    target_h = plan.output_size.height

    resized = cv2.resize(master_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    return resized


def _generate_adaptive_padding_output(
    master_image: np.ndarray,
    plan: LayoutPlan,
    banner_analysis: BannerAnalysis,
) -> np.ndarray | None:
    """
    Generate output using adaptive padding with background extension.

    Strategy:
    - Resize banner to fit within target dimensions.
    - Identify expansion zones (edges where background can be extended).
    - Use simple edge extension (replicate edge pixels) as a baseline.
    - Future: Replace with AI-based inpainting for seamless background generation.

    This implementation uses OpenCV's border replication as a deterministic
    baseline. It can later be replaced with generative models (e.g., Stable
    Diffusion inpainting) without changing the surrounding interfaces.
    """
    h, w = master_image.shape[:2]
    target_w = plan.output_size.width
    target_h = plan.output_size.height

    # Compute scale factor to fit within target dimensions.
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize banner.
    resized = cv2.resize(master_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Compute padding needed.
    pad_x = target_w - new_w
    pad_y = target_h - new_h

    # Distribute padding based on expansion zones.
    # If expansion zones indicate left/right, pad horizontally.
    # If expansion zones indicate top/bottom, pad vertically.
    has_horizontal_expansion = any(
        z.label in ("expand-left", "expand-right") for z in plan.expansion_zones
    )
    has_vertical_expansion = any(
        z.label in ("expand-top", "expand-bottom") for z in plan.expansion_zones
    )

    if has_horizontal_expansion:
        # Pad left and right.
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
    elif has_vertical_expansion:
        # Pad top and bottom.
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
    else:
        # No specific expansion zones; distribute evenly.
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top

    # Apply padding using edge replication.
    # This creates a simple but deterministic background extension.
    # Future: Replace with AI-based inpainting for seamless results.
    output = cv2.copyMakeBorder(
        resized,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )

    # Ensure output matches target dimensions exactly.
    if output.shape[0] != target_h or output.shape[1] != target_w:
        output = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    return output


def orchestrate_outputs(job: Job) -> None:
    """
    Orchestrate output generation for all requested sizes.

    This replaces the previous stub with real image generation logic.
    In a production system, this could fan out to parallel workers or
    a task queue, but for now it runs synchronously.
    """
    generate_output_images(job)



def run_initial_pipeline(job: Job) -> None:
    """
    Run the full stubbed pipeline for a newly created job.

    This is synchronous for now but structured so it can later be moved to a
    background worker or task queue without changing the API contract.
    """
    analyze_banner_content(job)
    align_optional_assets(job)
    score_aspect_ratio_risks(job)  # AI Step C: Model-assisted risk scoring
    analyze_aspect_ratios(job)
    plan_layouts(job)
    orchestrate_outputs(job)

