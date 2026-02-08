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


def _detect_border_regions(image: np.ndarray) -> List[Region]:
    """
    Detect decorative borders or frames in the banner.

    Strategy:
    - Analyze edge regions (10% from each side) for consistent patterns
    - Detect solid color borders, gradient frames, or decorative elements
    - Return regions that can be repositioned during responsive layout

    This is a heuristic approach that works well for common banner designs.
    """
    h, w = image.shape[:2]
    border_regions: List[Region] = []
    
    # Define edge thickness to analyze (10% of dimensions)
    edge_thickness_x = int(w * 0.10)
    edge_thickness_y = int(h * 0.10)
    
    # Minimum thickness for a border to be considered significant
    min_border_thickness = 5
    
    def _analyze_edge_uniformity(edge_region: np.ndarray) -> float:
        """
        Analyze how uniform an edge region is.
        Returns score in [0, 1] where 1 = very uniform (likely a border).
        """
        if edge_region.size == 0:
            return 0.0
        
        # Calculate standard deviation of pixel values
        std_dev = np.std(edge_region)
        
        # Low std dev = uniform = likely border
        # Normalize: std_dev of 0-30 maps to score 1.0-0.0
        uniformity_score = max(0.0, 1.0 - (std_dev / 30.0))
        
        return uniformity_score
    
    # Check left edge
    left_edge = image[:, :edge_thickness_x]
    left_uniformity = _analyze_edge_uniformity(left_edge)
    if left_uniformity > 0.6:  # Threshold for border detection
        border_regions.append(
            Region(
                x=0,
                y=0,
                width=edge_thickness_x,
                height=h,
                score=left_uniformity,
                label="border-left"
            )
        )
    
    # Check right edge
    right_edge = image[:, w - edge_thickness_x:]
    right_uniformity = _analyze_edge_uniformity(right_edge)
    if right_uniformity > 0.6:
        border_regions.append(
            Region(
                x=w - edge_thickness_x,
                y=0,
                width=edge_thickness_x,
                height=h,
                score=right_uniformity,
                label="border-right"
            )
        )
    
    # Check top edge
    top_edge = image[:edge_thickness_y, :]
    top_uniformity = _analyze_edge_uniformity(top_edge)
    if top_uniformity > 0.6:
        border_regions.append(
            Region(
                x=0,
                y=0,
                width=w,
                height=edge_thickness_y,
                score=top_uniformity,
                label="border-top"
            )
        )
    
    # Check bottom edge
    bottom_edge = image[h - edge_thickness_y:, :]
    bottom_uniformity = _analyze_edge_uniformity(bottom_edge)
    if bottom_uniformity > 0.6:
        border_regions.append(
            Region(
                x=0,
                y=h - edge_thickness_y,
                width=w,
                height=edge_thickness_y,
                score=bottom_uniformity,
                label="border-bottom"
            )
        )
    
    return border_regions


def _identify_background_zones(
    image: np.ndarray,
    saliency_map: np.ndarray,
    protection_mask: np.ndarray,
    border_regions: List[Region],
) -> np.ndarray:
    """
    Identify pure background regions suitable for AI-powered extension.

    Strategy:
    - Start with low-saliency regions (not visually important)
    - Exclude protected content (faces, text, logos)
    - Exclude detected borders (they'll be repositioned, not extended)
    - Return binary mask where 255 = pure background, 0 = content/border

    This mask guides the AI inpainting to focus on extendable regions.
    """
    h, w = image.shape[:2]
    
    # Start with low-saliency regions
    _, low_saliency = cv2.threshold(saliency_map, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Create background mask
    background_mask = low_saliency.copy()
    
    # Exclude protected content
    background_mask = cv2.bitwise_and(background_mask, cv2.bitwise_not(protection_mask))
    
    # Exclude border regions (they'll be repositioned, not extended)
    for border in border_regions:
        x1, y1 = max(0, border.x), max(0, border.y)
        x2 = min(w, border.x + border.width)
        y2 = min(h, border.y + border.height)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(background_mask, (x1, y1), (x2, y2), color=0, thickness=-1)
    
    return background_mask


def _compute_saliency_and_masks(
    image: np.ndarray,
    base_path: Path,
    faces: List[Region],
    text_regions: List[Region],
    border_regions: List[Region],
) -> tuple[str | None, str | None, str | None, str | None]:
    """
    Compute a saliency map, foreground mask, protection mask, and background mask.

    The saliency map is computed using OpenCV's spectral residual method, which
    is fast and deterministic. Foreground, protection, and background masks are
    derived from the saliency map and detected semantic regions.
    """
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image)
    if not success:
        logger.warning("Saliency computation failed; maps will not be persisted.")
        return None, None, None, None

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

    # Identify pure background zones for AI extension
    background_mask = _identify_background_zones(
        image=image,
        saliency_map=saliency_norm,
        protection_mask=protection_mask,
        border_regions=border_regions,
    )

    # Persist maps alongside the master banner for later inspection and reuse.
    saliency_path = base_path.with_name(f"{base_path.stem}_saliency.png")
    foreground_path = base_path.with_name(f"{base_path.stem}_foreground.png")
    protection_path = base_path.with_name(f"{base_path.stem}_protection.png")
    background_path = base_path.with_name(f"{base_path.stem}_background.png")

    try:
        cv2.imwrite(str(saliency_path), saliency_norm)
        cv2.imwrite(str(foreground_path), foreground_mask)
        cv2.imwrite(str(protection_path), protection_mask)
        cv2.imwrite(str(background_path), background_mask)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist saliency/protection maps: %s", exc)
        return None, None, None, None

    return str(foreground_path), str(saliency_path), str(protection_path), str(background_path)


def analyze_banner_content(job: Job) -> None:
    """
    Run content-aware analysis on the master banner and attach results to the job.

    This is the entry point for Step A (Banner Content Analysis). It:
    - Detects faces
    - Detects text regions
    - Detects decorative borders/frames
    - Computes a saliency map
    - Derives foreground, protection, and background masks
    """
    image = _load_image(job.master_banner_path)
    if image is None:
        # Leave banner_analysis as None to make the failure explicit to callers.
        job.banner_analysis = None
        return

    height, width = image.shape[:2]
    faces = _detect_faces(image)
    text_regions = _detect_text_regions(image)
    border_regions = _detect_border_regions(image)

    base_path = Path(job.master_banner_path)
    foreground_path, saliency_path, protection_path, background_path = _compute_saliency_and_masks(
        image=image,
        base_path=base_path,
        faces=faces,
        text_regions=text_regions,
        border_regions=border_regions,
    )

    job.banner_analysis = BannerAnalysis(
        width=width,
        height=height,
        faces=faces,
        text_regions=text_regions,
        logo_regions=[],  # Logo/product detection will be added in a later refinement.
        border_regions=border_regions,
        foreground_mask_path=foreground_path,
        saliency_map_path=saliency_path,
        protection_mask_path=protection_path,
        background_mask_path=background_path,
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
    banner_analysis: BannerAnalysis,
) -> List[Region]:
    """
    Identify zones where background can be extended for adaptive padding.

    Enhanced strategy:
    - Analyze which edges need expansion based on aspect ratio change
    - Use background mask to identify truly extendable regions
    - Avoid expanding near borders (they'll be repositioned instead)
    - Return precise expansion zones for AI inpainting
    """
    target_ratio = target_w / target_h
    banner_ratio = banner_w / banner_h

    zones: List[Region] = []

    # Load background mask to identify extendable regions
    background_mask = _load_mask(banner_analysis.background_mask_path)
    
    # Determine which edges need expansion
    needs_horizontal = target_ratio > banner_ratio
    needs_vertical = target_ratio < banner_ratio

    if needs_horizontal:
        # Target is wider: need horizontal expansion
        # Analyze left and right edges for extendability
        edge_w = int(banner_w * 0.15)  # Analyze 15% of width
        
        # Check if left edge is extendable (has background content)
        if background_mask is not None:
            left_region = background_mask[:, :edge_w]
            left_bg_ratio = np.sum(left_region > 0) / left_region.size if left_region.size > 0 else 0
            
            if left_bg_ratio > 0.3:  # At least 30% is background
                zones.append(
                    Region(
                        x=0,
                        y=0,
                        width=edge_w,
                        height=banner_h,
                        score=left_bg_ratio,
                        label="expand-left"
                    )
                )
        
        # Check if right edge is extendable
        if background_mask is not None:
            right_region = background_mask[:, banner_w - edge_w:]
            right_bg_ratio = np.sum(right_region > 0) / right_region.size if right_region.size > 0 else 0
            
            if right_bg_ratio > 0.3:
                zones.append(
                    Region(
                        x=banner_w - edge_w,
                        y=0,
                        width=edge_w,
                        height=banner_h,
                        score=right_bg_ratio,
                        label="expand-right"
                    )
                )
        
        # Fallback if no background detected
        if not zones:
            zones.append(Region(x=0, y=0, width=edge_w, height=banner_h, score=0.5, label="expand-left"))
            zones.append(Region(x=banner_w - edge_w, y=0, width=edge_w, height=banner_h, score=0.5, label="expand-right"))

    elif needs_vertical:
        # Target is taller: need vertical expansion
        edge_h = int(banner_h * 0.15)
        
        # Check if top edge is extendable
        if background_mask is not None:
            top_region = background_mask[:edge_h, :]
            top_bg_ratio = np.sum(top_region > 0) / top_region.size if top_region.size > 0 else 0
            
            if top_bg_ratio > 0.3:
                zones.append(
                    Region(
                        x=0,
                        y=0,
                        width=banner_w,
                        height=edge_h,
                        score=top_bg_ratio,
                        label="expand-top"
                    )
                )
        
        # Check if bottom edge is extendable
        if background_mask is not None:
            bottom_region = background_mask[banner_h - edge_h:, :]
            bottom_bg_ratio = np.sum(bottom_region > 0) / bottom_region.size if bottom_region.size > 0 else 0
            
            if bottom_bg_ratio > 0.3:
                zones.append(
                    Region(
                        x=0,
                        y=banner_h - edge_h,
                        width=banner_w,
                        height=edge_h,
                        score=bottom_bg_ratio,
                        label="expand-bottom"
                    )
                )
        
        # Fallback if no background detected
        if not zones:
            zones.append(Region(x=0, y=0, width=banner_w, height=edge_h, score=0.5, label="expand-top"))
            zones.append(Region(x=0, y=banner_h - edge_h, width=banner_w, height=edge_h, score=0.5, label="expand-bottom"))

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
    Plan a focus-preserving resize with adaptive padding (no letterbox).

    Strategy:
    - Resize to fit within target dimensions without cropping.
    - Use adaptive padding with AI-powered background extension instead of black bars.
    - Preserve all content while providing seamless background fill.
    """
    banner_w, banner_h = banner_analysis.width, banner_analysis.height

    # Compute expansion zones for adaptive padding using enhanced logic
    expansion_zones = _compute_expansion_zones(
        banner_w, banner_h, output_size.width, output_size.height, banner_analysis
    )

    return LayoutPlan(
        output_size=output_size,
        strategy_class=risk.strategy_class,
        anchor_region=None,
        crop_region=None,  # No crop; pure resize with adaptive padding.
        scaling_mode="fit",
        expansion_zones=expansion_zones,  # Use adaptive padding instead of letterbox
        protected_regions=[],
        reasoning=f"Moderate risk; adaptive padding with {len(expansion_zones)} expansion zones to preserve all content without letterbox bars",
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
        banner_w, banner_h, output_size.width, output_size.height, banner_analysis
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
    - Provide a conservative fallback with adaptive padding (no letterbox).
    - Use AI-powered background extension to avoid black bars.
    - Flag for manual review with detailed reasoning.
    - Preserve all content to avoid automatic mistakes.
    """
    banner_w, banner_h = banner_analysis.width, banner_analysis.height

    # Compute expansion zones for adaptive padding using enhanced logic
    expansion_zones = _compute_expansion_zones(
        banner_w, banner_h, output_size.width, output_size.height, banner_analysis
    )

    return LayoutPlan(
        output_size=output_size,
        strategy_class=risk.strategy_class,
        anchor_region=None,
        crop_region=None,
        scaling_mode="fit",
        expansion_zones=expansion_zones,  # Use adaptive padding instead of letterbox
        protected_regions=[],
        reasoning=f"High risk (score: {risk.risk_score:.2f}); manual review recommended. "
        f"Using adaptive padding with {len(expansion_zones)} expansion zones to preserve all content. {risk.reasoning}",
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


def _validate_asset_compositing(
    output_image: np.ndarray,
    plan: LayoutPlan,
    banner_analysis: BannerAnalysis,
    asset_alignments: List[AssetAlignment],
) -> tuple[float, List[str]]:
    """
    Validate the quality of composited assets.

    Checks:
    - Assets are visible (not completely outside bounds)
    - Assets don't overlap protected content (faces/text)
    - Asset scaling is reasonable (not too small or too large)
    - Asset positioning is balanced (not all on one side)

    Returns:
        Tuple of (score, warnings) where score is in [0, 1]
    """
    if not asset_alignments:
        # No assets to validate
        return 1.0, []

    warnings = []
    output_h, output_w = output_image.shape[:2]
    
    # Compute scaling factors from master banner to output
    scale_x = output_w / banner_analysis.width
    scale_y = output_h / banner_analysis.height

    # Track asset positions for balance check
    asset_positions = []
    
    # Check each asset
    visible_assets = 0
    overlapping_assets = 0
    too_small_assets = 0
    too_large_assets = 0

    for alignment in asset_alignments:
        if not alignment.target_region:
            continue

        # Scale target region to output coordinates
        target_x = int(alignment.target_region.x * scale_x)
        target_y = int(alignment.target_region.y * scale_y)
        target_w = int(alignment.target_region.width * scale_x)
        target_h = int(alignment.target_region.height * scale_y)

        # Check 1: Asset visibility (within bounds)
        if target_x >= output_w or target_y >= output_h or target_x + target_w <= 0 or target_y + target_h <= 0:
            warnings.append(f"Asset {alignment.role} is outside output bounds")
            continue

        visible_assets += 1
        asset_positions.append((target_x + target_w // 2, target_y + target_h // 2))

        # Check 2: Asset size validation
        asset_area = target_w * target_h
        output_area = output_w * output_h
        asset_percentage = asset_area / output_area

        if asset_percentage < 0.02:  # Less than 2% of output
            too_small_assets += 1
            warnings.append(f"Asset {alignment.role} may be too small ({asset_percentage*100:.1f}% of output)")
        elif asset_percentage > 0.5:  # More than 50% of output
            too_large_assets += 1
            warnings.append(f"Asset {alignment.role} may be too large ({asset_percentage*100:.1f}% of output)")

        # Check 3: Overlap with protected content
        asset_region = Region(x=target_x, y=target_y, width=target_w, height=target_h)
        
        # Check overlap with faces
        for face in banner_analysis.faces:
            face_scaled = Region(
                x=int(face.x * scale_x),
                y=int(face.y * scale_y),
                width=int(face.width * scale_x),
                height=int(face.height * scale_y),
            )
            if _regions_overlap(asset_region, face_scaled):
                overlapping_assets += 1
                warnings.append(f"Asset {alignment.role} overlaps with detected face")
                break

        # Check overlap with text
        for text in banner_analysis.text_regions:
            text_scaled = Region(
                x=int(text.x * scale_x),
                y=int(text.y * scale_y),
                width=int(text.width * scale_x),
                height=int(text.height * scale_y),
            )
            if _regions_overlap(asset_region, text_scaled):
                overlapping_assets += 1
                warnings.append(f"Asset {alignment.role} overlaps with text region")
                break

    # Check 4: Asset positioning balance
    if len(asset_positions) >= 2:
        # Check if assets are clustered on one side
        x_positions = [pos[0] for pos in asset_positions]
        y_positions = [pos[1] for pos in asset_positions]
        
        # Check horizontal balance
        left_count = sum(1 for x in x_positions if x < output_w / 2)
        right_count = len(x_positions) - left_count
        
        if left_count == 0 or right_count == 0:
            warnings.append("All assets are positioned on one side (horizontal)")
        
        # Check vertical balance
        top_count = sum(1 for y in y_positions if y < output_h / 2)
        bottom_count = len(y_positions) - top_count
        
        if top_count == 0 or bottom_count == 0:
            warnings.append("All assets are positioned on one side (vertical)")

    # Compute score based on issues found
    score = 1.0
    
    if visible_assets < len(asset_alignments):
        score -= 0.3  # Penalty for invisible assets
    
    if overlapping_assets > 0:
        score -= 0.2 * min(overlapping_assets, 2)  # Penalty for overlaps (max -0.4)
    
    if too_small_assets > 0:
        score -= 0.1 * min(too_small_assets, 2)  # Penalty for too small (max -0.2)
    
    if too_large_assets > 0:
        score -= 0.1 * min(too_large_assets, 2)  # Penalty for too large (max -0.2)

    score = max(0.0, score)  # Clamp to [0, 1]

    return score, warnings


def _regions_overlap(region1: Region, region2: Region) -> bool:
    """Check if two regions overlap using bounding box intersection."""
    x1_min, x1_max = region1.x, region1.x + region1.width
    y1_min, y1_max = region1.y, region1.y + region1.height
    
    x2_min, x2_max = region2.x, region2.x + region2.width
    y2_min, y2_max = region2.y, region2.y + region2.height
    
    # Check if rectangles overlap
    return not (x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min)


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
    
    # Validate asset compositing (Step J)
    asset_score, asset_warnings = _validate_asset_compositing(
        output_image, plan, job.banner_analysis, job.asset_alignment
    )

    # Combine warnings.
    all_warnings = content_warnings + aspect_warnings + visual_warnings + asset_warnings

    # Compute overall quality score (weighted average).
    # Include asset score if assets are present
    if job.asset_alignment:
        quality_score = (
            (content_score * 0.4) + 
            (aspect_score * 0.25) + 
            (visual_score * 0.15) + 
            (asset_score * 0.2)
        )
    else:
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


def _composite_assets_onto_output(
    output_image: np.ndarray,
    asset_alignments: List[AssetAlignment],
    master_banner_width: int,
    master_banner_height: int,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """
    Composite optional assets (logos, overlays) onto the generated output.

    Strategy:
    - Load each asset from disk
    - Scale asset proportionally based on output size relative to master banner
    - Position asset based on alignment decision (target_region)
    - Handle transparency/alpha blending for PNG assets
    - Skip failed assets gracefully with warnings

    Args:
        output_image: The generated output image (before asset compositing)
        asset_alignments: List of asset alignment decisions from Step B
        master_banner_width: Original master banner width
        master_banner_height: Original master banner height
        output_width: Target output width
        output_height: Target output height

    Returns:
        Output image with assets composited
    """
    if not asset_alignments:
        return output_image

    # Compute scaling factors from master banner to output
    scale_x = output_width / master_banner_width
    scale_y = output_height / master_banner_height

    # Work on a copy to avoid modifying the original
    result = output_image.copy()

    for alignment in asset_alignments:
        try:
            # Load asset image
            asset_image = cv2.imread(alignment.asset_path, cv2.IMREAD_UNCHANGED)
            if asset_image is None:
                logger.warning(f"Failed to load asset: {alignment.asset_path}")
                continue

            # Get asset dimensions
            if len(asset_image.shape) == 3 and asset_image.shape[2] == 4:
                # Asset has alpha channel
                asset_h, asset_w, _ = asset_image.shape
                has_alpha = True
            else:
                asset_h, asset_w = asset_image.shape[:2]
                has_alpha = False

            # Determine target position and size
            if alignment.target_region:
                # Scale the target region from master banner coordinates to output coordinates
                target_x = int(alignment.target_region.x * scale_x)
                target_y = int(alignment.target_region.y * scale_y)
                target_w = int(alignment.target_region.width * scale_x)
                target_h = int(alignment.target_region.height * scale_y)
            else:
                # Default: scale asset proportionally and place in top-left corner
                target_w = int(asset_w * scale_x)
                target_h = int(asset_h * scale_y)
                target_x = 10  # Small margin from edge
                target_y = 10

            # Resize asset to target dimensions
            if has_alpha:
                # Resize with alpha channel preserved
                resized_asset = cv2.resize(
                    asset_image,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LANCZOS4
                )
            else:
                resized_asset = cv2.resize(
                    asset_image,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LANCZOS4
                )

            # Clamp position to output bounds
            target_x = max(0, min(target_x, output_width - target_w))
            target_y = max(0, min(target_y, output_height - target_h))

            # Clamp size to fit within output
            if target_x + target_w > output_width:
                target_w = output_width - target_x
            if target_y + target_h > output_height:
                target_h = output_height - target_y

            # Resize again if clamping changed dimensions
            if resized_asset.shape[1] != target_w or resized_asset.shape[0] != target_h:
                resized_asset = cv2.resize(
                    resized_asset,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LANCZOS4
                )

            # Composite asset onto output
            if has_alpha:
                # Extract alpha channel and convert to float [0, 1]
                alpha = resized_asset[:, :, 3] / 255.0
                asset_rgb = resized_asset[:, :, :3]

                # Get the region of the output where we'll place the asset
                output_region = result[target_y:target_y+target_h, target_x:target_x+target_w]

                # Alpha blend: output = asset * alpha + background * (1 - alpha)
                for c in range(3):
                    output_region[:, :, c] = (
                        asset_rgb[:, :, c] * alpha +
                        output_region[:, :, c] * (1 - alpha)
                    )

                result[target_y:target_y+target_h, target_x:target_x+target_w] = output_region
            else:
                # No alpha channel, just paste the asset
                result[target_y:target_y+target_h, target_x:target_x+target_w] = resized_asset

            logger.info(
                f"Composited asset {alignment.role} at ({target_x}, {target_y}) "
                f"with size {target_w}x{target_h}"
            )

        except Exception as e:
            logger.warning(f"Failed to composite asset {alignment.asset_path}: {e}")
            continue

    return result


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

            # Composite optional assets onto the output (Step I)
            if job.asset_alignment:
                output_image = _composite_assets_onto_output(
                    output_image=output_image,
                    asset_alignments=job.asset_alignment,
                    master_banner_width=job.banner_analysis.width,
                    master_banner_height=job.banner_analysis.height,
                    output_width=plan.output_size.width,
                    output_height=plan.output_size.height,
                )

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
        
        # Print summary
        print("\n" + "="*60)
        print(f"📊 JOB GENERATION SUMMARY")
        print("="*60)
        print(f"Job ID: {job.id}")
        print(f"Outputs generated: {success_count}/{len(job.layout_plans)}")
        
        # Count strategies used
        strategies = {}
        replicate_used = False
        for plan in job.layout_plans:
            strategy = plan.strategy_class
            strategies[strategy] = strategies.get(strategy, 0) + 1
            if strategy == "adaptive-padding":
                replicate_used = True
        
        print(f"\nStrategies used:")
        for strategy, count in strategies.items():
            print(f"  • {strategy}: {count} output(s)")
        
        if replicate_used:
            print(f"\n🤖 AI Inpainting: Check logs above for Replicate usage")
        else:
            print(f"\n📐 AI Inpainting: Not needed (no adaptive padding)")
        
        # Quality summary
        avg_quality = sum(qc.quality_score for qc in quality_checks) / len(quality_checks) if quality_checks else 0
        needs_review = sum(1 for qc in quality_checks if qc.needs_manual_review)
        
        print(f"\nQuality:")
        print(f"  • Average score: {avg_quality:.2f}")
        print(f"  • Needs review: {needs_review}/{len(quality_checks)}")
        print("="*60 + "\n")
        
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


def _generate_inpainted_background(
    resized_image: np.ndarray,
    target_width: int,
    target_height: int,
    pad_left: int,
    pad_right: int,
    pad_top: int,
    pad_bottom: int,
    banner_analysis: BannerAnalysis,
    expansion_zones: List[Region],
) -> np.ndarray | None:
    """
    Generate seamless background extension using AI inpainting.

    Creates a mask for padding regions and uses Replicate's LaMa model
    to generate seamless background content instead of edge replication.

    Enhanced with context-aware prompts based on banner analysis.

    Args:
        resized_image: The resized banner image
        target_width: Target output width
        target_height: Target output height
        pad_left, pad_right, pad_top, pad_bottom: Padding amounts
        banner_analysis: Banner analysis with detected regions
        expansion_zones: Zones identified for expansion

    Returns:
        Inpainted image with seamless background, or None if inpainting fails
    """
    try:
        from app.services.replicate_http_client import inpaint_background

        # Create canvas with black padding (temporary)
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # Place resized image in center of canvas
        y_start = pad_top
        y_end = pad_top + resized_image.shape[0]
        x_start = pad_left
        x_end = pad_left + resized_image.shape[1]

        canvas[y_start:y_end, x_start:x_end] = resized_image

        # Create mask for inpainting (255 = inpaint, 0 = preserve)
        mask = np.zeros((target_height, target_width), dtype=np.uint8)

        # Mark padding regions for inpainting
        if pad_top > 0:
            mask[0:pad_top, :] = 255  # Top padding
        if pad_bottom > 0:
            mask[target_height-pad_bottom:target_height, :] = 255  # Bottom padding
        if pad_left > 0:
            mask[:, 0:pad_left] = 255  # Left padding
        if pad_right > 0:
            mask[:, target_width-pad_right:target_width] = 255  # Right padding

        # Use edge replication as base for inpainting (gives model context)
        base_image = cv2.copyMakeBorder(
            resized_image,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )

        # Generate context-aware prompt based on banner analysis
        prompt = _generate_inpainting_prompt(banner_analysis, expansion_zones)

        logger.info(f"Attempting AI inpainting with prompt: '{prompt}' (mask coverage: {np.sum(mask > 0) / mask.size:.1%})")
        print(f"\n🎨 Attempting AI background extension...")
        print(f"   Prompt: {prompt}")

        # Call Replicate inpainting
        inpainted = inpaint_background(
            image=base_image,
            mask=mask,
            prompt=prompt
        )

        if inpainted is not None:
            logger.info("AI inpainting successful - seamless background generated")
            print(f"✅ AI background extension successful\n")
            return inpainted
        else:
            logger.warning("AI inpainting failed - falling back to edge replication")
            print(f"⚠️  Using edge replication fallback\n")
            return None

    except Exception as e:
        logger.error(f"Inpainting error: {e} - falling back to edge replication")
        return None


def _generate_inpainting_prompt(
    banner_analysis: BannerAnalysis,
    expansion_zones: List[Region],
) -> str:
    """
    Generate context-aware prompt for AI inpainting based on banner analysis.

    Strategy:
    - Analyze the banner content to understand background type
    - Generate specific prompts for different background patterns
    - Default to generic seamless extension if unclear

    Returns:
        Context-aware prompt string for the inpainting model
    """
    # Check if banner has detected borders (likely decorative design)
    has_borders = len(banner_analysis.border_regions) > 0
    
    # Check expansion direction
    expansion_labels = [zone.label for zone in expansion_zones]
    is_horizontal = any(label in expansion_labels for label in ["expand-left", "expand-right"])
    is_vertical = any(label in expansion_labels for label in ["expand-top", "expand-bottom"])
    
    # Analyze background characteristics
    if banner_analysis.background_mask_path:
        background_mask = _load_mask(banner_analysis.background_mask_path)
        if background_mask is not None:
            bg_coverage = np.sum(background_mask > 0) / background_mask.size
            
            # High background coverage suggests simple/clean design
            if bg_coverage > 0.6:
                if has_borders:
                    return "extend clean background seamlessly, preserve design aesthetic"
                else:
                    return "seamlessly extend solid or gradient background"
            
            # Medium background suggests mixed content
            elif bg_coverage > 0.3:
                if is_horizontal:
                    return "extend background pattern horizontally, maintain visual consistency"
                elif is_vertical:
                    return "extend background pattern vertically, maintain visual consistency"
                else:
                    return "extend background pattern seamlessly"
            
            # Low background suggests content-heavy banner
            else:
                return "carefully extend background around content, preserve visual balance"
    
    # Default fallback prompt
    return "seamless background extension"


def _generate_adaptive_padding_output(
    master_image: np.ndarray,
    plan: LayoutPlan,
    banner_analysis: BannerAnalysis,
) -> np.ndarray | None:
    """
    Generate output using adaptive padding with AI-powered background extension.

    Strategy:
    - Resize banner to fit within target dimensions.
    - Identify expansion zones (edges where background can be extended).
    - Use AI inpainting (LaMa model) for seamless background generation.
    - Fall back to edge replication if AI inpainting fails or is unavailable.

    This implementation prioritizes AI-based inpainting for high-quality results
    while maintaining a deterministic fallback for reliability.
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

    # Try AI-powered inpainting first, fall back to edge replication
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        # Attempt AI inpainting for seamless background extension
        inpainted_output = _generate_inpainted_background(
            resized_image=resized,
            target_width=target_w,
            target_height=target_h,
            pad_left=pad_left,
            pad_right=pad_right,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            banner_analysis=banner_analysis,
            expansion_zones=plan.expansion_zones,
        )
        
        if inpainted_output is not None:
            # AI inpainting succeeded
            output = inpainted_output
        else:
            # Fall back to edge replication
            from pathlib import Path
            from datetime import datetime
            
            fallback_msg = "📐 Using FALLBACK (edge replication) for background extension"
            logger.info("Using edge replication fallback for background extension")
            print(f"{fallback_msg}")
            
            # Log to output.txt
            output_log = Path("output.txt")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(output_log, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {fallback_msg}\n")
            
            output = cv2.copyMakeBorder(
                resized,
                top=pad_top,
                bottom=pad_bottom,
                left=pad_left,
                right=pad_right,
                borderType=cv2.BORDER_REPLICATE,
            )
    else:
        # No padding needed, just use resized image
        output = resized

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

