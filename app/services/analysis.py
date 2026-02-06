from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
import pytesseract

from app.api.v1.schemas import JobStatus, OutputSize
from app.models.jobs import AssetAlignment, BannerAnalysis, Job, Region


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


def plan_layouts(job: Job) -> None:
    """
    Derive simple, deterministic layout strategy placeholders per bucket.

    These are NOT real layout computations; they only provide structure that
    later AI-driven logic can plug into.
    """
    strategies: Dict[str, str] = {}

    if job.aspect_ratio_buckets.get("similar"):
        strategies["similar"] = "centered-safe"
    if job.aspect_ratio_buckets.get("moderate"):
        strategies["moderate"] = "focus-preserving-crop"
    if job.aspect_ratio_buckets.get("extreme"):
        strategies["extreme"] = "edge-anchored-with-padding"

    job.layout_plan = strategies
    job.status = JobStatus.PLANNING


def orchestrate_outputs(job: Job) -> None:
    """
    Stubbed output-generation orchestration.

    In a real system this would fan out asynchronous resize tasks per output
    size and layout plan. For now, it simply marks the job as completed to
    reflect that the planning pipeline has finished.
    """
    # Placeholder for future parallel AI/image-processing calls.
    job.status = JobStatus.COMPLETED


def run_initial_pipeline(job: Job) -> None:
    """
    Run the full stubbed pipeline for a newly created job.

    This is synchronous for now but structured so it can later be moved to a
    background worker or task queue without changing the API contract.
    """
    analyze_banner_content(job)
    align_optional_assets(job)
    analyze_aspect_ratios(job)
    plan_layouts(job)
    orchestrate_outputs(job)

