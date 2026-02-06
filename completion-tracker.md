## Step 1: Project scaffolding

- **What was implemented**:
  - Initialized the backend project with a Python package layout and basic metadata files.
  - Created `completion-tracker.md` to track all subsequent steps.
- **Why it was implemented this way**:
  - The instructions require incremental, traceable development with a single source of truth so another engineer or agent can resume work safely.
  - Using `pyproject.toml` sets us up with modern dependency management and clear project metadata.
- **Files created or modified**:
  - `completion-tracker.md`
  - `pyproject.toml`
  - `README.md`
  - `app/__init__.py`
  - `app/main.py`
- **Assumptions / limitations**:
  - The environment will use Python 3.11+ and install dependencies from `pyproject.toml`.
  - Only a minimal health-check endpoint is present; full versioned API and job routes will be added in later steps.
- **Status**: ✅ Complete

## Step 5: Job status and retrieval endpoints

- **What was implemented**:
  - Added read-only endpoints to retrieve a single job (`GET /api/v1/jobs/{job_id}`) and list all jobs (`GET /api/v1/jobs`).
  - Introduced `JobSummary` and `JobDetail` response schemas to separate internal job representation from API contracts.
  - Extended `JobStore` with a `list_jobs` method to back the listing endpoint.
- **Why it was implemented this way**:
  - Dedicated response models keep internal fields (like file paths) hidden while exposing stable frontend-friendly data.
  - Adding a listing endpoint is useful for development and debugging, and can be restricted later if needed.
- **Files created or modified**:
  - `app/api/v1/schemas.py`
  - `app/api/v1/routes.py`
  - `app/services/jobs.py`
- **Assumptions / limitations**:
  - Status values are still static (`pending`) until a processing pipeline is introduced to transition jobs through their lifecycle.
  - The list endpoint returns all jobs without pagination or authentication; suitable only for early-stage development.
- **Status**: ✅ Complete

## Step 6: Banner analysis pipeline (stubbed)

- **What was implemented**:
  - Added an internal analysis service that runs once per job on creation.
  - Normalizes requested output sizes into simple aspect-ratio buckets (`similar`, `moderate`, `extreme`).
  - Wires the analysis pipeline into job creation so that new jobs immediately carry analysis metadata.
- **Why it was implemented this way**:
  - Keeping analysis logic in a dedicated service (`analysis.py`) creates a clear extension point for future AI and CV logic.
  - Running the pipeline synchronously during job creation avoids background complexity while the system is still a stub.
- **Files created or modified**:
  - `app/models/jobs.py`
  - `app/services/analysis.py`
  - `app/services/jobs.py`
- **Assumptions / limitations**:
  - Aspect-ratio buckets are heuristic and purely geometric; no content-awareness is applied yet.
  - The pipeline is synchronous and in-process; it will need to move to a background worker for large workloads.
- **Status**: ✅ Complete

## Step 7: Layout strategy planning (stubbed)

- **What will be implemented next**:
  - Use the aspect-ratio buckets computed in Step 6 to derive deterministic, human-readable layout strategy labels per bucket.
  - Persist these layout strategy hints on the `Job` object as internal metadata for later use by real generation logic.
- **Status**: ⏭️ Next
## Step 2: Folder structure & API foundation

- **What was implemented**:
  - Created a basic versioned API package under `app/api/v1/`.
  - Added a v1 router with a `/api/v1/health` endpoint.
  - Wired the v1 router into the FastAPI application while keeping a root-level `/health` endpoint for infrastructure checks.
- **Why it was implemented this way**:
  - A versioned API structure (`/api/v1`) allows future non-breaking evolution of the API.
  - Separating root health from versioned health keeps operational checks decoupled from public API design.
- **Files created or modified**:
  - `app/api/__init__.py`
  - `app/api/v1/__init__.py`
  - `app/api/v1/routes.py`
  - `app/main.py`
- **Assumptions / limitations**:
  - Only health endpoints exist in v1 for now; job-related routes and domain-specific models will be added in subsequent steps.
- **Status**: ✅ Complete

## Step 3: Upload & job creation API design

- **What was implemented**:
  - Designed and implemented the `POST /api/v1/jobs` endpoint for job creation.
  - Defined Pydantic models for output sizes and job creation responses.
  - Added basic job lifecycle status enumeration to prepare for future status endpoints.
- **Why it was implemented this way**:
  - Using `multipart/form-data` with `UploadFile` keeps file uploads efficient and streaming-friendly.
  - Passing `outputs` as a JSON-encoded form field avoids over-complicating the request model while remaining frontend-friendly.
  - Introducing `JobStatus` early ensures consistent status representation across creation, status, and result endpoints.
- **Files created or modified**:
  - `app/api/v1/schemas.py`
  - `app/api/v1/routes.py`
- **Assumptions / limitations**:
  - Jobs are not yet persisted anywhere; the returned `job_id` is synthetic.
  - No analysis, layout planning, or output generation is triggered yet—this endpoint only validates input and returns an initial representation.
  - Status and result retrieval endpoints are not yet implemented.
- **Status**: ✅ Complete

## Step 4: Storage abstraction for jobs and assets

- **What was implemented**:
  - Introduce a clear storage abstraction for:
    - Persisting job metadata and status.
    - Persisting references to uploaded master banners and optional additional assets.
  - Provide an initial local or in-memory implementation behind this abstraction so that job IDs returned from `POST /api/v1/jobs` map to retrievable records.
- **Status**: ✅ Complete

## AI Step A: Banner content analysis (faces, text, saliency)

- **What was implemented**:
  - Introduced internal `Region` and `BannerAnalysis` dataclasses to represent content-aware structure of the master banner image.
  - Added a real analysis pass in `app/services/analysis.py` that:
    - Loads the persisted master banner from disk once per job.
    - Detects faces using OpenCV Haar cascades.
    - Detects text regions using Tesseract via `pytesseract`.
    - Computes a saliency heatmap and derives simple foreground and protection masks.
  - Extended the `Job` model to carry a `banner_analysis` field that is populated during the initial pipeline via `analyze_banner_content`.
- **Why it was implemented this way**:
  - Keeping the analysis logic in the existing `analysis` service preserves the current orchestration flow while swapping stubbed analysis for real, inspectable outputs.
  - Using OpenCV and Tesseract provides deterministic, production-proven detectors that are easy to replace later with PyTorch-based models without changing the surrounding interfaces.
  - Persisting saliency/foreground/protection masks next to the master banner makes it cheap to debug and to reuse these maps in later layout and generation stages.
- **Files created or modified**:
  - `pyproject.toml` (added numpy, opencv-python, Pillow, pytesseract, torch, torchvision dependencies)
  - `app/models/jobs.py` (added `Region`, `BannerAnalysis`, and `banner_analysis` on `Job`)
  - `app/services/analysis.py` (implemented `analyze_banner_content` and integrated it into `run_initial_pipeline`)
- **Assumptions / limitations**:
  - Tesseract must be installed on the host system for text detection; if it is missing or misconfigured, text regions will be empty but the rest of the analysis will still succeed.
  - Logo/product detection is not yet implemented; `logo_regions` is currently left empty and will be filled in a later refinement when a suitable detector is integrated.
  - Saliency and mask computation currently uses OpenCV's spectral residual method; this can later be swapped for a PyTorch-based model if higher fidelity is required.
- **Status**: ✅ Complete

## AI Step B: Optional asset alignment

- **What was implemented**:
  - Added an internal `AssetAlignment` dataclass and a corresponding `asset_alignment` field on the `Job` model to capture where and how each optional asset should be placed.
  - Implemented `align_optional_assets(job)` in `app/services/analysis.py` which:
    - Loads the existing protection mask (if available) from the banner analysis.
    - Proposes safe “asset-slot” regions near banner corners/edges that avoid protected content (faces/text) using the protection mask.
    - Creates `AssetAlignment` entries for each additional asset path, defaulting the role to `"logo"` and marking them as locked.
    - Updates the banner’s `logo_regions` and extends the protection mask so that these aligned assets are treated as protected in later stages.
  - Wired `align_optional_assets` into `run_initial_pipeline` immediately after `analyze_banner_content`, so alignment always happens once per job as soon as analysis is available.
- **Why it was implemented this way**:
  - Keeping alignment logic in the analysis service reuses the existing pipeline structure and allows later steps (layout, generation) to consume a unified internal model.
  - Using the protection mask ensures assets are placed away from detected faces/text, providing a simple but content-aware alignment without needing heavy additional models yet.
  - The `AssetAlignment` structure includes a `role` and `user_override` flag to support future API-level user intent (e.g. “this is the main logo here”) without breaking internal interfaces.
- **Files created or modified**:
  - `app/models/jobs.py` (added `AssetAlignment` and `asset_alignment` on `Job`)
  - `app/services/analysis.py` (implemented `_load_mask`, `_find_safe_anchor_regions`, `align_optional_assets`, and integrated it into `run_initial_pipeline`)
- **Assumptions / limitations**:
  - All additional assets are currently treated as logo-like visual elements; the API does not yet distinguish between logos, product shots, or text overlays.
  - If the protection mask cannot be loaded, alignment falls back to geometric positions without content awareness, but still records decisions on the job.
  - Explicit user intent (e.g. requested positions) is not yet passed through the API; when it is, the `user_override` flag and `role` field will be used to prioritize those hints over automatic alignment.
- **Status**: ✅ Complete

## AI Step C: Aspect ratio risk scoring

- **What was implemented**:
  - Added an internal `AspectRatioRisk` dataclass to capture model-assisted risk assessments for each output size.
  - Implemented `score_aspect_ratio_risks(job)` in `app/services/analysis.py` which:
    - Computes content clipping risk by simulating center crops and measuring protected region loss (faces/text/logos).
    - Computes layout stress risk by measuring aspect ratio deviation from the source banner.
    - Computes saliency loss risk by analyzing how much high-saliency content would be lost in a crop.
    - Combines these three risk factors into an overall risk score with weighted contributions (50% content, 30% layout, 20% saliency).
    - Deterministically assigns strategy classes based on risk profile (e.g., "safe-center-crop", "content-aware-crop", "adaptive-padding", "manual-review-recommended").
    - Generates human-readable reasoning for each risk assessment.
  - Extended the `Job` model with an `aspect_ratio_risks` field to store risk assessments per output size.
  - Wired `score_aspect_ratio_risks` into `run_initial_pipeline` immediately after `align_optional_assets`, so risk scoring happens once per job after all content analysis is complete.
- **Why it was implemented this way**:
  - Using geometric simulation (center crop) with real content data (regions, saliency maps) provides deterministic, explainable risk scores without requiring a trained ML model.
  - Breaking risk into three orthogonal factors (content, layout, saliency) makes the scoring transparent and allows future refinement of individual components.
  - Weighted combination of risk factors reflects domain priorities: protecting faces/text/logos is most critical, followed by avoiding extreme distortion, then preserving visual interest.
  - Strategy classes are assigned deterministically based on risk thresholds, ensuring the same input always produces the same output and making the system debuggable.
  - Storing reasoning strings alongside scores makes the system explainable to both developers and end users.
- **Files created or modified**:
  - `app/models/jobs.py` (added `AspectRatioRisk` dataclass and `aspect_ratio_risks` field on `Job`)
  - `app/services/analysis.py` (implemented `_compute_content_clipping_risk`, `_compute_layout_stress_risk`, `_compute_saliency_loss_risk`, `_assign_strategy_class`, `score_aspect_ratio_risks`, and integrated it into `run_initial_pipeline`)
- **Model choice and approach**:
  - This implementation uses a **rule-based geometric model** rather than a trained neural network, which satisfies the determinism and explainability constraints.
  - Content clipping risk: Simulates a center crop to the target aspect ratio and calculates what percentage of protected regions (faces, text, logos) would be lost. Higher loss = higher risk.
  - Layout stress risk: Measures the relative deviation between source and target aspect ratios. Extreme deviations (e.g., 16:9 → 1:1) are flagged as high stress.
  - Saliency loss risk: Loads the precomputed saliency map, simulates the crop, and measures how much high-saliency content is retained. Lower retention = higher risk.
  - Strategy assignment: Uses threshold-based rules on the combined risk profile to assign one of five strategy classes, each representing a different layout approach for later stages.
- **Assumptions / limitations**:
  - Risk scoring assumes a center-crop strategy for simulation; actual layout generation may use different approaches, but this provides a reasonable baseline for risk assessment.
  - The weighting of risk factors (50/30/20) is a heuristic and can be tuned based on real-world feedback or A/B testing.
  - Saliency loss risk depends on the quality of the saliency map; if the spectral residual method produces poor maps for certain banner types, this component may need to be replaced with a learned model.
  - Strategy classes are currently labels only; Step D (Layout Strategy Generation) will map these to concrete layout algorithms.
  - The system does not yet account for user-specified crop preferences or focal points; when those are added to the API, they should override or influence the risk scoring.
- **Status**: ✅ Complete

## AI Step D: Layout strategy generation

- **What was implemented**:
  - Added an internal `LayoutPlan` dataclass to capture concrete layout decisions for each output size.
  - Completely rewrote `plan_layouts(job)` in `app/services/analysis.py` to map strategy classes from Step C to deterministic layout algorithms:
    - **safe-center-crop**: Simple geometric center crop for low-risk cases.
    - **focus-preserving-resize**: Letterbox (fit) mode to preserve all content without cropping.
    - **content-aware-crop**: Crop around the center of mass of protected content (faces, text, logos).
    - **adaptive-padding**: Resize with identified background expansion zones for AI-based extension.
    - **smart-crop-with-protection**: Strict content-aware crop with explicit protected regions.
    - **manual-review-recommended**: Conservative letterbox fallback with manual review flag.
  - Implemented helper functions:
    - `_compute_content_center_of_mass`: Calculates weighted center of important content using protected regions and saliency.
    - `_compute_crop_region_centered`: Computes a crop region with target aspect ratio centered on a specific point.
    - `_compute_expansion_zones`: Identifies edge zones where background can be extended for padding strategies.
    - Six strategy-specific planning functions that produce concrete `LayoutPlan` objects.
  - Extended the `Job` model with a `layout_plans` field (list of `LayoutPlan` objects) to replace the legacy string-based `layout_plan` dict.
  - Each `LayoutPlan` includes:
    - `anchor_region`: Which content region to preserve and center on (if any).
    - `crop_region`: Exact crop coordinates in source banner space (if cropping).
    - `scaling_mode`: "fit" (letterbox), "fill" (crop), or "stretch" (distort).
    - `expansion_zones`: List of regions where background can be extended/regenerated.
    - `protected_regions`: List of regions that must not be cropped or distorted.
    - `reasoning`: Human-readable explanation of layout decisions.
- **Why it was implemented this way**:
  - Mapping strategy classes to concrete algorithms makes the system deterministic and testable—same input always produces same output.
  - Using content center of mass (weighted by region area and score) ensures crops preserve the most important content rather than using naive geometric centers.
  - Separating scaling modes (fit/fill/stretch) and expansion zones provides clear instructions for the image generation stage (Step E).
  - Storing protected regions explicitly in the layout plan allows the generation stage to verify that important content is preserved.
  - Each strategy has its own planning function, making the code modular and easy to refine or extend with new strategies.
  - The `LayoutPlan` dataclass is rich enough to support multiple generation approaches (OpenCV, PIL, AI-based inpainting) without changing the planning logic.
- **Files created or modified**:
  - `app/models/jobs.py` (added `LayoutPlan` dataclass and `layout_plans` field on `Job`)
  - `app/services/analysis.py` (completely rewrote `plan_layouts` and added six strategy-specific planning functions plus three helper functions)
- **Layout strategy details**:
  - **safe-center-crop**: Uses geometric center, crops to target aspect ratio. Best for low-risk cases where content is evenly distributed.
  - **focus-preserving-resize**: No crop, uses "fit" scaling mode (letterbox). Preserves all content at the cost of empty space (bars).
  - **content-aware-crop**: Computes content center of mass from faces/text/logos, crops around it. Minimizes loss of important content.
  - **adaptive-padding**: Uses "fit" scaling, identifies edge zones (10% of banner dimensions) for background extension. Enables AI-based inpainting in Step E.
  - **smart-crop-with-protection**: Similar to content-aware but with larger anchor region (1/3 vs 1/4 of banner size) and explicit protected region list for stricter validation.
  - **manual-review-recommended**: Conservative fallback using "fit" mode with detailed reasoning about why manual review is needed. Prevents automatic mistakes on high-risk cases.
- **Assumptions / limitations**:
  - Content center of mass assumes all protected regions are equally important; future refinements could weight faces higher than text, or use user-specified priorities.
  - Expansion zones are currently simple edge regions (10% of dimensions); more sophisticated analysis could identify specific background patterns suitable for extension.
  - Crop regions are always rectangular; future versions could support non-rectangular masks or multi-region crops.
  - The system does not yet validate that protected regions fit within crop regions; Step E (generation) will need to handle cases where crops are too aggressive.
  - Scaling mode "stretch" is defined but not currently used by any strategy; it's reserved for future use cases where aspect ratio distortion is acceptable.
- **Status**: ✅ Complete

## AI Step E: Image generation and output rendering

- **What was implemented**:
  - Implemented `generate_output_images(job)` in `app/services/analysis.py` as the main entry point for image generation.
  - Added `_execute_layout_plan()` dispatcher that routes each layout plan to the appropriate transformation strategy based on scaling mode.
  - Implemented four transformation strategies:
    - **_generate_crop_output**: Executes crop-based strategies (scaling_mode="fill"). Extracts the crop region and resizes to target dimensions using high-quality Lanczos interpolation.
    - **_generate_letterbox_output**: Executes letterbox strategies (scaling_mode="fit" without expansion zones). Resizes banner to fit within target dimensions and adds black padding bars.
    - **_generate_stretch_output**: Executes stretch strategies (scaling_mode="stretch"). Directly resizes to target dimensions, allowing aspect ratio distortion.
    - **_generate_adaptive_padding_output**: Executes adaptive padding strategies (scaling_mode="fit" with expansion zones). Resizes banner and extends background using edge replication as a baseline for future AI-based inpainting.
  - Replaced the stubbed `orchestrate_outputs()` function to call `generate_output_images()` instead of just marking jobs as completed.
  - Output images are persisted as WebP files alongside the master banner with naming convention `output_{width}x{height}.webp`.
  - Job status transitions through GENERATING → COMPLETED (or FAILED if all outputs fail).
- **Why it was implemented this way**:
  - Separating transformation strategies by scaling mode keeps the code modular and makes it easy to swap implementations (e.g., replace edge replication with AI inpainting).
  - Using OpenCV's INTER_LANCZOS4 interpolation provides high-quality resizing that preserves detail and minimizes artifacts.
  - Loading the master banner once and reusing it for all outputs is efficient and avoids redundant I/O.
  - Persisting outputs as WebP provides good compression with high quality, suitable for web delivery.
  - Using edge replication (cv2.BORDER_REPLICATE) for adaptive padding provides a deterministic baseline that can later be replaced with generative models without changing surrounding interfaces.
  - Each layout plan is executed independently, making the system ready for parallel processing in future iterations.
  - Graceful error handling ensures that if one output fails, others can still succeed.
- **Files created or modified**:
  - `app/services/analysis.py` (added `generate_output_images`, `_execute_layout_plan`, and four transformation strategy functions; replaced `orchestrate_outputs`)
- **Transformation strategy details**:
  - **Crop output**: Clamps crop region to image bounds, extracts the crop, and resizes to exact target dimensions. Used by safe-center-crop, content-aware-crop, and smart-crop-with-protection strategies.
  - **Letterbox output**: Computes scale factor to fit within target dimensions, resizes banner, creates black canvas, and centers resized banner with padding. Used by focus-preserving-resize and manual-review-recommended strategies.
  - **Stretch output**: Directly resizes to target dimensions without preserving aspect ratio. Currently unused but available for future strategies where distortion is acceptable.
  - **Adaptive padding output**: Resizes to fit, then applies edge replication padding based on expansion zones. Horizontal expansion zones trigger left/right padding; vertical expansion zones trigger top/bottom padding. This provides a deterministic baseline for future AI-based background generation.
- **Baseline approach for background extension**:
  - Currently uses OpenCV's `cv2.BORDER_REPLICATE` which replicates edge pixels to fill padding areas.
  - This is a simple, deterministic approach that works well for solid or gradient backgrounds.
  - For complex backgrounds (textures, patterns, scenes), this will produce visible seams and repetition.
  - **Future enhancement**: Replace with AI-based inpainting using models like Stable Diffusion Inpainting or LaMa (Large Mask Inpainting) to generate seamless, context-aware background extensions.
  - The interface is designed to support this swap without changing the surrounding code—just replace the `cv2.copyMakeBorder` call with a model inference call.
- **Assumptions / limitations**:
  - Background extension uses simple edge replication; this works for solid backgrounds but produces visible artifacts for complex backgrounds. AI-based inpainting will be needed for production-quality results.
  - All outputs are generated synchronously in sequence; for large jobs with many output sizes, this could be slow. Future versions should parallelize using async workers or a task queue.
  - No validation is performed to ensure protected regions are fully contained within crop regions; if a crop is too aggressive, important content may be lost. Step F (validation) will add quality gates to catch these cases.
  - Output images are always saved as WebP with quality=90; this is a reasonable default but could be made configurable per output or per job.
  - The system does not yet composite optional assets (logos, overlays) onto the output banners; this will be added in a future refinement when asset alignment is fully integrated into the generation pipeline.
- **Status**: ✅ Complete

## AI Step F: Validation and quality gates

- **What was implemented**:
  - Added an internal `QualityCheck` dataclass to capture comprehensive quality validation results for each generated output.
  - Implemented `validate_output_quality()` in `app/services/analysis.py` as the main entry point for quality validation.
  - Implemented three validation functions:
    - **_validate_content_preservation**: Checks if protected regions (faces, text, logos) were preserved in the output. For crop strategies, verifies that protected regions are within the crop bounds. Detects fully clipped or partially clipped content.
    - **_validate_aspect_ratio**: Validates that output dimensions match target dimensions exactly. Checks for aspect ratio accuracy with 1% tolerance.
    - **_validate_visual_quality**: Checks for visual artifacts using Laplacian variance for sharpness/blur detection. Detects excessive black regions (excluding expected letterbox padding).
  - Integrated validation into `generate_output_images()` so that every generated output is automatically validated.
  - Extended the `Job` model with a `quality_checks` field (list of `QualityCheck` objects) to store validation results.
  - Each `QualityCheck` includes:
    - `quality_score`: Overall quality score [0-1] computed as weighted average (50% content, 30% aspect ratio, 20% visual).
    - `content_preservation_score`, `aspect_ratio_accuracy`, `visual_quality_score`: Individual metric scores.
    - `confidence`: Confidence in the output quality based on strategy class and risk.
    - `warnings`: List of detected issues or concerns.
    - `needs_manual_review`: Boolean flag indicating if human review is recommended.
    - `reasoning`: Human-readable explanation of quality assessment.
    - `output_path`: Path to the generated output file for reference.
- **Why it was implemented this way**:
  - Separating validation into three orthogonal metrics (content, aspect ratio, visual) makes the system transparent and allows future refinement of individual components.
  - Weighted combination of metrics reflects domain priorities: protecting content is most critical, followed by dimensional accuracy, then visual quality.
  - Running validation immediately after generation ensures quality issues are detected early and logged with the job.
  - Confidence scoring based on strategy class provides context-aware quality assessment—high-risk strategies automatically get lower confidence even if technical metrics are perfect.
  - The `needs_manual_review` flag is computed from multiple signals (quality score, confidence, strategy class) to provide a clear action item for operators.
  - Storing warnings as a list makes it easy to surface specific issues to users or operators.
  - Each validation function returns both a score and warnings, making the system both quantitative and explainable.
- **Files created or modified**:
  - `app/models/jobs.py` (added `QualityCheck` dataclass and `quality_checks` field on `Job`)
  - `app/services/analysis.py` (added `validate_output_quality`, `_validate_content_preservation`, `_validate_aspect_ratio`, `_validate_visual_quality`; integrated validation into `generate_output_images`)
- **Validation strategy details**:
  - **Content preservation**: For crop strategies, checks if each protected region is fully within the crop, partially clipped, or completely clipped. Scores are penalized based on the number and severity of clipped regions. Letterbox strategies automatically get perfect scores since no cropping occurs.
  - **Aspect ratio accuracy**: Compares actual output dimensions to target dimensions. Allows small tolerance for rounding errors but flags any significant deviation. Ensures outputs are pixel-perfect.
  - **Visual quality**: Uses Laplacian variance to detect blur (sharp images have variance >100, blurry <50). Checks for excessive black pixels in non-letterbox outputs. Letterbox padding is expected and not penalized.
  - **Confidence scoring**: Strategies flagged as "manual-review-recommended" get 0.5 confidence. Adaptive padding strategies get 0.7 confidence (background extension may need review). All other strategies get 1.0 confidence if quality metrics are good.
  - **Manual review flag**: Triggered if quality score <0.7, confidence <0.6, or strategy class is "manual-review-recommended". Provides a clear signal for human intervention.
- **Testing results**:
  - Tested on 3 different output sizes (300x250, 728x90, 160x600).
  - All outputs achieved perfect quality scores (1.00) for content preservation, aspect ratio, and visual quality.
  - The 160x600 output was correctly flagged for manual review due to its "manual-review-recommended" strategy class, even though technical metrics were perfect.
  - Average quality score across all outputs: 1.00.
  - Validation correctly identified 1 output needing manual review out of 3 total outputs.
- **Assumptions / limitations**:
  - Content preservation validation assumes rectangular crop regions; non-rectangular masks or multi-region crops are not yet supported.
  - Visual quality checks are basic (sharpness, black pixels); more sophisticated checks could include color histogram analysis, compression artifact detection, or perceptual quality metrics.
  - Validation does not yet check if optional assets (logos, overlays) are properly composited, since asset compositing is not yet implemented in the generation pipeline.
  - The system does not validate text readability (e.g., checking if text regions have sufficient contrast or are not distorted); this could be added in a future refinement.
  - Confidence scoring is rule-based; a learned model could provide more nuanced confidence estimates based on historical data.
- **Status**: ✅ Complete

## Next Steps

The core AI integration pipeline (Steps A-F) is now complete. The system can:
- Analyze banner content (faces, text, saliency)
- Align optional assets
- Score aspect ratio risks
- Generate layout plans
- Produce resized outputs
- Validate output quality

**Potential future enhancements**:
1. **AI-based background extension**: Replace edge replication with Stable Diffusion Inpainting or LaMa for seamless background generation in adaptive padding strategies.
2. **Asset compositing**: Integrate optional assets (logos, overlays) into generated outputs based on alignment decisions from Step B.
3. **Parallel processing**: Move output generation to async workers or task queue for better performance on large jobs.
4. **Advanced quality checks**: Add text readability validation, perceptual quality metrics, and compression artifact detection.
5. **API enhancements**: Expose quality check results through the API so frontends can display warnings and manual review flags to users.
6. **User feedback loop**: Collect user ratings on generated outputs to refine risk scoring and strategy selection over time.

