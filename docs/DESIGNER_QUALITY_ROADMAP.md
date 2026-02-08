# Designer-Quality Banner Resizing: Implementation Roadmap

## Role & Context

You are a **Senior ML Systems Engineer** responsible for evolving the banner resizing system from **technically functional** to **designer-acceptable**.

The system currently has:
- ✅ Complete content analysis pipeline (faces, text, saliency)
- ✅ Deterministic layout planning with 6 strategy classes
- ✅ High-quality image resizing (Lanczos interpolation)
- ✅ Comprehensive quality validation
- ❌ Primitive background extension (edge replication only)
- ❌ No asset compositing
- ❌ No text readability validation
- ❌ Black letterbox bars for extreme aspect ratios

Your mission: **Add AI-powered generation and asset rendering without breaking existing interfaces or functionality.**

---

## Non-Negotiable Constraints

1. **Preserve all existing APIs and data structures** — No breaking changes to routes, schemas, or internal models
2. **Maintain determinism where possible** — Same input must produce same output (except where AI models introduce variance)
3. **Keep the system debuggable** — Every decision must be explainable and logged
4. **Update `completion-tracker.md` after every step** — This is the source of truth
5. **No silent failures** — All AI operations must have graceful degradation and clear error states
6. **Backward compatibility** — Old jobs must still work; new features are opt-in or automatic improvements

---

## Implementation Order (Mandatory)

You **must** implement in this exact order. Do not skip or reorder steps.

### Phase 1: AI-Powered Background Extension (Steps G & H)

#### Step G: Integrate AI Inpainting Model

**Objective**: Replace `cv2.BORDER_REPLICATE` with a real generative model for seamless background extension.

**What to implement**:
- Add a new `_generate_inpainted_background()` function in `app/services/analysis.py`
- Choose and integrate an inpainting model:
  - **Recommended**: LaMa (Large Mask Inpainting) — fast, deterministic, open-source
  - **Alternative**: Stable Diffusion Inpainting — higher quality but slower
  - **Fallback**: Keep edge replication as a safety net if model fails
- The function must:
  - Accept: resized banner, expansion zones (mask), target dimensions
  - Return: seamlessly extended banner with generated background
  - Handle failures gracefully (fall back to edge replication with warning)
  - Log confidence scores and generation time
- Update `_generate_adaptive_padding_output()` to use inpainting instead of edge replication
- Add model loading/caching to avoid reloading on every job

**Why this order**:
- Background extension is the most visible quality issue
- It's independent of other features (can be tested in isolation)
- It unblocks the letterbox → adaptive padding transition

**Files to modify**:
- `pyproject.toml` (add inpainting model dependencies)
- `app/services/analysis.py` (add inpainting logic)
- `app/models/jobs.py` (optional: add inpainting metadata to `LayoutPlan`)

**Testing**:
- Generate outputs for 3+ different aspect ratios
- Verify backgrounds are seamless (no visible seams or repetition)
- Verify protected regions are not altered
- Verify fallback to edge replication if model fails

**Status**: ⏭️ Next

---

#### Step H: Eliminate Letterbox Strategy (Replace with Adaptive Padding)

**Objective**: Remove black bars by always using adaptive padding + inpainting for extreme aspect ratios.

**What to implement**:
- Modify `_plan_manual_review_recommended()` to use adaptive padding instead of letterbox
- Update risk scoring to never recommend letterbox (always use adaptive padding or crop)
- Modify `plan_layouts()` to route all high-risk cases to adaptive padding
- Update validation to expect no black bars (except for intentional design choices)
- Add a configuration flag to allow users to opt-in to letterbox if they want it (for backward compatibility)

**Why this order**:
- Depends on Step G (inpainting must work first)
- Eliminates the most obvious "non-designer" output (black bars)
- Improves visual quality across the board

**Files to modify**:
- `app/services/analysis.py` (modify strategy planning functions)
- `app/models/jobs.py` (optional: add letterbox preference flag)

**Testing**:
- Generate outputs for extreme aspect ratios (1:1, 16:9 → 160:600, etc.)
- Verify no black bars appear
- Verify backgrounds are extended intelligently
- Verify protected content is preserved

**Status**: ⏭️ Next

---

### Phase 2: Asset Compositing (Steps I & J)

#### Step I: Implement Asset Rendering Pipeline

**Objective**: Composite optional assets (logos, overlays) onto generated outputs.

**What to implement**:
- Add a new `_composite_assets_onto_output()` function in `app/services/analysis.py`
- The function must:
  - Accept: generated output image, list of `AssetAlignment` objects, master banner
  - Load each asset from disk
  - Scale asset to fit the target output dimensions (proportional scaling)
  - Position asset based on alignment decision (corner, edge, or explicit region)
  - Handle transparency/alpha blending if asset has alpha channel
  - Return: composited output with assets rendered
  - Handle failures gracefully (skip failed assets, log warnings)
- Integrate into `generate_output_images()` so assets are composited after generation
- Add metadata to `QualityCheck` to track which assets were successfully composited

**Why this order**:
- Depends on Step G (background extension must work first)
- Requires asset alignment from Step B (already implemented)
- Enables logos/overlays to appear in final outputs

**Files to modify**:
- `app/services/analysis.py` (add asset compositing logic)
- `app/models/jobs.py` (optional: add compositing metadata to `QualityCheck`)

**Testing**:
- Generate outputs with 1+ optional assets
- Verify assets are scaled and positioned correctly
- Verify assets don't overlap protected content (faces/text)
- Verify transparency is preserved
- Verify fallback if asset file is missing

**Status**: ⏭️ Next

---

#### Step J: Validate Asset Compositing Quality

**Objective**: Add quality checks to ensure composited assets look professional.

**What to implement**:
- Add a new `_validate_asset_compositing()` function in `app/services/analysis.py`
- The function must:
  - Check if assets are visible (not completely outside bounds)
  - Check if assets overlap protected content (faces/text)
  - Check if asset scaling is reasonable (not too small or too large)
  - Check if asset positioning is balanced (not all on one side)
  - Return: warnings and confidence score for asset compositing
- Integrate into `validate_output_quality()` so asset validation runs automatically
- Update `QualityCheck` to include asset compositing warnings

**Why this order**:
- Depends on Step I (asset rendering must work first)
- Ensures composited assets meet quality standards
- Provides feedback to users about asset placement

**Files to modify**:
- `app/services/analysis.py` (add asset validation logic)
- `app/models/jobs.py` (optional: extend `QualityCheck` with asset warnings)

**Testing**:
- Generate outputs with assets in various positions
- Verify warnings are triggered for problematic placements
- Verify quality scores reflect asset quality
- Verify manual review flag is set when needed

**Status**: ⏭️ Next

---

### Phase 3: Text Readability & Advanced Validation (Steps K & L)

#### Step K: Implement Text Readability Validation

**Objective**: Ensure text remains legible after resizing.

**What to implement**:
- Add a new `_validate_text_readability()` function in `app/services/analysis.py`
- The function must:
  - Load detected text regions from `banner_analysis.text_regions`
  - For each text region, estimate final text size in the output
  - Check if text size is above minimum threshold (e.g., 8px for web)
  - Check if text contrast is sufficient (using saliency map or color analysis)
  - Check if text is distorted (aspect ratio change > 20%)
  - Return: warnings and readability score
- Integrate into `validate_output_quality()` so text validation runs automatically
- Update `QualityCheck` to include text readability warnings

**Why this order**:
- Depends on Steps G & H (background extension must work first)
- Depends on Step I (asset compositing must work first)
- Ensures text quality across all outputs
- Improves user confidence in the system

**Files to modify**:
- `app/services/analysis.py` (add text readability validation)
- `app/models/jobs.py` (optional: extend `QualityCheck` with text warnings)

**Testing**:
- Generate outputs with text regions
- Verify warnings for small text
- Verify warnings for low contrast
- Verify warnings for distorted text
- Verify quality scores reflect text readability

**Status**: ⏭️ Next

---

#### Step L: Implement Perceptual Quality Metrics

**Objective**: Add advanced quality checks for visual artifacts and anomalies.

**What to implement**:
- Add a new `_validate_perceptual_quality()` function in `app/services/analysis.py`
- The function must:
  - Detect compression artifacts (using frequency analysis or ML model)
  - Detect color shifts or banding (histogram analysis)
  - Detect blur or over-sharpening (Laplacian variance, already partially done)
  - Detect seams in inpainted regions (edge detection on expansion zones)
  - Return: warnings and perceptual quality score
- Integrate into `validate_output_quality()` so perceptual validation runs automatically
- Update `QualityCheck` to include perceptual quality warnings

**Why this order**:
- Depends on Steps G & H (inpainting must work first)
- Depends on Step K (text validation must work first)
- Catches subtle quality issues that users would notice
- Provides comprehensive quality assessment

**Files to modify**:
- `app/services/analysis.py` (add perceptual quality validation)
- `app/models/jobs.py` (optional: extend `QualityCheck` with perceptual warnings)

**Testing**:
- Generate outputs with various background types (solid, gradient, complex)
- Verify seam detection works for inpainted regions
- Verify artifact detection catches compression issues
- Verify quality scores reflect perceptual quality

**Status**: ⏭️ Next

---

### Phase 4: API Enhancements (Steps M & N)

#### Step M: Expose Quality Metadata Through API

**Objective**: Allow frontends to display quality warnings and manual review flags to users.

**What to implement**:
- Add new response schemas in `app/api/v1/schemas.py`:
  - `QualityCheckResponse` (expose quality scores, warnings, manual review flag)
  - `OutputMetadataResponse` (include quality check, asset info, strategy used)
- Add new API endpoint: `GET /api/v1/jobs/{job_id}/outputs/{output_id}/quality`
- Modify `GET /api/v1/jobs/{job_id}/outputs` to include quality metadata
- Ensure sensitive internal data (file paths, mask paths) is not exposed

**Why this order**:
- Depends on all previous steps (quality data must exist first)
- Allows frontends to inform users about output quality
- Enables users to make decisions about manual review

**Files to modify**:
- `app/api/v1/schemas.py` (add response schemas)
- `app/api/v1/routes.py` (add quality metadata endpoints)

**Testing**:
- Call quality endpoint and verify response structure
- Verify sensitive data is not exposed
- Verify quality warnings are included
- Verify manual review flags are set correctly

**Status**: ⏭️ Next

---

#### Step N: Add Configuration for Designer Preferences

**Objective**: Allow designers to customize behavior (e.g., prefer crop vs. padding, asset positioning).

**What to implement**:
- Add new request schema in `app/api/v1/schemas.py`:
  - `DesignerPreferences` (crop_preference, padding_preference, asset_positioning, etc.)
- Modify `POST /api/v1/jobs` to accept optional `preferences` field
- Update `Job` model to store preferences
- Modify risk scoring and layout planning to respect preferences
- Update `completion-tracker.md` to document preference options

**Why this order**:
- Depends on all previous steps (all strategies must be implemented first)
- Allows designers to customize behavior without code changes
- Enables A/B testing and user feedback collection

**Files to modify**:
- `app/api/v1/schemas.py` (add preference schema)
- `app/api/v1/routes.py` (accept preferences in job creation)
- `app/models/jobs.py` (add preferences field to `Job`)
- `app/services/analysis.py` (respect preferences in planning)

**Testing**:
- Create jobs with different preferences
- Verify layout planning respects preferences
- Verify outputs match preference settings

**Status**: ⏭️ Next

---

## Implementation Guidelines

### For Each Step

1. **Before coding**: Update `completion-tracker.md` with step title and objectives
2. **During coding**: 
   - Write clear, commented code explaining WHY not WHAT
   - Add logging at key decision points
   - Handle failures gracefully with fallbacks
   - Preserve all existing interfaces
3. **After coding**:
   - Run diagnostics to check for syntax/type errors
   - Test with real banner images (use existing test data in `storage/jobs/`)
   - Update `completion-tracker.md` with results and next step
   - Verify no existing functionality is broken

### Model Selection Rules

- **Inpainting**: Prefer LaMa (fast, deterministic) over Stable Diffusion (slower, more variance)
- **Fallback**: Always have a fallback strategy (edge replication for inpainting, letterbox for adaptive padding)
- **Caching**: Load models once and reuse across jobs
- **Logging**: Log model inference time and confidence scores

### Quality & Reliability

- **No silent failures**: Every operation must either succeed or fail loudly with a clear error message
- **Graceful degradation**: If a feature fails, fall back to a simpler approach and log a warning
- **Determinism**: Same input must produce same output (except where randomness is intentional)
- **Explainability**: Every decision must be logged and traceable

### Testing Strategy

- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test full pipeline with real banner images
- **Regression tests**: Verify existing functionality still works
- **Visual inspection**: Manually review generated outputs for quality

---

## Success Criteria

At the end of this roadmap, the system must:

✅ Generate seamless backgrounds without visible seams or repetition
✅ Composite optional assets onto outputs with proper scaling and positioning
✅ Validate text readability and warn if text becomes too small
✅ Detect and warn about visual artifacts and quality issues
✅ Expose quality metadata through the API
✅ Allow designers to customize behavior through preferences
✅ Maintain backward compatibility with existing jobs and APIs
✅ Handle all failures gracefully with clear error messages
✅ Remain deterministic and debuggable

---

## Completion Tracking

After each step, update `completion-tracker.md` with:
- Step number and title
- What was implemented
- Which models/libraries were integrated
- Files created or modified
- Assumptions and limitations
- Testing results
- Status (✅ Complete / ⏭️ Next)

If the tracker is not updated, the work is invalid.

---

## Start Condition

Before starting:
1. Read this entire document
2. Read `completion-tracker.md` to understand current state
3. Review `app/services/analysis.py` to understand existing code
4. Begin with **Step G: Integrate AI Inpainting Model**

Do not skip steps.
Do not rush.
Build this as if it will be maintained for years.

