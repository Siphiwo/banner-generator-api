import json
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.api.v1.schemas import (
    JobCreateResponse,
    JobDetail,
    JobOutputsResponse,
    JobStatus,
    JobSummary,
    OutputPlan,
    OutputSize,
)
from app.services.jobs import JobStorageError, get_job_store

router = APIRouter(prefix="/api/v1")


@router.get("/health", tags=["health"])
async def health_check() -> dict:
    """API v1 health check endpoint."""
    return {"status": "ok", "api_version": "v1"}


@router.post(
    "/jobs",
    response_model=JobCreateResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["jobs"],
    summary="Create a new banner resize job",
)
async def create_job(
    master_banner: UploadFile = File(..., description="Primary banner image (PNG, JPG, or PSD)."),
    additional_assets: list[UploadFile] | None = File(
        default=None,
        description="Optional additional assets such as logos or product images.",
    ),
    outputs: str = Form(
        ...,
        description=(
            "JSON-encoded list of output sizes, e.g. "
            "[{\"width\":1200,\"height\":628,\"label\":\"facebook\"}]."
        ),
    ),
) -> JobCreateResponse:
    """
    Create a new banner resizing job.

    The client must send a multipart/form-data request containing:
    - `master_banner`: required image file (PNG, JPG, or PSD).
      - For PSD files: Layer naming conventions are used to identify semantic
        regions (text, logos, backgrounds). See docs/PSD_INTEGRATION.md.
    - `additional_assets`: optional list of extra files.
    - `outputs`: JSON-encoded list of desired output sizes.

    PSD files are treated as source-of-truth layout documents. The parser
    extracts semantic regions based on layer names:
    - Background layers: `bg:`, `background:`, or group named `BG`/`BACKGROUND`
    - Text layers: `text:`, `copy:`, or native Photoshop text layers
    - Logo layers: `logo:`
    - Product layers: `product:`, `hero:`
    - Protected layers: `[protect]` or `[lock]` modifiers
    
    For standard image files (PNG/JPG), computer vision is used for analysis.
    """
    try:
        raw_outputs = json.loads(outputs)
        parsed_outputs = [OutputSize.model_validate(item) for item in raw_outputs]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid `outputs` payload. Expected JSON list of OutputSize objects.",
        ) from exc

    job_id = str(uuid4())
    store = get_job_store()

    try:
        job = await store.create_job(
            job_id=job_id,
            outputs=parsed_outputs,
            master_banner=master_banner,
            additional_assets=additional_assets or [],
        )
    except JobStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist job and associated assets.",
        ) from exc

    return JobCreateResponse(
        job_id=job.id,
        status=job.status,
        outputs=job.outputs,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobDetail,
    tags=["jobs"],
    summary="Get details for a specific job",
)
async def get_job(job_id: str) -> JobDetail:
    """
    Retrieve metadata and requested outputs for a single job.

    This does not expose internal file paths; it is designed for frontend
    consumption.
    """
    store = get_job_store()
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )

    return JobDetail(
        id=job.id,
        status=job.status,
        outputs=job.outputs,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
    )


@router.get(
    "/jobs",
    response_model=list[JobSummary],
    tags=["jobs"],
    summary="List jobs (development use)",
)
async def list_jobs() -> list[JobSummary]:
    """
    List all known jobs.

    Intended primarily for development and debugging; in a real multi-tenant
    system, this would likely be scoped or protected.
    """
    store = get_job_store()
    jobs = await store.list_jobs()
    return [JobSummary.model_validate(job) for job in jobs]


@router.get(
    "/jobs/{job_id}/outputs",
    response_model=JobOutputsResponse,
    tags=["jobs"],
    summary="Get planned outputs for a job",
)
async def get_job_outputs(job_id: str) -> JobOutputsResponse:
    """
    Return planned outputs and layout strategies for a job.

    This endpoint exposes metadata only; it does not serve any generated
    images. It is intended for frontends to understand what outputs will be
    (or have been) produced for a given job.
    """
    store = get_job_store()
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )

    # Build a mapping from OutputSize object id to its bucket for quick lookup.
    size_to_bucket: dict[int, str] = {}
    for bucket, sizes in job.aspect_ratio_buckets.items():
        for size in sizes:
            size_to_bucket[id(size)] = bucket

    plans: list[OutputPlan] = []
    for size in job.outputs:
        bucket = size_to_bucket.get(id(size), "unknown")
        strategy = job.layout_plan.get(bucket, "unknown")
        plans.append(
            OutputPlan(
                width=size.width,
                height=size.height,
                label=size.label,
                bucket=bucket,
                layout_strategy=strategy,
            )
        )

    return JobOutputsResponse(job_id=job.id, status=job.status, outputs=plans)

