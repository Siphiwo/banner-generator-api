from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class JobStatus(str, Enum):
    """High-level lifecycle states for a banner job."""

    PENDING = "pending"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class OutputSize(BaseModel):
    """Desired output size for a generated banner."""

    width: PositiveInt = Field(..., description="Target width in pixels.")
    height: PositiveInt = Field(..., description="Target height in pixels.")
    label: str | None = Field(
        default=None,
        description="Optional human-readable label, e.g. 'Facebook cover', 'Sidebar'.",
    )


class JobCreateResponse(BaseModel):
    """Response returned when a new job is created."""

    job_id: str = Field(..., description="Server-generated unique job identifier.")
    status: JobStatus = Field(
        default=JobStatus.PENDING,
        description="Initial status of the job (always 'pending' on creation).",
    )
    outputs: List[OutputSize] = Field(
        default_factory=list,
        description="List of requested output sizes for this job.",
    )


class JobSummary(BaseModel):
    """Lightweight view of a job suitable for listings."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Unique job identifier.")
    status: JobStatus = Field(..., description="Current lifecycle status for the job.")


class JobDetail(BaseModel):
    """Detailed view of a single job."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Unique job identifier.")
    status: JobStatus = Field(..., description="Current lifecycle status for the job.")
    outputs: List[OutputSize] = Field(
        default_factory=list,
        description="Requested output sizes for this job.",
    )
    created_at: str = Field(
        ...,
        description="Job creation timestamp in ISO 8601 format (UTC).",
    )
    updated_at: str = Field(
        ...,
        description="Last modification timestamp in ISO 8601 format (UTC).",
    )


class OutputPlan(BaseModel):
    """Planned output for a specific target size."""

    width: PositiveInt = Field(..., description="Target width in pixels.")
    height: PositiveInt = Field(..., description="Target height in pixels.")
    label: str | None = Field(
        default=None,
        description="Optional label carried from the requested output.",
    )
    bucket: str = Field(
        ...,
        description="Aspect ratio bucket assigned during analysis (similar/moderate/extreme).",
    )
    layout_strategy: str = Field(
        ...,
        description="Planned layout strategy for this bucket.",
    )


class JobOutputsResponse(BaseModel):
    """Metadata about all requested/Planned outputs for a job."""

    job_id: str = Field(..., description="Job identifier.")
    status: JobStatus = Field(..., description="Current job status.")
    outputs: List[OutputPlan] = Field(
        default_factory=list,
        description="Planned outputs with layout strategy hints.",
    )

