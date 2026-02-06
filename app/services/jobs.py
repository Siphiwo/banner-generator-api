from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from fastapi import UploadFile

from app.api.v1.schemas import JobStatus, OutputSize
from app.models.jobs import Job
from app.services.analysis import run_initial_pipeline


class JobStorageError(RuntimeError):
    """Raised when a storage operation fails in a non-recoverable way."""


class JobStore:
    """
    Simple in-memory job store with filesystem-backed asset storage.

    This is a minimal abstraction that can later be replaced by a database
    and object storage without changing the API surface.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._jobs: Dict[str, Job] = {}
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    async def create_job(
        self,
        job_id: str,
        outputs: List[OutputSize],
        master_banner: UploadFile,
        additional_assets: List[UploadFile] | None = None,
    ) -> Job:
        """
        Persist a new job and its associated files.

        Files are written to disk under `<base_dir>/<job_id>/`.
        """
        job_dir = self._base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            master_ext = os.path.splitext(master_banner.filename or "")[1] or ".bin"
            master_path = job_dir / f"master{master_ext}"
            contents = await master_banner.read()
            master_path.write_bytes(contents)

            additional_paths: List[str] = []
            if additional_assets:
                for index, asset in enumerate(additional_assets):
                    asset_ext = os.path.splitext(asset.filename or "")[1] or ".bin"
                    asset_path = job_dir / f"asset_{index}{asset_ext}"
                    asset_contents = await asset.read()
                    asset_path.write_bytes(asset_contents)
                    additional_paths.append(str(asset_path))
        except OSError as exc:  # noqa: PERF203
            raise JobStorageError("Failed to persist job files to disk.") from exc

        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            outputs=outputs,
            master_banner_path=str(master_path),
            additional_asset_paths=additional_paths,
        )
        # Run the initial (stubbed) analysis and planning pipeline once per job.
        run_initial_pipeline(job)
        self._jobs[job.id] = job
        return job

    async def get_job(self, job_id: str) -> Job | None:
        """Retrieve a job by its identifier, if it exists."""
        return self._jobs.get(job_id)

    async def list_jobs(self) -> List[Job]:
        """Return all known jobs. Intended for debugging and admin tooling."""
        return list(self._jobs.values())


_default_store = JobStore(base_dir=Path(os.getenv("BANNER_STORAGE_DIR", "storage/jobs")))


def get_job_store() -> JobStore:
    """
    Return the process-wide job store instance.

    Abstracted behind a function to make it easy to later swap out the
    implementation or inject different stores in tests.
    """
    return _default_store

