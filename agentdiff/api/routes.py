from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..db import get_session
from ..models import Finding, ReviewRun, RunStatus
from ..reviewer import LLMError, PROMPT_VERSION
from ..schemas import ReviewCreateRequest, ReviewRunRead

router = APIRouter(prefix="/api/v1")


def _now() -> datetime:
    return datetime.now(UTC)


async def _load_run(session: AsyncSession, run_id: UUID) -> ReviewRun:
    stmt = (
        select(ReviewRun)
        .options(selectinload(ReviewRun.findings))
        .where(ReviewRun.id == run_id)
        .execution_options(populate_existing=True)
    )
    run = (await session.execute(stmt)).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail={"error": "review run not found"})
    return run


@router.post("/reviews", status_code=201, response_model=ReviewRunRead)
async def create_review(
    payload: ReviewCreateRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> ReviewRun:
    run = ReviewRun(
        repo=payload.repo,
        base_sha=payload.base_sha,
        head_sha=payload.head_sha,
        model_id=request.app.state.settings.model_id,
        prompt_version=PROMPT_VERSION,
        status=RunStatus.RUNNING,
    )
    session.add(run)
    await session.commit()

    try:
        outcome = await request.app.state.reviewer.review(
            diff=payload.diff, repo_context=payload.repo_context
        )
    except LLMError as exc:
        run.status = RunStatus.FAILED
        run.completed_at = _now()
        await session.commit()
        raise HTTPException(status_code=exc.status_code, detail={"error": str(exc)}) from exc
    except Exception as exc:
        run.status = RunStatus.FAILED
        run.completed_at = _now()
        await session.commit()
        raise HTTPException(status_code=500, detail={"error": "internal error"}) from exc

    for item in outcome.result.findings:
        session.add(
            Finding(
                run_id=run.id,
                file_path=item.file_path,
                start_line=item.start_line,
                end_line=item.end_line,
                severity=item.severity,
                category=item.category,
                title=item.title,
                rationale=item.rationale,
                suggested_patch=item.suggested_patch,
                confidence=item.confidence,
            )
        )

    run.status = RunStatus.COMPLETE
    run.completed_at = _now()
    run.latency_ms = outcome.latency_ms
    run.input_tokens = outcome.input_tokens
    run.output_tokens = outcome.output_tokens
    run.cache_read_tokens = outcome.cache_read_tokens
    run.cost_usd = outcome.cost_usd
    await session.commit()

    return await _load_run(session, run.id)


@router.get("/reviews", response_model=list[ReviewRunRead])
async def list_reviews(
    repo: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
) -> list[ReviewRun]:
    stmt = (
        select(ReviewRun)
        .options(selectinload(ReviewRun.findings))
        .order_by(ReviewRun.created_at.desc())
        .limit(limit)
    )
    if repo is not None:
        stmt = stmt.where(ReviewRun.repo == repo)
    return list((await session.execute(stmt)).scalars().all())


@router.get("/reviews/{run_id}", response_model=ReviewRunRead)
async def get_review(
    run_id: UUID, session: AsyncSession = Depends(get_session)
) -> ReviewRun:
    return await _load_run(session, run_id)
