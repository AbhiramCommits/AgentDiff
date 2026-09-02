from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..db import get_session
from ..gate import CONFIDENCE_DOWNGRADE_FACTOR, GateExecutionError
from ..models import Finding, GateResult, ReviewRun, RunStatus
from ..reviewer import LLMError, PROMPT_VERSION
from ..schemas import ReviewCreateRequest, ReviewRunRead

router = APIRouter(prefix="/api/v1")


def _now() -> datetime:
    return datetime.now(UTC)


def _run_query():
    return (
        select(ReviewRun)
        .options(selectinload(ReviewRun.findings).selectinload(Finding.gate_result))
        .execution_options(populate_existing=True)
    )


async def _load_run(session: AsyncSession, run_id: UUID) -> ReviewRun:
    stmt = _run_query().where(ReviewRun.id == run_id)
    run = (await session.execute(stmt)).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail={"error": "review run not found"})
    return run


async def _apply_gate_results(
    session: AsyncSession, request: Request, run: ReviewRun
) -> None:
    decisions = await request.app.state.gate.gate_all(list(run.findings), run.head_sha)
    for finding, decision in decisions:
        if finding.gate_result is not None:
            continue
        session.add(
            GateResult(
                finding_id=finding.id,
                decision=decision.decision,
                reason=decision.reason,
                tests_passed=decision.tests_passed,
                tests_failed=decision.tests_failed,
                coverage_before=decision.coverage_before,
                coverage_after=decision.coverage_after,
                duration_ms=decision.duration_ms,
                verification=decision.verification,
            )
        )
        if decision.verification == "unverified":
            finding.confidence = round(
                finding.confidence * CONFIDENCE_DOWNGRADE_FACTOR, 2
            )
    await session.commit()


@router.post("/reviews", status_code=201, response_model=ReviewRunRead)
async def create_review(
    payload: ReviewCreateRequest,
    request: Request,
    gate: bool = Query(default=False),
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

    run = await _load_run(session, run.id)
    if gate:
        try:
            await _apply_gate_results(session, request, run)
        except GateExecutionError as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc)}) from exc
        run = await _load_run(session, run.id)
    return run


@router.get("/reviews", response_model=list[ReviewRunRead])
async def list_reviews(
    repo: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
) -> list[ReviewRun]:
    stmt = (
        _run_query()
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


@router.post("/reviews/{run_id}/gate", response_model=ReviewRunRead)
async def gate_review(
    run_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> ReviewRun:
    run = await _load_run(session, run_id)
    if run.status != RunStatus.COMPLETE:
        raise HTTPException(
            status_code=409,
            detail={"error": f"cannot gate run with status '{run.status.value}'"},
        )
    try:
        await _apply_gate_results(session, request, run)
    except GateExecutionError as exc:
        raise HTTPException(status_code=502, detail={"error": str(exc)}) from exc
    return await _load_run(session, run_id)
