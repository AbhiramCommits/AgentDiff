"""initial review schema

Revision ID: 0001
Revises:
Create Date: 2026-09-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "review_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("repo", sa.String(length=255), nullable=False),
        sa.Column("base_sha", sa.String(length=64), nullable=False),
        sa.Column("head_sha", sa.String(length=64), nullable=False),
        sa.Column("model_id", sa.String(length=255), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), server_default=sa.text("'1'"), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "running",
                "complete",
                "failed",
                name="runstatus",
                native_enum=False,
                length=32,
                create_constraint=False,
            ),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("cache_read_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Numeric(precision=10, scale=6), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_review_runs_status", "review_runs", ["status"])

    op.create_table(
        "findings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("file_path", sa.String(length=512), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=True),
        sa.Column("end_line", sa.Integer(), nullable=True),
        sa.Column(
            "severity",
            sa.Enum(
                "blocker",
                "major",
                "minor",
                "nit",
                name="severity",
                native_enum=False,
                length=16,
                create_constraint=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "category",
            sa.Enum(
                "correctness",
                "security",
                "performance",
                "maintainability",
                "test-gap",
                name="category",
                native_enum=False,
                length=32,
                create_constraint=False,
            ),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("suggested_patch", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["review_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_findings_run_id", "findings", ["run_id"])

    op.create_table(
        "gate_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("finding_id", sa.Uuid(), nullable=False),
        sa.Column(
            "decision",
            sa.Enum(
                "accepted",
                "rejected",
                name="decision",
                native_enum=False,
                length=16,
                create_constraint=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "reason",
            sa.Enum(
                "tests_passed",
                "tests_failed",
                "coverage_dropped",
                "patch_did_not_apply",
                "no_patch",
                name="gatereason",
                native_enum=False,
                length=32,
                create_constraint=False,
            ),
            nullable=False,
        ),
        sa.Column("tests_passed", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("tests_failed", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("coverage_before", sa.Float(), nullable=True),
        sa.Column("coverage_after", sa.Float(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["finding_id"], ["findings.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("finding_id"),
    )


def downgrade() -> None:
    op.drop_table("gate_results")
    op.drop_table("findings")
    op.drop_table("review_runs")
