"""initial

Revision ID: 001
Revises:
Create Date: 2024-03-19 10:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create users table
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(length=50), nullable=False),
        sa.Column("email", sa.String(length=100), nullable=False),
        sa.Column("hashed_password", sa.String(length=100), nullable=False),
        sa.Column("full_name", sa.String(length=100), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column("is_superuser", sa.Boolean(), nullable=False, default=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)

    # Create avatars table
    op.create_table(
        "avatars",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=True),
        sa.Column(
            "personality",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=False,
            default={},
        ),
        sa.Column(
            "appearance",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=False,
            default={},
        ),
        sa.Column(
            "voice", postgresql.JSON(astext_type=sa.Text()), nullable=False, default={}
        ),
        sa.Column(
            "current_emotion", postgresql.JSON(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "current_cognitive_state",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "current_physical_state",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_avatars_owner_id"), "avatars", ["owner_id"], unique=False)


def downgrade():
    op.drop_index(op.f("ix_avatars_owner_id"), table_name="avatars")
    op.drop_table("avatars")
    op.drop_index(op.f("ix_users_username"), table_name="users")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
