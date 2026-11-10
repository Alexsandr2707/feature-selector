"""HTTP request contracts shared without importing service applications."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelInput(BaseModel):
    """Published dataset and restricted template overrides."""

    name: str = Field(min_length=1, max_length=200)
    dataset_id: str
    template_id: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class PredictionInput(BaseModel):
    """Explicit temporal split to apply and evaluate."""

    split: Literal["train", "validation", "test"] = "test"


class RenameInput(BaseModel):
    """Display name update without mutation of object content."""

    name: str = Field(min_length=1, max_length=200)


class ObjectInput(RenameInput):
    """Draft content and optional private capability for a job reservation."""

    id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    references: dict[str, str] = Field(default_factory=dict)
    job_id: str | None = None
    reservation_token: str | None = Field(default=None, min_length=32)


class DraftUpdateInput(RenameInput):
    """Final name and metadata for a reserved, unpublished result."""

    metadata: dict[str, Any] = Field(default_factory=dict)
