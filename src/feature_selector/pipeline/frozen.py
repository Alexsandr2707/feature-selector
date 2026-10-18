"""Reuse a complete fitted transform prefix without updating any parameters."""

import pickle
from pathlib import Path
from typing import Self

from ..tracking import StepContext
from .base import Step
from .core import TransformArtifact
from .data import PipelineData, PredictionData


class FrozenPipeline(Step):
    """Load an exported nested pipeline once and perform transformations only."""

    artifact_: TransformArtifact

    def __init__(self, path: str | Path) -> None:
        """Store the trusted artifact path without opening it.

        Parameters:
            path (str | Path): File produced by ConfiguredPipeline.export_transform.

        Returns:
            None.
        """
        self.path = path

    def fit(
        self,
        train: PipelineData,
        validation: PipelineData | None = None,
        *,
        context: StepContext | None = None,
    ) -> Self:
        """Load embedded state without fitting or invoking the nested fit methods.

        Parameters:
            train (PipelineData): Input schema to verify against the saved pipeline.
            validation (PipelineData | None): Unused validation examples.
            context (StepContext | None): Records provenance rather than old training events.

        Returns:
            Self: Ready frozen transformer with embedded state.
        """
        with Path(self.path).open("rb") as stream:
            artifact = pickle.load(stream)
        if not isinstance(artifact, TransformArtifact) or artifact.format_version != 1:
            raise ValueError("Expected a current exported transform artifact")
        artifact.pipeline._check_input(train)
        self.artifact_ = artifact
        if context:
            context.log_metadata(
                {"source": str(self.path), "retrained": False, "provenance": artifact.provenance}
            )
        return self

    def transform(
        self,
        data: PipelineData,
        *,
        context: StepContext | None = None,
    ) -> PipelineData:
        """Apply saved preprocessing and encoding to current targets and metadata.

        Parameters:
            data (PipelineData): Current grid with the saved feature schema and frequency.
            context (StepContext | None): Parent scope for nested transformation events.

        Returns:
            PipelineData: Encoded examples; no source-file access or training occurs.
        """
        return self.artifact_.pipeline.transform(data, context=context)

    def inverse_prediction(self, prediction: PredictionData) -> PredictionData:
        """Restore downstream reconstructions through the saved transform prefix.

        Parameters:
            prediction (PredictionData): Scalar output or reconstructed encoded inputs.

        Returns:
            PredictionData: Restored output through the nested inverse path.
        """
        for _, step in reversed(self.artifact_.pipeline.steps):
            prediction = step.inverse_prediction(prediction)
        return prediction
