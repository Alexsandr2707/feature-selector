"""A fitted experiment and predictions through its original configured pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentConfig
from .data import FeatureData, TemporalData
from .evaluation import prediction_report
from .pipeline import ConfiguredPipeline
from .pipeline.data import prepare_data
from .temporal import Boundaries


@dataclass
class ExperimentRun:
    """Fitted pipeline, run metadata and explicitly evaluated prediction reports."""

    config: ExperimentConfig
    pipeline: ConfiguredPipeline
    boundaries: Boundaries
    predictions: dict[str, pd.DataFrame]
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    artifact_dir: Path

    @property
    def model(self) -> Any:
        """Expose the terminal numerical estimator for history and inspection.

        Returns:
            Any: Estimator wrapped by the final step, or the custom final step itself.
        """
        step = self.pipeline.steps[-1][1]
        return getattr(step, "estimator", step)

    @property
    def feature_names(self) -> list[Any]:
        """Expose the fitted pipeline's original input feature order.

        Returns:
            list[Any]: Raw feature labels.
        """
        return list(self.pipeline.input_names_)

    def predict(
        self,
        data: FeatureData | TemporalData,
        split: str = "validation",
    ) -> pd.DataFrame:
        """Apply the fitted pipeline without fitting and return one split report.

        Parameters:
            data (FeatureData | TemporalData): Compatible raw grid; targets are optional.
            split (str): Train, validation or test temporal membership.

        Returns:
            pd.DataFrame: Original-unit predictions and any observed references.
        """
        prepared = prepare_data(data, self.boundaries, self.config.data["freq"])
        prediction = self.pipeline.predict(prepared, splits=(split,))
        mask = prediction.samples["split"].to_numpy() == split
        return prediction_report(prediction, mask)
