"""Uniform step registration, cloning and concrete algorithm adapters."""

from dataclasses import replace
import unittest

from sklearn.base import clone

from feature_selector import Step, register_step
from feature_selector.factory import STEP_REGISTRY, build_pipeline
from feature_selector.pipeline import PipelineData
from feature_selector.tracking import StepContext
from support import sample_data


class Offset(Step):
    def __init__(self, amount: float = 1) -> None:
        """Store an additive feature offset.

        Parameters:
            amount (float): Constant added to every feature.

        Returns:
            None.
        """
        self.amount = amount

    def transform(
        self, data: PipelineData, *, context: StepContext | None = None,
    ) -> PipelineData:
        """Shift features without changing alignment.

        Parameters:
            data (PipelineData): Current values.
            context (StepContext | None): Unused execution context.

        Returns:
            PipelineData: Shifted representation.
        """
        return replace(data, X=data.X + self.amount)


class FactoryTests(unittest.TestCase):
    def test_custom_step_registration_and_clone(self) -> None:
        """Extend processing without editing orchestration or configuration validation.

        Returns:
            None.
        """
        register_step("test_offset", Offset)
        self.addCleanup(STEP_REGISTRY.pop, "test_offset")
        pipeline = build_pipeline([{"name": "offset", "type": "test_offset",
                                    "params": {"amount": 4}},
                                   {"name": "model", "type": "ridge"}])
        cloned = clone(pipeline)
        self.assertIsNot(cloned["offset"], pipeline["offset"])
        data = sample_data()
        pipeline.fit(data)
        self.assertEqual(pipeline.transform(data, through="offset").X.iloc[0, 0], 4)
        self.assertEqual(pipeline.predict(data).values.shape, (60,))

    def test_python_steps_are_cloned(self) -> None:
        """Keep constructor-owned Python components independent of fitted instances.

        Returns:
            None.
        """
        step = Offset(3)
        pipeline = build_pipeline([("offset", step), {"name": "model", "type": "ridge"}])
        self.assertIsNot(pipeline["offset"], step)

    def test_invalid_order_fails_at_use_and_reports_step(self) -> None:
        """Avoid class-specific order restrictions while preserving useful error context.

        Returns:
            None.
        """
        pipeline = build_pipeline([{"name": "first", "type": "windows", "params": {"lookback": 3}},
                                   {"name": "second", "type": "windows"},
                                   {"name": "model", "type": "ridge"}])
        with self.assertRaises(ValueError) as caught:
            pipeline.fit(sample_data())
        self.assertIn("pipeline/second", " ".join(caught.exception.__notes__))
        with self.assertRaises(ValueError):
            build_pipeline([{"name": "scale", "type": "standard"}])
