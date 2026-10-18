"""Construct ordered pipeline steps from a small extensible registry."""

from collections.abc import Callable
from importlib import import_module
from typing import Any

from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .models.autoencoder import ConvAutoencoder, GRUConvAutoencoder
from .models.gru import GRURegressor
from .models.ridge import WindowRidge
from .pipeline import ConfiguredPipeline, Step
from .pipeline.adapters import ModelStep, TargetScale, TransformerStep
from .pipeline.frozen import FrozenPipeline
from .pipeline.temporal import AlignTarget, ExcludeIntervals, Windows
from .selectors import ElasticNetSelector, PLSSelector, StaticSelector

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
STEP_REGISTRY: dict[str, Callable[..., Step]] = {}


def register_step(name: str, constructor: Callable[..., Step]) -> None:
    """Register or replace a named constructor before loading project configuration.

    Parameters:
        name (str): Type identifier used by YAML steps.
        constructor (Callable[..., Step]): Constructor returning a uniform pipeline step.

    Returns:
        None: Updates the project registry.
    """
    STEP_REGISTRY[name] = constructor


def _transformer(
    constructor: Callable[..., BaseEstimator], requires_y: bool = False
) -> Callable[..., Step]:
    """Adapt a numerical transformer constructor to the project step interface.

    Parameters:
        constructor (Callable[..., BaseEstimator]): Underlying sklearn-style constructor.
        requires_y (bool): Whether complete target pairs are needed for training.

    Returns:
        Callable[..., Step]: Step constructor accepting the estimator's parameters.
    """

    def create(**params: Any) -> Step:
        """Construct a feature transformer adapter.

        Parameters:
            **params (Any): Underlying estimator parameters.

        Returns:
            Step: New transformer adapter.
        """
        return TransformerStep(constructor(**params), requires_y=requires_y)

    return create


def _model(
    constructor: Callable[..., BaseEstimator], reconstructs: bool = False
) -> Callable[..., Step]:
    """Adapt a numerical model constructor without adding parameter validation.

    Parameters:
        constructor (Callable[..., BaseEstimator]): Numerical estimator constructor.
        reconstructs (bool): Whether training ignores external y and reconstructs X.

    Returns:
        Callable[..., Step]: Model-step constructor.
    """

    def create(**params: Any) -> Step:
        """Construct a model adapter.

        Parameters:
            **params (Any): Underlying model parameters.

        Returns:
            Step: New model adapter.
        """
        return ModelStep(constructor(**params), reconstructs=reconstructs)

    return create


def _target_scale(scaler: dict[str, Any] | None = None) -> Step:
    """Construct a target scaler using the same short scaler names as feature steps.

    Parameters:
        scaler (dict[str, Any] | None): Type and optional params, or standard scaling.

    Returns:
        Step: Target-scaling adapter.
    """
    spec = scaler or {"type": "standard"}
    return TargetScale(SCALERS[spec["type"]](**spec.get("params", {})))


for _name, _constructor in SCALERS.items():
    register_step(_name, _transformer(_constructor))
register_step("static", _transformer(StaticSelector))
register_step("elasticnet", _transformer(ElasticNetSelector, requires_y=True))
register_step("pls", _transformer(PLSSelector, requires_y=True))
register_step("ridge", _model(WindowRidge))
register_step("gru", _model(GRURegressor))
register_step("conv_autoencoder", _model(ConvAutoencoder, reconstructs=True))
register_step("gru_conv_autoencoder", _model(GRUConvAutoencoder, reconstructs=True))
register_step("target_scale", _target_scale)
register_step("exclude_intervals", ExcludeIntervals)
register_step("align_target", AlignTarget)
register_step("windows", Windows)
register_step("frozen_pipeline", FrozenPipeline)
register_step("passthrough", Step)


def _restore_parameter(value: Any) -> Any:
    """Rebuild nested estimator descriptions from a saved Python configuration.

    Parameters:
        value (Any): Parameter value, collection or class/params estimator description.

    Returns:
        Any: Parameter value with explicitly described estimators reconstructed.
    """
    if isinstance(value, dict):
        if set(value) == {"class", "params"}:
            return _python_component(value)
        return {key: _restore_parameter(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_parameter(item) for item in value]
    return value


def _python_component(spec: dict[str, Any]) -> BaseEstimator:
    """Construct an explicitly named importable estimator from a Python API snapshot.

    Parameters:
        spec (dict[str, Any]): Qualified top-level class name and constructor params.
            Nested estimators use the same class/params description.

    Returns:
        BaseEstimator: Unfitted estimator reconstructed from its constructor parameters.
    """
    module_name, _, class_name = spec["class"].rpartition(".")
    constructor = getattr(import_module(module_name), class_name)
    if not isinstance(constructor, type) or not issubclass(constructor, BaseEstimator):
        raise TypeError("Saved Python components must be estimator classes")
    return constructor(**_restore_parameter(spec.get("params", {})))


def build_pipeline(specs: list[dict[str, Any] | tuple[str, Step]]) -> ConfiguredPipeline:
    """Construct steps in config order, cloning Python component instances.

    Parameters:
        specs (list[dict[str, Any] | tuple[str, Step]]): Named type/params mappings,
            saved class/params mappings or pairs of names and project steps.
            Python classes must be importable; names must be unique and contain no slash.

    Returns:
        ConfiguredPipeline: Unfitted ordered pipeline ending in a prediction-capable step.
    """
    steps: list[tuple[str, Step]] = []
    names: set[str] = set()
    for spec in specs:
        name = spec[0] if isinstance(spec, tuple) else spec["name"]
        if not isinstance(name, str) or not name or "/" in name or name in names:
            raise ValueError(f"Invalid or duplicate pipeline name: {name!r}")
        try:
            if isinstance(spec, tuple):
                step = clone(spec[1])
            elif "class" in spec:
                step = _python_component(spec)
            else:
                step = STEP_REGISTRY[spec["type"]](**spec.get("params", {}))
            if not isinstance(step, Step):
                raise TypeError("Pipeline components must implement the Step contract")
        except Exception as exc:
            exc.add_note(f"Constructing pipeline step {name!r}")
            raise
        steps.append((name, step))
        names.add(name)
    if not steps or not callable(getattr(steps[-1][1], "predict", None)):
        raise ValueError("Pipeline must end in a step providing predict")
    return ConfiguredPipeline(steps)
