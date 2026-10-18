"""Select DataFrame or NumPy features by names, positions, ElasticNet, or PLS."""

from collections.abc import Sequence
from typing import Any, Literal, Self, cast, overload

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.utils.validation import check_is_fitted

type FloatArray = NDArray[np.float64]
type ColumnIndices = NDArray[np.intp]
type FeatureMatrix = pd.DataFrame | NDArray[Any]
type Threshold = Literal["median", "mean"] | float


class _Selector(TransformerMixin, BaseEstimator):
    """Share positional selection and input-schema checks."""

    selected_indices_: ColumnIndices
    selected_features_: list[Any]
    feature_names_in_: NDArray[Any]
    n_features_in_: int
    n_training_pairs_: int
    _input_columns_: pd.Index | None

    @staticmethod
    def _validate_input(X: FeatureMatrix) -> None:
        """Check feature matrix structure.

        Parameters:
            X (FeatureMatrix): Two-dimensional features.

        Returns:
            None.
        """
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("Features must be a DataFrame or NumPy array")
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("Features must be two-dimensional with at least one column")
        if isinstance(X, pd.DataFrame) and X.columns.has_duplicates:
            raise ValueError("Feature columns must be unique")

    def _record_features(self, X: FeatureMatrix) -> None:
        """Record the training schema, replacing metadata from a previous fit.

        Parameters:
            X (FeatureMatrix): Nonempty training features.

        Returns:
            None.
        """
        self._validate_input(X)
        if len(X) == 0:
            raise ValueError("Training features must be nonempty")
        self.n_features_in_ = X.shape[1]
        self._input_columns_ = X.columns.copy() if isinstance(X, pd.DataFrame) else None

        # sklearn reserves feature_names_in_ for string column names.
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_

        if isinstance(X, pd.DataFrame) and all(isinstance(name, str) for name in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

    def _output_names(self) -> NDArray[Any]:
        """Resolve names for positional selection.

        Returns:
            NDArray[Any]: Original column labels or generated x0, x1, ... names.
        """
        if self._input_columns_ is not None:
            return self._input_columns_.to_numpy()
        return np.asarray([f"x{index}" for index in range(self.n_features_in_)], dtype=object)

    def _set_selection(self, indices: ColumnIndices) -> None:
        """Store positions and compatibility labels in output order.

        Parameters:
            indices (ColumnIndices): Selected positions.

        Returns:
            None.
        """
        self.selected_indices_ = np.asarray(indices, dtype=np.intp).copy()
        self.selected_features_ = list(self._output_names()[self.selected_indices_])

    def _training_pairs(self, X: FeatureMatrix, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Extract complete numeric feature/target pairs.

        Parameters:
            X (FeatureMatrix): Training features.
            y (ArrayLike): One target per row; pandas indexes must match when both exist.

        Returns:
            tuple[FloatArray, FloatArray]: Filtered features and flat targets.
        """
        self._record_features(X)
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series) and not X.index.equals(y.index):
            raise ValueError("X and y must have identical indexes")

        target = np.asarray(y, dtype=float).reshape(-1)
        if len(target) != len(X):
            raise ValueError("X and y must have identical lengths")

        values = np.asarray(X, dtype=float)
        complete = np.isfinite(values).all(axis=1) & np.isfinite(target)
        self.n_training_pairs_ = int(complete.sum())

        if self.n_training_pairs_ < 3:
            raise ValueError("A learned selector needs at least three complete training pairs")
        return values[complete], target[complete]

    @overload
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...

    @overload
    def transform(self, X: NDArray[Any]) -> NDArray[Any]: ...

    def transform(self, X: FeatureMatrix) -> FeatureMatrix:
        """Select columns while preserving the input container and row order.

        Parameters:
            X (FeatureMatrix): Features in training order; NumPy columns are positional.

        Returns:
            FeatureMatrix: Selected columns; set_output may override the container.
        """
        check_is_fitted(self, "selected_indices_")
        self._validate_input(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count must match fit")

        if isinstance(X, pd.DataFrame):
            if self._input_columns_ is not None and not X.columns.equals(self._input_columns_):
                raise ValueError("Feature names and order must match fit")
            return X.iloc[:, self.selected_indices_].copy()

        return X[:, self.selected_indices_].copy()

    def get_selected_indices(self) -> ColumnIndices:
        """Return selected positions in output order.

        Returns:
            ColumnIndices: A copy of selected positions.
        """
        check_is_fitted(self, "selected_indices_")
        return self.selected_indices_.copy()

    def get_selection_mask(self) -> NDArray[np.bool_]:
        """Return selected membership in input column order.

        Returns:
            NDArray[np.bool_]: One boolean per input column; output order is not encoded.
        """
        check_is_fitted(self, "selected_indices_")
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask

    def get_feature_names_out(
        self, input_features: Sequence[Any] | NDArray[Any] | None = None
    ) -> NDArray[Any]:
        """Resolve output labels, generating x0, x1, ... for unnamed inputs.

        Parameters:
            input_features (Sequence[Any] | NDArray[Any] | None): Optional input names;
                must match training labels when those exist.

        Returns:
            NDArray[Any]: Selected names in output order.
        """
        check_is_fitted(self, "selected_indices_")
        names = self._output_names()
        if input_features is not None:
            supplied = np.asarray(input_features, dtype=object)
            if supplied.shape != (self.n_features_in_,):
                raise ValueError("input_features must have one name per input column")
            if self._input_columns_ is not None and not np.array_equal(supplied, names):
                raise ValueError("input_features must match fit")
            names = supplied
        return names[self.selected_indices_]


class StaticSelector(_Selector):
    """Select explicit labels or positions in the requested order."""

    def __init__(
        self,
        columns: list[Any] | tuple[Any, ...] | None = None,
        *,
        indices: Sequence[int] | ColumnIndices | None = None,
    ) -> None:
        """Configure exactly one selection mode.

        Parameters:
            columns (list[Any] | tuple[Any, ...] | None): Unique DataFrame labels.
            indices (Sequence[int] | ColumnIndices | None): Unique nonnegative positions.

        Returns:
            None.
        """
        self.columns = columns
        self.indices = indices

    def _validate_params(self) -> None:
        """Validate selection mode and requested labels or positions.

        Returns:
            None.
        """
        if (self.columns is None) == (self.indices is None):
            raise ValueError("Provide exactly one of columns or indices")

        if self.columns is not None:
            if not isinstance(self.columns, (list, tuple)) or not self.columns:
                raise ValueError("columns must be a nonempty list or tuple of distinct labels")
            try:
                unique = len(set(self.columns)) == len(self.columns)
            except TypeError as exc:
                raise ValueError("Column labels must be hashable") from exc

            if not unique:
                raise ValueError("Column labels must be distinct")

        else:
            indices = self.indices
            if not isinstance(indices, (Sequence, np.ndarray)):
                raise ValueError("indices must be a nonempty sequence of distinct positions")
            if isinstance(indices, np.ndarray) and indices.ndim != 1:
                raise ValueError("indices must be one-dimensional")
            if len(indices) == 0:
                raise ValueError("indices must be a nonempty sequence of distinct positions")
            if any(
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or index < 0
                for index in indices
            ):
                raise ValueError("indices must contain nonnegative integers")

            if len(set(indices)) != len(indices):
                raise ValueError("indices must be distinct")

    def fit(self, X: FeatureMatrix, y: ArrayLike | None = None) -> Self:
        """Resolve configured labels or positions against the training schema.

        Parameters:
            X (FeatureMatrix): Training features; label selection requires a DataFrame.
            y (ArrayLike | None): Unused target for sklearn compatibility.

        Returns:
            Self: Selector with selected_indices_ in requested order.
        """
        self._validate_params()
        self._record_features(X)

        if self.columns is not None:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Selection by column labels requires a DataFrame")
            requested_columns = pd.Index(self.columns)
            positions = X.columns.get_indexer(requested_columns)
            if (positions < 0).any():
                missing = [name for name, position in zip(self.columns, positions) if position < 0]
                raise ValueError(f"Unknown feature names: {missing}")
        else:
            positions = np.asarray(self.indices, dtype=np.intp)
            if (positions >= self.n_features_in_).any():
                raise ValueError("Selected indices exceed the feature count")

        self._set_selection(positions)
        return self


class ElasticNetSelector(_Selector):
    """Select features by absolute ElasticNet coefficients, retaining input order."""

    def __init__(
        self,
        alpha: float = 0.01,
        l1_ratio: float = 0.7,
        top_k: int | None = None,
        threshold: Threshold = "median",
        max_iter: int = 10000,
        random_state: int | np.random.RandomState | None = 0,
    ) -> None:
        """Configure coefficient-based selection.

        Parameters:
            alpha (float): Nonnegative ElasticNet regularization strength.
            l1_ratio (float): L1 penalty fraction between zero and one.
            top_k (int | None): Maximum nonzero coefficients to keep; overrides threshold.
            threshold (Threshold): Minimum magnitude, or median/mean coefficient magnitude.
            max_iter (int): Maximum solver iterations.
            random_state (int | np.random.RandomState | None): Solver seed.

        Returns:
            None.
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.top_k = top_k
        self.threshold = threshold
        self.max_iter = max_iter
        self.random_state = random_state

    def _make_model(self) -> ElasticNet:
        """Construct the underlying regression model.

        Returns:
            ElasticNet: Unfitted model with the configured solver parameters.
        """
        return ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def _validate_params(self) -> None:
        """Check solver and selection parameters independent of training data.

        Returns:
            None.
        """
        validate = getattr(self._make_model(), "_validate_params", None)
        if callable(validate):
            validate()
        if self.top_k is not None:
            if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
                raise ValueError("top_k must be a positive integer")
        elif not (
            isinstance(self.threshold, str)
            and self.threshold in ("median", "mean")
            or isinstance(self.threshold, (int, float))
            and self.threshold >= 0
        ):
            raise ValueError("threshold must be median, mean or a nonnegative number")

    def _selection_mask(self, importance: FloatArray) -> NDArray[np.bool_]:
        """Choose positive coefficients by rank or threshold.

        Parameters:
            importance (FloatArray): Absolute coefficient magnitudes.

        Returns:
            NDArray[np.bool_]: Selection mask in input column order.
        """
        if not (importance > 0).any():
            raise ValueError("ElasticNet selected no features: all coefficients are zero")
        if self.top_k is not None:
            # Stable ranking resolves equal magnitudes by input column order.
            chosen = np.argsort(-importance, kind="stable")[: self.top_k]
            selected = np.zeros(len(importance), dtype=bool)
            selected[chosen] = importance[chosen] > 0
        else:
            if self.threshold == "median":
                cutoff = np.median(importance)
            elif self.threshold == "mean":
                cutoff = np.mean(importance)
            else:
                cutoff = self.threshold
            cutoff = cast(float, cutoff)
            selected = (importance > 0) & (importance >= cutoff)

        if not selected.any():
            raise ValueError("ElasticNet threshold selected no features")

        return selected

    def fit(self, X: FeatureMatrix, y: ArrayLike) -> Self:
        """Fit ElasticNet on complete pairs and select nonzero coefficients.

        Parameters:
            X (FeatureMatrix): Training matrix; DataFrame columns must be unique.
            y (ArrayLike): Aligned targets, one per row.

        Returns:
            Self: Selector with model_, importances_, and selected_features_.
        """
        self._validate_params()
        values, target = self._training_pairs(X, y)
        if self.top_k is not None and self.top_k > self.n_features_in_:
            raise ValueError("top_k must be an integer between 1 and n_features")
        self.model_ = self._make_model()
        self.model_.fit(values, target)
        importance = np.abs(self.model_.coef_)
        selected = self._selection_mask(importance)
        self.importances_ = importance
        self._set_selection(np.flatnonzero(selected))
        return self


class PLSSelector(_Selector):
    """Partition features by PLS weights and keep one representative per cluster."""

    clusters_: list[list[Any]]
    cluster_indices_: list[ColumnIndices]

    def __init__(self, depth: int = 3) -> None:
        """Configure the maximum binary partition depth.

        Parameters:
            depth (int): Nonnegative depth; zero produces one cluster.

        Returns:
            None.
        """
        self.depth = depth

    def _validate_params(self) -> None:
        """Check the partition depth.

        Returns:
            None.
        """
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError("depth must be a nonnegative integer")

    @staticmethod
    def _fit_pls(values: FloatArray, target: FloatArray) -> PLSRegression:
        """Fit up to two latent components for a feature subset.

        Parameters:
            values (FloatArray): Complete training features.
            target (FloatArray): Corresponding targets.

        Returns:
            PLSRegression: Fitted PLS model.
        """
        return PLSRegression(n_components=min(2, values.shape[1])).fit(values, target)

    @staticmethod
    def _representative(values: FloatArray, target: FloatArray) -> int:
        """Select the feature with the highest individual PLS reconstruction score.

        Parameters:
            values (FloatArray): Features in one cluster.
            target (FloatArray): Corresponding targets.

        Returns:
            int: Local index with the highest score; ties choose the first.
        """
        if values.shape[1] == 1 or np.std(target) == 0:
            return 0
        pls = PLSSelector._fit_pls(values, target)

        # Match PLS sample scaling without relying on sklearn's private attributes.
        feature_std = values.std(axis=0, ddof=1)
        feature_std[feature_std == 0] = 1
        standardized = (values - values.mean(axis=0)) / feature_std
        target_std = target.std(ddof=1) or 1.0
        contributions = pls.x_rotations_ @ pls.y_loadings_.T
        predictions = standardized * contributions[:, 0] * target_std + target.mean()
        residual = ((target[:, None] - predictions) ** 2).sum(axis=0)
        total = ((target - target.mean()) ** 2).sum()
        scores = 1 - residual / total

        return int(np.argmax(scores))

    def _partition(
        self,
        values: FloatArray,
        target: FloatArray,
        columns: ColumnIndices,
        depth: int,
    ) -> list[ColumnIndices]:
        """Recursively split column indices by dominant PLS component weights.

        Parameters:
            values (FloatArray): Full training feature matrix.
            target (FloatArray): Corresponding targets.
            columns (ColumnIndices): Indices belonging to the current cluster.
            depth (int): Remaining split depth.

        Returns:
            list[ColumnIndices]: Leaf clusters in traversal order.
        """
        if depth == 0 or len(columns) == 1 or np.std(target) == 0:
            return [columns]
        pls = self._fit_pls(values[:, columns], target)
        first_weights = np.abs(pls.x_weights_[:, 0])
        second_weights = np.abs(pls.x_weights_[:, 1])
        left = first_weights > second_weights
        if not left.any() or left.all():
            # Degenerate weights still produce two nonempty, deterministic children.
            left = np.arange(len(columns)) < (len(columns) + 1) // 2
        return self._partition(values, target, columns[left], depth - 1) + self._partition(
            values, target, columns[~left], depth - 1
        )

    def fit(self, X: FeatureMatrix, y: ArrayLike) -> Self:
        """Build PLS clusters from complete pairs and select their representatives.

        Parameters:
            X (FeatureMatrix): Training matrix; DataFrame columns must be unique.
            y (ArrayLike): Aligned targets, one per row.

        Returns:
            Self: Selector with clusters_ and selected_features_ in traversal order.
        """
        self._validate_params()
        values, target = self._training_pairs(X, y)
        nonconstant = np.flatnonzero(values.std(axis=0) > 0)
        if len(nonconstant) == 0:
            raise ValueError("PLS needs at least one nonconstant feature")

        # Further splits cannot produce more leaves than there are features.
        clusters = self._partition(values, target, nonconstant, min(self.depth, len(nonconstant)))
        self.cluster_indices_ = [columns.copy() for columns in clusters]
        names = self._output_names()
        self.clusters_ = [list(names[columns]) for columns in clusters]
        chosen = [columns[self._representative(values[:, columns], target)] for columns in clusters]
        self._set_selection(np.asarray(chosen, dtype=np.intp))
        return self
