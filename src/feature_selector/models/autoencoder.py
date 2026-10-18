"""Convolutional and recurrent autoencoders for window reconstruction."""

from collections.abc import Callable
from typing import Any, Literal, Self

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from torch import Tensor, nn
from torch.nn import functional as F

from ..tracking import StepContext
from .base import EpochHistory, TorchModelRunner

type WindowArray = NDArray[np.float32]


class _Network(nn.Module):
    """Compress channels before reducing time, then reconstruct the input window."""

    def __init__(
        self, features: int, timesteps: int, hidden_channels: int, latent_channels: int
    ) -> None:
        """Configure independent encoder and decoder convolutions.

        Parameters:
            features (int): Positive number of input channels.
            timesteps (int): Positive window length restored by the decoder.
            hidden_channels (int): Positive width of the intermediate representation.
            latent_channels (int): Positive number of encoded channels.

        Returns:
            None.
        """
        super().__init__()
        self.timesteps = timesteps
        self.encoder = nn.Sequential(
            nn.Conv1d(features, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, features, kernel_size=1),
        )

    def encode(self, X: Tensor) -> Tensor:
        """Map windows to the network's latent sequence.

        Parameters:
            X (Tensor): Windows shaped (samples, timesteps, features).

        Returns:
            Tensor: Latent sequences with the temporal length set by the network architecture.
        """
        return self.encoder(X.transpose(1, 2)).transpose(1, 2)

    def decode(self, Z: Tensor) -> Tensor:
        """Restore the trained window length and input channels.

        Parameters:
            Z (Tensor): Encoded windows shaped (samples, encoded_timesteps, latent_channels).

        Returns:
            Tensor: Reconstructions shaped (samples, timesteps, features).
        """
        # An explicit size restores odd lengths without inventing a feature ordering.
        expanded = F.interpolate(
            Z.transpose(1, 2), size=self.timesteps, mode="linear", align_corners=False
        )
        return self.decoder(expanded).transpose(1, 2)

    def forward(self, X: Tensor) -> Tensor:
        """Reconstruct input windows through the latent representation.

        Parameters:
            X (Tensor): Windows shaped (samples, timesteps, features).

        Returns:
            Tensor: Reconstructions with the input shape and no output activation.
        """
        return self.decode(self.encode(X))


class _FeatureNetwork(_Network):
    """Apply shared feature compression independently at every timestep."""

    def __init__(self, features: int, hidden_channels: int, latent_channels: int) -> None:
        """Configure pointwise encoder and decoder convolutions.

        Parameters:
            features (int): Positive number of input features.
            hidden_channels (int): Positive intermediate channel count.
            latent_channels (int): Positive encoded channel count.

        Returns:
            None.
        """
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(
            nn.Conv1d(features, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, latent_channels, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, features, kernel_size=1),
        )

    def decode(self, Z: Tensor) -> Tensor:
        """Restore features independently at each latent timestep.

        Parameters:
            Z (Tensor): Latent sequences shaped (samples, timesteps, latent_channels).

        Returns:
            Tensor: Reconstructions shaped (samples, timesteps, features).
        """
        return self.decoder(Z.transpose(1, 2)).transpose(1, 2)


class _GRUConvNetwork(_Network):
    """Encode recurrent states into latent features and reconstruct through another GRU."""

    def __init__(self, features: int, hidden_channels: int, latent_channels: int) -> None:
        """Configure separate encoder and decoder recurrent networks.

        Parameters:
            features (int): Positive input and reconstructed feature count.
            hidden_channels (int): Positive hidden size of both single-layer GRUs.
            latent_channels (int): Positive encoded feature count.

        Returns:
            None.
        """
        nn.Module.__init__(self)
        self.encoder_gru = nn.GRU(features, hidden_channels, batch_first=True)
        self.encoder_projection = nn.Conv1d(hidden_channels, latent_channels, kernel_size=1)
        self.decoder_projection = nn.Sequential(
            nn.Conv1d(latent_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
        )
        self.decoder_gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)
        self.output_projection = nn.Linear(hidden_channels, features)

    def encode(self, X: Tensor) -> Tensor:
        """Compress recurrent features while preserving every timestep.

        Parameters:
            X (Tensor): Windows shaped (samples, timesteps, features).

        Returns:
            Tensor: Latent sequences shaped (samples, timesteps, latent_channels).
        """
        sequence, _ = self.encoder_gru(X)
        return self.encoder_projection(sequence.transpose(1, 2)).transpose(1, 2)

    def decode(self, Z: Tensor) -> Tensor:
        """Reconstruct windows using only the supplied latent sequence.

        Parameters:
            Z (Tensor): Latent sequences shaped (samples, timesteps, latent_channels).

        Returns:
            Tensor: Reconstructions shaped (samples, timesteps, features).
        """
        expanded = self.decoder_projection(Z.transpose(1, 2)).transpose(1, 2)
        # Start a fresh recurrent state so the encoder cannot bypass the latent bottleneck.
        sequence, _ = self.decoder_gru(expanded)
        return self.output_projection(sequence)


class ConvAutoencoder(BaseEstimator):
    """Learn window reconstruction and expose encoded sequences as array transformations."""

    network_: _Network
    runner_: TorchModelRunner
    window_shape_: tuple[int, int]
    latent_shape_: tuple[int, int]
    n_features_in_: int
    history_: list[EpochHistory]
    best_epoch_: int | None

    def __init__(
        self,
        hidden_channels: int = 32,
        latent_channels: int = 8,
        epochs: int = 200,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        min_lr: float = 0.0001,
        weight_decay: float = 0.01,
        patience: int = 30,
        scheduler: bool = True,
        random_state: int = 0,
        device: str = "cpu",
        prediction_batch_size: int = 1024,
        mode: Literal["reconstruct", "encode"] = "reconstruct",
        compression: Literal["features_time", "features"] = "features_time",
        max_grad_norm: float | None = None,
    ) -> None:
        """Configure channel widths and reconstruction training.

        Parameters:
            hidden_channels (int): Positive width of the intermediate representation.
            latent_channels (int): Positive encoded width; smaller than the input
                feature count is recommended for feature compression.
            epochs (int): Positive maximum number of training epochs.
            batch_size (int): Positive number of windows per training batch.
            learning_rate (float): Finite positive AdamW learning rate.
            min_lr (float): Finite scheduler minimum between zero and learning_rate.
            weight_decay (float): Finite nonnegative AdamW weight decay.
            patience (int): Positive number of stale validation epochs before stopping.
            scheduler (bool): Whether to apply cosine learning rate decay.
            random_state (int): Initialization and batch shuffle seed in [0, 2**63).
            device (str): PyTorch device used for training and inference.
            prediction_batch_size (int): Positive maximum windows per inference batch.
            mode (Literal["reconstruct", "encode"]): Output representation for transform/predict.
            compression (Literal["features_time", "features"]): Compress features with
                optional temporal downsampling; features preserves every timestep.
            max_grad_norm (float | None): Finite positive joint L2 gradient limit;
                None disables gradient clipping.

        Returns:
            None.
        """
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.scheduler = scheduler
        self.random_state = random_state
        self.device = device
        self.prediction_batch_size = prediction_batch_size
        self.mode = mode
        self.compression = compression
        self.max_grad_norm = max_grad_norm

    def _build_network(self, window_shape: tuple[int, int]) -> _Network:
        """Construct an untrained network for the selected convolutional architecture.

        Parameters:
            window_shape (tuple[int, int]): Positive timestep and feature counts.

        Returns:
            _Network: Network on CPU; the training routine selects its device and seed.
        """
        if self.compression == "features":
            return _FeatureNetwork(window_shape[1], self.hidden_channels, self.latent_channels)
        return _Network(
            window_shape[1], window_shape[0], self.hidden_channels, self.latent_channels
        )

    @staticmethod
    def _as_windows(
        X: NDArray[Any], *, shape: tuple[int, int] | None = None, allow_empty: bool = False
    ) -> WindowArray:
        """Validate real window arrays and convert them for float32 computation.

        Parameters:
            X (NDArray[Any]): Real numeric array shaped (samples, timesteps, channels).
                Values must remain finite when converted to float32.
            shape (tuple[int, int] | None): Required trailing dimensions, if known.
            allow_empty (bool): Whether a zero sample count is accepted for inference.

        Returns:
            WindowArray: Finite float32 windows with positive trailing dimensions.
        """
        if not isinstance(X, np.ndarray) or X.dtype.kind not in "iuf":
            raise ValueError("Windows must be a real numeric NumPy array")
        if X.ndim != 3 or min(X.shape[1:]) < 1 or (not allow_empty and not len(X)):
            raise ValueError("Require windows shaped (samples, timesteps, channels)")
        if shape is not None and X.shape[1:] != shape:
            raise ValueError(f"Window shape must match {shape}")
        # A copy also supports reversed views and read-only caller arrays in torch.
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.array(X, dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Windows must contain finite float32 values")
        return values

    def fit(
        self,
        X: NDArray[Any],
        y: Any = None,
        validation_data: NDArray[Any] | None = None,
        *,
        context: StepContext | None = None,
    ) -> Self:
        """Learn reconstruction using only training windows for weight updates.

        Parameters:
            X (NDArray[Any]): Nonempty finite windows shaped (samples, timesteps, features).
            y (Any): Ignored target accepted for sklearn pipeline compatibility.
            validation_data (NDArray[Any] | None): Nonempty finite windows with the
                training trailing shape, used to select weights and stop early.
            context (StepContext | None): Live runner telemetry for this execution.

        Returns:
            Self: Fitted estimator with history_ and best_epoch_; without validation,
                all epochs run, final weights are kept and best_epoch_ is None.
        """
        if self.mode not in ("reconstruct", "encode"):
            raise ValueError("Unknown autoencoder mode")
        if self.compression not in ("features_time", "features"):
            raise ValueError("Unknown autoencoder compression")
        values = self._as_windows(X)
        window_shape = (values.shape[1], values.shape[2])
        validation = None
        if validation_data is not None:
            valid_values = self._as_windows(validation_data, shape=window_shape)
            validation = (valid_values, valid_values)

        device = torch.device(self.device)
        devices = [device.index or 0] if device.type == "cuda" else []
        # Isolate initialization and training from the caller's torch random state.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.random_state)
            network = self._build_network(window_shape).to(device)
            optimizer = torch.optim.AdamW(
                network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            scheduler = None
            if self.scheduler:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.epochs, eta_min=self.min_lr
                )
            runner = TorchModelRunner(network, optimizer, scheduler, self.device)
            runner.fit(
                values,
                values,
                validation,
                epochs=self.epochs,
                batch_size=self.batch_size,
                patience=self.patience,
                random_state=self.random_state,
                context=context,
                max_grad_norm=self.max_grad_norm,
            )

        self.network_ = network
        self.runner_ = runner
        self.window_shape_ = window_shape
        latent_timesteps = (
            window_shape[0] if self.compression == "features" else (window_shape[0] + 1) // 2
        )
        self.latent_shape_ = (latent_timesteps, self.latent_channels)
        self.n_features_in_ = window_shape[1]
        self.history_ = runner.history_
        self.best_epoch_ = runner.best_epoch_
        return self

    @torch.no_grad()
    def _infer(
        self, X: WindowArray, operation: Callable[[Tensor], Tensor], shape: tuple[int, int]
    ) -> WindowArray:
        """Apply an encoder or decoder in sample order with bounded batch sizes.

        Parameters:
            X (WindowArray): Validated input windows; zero samples are allowed.
            operation (Callable[[Tensor], Tensor]): Fitted network operation to evaluate.
            shape (tuple[int, int]): Trailing output dimensions for an empty input.

        Returns:
            WindowArray: Float32 outputs in input sample order.
        """
        size = self.prediction_batch_size
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("prediction_batch_size must be a positive integer")
        if not len(X):
            return np.empty((0, *shape), dtype=np.float32)
        self.network_.eval()
        predictions: list[WindowArray] = []
        for start in range(0, len(X), size):
            values = torch.as_tensor(X[start : start + size], device=self.runner_.device)
            predictions.append(operation(values).cpu().numpy())
        return np.concatenate(predictions)

    def encode(self, X: NDArray[Any]) -> WindowArray:
        """Encode windows using the fitted compression architecture.

        Parameters:
            X (NDArray[Any]): Finite windows with the fitted trailing shape;
                a zero sample count is allowed.

        Returns:
            WindowArray: Encoded windows shaped (samples, *latent_shape_); feature-only
                compression preserves the input temporal length.
        """
        check_is_fitted(self, "runner_")
        values = self._as_windows(X, shape=self.window_shape_, allow_empty=True)
        return self._infer(values, self.network_.encode, self.latent_shape_)

    def decode(self, Z: NDArray[Any]) -> WindowArray:
        """Reconstruct windows from encoded sequences.

        Parameters:
            Z (NDArray[Any]): Finite latent windows with the fitted encoded shape;
                a zero sample count is allowed.

        Returns:
            WindowArray: Reconstructions shaped (samples, fitted_timesteps, fitted_features).
        """
        check_is_fitted(self, "runner_")
        values = self._as_windows(Z, shape=self.latent_shape_, allow_empty=True)
        return self._infer(values, self.network_.decode, self.window_shape_)

    def predict(self, X: NDArray[Any]) -> WindowArray:
        """Apply the configured output representation.

        Parameters:
            X (NDArray[Any]): Complete windows matching the fitted input shape.

        Returns:
            WindowArray: Reconstructions or latent sequences selected by mode.
        """
        if self.mode == "encode":
            return self.encode(X)
        check_is_fitted(self, "runner_")
        values = self._as_windows(X, shape=self.window_shape_, allow_empty=True)
        return self._infer(values, self.network_.forward, self.window_shape_)

    def transform(self, X: NDArray[Any]) -> WindowArray:
        """Apply the configured representation for downstream pipeline steps.

        Parameters:
            X (NDArray[Any]): Complete windows matching the fitted input shape.

        Returns:
            WindowArray: Reconstructions or latent sequences selected by mode.
        """
        return self.predict(X)

    def inverse_transform(self, Z: NDArray[Any]) -> WindowArray:
        """Return approximate input windows from latent pipeline outputs.

        Parameters:
            Z (NDArray[Any]): Finite encoded windows with the fitted latent shape.

        Returns:
            WindowArray: Reconstructions with the fitted input trailing shape.
        """
        return self.decode(Z)

    def fit_transform(
        self,
        X: NDArray[Any],
        y: Any = None,
        validation_data: NDArray[Any] | None = None,
    ) -> WindowArray:
        """Train reconstruction and return the representation selected by mode.

        Parameters:
            X (NDArray[Any]): Nonempty finite windows shaped (samples, timesteps, features).
            y (Any): Ignored target accepted for sklearn pipeline compatibility.
            validation_data (NDArray[Any] | None): Optional validation windows with
                the same trailing shape, used for early stopping.

        Returns:
            WindowArray: Reconstructed or encoded training windows selected by mode.
        """
        return self.fit(X, y, validation_data).transform(X)


class GRUConvAutoencoder(ConvAutoencoder):
    """Compress recurrent features with pointwise convolutions, preserving window length."""

    def __init__(
        self,
        hidden_channels: int = 32,
        latent_channels: int = 8,
        epochs: int = 200,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        min_lr: float = 0.0001,
        weight_decay: float = 0.01,
        patience: int = 30,
        scheduler: bool = True,
        random_state: int = 0,
        device: str = "cpu",
        prediction_batch_size: int = 1024,
        mode: Literal["reconstruct", "encode"] = "reconstruct",
        max_grad_norm: float | None = None,
    ) -> None:
        """Configure GRU widths and reconstruction training on complete windows.

        Parameters:
            hidden_channels (int): Positive hidden size of both single-layer GRUs.
            latent_channels (int): Positive encoded width; use fewer channels than
                input features for feature compression.
            epochs (int): Positive maximum training epoch count.
            batch_size (int): Positive training window count per batch.
            learning_rate (float): Positive AdamW learning rate.
            min_lr (float): Scheduler minimum between zero and learning_rate.
            weight_decay (float): Nonnegative AdamW weight decay.
            patience (int): Positive stale validation epoch limit for early stopping.
            scheduler (bool): Whether to apply cosine learning rate decay.
            random_state (int): Initialization and shuffle seed in [0, 2**63).
            device (str): PyTorch device for training and inference.
            prediction_batch_size (int): Positive inference window count per batch.
            mode (Literal["reconstruct", "encode"]): Representation returned by
                predict and transform; encode and decode remain available in either mode.
            max_grad_norm (float | None): Finite positive joint L2 gradient limit;
                None disables clipping.

        Returns:
            None.
        """
        super().__init__(
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            min_lr=min_lr,
            weight_decay=weight_decay,
            patience=patience,
            scheduler=scheduler,
            random_state=random_state,
            device=device,
            prediction_batch_size=prediction_batch_size,
            mode=mode,
            compression="features",
            max_grad_norm=max_grad_norm,
        )

    def _build_network(self, window_shape: tuple[int, int]) -> _GRUConvNetwork:
        """Construct independent GRU encoder and decoder around a feature bottleneck.

        Parameters:
            window_shape (tuple[int, int]): Positive timestep and feature counts.

        Returns:
            _GRUConvNetwork: Untrained network preserving the input temporal length.
        """
        return _GRUConvNetwork(window_shape[1], self.hidden_channels, self.latent_channels)
