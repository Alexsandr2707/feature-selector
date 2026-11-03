"""Autoencoders with dense feature projections and optional temporal processing."""

from collections.abc import Callable
from math import isfinite
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
        """Configure dense feature projections and temporal encoder/decoder convolutions.

        Parameters:
            features (int): Positive number of input features.
            timesteps (int): Positive window length restored by the decoder.
            hidden_channels (int): Positive width of the intermediate representation.
            latent_channels (int): Positive number of encoded channels.

        Returns:
            None.
        """
        super().__init__()
        self.timesteps = timesteps
        self.input_projection = nn.Sequential(
            nn.Linear(features, hidden_channels),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_channels, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output_projection = nn.Linear(hidden_channels, features)

    def encode(self, X: Tensor) -> Tensor:
        """Project features at each timestep and downsample the temporal sequence.

        Parameters:
            X (Tensor): Windows shaped (samples, timesteps, features).

        Returns:
            Tensor: Latent sequences shaped (samples, ceil(timesteps / 2), latent_channels).
        """
        projected = self.input_projection(X)
        # Temporal convolutions require channels before the time dimension.
        return self.encoder(projected.transpose(1, 2)).transpose(1, 2)

    def decode(self, Z: Tensor) -> Tensor:
        """Interpolate latent sequences and restore the trained input features.

        Parameters:
            Z (Tensor): Encoded windows shaped (samples, encoded_timesteps, latent_channels).

        Returns:
            Tensor: Reconstructions shaped (samples, timesteps, features).
        """
        # An explicit size restores odd lengths without inventing a feature ordering.
        expanded = F.interpolate(
            Z.transpose(1, 2), size=self.timesteps, mode="linear", align_corners=False
        )
        sequence = self.decoder(expanded).transpose(1, 2)
        return self.output_projection(sequence)

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
        """Configure dense encoder and decoder layers shared across timesteps.

        Parameters:
            features (int): Positive number of input features.
            hidden_channels (int): Positive intermediate channel count.
            latent_channels (int): Positive encoded channel count.

        Returns:
            None.
        """
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(
            nn.Linear(features, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, latent_channels),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, features),
        )

    def encode(self, X: Tensor) -> Tensor:
        """Compress features independently at each input timestep.

        Parameters:
            X (Tensor): Windows shaped (samples, timesteps, features).

        Returns:
            Tensor: Latent sequences shaped (samples, timesteps, latent_channels).
        """
        return self.encoder(X)

    def decode(self, Z: Tensor) -> Tensor:
        """Restore features independently at each latent timestep.

        Parameters:
            Z (Tensor): Latent sequences shaped (samples, timesteps, latent_channels).

        Returns:
            Tensor: Reconstructions shaped (samples, timesteps, features).
        """
        return self.decoder(Z)


class _GRUNetwork(_Network):
    """Encode recurrent states into latent features and reconstruct through another GRU."""

    def __init__(self, features: int, hidden_channels: int, latent_channels: int) -> None:
        """Configure separate recurrent networks with dense feature projections.

        Parameters:
            features (int): Positive input and reconstructed feature count.
            hidden_channels (int): Positive hidden size of both single-layer GRUs.
            latent_channels (int): Positive encoded feature count.

        Returns:
            None.
        """
        nn.Module.__init__(self)
        self.encoder_gru = nn.GRU(features, hidden_channels, batch_first=True)
        self.encoder_projection = nn.Linear(hidden_channels, latent_channels)
        self.decoder_projection = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
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
        return self.encoder_projection(sequence)

    def decode(self, Z: Tensor) -> Tensor:
        """Reconstruct windows using only the supplied latent sequence.

        Parameters:
            Z (Tensor): Latent sequences shaped (samples, timesteps, latent_channels).

        Returns:
            Tensor: Reconstructions shaped (samples, timesteps, features).
        """
        expanded = self.decoder_projection(Z)
        # Start a fresh recurrent state so the encoder cannot bypass the latent bottleneck.
        sequence, _ = self.decoder_gru(expanded)
        return self.output_projection(sequence)


class _S4SequenceBlock(nn.Module):
    """Apply unidirectional S4 to features within each window."""

    def __init__(
        self, channels: int, state_size: int, ssm_learning_rate: float, dropout: float
    ) -> None:
        """Configure a full S4 block without sharing state between windows.

        Parameters:
            channels (int): Positive input and output feature width.
            state_size (int): Positive even state size for the conjugate-pair kernel.
            ssm_learning_rate (float): Finite positive learning rate for SSM parameters.
            dropout (float): Probability in [0, 1) for dropout tied across timesteps.

        Returns:
            None.
        """
        super().__init__()
        # Load the standalone implementation only when constructing an S4 model.
        from .s4 import S4Block

        # self.norm = nn.LayerNorm(channels)
        self.s4 = S4Block(
            d_model=channels,
            d_state=state_size,
            transposed=False,
            dropout=dropout,
            tie_dropout=True,
            bidirectional=False,
            mode="diag",
            lr=ssm_learning_rate,
            wd=0.0,
            verbose=False,
        )
        # These tables are populated at initialization; their lambda factories break pickle.
        kernel = self.s4.layer.kernel
        kernel.lr_dict.default_factory = None
        kernel.wd_dict.default_factory = None

    def forward(self, X: Tensor) -> Tensor:
        """Transform features using past and current points within each window.

        Parameters:
            X (Tensor): Windows shaped (samples, timesteps, channels).

        Returns:
            Tensor: Features with the input shape and causal temporal context.
        """
        # sequence, _ = self.s4(self.norm(X))
        sequence, _ = self.s4(X)
        # Recurrent-only views become stale after weight updates and prevent deepcopy export.
        self.s4.layer.kernel.step_params = {}
        return sequence


class _S4Network(_Network):
    """Compress S4 features without changing temporal length."""

    def __init__(
        self,
        features: int,
        latent_channels: int,
        state_size: int,
        ssm_learning_rate: float,
        dropout: float,
    ) -> None:
        """Configure S4 encoding and decoding at the input feature width.

        Parameters:
            features (int): Positive input and reconstructed feature count.
            latent_channels (int): Positive encoded feature count.
            state_size (int): Positive even state size in both S4 blocks.
            ssm_learning_rate (float): Finite positive learning rate for SSM parameters.
            dropout (float): Probability in [0, 1) for tied dropout in both S4 blocks.

        Returns:
            None.
        """
        nn.Module.__init__(self)
        self.encoder = _S4SequenceBlock(features, state_size, ssm_learning_rate, dropout)
        self.encoder_projection = nn.Linear(features, latent_channels)
        self.decoder_projection = nn.Sequential(
            nn.Linear(latent_channels, features),
            nn.GELU(),
        )
        self.decoder = _S4SequenceBlock(features, state_size, ssm_learning_rate, dropout)
        self.output_projection = nn.Linear(features, features)

    def encode(self, X: Tensor) -> Tensor:
        """Apply S4 directly to input features and project to latent channels.

        Parameters:
            X (Tensor): Windows shaped (samples, timesteps, features).

        Returns:
            Tensor: Latent sequences shaped (samples, timesteps, latent_channels).
        """
        sequence = self.encoder(X)
        return self.encoder_projection(sequence)

    def decode(self, Z: Tensor) -> Tensor:
        """Reconstruct every input timestep using only the supplied latent sequence.

        Parameters:
            Z (Tensor): Latent windows shaped (samples, timesteps, latent_channels).

        Returns:
            Tensor: Reconstructions shaped (samples, timesteps, features).
        """
        sequence = self.decoder(self.decoder_projection(Z))
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
            batch_size (int): Positive number of windows per complete training batch;
                must not exceed the training window count. Incomplete batches are discarded.
            learning_rate (float): Finite positive AdamW learning rate.
            min_lr (float): Cosine scheduler minimum; used only when scheduler is enabled.
            weight_decay (float): Finite nonnegative AdamW weight decay.
            patience (int): Positive number of stale validation epochs before stopping.
            scheduler (bool): Whether to apply cosine learning rate decay.
            random_state (int): PyTorch initialization and batch shuffling seed.
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
        """Construct a dense feature autoencoder with optional temporal convolutions.

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

    def _build_optimizer(self, network: _Network) -> torch.optim.AdamW:
        """Create the shared AdamW optimizer for reconstruction training.

        Parameters:
            network (_Network): Initialized network on the training device.

        Returns:
            AdamW: Optimizer bound to all network parameters in a single group.
        """
        return torch.optim.AdamW(
            network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
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
            optimizer = self._build_optimizer(network)
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


class GRUAutoencoder(ConvAutoencoder):
    """Compress recurrent features with dense projections, preserving window length."""

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
            batch_size (int): Positive window count per complete training batch;
                must not exceed the training window count. Incomplete batches are discarded.
            learning_rate (float): Positive AdamW learning rate.
            min_lr (float): Cosine scheduler minimum; used only when scheduler is enabled.
            weight_decay (float): Nonnegative AdamW weight decay.
            patience (int): Positive stale validation epoch limit for early stopping.
            scheduler (bool): Whether to apply cosine learning rate decay.
            random_state (int): PyTorch initialization and batch shuffling seed.
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

    def _build_network(self, window_shape: tuple[int, int]) -> _GRUNetwork:
        """Construct independent GRU encoder and decoder around a feature bottleneck.

        Parameters:
            window_shape (tuple[int, int]): Positive timestep and feature counts.

        Returns:
            _GRUNetwork: Untrained network preserving the input temporal length.
        """
        return _GRUNetwork(window_shape[1], self.hidden_channels, self.latent_channels)


class S4Autoencoder(ConvAutoencoder):
    """Compress whole-window S4 features while preserving every timestep."""

    def __init__(
        self,
        latent_channels: int = 8,
        state_size: int = 64,
        epochs: int = 200,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        ssm_learning_rate: float = 0.001,
        min_lr: float = 0.0001,
        weight_decay: float = 0.01,
        patience: int = 30,
        scheduler: bool = True,
        random_state: int = 0,
        device: str = "cpu",
        prediction_batch_size: int = 1024,
        mode: Literal["reconstruct", "encode"] = "reconstruct",
        max_grad_norm: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Configure unidirectional S4 reconstruction with feature compression.

        Parameters:
            latent_channels (int): Positive encoded width; fewer channels than input
                features are recommended for feature compression.
            state_size (int): Positive even state size in each S4 block.
            epochs (int): Positive maximum training epoch count.
            batch_size (int): Positive window count per complete training batch;
                must not exceed the training window count. Incomplete batches are discarded.
            learning_rate (float): Positive AdamW learning rate for ordinary parameters.
            ssm_learning_rate (float): Finite positive initial learning rate for SSM
                parameters; their individual weight decay settings come from S4.
            min_lr (float): Common cosine scheduler minimum for all parameter groups;
                used only when scheduler is enabled.
            weight_decay (float): Nonnegative AdamW weight decay unless overridden by S4.
            patience (int): Positive stale validation epoch limit for early stopping.
            scheduler (bool): Whether to apply cosine learning rate decay.
            random_state (int): PyTorch initialization and batch shuffling seed.
            device (str): PyTorch device for training and inference.
            prediction_batch_size (int): Positive inference window count per batch.
            mode (Literal["reconstruct", "encode"]): Representation returned by
                predict and transform; encode and decode remain available in either mode.
            max_grad_norm (float | None): Finite positive joint L2 gradient limit;
                None disables clipping.
            dropout (float): Finite probability in [0, 1) for dropout in both S4 blocks.
                Masks are shared across timesteps during training only.

        Returns:
            None.
        """
        super().__init__(
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
        self.state_size = state_size
        self.ssm_learning_rate = ssm_learning_rate
        self.dropout = dropout

    def _build_network(self, window_shape: tuple[int, int]) -> _S4Network:
        """Construct independent S4 blocks after checking their integration parameters.

        Parameters:
            window_shape (tuple[int, int]): Positive timestep and feature counts.

        Returns:
            _S4Network: Untrained unidirectional network preserving temporal length.
        """
        if (
            isinstance(self.state_size, bool)
            or not isinstance(self.state_size, int)
            or self.state_size < 2
            or self.state_size % 2
        ):
            raise ValueError("state_size must be a positive even integer")
        if (
            isinstance(self.ssm_learning_rate, bool)
            or not isfinite(self.ssm_learning_rate)
            or self.ssm_learning_rate <= 0
        ):
            raise ValueError("ssm_learning_rate must be finite and positive")
        if (
            isinstance(self.dropout, bool)
            or not isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be finite and in [0, 1)")
        return _S4Network(
            window_shape[1],
            self.latent_channels,
            self.state_size,
            float(self.ssm_learning_rate),
            float(self.dropout),
        )

    def _build_optimizer(self, network: _Network) -> torch.optim.AdamW:
        """Honor the standalone S4 optimizer settings for each trainable parameter.

        Parameters:
            network (_Network): Initialized network on the training device.

        Returns:
            AdamW: Optimizer with parameters grouped by effective learning rate and
                weight decay; the ordinary group is first for existing LR telemetry.
        """
        ordinary = (self.learning_rate, self.weight_decay)
        groups: dict[tuple[float, float], list[nn.Parameter]] = {ordinary: []}
        for parameter in network.parameters():
            options = getattr(parameter, "_optim", {})
            settings = (
                options.get("lr", self.learning_rate),
                options.get("weight_decay", self.weight_decay),
            )
            groups.setdefault(settings, []).append(parameter)
        parameter_groups: list[dict[str, Any]] = [
            {"params": parameters, "lr": lr, "weight_decay": weight_decay}
            for (lr, weight_decay), parameters in groups.items()
            if parameters
        ]
        return torch.optim.AdamW(
            parameter_groups, lr=self.learning_rate, weight_decay=self.weight_decay
        )
