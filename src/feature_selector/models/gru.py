"""GRU regression on complete temporal windows."""

from typing import Any, Self

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from torch import Tensor, nn

from ..tracking import StepContext
from .base import TorchModelRunner


class _Network(nn.Module):
    def __init__(self, features: int, hidden_size: int, num_layers: int) -> None:
        """Configure the recurrent encoder and regression head.

        Parameters:
            features (int): Number of input features per timestep.
            hidden_size (int): Hidden state width.
            num_layers (int): Number of stacked GRU layers.

        Returns:
            None.
        """
        super().__init__()
        self.gru = nn.GRU(
            features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize recurrent weights and biases.

        Returns:
            None.
        """
        for name, parameter in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def forward(self, X: Tensor) -> Tensor:
        """Predict from the last timestep's recurrent output.

        Parameters:
            X (Tensor): Windows shaped (samples, timesteps, features).

        Returns:
            Tensor: Predictions shaped (samples,).
        """
        output, _ = self.gru(X)
        return self.head(output[:, -1]).squeeze(-1)


class GRURegressor(RegressorMixin, BaseEstimator):
    """Train a GRU with optional validation-based early stopping."""

    def __init__(
        self,
        hidden_size: int = 16,
        num_layers: int = 1,
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
        checkpoint_start_epoch: int = 1,
        max_grad_norm: float | None = None,
    ) -> None:
        """Configure network architecture and training.

        Parameters:
            hidden_size (int): Hidden state width.
            num_layers (int): Number of stacked GRU layers.
            epochs (int): Maximum training epochs.
            batch_size (int): Training samples per batch.
            learning_rate (float): Initial AdamW learning rate.
            min_lr (float): Minimum cosine scheduler learning rate.
            weight_decay (float): AdamW weight decay.
            patience (int): Eligible epochs without validation improvement before stopping.
            scheduler (bool): Whether to use cosine learning rate decay.
            random_state (int): Seed for initialization and batch shuffling.
            device (str): PyTorch device used for training and prediction.
            prediction_batch_size (int): Maximum windows per prediction batch.
            checkpoint_start_epoch (int): Positive first epoch eligible for validation
                checkpoint selection and early stopping; earlier losses are still logged.
            max_grad_norm (float | None): Finite positive joint L2 gradient limit;
                None disables gradient clipping.

        Returns:
            None.
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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
        self.checkpoint_start_epoch = checkpoint_start_epoch
        self.max_grad_norm = max_grad_norm

    def fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        validation_data: tuple[NDArray[Any], NDArray[Any]] | None = None,
        *, context: StepContext | None = None,
    ) -> Self:
        """Train on windows, selecting weights by validation loss when available.

        Parameters:
            X (NDArray[Any]): Finite windows shaped (samples, timesteps, features).
            y (NDArray[Any]): Finite targets shaped (samples,).
            validation_data (tuple[NDArray[Any], NDArray[Any]] | None): Validation windows
                and targets; None trains for all epochs and keeps final weights.
            context (StepContext | None): Live runner telemetry for this execution.

        Returns:
            Self: Fitted estimator with epoch and batch losses in history_
                and the one-based best_epoch_. If validation is absent or training ends
                before checkpoint_start_epoch, final weights are kept and best_epoch_ is None.
        """
        self.window_shape_ = X.shape[1:]

        # A private fork avoids changing the caller's global torch RNG state.
        devices = [] if self.device == "cpu" else [torch.device(self.device).index or 0]

        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.random_state)

            self.network_ = _Network(X.shape[2], self.hidden_size, self.num_layers).to(self.device)
            optimizer = torch.optim.AdamW(
                self.network_.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

            scheduler = None
            if self.scheduler:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.epochs, eta_min=self.min_lr
                )

            runner = TorchModelRunner(self.network_, optimizer, scheduler, self.device)
            runner.fit(
                X,
                y,
                validation_data,
                epochs=self.epochs,
                batch_size=self.batch_size,
                patience=self.patience,
                random_state=self.random_state,
                context=context,
                checkpoint_start_epoch=self.checkpoint_start_epoch,
                max_grad_norm=self.max_grad_norm,
            )

        self.runner_ = runner
        self.history_ = runner.history_
        self.best_epoch_ = runner.best_epoch_

        return self

    @torch.no_grad()
    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """Predict targets using the configured prediction batch size.

        Parameters:
            X (NDArray[Any]): Windows with the same trailing shape as training.

        Returns:
            NDArray[Any]: Predictions shaped (samples,).
        """
        check_is_fitted(self, "runner_")
        if X.shape[1:] != self.window_shape_:
            raise ValueError("Prediction window shape must match fit")
        return self.runner_.predict(X, batch_size=self.prediction_batch_size)
