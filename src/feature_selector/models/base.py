"""Shared training and inference for PyTorch regression and reconstruction models."""

import copy
from math import isfinite
from typing import Any, Self, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from ..tracking import StepContext


class BatchLoss(TypedDict):
    """Training batch loss measured before the optimizer step."""

    batch: int
    samples: int
    train_loss: float
    step: int
    learning_rate: float


class EpochHistory(TypedDict):
    """Epoch metrics and losses of its batches in processing order."""

    epoch: int
    train_loss: float
    validation_loss: float | None
    batches: list[BatchLoss]


class TorchModelRunner:
    """Train and run PyTorch models with scalar or multidimensional targets."""

    def __init__(
        self,
        network: nn.Module,
        optimizer: Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        device: str = "cpu",
    ) -> None:
        """Attach a model and its optimization components.

        Parameters:
            network (nn.Module): Model returning predictions with the target shape.
                Scalar targets also accept predictions shaped (samples, 1).
            optimizer (Optimizer): Optimizer bound to the model parameters.
            scheduler (LRScheduler | None): Optional scheduler stepped per epoch.
            device (str): Device holding the model and processing batches.

        Returns:
            None.
        """
        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss_fn = nn.MSELoss()

    def fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        validation_data: tuple[NDArray[Any], NDArray[Any]] | None = None,
        *,
        epochs: int,
        batch_size: int,
        patience: int,
        random_state: int,
        context: StepContext | None = None,
        checkpoint_start_epoch: int = 1,
        max_grad_norm: float | None = None,
    ) -> Self:
        """Train for all epochs, or use early stopping when validation is provided.

        Parameters:
            X (NDArray[Any]): Training inputs with a leading sample dimension.
            y (NDArray[Any]): Finite targets with a leading sample dimension and
                nonempty trailing dimensions, such as scalars or reconstruction windows.
            validation_data (tuple[NDArray[Any], NDArray[Any]] | None): Validation inputs
                and targets with the training target's trailing shape; None keeps
                final weights without early stopping.
            epochs (int): Maximum number of epochs.
            batch_size (int): Samples per training batch.
            patience (int): Eligible epochs without improvement before stopping.
            random_state (int): Batch shuffling seed.
            context (StepContext | None): Batch and epoch telemetry for this execution.
            checkpoint_start_epoch (int): Positive first epoch eligible for checkpoint
                selection and early stopping; earlier epochs are still logged.
            max_grad_norm (float | None): Finite positive limit on the joint L2 gradient
                norm before each optimizer update; None disables clipping.

        Returns:
            Self: Runner with epoch and batch losses and the best eligible validation
                weights. Without validation or eligible epochs, final weights are kept
                and best_epoch_ is None; validation_loss is None only without validation.
        """
        for name, value in (
            ("epochs", epochs),
            ("batch_size", batch_size),
            ("patience", patience),
            ("checkpoint_start_epoch", checkpoint_start_epoch),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        if max_grad_norm is not None and (not isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be finite and positive")

        datasets = [(X, y)]
        if validation_data is not None:
            datasets.append(validation_data)
        for inputs, targets in datasets:
            if (
                inputs.ndim < 1
                or targets.ndim < 1
                or not len(inputs)
                or len(targets) != len(inputs)
                or not targets.size
            ):
                raise ValueError("Require nonempty inputs and one target per sample")

            if targets.shape[1:] != y.shape[1:]:
                raise ValueError("Validation target shape must match training targets")

            if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
                raise ValueError("Training and validation arrays must be finite")

        values = torch.as_tensor(X, dtype=torch.float32)
        targets = torch.as_tensor(y, dtype=torch.float32)

        validation: tuple[Tensor, Tensor] | None = None
        if validation_data is not None:
            validation = (
                torch.as_tensor(validation_data[0], dtype=torch.float32, device=self.device),
                torch.as_tensor(validation_data[1], dtype=torch.float32, device=self.device),
            )

        generator = torch.Generator().manual_seed(random_state)
        loader = DataLoader(
            TensorDataset(values, targets),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        best_loss = float("inf")
        stale_epochs = 0
        best_state = None
        self.history_: list[EpochHistory] = []
        self.best_epoch_: int | None = None

        context = context or StepContext()
        global_step = 0
        for epoch in range(1, epochs + 1):
            training_loss, batches = self._train_epoch(
                loader, context=context, epoch=epoch, global_step=global_step,
                max_grad_norm=max_grad_norm,
            )
            global_step += len(batches)
            validation_loss = self._validation_loss(*validation) if validation is not None else None

            if validation_loss is not None and not np.isfinite(validation_loss):
                raise ValueError("Nonfinite validation loss")

            self.history_.append(
                {
                    "epoch": epoch,
                    "train_loss": training_loss,
                    "validation_loss": validation_loss,
                    "batches": batches,
                }
            )

            context.log_metrics({"loss": training_loss}, split="train", epoch=epoch,
                                samples=len(X))
            if validation_loss is not None and validation_data is not None:
                context.log_metrics({"loss": validation_loss}, split="validation", epoch=epoch,
                                    samples=len(validation_data[0]))
                if epoch >= checkpoint_start_epoch:
                    if validation_loss < best_loss:
                        best_loss = validation_loss
                        stale_epochs = 0
                        self.best_epoch_ = epoch
                        # Copy tensors so later optimizer steps cannot change the checkpoint.
                        best_state = copy.deepcopy(self.network.state_dict())
                    else:
                        stale_epochs += 1

            if self.scheduler is not None:
                self.scheduler.step()

            if stale_epochs >= patience:
                break

        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()
        context.log_metadata({"best_epoch": self.best_epoch_,
                              "checkpoint_start_epoch": checkpoint_start_epoch,
                              "checkpoint_restored": best_state is not None,
                              "stopping_reason": "early_stopping" if stale_epochs >= patience
                              else "epochs_completed"})

        return self

    def _train_epoch(
        self, loader: DataLoader[Any], *, context: StepContext | None = None,
        epoch: int = 1, global_step: int = 0,
        max_grad_norm: float | None = None,
    ) -> tuple[float, list[BatchLoss]]:
        """Update weights and collect batch losses.

        Parameters:
            loader (DataLoader[Any]): Training input and target batches.
            context (StepContext | None): Live batch telemetry.
            epoch (int): One-based epoch coordinate.
            global_step (int): Optimizer steps completed in preceding epochs.
            max_grad_norm (float | None): Joint L2 gradient limit, or None to disable clipping.

        Returns:
            tuple[float, list[BatchLoss]]: Sample-weighted epoch loss and batch
                records with one-based batch numbers and actual batch sizes.
        """
        context = context or StepContext()
        self.network.train()
        total_loss = 0.0
        sample_count = 0
        batches: list[BatchLoss] = []

        for batch_number, (batch_x, batch_y) in enumerate(loader, start=1):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            prediction = self._prediction_for_targets(batch_x, batch_y)
            loss = self.loss_fn(prediction, batch_y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")

            learning_rate = self.optimizer.param_groups[0]["lr"]
            loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), max_grad_norm, error_if_nonfinite=True
                )
            self.optimizer.step()
            batch_loss = loss.item()
            batches.append(
                {
                    "batch": batch_number,
                    "samples": len(batch_x),
                    "train_loss": batch_loss,
                    "step": global_step + batch_number,
                    "learning_rate": learning_rate,
                }
            )

            context.log_metrics({"loss": batch_loss, "learning_rate": learning_rate},
                                split="train", epoch=epoch, batch=batch_number,
                                step=global_step + batch_number, samples=len(batch_x))
            total_loss += batch_loss * len(batch_x)
            sample_count += len(batch_x)

        return total_loss / sample_count, batches

    @torch.no_grad()
    def _validation_loss(self, X: Tensor, y: Tensor) -> float:
        """Calculate validation MSE.

        Parameters:
            X (Tensor): Validation inputs on the model's device.
            y (Tensor): Validation targets on the same device.

        Returns:
            float: Mean squared error.
        """
        self.network.eval()
        prediction = self._prediction_for_targets(X, y)
        return self.loss_fn(prediction, y).item()

    def _prediction_for_targets(self, X: Tensor, y: Tensor) -> Tensor:
        """Match scalar column outputs and reject incompatible prediction shapes.

        Parameters:
            X (Tensor): Input batch on the model's device.
            y (Tensor): Target batch with the same number of samples.

        Returns:
            Tensor: Predictions with exactly the target shape for samplewise MSE.
        """
        prediction = self.network(X)
        # Scalar regression heads may retain a singleton output dimension.
        if y.ndim == 1 and prediction.shape == (len(y), 1):
            prediction = prediction.squeeze(-1)
        if prediction.shape != y.shape:
            raise ValueError("Prediction shape must match target shape")
        return prediction

    @torch.no_grad()
    def predict(self, X: NDArray[Any], *, batch_size: int = 1024) -> NDArray[Any]:
        """Run inference in configurable batches.

        Parameters:
            X (NDArray[Any]): Inputs compatible with the attached model.
            batch_size (int): Maximum samples per prediction batch.

        Returns:
            NDArray[Any]: Predictions preserving the network's trailing dimensions.
                Scalar outputs shaped (samples, 1) become (samples,).
        """
        self.network.eval()
        predictions: list[NDArray[Any]] = []
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        batch_count = max(1, (len(X) + batch_size - 1) // batch_size)

        for batch in np.array_split(X, batch_count):
            values = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
            prediction = self.network(values)
            if prediction.ndim < 1 or prediction.shape[0] != len(batch):
                raise ValueError("Predictions must have one output per input sample")
            if prediction.ndim == 2 and prediction.shape[1] == 1:
                prediction = prediction.squeeze(-1)
            predictions.append(prediction.cpu().numpy())

        return np.concatenate(predictions)
