from __future__ import annotations

import json

from pathlib import Path

import numpy as np

import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import (
    compute_binary_metrics,
    print_metrics,
)

from src.training.checkpoint import (
    save_baseline_checkpoint,
)


def move_batch_to_device(
    batch: dict,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:

    pixel_values = (
        batch[
            "pixel_values"
        ]
        .to(
            device,
            non_blocking=True,
        )
    )

    labels = (
        batch[
            "labels"
        ]
        .to(
            device,
            non_blocking=True,
        )
    )

    return (
        pixel_values,
        labels,
    )


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    criterion,
    device: torch.device,
    gradient_clip_norm: float = 1.0,
    use_amp: bool = True,
) -> float:

    model.train()

    running_loss = 0.0

    sample_count = 0

    amp_enabled = (
        use_amp
        and device.type == "cuda"
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
        )
    )

    progress_bar = tqdm(
        dataloader,
        desc="Training",
    )

    for batch in progress_bar:

        (
            pixel_values,
            labels,
        ) = (
            move_batch_to_device(
                batch,
                device,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=
                device.type,

            dtype=
                torch.float16
                if device.type == "cuda"
                else torch.bfloat16,

            enabled=
                amp_enabled,
        ):

            logits = model(
                pixel_values
            )

            loss = criterion(
                logits,
                labels,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        if (
            gradient_clip_norm
            is not None
            and gradient_clip_norm > 0
        ):

            torch.nn.utils.clip_grad_norm_(
                model.get_trainable_parameters(),
                max_norm=
                    gradient_clip_norm,
            )

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = (
            labels.shape[
                0
            ]
        )

        running_loss += (
            loss.item()
            * batch_size
        )

        sample_count += (
            batch_size
        )

        current_loss = (
            running_loss
            / max(
                sample_count,
                1,
            )
        )

        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}"
        )

    epoch_loss = (
        running_loss
        / max(
            sample_count,
            1,
        )
    )

    return epoch_loss


@torch.no_grad()
def evaluate_model(
    model,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
    threshold: float = 0.5,
    description: str = "Evaluating",
) -> tuple[
    float,
    dict,
    dict,
]:

    model.eval()

    running_loss = 0.0

    sample_count = 0

    all_labels = []
    all_probabilities = []
    all_paths = []

    progress_bar = tqdm(
        dataloader,
        desc=description,
    )

    for batch in progress_bar:

        (
            pixel_values,
            labels,
        ) = (
            move_batch_to_device(
                batch,
                device,
            )
        )

        logits = model(
            pixel_values
        )

        loss = criterion(
            logits,
            labels,
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

        batch_size = (
            labels.shape[
                0
            ]
        )

        running_loss += (
            loss.item()
            * batch_size
        )

        sample_count += (
            batch_size
        )

        all_labels.extend(
            labels
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_probabilities.extend(
            probabilities
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        if (
            "image_path"
            in batch
        ):

            all_paths.extend(
                batch[
                    "image_path"
                ]
            )

    average_loss = (
        running_loss
        / max(
            sample_count,
            1,
        )
    )

    metrics = (
        compute_binary_metrics(
            labels=
                all_labels,

            probabilities=
                all_probabilities,

            threshold=
                threshold,
        )
    )

    raw_results = {
        "labels":
            np.asarray(
                all_labels
            ),

        "probabilities":
            np.asarray(
                all_probabilities
            ),

        "image_paths":
            all_paths,
    }

    return (
        average_loss,
        metrics,
        raw_results,
    )


def train_model(
    model,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_directory: str | Path,
    config: dict,
    threshold: float = 0.5,
    gradient_clip_norm: float = 1.0,
    use_amp: bool = True,
) -> list[dict]:

    checkpoint_directory = Path(
        checkpoint_directory
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    history = []

    best_auroc = (
        float(
            "-inf"
        )
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        print(
            "\n"
            "========================================"
        )

        print(
            f"EPOCH "
            f"{epoch}/{epochs}"
        )

        print(
            "========================================"
        )

        train_loss = (
            train_one_epoch(
                model=
                    model,

                dataloader=
                    train_dataloader,

                optimizer=
                    optimizer,

                criterion=
                    criterion,

                device=
                    device,

                gradient_clip_norm=
                    gradient_clip_norm,

                use_amp=
                    use_amp,
            )
        )

        (
            val_loss,
            val_metrics,
            _,
        ) = (
            evaluate_model(
                model=
                    model,

                dataloader=
                    val_dataloader,

                criterion=
                    criterion,

                device=
                    device,

                threshold=
                    threshold,

                description=
                    "Validation",
            )
        )

        print(
            f"\nTraining loss:   "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation loss: "
            f"{val_loss:.6f}"
        )

        print_metrics(
            val_metrics,
            title=(
                f"Validation "
                f"Epoch {epoch}"
            ),
        )

        epoch_record = {
            "epoch":
                epoch,

            "train_loss":
                float(
                    train_loss
                ),

            "val_loss":
                float(
                    val_loss
                ),

            **val_metrics,
        }

        history.append(
            epoch_record
        )

        # Keep an immutable checkpoint for every completed epoch.  This makes
        # it possible to compare or resume from any epoch, not only the latest.
        epoch_checkpoint = (
            checkpoint_directory
            / f"baseline_epoch_{epoch:03d}.pt"
        )

        save_baseline_checkpoint(
            path=epoch_checkpoint,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_metrics,
            config=config,
        )

        last_checkpoint = (
            checkpoint_directory
            / "baseline_last.pt"
        )

        save_baseline_checkpoint(
            path=
                last_checkpoint,

            model=
                model,

            optimizer=
                optimizer,

            epoch=
                epoch,

            metrics=
                val_metrics,

            config=
                config,
        )

        current_auroc = (
            val_metrics[
                "auroc"
            ]
        )

        if (
            not np.isnan(
                current_auroc
            )
            and current_auroc
            > best_auroc
        ):

            best_auroc = (
                current_auroc
            )

            best_checkpoint = (
                checkpoint_directory
                / "baseline_best.pt"
            )

            save_baseline_checkpoint(
                path=
                    best_checkpoint,

                model=
                    model,

                optimizer=
                    optimizer,

                epoch=
                    epoch,

                metrics=
                    val_metrics,

                config=
                    config,
            )

            print(
                "\nNew best checkpoint!"
            )

            print(
                f"Validation AUROC: "
                f"{best_auroc:.6f}"
            )

    history_path = (
        checkpoint_directory
        / "baseline_history.json"
    )

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
        )

    print(
        "\n========================================"
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"\nBest validation AUROC: "
        f"{best_auroc:.6f}"
    )

    print(
        "\nBest checkpoint:"
    )

    print(
        checkpoint_directory
        / "baseline_best.pt"
    )

    return history
