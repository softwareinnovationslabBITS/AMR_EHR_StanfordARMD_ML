"""
Data loading and loss evaluation helpers for the baseline TabTransformer final
loss analysis.

#migrate: extracted from tabtransformer_loss_evaluation.py
"""

import logging
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AMRDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cat_idx: List[int],
        cont_idx: List[int],
        bin_idx: List[int],
    ) -> None:

        self.X_cat = torch.as_tensor(
            X[:, cat_idx].astype(np.int64),
            dtype=torch.long,
        )

        self.X_cont = torch.as_tensor(
            X[:, cont_idx].astype(np.float32),
            dtype=torch.float32,
        )

        self.X_bin = torch.as_tensor(
            X[:, bin_idx].astype(np.float32),
            dtype=torch.float32,
        )

        self.y = torch.as_tensor(
            y.astype(np.float32),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        return (
            self.X_cat[index],
            self.X_cont[index],
            self.X_bin[index],
            self.y[index],
        )


@torch.no_grad()
def evaluate_mean_loss(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
) -> float:
    """
    Calculate mean loss over one complete dataset split.
    """

    model.eval()

    total_loss = 0.0
    total_observations = 0

    start_time = time.time()

    for batch_number, (
        x_cat,
        x_cont,
        x_bin,
        y_batch,
    ) in enumerate(loader, start=1):

        x_cat = x_cat.to(
            device,
            non_blocking=True,
        )

        x_cont = x_cont.to(
            device,
            non_blocking=True,
        )

        x_bin = x_bin.to(
            device,
            non_blocking=True,
        )

        y_batch = y_batch.to(
            device,
            non_blocking=True,
        )

        logits = model(
            x_cat,
            x_cont,
            x_bin,
        )

        batch_loss = criterion(
            logits,
            y_batch,
        )

        batch_size = y_batch.size(0)

        total_loss += batch_loss.item() * batch_size
        total_observations += batch_size

        if batch_number % 500 == 0:
            elapsed_minutes = (time.time() - start_time) / 60
            logger.info(
                "%s: processed %s rows (%.2f minutes)",
                split_name,
                f"{total_observations:,}",
                elapsed_minutes,
            )

    if total_observations == 0:
        raise ValueError(
            f"{split_name} loader contained no observations."
        )

    return total_loss / total_observations
