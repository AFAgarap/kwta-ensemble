# Winner-Take-All Ensemble Neural Network
# Copyright (C) 2021  Abien Fred Agarap
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""Implementation of model super class"""
from copy import deepcopy
import time
from typing import Dict, Tuple

import torch


class Model(torch.nn.Module):
    def __init__(
        self,
        optimizer: str,
        learning_rate: float,
        weight_decay: float = 1e-5,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available else "cpu"
        ),
    ):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                params=filter(
                    lambda parameters: parameters.requires_grad, self.parameters()
                ),
                lr=learning_rate,
                momentum=9e-1,
                weight_decay=weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=2, verbose=True, min_lr=1e-4, factor=1e-1
            )
        elif optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                params=filter(
                    lambda parameters: parameters.requires_grad, self.parameters()
                ),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=1, verbose=True, min_lr=1e-4, factor=1e-2
            )
        self.device = device
        self.to(self.device)
        self.train_loss = []
        self.train_accuracy = []
        self.valid_loss = []
        self.valid_accuracy = []

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reset_parameters(self, modules: torch.nn.Module) -> None:
        """
        Performs parameter reset to avoid
        use of identical weights for each expert network.

        Parameter
        ---------
        modules: torch.nn.Module
            The class layer whose weights will be reset.
        """
        if isinstance(modules, torch.nn.Linear) or isinstance(modules, torch.nn.Conv2d):
            modules.reset_parameters()

    def epoch_train(self, data_loaders: Dict, phase: str) -> Tuple[float, float]:
        raise NotImplementedError

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        epochs: int,
        show_every: int = 2,
    ) -> None:
        """
        Trains the model for a given epochs.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The training data loader to use.
        valid_loader: torch.utils.data.DataLoader
            The validation data loader to use.
        epochs: int
            The number of epochs to train the model.
        show_every: int
            The interval between training results displays.
        """
        start_time = time.time()
        best_model_weights = deepcopy(self.state_dict())
        best_accuracy = 0.0

        data_loaders = {"train": train_loader, "valid": valid_loader}

        for epoch in range(epochs):
            if (epoch + 1) % show_every == 0:
                print()
                print(f"Epoch {epoch + 1}/{epochs}")
                print("-" * 40)
            for phase in data_loaders.keys():
                if phase == "train":
                    self.train()
                else:
                    self.eval()

                epoch_loss, epoch_accuracy = self.epoch_train(
                    data_loaders=data_loaders, phase=phase
                )

                if phase == "train":
                    self.scheduler.step(epoch)

                if phase == "train":
                    self.train_loss.append(epoch_loss)
                    self.train_accuracy.append(epoch_accuracy)
                elif phase == "valid":
                    self.valid_loss.append(epoch_loss)
                    self.valid_accuracy.append(epoch_accuracy)

                if (epoch + 1) % show_every == 0:
                    print(
                        f"{phase.title()} Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}"
                    )

                if phase == "valid" and epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    best_model_weights = deepcopy(self.state_dict())

        time_elapsed = time.time() - start_time
        print()
        print(
            f"Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s"
        )
        print(f"Best Validation Accuracy: {best_accuracy:.4f}")
        print()

        self.load_state_dict(best_model_weights)
