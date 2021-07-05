# k-Winners-Take-All Ensemble Neural Network
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
import os
import time
from typing import Dict, Tuple

import torch


class Model(torch.nn.Module):
    """
    Super class for ensemble models.
    """

    def __init__(
        self,
        num_subnetworks: int,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available else "cpu"
        ),
    ):
        """
        Builds the super class for ensemble models.

        Parameters
        ----------
        optimizer: str
            The optimization algorithm to use.
        learning_rate: float
            The learning rate to use for optimization.
        weight_decay: float
            The weight decay to use for regularization.
        device: torch.device
            The device to use for computations.
        """
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_subnetworks = num_subnetworks
        self.device = device
        self.to(self.device)
        self.train_loss = []
        self.train_accuracy = []
        self.valid_loss = []
        self.valid_accuracy = []

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        The forward pass by the model.

        Parameter
        ---------
        features: torch.Tensor
            The input features.
        """
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
        """
        Performs single epoch training for model.

        Parameters
        ----------
        data_loaders: Dict
            The data loaders to use for
            training and validation.
        phase: str
            The phase of training,
            whether training or validation.

        Returns
        -------
        Tuple[float, float]
            epoch_loss
                The training or validation epoch loss.
            epoch_accuracy
                The training or validation epoch accuracy.
        """
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for features, labels in data_loaders.get(phase):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = self(features)
                loss = self.criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    self.optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += (outputs.argmax(1) == labels).sum().item() / len(labels)
        epoch_loss /= len(data_loaders.get(phase))
        epoch_accuracy /= len(data_loaders.get(phase))
        return epoch_loss, epoch_accuracy

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

    def score(self, data_loader: torch.utils.data.DataLoader) -> float:
        """
        Computes the accuracy of the model.

        Parameter
        ---------
        data_loader: torch.utils.data.DataLoader
            The data loader to use in evaluating the model.

        Returns
        -------
        accuracy: float
            The model accuracy.
        """
        self.eval()
        self.device = torch.device("cpu")
        self.to(self.device)
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                outputs = self.predict(features)
                correct = (outputs.argmax(1) == labels).sum().item()
                accuracy = correct / len(labels)
        accuracy *= 100.0
        return accuracy

    def save_model(self, filename: str) -> None:
        """
        Exports the trained model to
        outputs/models directory.

        Parameter
        ---------
        filename: str
            The filename for the exported model.
        """
        print("[INFO] Exporting trained model...")
        model_name = filename.split("-")[0]
        dataset_name = filename.split("-")[3]
        model_path = os.path.join("outputs", "models", model_name, dataset_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        filename = f"{filename}.pth"
        filename = os.path.join(model_path, filename)
        torch.save(self.state_dict(), filename)
        print(f"[SUCCESS] Trained model exported to {filename}")

    def load_model(self, filename: str) -> None:
        print("[INFO] Loading the trained model...")
        model_name = filename.split("-")[0]
        dataset_name = filename.split("-")[3]
        model_path = os.path.join("outputs", "models", model_name, dataset_name)
        if not filename.endswith(".pth"):
            filename = f"{filename}.pth"
        filename = os.path.join(model_path, filename)
        if os.path.isfile(filename):
            self.load_state_dict(torch.load(filename))
            print("[SUCCESS] Trained model ready for use.")
        else:
            print("[ERROR] Trained model not found.")
