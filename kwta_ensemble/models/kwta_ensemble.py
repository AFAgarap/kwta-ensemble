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
"""Implementation of kWTA-ENN"""
from copy import deepcopy
from typing import Dict, Tuple

import torch

from kwta_ensemble.models.base import Model
from kwta_ensemble.layers import WinnersTakeAllLayer


class kWTAEnsemble(Model):
    def __init__(
        self,
        num_classes: int,
        expert_model: torch.nn.Module,
        num_subnetworks: int = 2,
        sparsity: float = 0.75,
        competition_delay: int = 3,
        optimizer: str = "sgd",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__(
            num_subnetworks=num_subnetworks,
            optimizer=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self.num_subnetworks = num_subnetworks
        self.experts = torch.nn.Sequential()
        for index in range(self.num_subnetworks):
            expert_model = deepcopy(expert_model)
            expert_model.apply(self.reset_parameters)
            self.experts.add_module(f"expert_{index}", expert_model)
        self.competitive_layer = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=(num_classes * num_subnetworks), out_features=num_classes
            ),
            WinnersTakeAllLayer(sparsity=sparsity),
        )
        self.competition_delay = competition_delay
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

    def forward(self, features: torch.Tensor, epoch: int = 0) -> torch.Tensor:
        outputs = []
        for index in range(self.num_subnetworks):
            output = self.experts[index](features)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        activations = {}
        for index, layer in enumerate(self.competitive_layer):
            if index < 1:
                activations[index] = (
                    layer(outputs) if index == 0 else layer(activations.get(index - 1))
                )
            elif (index == 1 and epoch >= self.competition_delay) or not self.training:
                activations[index] = layer(activations.get(index - 1))
        outputs = activations.get(len(activations) - 1)
        return outputs

    def epoch_train(self, data_loaders: Dict, phase: str) -> Tuple[float, float]:
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
