# k-Winners-Take-All Ensemble Neural Network
# Copyright (C) 2021  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Implementation of ensemble neural network"""
from copy import deepcopy
from typing import Union

import torch

from kwta_ensemble.models.base import Model


class Ensemble(Model):
    def __init__(
        self,
        network: torch.nn.Module,
        feature_extractor: Union[torch.nn.Module, torch.nn.Sequential],
        use_feature_extractor: bool = False,
        num_subnetworks: int = 3,
        optimizer: str = "sgd",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Constructs an ensemble of neural networks.

        Parameters
        ----------
        network: torch.nn.Module:
            The learner architecture to use in ensemble.
        num_subnetworks: int
            The number of networks to use in ensemble.
        optimizer: str
            The optimizer to use.
        learning_rate: float
            The learning rate to use for optimization.
        weight_decay: float
            The weight decay parameter to use.
        device: torch.device
            The device to use for computation.
        """
        super().__init__(
            num_subnetworks=num_subnetworks,
            use_feature_extractor=use_feature_extractor,
            feature_extractor=feature_extractor,
        )
        self.model = torch.nn.Sequential()
        for index in range(self.num_subnetworks):
            network = deepcopy(network)
            network.apply(self.reset_parameters)
            self.model.add_module(f"network_{index}", network)
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
        self.to(self.device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        The forward pass by the model.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        logits: torch.Tensor
            The model outputs.
        """
        outputs = []
        for network in self.model:
            outputs.append(network(features))
        outputs = torch.stack(outputs, dim=1)
        logits = torch.mean(outputs, dim=1)
        return logits
