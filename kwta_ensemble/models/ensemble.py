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
"""Implementation of ensemble neural network"""
from copy import deepcopy

import torch


class Ensemble(torch.nn.Module):
    def __init__(
        self,
        network: torch.nn.Module,
        num_learners: int = 3,
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
        num_learners: int
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
            optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay
        )
        self.model = torch.nn.Sequential()
        for index in range(num_learners):
            network = deepcopy(network)
            network.apply(self.reset_parameters)
            self.model.add_module(f"network_{index}", network)
        self.device = device
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
