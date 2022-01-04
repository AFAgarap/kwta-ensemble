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
"""Implementation of kWTA-ENN"""
from copy import deepcopy
from typing import Dict, Tuple, Union

import torch

from kwta_ensemble.layers import WinnersTakeAllLayer
from kwta_ensemble.models.base import Model


class kWTAEnsemble(Model):
    def __init__(
        self,
        num_classes: int,
        expert_model: torch.nn.Module,
        feature_extractor: Union[torch.nn.Module, torch.nn.Sequential],
        use_feature_extractor: bool = False,
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
        """
        Constructs a kWTA ensemble of neural networks.

        Parameters
        ----------
        num_classes: int
            The number of classes to train on.
        expert_model: torch.nn.Module
            The architecture to use in ensemble.
        num_subnetworks: int
            The number of networks to use in ensemble.
        sparsity: float
            The percentage of winners to get
            from the competition phase.
            Defalt: 0.75
        competition_delay: int
            The number of epochs to skip before using
            the competitive layer.
            Default: 3
        optimizer: str
            The optimizer to use.
            Options: [sgd (default) | adamw]
        learning_rate: float
            The learning rate to use for optimization.
            Default: 1e-3
        weight_decay: float
            The weight decay parameter to use.
            Default: 1e-5
        device: torch.device
            The device to use for computation.
            Default: cuda:0
        """
        super().__init__(
            num_subnetworks=num_subnetworks,
            use_feature_extractor=use_feature_extractor,
            feature_extractor=feature_extractor,
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

        self.to(self.device)

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
        """
        The forward pass by the model.

        Parameters
        ----------
        features: torch.Tensor
            The input features.
        epoch: int
            The current training epoch.

        Returns
        -------
        outputs: torch.Tensor
            The model outputs.
        """
        if self.use_feature_extractor:
            if len(features.shape) > 2:
                features = features.view(features.shape[0], -1)
            # features = self.feature_extractor[:-1](features)
            features = self.feature_extractor(features)
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
