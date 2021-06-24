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
"""Implementation of ResNet"""
from typing import Tuple

import torch
import torchvision


class ResNet18(torch.nn.Module):
    """
    Pre-trained ResNet18 model.
    """

    def __init__(
        self,
        input_shape: Tuple,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        blocks_to_freeze: int = 3,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Loads a pretrained ResNet18 classifier.

        Parameters
        ----------
        num_classes: int
            The number of classes in the dataset.
        learning_rate: float
            The learning rate to use for optimization.
        device: torch.device
            The device to use for computations.
        """
        super().__init__()
        self.resnet = torchvision.models.resnet.resnet18(pretrained=True)
        if len(input_shape) < 4:
            self.resnet.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.freeze_blocks(num_blocks=blocks_to_freeze)
        self.resnet.fc = torch.nn.Linear(
            in_features=self.resnet.fc.in_features, out_features=num_classes
        )
        self.optimizer = torch.optim.SGD(
            params=(
                filter(lambda parameters: parameters.requires_grad, self.parameters())
            ),
            momentum=9e-1,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device
        self.resnet.to(self.device)

    def freeze_blocks(self, num_blocks: int = 3) -> None:
        """
        Freezes parameters of specified ResNet blocks.

        Parameter
        ---------
        num_blocks: int
            The number of blocks to freeze.
        """
        assert num_blocks <= 4, "There are only 4 blocks in ResNet."
        for block_index, child in enumerate(list(self.resnet.named_children())[4:8]):
            if block_index < num_blocks:
                for param in child[1].parameters():
                    param.requires_grad = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass by the model.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        torch.Tensor
            The model output.
        """
        return self.resnet.forward(features)
