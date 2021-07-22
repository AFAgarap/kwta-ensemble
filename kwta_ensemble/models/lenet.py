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
"""Implementation of LeNet"""
from typing import Dict

import torch


class LeNet(torch.nn.Module):
    """
    LeNet CNN architecture that optimizes
    softmax cross entropy using a gradient-based method.
    """

    _conv1_params = {"out_channels": 6, "kernel_size": 5, "padding": 2, "stride": 1}
    _conv2_params = {"out_channels": 16, "kernel_size": 5, "padding": 0, "stride": 1}

    def __init__(
        self,
        dim: int = 28,
        channel_dim: int = 1,
        num_classes: int = 10,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Constructs a LeNet classifier.

        Parameters
        ----------
        dim: int
            The depth dimensionality of the image input.
        channel_dim: int
            The dimensionality of the input features.
        num_classes: int
            The number of classes in the dataset.
        device: torch.device
            The device to use for model computations.
        """
        super().__init__()
        conv1_out = self.compute_conv_out(dim, LeNet._conv1_params)
        max1_out = self.compute_max_out(conv1_out)
        conv2_out = self.compute_conv_out(max1_out, LeNet._conv2_params)
        max2_out = self.compute_max_out(conv2_out)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dim,
                out_channels=LeNet._conv1_params.get("out_channels"),
                kernel_size=LeNet._conv1_params.get("kernel_size"),
                stride=LeNet._conv1_params.get("stride"),
                padding=LeNet._conv1_params.get("padding"),
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            torch.nn.Conv2d(
                in_channels=LeNet._conv1_params.get("out_channels"),
                out_channels=LeNet._conv2_params.get("out_channels"),
                kernel_size=LeNet._conv2_params.get("kernel_size"),
                stride=LeNet._conv2_params.get("stride"),
                padding=LeNet._conv2_params.get("padding"),
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=int(
                    LeNet._conv2_params.get("out_channels") * max2_out * max2_out
                ),
                out_features=120,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=84, out_features=num_classes),
        )
        self.device = device
        self.to(self.device)

    @staticmethod
    def compute_conv_out(dim: int, params: Dict) -> int:
        """
        Computes the convolutional layer output size.

        Parameters
        ----------
        dim: int
            The dimensionality of the input to the convolutional layer.
        params: Dict
            The parameters of the convolutional layer.

        Returns
        -------
        int
            The output size of the convolutional layer.
        """
        return (
            dim - params.get("kernel_size") + 2 * params.get("padding")
        ) / params.get("stride") + 1

    @staticmethod
    def compute_max_out(dim: int) -> int:
        """
        Computes the max pooling layer output size.

        Parameters
        ----------
        dim: int
            The dimensionality of the input to the max pooling layer.

        Returns
        -------
        int
            The output size of the max pooling layer.
        """
        return int(dim - 2) / 2 + 1

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
        return self.layers(features)
