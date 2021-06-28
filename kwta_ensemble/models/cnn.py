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
"""Implementation of convolutional neural network"""
from typing import Dict, Tuple

import torch


class CNN(torch.nn.Module):
    """
    A convolutional neural network that optimizes
    softmax cross entropy using a gradient-based method.
    """

    _conv1_params = {"out_channels": 64, "kernel_size": 8, "padding": 1, "stride": 2}
    _conv2_params = {"out_channels": 128, "kernel_size": 6, "padding": 1, "stride": 2}

    def __init__(
        self,
        dim: int = 28,
        input_dim: int = 1,
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Constructs a convolutional neural network classifier.

        Parameters
        ----------
        dim: int
            The depth dimensionality of the image input.
        input_dim: int
            The dimensionality of the input features.
        num_classes: int
            The number of classes in the dataset.
        learning_rate: float
            The learning rate to use for optimization.
        """
        super().__init__()
        conv1_out = self.compute_conv_out(dim, CNN._conv1_params)
        conv2_out = self.compute_conv_out(conv1_out, CNN._conv2_params)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=input_dim,
                out_channels=CNN._conv1_params.get("out_channels"),
                kernel_size=CNN._conv1_params.get("kernel_size"),
                stride=CNN._conv1_params.get("stride"),
                padding=CNN._conv1_params.get("padding"),
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                in_channels=CNN._conv1_params.get("out_channels"),
                out_channels=CNN._conv2_params.get("out_channels"),
                kernel_size=CNN._conv2_params.get("kernel_size"),
                stride=CNN._conv2_params.get("stride"),
                padding=CNN._conv2_params.get("padding"),
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=int(
                    CNN._conv2_params.get("out_channels") * conv2_out * conv2_out
                ),
                out_features=50,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=50, out_features=512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=512, out_features=num_classes),
        )

        for index, layer in enumerate(self.layers):
            if index < (len(self.layers) - 1) and (
                isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d)
            ):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            elif index == (len(self.layers) - 1) and isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loss = []
        self.train_accuracy = []
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass by the model.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        logits: torch.Tensor
            The model output.
        """
        logits = self.layers(features)
        return logits
