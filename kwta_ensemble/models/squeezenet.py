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
"""Implementation of SqueezeNet"""
import torch
import torchvision


class SqueezeNet(torch.nn.Module):
    """
    Pre-trained SqueezeNet model
    """

    _supported_arch = ("1.0", "1.1")

    def __init__(
        self,
        num_classes: int = 10,
        arch: str = "1.1",
        learning_rate: float = 1e-3,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Loads a pretrained SqueezeNet classifier.

        Parameters
        ----------
        num_classes: int
            The number of classes in the dataset.
        arch: str
            The variant to use.
        learning_rate: float
            The learning rate to use for optimization.
        device: torch.device
            The device to use for computations.
        """
        super().__init__()
        assert arch in SqueezeNet._supported_arch
        if arch == "1.0":
            self.model = torchvision.models.squeezenet1_0(pretrained=True)
        elif arch == "1.1":
            self.model = torchvision.models.squeezenet1_1(pretrained=True)
        self.freeze_model()
        self.model.classifier[0] = torch.nn.BatchNorm2d(512)
        self.model.classifier[1] = torch.nn.Conv2d(
            in_channels=self.model.classifier[1].in_channels,
            out_channels=num_classes,
            kernel_size=self.model.classifier[1].kernel_size,
            stride=self.model.classifier[1].stride,
        )
        self.device = device
        self.model.to(self.device)

    def freeze_model(self):
        """
        Freezes all the parameters of the pre-trained model
        except for the last three layers of the feature extractor
        and the classification layer.
        """
        for parameters in self.model.parameters():
            parameters.requires_grad = False
        for parameters in self.model.features[-3:-1].parameters():
            parameters.requires_grad = True
        for parameters in self.model.classifier.parameters():
            parameters.requires_grad = True

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
        return self.model.forward(features)
