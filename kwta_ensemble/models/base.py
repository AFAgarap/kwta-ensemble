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
from typing import Dict, Tuple

import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

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
