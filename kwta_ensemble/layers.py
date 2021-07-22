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
"""Implementation of k-Winners-Take-All layer"""
import torch


class WinnersTakeAllLayer(torch.nn.Module):
    """
    The k-Winners-Take-All layer.
    """

    def __init__(self, sparsity: float = 3e-1):
        """
        Builds the k-Winners-Take-All (kWTA) layer.

        Parameter
        ---------
        sparsity: float
            The percentage of winners to take
            from the competition.
        """
        super().__init__()
        self.sparsity = sparsity

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of kWTA layer.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        activations: torch.Tensor
            The kWTA units.
        """
        top_k = int(self.sparsity * features.shape[1])
        winners = features.topk(top_k, dim=1)[0][:, -1]
        winners = winners.expand(features.shape[1], features.shape[0]).permute(1, 0)
        activations = (features >= winners).to(features)
        activations *= features
        return activations
