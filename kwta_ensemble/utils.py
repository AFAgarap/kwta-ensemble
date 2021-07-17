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
"""Utility functions module"""
import json
from math import ceil, floor
import os
from typing import Dict, List, Tuple

import random

from imblearn.over_sampling import RandomOverSampler
import numpy as np
from pt_datasets import load_dataset, create_dataloader
import torch


def set_global_seed(seed: int) -> None:
    """
    Sets the seed value for random number generators.

    Parameter
    ---------
    seed : int
        The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def create_dataloaders(
    dataset: str, vectorizer: str, ngram_range: Tuple, batch_size: int, seed: int
) -> Dict:
    """
    Splits the training data to training and validation datasets,
    then creates a data loader for each of the dataset split.

    Parameters
    ----------
    dataset: str
        The dataset to load.
    vectorizer: str
        The vectorization method to use.
        Options: [ngrams | tfidf]
    ngram_range: Tuple
        The n-grams range for text vectorization.
    batch_size: int
        The mini-batch size to use.
    seed: int
        The seed to use for reproducibility.

    Returns
    -------
    data_loaders: Dict
        A dictionary that consists of the data loaders
        for each dataset split, together with the
        dataset metadata.
        Access the metadata through the `meta` key.
    """
    train_data, test_data = load_dataset(
        name=dataset, normalize=False, vectorizer=vectorizer, ngram_range=ngram_range
    )

    if dataset == "wdbc":
        over_sampler = RandomOverSampler(random_state=seed)
        (train_data.data, train_data.targets) = over_sampler.fit_resample(
            train_data.data, train_data.targets
        )
        (test_data.data, test_data.targets) = over_sampler.fit_resample(
            test_data.data, test_data.targets
        )

    num_features = np.prod(train_data.data.shape[1:])
    if dataset != "svhn":
        input_shape = train_data.data.shape
        num_classes = len(train_data.classes)
    else:
        input_shape = train_data.data.transpose(0, 2, 3, 1).shape
        num_classes = len(np.unique(train_data.labels))

    train_data, valid_data = torch.utils.data.random_split(
        train_data,
        [ceil(len(train_data) * 0.90), floor(len(train_data) * 0.10)],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = create_dataloader(train_data, batch_size=batch_size, num_workers=4)
    valid_loader = create_dataloader(valid_data, batch_size=batch_size, num_workers=4)
    test_loader = create_dataloader(test_data, batch_size=len(test_data))
    data_loaders = {
        "meta": {
            "num_features": num_features,
            "input_shape": input_shape,
            "num_classes": num_classes,
        },
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
    }
    return data_loaders


def get_moe_filename(
    num_subnetwork: int,
    expert_gating_architecture: str,
    dataset: str,
    learning_rate: float,
    optimizer: str,
    batch_size: int,
) -> str:
    filename = f"moe-{num_subnetwork}-{expert_gating_architecture}"
    filename = f"{filename}-{dataset}"
    filename = f"{filename}-{learning_rate}-lr-opt-{optimizer}-{batch_size}-bsize"
    return filename


def get_ensemble_filename(
    num_subnetwork: int,
    subnetwork_architecture: str,
    dataset: str,
    learning_rate: float,
    optimizer: str,
    batch_size: int,
) -> str:
    filename = f"ensemble-{num_subnetwork}-{subnetwork_architecture}"
    filename = f"{filename}-{dataset}"
    filename = f"{filename}-{learning_rate}-lr-opt-{optimizer}-{batch_size}-bsize"
    return filename


def get_kwta_enn_filename(
    num_subnetwork: int,
    subnetwork_architecture: str,
    dataset: str,
    learning_rate: float,
    optimizer: str,
    batch_size: int,
    competition_delay: int,
    sparsity_factor: float,
) -> str:
    filename = f"ensemble-{num_subnetwork}-{subnetwork_architecture}"
    filename = f"{filename}-{dataset}"
    filename = f"{filename}-cd-{competition_delay}-s-{sparsity_factor}"
    filename = f"{filename}-{learning_rate}-lr-opt-{optimizer}-{batch_size}-bsize"
    return filename


def export_results(model_results: Dict, filename: str) -> None:
    """
    Exports the training results stored in model class to a JSON file.

    Parameters
    ----------
    model: torch.nn.Module
        The trained model object.
    accuracy: Dict
        The dictionary that consists of the
        model accuracy per seed, and the
        average, max, and std accuracy over all seeds.
    filename: str
        The filename of the JSON file to write.
    """
    dataset_name = filename.split("-")[3]
    results_path = os.path.join("outputs", "results", dataset_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    filename = f"{filename}.json"
    filename = os.path.join(results_path, filename)
    results = dict()
    for key, value in model_results.items():
        results[key] = value
    with open(filename, "w") as file:
        json.dump(results, file)


def compute_learner_accuracy(outputs: List, labels: torch.Tensor) -> List:
    """
    Computes the accuracy of each output in `outputs`.

    Parameters
    ----------
    outputs: List
        The list of model outputs.
    labels: torch.Tensor
        The target outputs tensor.

    Returns
    -------
    accuracies: List
        The list of accuracy of each output in `outputs`.
    """
    accuracies = [
        (output.argmax(1) == labels).sum().item() / len(labels) for output in outputs
    ]
    return accuracies
