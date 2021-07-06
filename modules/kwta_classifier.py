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
"""kWTA Ensemble classifier"""
import argparse

from kwta_ensemble.models import CNN, DNN, kWTAEnsemble, LeNet
from kwta_ensemble.utils import create_dataloaders, set_global_seed


def main(arguments: argparse.Namespace):
    (
        seeds,
        dataset,
        vectorizer,
        ngram_range,
        batch_size,
        optimizer,
        learning_rate,
        weight_decay,
        epochs,
        num_subnetworks,
        subnetwork_architecture,
        show_every,
        competition_delay,
        sparsity_factor,
    ) = (
        arguments.seeds,
        arguments.dataset,
        arguments.vectorizer,
        arguments.ngram_range,
        arguments.batch_size,
        arguments.optimizer,
        arguments.learning_rate,
        arguments.weight_decay,
        arguments.epochs,
        arguments.num_subnetworks,
        arguments.subnetwork_architecture,
        arguments.show_every,
        arguments.use_competition_after,
        arguments.sparsity_factor,
    )
    results = dict()
    for num_subnetwork in range(2, num_subnetworks + 1):
        accuracies = []
        for seed in seeds:
            print()
            print(f"[INFO] Dataset: {dataset}")
            print(f"[INFO] Number of learners: {num_subnetwork}")
            print(f"[INFO] Seed: {seed}")

            set_global_seed(seed=seed)

            data_loaders = create_dataloaders(
                dataset=dataset,
                vectorizer=vectorizer,
                ngram_range=ngram_range,
                batch_size=batch_size,
                seed=seed,
            )

            train_loader = data_loaders.get("train")
            valid_loader = data_loaders.get("valid")
            test_loader = data_loaders.get("test")
            num_features = data_loaders.get("meta").get("num_features")
            input_shape = data_loaders.get("meta").get("input_shape")
            num_classes = data_loaders.get("meta").get("num_classes")

            if subnetwork_architecture == "dnn":
                subnetwork = DNN(units=((num_features, 100), (100, num_classes)))
            elif subnetwork_architecture == "cnn":
                subnetwork = CNN(
                    dim=input_shape[1],
                    input_dim=(1 if len(input_shape) < 4 else input_shape[3]),
                    num_classes=num_classes,
                )
            elif subnetwork_architecture == "lenet":
                subnetwork = LeNet(
                    dim=input_shape[1],
                    channel_dim=(1 if len(input_shape) < 4 else input_shape[3]),
                    num_classes=num_classes,
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="kWTA Ensemble Neural Network classifier"
    )
    group = parser.add_argument_group("Parameters")
    group.add_argument(
        "-s",
        "--seeds",
        nargs="+",
        type=int,
        required=False,
        default=[1234, 42, 73],
        help="the list of random seeds to use, default: [1234, 42, 73]",
    )
    group.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="mnist",
        help="the dataset to use, default: [mnist]",
    )
    group.add_argument(
        "-v",
        "--vectorizer",
        type=str,
        default="ngrams",
        help="the vectorizer to use, options: [ngrams (default) | tfidf]",
    )
    group.add_argument(
        "-nr",
        "--ngram_range",
        nargs="+",
        required=False,
        default=(1, 5),
        type=int,
        help="the n-grams range to use for text vectorization, default: [(1, 5)]",
    )
    group.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=256,
        help="the mini-batch size to use, default: [256]",
    )
    group.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="the optimization algorithm to use, options: [sgd (default) | adamw]",
    )
    group.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=3e-4,
        help="the learning rate to use for optimization, default: [3e-4]",
    )
    group.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=1e-5,
        help="the weight decay to use during optimization, default: [1e-5]",
    )
    group.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="the number of epochs to train the model, default: [10]",
    )
    group.add_argument(
        "-se",
        "--show_every",
        type=int,
        default=1,
        help="the number of interval between training progress displays, default: [1]",
    )
    group.add_argument(
        "-nl",
        "--num_subnetworks",
        type=int,
        default=3,
        help="the number of subnetworks to instantiate, default: [3]",
    )
    group.add_argument(
        "--subnetwork_architecture",
        type=str,
        default="dnn",
        help="the architecture to use for an expert, options: [cnn | dnn (default) | lenet]",
    )
    group.add_argument(
        "-cd",
        "--use_competition_after",
        type=int,
        default=0,
        help="the number of epochs to reach before using kWTA, default: [0]",
    )
    group.add_argument(
        "-sf",
        "--sparsity_factor",
        type=float,
        default=0.75,
        help="the percentage of winners to get, default: [0.75]",
    )
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
