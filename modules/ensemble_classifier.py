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
"""Ensemble classifier"""
import argparse

from kwta_ensemble.utils import create_dataloaders, set_global_seed


def main(arguments):
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
        num_learners,
        learner_architecture,
        show_every,
        use_feature_extractor,
        feature_extractor_arch,
        code_dim,
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
        arguments.num_learners,
        arguments.learner_architecture,
        arguments.show_every,
        arguments.use_feature_extractor,
        arguments.feature_extractor_arch,
        arguments.code_dim,
    )

    results = dict()
    for num_learner in range(2, num_learners + 1):
        accuracies = []
        for seed in seeds:
            print()
            print(f"[INFO] Dataset: {dataset}")
            print(f"[INFO] Number of learners: {num_learner}")
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


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Neural Network classifier")
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
        "-nl",
        "--num_learners",
        type=int,
        default=3,
        help="the number of learners to instantiate, default: [3]",
    )
    group.add_argument(
        "--learner_architecture",
        type=str,
        default="dnn",
        help="the architecture to use for an expert, options: [cnn | dnn (default) | lenet]",
    )
    group.add_argument(
        "-se",
        "--show_every",
        type=int,
        default=1,
        help="the number of interval between training progress displays, default: [1]",
    )
    group.add_argument(
        "-x",
        "--use_feature_extractor",
        required=False,
        dest="use_feature_extractor",
        action="store_true",
    )
    group.add_argument(
        "--feature_extractor_arch",
        required=False,
        default="dnn",
        type=str,
        help="the architecture to use for feature extractor, options: [cnn | dnn (default) | resnet18]",
    )
    group.add_argument(
        "-c",
        "--code_dim",
        required=False,
        type=int,
        default=70,
        help="the dimensionality of the learned representation, default: [70]",
    )
    group.set_defaults(use_feature_extractor=False)
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
