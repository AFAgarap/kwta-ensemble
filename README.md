# k-Winners-Take-All Ensemble Neural Network

## Abstract

Ensembling is one approach that improves the performance of a neural network by
combining a group of them. It combines a number of independent neural networks
usually by averaging or summing up their individual outputs. In this work, we
modify the ensemble of independent neural networks by training the sub-networks
concurrently instead of independently. This concurrent training of sub-networks
lead them to cooperate with each other, and we refer to them as "cooperative
ensemble" of networks. Meanwhile, the mixture-of-experts approach improves the
performance of a neural network by dividing up a given dataset to a group of
neural networks; and with a gating network assigning a specialization to each
of its sub-networks called "experts". Instead of the aforementioned ways for
combining a group of neural networks, we propose to use a k-Winners-Take-All
(kWTA) activation function to act as the combination method for the outputs of
each sub-network in the ensemble, which we call "kWTA ensemble neural networks"
(kWTA-ENN). With kWTA, the losing neurons of the sub-networks are inhibited
while the winning neurons are retained, which results to sub-networks having
some form of specialization but also sharing knowledge with one another. We
compare our approach with the cooperative ensemble of networks and
mixture-of-experts, where we used a feed-forward network with one hidden
layer having 100 neurons as the sub-network architecture. Our approach
yielded a better performance than our baseline models, reaching the
following test accuracies on benchmark datasets: 98.34% on MNIST, 88.06% on
Fashion-MNIST, 91.61% on KMNIST, and 95.97% on WDBC.

## Usage

It is recommended to create a virtual environment for this repository.

```buildoutcfg
$ pyenv virtualenv 3.8.8 kwta-ensemble
$ pyenv local kwta-ensemble
```

Then, either `pip` or `poetry` could be used to set up its dependencies.

```
$ pip install -r requirements
$ # or
$ poetry install
```
