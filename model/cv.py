"""
TODO add to the main.
"""

import torch
from torch.utils.data import random_split
from .train import train_model, train_rnn, test_model, test_rnn

def k_fold_cross_validation(dataset, k=5):
    fold_size = len(dataset) // k
    remaining_size = len(dataset) % k
    sizes = [fold_size + (1 if i < remaining_size else 0) for i in range(k)]
    folds = random_split(dataset, sizes)
    cross_validation_sets = []
    for i in range(k):
        validation_set = folds[i]
        training_set = [folds[j] for j in range(k) if j != i]
        training_set = torch.utils.data.ConcatDataset(training_set)
        cross_validation_sets.append((training_set, validation_set))
    return cross_validation_sets

def cross_validation_training(dataset, model_type="transformer", **kwargs):
    cross_validation_sets = k_fold_cross_validation(dataset)
    train_function = train_model if model_type == "transformer" else train_rnn
    test_function = test_model if model_type == "transformer" else test_rnn
    results = []
    for i, (train_set, valid_set) in enumerate(cross_validation_sets):
        print(f"Training on fold {i + 1}...")
        train_function(train_set, **kwargs)
        validation_result = test_function(valid_set)
        fold_result = {'validation_result': validation_result}
        results.append(fold_result)
    return results
