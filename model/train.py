import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torch.nn as nn
import numpy as np
from .transformer import TransformerModel
from .gru import RNNModel
from .tempscalewrapper import TempScaledModel
from .result_metrics import ECEComputer
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import random

# def train_model(dataset, task="regression", target_index=0, epochs=10, learning_rate=0.001, batch_size=32):
#     """
#     target_index is the index of the target variable column in the y tensor.
#     """
#     # Check if CUDA is available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Determine the output dimension and loss function based on the task
#     if task == "regression":
#         output_dim = 1
#         loss_function = nn.MSELoss()
#     elif task == "classification":
#         output_dim = 2
#         loss_function = nn.NLLLoss()
#     else:
#         raise ValueError("Invalid task specified. Choose 'regression' or 'classification'.")
    
#     # Create DataLoader
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     # Determine input dimensions from a sample
#     sample_X, sample_s, _ = dataset[0]
#     s_dim = sample_s.shape[0]
#     X_seq_len, X_feat_dim = sample_X.shape
    
#     # Hyperparameters for the model
#     n_heads = 4 
#     hidden_dim = 64
    
#     # Create the model
#     model = TransformerModel(s_dim, X_seq_len, X_feat_dim, n_heads, hidden_dim, output_dim)
#     model.to(device) # Move model to GPU if available

#     # Optimizer
#     optimizer = Adam(model.parameters(), lr=learning_rate)
    
#     # Training loop
#     for epoch in range(epochs):
#         for batch_X, batch_s, batch_y in dataloader:
#             # Move data to GPU if available
#             batch_X, batch_s, batch_y = batch_X.to(device), batch_s.to(device), batch_y.to(device)

#             # Determine the target based on the task
#             target = batch_y[:, target_index] if task == "regression" else batch_y[:, target_index].long()
            
#             # Forward pass
#             predictions = model(batch_X, batch_s).squeeze()
#             if task == "classification":
#                 predictions = F.log_softmax(predictions, dim=1)
#             # Compute loss
#             loss = loss_function(predictions, target)
            
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
            
#             # Update weights
#             optimizer.step()
        
#         print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item()}")
    
#     return model

# def test_model(model, test_dataloader, task="classification", target_index=0):
#     # Check if CUDA is available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device) # Move model to GPU if available
#     model.eval()  # Set the model to evaluation mode

#     # Determine the loss function based on the task
#     if task == "regression":
#         loss_function = nn.MSELoss()
#     elif task == "classification":
#         loss_function = nn.NLLLoss()
#     else:
#         raise ValueError("Invalid task specified. Choose 'regression' or 'classification'.")

#     # Initialize variables to store test loss and accuracy (for classification)
#     test_loss = 0.0
#     correct_predictions = 0

#     # For AUROC
#     all_targets = []
#     all_predictions = []

#     # Iterate through the test DataLoader
#     with torch.no_grad():
#         for batch_X, batch_s, batch_y in test_dataloader:
#             # Move data to GPU if available
#             batch_X, batch_s, batch_y = batch_X.to(device), batch_s.to(device), batch_y.to(device)

#             # Determine the target based on the task
#             target = batch_y[:, target_index] if task == "regression" else batch_y[:, target_index].long()

#             # Forward pass
#             predictions = model(batch_X, batch_s).squeeze()
#             if task == "classification":
#                 predictions = F.log_softmax(predictions, dim=1)
#                 all_targets.extend(target.cpu().numpy())
#                 all_predictions.extend(torch.exp(predictions[:, 1]).cpu().numpy())  # Probabilities for the positive class

#             # Compute loss
#             loss = loss_function(predictions, target)
#             test_loss += loss.item()

#             # Compute accuracy for classification task
#             if task == "classification":
#                 _, predicted_labels = torch.max(predictions, 1)
#                 correct_predictions += (predicted_labels == target).sum().item()

#     # Compute average test loss
#     average_test_loss = test_loss / len(test_dataloader)

#     print(f"Test Loss: {average_test_loss}")

#     accuracy = -1.
#     auroc = -1.

#     # Print accuracy for classification task
#     if task == "classification":
#         accuracy = correct_predictions / len(test_dataloader.dataset)
#         print(f"Test Accuracy: {accuracy * 100}%")
#         auroc = roc_auc_score(all_targets, all_predictions)
#         print(f"Test AUROC: {auroc}")

#     return torch.tensor([average_test_loss, accuracy, auroc])



# def train_rnn(dataset, task="regression", target_index=0, model_type='GRU-dt', epochs=10, learning_rate=0.001, batch_size=32):
#     # Check if CUDA is available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Determine the output dimension and loss function based on the task
#     if task == "regression":
#         output_dim = 1
#         loss_function = nn.MSELoss()
#     elif task == "classification":
#         output_dim = 2
#         loss_function = nn.NLLLoss()
#     else:
#         raise ValueError("Invalid task specified. Choose 'regression' or 'classification'.")
    
#     # Create DataLoader
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     # Determine input dimensions from a sample
#     sample_X, _, _, sample_s, _ = dataset[0]
#     s_dim = sample_s.shape[0]
#     _, X_feat_dim = sample_X.shape
    
#     # Hyperparameters for the model
#     # n_heads = 4 
#     hidden_dim = 64
    
#     # Create the model
#     # model = TransformerModel(s_dim, X_seq_len, X_feat_dim, n_heads, hidden_dim, output_dim)
#     model = RNNModel(s_dim, X_feat_dim, hidden_dim, output_dim, model_type)
#     model.to(device) # Move model to GPU if available

#     # Optimizer
#     optimizer = Adam(model.parameters(), lr=learning_rate)
    
#     # Training loop
#     for epoch in range(epochs):
#         for batch_X, batch_M, batch_t, batch_s, batch_y in dataloader:
#             # Move data to GPU if available
#             # batch_X, batch_s, batch_y = batch_X.to(device), batch_s.to(device), batch_y.to(device)
#             batch_X, batch_M, batch_t, batch_s, batch_y = batch_X.to(device), batch_M.to(device), batch_t.to(device), batch_s.to(device), batch_y.to(device)

#             # Determine the target based on the task
#             target = batch_y[:, target_index] if task == "regression" else batch_y[:, target_index].long()

            
#             # Forward pass
#             predictions = model(batch_X, batch_M, batch_t, batch_s).squeeze()
#             if task == "classification":
#                 predictions = F.log_softmax(predictions, dim=1)
#             # Compute loss
#             loss = loss_function(predictions, target)
            
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
            
#             # Update weights
#             optimizer.step()
        
#         print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item()}")
    
#     return model

# def test_rnn(model, test_dataloader, task="classification", target_index=0):
#     # Check if CUDA is available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device) # Move model to GPU if available
#     model.eval()  # Set the model to evaluation mode

#     # Determine the loss function based on the task
#     if task == "regression":
#         loss_function = nn.MSELoss()
#     elif task == "classification":
#         loss_function = nn.NLLLoss()
#     else:
#         raise ValueError("Invalid task specified. Choose 'regression' or 'classification'.")

#     # Initialize variables to store test loss and accuracy (for classification)
#     test_loss = 0.0
#     correct_predictions = 0

#     # For AUROC
#     all_targets = []
#     all_predictions = []

#     # Iterate through the test DataLoader
#     with torch.no_grad():
#         for batch_X, batch_M, batch_t, batch_s, batch_y in test_dataloader:
#             # Move data to GPU if available
#             batch_X, batch_M, batch_t, batch_s, batch_y = batch_X.to(device), batch_M.to(device), batch_t.to(device), batch_s.to(device), batch_y.to(device)

#             # Determine the target based on the task
#             target = batch_y[:, target_index] if task == "regression" else batch_y[:, target_index].long()

#             # Forward pass
#             predictions = model(batch_X, batch_M, batch_t, batch_s).squeeze()
#             if task == "classification":
#                 predictions = F.log_softmax(predictions, dim=1)
#                 all_targets.extend(target.cpu().numpy())
#                 all_predictions.extend(torch.exp(predictions[:, 1]).cpu().numpy())  # Probabilities for the positive class

#             # Compute loss
#             loss = loss_function(predictions, target)
#             test_loss += loss.item()

#             # Compute accuracy for classification task
#             if task == "classification":
#                 _, predicted_labels = torch.max(predictions, 1)
#                 correct_predictions += (predicted_labels == target).sum().item()

#     # Compute average test loss
#     average_test_loss = test_loss / len(test_dataloader)

#     print(f"Test Loss: {average_test_loss}")

#     accuracy = -1.
#     auroc = -1.

#     # Print accuracy for classification task
#     if task == "classification":
#         accuracy = correct_predictions / len(test_dataloader.dataset)
#         print(f"Test Accuracy: {accuracy * 100}%")
#         auroc = roc_auc_score(all_targets, all_predictions)
#         print(f"Test AUROC: {auroc}")

#     return torch.tensor([average_test_loss, accuracy, auroc])

def get_mean_std(dataset):
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    sample_X, _, _, sample_s, _ = next(iter(dataloader))
    mean_dyn = sample_X.mean(dim=0)
    std_dyn = sample_X.std(dim=0)
    mean_stat = sample_s.mean(dim=0)
    std_stat = sample_s.std(dim=0)
    std_dyn[std_dyn == 0] = 1
    std_stat[std_stat == 0] = 1
    return mean_dyn, std_dyn, mean_stat, std_stat

def train_rnn_yale(dataset, task, target_index, model_type, rnn_type, epochs, learning_rate, batch_size, output_dim, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), **kwargs):
    # Check if CUDA is available

    loss_function = get_loss_fn(task)

    mean_dyn, std_dyn, mean_stat, std_stat = get_mean_std(dataset)
    mean_dyn, std_dyn, mean_stat, std_stat = mean_dyn.to(device), std_dyn.to(device), mean_stat.to(device), std_stat.to(device)
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Determine input dimensions from a sample
    sample_X, _, _, sample_s, _ = dataset[0]
    s_dim = sample_s.shape[0]
    _, X_feat_dim = sample_X.shape
    
    # Hyperparameters for the model
    # n_heads = 4 
    hidden_dim = 64
    
    # Create the model
    # model = TransformerModel(s_dim, X_seq_len, X_feat_dim, n_heads, hidden_dim, output_dim)
    model = RNNModel(s_dim, X_feat_dim, hidden_dim, output_dim, model_type, rnn_type, mean_dyn=mean_dyn, std_dyn=std_dyn, mean_stat=mean_stat, std_stat=std_stat)
    model.to(device) # Move model to GPU if available

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        for batch_X, batch_M, batch_t, batch_s, batch_y in dataloader:
            # Move data to GPU if available
            # batch_X, batch_s, batch_y = batch_X.to(device), batch_s.to(device), batch_y.to(device)
            if len(batch_y.shape) == 1:
                batch_y = batch_y.reshape(-1, 1)
            batch_X, batch_M, batch_t, batch_s, batch_y = batch_X.to(device), batch_M.to(device), batch_t.to(device), batch_s.to(device), batch_y.to(device)

            # Determine the target based on the task
            # target_indices = np.arange(target_index, target_index + output_dim)
            target = batch_y[:, target_index] if task == "regression" else batch_y[:, target_index].long() # just one column for classes.

            # import pdb; pdb.set_trace()
            # Forward pass
            predictions = model(batch_X, batch_M, batch_t, batch_s).squeeze()
            # if task == "classification":
            #     predictions = F.log_softmax(predictions, dim=1)
            # Compute loss
            loss = loss_function(predictions, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update weights
            optimizer.step()
        
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item()}")
    
    return model

def crps_loss(predictions, target):
    """
    Computes the Continuous Ranked Probability Score (CRPS) loss using broadcasting and F.mse_loss.
    [WARNING] the target must be an integer tensor with values in the range [0, num_classes - 1].
    
    Parameters:
    - predictions (torch.Tensor): The predicted probabilities for each class. Shape: (batch_size, num_classes)
    - target (torch.Tensor): The true class labels. Shape: (batch_size,)
    
    Returns:
    - torch.Tensor: The CRPS loss.
    """
    
    # Ensure the predictions are in the range [0, 1] by applying the softmax function
    predictions = F.softmax(predictions, dim=1)
    
    # Compute the cumulative probabilities for the predictions
    pred_cum_probs = predictions.cumsum(dim=1)
    
    # Construct the true cumulative probabilities using broadcasting
    true_cum_probs = (target.unsqueeze(1) <= torch.arange(predictions.size(1), device=predictions.device)).float()
    
    # Compute the mean squared error between the predicted and true cumulative probabilities
    loss = F.mse_loss(pred_cum_probs, true_cum_probs)
    
    return loss

#############
def test_rnn_yale(model, dataset, task, target_index, batch_size, calibration, n_bins_ece=10,
                  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), **kwargs):
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()

    loss_function = get_loss_fn(task)
    test_loss = 0.0
    correct_predictions = 0

    all_targets = []
    all_preds_ = []

    if calibration:
        ece_computer = ECEComputer(device=device, n_bins=n_bins_ece)

    with torch.no_grad():
        for batch_X, batch_M, batch_t, batch_s, batch_y in test_dataloader:
            if len(batch_y.shape) == 1:
                batch_y = batch_y.reshape(-1, 1)
            batch_X, batch_M, batch_t, batch_s, batch_y = batch_X.to(device), batch_M.to(device), batch_t.to(device), batch_s.to(device), batch_y.to(device)

            target = batch_y[:, target_index] if task == "regression" else batch_y[:, target_index].long()
            predictions = model(batch_X, batch_M, batch_t, batch_s).squeeze()
            model.apply_softmax = False
            loss = loss_function(predictions, target)
            test_loss += loss.item()
            if task in ["classification", "ordinal"]:
                predictions = F.softmax(predictions, dim=1)
                all_targets.extend(target.cpu().numpy())
                this_pred = predictions.detach().cpu().numpy()
                all_preds_.extend(this_pred)
                _, predicted_labels = torch.max(predictions, 1)
                correct_predictions += (predicted_labels == target).sum().item()

                if calibration:
                    ece_computer.update(predictions, target)

    average_test_loss = test_loss / len(test_dataloader)
    print(f"Test Loss: {average_test_loss}")

    accuracy = -1.
    auroc = -1.
    ece_value = -1.

    
    """
    commenting this because not needed.
    """
    # if task in ["classification", "ordinal"]:
    #     accuracy = correct_predictions / len(dataset)
    #     print(f"Test Accuracy: {accuracy * 100}%")
    #     auroc = roc_auc_score(all_targets, all_preds_, multi_class="ovr", average="macro")
    #     print(f"Test AUROC: {auroc}")

    #     if calibration:
    #         ece_value = ece_computer.compute()
    #         print(f"Test ECE: {ece_value}")

    return torch.tensor([average_test_loss, accuracy, auroc, ece_value])

#############

def get_loss_fn(task):
     # Determine the loss function based on the task
    if task == "regression":
        loss_function = nn.MSELoss()
    elif task == "classification":
        loss_function = nn.CrossEntropyLoss()
    elif task == "ordinal":
        loss_function = crps_loss
    else:
        raise ValueError("Invalid task specified. Choose 'regression' or 'classification' or 'ordinal'.")
    return loss_function

def calibrate(model, calibration_dataset, calibration_epochs, calibration_lr, batch_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), **kwargs):
    """
    Calibrate the temperature of the model using the given calibration dataset.

    Args:
    - model: The wrapped model with temperature scaling.
    - calibration_dataset: The dataset to be used for calibration.
    - calibration_epochs: Number of calibration_epochs for calibration.
    - calibration_lr: Learning rate for temperature optimization.
    - batch_size: Batch size for calibration.
    - device: Device to use ('cuda' or 'cpu').

    Returns:
    - The calibrated model.
    """

    # Set the temperature parameter to be learnable
    model.enable_temperature_grad()

    # Define the loss function and optimizer
    # loss_function = nn.NLLLoss()  # Assuming classification
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam([model.temperature], lr=calibration_lr)  # Only calibrate the temperature

    # Create DataLoader
    dataloader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=True)

    # Calibration loop
    for epoch in range(calibration_epochs):
        for batch_X, batch_M, batch_t, batch_s, batch_y in dataloader:
            # Move data to GPU if available
            batch_X, batch_M, batch_t, batch_s, batch_y = batch_X.to(device), batch_M.to(device), batch_t.to(device), batch_s.to(device), batch_y.to(device)

            # Extract target labels (assuming they're in the last column)
            target = batch_y[:, -1].long()
            
            # Forward pass
            predictions = model(batch_X, batch_M, batch_t, batch_s)
            # log_predictions = torch.log(predictions)
            # Compute loss
            # loss = loss_function(log_predictions, target)
            loss = loss_function(predictions, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update the temperature
            optimizer.step()

        print(f"Calibration Epoch {epoch + 1}/{calibration_epochs} - Loss: {loss.item()}")

    # Set the temperature parameter to be non-learnable after calibration
    model.enable_base_model_grad()

    return model

def cross_validate_rnn_yale(subsets, calibration, calibration_pct, seed=None, **train_args):
    """
    Perform k-fold cross-validation on the given dataset.
    
    Parameters:
    - dataset: The full dataset to be split into k folds.
    - k: The number of folds for cross-validation.
    - calibration: Whether to perform temperature scaling calibration.
    - calibration_pct: Percentage of the training data to be used for calibration.
    - **train_args: Arguments to be passed to the train_rnn_yale and test_rnn_yale functions.
    
    Returns:
    - A list of test metrics for each fold.
    """

    # if seed is not None:
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     # If you are using CUDA, also set the following
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    # fold_size = len(dataset) // k
    metrics = []
    train_datasets = []
    calib_datasets = []  # Calibration datasets
    test_datasets = []
    models = []
    
    # Split dataset into k parts
    # subsets = random_split(dataset, [fold_size] * (k-1) + [len(dataset) - fold_size * (k-1)])
    k = len(subsets)
    for i in range(k):
        print(f"Fold {i+1}/{k}")
        
        # Create test datasets for the current fold
        test_subset = subsets[i]
        train_subsets = [subsets[j] for j in range(k) if j != i]
        full_train_subset = ConcatDataset(train_subsets)
        
        # If calibration is enabled, further split the training data
        if calibration:
            calib_size = int(len(full_train_subset) * calibration_pct)
            train_size = len(full_train_subset) - calib_size
            train_subset, calib_subset = random_split(full_train_subset, [train_size, calib_size])
        else:
            train_subset = full_train_subset
            calib_subset = None
        
        # Train the model on the current fold
        base_model = train_rnn_yale(train_subset, **train_args)
        model = TempScaledModel(base_model, apply_scale=calibration, apply_softmax=False)  # Always wrap in TempScaledModel

        # Calibrate the model if calibration is enabled
        if calibration:
            model = calibrate(model, calib_subset, **train_args)
        
        # Test the model
        fold_metrics = test_rnn_yale(model, test_subset, calibration=calibration, **train_args)
        
        metrics.append(fold_metrics)
        train_datasets.append(train_subset)
        if calibration:
            calib_datasets.append(calib_subset)
        test_datasets.append(test_subset)
        models.append(model)
        
    return metrics, train_datasets, calib_datasets, test_datasets, models