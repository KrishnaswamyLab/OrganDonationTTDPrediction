import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
import re
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_recall_curve, auc


def get_results_df(extracted_results_path):
    # Initialize an empty list to store the rows of the DataFrame
    df_rows = []
    # Iterate through each subfolder and read the "test_loss.pt" file
    for subfolder in os.listdir(extracted_results_path):
        subfolder_path = os.path.join(extracted_results_path, subfolder)
        
        # Check if the subfolder contains "test_loss.pt"
        if "test_loss.pt" in os.listdir(subfolder_path):
            loss_file_path = os.path.join(subfolder_path, "test_loss.pt")
            
            # Load the tensor using torch.load
            loss_tensor = torch.load(loss_file_path)
            
            # Convert the tensor to a list
            loss_values = loss_tensor.tolist()
            
            # Split the subfolder name by "_"
            subfolder_parts = subfolder.split("_")
            
            # Create a row based on the number of parts in the subfolder name
            if len(subfolder_parts) == 3:
                row = subfolder_parts + loss_values
            elif len(subfolder_parts) == 6:
                row = [subfolder_parts[-1], subfolder_parts[-3], subfolder_parts[-2]] + loss_values
            else:
                continue  # Skip subfolders that don't match the expected formats
            
            # Append the row to the list
            df_rows.append(row)

    # Create a DataFrame from the list of rows
    column_names = ["Part1", "Part2", "Part3", "Loss1", "Loss2", "Loss3"]
    df = pd.DataFrame(df_rows, columns=column_names)

    # Save the DataFrame to a CSV file
    # csv_file_path = 'results_agg_dataframe.csv'
    # df.to_csv(csv_file_path, index=False)

    # Show the first few rows of the DataFrame and the path to the CSV file
    # df.head(), csv_file_path
    # target = pd.read_csv('output_data/preprocessed3/static_target.csv')
    # vary = target["time_extub_to_death_hours"].var()
    df = df.sort_values(by=['Part1', 'Part3'])
    # df.loc[df["Part1"] == "regression", "Loss1"] = 1 - df.loc[df["Part1"] == "regression", "Loss1"]/vary
    return df

def get_cm_auroc(res_path, model_path="model.pt", test_dataset_path="test_dataset.pt"):
    model = torch.load(res_path + f"/{model_path}", map_location=torch.device('cpu'))
    test_dataset = torch.load(res_path + f"/{test_dataset_path}")
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    batch_X, batch_M, batch_t, batch_s, batch_y = next(iter(test_loader))
    target = batch_y[:, 1].long()
    # Forward pass
    with torch.no_grad():
        predictions = model(batch_X, batch_M, batch_t, batch_s).squeeze()
    # predictions = F.log_softmax(predictions, dim=1)
    predictions = F.softmax(predictions, dim=1)
    cm = confusion_matrix(target.numpy(), predictions.argmax(dim=1).numpy())
    auroc_macro = roc_auc_score(target.numpy(), predictions.numpy(), multi_class='ovr', average='macro')
    accuracy = cm.trace()/cm.sum()
    return dict(model=model_path, cm=cm, auroc_macro=auroc_macro, accuracy=accuracy)

def plot_cm(cm, classlabels=["<30", "30~60", "60~120", ">120"], title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2) # for label size
    ax = sns.heatmap(cm, annot=cm, fmt='g', cmap="Blues", xticklabels=classlabels, yticklabels=classlabels)
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    plt.title(title)
    plt.show()

def plot_cm_auroc(res_path, model_path="model.pt", test_dataset_path="test_dataset.pt"):
    res_this = get_cm_auroc(res_path, model_path, test_dataset_path)
    cm = res_this["cm"]
    model = res_this["model"].split("_")[-1]
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2) # for label size
    ax = sns.heatmap(cm, annot=cm, fmt='g', cmap="Blues", xticklabels=["<30", "30~60", "60~120", ">120"], yticklabels=["<30", "30~60", "60~120", ">120"])
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    plt.title(f"Confusion Matrix of {model}, AUROC={res_this['auroc_macro']:.3f}, Accuracy={res_this['accuracy']:.3f}")
    plt.show()


def plot_cm_auroc_subplots(res_path, model_path="model.pt", test_dataset_path="test_dataset.pt", ax=None, fontsize=1.2, annotsize=1.2):
    res_this = get_cm_auroc(res_path, model_path, test_dataset_path)
    cm = res_this["cm"]
    model = res_this["model"].split("_")[-1].strip(".pt")
    sns.set(font_scale=fontsize)  # for label size
    sns.heatmap(cm, annot=cm, fmt='g', cmap="Blues", xticklabels=["<30", "30~60", "60~120", ">120"], yticklabels=["<30", "30~60", "60~120", ">120"], ax=ax, cbar=False, annot_kws={'size': annotsize})
    ax.set_xlabel("Predicted Values", fontsize=fontsize)
    ax.set_ylabel("Actual Values", fontsize=fontsize)
    ax.set_title(f"AUC={res_this['auroc_macro']:.3f}, Acc={res_this['accuracy']:.3f}", fontsize=fontsize)
    ax.title.set_size(fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)


def get_cv_results_df(extracted_results_path):
    subfolders = []
    # Iterate through each subfolder and read the "test_loss.pt" file
    for subfolder in os.listdir(extracted_results_path):
        if subfolder.endswith("crossval"):
            # print(subfolder)
            subfolders.append(subfolder)
    metrics_all = []
    for fdr in subfolders:
        metrics_list = []
        for file in os.listdir(extracted_results_path + fdr):
            if re.match(r".*metrics_fold\d+\.pt$", file):
                # print(file)
                metrics_list.append(torch.load(extracted_results_path + fdr + "/" + file))
        metrics = torch.stack(metrics_list)
        metrics_means = torch.stack(metrics_list).mean(dim=0).numpy()
        metrics_stds = torch.stack(metrics_list).std(dim=0).numpy()
        metrics_dict = dict(model=fdr, metrics_means=metrics_means, metrics_stds=metrics_stds)
        metrics_all.append(metrics_dict)
    # Convert metrics_all to a list of rows for the dataframe
    df_rows = [{
        'model': metrics_dict['model'],
        'metric1': f"{metrics_dict['metrics_means'][0]:.3f} ({metrics_dict['metrics_stds'][0]:.3f})",
        'metric2': f"{metrics_dict['metrics_means'][1]:.3f} ({metrics_dict['metrics_stds'][1]:.3f})",
        'metric3': f"{metrics_dict['metrics_means'][2]:.3f} ({metrics_dict['metrics_stds'][2]:.3f})"
    } for metrics_dict in metrics_all]

    # Convert the list of rows to a dataframe
    df = pd.DataFrame(df_rows)
    df.sort_values(by=['model'], inplace=True)
    return df

def compute_ece(probs, labels, n_bins=10):
    """
    Compute the Expected Calibration Error (ECE) using broadcasting.
    
    Args:
    - probs (torch.Tensor): Tensor of predicted probabilities of shape (n_samples, n_classes).
    - labels (torch.Tensor): True labels of shape (n_samples).
    - n_bins (int): Number of bins to use for ECE computation.
    
    Returns:
    - ece (float): The computed ECE.
    """
    
    # Get the predicted confidence (max probability)
    confidences, predictions = torch.max(probs, 1)
    
    # Define bin edges
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Use broadcasting to determine the bin index for each confidence
    bin_idxs = ((confidences[:, None] > bin_lowers) & (confidences[:, None] <= bin_uppers)).float()
    
    # Compute bin count
    bin_counts = bin_idxs.sum(dim=0)
    
    # Compute the accuracy for each bin
    correct = predictions == labels
    bin_corrects = (bin_idxs * correct[:, None]).sum(dim=0)
    bin_accuracy = bin_corrects / bin_counts.clamp(min=1)  # clamp to avoid division by 0
    
    # Compute average confidence for each bin
    bin_confidences = (bin_idxs * confidences[:, None]).sum(dim=0)
    bin_avg_confidence = bin_confidences / bin_counts.clamp(min=1)  # clamp to avoid division by 0
    
    # Compute ECE
    ece = torch.sum(bin_counts * torch.abs(bin_avg_confidence - bin_accuracy)) / len(probs)
    
    return ece.item()

def expected_calibration_error(samples, true_labels, M=5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece

def brier_score(predictions, targets):
    """
    Compute the Brier Score for multi-class classification.
    
    Parameters:
    - predictions (torch.Tensor): The predicted probabilities with shape (batch_size, num_classes)
    - targets (torch.Tensor): The true class labels with shape (batch_size,)
    
    Returns:
    - float: The Brier Score
    """
    n_samples, n_classes = predictions.shape
    
    # Convert targets to one-hot encoding
    targets_one_hot = torch.zeros(n_samples, n_classes).scatter_(1, targets.unsqueeze(1), 1)
    
    # Compute Brier score
    score = torch.mean(torch.sum((predictions - targets_one_hot) ** 2, dim=1))
    
    return score.item()

def combined_calibration_plot(probabilities, probabilities_calibrated, targets, ax=None, title=None, n_bins=10):
    """
    Plot the calibration curves for the original and calibrated probabilities in a 2x2 grid.
    
    Parameters:
    - probabilities (torch.Tensor): Original predicted probabilities with shape (batch_size, num_classes).
    - probabilities_calibrated (torch.Tensor): Calibrated predicted probabilities with shape (batch_size, num_classes).
    - targets (torch.Tensor): True class labels with shape (batch_size,).
    - ax (array-like): Array of subplot axes for 2x2 grid.
    - title (str): Main title for the 2x2 grid.
    - n_bins (int): Number of bins to use for calibration plot.
    """
    n_samples, n_classes = probabilities.shape
    if n_classes != 4:
        raise ValueError("This function is designed for 4 classes. Adjust the code for a different number of classes.")
    
    # Convert targets to one-hot encoding
    targets_one_hot = torch.zeros(n_samples, n_classes).scatter_(1, targets.unsqueeze(1), 1)
    
    if ax is None:
        _, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    for class_idx in range(n_classes):
        current_ax = ax[class_idx // 2, class_idx % 2]
        
        # Function to plot calibration for given probabilities
        def plot_calibration(class_probabilities, label, linestyle, marker):
            # Bin the probabilities
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            true_proportions = []
            pred_means = []
            bin_counts = []

            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Indices of samples that fall into the current bin
                bin_indices = (class_probabilities > bin_lower) & (class_probabilities <= bin_upper)
                bin_count = bin_indices.sum().item()

                if bin_count > 0:
                    true_proportions.append(targets_one_hot[:, class_idx][bin_indices].float().mean().item())
                    pred_means.append(class_probabilities[bin_indices].mean().item())
                    bin_counts.append(bin_count)

            # Plot and annotate
            current_ax.plot(pred_means, true_proportions, linestyle=linestyle, marker=marker, label=label)
            for pred_mean, true_prop, count in zip(pred_means, true_proportions, bin_counts):
                current_ax.annotate(f"{count}", (pred_mean, true_prop), textcoords="offset points", xytext=(0,10), ha='center')
        
        # Plot calibration for original and calibrated probabilities
        plot_calibration(probabilities[:, class_idx], "Original", linestyle='-', marker='s')
        plot_calibration(probabilities_calibrated[:, class_idx], "Calibrated", linestyle='-', marker='^')
        
        # Plot settings
        current_ax.plot([0, 1], [0, 1], 'k:', label="Perfectly calibrated")
        current_ax.set_xlabel('Mean predicted probability')
        current_ax.set_ylabel('Fraction of positives')
        current_ax.set_title(f'Class {class_idx}')
        current_ax.legend()

    if title:
        plt.suptitle(title, fontsize=16, y=1.05)
    plt.tight_layout()

class ECEComputer:
    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), n_bins=10):
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=device)
        self.bin_lowers = self.bin_boundaries[:-1]
        self.bin_uppers = self.bin_boundaries[1:]
        
        # Accumulators
        self.bin_counts = torch.zeros(n_bins, device=device)
        self.bin_corrects = torch.zeros(n_bins, device=device)
        self.bin_confidences = torch.zeros(n_bins, device=device)

    def update(self, probs, labels):
        confidences, predictions = torch.max(probs, 1)
        bin_idxs = ((confidences[:, None] > self.bin_lowers) & (confidences[:, None] <= self.bin_uppers)).float()
        
        self.bin_counts += bin_idxs.sum(dim=0)
        
        correct = predictions == labels
        self.bin_corrects += (bin_idxs * correct[:, None]).sum(dim=0)
        
        self.bin_confidences += (bin_idxs * confidences[:, None]).sum(dim=0)

    def compute(self):
        bin_accuracy = self.bin_corrects / self.bin_counts.clamp(min=1)
        bin_avg_confidence = self.bin_confidences / self.bin_counts.clamp(min=1)
        
        ece = torch.sum(self.bin_counts * torch.abs(bin_avg_confidence - bin_accuracy)) / self.bin_counts.sum()
        return ece.item()




def compute_custom_auroc(y_true, y_probs):
    # For class 0 vs rest
    binary_labels = (y_true == 0).astype(int)
    auroc_0_vs_rest = roc_auc_score(binary_labels, y_probs[:, 0])
    
    # For classes 0,1 vs 2,3
    binary_labels = (y_true <= 1).astype(int)
    auroc_01_vs_23 = roc_auc_score(binary_labels, y_probs[:, 0] + y_probs[:, 1])
    
    # For classes 0,1,2 vs 3
    binary_labels = (y_true <= 2).astype(int)
    auroc_012_vs_3 = roc_auc_score(binary_labels, y_probs[:, 0] + y_probs[:, 1] + y_probs[:, 2])
    
    return auroc_0_vs_rest, auroc_01_vs_23, auroc_012_vs_3

def compute_custom_aucpr(y_true, y_probs):

    # Use AUC function to calculate the area under the curve of precision recall curve

    # For class 0 vs rest
    binary_labels = (y_true == 0).astype(int)
    precision, recall, thresholds = precision_recall_curve(binary_labels, y_probs[:,0])
    aucpr_0_vs_rest = auc(recall, precision)
    
    # For classes 0,1 vs 2,3
    binary_labels = (y_true <= 1).astype(int)
    precision, recall, thresholds = precision_recall_curve(binary_labels,  y_probs[:, 0] + y_probs[:, 1])
    aucpr_01_vs_23 = auc(recall, precision)
    
    # For classes 0,1,2 vs 3

    binary_labels = (y_true <= 2).astype(int)
    precision, recall, thresholds = precision_recall_curve(binary_labels,  y_probs[:, 0] + y_probs[:, 1] + y_probs[:, 2])
    aucpr_012_vs_3 = auc(recall, precision)
    
    return aucpr_0_vs_rest, aucpr_01_vs_23, aucpr_012_vs_3

## DEPRECATED
def compute_custom_accuracy(y_true, y_probs):
    # For class 0 vs rest
    binary_labels = (y_true == 0).astype(int)
    preds_0_vs_rest = (y_probs[:, 0] > 0.5).astype(int)
    acc_0_vs_rest = accuracy_score(binary_labels, preds_0_vs_rest)
    
    # For classes 0,1 vs 2,3
    binary_labels = (y_true <= 1).astype(int)
    preds_01_vs_23 = ((y_probs[:, 0] + y_probs[:, 1]) > 0.5).astype(int)
    acc_01_vs_23 = accuracy_score(binary_labels, preds_01_vs_23)
    
    # For classes 0,1,2 vs 3
    binary_labels = (y_true <= 2).astype(int)
    preds_012_vs_3 = ((y_probs[:, 0] + y_probs[:, 1] + y_probs[:, 2]) > 0.5).astype(int)
    acc_012_vs_3 = accuracy_score(binary_labels, preds_012_vs_3)
    
    return acc_0_vs_rest, acc_01_vs_23, acc_012_vs_3

def compute_custom_accuracy_combine_classes(y_true, y_pred):
    # For class 0 vs rest
    binary_labels = (y_true == 0).astype(int)
    preds_0_vs_rest = (y_pred == 0).astype(int)
    acc_0_vs_rest = (binary_labels == preds_0_vs_rest).mean()
    
    # For classes 0,1 vs 2,3
    binary_labels = (y_true <= 1).astype(int)
    preds_01_vs_23 = (y_pred <= 1).astype(int)
    acc_01_vs_23 = (binary_labels == preds_01_vs_23).mean()
    
    # For classes 0,1,2 vs 3
    binary_labels = (y_true <= 2).astype(int)
    preds_012_vs_3 = (y_pred <= 2).astype(int)
    acc_012_vs_3 = (binary_labels == preds_012_vs_3).mean()
    
    return acc_0_vs_rest, acc_01_vs_23, acc_012_vs_3

def compute_custom_f1(y_true, y_probs):
    ## FIXME this is not correct!!!
    # For class 0 vs rest
    binary_labels = (y_true == 0).astype(int)
    preds_0_vs_rest = (y_probs[:, 0] > 0.5).astype(int)
    f1_0_vs_rest = f1_score(binary_labels, preds_0_vs_rest)
    
    # For classes 0,1 vs 2,3
    binary_labels = (y_true <= 1).astype(int)
    preds_01_vs_23 = ((y_probs[:, 0] + y_probs[:, 1]) > 0.5).astype(int)
    f1_01_vs_23 = f1_score(binary_labels, preds_01_vs_23)
    
    # For classes 0,1,2 vs 3
    binary_labels = (y_true <= 2).astype(int)
    preds_012_vs_3 = ((y_probs[:, 0] + y_probs[:, 1] + y_probs[:, 2]) > 0.5).astype(int)
    f1_012_vs_3 = f1_score(binary_labels, preds_012_vs_3)
    
    return f1_0_vs_rest, f1_01_vs_23, f1_012_vs_3

def plot_custom_curves(y_true, y_probs, title=""):
    plt.figure(figsize=(15, 12))

    scenarios = [
        {"label": "Class 0 vs Rest", "classes": [0]},
        {"label": "Class 0,1 vs 2,3", "classes": [0, 1]},
        {"label": "Class 0,1,2 vs 3", "classes": [0, 1, 2]}
    ]

    for idx, scenario in enumerate(scenarios):
        binary_labels = np.isin(y_true, scenario["classes"]).astype(int)
        score = y_probs[:, scenario["classes"]].sum(axis=1)
        
        fpr, tpr, _ = roc_curve(binary_labels, score)
        precision, recall, _ = precision_recall_curve(binary_labels, score)
        
        roc_auc = auc(fpr, tpr)
        avg_precision = average_precision_score(binary_labels, score)
        
        plt.subplot(3, 2, 2*idx + 1)
        plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC: {scenario["label"]}')
        plt.legend(loc='lower right')
        plt.grid(True)

        plt.subplot(3, 2, 2*idx + 2)
        plt.plot(recall, precision, color='b', lw=2, label=f'Avg Precision = {avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall: {scenario["label"]}')
        plt.legend()
        plt.grid(True)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

def ensemble_predict(models, test_dataset, device, apply_scale=False, softmax_results=False):
    """
    Perform ensemble prediction on the validation set by averaging logits.
    
    Args:
    - models (list): A list of trained PyTorch models.
    - val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
    - device (str): Device to which models and data should be moved before prediction.
    
    Returns:
    - ensemble_predictions (torch.Tensor): A tensor of ensemble predictions for the validation set.
    """

    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    batch_X, batch_M, batch_t, batch_s, batch_y = next(iter(test_loader))
    batch_X, batch_M, batch_t, batch_s, batch_y = batch_X.to(device), batch_M.to(device), batch_t.to(device), batch_s.to(device), batch_y.to(device)
    X_test_tuple = (batch_X, batch_M, batch_t, batch_s)
    target = batch_y[:, 1].long()

    # Ensure models are in eval mode and moved to the desired device
    for model in models:
        model.eval()
        model.to(device)
        model.apply_softmax = False
        model.apply_scale = apply_scale
    
    # Store all predictions
    # all_predictions = []
    
    # Iterate over validation data
    # for inputs, _ in val_loader:
    #     inputs = inputs.to(device)
        
    # Predict using each model and store predictions
    logits = [model(batch_X, batch_M, batch_t, batch_s).squeeze() for model in models]
    
    # Average the logits
    avg_logit = torch.stack(logits).mean(dim=0)
    
    # Apply sigmoid to the averaged logits
    # avg_prediction = F.sigmoid(avg_logit)
    avg_prediction = avg_logit
    if softmax_results:
        avg_prediction = F.softmax(avg_prediction, dim=1)
    # all_predictions.append(avg_prediction)
    
    # Concatenate all predictions
    # ensemble_predictions = torch.cat(all_predictions, dim=0)
    
    return avg_prediction, target

# Example usage:
# device = 'cuda'  # or 'cpu' or any other specific GPU like 'cuda:0'
# models = [torch.load("model_fold1.pth"), torch.load("model_fold2.pth"), ...]
# val_dataset = YourValidationDataset(...)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
# predictions = ensemble_predict(models, val_loader, device)


from scipy.stats import chi2_contingency

def chi2_test(true_labels, predicted_labels, num_classes):
    """
    Perform a Chi-square test to compare the observed frequencies of predicted labels
    with the true labels across different classes.

    Args:
    true_labels (list): List of actual true labels.
    predicted_labels (list): List of predicted labels from the model.
    num_classes (int): Number of different classes in the dataset.

    Returns:
    tuple: chi2 statistic and p-value from the Chi-square test.
    """
    # Constructing the contingency table
    contingency_table = np.zeros((num_classes, num_classes))
    for true, pred in zip(true_labels, predicted_labels):
        contingency_table[true][pred] += 1

    # Performing the Chi-square test
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    return chi2, p_value

def compute_custom_chi2(y_true, y_pred):
    # For class 0 vs rest
    binary_labels = (y_true == 0).astype(int)
    preds_0_vs_123 = (y_pred == 0).astype(int)
    chi2_0_vs_123, pval_0_vs_123 = chi2_test(binary_labels, preds_0_vs_123, 2)
    
    # For classes 0,1 vs 2,3
    binary_labels = (y_true <= 1).astype(int)
    preds_01_vs_23 = (y_pred <= 1).astype(int)
    chi2_01_vs_23, pval_01_vs_23 = chi2_test(binary_labels, preds_01_vs_23, 2)
    
    # For classes 0,1,2 vs 3
    binary_labels = (y_true <= 2).astype(int)
    preds_012_vs_3 = (y_pred <= 2).astype(int)
    chi2_012_vs_3, pval_012_vs_3 = chi2_test(binary_labels, preds_012_vs_3, 2)

    return (chi2_0_vs_123, pval_0_vs_123), (chi2_01_vs_23, pval_01_vs_23), (chi2_012_vs_3, pval_012_vs_3)

def get_all_metrics(predictions, targets):

    pred_labels = np.argmax(predictions, 1)
    acc_uncalib = accuracy_score(targets, pred_labels)
    try:
        auroc_0_vs_rest, auroc_01_vs_23, auroc_012_vs_3 = compute_custom_auroc(targets, predictions)
    except ValueError:
        auroc_0_vs_rest, auroc_01_vs_23, auroc_012_vs_3 = np.nan, np.nan, np.nan
    try: 
        aucpr_0_vs_rest, aucpr_01_vs_23, aucpr_012_vs_3 = compute_custom_aucpr(targets, predictions)
    except ValueError:
        aucpr_0_vs_rest, aucpr_01_vs_23, aucpr_012_vs_3 = np.nan, np.nan, np.nan
    acc_0_vs_rest, acc_01_vs_23, acc_012_vs_3 = compute_custom_accuracy_combine_classes(targets, pred_labels)
    try:
        f1_0_vs_rest, f1_01_vs_23, f1_012_vs_3 = compute_custom_f1(targets, predictions)
    except ValueError:
        f1_0_vs_rest, f1_01_vs_23, f1_012_vs_3 = np.nan, np.nan, np.nan
    try:
        ece = expected_calibration_error(predictions, targets, M=10)
    except ValueError:
        ece = np.nan
    
    res_dict = {"Accuracy": acc_uncalib,
                "Accuracy_0_rest": acc_0_vs_rest,
                "Accuracy_01_23": acc_01_vs_23,
                "Accuracy_012_3": acc_012_vs_3,
                "ROC_AUC_0_rest": auroc_0_vs_rest,
                "ROC_AUC_01_23": auroc_01_vs_23,
                "ROC_AUC_012_3": auroc_012_vs_3,
                "PR_AUC_0_rest": aucpr_0_vs_rest,
                "PR_AUC_01_23": aucpr_01_vs_23,
                "PR_AUC_012_3": aucpr_012_vs_3,
                "F1_0_rest": f1_0_vs_rest,
                "F1_01_23": f1_01_vs_23,
                "F1_012_3": f1_012_vs_3,
                "ECE": ece.item()}
    return res_dict