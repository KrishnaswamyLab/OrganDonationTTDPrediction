import torch
import pickle
from torch.utils.data import DataLoader, Subset
from model.dataset import train_test_split, YaleDatasetWithMissingnessInfo
from model.train import train_rnn_yale, test_rnn_yale, cross_validate_rnn_yale
import argparse
from pathlib import Path
import pathlib
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import random
import numpy as np
import os

def seed_everything(seed=42):
    """
    Seed everything to make code more deterministic.
    """
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy module
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python's built-in `hashlib` and others
    
    torch.manual_seed(seed)  # PyTorch random number generator for CPU
    torch.cuda.manual_seed(seed)  # PyTorch random number generator for all GPUs
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU, this is needed as well
    
    # Configure PyTorch to behave deterministically
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("Arguments:")
    print(cfg)
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if cfg.logger.use_wandb:
        run = wandb.init(
            entity=cfg.logger.entity,
            project=cfg.logger.project,
            tags=cfg.logger.tags,
            reinit=True,
            config=config,
            settings=wandb.Settings(start_method="thread"),
        )
    seed_everything(seed=cfg.training.seed)
    # yaledataset = torch.load(f"{cfg.paths.root}/{cfg.paths.dataset}")
    path_prepend = f"{cfg.paths.root}/{cfg.paths.result}/{cfg.paths.dataset.split('/')[-1].split('.')[0].split('_')[-1]}/{cfg.training.seed}/{cfg.model.task}_target{cfg.model.targetidx}dim{cfg.model.output_dim}_{cfg.model.type}{cfg.model.rnn_type}"
    yaledataset = torch.load(f"{cfg.paths.root}/{cfg.paths.dataset}")

    if cfg.validation.mode == "cross_val":
        cross_val(cfg, yaledataset, path_prepend)
    elif cfg.validation.mode == "train_only":
        train_only(cfg, yaledataset, path_prepend)
    else:
        raise ValueError(f"Unknown validation mode: {cfg.validation.mode}")
    
    if cfg.logger.use_wandb:
        run.finish()

def cross_val(cfg, yaledataset, path_prepend):
    # Load with pickle
    with open(f'{cfg.paths.root}/{cfg.paths.fold_idx}', 'rb') as f:
        fold_idx = pickle.load(f)
    datasets = [Subset(yaledataset, fold_id) for fold_id in fold_idx]
    # datasets = load_datasets(f"{cfg.paths.root}/{cfg.paths.cv}", cfg.validation.k_folds)
    metrics, train_datasets, calib_datasets, test_datasets, models = cross_validate_rnn_yale(datasets, model_type=cfg.model.type, rnn_type=cfg.model.rnn_type, task=cfg.model.task, target_index=cfg.model.targetidx, epochs=cfg.training.epochs, batch_size=cfg.training.batch_size, learning_rate=cfg.training.learning_rate, output_dim=cfg.model.output_dim, calibration=cfg.calibration.enabled, calibration_pct=cfg.calibration.pct, calibration_epochs=cfg.calibration.epochs, calibration_lr=cfg.calibration.lr, n_bins_ece=cfg.calibration.n_bins_ece, seed=cfg.training.seed)
    savepath = f"{path_prepend}_crossval"
    Path(f"{savepath}").mkdir(parents=True, exist_ok=True)
    for i in range(len(metrics)):
        torch.save(metrics[i], f"{savepath}/metrics_fold{i}.pt")
        torch.save(train_datasets[i], f"{savepath}/train_dataset_fold{i}.pt")
        if cfg.calibration.enabled:
            torch.save(calib_datasets[i], f"{savepath}/calib_dataset_fold{i}.pt")
        torch.save(test_datasets[i], f"{savepath}/test_dataset_fold{i}.pt")
        torch.save(models[i], f"{savepath}/model_fold{i}.pt")
        torch.save(models[i].state_dict(), f"{savepath}/model_state_dict_fold{i}.pt")

def train_only(cfg, yaledataset, path_prepend):
    savepath = f"{path_prepend}_trainonly"
    pathlib.Path(f"{savepath}").mkdir(parents=True, exist_ok=True)
    model = train_rnn_yale(yaledataset, model_type=cfg.model.type, rnn_type=cfg.model.rnn_type, task=cfg.model.task, target_index=cfg.model.targetidx, epochs=cfg.training.epochs, batch_size=cfg.training.batch_size, learning_rate=cfg.training.learning_rate, output_dim=cfg.model.output_dim)
    torch.save(model, f"{savepath}/model.pt")

#################
## THIS IS NO LONGER USED!
# def external_validation(cfg, yaledataset, path_prepend):
#     savepath = f"{path_prepend}_external_validation_{cfg.validation.split_train}"
#     pathlib.Path(f"{savepath}").mkdir(parents=True, exist_ok=True)
#     # no calibration. no cross validation
#     if cfg.validation.split_train:
#         train_dataset, test_dataset = train_test_split(yaledataset, test_size=cfg.training.test_size, random_seed=cfg.training.seed)
#         torch.save(train_dataset, f"{savepath}/train_dataset.pt")
#         torch.save(test_dataset, f"{savepath}/test_dataset.pt")    # other_dataset = torch.load(external_validation_path)
#     else:
#         train_dataset = yaledataset
#         test_dataset = torch.load(cfg.paths.external_validation)
#     model = train_rnn_yale(train_dataset, model_type=model_type, rnn_type=rnn_type, task=task, target_index=target_index, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, output_dim=output_dim)
#     test_metrics = test_rnn_yale(model, test_dataset, task=task, target_index=target_index, output_dim=output_dim, batch_size=batch_size, calibration=calibration) # still compute the ece if calibration is true.
#     pathlib.Path(f"{savepath}").mkdir(parents=True, exist_ok=True)
#     torch.save(test_metrics, f"{savepath}/metrics.pt")
#     torch.save(model, f"{savepath}/model.pt")




if __name__ == "__main__":
    main()