"""
This script sets up an Optuna-based hyperparameter tuning pipeline using PyTorch Lightning.
It handles configuration instantiation, callback creation, training, and evaluation of a 
multi-task model. Hyperparameters are tuned via Optuna and the best parameters/metric are 
saved to a YAML file.
"""

import os
import argparse
import yaml
import importlib
import functools
import optuna

import torch
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelSummary
# ModelCheckpoint is not imported since we do not want checkpointing.
import lightning.pytorch.loggers as pl_loggers  # for WandbLogger, etc.

# Import your custom code
from src.data.multitask_datamodule import MultiTaskDataModule
from src.models.multitask_module import MultiTaskModule
from src.utils import get_pylogger

from optuna.samplers import TPESampler
# 1) Import MedianPruner + PruningCallback
from optuna.pruners import MedianPruner
from optuna.integration import PyTorchLightningPruningCallback

log = get_pylogger(__name__)

###############################################################################
# 1) Mini-Recursive "instantiate" to handle _target_ (and _partial_) keys
###############################################################################
def import_class(path: str):
    """
    Split a full path like 'torchvision.transforms.Compose' into
    module part and class name, then import and return the class.

    Args:
        path (str): The full path to the class.

    Returns:
        type: The imported class.
    """
    parts = path.split(".")
    module_path = ".".join(parts[:-1])
    class_name = parts[-1]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def instantiate_recursive(cfg):
    """
    Recursively walk through 'cfg'. If a dictionary contains a '_target_' key,
    import and instantiate that object. If '_partial_' == true, return a functools.partial
    instead of a full instance.

    Args:
        cfg (dict or list or any): The configuration structure.

    Returns:
        The instantiated configuration object.
    """
    if isinstance(cfg, list):
        # Recursively instantiate each item in the list
        return [instantiate_recursive(item) for item in cfg]
    elif isinstance(cfg, dict):
        if "_target_" in cfg:
            cls = import_class(cfg["_target_"])
            partial_mode = cfg.get("_partial_", False)
            kwargs = {
                k: instantiate_recursive(v)
                for k, v in cfg.items() if k not in ("_target_", "_partial_")
            }
            if partial_mode:
                return functools.partial(cls, **kwargs)
            else:
                return cls(**kwargs)
        else:
            return {k: instantiate_recursive(v) for k, v in cfg.items()}
    else:
        return cfg


###############################################################################
# 2) Argument parsing
###############################################################################
def parse_args():
    """
    Parse command-line arguments for the hyperparameter tuning.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Optuna-based Hyperparameter Tuning"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML config file"
    )
    parser.add_argument(
        "--n_trials", type=int, default=10,
        help="Number of Optuna trials"
    )
    parser.add_argument(
        "--study_name", type=str, default="optuna_study",
        help="Optuna study name"
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help="Optuna storage (e.g. sqlite:///example.db)"
    )
    return parser.parse_args()


def load_config(cfg_path: str):
    """
    Loads a YAML config file and returns the raw configuration dictionary.

    Args:
        cfg_path (str): Path to the YAML configuration file.

    Returns:
        dict: The loaded configuration dictionary.
    """
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    return config


###############################################################################
# 3) Create Callbacks & Logger
###############################################################################
def create_callbacks(cfg):
    """
    Create a list of Lightning callbacks based on the configuration.

    Args:
        cfg (dict): The configuration dictionary.

    Returns:
        list: A list of instantiated callback objects.
    """
    cb_list = []
    # Note: We have removed ModelCheckpoint instantiation to avoid checkpointing.
    # EarlyStopping
    if "early_stopping" in cfg["callbacks"]:
        es_cfg = cfg["callbacks"]["early_stopping"]
        earlystop_cb = EarlyStopping(
            monitor=es_cfg["monitor"],
            patience=es_cfg["patience"],
            mode=es_cfg["mode"]
        )
        cb_list.append(earlystop_cb)
    # ModelSummary
    if "model_summary" in cfg["callbacks"]:
        ms_cfg = cfg["callbacks"]["model_summary"]
        model_summary_cb = ModelSummary(max_depth=ms_cfg["max_depth"])
        cb_list.append(model_summary_cb)
    # LR Monitor
    #if "lr_monitor" in cfg["callbacks"]:
    #   lr_cb = LearningRateMonitor(logging_interval='epoch')
    #   cb_list.append(lr_cb)
    return cb_list


def instantiate_logger(cfg):
    """
    Disable external logging by returning None.

    Returns:
        None
    """
    return None


###############################################################################
# 4) Main Training Function
###############################################################################
def train_once(config, trial=None):
    """
    1) Recursively instantiate configuration objects (transforming all '_target_' keys).
    2) Build the DataModule, Model, and Trainer.
    3) Run training (and optionally testing).
    4) Return the primary metric.

    If a trial is provided, attach a PyTorch Lightning pruning callback.

    Args:
        config (dict): The base configuration dictionary.
        trial (optuna.trial.Trial, optional): The Optuna trial object for hyperparameter suggestions.

    Returns:
        float: The final metric value from the training run.
    """
    if config.get("seed") is not None:
        L.seed_everything(config["seed"], workers=True)

    instantiated_cfg = instantiate_recursive(config)
    datamodule = instantiated_cfg["data"]
    model = instantiated_cfg["model"]

    callbacks = create_callbacks(instantiated_cfg)
    loggers = instantiate_logger(instantiated_cfg)

    metric_name = "val/weighted_f1_r2"
    if trial is not None:
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor=metric_name)
        callbacks.append(pruning_callback)

    trainer_cfg = instantiated_cfg["trainer"]
    trainer = Trainer(
        default_root_dir=trainer_cfg["default_root_dir"],
        min_epochs=trainer_cfg["min_epochs"],
        max_epochs=trainer_cfg["max_epochs"],
        accelerator=trainer_cfg["accelerator"],
        devices=trainer_cfg["devices"],
        precision=trainer_cfg["precision"],
        check_val_every_n_epoch=trainer_cfg["check_val_every_n_epoch"],
        log_every_n_steps=trainer_cfg["log_every_n_steps"],
        deterministic=trainer_cfg["deterministic"],
        enable_checkpointing=trainer_cfg["enable_checkpointing"],
        callbacks=callbacks,
        logger=False
    )

    if instantiated_cfg.get("compile", False):
        log.info("Compiling model with torch.compile()!")
        model = torch.compile(model)

    if instantiated_cfg.get("train", True):
        trainer.fit(model=model, datamodule=datamodule)

    val_metrics = trainer.callback_metrics
    final_score = val_metrics.get(metric_name, 0.0)
    return float(final_score)


###############################################################################
# 5) Objective + Main
###############################################################################
def objective(trial, base_config):
    """
    Modify the base configuration in-place using trial.suggest_*,
    then call train_once(base_config) to train the model.
    
    Args:
        trial (optuna.trial.Trial): The trial object for hyperparameter suggestions.
        base_config (dict): The base configuration dictionary.
        
    Returns:
        float: The metric value resulting from this training trial.
    """
    # 1) Optimizer => only Adam and AdamW
    optimizer_name = trial.suggest_categorical(
        "optimizer_name", ["torch.optim.Adam", "torch.optim.AdamW"]
    )
    base_config["model"]["optimizer_cfg"]["_target_"] = optimizer_name

    # 2) LR
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    base_config["model"]["optimizer_cfg"]["lr"] = lr

    # 3) Weight decay
    weight_decay = trial.suggest_float("weight_decay", 1e-9, 1e-4, log=True)
    base_config["model"]["optimizer_cfg"]["weight_decay"] = weight_decay

    # 4) Unfreeze layers (for DenseNet)
    unfreeze_count = trial.suggest_int("unfreeze_count", 60, 110)
    base_config["model"]["modal_nets"]["images"]["unfreeze_layer_count"] = unfreeze_count

    # 5) Batch size (User wants only 8)
    batch_size = trial.suggest_categorical("batch_size", [8])
    base_config["data"]["batch_size"] = batch_size

    # 6) LR Scheduler factor + patience
    factor = trial.suggest_float("factor", 0.1, 0.9)
    patience = trial.suggest_categorical("patience", [2, 4, 6, 8])
    base_config["model"]["scheduler_cfg"]["factor"] = factor
    base_config["model"]["scheduler_cfg"]["patience"] = patience

    # 7) Fusion type
    fusion_type = trial.suggest_categorical("fusion_type", ["concat", "mean"])
    base_config["model"]["fusion_type"] = fusion_type

    # 8) GRU hyperparameters
    hidden_size = trial.suggest_int("hidden_size", 80, 200)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    rnn_dropout = trial.suggest_float("rnn_dropout", 0.0, 0.3)
    bidirectional = trial.suggest_categorical("bidirectional", [False, True])
    fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.2)
    base_config["model"]["modal_nets"]["forces"]["hidden_size"] = hidden_size
    base_config["model"]["modal_nets"]["forces"]["n_layers"] = n_layers
    base_config["model"]["modal_nets"]["forces"]["rnn_dropout"] = rnn_dropout
    base_config["model"]["modal_nets"]["forces"]["bidirectional"] = bidirectional
    base_config["model"]["modal_nets"]["forces"]["fc_dropout"] = fc_dropout

    # 8) InceptionTimePlus hyperparams
    #nf = trial.suggest_categorical("nf", [16, 32, 64])
    #depth = trial.suggest_int("depth", 6, 12)
    #bottleneck = trial.suggest_categorical("bottleneck", [True, False])
    #base_config["model"]["modal_nets"]["forces"]["nf"] = nf
    #base_config["model"]["modal_nets"]["forces"]["depth"] = depth
    #base_config["model"]["modal_nets"]["forces"]["bottleneck"] = bottleneck

    # 8) RESCNN hyperparameters (commented out)
    #coord = trial.suggest_categorical("coord", [True, False])
    #separable = trial.suggest_categorical("separable", [True, False])
    #zero_norm = trial.suggest_categorical("zero_norm", [True, False])
    #base_config["model"]["modal_nets"]["forces"]["coord"] = coord
    #base_config["model"]["modal_nets"]["forces"]["separable"] = separable
    #base_config["model"]["modal_nets"]["forces"]["zero_norm"] = zero_norm

    # 9) Weighted Metric (Alpha)
    alpha = trial.suggest_float("alpha_weight", 0.0, 0.4)
    base_config["model"]["weight_f1"] = alpha
    base_config["model"]["weight_r2"] = 1.0 - alpha

    # New hyperparameters for data augmentation and transforms
    force_augment_times = trial.suggest_int("force_augment_times", 1, 2)
    base_config["data"]["force_augment_times"] = force_augment_times
    force_noise_magnitude = trial.suggest_float("force_noise_magnitude", 0.01, 0.1)
    base_config["data"]["force_noise_magnitude"] = force_noise_magnitude
    gaussian_blur_kernel = trial.suggest_categorical("gaussian_blur_kernel", [13, 15, 17, 19])
    base_config["data"]["image_transform_train"]["transforms"][3]["transforms"][0]["kernel_size"] = gaussian_blur_kernel

    # Train once and return the metric (pass trial for pruning)
    score = train_once(base_config, trial=trial)
    return score


def main():
    """
    Main function to perform Optuna-based hyperparameter tuning.

    Loads the configuration, creates an Optuna study, runs the optimization over a
    number of trials, prints the best parameters and metric, and saves them to a YAML file.
    """
    args = parse_args()  # parse_args() reads command-line arguments, including --config <yaml_file>
    base_config = load_config(args.config)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(),
        pruner=MedianPruner(n_warmup_steps=5, n_startup_trials=5)
    )
    study.optimize(lambda trial: objective(trial, base_config), n_trials=args.n_trials)
    print("Optuna study completed!")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")

    best_metric = study.best_value
    best_params = study.best_params
    output_dict = {
        "best_metric": float(best_metric),
        "best_params": best_params
    }
    best_params_file = "optuna_best_params_Densenet121_GRU.yaml"
    with open(best_params_file, "w") as f:
        yaml.safe_dump(output_dict, f)
    log.info(f"Best parameters and metric saved to: {best_params_file}")


if __name__ == "__main__":
    main()
