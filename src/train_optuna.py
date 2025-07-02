"""
This script sets up an Optuna-based hyperparameter tuning pipeline using PyTorch Lightning.
It handles configuration instantiation, callback and logger creation, training, and
evaluation of a multi-task model. Hyperparameters are tuned via Optuna and the best
parameters are saved to a YAML file.

The main components are:
    - Recursive instantiation of objects from configuration dictionaries.
    - Creation of callbacks and loggers for training.
    - A train_once() function to run a single training session.
    - An objective function for Optuna to optimize.
    - A main() function to run the entire process.
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
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, ModelSummary
)
from lightning.pytorch.loggers import WandbLogger  # or general pl_loggers if you prefer
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
# Import the pruning callback:
from optuna.integration import PyTorchLightningPruningCallback

# Import your custom code
from src.data.multitask_datamodule import MultiTaskDataModule
from src.models.multitask_module import MultiTaskModule
from src.utils import get_pylogger

log = get_pylogger(__name__)


###############################################################################
# 1) Mini-Recursive "instantiate" to handle _target_ (and _partial_) keys
###############################################################################
def import_class(path: str):
    """
    Import and return a class from its full path.

    Splits a full path like 'torchvision.transforms.Compose' into its module and class
    parts, then imports and returns the class.

    Args:
        path (str): The full path to the class.

    Returns:
        type: The class that was imported.
    """
    parts = path.split(".")
    module_path = ".".join(parts[:-1])
    class_name = parts[-1]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def instantiate_recursive(cfg):
    """
    Recursively instantiate objects defined in the configuration dictionary.

    If a dictionary contains the key '_target_', the function will import and instantiate
    the corresponding class. If '_partial_' is set to True, a functools.partial is returned
    instead of a fully instantiated object.

    Args:
        cfg (Any): The configuration, which may be a dict, list, or base type.

    Returns:
        Any: The instantiated configuration with all targets replaced by actual objects.
    """
    if isinstance(cfg, list):
        # Recursively instantiate each item in the list.
        return [instantiate_recursive(item) for item in cfg]

    elif isinstance(cfg, dict):
        # If this dict has a _target_, import and instantiate the object.
        if "_target_" in cfg:
            cls = import_class(cfg["_target_"])
            partial_mode = cfg.get("_partial_", False)
            # Prepare constructor kwargs (ignoring _target_ and _partial_ keys).
            kwargs = {
                k: instantiate_recursive(v)
                for k, v in cfg.items()
                if k not in ("_target_", "_partial_")
            }
            if partial_mode:
                # Return a partial, so that some parameters may be supplied later.
                return functools.partial(cls, **kwargs)
            else:
                return cls(**kwargs)
        else:
            # Recursively instantiate all dictionary values.
            return {k: instantiate_recursive(v) for k, v in cfg.items()}

    else:
        # Base case: if cfg is a base type (str, int, float, etc.), return it as is.
        return cfg


###############################################################################
# 2) Argument Parsing
###############################################################################
def parse_args():
    """
    Parse command-line arguments.

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
    Load a YAML configuration file and return its contents as a dictionary.

    Args:
        cfg_path (str): The path to the YAML configuration file.

    Returns:
        dict: The configuration dictionary.
    """
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    return config


###############################################################################
# 3) Create Callbacks & Logger
###############################################################################
def create_callbacks(cfg):
    """
    Create a list of callbacks based on the configuration.

    The function checks for keys (e.g., 'model_checkpoint', 'early_stopping')
    in the callbacks section of the configuration.

    Args:
        cfg (dict): The configuration dictionary.

    Returns:
        list: A list of instantiated callbacks.
    """
    cb_list = []

    # ModelCheckpoint callback
    if "model_checkpoint" in cfg["callbacks"]:
        mc_cfg = cfg["callbacks"]["model_checkpoint"]
        checkpoint_cb = ModelCheckpoint(
            dirpath=mc_cfg["dirpath"],
            filename=mc_cfg["filename"],
            monitor=mc_cfg["monitor"],
            mode=mc_cfg["mode"],
            save_last=mc_cfg["save_last"],
            auto_insert_metric_name=mc_cfg["auto_insert_metric_name"]
        )
        cb_list.append(checkpoint_cb)

    # EarlyStopping callback
    if "early_stopping" in cfg["callbacks"]:
        es_cfg = cfg["callbacks"]["early_stopping"]
        earlystop_cb = EarlyStopping(
            monitor=es_cfg["monitor"],
            patience=es_cfg["patience"],
            mode=es_cfg["mode"]
        )
        cb_list.append(earlystop_cb)

    # ModelSummary callback
    if "model_summary" in cfg["callbacks"]:
        ms_cfg = cfg["callbacks"]["model_summary"]
        model_summary_cb = ModelSummary(max_depth=ms_cfg["max_depth"])
        cb_list.append(model_summary_cb)

    # LearningRateMonitor callback (commented out by default)
    # if "lr_monitor" in cfg["callbacks"]:
    #     lr_cb = LearningRateMonitor(logging_interval='epoch')
    #     cb_list.append(lr_cb)

    return cb_list


def instantiate_logger(cfg):
    """
    Instantiate loggers as defined in the configuration.

    If the configuration contains a 'logger' key with a 'wandb' entry,
    a WandbLogger is instantiated.

    Args:
        cfg (dict): The configuration dictionary.

    Returns:
        list or None: A list of logger instances or None if no loggers are defined.
    """
    if "logger" not in cfg or cfg["logger"] is None:
        return None

    logger_cfg = cfg["logger"]
    if "wandb" in logger_cfg and logger_cfg["wandb"] is not None:
        wandb_cfg = logger_cfg["wandb"]
        wandb_logger = WandbLogger(
            project=wandb_cfg.get("project", "optuna-project"),
            name=wandb_cfg.get("name", None),
            save_dir=wandb_cfg.get("save_dir", "wandb_logs")
            # entity=wandb_cfg.get("entity", None),
        )
        return [wandb_logger]
    else:
        return None


###############################################################################
# 4) Main Training Function
###############################################################################
def train_once(config, trial=None):
    """
    Train the model once using the given configuration.

    Steps:
      1. Recursively instantiate the configuration to transform all _target_ keys.
      2. Build the DataModule, Model, Trainer, and other required components.
      3. Run the training (and optionally testing).
      4. Return the metric used for optimization.

    A pruning callback is attached if a trial object is provided.

    Args:
        config (dict): The base configuration dictionary.
        trial (optuna.trial.Trial, optional): An Optuna trial object for pruning.

    Returns:
        float: The final metric value for the trial.
    """
    # Fix seed if provided in the configuration.
    if config.get("seed") is not None:
        L.seed_everything(config["seed"], workers=True)

    # Recursively instantiate all objects in the configuration.
    instantiated_cfg = instantiate_recursive(config)

    # Instantiate DataModule and Model.
    datamodule = instantiated_cfg["data"]
    model = instantiated_cfg["model"]

    # Create callbacks and loggers.
    callbacks = create_callbacks(instantiated_cfg)
    loggers = instantiate_logger(instantiated_cfg)

    # Define the main metric for optimization.
    metric_name = "val/weighted_f1_r2"

    # Add the Optuna pruning callback if a trial is provided.
    if trial is not None:
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor=metric_name)
        callbacks.append(pruning_callback)

    # Set up the Trainer using the provided configuration.
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

    # Optionally compile the model with torch.compile().
    if instantiated_cfg.get("compile", False):
        log.info("Compiling model with torch.compile()!")
        model = torch.compile(model)

    # Run training if enabled.
    if instantiated_cfg.get("train", True):
        trainer.fit(model=model, datamodule=datamodule)

    # Retrieve validation metrics.
    val_metrics = trainer.callback_metrics
    final_score = val_metrics.get(metric_name, 0.0)
    return float(final_score)


###############################################################################
# 5) Objective Function and Main
###############################################################################
def objective(trial, base_config):
    """
    Objective function for Optuna hyperparameter tuning.

    This function modifies the base configuration in-place using suggestions from
    the Optuna trial, then calls train_once() to train the model. The resulting metric
    is returned to Optuna for optimization.

    Args:
        trial (optuna.trial.Trial): The trial object for hyperparameter suggestions.
        base_config (dict): The base configuration dictionary.

    Returns:
        float: The metric value from the training run.
    """
    # 1) Optimizer selection.
    optimizer_name = trial.suggest_categorical(
        "optimizer_name", ["torch.optim.Adam", "torch.optim.AdamW"]
    )
    base_config["model"]["optimizer_cfg"]["_target_"] = optimizer_name

    # 2) Learning rate.
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    base_config["model"]["optimizer_cfg"]["lr"] = lr

    # 3) Weight decay.
    weight_decay = trial.suggest_float("weight_decay", 1e-9, 1e-4, log=True)
    base_config["model"]["optimizer_cfg"]["weight_decay"] = weight_decay

    # 4) Unfreeze layers (for DenseNet).
    unfreeze_count = trial.suggest_int("unfreeze_count", 60, 110)
    base_config["model"]["modal_nets"]["images"]["unfreeze_layer_count"] = unfreeze_count

    # 5) Batch size.
    batch_size = trial.suggest_categorical("batch_size", [8])
    base_config["data"]["batch_size"] = batch_size

    # 6) Learning rate scheduler factor and patience.
    factor = trial.suggest_float("factor", 0.1, 0.9)
    patience = trial.suggest_categorical("patience", [2, 4, 6, 8])
    base_config["model"]["scheduler_cfg"]["factor"] = factor
    base_config["model"]["scheduler_cfg"]["patience"] = patience

    # 7) Fusion type selection.
    fusion_type = trial.suggest_categorical("fusion_type", ["concat", "mean"])
    base_config["model"]["fusion_type"] = fusion_type

    # 8) (Optional) Additional hyperparameters for force networks
    # Example for RESCNN:
    coord = trial.suggest_categorical("coord", [True, False])
    separable = trial.suggest_categorical("separable", [True, False])
    zero_norm = trial.suggest_categorical("zero_norm", [True, False])
    base_config["model"]["modal_nets"]["forces"]["coord"] = coord
    base_config["model"]["modal_nets"]["forces"]["separable"] = separable
    base_config["model"]["modal_nets"]["forces"]["zero_norm"] = zero_norm

    #InceptionTimePlus hyperparams
    #nf = trial.suggest_categorical("nf", [16, 32, 64])
    #depth = trial.suggest_int("depth", 6, 12)
    #bottleneck = trial.suggest_categorical("bottleneck", [True, False])
    #base_config["model"]["modal_nets"]["forces"]["nf"] = nf
    #base_config["model"]["modal_nets"]["forces"]["depth"] = depth
    #base_config["model"]["modal_nets"]["forces"]["bottleneck"] = bottleneck
 
    #GRU
    #hidden_size = trial.suggest_int("hidden_size", 100, 300)
    #n_layers = trial.suggest_int("n_layers", 1, 3)
    #rnn_dropout = trial.suggest_float("rnn_dropout", 0.0, 0.3)
    #bidirectional = trial.suggest_categorical("bidirectional", [False, True])
    #fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.2)

    #base_config["model"]["modal_nets"]["forces"]["hidden_size"] = hidden_size
    #base_config["model"]["modal_nets"]["forces"]["n_layers"] = n_layers
    #base_config["model"]["modal_nets"]["forces"]["rnn_dropout"] = rnn_dropout
    #base_config["model"]["modal_nets"]["forces"]["bidirectional"] = bidirectional
    #base_config["model"]["modal_nets"]["forces"]["fc_dropout"] = fc_dropout

    # 9) Weighted metric: combine F1 and R2.
    alpha = trial.suggest_float("alpha_weight", 0.0, 0.4)
    base_config["model"]["weight_f1"] = alpha
    base_config["model"]["weight_r2"] = 1.0 - alpha

    # Train the model once, passing the trial for pruning.
    score = train_once(base_config, trial=trial)

    return score


def main():
    """
    Main function to run the hyperparameter tuning using Optuna.

    Loads the configuration, creates an Optuna study, optimizes the objective
    over a number of trials, and saves the best parameters and metric to a YAML file.
    """
    args = parse_args()

    # Load configuration from YAML file.
    base_config = load_config(args.config)

    # Create an Optuna study (direction is set to maximize the metric).
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(),
        pruner=MedianPruner(n_warmup_steps=0, n_startup_trials=5)
    )

    # Optimize the objective function over the specified number of trials.
    study.optimize(lambda trial: objective(trial, base_config), n_trials=args.n_trials)

    # Log the results.
    log.info("Optuna study completed!")
    log.info(f"Best value: {study.best_value}")
    log.info(f"Best params: {study.best_params}")

    # Save the best parameters and metric to a YAML file.
    best_metric = study.best_value
    best_params = study.best_params

    output_dict = {
        "best_metric": float(best_metric),
        "best_params": best_params
    }

    best_params_file = "optuna_best_params_densenet_Rescnn_withpruner.yaml"
    with open(best_params_file, "w") as f:
        yaml.safe_dump(output_dict, f)

    log.info(f"Best parameters and metric saved to: {best_params_file}")


if __name__ == "__main__":
    main()
