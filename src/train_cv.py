# File: train_cv.py

from typing import List, Optional, Tuple
import copy
import hydra
import lightning as L
import pyrootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

# Setup project root (so local imports and path resolution work)
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Local imports AFTER pyrootutils
from src import utils
from src.data.multitask_datamodule_cv import MultiTaskDataModule
from src.data.components.multitask_dataset import MultiTaskWindingDataset

import numpy as np
from torch.utils.data import random_split
from sklearn.model_selection import KFold

log = utils.get_pylogger(__name__)

# Initialize any custom resolvers for OmegaConf if you have them
for name, resolver in utils.all_custom_resolvers.items():
    OmegaConf.register_new_resolver(name, resolver)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """
    This function ALWAYS performs K-fold cross-validation.
    The old single-split logic is removed.
    1) Perform K-Fold CV => gather each fold's val/f1_best.
    2) Print mean ± std of val/f1_best across folds.
    3) Retrain on the entire train_val dataset => final model => test once.
    4) Print final test/f1 and compare to mean CV f1_best if available.
    """

    # 1) Set seed if given
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # 2) Instantiate transforms
    log.info("Instantiating transforms from config...")
    transform_train = hydra.utils.instantiate(cfg.data.image_transform_train)
    transform_val   = hydra.utils.instantiate(cfg.data.image_transform_val)
    transform_test  = hydra.utils.instantiate(cfg.data.image_transform_test)

    # 3) Always do cross-validation
    log.info("===== Always Running Cross-Validation (CV) =====")
    metric_dict, object_dict = _cross_val_training(
        cfg, transform_train, transform_val, transform_test
    )

    return metric_dict, object_dict


def _cross_val_training(
    cfg: DictConfig,
    transform_train,
    transform_val,
    transform_test
) -> Tuple[dict, dict]:
    """
    K-Fold cross-validation on the entire 'train_val' dataset, then a single final test.
    Steps:
      1) Create a 'full_dataset' (train_val) with no augmentation for splitting.
      2) Perform K-Fold, each fold trains a model on fold_{train} and validates on fold_{val}.
      3) Collect val/f1_best from each fold => compute mean/stdev.
      4) Train a final model on the entire train_val dataset (no val).
      5) Test once with that final model => compare final test/f1 vs. mean CV f1.
    """

    # 1) Load the entire train_val dataset with NO augmentation for splitting
    log.info("Creating the full_dataset (train_val) with val transform (no augmentation).")
    full_dataset = MultiTaskWindingDataset(
        root_dir=cfg.data.data_dir,
        subset="train_val",
        csv_filename=cfg.data.csv_filename_trainval,
        coil_images_indices=cfg.data.coil_images_indices,
        coil_force_column=cfg.data.coil_force_column,
        class_label_column=cfg.data.class_label_column,
        regression_label_column=cfg.data.regression_label_column,
        material_batch=cfg.data.material_batch,
        downsample_factor=cfg.data.downsample_factor,
        num_force_layers=cfg.data.num_force_layers,
        force_augment=False,
        force_augment_times=0,
        image_transform=transform_val,  # No augmentation => just center crop, etc.
        force_noise_magnitude=0.0,
    )
    n_samples = len(full_dataset)
    indices = np.arange(n_samples)

    # 2) Create KFold object
    k = cfg.cross_validation.get("n_splits", 5)
    shuffle = cfg.cross_validation.get("shuffle", True)
    seed = cfg.cross_validation.get("seed", 42)
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=seed)

    fold_val_f1_bests = []

    # 3) Loop over folds
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        log.info(f"\n===== Fold {fold_idx + 1} / {k} =====")

        # 3.1) Build a custom datamodule for this fold
        dm = _make_fold_datamodule(
            cfg, full_dataset, train_idx, val_idx, transform_train, transform_val
        )

        # 3.2) Instantiate model
        model: LightningModule = hydra.utils.instantiate(cfg.model)
        if cfg.get("compile"):
            model = torch.compile(model)

        # 3.3) Instantiate callbacks/loggers and trainer
        callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))
        logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))
        trainer: Trainer = hydra.utils.instantiate(
            cfg.get("trainer"),
            callbacks=callbacks,
            logger=logger
        )

        # 3.4) Train on this fold
        trainer.fit(model=model, datamodule=dm)

        # 3.5) Validate on this fold's val set
        trainer.validate(model=model, datamodule=dm, verbose=False)
        # We'll get the best val_f1 from the trainer callback metrics
        val_f1_best = trainer.callback_metrics.get("val/f1_best", None)
        if val_f1_best is not None:
            fold_val_f1_bests.append(float(val_f1_best.cpu().item()))
        log.info(f"Fold {fold_idx + 1} => val/f1_best: {val_f1_best}")

    # 4) Summarize CV results
    if len(fold_val_f1_bests) > 0:
        mean_f1_best = float(np.mean(fold_val_f1_bests))
        std_f1_best = float(np.std(fold_val_f1_bests))
        log.info(
            f"=== Cross-validation val/f1_best across {k} folds: "
            f"{mean_f1_best:.4f} ± {std_f1_best:.4f}"
        )
    else:
        mean_f1_best, std_f1_best = None, None
        log.warning("No val/f1_best metrics found. Did you log `val/f1_best`?")

    # 5) Train on ENTIRE train_val => final model => test
    log.info("\nCross-validation complete. Now training on ALL train_val data (no val split) for final test.")
    dm_final = _make_datamodule_entire_train(
        cfg, full_dataset, transform_train, transform_val, transform_test
    )

    model_final: LightningModule = hydra.utils.instantiate(cfg.model)
    if cfg.get("compile"):
        model_final = torch.compile(model_final)

    callbacks_final: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))
    logger_final: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    # skip val sanity check => we have no val set in final training
    trainer_final: Trainer = hydra.utils.instantiate(
        cfg.get("trainer"),
        callbacks=callbacks_final,
        logger=logger_final,
        num_sanity_val_steps=0
    )

    trainer_final.fit(model=model_final, datamodule=dm_final)

    final_test_metrics = {}
    if cfg.get("test"):
        log.info("Starting single final test with best ckpt from final training.")
        ckpt_path = trainer_final.checkpoint_callback.best_model_path
        if ckpt_path == "":
            ckpt_path = None
        trainer_final.test(model=model_final, datamodule=dm_final, ckpt_path=ckpt_path)
        final_test_metrics = trainer_final.callback_metrics
        log.info(f"Final best ckpt path: {ckpt_path}")

    test_f1 = final_test_metrics.get("test/f1", None)
    if test_f1 is not None and mean_f1_best is not None:
        log.info(
            f"Comparison: CV mean val/f1_best = {mean_f1_best:.4f}, final test/f1 = {test_f1:.4f}"
        )

    # Return a dict with CV stats + final test metrics
    metric_dict = {
        "cv_mean_f1_best": mean_f1_best,
        "cv_std_f1_best": std_f1_best,
        **final_test_metrics,
    }
    object_dict = {}
    return metric_dict, object_dict


def _make_fold_datamodule(
    cfg: DictConfig,
    full_dataset: MultiTaskWindingDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    xform_train,
    xform_val
) -> LightningDataModule:
    """
    Builds a mini DataModule for one fold. We replicate the logic of your MultiTaskDataModule's
    splitting, but do it manually with train_idx/val_idx.
    """
    # 1) Subset metadata
    train_meta = full_dataset.metadata.iloc[train_idx].reset_index(drop=True)
    val_meta   = full_dataset.metadata.iloc[val_idx].reset_index(drop=True)

    # 2) Create train/val subsets with the appropriate transforms
    train_dataset = MultiTaskWindingDataset(
        root_dir=cfg.data.data_dir,
        subset="train_val",
        csv_filename=cfg.data.csv_filename_trainval,
        coil_images_indices=cfg.data.coil_images_indices,
        coil_force_column=cfg.data.coil_force_column,
        class_label_column=cfg.data.class_label_column,
        regression_label_column=cfg.data.regression_label_column,
        material_batch=cfg.data.material_batch,
        downsample_factor=cfg.data.downsample_factor,
        num_force_layers=cfg.data.num_force_layers,
        force_augment=cfg.data.force_augment,
        force_augment_times=cfg.data.force_augment_times,
        image_transform=xform_train,  # augmented
        force_noise_magnitude=cfg.data.force_noise_magnitude,
    )
    train_dataset.metadata = train_meta

    val_dataset = MultiTaskWindingDataset(
        root_dir=cfg.data.data_dir,
        subset="train_val",
        csv_filename=cfg.data.csv_filename_trainval,
        coil_images_indices=cfg.data.coil_images_indices,
        coil_force_column=cfg.data.coil_force_column,
        class_label_column=cfg.data.class_label_column,
        regression_label_column=cfg.data.regression_label_column,
        material_batch=cfg.data.material_batch,
        downsample_factor=cfg.data.downsample_factor,
        num_force_layers=cfg.data.num_force_layers,
        force_augment=False,
        force_augment_times=0,
        image_transform=xform_val,  # no augmentation
        force_noise_magnitude=0.0,
    )
    val_dataset.metadata = val_meta

    class _FoldDataModule(LightningDataModule):
        def __init__(self, train_ds, val_ds, cfg):
            super().__init__()
            self.train_ds = train_ds
            self.val_ds   = val_ds
            self.cfg      = cfg
            self.test_ds  = None

        def setup(self, stage=None):
            pass

        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                self.train_ds,
                batch_size=self.cfg.data.batch_size,
                shuffle=True,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

        def val_dataloader(self):
            return torch.utils.data.DataLoader(
                self.val_ds,
                batch_size=self.cfg.data.batch_size,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

        def test_dataloader(self):
            return []

    return _FoldDataModule(train_dataset, val_dataset, cfg)


def _make_datamodule_entire_train(
    cfg: DictConfig,
    full_dataset: MultiTaskWindingDataset,
    xform_train,
    xform_val,
    xform_test
) -> LightningDataModule:
    """
    Create a DataModule that trains on the ENTIRE 'train_val' dataset (no val split),
    but still uses the normal test set from MultiTaskDataModule for final testing.
    """
    class _EntireTrainDataModule(MultiTaskDataModule):
        def setup(self, stage=None):
            if stage == "fit" or stage is None:
                # entire train_val => with train transform
                from src.data.components.multitask_dataset import MultiTaskWindingDataset
                self.train_dataset = MultiTaskWindingDataset(
                    root_dir=self.hparams.data_dir,
                    subset="train_val",
                    csv_filename=self.hparams.csv_filename_trainval,
                    coil_images_indices=self.hparams.coil_images_indices,
                    coil_force_column=self.hparams.coil_force_column,
                    class_label_column=self.hparams.class_label_column,
                    regression_label_column=self.hparams.regression_label_column,
                    material_batch=self.hparams.material_batch,
                    downsample_factor=self.hparams.downsample_factor,
                    num_force_layers=self.hparams.num_force_layers,
                    force_augment=self.hparams.force_augment,
                    force_augment_times=self.hparams.force_augment_times,
                    image_transform=xform_train,
                    force_noise_magnitude=self.hparams.force_noise_magnitude,
                )
                # No val set
                self.val_dataset = None

            if stage == "test" or stage is None:
                # normal logic for test dataset
                super().setup(stage="test")
                if self.test_dataset is not None:
                    self.test_dataset.image_transform = xform_test

        def val_dataloader(self):
            return []

    dm = _EntireTrainDataModule(
        data_dir=cfg.data.data_dir,
        csv_filename_trainval=cfg.data.csv_filename_trainval,
        csv_filename_test=cfg.data.csv_filename_test,
        coil_images_indices=cfg.data.coil_images_indices,
        coil_force_column=cfg.data.coil_force_column,
        class_label_column=cfg.data.class_label_column,
        regression_label_column=cfg.data.regression_label_column,
        batch_size=cfg.data.batch_size,
        val_split=0.0,
        num_workers=cfg.data.num_workers,
        downsample_factor=cfg.data.downsample_factor,
        num_force_layers=cfg.data.num_force_layers,
        force_augment=cfg.data.force_augment,
        force_augment_times=cfg.data.force_augment_times,
        force_noise_magnitude=cfg.data.force_noise_magnitude,
        material_batch=cfg.data.material_batch,
        image_transform_train=xform_train,
        image_transform_val=xform_val,
        image_transform_test=xform_test,
    )
    return dm


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # Apply optional utilities (e.g., prompting for tags, printing config, etc.)
    utils.extras(cfg)

    # Run cross-validation training
    metric_dict, _ = train(cfg)

    # Optionally retrieve a single optimized metric if needed for e.g. HPO
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict,
        metric_name=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()
