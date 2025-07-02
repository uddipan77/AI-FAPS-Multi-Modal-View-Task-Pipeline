import os
import copy
import torch
import numpy as np
import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from src.data.components.multitask_dataset import MultiTaskWindingDataset

class MultiTaskDataModule(LightningDataModule):
    """
    DataModule that returns samples with both classification and regression labels.
    """

    def __init__(
        self,
        data_dir: str,
        csv_filename_trainval: str,
        csv_filename_test: str,
        coil_images_indices: list,
        coil_force_column: str,
        class_label_column: str,
        regression_label_column: str,
        batch_size: int = 8,
        val_split: float = 0.2,
        num_workers: int = 4,
        downsample_factor: int = 1,
        num_force_layers: int = 5,
        force_augment: bool = False,
        force_augment_times: int = 0,
        force_noise_magnitude: float = 0.0,
        material_batch: int = 1,
        image_transform_train=None,
        image_transform_val=None,
        image_transform_test=None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.csv_filename_trainval = csv_filename_trainval
        self.csv_filename_test = csv_filename_test
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # 1) Base dataset
            base_dataset = MultiTaskWindingDataset(
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
                force_augment=False,
                force_augment_times=0,
                image_transform=self.hparams.image_transform_val,
                force_noise_magnitude=0.0,
            )

            # If val_split==0 or dataset is forcibly overridden, we might skip splitting.
            if self.val_split > 0:
                val_size = int(len(base_dataset) * self.hparams.val_split)
                train_size = len(base_dataset) - val_size
                split_train, split_val = random_split(base_dataset, [train_size, val_size])
                train_indices, val_indices = split_train.indices, split_val.indices

                # 2) Train dataset with augmentation
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
                    image_transform=self.hparams.image_transform_train,
                    force_noise_magnitude=self.hparams.force_noise_magnitude,
                )
                self.train_dataset.metadata = self.train_dataset.metadata.iloc[train_indices].reset_index(drop=True)

                # 3) Val dataset
                self.val_dataset = MultiTaskWindingDataset(
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
                    force_augment=False,
                    force_augment_times=0,
                    image_transform=self.hparams.image_transform_val,
                    force_noise_magnitude=0.0,
                )
                self.val_dataset.metadata = self.val_dataset.metadata.iloc[val_indices].reset_index(drop=True)
            else:
                # val_split=0 => entire set is training, no val
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
                    image_transform=self.hparams.image_transform_train,
                    force_noise_magnitude=self.hparams.force_noise_magnitude,
                )
                self.val_dataset = None

        if stage == "test" or stage is None:
            # Test dataset
            self.test_dataset = MultiTaskWindingDataset(
                root_dir=self.hparams.data_dir,
                subset="test",
                csv_filename=self.hparams.csv_filename_test,
                coil_images_indices=self.hparams.coil_images_indices,
                coil_force_column=self.hparams.coil_force_column,
                class_label_column=self.hparams.class_label_column,
                regression_label_column=self.hparams.regression_label_column,
                material_batch=self.hparams.material_batch,
                downsample_factor=self.hparams.downsample_factor,
                num_force_layers=self.hparams.num_force_layers,
                force_augment=False,
                force_augment_times=0,
                image_transform=self.hparams.image_transform_test,
                force_noise_magnitude=0.0,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
