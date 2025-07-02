"""
Module for multitask data loading.

This module provides the MultiTaskDataModule class, which is a PyTorch Lightning
DataModule that returns samples with both classification and regression labels.
It leverages the MultiTaskWindingDataset to prepare training, validation, and test
datasets.
"""

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

    This DataModule prepares and serves the training, validation, and test datasets for
    multitask learning using the MultiTaskWindingDataset.
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
        # You may add seed or other parameters as needed
        **kwargs
    ):
        """
        Initialize the MultiTaskDataModule with the provided parameters.

        Args:
            data_dir (str): Root directory for the dataset.
            csv_filename_trainval (str): CSV filename for the train/validation data.
            csv_filename_test (str): CSV filename for the test data.
            coil_images_indices (list): List of indices for coil images.
            coil_force_column (str): Name of the column for coil force.
            class_label_column (str): Name of the column for classification labels.
            regression_label_column (str): Name of the column for regression labels.
            batch_size (int, optional): Batch size for DataLoader. Defaults to 8.
            val_split (float, optional): Fraction of data reserved for validation.
                                         Defaults to 0.2.
            num_workers (int, optional): Number of worker processes for DataLoader.
                                         Defaults to 4.
            downsample_factor (int, optional): Factor to downsample images. Defaults to 1.
            num_force_layers (int, optional): Number of force layers. Defaults to 5.
            force_augment (bool, optional): Whether to perform force augmentation.
                                            Defaults to False.
            force_augment_times (int, optional): Number of times to apply augmentation.
                                                 Defaults to 0.
            force_noise_magnitude (float, optional): Magnitude of force noise to add.
                                                       Defaults to 0.0.
            material_batch (int, optional): Batch size for materials. Defaults to 1.
            image_transform_train (optional): Image transformations for training.
            image_transform_val (optional): Image transformations for validation.
            image_transform_test (optional): Image transformations for testing.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.csv_filename_trainval = csv_filename_trainval
        self.csv_filename_test = csv_filename_test
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

        # These datasets will be defined in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """
        Set up the datasets for training, validation, and testing.

        Depending on the stage, this method initializes the corresponding datasets:
        - For stage "fit" (or None): Creates training and validation datasets.
        - For stage "test" (or None): Creates the test dataset.

        Args:
            stage (str, optional): Stage of setup ('fit' or 'test').
                                   If None, both training and testing datasets are set up.
        """
        if stage == "fit" or stage is None:
            # 1) Base dataset for train+val
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
            val_size = int(len(base_dataset) * self.hparams.val_split)
            train_size = len(base_dataset) - val_size

            train_indices, val_indices = random_split(
                base_dataset, [train_size, val_size]
            )
            train_indices, val_indices = train_indices.indices, val_indices.indices

            # 2) Create actual train dataset (with augmentation if desired)
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
            # Manually reduce to train_indices
            self.train_dataset.metadata = self.train_dataset.metadata.iloc[
                train_indices
            ].reset_index(drop=True)

            # 3) Create validation dataset
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
                force_augment=False,  # no augmentation in validation
                force_augment_times=0,
                image_transform=self.hparams.image_transform_val,
                force_noise_magnitude=0.0,
            )
            self.val_dataset.metadata = self.val_dataset.metadata.iloc[
                val_indices
            ].reset_index(drop=True)

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
        """
        Create and return the training DataLoader.

        Returns:
            DataLoader: The DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        Create and return the validation DataLoader.

        Returns:
            DataLoader: The DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """
        Create and return the test DataLoader.

        Returns:
            DataLoader: The DataLoader for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
