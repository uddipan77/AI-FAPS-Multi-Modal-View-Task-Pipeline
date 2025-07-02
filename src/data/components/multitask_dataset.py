"""
Module for MultiTaskWindingDataset.

This module defines the MultiTaskWindingDataset class, which is a PyTorch
dataset that returns samples with:
  - a list of 4 images per sample,
  - force data tensor of shape [5, seg_len],
  - a classification label (0/1),
  - a regression label (float).

The dataset also applies optional data augmentations to the force data.
"""

import os
import copy
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tsai.data.preprocessing import TSNormalize
from tsai.data.transforms import TSMagMulNoise


class MultiTaskWindingDataset(Dataset):
    """
    A combined dataset that returns images, forces, classification and regression labels.

    Each sample in the dataset contains:
      - images (list of 4 images)
      - forces (a single tensor of shape [5, seg_len])
      - classification label (0/1)
      - regression label (float)
    """

    def __init__(
        self,
        root_dir: str,
        subset: str,
        csv_filename: str,
        coil_images_indices: list = [957, 958, 959, 960],
        coil_force_column: str = "force",
        class_label_column: str = "label_geom_error_overall",
        regression_label_column: str = "label_dc_resistance",
        material_batch: int = 1,
        downsample_factor: int = 4,  # e.g. 4 if you want ~2881 per segment
        num_force_layers: int = 5,
        force_augment: bool = False,
        force_augment_times: int = 0,
        image_transform=None,
        force_noise_magnitude: float = 0.0,
    ):
        """
        Initialize the MultiTaskWindingDataset.

        Args:
            root_dir (str): Root directory where data is stored.
            subset (str): Subset of the data (e.g., 'train_val' or 'test').
            csv_filename (str): Filename of the CSV containing metadata.
            coil_images_indices (list, optional): Indices for selecting coil images.
                Defaults to [957, 958, 959, 960].
            coil_force_column (str, optional): Column name for the coil force.
                Defaults to "force".
            class_label_column (str, optional): Column name for the classification label.
                Defaults to "label_geom_error_overall".
            regression_label_column (str, optional): Column name for the regression label.
                Defaults to "label_dc_resistance".
            material_batch (int, optional): Material batch to filter the dataset.
                Defaults to 1.
            downsample_factor (int, optional): Factor by which to downsample the force data.
                Defaults to 4.
            num_force_layers (int, optional): Number of segments (force layers) to produce.
                Defaults to 5.
            force_augment (bool, optional): Whether to apply force data augmentation.
                Defaults to False.
            force_augment_times (int, optional): Number of augmented copies per sample.
                Defaults to 0.
            image_transform (callable, optional): Transformation to apply to images.
                Defaults to None.
            force_noise_magnitude (float, optional): Magnitude of noise for force augmentation.
                Defaults to 0.0.
        """
        super().__init__()
        self.root_dir = root_dir
        self.subset = subset
        self.coil_images_indices = coil_images_indices
        self.coil_force_column = coil_force_column
        self.class_label_column = class_label_column
        self.regression_label_column = regression_label_column
        self.downsample_factor = downsample_factor
        self.num_force_layers = num_force_layers
        self.force_augment = force_augment
        self.force_augment_times = force_augment_times

        # 1) Read CSV
        csv_path = os.path.join(root_dir, subset, csv_filename)
        self.metadata = pd.read_csv(csv_path)

        # 2) Filter only the desired material batch
        self.metadata = self.metadata[
            self.metadata["material_batch"] == material_batch
        ].reset_index(drop=True)

        # 3) Basic transform for images
        self.image_transform = image_transform

        # 4) Force transform: normalization, plus optional noise
        self.force_normalize = TSNormalize(min=None, max=None, range=[-1, 1])
        self.force_augment_transform = (
            TSMagMulNoise(magnitude=force_noise_magnitude)
            if force_augment
            else None
        )

    def __len__(self):
        """
        Return the number of samples in the dataset.

        If force augmentation is enabled, each sample is repeated with augmented copies.

        Returns:
            int: Total number of samples (augmented copies included if applicable).
        """
        if self.force_augment and self.force_augment_times > 0:
            return len(self.metadata) * (1 + self.force_augment_times)
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Retrieve a sample corresponding to the given index.

        Each sample contains images, a force tensor, a classification label,
        and a regression label. Optionally, force augmentation is applied.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing:
                - "images": list of 4 image Tensors
                - "forces": Tensor of shape [5, seg_len]
                - "class_label": Classification label (Tensor)
                - "reg_label": Regression label (Tensor)
                - "coil_id": Identifier for the coil
        """
        # Handle "augmented copies"
        if self.force_augment and self.force_augment_times > 0:
            original_idx = idx // (1 + self.force_augment_times)
            copy_idx = idx % (1 + self.force_augment_times)
            is_augmented = (copy_idx > 0)
        else:
            original_idx = idx
            is_augmented = False

        row = self.metadata.iloc[original_idx]
        coil_id = row["coil_id"]

        # -------------------------
        # 1) Load images (4 images)
        # -------------------------
        images = []
        coil_images_folder = os.path.join(
            self.root_dir, self.subset, coil_id, coil_id + "_images"
        )
        for img_idx in self.coil_images_indices:
            img_path = os.path.join(
                coil_images_folder, f"{coil_id}_image_{img_idx:04d}.jpg"
            )
            img = Image.open(img_path).convert("RGB")
            if self.image_transform:
                img = self.image_transform(img)
            images.append(img)

        # -------------------------
        # 2) Load force data
        # -------------------------
        curve_file = os.path.join(
            self.root_dir, self.subset, coil_id, f"{coil_id}_force_and_displacement_curves.pkl"
        )
        curve_df = pd.read_pickle(curve_file)
        force_data = curve_df[self.coil_force_column].values.astype(np.float32)

        # Downsample if needed
        if self.downsample_factor > 1:
            force_data = self._downsample(force_data, self.downsample_factor)

        # Split into segments for each force layer (approx. seg_len per segment)
        segments = self._segment_force(force_data, self.num_force_layers)

        # Possibly apply augmentation and normalization
        channel_tensors = []
        for seg in segments:
            seg_t = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)  # shape: [1, seg_len]
            if is_augmented and self.force_augment_transform:
                seg_t = self.force_augment_transform(seg_t)
            seg_t = self.force_normalize(seg_t)
            seg_t = seg_t.squeeze(0)  # shape: [seg_len]
            channel_tensors.append(seg_t)

        # Stack to form a force tensor with shape [5, seg_len]
        force_tensor = torch.stack(channel_tensors, dim=0)

        # -------------------------
        # 3) Classification label
        # -------------------------
        class_label = torch.tensor(row[self.class_label_column], dtype=torch.long)

        # -------------------------
        # 4) Regression label
        # -------------------------
        regression_label = torch.tensor(row[self.regression_label_column], dtype=torch.float32)

        return {
            "images": images,           # list of 4 image Tensors
            "forces": force_tensor,     # tensor of shape [5, seg_len]
            "class_label": class_label,
            "reg_label": regression_label,
            "coil_id": coil_id,
        }

    def _downsample(self, data, factor):
        """
        Downsample the input data by taking the mean over non-overlapping windows.

        Args:
            data (numpy.ndarray): 1D array of force data.
            factor (int): Downsampling factor.

        Returns:
            numpy.ndarray: Downsampled force data.
        """
        n = len(data)
        trimmed_len = n - (n % factor)
        data = data[:trimmed_len]
        data = data.reshape(-1, factor).mean(axis=1)
        return data

    def _segment_force(self, force_data, num_layers):
        """
        Segment the force data into a given number of layers.

        Args:
            force_data (numpy.ndarray): 1D array of force data.
            num_layers (int): Number of segments (layers) to split the data into.

        Returns:
            list: A list of numpy.ndarray segments of the force data.
        """
        length = len(force_data)
        step = length // num_layers
        segments = []
        start = 0
        for i in range(num_layers):
            end = start + step
            if i == num_layers - 1:
                end = length
            seg = force_data[start:end]
            if len(seg) == 0:
                seg = np.zeros(1, dtype=np.float32)
            segments.append(seg)
            start = end
        return segments
