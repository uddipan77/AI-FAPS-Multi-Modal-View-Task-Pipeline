import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import logging

# tsai transformations
from tsai.data.preprocessing import TSNormalize
from tsai.data.transforms import TSMagMulNoise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoilDataset(Dataset):
    def __init__(
        self,
        data_dir,
        subset,
        csv_file_path,
        material_batch=1,
        downsample_factor=4,
        num_layers=5,
        augment=False,
        augment_times=0,
        indices=None
    ):
        """
        Dataset class for coil data that combines downsampling + augmentation.

        Args:
            data_dir (str): Root directory containing the data.
            subset (str): Subset folder name (e.g., "train_val", "test").
            csv_file_path (str): Path to the CSV file with coil metadata.
            material_batch (int): Material batch to filter the data.
            downsample_factor (int): Factor by which to downsample the force data.
            num_layers (int): Number of subsegments to split the force data into.
            augment (bool): Whether to apply data augmentation.
            augment_times (int): Number of augmented copies per original sample.
            indices (list or np.ndarray, optional): Indices for subsetting the CSV data.
        """
        self.data_dir = data_dir
        self.subset = subset
        self.csv_data = pd.read_csv(csv_file_path)

        # Filter by material_batch
        self.csv_data = self.csv_data[self.csv_data["material_batch"] == material_batch].reset_index(drop=True)

        # If a list of indices is provided (e.g., from a random_split), subset the CSV
        if indices is not None:
            self.csv_data = self.csv_data.iloc[indices].reset_index(drop=True)

        # Downsampling & segmentation parameters
        self.downsample_factor = downsample_factor
        self.num_layers = num_layers

        # Augmentation parameters
        self.augment = augment
        self.augment_times = augment_times
        # Define the transform for augmentation
        self.augment_transform = TSMagMulNoise(magnitude=0.1) if self.augment else None

        # Normalization transform (applied to each subsegment)
        self.force_transform = TSNormalize(min=None, max=None, range=[-1, 1])

    def __len__(self):
        """
        If augmentation is turned on (augment=True) and augment_times > 0,
        the length is multiplied by (1 + augment_times).
        """
        if self.augment and self.augment_times > 0:
            return len(self.csv_data) * (1 + self.augment_times)
        return len(self.csv_data)

    def __getitem__(self, idx):
        """
        Returns:
            stacked_curves (torch.Tensor): Shape = [num_layers, segment_length].
            resistance_label (torch.Tensor): DC resistance value.
        """
        # If augmenting, figure out if this index corresponds to an "original" sample or an "augmented" copy
        if self.augment and self.augment_times > 0:
            original_idx = idx // (1 + self.augment_times)
            copy_idx = idx % (1 + self.augment_times)  # 0 = original, 1..augment_times = augmented copy
            is_augmented = copy_idx > 0
        else:
            original_idx = idx
            is_augmented = False

        row = self.csv_data.iloc[original_idx]

        coil_id = row["coil_id"]
        resistance_label = torch.tensor(row["label_dc_resistance"], dtype=torch.float32)

        # Load the pickle file containing the force data
        curve_file_path = os.path.join(
            self.data_dir, self.subset, coil_id, f"{coil_id}_force_and_displacement_curves.pkl"
        )
        try:
            curve_data = pd.read_pickle(curve_file_path)
        except Exception as e:
            logger.error(f"Error loading PKL file for coil_id {coil_id}: {e}")
            raise e

        # Extract force data
        force_values = curve_data.get("force")
        if force_values is None:
            raise ValueError(f"'force' data not found in PKL file for coil_id {coil_id}.")

        force_values = np.array(force_values, dtype=np.float32)

        # ----- 1) Downsampling -----
        if self.downsample_factor > 1:
            force_values = self.downsample_time_series(force_values, self.downsample_factor)

        # ----- 2) Ensure divisibility by num_layers -----
        # If the total length isn't divisible by num_layers, trim the end
        if len(force_values) % self.num_layers != 0:
            trim_amount = len(force_values) % self.num_layers
            force_values = force_values[:-trim_amount]

        # ----- 3) Split force data into num_layers subsegments -----
        subsegments = np.array_split(force_values, self.num_layers)

        curves = []
        for seg_id, layer_data_np in enumerate(subsegments):
            if len(layer_data_np) == 0:
                # If any subsegment is empty, fill with a zero
                layer_data_np = np.zeros(1, dtype=np.float32)

            # Convert to Torch Tensor with a "channel" dimension
            layer_data = torch.tensor(layer_data_np, dtype=torch.float32).unsqueeze(0)
            
            # Apply augmentation if needed (only for augmented copies)
            if is_augmented and self.augment_transform:
                layer_data = self.augment_transform(layer_data)

            # Apply normalization
            layer_data = self.force_transform(layer_data)

            # Remove channel dimension
            layer_data = layer_data.squeeze(0)
            curves.append(layer_data)

        # Stack subsegments into a single tensor: shape [num_layers, segment_length]
        stacked_curves = torch.stack(curves, dim=0)

        return stacked_curves, resistance_label

    @staticmethod
    def downsample_time_series(data, factor):
        """
        Downsamples the time series data by averaging over non-overlapping windows.
        Ensures the length of 'data' is divisible by 'factor' by trimming the extra points.
        """
        n = len(data)
        trimmed_length = n - (n % factor)
        if trimmed_length != n:
            logger.info(f"Trimming {n - trimmed_length} points for downsampling by factor={factor}.")
        data_trimmed = data[:trimmed_length]

        data_reshaped = data_trimmed.reshape(-1, factor)
        data_downsampled = data_reshaped.mean(axis=1)

        return data_downsampled
