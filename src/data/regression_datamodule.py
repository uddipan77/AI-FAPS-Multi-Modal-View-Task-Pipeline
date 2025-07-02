from torch.utils.data import DataLoader, random_split
#import pytorch_lightning as pl
from lightning import LightningDataModule
import numpy as np

# Import the regression dataset class
from src.data.components.regression_dataset import CoilDataset

class CoilDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        subset,
        csv_file_path,
        batch_size=8,
        val_split=0.2,
        material_batch=1,
        downsample_factor=4,
        num_layers=5,
        augment_times=1,  # e.g., 1 or 2 or 3, etc.
        test_subset=None,
        test_csv_file_path=None,
    ):
        """
        DataModule for regression tasks.

        Args:
            data_dir (str): Root directory containing the data.
            subset (str): Which subset to load (e.g., 'train_val').
            csv_file_path (str): Path to the CSV file with coil metadata.
            batch_size (int): Batch size for DataLoader.
            val_split (float): Fraction of data to use for validation.
            material_batch (int): Material batch number to filter the data.
            downsample_factor (int): Factor by which to downsample the force data.
            num_layers (int): Number of subsegments per sample.
            augment_times (int): Number of augmented copies per original sample in training.
            test_subset (str, optional): Subset name for testing (e.g., 'test').
            test_csv_file_path (str, optional): Path to the test CSV file.
        """
        super().__init__()
        self.data_dir = data_dir
        self.subset = subset
        self.csv_file_path = csv_file_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.material_batch = material_batch
        self.downsample_factor = downsample_factor
        self.num_layers = num_layers
        self.augment_times = augment_times
        self.test_subset = test_subset
        self.test_csv_file_path = test_csv_file_path

    def setup(self, stage=None):
        """
        Set up datasets based on the stage (fit, validate, test, predict).
        """
        if stage == "fit" or stage is None:
            # Create a base dataset for indexing
            base_dataset = CoilDataset(
                data_dir=self.data_dir,
                subset=self.subset,
                csv_file_path=self.csv_file_path,
                material_batch=self.material_batch,
                downsample_factor=self.downsample_factor,
                num_layers=self.num_layers,
                augment=False,
                augment_times=0,
            )

            val_size = int(len(base_dataset) * self.val_split)
            train_size = len(base_dataset) - val_size

            # Split into train and validation subsets
            train_indices, val_indices = random_split(
                base_dataset, [train_size, val_size], generator=None
            )
            train_indices, val_indices = train_indices.indices, val_indices.indices

            # Create training dataset with augmentation
            self.train_dataset = CoilDataset(
                data_dir=self.data_dir,
                subset=self.subset,
                csv_file_path=self.csv_file_path,
                material_batch=self.material_batch,
                downsample_factor=self.downsample_factor,
                num_layers=self.num_layers,
                augment=True,
                augment_times=self.augment_times,
                indices=train_indices,
            )

            # Validation dataset (no augmentation)
            self.val_dataset = CoilDataset(
                data_dir=self.data_dir,
                subset=self.subset,
                csv_file_path=self.csv_file_path,
                material_batch=self.material_batch,
                downsample_factor=self.downsample_factor,
                num_layers=self.num_layers,
                augment=False,
                augment_times=0,
                indices=val_indices,
            )

        if stage == "test" or stage is None:
            if self.test_subset and self.test_csv_file_path:
                self.test_dataset = CoilDataset(
                    data_dir=self.data_dir,
                    subset=self.test_subset,
                    csv_file_path=self.test_csv_file_path,
                    material_batch=self.material_batch,
                    downsample_factor=self.downsample_factor,
                    num_layers=self.num_layers,
                    augment=False,
                    augment_times=0,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
        raise ValueError("Test dataset is not set. Ensure 'test_subset' and 'test_csv_file_path' are provided in YAML.")

    def prepare_data(self):
        pass

    def prepare_data_per_node(self):
        pass

    def _log_hyperparams(self):
        return False
