import os
import copy
import torch
import numpy as np
from rich.table import Table
from collections import Counter
from operator import itemgetter
from rich.console import Console
from torchvision.datasets import MNIST
from lightning import LightningDataModule
from typing import Any, Dict, Optional, Tuple
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset

from src.data.components.linear_winding_dataset import LinearWindingDataset
from src.utils.file_utils import copy_files_parallel


class LinearWindingDataModule(LightningDataModule):
    """LightningDataModule for Linear Winding dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html

    :param data_dir: Path to the data directory.
    :param batch_size: Number of samples in each batch.
    :param num_workers: Number of workers for the dataloaders.
    :param pin_memory: Whether to pin memory in dataloaders.
    :param train_val_split: Tuple with the train and validation split ratios. Default is 80:20.
    :param copy_data: Whether to copy the data to a temporary directory in the executing node. Useful for faster data loading in GPU nodes.
    :param overwrite: Whether to overwrite the data if it already exists. Only used if `copy_data` is True.
    :param coil_images_indices: List of indices for the coil images. Specific to the Linear Winding dataset.
    :param coil_force_columns: List of columns for the coil force. Specific to the Linear Winding dataset.
    :param train_transforms: Transforms to apply to the training dataset.
    :param val_transforms: Transforms to apply to the validation dataset.
    :param test_transforms: Transforms to apply to the test dataset.
    :param seed: Seed for reproducibility.

    """

    def __init__(
        self,
        data_dir: str = os.path.join('data', 'LinearWinding'),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_split: Tuple[float, float] = (0.8, 0.2),
        copy_data: bool = False,
        overwrite: bool = False,
        coil_images_indices: list = [957, 958, 959, 960],
        coil_force_columns: list = ["force"],
        train_transforms = None,
        val_transforms = None,
        test_transforms = None,
        seed: int = 42,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms if val_transforms else self.train_transforms
        self.test_transforms = test_transforms if test_transforms else self.train_transforms

        # datasets split
        self.train_dataset: Optional[Dataset] = None
        self.validation_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    @property
    def num_classes(self):
        # Return the number of classes in the dataset
        return 2
    
    def count_labels(self, dataset):
        """Count the number of samples for each class in the dataset.
        
        :param dataset: Dataset to count the labels.
        :return: Counter with the label counts.
        
        """
        label_counter = Counter()
        for data in dataset:
            label = data['labels']
            label_counter[label] += 1
        return label_counter

    def print_label_counts(self):
        """Print the sample counts per label for the train, validation and test datasets."""
        
        train_label_count = self.count_labels(self.train_dataset)
        val_label_count = self.count_labels(self.validation_dataset)
        test_label_count = self.count_labels(self.test_dataset)
        
        train_label_count = [train_label_count[label] for label in self.train_dataset.dataset.labels]
        val_label_count = [val_label_count[label] for label in self.validation_dataset.dataset.labels]
        test_label_count = [test_label_count[label] for label in self.test_dataset.labels]
        
        # Create a rich table with the sample counts
        table = Table(title="Sample counts by class")
        
        table.add_column("Dataset", style='table.header')
        for label in self.train_dataset.dataset.labels:
            table.add_column(f"Label {label}")
        table.add_column("Total Samples", justify="right")
        
        # Add the rows for each dataset, also convert to string for rich table
        table.add_row("Train", *list(map(str, train_label_count)), str(len(self.train_dataset)))
        table.add_row("Validation", *list(map(str, val_label_count)), str(len(self.validation_dataset)))
        table.add_row("Test", *list(map(str, test_label_count)), str(len(self.test_dataset)), end_section=True)
        
        console = Console()
        console.print(table)
    
    def prepare_data(self):
        """Prepare data for the experiment."""
        
        # Copy data to temporary directory if needed
        if torch.cuda.is_available() and self.hparams.copy_data:
            
            job_dir = os.path.join(os.environ['TMPDIR'], 'LinearWinding')
            
            # Filter regex for the files to copy
            filter_regex = '|'.join(f'_image_{index:04}.jpg' for index in self.hparams.coil_images_indices) + '|.csv|.pkl'
            
            # Copy the files in parallel - custom utility function
            copy_files_parallel(self.hparams.data_dir, job_dir, filter_regex=filter_regex, overwrite=self.hparams.overwrite)
            
            # Update the data directory to the temporary directory
            self.hparams.data_dir = job_dir
            print(self.hparams.data_dir)
        else:
            print(f'Not copying data! GPU: {torch.cuda.is_available()}, Copy: {self.hparams.copy_data}')
        

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.train_dataset and not self.validation_dataset and not self.test_dataset:
            
            # Setting up the entire train_val dataset
            train_val_dataset = LinearWindingDataset(
                root_dir=self.hparams.data_dir,
                subset="train_val",
                images_per_sample=self.hparams.coil_images_indices,
                curve_columns=self.hparams.coil_force_columns,
                label_column="label_geom_error_overall"
            )

         #note that only the map function and the dataloaders call the getitem fucntion from the dataset class
            # List of all labels in the train_val dataset (for stratified split)
            get_label = itemgetter('labels')  
            #itemgetter is a utility function from Python's operator module
            #Essentially, get_label is now a function that extracts the value with key 'labels'  from any dictionary passed to it.
            labels = list(map(get_label, train_val_dataset)) #mainly needed cause of tsratification
            #applies the get_label function to each item in train_val_dataset.
            #map calls getitem method of train_val_dataset which is the object of LinearWindingDataset
            #extract the 'labels' field from each sample (which is a dictionary) using the get_label function.
            #finally i convert it to a list and this will start from 0 till len(train_val_dataset)-1
            #len(train_val_dataset) calls the __len__ method of the LinearWindingDataset

            # Stratified split of train and validation datasets
            train_split, val_split = self.hparams.train_val_split
            indices = list(range(len(train_val_dataset)))
            #This line creates a list of indices ranging from 0 to len(train_val_dataset) - 1

            train_indices, val_indices, _, _ = train_test_split(
                indices, labels, test_size=val_split, stratify=labels, random_state=self.hparams.seed
            )
            #train_test_split returns train_indices, val_indices, train_labels and val_labels
            
            # Use Subset to create the train and validation datasets
            self.train_dataset = Subset(train_val_dataset, train_indices)
            self.validation_dataset = Subset(train_val_dataset, val_indices)
            
            # Deepcopy the datasets to avoid transforms affecting each other
            # Separate transforms for train and validation datasets
            self.train_dataset.dataset = copy.deepcopy(train_val_dataset)
            self.validation_dataset.dataset = copy.deepcopy(train_val_dataset)
            self.train_dataset.dataset.transform = self.train_transforms
            self.validation_dataset.dataset.transform = self.val_transforms

            # Create the test dataset
            self.test_dataset = LinearWindingDataset(
                root_dir=self.hparams.data_dir, 
                subset="test",
                images_per_sample=self.hparams.coil_images_indices,
                curve_columns=self.hparams.coil_force_columns,
                label_column="label_geom_error_overall",
                transforms=self.test_transforms
            )
            
            # Print the dataset sizes and label counts
            self.print_label_counts()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = LinearWindingDataModule()


#Each DataLoader will return a batch of dictionaries, where each dictionary corresponds to a sample