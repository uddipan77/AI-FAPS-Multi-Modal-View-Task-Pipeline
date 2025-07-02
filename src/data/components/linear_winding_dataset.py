import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from tsfresh.utilities.dataframe_functions import impute
from tsai.data.preprocessing import TSNormalize  # Import TSNormalize

class LinearWindingDataset(Dataset):
    """A custom PyTorch Dataset for loading multimodal data consisting of images and force curves.
    Each sample in the dataset has multiple images, a force curve, a label, and a unique coil ID.
    
    :param root_dir: Root directory containing data.
    :param subset: Specifies which subset of data to load. Default is 'train_val'. Options are 'train_val', 'test', or 'excluded'.
    :param images_per_sample: Specific image numbers to load per sample.
    :param curve_columns: Names of columns from the curve pkl files to include.
    :param label_column: The column in the CSV file to use as the label.
    :param transforms: Transform pipeline to apply to images. Default is None.

    :return: A custom PyTorch Dataset for loading multimodal data.
    """
    
    def __init__(self, root_dir, subset, images_per_sample, curve_columns, label_column, transforms=None):
        # Load metadata of coil data with the labels
        csv_file = os.path.join(root_dir, subset, f"labels_and_metadata_coil_level_{subset}_2023-09-15.CSV")
        self.metadata = pd.read_csv(csv_file)

        # Filter to only use rows where material_batch = 1
        self.metadata = self.metadata[self.metadata["material_batch"] == 1].reset_index(drop=True)

        # Print shape of the filtered data to verify
        print(f"[LinearWindingDataset] Loaded {len(self.metadata)} rows after filtering material_batch = 1")

        
        
        self.root_dir = root_dir
        self.subset = subset
        self.images_per_sample = images_per_sample
        self.curve_columns = curve_columns
        self.label_column = label_column
        self.transform = transforms
        self.force_transform = TSNormalize(min=None, max=None, range=[-1, 1])  # Initialize TSNormalize for forces
        
        # Load force features from tsfresh
        force_features_file = os.path.join(root_dir, subset, f"tsfresh_force_features_coil_level_{subset}_2024-01-16.CSV")        
        force_tsfresh_features = pd.read_csv(force_features_file)
        self.force_features = self.prepare_tsfresh_features(force_tsfresh_features)
        
        self.labels = [0, 1]
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get the coil metadata, row wise
        #final idx value will be len(train_val_dataset) - 1.
        coil_data = self.metadata.iloc[idx]
        #it will pull the corresponding row from this DataFrame and store it as coil_data.
        #the result is a pandas Series object.
        
        # Load Images
        coil_images_folder = os.path.join(self.root_dir, self.subset, coil_data["coil_id"], coil_data["coil_id"] + "_images")
        images = []
        for img_num in self.images_per_sample:
            img_name = os.path.join(coil_images_folder, coil_data["coil_id"] + "_image_{:04d}.jpg".format(img_num))
            img = Image.open(img_name).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        # Load Curves and apply TSNormalize to force data
        curve_file = os.path.join(self.root_dir, self.subset, coil_data["coil_id"], coil_data["coil_id"] + "_force_and_displacement_curves.pkl")
        curve_df = pd.read_pickle(curve_file)
        curve_data = curve_df[self.curve_columns].values  # Get the force data
        
        curves = []
        num_layers = 5
        start, end = 0, len(curve_data)
        step = ((end - start) // num_layers)

        layer_indices = list(range(start, end, step))
        if layer_indices[-1] + step < end:
            layer_indices.append(layer_indices[-1] + step)
        
        layer_indices[-1] = min(layer_indices[-1], end)
    
        for peak_i, peak_j in zip(layer_indices[:-1], layer_indices[1:]):
            layer_data = curve_data[peak_i : peak_j]
            layer_data = torch.tensor(layer_data, dtype=torch.float32).permute(1, 0)
            layer_data = self.force_transform(layer_data)  # Apply TSNormalize to each layer
            curves.append(layer_data)
        
        # Load force features
        force_features = torch.tensor(
            self.force_features[self.force_features["coil_id"] == coil_data["coil_id"]].fillna(0).drop(columns=["coil_id", "label"]).values,
            dtype=torch.float32
        ).squeeze()
        
        # Load Labels
        label = coil_data[self.label_column]
        
        return {
            'images': images,
            'forces': curves,
            'forces_tsfresh_features': force_features,
            'labels': label,
            'id': coil_data["coil_id"]
        }
    
    def prepare_tsfresh_features(self, features):
        """Prepare the tsfresh features."""
        
        # Select numerical features
        numerical_cols = features.select_dtypes(include=['number']).columns
        features_numerical = features[numerical_cols]

        # Normalize the numerical features
        scaler = StandardScaler()
        features_numerical_scaled = scaler.fit_transform(features_numerical)

        # Select string features
        string_cols = features.select_dtypes(include=['object']).columns
        features_string = features[string_cols]
        
        return pd.concat([features_string, pd.DataFrame(features_numerical_scaled, columns=numerical_cols)], axis=1)
