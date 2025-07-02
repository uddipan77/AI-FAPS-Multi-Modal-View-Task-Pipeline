import pandas as pd
import numpy as np
from torchvision import transforms
from src.data.components.linear_winding_dataset import LinearWindingDataset
from tsfresh.feature_extraction import extract_features, EfficientFCParameters, MinimalFCParameters


def extract_force_features(dataset_type='train_val', n_jobs=10, chunksize=1000):

    all_data = []
    labels = []
    coil_ids = []
    
    train_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_val_dataset = LinearWindingDataset(
        '/home/hpc/iwfa/iwfa028h/dev/faps/AI-FAPS_Vishnudev_Krishnadas/data/LinearWinding/dataset_linear_winding_multimodal', 
        dataset_type,
        [957, 958, 959, 960],
        ['force'],
        "label_geom_error_overall",
        transforms=train_transforms
    )
    
    for i, sample in enumerate(train_val_dataset):
        
        force_data = np.concatenate([layer_data.numpy()[0] for layer_data in sample['forces']])
        
        sample_df = pd.DataFrame({'force': force_data})
        sample_df['coil_id'] = sample['id']
                
        all_data.append(sample_df)
        
        if i == 1:
            break

    all_data_df = pd.concat(all_data)
    print(all_data_df)
    features = extract_features(all_data_df, column_value='force', column_id='coil_id', chunksize=chunksize, n_jobs=n_jobs, default_fc_parameters=EfficientFCParameters(), disable_progressbar=False)
    print(features)
    # root_dir = f'/home/hpc/iwfa/iwfa028h/work_data/data/LinearWindingDataset/fabian/dataset_linear_winding_multimodal/{dataset_type}/'
    # all_data_df.to_csv(root_dir + f'tsfresh_force_features_coil_level_{dataset_type}_2024-01-16.CSV', index=False)


if __name__ == '__main__':
    # NOTE: This is a very time consuming process.
    # Works the best with salloc tinyfat with 10 tasks per core
    
    # extract_force_features('train_val', n_jobs=10, chunksize=1000)
    extract_force_features('test', n_jobs=4, chunksize=None)
