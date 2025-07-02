import pandas as pd
import numpy as np
from torchvision import transforms
from src.data.components.linear_winding_dataset import LinearWindingDataset
from tsfresh.feature_extraction import extract_features, EfficientFCParameters, MinimalFCParameters
from tsfresh import extract_relevant_features, select_features


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
        
        # sample_df = pd.DataFrame({'force': force_data})
        # sample_df['coil_id'] = sample['id']

        all_data.append(np.stack([force_data, np.repeat(sample['id'], len(force_data))], axis=1))
        # all_data.append(sample_df)
        print(i)
        if i == 10:
            break

    all_data_df = pd.DataFrame(np.concatenate(all_data), columns=['force', 'coil_id'])
    all_data_df['force'] = all_data_df['force'].astype(float)
    X = all_data_df[['force', 'coil_id']]
    # print(all_data_df.info())
    
    label_df = pd.read_csv('/home/hpc/iwfa/iwfa028h/dev/faps/AI-FAPS_Vishnudev_Krishnadas/data/LinearWinding/dataset_linear_winding_multimodal/train_val/labels_and_metadata_coil_level_train_val_2023-09-15.CSV')
    resistance = pd.Series(dict(zip(label_df['coil_id'], label_df['label_dc_resistance'])))
    mask = resistance.index.isin(X.coil_id)
    y = resistance[mask]
    # print(y)
    
    features = extract_features(X, column_value='force', column_id='coil_id', chunksize=chunksize, n_jobs=n_jobs, default_fc_parameters=EfficientFCParameters(), disable_progressbar=False)
    features.dropna(axis=1, how='any', inplace=True)
    print(features.shape)
    
    features_sel = select_features(
        features,
        y,
        n_jobs=n_jobs,
        chunksize=chunksize,
        ml_task='regression',
    )
    print(features_sel)

    # features = extract_relevant_features(
    #     X,
    #     y,
    #     column_value='force',
    #     column_id="coil_id",
    #     n_jobs=n_jobs,
    #     chunksize=chunksize,
    #     ml_task='regression',
    #     profile=False,
    #     default_fc_parameters=EfficientFCParameters(),
    # )
    
    features.to_csv('/home/hpc/iwfa/iwfa028h/dev/faps/AI-FAPS_Vishnudev_Krishnadas/data/LinearWinding/dataset_linear_winding_multimodal/train_val/tsfresh_relevant_features_coil_level_train_val_2024-02-01.CSV', index=False)
    # features = extract_features(all_data_df, column_value='force', column_id='coil_id', chunksize=chunksize, n_jobs=n_jobs, default_fc_parameters=EfficientFCParameters(), disable_progressbar=False)
    print(features)
    

if __name__ == '__main__':
    extract_force_features('train_val', n_jobs=8, chunksize=1)