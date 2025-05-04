"""
Combine and preprocess data.
make pytorch dataset.
data sources:
    - dynamic
    - static
    - target
data preprocessing:
    do nothing for now.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import argparse
import pathlib

class MIMICDataset(Dataset):
    def __init__(self, static_file_path, dynamic_folder_path):
        # Load the static data
        static_data = pd.read_csv(static_file_path)
        
        # Pre-load all the dynamic and static data into memory
        self.data = []
        for idx, static_row in static_data.iterrows():
            stay_id = int(static_row['stay_id'])
            dynamic_file_path = os.path.join(dynamic_folder_path, f'{stay_id}.csv')
            
            # Check if the corresponding dynamic file exists
            if os.path.exists(dynamic_file_path):
                # Load the dynamic data
                try:
                    dynamic_data = pd.read_csv(dynamic_file_path)
                except pd.errors.EmptyDataError:
                    print(f'Empty data for {stay_id}')
                    continue
                X = torch.tensor(dynamic_data.drop(['stay_id', 'time_bin'], axis=1).values, dtype=torch.float32)
                
                # Extract the static features and targets
                y = torch.tensor([static_row['has_ventilator'], static_row['hospital_expire_flag'], static_row['charlson_charlson_comorbidity_index']], dtype=torch.float32)
                s = torch.tensor(static_row.drop(['stay_id', 'has_ventilator', 'hospital_expire_flag', 'charlson_charlson_comorbidity_index']).values, dtype=torch.float32)
                
                # Append the data to the list
                self.data.append((X, s, y))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MIMICDatasetWithMissingnessInfo(Dataset):
    def __init__(self, static_file_path, dynamic_folder_path, missing_info_folder_path, max_len=100):
        # Load the static data
        static_data = pd.read_csv(static_file_path)
        
        # Pre-load all the dynamic and static data into memory
        self.data = []
        for idx, static_row in static_data.iterrows():
            stay_id = int(static_row['stay_id'])
            dynamic_file_path = os.path.join(dynamic_folder_path, f'{stay_id}.csv')
            missing_info_file_path = os.path.join(missing_info_folder_path, f'{stay_id}.csv')
            
            # Check if the corresponding dynamic file exists
            if os.path.exists(dynamic_file_path) and os.path.exists(missing_info_file_path):
                # Load the dynamic data
                try:
                    dynamic_data = pd.read_csv(dynamic_file_path)
                except pd.errors.EmptyDataError:
                    print(f'Empty data for {stay_id}.')
                    continue
                try:
                    missing_info_data = pd.read_csv(missing_info_file_path)
                except pd.errors.EmptyDataError:
                    print(f'Empty missing info data for {stay_id}.')
                    continue
                X = torch.tensor(dynamic_data.drop(['stay_id', 'charttime', 'time_to_last_rec'], axis=1).values, dtype=torch.float32)
                t = torch.tensor(dynamic_data['time_to_last_rec'].values.reshape(-1, 1), dtype=torch.float32)
                # need to make sure they contain the same columns.
                M = torch.tensor(missing_info_data[dynamic_data.columns].drop(['stay_id', 'charttime', 'time_to_last_rec'], axis=1).values, dtype=torch.float32)
                # entry (i, j) of M is 1 if the record at the i-th time of the j-th patient is not missing in the raw data else 0.
                # Extract the static features and targets
                y = torch.tensor([static_row['has_ventilator'], static_row['hospital_expire_flag'], static_row['charlson_charlson_comorbidity_index']], dtype=torch.float32)
                s = torch.tensor(static_row.drop(['stay_id', 'has_ventilator', 'hospital_expire_flag', 'charlson_charlson_comorbidity_index']).values, dtype=torch.float32)
                # pad or truncate the dynamic data
                X = pad_or_truncate(X, max_len, pad_with_zeros=False) # fill-forward
                M = pad_or_truncate(M, max_len, pad_with_zeros=True) # all padded data is missing, so set to 0.
                t = pad_or_truncate(t, max_len, pad_with_zeros=False) # fill-forward with the earliest time.
                # Append the data to the list
                self.data.append((X, M, t, s, y))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class YaleDataset(Dataset):
    def __init__(self, static_file_path, dynamic_file_path, max_len=500):
        # Load the static data
        static_data = pd.read_csv(static_file_path)
        dynamic_data = pd.read_csv(dynamic_file_path)
        # Pre-load all the dynamic and static data into memory
        self.data = []
        print("loading started...")
        for idx, static_row in static_data.iterrows():
            # print(f"{idx}: {static_row['PAT_ID']} loading...")
            # stay_id = int(static_row['stay_id'])
            # PAT_ID = static_row['PAT_ID']
            # dynamic_file_path = os.path.join(dynamic_folder_path, f'{PAT_ID}.csv')
            
            # Check if the corresponding dynamic file exists
            # if os.path.exists(dynamic_file_path):
            #     # Load the dynamic data
            #     try:
            #         dynamic_data = pd.read_csv(dynamic_file_path)
            #     except pd.errors.EmptyDataError:
            #         print(f'Empty data for {PAT_ID}.')
            #         continue
            dynamic_data_group = dynamic_data[dynamic_data['PAT_ID'] == static_row['PAT_ID']]
            X = torch.tensor(dynamic_data_group.drop(['PAT_ID', 'time_to_extube_hours'], axis=1).values, dtype=torch.float32)
            t = torch.tensor(dynamic_data_group['time_to_extube_hours'].values.reshape(-1, 1), dtype=torch.float32)
            # need to make sure they contain the same columns.
            # entry (i, j) of M is 1 if the record at the i-th time of the j-th patient is not missing in the raw data else 0.
            # Extract the static features and targets
            y = torch.tensor(static_row["time_extub_to_death_hours"], dtype=torch.float32)
            s = torch.tensor(static_row.drop(['PAT_ID', 'time_extub_to_death_hours']).astype(float).values, dtype=torch.float32)
            # pad or truncate the dynamic data
            X = pad_or_truncate(X, max_len, pad_with_zeros=False) # fill-forward
            t = pad_or_truncate(t, max_len, pad_with_zeros=False) # fill-forward with the earliest time.
            # Append the data to the list
            self.data.append((X, t, s, y))
            if idx % 100 == 0:
                print(f"{idx}: {static_row['PAT_ID']} finished.")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class YaleDatasetWithMissingnessInfo(Dataset):
    def __init__(self, static_file_path, dynamic_file_path, missing_file_path, max_len=100, x_vars=None):
        # Load the static data
        static_data = pd.read_csv(static_file_path)
        dynamic_data = pd.read_csv(dynamic_file_path)
        missing_data = pd.read_csv(missing_file_path)
        # Pre-load all the dynamic and static data into memory
        self.data = []
        print("loading started...")
        for idx, static_row in static_data.iterrows():
            # print(f"{idx}: {static_row['PAT_ID']} loading...")
            dynamic_data_group = dynamic_data[dynamic_data['PAT_ID'] == static_row['PAT_ID']]
            missing_data_group = missing_data[missing_data['PAT_ID'] == static_row['PAT_ID']]
            if x_vars is None:
                X = torch.tensor(dynamic_data_group.drop(['PAT_ID', 'time_to_extube_hours'], axis=1).values, dtype=torch.float32)
                # need to make sure they contain the same columns.
                M = torch.tensor(missing_data_group.drop(['PAT_ID', 'time_to_extube_hours'], axis=1).values, dtype=torch.float32)
            else:
                X = torch.tensor(dynamic_data_group[x_vars].values, dtype=torch.float32)
                M = torch.tensor(missing_data_group[x_vars].values, dtype=torch.float32)
            # entry (i, j) of M is 1 if the record at the i-th time of the j-th patient is not missing in the raw data else 0.
            # Extract the static features and targets
            t = torch.tensor(dynamic_data_group['time_to_extube_hours'].values.reshape(-1, 1), dtype=torch.float32)
            y = torch.tensor(static_row[["time_extub_to_death_hours", "time_range"]], dtype=torch.float32)
            s = torch.tensor(static_row.drop(['PAT_ID', 'time_extub_to_death_hours', "time_range"]).astype(float).values, dtype=torch.float32)
            # pad or truncate the dynamic data
            X = pad_or_truncate(X, max_len, pad_with_zeros=False) # fill-forward
            M = pad_or_truncate(M, max_len, pad_with_zeros=True) # all padded data is missing, so set to 0.
            t = pad_or_truncate(t, max_len, pad_with_zeros=False) # fill-forward with the earliest time.
            # Append the data to the list
            self.data.append((X, M, t, s, y))
            if idx % 100 == 0:
                print(f"{idx}: {static_row['PAT_ID']} finished.")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pad_or_truncate(tensor, target_rows, pad_with_zeros=False):
    k = tensor.size(0)
    if k < target_rows:
        if pad_with_zeros:
            padding = torch.zeros(target_rows - k, tensor.size(1), dtype=tensor.dtype, device=tensor.device)
        else:
            padding = tensor[0].unsqueeze(0).repeat(target_rows - k, 1)
        return torch.cat((padding, tensor), dim=0)
    elif k > target_rows:
        return tensor[-target_rows:]
    else:
        return tensor

def train_test_split(dataset, test_size=0.2, random_seed=None):
    # Calculate the number of samples in the test set
    test_length = int(len(dataset) * test_size)
    train_length = len(dataset) - test_length

    # Split the dataset into training and test sets using random_split
    train_dataset, test_dataset = random_split(dataset, [train_length, test_length], generator=torch.Generator().manual_seed(random_seed))

    return train_dataset, test_dataset

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Specify the number of bins (nbins) for the dataset.")
#     parser.add_argument('--nbins', type=str, default='example', help="Number of bins for the dataset.")
#     parser.add_argument('--missing_info', type=bool, default=False, help="Whether to use missing info.")
#     args = parser.parse_args()
#     nbin = args.nbins
#     missing_info = args.missing_info

#     # static_file_path = '../data/static_selected.csv'
#     static_file_path = '../data/stat_sel_vent.csv'
#     # nbin = 24
#     # nbin = "example"
#     if missing_info:
#         missing_info_folder_path = '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/mimic_data/mimiciv_all/no_bin_data_sep_missing_info'
#         dynamic_folder_path = '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/mimic_data/mimiciv_all/no_bin_data_sep'

#         # missing_info_folder_path = '../data/missing_info_example'
#         # dynamic_folder_path = '../data/no_bin_example'

#         # Creating an instance of the preloaded dataset
#         dataset = MIMICDatasetWithMissingnessInfo(static_file_path, dynamic_folder_path, missing_info_folder_path)

#         # Saving the preloaded dataset to a file using torch.save
#         dataset_file_path = f'../dataset/mimiciv50k_no_bin.pt'
#         # dataset_file_path = f'../dataset/mimiciv50k_no_bin_example.pt'
#     else:
#         dynamic_folder_path = f'../data/binned_data_sep_{nbin}'

#         # Creating an instance of the preloaded dataset
#         dataset = MIMICDataset(static_file_path, dynamic_folder_path)

#         # Saving the preloaded dataset to a file using torch.save
#         dataset_file_path = f'../dataset/mimiciv50k_bin_{nbin}.pt'
#     torch.save(dataset, dataset_file_path)

# if __name__ == "__main__":
#     static_file_path = '../output_data/preprocessed/static_target.csv'
#     dynamic_file_path = '../output_data/preprocessed/dynamic.csv'
#     dataset = YaleDataset(static_file_path, dynamic_file_path)
#     torch.save(dataset, "../dataset/yale_imputed.pt")

if __name__ == "__main__":
    # static_file_path = '../output_data/preprocessed3/static_target.csv'
    # dynamic_file_path = '../output_data/preprocessed3/dynamic.csv'
    # missing_file_path = '../output_data/preprocessed3/missing.csv'
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, "../dataset/yale.pt")

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])
    # torch.save(dataset, "../dataset/yale_refl_gcs.pt")
    
    # static_file_path = '../output_data/other_hosp/static_target.csv'
    # dynamic_file_path = '../output_data/other_hosp/dynamic.csv'
    # missing_file_path = '../output_data/other_hosp/missing.csv'
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, "../dataset/other_hosp.pt")

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])
    # torch.save(dataset, "../dataset/other_refl_gcs.pt")

    """
    external validation
    """
    # pathlib.Path("../dataset/valid_split/all/").mkdir(parents=True, exist_ok=True)
    # pathlib.Path("../dataset/valid_split/refl_gcs/").mkdir(parents=True, exist_ok=True)

    # static_file_path = '../output_data/valid_split/yale/static_target.csv'
    # dynamic_file_path = '../output_data/valid_split/yale/dynamic.csv'
    # missing_file_path = '../output_data/valid_split/yale/missing.csv'

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])
    # torch.save(dataset, "../dataset/valid_split/refl_gcs/yale.pt")

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, "../dataset/valid_split/all/yale.pt")

    # static_file_path = '../output_data/valid_split/other_hosp/static_target.csv'
    # dynamic_file_path = '../output_data/valid_split/other_hosp/dynamic.csv'
    # missing_file_path = '../output_data/valid_split/other_hosp/missing.csv'
    
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])
    # torch.save(dataset, "../dataset/valid_split/refl_gcs/other_hosp.pt")

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, "../dataset/valid_split/all/other_hosp.pt")

    """
    external validation + time split
    """
    # pathlib.Path("../dataset/valid_split_time/all/").mkdir(parents=True, exist_ok=True)
    # pathlib.Path("../dataset/valid_split_time/refl_gcs/").mkdir(parents=True, exist_ok=True)

    # static_file_path = '../output_data/valid_split_time/yale_bf21/static_target.csv'
    # dynamic_file_path = '../output_data/valid_split_time/yale_bf21/dynamic.csv'
    # missing_file_path = '../output_data/valid_split_time/yale_bf21/missing.csv'

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])
    # torch.save(dataset, "../dataset/valid_split_time/refl_gcs/yale_bf21.pt")

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, "../dataset/valid_split_time/all/yale_bf21.pt")

    # static_file_path = '../output_data/valid_split_time/yale_af21/static_target.csv'
    # dynamic_file_path = '../output_data/valid_split_time/yale_af21/dynamic.csv'
    # missing_file_path = '../output_data/valid_split_time/yale_af21/missing.csv'

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])
    # torch.save(dataset, "../dataset/valid_split_time/refl_gcs/yale_af21.pt")

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, "../dataset/valid_split_time/all/yale_af21.pt")

    # static_file_path = '../output_data/valid_split_time/other_hosp/static_target.csv'
    # dynamic_file_path = '../output_data/valid_split_time/other_hosp/dynamic.csv'
    # missing_file_path = '../output_data/valid_split_time/other_hosp/missing.csv'
    
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])
    # torch.save(dataset, "../dataset/valid_split_time/refl_gcs/other_hosp.pt")

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, "../dataset/valid_split_time/all/other_hosp.pt")

    """
    external validation + time split 24hr
    """
    # pathlib.Path("../dataset/valid_split_time_last_24_100/all/").mkdir(parents=True, exist_ok=True)

    # static_file_path = '../output_data/valid_split_time_last_24/yale_bf21/static_target.csv'
    # dynamic_file_path = '../output_data/valid_split_time_last_24/yale_bf21/dynamic.csv'
    # missing_file_path = '../output_data/valid_split_time_last_24/yale_bf21/missing.csv'

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, max_len=100)
    # torch.save(dataset, "../dataset/valid_split_time_last_24_100/all/yale_bf21.pt")

    # static_file_path = '../output_data/valid_split_time_last_24/yale_af21/static_target.csv'
    # dynamic_file_path = '../output_data/valid_split_time_last_24/yale_af21/dynamic.csv'
    # missing_file_path = '../output_data/valid_split_time_last_24/yale_af21/missing.csv'

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, max_len=100)
    # torch.save(dataset, "../dataset/valid_split_time_last_24_100/all/yale_af21.pt")

    # static_file_path = '../output_data/valid_split_time_last_24/other_hosp/static_target.csv'
    # dynamic_file_path = '../output_data/valid_split_time_last_24/other_hosp/dynamic.csv'
    # missing_file_path = '../output_data/valid_split_time_last_24/other_hosp/missing.csv'

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, max_len=100)
    # torch.save(dataset, "../dataset/valid_split_time_last_24_100/all/other_hosp.pt")


    """
    external validation + time split 500
    """
    # pathlib.Path("../dataset/valid_split_time_500/all/").mkdir(parents=True, exist_ok=True)

    # static_file_path = '../output_data/valid_split_time/yale_bf21/static_target.csv'
    # dynamic_file_path = '../output_data/valid_split_time/yale_bf21/dynamic.csv'
    # missing_file_path = '../output_data/valid_split_time/yale_bf21/missing.csv'

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, max_len=500)
    # torch.save(dataset, "../dataset/valid_split_time_500/all/yale_bf21.pt")

    # static_file_path = '../output_data/valid_split_time/yale_af21/static_target.csv'
    # dynamic_file_path = '../output_data/valid_split_time/yale_af21/dynamic.csv'
    # missing_file_path = '../output_data/valid_split_time/yale_af21/missing.csv'

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, max_len=500)
    # torch.save(dataset, "../dataset/valid_split_time_500/all/yale_af21.pt")

    # static_file_path = '../output_data/valid_split_time/other_hosp/static_target.csv'
    # dynamic_file_path = '../output_data/valid_split_time/other_hosp/dynamic.csv'
    # missing_file_path = '../output_data/valid_split_time/other_hosp/missing.csv'

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, max_len=500)
    # torch.save(dataset, "../dataset/valid_split_time_500/all/other_hosp.pt")


    """
    2hr before extubation
    """
    # pathlib.Path("../dataset/time_gap/").mkdir(parents=True, exist_ok=True)

    # name = 'yale_bf21'
    # static_file_path = f'../output_data/time_gap/{name}/static_df_kept.csv'

    # time = 0
    # dynamic_file_path = f'../output_data/time_gap/{name}/dynamic_df_kept_{time}.csv'
    # missing_file_path = f'../output_data/time_gap/{name}/missing_df_kept_{time}.csv'
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, f"../dataset/time_gap/{name}_{time}.pt")

    # time = 60
    # dynamic_file_path = f'../output_data/time_gap/{name}/dynamic_df_kept_{time}.csv'
    # missing_file_path = f'../output_data/time_gap/{name}/missing_df_kept_{time}.csv'
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, f"../dataset/time_gap/{name}_{time}.pt")

    # time = 120
    # dynamic_file_path = f'../output_data/time_gap/{name}/dynamic_df_kept_{time}.csv'
    # missing_file_path = f'../output_data/time_gap/{name}/missing_df_kept_{time}.csv'
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, f"../dataset/time_gap/{name}_{time}.pt")

    # name = 'yale_af21'
    # static_file_path = f'../output_data/time_gap/{name}/static_df_kept.csv'

    # time = 0
    # dynamic_file_path = f'../output_data/time_gap/{name}/dynamic_df_kept_{time}.csv'
    # missing_file_path = f'../output_data/time_gap/{name}/missing_df_kept_{time}.csv'
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, f"../dataset/time_gap/{name}_{time}.pt")

    # time = 60
    # dynamic_file_path = f'../output_data/time_gap/{name}/dynamic_df_kept_{time}.csv'
    # missing_file_path = f'../output_data/time_gap/{name}/missing_df_kept_{time}.csv'
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, f"../dataset/time_gap/{name}_{time}.pt")

    # time = 120
    # dynamic_file_path = f'../output_data/time_gap/{name}/dynamic_df_kept_{time}.csv'
    # missing_file_path = f'../output_data/time_gap/{name}/missing_df_kept_{time}.csv'
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, f"../dataset/time_gap/{name}_{time}.pt")

    # name = 'other_hosp'
    # static_file_path = f'../output_data/time_gap/{name}/static_df_kept.csv'

    # time = 0
    # dynamic_file_path = f'../output_data/time_gap/{name}/dynamic_df_kept_{time}.csv'
    # missing_file_path = f'../output_data/time_gap/{name}/missing_df_kept_{time}.csv'
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, f"../dataset/time_gap/{name}_{time}.pt")

    # time = 60
    # dynamic_file_path = f'../output_data/time_gap/{name}/dynamic_df_kept_{time}.csv'
    # missing_file_path = f'../output_data/time_gap/{name}/missing_df_kept_{time}.csv'
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, f"../dataset/time_gap/{name}_{time}.pt")

    # time = 120
    # dynamic_file_path = f'../output_data/time_gap/{name}/dynamic_df_kept_{time}.csv'
    # missing_file_path = f'../output_data/time_gap/{name}/missing_df_kept_{time}.csv'
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, f"../dataset/time_gap/{name}_{time}.pt")

    # static_file_path = '../output_data/valid_split_time/yale_af21/static_target.csv'
    # dynamic_file_path = '../output_data/valid_split_time/yale_af21/dynamic.csv'
    # missing_file_path = '../output_data/valid_split_time/yale_af21/missing.csv'

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, "../dataset/valid_split_time/all/yale_af21.pt")

    # static_file_path = '../output_data/valid_split_time/other_hosp/static_target.csv'
    # dynamic_file_path = '../output_data/valid_split_time/other_hosp/dynamic.csv'
    # missing_file_path = '../output_data/valid_split_time/other_hosp/missing.csv'
    
    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path, x_vars=["dynamic_cornealreflex", "dynamic_gagreflex", "dynamic_gcs"])

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, "../dataset/valid_split_time/all/other_hosp.pt")

    """
    external validation + time split remove <0
    """
    # pathlib.Path("../dataset/valid_split_time_remove_negative_targ/all/").mkdir(parents=True, exist_ok=True)

    # static_file_path = '../output_data/valid_split_time_remove_negative_targ/yale_bf21/static_target.csv'
    # dynamic_file_path = '../output_data/valid_split_time_remove_negative_targ/yale_bf21/dynamic.csv'
    # missing_file_path = '../output_data/valid_split_time_remove_negative_targ/yale_bf21/missing.csv'

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, "../dataset/valid_split_time_remove_negative_targ/all/yale_bf21.pt")

    # static_file_path = '../output_data/valid_split_time_remove_negative_targ/yale_af21/static_target.csv'
    # dynamic_file_path = '../output_data/valid_split_time_remove_negative_targ/yale_af21/dynamic.csv'
    # missing_file_path = '../output_data/valid_split_time_remove_negative_targ/yale_af21/missing.csv'

    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, "../dataset/valid_split_time_remove_negative_targ/all/yale_af21.pt")

    # static_file_path = '../output_data/valid_split_time_remove_negative_targ/other_hosp/static_target.csv'
    # dynamic_file_path = '../output_data/valid_split_time_remove_negative_targ/other_hosp/dynamic.csv'
    # missing_file_path = '../output_data/valid_split_time_remove_negative_targ/other_hosp/missing.csv'


    # dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    # torch.save(dataset, "../dataset/valid_split_time_remove_negative_targ/all/other_hosp.pt")

    """
    external validation + time split remove <0;
    only keep time in advance (6 hours)
    """
    keep_time_in_advance = 6
    in_path_root = '../output_data/sci_rep_valid_split_time_remove_negative_targ_in_advance'
    save_path_root = '../dataset/sci_rep_valid_split_time_remove_negative_targ_in_advance'
    in_path = f'{in_path_root}_{keep_time_in_advance}/'
    save_path = f'{save_path_root}_{keep_time_in_advance}/'

    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    static_file_path = f'{in_path}/yale_bf21/static_target.csv'
    dynamic_file_path = f'{in_path}/yale_bf21/dynamic.csv'
    missing_file_path = f'{in_path}/yale_bf21/missing.csv'

    dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    torch.save(dataset, f"{save_path}/yale_bf21.pt")

    static_file_path = f'{in_path}/yale_af21/static_target.csv'
    dynamic_file_path = f'{in_path}/yale_af21/dynamic.csv'
    missing_file_path = f'{in_path}/yale_af21/missing.csv'

    dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    torch.save(dataset, f"{save_path}/yale_af21.pt")

    static_file_path = f'{in_path}/other_hosp/static_target.csv'
    dynamic_file_path = f'{in_path}/other_hosp/dynamic.csv'
    missing_file_path = f'{in_path}/other_hosp/missing.csv'


    dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    torch.save(dataset, f"{save_path}/other_hosp.pt")

    """
    external validation + time split remove <0;
    only keep time in advance (12 hours)
    """
    keep_time_in_advance = 12
    in_path_root = '../output_data/sci_rep_valid_split_time_remove_negative_targ_in_advance'
    save_path_root = '../dataset/sci_rep_valid_split_time_remove_negative_targ_in_advance'
    in_path = f'{in_path_root}_{keep_time_in_advance}/'
    save_path = f'{save_path_root}_{keep_time_in_advance}/'

    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    static_file_path = f'{in_path}/yale_bf21/static_target.csv'
    dynamic_file_path = f'{in_path}/yale_bf21/dynamic.csv'
    missing_file_path = f'{in_path}/yale_bf21/missing.csv'

    dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    torch.save(dataset, f"{save_path}/yale_bf21.pt")

    static_file_path = f'{in_path}/yale_af21/static_target.csv'
    dynamic_file_path = f'{in_path}/yale_af21/dynamic.csv'
    missing_file_path = f'{in_path}/yale_af21/missing.csv'

    dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    torch.save(dataset, f"{save_path}/yale_af21.pt")

    static_file_path = f'{in_path}/other_hosp/static_target.csv'
    dynamic_file_path = f'{in_path}/other_hosp/dynamic.csv'
    missing_file_path = f'{in_path}/other_hosp/missing.csv'


    dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    torch.save(dataset, f"{save_path}/other_hosp.pt")

    """
    external validation + time split remove <0;
    only keep time in advance (24 hours)
    """
    keep_time_in_advance = 24
    in_path_root = '../output_data/sci_rep_valid_split_time_remove_negative_targ_in_advance'
    save_path_root = '../dataset/sci_rep_valid_split_time_remove_negative_targ_in_advance'
    in_path = f'{in_path_root}_{keep_time_in_advance}/'
    save_path = f'{save_path_root}_{keep_time_in_advance}/'

    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    static_file_path = f'{in_path}/yale_bf21/static_target.csv'
    dynamic_file_path = f'{in_path}/yale_bf21/dynamic.csv'
    missing_file_path = f'{in_path}/yale_bf21/missing.csv'

    dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    torch.save(dataset, f"{save_path}/yale_bf21.pt")

    static_file_path = f'{in_path}/yale_af21/static_target.csv'
    dynamic_file_path = f'{in_path}/yale_af21/dynamic.csv'
    missing_file_path = f'{in_path}/yale_af21/missing.csv'

    dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    torch.save(dataset, f"{save_path}/yale_af21.pt")

    static_file_path = f'{in_path}/other_hosp/static_target.csv'
    dynamic_file_path = f'{in_path}/other_hosp/dynamic.csv'
    missing_file_path = f'{in_path}/other_hosp/missing.csv'


    dataset = YaleDatasetWithMissingnessInfo(static_file_path, dynamic_file_path, missing_file_path)
    torch.save(dataset, f"{save_path}/other_hosp.pt")
