import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class UserLevelDataset(Dataset):
    """
    Unified Dataset class for sequence recommendation systems, supporting both 
    training and testing.
    """

    def __init__(self, 
                 root: str = None, 
                 task_name: str = None, 
                 type_list: list = None, 
                 batch_size: int = 25, 
                 shuffle: bool = True):
        self.inter_df = None
        self.item_df = None
        self.item_profile = None
        self.embedding = None

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.user_idx = 0

        self.root = root
        self.task_name = task_name
        self.type_list = type_list or []

        self._check_dataset()

    def _check_dataset(self) -> None:
        """
        Check the existence of required dataset files.
        Load them into memory if present.
        """
        data_path = os.path.join(self.root, self.task_name)

        # Load each requested data type
        for data_type in self.type_list:
            if data_type in ["inter", "item"]:
                file_type = "csv"
            elif data_type == "profile":
                file_type = "json"
            elif data_type == "embedding":
                file_type = "pkl"
            else:
                raise ValueError(f"Data type '{data_type}' not supported.")

            # Example path: dataset/Movies_and_TV/movies_and_tv_inter.csv
            file_path = os.path.join(data_path, f"{self.task_name.lower()}_{data_type}.{file_type}")
            # Attempt alternative naming
            if not os.path.exists(file_path):
                file_path = os.path.join(data_path, f"{self.task_name}_{data_type}.{file_type}")

            if not os.path.exists(file_path):
                msg = f"'{self.task_name.lower()}_{data_type}.{file_type}' not found in '{data_path}'."
                raise FileNotFoundError(msg)
            else:
                if data_type == "inter":
                    # Example appended filename with parameters
                    appended_file = f"{file_path}_{config['initial_user_seq_length']}_{config['pretrain_model']}"
                    if not os.path.exists(appended_file):
                        raise FileNotFoundError(
                            f"Appended inter file not found: {appended_file}"
                        )
                    self.inter_df = pd.read_csv(appended_file, sep='\t')

                elif data_type == "item":
                    self.item_df = pd.read_csv(file_path, sep='\t')

                elif data_type == "profile":
                    with open(file_path, 'r') as f:
                        self.item_profile = json.load(f)
                    # Convert each key to int
                    self.item_profile = {int(k): v for k, v in self.item_profile.items()}

                elif data_type == "embedding":
                    self.embedding = pd.read_pickle(file_path)

        # Basic user-level info
        self.users = list(self.inter_df['user_id:token'])
        self.indices = np.arange(len(self.users))

        # Load precomputed ranking results
        ranking_file = os.path.join(data_path, f"{config['pretrain_model']}_Ranking.json")
        if not os.path.exists(ranking_file):
            raise FileNotFoundError(f"Ranking file not found: {ranking_file}")
        with open(ranking_file, 'r') as f:
            self.SR_ranking = json.load(f)

    def __len__(self) -> int:
        """
        Return total number of users.
        """
        return len(self.users)

    def __iter__(self):
        """
        Iterator reset (optionally shuffles the user indices).
        """
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.user_idx = 0
        return self

    def __next__(self) -> list:
        """
        Generate a batch of data.
        """
        if self.user_idx >= len(self.users):
            raise StopIteration

        batch_data = []
        for _ in range(self.batch_size):
            if self.user_idx >= len(self.users):
                break

            user_id = self.users[self.indices[self.user_idx]]
            sr_ranking_list = self.SR_ranking[str(user_id)]

            item_id_series = self.inter_df[self.inter_df['user_id:token'] == user_id]['item_id_list:token_seq']
            item_id_list = item_id_series.values[0].split()
            total_length = len(item_id_list)

            current_data_dict = {
                'user_id': user_id,
                'item_id_list': item_id_list,
                'user_sequence_length': total_length,
                'sr_ranking_list': sr_ranking_list
            }
            batch_data.append(current_data_dict)
            self.user_idx += 1

        if not batch_data:
            raise StopIteration
        
        return batch_data

    def reset(self) -> None:
        """
        Reset the iterator to the beginning.
        """
        self.user_idx = 0
        self.indices = np.arange(len(self.users))
