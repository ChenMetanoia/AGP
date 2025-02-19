import pandas as pd
import numpy as np
import torch
import re
import random
import math
from tqdm import tqdm
from recbole.data.utils import create_dataset
from recbole.config import Config
from utils.util import data_preparation


def load_dataset(config_dict):
    config = Config(model='SASRec', config_dict=config_dict, config_file_list=['./props/SASRec.yaml'])
    dataset = create_dataset(config)
    return data_preparation(config, dataset)


def mask_items(item_profiles, mask_ratio):
    num_to_mask = math.floor(len(item_profiles) * mask_ratio)
    indices_to_mask = random.sample(range(len(item_profiles)), num_to_mask)
    for idx in indices_to_mask:
        item_profiles[idx] = "[MASK]"
    return item_profiles


def get_user_seq(data, index):
    user_id = data.dataset.inter_feat['user_id'][index].item()
    user_seq = data.dataset.inter_feat['item_id_list'][index].tolist()
    user_seq = user_seq[:user_seq.index(0)] if 0 in user_seq else user_seq
    ground_truth = data.dataset.inter_feat['item_id'][index].item()
    return user_id, user_seq, ground_truth


def generate_negative_samples(ground_truth, data, num_samples=10):
    item_list = data.dataset.inter_feat['item_id'].tolist()
    return random.sample([item for item in item_list if item != ground_truth], num_samples)


def generate_movie_profile_seq(user_seq, movie_profile):
    return [movie_profile[str(item_id)] for item_id in user_seq if str(item_id) in movie_profile]


class Interpreter:
    def __init__(self, item_text_info_path):
        self.item_text_dict = self._load_item_text_info(item_text_info_path)

    def _load_item_text_info(self, item_text_info_path):
        item_text_df = pd.read_csv(item_text_info_path, sep='\t')
        return {
            row['item_id:token']: {
                'title': row['title:token_seq'],
                'brand': row['brand:token_seq'],
                'category': row['category:token_seq']
            } for _, row in item_text_df.iterrows()
        }

    def interpret_seq(self, seq, text_fields):
        seq = seq if isinstance(seq, list) else [seq]
        return [{
            'item_id': item_id,
            **{field: self.item_text_dict.get(item_id, {}).get(field, 'N/A') for field in text_fields}
        } for item_id in seq]


def generate_item_profile_embedding(item_profile, plm, gpu_id='0'):
    return generate_text_embedding(list(item_profile.keys()), list(item_profile.values()), plm, gpu_id)


def load_plm(model_name='bert-base-uncased'):
    from transformers import AutoModel, AutoTokenizer
    model_map = {
        'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
        'all-mpnet-base-v2': 'sentence-transformers/all-mpnet-base-v2',
        'bge-base-en-v1.5': 'BAAI/bge-base-en-v1.5'
    }
    model_name = model_map.get(model_name, model_name)
    return AutoTokenizer.from_pretrained(model_name), AutoModel.from_pretrained(model_name)


def set_device(gpu_id):
    return torch.device(f'cuda:{gpu_id}' if gpu_id != '-1' and torch.cuda.is_available() else 'cpu')


def get_ranking_results(user_summary, item_profile_embedding, plm, gpu_id='0'):
    device = set_device(gpu_id)
    tokenizer, model = load_plm(plm)
    model.to(device).eval()
    user_summary_embedding = generate_text_embedding([0], [user_summary], plm, gpu_id)['item_emb:float_seq'].values[0]
    user_summary_embedding = torch.tensor(user_summary_embedding).to(device)
    similarity_scores = torch.matmul(user_summary_embedding, item_profile_embedding.to(device).T)
    return torch.argsort(similarity_scores, descending=True)


def parse_rerank_output(output: str, bos: str, eos: str) -> list:
    if not (output.startswith(bos) and output.endswith(eos)):
        raise ValueError("Output format incorrect: Missing BOS or EOS tokens.")
    content = output[len(bos):-len(eos)].strip()
    if not re.match(r"^\d+(,\d+)*$", content):
        raise ValueError("Output format incorrect: Expected a comma-separated list of integers.")
    return list(map(int, content.split(',')))


