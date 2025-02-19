```python
import importlib
import torch
import pickle
import numpy as np
import os
from tqdm import tqdm
from recbole.data.dataloader import *
from recbole.sampler import KGSampler, Sampler, RepeatableSampler
from recbole.utils import ModelType, set_color


def extract_last_block(file_path):
    content = []
    inside_block = False  # Track if we're inside the block
    with open(file_path, 'r') as file:
        for line in file:
            if 'Updated prompt accepted:' in line:
                inside_block = True
                content = []  # Reset content to capture last block
                continue
            elif 'original prompt:' in line or 'Batch completed.' in line:
                inside_block = False
                continue
            if inside_block:
                content.append(line.strip())
    return '\n'.join(content)


def ndcg_score(target_item, predicted_list):
    if target_item not in predicted_list:
        return 0.0
    rank = predicted_list.index(target_item)
    dcg = 1 / np.log2(rank + 2)
    idcg = 1 / np.log2(2)
    return dcg / idcg


def save_tensor_to_file(file_path, tensor):
    with open(file_path, 'ab') as file:
        torch.save(tensor, file)


def empty_file(file_path):
    open(file_path, 'w').close()


def import_class(model_name):
    model_root_path = 'recbole_gnn.model.general_recommender'
    module_path = f'{model_root_path}.{model_name.lower()}' if model_name == 'LightGCN' else f'{model_root_path}.basic_gnn'
    module = importlib.import_module(module_path)
    return getattr(module, model_name)


def data_preparation(config, dataset):
    built_datasets = dataset.build()
    datasets = built_datasets[:3] if len(built_datasets) == 3 else built_datasets
    samplers = create_samplers(config, dataset, built_datasets, cold=len(datasets) == 4)
    
    train_data = get_dataloader(config, "train")(config, datasets[0], samplers[0], shuffle=config["shuffle"])
    valid_data = get_dataloader(config, "valid")(config, datasets[1], samplers[1], shuffle=False)
    test_data = get_dataloader(config, "test")(config, datasets[2], samplers[2], shuffle=False)
    
    if len(datasets) == 4:
        cold_data = get_dataloader(config, "test")(config, datasets[3], samplers[3], shuffle=False)
        return train_data, valid_data, test_data, cold_data
    return train_data, valid_data, test_data


def get_dataloader(config, phase):
    ae_models = {"MultiDAE", "MultiVAE", "MacridVAE", "CDAE", "ENMF", "RaCT", "RecVAE"}
    if config["model"] in ae_models:
        return _get_AE_dataloader(config, phase)
    if phase == "train":
        return TrainDataLoader if config["MODEL_TYPE"] != ModelType.KNOWLEDGE else KnowledgeBasedDataLoader
    return FullSortEvalDataLoader if config["eval_args"]["mode"][phase] == "full" else NegSampleEvalDataLoader


def _get_AE_dataloader(config, phase):
    return UserDataLoader if phase == "train" else FullSortEvalDataLoader if config["eval_args"]["mode"] == "full" else NegSampleEvalDataLoader


def _create_sampler(dataset, built_datasets, distribution, repeatable, alpha=1.0, base_sampler=None):
    if distribution == "none":
        return None
    if base_sampler:
        base_sampler.set_distribution(distribution)
        return base_sampler
    sampler_cls = RepeatableSampler if repeatable else Sampler
    return sampler_cls(["train", "valid", "test"], dataset, distribution, alpha)


def create_samplers(config, dataset, built_datasets, cold=False):
    train_sampler = _create_sampler(dataset, built_datasets, config["train_neg_sample_args"]["distribution"], config["repeatable"], config["train_neg_sample_args"]["alpha"])
    valid_sampler = _create_sampler(dataset, built_datasets, config["valid_neg_sample_args"]["distribution"], config["repeatable"], base_sampler=train_sampler)
    test_sampler = _create_sampler(dataset, built_datasets, config["test_neg_sample_args"]["distribution"], config["repeatable"], base_sampler=train_sampler)
    if cold:
        cold_sampler = _create_sampler(dataset, built_datasets, config["test_neg_sample_args"]["distribution"], config["repeatable"], base_sampler=train_sampler)
        return train_sampler, valid_sampler, test_sampler, cold_sampler
    return train_sampler, valid_sampler, test_sampler
```

