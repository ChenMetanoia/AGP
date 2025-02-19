import pandas as pd
import numpy as np
import torch
import re
from tqdm import tqdm
from recbole.data.utils import create_dataset
from recbole.config import Config
from utils.util import data_preparation
import math
import random
def load_dataset(config_dict):
    config_file_list = ['./props/SASRec.yaml']
    config = Config(model='SASRec',
                    config_dict=config_dict,
                    config_file_list=config_file_list)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    return train_data, valid_data, test_data


def mask_items(item_profiles, n):
    # Calculate the number of elements to mask
    num_to_mask = math.floor(len(item_profiles) * n)
    
    # Randomly select indices to mask
    indices_to_mask = random.sample(range(len(item_profiles)), num_to_mask)
    
    # Mask the selected indices
    for idx in indices_to_mask:
        item_profiles[idx] = "[MASK]"
    
    return item_profiles

def get_user_seq(data, index):
    user_id = data.dataset.inter_feat['user_id'][index].item()
    user_seq = data.dataset.inter_feat['item_id_list'][index].tolist()
    # truncate the user_seq once the item_id is 0
    if 0 in user_seq:
        user_seq = user_seq[:user_seq.index(0)]
    ground_truth = data.dataset.inter_feat['item_id'][index].item()
    return user_id, user_seq, ground_truth


def generate_negative_samples(ground_truth, data, num_samples=10):
    negative_samples = []
    item_list = data.dataset.inter_feat['item_id'].tolist()
    for i in range(num_samples):
        random_idx = np.random.randint(len(item_list))
        while item_list[random_idx] == ground_truth:
            random_idx = np.random.randint(len(item_list))
        negative_samples.append(item_list[random_idx])
    return negative_samples

def generate_movie_profile_seq(user_seq, movie_profile):
    movie_profile_seq = []
    for item_id in user_seq:
        if str(item_id) in movie_profile:
            movie_profile_seq.append(movie_profile[str(item_id)])
    return movie_profile_seq

class Interpreter:
    def __init__(self, item_text_info_path):
        self.item_text_dict = self.load_item_text_info(item_text_info_path)
        
    def load_item_text_info(self, item_text_info_path):
        item_text_df = pd.read_csv(item_text_info_path, sep='\t')
        item_text_dict = {}
        # item_text_df has three columns: item_id:token, title:token_seq, category:token_seq
        # convert item_text_df into a dictionary
        for idx, row in item_text_df.iterrows():
            item_text_dict[row['item_id:token']] = {'title': row['title:token_seq'], 
                                                    'brand': row['brand:token_seq'],
                                                    'category': row['category:token_seq']}
        return item_text_dict
    
    def interpret_user_seq(self, user_seq, ground_truth, text_fields):
        # user_seq is a list of item_id
        # ground_truth is an item_id
        # text_fields is a list of text fields to be interpreted
        # return a list of dictionaries, each dictionary contains the interpretation of a item_id in user_seq based on text_fields
        # return a dictionary containing the interpretation of the ground_truth item_id
        interpretations = []
        for item_id in user_seq:
            if item_id in self.item_text_dict:
                interpretation = {}
                interpretation['item_id'] = item_id
                for field in text_fields:
                    try:
                        interpretation[field] = self.item_text_dict[item_id][field]
                    except KeyError:
                        interpretation[field] = 'N/A'
                interpretations.append(interpretation)
        ground_truth_interpretation = {}
        if ground_truth in self.item_text_dict:
            ground_truth_interpretation['item_id'] = ground_truth
            for field in text_fields:
                ground_truth_interpretation[field] = self.item_text_dict[ground_truth][field]
        return interpretations, ground_truth_interpretation

    def interpret_seq(self, seq, text_fields):
        # seq is a list of item_id
        # text_fields is a list of text fields to be interpreted
        # return a list of dictionaries, each dictionary contains the interpretation of a item_id in seq based on text_fields
        interpretations = []
        # check seq is a list or one item_id, if one item_id, convert it into a list
        if not isinstance(seq, list):
            seq = [seq]
        for item_id in seq:
            if item_id in self.item_text_dict:
                interpretation = {}
                interpretation['item_id'] = item_id
                for field in text_fields:
                    try:
                        interpretation[field] = self.item_text_dict[item_id][field]
                    except KeyError:
                        interpretation[field] = 'N/A'
                interpretations.append(interpretation)
        return interpretations
    
    
def generate_item_profile_embedding(item_profile, plm, gpu_id='0'):
    plm = plm.lower()
    items = list(item_profile.keys())
    texts = [item_profile[item] for item in items]
    embeddings_df = generate_text_embedding(items, texts, plm, gpu_id=gpu_id)
    return embeddings_df

def load_plm(model_name='bert-base-uncased'):
    from transformers import AutoModel, AutoTokenizer
    if model_name == 'bert-base-uncased':
        model_name = model_name
    elif model_name == 'all-MiniLM-L6-v2' or model_name == 'all-mpnet-base-v2':
        model_name = f'sentence-transformers/{model_name}'
    elif model_name == 'bge-base-en-v1.5':
        model_name = 'BAAI/bge-base-en-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def generate_text_embedding(ids, texts, plm, gpu_id='0'):
    if plm == 'instructor-xl' or plm == 'all-MiniLM-L6-v2' or plm=='all-mpnet-base-v2':
        if plm == 'instructor-xl':
            from InstructorEmbedding import INSTRUCTOR
            model = INSTRUCTOR('hkunlp/instructor-xl')
        elif plm == 'all-MiniLM-L6-v2':
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
        elif plm == 'all-mpnet-base-v2':
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-mpnet-base-v2')
        else:
            raise NotImplementedError('plm {} is not supported'.format(plm))
        
        if plm == 'instructor-xl':
            instruction = "Represent the user profile: "
            sentence = []
            for t in texts:
                sentence.append(instruction + t) 
        else:
            sentence = texts
        embeddings = model.encode(sentence, 
                                    batch_size=32, 
                                    show_progress_bar=True,
                                    convert_to_numpy=True,
                                    device='cuda:{}'.format(gpu_id),
        )
    elif plm == '   jxm/cde-small-v1':
        pass
        # model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)
        # dataset_embeddings = model.encode(
        #     texts,
        #     prompt_name="document",
        #     convert_to_tensor=True
        # )
    elif plm == 'voyageai':
        import voyageai

        vo = voyageai.Client()
        # This will automatically use the environment variable VOYAGE_API_KEY.
        # Alternatively, you can use vo = voyageai.Client(api_key="<your secret key>")

        batch_size = 128
        embeddings = [
            vo.embed(
                texts[i : i + batch_size],
                model="voyage-large-2-instruct",
                input_type="document",
            ).embeddings
            for i in range(0, len(texts), batch_size)
        ]
    else:
        device = set_device(gpu_id)
        tokenizer, model = load_plm(plm)
        model = model.to(device)
        model.eval()  # Set the model to evaluation mode
        
        embeddings = []
        batch_size = 4
        
        with torch.no_grad():  # Disable gradient computation for inference
            if plm == 'bert-base-uncased':
                for i in tqdm(range(0, len(texts), batch_size), desc='Generating Embeddings', unit='batches'):
                    batch_texts = texts[i:i+batch_size]
                    encoded_batch = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
                    encoded_batch = encoded_batch.to(device)
                    outputs = model(**encoded_batch)
                    cls_output = outputs.last_hidden_state[:, 0, ].cpu().tolist()
                    embeddings.extend(cls_output)
            elif plm == 'bge-base-en-v1.5':
                for i in tqdm(range(0, len(texts), batch_size), desc='Generating Embeddings', unit='batches'):
                    batch_texts = texts[i:i+batch_size]
                    encoded_batch = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
                    encoded_batch = encoded_batch.to(device)
                    model_output = model(**encoded_batch)
                    # Perform pooling. In this case, cls pooling.
                    sentence_embeddings = model_output[0][:, 0]
                    embeddings.extend(sentence_embeddings)
                # normalize embeddings
                embeddings = torch.stack(embeddings, dim=0)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Convert embeddings to a DataFrame
    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()
    elif isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().tolist()
    elif isinstance(embeddings, list):
        pass
    else:
        raise NotImplementedError('embeddings type {} is not supported'.format(type(embeddings)))
    embeddings_df = pd.DataFrame({"item_id:token": ids, "item_emb:float_seq": embeddings})
    
    # Convert embeddings to a string representation
    # embeddings_df['item_emb:float_seq'] = embeddings_df['item_emb:float_seq'].apply(lambda x: ' '.join(map(str, x)))
    
    return embeddings_df


def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
        

def get_ranking_results(user_summary, item_profile_embedding, plm, gpu_id='0'):
    # user_summary is a text string
    # item_profile_embedding is a DataFrame with two columns: item_id:token, item_emb:float_seq
    device = set_device(gpu_id)
    tokenizer, model = load_plm(plm)
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    user_summary_embedding = generate_text_embedding([0], [user_summary], plm, gpu_id=gpu_id)
    user_summary_embedding = user_summary_embedding['item_emb:float_seq'].values[0]
    user_summary_embedding = torch.tensor(user_summary_embedding).to(device)
    item_profile_embedding = item_profile_embedding.to(device)
    similarity_scores = []
    with torch.no_grad():  # Disable gradient computation for inference
        similarity_scores = torch.matmul(user_summary_embedding, item_profile_embedding.T)
    sort_idx = torch.argsort(similarity_scores, descending=True)
    return sort_idx

def parse_rerank_output(self, output: str) -> list:
    """
    Parses the rerank output generated by the model.

    Args:
        output (str): The generated output containing the reranked list.

    Returns:
        list: A list of integers representing the reranked item indices.
    """
    # Ensure output starts with BOS and ends with EOS
    if not (output.startswith(self.BOS) and output.endswith(self.EOS)):
        raise ValueError("Output format incorrect: Missing BOS or EOS tokens.")

    # Extract content between BOS and EOS
    content = output[len(self.BOS):-len(self.EOS)].strip()

    # Validate and extract the list of integers
    match = re.match(r"^\d+(,\d+)*$", content)
    if not match:
        raise ValueError("Output format incorrect: Expected a comma-separated list of integers.")

    # Convert to list of integers
    return list(map(int, content.split(',')))
    
