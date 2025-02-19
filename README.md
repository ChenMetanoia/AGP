# AGP: Auto-Guided Prompt Refinement for Personalized Reranking in Recommender Systems

Welcome to the anonymous repository for **AGP: Auto-Guided Prompt Refinement for Personalized Reranking in Recommender Systems**.

## Overview
AGP introduces an automated prompt optimization framework designed to enhance personalized reranking in recommender systems. By iteratively refining prompts based on feedback from model performance, AGP dynamically adapts to improve recommendation quality.

## Features
- **Automated Prompt Refinement**: Iterative optimization of prompts based on model feedback.
- **Personalized Reranking**: Tailors reranking strategies for improved user experience.
- **Seamless Integration**: Compatible with existing recommendation models.

## Dataset Availability
Preprocessed datasets will be published after acceptance:
- **Movies and TV Dataset**
- **Yelp Dataset**
- **Goodreads Dataset**

## OpenAI API Key Setup
To configure OpenAI API access, set `openai.api_key` and `openai.base_url` in the `item_model()` function.

Training logs are automatically saved under `./log`.

## Hyperparameter Configuration
Key hyperparameters for fine-tuning are defined in the third cell (# Configuration):

### 1. `verify_step`
- Defines the number of steps for optimizing the prompt on each user.
- Example: If `verify_step=3`, the prompt is optimized for each user three times, retaining it only if it improves reranking performance.

### 2. `batch_size`
- Determines the number of users whose interaction data is used for prompt optimization in each batch.

### 3. `hist_seq_length`
- Specifies the number of historical items per user used for prompt optimization.

### 4. `train_user_batch` & `test_user_batch`
- Training users per epoch = `train_user_batch * batch_size`
- Test users per epoch = `test_user_batch * batch_size`

### 5. `pretrain_model`
- Supports `SASRec` and `LightGCN` as pretrained collaborative filtering models.

## Getting Started
To use AGP, follow these steps:
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run an example experiment:
   ```bash
   python prompt_optimize_movie_recommendation.py --llm_model gpt-4o-mini-2024-07-18 --cot 1 --task_name Movies_and_TV --minimum_train 0 --verify_steps 1 --hist_seq_length 5 --use_distance 0 --batch_size 10
   ```

---
**Note**: This repository is maintained anonymously for the peer-review process.

