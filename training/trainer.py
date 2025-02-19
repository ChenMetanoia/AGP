import os
import json
import numpy as np
import concurrent.futures
from datetime import datetime
from tqdm import tqdm

from utils.tools import mask_items
from utils.util import ndcg_score, extract_last_block
from models.item_model import item_model
from prompts.prompt_bank import PromptBank
# from your_code import BOS, EOS, increment  # or define them in this file

class Trainer:
    """
    The Trainer class encapsulates the main training logic for iterative
    prompt optimization and testing.
    """

    def __init__(
        self,
        config: dict,
        data_loader,
        memory_bank,
        prompt_bank,
        bos_token=None,
        eos_token=None,
    ):
        """
        Initialize the Trainer with configuration, dataset, and prompt bank.

        Args:
            config (dict): Configuration dictionary (parsed arguments, etc.).
            data_loader (iterable): An iterable dataset object yielding batches.
            memory_bank: An object with methods to retrieve item profiles & embeddings.
            prompt_bank: A PromptBank instance providing all relevant prompt templates.
            bos_token (str): Beginning-of-string token (e.g., '|beginoftext|').
            eos_token (str): End-of-string token (e.g., '|endoftext|').
        """
        self.config = config
        self.data_loader = data_loader
        self.memory_bank = memory_bank
        self.prompt_bank = prompt_bank

        # Tokens (if needed for parsing model outputs)
        self.BOS = bos_token or BOS
        self.EOS = eos_token or EOS

        # Set up logging
        self._setup_logging()

        # Some metrics or state
        self.metrics = {
            'train_ranking': [],
            'test_ranking': [],
            'train_ndcg': [],
            'test_ndcg': [],
            'sr_ranking': [],
            'sr_ndcg': []
        }

        # Decide initial prompt
        self.prompt_to_optimize = (
            user_profile_generation_prompt
            if self.config['generate_user_profile'] else item_profile_generation_prompt
        )

        # Optionally load from checkpoint
        if self.config['load_check_point']:
            loaded = extract_last_block(
                os.path.join(self.log_file_path, self.config['load_check_point'])
            )
            self.prompt_to_optimize = loaded
            print('Loaded checkpoint prompt:\n', loaded)

    # --------------------------------------------------------------------------
    # HELPER METHODS
    # --------------------------------------------------------------------------
    def _setup_logging(self) -> None:
        """
        Prepare output/log directories and file names based on configuration.
        """
        suffix = (
            'profile_generation'
            if self.config['generate_user_profile'] else 'reranking'
        )

        self.log_file_path = f"./logs/{self.config['pretrain_model']}/{self.config['task_name']}/{suffix}/"
        self.profile_path = os.path.join(self.log_file_path, "saved_user_profile")
        self.apicall_path = os.path.join(self.log_file_path, "api_call_logs")

        os.makedirs(self.log_file_path, exist_ok=True)
        os.makedirs(self.profile_path, exist_ok=True)
        os.makedirs(self.apicall_path, exist_ok=True)

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = (
            f"llmModel{self.config['llm_model']}_epochs{self.config['epochs']}"
            f"_verifySteps{self.config['verify_steps']}_batchSize{self.config['batch_size']}"
            f"_histSeqLength{self.config['hist_seq_length']}_useGeneralization{self.config['use_generalization']}"
            f"_minimumTrain{self.config['minimum_train']}_cot{self.config['cot']}"
            f"_histseqmask{self.config['hist_seq_mask']}_{current_time}.txt"
        )

        self.log_file_name = os.path.join(self.log_file_path, file_name)
        self.prefix = (
            f"llmModel{self.config['llm_model']}_epochs{self.config['epochs']}"
            f"_verifySteps{self.config['verify_steps']}_batchSize{self.config['batch_size']}"
            f"_histSeqLength{self.config['hist_seq_length']}_useGeneralization{self.config['use_generalization']}"
            f"_minimumTrain{self.config['minimum_train']}"
        )

        self.profile_save_begin = os.path.join(self.profile_path, "begin_" + file_name)
        self.profile_save_end = os.path.join(self.profile_path, "end_" + file_name)
        self.apicall_file = os.path.join(self.apicall_path, file_name)

    # --------------------------------------------------------------------------
    # MAIN PUBLIC METHOD
    # --------------------------------------------------------------------------
    def run(self) -> None:
        """
        Entry point to run the entire training procedure:
        1. Determine train/test user splits.
        2. For each epoch, run training batches, optimize prompt, then run tests.
        3. Log metrics and finalize.
        """
        # Determine # of train/test users based on config
        if self.config['pretrain_model'] == 'LightGCN' and self.config['task_name'] == 'Movies_and_TV':
            train_user, test_user, minimum_ratio = 800, 200, 8
        else:
            train_user, test_user, minimum_ratio = 700, 300, 7

        train_num = train_user // self.config['batch_size']
        test_num = test_user // self.config['batch_size']

        with open(self.log_file_name, 'w') as log_file:
            for epoch in range(self.config['epochs']):
                train_ranking, test_ranking = [], []
                train_ndcg, test_ndcg = [], []
                test_sr_ranking, test_sr_ndcg = [], []

                # data_loader is an iterator that yields batch_data
                for steps, batch_data in enumerate(tqdm(self.data_loader, position=0)):

                    # ------------------
                    # TRAIN
                    # ------------------
                    if steps < train_num:
                        # Skip if minimum_train is set
                        if self.config['minimum_train'] == 1 and steps >= train_num // minimum_ratio:
                            continue

                        # Attempt to refine prompt verify_steps times
                        for _ in range(self.config['verify_steps']):
                            # Keep old prompt in case we revert
                            old_prompt = self.prompt_to_optimize

                            # 1) Evaluate model on batch_data (loss, ranking, NDCG, etc.)
                            total_loss, total_ranking_val, total_ndcg_val, saved_profile = \
                                self.parallel_process_batches(batch_data, calculate_loss=True)

                            if set(total_loss) == {None}:
                                # All perfect? No need to optimize
                                continue

                            avg_ranking = total_ranking_val

                            # 2) Possibly generalize reasons for failures
                            generalized_failed_cases = None
                            if self.config['use_generalization']:
                                filtered_instructions = [
                                    f"Instruction {i}: {loss}" 
                                    for i, loss in enumerate(total_loss) 
                                    if loss is not None
                                ]
                                gen_prompt = self.prompt_bank.generalize_failed_cases_prompt(filtered_instructions)
                                generalized_failed_cases = item_model(gen_prompt, self.config)
                                increment()

                            # 3) Optimize prompt
                            if self.config['use_distance']:
                                optimization_prompt = self.prompt_bank.generate_optimization_prompt(
                                    self.prompt_to_optimize,
                                    total_loss,
                                    avg_ranking,
                                    generalized_failed_cases
                                )
                            else:
                                # If not using distance, pass None
                                optimization_prompt = self.prompt_bank.generate_optimization_prompt(
                                    self.prompt_to_optimize,
                                    total_loss,
                                    None,
                                    generalized_failed_cases
                                )

                            updated_prompt_text = item_model(optimization_prompt, self.config)
                            increment()

                            # Extract the refined prompt from BOS/EOS if present
                            try:
                                updated_prompt_text = (
                                    updated_prompt_text
                                    .split(self.BOS, 1)[1]
                                    .split(self.EOS, 1)[0]
                                    .strip()
                                )
                            except Exception:
                                print('Updated prompt in illegal format:\n', updated_prompt_text)

                            # Temporarily accept
                            self.prompt_to_optimize = updated_prompt_text
                            log_file.write(f"\nOriginal prompt:\n{old_prompt}\n")
                            log_file.write(f"Average Ranking: {avg_ranking}\n")

                            # 4) Re-check performance with updated prompt
                            _, updated_avg_ranking, updated_ndcg_val, _ = \
                                self.parallel_process_batches(batch_data, calculate_loss=False)

                            # If performance doesn't improve, revert
                            if updated_avg_ranking >= avg_ranking:
                                self.prompt_to_optimize = old_prompt
                                log_file.write(f"Updated prompt **discarded**:\n{old_prompt}\n")
                            else:
                                log_file.write(f"Updated prompt **accepted**:\n{self.prompt_to_optimize}\n")
                                total_ndcg_val = updated_ndcg_val
                                avg_ranking = updated_avg_ranking

                        # Save user profiles if relevant
                        self._save_intermediate_profiles(
                            saved_profile, steps, train_num
                        )

                        train_ranking.append(avg_ranking)
                        train_ndcg.append(total_ndcg_val)
                        log_file.write('Train batch completed.\n')

                        with open(self.apicall_file, "a") as api_log:
                            api_log.write(f"Train batch completed with API calls: {counter}\n")

                    # ------------------
                    # TEST
                    # ------------------
                    elif steps < (train_num + test_num):
                        loss_vals, batched_test_ranking, batched_test_ndcg, _ = \
                            self.parallel_process_batches(batch_data, calculate_loss=False)

                        # loss_vals[0], loss_vals[1] = [ sr_ranking, sr_ndcg ]
                        sr_ranking_val, sr_ndcg_val = loss_vals[0], loss_vals[1]

                        print(f"Original Ranking (SR): {sr_ranking_val}")
                        print(f"Test Reranking: {batched_test_ranking}")

                        test_sr_ranking.append(sr_ranking_val)
                        test_sr_ndcg.append(sr_ndcg_val)
                        test_ranking.append(batched_test_ranking)
                        test_ndcg.append(batched_test_ndcg)

                        log_file.write(f"Original SR Ranking: {sr_ranking_val}\n")
                        log_file.write(f"Test Reranking: {batched_test_ranking}\n")

                        with open(self.apicall_file, "w") as api_log:
                            api_log.write(f"Test batch completed with API calls: {counter}\n")

                    # ------------------
                    # EPOCH COMPLETION
                    # ------------------
                    else:
                        # Summarize metrics for this epoch
                        self.metrics['train_ranking'].append(np.mean(train_ranking))
                        self.metrics['train_ndcg'].append(np.mean(train_ndcg))
                        self.metrics['test_ranking'].append(np.mean(test_ranking))
                        self.metrics['test_ndcg'].append(np.mean(test_ndcg))
                        self.metrics['sr_ranking'].append(np.mean(test_sr_ranking))
                        self.metrics['sr_ndcg'].append(np.mean(test_sr_ndcg))

                        epoch_log = (
                            f"Epoch {epoch} completed with:\n"
                            f"  train_ranking = {self.metrics['train_ranking'][-1]:.4f},\n"
                            f"  train_ndcg    = {self.metrics['train_ndcg'][-1]:.4f},\n"
                            f"  test_ranking  = {self.metrics['test_ranking'][-1]:.4f},\n"
                            f"  test_ndcg     = {self.metrics['test_ndcg'][-1]:.4f},\n"
                            f"  sr_ranking    = {self.metrics['sr_ranking'][-1]:.4f},\n"
                            f"  sr_ndcg       = {self.metrics['sr_ndcg'][-1]:.4f}.\n"
                        )
                        print(epoch_log)
                        log_file.write(epoch_log)
                        break  # Move to next epoch

    # --------------------------------------------------------------------------
    # CORE LOGIC METHODS
    # --------------------------------------------------------------------------
    def get_train_item_only(
        self,
        item_id_list,
        sr_ranking_list,
        initial_user_seq_length: int,
        hist_seq_length: int
    ):
        """
        Extract training/historical items and the target item from a user's sequence.
        Returns (x, y) dicts used to build the model prompt or evaluate the output.
        """
        train_item_id_list = item_id_list[:-1][-hist_seq_length:]
        item_profiles = [
            self.memory_bank.get_item_profile(int(item_id))
            for item_id in train_item_id_list
        ]

        if self.config['hist_seq_mask'] != 0:
            item_profiles = mask_items(item_profiles, self.config['hist_seq_mask'])

        sr_item_profiles = [
            f"<{idx}: {self.memory_bank.get_item_profile(item_id)}>"
            for idx, item_id in enumerate(sr_ranking_list[-initial_user_seq_length:])
        ]

        x_data = {
            'historical_item_profiles': "\n".join(item_profiles)
                if len(item_profiles) > 1
                else item_profiles[0],
            'sr_item_profiles': "\n".join(sr_item_profiles)
                if len(sr_item_profiles) > 1
                else sr_item_profiles[0],
        }

        test_item_id = int(item_id_list[-1])
        target_item_pos = sr_ranking_list.index(test_item_id)

        y_data = {
            'target_item_profile': self.memory_bank.get_item_profile(test_item_id),
            'target_item_id': str(target_item_pos)
        }
        return x_data, y_data

    def process_user_data(self, user_data: dict, calculate_loss: bool = True):
        """
        Process a single user's data. 
        - Builds the relevant prompt (user profile or item profile).
        - Gets the model response (reranked items).
        - Evaluates ranking (NDCG, position).
        - Optionally calculates a "loss" or "reason" prompt if the top item is incorrect.

        Returns: (loss, total_ranking, ndcg, saved_prompt)
        """
        total_ranking = 0
        saved_prompt = {}

        item_id_list = user_data['item_id_list']
        sr_ranking_list = user_data['sr_ranking_list']
        initial_len = self.config['initial_user_seq_length']
        hist_len = self.config['hist_seq_length']

        x, y = self.get_train_item_only(item_id_list, sr_ranking_list, initial_len, hist_len)

        y_item_id = y['target_item_id']
        y_item_profile = y['target_item_profile']
        rerank_length = len(sr_ranking_list[-initial_len:])

        # Generate user profile vs. direct item prompt
        if self.config['generate_user_profile']:
            # Step 1: Generate user profile
            user_prompt = self.prompt_bank.create_user_profile_prompt(
                user_profile_generation_prompt,
                x['historical_item_profiles']
            )
            y_user_profile = item_model(user_prompt, self.config)
            increment()

            # Step 2: Rerank
            rerank_prompt = self.prompt_bank.rerank_by_user_profile_prompt(
                y_user_profile,
                x['sr_item_profiles'],
                rerank_length
            )
            item_response = item_model(rerank_prompt, self.config)
            increment()

            saved_prompt = {
                'profile_generation_prompt': user_profile_generation_prompt,
                'user_historical_items': x['historical_item_profiles'],
                'generated_profile': y_user_profile
            }
        else:
            if self.config['cot'] == 0:
                prompt_text = self.prompt_bank.inject_item_profile_generation_prompt_recall_single(
                    item_profile_generation_prompt,
                    x['historical_item_profiles'],
                    x['sr_item_profiles'],
                    rerank_length
                )
            elif self.config['cot'] == 1:
                prompt_text = self.prompt_bank.rerank_by_cot_prompt(
                    x['historical_item_profiles'],
                    x['sr_item_profiles'],
                    rerank_length
                )
            else:  # config['cot'] == 2
                prompt_text = self.prompt_bank.rerank_directly_prompt(
                    x['historical_item_profiles'],
                    x['sr_item_profiles'],
                    rerank_length
                )
            item_response = item_model(prompt_text, self.config)
            increment()

        # Parse the model response
        predicted_ranking_list = find_first_list(item_response)
        if predicted_ranking_list is not None:
            predicted_item_id = str(predicted_ranking_list[0])
            gt_item_id = int(y_item_id)
            ndcg_val = ndcg_score(gt_item_id, predicted_ranking_list)

            if gt_item_id in predicted_ranking_list:
                predicted_rank = predicted_ranking_list.index(gt_item_id) + 1
                total_ranking += predicted_rank
            else:
                total_ranking += initial_len
        else:
            # Fallback if invalid
            predicted_item_id = None
            if self.config['llm_model'] == 'deepseek':
                ndcg_val = ndcg_score(1, [0,0,0,0,0,0,0,0,0,1])
            else:
                ndcg_val = 0
            predicted_ranking_list = 'Invalid output which is not a list.'
            total_ranking += initial_len

        # Calculate or skip "loss"
        if calculate_loss:
            if y_item_id == predicted_item_id:
                loss = None
            else:
                # Build an explanation prompt for errors
                if self.config['generate_user_profile']:
                    # uses the user profile route
                    reason_prompt = self.prompt_bank.user_profile_evaluation_reason(
                        y_item_profile,
                        x['sr_item_profiles'],
                        saved_prompt.get('generated_profile', ''),
                        rerank_prompt
                    )
                else:
                    # direct reranking route
                    reason_prompt = self.prompt_bank.reranking_evaluation_reason(
                        predicted_ranking_list,
                        y_item_id,
                        y_item_profile,
                        x['sr_item_profiles'],
                        x['historical_item_profiles']
                    )
                loss = item_model(reason_prompt, self.config)
        else:
            # Return (position, baseline ndcg) for reference
            baseline_item = int(item_id_list[-1])
            baseline_ndcg = ndcg_score(
                baseline_item, sr_ranking_list[-initial_len:]
            )
            loss = (int(y['target_item_id']) + 1, baseline_ndcg)

        return loss, total_ranking, ndcg_val, saved_prompt

    def parallel_process_batches(self, batch_data, calculate_loss=True):
        """
        Process a batch of user data in parallel using concurrent.futures.

        Returns:
            total_loss, avg_ranking, avg_ndcg, saved_profile
        """
        losses = []
        saved_profile = []
        total_ranking = 0
        total_ndcg = 0

        batch_size = len(batch_data)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_user_data = {
                executor.submit(
                    self.process_user_data,
                    user_info,
                    calculate_loss
                ): user_info 
                for user_info in batch_data
            }

            for future in concurrent.futures.as_completed(future_to_user_data):
                try:
                    batch_loss, batch_ranking, batch_ndcg, batch_saved = future.result()
                    losses.append(batch_loss)
                    saved_profile.append(batch_saved)
                    total_ranking += batch_ranking
                    total_ndcg += batch_ndcg
                except Exception as exc:
                    print(f'Generated an exception: {exc}')
                    batch_size -= 1

        # Avoid division by zero
        batch_size = max(batch_size, 1)

        if calculate_loss:
            total_loss = losses
        else:
            # If not computing detailed losses, we only store aggregated metrics
            total_loss = np.mean(losses, axis=0)

        return total_loss, total_ranking / batch_size, total_ndcg / batch_size, saved_profile

    def _save_intermediate_profiles(self, saved_profile, steps, train_num) -> None:
        """
        Helper for storing user profiles at different intervals.
        """
        # If minimum_train is on, save them at the beginning
        if self.config['minimum_train'] == 1:
            with open(self.profile_save_begin, "a") as file:
                json.dump(saved_profile, file, indent=4)

        # If generating user profiles, optionally store at early or late steps
        elif self.config['generate_user_profile']:
            if steps < train_num * 0.1:
                with open(self.profile_save_begin, "a") as file:
                    json.dump(saved_profile, file, indent=4)
            elif steps >= train_num * 0.9:
                with open(self.profile_save_end, "a") as file:
                    json.dump(saved_profile, file, indent=4)
