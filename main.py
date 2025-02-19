import argparse
import os
import sys

from config import Config  # or you can import direct constants
from data.dataset import UserLevelDataset
from data.memory_bank import MemoryBank
from training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='dataset')
    parser.add_argument('--task_name', type=str, default='Movies_and_TV')
    parser.add_argument('--pretrain_model', type=str, default='LightGCN')
    parser.add_argument('--type_list', nargs='+', default=["inter", "profile"])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--verify_steps', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--initial_user_seq_length', type=int, default=10)
    parser.add_argument('--hist_seq_length', type=int, default=20)
    parser.add_argument('--train_user_batch', type=int, default=70)
    parser.add_argument('--test_user_batch', type=int, default=30)
    parser.add_argument('--llm_model', type=str, default='gpt-4o')
    parser.add_argument('--load_check_point', type=str, default='')
    parser.add_argument('--generate_user_profile', type=int, default=1)
    parser.add_argument('--use_distance', type=int, default=1)
    parser.add_argument('--minimum_train', type=int, default=0)
    parser.add_argument('--use_generalization', type=int, default=1)
    parser.add_argument('--cot', type=int, default=0)
    parser.add_argument('--hist_seq_mask', type=float, default=0)

    return parser.parse_args()

def main():
    args = parse_args()
    config = vars(args)

    # Optionally merge Config from config.py
    # config = {**Config.DEFAULTS, **config}

    # Initialize dataset
    data_loader = UserLevelDataset(
        root=config['root'],
        task_name=config['task_name'],
        type_list=config['type_list'],
        batch_size=config['batch_size'],
        shuffle=False
    )

    # Initialize memory bank
    memory_bank = MemoryBank(data_loader)

    # Initialize trainer and run
    trainer = Trainer(
        config=config,
        data_loader=data_loader,
        memory_bank=memory_bank,
        prompt_bank=prompt_bank,
        bos_token='|beginoftext|',
        eos_token='|endoftext|'
    )
    trainer.run()

if __name__ == "__main__":
    main()
