"""
Global configuration or default constants.
"""

class Config:
    """
    You can store global default settings or environment logic here.
    """
    DEFAULTS = {
        'root': 'dataset',
        'task_name': 'Movies_and_TV',
        'pretrain_model': 'LightGCN',
        'type_list': ['inter', 'profile'],
        'epochs': 10,
        'verify_steps': 3,
        'batch_size': 10,
        # ...
    }

# Optionally, parse environment variables here, etc.
