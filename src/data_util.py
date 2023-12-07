import json
import datasets
import numpy as np

def load_data(dataset_name):
    if dataset_name == 'ambigqa':
        data_path = 'logs/dataset/ambigqa/ambigqa_dev_balance.json'
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    elif dataset_name == 'ambig_inst':
        data_path = 'logs/dataset/ambiginst.json'
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    elif dataset_name == 'nq_open':
        all_logs = datasets.load_dataset('nq_open', cache_dir = 'dataset_cache')
        test_data = all_logs['validation'].select(np.arange(200))
        test_data = [test_data[x] for x in range(200)]
        return test_data
    elif dataset_name == 'gsm8k':
        gsm8k = datasets.load_dataset('gsm8k', 'main')
        test_data = gsm8k['test'].select(np.arange(200))
        test_data = [test_data[x] for x in range(200)]
        return test_data
