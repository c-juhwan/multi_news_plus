# Standard Library Modules
import os
import sys
import pickle
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
from torch.utils.data.dataset import Dataset
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class SummarizationDataset(Dataset):
    def __init__(self, args, data_path:str, split:str) -> None:
        super(SummarizationDataset, self).__init__()
        self.args = args
        self.split = split
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        self.data_list = []

        for idx in tqdm(range(len(data_['source_text'])), desc=f'Loading data from {data_path}'):
            self.data_list.append({
                'index': idx,
                'source_text': data_['source_text'][idx],
                'target_text': data_['target_text'][idx],
            })

        del data_

    def __getitem__(self, idx:int) -> dict:
        # Tokenize input text
        index = self.data_list[idx]['index']
        source_text = "Summarize: " + self.data_list[idx]['source_text'] if self.args.model_type == 't5' else self.data_list[idx]['source_text']
        target_text = self.data_list[idx]['target_text']

        return {
            'index': index,
            'source_text': source_text,
            'target_text': target_text,
        }

    def __len__(self) -> int:
        return len(self.data_list)

def collate_fn(data):
    # Merge a list of samples to form a mini-batch
    index = [sample['index'] for sample in data] # list of integers (batch_size)
    source_text = [sample['source_text'] for sample in data] # list of strings
    target_text = [sample['target_text'] for sample in data] # list of strings

    return {
        'index': index,
        'source_text': source_text,
        'target_text': target_text,
    }