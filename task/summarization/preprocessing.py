# Standard Library Modules
import os
import sys
import json
import pickle
import random
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarning for pandas
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path

def preprocessing(args: argparse.Namespace) -> pd.DataFrame:
    if args.task_dataset == 'multi_news':
        dataset_path = os.path.join(args.data_path, 'raw_data.json')
    elif args.task_dataset == 'multi_news_plus':
        dataset_path = os.path.join(args.data_path, 'cleansed_data.json')

    # Load the dataset - read json file into a pandas dataframe
    data_df = pd.read_json(dataset_path, orient='records')

    data_dict = {
        'train': {
            'source_text': [],
            'target_text': []
        },
        'val': {
            'source_text': [],
            'target_text': []
        },
        'test': {
            'source_text': [],
            'target_text': []
        }
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset)
    check_path(preprocessed_path)

    for idx in tqdm(range(len(data_df)), desc='Preprocessing data'):
        if data_df['document_num'][idx] == 0:
            continue # Exclude the data with no document

        # Extract data
        target_text = data_df['summary'][idx]
        split = data_df['split'][idx]

        # Load source documents - list of strings
        source_documents = data_df['document'][idx] if args.task_dataset == 'multi_news' else data_df['cleansed_document'][idx]
        # Remove "" from the list of strings
        source_documents = [doc for doc in source_documents if doc != ""]

        # Concatenate source documents into one string - use <multidoc_sep> as separator token. Give space before and after the separator token
        source_text = f" {args.doc_sep_token} ".join(source_documents)
        # source_text = " ".join(source_documents) # Just concatenate the source documents without separator token
        # source_text = "\n".join(source_documents) # Concatenate the source documents with newline character

        # Append data to the dictionary
        data_dict[split]['source_text'].append(source_text)
        data_dict[split]['target_text'].append(target_text)

    # Change the key name from 'val' to 'valid'
    data_dict['valid'] = data_dict.pop('val')

    assert len(data_dict['train']['source_text']) == len(data_dict['train']['target_text']), "Train data length mismatch"
    assert len(data_dict['valid']['source_text']) == len(data_dict['valid']['target_text']), "Valid data length mismatch"
    assert len(data_dict['test']['source_text']) == len(data_dict['test']['target_text']), "Test data length mismatch"

    # Save the data_dict for each split as pickle file
    for split in ['train', 'valid', 'test']:
        with open(os.path.join(preprocessed_path, f'{split}.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)
