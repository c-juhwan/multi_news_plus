# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import logging
import argparse
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
from nlgeval import NLGEval
from bert_score import BERTScorer
from BARTScore.bart_score import BARTScorer
# Pytorch Modules
import torch
torch.set_num_threads(2)
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Huggingface Modules
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.summarization.dataset import SummarizationDataset, collate_fn
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, check_path, get_huggingface_model_name

def testing(args: argparse.Namespace) -> None:
    device = get_torch_device(args.device)

    # Define logger
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    # Initialize tensorboard writer
    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.log_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    # Load dataset and define dataloader
    write_log(logger, "Loading dataset...")
    dataset_test = SummarizationDataset(args, os.path.join(args.preprocess_path, args.task, 'multi_news_plus', 'test.pkl'), 'test') # For fair comparison, share multi_news_plus test data for all models
    dataloader_test = DataLoader(dataset_test, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    # Get model instance
    write_log(logger, "Building model")
    model_name = get_huggingface_model_name(args.model_type)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, additional_special_tokens=[args.doc_sep_token], model_max_length=args.max_seq_len) # Add special token for input
    # avoid 'sep_token' to be added multiple times - T5 already has '<sep>' token
    model.resize_token_embeddings(len(tokenizer)) # Resize model embedding to fit new tokenizer

    # Load model weights
    write_log(logger, "Loading model weights")
    load_model_name = os.path.join(args.model_path, args.task, args.task_dataset,
                                   f'final_model_{args.model_type}.pt')
    model = model.to('cpu')
    checkpoint = torch.load(load_model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    write_log(logger, f"Loaded model weights from {load_model_name}")

    # Load Wandb
    if args.use_wandb:
        import wandb
        from wandb import AlertLevel
        wandb.init(project=args.proj_name,
                   name=get_wandb_exp_name(args),
                   config=args,
                   notes=args.description,
                   tags=["TEST",
                         f"Dataset: {args.task_dataset}",
                         f"Model: {args.model_type}"])

    del checkpoint

    # Test - Start evaluation
    model = model.eval()
    result_list = []
    ref_list = []
    hyp_list = []

    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc=f'Testing')):
        # Test - Get data from batch
        source_text = data_dicts['source_text']
        target_text = data_dicts['target_text']
        model_inputs = tokenizer(source_text, text_target=None, # Test data does not have target text
                                    padding='max_length', truncation=True,
                                    max_length=args.max_seq_len, return_tensors='pt')
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        with torch.no_grad():
            generated_tokens = model.generate(**model_inputs, max_length=args.max_seq_len, early_stopping=True)
        generated_target = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for idx, (source, target, generated) in enumerate(zip(source_text, target_text, generated_target)):
            result_list.append({
                'source': source,
                'target': target,
                'generated': generated,
            })

        ref_list.extend(target_text)
        hyp_list.extend(generated_target)

    # Delete model to save memory
    model = model.to('cpu')
    del model

    # Test - nlg-eval
    write_log(logger, "TEST - Calculating NLG-eval metrics...")
    Eval = NLGEval(metrics_to_omit=['SPICE', 'CIDEr', 'SkipThoughtCS', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore'])
    BERT_Eval = BERTScorer(device=args.device, model_type='bert-base-uncased')
    BART_Eval = BARTScorer(device=args.device, checkpoint='facebook/bart-large-cnn')

    # I don't know why but we need this
    _strip = lambda x: x.strip()
    ref_list2 = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    metrics_dict = Eval.compute_metrics(ref_list2, hyp_list)
    bert_score_P, bert_score_R, bert_score_F1, bart_score_total = 0, 0, 0, 0

    write_log(logger, "TEST - Calculating BERTScore metrics...")
    bert_score_P, bert_score_R, bert_score_F1 = BERT_Eval.score(ref_list, hyp_list, batch_size=args.test_batch_size)

    write_log(logger, "TEST - Calculating BARTScore metrics...")
    bart_score_total = BART_Eval.score(ref_list, hyp_list, batch_size=args.test_batch_size)

    bert_score_P = bert_score_P.mean().item()
    bert_score_R = bert_score_R.mean().item()
    bert_score_F1 = bert_score_F1.mean().item()
    bart_score_total = sum(bart_score_total) / len(bart_score_total)

    # Final - End of testing
    write_log(logger, f"TEST - Bleu_1: {metrics_dict['Bleu_1']:.4f}")
    write_log(logger, f"TEST - Bleu_2: {metrics_dict['Bleu_2']:.4f}")
    write_log(logger, f"TEST - Bleu_3: {metrics_dict['Bleu_3']:.4f}")
    write_log(logger, f"TEST - Bleu_4: {metrics_dict['Bleu_4']:.4f}")
    write_log(logger, f"TEST - Bleu_avg: {(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4:.4f}")
    write_log(logger, f"TEST - Rouge_L: {metrics_dict['ROUGE_L']:.4f}")
    write_log(logger, f"TEST - Meteor: {metrics_dict['METEOR']:.4f}")
    write_log(logger, f"TEST - BERTScore_Precision: {bert_score_P:.4f}")
    write_log(logger, f"TEST - BERTScore_Recall: {bert_score_R:.4f}")
    write_log(logger, f"TEST - BERTScore_F1: {bert_score_F1:.4f}")
    write_log(logger, f"TEST - BARTScore: {bart_score_total:.4f}")

    # Save data as json file
    save_path = os.path.join(args.result_path, args.task, args.task_dataset)
    check_path(save_path)

    result_dict = {
        'args': vars(args),
        'Bleu_1': metrics_dict['Bleu_1'],
        'Bleu_2': metrics_dict['Bleu_2'],
        'Bleu_3': metrics_dict['Bleu_3'],
        'Bleu_4': metrics_dict['Bleu_4'],
        'Bleu_avg': (metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4,
        'Rouge_L': metrics_dict['ROUGE_L'],
        'Meteor': metrics_dict['METEOR'],
        'BERTScore_Precision': bert_score_P,
        'BERTScore_Recall': bert_score_R,
        'BERTScore_F1': bert_score_F1,
        'BARTScore': bart_score_total,
        'result_list': result_list,
    }
    save_name = os.path.join(save_path, f'test_result_{args.model_type}.json')
    with open(save_name, 'w') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    if args.use_tensorboard:
        writer.add_scalar('TEST/Bleu_1', metrics_dict['Bleu_1'], global_step=0)
        writer.add_scalar('TEST/Bleu_2', metrics_dict['Bleu_2'], global_step=0)
        writer.add_scalar('TEST/Bleu_3', metrics_dict['Bleu_3'], global_step=0)
        writer.add_scalar('TEST/Bleu_4', metrics_dict['Bleu_4'], global_step=0)
        writer.add_scalar('TEST/Bleu_avg', (metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4, global_step=0)
        writer.add_scalar('TEST/Rouge_L', metrics_dict['ROUGE_L'], global_step=0)
        writer.add_scalar('TEST/Meteor', metrics_dict['METEOR'], global_step=0)
        writer.add_scalar('TEST/BERTScore_Precision', bert_score_P, global_step=0)
        writer.add_scalar('TEST/BERTScore_Recall', bert_score_R, global_step=0)
        writer.add_scalar('TEST/BERTScore_F1', bert_score_F1, global_step=0)
        writer.add_scalar('TEST/BARTScore', bart_score_total, global_step=0)

        writer.close()
    if args.use_wandb:
        wandb_df = pd.DataFrame({
            'Dataset': [args.task_dataset],
            'Model': [args.model_type],
            'Bleu_1': [metrics_dict['Bleu_1']],
            'Bleu_2': [metrics_dict['Bleu_2']],
            'Bleu_3': [metrics_dict['Bleu_3']],
            'Bleu_4': [metrics_dict['Bleu_4']],
            'Bleu_avg': [(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4],
            'Rouge_L': [metrics_dict['ROUGE_L']],
            'Meteor': [metrics_dict['METEOR']],
            'BERTScore_Precision': [bert_score_P],
            'BERTScore_Recall': [bert_score_R],
            'BERTScore_F1': [bert_score_F1],
            'BARTScore': [bart_score_total],
        })
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({"TEST_Result": wandb_table})
        wandb.save(save_name)

        wandb.finish()

    return metrics_dict
