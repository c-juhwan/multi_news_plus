# Standard Library Modules
import os
import argparse
# Custom Modules
from utils.utils import parse_bool

class ArgParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.user_name = os.getlogin()
        self.proj_name = 'MultiNewsPlus'

        # Task arguments
        task_list = ['summarization']
        self.parser.add_argument('--task', type=str, choices=task_list, default='summarization',
                                 help='Task to do; Must be given.')
        job_list = ['preprocessing', 'training', 'resume_training', 'testing']
        self.parser.add_argument('--job', type=str, choices=job_list, default='training',
                                 help='Job to do; Must be given.')
        dataset_list = ['multi_news', 'multi_news_plus']
        self.parser.add_argument('--task_dataset', type=str, choices=dataset_list, default='multi_news',
                                 help='Dataset for the task; Must be given.')
        self.parser.add_argument('--description', type=str, default='default',
                                 help='Description of the experiment; Default is "default"')

        # Path arguments - Modify these paths to fit your environment
        self.parser.add_argument('--data_path', type=str, default=f'./cleansing',
                                 help='Path to the dataset.')
        self.parser.add_argument('--preprocess_path', type=str, default=f'./preprocessed/{self.proj_name}',
                                 help='Path to the preprocessed dataset.')
        self.parser.add_argument('--model_path', type=str, default=f'./model_final/{self.proj_name}',
                                 help='Path to the model after training.')
        self.parser.add_argument('--checkpoint_path', type=str, default=f'./model_checkpoint/{self.proj_name}')
        self.parser.add_argument('--result_path', type=str, default=f'./results/{self.proj_name}',
                                 help='Path to the result after testing.')
        self.parser.add_argument('--log_path', type=str, default=f'./tensorboard_log/{self.proj_name}',
                                 help='Path to the tensorboard log file.')

        # Model - Basic arguments
        self.parser.add_argument('--proj_name', type=str, default=self.proj_name,
                                 help='Name of the project.')
        model_type_list = ['bart', 't5']
        self.parser.add_argument('--model_type', type=str, choices=model_type_list, default='bart',
                                 help='Type of the classification model to use.')
        self.parser.add_argument('--model_ispretrained', type=parse_bool, default=True,
                                 help='Whether to use pretrained model; Default is True')
        self.parser.add_argument('--min_seq_len', type=int, default=4,
                                 help='Minimum sequence length of the input; Default is 4')
        self.parser.add_argument('--max_seq_len', type=int, default=1024,
                                 help='Maximum sequence length of the input; Default is 1024')
        self.parser.add_argument('--dropout_rate', type=float, default=0.0,
                                 help='Dropout rate of the model; Default is 0.0')
        self.parser.add_argument('--doc_sep_token', type=str, default='<multidoc_sep>',
                                 help='Separator token for input; Default is <multidoc_sep>')

        # Model - Optimizer & Scheduler arguments
        optim_list = ['SGD', 'AdaDelta', 'Adam', 'AdamW']
        scheduler_list = ['None', 'StepLR', 'LambdaLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau']
        self.parser.add_argument('--optimizer', type=str, choices=optim_list, default='Adam',
                                 help="Optimizer to use; Default is Adam")
        self.parser.add_argument('--scheduler', type=str, choices=scheduler_list, default='None',
                                 help="Scheduler to use for classification; If None, no scheduler is used; Default is None")

        # Training arguments 1
        self.parser.add_argument('--num_epochs', type=int, default=3,
                                 help='Training epochs; Default is 3')
        self.parser.add_argument('--learning_rate', type=float, default=2e-5,
                                 help='Learning rate of optimizer; Default is 2e-5')
        # Training arguments 2
        self.parser.add_argument('--num_workers', type=int, default=2,
                                 help='Num CPU Workers; Default is 2')
        self.parser.add_argument('--batch_size', type=int, default=4,
                                 help='Batch size; Default is 4')
        self.parser.add_argument('--grad_accum_steps', type=int, default=4,
                                 help='Gradient accumulation steps; Default is 4')
        self.parser.add_argument('--weight_decay', type=float, default=0,
                                 help='Weight decay; Default is 0; If 0, no weight decay')
        self.parser.add_argument('--clip_grad_norm', type=int, default=0,
                                 help='Gradient clipping norm; Default is 0')
        self.parser.add_argument('--early_stopping_patience', type=int, default=10,
                                 help='Early stopping patience; No early stopping if None; Default is None')
        self.parser.add_argument('--train_valid_split', type=float, default=0.1,
                                 help='Train/Valid split ratio; Default is 0.1')
        objective_list = ['loss']
        self.parser.add_argument('--optimize_objective', type=str, choices=objective_list, default='loss',
                                 help='Objective to optimize; Default is loss')

        # Testing/Inference arguments
        self.parser.add_argument('--test_batch_size', default=4, type=int,
                                 help='Batch size for test; Default is 4')
        self.parser.add_argument('--num_beams', default=3, type=int,
                                 help='Number of beams for beam search; Default is 3')

        # Other arguments - Device, Seed, Logging, etc.
        self.parser.add_argument('--device', type=str, default='cuda',
                                 help='Device to use for training; Default is cuda')
        self.parser.add_argument('--seed', type=int, default=None,
                                 help='Random seed; Default is None')
        self.parser.add_argument('--use_tensorboard', type=parse_bool, default=True,
                                 help='Using tensorboard; Default is True')
        self.parser.add_argument('--use_wandb', type=parse_bool, default=True,
                                 help='Using wandb; Default is True')
        self.parser.add_argument('--log_freq', default=500, type=int,
                                 help='Logging frequency; Default is 500')

    def get_args(self):
        return self.parser.parse_args()
