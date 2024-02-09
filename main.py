# Standard Library Modules
import time
import argparse
# Custom Modules
from utils.arguments import ArgParser
from utils.utils import check_path, set_random_seed

def main(args: argparse.Namespace) -> None:
    # Set random seed
    if args.seed not in [None, 'None']:
        set_random_seed(args.seed)

    start_time = time.time()

    # Check if the path exists
    for path in []:
        check_path(path)

    # Get the job to do
    if args.job == None:
        raise ValueError('Please specify the job to do.')
    else:
        if args.task == 'summarization':
            if args.job == 'preprocessing':
                from task.summarization.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.summarization.train import training as job
            elif args.job == 'testing':
                from task.summarization.test import testing as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        else:
            raise ValueError(f'Invalid task: {args.task}')

    # Do the job
    job(args)

    elapsed_time = time.time() - start_time
    print(f'Completed {args.job}; Time elapsed: {elapsed_time / 60:.2f} minutes')

if __name__ == '__main__':
    # Parse arguments
    parser = ArgParser()
    args = parser.get_args()

    # Run the main function
    main(args)
