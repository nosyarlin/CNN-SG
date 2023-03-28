import os
import torch
import argparse
import pandas as pd
from models import get_model
from shared_funcs import load_checkpoint
from config import MODEL_FILEPATH, HYPERPARAMETERS_FILEPATH, RESULTS_DIR


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Process Command-line Arguments')
    parser.add_argument(
        '--model_path',
        default=MODEL_FILEPATH,
        help='Path to saved model weights'
    )
    parser.add_argument(
        '--hyperparameters',
        default=HYPERPARAMETERS_FILEPATH,
        help='Path to hyperparameters dataframe'
    )
    parser.add_argument(
        '--results_path',
        default=RESULTS_DIR,
        help='Path to the directory to save the model, hyperparameters ' \
             'and results'
    )
    return parser


if __name__ == '__main__':
    
    parser = get_arg_parser()
    args = parser.parse_args()

    # Get hyperparameters of the saved model
    hp = pd.read_csv(args.hyperparameters)
    archi = hp.loc[hp['Hyperparameters']
                    == 'Architecture', 'Values'].item()
    num_classes = int(hp.loc[hp['Hyperparameters']
                        == 'NumClasses', 'Values'].item())
    dropout = float(hp.loc[hp['Hyperparameters']
                    == 'Dropout', 'Values'].item())

    model, _ = get_model(archi, num_classes, False, False, dropout)
    load_checkpoint(args.model_path, model)

    model.eval()
    example = torch.rand(1, 3, 320, 480)
    traced_script_module = torch.jit.trace(model, example)

    model_save_path = os.path.join(args.results_path, "model.pt")
    traced_script_module.save(model_save_path)
