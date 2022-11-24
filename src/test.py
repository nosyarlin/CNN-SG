import os
import sys
import torch
import argparse
import pandas as pd
from torch import nn
from models import get_model
from dataset import get_dataloader
from shared_funcs import evaluate_model, load_checkpoint, check_img_size
from config import (
    PREPROCESSED_IMAGE_DIR, MODEL_FILEPATH, TEST_FILEPATH,
    HYPERPARAMETERS_FILEPATH, RESULTS_DIR)


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Process Command-line Arguments')
    parser.add_argument(
        '--image_dir',
        default=PREPROCESSED_IMAGE_DIR,
        help='Path to the directory containing the images')
    parser.add_argument(
        '--model_path',
        default=MODEL_FILEPATH,
        help='Path to saved model weights')
    parser.add_argument(
        '--xy_test', default=TEST_FILEPATH,
        help='Path to xy dataframe')
    parser.add_argument(
        '--hyperparameters',
        default=HYPERPARAMETERS_FILEPATH,
        help='Path to hyperparameters dataframe')
    parser.add_argument(
        '--path_to_save_results',
        default=RESULTS_DIR,
        help='Path to the directory to save the model, hyperparameters and \
            results')
    parser.add_argument(
        '--num_workers', default='4', type=int,
        help='Number of threads to be set for loading data')
    parser.add_argument(
        '--img_resize', action='store_true',
        help='Resize each image before training/testing')
    parser.add_argument(
        '--use_cpu', action='store_true',
        help='Using CPU for processing'
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
    batch_size = int(hp.loc[hp['Hyperparameters']
                     == 'BatchSize', 'Values'].item())
    img_size = int(hp.loc[hp['Hyperparameters']
                   == 'ImgSize', 'Values'].item())
    crop_size = int(hp.loc[hp['Hyperparameters']
                    == 'CropSize', 'Values'].item())
    
    # Check that paths to save results exist and is empty
    if os.path.exists(args.path_to_save_results) and \
       len(os.listdir(args.path_to_save_results)) == 0:
        print("\nSaving results in {}\n".format(args.path_to_save_results))
    else:
        sys.exit(
            "\nError: File path to save results do not exist, or directory is " \
            "not empty"
        )

    # Get test data
    xy_test = pd.read_csv(args.xy_test)
    if not args.img_resize:
        # Check img size of first image
        check_img_size(xy_test.FileName[0], "testing set", img_size)

    test_dl = get_dataloader(
        xy_test.FileName, xy_test.SpeciesCode, batch_size, args.image_dir,
        crop_size, False, args.num_workers, args.img_resize, img_size
    )

    print("Dataset to be used includes {} testing images.".format(
        len(xy_test.FileName)))

    # Create Model
    model, _ = get_model(archi, num_classes, False, False, dropout)
    load_checkpoint(args.model_path, model)

    if not args.use_cpu:
        model.cuda()
        device = torch.device('cuda')
        print(
            "\nUsing {} with cuDNN version {} for" \
            "testing with {} architecture."
            .format(
                device, torch.backends.cudnn.version(), archi
            ))
    else:
        device = torch.device('cpu')
        print(
            "\nUsing {} WITHOUT cuDNN for testing with {} architecture."
            .format(
                device, archi
            ))

    # Run data through model
    loss_func = nn.CrossEntropyLoss()
    test_acc, test_loss, probabilities = evaluate_model(
        model, test_dl, loss_func, device
    )
    print("\nTesting complete. Test acc: {}, Test loss: {}\n".format(
        test_acc, test_loss))

    # Saving results, probabilities and metadata
    probabilities = probabilities.T.tolist()
    test_probs_df = pd.DataFrame({
        'file_name': xy_test.FileName,
        'prob_empty': probabilities[0],
        'prob_human': probabilities[1],
        'prob_animal': probabilities[2]}
    )
    test_probs_df.to_csv(
        index=False,
        path_or_buf=os.path.join(
            args.path_to_save_results,
            'test_probabilities.csv'
        )
    )

    test_results_df = pd.DataFrame({'Acc': [test_acc], 'Loss': [test_loss]})
    test_results_df.to_csv(
        index=False,
        path_or_buf=os.path.join(
            args.path_to_save_results,
            'test_results.csv'
        )
    )

    test_metadata_keys = (
        "TrainedModel", "ModelID", "TestSet", "TestSetSize",
        "HyperparametersPath"
    )
    test_metadata_values = (
        args.model_path,
        args.xy_test.split("\\")[-1],
        len(xy_test.FileName),
        args.hyperparameters,
    )

    test_metadata = pd.DataFrame(
        {'Metadata': test_metadata_keys, 'Values': test_metadata_values}
    )
    test_metadata.to_csv(index=False, path_or_buf=os.path.join(
        args.path_to_save_results, 'testing_metadata.csv'))
