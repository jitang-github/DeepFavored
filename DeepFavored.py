import argparse
import sys

from identify import IdentifyWithArgs
from train import TrainWithArgs


def full_parser():
    parser = argparse.ArgumentParser(
        description="DeepFavored: Deep learning based method identifying favored(adaptive) mutations.")
    subparsers = parser.add_subparsers(help="sub-commands")

    train_parser = subparsers.add_parser('train', help="train model")
    train_parser.add_argument('--hyparams', action='store', type=str,
                              help="path to the file documenting hyperparameters for training deepfavored")
    train_parser.add_argument('--trainData', action='store', type=str,
                              help="path to the directory loading training data")
    train_parser.add_argument('--testData', action='store', type=str, help="path to the directory loading test data")
    # train_parser.add_argument('--valid_prop', action='store', type=str, help="proportion of validation data")
    train_parser.add_argument('--modelDir', action='store', type=str,
                              help="path to the directory saving training output")

    identify_parser = subparsers.add_parser('identify',
                                            help="run trained model to identify favored mutation sites")
    identify_parser.add_argument('--modelDir', action='store', type=str,
                                 help="path to the directory saving training output")
    identify_parser.add_argument('--input', action='store', type=str,
                                 help="path to a directory or a file loading sites to be identified")
    identify_parser.add_argument('--outDir', action='store', type=str,
                                 help="path to the directory saving identification output")
    return parser


if __name__ == '__main__':
    runparser = full_parser()
    args = runparser.parse_args()

    # if called with no arguments, print help
    if len(sys.argv) == 1:
        runparser.parse_args(['--help'])

    if sys.argv[1] == 'train':
        TrainWithArgs(args)
    if sys.argv[1] == 'identify':
        IdentifyWithArgs(args)
