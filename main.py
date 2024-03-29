"""
Module: main
Author: Brendan O'Connor
Date: June 5th, 2023

Description: This module processes a set of arguments as appropriate configurations,
creates destination directories, and calls the singer identity encoder model for training.
"""

import argparse
import os
from pathlib import Path
import sys
import warnings

if os.path.abspath('../my_utils') not in sys.path:
    sys.path.insert(1, os.path.abspath('../my_utils'))

from utils import print_args
from solver import SieEncoderTrainer

warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")


def str2bool(v):
    """
    Converts a string representation of a boolean value to its corresponding boolean value.

    Args:
        v (str): String representation of a boolean value.

    Returns:
        bool: Converted boolean value.
    """
    return v.lower() in ("true")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processes a set of arguments to run the singer identity encoder model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # path specifications
    parser.add_argument(
        "-rid",
        "--run_id",
        type=str,
        default="default_model",
        help="Name of destination model directory and associated files. \
        If --new_run_id specified, this becomes the name of the model directory from which ckpt is extracted for pretrained weights",
    )
    parser.add_argument(
        "-nrid",
        "--new_run_id",
        type=str,
        default=None,
        help="If not None, this becomes the name of the new destination model directory and associated files, trained using ckpt from model specified in -run_id.",
    )
    parser.add_argument(
        "-fd",
        "--feature_dir",
        type=Path,
        default="example_feats",
        help="Path to directory of to feature dataset, which must contain train, val directories and feat_params.yaml file",
    )
    parser.add_argument(
        "-pd",
        "--pitch_dir",
        type=Path,
        default="pitch_feats",
        help="Path to directory to pitch feature dataset, which must contain train, val directories and feat_params.yaml file",
    )
    parser.add_argument(
        "-md",
        "--models_dir",
        type=Path,
        default="./sie_models",
        help="Define the parent directory for all model directories",
    )
    parser.add_argument("-a", "--ask", default=True, type=str2bool)

    # schedulers (ints)
    parser.add_argument(
        "-ugi",
        "--use_given_iters",
        type=str2bool,
        default=True,
        help="Uses the arguments for train_iters and val_iters. Otherwise automatically calculated",
    )
    parser.add_argument(
        "-ti",
        "--train_iters",
        type=int,
        default=40,
        help="Default values taken from Damp iters calculation",
    )  # was 2124
    parser.add_argument(
        "-vi",
        "--val_iters",
        type=int,
        default=40,
        help="Default values taken from Damp iters calculation",
    )  # was 308
    parser.add_argument(
        "-p",
        "--patience",
        type=int,
        default=50,
        help="Determines how long EarlyStopping waits before ceasing training",
    )
    parser.add_argument(
        "-et",
        "--earlystop_thresh",
        type=float,
        default=0.0,
        help="Determines how long EarlyStopping waits before ceasing training",
    )
    parser.add_argument(
        "-stp",
        "--stop_at_step",
        type=int,
        default=100,
        help="Upper limit for number of steps before ceasing training",
    )

    # framework setup (ints)
    parser.add_argument(
        "-lri",
        "--learning_rate_init",
        type=int,
        default=1e-4,
        help="Choose which cuda driver to use.",
    )
    parser.add_argument(
        "-spb",
        "--speakers_per_batch",
        type=int,
        default=8,
        help="Choose which cuda driver to use.",
    )
    parser.add_argument(
        "-ups",
        "--utterances_per_speaker",
        type=int,
        default=10,
        help="Choose which cuda driver to use.",
    )
    parser.add_argument(
        "-wc",
        "--which_cuda",
        type=int,
        default=0,
        help="Choose which cuda driver to use.",
    )
    parser.add_argument(
        "-ul",
        "--use_loss",
        type=str,
        default="ge2e",
        help="Choose mode for determining loss value",
    )
    parser.add_argument(
        "-w",
        "--workers",
        default=20,
        type=int,
        help="Number of workers for parallel processing",
    )
    parser.add_argument("-pc", "--pitch_condition", default=False, type=str2bool)

    # model setup (ints)
    parser.add_argument(
        "-hs",
        "--model_hidden_size",
        type=int,
        default=768,
        help="Number of dimensions in hidden layer.",
    )
    parser.add_argument(
        "-es",
        "--model_embedding_size",
        type=int,
        default=256,
        help="Model embedding size.",
    )
    parser.add_argument(
        "-nl",
        "--num_layers",
        type=int,
        default=3,
        help="Number of LSTM stacks in model.",
    )
    parser.add_argument(
        "-nt",
        "--num_timesteps",
        type=int,
        default=128,
        help="Number of timesteps used in feature example fed to network",
    )
    parser.add_argument("-tr", "--tiny_run", default=False, action="store_true")
    parser.add_argument("-eo", "--eval_only", default=False, action="store_true")
    parser.add_argument("-ri", "--restart_iters", default=False, action="store_true")
    parser.add_argument("-uqp", "--using_qian_pretrained", default=False, action="store_true")

    # feat params (bool, str, int)
    parser.add_argument("-ua", "--use_audio", default=False, type=str2bool)
    parser.add_argument("-ft", "--feats_type", default="mel", type=str)

    parser.add_argument("-fdm", "--frame_dur_ms", default=16, type=int)
    parser.add_argument("-nhf", "--num_harm_feats", default=80, type=int)
    parser.add_argument("-naf", "--num_aper_feats", default=0, type=int)
    parser.add_argument("-sr", "--sampling_rate", default=16000, type=int)
    parser.add_argument("-fs", "--fft_size", default=1024, type=int)

    parser.add_argument("-uaf", "--use_aper_feats", default=False, type=str2bool)
    parser.add_argument("-nm", "--norm_method", default=None, type=str)
    parser.add_argument("-wp", "--w2w_process", default="wav2world", type=str)
    parser.add_argument("-drm", "--dim_red_method", default="chandna", type=str)

    # change the fmin/fmax values to 90/7600
    parser.add_argument(
        "-fmin", default=90, type=int
    )  # 50 chosen by me, 71 chosen by default params
    parser.add_argument(
        "-fmax", default=7600, type=int
    )  # 1100 chosen by me, 800 chosen by default params
    parser.add_argument(
        "-n",
        "--notes",
        type=str,
        default="",
        help="Add these notes which will be saved to a config text file that gets saved in your saved directory",
    )

    config = parser.parse_args()

    if config.use_audio == True:
        feat_params = {
            "w2w_process": config.w2w_process,
            "dim_red_method": config.dim_red_method,
            "fmin": config.fmin,
            "fmax": config.fmax,
            "num_harm_feats": config.num_harm_feats,
            "num_aper_feats": config.num_aper_feats,
            "use_aper_feats": config.use_aper_feats,
            "frame_dur_ms": config.frame_dur_ms,
            "sr": config.sampling_rate,
            "fft_size": config.fft_size,
        }

    # Process arguments
    if config.new_run_id is not None:
        config.new_run_id = os.path.join(config.models_dir, config.new_run_id)

    config.models_dir.mkdir(exist_ok=True)
    config.string_sum = str(config)
    print_args(config, parser)

    # initiate and train model till finish
    if config.use_audio == True:
        encoder = SieEncoderTrainer(config, feat_params)
    else:
        encoder = SieEncoderTrainer(config)
    encoder.train()
