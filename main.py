from utils.argutils import print_args
from encoder.solver import SingerIdentityEncoder
from pathlib import Path
import argparse, os, pdb

def str2bool(v):
    return v.lower() in ('true')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains the speaker encoder. You must have run encoder_preprocess.py first.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-r", "--run_id", type=str, default='testRuns', help= \
        "Name for this model instance. If there is already a model named same as this in models_dir, then it uses the params from that")
    parser.add_argument("-nr", "--new_run_id", type=str, default=None, help= \
        "Name for new model directory. If model is None, error will occur"
        "restart from scratch.")
    parser.add_argument("-d", "--clean_data_root", type=Path, default="/homes/bdoc3/my_data/spmel_data/vocalSet_subset_unnormed/train", help= \
        "Path to the output directory of encoder_preprocess.py. If you left the default "
        "output directory when preprocessing, it should be <datasets_root>/SV2TTS/encoder/.")
    parser.add_argument("-m", "--models_dir", type=Path, default="encoder/saved_models/", help=\
        "Path to the output directory that will contain the saved model weights, as well as "
        "backups of those weights and plots generated during training.")
    parser.add_argument("-tb", "--tb_every", type=int, default=10, help= \
        "Number of steps between updates of the loss and the plots.")
    parser.add_argument("-s", "--save_every", type=int, default=500, help= \
        "Number of steps between updates of the model on the disk. Set to 0 to never save the "
        "model.")
    parser.add_argument("-t", "--train_iters", type=int, default=200)
    parser.add_argument("-v", "--val_iters", type=int, default=10)
    # parser.add_argument("-b", "--backup_every", type=int, default=7500, help= \
        # "Number of steps between backups of the model. Set to 0 to never make backups of the "
        # "model.")
    parser.add_argument("-c", "--which_cuda", type=int, default=0, help= \
        "Do not load any saved model.")
    parser.add_argument("-l", "--use_loss", type=str, default='both')
    parser.add_argument("-stp", "--stop_at_step", type=int, default=1000)
    parser.add_argument("-n", "--notes", type=str, default='', help= \
        "Add these notes which will be saved to a config text file that gets saved in your saved directory")
    config = parser.parse_args()
    
    # Process the arguments
    config.models_dir.mkdir(exist_ok=True)
    config.string_sum = str(config)
    print_args(config, parser)

    encoder = SingerIdentityEncoder(config)

    encoder.train()
    