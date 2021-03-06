import sys

# for some reason there is an unwatend path in sys.path. Must figure out how to remove this
for i in sys.path:
    if i == '/homes/bdoc3/wavenet_vocoder':
        sys.path.remove(i)

sys.path.insert(1, '/homes/bdoc3/my_utils')

from utils import print_args
from solver import SingerIdentityEncoder
from pathlib import Path
import argparse, pdb


def str2bool(v):
    return v.lower() in ('true')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processes a set of arguments to run the singer identity encoder model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # path specifications    
    parser.add_argument("-rid", "--run_id", type=str, default='testRuns', help= "Name of destination model directory and associated files.\
        If --new_run_id specified,this becomes the name of the model directory from which ckpt is extracted for pretrained weights")
    parser.add_argument("-nrid", "--new_run_id", type=str, default=None, help= \
        "If not None, this becomes the name of the new destination model directory and associated files, trained using ckpt from model specified in -run_id.")
    parser.add_argument("-fd", "--feature_dir", type=Path, default="/homes/bdoc3/my_data/audio_data/deslienced_concat_DAMP", help= \
        "Path to directory of to feature dataset, which must contain train, val directories and feat_params.yaml file")
    parser.add_argument("-md", "--models_dir", type=Path, default="/homes/bdoc3/my_data/autovc_models/singer_identity_encoder/", help=\
        "Define the parent directory for all model directories")
    
    #schedulers (ints)
    parser.add_argument("-te", "--tb_every", type=int, default=10, help= \
        "Number of steps between updates of the loss and the plots for in tensorboard.")
    parser.add_argument("-se", "--save_every", type=int, default=1000, help= \
        "Number of steps between updates of the model on the disk. Overwritten at every save")
    parser.add_argument("-ti", "--train_iters", type=int, default=200, help= "Number of training steps to take before passing back to validation steps")
    parser.add_argument("-vi", "--val_iters", type=int, default=10, help= "Number of validation steps to take before passing back to training steps")
    parser.add_argument("-p", "--patience", type=int, default=35, help= "Determines how long EarlyStopping waits before ceasing training")
    parser.add_argument("-stp", "--stop_at_step", type=int, default=100000, help= "Upper limit for number of steps before ceasing training")

    #framework setup (ints)
    parser.add_argument("-lri", "--learning_rate_init", type=int, default=1e-4, help= "Choose which cuda driver to use.")
    parser.add_argument("-spb", "--speakers_per_batch", type=int, default=8, help= "Choose which cuda driver to use.")
    parser.add_argument("-ups", "--utterances_per_speaker", type=int, default=10, help= "Choose which cuda driver to use.")
    parser.add_argument("-wc", "--which_cuda", type=int, default=0, help= "Choose which cuda driver to use.")
    parser.add_argument("-ul", "--use_loss", type=str, default='ge2e', help= "Choose mode for determining loss value")
    parser.add_argument('-w','--workers', default=16, type=int, help='Number of workers for parallel processing')
    parser.add_argument('-pc','--pitch_condition', default=False, type=str2bool)

    #model setup (ints)
    parser.add_argument("-hs", "--model_hidden_size", type=int, default=256, help= "Number of dimensions in hidden layer.")
    parser.add_argument("-es", "--model_embedding_size", type=int, default=256, help= "Model embedding size.")
    parser.add_argument("-nl", "--num_layers", type=int, default=3, help= "Number of LSTM stacks in model.")
    parser.add_argument("-nt", "--num_timesteps", type=int, default=307, help= "Number of timesteps used in feature example fed to network")
    
    #feat params (bool, str, int)
    parser.add_argument('-ua','--use_audio', default=False, type=str2bool)
    parser.add_argument('-ft','--feats_type', default='world', type=str)
    parser.add_argument('-uaf','--use_aper_feats', default=False, type=str2bool)    
    parser.add_argument('-nm','--norm_method', default=None, type=str)
    parser.add_argument('-wp','--w2w_process', default='wav2world', type=str)
    parser.add_argument('-drm','--dim_red_method', default='chandra', type=str)
    parser.add_argument('-fdm','--frame_dur_ms', default=5, type=int)    
    parser.add_argument('-nhf','--num_harm_feats', default=40, type=int)
    parser.add_argument('-naf','--num_aper_feats', default=4, type=int)
    parser.add_argument('-sr','--sampling_rate', default=16000, type=int)
    parser.add_argument('-fs','--fft_size', default=None, type=int)
    parser.add_argument('-fmin', default=50, type=int) #50 chosen by me, 71 chosen by default params   
    parser.add_argument('-fmax', default=1100, type=int) #1100 chosen by me, 800 chosen by default params 
    parser.add_argument("-n", "--notes", type=str, default='', help= "Add these notes which will be saved to a config text file that gets saved in your saved directory")
    
    config = parser.parse_args()
    
    if config.use_audio ==True:
        feat_params = feat_params = {"w2w_process":config.w2w_process,
                                "dim_red_method":config.dim_red_method,
                                "fmin":config.fmin,
                                "fmax":config.fmax,
                                'num_harm_feats':config.num_harm_feats,
                                'num_aper_feats':config.num_aper_feats,
                                'use_aper_feats':config.use_aper_feats,
                                'frame_dur_ms':config.frame_dur_ms,
                                'sr':config.sampling_rate,
                                'fft_size':config.fft_size}
        

    # Process arguments
    config.models_dir.mkdir(exist_ok=True)
    config.string_sum = str(config)
    print_args(config, parser)

    # initiate and train model till finish
    if config.use_audio ==True:
        encoder = SingerIdentityEncoder(config, feat_params)
    else:
        encoder = SingerIdentityEncoder(config)
    encoder.train()
    