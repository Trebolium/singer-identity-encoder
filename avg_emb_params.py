import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

SIE_ckpt_path = './default_trained'

# subset = 'train'
ds_dir_path = 'example_feats'
use_aper_feats = False
which_cuda = 0

window_timesteps = 128 # 384 if using world (hopsize 5ms), 128 if using mel (hopsize 16ms)
model_hidden_size = 768
model_embedding_size = 256
model_num_layers = 3
adam_init = 0.0001

min_tracks_pp = 1
max_tracks_pp = 100
max_embs_pp = 100
max_num_singers = None