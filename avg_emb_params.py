
SIE_ckpt_path = '/homes/bdoc3/my_data/autovc_models/singer_identity_encoder/bestPerformingSIE_mel80'
subset = 'test'
# parent_dir = '/import/c4dm-02/bdoc3/spmel/damp_qianParams'
parent_dir = '/import/c4dm-02/bdoc3/world_data/damp_80_16ms'
# parent_dir = '/homes/bdoc3/my_data/world_vocoder_data/damp_new'

use_aper_feats = False
which_cuda = 0

window_timesteps = 128 # 384 if using world (hopsize 5ms), 128 if using mel (hopsize 16ms)
model_hidden_size = 768
model_embedding_size = 256
model_num_layers = 3
adam_init = 0.0001

min_tracks_pp = 1
max_tracks_pp = 30

max_num_singers = None