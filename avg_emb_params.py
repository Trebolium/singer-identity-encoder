import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SIE_ckpt_path = '/homes/bdoc3/my_data/autovc_models/singer-identity-encoder/qianPretrainedSie_LibriVox1_Mels80' #"./sie_models/default_model" # bestPerformingSIE_mel80 # qianPretrainedSie_LibriVox1_Mels80

if os.path.basename(SIE_ckpt_path) == 'qianPretrainedSie_LibriVox1_Mels80' or os.path.basename(SIE_ckpt_path) == 'qianPretrainedSie_LibriVox1_Mels80':
    qians_pretrained_model = True
else:
    qians_pretrained_model = False

ds_dir_path = "/homes/bdoc3/my_data/spmel_data/damp_sized_libri" #"example_feats"
ds_dir_path = "/import/c4dm-02/bdoc3/spmel/damp_qianParams" #"example_feats"
subsets = ["test"]
use_aper_feats = False
which_cuda = 0

window_timesteps = (
    128  # 384 if using world (hopsize 5ms), 128 if using mel (hopsize 16ms)
)
model_hidden_size = 768
model_embedding_size = 256
model_num_layers = 3
adam_init = 0.0001

min_tracks_pp = 1
max_tracks_pp = 100
max_embs_pp = 50
max_num_singers = 320
