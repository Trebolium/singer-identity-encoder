
SIE_ckpt_path = '/homes/bdoc3/my_data/autovc_models/singer-identity-encoder/bestPerformingSIE_mel80'
# SIE_ckpt_path = '/homes/bdoc3/my_data/autovc_models/singer-identity-encoder/autoVc_pretrainedOnVctk_Mels80'

subset = ''
# ds_dir_path = '/import/c4dm-02/bdoc3/spmel/damp_qianParams'
# ds_dir_path = '/import/c4dm-02/bdoc3/spmel/vctk_qianParams'
# ds_dir_path = '/import/c4dm-02/bdoc3/world_data/damp_80_16ms'
# ds_dir_path = '/homes/bdoc3/my_data/spmel_data/vocadito'
ds_dir_path = '/homes/bdoc3/my_data/spmel_data/vocalset/vocalSet_subset_unnormed'
use_aper_feats = False
which_cuda = 1

window_timesteps = 128 # 384 if using world (hopsize 5ms), 128 if using mel (hopsize 16ms)
model_hidden_size = 768
model_embedding_size = 256
model_num_layers = 3
adam_init = 0.0001

min_tracks_pp = 1
max_tracks_pp = 100
max_embs_pp = 100
max_num_singers = None