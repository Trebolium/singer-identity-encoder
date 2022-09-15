import math, random, yaml, os, torch, sys, pickle, pdb
from torch import nn
import numpy as np
from avg_emb_params import *
sys.path.insert(1, SIE_ckpt_path)
sys.path.insert(1, '/homes/bdoc3/my_utils')
from utils import build_SIE_model
from my_os import recursive_file_retrieval
from my_arrays import fix_feat_length
from tqdm import tqdm



def process_all_vocalisations(SIE):

    """
    Does:
        Go through each vocalisation in each voice dir
            Save voice_id
            Divide into chunks
            Save SIE embedding and track_id for each chunk
            Get average of SIE embeddings per singer
    
    Return:
        all_singer_embs: Voice embeddings for each chunk, grouped by each vocalist
        singer_meta_data: List of tuples (voice_id, averaged voice embedding, list of track names used)

    """
    all_singer_embs = []
    singer_meta_data = []
    for label, vd in enumerate(multisong_voice_dirs):
        singer_track_name_dirs = []
        track_pp = len(os.listdir(vd))
        print(f'Voice: {vd}, {label}/{len(multisong_voice_dirs)}')
        avg_pitch = 0
        singer_pitches = []
        singer_embs = []
        vd_tracks = os.listdir(vd)
        random.shuffle(vd_tracks)
        # go through this voices tracks in random order
        for i, fn in enumerate(vd_tracks[:max_tracks_pp]):
            print(f'File: {fn}, {i}/{min(len(os.listdir(vd)), max_tracks_pp)}')
            fp = os.path.join(vd, fn)
            if not fp.endswith('npy'):
                continue
            feats = np.load(fp)
            voiced = feats[:,-1].astype(int) == 0
            singer_pitches.extend(feats[:,-2][voiced])
            # scan through this track in chunks
            for start in range(0, len(feats)-window_timesteps, window_timesteps):
                trimmed_feats, _ = fix_feat_length(feats, window_timesteps, start)
                spectral_feats = trimmed_feats[:,:num_feats_used]
                SIE_input = torch.from_numpy(spectral_feats).to(device).float().unsqueeze(0)
                singer_emb = SIE(SIE_input)
                singer_emb = singer_emb.squeeze(0).cpu().detach().numpy()
                singer_embs.append(singer_emb)
                singer_track_name_dirs.append(fn)
                # print(start, len(singer_embs))
        
        all_singer_embs.append(singer_embs)
        singer_avg = np.mean(np.asarray(singer_embs), axis=0)
        singer_meta_data.append([os.path.basename(vd), singer_avg] + singer_track_name_dirs) 

    return all_singer_embs, singer_meta_data


def threshold_subdir_retrieval():
    # collect all paths from subset that possess above min_tracks_pp variable
    multisong_voice_dirs = []
    subdir = os.path.join(ds_dir_path, subset)
    r, voice_dirs, fps = next(os.walk(subdir))
    for vd in voice_dirs:
        if len(os.listdir(os.path.join(r, vd))) >= min_tracks_pp:
            print(os.path.join(r, vd))
            multisong_voice_dirs.append(os.path.join(r, vd))

    return multisong_voice_dirs


    
def dir_data_handling():
    # creates directory for list
    SIE_name = os.path.basename(SIE_ckpt_path)
    ds_name = os.path.basename(ds_dir_path)
    meta_dir = os.path.join('/homes/bdoc3/my_data/voice_embs_visuals_metadata', SIE_name, ds_name, subset)
    if not os.path.exists(meta_dir):
        print(f'making dirs for: {meta_dir}')
        os.makedirs(meta_dir)

    # calculates num_feats_used to use
    with open(os.path.join(ds_dir_path, 'feat_params.yaml'), 'rb') as handle:
        feat_params = yaml.load(handle, Loader=yaml.FullLoader)
    if use_aper_feats: num_feats_used = feat_params['num_harm_feats'] + feat_params['num_aper_feats']
    else: num_feats_used = feat_params['num_harm_feats']

    return meta_dir, num_feats_used


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device(f'cuda:{which_cuda}' if use_cuda else 'cpu')
    # get dst_dir, num feats used for training, and list of voice dirs
    dst_dir, num_feats_used = dir_data_handling()
    multisong_voice_dirs = threshold_subdir_retrieval()

    # shuffle list
    this_seed = 0
    random.seed(this_seed)
    if max_num_singers == None:
        random.shuffle(multisong_voice_dirs)
    else:
        multisong_voice_dirs = random.sample(multisong_voice_dirs, k=max_num_singers)

    SIE = build_SIE_model(num_feats_used, device) # make sure model layer params in avg_emb_params are correct
    all_singer_embs, singer_meta_data = process_all_vocalisations(SIE)

    print('Saving data to disk...')
    with open(os.path.join(dst_dir, 'voices_metadata.pkl'), 'wb') as handle:
        pickle.dump(singer_meta_data, handle)

    with open(os.path.join(dst_dir, 'voices_chunks_embs.pkl'), 'wb') as handle:
        pickle.dump(all_singer_embs, handle)

