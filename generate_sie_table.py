
"""
This module processes vocalizations and extracts voice embeddings using the Speaker Identity Embedding (SIE) model.

It provides functions for processing vocalizations, dividing them into chunks, extracting SIE embeddings, and calculating average embeddings per singer.

Module Functions:
- process_all_vocalisations(SIE): Processes all vocalizations and extracts voice embeddings for each vocalist.
- threshold_subdir_retrieval(subset): Retrieves voice directories that meet the minimum number of tracks threshold.
- dir_data_handling(): Handles directory data and calculates the number of feature used for training.
- main(): Main entry point of the module, performs the processing and saves the extracted embeddings and metadata to disk.
"""

import math
import random
import yaml
import os
import torch
import sys
import pickle
import numpy as np
from avg_emb_params import *

sys.path.insert(1, SIE_ckpt_path)
if os.path.abspath("../my_utils") not in sys.path:
    sys.path.insert(1, os.path.abspath("../my_utils"))
from utils import build_SIE_model



def process_all_vocalisations(SIE):
    """
    Process all vocalizations for each voice directory.
    
    Args:
        SIE: The SIE model used for embedding extraction.
        
    Returns:
        all_singer_embs: Voice embeddings for each chunk, grouped by each vocalist.
        singer_meta_data: List of tuples (voice_id, averaged voice embedding, list of track names used).
    """
    all_singer_embs = []
    singer_meta_data = []

    for label, vd in enumerate(multisong_voice_dirs):
        singer_track_name_dirs = []
        track_pp = len(os.listdir(vd))
        print(f"Voice: {vd}, {label}/{len(multisong_voice_dirs)}")
        avg_pitch = 0
        singer_pitches = []
        singer_embs = []
        vd_tracks = os.listdir(vd)
        vd_tracks = [
            i for i in vd_tracks if not i.startswith(".") or not i.endswith(".npy")
        ]
        random.shuffle(vd_tracks)
        count = 0
        emb_per_track = math.ceil(max_embs_pp / len(vd_tracks))
        for i, fn in enumerate(vd_tracks):
            print(f"File: {fn}, {i}/{min(len(os.listdir(vd)), max_tracks_pp)}")
            fp = os.path.join(vd, fn)
            if not fp.endswith("npy") or fp.startswith("."):
                continue
            feats = np.load(fp)
            if feats.shape[0] <= window_timesteps:
                continue

            voiced = feats[:, -1].astype(int) == 0
            singer_pitches.extend(feats[:, -2][voiced])

            for j in range(emb_per_track):
                left = np.random.randint(0, feats.shape[0] - window_timesteps)
                cropped_feats = (
                    torch.from_numpy(
                        feats[np.newaxis, left : left + window_timesteps, :]
                    )
                    .to(device)
                    .float()
                )
                singer_emb = SIE(cropped_feats)
                singer_emb = singer_emb.squeeze(0).cpu().detach().numpy()
                singer_embs.append(singer_emb)
                singer_track_name_dirs.append(fn)
                count += 1
            if count >= max_tracks_pp:
                break

        all_singer_embs.append(singer_embs)
        singer_avg = np.mean(np.asarray(singer_embs), axis=0)
        singer_meta_data.append(
            [os.path.basename(vd), singer_avg] + singer_track_name_dirs
        )

    return all_singer_embs, singer_meta_data


def threshold_subdir_retrieval(subset):
    """
    Retrieve voice directories that have more than min_tracks_pp tracks in the given subset.
    
    Args:
        subset: The subset of the dataset.
        
    Returns:
        multisong_voice_dirs: List of voice directories that meet the track count threshold.
    """
    multisong_voice_dirs = []
    subdir = os.path.join(ds_dir_path, subset)
    r, voice_dirs, fps = next(os.walk(subdir))
    for vd in voice_dirs:
        if vd.startswith("."):
            continue
        if len(os.listdir(os.path.join(r, vd))) >= min_tracks_pp:
            print(os.path.join(r, vd))
            multisong_voice_dirs.append(os.path.join(r, vd))

    return multisong_voice_dirs


def dir_data_handling():
    """
    Handle directory data and return necessary information.
    
    Returns:
        meta_dir: Directory path for metadata.
        num_feats_used: Number of features used for training.
    """
    SIE_name = os.path.basename(SIE_ckpt_path)
    ds_name = os.path.basename(ds_dir_path)
    meta_dir = os.path.join("../voice_embs_visuals_metadata", SIE_name, ds_name, subset)
    if not os.path.exists(meta_dir):
        print(f"making dirs for: {meta_dir}")
        os.makedirs(meta_dir)

    with open(os.path.join(ds_dir_path, "feat_params.yaml"), "rb") as handle:
        feat_params = yaml.load(handle, Loader=yaml.FullLoader)
    if use_aper_feats:
        num_feats_used = feat_params["num_harm_feats"] + feat_params["num_aper_feats"]
    else:
        num_feats_used = feat_params["num_harm_feats"]

    return meta_dir, num_feats_used


if __name__ == "__main__":
    subsets = ["train", "val"]
    for subset in subsets:
        use_cuda = torch.cuda.is_available()
        device = torch.device(f"cuda:{which_cuda}" if use_cuda else "cpu")
        dst_dir, num_feats_used = dir_data_handling()
        multisong_voice_dirs = threshold_subdir_retrieval(subset)

        this_seed = 0
        random.seed(this_seed)
        if max_num_singers == None:
            random.shuffle(multisong_voice_dirs)
        else:
            multisong_voice_dirs = random.sample(
                multisong_voice_dirs, k=max_num_singers
            )
        SIE = build_SIE_model(
            num_feats_used, device
        )
        all_singer_embs, singer_meta_data = process_all_vocalisations(SIE)

        print("Saving data to disk...")
        with open(
            os.path.join(dst_dir, f"voices_metadata_{max_tracks_pp}avg.pkl"), "wb"
        ) as handle:
            pickle.dump(singer_meta_data, handle)

        with open(
            os.path.join(dst_dir, f"voices_chunks_embs_{max_tracks_pp}avg.pkl"), "wb"
        ) as handle:
            pickle.dump(all_singer_embs, handle)
