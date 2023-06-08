
"""
Module: utterance.py

This module defines an Utterance to represent individual utterances for speakers.

Minimally altered code from https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/master/encoder/data_objects
"""

import math
from pathlib import Path
import pdb

from librosa.filters import mel
import numpy as np

from my_normalise import norm_feat_arr
from my_audio.mel import audio_to_mel_autovc, db_normalize, audio_to_mel_librosa
from my_audio.world import freq_to_vuv_midi, onehotmidi_from_world_fp, get_world_feats, onehotmidi_from_world_pitch_feats
from my_audio.utils import audio_io
import pyworld as pw


class Utterance:
    """Generate a set of features for a vocalists recording.

    This class finds the relevant features using its assigned path.
    It randomly chooses a slice from the feature set,
    making necessary adjustments.

    """

    def __init__(self, featpath, audiopath, config, feat_params, normstats, augment_chain):
        """Initialize the Utterance object.

        Args:
            featpath (str): Path to the relevant feature file.
            audiopath (str): Path to the relevant audio file.
            config (argparaser): Argparser object with configurations.
            feat_params (dict): Dictionary of configs for audio transformation.
            normstats (dict): Normalization statistics for feature normalization.
            augment_chain: Augmentation chain object for data augmentation. 

        """
        self.normstats = normstats
        self.featpath = featpath
        self.audiopath = audiopath
        self.config = config
        self.feat_params = feat_params
        if config.feats_type == "mel":
            num_total_feats = feat_params["num_harm_feats"]
            self.mel_filter = mel(
                sr=config.sampling_rate,
                n_fft=config.fft_size,
                fmin=config.fmin,
                fmax=config.fmax,
                n_mels=num_total_feats,
            ).T
            self.min_level = np.exp(-100 / 20 * np.log(10))
            self.hop_size = int(
                (self.config.frame_dur_ms / 1000) * self.config.sampling_rate
            )

    def get_chunk(self, frames, n_frames, start=None):
        """Select a temporal slice from the input feature array.
    
        Args:
            frames (array): An array of features to be sliced.
            n_frames (int): Temporal length of the desired slice.
            start (int): Starting index of slicing (optional).

        Returns:
            array: Sliced version of the input feature array.
            tuple: Integers representing the start and end indices for slicing.

        """
        if frames.shape[0] > n_frames:
            if start == None:
                start = np.random.randint(0, frames.shape[0] - n_frames)
        else:
            start = 0
            pad_size = math.ceil((n_frames - frames.shape[0]) / 2)
            if frames.ndim == 1:
                pad_vec = np.full((pad_size), np.min(frames))
            else:
                pad_vec = np.full((pad_size,
                                   frames.shape[1]),
                                   np.amin(frames, axis=0))
            frames = np.concatenate((pad_vec, frames, pad_vec))

        end = start + n_frames
        # print('start', start)
        return frames[start:end], (start, end)

    # get features, either from audio or precomputed npy arrays.
    def get_frames(self, n_frames, start=None):
        """Get feature array from given pathway.

        Args:
            n_frames (int): Temporal length of the desired slice
            start (int): if not None, determines starting index of slicing

        Returns:
            array: slice of feature array
            tuple: Ints representing the start & end indices for slicing

        """
        if self.config.use_audio:
            y = audio_io(self.featpath, self.feat_params["sr"])
            samps_per_frame = (
                self.feat_params["frame_dur_ms"] / 1000
            ) * self.feat_params["sr"]
            required_size = int(samps_per_frame * n_frames) - 1
            if y.shape[0] < 1:
                # '+2' at end is for f0_estimation vectors
                frames = np.zeros(
                    (
                        n_frames,
                        (
                            self.feat_params["num_harm_feats"]
                            + self.feat_params["num_aper_feats"]
                            + 2
                        ),
                    )
                )
                start_end = (0, required_size)
            else:
                counter = 0
                looper = True
                while looper:
                    if counter < 10:
                        try:
                            if start == None:
                                y_chunk, start_end = self.get_chunk(y, required_size)
                            else:
                                y_chunk, start_end = self.get_chunk(
                                    y, required_size, start
                                )
                            if self.config.feats_type == "mel":
                                # for fair comparison, this block to restrict mel selections same way as world features - maybe try crepe in future
                                f0, t_stamp = pw.dio(
                                    y_chunk.astype(np.double),
                                    self.feat_params["sr"],
                                    self.feat_params["fmin"],
                                    self.feat_params["fmax"],
                                    frame_period = self.feat_params['frame_dur_ms']
                                )
                                refined_f0 = pw.stonemask(
                                    y_chunk.astype(np.double),
                                    f0,
                                    t_stamp,
                                    self.feat_params["sr"],
                                )
                                refined_f0 = freq_to_vuv_midi(refined_f0)

                                db_unnormed_melspec = audio_to_mel_autovc(
                                    y_chunk,
                                    self.config.fft_size,
                                    self.hop_size,
                                    self.mel_filter,
                                )
                                frames = db_normalize(
                                    db_unnormed_melspec, self.min_level
                                )

                                onehot_midi = onehotmidi_from_world_pitch_feats(refined_f0, 0, n_frames, self.config.midi_range)
                                frames = np.concatenate((frames, onehot_midi), axis=1)

                            elif self.config.feats_type == "world":
                                frames = get_world_feats(
                                    y_chunk.astype("double"),
                                    self.feat_params,
                                    self.config,
                                )

                            looper = False
                        except ValueError as e:
                            print(
                                f"ValueError: {e}. Trying another random chunk from uttr: {self.featpath}"
                            )
                            counter += 1
                    else:
                        print(
                            f"Could not find vocal segments. Returning zero'd array instead"
                        )
                        frames = np.zeros(
                            (
                                n_frames,
                                (
                                    self.feat_params["num_harm_feats"]
                                    + self.feat_params["num_aper_feats"]
                                    + 2
                                ),
                            )
                        )  # might need to alter if making aper gens conditional of config.use_aper_feats
                        start_end = (0, required_size)
                        looper = False
        else:
            # if certain numpy files are corrupt, we must know about it
            try:
                frames = np.load(self.featpath)
                frames, start_end = self.get_chunk(frames, n_frames)
            except Exception as e:
                print(e)
                pdb.set_trace

        # print('another utterance processed', (time.time() - stime))
        return frames[:n_frames], start_end

    def random_partial(self, n_frames, num_total_feats):
        """Crop the frames into a partial utterance of n_frames.

        Args:
            n_frames (int): The number of frames of the partial utterance

        Returns:
            the partial utterance frames and a tuple indicating the start and end of the
            partial utterance in the complete utterance.

        """
        all_feats, start_end = self.get_frames(n_frames)
        spec_feats = all_feats[:, :num_total_feats]

        if (
            self.config.norm_method == "schluter"
            or self.config.norm_method == "global_unit_var"
        ):
            spec_feats = norm_feat_arr(
                spec_feats, self.config.norm_method, self.normstats
            )
        else:
            spec_feats = norm_feat_arr(spec_feats, self.config.norm_method)

        if self.config.pitch_condition:
            subset_vocalist_performance_path = Path(self.featpath).relative_to(
                self.config.feature_dir
            )
            pitch_feat_path = self.config.pitch_dir / subset_vocalist_performance_path
            onehot_midi = onehotmidi_from_world_fp(
                pitch_feat_path, start_end[0], n_frames, self.config.midi_range
            )
            final_feats = np.concatenate((spec_feats, onehot_midi), axis=1)
        else:
            final_feats = spec_feats

        return final_feats, start_end

    def specific_partial(self, n_frames, num_total_feats, start):
        """Crop the frames into a partial utterance of n_frames.

        Args:
            n_frames (int): The number of frames of the partial utterance

        Returns:
            the partial utterance frames and a tuple indicating the start and end of the
            partial utterance in the complete utterance.

        """

        frames, start_end = self.get_frames(n_frames, start)
        frames = frames[:, :num_total_feats]

        return frames, start_end
