from concurrent.futures.process import _get_chunks
import numpy as np
import math, librosa, pdb, sys
from librosa.filters import mel
from scipy.io import wavfile
import soundfile as sf
from sklearn.preprocessing import normalize
import time
import pyworld as pw

from utils import get_world_feats 


from my_audio.mel import audio_to_mel_autovc, db_normalize
from my_audio.world import freq_to_vuv_midi
from my_audio.pitch import midi_as_onehot

"""Minimally altered code from https://github.com/Trebolium/Real-Time-Voice-Cloning/tree/master/encoder/data_objects"""

class Utterance:
    def __init__(self, frames_fpath, wave_fpath, config, feat_params):
        self.frames_fpath = frames_fpath
        self.wave_fpath = wave_fpath
        self.config = config
        self.feat_params = feat_params
        if config.feats_type == 'mel':
            num_total_feats = feat_params['num_harm_feats']
            self.mel_filter = mel(config.sampling_rate, config.fft_size, fmin=config.fmin, fmax=config.fmax, n_mels=num_total_feats).T
            # self.mel_filter = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
            self.min_level = np.exp(-100 / 20 * np.log(10))
            self.hop_size = int((self.config.frame_dur_ms/1000) * self.config.sampling_rate)

    def get_chunk(self, frames, n_frames, start=None):

        if frames.shape[0] > n_frames:
            if start == None:
                start = np.random.randint(0, frames.shape[0] - n_frames)
        else:
#             print(f'frames.shape[0] {frames.shape[0]}, n_frames {n_frames}')
            start = 0
            pad_size = math.ceil((n_frames - frames.shape[0])/2)
            if frames.ndim == 1:
                pad_vec = np.full((pad_size), np.min(frames))
            else:
                pad_vec = np.full((pad_size, frames.shape[1]), np.amin(frames, axis=0))
            frames = np.concatenate((pad_vec, frames, pad_vec))
            
        end = start + n_frames
        # print('start', start)
        return frames[start:end], (start, end)

# get features, either from audio or precomputed npy arrays.
    def get_frames(self, n_frames, start=None):

        # stime = time.time()
        if self.config.use_audio:
            # _, y = wavfile.read(self.frames_fpath) # WAVFILE OUTPUTS PCM VALUES (INTS - CAUSES MEL GEN ISSUES)
            y, _ = sf.read(self.frames_fpath)
            samps_per_frame = (self.feat_params['frame_dur_ms']/1000) * self.feat_params['sr']
            required_size =  int(samps_per_frame * n_frames)
            if y.shape[0] < 1:
                # '+2' at end is for f0_estimation vectors
                frames = np.zeros((n_frames, (self.feat_params['num_harm_feats']+self.feat_params['num_aper_feats']+2)))
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
                                y_chunk, start_end = self.get_chunk(y, required_size, start)
                            if self.config.feats_type == 'mel':
                               
                                # for fair comparison, this block to restrict mel selections same way as world features - maybe try crepe in future
                                f0, t_stamp = pw.dio(y_chunk, self.feat_params['sr'], self.feat_params['fmin'], self.feat_params['fmax'])
                                refined_f0 = pw.stonemask(y_chunk, f0, t_stamp, self.feat_params['sr'])
                                refined_f0 = freq_to_vuv_midi(refined_f0)

                                db_unnormed_melspec = audio_to_mel_autovc(y_chunk, self.config.fft_size, self.hop_size, self.mel_filter)
                                frames = db_normalize(db_unnormed_melspec, self.min_level)
                            elif self.config.feats_type == 'world':
                                frames = get_world_feats(y_chunk.astype('double'), self.feat_params, self.config)
                            
                            looper = False
                        except ValueError as e:
                            print(f'ValueError: {e}. Trying another random chunk from uttr: {self.frames_fpath}')
                            counter +=1
                    else:
                        print(f'Could not find vocal segments. Returning zero\'d array instead')
                        frames = np.zeros((n_frames, (self.feat_params['num_harm_feats']+self.feat_params['num_aper_feats']+2))) # might need to alter if making aper gens conditional of config.use_aper_feats
                        start_end = (0, required_size)
                        looper = False
        else:
            # if certain numpy files are corrupt, we must know about it
            try:
                frames = np.load(self.frames_fpath)
                frames, start_end = self.get_chunk(frames, n_frames)
            except Exception as e:
                print(e)
                pdb.set_trace
        # print('another utterance processed', (time.time() - stime))
        return frames[:n_frames], start_end

    def random_partial(self, n_frames, num_total_feats):
        """
        Crops the frames into a partial utterance of n_frames
        
        :param n_frames: The number of frames of the partial utterance
        :return: the partial utterance frames and a tuple indicating the start and end of the 
        partial utterance in the complete utterance.
        """

        all_feats, start_end = self.get_frames(n_frames)
        final_feats = all_feats[:,:num_total_feats]
        
        if self.config.pitch_condition:
            midi_contour = all_feats[:,-2]
            unvoiced = all_feats[:,-1].astype(int) == 1
            # remove the interpretted values generated because of unvoiced sections
            midi_contour[unvoiced] = 0
            try:
                onehot_midi = midi_as_onehot(midi_contour, self.config.midi_range)
            except Exception as e:
                print(e)
                pdb.set_trace()
                onehot_midi = midi_as_onehot(midi_contour, self.config.midi_range)
            final_feats = np.concatenate((final_feats, onehot_midi), axis=1)

#         pitches = frames[:,-2:]
#         one_hot_pitches = midi_as_onehot(pitches, self.config.vocal_range)

        # frames = (frames - frames.mean()) / frames.std() # normalise from 0-1 across entire numpy
        # frames = (frames - frames.mean(axis=0)) / frames.std(axis=0) # normalise from 0-1 across features
        return final_feats, start_end

    def specific_partial(self, n_frames, num_total_feats, start):
        """
        Crops the frames into a partial utterance of n_frames
        
        :param n_frames: The number of frames of the partial utterance
        :return: the partial utterance frames and a tuple indicating the start and end of the 
        partial utterance in the complete utterance.
        """
        # pdb.set_trace()

        frames, start_end = self.get_frames(n_frames, start)
        frames = frames[:,:num_total_feats]

        # frames = (frames - frames.mean()) / frames.std() # normalise from 0-1 across entire numpy
        # frames = (frames - frames.mean(axis=0)) / frames.std(axis=0) # normalise from 0-1 across features
        # pdb.set_trace()   
        return frames, start_end 