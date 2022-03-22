from concurrent.futures.process import _get_chunks
import numpy as np
import math, librosa, pdb
import soundfile as sf
from sklearn.preprocessing import normalize

from utils import process_data

"""Minimally altered code from https://github.com/Trebolium/Real-Time-Voice-Cloning/tree/master/encoder/data_objects"""

class Utterance:
    def __init__(self, frames_fpath, wave_fpath, config, feat_params):
        self.frames_fpath = frames_fpath
        self.wave_fpath = wave_fpath
        self.config = config
        self.feat_params = feat_params

    def get_chunk(self, frames, n_frames, start=None):

        if frames.shape[0] > n_frames:
            if start == None:
                start = np.random.randint(0, frames.shape[0] - n_frames)
        else:
#             print(f'frames.shape[0] {frames.shape[0]}, n_frames {n_frames}')
            start = 0
            pad_size = math.ceil(n_frames - frames.shape[0]/2)
            if frames.ndim == 1:
                pad_vec = np.full((pad_size), np.min(frames))
            else:
                pad_vec = np.full((pad_size, frames.shape[1]), np.min(frames))
            frames = np.concatenate((pad_vec, frames, pad_vec))
            
        end = start + n_frames
#         print('start', start)
        return frames[start:end], (start, end)

# get features, either from audio or precomputed npy arrays.
    def get_frames(self, n_frames, start=None):

        if self.config.use_audio:
            y, _ = sf.read(self.frames_fpath)
            samps_per_frame = (self.feat_params['frame_dur_ms']/1000) * self.feat_params['sr']
            required_size =  int(samps_per_frame * n_frames)
            if y.shape[0] < 1:
                # '+2' at end is for f0_estimation vector
                frames = np.zeros((n_frames, (self.feat_params['num_feats']+self.feat_params['num_aper_feats']+2)))
                start_end = (0, required_size)
            else:
                counter = 0
                looper = True
                while looper:
                    if counter > 10:
                        raise Exception(f'Could not find vocal segments after randomly selecting 10 segments of length {n_frames}.')
                    try:
                        if start == None:
                            y_chunk, start_end = self.get_chunk(y, required_size)
                        else:
                            y_chunk, start_end = self.get_chunk(y, required_size, start)
                        frames = process_data(y_chunk.astype('double'), self.feat_params, self.config)
                        looper = False
                    except ValueError as e:
                        print(e, 'Trying another random chunk')
                        counter +=1

        else:
            frames = np.load(self.frames_fpath)
            frames, start_end = self.get_chunk(frames, n_frames)
        # print('another utterance processed')
        return frames[:n_frames], start_end

    def random_partial(self, n_frames, num_feats):
        """
        Crops the frames into a partial utterance of n_frames
        
        :param n_frames: The number of frames of the partial utterance
        :return: the partial utterance frames and a tuple indicating the start and end of the 
        partial utterance in the complete utterance.
        """
        # pdb.set_trace()

        frames, start_end = self.get_frames(n_frames)
        frames = frames[:,:num_feats]

        # frames = (frames - frames.mean()) / frames.std() # normalise from 0-1 across entire numpy
        # frames = (frames - frames.mean(axis=0)) / frames.std(axis=0) # normalise from 0-1 across features
        # pdb.set_trace()   
        return frames, start_end

    def specific_partial(self, n_frames, num_feats, start):
        """
        Crops the frames into a partial utterance of n_frames
        
        :param n_frames: The number of frames of the partial utterance
        :return: the partial utterance frames and a tuple indicating the start and end of the 
        partial utterance in the complete utterance.
        """
        # pdb.set_trace()

        frames, start_end = self.get_frames(n_frames, start)
        frames = frames[:,:num_feats]

        # frames = (frames - frames.mean()) / frames.std() # normalise from 0-1 across entire numpy
        # frames = (frames - frames.mean(axis=0)) / frames.std(axis=0) # normalise from 0-1 across features
        # pdb.set_trace()   
        return frames, start_end 
