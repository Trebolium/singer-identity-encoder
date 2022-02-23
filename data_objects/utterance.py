import numpy as np
import math, pdb
from sklearn.preprocessing import normalize

"""Minimally altered code from https://github.com/Trebolium/Real-Time-Voice-Cloning/tree/master/encoder/data_objects"""

class Utterance:
    def __init__(self, frames_fpath, wave_fpath):
        self.frames_fpath = frames_fpath
        self.wave_fpath = wave_fpath
        
    def get_frames(self):
        try:
            frames = np.load(self.frames_fpath)
        except ValueError as e:
            frames = np.load(self.frames_fpath, allow_pickle=True)
        return frames

    def random_partial(self, n_frames, num_feats):
        """
        Crops the frames into a partial utterance of n_frames
        
        :param n_frames: The number of frames of the partial utterance
        :return: the partial utterance frames and a tuple indicating the start and end of the 
        partial utterance in the complete utterance.
        """
        frames = self.get_frames()[:,:num_feats]

        # frames = (frames - frames.mean()) / frames.std() # normalise from 0-1 across entire numpy
        # frames = (frames - frames.mean(axis=0)) / frames.std(axis=0) # normalise from 0-1 across features
        # pdb.set_trace()
        
        if frames.shape[0] > n_frames:
            start = np.random.randint(0, frames.shape[0] - n_frames)
        else:
            # new section - pad the sides to make up for chunks thata are too small
            start = 0
            pad_size = math.ceil(n_frames - frames.shape[0]/2)
            pad_vec = np.full((pad_size, frames.shape[1]), np.min(frames))
            frames = np.concatenate((pad_vec, frames, pad_vec))
        end = start + n_frames
        return frames[start:end], (start, end)