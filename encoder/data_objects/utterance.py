import numpy as np
import math, pdb
from sklearn.preprocessing import normalize


class Utterance:
    def __init__(self, frames_fpath, wave_fpath):
        self.frames_fpath = frames_fpath
        self.wave_fpath = wave_fpath
        
    def get_frames(self):
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames):
        """
        Crops the frames into a partial utterance of n_frames
        
        :param n_frames: The number of frames of the partial utterance
        :return: the partial utterance frames and a tuple indicating the start and end of the 
        partial utterance in the complete utterance.
        """
        frames = self.get_frames()
        
        # frames = (frames - frames.mean()) / frames.std() # normalise from 0-1
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