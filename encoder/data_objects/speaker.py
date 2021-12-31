from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.utterance import Utterance
from pathlib import Path
import pdb
import numpy as np

# Contains the set of utterances of a single speaker
class Speaker:
    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None
        
    def _load_utterances(self):
        """ Utterance stores mel and wav paths, but does not make them npy objects until random_partial() is called"""
        spkr_uttrs_list = [f.name[:-4] for f in self.root.glob('*.npy')]
        self.utterances = [Utterance(self.root.joinpath(f+'.npy'), f+'.wav') for f in spkr_uttrs_list]

        try:
            self.utterance_cycler = RandomCycler(self.utterances)
        except:
            pdb.set_trace()       
    def random_partial(self, count, n_frames):
        """
        Samples a batch of <count> unique partial utterances from the disk in a way that all 
        utterances come up at least once every two cycles and in a random order every time.
        
        :param count: The number of partial utterances to sample from the set of utterances from 
        that speaker. Utterances are guaranteed not to be repeated if <count> is not larger than 
        the number of utterances available.
        :param n_frames: The number of frames in the partial utterance.
        :return: A list of tuples (utterance, frames, range) where utterance is an Utterance, 
        frames are the frames of the partial utterances and range is the range of the partial 
        utterance with regard to the complete utterance.
        """
        if self.utterances is None:
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)
        """Utterance random_partial returns: (splice of numpy, (start/end values))"""
        a = [(u,) + u.random_partial(n_frames) for u in utterances]

        """ a is a list of tuples (uttr objects, spliced uttr_features, start/end coords), size of count"""
        return a
