from data_objects.random_cycler import RandomCycler
from data_objects.utterance import Utterance
from pathlib import Path
import pdb

"""Minimally altered code from https://github.com/Trebolium/Real-Time-Voice-Cloning/tree/master/encoder/data_objects"""


# Contains the set of utterances of a single speaker
class Speaker:
    def __init__(self, root: Path, config, feat_params, norm_stats, augment_chain):
        self.augment_chain = augment_chain
        self.norm_stats = norm_stats
        self.root = root
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None
        self.config = config
        self.feat_params = feat_params
        
    def _load_utterances(self):
        if self.config.use_audio == True:
            ext = '.m4a'
        else:
            ext = '.npy'
        """ Utterance stores mel and wav paths, but does not make them npy objects until random_partial() is called"""
        spkr_uttrs_list = [f.name[:-4] for f in self.root.glob('*' +ext)]
        self.utterances = [Utterance(self.root.joinpath(f+ext), f+ext, self.config, self.feat_params, self.norm_stats, self.augment_chain) for f in spkr_uttrs_list]
        try:
            self.utterance_cycler = RandomCycler(self.utterances)
        except Exception as e:
            print(e)
            pdb.set_trace()

    def random_partial(self, count, n_frames, num_total_feats):
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
            # print(f'loading utterances for speak {self.name}')
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)
        """Utterance random__partial returns: (splice of numpy, (start/end values))"""
        a = [(u,) + u.random_partial(n_frames, num_total_feats) for u in utterances]

        # print(f'{len(a)} utterances for speaker {self.name} collected')
        """ a is a list of tuples (uttr objects, spliced uttr_features, start/end coords), size of count"""
        return a
