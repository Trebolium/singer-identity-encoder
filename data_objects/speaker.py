"""

MOdule Description: This module defines the Speaker class, which represents a single speaker
             and contains a set of utterances from which features can be found and extracted.

Usage:
    from speaker import Speaker

    # Create a Speaker object
    speaker = Speaker(root, config, feat_params, norm_stats, augment_chain)

    # Load utterances for the speaker
    speaker.load_utterances()

    # Sample a batch of random partial utterances from the speaker
    partial_utterances = speaker.random_partial(count, n_frames, num_total_feats)

Classes:
    - Speaker: Represents a single speaker and contains a set of utterances.

Dependencies:
    - data_objects.random_cycler.RandomCycler
    - data_objects.utterance.Utterance
    - pathlib.Path
    - pdb

"""

from data_objects.random_cycler import RandomCycler
from data_objects.utterance import Utterance
from pathlib import Path
import pdb


class Speaker:
    def __init__(self, root: Path, config, feat_params, norm_stats, augment_chain):
        """
        Initialize a Speaker object.

        Args:
            root (Path): The root path of the speaker's data directory.
            config: The configuration object.
            feat_params: The feature parameters.
            norm_stats: The normalization statistics.
            augment_chain: The augmentation chain.

        Returns:
            None
        """
        self.augment_chain = augment_chain
        self.norm_stats = norm_stats
        self.root = root
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None
        self.config = config
        self.feat_params = feat_params

    def load_utterances(self):
        """
        Load utterances for the speaker from the disk.

        Returns:
            None
        """
        spkr_uttrs_list = [f.name for f in self.root.glob("*")]
        self.utterances = [
            Utterance(
                self.root.joinpath(f),
                f,
                self.config,
                self.feat_params,
                self.norm_stats,
                self.augment_chain,
            )
            for f in spkr_uttrs_list
        ]
        try:
            self.utterance_cycler = RandomCycler(self.utterances)
        except Exception as e:
            print(e)
            pdb.set_trace()

    def random_partial(self, count, n_frames, num_total_feats):
        """
        Sample a batch of random partial utterances from the speaker.

        Args:
            count (int): The number of partial utterances to sample from the set of utterances
                         from that speaker.
            n_frames (int): The number of frames in the partial utterance.
            num_total_feats: The total number of features.

        Returns:
            list: A list of tuples (utterance, frames, range) where utterance is an Utterance object,
                  frames are the frames of the partial utterances, and range is the range of the
                  partial utterance with regard to the complete utterance.
        """
        if self.utterances is None:
            self.load_utterances()

        utterances = self.utterance_cycler.sample(count)
        a = [(u,) + u.random_partial(n_frames, num_total_feats) for u in utterances]

        return a
