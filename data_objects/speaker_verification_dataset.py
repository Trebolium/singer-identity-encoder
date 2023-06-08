import os
import pdb

import audiomentations
from torch.utils.data import Dataset, DataLoader

from data_objects.random_cycler import RandomCycler
from data_objects.speaker_batch import SpeakerBatch
from data_objects.speaker import Speaker
from my_normalise import get_norm_stats

"""
Altered code from https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/master/encoder/data_objects
"""

def get_augs():
    """
    Get the audio augmentations chain.

    Returns:
        audiomentations.Compose: The audio augmentations chain.
    """

    augment_chain = audiomentations.Compose(
        [
            audiomentations.AddGaussianSNR(min_snr_in_db=20, max_snr_in_db=50, p=1.0),
            audiomentations.SevenBandParametricEQ(p=1.0),
        ]
    )

    return augment_chain


class SpeakerVerificationDataset(Dataset):
    """
    Dataset for speaker verification.

    Args:
        datasets_root (str): Path to the root directory of preprocessed speaker directories.
        config: Configuration object.
        feat_params: Feature parameters.
        num_total_feats (int): Total number of features.
        norm_stats: Normalization statistics.
    """

    def __init__(self, datasets_root, config, feat_params, num_total_feats, norm_stats=None):
        if config.use_audio:
            augment_chain = get_augs()
        else:
            augment_chain = None

        self.root = datasets_root
        speaker_dirs = [
            f
            for f in self.root.glob("*")
            if f.is_dir()
            and not str(f).startswith(".")
            and not os.path.basename(f).startswith(".")
        ]
        if len(speaker_dirs) == 0:
            raise Exception(
                "No speakers found. Make sure you are pointing to the directory "
                "containing all preprocessed speaker directories."
            )

        if norm_stats is None:
            if config.norm_method == "schluter" or config.norm_method == "global_unit_var":
                norm_stats = get_norm_stats(datasets_root, num_total_feats)
        self.norm_stats = norm_stats
        self.speakers = [
            (Speaker(speaker_dir, config, feat_params, norm_stats, augment_chain), i)
            for i, speaker_dir in enumerate(speaker_dirs)
        ]
        self.num_speakers = len(self.speakers)
        self.speaker_cycler = RandomCycler(self.speakers)

    def get_stats(self):
        """
        Get the normalization statistics.

        Returns:
            object: The normalization statistics.
        """
        return self.norm_stats

    def num_voices(self):
        """
        Get the number of speakers.

        Returns:
            int: The number of speakers.
        """
        return self.num_speakers

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return int(1e10)

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: A tuple containing the Speaker object and its index.
        """
        return next(self.speaker_cycler)

    def get_logs(self):
        """
        Get the logs from the dataset.

        Returns:
            str: The logs from the dataset.
        """
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string


class SpeakerVerificationDataLoader(DataLoader):
    """
    DataLoader for speaker verification.

    Args:
        dataset: SpeakerVerificationDataset object.
        speakers_per_batch (int): Number of speakers per batch.
        utterances_per_speaker (int): Number of utterances per speaker.
        partials_n_frames: Partial frames.
        num_total_feats (int): Total number of features.
        sampler: Sampler object.
        batch_sampler: BatchSampler object.
        num_workers (int): Number of workers.
        pin_memory (bool): Whether to pin memory.
        timeout (float): Timeout value.
        worker_init_fn: Worker initialization function.
    """

    def __init__(
        self,
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        partials_n_frames,
        num_total_feats,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        timeout=0,
        worker_init_fn=None,
    ):
        self.partials_n_frames = partials_n_frames
        self.utterances_per_speaker = utterances_per_speaker
        self.num_total_feats = num_total_feats

        super().__init__(
            dataset=dataset,
            batch_size=speakers_per_batch,
            shuffle=False,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=self.collate,
            pin_memory=pin_memory,
            drop_last=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )

    def collate(self, speaker_data):
        """
        Collate function for the DataLoader.

        Args:
            speaker_data (list): List of Speaker objects.

        Returns:
            SpeakerBatch: The collated SpeakerBatch object.
        """
        return SpeakerBatch(
            speaker_data,
            self.utterances_per_speaker,
            self.partials_n_frames,
            self.num_total_feats,
        )
