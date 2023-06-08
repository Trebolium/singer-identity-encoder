import numpy as np
from typing import List

"""Code from https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/master/encoder/data_objects"""


class SpeakerBatch:
    """
    Represents a batch of speakers' data for training.

    Args:
        speakers_data (List): List of tuples containing Speaker objects and their indices.
        utterances_per_speaker (int): Number of utterances per speaker.
        n_frames (int): Number of frames.
        num_total_feats (int): Total number of features.

    Attributes:
        partials (dict): Dictionary of speaker lists (utterance objects, evenly spliced utterance features).
        data (tuple): Tuple containing input data array and target data array.

    """

    def __init__(
        self,
        speakers_data: List,
        utterances_per_speaker: int,
        n_frames: int,
        num_total_feats: int,
    ):
        """
        Initialize SpeakerBatch with the given parameters.
        """
        # print("Speaker Batch initiated")
        self.partials = {
            s.name: s.random_partial(utterances_per_speaker, n_frames, num_total_feats)
            for s, _ in speakers_data
        }
        # print("utterances per speaker generated")
        x_data = np.array(
            [
                uttr_data[1]
                for s, _ in speakers_data
                for uttr_data in self.partials[s.name]
            ]
        )
        y_data = np.array(
            [speakers_data[i // utterances_per_speaker][1] for i in range(len(x_data))]
        )
        self.data = x_data, y_data
