import numpy as np
from typing import List

from numpy.core.records import array
#from encoder.data_objects.speaker import Speaker
import pdb

# gets random 160-frame slice 

class SpeakerBatch:
    def __init__(self, speakers_data: List, utterances_per_speaker: int, n_frames: int):
        # self.speakers = speakers_data[0]
        
        """ dict of speaker lists (uttr objects, evenly spliced uttr_features)"""
        self.partials = {s.name: s.random_partial(utterances_per_speaker, n_frames) for s,_ in speakers_data}
        
        """ Array of shape (n_speakers * n_utterances, n_frames, mel_n), e.g. for 3 speakers with
        4 utterances each of 160 frames of 40 mel coefficients: (12, 160, 40)"""
        x_data = np.array([uttr_data[1] for s,_ in speakers_data for uttr_data in self.partials[s.name]])
        y_data = np.array([speakers_data[i//utterances_per_speaker][1] for i in range(len(x_data))])
        self.data = x_data, y_data