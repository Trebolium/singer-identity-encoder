from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.utterance import Utterance
import pdb

def collater(dataset):
    # warning: If using RandomCycler, dataset will be infinite
    for i, speaker in enumerate(dataset):
        spkr_uttrs_list = [speaker.name[:-4] for f in speaker.root.glob('*.npy')]
        utterances = [Utterance(speaker.root.joinpath(f+'.npy'), f+'.wav') for f in spkr_uttrs_list]
        print(f"utterance {i} length is: {len(utterances)}")
        try:
            utterance_cycler = RandomCycler(utterances)
        except:
            pdb.set_trace()