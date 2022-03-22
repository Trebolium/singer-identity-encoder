from data_objects.random_cycler import RandomCycler
from data_objects.utterance import Utterance
import numpy as np
import pdb

def collater(dataset, utterances_per_speaker, n_frames, num_feats, config, feat_params):
    # warning: If using RandomCycler, dataset will be infinite
    for i, (speaker, idx) in enumerate(dataset):
        try:
            # load_partial from Speaker
            if self.config.use_audio == True:
                ext = '.wav'
            else:
                ext = '.npy'
            spkr_uttrs_list = [f.name[:-4] for f in self.root.glob('*' +ext)]
            utterances = [Utterance(self.root.joinpath(f+ext), f+ext, self.config, self.feat_params) for f in spkr_uttrs_list]
            print(f"Number of utterances for speaker {speaker.name} is: {len(utterances)}")
            # load_partial utterances
            utterance_cycler = RandomCycler(utterances)
            utterances = utterance_cycler.sample(config.utterances_per_speaker)
            a = [(u,) + u.random_partial(n_frames, num_feats) for u in utterances]
        except:
            pdb.set_trace()
    speakers_data = [dataset[2], dataset[4], dataset[7]]
    partials = {s.name: s.random_partial(utterances_per_speaker, n_frames) for s,_ in speakers_data}
    x_data = np.array([uttr_data[1] for s,_ in speakers_data for uttr_data in partials[s.name]])
    y_data = np.array([speakers_data[i//utterances_per_speaker][1] for i in range(len(x_data))])
    