import math, random, yaml, os, sys, torch, pickle, pdb
from torch import nn
import numpy as np
from collections import OrderedDict
from avg_emb_params import *
sys.path.insert(1, SIE_ckpt_path)
sys.path.insert(1, '/homes/bdoc3/my_utils')
from my_os import recursive_file_retrieval
from my_arrays import fix_feat_length
from tqdm import tqdm

from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq


class SingerIdEncoder(nn.Module):
    def __init__(self, device, loss_device, num_feats):
        super().__init__()
        self.loss_device = loss_device
        
        # Network defition
        self.lstm = nn.LSTM(input_size=num_feats,
                            hidden_size=model_hidden_size, 
                            num_layers=model_num_layers, 
                            batch_first=True).to(device)
        self.linear = nn.Linear(in_features=model_hidden_size, 
                                out_features=model_embedding_size).to(device)
        self.relu = torch.nn.ReLU().to(device)
        
        # self.class_layer = nn.Linear(model_embedding_size, class_num).to(device)
            # ,nn.Dropout(self.dropout)
            # ,nn.BatchNorm1d(512)
            # ,nn.ReLU()
            
        
        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(loss_device)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)
        
    def do_gradient_ops(self, loss):
        # Gradient scale
        if loss.is_cuda != True: # if not true, its going to be the 
            self.similarity_weight.grad *= 0.01
            self.similarity_bias.grad *= 0.01
            
        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.
        
        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape 
        (batch_size, n_frames, n_channels) 
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers, 
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        # Pass the input through the LSTM layers and retrieve all outuuuputs, the final hidden state
        # and the final cell state.
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        # L2-normalize it
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        
        # class_preds = self.class_layer(embeds)

        # return embeds, class_preds
        return embeds
    
    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(self.loss_device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        ## Even more vectorized version (slower maybe because of transpose)
        # sim_matrix2 = torch.zeros(speakers_per_batch, speakers_per_batch, utterances_per_speaker
        #                           ).to(self.loss_device)
        # eye = np.eye(speakers_per_batch, dtype=np.int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)
        
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix
    
    def loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.
        
        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        loss = self.loss_fn(sim_matrix, target)
        
        # EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # Snippet from https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer


def build_model(total_num_feats):

    # load checkpoint, and dependant strings
    print('Building model...\n')

    if not SIE_ckpt_path.endswith('untrained'):
        print('using trained model')
        sie_checkpoint = torch.load(os.path.join(SIE_ckpt_path, 'saved_model.pt'))
        new_state_dict = OrderedDict()
        if SIE_ckpt_path.endswith('autoVc_pretrained'):
            model_state = 'model_b'
            sie_num_feats_used = sie_checkpoint[model_state]['module.lstm.weight_ih_l0'].shape[1]
        else:
            model_state = 'model_state'
            sie_num_feats_used = sie_checkpoint[model_state]['lstm.weight_ih_l0'].shape[1]
        # sie_num_voices_used = sie_checkpoint['model_state']['class_layer.weight'].shape[0]

        # transfer checkpoint info to SIE model
        assert total_num_feats == sie_num_feats_used
        sie = SingerIdEncoder(device, loss_device, sie_num_feats_used)
    
        # prepend initiated weights from freshly loaded SIE to new_state_dict
        if 'autoVc_pretrained' in SIE_ckpt_path:
            new_state_dict['similarity_weight'] = sie.similarity_weight
            new_state_dict['similarity_bias'] = sie.similarity_bias

        # incrementally update new_state_dict with checkpoint params, and load
        for (key, val) in sie_checkpoint[model_state].items():
            # condtional if a certain recognised type of SIE, keys() will be different
            if SIE_ckpt_path.endswith('autoVc_pretrained'):
                key = key[7:] # gets right of the substring 'module'
                if key.startswith('embedding'):
                    key = 'linear.' +key[10:]
            new_state_dict[key] = val
        sie.load_state_dict(new_state_dict)

    else:
        print('using untrained model')
        sie = SingerIdEncoder(device, loss_device, total_num_feats)

    for param in sie.parameters():
        param.requires_grad = False
    sie_optimizer = torch.optim.Adam(sie.parameters(), adam_init)

    # ensure all tensors in optimizer are on same device
    for state in sie_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(device)

    sie.to(device)
    sie.eval()
    return sie

# creates directory for list
SIE_name = os.path.basename(SIE_ckpt_path)
SIE_meta_dir = os.path.join('metadata', SIE_name)
if not os.path.exists(SIE_meta_dir):
    print(f'making dir {SIE_meta_dir}')
    os.mkdir(SIE_meta_dir)

with open(os.path.join(parent_dir, 'feat_params.yaml'), 'rb') as handle:
    feat_params = yaml.load(handle, Loader=yaml.FullLoader)
if use_aper_feats:
    total_num_feats = feat_params['num_harm_feats'] + feat_params['num_aper_feats']
else:
    total_num_feats = feat_params['num_harm_feats']

loss_device = torch.device("cpu")
use_cuda = torch.cuda.is_available()
device = torch.device(f'cuda:{which_cuda}' if use_cuda else 'cpu')
SIE = build_model(total_num_feats) # make sure model layer params in avg_emb_params are correct


multisong_voice_dirs = []

# collect all paths from subset that posses above a designated number of audio files
subdir = os.path.join(parent_dir, subset)
r, voice_dirs, fps = next(os.walk(subdir))
for vd in voice_dirs:
    if len(os.listdir(os.path.join(r, vd))) >= min_tracks_pp:
        print(os.path.join(r, vd))
        multisong_voice_dirs.append(os.path.join(r, vd))

# either shuffle voicedir list or choose a random number of them
this_seed = 0
random.seed(this_seed)
if max_num_singers == None:
    random.shuffle(multisong_voice_dirs)
else:
    # pdb.set_trace()
    multisong_voice_dirs = random.sample(multisong_voice_dirs, k=max_num_singers)

num_classes = len(multisong_voice_dirs)

# go through chosen/randomised voicedir list, get feats, get embs, save in grouped list
all_singer_embs = []
singer_meta_data = []

# go through all voice dirs
for label, vd in enumerate(multisong_voice_dirs):
    singer_track_name_dirs = []
    track_pp = len(os.listdir(vd))
    print(f'Voice: {vd}, {label}/{len(multisong_voice_dirs)}')
    avg_pitch = 0
    singer_pitches = []
    singer_embs = []
    # go through this voices tracks
    for i, fn in enumerate(os.listdir(vd)[:max_tracks_pp]):
        print(f'File: {fn}, {i}/{min(len(os.listdir(vd)), max_tracks_pp)}')
        fp = os.path.join(vd, fn)
        if not fp.endswith('npy'):
            continue
        feats = np.load(fp)
        voiced = feats[:,-1].astype(int) == 0
        singer_pitches.extend(feats[:,-2][voiced])
        # scan through this track in chunks
        # if i ==3:
        #     pdb.set_trace()
        for start in range(0, len(feats)-window_timesteps, window_timesteps):
            trimmed_feats, _ = fix_feat_length(feats, window_timesteps, start)
            spectral_feats = trimmed_feats[:,:total_num_feats]
            SIE_input = torch.from_numpy(spectral_feats).to(device).float().unsqueeze(0)
            singer_emb = SIE(SIE_input)
            singer_emb = singer_emb.squeeze(0).cpu().detach().numpy()
            singer_embs.append(singer_emb)
        singer_track_name_dirs.append(os.path.join(os.path.basename(vd), fn))

    
    all_singer_embs.append(singer_embs)
    
    singer_avg = np.mean(np.asarray(singer_embs), axis=0)
    # singer_pitches = np.asarray(singer_pitches)
    # pitch_stats = [np.mean(singer_pitches), np.std(singer_pitches)]
    single_meta_entry = [os.path.basename(vd), singer_avg]    
    for i in singer_track_name_dirs:
        single_meta_entry.append(i)
    
    singer_meta_data.append(single_meta_entry)
    # pdb.set_trace()

print('Saving data to disk...')
ds_name = os.path.basename(parent_dir)
singer_metad_fn = f'{ds_name}_{subset}_singers_metadata.pkl'

####-THIS SECTION WAS FOR CONVERTING LIST INTO DICT-####

# # save metadata as dictionary to either new file, or overwrite previous file
# metadic = {}
# metadic['global_info'] = { 'keys':'singer_id', 'contents':('average_SIE_embs', ('pitch_average', 'pitch_std')), 'SIE_ckpt_path':SIE_ckpt_path, 'data_dir_used':parent_dir}
# for i, singer_met in enumerate(singer_meta_data):
#     metadic[singer_met[0]] = (singer_met[1], singer_met[2])
# with open(f'{os.path.join(SIE_meta_dir, singer_metad_fn)}', 'wb') as handle:
#     pickle.dump(metadic, handle)

####-THIS SECTION WAS FOR CONVERTING LIST INTO DICT-####

with open(f'{os.path.join(SIE_meta_dir, singer_metad_fn)}', 'wb') as handle:
    pickle.dump(singer_meta_data, handle)

singer_metad_fn = f'{ds_name}_{subset}_singers_chunks_embs.pkl'
with open(f'{os.path.join(SIE_meta_dir, singer_metad_fn)}', 'wb') as handle:
    pickle.dump(all_singer_embs, handle)
# else:
#     current_all_embs = pickle.load(open(existing_all_singer_embs_path, 'rb'))
#     for singer_embs in all_singer_embs:
#         current_all_embs.append(singer_embs)
#     with open(existing_all_singer_embs_path, 'wb') as handle:
#         pickle.dump(current_all_embs, handle)
