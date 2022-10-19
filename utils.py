from pathlib import Path
import argparse, sys, random, pdb, os
import numpy as np
from collections import OrderedDict

import pyworld as pw
from avg_emb_params import *
from my_normalise import zero_one_mapped, unit_var
# sys.path.insert(1, '/homes/bdoc3/my_utils')

# for some reason there is an unwatend path in sys.path. Must figure out how to remove this
# for i in sys.path:
#     if i == '/homes/bdoc3/wavenet_vocoder':
#         sys.path.remove(i)

from my_audio.world import code_harmonic, sp_to_mfsc, freq_to_vuv_midi

_type_priorities = [    # In decreasing order
    Path,
    str,
    int,
    float,
    bool,
]

def _priority(o):
    p = next((i for i, t in enumerate(_type_priorities) if type(o) is t), None) 
    if p is not None:
        return p
    p = next((i for i, t in enumerate(_type_priorities) if isinstance(o, t)), None) 
    if p is not None:
        return p
    return len(_type_priorities)

def print_args(args: argparse.Namespace, parser=None):
    args = vars(args)
    if parser is None:
        priorities = list(map(_priority, args.values()))
    else:
        all_params = [a.dest for g in parser._action_groups for a in g._group_actions ]
        priority = lambda p: all_params.index(p) if p in all_params else len(all_params)
        priorities = list(map(priority, args.keys()))
    
    pad = max(map(len, args.keys())) + 3
    indices = np.lexsort((list(args.keys()), priorities))
    items = list(args.items())
    
    print("Arguments:")
    for i in indices:
        param, value = items[i]
        print("    {0}:{1}{2}".format(param, ' ' * (pad - len(param)), value))
    print("")

class EarlyStopping():
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.lowest_loss = float('inf')

    def check(self, loss):
                
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            else:
                return False


def get_world_feats(y, feat_params, config):
    if config.w2w_process == 'wav2world':
        feats=pw.wav2world(y, feat_params['sr'],frame_period=feat_params['frame_dur_ms'])
        harm = feats[1]
        aper = feats[2]
        refined_f0 = feats[0]

    else:
        if config.w2w_process == 'harvest':
            f0, t_stamp = pw.harvest(y, feat_params['sr'], feat_params['fmin'], feat_params['fmax'], feat_params['frame_dur_ms'])
        elif config.w2w_process =='dio':
            f0, t_stamp = pw.dio(y, feat_params['sr'], feat_params['fmin'], feat_params['fmax'], frame_period = feat_params['frame_dur_ms'])
        refined_f0 = pw.stonemask(y, f0, t_stamp, feat_params['sr'])
        harm = pw.cheaptrick(y, refined_f0, t_stamp, feat_params['sr'], f0_floor=feat_params['fmin'])
        aper = pw.d4c(y, refined_f0, t_stamp, feat_params['sr'])
    refined_f0 = freq_to_vuv_midi(refined_f0) # <<< this can be done at training time
    

    if config.dim_red_method == 'code-h':
        harm = code_harmonic(harm, feat_params['num_harm_feats'])
        aper = code_harmonic(aper, feat_params['num_aper_feats'])
    elif config.dim_red_method == 'world':
        harm = pw.code_spectral_envelope(harm, feat_params['sr'], feat_params['num_harm_feats'])
        aper = pw.code_aperiodicity(aper, feat_params['num_harm_feats'])
    elif config.dim_red_method == 'chandna':
        harm = 10*np.log10(harm) # previously, using these logs was a separate optional process to 'chandna'
        aper = 10*np.log10(aper**2)
        harm = sp_to_mfsc(harm, feat_params['num_harm_feats'], 0.45)
        aper =sp_to_mfsc(aper, feat_params['num_aper_feats'], 0.45)
    else:
        raise Exception("The value for dim_red_method was not recognised")

    out_feats=np.concatenate((harm,aper,refined_f0),axis=1)

    return out_feats

def norm_feat_arr(arr, config):
    if config.norm_method == 'zero_one':
        arr = zero_one_mapped(arr)
    elif config.norm_method == 'unit_var':
        arr = unit_var(arr)
    return arr


import torch
from torch import nn
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq

#FIXME: Why don't we just use the SpeakerEncoder instead?
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


def build_SIE_model(total_num_feats, device):

    """
    Loads model parameters from model path, or loads an untrained model
    Designed specifically for specific models due to following assumptions:
        If model ends with 'autoVc_pretrained', we know:
            1. Its named_parameters are stored in dictionary name 'model_b'
            2. The names of the models parameters would start with 'module'
            3. That we must rename its layers to match those of our
                SingerIdEncoder's before transferring parameter information.
            4. Use the names of its initial layers through prior knowledge to
                to verify the number features it was trained on/designed for.
    Sets up model for evaluation with .eval() and requires_grad==False
    Loads all parameters to a specific device (GPU/CPU)

    Args:
        total_num_feats: the number of features we intend to use
        NB: All variables are currently provided from a parameter python file
    
    """
    # load checkpoint, and dependant strings
    print('Building model...\n')
    loss_device = torch.device("cpu")
    
    if SIE_ckpt_path != None:
        print('using trained model')
        sie_checkpoint = torch.load(os.path.join(SIE_ckpt_path, 'saved_model.pt'))
        new_state_dict = OrderedDict()

        # verify the number of features (sie_num_feats_used) model was trained on matches our intended use
        if SIE_ckpt_path.endswith('autoVc_pretrainedOnVctk_Mels80'):
            model_state = 'model_b'
            sie_num_feats_used = sie_checkpoint[model_state]['module.lstm.weight_ih_l0'].shape[1]
        else:
            model_state = 'model_state'
            sie_num_feats_used = sie_checkpoint[model_state]['lstm.weight_ih_l0'].shape[1]
        assert total_num_feats == sie_num_feats_used

        sie = SingerIdEncoder(device, loss_device, sie_num_feats_used)
        # add initiated weights from freshly loaded SIE to new_state_dict
        if 'autoVc_pretrained' in SIE_ckpt_path:
            new_state_dict['similarity_weight'] = sie.similarity_weight
            new_state_dict['similarity_bias'] = sie.similarity_bias

        # incrementally update new_state_dict with checkpoint params, and load
        for (key, val) in sie_checkpoint[model_state].items():
            # condtional if a certain recognised type of SIE, keys() will be different
            if SIE_ckpt_path.endswith('autoVc_pretrainedOnVctk_Mels80'):
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