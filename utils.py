from pathlib import Path
import argparse, sys, random, pdb
import numpy as np
import pyworld as pw
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