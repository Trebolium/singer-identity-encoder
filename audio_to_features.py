import os, time, sys, argparse
from librosa.filters import mel
import numpy as np
if os.path.abspath('../my_utils') not in sys.path: sys.path.insert(1, os.path.abspath('../my_utils'))
from my_os import recursive_file_retrieval
from my_audio.utils import audio2feats_process
from my_container import substring_exclusion, substring_inclusion, balance_by_strings, separate_by_starting_substring
from my_datasets.utils import make_dataset_dir
from my_threads import multithread_chunks

"""Script for collecting wav files from directory,
and converting them to features
"""

def str2bool(v):
    return v.lower() in ('true')

def nus_filter(rootDir, config):
    config.audio_ext = 'wav'

    _, fileList = recursive_file_retrieval(rootDir)
    fileList = [fp for fp in fileList if fp.endswith(config.audio_ext) and not fp.startswith('.')]
    return fileList    


def vctk_filter(rootDir, config):

    config.audio_ext = '.flac'
    _, fileList = recursive_file_retrieval(rootDir)
    fileList = [fp for fp in fileList if fp.endswith(config.audio_ext) and not fp.startswith('.')]
    filtered_list = [fp for fp in fileList if fp[:-len(config.audio_ext)][-1] != '2']
    return filtered_list

def damp_filter(rootDir, config):

    config.audio_ext = '.m4a'
    _, fileList = recursive_file_retrieval(rootDir)
    filtered_list = [fp for fp in fileList if fp.endswith(config.audio_ext) and not fp.startswith('.')]
    return filtered_list

def wav_filter(rootDir, config):
    
    config.audio_ext = '.wav'
    _, fileList = recursive_file_retrieval(rootDir)
    filtered_list = [fp for fp in fileList if fp.endswith(config.audio_ext) and not fp.startswith('.')]
    return filtered_list

def vocalset_filter(rootDir, config):

    config.audio_ext = '.wav'
    # declare variables
    balance_list = True
    use_subclass = True

    _, fileList = recursive_file_retrieval(rootDir)
    fileList = [fp for fp in fileList if fp.endswith(config.audio_ext) and not fp.startswith('.')]

    if use_subclass:
        class_list=['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
        exclude_list = ['caro','row','long','dona']
        exclusive_list = substring_exclusion(fileList, exclude_list) 
        inclusive_list = substring_inclusion(exclusive_list, class_list)
        fileList = inclusive_list

    filtered_list  = [f for f in fileList if f[-6] == '_'] # to ensure that the second last char of the filename is '_', as we are not interested in deviating formats
    
    if balance_list:
        singer_list = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10','m11','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
        files_by_singers = separate_by_starting_substring(filtered_list, singer_list)
        total_balanced_list = []
        for f_by_singer in files_by_singers:
            total_balanced_list.extend(balance_by_strings(f_by_singer, class_list, 100))
        return total_balanced_list  
    else:
        return filtered_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='params for converting audio to spectral using world', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dd','--dst_dir', default='example_feats', type=str)
    parser.add_argument('-sd','--src_dir', default='example_audio', type=str)
    parser.add_argument('-de','--dst_ext', default='.npy', type=str)
    parser.add_argument('-fd','--find_diff', default=1, type=int)
    #multithread
    parser.add_argument('-np','--num_processes', default=16, type=int)    
    parser.add_argument('-um','--use_multithread', default=1, type=int)    
    parser.add_argument('-wd','--which_dataset', default='damp', type=str)
    parser.add_argument('-ne','--numpy_ext', default='.npy', type=str)
    parser.add_argument('-cc','--channel_choice', default='left', type=str)
    parser.add_argument('-ds','--desilence', default=0, type=int)
    # feat params
    parser.add_argument('-sr','--sampling_rate', default=16000, type=int)
    parser.add_argument('-fdm','--frame_dur_ms', default=16, type=int)    
    parser.add_argument('-nhf','--num_harm_feats', default=80, type=int)
    parser.add_argument('-ft','--feat_type', default='mel', type=str)
    # ensure these are correct for the given type of features
    parser.add_argument('-fs','--fft_size', default=1024, type=int)
    parser.add_argument('-fmin', default=90, type=int)
    parser.add_argument('-fmax', default=7600, type=int)
    # world specific params
    parser.add_argument('-wp','--w2w_process', default=None, type=str)
    parser.add_argument('-drm','--dim_red_method', default=None, type=str)
    parser.add_argument('-naf','--num_aper_feats', default=None, type=int)


    config = parser.parse_args()

    time_start = time.time()

    feat_params = {'w2w_process':config.w2w_process,                           
                    'dim_red_method':config.dim_red_method,
                    'num_harm_feats':config.num_harm_feats,
                    'num_aper_feats':config.num_aper_feats,
                    'frame_dur_ms':config.frame_dur_ms,
                    'sr':config.sampling_rate,
                    'fft_size':config.fft_size,
                    "fmin":config.fmin,
                    "fmax":config.fmax
                    }
    
    make_dataset_dir(config.dst_dir, feat_params)
    file_path_issues = []
    if 'vocalset' in config.which_dataset.lower():
        filtered_list = vocalset_filter(config.src_dir, config)
    elif 'vctk' in config.which_dataset.lower():
        filtered_list = vctk_filter(config.src_dir, config)
    elif 'damp' in config.which_dataset.lower():
        filtered_list = damp_filter(config.src_dir, config)
    elif 'nus' in config.which_dataset.lower():
        filtered_list = nus_filter(config.src_dir, config)
    elif 'mir1k' in config.which_dataset.lower() or 'vocadito' in config.which_dataset.lower():
        filtered_list = wav_filter(config.src_dir, config)
    else:
        raise NotImplementedError

    

    if config.find_diff:
        _, dst_dir_all_fps = recursive_file_retrieval(config.dst_dir)
        dst_dir_all_fns = [os.path.basename(path)[:-len(config.dst_ext)] for path in dst_dir_all_fps]
        filtered_list = [path for path in filtered_list if os.path.basename(path)[:-len(config.audio_ext)] not in dst_dir_all_fns]

    if config.feat_type == 'mel':
        mel_filter = mel(sr=feat_params['sr'], n_fft=feat_params['fft_size'], fmin=feat_params['fmin'], fmax=feat_params['fmax'], n_mels=feat_params['num_harm_feats']).T
        min_level = np.exp(-100 / 20 * np.log(10))
        hop_size = int((feat_params['frame_dur_ms']/1000) * feat_params['sr'])
        args = [config, feat_params, mel_filter, min_level, hop_size]
    elif config.feat_type == 'world' or config.feat_type == 'crepe':
        args = [config, feat_params]
    else:
        raise Exception('feat_type param was not recognised.')

    if config.use_multithread:
        multithread_chunks(audio2feats_process, filtered_list, config.num_processes, args)
    else:
        for i, fp in enumerate(filtered_list):
            print(i, '/', len(filtered_list))
            arg_list = [fp] + args
            audio2feats_process(arg_list)

    print(file_path_issues)

