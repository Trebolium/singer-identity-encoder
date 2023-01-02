import pickle, os, random, pdb, argparse, time, math, csv
from re import A
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
sys.path.insert(1, '/homes/bdoc3/my_utils')
from my_container import substring_inclusion, reorder_truncate, flatten_and_label
from my_csv import vctk_id_gender_list


def get_vocalset_gender_techs():

    singing_techniques = ['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
    gender_group_labels_arr = []
    technique_group_labels_arr = []
    for voice_meta in metad_by_singer_list:
        voice_fns = substring_inclusion(voice_meta[2:], singing_techniques)
        print(len(voice_fns))

        for fn in voice_fns:

            # count += 1

            if fn.startswith('m'):
                gender_group_labels_arr.append('male')
            elif fn.startswith('f'):
                gender_group_labels_arr.append('female')
            
            st_found = False
            for st_i, st in enumerate(singing_techniques):
                if st in fn:
                    technique_group_labels_arr.append(st_i)
                    st_found = True
            if not st_found:
                pdb.set_trace()
                raise Exception('St not found')
                
    gender_group_labels_arr = np.asarray(gender_group_labels_arr)
    all_labels_arrs.append(gender_group_labels_arr)
    all_label_names.append('gender')
    all_labels_class_sizes.append(2)

    technique_group_labels_arr = np.asarray(technique_group_labels_arr)
    all_labels_arrs.append(technique_group_labels_arr)
    all_label_names.append('singing_technique')
    all_labels_class_sizes.append(config.max_num_techs)
    return gender_group_labels_arr, technique_group_labels_arr


def get_vocadito_gender():
    gender_group_labels_arr = []
    csv_path = '/homes/bdoc3/my_data/text_data/vocadito/vocadito_metadata.csv'
    f = open(csv_path, 'r')
    reader = csv.reader(f)
    header = next(reader)
    singer_meta = [row for row in reader]
    perf_key_meta_list = [row[0] for row in singer_meta]
    gender_meta_list = [row[4] for row in singer_meta]

    for voice_meta in metad_by_singer_list:

        uttrs_fps = voice_meta[2:]
        for fp in uttrs_fps:
            track_name = os.path.basename(fp)[:-4]
            track_int = track_name.split('_')[1]
            try:

                idx = perf_key_meta_list.index(track_int)
            except ValueError as e:
                print(e)
                continue

            gender = gender_meta_list[idx]
            if 'm' in gender.lower():
                gender_group_labels_arr.append('male')
            elif 'f' in gender.lower():
                gender_group_labels_arr.append('female')
            else:
                raise Exception(f'Gender value not recognised for excerpt {track_name} in csv row {idx}')

    gender_group_labels_arr = np.asarray(gender_group_labels_arr)
    all_labels_arrs.append(gender_group_labels_arr)
    all_label_names.append('gender')
    all_labels_class_sizes.append(1)
    return gender_group_labels_arr


def get_vctk_gender():
    id_list, gender_list = vctk_id_gender_list()
    gender_group_labels_arr = []
    for voice_meta in metad_by_singer_list:
        uttrs_fps = voice_meta[2:]
        for fp in uttrs_fps:
            track_name = os.path.basename(fp)[:-4]
            singer_id = track_name.split('_')[0]
            idx = id_list.index(singer_id)

            gender = gender_list[idx]
            if 'm' in gender.lower():
                gender_group_labels_arr.append('male')
            elif 'f' in gender.lower():
                gender_group_labels_arr.append('female')
            else:
                raise Exception(f'Gender value not recognised for excerpt {track_name} in csv row {idx}')

    gender_group_labels_arr = np.asarray(gender_group_labels_arr)
    all_labels_arrs.append(gender_group_labels_arr)
    all_label_names.append('gender')
    all_labels_class_sizes.append(1)
    return gender_group_labels_arr


def get_damp_gender():
    """
    Get entries from gender csv file
    Add these to a gender array as 0, 1 or 2 if entry is None
    """
    gender_group_labels_arr = []
    csv_path = '/homes/bdoc3/my_data/text_data/damp/intonation_metadata.csv'
    f = open(csv_path, 'r')
    reader = csv.reader(f)
    header = next(reader)
    singer_meta = [row for row in reader]
    perf_key_meta_list = [row[0] for row in singer_meta]
    gender_meta_list = [row[8] for row in singer_meta]
    perf_key_set = set()

    for voice_meta in metad_by_singer_list:

        uttrs_fps = voice_meta[2:]
        for fp in uttrs_fps:
            perf_key = os.path.basename(fp)[:-4]
            perf_key_set.add(perf_key)

            try:
                idx = perf_key_meta_list.index(perf_key)
            except ValueError as e:
                print(e)
                continue

            gender = gender_meta_list[idx]
            if 'm' in gender.lower():
                gender_group_labels_arr.append('male')
            elif 'f' in gender.lower():
                gender_group_labels_arr.append('female')
            elif 'none' in gender.lower():
                gender_group_labels_arr.append('unknown')
            else:
                raise Exception(f'Gender value not recognised for excerpt {perf_key} in csv row {idx}')

    gender_group_labels_arr = np.asarray(gender_group_labels_arr)
    all_labels_arrs.append(gender_group_labels_arr)
    all_label_names.append('gender')
    #FIXME: put the len-set trick in the process function
# if len(set(all_labels_class_sizes))
# if 2 in gender_group_labels_arr:
#     all_labels_class_sizes.append(3)
# else:
    all_labels_class_sizes.append(2)
    return gender_group_labels_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # path specifications    
    parser.add_argument("-us", "--use_subset", type=str, default='')
    parser.add_argument("-sn", "--sie_name", type=str, default='')
    parser.add_argument("-dn", "--ds_name", type=str, default='')
    parser.add_argument("-pfe", "--pkl_fn_extras", type=str, default='')
    parser.add_argument("-mi", "--max_ids", type=int, default=10)
    parser.add_argument("-mc", "--max_clips", type=int, default=100)
    parser.add_argument("-mt", "--max_techs", type=int, default=6)

    config = parser.parse_args()

    metad_dir = os.path.join('/homes/bdoc3/my_data/voice_embs_visuals_metadata', config.sie_name, config.ds_name, config.use_subset)
    if not os.path.exists(metad_dir):
        os.makedirs(metad_dir)

    # get embeddings and labels
    print('loading data...')
    chunk_embs_by_singer_list = pickle.load(open(os.path.join(metad_dir, f'voices_chunks_embs{config.pkl_fn_extras}.pkl'), 'rb'))
    chunk_embs_by_singer_list = [singer_embs[:config.max_clips] for singer_embs in chunk_embs_by_singer_list]
    metad_by_singer_list = pickle.load(open(os.path.join(metad_dir, f'voices_metadata{config.pkl_fn_extras}.pkl'), 'rb'))
    metad_by_singer_list = [singer_embs[:config.max_clips+2] for singer_embs in metad_by_singer_list]


    # randomise and truncate voices data based on max_num_ids given
    random.seed(1)
    assert len(chunk_embs_by_singer_list) == len(metad_by_singer_list)
    if config.max_ids == math.inf:
        choosen_idxs = random.sample(range(0, len(chunk_embs_by_singer_list)-1), len(chunk_embs_by_singer_list)-1)
    else:
        choosen_idxs = random.sample(range(0, len(chunk_embs_by_singer_list)-1), config.max_ids)
    chunk_embs_by_singer_list = [chunk_embs_by_singer_list[i] for i in choosen_idxs]
    metad_by_singer_list = [metad_by_singer_list[i] for i in choosen_idxs]

    print('selecting data...')
    all_labels_arrs = []
    all_label_names = []
    all_labels_class_sizes = []
    # reorder the grouped list by how many emb entries there are per singer
    num_embs_per_singer = [len(embs) for embs in chunk_embs_by_singer_list]

    # generate voice ID labels and truncate based on these
    all_singer_embs_arr, id_group_labels_arr = flatten_and_label(chunk_embs_by_singer_list)
    id_group_names = np.asarray([metad_by_singer_list[label][0] for label in id_group_labels_arr])
    if config.max_ids == math.inf:
        config.max_ids = len(set(id_group_labels_arr))

    all_labels_arrs.append(id_group_labels_arr)
    all_label_names.append('id')
    if config.max_ids == None:
        all_labels_class_sizes.append(len(set(id_group_labels_arr)))
    else:
        all_labels_class_sizes.append(config.max_ids)
    all_singer_embs_arr, id_group_labels_arr = reorder_truncate(all_singer_embs_arr, id_group_labels_arr, config.max_ids)

    print('formatting data...')
    # make pandas dataframe columns for features, labels
    feat_cols = [ 'timbreFeat'+str(i) for i in range(all_singer_embs_arr.shape[1]) ]
    df = pd.DataFrame(all_singer_embs_arr, columns = feat_cols)
    df['id'] = id_group_names

    # get more labels, depending on dataset used
    if 'vocalset' in config.ds_name.lower():
        gender_group_labels_arr, technique_group_labels_arr = get_vocalset_gender_techs()
        df['gender'] = gender_group_labels_arr
        df['technique'] = technique_group_labels_arr
    else:
        if 'damp' in config.ds_name.lower():
            gender_group_labels_arr = get_damp_gender()
        elif 'vocadito' in config.ds_name.lower():
            gender_group_labels_arr = get_vocadito_gender()
        elif 'vctk' in config.ds_name.lower():
            gender_group_labels_arr = get_vctk_gender()
        else:
            raise NotImplementedError
        df['gender'] = gender_group_labels_arr
        

    print('Making and saving PCA plot...')
    pca = PCA(n_components=2)
    # load df into transform using the feat_cols key we made earlier which ONLY refers to the vocalFeats
    pca_result = pca.fit_transform(df[feat_cols].values)
    # create new columns and add pca results to them
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1]
    # df['pca-three'] = pca_result[:,2]

    print('plotting t-SNE...')
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df[feat_cols].values)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    plt.figure(figsize=(16,10))
    sns.scatterplot(x="tsne-2d-one",
                    y="tsne-2d-two",
                    hue="id",
                    palette=sns.color_palette("bright", len(np.unique(id_group_names))),
                    data=df,
                    legend="full",
                    alpha=0.3,
                    style='gender')
    fp = os.path.join(metad_dir, f'id-gender-2tsne{config.pkl_fn_extras}.png')
    print('saving tsne plot as: ', fp)
    plt.savefig(fp)
    if 'vocalset' in config.ds_name.lower():
        sns.scatterplot(x="tsne-2d-one",
                        y="tsne-2d-two",
                        hue="id",
                        palette=sns.color_palette("bright", len(np.unique(id_group_names))),
                        data=df,
                        legend="full",
                        alpha=0.3,
                        size='gender',
                        style='technique')
        fp = os.path.join(metad_dir, f'id-technique-2tsne{config.pkl_fn_extras}.png')    
        print('saving pca plot as: ', fp)
        plt.savefig(fp)
    # exit(0)

    # for i in range(len(all_labels_arrs)):
    #     label_arr = all_labels_arrs[i]
    #     label_name = all_label_names[i]
    #     num_classes = len(set(label_arr))
    #     # num_classes = all_labels_class_sizes[i]

    #     #FIXME: Since we are not using 

    #     #     voice_embs_classes, label_arr = reorder_truncate(all_singer_embs_arr, label_arr, num_classes)
    #     #     return voice_embs_classes, label_arr
    #     df['y'] = label_arr.astype(int)

    #     # print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    #     plt.figure(figsize=(16,10))
    #     try:
    #         pdb.set_trace()
    #         # sns.scatterplot(x="pca-one", y="pca-two", hue="y", palette=sns.color_palette("hls", num_classes), data=df, legend="full", alpha=0.3)
    #         sns.scatterplot(
    #             x="pca-one", y="pca-two",
    #             hue="y",
    #             palette=sns.color_palette("hls", num_classes),
    #             data=df,
    #             legend="full",
    #             alpha=0.3
    #             )
    #     except:
    #         pdb.set_trace()

    #     # print('saving pca plot as: ', os.path.join(metad_dir, config.ds_name, f'{config.use_subset}-{label_name}-2dPca.png'))
    #     print('saving pca plot as: ', os.path.join(metad_dir, f'{label_name}-{config.max_ids}voices-2dPca.png'))
    #     plt.savefig(os.path.join(metad_dir, f'{label_name}-{config.max_ids}voices-2dPca.png'))


    #     print('making and saving plot...')
    #     plt.figure(figsize=(16,10))
    #     sns.scatterplot(
    #         x="tsne-2d-one", y="tsne-2d-two",
    #         hue="y",
    #         palette=sns.color_palette("hls", num_classes),
    #         data=df,
    #         legend="full",
    #         alpha=0.3
    #         )
            
    #     print('saving pca plot as: ', os.path.join(metad_dir, f'{label_name}-{config.max_ids}voices-2dTsne.png'))
    #     plt.savefig(os.path.join(metad_dir, f'{label_name}-{config.max_ids}voices-2dTsne.png'))