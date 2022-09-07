import pickle, os, random, pdb, argparse, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

def order_by_num_entries(grouped_by_classes, group_size_list):
    # return indices of chunk_size list, ordered by contents (number of chunks)
    indices_ordered_by_size = np.argsort(group_size_list)
    # rearrange list_grouped_by_singer by these indices to get examples of singers in order of number of emb_chunks provided
    ordered_list_grouped_by_classes = [grouped_by_classes[i] for i in indices_ordered_by_size]
    return ordered_list_grouped_by_classes


def chunk_lists_to_arrays(grouped_list):
    # manually create list of labels for embeddings, based on how they are grouped in grouped_list list
    group_labels = []
    list_of_sizes = []
    for i, singer_embs in enumerate(grouped_list):
        num_chunks = len(singer_embs)
        list_of_sizes.append(num_chunks)
        group_labels.extend([i for _ in range(num_chunks)])

    # convert this list to array
    group_labels_arr = np.asarray(group_labels)

    
    grouped_list_arr = np.asarray([singer_emb for singer_embs in grouped_list for singer_emb in singer_embs])

    # pay attention ot the order of the output. Quite relevant for later
    return grouped_list_arr, group_labels_arr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # path specifications    
    parser.add_argument("-us", "--use_subset", type=str, default='val')
    parser.add_argument("-sn", "--sie_name", type=str, default='untrained')
    parser.add_argument("-dn", "--ds_name", type=str, default='damp_inton_80')
    parser.add_argument("-nc", "--num_classes", type=int, default=20)
    config = parser.parse_args()

    
    SIE_meta_dir = os.path.join('metadata', config.sie_name)
    if not os.path.exists(SIE_meta_dir):
        os.mkdir(SIE_meta_dir)

    # get embeddings and labels
    print('loading data...')
    singer_metad_fn = f'{config.ds_name}_{config.use_subset}_singers_chunks_embs.pkl'
    list_grouped_by_singer = pickle.load(open(f'{os.path.join(SIE_meta_dir, singer_metad_fn)}', 'rb'))

    print('selecting data...')
    # reorder the grouped list by how many emb entries there are per singer
    num_embs_per_singer = [len(embs) for embs in list_grouped_by_singer]
    ordered_list_grouped_by_singers = order_by_num_entries(list_grouped_by_singer, num_embs_per_singer)
    # reverses list in place
    ordered_list_grouped_by_singers.reverse()
    # convert reorderd list to array and accompanying label array
    all_singer_embs_arr, group_labels_arr = chunk_lists_to_arrays(ordered_list_grouped_by_singers)
    unique, counts = np.unique(group_labels_arr, return_counts=True)

    print('formatting data...')
    # get embs from singers with most data / emb examples
    selected_singer_embs_arr = all_singer_embs_arr[np.where(group_labels_arr < config.num_classes)]
    selected_group_labels_arr = group_labels_arr[np.where(group_labels_arr < config.num_classes)]
    selected_group_labels_arr = np.expand_dims(selected_group_labels_arr, axis=1)
    choosen_label_embs_arr = np.concatenate((selected_singer_embs_arr, selected_group_labels_arr), axis=1)
    # make pandas dataframe columns for features, labels
    feat_cols = [ 'timbreFeat'+str(i) for i in range(choosen_label_embs_arr[:,:-1].shape[1]) ]
    df = pd.DataFrame(choosen_label_embs_arr[:,:-1], columns = feat_cols)
    df['y'] = choosen_label_embs_arr[:,-1].astype(int)
    df['label'] = df['y'].apply(lambda i: str(i)) # not sure about the point of this line...
    X, y = None, None

    print('Making and saving PCA plot...')
    pca = PCA(n_components=3)
    # load df into transform using the feat_cols key we made earlier which ONLY refers to the vocalFeats
    pca_result = pca.fit_transform(df[feat_cols].values)
    # create new columns and add pca results to them
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]
    # print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", config.num_classes),
        data=df,
        legend="full",
        alpha=0.3
        )
    plt.savefig(os.path.join(SIE_meta_dir, f'{config.ds_name}_{config.use_subset}_2d_pca.png'))

    print('plotting t-SNE...')
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df[feat_cols].values)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    print('making and saving plot...')
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", config.num_classes),
        data=df,
        legend="full",
        alpha=0.3
        )
    plt.savefig(os.path.join(SIE_meta_dir, f'{config.ds_name}_{config.use_subset}_2d_tsne.png'))