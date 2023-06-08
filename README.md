# Singer Identity Embedding Encoder

## An extension of [CorentinJ's work](https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/master/encoder) on voice cloning

This repository comes with a small set of audio files for training and validation. These are found in the ```singer-identity-encoder/example_audio``` directory. To train the network, you will need to replace it with a more substantial dataset. The training-validation cycles and number of iterations are by default set very low, intending to demonstrating proof of concept quickly. These hyperparameters, along with many others and pathways are configurable although the interface used to change these is different between the singer-identity-encoder and autoSvc repositories (see original documentation for more details on this).

When using this framework to train your own models, please take car eto ensure you examine the argparse argument options, or the parameter files. The current default values are set to make the repository work quickly.

## Generate audio features

To convert the audio files to melspectrogram features, run ```python singer-identity-encoder/audio_to_features.py```. This will automatically save the feature numpy files to ```singer-identity-encoder/example_feats```.

## Pretrain SIE encoder

To train the singer identity encoder using these generated features, run ```python singer-identity-encoder/main.py```. This will train an Singer Identity Embedding (SIE) model and automatically save it to ```singer-identity-encoder/sie_models```, populated with input feature examples, configuration text file, a ```saved_model.pt``` file created when validation loss improves, and a ```ckpt_#.ckpt``` file created when training is finished.

## Generate SIE lookup table

Now that the SIE model is trained, we can generate an average SIE for each singer across all of their recordings in the given dataset. To do this, run ```python singer-identity-encoder/generate_sie_table.py```, which saves the resulting SIEs to a directory at ```../voice_embs_visuals_metadata/default_model/example_feats``` (assuming variables remain at default settings). Unlike ```singer-identity-encoder/main.py```, this script uses the parameter file ```avg_emb_params.py``` for its arguments (don't judge me, I have my reasons :P).

# Plot your embeddings on a 2D plane

Plot the embeddings of each singer you have generated SIEs for on a 2D plane to verify salience between singers, implying how robust the SIEs descriminating capabilities are. To do this, run ```python singer-identity-encoder/plot_enbs_tsne.py```. Once again, this script uses ```argparse``` as a interface for providing variable values as arguments. Output visualisations are sent to their default location at ```../voice_embs_visuals_metadata/default_model/example_feats/train``` (assuming variables remain at default settings).