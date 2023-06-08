"""
This module implements the SpeakerEncoder class for extracting descriminative
speaker embeddings which can be achieved through training with  GE2E loss.

SpeakerEncoder Class:
- __init__(self, device, loss_device, class_num, num_total_feats, model_hidden_size, model_embedding_size,
           model_num_layers, use_classify=True): Initializes the SpeakerEncoder object with the specified parameters.
- do_gradient_ops(self, loss): Performs gradient operations including scaling and clipping.
- forward(self, utterances, hidden_init=None): Computes the embeddings of a batch of utterance spectrograms.
- similarity_matrix(self, embeds): Computes the similarity matrix for the embeddings.
- loss(self, embeds): Computes the softmax loss and equal error rate (EER) for the embeddings.

Minimally altered code from https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/master/encoder/data_objects
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from torch.nn.utils import clip_grad_norm_


class SpeakerEncoder(nn.Module):
    """
    SpeakerEncoder class for extracting speaker embeddings and performing speaker classification.

    Parameters:
    - device (str): Device to be used for computation (e.g., 'cuda' for GPU, 'cpu' for CPU).
    - loss_device (str): Device to be used for loss computation (e.g., 'cuda' for GPU, 'cpu' for CPU).
    - class_num (int): Number of speaker classes for classification.
    - num_total_feats (int): Total number of input features.
    - model_hidden_size (int): Size of the hidden layer in the LSTM.
    - model_embedding_size (int): Size of the output embeddings.
    - model_num_layers (int): Number of layers in the LSTM model.
    - use_classify (bool): Flag indicating whether to perform speaker classification.

    Methods:
    - do_gradient_ops(self, loss): Performs gradient operations including scaling and clipping.
    - forward(self, utterances, hidden_init=None): Computes the embeddings of a batch of utterance spectrograms.
    - similarity_matrix(self, embeds): Computes the similarity matrix for the embeddings.
    - loss(self, embeds): Computes the softmax loss and equal error rate (EER) for the embeddings.
    """

    def __init__(
        self,
        device,
        loss_device,
        class_num,
        num_total_feats,
        model_hidden_size,
        model_embedding_size,
        model_num_layers,
        use_classify=True,
    ):
        super().__init__()
        self.device = device
        self.loss_device = loss_device
        self.use_classify = use_classify

        # Network definition
        self.lstm = nn.LSTM(
            input_size=num_total_feats,
            hidden_size=model_hidden_size,
            num_layers=model_num_layers,
            batch_first=True,
        ).to(device)

        self.linear = nn.Linear(
            in_features=model_hidden_size, out_features=model_embedding_size
        ).to(device)

        self.relu = nn.ReLU().to(device)

        if use_classify:
            self.class_layer = nn.Linear(model_embedding_size, class_num).to(device)
        else:
            self.class_num = class_num

        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.0])).to(loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0])).to(loss_device)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)

    def do_gradient_ops(self, loss):
        """
        Perform gradient operations including scaling and clipping.

        Parameters:
        - loss (torch.Tensor): Loss tensor.

        Returns:
        - None
        """
        # Gradient scale
        if loss.is_cuda != True:
            self.similarity_weight.grad *= 0.01
            self.similarity_bias.grad *= 0.01

        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)

    def forward(self, utterances, hidden_init=None):
        """
        Compute the embeddings of a batch of utterance spectrograms.

        Parameters:
        - utterances (torch.Tensor): Batch of mel-scale filterbanks of shape (batch_size, n_frames, n_channels).
        - hidden_init (torch.Tensor): Initial hidden state of the LSTM as a tensor of shape
          (num_layers, batch_size, hidden_size). Defaults to None.

        Returns:
        - embeds (torch.Tensor): Embeddings as a tensor of shape (batch_size, embedding_size).
        - class_preds (torch.Tensor): Class predictions as a tensor of shape (batch_size, class_num).
        """
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state,
        # and the final cell state.
        out, (hidden, cell) = self.lstm(utterances, hidden_init)

        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))

        # L2-normalize it
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)

        if self.use_classify:
            class_preds = self.class_layer(embeds)
        else:
            class_preds = torch.zeros((len(embeds), self.class_num)).to(self.device)
            class_preds[:, 0] = 1.0

        return embeds, class_preds

    def similarity_matrix(self, embeds):
        """
        Compute the similarity matrix according to section 2.1 of GE2E.

        Parameters:
        - embeds (torch.Tensor): Embeddings as a tensor of shape (speakers_per_batch,
          utterances_per_speaker, embedding_size).

        Returns:
        - sim_matrix (torch.Tensor): Similarity matrix as a tensor of shape (speakers_per_batch,
          utterances_per_speaker, speakers_per_batch).
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (
            torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5
        )

        # Exclusive centroids (1 per utterance)
        centroids_excl = torch.sum(embeds, dim=1, keepdim=True) - embeds
        centroids_excl /= utterances_per_speaker - 1
        centroids_excl = centroids_excl.clone() / (
            torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5
        )

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(
            speakers_per_batch, utterances_per_speaker, speakers_per_batch
        ).to(self.loss_device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)

        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias

        return sim_matrix

    def loss(self, embeds):
        """
        Compute the softmax loss according to section 2.1 of GE2E.

        Parameters:
        - embeds (torch.Tensor): Embeddings as a tensor of shape (speakers_per_batch,
          utterances_per_speaker, embedding_size).

        Returns:
        - loss (torch.Tensor): Loss for this batch of embeddings.
        - eer (float): Equal error rate (EER) for this batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape(
            (speakers_per_batch * utterances_per_speaker, speakers_per_batch)
        )
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
            eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

        return loss, eer
