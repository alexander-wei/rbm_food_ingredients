"""train.py"""

from typing import List
from tqdm.auto import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from rbm import RBM
from ingred2vec import Binarizer

class IngredientsRBM(RBM):
    """Extend Gabriel M. Bianconi's with density penalizers, training loop, and unicode
    wrapping functions"""
    data_loader: DataLoader

    def __init__(self, data_loader: DataLoader, num_visible, num_hidden, k, learning_rate=0.001, 
                 momentum_coefficient=0.5, weight_decay=0.0001, use_cuda=True):
        super().__init__(num_visible, num_hidden, k, learning_rate, momentum_coefficient,
                         weight_decay, use_cuda)
        self.data_loader = data_loader

    def fit(self, n_epochs: int=4, batch_size: int=4):
        for epoch in range(n_epochs):        
            train_loader = self.data_loader
            epoch_error = 0.0

            frame_log = tqdm(total=0, position=2, bar_format='{desc}')
            for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                batch = batch.view(batch_size, self.num_visible)
                batch_error = self.contrastive_divergence(batch.cuda())
                batch_error += 1 *torch.norm(self.hidden_bias, 1)
                batch_error += 1 *torch.norm(self.negative_visible_probabilities, 1)
                epoch_error += batch_error.cpu()
                frame_log.set_description_str("bat %d: %.4f" % (batch_idx, epoch_error / batch_idx))

            print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))

    def str_sample_hidden(self, x: List[List[str]], binarizer: Binarizer, external=True):
        """unicode string to embedding (RBM hidden activations)"""
        mh_encoded_string = torch.Tensor(
            np.array([binarizer.transform([u])\
                      for u in x])[:,0,:]
        )
        rbm_embeds = super().sample_hidden(mh_encoded_string.cuda())
        if external:
            rbm_embeds = rbm_embeds.detach().cpu().numpy()

        return rbm_embeds

    def str_predict(self, x: List[List[str]], binarizer: Binarizer, temperature=.01):
        """embedding (RBM hidden activations) to unicode string"""
        rbm_embeds = self.str_sample_hidden(x, binarizer, external=False)
        return binarizer.inverse_transform(gate(self.sample_visible(
            rbm_embeds), temperature))

def gate_(x, temperature=.01):
    """filter visible units according to threshhold "temperature" """
    for u in x:
        if u > temperature:
            yield 1.
            continue
        yield 0

def gate(x, temperature=.01):
    """reconstruct ingredients by activations of visible units"""
    def part(x):
        for v in x:
            yield list(gate_(
                v, temperature))
    return np.array(list(part(x)))
