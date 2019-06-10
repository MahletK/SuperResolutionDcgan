import tensorflow as tf

from numpy.random import permutation

# from discriminator import discriminator
# from generator import generator
# from costs_and_vars import costs_and_vars
# #from BatchGenerator import BatchGenerator
# from batch_norm import batch_norm
# from conv2d import conv2d

class BatchGenerator:
    '''Generator class returning list of indexes at every iteration.'''
    def __init__(self, batch_size, dataset_size):
        self.batch_size   = batch_size
        self.dataset_size = dataset_size

        assert (self.dataset_size > 0)               , 'Dataset is empty.'
        assert (self.dataset_size >= self.batch_size), 'Invalid bathc_size.'
        assert (self.batch_size > 0)                 , 'Invalid bathc_size.'

        self.last_idx = -1
        self.idxs     = permutation(dataset_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.last_idx + self.batch_size <= self.dataset_size - 1:
            start = self.last_idx + 1
            self.last_idx += self.batch_size

            return self.idxs[start: self.last_idx + 1]

        else:
            if self.last_idx == self.dataset_size - 1:
                raise StopIteration

            start = self.last_idx + 1
            self.last_idx = self.dataset_size - 1

            return self.idxs[start: self.last_idx + 1]

