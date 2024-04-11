from  __future__ import absolute_import, division, print_function

import numpy as np

from core.base.base_generator import BaseGenerator


class Generator(BaseGenerator):
    def __init__(self, config, mode, X=None, ADV=None):
        super(Generator, self).__init__(config, mode)
        self.build_generator(X=X, ADV=ADV)

    def save_data(self, iter):
        return super().save_data(iter)

    def generate_random_X(self, shape):
        return np.random.uniform(2,3, shape)

    def generate_random_ADV(self, shape):
        return np.random.uniform(2,3, shape)