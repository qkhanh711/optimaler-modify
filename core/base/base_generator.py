import os

import numpy as np
import random 




class BaseGenerator(object):
    def __init__(self, config, mode="train"):
        self.config = config
        self.mode = mode
        self.num_agents = config.num_agents
        self.num_items = config.num_items
        self.num_instances = config[self.mode].num_batches * config[self.mode].batch_size
        self.num_misreports = config[self.mode].num_misreports
        self.batch_size = config[self.mode].batch_size
        self.X = self.generate_random_X([self.num_instances, self.num_agents, self.num_items])
        self.ADV = self.generate_random_ADV([self.num_misreports, self.num_instances, self.num_agents, self.num_items])
        
        x, y, z = self.X.shape
        N = x * y * z
        
        Q_0 = [i for i in range(20000, 25000)]
        Q_i = [random.randint(200, 2000) for i in range(20000, 25000)]
        q_0, q_i = [], []
        for i in range(x):
            for j in range(y):
                for ij in range(z):
                    inds = random.sample(range(len(Q_0)), random.randint(1, 3))
                    q_0.append(sum([Q_0[_] for _ in inds]))
                    q_i.append(sum([Q_i[_] for _ in inds]))
                    
        Q_image_0 = np.array(q_0, dtype=float)
        Q_image = np.array(q_i, dtype=float)
        t_1 = 0.6
        data_rate = {2: [1012, 1725],
                    3: [1491, 1951],
                    4: [1503, 2023],
                    5: [1520, 2035],
                    6: [1545, 2051],
                    7: [1562, 2079],
                    8: [1588, 2094],
                    9: [1600, 2101]}
        
        Q_user = np.array([random.randint(data_rate[self.num_agents][0],
                            data_rate[self.num_agents][1])* 10 * t_1 for i in range (N)],
                            dtype = float)
        
        R_i = []
        for i in range(N):
            W_B = 12 * 15 * 2**2 * 10^3
            p_i = 0.15
            sigma_2 = 10**(-17)
            d_i = random.randint(20, 300)
            h_i = random.random()
            r_i = W_B * np.log2(1 + (p_i * h_i * d_i ** -2) / (W_B * sigma_2))
            R_i.append(r_i)
        
        R_i = np.array(R_i, dtype=float)
        k_values = np.array([0.6, 0.6, 3.9, 1.0, 1.9, 0.5, 0.3, 4.9, 0.4])
        tau_2 = 0.045
        
        k0, k1, k2, k3, k4, k5, k6, k7, k8 = k_values
        val = [0.0] * len(Q_image)

        condition_1 = (Q_image + Q_user) / R_i <= tau_2
        condition_2 =  (Q_user / R_i) <= tau_2
        condition_3 = (Q_image / R_i) <= tau_2
        val_1, val_2, val_3, val_4 = [], [], [], []
        # print(len(Q_user))
        # print(len(Q_image))
        # print(len(Q_image_0))
        # print(len(R_i))
        for i in range(len(Q_image)):
            if condition_1[i]:
                val[i] = k0 * k1 ** (tau_2 - (Q_image[i] + Q_user[i]) / R_i[i]) + k2 * np.log(1 + Q_image[i] / Q_image_0[i]) + k3
                val_1.append(val[i])
            elif condition_2[i]:
                val[i] = k0 * k1 ** (tau_2 - Q_user[i] / R_i[i]) + k4 * np.log(1 + Q_user[i] / (Q_user[i] + Q_image_0[i])) + k5
                val_2.append(val[i])
            elif condition_3[i]:
                val[i] = k6 * k1**(tau_2 - Q_image[i] / R_i[i]) + k7 * np.log(1 + Q_image[i] / Q_image_0[i]) + k8
                val_3.append(val[i])
            else:
                val[i] = 0.0
                val_4.append(0)
        
        print(np.array(val).shape)
        val = np.arange(N)
        
        
        XX = val.reshape(x, y, z)
        ADV_arr = []
        new_XX = np.random.permutation(XX.ravel()).reshape(x, y, z)
        ADV_arr.append(new_XX)
        if self.num_misreports > 1:
            for i in range (self.num_misreports - 1):
                perm_X = np.random.permutation(XX.ravel()).reshape(x, y, z)
                ADV_arr.append(perm_X)
                
        ADV_arr = np.stack(ADV_arr)
        print(XX.shape)
        print(ADV_arr.shape)
        
        self.X = XX
        self.ADV = ADV_arr
        
    def build_generator(self, X=None, ADV=None):
        if self.mode == "train":
            if self.config.train.data == "fixed":
                self.get_data(X, ADV)
                if self.config.train.restore_iter == 0:
                    self.get_data(X, ADV)
                else:
                    self.load_data_from_file(self.config.train.restore_iter)
                self.gen_func = self.gen_fixed()
            else:
                self.gen_func = self.gen_online()

        else:
            if self.config[self.mode].data == "fixed" or X is not None:
                self.get_data(X, ADV)
                self.gen_func = self.gen_fixed()
            else:
                self.gen_func = self.gen_online()

    def get_data(self, X=None, ADV=None):
        """Generates data"""
        x_shape = [self.num_instances, self.num_agents, self.num_items]
        adv_shape = [self.num_misreports, self.num_instances, self.num_agents, self.num_items]

        if X is None:
            X = self.generate_random_X(x_shape)
        if ADV is None:
            ADV = self.generate_random_ADV(adv_shape)

        self.X = X
        self.ADV = ADV

    def load_data_from_file(self, iter):
        """Loads data from disk"""
        print("Loading data from disk")
        print(os.path.join(self.config.dir_name, "X.npy"))
        self.X = np.load(os.path.join(self.config.dir_name, "X.npy"))
        self.ADV = np.load(os.path.join(self.config.dir_name, "ADV_" + str(iter) + ".npy"))

    def save_data(self, iter):
        """Saved data to disk"""
        if self.config.save_data is None:
            return
            

        if iter == 0:
            np.save(os.path.join(self.config.dir_name, "X"), self.X)
        else:
            np.save(os.path.join(self.config.dir_name, "ADV_" + str(iter)), self.ADV)

    def gen_fixed(self):
        i = 0
        if self.mode == "train":
            perm = np.random.permutation(self.num_instances)
        else:
            perm = np.arange(self.num_instances)

        while True:
            idx = perm[i * self.batch_size : (i + 1) * self.batch_size]
            yield self.X[idx], self.ADV[:, idx, :, :], idx
            i += 1
            if i * self.batch_size == self.num_instances:
                i = 0
                if self.mode is "train":
                    perm = np.random.permutation(self.num_instances)
                else:
                    perm = np.arange(self.num_instances)

    def gen_online(self):
        x_batch_shape = [self.batch_size, self.num_agents, self.num_items]
        adv_batch_shape = [self.num_misreports, self.batch_size, self.num_agents, self.num_items]
        while True:
            X = self.generate_random_X(x_batch_shape)
            ADV = self.generate_random_ADV(adv_batch_shape)
            yield X, ADV, None

    def update_adv(self, idx, adv_new):
        """Updates ADV for caching"""
        self.ADV[:, idx, :, :] = adv_new

    def generate_random_X(self, shape):
        """Rewrite this for new distributions"""
        raise NotImplementedError

    def generate_random_ADV(self, shape):
        """Rewrite this for new distributions"""
        raise NotImplementedError

# if __name__ == "__main__":
    # BaseGenerator().save_data(1)