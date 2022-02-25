import sys
sys.path.extend(['../'])

import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from feeders import tools
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d




def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

class Feeder(Dataset):
    # OUR CONTRIBUTION

    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 personality_index,
                 data_path,
                 label_path,
                 laban_path,
                 info_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 random_shift=False,
                 normalization=False,
                 debug=False,
                 mmap=True,
                 is_train=False,):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.laban_path = laban_path
        self.info_path = info_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.random_shift = random_shift
        self.normalization = normalization
        self.window_size = window_size
        self.sample_name = []
        self.is_train = is_train
        self.personality_index = personality_index
        self.load_data(mmap)

        if self.normalization:
            self.get_mean_map()

    def load_data(self, mmap):
        # data: N C V T M

        self.label = np.load(self.label_path)

        print("Label", self.label.min(0), self.label.max(0), self.label.mean(0))
        self.label_mean = [ 0.248201, 0.17728447, -0.04173013, 0.00274649, -0.34684816]
        print("Label mean", self.label_mean)
        print("Label MSE", np.square(self.label - self.label_mean).mean(axis=0).mean())
        self.info = np.load(self.info_path)
        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        self.data = np.load(self.data_path)
        self.laban = np.load(self.laban_path)
        self.laban = self.laban[..., [0,1,2,3,4,5,6,7,8,9,10,11]] #
        self.weight = np.zeros_like(self.label)

        reweight = 'sqrt_inv'
        lds = True
        lds_kernel = 'gaussian'
        lds_ks = 5  # 5
        lds_sigma = 2  # 2


        min = None
        max = None
        laban_min = None
        laban_max = None
        if self.is_train:
            min = np.min(self.data, axis=(0, 1, 2))
            max = np.max(self.data, axis=(0, 1, 2))
            np.save("min_max.npy", np.array([min, max]))

            laban_min = np.min(self.laban, axis=(0, 1, 2))
            laban_max = np.max(self.laban, axis=(0, 1, 2))
            np.save("laban_min_max.npy", np.array([laban_min, laban_max]))

        else:
            min, max = np.load("min_max.npy")
            laban_min, laban_max = np.load("laban_min_max.npy")

        print("X", self.data.shape, self.data.min(axis=(0, 1, 2)), self.data.max(axis=(0, 1, 2)))


        self.new_data = []
        self.new_label = []
        self.new_laban = []
        self.new_info = self.info[:]
        start = 0
        for i, info in enumerate(self.info):
            count = int(info[2])
            end = start + count
            self.new_data.append(self.data[start:end].reshape((-1, *self.data.shape[2:])))
            self.new_laban.append(self.laban[start:end].reshape((-1, *self.laban.shape[2:])))
            self.new_label.append(self.label[start])
            # self.new_info[i][2] = 1
            start = end
        print(len(self.new_data), len(self.new_laban), len(self.new_label), len(self.new_info))

        # self.data = (self.data - min) / (max - min)
        self.data = np.expand_dims(self.data, 4)
        self.data = np.transpose(self.data, (0, 3, 1, 2, 4))

        # self.laban = (self.laban - laban_min) / (laban_max - laban_min)
        self.laban = np.expand_dims(self.laban, 4)
        self.laban = np.transpose(self.laban, (0, 3, 1, 2, 4))

        if self.data.shape[1] > 3:
            self.data = self.data[:, :3, :, :, :]

        # LOWER_BODY_INDICES = [0, 1, 2, 4, 5, 7, 8, 10, 11]
        # self.data[:, :, :, LOWER_BODY_INDICES, :] *= 0.001


        # self.label = self.label[:, self.personality_index : self.personality_index + 1]

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]
        print(self.data.shape, self.data.min(axis=(0, 2, 3, 4)), self.data.max(axis=(0, 2, 3, 4)))
        print(self.label.shape)
        print(self.laban.shape, self.laban.min(axis=(0, 2, 3, 4)), self.laban.max(axis=(0, 2, 3, 4)))
        print(self.info.shape)
        print("np.isnan(self.data)", np.isnan(self.data).any())
        print("np.isnan(self.label)", np.isnan(self.label).any())
        self.video_lengths = self.info[:, 2].astype(int)

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.new_data)

    def __getitem__(self, index):
        # get data
        # starting_index = self.video_lengths[:index].sum()
        # end_index = starting_index + self.video_lengths[index]
        data_numpy = self.new_data[index]
        laban_numpy = self.new_laban[index]
        label = self.new_label[index]
        weight = np.tile(self.weight[index], (5))


        # # processing
        # if self.normalization:
        #     data_numpy = (data_numpy - self.mean_map) / self.std_map
        # if self.random_shift:
        #     data_numpy = tools.random_shift(data_numpy)
        # if self.random_choose:
        #     data_numpy = tools.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        # if self.random_move:
        #     data_numpy = tools.random_move(data_numpy)
        if np.isnan(data_numpy).any():
            print("Encountered NaN in the input")


        return data_numpy, laban_numpy, label, weight
    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def top_k(self, score, top_k):
        return 0

class FeederOld(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return 50

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)


if __name__ == '__main__':
    import os
    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)
