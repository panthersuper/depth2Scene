import os
import numpy as np
import scipy.misc
# import h5py
np.random.seed(123)
import math
import os
from fnmatch import fnmatch
import scipy.io as spio
import json

# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):

        # self.load_size = int(kwargs['load_size'])
        # self.fine_size = int(kwargs['fine_size'])

        self.load_w = int(kwargs['load_w'])
        self.load_h = int(kwargs['load_h'])
        self.fine_w = int(kwargs['fine_w'])
        self.fine_h = int(kwargs['fine_h'])

        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.data_root = os.path.join(kwargs['data_root'])
        self.labList = json.load(open(os.path.join(self.data_root,'labels.txt'), "r"))

        # read data info from lists
        self.list_im = []
        self.list_lab = []

        labs = []

        #get data list
        for path, subdirs, files in os.walk(self.data_root):
            for name in files:
                if fnmatch(name, 'scene.txt'):
                    # print(path)

                    lab_dir = os.path.join(path,'scene.txt')

                    for path2,subdirs2,files2 in os.walk(path):
                        if "depth_bfx" in path2:
                            for name2 in files2:
                                img_dir = os.path.join(path2,name2)

                                self.list_im.append(img_dir)
                                self.list_lab.append(lab_dir)

        self.list_im = np.array(self.list_im, np.object)
        self.list_lab = np.array(self.list_lab, np.object)

        self.num = self.list_im.shape[0]

        print('# Images found:',self.num)

        # permutation
        perm = np.random.permutation(self.num)
        self.list_im[:, ...] = self.list_im[perm, ...]
        self.list_lab[:, ...] = self.list_lab[perm, ...]

        self._idx = 0

    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.fine_h, self.fine_w))
        labels_batch = np.zeros((batch_size))
        
        for i in range(batch_size):
            image = scipy.misc.imread(self.list_im[self._idx])
            image = scipy.misc.imresize(image, (self.load_h, self.load_w))
            image = image.astype(np.float32)/255.
            image = image - self.data_mean

            if self.randomize:
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    # image = image[:,::-1,:]
                    image = image[:,::-1]

                offset_h = np.random.random_integers(0, self.load_h-self.fine_h)
                offset_w = np.random.random_integers(0, self.load_w-self.fine_w)
            else:
                offset_h = math.floor((self.load_h-self.fine_h)/2)
                offset_w = math.floor((self.load_w-self.fine_w)/2)

            lab = open(self.list_lab[self._idx], "r").read()
            lab = self.labList.index(lab)

            images_batch[i, ...] = image[offset_h:offset_h+self.fine_h, offset_w:offset_w+self.fine_w]
            labels_batch[i] = lab # downsampling

            self._idx += 1
            if self._idx == self.num:
                self._idx = 0

        return images_batch, labels_batch


    def size(self):
        return self.num

    def reset(self):
        self._idx = 0
