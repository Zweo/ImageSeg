import os
import torch
import numpy as np
import cv2 as cv
from os.path import join
from torch.utils.data.dataset import Dataset


class DataTrain(Dataset):
    def __init__(self, data_path):
        self.data_dir = data_path
        self.image_list = os.listdir(join(data_path, 'images'))
        files_len = len(self.image_list)
        try:
            imgs = np.zeros(shape=(files_len, 256, 256, 3), dtype=np.uint8)
            labels = np.zeros(shape=(files_len, 256, 256), dtype=np.uint8)
            for idx, file in enumerate(self.image_list):
                fname = file.split('.')[0]
                img = cv.imread(join(self.data_dir, 'images', fname + '.tif'))
                img = np.asarray(img, dtype=np.uint8)
                label = cv.imread(
                    join(self.data_dir, 'labels', fname + '.png'),
                    cv.IMREAD_UNCHANGED)
                label = np.asarray(label, dtype=np.uint8) % 100
                imgs[idx, :, :, :] = img
                labels[idx, :, :] = label
            self.images = np.transpose(imgs, (0, 3, 1, 2))
            self.labels = labels
        except Exception:
            raise Exception('read error')

    def __getitem__(self, index):
        img = torch.from_numpy(self.images[index]).float()
        return img, self.labels[index]

    def __len__(self):
        return len(self.image_list)
