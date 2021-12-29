from torch.utils.data import Dataset
import torch
import os
import numpy as np

class CARDataset(Dataset):

    def __init__(self, dir, point_num, data_mode='spher', transform=None):

        super(CTDataset, self).__init__()

        self.data_dir = dir
        self.point_num = point_num
        self.data_mode = data_mode

        self.parameter_dir = self.data_dir + '/parameter'

        self.moving_dir = self.data_dir + '/moving_' + self.data_mode
        self.target_dir = self.data_dir + '/target_' + self.data_mode
        self.moving_start_dir = self.data_dir + '/moving_start'
        self.target_start_dir = self.data_dir + '/target_start'

        self.moving_mean = np.load(self.parameter_dir + '/moving_' + self.data_mode + '_mean.npy').squeeze()
        self.moving_std = np.load(self.parameter_dir + '/moving_' + self.data_mode + '_std.npy').squeeze()
        self.moving_start_mean = np.load(self.parameter_dir + '/moving_start_mean.npy').squeeze()
        self.moving_start_std = np.load(self.parameter_dir + '/moving_start_std.npy').squeeze()

        self.target_mean = np.load(self.parameter_dir + '/target_' + self.data_mode + '_mean.npy').squeeze()
        self.target_std = np.load(self.parameter_dir + '/target_' + self.data_mode + '_std.npy').squeeze()
        self.target_start_mean = np.load(self.parameter_dir + '/target_start_mean.npy').squeeze()
        self.target_start_std = np.load(self.parameter_dir + '/target_start_std.npy').squeeze()

        self.transform = transform

        self.sample_list = os.listdir(self.moving_dir)

    def __len__(self):

        return len(self.sample_list)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        moving = np.load(self.moving_dir + '/' + self.sample_list[idx])
        target = np.load(self.target_dir + '/' + self.sample_list[idx])
        moving_start = np.load(self.moving_start_dir + '/' + self.sample_list[idx])
        target_start = np.load(self.target_start_dir + '/' + self.sample_list[idx])

        sample_mean = np.concatenate((self.moving_mean, self.target_mean))
        sample_std = np.concatenate((self.moving_std, self.target_std))
        start_mean = np.concatenate((self.moving_start_mean, self.target_start_mean))
        start_std = np.concatenate((self.moving_start_std, self.target_start_std))

        sample = np.concatenate((moving, target), axis=-1)
        sample_norm = (sample - sample_mean) / sample_std
        start = np.concatenate((moving_start, target_start), axis=-1)
        start_norm = (start - start_mean) / start_std

        sample = {'sample': sample,
                  'sample_norm': sample_norm,
                  'start': start,
                  'start_norm': start_norm,
                  'sample_mean': sample_mean,
                  'sample_std': sample_std,
                  'start_mean': start_mean,
                  'start_std': start_std}

        return sample


class ToTensor(object):

    def __call__(self, sample):
#
        return {'sample': torch.from_numpy(sample['sample'].copy()),
                'sample_mean': torch.from_numpy(sample['sample_mean'].copy()),
                'sample_std': torch.from_numpy(sample['sample_std'].copy()),
                'sample_norm': torch.from_numpy(sample['sample_norm'].copy()),
                'start': torch.from_numpy(sample['start'].copy()),
                'start_mean': torch.from_numpy(sample['start_mean'].copy()),
                'start_std': torch.from_numpy(sample['start_std'].copy()),
                'start_norm': torch.from_numpy(sample['start_norm'].copy())}




