import mne
import torch
import numpy as np

from torch.utils.data import Dataset

class TorchDataset(Dataset):
    def __init__(self, paths, temporal_context_length, window_size,
                 sfreq: int, rfreq: int):
        super().__init__()
        self.sfreq, self.rfreq = sfreq, rfreq
        self.info = mne.create_info(sfreq=sfreq, ch_types='eeg', ch_names=['Fp1'])
        self.scaler = mne.decoding.Scaler(self.info, scalings='median')
        self.x, self.y = self.get_data(paths,
                                       temporal_context_length=temporal_context_length,
                                       window_size=window_size)
        self.x, self.y = torch.tensor(self.x, dtype=torch.float32), torch.tensor(self.y, dtype=torch.long)

    def get_data(self, paths, temporal_context_length, window_size):
        total_x, total_y = [], []
        for path in paths:
            data = np.load(path)
            x, y = data['x'], data['y']
            x = np.expand_dims(x, axis=1)
            x = mne.EpochsArray(x, info=self.info)
            # band pass
            x = x.filter(0.5, 40)
            x = x.resample(self.rfreq)
            x = x.get_data(copy=True)
            # norm
            x = self.scaler.fit_transform(x)
            x = x.squeeze()
            x = self.many_to_many(x, temporal_context_length, window_size)
            y = self.many_to_many(y, temporal_context_length, window_size)
            total_x.append(x)
            total_y.append(y)
        total_x, total_y = np.concatenate(total_x), np.concatenate(total_y)
        return total_x, total_y

    @staticmethod
    def many_to_many(elements, temporal_context_length, window_size):
        size = len(elements)
        total = []
        if size <= temporal_context_length:
            return elements
        for i in range(0, size-temporal_context_length+1, window_size):
            temp = np.array(elements[i:i+temporal_context_length])
            total.append(temp)
        total.append(elements[size-temporal_context_length:size])
        total = np.array(total)
        return total

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x = self.x[item].clone().detach().requires_grad_(False)
        y = self.y[item].clone().detach().requires_grad_(False)
        return x, y
