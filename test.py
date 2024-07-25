import os
from utils.data import TorchDataset
from models.encoder import TimeSeriesBertEncoder

dset_root = "./dset/Sleep-EDF-2018/npz/Fpz-Cz/"
paths = os.listdir(dset_root)
paths = [os.path.join(dset_root, x) for x in paths]

dset = TorchDataset(paths, 20, 10, 100, 100)

x, y = dset[:10]
x = x.reshape(-1, 3000)
x = x.unsqueeze(1)
encoder = TimeSeriesBertEncoder(in_channel=1, 
                                h_dim=256, 
                                vocab_size=10000,
                                beta=0.5,)

last_hidden_state, pooler_output, embedding_loss, min_encoding_idx = encoder(x)
print(pooler_output.shape)
print(last_hidden_state.shape)