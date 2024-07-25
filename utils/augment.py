import torch
import numpy as np


class AugmentMethod(object):
    """
    Augmentation method.
    """
    def __init__(self):
        self.data_augment = [self.RandomErasing,
                             self.Jittering,
                             self.Flip,
                             self.Scaling]

    def RandomErasing(self, xs, value=0, prop=0.5):
        """Randomly selects prop number of element to erase.
        Args:
            xs (tensor): input tensor, (sequence_lenth,) for unbatched data, 
                         (batch_size, sequence_length) for batched data.
            value (int): value to fill erased region.
            prop (float): proportion of erasing.
        """
        if len(xs.shape) == 1:
            sequence_length = xs.size
            idxs = np.random.choice(np.arange(sequence_length),
                                    replace=False,
                                    size=int(sequence_length * prop))
            xs[idxs] = value
            return xs
        else:
            _, sequence_length = xs.shape

            for x in xs:
                idxs = np.random.choice(np.arange(sequence_length),
                                        replace=False,
                                        size=(int(sequence_length * prop)))
                x[idxs] = value
            return xs
        
    def Jittering(self, xs, mean=0.,  std=1.):
        return xs + np.random.normal(loc=mean, scale=std, size=xs.shape)

    def Flip(self, xs):
        return torch.flip(xs, dims=[1])
    
    def Scaling(self, xs, sigma=0.1):
        n_scale = np.random.normal(loc=1, scale=sigma, size=(xs.shape[0], xs.shape[1]))
        return xs * torch.tensor(n_scale)

    def __call__(self, xs):
        """
        Args:
            xs (tensor): input tensor.
        Returns:
            tensor: augmented data.
        """
        augment = np.random.choice(self.data_augment, 1)[0]
        xs = augment(xs)
        return xs

