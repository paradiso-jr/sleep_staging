# @source: https://github.com/MishaLaskin/vqvae
# @modified: wujiarong
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - vocab_size : number of embeddings
    - embedding_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, vocab_size, embedding_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, _freeze=True)
        self.embedding.weight.data.uniform_(-1.0 / self.vocab_size, 1.0 / self.vocab_size)
        # self.embedding.weight.data.uniform_(-1.0, 1.0)
    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, hidden_dim, squence_length)

        quantization pipeline:

            1. get encoder input (batch, hidden_dim, squence_length)
            2. flatten input to (BS*sequence_length, hidden_dim)

        """
        # reshape z -> (batch_size, z, sequence_length) and flatten
        z = z.permute(0, 2, 1).contiguous()
        # the embedding_dim should equal to hidden_dim
        z_flattened = z.view(-1, self.embedding_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.vocab_size).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # unflatten min_encoding_indices to (batch_size, sequence_length)
        min_encoding_indices = min_encoding_indices.view(z.shape[0], -1)
        
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        # z_q = z_q.contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices