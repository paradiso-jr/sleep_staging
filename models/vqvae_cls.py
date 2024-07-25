import torch
import numpy as np
import torch.nn as nn

from models.encoder import Encoder
from models.quantizer import VectorQuantizer

from utils.loss import info_nce_loss

class VQVAE_cls(nn.Module):
    def __init__(self,
                in_channel
                h_dim, 
                res_h_dim, 
                n_res_layers,
                n_embeddings, 
                embedding_dim, 
                beta, 
                decoder_model, 
                num_labels,
                pretrain,
                device,
                ):
        """
        Initialize the VQVAE_cls model.
        
        Args:
            h_dim (int): The hidden dimension of the encoder.
            res_h_dim (int): The hidden dimension of the residual blocks in the encoder.
            n_res_layers (int): The number of residual blocks in the encoder.
            n_embeddings (int): The number of embeddings in the Vector Quantizer.
            embedding_dim (int): The dimension of the embeddings in the Vector Quantizer.
            beta (float): The commitment loss weight for the Vector Quantizer.
            decoder_model (nn.Module): The decoder model to be used.
            num_labels (int): The number of labels for the classification task.
            pretrain (bool): Whether to pretrain the model.
            device (torch.device): The device to be used for the model.
        Returns:
            None
        """
        super(VQVAE_cls, self).__init__()
        # encode image into continuous latent space
        self.encoder = nn.Sequential(Encoder(1, h_dim, n_res_layers, res_h_dim),
                                    nn.Conv1d(h_dim, embedding_dim, kernel_size=1, stride=1))

        # pass continuous latent vector through discretization bottleneck
        # n_embeddings also represents the number of tokens
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta, device)

        # decode the discrete latent representation
        self.decoder = decoder_model.encoder

        # MLP classification layer
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(decoder_model.config.d_model, num_labels)
        self.device = device
        self.pretrain = pretrain
    def freeze_encoder(self, enable=True):
        """Freeze the encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = (False if enable else True)
    def freeze_decoer(self, enable=True):
        """Freeze the decoder."""
        for param in self.decoder.parameters():
            param.requires_grad = (False if enable else True)
        for params in self.decoder.layers[5].parameters():
            params.requires_grad = True
    
    def freeze_cls(self, enable=True):
        """Freeze the classification layer."""
        for param in self.fc.parameters():
            param.requires_grad = (False if enable else True)
    def enable_pretrain(self, enable=True):
        """Enable the pre-training mode."""
        self.pretrain = enable
    def forward(self, x, verbose=False):
        z_e = self.encoder(x)

        embedding_loss, z_q, perplexity, _, min_encoding_idx = self.vector_quantization(
            z_e)
        
        if self.pretrain:
            # pre-training the encoder and decoder
            return embedding_loss, z_q, perplexity, min_encoding_idx
        
        # add special tokens for bos and eos
        bs = min_encoding_idx.shape[0]
        # avoid index 1, 2, which is reserved for bos and eos tokens
        min_encoding_idx = min_encoding_idx + 2
        bos_tokens = torch.tensor([self.decoder.config.bos_token_id]*bs).view(bs, -1).to(min_encoding_idx.device)
        eos_tokens = torch.tensor([self.decoder.config.eos_token_id]*bs).view(bs, -1).to(min_encoding_idx.device)
        min_encoding_idx = torch.cat((bos_tokens, min_encoding_idx), dim=1)
        min_encoding_idx = torch.cat((min_encoding_idx, eos_tokens), dim=1)

        # use eos_token to get the classification embeddings
        cls_embedding = self.decoder(min_encoding_idx).last_hidden_state[:, -1, :]
        logits = self.fc(cls_embedding)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('quantization data shape:', min_encoding_idx.shape)
            print("cls embedding shape:", cls_embedding.shape)
            print('logits shape:', logits.shape)

        return embedding_loss, cls_embedding, logits, perplexity