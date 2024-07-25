import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embedding import TimeSeriesCnnEmbedding
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BertModel, AutoModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 3000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: torch.Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TimeSeriesEncoder(nn.Module):
    """
    TimeSeriesEncoder is the encoder of the Transformer model.
    Args:
        in_channel (int): The number of input channels.
        conv_h_dim (int): The hidden dimension of the convolutional layer.
        n_res_layers (int): The number of residual layers.
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the embedding layer.
        beta (float): The hyperparameter beta.
        d_model (int): The dimension of the Transformer model.
        nhead (int): The number of heads in the Transformer model.
        d_hid (int): The hidden dimension of the Transformer model.
        nlayers (int): The number of encoder layers in the Transformer model.
        dropout (float, optional): The dropout rate. Defaults to 0.5.
        """
    def __init__(self,
                in_channel: int, 
                conv_h_dim: int, 
                n_res_layers: int, 
                vocab_size: int, 
                embedding_dim: int, 
                beta: float,     
                d_model: int, 
                nhead: int, 
                d_hid: int,
                nlayers: int, 
                dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.embedding = TimeSeriesRnnEmbedding(in_channel=in_channel,
                                            h_dim=conv_h_dim,
                                            n_res_layers=n_res_layers,
                                            vocab_size=vocab_size,
                                            embedding_dim=embedding_dim,
                                            beta=beta)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        model = BertModel.from_pretrained("bert-base-uncased")

        self.in_channel = in_channel
        self.conv_h_dim = conv_h_dim
        self.n_res_layers = n_res_layers
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.d_model = d_model
    
    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Arguments:
            src: torch.Tensor, shape ``[seq_len, batch_size, 1]``
            src_mask: torch.Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output torch.Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        embedding_loss, src, perplexity, min_encodings, min_encoding_idx = self.embedding(src)
        
        # add special tokens for bos and eos
        bs = min_encoding_idx.shape[0]
        # avoid index 1, 2, which is reserved for bos and eos tokens
        min_encoding_idx = min_encoding_idx + 2
        bos_tokens = torch.tensor([self.decoder.config.bos_token_id]*bs).view(bs, -1).to(min_encoding_idx.device)
        eos_tokens = torch.tensor([self.decoder.config.eos_token_id]*bs).view(bs, -1).to(min_encoding_idx.device)
        min_encoding_idx = torch.cat((bos_tokens, min_encoding_idx), dim=1)
        min_encoding_idx = torch.cat((min_encoding_idx, eos_tokens), dim=1)

        return output, embedding_loss, perplexity, min_encodings, min_encoding_idx


class TimeSeriesBertEncoder(nn.Module):
    """
    TimeSeriesEncoder is the encoder of the Transformer model.
    Args:
        in_channel (int): The number of input channels.
        h_dim (int): The hidden dimension of the RNN layer.
        n_res_layers (int): The number of residual layers.
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the embedding layer.
        beta (float): The hyperparameter beta.

        """
    def __init__(self,
                in_channel: int, 
                h_dim: int, 
                vocab_size: int, 
                beta: float,):
        super().__init__()
        self.model_type = 'Bert'
        self.embedding = TimeSeriesCnnEmbedding(in_channel=in_channel,
                                            h_dim=h_dim,
                                            vocab_size=vocab_size,
                                            embedding_dim=h_dim,
                                            beta=beta)

        self.model = BertModel.from_pretrained("bert-base-uncased")
        # try bert tiny
        # self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        self.cls_token_id = 101
        self.sep_token_id = 102

        self.in_channel = in_channel
        self.h_dim = h_dim
        self.vocab_size = vocab_size
        self.beta = beta

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            src: torch.Tensor, shape ``[seq_len, batch_size, 1]``

        Returns:
            output
        """

        embedding_loss, _, _, _, min_encoding_idx = self.embedding(src)

        bs = min_encoding_idx.shape[0]
        # avoid special_token_id, which is reserved for bos and eos tokens
        min_encoding_idx = min_encoding_idx + 200
        cls_tokens = torch.tensor([self.cls_token_id]*bs).view(bs, -1).to(min_encoding_idx.device)
        sep_tokens = torch.tensor([self.sep_token_id]*bs).view(bs, -1).to(min_encoding_idx.device)
        min_encoding_idx = torch.cat((cls_tokens, min_encoding_idx), dim=1)
        min_encoding_idx = torch.cat((min_encoding_idx, sep_tokens), dim=1)

        output = self.model(min_encoding_idx)
        return output.last_hidden_state, output.pooler_output, embedding_loss, min_encoding_idx