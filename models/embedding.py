# @source: https://github.com/MishaLaskin/vqvae
# @modified: wujiarong
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.residual import ResidualStack
from models.quantizer import VectorQuantizer

class TimeSeriesCnnEmbedding(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ-VAE, q_theta outputs parameters of a categorical distribution.

    Args:
    - in_channel(int) : the input dimension
    - h_dim(int) : the dimension of hidden conv layers
    - vocab_size(int) : the number of embedding vectors
    - embedding_dim(int) : the dimension of each embedding vector
    - beta(float) : the temperature parameter for softmax
    """

    def __init__(self, 
                in_channel,
                h_dim,
                vocab_size, 
                embedding_dim, 
                beta):
        super(TimeSeriesCnnEmbedding, self).__init__()
        
        self.inplanes3 = 16
        self.inplanes5 = 16
        self.inplanes7 = 16
        self.seq_len = 3000 >> 3
        self.out_channel = 32
        self.layers = [1, 1, 1, 1]
        self.conv1 = nn.Conv1d(in_channel, 16, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.SiLU()


        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 16, self.layers[0], stride=1)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 16, self.layers[1], stride=1)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 24, self.layers[2], stride=2)
        self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 32, self.layers[3], stride=2)
        self.avgpool3 = nn.AvgPool1d(kernel_size=2)
        
        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 16, self.layers[0], stride=1)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 16, self.layers[1], stride=1)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 24, self.layers[2], stride=2)
        self.layer5x5_4 = self._make_layer5(BasicBlock5x5, 32, self.layers[3], stride=2)
        self.avgpool5 = nn.AvgPool1d(kernel_size=2)
        
        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 16, self.layers[0], stride=1)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 16, self.layers[1], stride=1)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 24, self.layers[2], stride=2)
        self.layer7x7_4 = self._make_layer7(BasicBlock7x7, 32, self.layers[3], stride=2)
        self.avgpool7 = nn.AvgPool1d(kernel_size=2)
        
        #self.linear_out = nn.Linear(h_dim*self.seq_len*1, h_dim*self.seq_len)

        self.quantizer = VectorQuantizer(vocab_size, 
                                        embedding_dim, 
                                        beta)
    def forward(self, x0):
        b = x0.shape[0]
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.layer3x3_1(x0)
        x1 = self.layer3x3_2(x1)
        x1 = self.layer3x3_3(x1)
        x1 = self.layer3x3_4(x1)
        x1 = self.avgpool3(x1)
        
        x2 = self.layer5x5_1(x0)
        x2 = self.layer5x5_2(x2)
        x2 = self.layer5x5_3(x2)
        x2 = self.layer5x5_4(x2)
        x2 = self.avgpool5(x2)
        
        x3 = self.layer7x7_1(x0)
        x3 = self.layer7x7_2(x3)
        x3 = self.layer7x7_3(x3)
        x3 = self.layer7x7_4(x3)
        x3 = self.avgpool7(x3)
        
        x = torch.stack([x1, x2, x3], dim=-1)
        x = torch.mean(x, dim=-1)

        x = x.reshape(b, self.out_channel, self.seq_len)
        embedding_loss, z_q, perplexity, min_encodings, min_encoding_idx = self.quantizer(x)
        
        return embedding_loss, z_q, perplexity, min_encodings, min_encoding_idx

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)

    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                    padding=2, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                    padding=3, bias=False)


class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.SiLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.SiLU()
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        #d = residual.shape[2] - out.shape[2]
        #out1 = residual[:, :, 0:-d] + out
        #out1 = self.relu(out1)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.SiLU()
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        #d = residual.shape[2] - out.shape[2]
        #out1 = residual[:, :, 0:-d] + out
        #out1 = self.relu(out1)
        out += residual
        out = self.relu(out)
        return out