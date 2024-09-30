import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import utils


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, embedding_dim, heads, dropout=0.1):
        super().__init__()

        assert embedding_dim % heads == 0

        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

        self.drop = nn.Dropout(dropout)

        self.num_heads = heads

    def forward(self, key, query, value):

        # because it should be self attention
        assert key.shape == query.shape and key.shape == value.shape

        # N = batchsize, S = sequence length (num patches), E = embed_dim
        N, S, E = key.shape
        H = self.num_heads

        # Get projected key, query, and value in shape (N, H, S, E//H)
        K = self.key(key).view(N, S, H, E // H).moveaxis(1, 2)
        Q = self.query(query).view(N, S, H, E // H).moveaxis(1, 2)
        V = self.value(value).view(N, S, H, E // H).moveaxis(1, 2)

        scale = math.sqrt(E / H)

        #   (N, H, S, E//H) @ (N,H, E//H, S) =>  (N,H,S,S)
        Y = Q @ K.transpose(2, 3) / scale

        #   (N,H,S,S) @ (N,H,S, E//H) ==> (N,H,S,E//h)
        Y = self.drop(F.softmax(Y, dim=-1)) @ V

        Y = Y.swapaxes(1, 2).reshape(N, S, E)

        Y = self.output_proj(Y)

        return Y


class SelfAttentionBlock(nn.Module):
    def __init__(self, mlp_hidden_dim, embedding_dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.mha = MultiHeadedSelfAttention(embedding_dim, heads=4)

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(nn.Linear(embedding_dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, embedding_dim))

    def forward(self, x):
        x_residual = x

        x = self.norm1(x)
        x = self.mha(x, x, x)

        x += x_residual
        x_residual = x

        x = self.norm2(x)
        x = self.mlp(x)

        x += x_residual

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embedding_dim, dropout=0.1):
        super().__init__()

        assert embedding_dim % 2 == 0

        pe = torch.zeros(1, num_patches, embedding_dim)

        i = torch.arange(num_patches).unsqueeze(1)
        pows = torch.pow(10000, -torch.arange(0, embedding_dim, 2) / embedding_dim)

        pe[0, :, 0::2] = torch.sin(i * pows)
        pe[0, :, 1::2] = torch.cos(i * pows)

        self.drop = nn.Dropout(dropout)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.drop(x + self.pe)


class ReconstructImage(nn.Module):
    def __init__(self, enc_dim, dec_dim, img_size, patch_size, stride):
        super().__init__()

        # self.patch_size = patch_size
        # self.stride = stride
        # self.in_channels = in_channels
        # self.num_patches = ((224 - patch_size) // stride + 1) ** 2

        self.proj = nn.Linear(enc_dim, dec_dim)

        encoders = []
        for i in range(4):
            encoders.append(SelfAttentionBlock(4 * dec_dim, dec_dim))

        self.self_attn = nn.Sequential(*encoders)

        self.head = nn.Sequential(nn.Linear(dec_dim, enc_dim // 2), nn.Linear(enc_dim // 2, enc_dim))

        self.fold = nn.Fold((img_size, img_size), kernel_size=patch_size, stride=stride)

        # self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=4, stride=2, padding=1), nn.Conv2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1))

        # self.upsample = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), nn.ConvTranspose2d(in_channels=64, out_channels=in_channels, kernel_size=4, stride=2, padding=1))

    def forward(self, x):

        x = self.proj(x)

        x = self.self_attn(x)

        x = self.head(x)

        x = self.fold(x.swapaxes(1, 2))

        return x


class VisionTransformer(nn.Module):
    def __init__(self, embedding_dim, img_size, in_channels, masked_patches=0.75, patch_size=16, stride=None, downstream_task: nn.Module = None):
        """Vision Transformer class. Encoder is a ViT, and Decoder is a

        Args:
            embedding_dim (int): the dimension of the embedding
            img_size (int): the size of the image. Should be a square image so only one number passed
            in_channels (int, optional): Number of input channels.
            masked_patches (float, optional): % of patches you want masked.
                    Will always round down if percentage doesn't get a whole number of patches.
                    Defaults to 0.75. Note that if there is a downstream_task, no patches will be masked.
            patch_size (int, optional): Side length of patches. Defaults to 16.
            stride (int, optional): Stride when patchifying.
                    If not given, it will automatically be set to patch_size
            downstream_task (bool, optional): The downstream task the ViT should use.
                    If not given, it means that we are pretraining, so our downstream task becomes image reconstruction.
        """
        super().__init__()

        total_patches = (img_size // patch_size) ** 2

        self.stride = patch_size if stride is None else stride

        self.n_masked_patches = int(masked_patches * masked_patches)
        self.patchify = nn.Unfold(kernel_size=patch_size, stride=self.stride)
        self.masktokens = nn.Parameter(torch.zeros(1, self.n_masked_patches, in_channels * patch_size * patch_size))

        self.proj = nn.Linear(patch_size * patch_size * in_channels, embedding_dim)
        self.positional_encode = PositionalEncoding(total_patches, embedding_dim)

        attn_blocks = []
        for i in range(12):
            attn_blocks.append(SelfAttentionBlock(4 * embedding_dim, embedding_dim))

        self.enconder = nn.Sequential(*attn_blocks)

        if downstream_task is None:
            self.decoder = ReconstructImage(embedding_dim, embedding_dim // 4, img_size, patch_size, self.stride)
        else:
            self.decoder = downstream_task

    def forward(self, x):
        # patchify and mask
        x = self.patchify(x)
        x = x.swapaxes(1, 2)
        patch_indicies = np.random.permutation(range(196))[: self.n_masked_patches]
        x[:, patch_indicies] = self.masktokens.repeat(x.shape[0], 1, 1)

        # project and encode position
        x = nn.Flatten(start_dim=2, end_dim=-1)(x)
        x = self.proj(x)
        x = self.positional_encode(x)

        # self attention blocks
        x = self.enconder(x)

        # mlp_head
        x = self.decoder(x)

        return x
