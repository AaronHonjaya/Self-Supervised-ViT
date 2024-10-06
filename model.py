import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import Block
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

    def forward(self, x):
        key = query = value = x
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

        x = x + self.mha(self.norm1(x))
      
        x = x + self.mlp(self.norm2(x))


        return x


# class PositionalEncoding(nn.Module):
#     def __init__(self, embedding_dim, max_len = 6000, dropout=0.1):
#         super().__init__()

#         assert embedding_dim % 2 == 0

#         self.embedding_dim = embedding_dim
#         self.encodings = nn.Parameter(torch.randn(1, embedding_dim, 256))
        
    
#         self.drop = nn.Dropout(dropout)

#     def forward(self, x):
        
#         num_patches = x.shape[1]
        
        
#         encodings = nn.functional.interpolate(self.encodings, size=num_patches, mode='linear', align_corners=False)
        
        
#         return self.drop(x + encodings.swapaxes(1,2))
    


# class ReconstructImage(nn.Module):
#     def __init__(self, enc_dim, dec_dim, out_dim):
#         super().__init__()



       
#         self.head = nn.Sequential(
#             nn.Linear(dec_dim, dec_dim * 2), 
#             nn.ReLU(),
#             nn.Linear(dec_dim * 2, out_dim),
#             nn.Sigmoid()
#         )


#         # self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=4, stride=2, padding=1), nn.Conv2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1))

#         # self.upsample = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), nn.ConvTranspose2d(in_channels=64, out_channels=in_channels, kernel_size=4, stride=2, padding=1))

#     def forward(self, x):

#         x = self.proj(x)

#         x = self.transformer_blocks(x)

#         x = self.head(x)

#         # x = self.fold(x.swapaxes(1, 2))

#         return x






    
    
class VisionTransformer(nn.Module):
    def __init__(self, encoder_dim = 512, decoder_dim = 256, in_channels = 3, num_mask_tokens = 1, mask_precent=0.5, patch_size=8, stride=None, downstream_task: nn.Module = None):
        """Vision Transformer class. Encoder is a ViT, and Decoder is a

        Args:
            embedding_dim (int): the dimension of the embedding
            img_size (int): the size of the image. Should be a square image so only one number passed
            in_channels (int, optional): Number of input channels.
            mask_precent (float, optional): % of patches you want masked.
                    Will always round down if percentage doesn't get a whole number of patches.
                    Defaults to 0.75. Note that if there is a downstream_task, no patches will be masked.
            patch_size (int, optional): Side length of patches. Defaults to 16.
            stride (int, optional): Stride when patchifying.
                    If not given, it will automatically be set to patch_size
            downstream_task (bool, optional): The downstream task the ViT should use.
                    If not given, it means that we are pretraining, so our downstream task becomes image reconstruction.
        """
        super().__init__()

        

        self.stride = patch_size if stride is None else stride
        self.embedding_dim = encoder_dim
        self.n_mask_tokens = num_mask_tokens
        self.mask_precent = mask_precent
        self.masktokens = nn.Parameter(torch.zeros(1, self.n_mask_tokens, encoder_dim))


        # *********** #
        # Encoder
        # *********** #
        self.proj_enc = nn.Linear(patch_size * patch_size * in_channels, encoder_dim)
        
        num_patches = 64
        self.enc_embed = nn.Parameter(torch.zeros(1, 64, encoder_dim), requires_grad=False)
        enc_pos_embed = utils.get_2d_sincos_pos_embed(self.enc_embed.shape[-1], int(num_patches**.5), cls_token=False)
        self.enc_embed.data.copy_(torch.from_numpy(enc_pos_embed).float().unsqueeze(0))
        
        
        
        attn_blocks = []
        for i in range(12):
            attn_blocks.append(Block(encoder_dim, 16, 4, qkv_bias=True, norm_layer=nn.LayerNorm))
            # attn_blocks.append(SelfAttentionBlock(4 * encoder_dim, encoder_dim))

        self.enconder = nn.Sequential(*attn_blocks)
        
        self.enc_norm = nn.LayerNorm(encoder_dim)

        # *********** #
        # Decoder 
        # *********** #
        self.proj_dec = nn.Linear(self.embedding_dim, decoder_dim)

        self.dec_embed = nn.Parameter(torch.zeros(1, 64, decoder_dim), requires_grad=False)
        dec_pos_embed = utils.get_2d_sincos_pos_embed(self.dec_embed.shape[-1], int(num_patches**.5), cls_token=False)
        self.dec_embed.data.copy_(torch.from_numpy(dec_pos_embed).float().unsqueeze(0))
        
        blocks = []
        for i in range(4):
            attn_blocks.append(Block(decoder_dim, 16, 4, qkv_bias=True, norm_layer=nn.LayerNorm))
            # blocks.append(SelfAttentionBlock(4 * decoder_dim, decoder_dim))

        self.decoder = nn.Sequential(*blocks)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * in_channels, bias=True)

       

    def forward(self, x):
        # patchify and mask
        
        # x = self.patchify(x)
        # x = x.swapaxes(1, 2)
        
        num_patches = x.shape[1]

        patch_indicies = torch.randperm(num_patches, device=x.device)
        num_masked = int(num_patches * self.mask_precent)
        
        
        selected = x[:, patch_indicies[num_masked:]] # Unmasked patches
        selected = self.proj_enc(selected) #
        
        
        seq = torch.zeros(x.shape[0], x.shape[1], self.embedding_dim, dtype=x.dtype, device=x.device)
        seq[:, patch_indicies[num_masked:]] = selected
        
        # substitute in masks
        mask_token_indicies = torch.randint(0, self.n_mask_tokens, (num_masked,), device=x.device)
        seq[:, patch_indicies[ :num_masked]] = self.masktokens[:, mask_token_indicies]
        
    
        
        seq = seq + self.enc_embed

        # self attention blocks
        seq= self.enconder(seq)

        
        
        seq = self.proj_dec(seq)
        seq = seq + self.dec_embed
        seq = self.decoder(seq)
        seq = self.decoder_norm(seq)
        seq = self.decoder_pred(seq)
        
        return seq, patch_indicies[:num_masked]
