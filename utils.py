import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def process_image_for_display(img: torch.Tensor):
    img = img.unsqueeze(0)
    img = img.moveaxis(1, 2) # (1, L, D) -> (1, D, L)
    
    img = F.fold(img, (64, 64), kernel_size=8, stride=8)
    print(img.shape)
    img = img.squeeze(0)
    if img.ndim == 3 and img.shape[0] in [1, 3]:  # Check if it's a single-channel or three-channel image
        img = img.moveaxis(0, 1).moveaxis(1, 2)  # Rearranging to (H, W, C)

    
    img = img.cpu()
    img = img.numpy()

    if np.max(img) <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def display_data(pred, target):
    pred = process_image_for_display(pred)
    target = process_image_for_display(target)

    fig, axes = plt.subplots(1,2, figsize = (10,5))
    axes[0].imshow(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB))
    axes[0].axis("off")
    axes[0].set_title("prediction")
    
    axes[1].imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
    axes[1].axis("off")
    axes[1].set_title("original")
    
    plt.tight_layout()
    plt.show()
    


def calc_psnr(pred, target):
    if torch.max(pred) <= 1.0:
        pred = pred * 255.0
    if torch.max(target) <= 1.0:
        target = target * 255.0
    
    mse = F.mse_loss(pred, target)
    
    if(mse == 0):
        return float('inf')
    
    R = torch.tensor(255**2, device=mse.device, dtype=mse.dtype)
    
    return torch.log10(R/mse) * 10


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb