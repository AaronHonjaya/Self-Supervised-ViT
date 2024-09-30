import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224

NUM_POINTS = 68

batch_size = 64


def display_data(img: torch.Tensor, pts=None):

    if img.ndim == 3 and img.shape[0] in [1, 3]:  # Check if it's a single-channel or three-channel image
        img = img.moveaxis(0, 1).moveaxis(1, 2)  # Rearranging to (H, W, C)

    img = img.cpu()
    img = img.numpy()

    if np.max(img) <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if pts is not None:
        pts *= img.shape[0]
        for x, y in pts:
            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()
