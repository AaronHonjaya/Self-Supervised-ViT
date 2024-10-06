import os
import numpy as np
from prefetch_generator import BackgroundGenerator
import torch

from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from torch.nn.functional import unfold, mse_loss
from dataset import TrainDatasetImageNet
import utils

from torchvision import datasets, transforms

# Define your transformations



from model import VisionTransformer



def train(model, train_loader, val_loader, logs_path, device, n_epochs):
    model.to(device)

    # trainset = TrainDatasetImageNet("./tiny-imagenet-200/train", 8)
    # trainset, valset = random_split(trainset, [0.85, 0.15])
    
    # train_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
    # val_loader = DataLoader(valset, batch_size=16, shuffle=False, num_workers=4)

    # optim = torch.optim.AdamW(model.parameters(), lr=0.00025, betas=(0.9, 0.95))
    

    mse = torch.nn.MSELoss()
    train_losses = []
    train_psnrs = []
    val_psnrs = []
    best_val_psnr = -1
    for epoch in range(0, n_epochs):
        model.train()

        print(f"\n{epoch}.)")
        
        total_loss = 0
        total_psnr = 0
        pbar = tqdm(enumerate(BackgroundGenerator(train_loader)), total=len(train_loader))
        for itr, (train_images, _label) in pbar:
            train_images = train_images.to(device=device, dtype=torch.float32)
            
            pred, indicies = model(train_images)
            loss = mse(pred[:, indicies] * 255.0, train_images[:, indicies] *255.0)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            
            train_psnr = utils.calc_psnr(pred[: , indicies], train_images[:, indicies])
            total_psnr += train_psnr.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_train_psnr = total_psnr / len(train_loader)
        train_psnrs.append(avg_train_psnr)
            
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            torch.save(model.state_dict(), os.path.join(logs_path, "checkpoints", f'{epoch}.pth'))

        with torch.no_grad():
            model.eval()
            pbar = tqdm(enumerate(BackgroundGenerator(val_loader)), total=len(val_loader))

            total_psnr = 0
            for itr, (val_images, _label) in pbar:
                val_images = val_images.to(device=device, dtype=torch.float32)

                pred, indicies = model(val_images)
                val_psnr = utils.calc_psnr(pred[: , indicies], val_images[:, indicies])
                total_psnr += val_psnr.item()

            avg_val_psnr = total_psnr / len(val_loader)
            val_psnrs.append(avg_val_psnr)
            print(f"train loss = {avg_train_loss}  | train psnr = {avg_train_psnr} | val psnr = {avg_val_psnr}")
            if best_val_psnr < avg_val_psnr:
                best_val_psnr = avg_val_psnr
                torch.save(model.state_dict(), os.path.join(logs_path, "best_models", "best_model.pth"))
        
    train_losses = np.array(train_losses, dtype=np.float32)
    val_psnrs = np.array(val_psnrs, dtype=np.float32)
    train_psnrs = np.array(train_psnrs, dtype=np.float32)
    losses_path = os.path.join(logs_path, "losses")
    np.save(os.path.join(losses_path, "train_losses"), train_losses)
    np.save(os.path.join(losses_path, "train_psnrs"), train_psnrs)
    np.save(os.path.join(losses_path, "val_psnrs"), val_psnrs)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(256),            # Resize to 256x256
        transforms.CenterCrop(224),       # Center crop to 224x224
        transforms.ToTensor(),             # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    coco_full = datasets
    model = VisionTransformer()
    train(model, "./logs", "cuda", 50)
