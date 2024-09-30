import os
import numpy as np
from prefetch_generator import BackgroundGenerator
import torch
from tqdm import tqdm
from DataSet import DataSet300W
from torch.utils.data import random_split, DataLoader
from torch.nn.functional import unfold, mse_loss
import utils

from model import VisionTransformer


def train(model, device, n_epochs):
    model.to(device)

    trainset = DataSet300W("./Face Detection/Data/lfpw/trainset_npy", apply_augment=False)
    trainset, valset = random_split(trainset, [0.8, 0.2])

    train_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=4, shuffle=False, num_workers=4)

    optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

    mse = torch.nn.MSELoss()
    losses = []
    best_val_loss = float("inf")
    for epoch in range(0, n_epochs):
        model.train()

        print(f"\n{epoch}.)")
        total_loss = 0
        pbar = tqdm(enumerate(BackgroundGenerator(train_loader)), total=len(train_loader))
        for itr, (train_images, _points) in pbar:
            train_images = train_images.to(device=device, dtype=torch.float32)

            pred = model(train_images)
            loss = mse(pred, train_images)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        losses.append(avg_train_loss)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            torch.save(model.state_dict(), os.path.join("./Face Detection/self-supervised/logs/checkpoints", str(epoch)))

        with torch.no_grad():
            model.eval()
            pbar = tqdm(enumerate(BackgroundGenerator(val_loader)), total=len(val_loader))
            total_loss = 0
            for itr, (val_images, _points) in pbar:
                val_images = val_images.to(device=device, dtype=torch.float32)

                pred = model(val_images)
                val_mse = mse(pred, val_images)

                total_loss += val_mse

            avg_val_loss = total_loss / len(val_loader)

            print(f"train loss = {avg_train_loss}  |  val loss = {avg_val_loss}")
            if best_val_loss > avg_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join("./self-supervised/logs/best_models", "best_model"))


if __name__ == "__main__":
    model = VisionTransformer(16 * 16 * 3, 224, 3)
    train(model, "cuda", 100)
