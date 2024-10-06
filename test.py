from prefetch_generator import BackgroundGenerator
import torch
from torch.utils.data import DataLoader
from dataset import TrainDatasetImageNet
from model import VisionTransformer
from utils import display_data

def test():
    trainset = TrainDatasetImageNet("./tiny-imagenet-200/train", 8)
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)

    device = "cuda"
    model = VisionTransformer(256, in_channels=3, patch_size=8)

    model.load_state_dict(torch.load("./logs/best_models/best_model.pth"))
    
    model.to(device=device)
    model.eval()
    with torch.no_grad():
        for img, _ in train_loader:
            img = img.to(device)
            pred, indicies = model(img)
            display_data(pred[0], img[0])
        
if __name__ == "__main__":
    test()