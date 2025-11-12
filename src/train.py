import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from preprocess import BrainMRIDataset
from model import UNet

# ────────────────────────────────
# Helper: Dice Coefficient + Loss
# ────────────────────────────────
def dice_coeff(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    return ((2. * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)).mean()

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, preds, targets):
        dice = dice_coeff(preds, targets)
        bce = self.bce(preds, targets)
        return 1 - dice + bce

# ────────────────────────────────
# Training Loop
# ────────────────────────────────
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = torch.tensor(images, dtype=torch.float32).to(device)
        masks = torch.tensor(masks, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    dice_score = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating", leave=False):
            images = torch.tensor(images, dtype=torch.float32).to(device)
            masks = torch.tensor(masks, dtype=torch.float32).to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()
            dice_score += dice_coeff(outputs, masks).item()
    return val_loss / len(loader), dice_score / len(loader)

# ────────────────────────────────
# Main Training
# ────────────────────────────────
def main():
    base_dir = "/Users/maheenadeeb/Downloads/medical-image-analysis/data/kaggle_3m"
    csv_path = os.path.join(base_dir, "image_mask_pairs.csv")

    # Dataset & split
    dataset = BrainMRIDataset(csv_path, base_dir)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    # Model, loss, optimizer
    model = UNet(n_channels=1, n_classes=1).to(device)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    best_dice = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice = validate_one_epoch(model, val_loader, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), f"checkpoints/unet_best.pth")
            print(f"💾 Saved new best model with Dice: {best_dice:.4f}")

    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
