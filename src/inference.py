import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from pathlib import Path

from model import UNet
from preprocess import BrainMRIDataset

# ────────────────────────────────
# Helper functions
# ────────────────────────────────
def save_prediction_images(image, mask, pred, save_prefix):
    """Save input, ground truth, and prediction as PNGs."""
    image = (image.squeeze() * 255).astype(np.uint8)
    mask = (mask.squeeze() * 255).astype(np.uint8)
    pred = (pred.squeeze() * 255).astype(np.uint8)

    cv2.imwrite(f"{save_prefix}_input.png", image)
    cv2.imwrite(f"{save_prefix}_mask.png", mask)
    cv2.imwrite(f"{save_prefix}_pred.png", pred)

def dice_coeff_numpy(pred, target, smooth=1e-6):
    pred = (pred > 0.5).astype(np.float32)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# ────────────────────────────────
# Inference pipeline
# ────────────────────────────────
def run_inference(save_outputs=True, num_samples=5):
    base_dir = "/Users/maheenadeeb/Downloads/medical-image-analysis/data/kaggle_3m"
    csv_path = os.path.join(base_dir, "image_mask_pairs.csv")
    checkpoint_path = "checkpoints/unet_best.pth"

    # Prepare output folder
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Load dataset
    dataset = BrainMRIDataset(csv_path, base_dir)
    print(f"✅ Loaded dataset with {len(dataset)} samples")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    # Load model
    model = UNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Pick random samples
    samples = random.sample(range(len(dataset)), num_samples)
    results = []

    for idx in samples:
        image, mask = dataset[idx]
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(image_tensor)
        pred = pred.cpu().numpy().squeeze()

        dice = dice_coeff_numpy(pred, mask.squeeze())
        img_name = Path(dataset.df.iloc[idx]['image']).stem

        # Save results
        save_prefix = output_dir / img_name
        save_prediction_images(image, mask, pred, str(save_prefix))
        results.append({"image": img_name, "dice_score": dice})

        print(f"🩻 Saved {img_name}  |  Dice: {dice:.4f}")

    # Save summary CSV
    summary_csv = output_dir / "summary.csv"
    pd.DataFrame(results).to_csv(summary_csv, index=False)
    print(f"\n📁 Predictions saved in: {output_dir}")
    print(f"📊 Summary CSV saved as: {summary_csv}")

    # Optional: display one overlay
    sample_idx = samples[0]
    image, mask = dataset[sample_idx]
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(image_tensor)
    pred = (pred.cpu().numpy() > 0.5).astype(np.float32)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(image.squeeze(), cmap="gray"); plt.title("MRI Image"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(mask.squeeze(), cmap="gray"); plt.title("Ground Truth"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(image.squeeze(), cmap="gray"); plt.imshow(pred.squeeze(), alpha=0.5, cmap="Reds")
    plt.title("Prediction Overlay"); plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_inference()
