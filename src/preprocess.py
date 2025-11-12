import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

class BrainMRIDataset(Dataset):
    """
    Custom Dataset for Brain MRI (.tif) images and their segmentation masks.
    """
    def __init__(self, csv_path, base_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.base_dir = Path(base_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.base_dir / row['image']
        mask_path = self.base_dir / row['mask']

        # --- Load TIF image ---
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        # --- Load segmentation mask ---
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # --- Resize both to same dimensions ---
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256)) / 255.0

        # --- Add channel dimension (1, 256, 256) ---
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return image, mask


if __name__ == "__main__":
    base_dir = "/Users/maheenadeeb/Downloads/medical-image-analysis/data/kaggle_3m"
    csv_path = os.path.join(base_dir, "image_mask_pairs.csv")

    dataset = BrainMRIDataset(csv_path, base_dir)
    print(f"✅ Dataset loaded: {len(dataset)} samples")

    # Preview one example
    img, msk = dataset[0]
    print("Image shape:", img.shape, "| Mask shape:", msk.shape)
