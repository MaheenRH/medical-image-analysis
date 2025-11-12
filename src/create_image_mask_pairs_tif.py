import os
import pandas as pd
from pathlib import Path

def create_mapping(base_dir):
    base_path = Path(base_dir)
    records = []

    # Each patient folder (e.g. TCGA_CS_6667_20011105)
    for patient_folder in sorted(base_path.glob("*")):
        if not patient_folder.is_dir():
            continue

        # All .tif files inside (excluding mask files for now)
        for img_file in patient_folder.glob("*.tif"):
            if "_mask" in img_file.name:
                continue  # skip masks

            # Try to find corresponding mask
            mask_name = img_file.stem + "_mask.tif"
            mask_path = patient_folder / mask_name

            if mask_path.exists():
                records.append({
                    "image": str(img_file.relative_to(base_path)),
                    "mask": str(mask_path.relative_to(base_path))
                })

    # Save as CSV
    df = pd.DataFrame(records)
    csv_out = base_path / "image_mask_pairs.csv"
    df.to_csv(csv_out, index=False)
    print(f"✅ Created CSV with {len(df)} image-mask pairs at {csv_out}")

if __name__ == "__main__":
    base_dir = "/Users/maheenadeeb/Downloads/medical-image-analysis/data/kaggle_3m"
    create_mapping(base_dir)
