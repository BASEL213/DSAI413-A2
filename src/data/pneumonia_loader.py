"""
Loader for the Kaggle Chest X-Ray Pneumonia dataset.
Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Folder structure expected:
    <images_dir>/
        train/
            NORMAL/    (1,341 images)
            PNEUMONIA/ (3,875 images)
        val/
            NORMAL/    (8 images)
            PNEUMONIA/ (8 images)
        test/
            NORMAL/    (234 images)
            PNEUMONIA/ (390 images)

Since this dataset provides classification labels but no free-text radiology
reports, synthetic impressions are generated from the class label so the
downstream RAG retrieval and generation components work unchanged.
"""
import os
import glob
import pandas as pd
from pathlib import Path


class PneumoniaLoader:
    """Loads the Chest X-Ray Pneumonia dataset and produces a DataFrame
    compatible with the rest of the CXR RAG pipeline."""

    # Synthetic impressions keyed by image subtype
    IMPRESSIONS = {
        "NORMAL": (
            "No acute cardiopulmonary disease. The lungs are clear bilaterally. "
            "No pleural effusion or pneumothorax identified. "
            "Cardiac silhouette within normal limits."
        ),
        "PNEUMONIA_BACTERIAL": (
            "Focal airspace consolidation consistent with bacterial pneumonia. "
            "Increased opacity identified in the affected lung region. "
            "No significant pleural effusion. Cardiac silhouette within normal limits."
        ),
        "PNEUMONIA_VIRAL": (
            "Bilateral interstitial infiltrates and peribronchial thickening "
            "consistent with viral pneumonia. No focal consolidation. "
            "No pleural effusion or pneumothorax. Cardiac silhouette within normal limits."
        ),
    }

    FINDINGS = {
        "NORMAL": (
            "Lungs: Clear bilaterally. No focal opacity, consolidation, or atelectasis. "
            "Heart and Mediastinum: Normal cardiac silhouette, mediastinum not widened. "
            "Pleura: No pleural effusion or pneumothorax. "
            "Bones and Soft Tissues: No acute osseous abnormality."
        ),
        "PNEUMONIA_BACTERIAL": (
            "Lungs: Focal airspace consolidation in the affected lobe, consistent "
            "with bacterial pneumonia. No contralateral involvement. "
            "Heart and Mediastinum: Cardiac silhouette within normal limits. "
            "Pleura: No significant pleural effusion or pneumothorax. "
            "Bones and Soft Tissues: No acute osseous abnormality."
        ),
        "PNEUMONIA_VIRAL": (
            "Lungs: Bilateral interstitial infiltrates with peribronchial thickening, "
            "consistent with viral pneumonia. No focal consolidation. "
            "Heart and Mediastinum: Cardiac silhouette within normal limits. "
            "Pleura: No pleural effusion or pneumothorax. "
            "Bones and Soft Tissues: No acute osseous abnormality."
        ),
    }

    def __init__(self, images_dir: str):
        """
        images_dir: path to the root chest_xray folder (contains train/, val/, test/).
        On Kaggle: /kaggle/input/chest-xray-pneumonia/chest_xray
        """
        self.images_dir = images_dir

    def _subtype(self, label: str, filename: str) -> str:
        """Determine subtype from label and filename."""
        if label == "NORMAL":
            return "NORMAL"
        name = Path(filename).stem.upper()
        if "BACTERIA" in name:
            return "PNEUMONIA_BACTERIAL"
        return "PNEUMONIA_VIRAL"

    def load(self, splits: list | None = None) -> pd.DataFrame:
        """Load images from the given splits (default: all three).

        Returns a DataFrame with columns:
            study_id, label, subtype, impression, findings, image_path, split
        """
        if splits is None:
            splits = ["train", "val", "test"]

        rows = []
        for split in splits:
            split_dir = os.path.join(self.images_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for label in ("NORMAL", "PNEUMONIA"):
                label_dir = os.path.join(split_dir, label)
                if not os.path.isdir(label_dir):
                    continue
                for pattern in ("*.jpeg", "*.jpg", "*.png"):
                    for img_path in glob.glob(os.path.join(label_dir, pattern)):
                        subtype = self._subtype(label, img_path)
                        rows.append({
                            "study_id": Path(img_path).stem,
                            "label": label,
                            "subtype": subtype,
                            "impression": self.IMPRESSIONS[subtype],
                            "findings": self.FINDINGS[subtype],
                            "image_path": img_path,
                            "split": split,
                        })

        if not rows:
            raise FileNotFoundError(
                f"No images found under '{self.images_dir}'. "
                "On Kaggle add the dataset via: + Add Input → "
                "paultimothymooney/chest-xray-pneumonia"
            )

        return pd.DataFrame(rows).reset_index(drop=True)

    def load_split(self, split: str) -> pd.DataFrame:
        return self.load(splits=[split])

    def train_val_test(self) -> tuple:
        """Return (train_df, val_df, test_df) using the dataset's native splits."""
        full = self.load()
        return (
            full[full["split"] == "train"].reset_index(drop=True),
            full[full["split"] == "val"].reset_index(drop=True),
            full[full["split"] == "test"].reset_index(drop=True),
        )

    def stats(self) -> dict:
        """Return a summary dict of image counts per split and class."""
        df = self.load()
        return {
            "total": len(df),
            "by_split": df.groupby("split").size().to_dict(),
            "by_label": df.groupby("label").size().to_dict(),
            "by_subtype": df.groupby("subtype").size().to_dict(),
        }
