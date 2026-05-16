import xml.etree.ElementTree as ET
import pandas as pd
import os
import re
import glob
from pathlib import Path


class OpenILoader:
    """Loads the Indiana University / OpenI chest X-ray dataset.

    Primary method: load_from_kaggle_csvs() — uses the CSV files that come
    with the Kaggle dataset (raddar/chest-xrays-indiana-university).
    Fallback method: load() — parses the original OpenI XML report files.
    """

    COMPARISON_PATTERNS = [
        r'\bcompared to( the)? (previous|prior|old|last)\b.*?[.\n]',
        r'\b(unchanged|stable|no change)\b',
        r'\b(new|interval)\b(?= (development|appearance|finding))',
        r'\bsince( the)? (prior|previous|last)\b.*?[.\n]',
    ]

    def __init__(self, images_dir: str, reports_dir: str = ""):
        self.images_dir = images_dir
        self.reports_dir = reports_dir

    # ── Primary: Kaggle CSV method ────────────────────────────────────────────

    def load_from_kaggle_csvs(self, kaggle_dir: str) -> pd.DataFrame:
        """Load from the CSVs bundled with the Kaggle dataset.

        Expects:
            kaggle_dir/indiana_reports.csv
            kaggle_dir/indiana_projections.csv
        """
        reports_path = os.path.join(kaggle_dir, "indiana_reports.csv")
        proj_path    = os.path.join(kaggle_dir, "indiana_projections.csv")

        if not os.path.exists(reports_path) or not os.path.exists(proj_path):
            raise FileNotFoundError(
                f"CSV files not found in {kaggle_dir}. "
                "Run: kaggle datasets download -d raddar/chest-xrays-indiana-university"
            )

        reports = pd.read_csv(reports_path)
        projections = pd.read_csv(proj_path)

        # Keep frontal views only (PA / AP)
        frontal = projections[
            projections["projection"].str.lower() == "frontal"
        ].copy()

        # Build full image paths and keep only files that actually exist
        frontal["image_path"] = frontal["filename"].apply(
            lambda fn: os.path.join(self.images_dir, str(fn))
        )
        frontal = frontal[frontal["image_path"].apply(os.path.exists)]

        # Merge: one frontal image per study
        frontal_first = frontal.drop_duplicates(subset="uid", keep="first")
        merged = reports.merge(
            frontal_first[["uid", "image_path"]], on="uid", how="inner"
        )

        # Clean text columns
        merged["impression"] = (
            merged["impression"]
            .fillna("")
            .astype(str)
            .apply(lambda t: self.strip_comparison_language(t.strip()))
        )
        merged["findings"] = (
            merged["findings"]
            .fillna("")
            .astype(str)
            .apply(lambda t: self.strip_comparison_language(t.strip()))
        )

        # Drop rows with no usable impression
        merged = merged[merged["impression"].str.len() > 10]
        merged = merged.rename(columns={"uid": "study_id"})

        return merged[["study_id", "impression", "findings", "image_path"]].reset_index(drop=True)

    # ── Fallback: XML method ──────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Parse original OpenI XML report files (fallback if CSVs not present)."""
        if not self.reports_dir:
            raise ValueError("reports_dir must be set to use load()")

        rows = []
        xml_files = glob.glob(os.path.join(self.reports_dir, "*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No XML files found in {self.reports_dir}")

        for xml_path in xml_files:
            row = self._parse_xml(xml_path)
            if row:
                rows.append(row)

        if not rows:
            raise ValueError(
                "XML parsing returned 0 valid rows. "
                "Check that images_dir contains matching PNG files."
            )

        df = pd.DataFrame(rows)
        df = df.dropna(subset=["impression", "image_path"])
        df = df[df["impression"].str.strip().str.len() > 10]
        return df.reset_index(drop=True)

    def _parse_xml(self, xml_path: str) -> dict | None:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            return None

        # uId is stored as an attribute, not element text
        uid_el = root.find(".//uId")
        uid = (uid_el.get("id") if uid_el is not None else None) or Path(xml_path).stem

        impression = root.findtext(".//AbstractText[@Label='IMPRESSION']") or ""
        findings   = root.findtext(".//AbstractText[@Label='FINDINGS']") or ""

        image_ids = [fig.get("id", "") for fig in root.findall(".//parentImage")]
        frontal_path = self._find_frontal_image(image_ids)

        if not impression.strip() or not frontal_path:
            return None

        return {
            "study_id":  uid,
            "impression": self.strip_comparison_language(impression.strip()),
            "findings":   self.strip_comparison_language(findings.strip()),
            "image_path": frontal_path,
        }

    def _find_frontal_image(self, image_ids: list[str]) -> str | None:
        for img_id in image_ids:
            # Try with and without extension, strip .dcm suffix if present
            img_id_clean = img_id.replace(".dcm", "")
            for ext in (".png", ".jpg", ".jpeg"):
                path = os.path.join(self.images_dir, img_id_clean + ext)
                if os.path.exists(path):
                    return path
        return None

    def strip_comparison_language(self, text: str) -> str:
        for pattern in self.COMPARISON_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return re.sub(r'\s{2,}', ' ', text).strip()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def train_val_test_split(
        self, df: pd.DataFrame, train: float = 0.8, val: float = 0.1
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df)
        n_train = int(n * train)
        n_val   = int(n * val)
        return (
            df.iloc[:n_train].assign(split="train"),
            df.iloc[n_train:n_train + n_val].assign(split="val"),
            df.iloc[n_train + n_val:].assign(split="test"),
        )
