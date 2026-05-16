import os
import pickle
import torch
from pathlib import Path
from PIL import Image
from typing import List, Dict


class ColPaliRetriever:
    """
    Direct ColPali implementation using colpali-engine (no byaldi).
    Much faster than byaldi which saves entire index after every image.
    """

    MODEL_NAME = "vidore/colpali-v1.3"
    BATCH_SIZE = 1

    def __init__(self, device: str = "cuda"):
        from colpali_engine.models import ColPali, ColPaliProcessor

        self.device = device
        self.model = ColPali.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(self.MODEL_NAME)
        self.image_embeddings = None  # Will hold (N, num_patches, dim)
        self.image_paths: List[str] = []

    def build_index(self, images_dir: str = None, image_paths: List[str] = None, index_save_dir: str = None):
        """Index images. Either pass directory or list of paths."""
        if image_paths is None:
            if images_dir is None:
                raise ValueError("Either images_dir or image_paths required")
            image_paths = sorted([
                str(p) for p in Path(images_dir).rglob("*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ])

        if not image_paths:
            raise ValueError("No images found to index")

        print(f"Indexing {len(image_paths)} images with batch size {self.BATCH_SIZE}...")

        all_embeddings = []
        from tqdm import tqdm

        for i in tqdm(range(0, len(image_paths), self.BATCH_SIZE)):
            batch_paths = image_paths[i:i + self.BATCH_SIZE]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]

            with torch.no_grad():
                batch_inputs = self.processor.process_images(batch_images).to(self.device)
                # Remove 'labels' if present (prevents OOM loss computation)
                batch_inputs.pop('labels', None)
                batch_embeddings = self.model(**batch_inputs)

            all_embeddings.append(batch_embeddings.cpu())

            for img in batch_images:
                img.close()

            # Periodic cleanup to prevent VRAM fragmentation
            if i % 100 == 0:
                torch.cuda.empty_cache()

        self.image_embeddings = torch.cat(all_embeddings, dim=0)
        self.image_paths = image_paths

        print(f"Index built: {self.image_embeddings.shape}")

        if index_save_dir:
            self.save_index(index_save_dir)

    def save_index(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.image_embeddings, os.path.join(save_dir, "colpali_embeddings.pt"))
        with open(os.path.join(save_dir, "colpali_paths.pkl"), "wb") as f:
            pickle.dump(self.image_paths, f)
        print(f"Index saved to {save_dir}")

    def load_index(self, save_dir: str):
        self.image_embeddings = torch.load(os.path.join(save_dir, "colpali_embeddings.pt"))
        with open(os.path.join(save_dir, "colpali_paths.pkl"), "rb") as f:
            self.image_paths = pickle.load(f)
        print(f"Index loaded: {self.image_embeddings.shape}")

    @classmethod
    def from_index(cls, index_dir: str) -> "ColPaliRetriever":
        instance = cls.__new__(cls)
        from colpali_engine.models import ColPali, ColPaliProcessor
        instance.device = "cuda"
        instance.model = ColPali.from_pretrained(
            cls.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        instance.processor = ColPaliProcessor.from_pretrained(cls.MODEL_NAME)
        instance.load_index(index_dir)
        return instance

    def load_path_map(self, save_dir: str):
        """Backward compat: paths are already loaded in load_index."""
        pass

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search using late-interaction (MaxSim)."""
        with torch.no_grad():
            query_inputs = self.processor.process_queries([query]).to(self.device)
            query_embedding = self.model(**query_inputs).cpu()

        # Late-interaction: for each image, sum of max similarities per query token
        scores = []
        for img_emb in self.image_embeddings:
            # img_emb: (num_patches, dim), query_embedding: (1, num_query_tokens, dim)
            sim = torch.einsum("qd,pd->qp", query_embedding[0].float(), img_emb.float())
            score = sim.max(dim=-1).values.sum().item()
            scores.append(score)

        scores_tensor = torch.tensor(scores)
        top_k = scores_tensor.topk(min(k, len(scores)))

        results = []
        for idx, score in zip(top_k.indices.tolist(), top_k.values.tolist()):
            path = self.image_paths[idx]
            pil_image = Image.open(path).convert("RGB") if os.path.exists(path) else None
            results.append({
                "doc_id": idx,
                "score": score,
                "image_path": path,
                "image": pil_image,
            })
        return results
