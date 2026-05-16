import os
import pickle
import numpy as np
import torch
from pathlib import Path
from PIL import Image


class CLIPRetriever:
    """
    Baseline retriever using CLIP ViT-L/14 with FAISS IndexFlatIP.
    Encodes images globally (one embedding per image) and searches by text or image query.
    Used to compare against ColPali's patch-level late-interaction retrieval.
    """

    MODEL_NAME = "ViT-L-14"
    PRETRAINED = "openai"

    def __init__(self):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME, pretrained=self.PRETRAINED
        )
        self.tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self._index = None
        self._image_paths: list[str] = []

    def build_index(self, image_paths: list[str], batch_size: int = 64):
        """Encode all images and build a FAISS cosine similarity index."""
        import faiss
        from tqdm import tqdm
        all_embeddings = []
        print(f'Encoding {len(image_paths)} images...')
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i: i + batch_size]
            imgs = []
            valid_paths = []
            for p in batch_paths:
                try:
                    imgs.append(self.preprocess(Image.open(p).convert("RGB")))
                    valid_paths.append(p)
                except Exception:
                    pass
            if not imgs:
                continue
            tensor = torch.stack(imgs).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(emb.cpu().numpy().astype("float32"))
            self._image_paths.extend(valid_paths)

        embeddings = np.vstack(all_embeddings)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # inner product == cosine on L2-normalised vecs
        self._index.add(embeddings)

    def save_index(self, save_dir: str):
        import faiss
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self._index, os.path.join(save_dir, "clip_faiss.index"))
        with open(os.path.join(save_dir, "clip_paths.pkl"), "wb") as f:
            pickle.dump(self._image_paths, f)

    def load_index(self, save_dir: str):
        import faiss
        self._index = faiss.read_index(os.path.join(save_dir, "clip_faiss.index"))
        with open(os.path.join(save_dir, "clip_paths.pkl"), "rb") as f:
            self._image_paths = pickle.load(f)

    def _encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype("float32")

    def _encode_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype("float32")

    def _search(self, query_emb: np.ndarray, k: int) -> list[dict]:
        scores, indices = self._index.search(query_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            path = self._image_paths[idx]
            pil_image = None
            if os.path.exists(path):
                pil_image = Image.open(path).convert("RGB")
            results.append({"score": float(score), "image_path": path, "image": pil_image})
        return results

    def search_by_text(self, query: str, k: int = 3) -> list[dict]:
        return self._search(self._encode_text(query), k)

    def search_by_image(self, image: Image.Image, k: int = 3) -> list[dict]:
        return self._search(self._encode_image(image), k)
