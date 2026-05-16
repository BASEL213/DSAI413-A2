import json
import time
import re
from pathlib import Path
from tqdm import tqdm
from groq import Groq


# 15 clinical categories with 6 question templates each.
# Methodology follows MIMIC-CXR-VQA paper (MIDL 2026).
CLINICAL_QUESTIONS: dict[str, list[str]] = {
    "Atelectasis": [
        "Is atelectasis observed in this chest X-ray?",
        "Can atelectasis be identified in this image?",
        "Are there signs of atelectasis?",
        "Does this radiograph show evidence of atelectasis?",
        "What evidence of atelectasis, if any, is present?",
        "Are there indications of pulmonary atelectasis?",
    ],
    "Cardiomegaly": [
        "Is cardiomegaly present in this chest X-ray?",
        "Does the heart appear enlarged on this radiograph?",
        "Are there signs of an enlarged cardiac silhouette?",
        "Is the cardiothoracic ratio abnormal?",
        "Does this image show evidence of cardiomegaly?",
        "Are there indications of cardiac enlargement?",
    ],
    "Consolidation": [
        "Is there consolidation visible in this chest X-ray?",
        "Can pulmonary consolidation be identified?",
        "Are there signs of airspace consolidation?",
        "Does this radiograph show evidence of consolidation?",
        "What evidence of consolidation is present?",
        "Are there indications of pulmonary consolidation?",
    ],
    "Edema": [
        "Is pulmonary edema present in this chest X-ray?",
        "Are there signs of pulmonary edema?",
        "Can pulmonary edema be identified in this image?",
        "Does this radiograph show evidence of edema?",
        "Are there indications of interstitial or alveolar edema?",
        "What evidence of pulmonary edema is visible?",
    ],
    "Enlarged Mediastinum": [
        "Is the mediastinum widened in this chest X-ray?",
        "Are there signs of mediastinal widening?",
        "Does this radiograph show an enlarged mediastinum?",
        "Can mediastinal enlargement be identified?",
        "Is there evidence of mediastinal widening?",
        "Are there indications of an abnormal mediastinal contour?",
    ],
    "Fracture": [
        "Is there evidence of a fracture in this chest X-ray?",
        "Can any rib or clavicular fractures be identified?",
        "Are there signs of bony fracture?",
        "Does this radiograph show evidence of fracture?",
        "What evidence of fracture is present?",
        "Are there indications of skeletal injury?",
    ],
    "Lung Lesion": [
        "Is there a lung lesion visible in this chest X-ray?",
        "Can a pulmonary nodule or mass be identified?",
        "Are there signs of a lung lesion?",
        "Does this radiograph show evidence of a lung lesion?",
        "What evidence of pulmonary lesion is present?",
        "Are there indications of a nodule, mass, or opacity?",
    ],
    "Lung Opacity": [
        "Is there lung opacity visible in this chest X-ray?",
        "Can pulmonary opacity be identified?",
        "Are there signs of lung opacity?",
        "Does this radiograph show evidence of opacity?",
        "What evidence of lung opacity is present?",
        "Are there indications of airspace or interstitial opacity?",
    ],
    "No Finding": [
        "Is this chest X-ray normal?",
        "Are there any abnormal findings in this radiograph?",
        "Does this chest X-ray appear within normal limits?",
        "Are there any significant pathological findings?",
        "Is this a normal chest radiograph?",
        "Are there any notable findings in this chest X-ray?",
    ],
    "Pleural Effusion": [
        "Is there evidence of pleural effusion in this chest X-ray?",
        "Can pleural effusion be identified in this image?",
        "Are there signs of pleural effusion?",
        "Does this radiograph show evidence of fluid in the pleural space?",
        "What evidence of pleural effusion is visible?",
        "Are there indications of pleural fluid accumulation?",
    ],
    "Pleural Other": [
        "Are there any pleural abnormalities in this chest X-ray?",
        "Can pleural thickening or calcification be identified?",
        "Are there signs of pleural disease?",
        "Does this radiograph show evidence of pleural abnormality?",
        "What pleural findings are present?",
        "Are there indications of pleural pathology other than effusion?",
    ],
    "Pneumonia": [
        "Is there evidence of pneumonia in this chest X-ray?",
        "Can pneumonia be identified in this image?",
        "Are there signs of pneumonic infiltrate?",
        "Does this radiograph show evidence of pneumonia?",
        "What evidence of pneumonia is present?",
        "Are there indications of infectious or inflammatory lung consolidation?",
    ],
    "Pneumothorax": [
        "Is there a pneumothorax visible in this chest X-ray?",
        "Can pneumothorax be identified in this image?",
        "Are there signs of pneumothorax?",
        "Does this radiograph show evidence of pleural air?",
        "What evidence of pneumothorax is present?",
        "Are there indications of a collapsed lung?",
    ],
    "Support Devices": [
        "Are there any support devices visible in this chest X-ray?",
        "Can any lines, tubes, or wires be identified?",
        "Are there signs of medical equipment or support devices?",
        "Does this radiograph show any implanted or external devices?",
        "What support devices, if any, are present?",
        "Are there indications of endotracheal tube, central line, or other device placement?",
    ],
    "Subcutaneous Emphysema": [
        "Is there subcutaneous emphysema visible in this chest X-ray?",
        "Can subcutaneous air be identified?",
        "Are there signs of subcutaneous emphysema?",
        "Does this radiograph show evidence of soft tissue air?",
        "What evidence of subcutaneous emphysema is present?",
        "Are there indications of air in the soft tissues?",
    ],
}

KEYWORD_MAP: dict[str, list[str]] = {
    "Atelectasis": ["atelectasis", "atelectatic", "collapse", "subsegmental"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement", "cardiomeg"],
    "Consolidation": ["consolidation", "consolidative", "airspace disease"],
    "Edema": ["edema", "oedema", "vascular congestion", "interstitial"],
    "Enlarged Mediastinum": ["mediastinal widening", "widened mediastinum", "mediastinum"],
    "Fracture": ["fracture", "fractured", "break", "rib fracture"],
    "Lung Lesion": ["nodule", "mass", "lesion", "opacity", "neoplasm"],
    "Lung Opacity": ["opacity", "haziness", "infiltrate", "opacification"],
    "No Finding": ["no acute", "normal", "within normal limits", "unremarkable", "no significant"],
    "Pleural Effusion": ["effusion", "pleural fluid", "blunting", "costophrenic"],
    "Pleural Other": ["pleural thickening", "pleural calcification", "pleural disease"],
    "Pneumonia": ["pneumonia", "pneumonic", "infection", "infectious"],
    "Pneumothorax": ["pneumothorax", "ptx", "pleural air"],
    "Support Devices": ["tube", "line", "catheter", "wire", "pacemaker", "device", "support"],
    "Subcutaneous Emphysema": ["subcutaneous emphysema", "subcutaneous air", "soft tissue air"],
}

SYSTEM_PROMPT = (
    "You are a radiologist answering clinical questions about chest X-rays.\n"
    "Answer STRICTLY based on the provided radiographic findings and impression.\n\n"
    "Rules:\n"
    "- Do not mention prior studies, comparisons, follow-up, or temporal changes\n"
    "- Refer to 'the radiograph' or 'the image', NOT 'the report'\n"
    "- Give concise, evidence-based answers (1-2 sentences)\n"
    "- If a finding is not mentioned, state it is not observed\n"
    "- Never infer beyond what is explicitly stated in the findings"
)


class QACreator:
    """Creates QA pairs from radiology reports using Groq LLaMA 3.1 8B Instant."""

    MODEL = "llama-3.1-8b-instant"
    MAX_RETRIES = 3
    REQUESTS_PER_MINUTE = 28  # Groq free tier: 30/min, leave headroom

    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self._request_count = 0
        self._minute_start = time.time()

    def _select_questions(self, impression: str, findings: str) -> list[tuple[str, str]]:
        """Returns (category, question) pairs relevant to the impression."""
        text = (impression + " " + findings).lower()
        selected = []
        for category, keywords in KEYWORD_MAP.items():
            if any(kw in text for kw in keywords):
                # Pick first 3 questions for matched categories
                for q in CLINICAL_QUESTIONS[category][:3]:
                    selected.append((category, q))
        # Always include 1 support devices question and 1 no-finding check
        if not any(c == "Support Devices" for c, _ in selected):
            selected.append(("Support Devices", CLINICAL_QUESTIONS["Support Devices"][0]))
        # Cap to 8 questions per study to keep Groq costs low
        return selected[:8]

    def _rate_limit(self):
        self._request_count += 1
        elapsed = time.time() - self._minute_start
        if self._request_count > 20:
            target_interval = 60.0 / self.REQUESTS_PER_MINUTE
            time_per_request = elapsed / self._request_count
            if time_per_request < target_interval:
                time.sleep(target_interval - time_per_request)
        if elapsed > 60:
            self._request_count = 1
            self._minute_start = time.time()

    def _call_groq(self, question: str, impression: str, findings: str) -> str:
        user_content = (
            f"Findings: {findings}\n"
            f"Impression: {impression}\n\n"
            f"Question: {question}"
        )
        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_limit()
                resp = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.1,
                    max_tokens=150,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt == self.MAX_RETRIES - 1:
                    print(f"Failed after {self.MAX_RETRIES} retries: {str(e)[:80]}")
                    return ""
                backoff = min(10 * (2 ** attempt), 120)
                print(f"Groq error (attempt {attempt + 1}/{self.MAX_RETRIES}): waiting {backoff}s...")
                time.sleep(backoff)
        return ""

    def generate_for_study(
        self, study_id: str, impression: str, findings: str, image_path: str, split: str = "train"
    ) -> list[dict]:
        pairs = []
        for category, question in self._select_questions(impression, findings):
            answer = self._call_groq(question, impression, findings)
            if answer:
                pairs.append({
                    "study_id": study_id,
                    "image_path": image_path,
                    "category": category,
                    "question": question,
                    "answer": answer,
                    "split": split,
                })
        return pairs

    def generate_dataset(
        self, df, output_path: str, max_studies: int | None = None
    ) -> list[dict]:
        if max_studies:
            df = df.head(max_studies)
        all_pairs = []
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating QA pairs"):
                pairs = self.generate_for_study(
                    study_id=row["study_id"],
                    impression=row["impression"],
                    findings=row.get("findings", ""),
                    image_path=row["image_path"],
                    split=row.get("split", "train"),
                )
                for pair in pairs:
                    f.write(json.dumps(pair) + "\n")
                all_pairs.extend(pairs)

        print(f"Generated {len(all_pairs)} QA pairs → {output}")
        return all_pairs
