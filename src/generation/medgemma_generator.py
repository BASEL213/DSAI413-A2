import json
import torch
from PIL import Image
from src.generation.prompts import (
    REPORT_DIRECT_PROMPT,
    REPORT_RAG_PROMPT,
    QA_PROMPT,
    STRUCTURED_REPORT_PROMPT,
)


class MedGemmaGenerator:
    """
    Wraps google/medgemma-1.5-4b-it for dual-mode CXR inference.
    Loaded in 4-bit NF4 quantization to run on a single T4 GPU (~3 GB VRAM).
    """

    MODEL_ID = "google/medgemma-1.5-4b-it"

    def __init__(self, hf_token: str, load_in_4bit: bool = True):
        from transformers import (
            AutoProcessor,
            AutoModelForImageTextToText,
            BitsAndBytesConfig,
        )

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, token=hf_token)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
        )
        self.model.eval()

    def generate_report(
        self,
        image: Image.Image,
        context_reports: list[str] | None = None,
        structured: bool = False,
        max_new_tokens: int = 300,
    ) -> str:
        """
        Mode 1: Report Generation.
        context_reports=None → direct (no RAG). Provided → RAG-augmented.
        structured=True → returns JSON-formatted report.
        """
        if structured:
            prompt = STRUCTURED_REPORT_PROMPT
        elif context_reports:
            context = "\n---\n".join(context_reports)
            prompt = REPORT_RAG_PROMPT.format(context=context)
        else:
            prompt = REPORT_DIRECT_PROMPT

        return self._generate([image], prompt, max_new_tokens)

    def answer_question(
        self,
        image: Image.Image,
        question: str,
        context_reports: list[str],
        max_new_tokens: int = 200,
    ) -> str:
        """Mode 2: RAG-based QA. Always requires retrieved context."""
        context = "\n---\n".join(context_reports)
        prompt = QA_PROMPT.format(context=context, question=question)
        return self._generate([image], prompt, max_new_tokens)

    def _generate(
        self, images: list[Image.Image], text: str, max_new_tokens: int
    ) -> str:
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def parse_structured_report(self, raw_output: str) -> dict:
        """Extract JSON from structured report output, with fallback."""
        try:
            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            return json.loads(raw_output[start:end])
        except (json.JSONDecodeError, ValueError):
            return {"raw": raw_output, "parse_error": True}
