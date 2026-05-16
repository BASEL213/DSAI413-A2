import numpy as np
from typing import Sequence


class Evaluator:
    """Evaluation metrics for radiology report generation and QA."""

    # ── Report generation metrics ─────────────────────────────────────────────

    def bertscore(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        model_type: str = "microsoft/deberta-xlarge-mnli",
    ) -> dict[str, float]:
        """Token-level semantic similarity using contextual embeddings."""
        from bert_score import score as _score
        P, R, F1 = _score(
            list(predictions),
            list(references),
            lang="en",
            model_type=model_type,
            verbose=False,
        )
        return {
            "BERTScore_P": P.mean().item(),
            "BERTScore_R": R.mean().item(),
            "BERTScore_F1": F1.mean().item(),
        }

    def rouge_l(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> float:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        return float(np.mean(scores))

    def radgraph_f1(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> float:
        """Clinical entity and relation F1 using the RadGraph model."""
        from radgraph import F1RadGraph
        scorer = F1RadGraph(reward_level="partial")
        result = scorer(hyps=list(predictions), refs=list(references))
        # radgraph returns (mean_f1, list_of_f1s) depending on version
        if isinstance(result, tuple):
            return float(result[0])
        return float(result)

    # ── QA metrics ────────────────────────────────────────────────────────────

    def bleu(
        self, predictions: Sequence[str], references: Sequence[str], n: int = 4
    ) -> float:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        smoothie = SmoothingFunction().method1
        refs = [[ref.split()] for ref in references]
        hyps = [pred.split() for pred in predictions]
        weights = tuple(1.0 / n for _ in range(n))
        return corpus_bleu(refs, hyps, weights=weights, smoothing_function=smoothie)

    # ── Aggregate report ──────────────────────────────────────────────────────

    def evaluate_report_generation(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> dict[str, float]:
        results = {}
        results.update(self.bertscore(predictions, references))
        results["ROUGE-L"] = self.rouge_l(predictions, references)
        try:
            results["RadGraph_F1"] = self.radgraph_f1(predictions, references)
        except Exception as e:
            results["RadGraph_F1"] = float("nan")
            print(f"[RadGraph skipped] {e}")
        return results

    def evaluate_qa(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> dict[str, float]:
        results = {}
        results.update(self.bertscore(predictions, references))
        results["BLEU-4"] = self.bleu(predictions, references)
        results["ROUGE-L"] = self.rouge_l(predictions, references)
        return results
