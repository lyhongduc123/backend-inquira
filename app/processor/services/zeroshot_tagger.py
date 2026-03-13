import numpy as np
from sentence_transformers import CrossEncoder

TAG_TEMPLATES = {
    "methodology": [
        "The research methodology used in this study is a {}.",
        "This academic paper conducts a {}.",
        "The experimental design of this research is based on a {}."
    ],
    "topic": [
        "The primary topic of this academic paper is {}.",
        "This research paper contributes to the field of {}.",
        "This text discusses concepts related to {}."
    ],
    "resource": [
        "This research paper presents or provides {}.",
        "The authors of this study release {}.",
    ],
    "general": [
        "This paper is about {}.",
        "The main focus of this document is {}."
    ]
}

class ZeroShotTaggerService:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/nli-deberta-v3-xsmall")

    def _softmax(self, x):
        """Convert logits to probabilities (0.0 -> 1.0)"""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def compute_tags(self, abstract: str, candidate_labels: list, category: str = "general") -> list:
        """
        Compute relevance scores for candidate labels using zero-shot classification.
        """
        templates = TAG_TEMPLATES.get(category, TAG_TEMPLATES["general"])
        final_results = []
        for label in candidate_labels:
            pairs = [[abstract, template.format(label)] for template in templates]
            raw_logits = self.model.predict(pairs)
            probabilities = self._softmax(raw_logits)
            
            entailment_probs = probabilities[:, 1]
            avg_confidence = np.mean(entailment_probs)
            
            final_results.append({
                "tag": label,
                "confidence": round(float(avg_confidence) * 100, 2)
            })

        final_results.sort(key=lambda x: x["confidence"], reverse=True)
        return final_results