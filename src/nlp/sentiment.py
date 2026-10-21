"""Multi-Dimensional Sentiment and Emotion Inference Engine."""

import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("EmotionAnalyzer")

# Default Hugging Face emotion classifier model
DEFAULT_MODEL_NAME = os.getenv(
    "EMOTION_MODEL_NAME", "cardiffnlp/twitter-roberta-base-emotion-latest"
)


class EmotionAnalyzer:
    """
    Multi-dimensional emotion and stance analyzer.
    Extracts 5 key sentiment dimensions:
    - Sarcasm
    - Anxiety
    - Excitement
    - Support
    - Opposition
    """

    _instance: Optional["EmotionAnalyzer"] = None

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, use_gpu: bool = False) -> None:
        self.model_name = model_name
        self.use_gpu = use_gpu
        self._pipeline = None
        self._model_failed = False

    @classmethod
    def get_instance(cls) -> "EmotionAnalyzer":
        """Singleton accessor for emotion analyzer."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        """Lazy load Hugging Face pipeline."""
        if self._pipeline is not None or self._model_failed:
            return self._pipeline

        try:
            logger.info(f"Loading Hugging Face emotion model '{self.model_name}'...")
            from transformers import pipeline
            import torch

            device = 0 if (self.use_gpu and torch.cuda.is_available()) else -1
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                device=device,
                truncation=True,
                max_length=512,
            )
            logger.info(f"Emotion model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load Hugging Face model '{self.model_name}': {e}. "
                "Using high-performance heuristic analyzer."
            )
            self._model_failed = True
            self._pipeline = None

        return self._pipeline

    def _heuristic_analysis(self, text: str) -> Dict[str, float]:
        """
        Fast lexical and semantic rule-based analyzer used when model is offline
        or for hybrid score adjustments.
        """
        clean_text = text.lower()
        words = set(re.findall(r"\b\w+\b", clean_text))

        # Lexicons
        excitement_lexicon = {
            "amazing", "awesome", "great", "excellent", "super", "fantastic", "wonderful",
            "thrilled", "excited", "brilliant", "love", "outstanding", "incredible", "happy",
            "victory", "win", "yay", "wow", "hurray", "breakthrough", "congratulations",
            "best", "success", "innovative", "proud"
        }
        anxiety_lexicon = {
            "worried", "anxious", "fear", "scared", "danger", "risk", "panic", "threat",
            "warning", "terrified", "concern", "crisis", "nervous", "dread", "alarm",
            "breach", "fail", "failure", "trouble", "emergency", "catastrophe", "vulnerable"
        }
        support_lexicon = {
            "support", "agree", "endorse", "recommend", "praise", "good", "helpful",
            "beneficial", "positive", "legitimate", "protect", "solid", "true", "trust",
            "approved", "backed", "vote", "respect", "appreciate", "welcome", "right"
        }
        opposition_lexicon = {
            "oppose", "against", "disagree", "reject", "condemn", "boycott", "protest",
            "corrupt", "hate", "terrible", "worst", "ban", "shame", "fraud", "fake",
            "lies", "liar", "attack", "enemy", "bad", "illegal", "evil", "disaster", "stop"
        }
        sarcasm_markers = {
            "/s", "sarcasm", "yeah right", "as if", "surely", "totally normal", "obviously",
            "great job...", "brilliant idea...", "oh sure"
        }

        # Calculate keyword match densities
        excitement_matches = len(words.intersection(excitement_lexicon))
        anxiety_matches = len(words.intersection(anxiety_lexicon))
        support_matches = len(words.intersection(support_lexicon))
        opposition_matches = len(words.intersection(opposition_lexicon))

        # Sarcasm markers & structural checks
        sarcasm_score = 0.05
        for marker in sarcasm_markers:
            if marker in clean_text:
                sarcasm_score += 0.45

        # Check sarcastic quotes around positive words (e.g. "great", "expert")
        if re.search(r'"(great|amazing|expert|genius|secure|safe|perfect)"', clean_text):
            sarcasm_score += 0.4

        # Compute normalized scores with baseline priors
        excl_count = min(text.count("!"), 3) * 0.1
        caps_ratio = min(sum(1 for c in text if c.isupper()) / max(len(text), 1), 0.5) * 0.4

        excitement_score = min(0.05 + (excitement_matches * 0.3) + excl_count + caps_ratio, 0.98)
        anxiety_score = min(0.05 + (anxiety_matches * 0.3) + (0.1 if "?" in text and anxiety_matches else 0.0), 0.95)
        support_score = min(0.05 + (support_matches * 0.25) + (excitement_matches * 0.15), 0.95)
        opposition_score = min(0.05 + (opposition_matches * 0.25) + (caps_ratio * 0.2), 0.95)

        # Baseline adjustment for neutral / low-signal text
        if not (excitement_matches or anxiety_matches or support_matches or opposition_matches):
            return {
                "sarcasm_score": round(float(sarcasm_score), 4),
                "anxiety_score": 0.05,
                "excitement_score": 0.05,
                "support_score": 0.10,
                "opposition_score": 0.05,
            }

        return {
            "sarcasm_score": round(float(min(max(sarcasm_score, 0.0), 1.0)), 4),
            "anxiety_score": round(float(min(max(anxiety_score, 0.0), 1.0)), 4),
            "excitement_score": round(float(min(max(excitement_score, 0.0), 1.0)), 4),
            "support_score": round(float(min(max(support_score, 0.0), 1.0)), 4),
            "opposition_score": round(float(min(max(opposition_score, 0.0), 1.0)), 4),
        }

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze text and return scores for Sarcasm, Anxiety, Excitement, Support, Opposition.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "sarcasm_score": 0.0,
                "anxiety_score": 0.0,
                "excitement_score": 0.0,
                "support_score": 0.0,
                "opposition_score": 0.0,
            }

        pipe = self._load_model()
        if pipe is None:
            return self._heuristic_analysis(text)

        try:
            results = pipe(text)
            # Flatten top_k results if nested
            if results and isinstance(results[0], list):
                scores_list = results[0]
            elif results and isinstance(results, list):
                scores_list = results
            else:
                scores_list = []

            # Map raw labels (e.g., joy, optimism, anger, sadness, fear)
            raw_scores = {}
            for item in scores_list:
                label = item.get("label", "").lower()
                score = float(item.get("score", 0.0))
                raw_scores[label] = score

            # Normalize to target dimensions
            # Joy / Optimism -> Excitement & Support
            joy = raw_scores.get("joy", raw_scores.get("positive", 0.0))
            optimism = raw_scores.get("optimism", 0.0)
            surprise = raw_scores.get("surprise", 0.0)

            # Fear / Sadness -> Anxiety
            fear = raw_scores.get("fear", 0.0)
            sadness = raw_scores.get("sadness", 0.0)

            # Anger / Disgust -> Opposition
            anger = raw_scores.get("anger", raw_scores.get("negative", 0.0))
            disgust = raw_scores.get("disgust", 0.0)

            # Lexical sarcasm detection combined with sentiment contrast
            heuristic = self._heuristic_analysis(text)

            excitement_score = min(max(joy * 0.8 + optimism * 0.5 + surprise * 0.3 + heuristic["excitement_score"] * 0.2, 0.0), 1.0)
            anxiety_score = min(max(fear * 0.8 + sadness * 0.5 + heuristic["anxiety_score"] * 0.2, 0.0), 1.0)
            support_score = min(max(joy * 0.5 + optimism * 0.7 + (1.0 - anger) * 0.2 + heuristic["support_score"] * 0.3, 0.0), 1.0)
            opposition_score = min(max(anger * 0.8 + disgust * 0.6 + heuristic["opposition_score"] * 0.3, 0.0), 1.0)
            sarcasm_score = heuristic["sarcasm_score"]

            return {
                "sarcasm_score": round(float(sarcasm_score), 4),
                "anxiety_score": round(float(anxiety_score), 4),
                "excitement_score": round(float(excitement_score), 4),
                "support_score": round(float(support_score), 4),
                "opposition_score": round(float(opposition_score), 4),
            }
        except Exception as e:
            logger.warning(f"Error during transformer emotion inference: {e}. Falling back to heuristic.")
            return self._heuristic_analysis(text)


def analyze_emotion(text: str) -> Dict[str, float]:
    """
    Convenience function to analyze emotion dimensions of input text.

    Returns:
        dict with keys:
        - sarcasm_score (float, 0.0 to 1.0)
        - anxiety_score (float, 0.0 to 1.0)
        - excitement_score (float, 0.0 to 1.0)
        - support_score (float, 0.0 to 1.0)
        - opposition_score (float, 0.0 to 1.0)
    """
    analyzer = EmotionAnalyzer.get_instance()
    return analyzer.analyze(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = "This project is amazing! Great work team."
    res = analyze_emotion(sample)
    print(f"Text: '{sample}'")
    print(f"Scores: {res}")
