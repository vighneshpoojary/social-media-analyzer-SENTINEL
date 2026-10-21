"""Unit tests for Multi-Dimensional Sentiment & Emotion Inference Engine."""

import unittest
from src.nlp.sentiment import EmotionAnalyzer, analyze_emotion


class TestEmotionAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = EmotionAnalyzer.get_instance()

    def test_positive_excitement_text(self):
        """Verify 'This project is amazing!' generates high excitement and support scores."""
        text = "This project is amazing!"
        scores = analyze_emotion(text)

        # Verify all 5 keys exist
        expected_keys = {
            "sarcasm_score",
            "anxiety_score",
            "excitement_score",
            "support_score",
            "opposition_score",
        }
        self.assertEqual(set(scores.keys()), expected_keys)

        # Verify values are floats in [0.0, 1.0]
        for k, v in scores.items():
            self.assertIsInstance(v, float, f"Score for {k} must be float")
            self.assertGreaterEqual(v, 0.0, f"{k} must be >= 0.0")
            self.assertLessEqual(v, 1.0, f"{k} must be <= 1.0")

        # Excitement & support should be prominent
        self.assertGreater(scores["excitement_score"], 0.2)
        self.assertGreater(scores["support_score"], 0.1)

    def test_anxiety_fear_text(self):
        """Verify anxious/critical warning text generates high anxiety score."""
        text = "Severe emergency and catastrophic crisis! Huge security breach and danger detected."
        scores = analyze_emotion(text)

        self.assertGreater(scores["anxiety_score"], 0.3)
        self.assertLess(scores["excitement_score"], 0.3)

    def test_opposition_protest_text(self):
        """Verify dissent and opposition generates high opposition score."""
        text = "We completely oppose and reject this terrible decision. Protest and boycott now!"
        scores = analyze_emotion(text)

        self.assertGreater(scores["opposition_score"], 0.3)
        self.assertLess(scores["support_score"], 0.2)

    def test_sarcasm_markers(self):
        """Verify sarcastic markers produce elevated sarcasm score."""
        text = 'Oh "genius" plan, great job... obviously a total success /s'
        scores = analyze_emotion(text)

        self.assertGreater(scores["sarcasm_score"], 0.4)

    def test_empty_and_whitespace_text(self):
        """Verify empty strings return zeroed scores without errors."""
        empty_scores = analyze_emotion("")
        self.assertEqual(empty_scores["excitement_score"], 0.0)
        self.assertEqual(empty_scores["sarcasm_score"], 0.0)

        whitespace_scores = analyze_emotion("   \n\t  ")
        self.assertEqual(whitespace_scores["anxiety_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
