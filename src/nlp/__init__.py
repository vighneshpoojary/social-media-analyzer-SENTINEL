"""NLP, Emotion Inference, Topic Modeling, Demographic Profiling, and Bot Detection Package."""

from src.nlp.sentiment import EmotionAnalyzer, analyze_emotion
from src.nlp.topic_modeling import TopicModeler, extract_trending_topics
from src.nlp.demographics import DemographicProfiler, infer_user_demographics
from src.nlp.bot_detector import BotDetector, detect_bots

__all__ = [
    "EmotionAnalyzer",
    "analyze_emotion",
    "TopicModeler",
    "extract_trending_topics",
    "DemographicProfiler",
    "infer_user_demographics",
    "BotDetector",
    "detect_bots",
]
