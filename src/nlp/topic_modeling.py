"""Real-Time Topic Modeling, Keyword Extraction, and Trend Velocity Engine."""

import collections
import logging
import math
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("TopicModeling")

# Default SentenceTransformers model for BERTopic
DEFAULT_EMBEDDING_MODEL = os.getenv("TOPIC_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Common social media stopwords
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
    "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves",
    # Social media artifacts
    "https", "http", "t", "co", "rt", "via", "amp", "com", "www", "news", "post", "update", "today",
}


def preprocess_text(text: str) -> str:
    """Clean and normalize social media raw text."""
    if not text or not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove mentions
    text = re.sub(r"@\w+", "", text)
    # Normalize whitespace and lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def extract_ngrams(text: str, n: int = 1) -> List[str]:
    """Extract clean unigrams, bigrams, or trigrams excluding stopwords."""
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    if n == 1:
        return tokens
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


class TopicModeler:
    """
    Topic Modeler supporting BERTopic semantic clustering
    with high-performance TF-IDF/N-Gram heuristic fallback.
    """

    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.embedding_model = embedding_model
        self._bertopic_model = None
        self._bertopic_failed = False

    def _get_bertopic(self):
        """Lazy load BERTopic model."""
        if self._bertopic_model is not None or self._bertopic_failed:
            return self._bertopic_model

        try:
            logger.info(f"Loading BERTopic with embedding model '{self.embedding_model}'...")
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            from sklearn.feature_extraction.text import CountVectorizer

            embedding_net = SentenceTransformer(self.embedding_model)
            vectorizer_model = CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))

            self._bertopic_model = BERTopic(
                embedding_model=embedding_net,
                vectorizer_model=vectorizer_model,
                min_topic_size=3,
                verbose=False,
            )
            logger.info("BERTopic model successfully initialized.")
        except Exception as e:
            logger.warning(f"Could not load BERTopic: {e}. Falling back to TF-IDF / N-gram clustering.")
            self._bertopic_failed = True
            self._bertopic_model = None

        return self._bertopic_model

    def _fallback_extract_topics(
        self, posts: List[Dict[str, Any]], top_n: int = 10, time_window_hours: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        TF-IDF and N-Gram heuristic topic clustering engine.
        Groups posts by dominant semantic n-grams and calculates keyword statistics.
        """
        if not posts:
            return []

        cleaned_docs = []
        for p in posts:
            txt = preprocess_text(p.get("raw_text", ""))
            cleaned_docs.append(txt)

        # Count document frequencies and term frequencies
        doc_count = len(cleaned_docs)
        term_doc_freq = collections.defaultdict(set)
        term_total_freq = collections.defaultdict(int)

        for doc_idx, doc in enumerate(cleaned_docs):
            unigrams = extract_ngrams(doc, 1)
            bigrams = extract_ngrams(doc, 2)
            all_terms = unigrams + bigrams

            for term in all_terms:
                term_doc_freq[term].add(doc_idx)
                term_total_freq[term] += 1

        if not term_doc_freq:
            return []

        # Calculate TF-IDF style relevance scores for keywords
        scored_terms = []
        for term, doc_set in term_doc_freq.items():
            df = len(doc_set)
            tf = term_total_freq[term]
            idf = math.log((doc_count + 1) / (df + 1)) + 1.0
            # Boost bigrams slightly for better readability
            bigram_boost = 1.2 if " " in term else 1.0
            score = tf * idf * bigram_boost
            scored_terms.append((term, score, df, doc_set))

        scored_terms.sort(key=lambda x: x[1], reverse=True)

        # Cluster posts around top seed terms
        assigned_docs: Set[int] = set()
        topics_result = []
        topic_id_counter = 1

        for term, score, df, doc_set in scored_terms:
            if len(topics_result) >= top_n:
                break

            # Find matching docs not yet assigned or primary
            current_cluster_docs = doc_set - assigned_docs
            # If overlap is high, allow shared docs
            if len(current_cluster_docs) < 2 and len(doc_set) >= 2:
                current_cluster_docs = doc_set

            if not current_cluster_docs:
                continue

            assigned_docs.update(current_cluster_docs)

            # Find co-occurring top keywords in this cluster
            cluster_terms = collections.defaultdict(int)
            for d_idx in current_cluster_docs:
                for t in extract_ngrams(cleaned_docs[d_idx], 1) + extract_ngrams(cleaned_docs[d_idx], 2):
                    if t != term and not any(part in STOPWORDS for part in t.split()):
                        cluster_terms[t] += 1

            top_co_occurring = [
                k for k, _ in sorted(cluster_terms.items(), key=lambda x: x[1], reverse=True)[:4]
            ]
            all_keywords = [term] + top_co_occurring

            post_count = len(current_cluster_docs)
            # Velocity score: posts per hour within time window, scaled by count
            velocity_score = round(float(post_count / max(time_window_hours, 0.1)), 2)

            # Synthesize title label
            primary_label = " ".join(w.capitalize() for w in term.split())
            if top_co_occurring:
                secondary_label = " ".join(w.capitalize() for w in top_co_occurring[0].split())
                label = f"{primary_label} & {secondary_label}"
            else:
                label = f"Topic: {primary_label}"

            topics_result.append({
                "topic_id": topic_id_counter,
                "label": label,
                "keywords": all_keywords[:5],
                "post_count": post_count,
                "velocity_score": velocity_score,
            })
            topic_id_counter += 1

        return topics_result

    def extract_topics(
        self, posts: List[Dict[str, Any]], top_n: int = 10, time_window_hours: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Extract trending topics from post list using BERTopic or heuristic fallback.
        """
        if not posts:
            return []

        # If sample size is small (< 10), use heuristic fallback directly for speed and stability
        if len(posts) < 10:
            return self._fallback_extract_topics(posts, top_n=top_n, time_window_hours=time_window_hours)

        model = self._get_bertopic()
        if model is None:
            return self._fallback_extract_topics(posts, top_n=top_n, time_window_hours=time_window_hours)

        try:
            docs = [preprocess_text(p.get("raw_text", "")) for p in posts]
            # Filter out near-empty docs
            valid_docs = [d for d in docs if len(d) > 5]
            if len(valid_docs) < 8:
                return self._fallback_extract_topics(posts, top_n=top_n, time_window_hours=time_window_hours)

            topics, _ = model.fit_transform(valid_docs)
            topic_info = model.get_topic_info()

            results = []
            for _, row in topic_info.iterrows():
                t_id = int(row["Topic"])
                if t_id == -1:  # Outlier cluster in BERTopic
                    continue
                if len(results) >= top_n:
                    break

                count = int(row["Count"])
                topic_words = [word for word, _ in model.get_topic(t_id)][:5]
                label_words = [w.capitalize() for w in topic_words[:2]]
                label = f"Topic: {' / '.join(label_words)}" if label_words else f"Topic #{t_id}"

                velocity = round(float(count / max(time_window_hours, 0.1)), 2)

                results.append({
                    "topic_id": len(results) + 1,
                    "label": label,
                    "keywords": topic_words,
                    "post_count": count,
                    "velocity_score": velocity,
                })

            # If BERTopic produced 0 or 1 topic due to strict min_size, supplement with heuristic
            if len(results) < 2:
                fallback_results = self._fallback_extract_topics(
                    posts, top_n=top_n, time_window_hours=time_window_hours
                )
                if len(fallback_results) > len(results):
                    return fallback_results

            return results
        except Exception as e:
            logger.warning(f"BERTopic inference encountered error: {e}. Using fallback.")
            return self._fallback_extract_topics(posts, top_n=top_n, time_window_hours=time_window_hours)


def extract_trending_topics(
    posts: List[Dict[str, Any]], top_n: int = 10, time_window_hours: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Convenience function to extract trending topics from a list of post dictionaries.

    Returns:
        list of dicts containing:
        - topic_id (int)
        - label (str)
        - keywords (list of str)
        - post_count (int)
        - velocity_score (float)
    """
    modeler = TopicModeler()
    return modeler.extract_topics(posts, top_n=top_n, time_window_hours=time_window_hours)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_posts = [
        {"raw_text": "Cybersecurity breach reported in central servers. Zero-day vulnerability exploited."},
        {"raw_text": "Critical security patch released for the zero-day exploit. Upgrade immediately."},
        {"raw_text": "Ransomware and cyber attacks surge across critical infrastructure."},
        {"raw_text": "Economy shows steady growth in the third quarter with market rally."},
        {"raw_text": "Stock markets reach all-time high amid robust quarterly economic earnings."},
        {"raw_text": "Central bank announces interest rate stabilization plan for economic support."},
    ]
    topics = extract_trending_topics(sample_posts, top_n=5)
    print("Extracted Trending Topics:")
    for t in topics:
        print(f"  [{t['topic_id']}] {t['label']} (Count: {t['post_count']}, Velocity: {t['velocity_score']}) -> Keywords: {t['keywords']}")
