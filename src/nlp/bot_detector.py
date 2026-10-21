"""Bot Detection Engine using Isolation Forest and Statistical Behavioral Heuristics."""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("BotDetector")

# Thresholds for rule-based anomaly flags
BOT_SCORE_THRESHOLD = 0.6      # Scores at or above this classify as a bot
HIGH_FREQ_POSTS_PER_HOUR = 20  # Suspiciously high posting rate
HIGH_REPETITION_RATE = 0.75    # 75%+ identical/near-identical posts
LOW_UNIQUE_RECIPIENTS = 3      # Fewer than 3 distinct interaction targets


class BotDetector:
    """
    Bot Detection via Isolation Forest (scikit-learn) with resilient statistical heuristic fallback.

    Features analyzed:
    - posts_per_hour (float):        Average posting frequency
    - repetition_rate (float):       Ratio of near-duplicate posts (0.0–1.0)
    - unique_recipients (int):       Number of distinct user interaction targets
    - avg_response_time_sec (float): Average seconds from trigger to response
    - follower_following_ratio (float): Follower/following balance
    - account_age_days (float):      Age of account in days
    - burst_flag (int 0/1):          1 if short bursts of high-volume posting detected
    """

    _instance: Optional["BotDetector"] = None

    def __init__(self, contamination: float = 0.15) -> None:
        """
        Args:
            contamination: Expected fraction of bots in the population (0.0–0.5).
        """
        self.contamination = contamination
        self._model = None
        self._is_fitted = False
        self._sklearn_failed = False

    @classmethod
    def get_instance(cls) -> "BotDetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _extract_features(self, activity: Dict[str, Any]) -> List[float]:
        """Convert a user activity dict to a feature vector."""
        return [
            float(activity.get("posts_per_hour", 0.0)),
            float(activity.get("repetition_rate", 0.0)),
            float(activity.get("unique_recipients", 10)),
            float(activity.get("avg_response_time_sec", 300.0)),
            float(activity.get("follower_following_ratio", 1.0)),
            float(activity.get("account_age_days", 365.0)),
            float(activity.get("burst_flag", 0)),
        ]

    def _build_model(self):
        """Lazy-import and build the Isolation Forest model with safe fallback."""
        if self._sklearn_failed:
            return None
        try:
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(
                n_estimators=200,
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1,
            )
            return model
        except Exception as e:
            logger.debug(f"Could not load IsolationForest from sklearn: {e}. Using statistical heuristics.")
            self._sklearn_failed = True
            return None

    def fit(self, activity_logs: List[Dict[str, Any]]) -> None:
        """
        Fit the Isolation Forest on a population of user activity logs.
        Used to establish a normal behavioral baseline.
        """
        if len(activity_logs) < 5:
            logger.warning("Too few samples to fit Isolation Forest. Minimum 5 required.")
            return

        model = self._build_model()
        if model is not None:
            try:
                X = np.array([self._extract_features(a) for a in activity_logs])
                model.fit(X)
                self._model = model
                self._is_fitted = True
                logger.info(f"BotDetector IsolationForest fitted on {len(X)} user activity samples.")
                return
            except Exception as e:
                logger.debug(f"IsolationForest fit failed: {e}. Falling back to rule-based scoring.")
                self._model = None

        self._is_fitted = True
        logger.info(f"BotDetector baseline established for {len(activity_logs)} user activity samples using statistical heuristics.")

    def score_user(self, activity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single user's activity for bot-like behavior.

        Returns dict with:
        - bot_score (float 0.0–1.0): Higher = more bot-like
        - is_bot (bool)
        - anomaly_flags (list[str]): Human-readable anomaly reasons
        - method (str): 'isolation_forest' or 'heuristic'
        """
        flags: List[str] = []

        # ── Rule-based heuristic flags ──────────────────────────────
        pph = float(activity.get("posts_per_hour", 0.0))
        rep = float(activity.get("repetition_rate", 0.0))
        uniq = int(activity.get("unique_recipients", 10))
        burst = int(activity.get("burst_flag", 0))
        age = float(activity.get("account_age_days", 365.0))
        ffr = float(activity.get("follower_following_ratio", 1.0))

        if pph >= HIGH_FREQ_POSTS_PER_HOUR:
            flags.append(f"HIGH_POST_FREQUENCY:{pph:.1f}/hr")
        if rep >= HIGH_REPETITION_RATE:
            flags.append(f"HIGH_REPETITION_RATE:{rep:.0%}")
        if uniq <= LOW_UNIQUE_RECIPIENTS:
            flags.append(f"LOW_UNIQUE_RECIPIENTS:{uniq}")
        if burst:
            flags.append("BURST_POSTING_DETECTED")
        if age < 30:
            flags.append(f"NEW_ACCOUNT:{age:.0f}d")
        if ffr > 20:
            flags.append(f"ABNORMAL_FOLLOWER_RATIO:{ffr:.1f}x")

        # Compute weighted heuristic score
        raw_score = (
            (min(pph / 40.0, 1.0) * 0.35) +
            (min(rep, 1.0) * 0.30) +
            (1.0 if burst else 0.0) * 0.20 +
            (0.15 if uniq <= LOW_UNIQUE_RECIPIENTS else 0.0)
        )
        heuristic_score = min(1.0, max(0.0, raw_score))

        # ── Isolation Forest score (if available) ───────────────────
        if self._is_fitted and self._model is not None:
            try:
                X = np.array([self._extract_features(activity)])
                raw = self._model.decision_function(X)[0]
                bot_score = float(np.clip(0.5 - raw, 0.0, 1.0))
                method = "isolation_forest"
            except Exception:
                bot_score = heuristic_score
                method = "heuristic"
        else:
            bot_score = heuristic_score
            method = "heuristic"

        is_bot = bot_score >= BOT_SCORE_THRESHOLD or len(flags) >= 3

        return {
            "bot_score": round(bot_score, 4),
            "is_bot": is_bot,
            "anomaly_flags": flags,
            "method": method,
        }

    def detect_bots(
        self, user_activity_logs: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze a batch of user activity logs for bot behavior.

        Each log must have a 'hashed_user_id' key plus behavioral metrics.

        Returns:
            dict mapping hashed_user_id -> score result dict
        """
        if not user_activity_logs:
            return {}

        # Fit on entire batch if not already fitted
        if not self._is_fitted:
            self.fit(user_activity_logs)

        results: Dict[str, Dict[str, Any]] = {}
        bot_count = 0

        for activity in user_activity_logs:
            uid = str(activity.get("hashed_user_id", "unknown"))
            result = self.score_user(activity)
            results[uid] = result
            if result["is_bot"]:
                bot_count += 1
                logger.info(
                    f"  [BOT FLAGGED] {uid[:12]}... score={result['bot_score']:.3f} "
                    f"flags={result['anomaly_flags']}"
                )

        logger.info(f"Bot detection complete: {bot_count}/{len(results)} flagged as bots.")
        return results


def detect_bots(user_activity_logs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to detect bots in a list of user activity logs.

    Args:
        user_activity_logs: List of dicts with keys:
            - hashed_user_id (str)
            - posts_per_hour (float)
            - repetition_rate (float 0.0–1.0)
            - unique_recipients (int)
            - avg_response_time_sec (float)
            - follower_following_ratio (float)
            - account_age_days (float)
            - burst_flag (int 0/1)

    Returns:
        dict mapping hashed_user_id -> {bot_score, is_bot, anomaly_flags, method}
    """
    detector = BotDetector.get_instance()
    return detector.detect_bots(user_activity_logs)
