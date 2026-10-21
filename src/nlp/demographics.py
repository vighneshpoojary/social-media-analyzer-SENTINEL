"""Automated Demographic Profiling Engine (NER, Age Bracket, Role, and Language)."""

import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("DemographicProfiler")

CURRENT_YEAR = 2026

# Known geopolitical entities (GPE) gazetteer for high accuracy matching
KNOWN_GPE_GAZETTEER = {
    # Countries
    "india", "united states", "usa", "us", "uk", "united kingdom", "canada", "australia",
    "germany", "france", "russia", "china", "japan", "israel", "uae", "singapore",
    # Indian Metros and States
    "new delhi", "delhi", "mumbai", "bengaluru", "bangalore", "hyderabad", "chennai",
    "kolkata", "pune", "ahmedabad", "noida", "gurgaon", "gurugram", "chandigarh",
    "kerala", "karnataka", "maharashtra", "punjab", "tamil nadu", "uttar pradesh",
    # Global Metros
    "london", "new york", "san francisco", "washington", "tokyo", "paris", "berlin",
    "dubai", "toronto", "sydney", "geneva", "beijing", "tel aviv"
}


class DemographicProfiler:
    """
    Infers demographic profiles:
    - Age bracket (18-24, 25-34, 35-54, 55+, Unknown)
    - Location GPE (Geopolitical Entity)
    - Professional Role
    - Primary Language
    """

    _instance: Optional["DemographicProfiler"] = None

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self._nlp = None
        self._spacy_failed = False

    @classmethod
    def get_instance(cls) -> "DemographicProfiler":
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_spacy(self):
        """Lazy load spaCy model with error handling."""
        if self._nlp is not None or self._spacy_failed:
            return self._nlp

        try:
            import spacy
            self._nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model '{self.model_name}'.")
        except Exception as e:
            logger.warning(
                f"Could not load spaCy model '{self.model_name}': {e}. "
                "Using regex and gazetteer NER fallback."
            )
            self._spacy_failed = True
            self._nlp = None

        return self._nlp

    def extract_location(self, text: str, explicit_location: Optional[str] = None) -> str:
        """Extract Location (GPE) using gazetteer matching and spaCy NER."""
        if explicit_location and isinstance(explicit_location, str) and len(explicit_location.strip()) > 1:
            return explicit_location.strip()

        if not text or not isinstance(text, str):
            return "Unknown"

        # 1. Gazetteer check (fast and precise)
        clean_lower = text.lower()
        for loc in sorted(KNOWN_GPE_GAZETTEER, key=len, reverse=True):
            pattern = rf"\b{re.escape(loc)}\b"
            if re.search(pattern, clean_lower):
                return loc.title()

        # 2. Try spaCy NER
        nlp = self._load_spacy()
        if nlp is not None:
            try:
                doc = nlp(text)
                gpe_ents = [ent.text.strip() for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
                if gpe_ents:
                    return gpe_ents[0]
            except Exception as e:
                logger.debug(f"spaCy location extraction error: {e}")

        # 3. Regex location markers
        marker_match = re.search(
            r"(?:📍|based in|living in|from|located in)\s+([A-Za-z\s,]+?)(?=[|•\.\n\r]|$)",
            text,
            re.IGNORECASE,
        )
        if marker_match:
            candidate = marker_match.group(1).strip()
            # If candidate contains stop words or verbs, take only the first 1-2 words
            words = candidate.split()
            if words:
                loc_words = []
                for w in words[:3]:
                    if w.lower() in {"building", "working", "studying", "making", "and", "with", "for"}:
                        break
                    loc_words.append(w)
                if loc_words:
                    return " ".join(loc_words).title()

        return "Unknown"

    def infer_age_bracket(self, text: str) -> str:
        """
        Infer age bracket: '18-24', '25-34', '35-54', '55+', or 'Unknown'.
        """
        if not text or not isinstance(text, str):
            return "Unknown"

        clean = text.lower()

        # 1. Direct age patterns: "age: 42", "age 28", "22yo", "25 y/o", "45 years old"
        direct_age_patterns = [
            r"\bage:?\s*(\d{2})\b",
            r"\b(\d{2})\s*(?:yo|y/o|years?\s*old|years?\s*young)\b",
            r"\b(\d{2})\s*yr\s*old\b",
        ]
        for pat in direct_age_patterns:
            m = re.search(pat, clean)
            if m:
                age = int(m.group(1))
                if 15 <= age <= 24:
                    return "18-24"
                elif 25 <= age <= 34:
                    return "25-34"
                elif 35 <= age <= 54:
                    return "35-54"
                elif age >= 55:
                    return "55+"

        # 2. Birth year patterns: "born in 1998", "b. 2004", "'95 kid", "dob: 1982"
        birth_match = re.search(
            r"\b(?:born\s*(?:in\s*)?|b\.\s*|dob:?\s*)(19\d{2}|20\d{2})\b",
            clean,
        )
        if birth_match:
            birth_year = int(birth_match.group(1))
            age = CURRENT_YEAR - birth_year
            if 15 <= age <= 24:
                return "18-24"
            elif 25 <= age <= 34:
                return "25-34"
            elif 35 <= age <= 54:
                return "35-54"
            elif age >= 55:
                return "55+"

        # 3. Graduation cohort patterns
        grad_match = re.search(
            r"\b(?:class of|grad(?:uating)?(?:\s*in)?)\s*(20\d{2}|19\d{2})\b",
            clean,
        )
        if grad_match:
            grad_year = int(grad_match.group(1))
            if grad_year >= 2024:
                return "18-24"
            elif 2016 <= grad_year <= 2023:
                return "25-34"
            elif 1995 <= grad_year <= 2015:
                return "35-54"
            elif grad_year < 1995:
                return "55+"

        # 4. Life-stage keywords
        if any(w in clean for w in ["student", "undergrad", "college sophomore", "college senior", "freshman", "fresher", "gen z"]):
            return "18-24"
        if any(w in clean for w in ["phd scholar", "postdoc", "early career", "young professional", "millennial"]):
            return "25-34"
        if any(w in clean for w in ["father of", "mother of", "dad of", "mom of", "15+ years experience", "18+ years", "20+ years", "senior director"]):
            return "35-54"
        if any(w in clean for w in ["retired", "veteran 30+ years", "veteran 35+ years", "emeritus", "grandpa", "grandmother", "ret."]):
            return "55+"

        return "Unknown"

    def infer_professional_role(self, text: str) -> str:
        """Infer professional role from bio or post text."""
        if not text or not isinstance(text, str):
            return "General / Other"

        clean = text.lower()

        # Prioritized roles (Journalism & Security prioritized over broad keywords)
        role_patterns = [
            ("Journalism & Media", [
                "journalist", "reporter", "correspondent", "editor", "columnist", "anchor",
                "news editor", "author", "writer", "broadcaster", "press", "investigative journalist",
                "defence correspondent", "security correspondent"
            ]),
            ("Cybersecurity & Intelligence", [
                "cybersecurity", "cyber security", "security researcher", "infosec", "threat intelligence",
                "osint", "soc analyst", "pentester", "ethical hacker", "incident response",
                "malware analyst", "security engineer", "ciso", "intelligence officer"
            ]),
            ("Software & Data Engineering", [
                "software engineer", "developer", "data scientist", "devops", "cloud architect",
                "full stack", "backend", "frontend", "machine learning", "ai engineer",
                "programmer", "coder", "tech lead", "cto"
            ]),
            ("Defence & Public Sector", [
                "military officer", "army officer", "naval officer", "air force", "police", "civil servant",
                "diplomat", "public policy", "government official", "armed forces", "bureaucrat",
                "defence", "military", "army", "navy"
            ]),
            ("Academic & Research", [
                "professor", "researcher", "phd candidate", "student", "scholar", "lecturer",
                "scientist", "academic", "fellow", "educator"
            ]),
            ("Executive & Business", [
                "ceo", "founder", "director", "vp", "executive", "entrepreneur", "investor",
                "venture capitalist", "managing director", "co-founder", "manager"
            ]),
        ]

        for category, keywords in role_patterns:
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", clean):
                    return category

        return "General / Other"

    def detect_language(self, text: str) -> str:
        """Detect primary language based on script and characters."""
        if not text or not isinstance(text, str):
            return "en"

        # Check for Devanagari script (Hindi, Marathi, etc.)
        if re.search(r"[\u0900-\u097F]", text):
            return "hi"
        # Check for Arabic script (Urdu, Arabic, Persian)
        if re.search(r"[\u0600-\u06FF]", text):
            return "ar"
        # Check for Cyrillic script
        if re.search(r"[\u0400-\u04FF]", text):
            return "ru"
        # Check for CJK characters (Chinese)
        if re.search(r"[\u4E00-\u9FFF]", text):
            return "zh"

        return "en"

    def profile_user(self, user_metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Analyze user metadata and return demographic attributes:
        - age_bracket
        - location_gpe
        - primary_language
        - professional_role
        """
        bio_text = str(user_metadata.get("bio", "") or user_metadata.get("description", "") or "")
        recent_text = str(user_metadata.get("raw_text", "") or user_metadata.get("posts_text", "") or "")
        explicit_loc = user_metadata.get("location")

        combined_text = f"{bio_text} {recent_text}".strip()

        location_gpe = self.extract_location(combined_text, explicit_location=explicit_loc)
        age_bracket = self.infer_age_bracket(combined_text)
        professional_role = self.infer_professional_role(combined_text)
        primary_language = self.detect_language(combined_text)

        return {
            "age_bracket": age_bracket,
            "location_gpe": location_gpe,
            "primary_language": primary_language,
            "professional_role": professional_role,
        }


def infer_user_demographics(user_metadata: Dict[str, Any]) -> Dict[str, str]:
    """
    Convenience function to infer demographics for a user profile dictionary.

    Returns:
        dict with keys:
        - age_bracket (str: '18-24', '25-34', '35-54', '55+', or 'Unknown')
        - location_gpe (str)
        - primary_language (str)
        - professional_role (str)
    """
    profiler = DemographicProfiler.get_instance()
    return profiler.profile_user(user_metadata)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_user = {
        "bio": "28yo Senior Cybersecurity Analyst @NTRO. Based in New Delhi 📍. OSINT & Threat Intel.",
        "location": "New Delhi, India",
    }
    demo = infer_user_demographics(sample_user)
    print(f"Sample User: {sample_user}")
    print(f"Demographics: {demo}")
