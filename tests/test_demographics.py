"""Unit tests for Automated Demographic Profiling and ClickHouse Demographics Storage."""

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.jobs.demographic_profiler import DemographicProfilerJob, ensure_sha256
from src.nlp.demographics import (
    DemographicProfiler,
    infer_user_demographics,
)
from src.storage.clickhouse_client import ClickHouseStorageClient


class TestDemographicProfiling(unittest.TestCase):
    def setUp(self):
        self.profiler = DemographicProfiler.get_instance()

    def test_location_extraction(self):
        """Test location extraction from explicit field and bio text."""
        # Explicit location
        loc1 = self.profiler.extract_location("Senior Analyst", explicit_location="New Delhi, India")
        self.assertEqual(loc1, "New Delhi, India")

        # Regex marker location
        loc2 = self.profiler.extract_location("Cybersecurity Researcher 📍 London. Threat Hunter.")
        self.assertEqual(loc2, "London")

        # Gazetteer matching
        loc3 = self.profiler.extract_location("Software developer based in Bengaluru building distributed tools.")
        self.assertEqual(loc3, "Bengaluru")

    def test_age_bracket_inference(self):
        """Test age bracket deduction from direct age, birth years, and life stages."""
        # Direct age mentions
        self.assertEqual(self.profiler.infer_age_bracket("22yo tech enthusiast"), "18-24")
        self.assertEqual(self.profiler.infer_age_bracket("29 y/o software engineer"), "25-34")
        self.assertEqual(self.profiler.infer_age_bracket("Age: 42. Director of Ops."), "35-54")
        self.assertEqual(self.profiler.infer_age_bracket("62 years old retired officer"), "55+")

        # Birth year mentions (assuming 2026 current year)
        self.assertEqual(self.profiler.infer_age_bracket("born in 2004, coder"), "18-24")
        self.assertEqual(self.profiler.infer_age_bracket("b. 1995, building web tools"), "25-34")
        self.assertEqual(self.profiler.infer_age_bracket("DOB: 1982"), "35-54")
        self.assertEqual(self.profiler.infer_age_bracket("born in 1960"), "55+")

        # Life stage keywords
        self.assertEqual(self.profiler.infer_age_bracket("College undergrad student @IIT"), "18-24")
        self.assertEqual(self.profiler.infer_age_bracket("PhD scholar & early career researcher"), "25-34")
        self.assertEqual(self.profiler.infer_age_bracket("Father of 2, 18+ years experience in telecom"), "35-54")
        self.assertEqual(self.profiler.infer_age_bracket("Retired Army General (ret.)"), "55+")

    def test_professional_role_inference(self):
        """Test role inference across security, engineering, journalism, defense, academia."""
        self.assertEqual(
            self.profiler.infer_professional_role("OSINT researcher & threat intelligence analyst"),
            "Cybersecurity & Intelligence",
        )
        self.assertEqual(
            self.profiler.infer_professional_role("Senior Full Stack Software Engineer and Cloud Architect"),
            "Software & Data Engineering",
        )
        self.assertEqual(
            self.profiler.infer_professional_role("Defence correspondent and investigative journalist"),
            "Journalism & Media",
        )
        self.assertEqual(
            self.profiler.infer_professional_role("Military officer in armed forces defence department"),
            "Defence & Public Sector",
        )
        self.assertEqual(
            self.profiler.infer_professional_role("Professor of Computer Science and Academic Researcher"),
            "Academic & Research",
        )
        self.assertEqual(
            self.profiler.infer_professional_role("CEO & Founder at TechVentures"),
            "Executive & Business",
        )

    def test_full_user_demographic_pipeline(self):
        """Test end-to-end infer_user_demographics utility."""
        user = {
            "bio": "24yo Threat Intelligence Analyst @CERT-In. Based in New Delhi 📍. Ethical Hacking.",
            "location": "New Delhi, India",
        }
        res = infer_user_demographics(user)

        self.assertEqual(res["age_bracket"], "18-24")
        self.assertEqual(res["location_gpe"], "New Delhi, India")
        self.assertEqual(res["professional_role"], "Cybersecurity & Intelligence")
        self.assertEqual(res["primary_language"], "en")


class TestDemographicsStorage(unittest.TestCase):
    def setUp(self):
        self.client = ClickHouseStorageClient(
            host="localhost", port=9000, user="default", password="", database="ntro_analytics"
        )

    def test_demographics_table_init(self):
        """Verify user_demographics table create query."""
        with patch.object(self.client, "get_client") as mock_get_client:
            mock_ch = MagicMock()
            mock_get_client.return_value = mock_ch

            self.client.init_tables()

            calls = [call[0][0] for call in mock_ch.execute.call_args_list]
            demo_query = [c for c in calls if "CREATE TABLE IF NOT EXISTS ntro_analytics.user_demographics" in c][0]

            self.assertIn("hashed_user_id String", demo_query)
            self.assertIn("age_bracket String", demo_query)
            self.assertIn("location_gpe String", demo_query)
            self.assertIn("primary_language String", demo_query)
            self.assertIn("professional_role String", demo_query)
            self.assertIn("ENGINE = ReplacingMergeTree", demo_query)

    def test_upsert_demographics(self):
        """Verify upsert operation on user_demographics."""
        with patch.object(self.client, "get_client") as mock_get_client:
            mock_ch = MagicMock()
            mock_get_client.return_value = mock_ch

            records = [
                {
                    "hashed_user_id": "a" * 64,
                    "age_bracket": "25-34",
                    "location_gpe": "London",
                    "primary_language": "en",
                    "professional_role": "Cybersecurity & Intelligence",
                    "updated_at": "2026-08-29T12:00:00Z",
                }
            ]

            count = self.client.upsert_user_demographics(records)
            self.assertEqual(count, 1)
            mock_ch.execute.assert_called()

            call_args = mock_ch.execute.call_args
            query = call_args[0][0]
            data = call_args[0][1]

            self.assertIn("INSERT INTO ntro_analytics.user_demographics", query)
            self.assertEqual(data[0][0], "a" * 64)
            self.assertEqual(data[0][1], "25-34")
            self.assertEqual(data[0][2], "London")
            self.assertEqual(data[0][4], "Cybersecurity & Intelligence")


class TestDemographicProfilerJob(unittest.TestCase):
    def test_job_run_once(self):
        """Test DemographicProfilerJob execution and hashing."""
        mock_storage = MagicMock()
        mock_storage.check_connection.return_value = True

        mock_users = [
            {
                "user_id": "user_raw_123",
                "bio": "22yo college student in Mumbai. Passionate about AI.",
            }
        ]

        job = DemographicProfilerJob(batch_limit=10, clickhouse_client=mock_storage)
        results = job.run_once(mock_users=mock_users)

        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["hashed_user_id"], ensure_sha256("user_raw_123"))
        self.assertEqual(res["age_bracket"], "18-24")
        self.assertEqual(res["location_gpe"], "Mumbai")


if __name__ == "__main__":
    unittest.main()
