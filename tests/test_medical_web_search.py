from __future__ import annotations

import sys
import unittest
from os import environ
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.LLM.web_search.medical_google_cse import (
    DEFAULT_ALLOWED_DOMAINS,
    DEFAULT_BLOCKED_DOMAINS,
    _is_allowed_domain,
    _with_domain_controls,
    search_medical_web,
)


class MedicalWebSearchTests(unittest.TestCase):
    def test_search_is_noop_without_google_credentials(self) -> None:
        old_key = environ.pop("GOOGLE_API_KEY", None)
        old_cx = environ.pop("GOOGLE_CX", None)
        try:
            self.assertEqual(search_medical_web("kidney disease", num_results=1), [])
        finally:
            if old_key is not None:
                environ["GOOGLE_API_KEY"] = old_key
            if old_cx is not None:
                environ["GOOGLE_CX"] = old_cx

    def test_domain_allowlist_blocks_wikipedia_and_social(self) -> None:
        self.assertTrue(_is_allowed_domain("cdc.gov", DEFAULT_ALLOWED_DOMAINS, DEFAULT_BLOCKED_DOMAINS))
        self.assertFalse(_is_allowed_domain("wikipedia.org", DEFAULT_ALLOWED_DOMAINS, DEFAULT_BLOCKED_DOMAINS))
        self.assertFalse(_is_allowed_domain("facebook.com", DEFAULT_ALLOWED_DOMAINS, DEFAULT_BLOCKED_DOMAINS))

    def test_query_adds_blocked_site_exclusions(self) -> None:
        query = _with_domain_controls("viêm cầu thận", DEFAULT_BLOCKED_DOMAINS)

        self.assertIn("-site:wikipedia.org", query)
        self.assertIn("-site:facebook.com", query)
        self.assertIn("medical kidney nephrology guideline", query)


if __name__ == "__main__":
    unittest.main()
