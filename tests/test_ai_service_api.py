from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.app import app, get_answerer


class FakeAnswerer:
    async def answer(self, **_: Any) -> dict[str, Any]:
        return {
            "query": "Xin chào",
            "answer": "Xin chào",
            "route": "direct",
            "sources": [],
        }

    async def stream_answer(self, **_: Any):
        yield {"event": "token", "token": "Xin"}
        yield {"event": "token", "token": " chào"}
        yield {
            "event": "done",
            "query": "Xin chào",
            "answer": "Xin chào",
            "route": "direct",
            "sources": [],
        }

    async def summarize_memory(self, **_: Any) -> str:
        return "Người dùng đang hỏi tiếp về bệnh thận."


class AiServiceApiTests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides[get_answerer] = lambda: FakeAnswerer()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_sync_answer_endpoint(self) -> None:
        response = self.client.post("/chat/answer", json={"query": "Xin chào"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Xin chào")

    def test_stream_endpoint_emits_sse_tokens_and_done_event(self) -> None:
        with self.client.stream("POST", "/chat/stream", json={"query": "Xin chào"}) as response:
            body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: token", body)
        self.assertIn('"token": "Xin"', body)
        self.assertIn('"token": " chào"', body)
        self.assertIn("event: done", body)
        self.assertIn('"answer": "Xin chào"', body)

    def test_memory_summary_endpoint(self) -> None:
        response = self.client.post(
            "/memory/summarize",
            json={
                "previous_summary": "",
                "question": "Hội chứng thận hư là gì?",
                "answer": "Đây là bệnh lý thận.",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("bệnh thận", response.json()["summary"])


if __name__ == "__main__":
    unittest.main()
