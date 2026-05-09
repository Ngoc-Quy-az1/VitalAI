from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from src.LLM.tools.medical_tools.constants import ALLOWED_ENDPOINTS, DEFAULT_CONTRACT_PATH


class MedicalToolsClient:
    """Minimal async wrapper around the deployable medical tools HTTP service."""

    def __init__(self, base_url: str, timeout_seconds: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_seconds = timeout_seconds

    async def evaluate(self, parameters: dict[str, Any], endpoint: str = "/mcp/medical-tools/evaluate") -> dict[str, Any]:
        """Call medical tools evaluate endpoint and return parsed JSON."""

        if endpoint not in ALLOWED_ENDPOINTS:
            return {
                "tool_status": "blocked",
                "error": f"Endpoint không được phép: {endpoint}",
            }
        return await asyncio.to_thread(self._post_json, endpoint, parameters)

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = urljoin(self.base_url, endpoint.lstrip("/"))
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            return {"tool_status": "http_error", "status_code": exc.code, "error": str(exc)}
        except URLError as exc:
            return {"tool_status": "unavailable", "error": str(exc.reason)}
        except TimeoutError:
            return {"tool_status": "timeout", "error": "Medical tools service timeout"}
        except json.JSONDecodeError as exc:
            return {"tool_status": "bad_json", "error": str(exc)}


def load_medical_tools_contract(path: str | Path | None = None) -> str:
    """Read runtime MCP contract markdown for the router agent."""

    contract_path = Path(path) if path else DEFAULT_CONTRACT_PATH
    return contract_path.read_text(encoding="utf-8")
