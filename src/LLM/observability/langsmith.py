from __future__ import annotations

"""Cấu hình LangSmith tracing từ biến môi trường.

LangChain/LangSmith hiện hỗ trợ nhóm biến `LANGSMITH_*`, trong khi một số
integration vẫn đọc alias cũ `LANGCHAIN_*`. Module này chuẩn hóa cả hai nhóm để
tracing hoạt động nhất quán trong script, API server và notebook.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class LangSmithConfig:
    """Trạng thái LangSmith sau khi normalize environment."""

    tracing_enabled: bool
    project: str | None
    endpoint: str | None
    has_api_key: bool


def configure_langsmith_from_env(
    dotenv_path: str | Path | None = None,
    *,
    override: bool = False,
) -> LangSmithConfig:
    """Load `.env` và map `LANGSMITH_*` sang alias `LANGCHAIN_*`.

    Các biến chính trong `.env`:
    - `LANGSMITH_TRACING=true|false`
    - `LANGSMITH_API_KEY=...`
    - `LANGSMITH_PROJECT=...`
    - `LANGSMITH_ENDPOINT=...` optional

    Hàm này không validate API key với network. Nó chỉ đảm bảo các biến cần
    thiết có mặt trong `os.environ` trước khi LangChain tạo run trace.
    """

    env_path = Path(dotenv_path) if dotenv_path else None
    load_dotenv(dotenv_path=env_path, override=override)
    for key, value in _read_observability_env(env_path).items():
        if override or key not in os.environ:
            os.environ[key] = value

    api_key = _first_env("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
    project = _first_env("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")
    endpoint = _first_env("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT")
    tracing_raw = _first_env("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
    tracing_enabled = _is_truthy(tracing_raw)

    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGCHAIN_API_KEY"] = api_key
    if project:
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ["LANGCHAIN_PROJECT"] = project
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint

    if tracing_raw is not None:
        normalized = "true" if tracing_enabled else "false"
        os.environ["LANGSMITH_TRACING"] = normalized
        os.environ["LANGCHAIN_TRACING_V2"] = normalized

    return LangSmithConfig(
        tracing_enabled=tracing_enabled,
        project=project,
        endpoint=endpoint,
        has_api_key=bool(api_key),
    )


def _first_env(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return None


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().strip('"\'').lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return False


def _read_observability_env(dotenv_path: Path | None) -> dict[str, str]:
    path = dotenv_path or Path(".env")
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key.startswith(("LANGSMITH_", "LANGCHAIN_")):
            continue

        value = _strip_inline_comment(value.strip())
        values[key] = _strip_optional_quotes(value)
    return values


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if value.startswith(("'", '"')) and not value.endswith(value[0]):
        return value[1:]
    return value


def _strip_inline_comment(value: str) -> str:
    if value.startswith(("'", '"')):
        return value
    return value.split(" #", 1)[0].strip()
