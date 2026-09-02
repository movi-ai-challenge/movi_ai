"""테스트 대상이 외부 SDK를 실제 호출하지 않도록 최소 import stub을 제공한다."""

from __future__ import annotations

import sys
from types import ModuleType


def install_optional_dependency_stubs() -> None:
    try:
        import openai  # noqa: F401
    except ModuleNotFoundError:
        openai_stub = ModuleType("openai")

        class OpenAI:
            pass

        class APIConnectionError(Exception):
            pass

        class APITimeoutError(Exception):
            pass

        class RateLimitError(Exception):
            pass

        openai_stub.OpenAI = OpenAI
        openai_stub.APIConnectionError = APIConnectionError
        openai_stub.APITimeoutError = APITimeoutError
        openai_stub.RateLimitError = RateLimitError
        sys.modules["openai"] = openai_stub

    try:
        import dotenv  # noqa: F401
    except ModuleNotFoundError:
        dotenv_stub = ModuleType("dotenv")
        dotenv_stub.load_dotenv = lambda: None
        sys.modules["dotenv"] = dotenv_stub
