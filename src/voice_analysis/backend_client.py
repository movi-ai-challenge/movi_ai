from __future__ import annotations

import os
from typing import Any

import requests


# ============================================================
# Backend Config
# ============================================================

BACKEND_BASE_URL = os.getenv(
    "BACKEND_BASE_URL",
    "https://moviback.duckdns.org",
)

VOICE_COMMAND_ENDPOINT = os.getenv(
    "VOICE_COMMAND_ENDPOINT",
    "/api/v1/voice/command",
)

BACKEND_TIMEOUT = float(
    os.getenv(
        "BACKEND_TIMEOUT",
        "5",
    )
)


# ============================================================
# Exception
# ============================================================

class BackendClientError(RuntimeError):
    """
    Backend 통신 중 발생하는 오류.
    """

    pass


# ============================================================
# URL 생성
# ============================================================

def build_voice_command_url() -> str:

    return (
        f"{BACKEND_BASE_URL.rstrip('/')}"
        f"/{VOICE_COMMAND_ENDPOINT.lstrip('/')}"
    )


# ============================================================
# Voice Command POST
# ============================================================

def send_voice_command(
    request_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Voice AI 분석 결과를 Spring Backend로 전달한다.

    Parameters
    ----------
    request_data
        request_mapper.py에서 생성한 Backend Request JSON

    Returns
    -------
    dict
        Backend JSON Response
    """

    url = build_voice_command_url()

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


    try:

        response = requests.post(
            url,
            json=request_data,
            headers=headers,
            timeout=BACKEND_TIMEOUT,
        )

    # ========================================================
    # Timeout
    # ========================================================

    except requests.Timeout as error:

        raise BackendClientError(
            f"Backend 요청 시간이 초과되었습니다. "
            f"url={url}"
        ) from error


    # ========================================================
    # Connection Error
    # ========================================================

    except requests.ConnectionError as error:

        raise BackendClientError(
            f"Backend 서버에 연결할 수 없습니다. "
            f"url={url}"
        ) from error


    # ========================================================
    # 기타 Request Error
    # ========================================================

    except requests.RequestException as error:

        raise BackendClientError(
            f"Backend 요청 중 오류가 발생했습니다: "
            f"{error}"
        ) from error


    # ========================================================
    # HTTP Error
    # ========================================================

    if not response.ok:

        raise BackendClientError(
            f"Backend 응답 오류\n"
            f"status_code={response.status_code}\n"
            f"url={url}\n"
            f"body={response.text}"
        )


    # ========================================================
    # Empty Response
    # ========================================================

    if not response.content:

        return {
            "status": "success"
        }


    # ========================================================
    # JSON Response
    # ========================================================

    try:

        return response.json()

    except ValueError:

        return {
            "status": "success",
            "message": response.text,
        }