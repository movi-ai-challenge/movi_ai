from __future__ import annotations

import os
from typing import Any

import requests


BACKEND_BASE_URL = os.getenv(
    "BACKEND_BASE_URL",
    "https://moviback.duckdns.org",
)

VOICE_COMMAND_ENDPOINT = os.getenv(
    "VOICE_COMMAND_ENDPOINT",
    "/api/v1/voice/command",
)


class BackendClientError(
    RuntimeError
):
    pass


def send_voice_command(
    request_data: dict[str, Any],
) -> dict[str, Any]:

    url = (
        f"{BACKEND_BASE_URL.rstrip('/')}"
        f"/{VOICE_COMMAND_ENDPOINT.lstrip('/')}"
    )

    try:

        response = requests.post(
            url,
            json=request_data,
            timeout=5,
        )

    except requests.Timeout as error:

        raise BackendClientError(
            "Backend 요청 Timeout"
        ) from error

    except requests.ConnectionError as error:

        raise BackendClientError(
            "Backend 연결 실패"
        ) from error

    except requests.RequestException as error:

        raise BackendClientError(
            f"Backend 요청 실패: {error}"
        ) from error


    if not response.ok:

        raise BackendClientError(
            f"Backend 응답 오류 "
            f"status={response.status_code}, "
            f"body={response.text}"
        )


    if not response.content:

        return {
            "status": "success"
        }


    try:

        return response.json()

    except ValueError:

        return {
            "status": "success",
            "body": response.text,
        }