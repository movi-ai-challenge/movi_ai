"""Legacy FDS 결과 push client.

현재 공식 계약은 Spring이 FDS API를 호출하고 HTTP 응답으로 결과를 받는
동기 구조다. 이 모듈은 기존 연결 실험의 호환성을 위해 유지한다.
"""

from __future__ import annotations

import os
from typing import Any

import requests


BACKEND_BASE_URL = os.getenv(
    "BACKEND_BASE_URL",
    "https://moviback.duckdns.org",
)

BACKEND_FRAUD_ENDPOINT = os.getenv(
    "BACKEND_FRAUD_ENDPOINT",
    "/api/v1/fraud/result",
)

BACKEND_API_KEY = os.getenv(
    "BACKEND_API_KEY",
)


class BackendClientError(RuntimeError):
    pass


def send_fraud_result(
    *,
    transaction_id: str,
    anomaly_score: float,
    threshold: float,
    is_fraud: bool,
    risk_level: str,
    model: str = "isolation_forest",
    risk_score: float | None = None,
    triggered_rules: list[str] | None = None,
) -> dict[str, Any]:
    """
    FDS 결과를 Spring Backend로 전달한다.
    """

    url = (
        f"{BACKEND_BASE_URL.rstrip('/')}"
        f"/{BACKEND_FRAUD_ENDPOINT.lstrip('/')}"
    )

    payload = {
        "transaction_id": transaction_id,
        "anomaly_score": anomaly_score,
        "threshold": threshold,
        "is_fraud": is_fraud,
        "risk_level": risk_level,
        "model": model,
    }

    if risk_score is not None:
        payload["risk_score"] = risk_score

    if triggered_rules is not None:
        payload["triggered_rules"] = triggered_rules


    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


    # 백엔드에서 API Key 인증을 사용하는 경우
    if BACKEND_API_KEY:
        headers["X-API-KEY"] = BACKEND_API_KEY


    try:

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=5,
        )

    except requests.Timeout as error:

        raise BackendClientError(
            "Backend 요청 Timeout"
        ) from error

    except requests.ConnectionError as error:

        raise BackendClientError(
            "Backend 서버 연결 실패"
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
