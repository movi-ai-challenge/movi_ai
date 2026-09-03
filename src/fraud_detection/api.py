"""MOVI Fraud Detection HTTP API."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from .fraud_service import (
    FraudDetectionService,
    FraudDetectionServiceError,
    InvalidTransactionRequest,
)
from .config import SERVICE_VERSION
from .schemas import FraudDetectionRequest, FraudDetectionResponse


logger = logging.getLogger(__name__)

app = FastAPI(
    title="MOVI Fraud Detection API",
    description="Isolation Forest + Rule Engine 기반 MOVI 이상거래 탐지 API",
    version=SERVICE_VERSION,
)

fraud_service = FraudDetectionService()


def fraud_service_ready() -> bool:
    return fraud_service is not None and fraud_service.ready()


def require_fraud_service() -> FraudDetectionService:
    if not fraud_service_ready():
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_NOT_READY",
                "message": "이상거래 탐지 모델이 준비되지 않았습니다.",
            },
        )
    return fraud_service


@app.get("/")
def root():
    return {
        "service": "MOVI Fraud Detection API",
        "status": "running",
        "version": SERVICE_VERSION,
        "components": [
            "isolation_forest",
            "rule_engine",
            "risk_score",
        ],
    }


@app.get("/health")
def health_check():
    ready = fraud_service_ready()
    return {
        "status": "ok" if ready else "degraded",
        "service": "fraud-detection",
        "model_loaded": ready,
        "model": "isolation_forest",
        "rule_engine_loaded": (
            fraud_service is not None
            and fraud_service.rule_engine is not None
        ),
        "risk_score_enabled": True,
        "threshold": (
            round(fraud_service.threshold, 6)
            if fraud_service is not None
            and fraud_service.threshold is not None
            else None
        ),
        "error_code": None if ready else "MODEL_RESOURCE_LOAD_FAILED",
    }


@app.get("/ready")
def readiness_check():
    require_fraud_service()
    return {
        "status": "ready",
        "service": "fraud-detection",
    }


@app.post(
    "/api/v1/fraud/detect",
    response_model=FraudDetectionResponse,
)
def detect_fraud(request: FraudDetectionRequest):
    try:
        service = require_fraud_service()
        return FraudDetectionResponse(**service.detect(request))

    except HTTPException:
        raise

    except InvalidTransactionRequest as error:
        logger.warning(
            "FDS request rejected: error_type=%s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_TRANSACTION_REQUEST",
                "message": "거래 탐지 요청이 올바르지 않습니다.",
            },
        ) from error

    except FraudDetectionServiceError as error:
        logger.exception("FDS service failed during inference")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_FDS_ERROR",
                "message": "이상거래 분석 중 내부 오류가 발생했습니다.",
            },
        ) from error

    except Exception as error:
        logger.exception("Unexpected error during FDS inference")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_FDS_ERROR",
                "message": "이상거래 분석 중 내부 오류가 발생했습니다.",
            },
        ) from error
