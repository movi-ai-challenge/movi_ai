from __future__ import annotations

from dataclasses import dataclass

from google.api_core.client_options import ClientOptions
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech

from .config import (
    PROJECT_ID,
    RECOGNIZER_PATH,
    STT_API_ENDPOINT,
    STT_MODEL,
    LANGUAGE_CODE,
)


# ============================================================
# 오디오 제약
#
# ai-api-contract.md 2.1:
#   WebM/Opus 또는 WAV, 최대 5MB / 15초
# ============================================================

MAX_AUDIO_BYTES = 5 * 1024 * 1024


class SttError(RuntimeError):
    """
    STT 처리 실패.

    code 는 ai-api-contract.md 2.7 의 내부 코드를 그대로 쓴다.
    """

    def __init__(
        self,
        code: str,
        message: str,
    ):
        super().__init__(message)
        self.code = code


@dataclass
class SttResult:
    transcript: str
    confidence: float


class SttBatchService:
    """
    Google Cloud Speech-to-Text V2 배치 인식.

    스트리밍(stt_stream_service.py)과 달리 업로드된 오디오 파일
    한 건을 그대로 인식한다. 백엔드는 multipart 로 파일을 보내므로
    내부 Voice API 는 이쪽을 쓴다.
    """

    def __init__(self):

        if not PROJECT_ID:
            raise SttError(
                "STT_PROVIDER_ERROR",
                "GOOGLE_CLOUD_PROJECT 환경변수가 설정되지 않았습니다.",
            )

        client_options = ClientOptions(
            api_endpoint=STT_API_ENDPOINT
        )

        self.client = speech_v2.SpeechClient(
            client_options=client_options
        )

    # ========================================================
    # 인식
    # ========================================================

    def transcribe(
        self,
        audio_bytes: bytes,
    ) -> SttResult:
        """
        오디오 bytes 를 텍스트로 변환한다.

        컨테이너 포맷(WebM/Opus, WAV 등)은 Google 의
        auto decoding 에 맡긴다. 프런트가 어떤 코덱으로 녹음할지
        확정되지 않았고, 직접 판별하면 포맷이 늘 때마다 코드를 고쳐야 한다.
        """

        if not audio_bytes:
            raise SttError(
                "UNSUPPORTED_AUDIO_FORMAT",
                "오디오 데이터가 비어 있습니다.",
            )

        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise SttError(
                "AUDIO_TOO_LONG",
                f"오디오가 최대 크기({MAX_AUDIO_BYTES} bytes)를 초과했습니다.",
            )

        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=(
                cloud_speech.AutoDetectDecodingConfig()
            ),
            model=STT_MODEL,
            language_codes=[LANGUAGE_CODE],
        )

        request = cloud_speech.RecognizeRequest(
            recognizer=RECOGNIZER_PATH,
            config=config,
            content=audio_bytes,
        )

        try:
            response = self.client.recognize(request=request)

        except Exception as error:
            raise SttError(
                "STT_PROVIDER_ERROR",
                f"Google STT 호출 실패: {error}",
            ) from error

        return self._collect(response)

    # ========================================================
    # 결과 병합
    # ========================================================

    def _collect(
        self,
        response,
    ) -> SttResult:
        """
        Google 은 발화를 여러 result 로 쪼개서 준다.
        transcript 는 이어붙이고 confidence 는 평균을 낸다.
        """

        parts: list[str] = []
        confidences: list[float] = []

        for result in response.results:

            if not result.alternatives:
                continue

            alternative = result.alternatives[0]

            text = alternative.transcript.strip()

            if not text:
                continue

            parts.append(text)

            # chirp 계열은 confidence 를 주지 않는 경우가 있다.
            # 0.0 을 평균에 넣으면 신뢰도가 실제보다 낮게 나오므로 제외한다.
            if alternative.confidence > 0.0:
                confidences.append(
                    float(alternative.confidence)
                )

        transcript = " ".join(parts).strip()

        if not transcript:
            raise SttError(
                "EMPTY_TRANSCRIPT",
                "음성에서 텍스트를 인식하지 못했습니다.",
            )

        if confidences:
            confidence = sum(confidences) / len(confidences)
        else:
            # 모델이 confidence 를 주지 않은 경우.
            # 백엔드 계약상 sttConfidence 는 null 이 될 수 없어
            # 중립값을 넣고, 백엔드의 신뢰도 정책이 이를 낮은 값으로
            # 취급하지 않도록 0.5 를 쓴다.
            confidence = 0.5

        return SttResult(
            transcript=transcript,
            confidence=min(max(confidence, 0.0), 1.0),
        )
