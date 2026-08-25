from __future__ import annotations

from typing import Any, Optional

from .wake_word_detector import WakeWordDetector
from .requirement_analyzer import RequirementAnalyzer
from .follow_up_entity_parser import FollowUpEntityParser

from .request_mapper import (
    create_request_id,
    build_backend_request,
)

from .backend_client import (
    send_voice_command,
    BackendClientError,
)


class VoicePipeline:
    """
    MOVI 음성 요구사항 분석 Pipeline.

    책임
    ----
    Python:
        - Wake Word
        - Intent 분석
        - Entity 추출
        - Follow-up Entity 분석
        - Backend 전송

    Backend:
        - 필수값 검증
        - 빈 값 판단
        - 다음 질문 결정
        - 실제 금융 기능 실행
    """

    def __init__(self):

        self.wake_word_detector = (
            WakeWordDetector()
        )

        self.requirement_analyzer = (
            RequirementAnalyzer()
        )

        self.follow_up_parser = (
            FollowUpEntityParser()
        )

        # 하나의 Multi-turn 요청에서 유지
        self.request_id: Optional[str] = None

        self.current_intent: Optional[str] = None

        self.current_entities: dict[str, Any] = {}

        # Backend가 다음에 요구한 Field
        self.requested_field: Optional[str] = None


    # ============================================================
    # 최초 사용자 명령
    # ============================================================

    def process_text(
        self,
        text: str,
    ) -> dict[str, Any]:

        if not text or not text.strip():

            return self._error_response(
                code="EMPTY_INPUT",
                message="입력된 음성이 없습니다.",
            )


        # ========================================================
        # 1. Wake Word
        # ========================================================

        wake_result = (
            self.wake_word_detector.detect(
                text
            )
        )


        if not wake_result.activated:

            return {
                "status": "ignored",
                "reason": "wake_word_not_detected",
                "message": "호출어가 감지되지 않았습니다.",
            }


        command = wake_result.command


        if not command:

            return {
                "status": "awaiting_command",
                "message": "무엇을 도와드릴까요?",
            }


        # ========================================================
        # 2. 새로운 Request 시작
        # ========================================================

        self.reset()

        self.request_id = (
            create_request_id()
        )


        # ========================================================
        # 3. Requirement Analyzer
        # ========================================================

        try:

            analysis = (
                self.requirement_analyzer.analyze(
                    command
                )
            )

        except Exception as error:

            return self._error_response(
                code="REQUIREMENT_ANALYSIS_FAILED",
                message="사용자 요청을 분석하지 못했습니다.",
                detail=str(error),
            )


        # ========================================================
        # 4. Intent / Entity 저장
        # ========================================================

        self.current_intent = (
            analysis.intent
        )

        self.current_entities = (
            analysis.entities.model_dump()
        )


        # ========================================================
        # unknown
        # ========================================================

        if self.current_intent == "unknown":

            return {
                "status": "unsupported",
                "intent": "unknown",
                "message": "현재 지원하지 않는 요청입니다.",
            }


        # ========================================================
        # 5. Backend로 무조건 전달
        # ========================================================

        return self._send_to_backend(
            transcript=command
        )


    # ============================================================
    # Follow-up 처리
    # ============================================================

    def process_follow_up(
        self,
        text: str,
    ) -> dict[str, Any]:
        """
        Backend가 requested_field를 반환한 이후
        사용자가 추가로 말한 내용을 처리한다.

        Example
        -------
        Backend:
            requested_field = recipient_bank

        사용자:
            "국민은행이야"

        Python:
            recipient_bank = 국민은행

        이후 동일 request_id로 Backend에 재전송한다.
        """

        if not text or not text.strip():

            return self._error_response(
                code="EMPTY_FOLLOW_UP",
                message="추가 정보가 입력되지 않았습니다.",
            )


        if self.request_id is None:

            return self._error_response(
                code="NO_ACTIVE_REQUEST",
                message="진행 중인 요청이 없습니다.",
            )


        # ========================================================
        # Backend가 특정 Field를 요청한 경우
        # ========================================================

        if self.requested_field:

            try:

                parsed = (
                    self.follow_up_parser.parse(
                        field_name=self.requested_field,
                        user_text=text,
                    )
                )

            except Exception as error:

                return self._error_response(
                    code="FOLLOW_UP_ANALYSIS_FAILED",
                    message="추가 정보를 분석하지 못했습니다.",
                    detail=str(error),
                )


            # ====================================================
            # Entity 추출 성공
            # ====================================================

            if parsed.success:

                self.current_entities[
                    parsed.field_name
                ] = parsed.value


        # ========================================================
        # requested_field가 없을 경우
        #
        # Backend가 confirm / deny 등의 응답을 기다리는
        # 구조로 확장할 수 있다.
        #
        # 현재는 그대로 transcript만 다시 전달한다.
        # ========================================================


        return self._send_to_backend(
            transcript=text
        )


    # ============================================================
    # Backend 전송
    # ============================================================

    def _send_to_backend(
        self,
        *,
        transcript: str,
    ) -> dict[str, Any]:

        if self.request_id is None:

            return self._error_response(
                code="NO_REQUEST_ID",
                message="Request ID가 없습니다.",
            )


        if self.current_intent is None:

            return self._error_response(
                code="NO_INTENT",
                message="Intent가 없습니다.",
            )


        # ========================================================
        # Request JSON 생성
        # ========================================================

        backend_request = (
            build_backend_request(
                request_id=self.request_id,
                transcript=transcript,
                intent=self.current_intent,
                entities=self.current_entities,
            )
        )


        # ========================================================
        # Backend POST
        # ========================================================

        try:

            backend_response = (
                send_voice_command(
                    backend_request
                )
            )

        except BackendClientError as error:

            return self._error_response(
                code="BACKEND_REQUEST_FAILED",
                message="Backend 서버 요청에 실패했습니다.",
                detail=str(error),
            )


        # ========================================================
        # Backend가 요구하는 다음 Field 저장
        # ========================================================

        self.requested_field = (
            backend_response.get(
                "requested_field"
            )
        )


        # ========================================================
        # Backend 응답 반환
        # ========================================================

        return {
            "status": backend_response.get(
                "status",
                "success",
            ),

            "request_id": self.request_id,

            "intent": self.current_intent,

            "entities": self.current_entities.copy(),

            "requested_field": self.requested_field,

            "message": backend_response.get(
                "message"
            ),

            "backend_request": backend_request,

            "backend_response": backend_response,
        }


    # ============================================================
    # 현재 상태 확인
    # ============================================================

    def get_current_state(
        self,
    ) -> dict[str, Any]:

        return {
            "request_id": self.request_id,
            "intent": self.current_intent,
            "entities": self.current_entities.copy(),
            "requested_field": self.requested_field,
        }


    # ============================================================
    # Error
    # ============================================================

    def _error_response(
        self,
        code: str,
        message: str,
        detail: Optional[str] = None,
    ) -> dict[str, Any]:

        response = {
            "status": "error",

            "error": {
                "code": code,
                "message": message,
            }
        }


        if detail:

            response[
                "error"
            ][
                "detail"
            ] = detail


        return response


    # ============================================================
    # Reset
    # ============================================================

    def reset(
        self,
    ) -> None:

        self.request_id = None

        self.current_intent = None

        self.current_entities = {}

        self.requested_field = None