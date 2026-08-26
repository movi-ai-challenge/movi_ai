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

    Python 담당
    -----------
    1. Wake Word 감지
    2. Intent 분석
    3. Entity 추출
    4. Backend Request 생성
    5. Backend Response 처리
    6. Follow-up Entity 추출
    7. Multi-turn Context 유지

    Backend 담당
    --------------
    1. 필수 Entity 판단
    2. 빈 값 판단
    3. 다음 요청 Field 결정
    4. 실제 금융 기능 처리
    """

    def __init__(self):

        self.wake_word_detector = WakeWordDetector()

        self.requirement_analyzer = RequirementAnalyzer()

        self.follow_up_parser = FollowUpEntityParser()

        # ====================================================
        # 현재 진행 중인 요청 상태
        # ====================================================

        self.request_id: Optional[str] = None

        self.current_intent: Optional[str] = None

        self.current_entities: dict[str, Any] = {}

        # Backend가 추가로 요구한 Entity
        self.requested_field: Optional[str] = None


    # ============================================================
    # 최초 명령 처리
    # ============================================================

    def process_text(
        self,
        text: str,
    ) -> dict[str, Any]:
        """
        최초 사용자 음성 명령 처리.

        Example
        -------
        "모비야 김민수한테 오만원 보내줘"
        """

        if not text or not text.strip():

            return self._error_response(
                code="EMPTY_INPUT",
                message="입력된 음성이 없습니다.",
            )


        # ========================================================
        # 1. Wake Word 감지
        # ========================================================

        wake_result = self.wake_word_detector.detect(
            text
        )


        if not wake_result.activated:

            return {
                "status": "ignored",
                "reason": "wake_word_not_detected",
                "message": "호출어가 감지되지 않았습니다.",
            }


        command = wake_result.command


        # "모비야"만 말한 경우
        if not command:

            return {
                "status": "awaiting_command",
                "message": "무엇을 도와드릴까요?",
            }


        # ========================================================
        # 2. 새 요청 시작
        # ========================================================

        self.reset()

        self.request_id = create_request_id()


        # ========================================================
        # 3. Requirement 분석
        # ========================================================

        try:

            analysis = self.requirement_analyzer.analyze(
                command
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

        self.current_intent = analysis.intent


        try:

            self.current_entities = (
                analysis.entities.model_dump()
            )

        except AttributeError:

            self.current_entities = dict(
                analysis.entities
            )


        # ========================================================
        # 5. 지원하지 않는 Intent
        # ========================================================

        if self.current_intent == "unknown":

            return {
                "status": "unsupported",
                "intent": "unknown",
                "message": "현재 지원하지 않는 요청입니다.",
            }


        # ========================================================
        # 6. Backend 전송
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
        Backend에서 requested_field가 반환된 이후
        사용자의 추가 음성을 처리한다.

        Example
        -------
        requested_field:
            recipient_bank

        user:
            "국민은행이야"
        """

        if not text or not text.strip():

            return self._error_response(
                code="EMPTY_FOLLOW_UP",
                message="추가 정보가 입력되지 않았습니다.",
            )


        # 진행 중인 요청 없음
        if self.request_id is None:

            return self._error_response(
                code="NO_ACTIVE_REQUEST",
                message="진행 중인 요청이 없습니다.",
            )


        # ========================================================
        # Backend가 특정 Field를 요구한 경우
        # ========================================================

        if self.requested_field:

            try:

                parsed = self.follow_up_parser.parse(
                    field_name=self.requested_field,
                    user_text=text,
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
        # Backend 재전송
        #
        # 같은 request_id 유지
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
        # 1. Backend Request 생성
        # ========================================================

        backend_request = build_backend_request(

            request_id=self.request_id,

            transcript=transcript,

            intent=self.current_intent,

            entities=self.current_entities,
        )


        # ========================================================
        # 2. Backend POST
        # ========================================================

        try:

            backend_response = send_voice_command(
                backend_request
            )

        except BackendClientError as error:

            return {
                "status": "backend_unavailable",

                "request_id": self.request_id,

                "intent": self.current_intent,

                "entities": self.current_entities.copy(),

                "backend_request": backend_request,

                "error": {
                    "code": "BACKEND_REQUEST_FAILED",
                    "message": str(error),
                },
            }


        # ========================================================
        # 3. Backend requested_field 저장
        # ========================================================

        self.requested_field = (
            backend_response.get(
                "requested_field"
            )
        )


        # ========================================================
        # 4. Backend 완료 시 Context 종료 여부
        # ========================================================

        backend_status = backend_response.get(
            "status",
            "success",
        )


        result = {

            "status": backend_status,

            "request_id": self.request_id,

            "intent": self.current_intent,

            "entities": self.current_entities.copy(),

            "requested_field": self.requested_field,

            "message": backend_response.get(
                "message"
            ),

            "data": backend_response.get(
                "data"
            ),

            "backend_request": backend_request,

            "backend_response": backend_response,
        }


        # completed면 요청 종료
        if backend_status == "completed":

            self.reset()


        return result


    # ============================================================
    # 현재 상태 확인
    # ============================================================

    def get_current_state(
        self,
    ) -> dict[str, Any]:

        return {

            "request_id": self.request_id,

            "intent": self.current_intent,

            "entities": (
                self.current_entities.copy()
            ),

            "requested_field": (
                self.requested_field
            ),
        }


    # ============================================================
    # Error Response
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

            response["error"]["detail"] = (
                detail
            )


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