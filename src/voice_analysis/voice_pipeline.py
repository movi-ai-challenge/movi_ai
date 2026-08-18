from typing import Any, Optional

from .wake_word_detector import WakeWordDetector
from .requirement_analyzer import RequirementAnalyzer
from .requirement_validator import RequirementValidator
from .conversation_context import ConversationContext
from .follow_up_entity_parser import FollowUpEntityParser


class VoicePipeline:
    """
    MOVI 음성 요구사항 분석 통합 Pipeline.

    현재 단계에서는 STT 이후의 텍스트를 입력으로 받는다.

    역할
    ----
    1. Wake Word 감지
    2. 최초 요구사항 분석
    3. Requirement 검증
    4. 누락 정보 관리
    5. Follow-up 답변 해석 및 병합
    """

    def __init__(self):

        self.wake_word_detector = WakeWordDetector()

        self.requirement_analyzer = RequirementAnalyzer()

        self.requirement_validator = RequirementValidator()

        self.context = ConversationContext(
            validator=self.requirement_validator
        )

        self.follow_up_parser = FollowUpEntityParser()

    # ============================================================
    # 최초 명령 처리
    # ============================================================

    def process_text(
        self,
        text: str,
    ) -> dict[str, Any]:
        """
        최초 사용자 명령 처리.

        Example
        -------
        입력:
            "모비야 김민수한테 오만원 보내줘"

        반환:
            {
                "status": "need_more_info",
                "intent": "transfer_money",
                ...
            }
        """

        if not text or not text.strip():
            return self._error_response(
                code="EMPTY_INPUT",
                message="입력된 음성이 없습니다.",
            )

        # --------------------------------------------------------
        # 1. Wake Word 감지
        # --------------------------------------------------------

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

        # 새로운 명령이 들어왔으므로
        # 기존 Context 초기화
        self.context.clear()

        # --------------------------------------------------------
        # 2. GPT 요구사항 분석
        # --------------------------------------------------------

        try:

            analysis = self.requirement_analyzer.analyze(
                command
            )

        except Exception as e:

            return self._error_response(
                code="REQUIREMENT_ANALYSIS_FAILED",
                message="사용자 요청을 분석하지 못했습니다.",
                detail=str(e),
            )

        # --------------------------------------------------------
        # 3. Validator
        # --------------------------------------------------------

        validated = self.requirement_validator.validate(
            analysis
        )

        # --------------------------------------------------------
        # 4. Context 시작
        # --------------------------------------------------------

        self.context.start(
            validated
        )

        # --------------------------------------------------------
        # 5. 결과 생성
        # --------------------------------------------------------

        return self._build_current_response()

    # ============================================================
    # Follow-up 처리
    # ============================================================

    def process_follow_up(
        self,
        text: str,
    ) -> dict[str, Any]:
        """
        누락정보에 대한 사용자의 추가 답변 처리.

        Example
        -------
        현재 missing:
            recipient_bank

        사용자:
            "국민은행으로 해줘"
        """

        if not text or not text.strip():

            return self._error_response(
                code="EMPTY_FOLLOW_UP",
                message="추가 정보가 입력되지 않았습니다.",
            )

        if not self.context.is_active():

            return self._error_response(
                code="NO_ACTIVE_CONTEXT",
                message="진행 중인 요청이 없습니다.",
            )

        field_name = (
            self.context.get_next_missing_field()
        )

        if field_name is None:

            return self._build_current_response()

        # --------------------------------------------------------
        # GPT Follow-up parser
        # --------------------------------------------------------

        try:

            parsed = self.follow_up_parser.parse(
                field_name=field_name,
                user_text=text,
            )

        except Exception as e:

            return self._error_response(
                code="FOLLOW_UP_ANALYSIS_FAILED",
                message="추가 정보를 분석하지 못했습니다.",
                detail=str(e),
            )

        # 추출 실패
        if not parsed.success:

            return {
                "status": "need_more_info",
                "intent": self._current_intent(),
                "missing_fields": (
                    self.context.get_missing_fields()
                ),
                "requested_field": field_name,
                "next_question": (
                    self.context.get_follow_up_question()
                ),
                "message": (
                    "입력한 정보를 정확히 이해하지 못했습니다. "
                    "다시 말씀해주세요."
                ),
            }

        # --------------------------------------------------------
        # Context 업데이트
        # --------------------------------------------------------

        try:

            self.context.update_entity(
                parsed.field_name,
                parsed.value,
            )

        except Exception as e:

            return self._error_response(
                code="CONTEXT_UPDATE_FAILED",
                message="요구사항 정보를 갱신하지 못했습니다.",
                detail=str(e),
            )

        return self._build_current_response()

    # ============================================================
    # 현재 Context 결과
    # ============================================================

    def _build_current_response(
        self,
    ) -> dict[str, Any]:

        requirement = self.context.get_current()

        if requirement is None:

            return self._error_response(
                code="NO_REQUIREMENT",
                message="분석된 요구사항이 없습니다.",
            )

        missing_fields = (
            self.context.get_missing_fields()
        )

        # --------------------------------------------------------
        # 지원하지 않는 요청
        # --------------------------------------------------------

        if requirement.intent == "unknown":

            return {
                "status": "unsupported",
                "intent": "unknown",
                "message": (
                    "현재 지원하지 않는 요청입니다."
                ),
                "requirement": (
                    requirement.model_dump()
                ),
            }

        # --------------------------------------------------------
        # 누락정보 존재
        # --------------------------------------------------------

        if missing_fields:

            return {
                "status": "need_more_info",

                "intent": requirement.intent,

                "entities": (
                    requirement.entities.model_dump()
                ),

                "missing_fields": missing_fields,

                "requested_field": (
                    self.context.get_next_missing_field()
                ),

                "next_question": (
                    self.context.get_follow_up_question()
                ),

                "requirement": (
                    requirement.model_dump()
                ),
            }

        # --------------------------------------------------------
        # Requirement 완성
        # --------------------------------------------------------

        return {
            "status": "ready",

            "intent": requirement.intent,

            "entities": (
                requirement.entities.model_dump()
            ),

            "missing_fields": [],

            "requirement": (
                requirement.model_dump()
            ),
        }

    # ============================================================
    # Helper
    # ============================================================

    def _current_intent(
        self,
    ) -> Optional[str]:

        requirement = self.context.get_current()

        if requirement is None:
            return None

        return requirement.intent

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
            response["error"]["detail"] = detail

        return response

    # ============================================================
    # Context 초기화
    # ============================================================

    def reset(self) -> None:
        """
        현재 진행 중인 사용자 요청 초기화.
        """

        self.context.clear()