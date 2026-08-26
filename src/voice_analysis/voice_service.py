from __future__ import annotations

from typing import Any

from .wake_word_detector import (
    WakeWordDetector,
)

from .requirement_analyzer import (
    RequirementAnalyzer,
)

from .follow_up_entity_parser import (
    FollowUpEntityParser,
)


class VoiceAnalysisService:
    """
    MOVI Voice AI Service.

    담당
    ----
    - Wake Word
    - Intent 분석
    - Entity 추출
    - Follow-up Entity 추출

    담당하지 않음
    ------------
    - 빈 값 판단
    - 필수 필드 판단
    - DB 조회
    - 금융 기능 실행

    위 기능은 Spring Backend 담당.
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


    # ========================================================
    # 최초 명령
    # ========================================================

    def analyze(
        self,
        text: str,
    ) -> dict[str, Any]:

        if not text or not text.strip():

            return {
                "status": "error",
                "message": (
                    "입력된 음성이 없습니다."
                ),
            }


        # ----------------------------------------------------
        # Wake Word
        # ----------------------------------------------------

        wake_result = (
            self.wake_word_detector.detect(
                text
            )
        )


        if not wake_result.activated:

            return {
                "status": "ignored",
                "message": (
                    "호출어가 감지되지 않았습니다."
                ),
            }


        command = wake_result.command


        if not command:

            return {
                "status": "awaiting_command",
                "message": (
                    "무엇을 도와드릴까요?"
                ),
            }


        # ----------------------------------------------------
        # Intent / Entity
        # ----------------------------------------------------

        analysis = (
            self.requirement_analyzer.analyze(
                command
            )
        )


        try:

            entities = (
                analysis
                .entities
                .model_dump()
            )

        except AttributeError:

            entities = dict(
                analysis.entities
            )


        return {

            "status": (
                "unsupported"
                if analysis.intent == "unknown"
                else "analyzed"
            ),

            "intent": (
                analysis.intent
            ),

            "transcript": (
                command
            ),

            "entities": (
                entities
            ),
        }


    # ========================================================
    # Follow-up
    # ========================================================

    def analyze_follow_up(
        self,
        *,
        requested_field: str,
        text: str,
        entities: dict[str, Any],
    ) -> dict[str, Any]:

        parsed = (
            self.follow_up_parser.parse(

                field_name=(
                    requested_field
                ),

                user_text=text,
            )
        )


        updated_entities = (
            entities.copy()
        )


        if parsed.success:

            updated_entities[
                parsed.field_name
            ] = parsed.value


        return {

            "status": (
                "analyzed"
                if parsed.success
                else "parse_failed"
            ),

            "transcript": text,

            "entities": (
                updated_entities
            ),
        }