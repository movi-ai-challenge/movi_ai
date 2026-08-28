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

            "intent_confidence": (
                getattr(
                    analysis,
                    "intent_confidence",
                    0.5,
                )
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


    # ========================================================
    # 내부 API 계약 경로
    #
    # 백엔드가 세션을 이미 열어둔 상태에서 호출하므로
    # 호출어("모비야")를 요구하지 않는다. 재질문 답변("오만 원")에는
    # 호출어가 없고, 요구하면 전부 ignored 로 떨어진다.
    # 호출어가 붙어 있으면 제거만 한다.
    # ========================================================

    def analyze_command(
        self,
        text: str,
    ):
        """
        호출어 유무와 무관하게 명령 본문을 분석한다.

        Returns:
            RequirementAnalysis
        """

        wake_result = (
            self.wake_word_detector.detect(text)
        )

        if wake_result.activated and wake_result.command:
            command = wake_result.command
        else:
            command = text.strip()

        if not command:
            raise ValueError(
                "분석할 명령이 비어 있습니다."
            )

        return (
            self.requirement_analyzer.analyze(command)
        )


    def parse_follow_up(
        self,
        *,
        field_name: str,
        text: str,
    ):
        """
        재질문 답변에서 지정된 필드 하나만 추출한다.

        Returns:
            FollowUpEntityResult
        """

        return self.follow_up_parser.parse(
            field_name=field_name,
            user_text=text,
        )
