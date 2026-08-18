from copy import deepcopy

from .schemas import RequirementAnalysis


class RequirementValidator:
    """
    GPT 분석 결과를 규칙 기반으로 재검증한다.

    목적
    ----
    - LLM의 missing_fields 누락/오판 보정
    - Intent별 필수 Entity를 코드로 강제
    - 금융 요청 실행 전 최소 정보 확보
    """

    REQUIRED_FIELDS = {
        "transfer_money": [
            "recipient_name",
            "recipient_bank",
            "recipient_account",
            "amount",
        ],
        "read_screen": [],
        "check_history": [],
        "check_savings": [],
        "confirm": [],
        "deny": [],
        "cancel": [],
        "unknown": [],
    }

    def validate(
        self,
        analysis: RequirementAnalysis,
    ) -> RequirementAnalysis:
        """
        RequirementAnalysis를 검증하고
        missing_fields를 코드 기준으로 다시 계산한다.
        """

        result = analysis.model_copy(deep=True)

        required_fields = self.REQUIRED_FIELDS.get(
            result.intent,
            [],
        )

        missing_fields: list[str] = []

        for field_name in required_fields:
            value = getattr(
                result.entities,
                field_name,
                None,
            )

            if self._is_missing(value):
                missing_fields.append(field_name)

        result.missing_fields = missing_fields

        return result

    def _is_missing(self, value) -> bool:
        """
        필드가 실제로 비어 있는지 판단.
        """

        if value is None:
            return True

        if isinstance(value, str):
            return not value.strip()

        return False

    def is_complete(
        self,
        analysis: RequirementAnalysis,
    ) -> bool:
        """
        현재 요구사항이 실행 가능한 상태인지 확인.
        """

        validated = self.validate(analysis)

        return len(validated.missing_fields) == 0