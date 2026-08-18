from typing import Optional

from .schemas import (
    RequirementAnalysis,
    RequirementEntities,
)
from .requirement_validator import RequirementValidator


class ConversationContext:
    """
    사용자와의 Multi-turn 금융 명령 Context 관리.

    역할
    ----
    1. 최초 RequirementAnalysis 저장
    2. 현재 누락 필드 확인
    3. 누락 필드에 대한 질문 생성
    4. 사용자 추가 답변을 기존 Entity에 병합
    5. 요구사항 완성 여부 판단
    """

    # --------------------------------------------------------
    # 누락 필드별 질문 문구
    # --------------------------------------------------------

    FOLLOW_UP_MESSAGES = {
        "recipient_name":
            "받는 분의 이름을 말씀해주세요.",

        "recipient_bank":
            "받는 분의 은행을 말씀해주세요.",

        "recipient_account":
            "받는 분의 계좌번호를 말씀해주세요.",

        "amount":
            "송금할 금액을 말씀해주세요.",
    }

    def __init__(
        self,
        validator: RequirementValidator,
    ):
        self.validator = validator

        self.current_requirement: Optional[
            RequirementAnalysis
        ] = None

    # ========================================================
    # Context 시작
    # ========================================================

    def start(
        self,
        requirement: RequirementAnalysis,
    ) -> RequirementAnalysis:
        """
        새로운 사용자 명령 Context 시작.
        """

        validated = self.validator.validate(
            requirement
        )

        self.current_requirement = validated

        return validated

    # ========================================================
    # 현재 상태
    # ========================================================

    def get_current(
        self,
    ) -> Optional[RequirementAnalysis]:
        """
        현재 진행 중인 Requirement 반환.
        """

        return self.current_requirement

    def is_active(self) -> bool:
        """
        현재 진행 중인 명령이 있는지 확인.
        """

        return self.current_requirement is not None

    def is_complete(self) -> bool:
        """
        현재 요구사항이 완성되었는지 확인.
        """

        if self.current_requirement is None:
            return False

        return self.validator.is_complete(
            self.current_requirement
        )

    # ========================================================
    # 누락 정보
    # ========================================================

    def get_missing_fields(
        self,
    ) -> list[str]:
        """
        현재 누락 필드 목록 반환.
        """

        if self.current_requirement is None:
            return []

        validated = self.validator.validate(
            self.current_requirement
        )

        self.current_requirement = validated

        return validated.missing_fields

    def get_next_missing_field(
        self,
    ) -> Optional[str]:
        """
        다음으로 요청할 누락 필드 반환.

        현재는 validator의 순서를 그대로 사용한다.
        """

        missing_fields = self.get_missing_fields()

        if not missing_fields:
            return None

        return missing_fields[0]

    # ========================================================
    # Follow-up 질문
    # ========================================================

    def get_follow_up_question(
        self,
    ) -> Optional[str]:
        """
        다음 누락 필드에 대한 사용자 질문 반환.
        """

        field_name = self.get_next_missing_field()

        if field_name is None:
            return None

        return self.FOLLOW_UP_MESSAGES.get(
            field_name,
            "추가 정보를 말씀해주세요.",
        )

    # ========================================================
    # Entity 병합
    # ========================================================

    def update_entity(
        self,
        field_name: str,
        value,
    ) -> RequirementAnalysis:
        """
        특정 Entity 값을 현재 Requirement에 병합한다.

        Example
        -------
        update_entity(
            "recipient_bank",
            "국민은행"
        )
        """

        if self.current_requirement is None:
            raise RuntimeError(
                "진행 중인 대화 Context가 없습니다."
            )

        if not hasattr(
            self.current_requirement.entities,
            field_name,
        ):
            raise ValueError(
                f"지원하지 않는 Entity입니다: {field_name}"
            )

        setattr(
            self.current_requirement.entities,
            field_name,
            value,
        )

        # 병합 후 다시 검증
        self.current_requirement = (
            self.validator.validate(
                self.current_requirement
            )
        )

        return self.current_requirement

    # ========================================================
    # Follow-up 응답 병합
    # ========================================================

    def apply_follow_up(
        self,
        value,
    ) -> RequirementAnalysis:
        """
        현재 가장 우선순위가 높은 누락 필드에
        사용자 응답을 자동 병합한다.

        예:
            현재 missing = recipient_bank

            사용자의 추가 답변 = "국민은행"

            → recipient_bank = "국민은행"
        """

        field_name = self.get_next_missing_field()

        if field_name is None:
            raise RuntimeError(
                "현재 누락된 정보가 없습니다."
            )

        return self.update_entity(
            field_name,
            value,
        )

    # ========================================================
    # Context 종료
    # ========================================================

    def clear(self) -> None:
        """
        현재 대화 Context 초기화.
        """

        self.current_requirement = None