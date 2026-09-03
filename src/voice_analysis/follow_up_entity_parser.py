from openai import OpenAI

from .config import OPENAI_MAX_RETRIES, OPENAI_MODEL, OPENAI_TIMEOUT_SECONDS
from .schemas import FollowUpEntityResult


SYSTEM_PROMPT = """
너는 금융 음성 비서 MOVI의 추가 정보 분석기다.

사용자가 이전 금융 요청에서 누락된 정보에 대해
추가로 답변한 짧은 한국어 문장을 분석한다.

현재 시스템이 요구하는 field_name 하나가 함께 제공된다.

반드시 그 field에 해당하는 값만 추출한다.


[지원 field]

recipient_name
- 받는 사람 이름
- 예: "김민수야" -> "김민수"

recipient_bank
- 받는 사람의 은행
- 예: "국민은행이야" -> "국민은행"
- 예: "신한으로 해줘" -> "신한은행"

recipient_account
- 받는 사람의 계좌번호
- 숫자만 추출한다.
- 공백이나 '-'는 제거한다.
- 예:
  "123-456-7890이야"
  -> "1234567890"

amount
- 송금 금액
- 반드시 원 단위 정수로 변환한다.
- 예:
  "오만원"
  -> 50000

  "십만원이야"
  -> 100000

  "5만 원"
  -> 50000


[중요 규칙]

1. 요청받은 field_name에 해당하는 값만 추출한다.

2. 사용자가 명확하게 말하지 않은 값은 추측하지 않는다.

3. 값을 추출할 수 없으면:
   success = false
   value = null

4. field_name은 입력받은 값을 그대로 반환한다.

5. 계좌번호를 임의로 생성하지 않는다.

6. 은행명은 가능하면 공식적인 이름으로 정규화한다.

예:
"국민" -> "국민은행"
"신한" -> "신한은행"
"우리" -> "우리은행"
"하나" -> "하나은행"

7. original_text에는 사용자의 답변을 그대로 넣는다.
"""


class FollowUpEntityParser:
    """
    누락 Entity에 대한 사용자 추가 답변을
    GPT로 구조화한다.
    """

    SUPPORTED_FIELDS = {
        "recipient_name",
        "recipient_bank",
        "recipient_account",
        "amount",
    }

    def __init__(self):
        # OpenAI 클라이언트는 첫 호출 때 만든다.
        # 생성자에서 만들면 OPENAI_API_KEY 가 없는 환경에서
        # import 만으로 죽어 CI 테스트 수집이 실패한다.
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                timeout=OPENAI_TIMEOUT_SECONDS,
                max_retries=OPENAI_MAX_RETRIES,
            )
        return self._client

    def parse(
        self,
        field_name: str,
        user_text: str,
    ) -> FollowUpEntityResult:
        """
        현재 필요한 field에 맞춰
        사용자의 추가 답변을 분석한다.
        """

        if field_name not in self.SUPPORTED_FIELDS:
            raise ValueError(
                f"지원하지 않는 Follow-up field입니다: {field_name}"
            )

        if not user_text or not user_text.strip():
            raise ValueError(
                "사용자 추가 응답이 비어 있습니다."
            )

        response = self.client.responses.parse(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"현재 필요한 field_name: {field_name}\n"
                        f"사용자 답변: {user_text.strip()}"
                    ),
                },
            ],
            text_format=FollowUpEntityResult,
        )

        result = response.output_parsed

        if result is None:
            raise RuntimeError(
                "Follow-up 응답을 파싱하지 못했습니다."
            )

        # 모델이 field_name을 바꾸는 경우 방어
        result.field_name = field_name

        return result
