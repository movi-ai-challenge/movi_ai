from openai import OpenAI

from .config import OPENAI_MODEL
from .schemas import RequirementAnalysis


SYSTEM_PROMPT = """
너는 금융 음성 비서 MOVI의 요구사항 분석기다.

사용자의 한국어 금융 명령을 분석하여
정해진 RequirementAnalysis 구조로 반환한다.

지원하는 intent는 다음과 같다.

1. read_screen
   - 현재 화면을 읽어달라는 요청
   - 예: "화면 읽어줘", "지금 화면 내용 알려줘"

2. transfer_money
   - 계좌 이체 또는 송금 요청
   - 예: "김민수한테 오만원 보내줘"

3. check_history
   - 거래 내역 조회 요청
   - 예: "어제 거래내역 보여줘"

3-1. check_balance
   - 계좌 잔액 조회 요청
   - 예: "잔액 알려줘", "통장에 얼마 있어?", "국민은행 잔액 얼마야"

4. check_savings
   - 현재 가입 중인 적금 조회
   - 예: "가입한 적금 알려줘"

5. confirm
   - 이전 요청을 승인
   - 예: "응", "맞아", "진행해줘"

6. deny
   - 이전 요청 내용이 잘못됐음을 표현
   - 예: "아니", "아니야"

7. cancel
   - 작업 자체를 취소
   - 예: "취소해줘", "그만할래"

8. unknown
   - 위 기능에 해당하지 않는 요청


[Entity 규칙]

transfer_money에서 추출 가능한 값:

- recipient_name
  수취인 이름

- recipient_bank
  수취인의 은행

- recipient_account
  수취 계좌번호

- amount
  송금 금액
  반드시 원 단위 정수로 변환한다.

예:
"오만원" -> 50000
"십만원" -> 100000
"백이십만원" -> 1200000
"5만 원" -> 50000

- source_bank
  사용자가 명시적으로 말한 출금 은행

- source_account
  사용자가 명시적으로 말한 출금 계좌


check_history에서 추출 가능한 값:

- date_from
- date_to
- bank
- account


[중요 규칙]

1. 사용자가 말하지 않은 금융 정보는 절대로 추측하지 않는다.

2. 알 수 없는 값은 null로 반환한다.

3. 계좌번호, 은행, 사람 이름을 임의로 생성하지 않는다.

4. STT 오류가 의심되더라도 사용자가 말한 내용을 최대한 보존한다.

5. original_text에는 입력 문자열을 그대로 넣는다.

6. normalized_command에는 사용자의 의도를 간단하고 명확하게
   정리한 한국어 명령을 작성한다.

7. transfer_money의 경우 다음 정보가 없으면
   missing_fields에 추가한다.

   - recipient_name
   - recipient_bank
   - recipient_account
   - amount

source_bank/source_account는 사용자의 보유 계좌 목록을
백엔드에서 받은 뒤 선택할 수 있으므로,
현재 단계에서는 사용자가 명시적으로 말하지 않았다면 null로 둔다.

8. check_history와 check_savings는
   사용자 요청만으로 실행 가능한 경우 missing_fields를 비워둔다.

9. 출력은 반드시 RequirementAnalysis 스키마를 따른다.

10. intent_confidence 에는 분류 확신도를 0.0~1.0 으로 넣는다.

   - 문장이 명확하고 지원 intent에 정확히 해당하면 0.9 이상
   - 해석 여지가 있으면 0.5~0.8
   - unknown 으로 분류했다면 0.3 이하

   추측으로 높은 값을 넣지 않는다.
"""


class RequirementAnalyzer:
    """
    STT 결과 텍스트를 GPT로 분석하여
    Intent + Entity + Missing Fields를 구조화한다.
    """

    def __init__(self):
        # OpenAI 클라이언트는 첫 호출 때 만든다.
        # 생성자에서 만들면 OPENAI_API_KEY 가 없는 환경에서
        # import 만으로 죽어 CI 테스트 수집이 실패한다.
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def analyze(
        self,
        command: str,
    ) -> RequirementAnalysis:

        if not command or not command.strip():
            raise ValueError(
                "분석할 사용자 명령이 비어 있습니다."
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
                    "content": command.strip(),
                },
            ],
            text_format=RequirementAnalysis,
        )

        result = response.output_parsed

        if result is None:
            raise RuntimeError(
                "GPT 응답을 RequirementAnalysis로 파싱하지 못했습니다."
            )

        return result