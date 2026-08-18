| 순서     | 태스크                            | 핵심 내용                                                  | 상태   |
| ------ | ------------------------------ | ------------------------------------------------------ | ---- |
| 1      | Google Streaming STT           | 실시간 음성 → 텍스트                                           | ✅ 완료 |
| 2      | Wake Word 감지                   | `모비야` 감지                                               | ✅ 완료 |
| 3      | Requirement Schema             | Intent / Entity 구조 정의                                  | ✅ 완료 |
| 4      | GPT 요구사항 분석                    | Intent + Entity + 금액/은행 등 구조화                          | ✅ 완료 |
| 5      | Requirement Validator          | 누락 필드 코드 검증                                            | ✅ 완료 |
| 6      | Conversation Context           | 누락정보 상태 유지                                             | ✅ 완료 |
| 7      | Follow-up Entity Parser        | 추가 답변을 현재 누락 필드에 맞게 해석                                 | ✅ 완료 |
| **8**  | **출금 계좌 선택 흐름**                | 본인 보유 계좌 목록에서 `source_bank/source_account` 선택          | ⏳ 다음 |
| **9**  | **Confirmation Builder**       | 최종 요구사항을 사용자 확인 문장으로 생성                                | ⏳    |
| **10** | **TTS**                        | 확인 문장 및 조회 결과를 음성으로 변환                                 | ⏳    |
| **11** | **Confirm / Deny / Cancel 처리** | `"응"`, `"아니"`, `"취소"`에 따라 상태 전이                        | ⏳    |
| **12** | **Voice Pipeline 통합**          | STT → GPT → Validator → Follow-up → Confirmation 전체 연결 | ⏳    |
| **13** | **기능 0 화면 읽기 연결**              | `read_screen` 결과와 프런트 화면 텍스트 연결 규격                     | ⏳    |
| **14** | **기능 2 거래내역 조회 연결**            | 기간/계좌 조건 → Spring 조회 요청 구조                             | ⏳    |
| **15** | **기능 3 적금 조회 연결**              | `check_savings` → Spring 적금 조회 요청                      | ⏳    |
| **16** | **Fraud Detection 연결**         | 송금 확정 직전 Risk Score / 이상거래 판단 호출                       | ⏳    |
| **17** | **Node.js / Spring 통합 규격**     | WebSocket + REST JSON 계약 정의                            | ⏳    |
| **18** | **End-to-End 테스트**             | 실제 음성부터 최종 요청까지 시나리오 검증                                | ⏳    |
