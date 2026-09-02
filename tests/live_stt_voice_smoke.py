"""실제 마이크 → Google STT → GPT 음성 분석 통합 스모크 테스트.

외부 API와 로컬 마이크를 사용하므로 자동 단위 테스트 검색 대상에서
제외한다. 프로젝트 루트에서 다음처럼 직접 실행한다.

    python tests/live_stt_voice_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        """python-dotenv가 없을 때 셸 환경변수만 사용한다."""

        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live_voice_smoke import CASES, validate  # noqa: E402


DEFAULT_TIMEOUT_SECONDS = 30.0


def environment_error() -> str | None:
    missing: list[str] = []

    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        missing.append("GOOGLE_CLOUD_PROJECT")

    if not missing:
        return None

    return "필수 환경변수가 없습니다: " + ", ".join(missing)


def safe_error_message(error: Exception) -> str:
    """키·경로 등 외부 SDK 상세값을 출력하지 않고 원인만 안내한다."""

    error_type = type(error).__name__

    if error_type in {"AuthenticationError", "Unauthenticated"}:
        return f"{error_type}: OpenAI 또는 Google 인증정보를 확인하세요."
    if error_type in {"PermissionDenied", "Forbidden"}:
        return f"{error_type}: 프로젝트 API 권한을 확인하세요."
    if error_type in {"PortAudioError"}:
        return f"{error_type}: 마이크 장치와 macOS 마이크 권한을 확인하세요."
    if isinstance(error, TimeoutError):
        return "TimeoutError: 제한 시간 안에 STT 최종 결과를 받지 못했습니다."

    return f"{error_type}: 통합 처리 중 오류가 발생했습니다."


async def capture_final_transcript(
    *,
    stt_service: Any,
    microphone_type: type,
    timeout_seconds: float,
) -> dict[str, Any]:
    """마이크 스트림에서 첫 번째 유효한 STT final 결과를 반환한다."""

    async with microphone_type() as microphone:

        async def wait_for_final() -> dict[str, Any]:
            async for result in stt_service.recognize(
                microphone.audio_generator()
            ):
                result_type = result.get("type")
                text = str(result.get("text", "")).strip()

                if result_type == "interim":
                    print(f"\r인식 중: {text:<50}", end="", flush=True)
                    continue

                if result_type == "final" and text:
                    print("\r" + " " * 65 + "\r", end="", flush=True)
                    return result

            raise RuntimeError("STT 스트림이 최종 결과 없이 종료되었습니다.")

        return await asyncio.wait_for(
            wait_for_final(),
            timeout=timeout_seconds,
        )


async def run() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    error_message = environment_error()
    if error_message:
        print(f"[설정 오류] {error_message}")
        print("MOVI/.env 또는 현재 셸 환경변수를 확인하세요.")
        return 2

    try:
        from src.voice_analysis.config import OPENAI_MODEL, STT_MODEL
        from src.voice_analysis.stt_stream_service import STTStreamService
        from src.voice_analysis.test_stream_stt import MicrophoneStream
        from src.voice_analysis.voice_service import VoiceAnalysisService

        stt_service = STTStreamService()
        voice_service = VoiceAnalysisService()
    except Exception as error:
        print(f"[초기화 실패] {safe_error_message(error)}")
        return 2

    timeout_seconds = float(
        os.getenv("VOICE_SMOKE_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    )

    print("=" * 72)
    print("MOVI 마이크 → Google STT → GPT 통합 스모크 테스트")
    print(f"STT Model: {STT_MODEL} / GPT Model: {OPENAI_MODEL}")
    print("문장을 끊지 말고 한 번에 말한 뒤 1초 정도 기다려주세요.")
    print("=" * 72)

    failed = 0

    for index, case in enumerate(CASES, start=1):
        print(f"\n[{index}/{len(CASES)}] {case.name}")
        print(f"말할 문장: {case.transcript}")

        try:
            input("준비되면 Enter를 누르고 바로 말하세요: ")
            stt_result = await capture_final_transcript(
                stt_service=stt_service,
                microphone_type=MicrophoneStream,
                timeout_seconds=timeout_seconds,
            )

            transcript = str(stt_result["text"]).strip()
            confidence = float(stt_result.get("confidence", 0.0))
            print(f"STT 결과 : {transcript}")
            print(f"신뢰도   : {confidence:.2f}")

            analysis = voice_service.analyze(transcript)
            errors = validate(case, analysis)
        except Exception as error:
            failed += 1
            print(f"결과     : FAIL ({safe_error_message(error)})")
            continue

        if errors:
            failed += 1
            print("결과     : FAIL")
            for validation_error in errors:
                print(f"  - {validation_error}")
            continue

        print(f"Intent   : {analysis['intent']}")
        print("결과     : PASS")

    print("\n" + "=" * 72)
    if failed:
        print(f"통합 검증 실패: {failed}/{len(CASES)}개 시나리오 실패")
        return 1

    print(f"통합 검증 성공: {len(CASES)}개 음성 시나리오 통과")
    return 0


def main() -> int:
    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        print("\n사용자가 테스트를 중단했습니다.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
