"""
pytest 실행 시 프로젝트 루트의 .env 를 자동으로 로드한다.

애플리케이션 코드는 os.getenv 만 사용하고 .env 를 직접 읽지 않기 때문에,
테스트에서는 여기서 환경변수를 미리 주입해 준다.

서버(컨테이너)에서는 docker --env-file 로 동일한 효과를 얻는다.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


# ============================================================
# 수집 제외
#
# 아래 두 파일은 test_ 로 시작하지만 pytest 테스트 함수가 없는
# 수동 실행 스크립트다. 로컬 마이크 / 스피커 / GCP 자격증명이 있어야
# 의미가 있고, CI(리눅스)에서는 sounddevice 가 시스템 libportaudio2
# 를 요구해 수집 단계에서 깨진다.
#
#   실행하려면:  python -m src.voice_analysis.test_stream_stt
# ============================================================

collect_ignore = [
    "src/voice_analysis/test_stream_stt.py",
    "src/voice_analysis/test_tts.py",
]
