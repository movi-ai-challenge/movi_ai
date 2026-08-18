import os


# ============================================================
# Google Cloud
# ============================================================

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")


# Chirp 3는 현재 us / eu multi-region에서 제공됨
STT_LOCATION = os.getenv(
    "GOOGLE_STT_LOCATION",
    "us"
)

STT_MODEL = os.getenv(
    "GOOGLE_STT_MODEL",
    "chirp_3"
)

LANGUAGE_CODE = "ko-KR"


# ============================================================
# Google STT V2 Recognizer
# ============================================================

RECOGNIZER_PATH = (
    f"projects/{PROJECT_ID}"
    f"/locations/{STT_LOCATION}"
    f"/recognizers/_"
    if PROJECT_ID
    else None
)

STT_API_ENDPOINT = (
    f"{STT_LOCATION}-speech.googleapis.com"
)


# ============================================================
# Audio Streaming
# ============================================================

# MVP / 테스트용 기본값
#
# 실제 프런트 통합 시에는 브라우저에서 전달되는
# sample rate와 정확히 일치하도록 변경 가능
AUDIO_SAMPLE_RATE = 16000

AUDIO_CHANNEL_COUNT = 1

# LINEAR16 = signed 16-bit PCM
AUDIO_ENCODING = "LINEAR16"

# Google 권장 streaming frame 기준
AUDIO_CHUNK_DURATION_MS = 100


# ============================================================
# Google TTS
# ============================================================

TTS_LANGUAGE_CODE = "ko-KR"

# 처음에는 특정 voice를 강제하지 않고
# language + gender 기준으로 선택
TTS_SPEAKING_RATE = 1.0
TTS_PITCH = 0.0

TTS_OUTPUT_ENCODING = "MP3"


# ============================================================
# Streaming Result
# ============================================================

ENABLE_INTERIM_RESULTS = True


# ============================================================
# Debug
# ============================================================

DEBUG = os.getenv(
    "VOICE_DEBUG",
    "true"
).lower() == "true"


# ============================================================
# OpenAI Requirement Analyzer
# ============================================================

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5-nano",
)


def print_config():
    """
    현재 STT 설정 확인용.
    실제 서비스 로직에서는 호출할 필요 없음.
    """

    if not DEBUG:
        return

    print("=" * 50)
    print("Google STT Configuration")
    print("=" * 50)

    print(f"PROJECT_ID       : {PROJECT_ID}")
    print(f"LOCATION         : {STT_LOCATION}")
    print(f"MODEL            : {STT_MODEL}")
    print(f"LANGUAGE         : {LANGUAGE_CODE}")

    print(f"SAMPLE_RATE      : {AUDIO_SAMPLE_RATE}")
    print(f"CHANNELS         : {AUDIO_CHANNEL_COUNT}")
    print(f"ENCODING         : {AUDIO_ENCODING}")
    print(f"CHUNK_DURATION   : {AUDIO_CHUNK_DURATION_MS}ms")

    print(f"RECOGNIZER       : {RECOGNIZER_PATH}")
    print(f"API_ENDPOINT     : {STT_API_ENDPOINT}")

    print("=" * 50)


if __name__ == "__main__":
    print_config()