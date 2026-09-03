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
    "long"
)

# 배치도 스트리밍과 같은 long 을 쓴다.
#
# chirp_3 는 중간 결과(interim)를 내보내지 않는다. 실측하면 2.6초 발화에
# interim 0건 / final 1건만 온다. 그러면 말이 끝나야 글자가 나와 실시간
# 표시가 성립하지 않는다.
#
# long 은 같은 발화에서 interim 25건을 내보내고, 호출어 '모비야'도 정확히
# 인식했다(chirp_3 는 '모비아', short 는 '모기야' 로 흘렸다).
#
# 배치는 interim 이 필요 없지만, 두 경로가 다른 모델을 쓰면 같은 말을 두고
# 인식 결과가 갈린다. 호출어부터 다르게 들리면 어느 경로로 들어왔느냐에 따라
# 되고 안 되고가 바뀐다. 모델을 하나로 맞춘다.
STT_STREAM_MODEL = os.getenv(
    "GOOGLE_STT_STREAM_MODEL",
    "long"
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
    "gpt-4o-mini",
)

# 음성 명령은 사용자가 응답을 기다리는 동기 경로다. 모델 호출이 지연될 때
# SDK의 긴 기본 대기/재시도로 전체 음성 세션이 멈추지 않도록 상한을 둔다.
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "5"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "0"))


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
