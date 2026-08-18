from pathlib import Path

from .tts_service import TTSService


def main():
    tts_service = TTSService()

    text = "김민수님에게 오만원을 송금하시겠습니까?"

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "tts_test.mp3"

    print("TTS 변환 시작")
    print(f"입력: {text}")

    tts_service.synthesize_to_file(
        text=text,
        output_path=str(output_path),
    )

    print("TTS 변환 완료")
    print(f"출력 파일: {output_path}")


if __name__ == "__main__":
    main()