import asyncio

import sounddevice as sd

from .config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNEL_COUNT,
    AUDIO_CHUNK_DURATION_MS,
)
from .stt_stream_service import STTStreamService


class MicrophoneStream:
    """
    테스트용 로컬 마이크 PCM Stream.

    Mac 마이크에서 LINEAR16 PCM을 읽어서
    asyncio.Queue를 통해 STTStreamService에 전달한다.

    실제 서비스에서는 이 부분이
    Node.js -> WebSocket 입력으로 교체된다.
    """

    def __init__(self):
        self.audio_queue = asyncio.Queue()
        self.stream = None
        self.loop = None

        # 16kHz 기준:
        # 100ms = 1600 samples
        self.block_size = int(
            AUDIO_SAMPLE_RATE
            * AUDIO_CHUNK_DURATION_MS
            / 1000
        )

    def _audio_callback(
        self,
        indata,
        frames,
        time,
        status,
    ):
        """
        sounddevice의 callback.

        별도 audio thread에서 호출되기 때문에
        asyncio loop에 thread-safe하게 데이터를 전달한다.
        """

        if status:
            print(f"[Audio Status] {status}")

        if self.loop is None:
            return

        audio_bytes = bytes(indata)

        self.loop.call_soon_threadsafe(
            self.audio_queue.put_nowait,
            audio_bytes,
        )

    async def __aenter__(self):
        self.loop = asyncio.get_running_loop()

        self.stream = sd.RawInputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            blocksize=self.block_size,
            dtype="int16",
            channels=AUDIO_CHANNEL_COUNT,
            callback=self._audio_callback,
        )

        self.stream.start()

        return self

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        if self.stream:
            self.stream.stop()
            self.stream.close()

        # generator 종료용
        await self.audio_queue.put(None)

    async def audio_generator(self):
        """
        STTStreamService가 받을 async audio stream.

        yield:
            raw LINEAR16 PCM bytes
        """

        while True:
            chunk = await self.audio_queue.get()

            if chunk is None:
                break

            yield chunk


async def run_stt_test():
    """
    로컬 마이크 -> Google STT V2
    실시간 인식 테스트.
    """

    print("=" * 60)
    print("Google STT V2 - Microphone Streaming Test")
    print("=" * 60)
    print()
    print("마이크 입력을 시작합니다.")
    print("한국어로 말을 해보세요.")
    print()
    print("종료: Ctrl + C")
    print("-" * 60)

    stt_service = STTStreamService()

    async with MicrophoneStream() as microphone:

        async for result in stt_service.recognize(
            microphone.audio_generator()
        ):

            result_type = result["type"]
            text = result["text"]

            if result_type == "interim":

                stability = result.get(
                    "stability",
                    0.0,
                )

                print(
                    f"\r[INTERIM] "
                    f"{text} "
                    f"(stability={stability:.2f})",
                    end="",
                    flush=True,
                )

            elif result_type == "final":

                confidence = result.get(
                    "confidence",
                    0.0,
                )

                print()
                print(
                    f"[FINAL] {text} "
                    f"(confidence={confidence:.2f})"
                )
                print("-" * 60)


async def main():
    try:
        await run_stt_test()

    except KeyboardInterrupt:
        print()
        print("STT 테스트를 종료합니다.")

    except Exception as e:
        print()
        print("[ERROR]")
        print(type(e).__name__)
        print(e)

        raise


if __name__ == "__main__":
    asyncio.run(main())