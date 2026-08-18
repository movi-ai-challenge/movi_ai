from collections.abc import AsyncIterator

from google.api_core.client_options import ClientOptions
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech

from .config import (
    RECOGNIZER_PATH,
    STT_API_ENDPOINT,
    STT_MODEL,
    LANGUAGE_CODE,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNEL_COUNT,
    ENABLE_INTERIM_RESULTS,
)


class STTStreamService:
    """
    Google Cloud Speech-to-Text V2 Streaming 서비스.

    역할
    ----
    - PCM audio chunk를 Google STT StreamingRecognize로 전달
    - interim / final transcript 구분
    - 이후 WebSocket 서버에서 그대로 재사용
    """

    def __init__(self):

        if not PROJECT_ID:
            raise ValueError(
                "Google STT 사용을 위해 "
                "GOOGLE_CLOUD_PROJECT 환경변수가 필요합니다."
            )

        client_options = ClientOptions(
            api_endpoint=STT_API_ENDPOINT
        )

        self.client = speech_v2.SpeechAsyncClient(
            client_options=client_options
        )

    def _build_streaming_config(
        self,
    ) -> cloud_speech.StreamingRecognitionConfig:
        """
        Google STT Streaming 설정 생성.
        """

        decoding_config = cloud_speech.ExplicitDecodingConfig(
            encoding=(
                cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16
            ),
            sample_rate_hertz=AUDIO_SAMPLE_RATE,
            audio_channel_count=AUDIO_CHANNEL_COUNT,
        )

        recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=decoding_config,
            model=STT_MODEL,
            language_codes=[LANGUAGE_CODE],
        )

        streaming_features = cloud_speech.StreamingRecognitionFeatures(
            interim_results=ENABLE_INTERIM_RESULTS,
        )

        return cloud_speech.StreamingRecognitionConfig(
            config=recognition_config,
            streaming_features=streaming_features,
        )

    async def _request_generator(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[cloud_speech.StreamingRecognizeRequest]:
        """
        Google STT에 전달할 request stream 생성.

        첫 번째 request:
            recognizer + config

        두 번째 이후:
            audio bytes
        """

        streaming_config = self._build_streaming_config()

        # 최초 요청에는 설정만 포함
        yield cloud_speech.StreamingRecognizeRequest(
            recognizer=RECOGNIZER_PATH,
            streaming_config=streaming_config,
        )

        # 이후 요청에는 audio만 포함
        async for audio_chunk in audio_stream:
            if not audio_chunk:
                continue

            yield cloud_speech.StreamingRecognizeRequest(
                audio=audio_chunk
            )

    async def recognize(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[dict]:
        """
        PCM audio stream을 Google STT로 전달하고
        인식 결과를 dict 형태로 반환한다.

        반환 예시

        interim:
        {
            "type": "interim",
            "text": "김민수에게 오만원",
            "stability": 0.82
        }

        final:
        {
            "type": "final",
            "text": "김민수에게 오만원 송금해줘",
            "confidence": 0.94
        }
        """

        requests = self._request_generator(audio_stream)

        responses = await self.client.streaming_recognize(
            requests=requests
        )

        async for response in responses:

            for result in response.results:

                if not result.alternatives:
                    continue

                alternative = result.alternatives[0]

                transcript = alternative.transcript.strip()

                if not transcript:
                    continue

                if result.is_final:

                    yield {
                        "type": "final",
                        "text": transcript,
                        "confidence": alternative.confidence,
                    }

                else:

                    yield {
                        "type": "interim",
                        "text": transcript,
                        "stability": result.stability,
                    }