from google.cloud import texttospeech

from .config import (
    TTS_LANGUAGE_CODE,
    TTS_SPEAKING_RATE,
    TTS_PITCH,
)


class TTSService:
    """
    Google Cloud Text-to-Speech 서비스.

    역할
    ----
    - 텍스트를 음성 데이터로 변환
    - MP3 bytes 반환
    - 향후 API/WebSocket 응답에서 재사용 가능
    """

    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()

    def synthesize(self, text: str) -> bytes:
        """
        입력 텍스트를 MP3 음성으로 변환한다.

        Args:
            text: 음성으로 변환할 문자열

        Returns:
            MP3 binary bytes
        """

        if not text or not text.strip():
            raise ValueError("TTS 입력 텍스트가 비어 있습니다.")

        synthesis_input = texttospeech.SynthesisInput(
            text=text.strip()
        )

        voice = texttospeech.VoiceSelectionParams(
            language_code=TTS_LANGUAGE_CODE,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=TTS_SPEAKING_RATE,
            pitch=TTS_PITCH,
        )

        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        return response.audio_content

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
    ) -> None:
        """
        TTS 결과를 MP3 파일로 저장한다.
        """

        audio_content = self.synthesize(text)

        with open(output_path, "wb") as audio_file:
            audio_file.write(audio_content)