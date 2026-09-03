from unittest.mock import Mock, patch

from src.voice_analysis.config import (
    OPENAI_MAX_RETRIES,
    OPENAI_TIMEOUT_SECONDS,
)
from src.voice_analysis.follow_up_entity_parser import FollowUpEntityParser
from src.voice_analysis.requirement_analyzer import RequirementAnalyzer


def test_requirement_analyzer_uses_bounded_openai_client():
    client = Mock()
    with patch(
        "src.voice_analysis.requirement_analyzer.OpenAI",
        return_value=client,
    ) as openai:
        analyzer = RequirementAnalyzer()

        assert analyzer.client is client
        openai.assert_called_once_with(
            timeout=OPENAI_TIMEOUT_SECONDS,
            max_retries=OPENAI_MAX_RETRIES,
        )


def test_follow_up_parser_uses_bounded_openai_client():
    client = Mock()
    with patch(
        "src.voice_analysis.follow_up_entity_parser.OpenAI",
        return_value=client,
    ) as openai:
        parser = FollowUpEntityParser()

        assert parser.client is client
        openai.assert_called_once_with(
            timeout=OPENAI_TIMEOUT_SECONDS,
            max_retries=OPENAI_MAX_RETRIES,
        )
