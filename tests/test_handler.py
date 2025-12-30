#!/usr/bin/env python3
"""
Unit tests for the Parakeet RunPod Serverless handler.

These tests validate the request handling, input validation, and
response formatting without requiring actual GPU/NeMo setup.
"""

import base64
import json
import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.validation import validate_audio_input, validate_timestamp_flag


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_audio_base64():
    """Generate a short base64-encoded audio sample."""
    # Create a minimal WAV file (44 bytes header + minimal data)
    wav_data = (
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00"
        b"\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    return base64.b64encode(wav_data).decode('utf-8')


@pytest.fixture
def sample_audio_url():
    """Sample S3 pre-signed URL."""
    return "https://s3.amazonaws.com/bucket/audio.wav?signature=..."



# =============================================================================
# Input Validation Tests
# =============================================================================

class TestValidateAudioInput:
    """Tests for validate_audio_input function."""

    def test_base64_input_valid(self, sample_audio_base64):
        """Test valid base64 input."""
        data = {"audio_base64": sample_audio_base64}
        input_type, audio_data = validate_audio_input(data)

        assert input_type == "base64"
        assert audio_data == sample_audio_base64

    def test_url_input_valid(self, sample_audio_url):
        """Test valid URL input."""
        data = {"audio_url": sample_audio_url}
        input_type, audio_data = validate_audio_input(data)

        assert input_type == "url"
        assert audio_data == sample_audio_url

    def test_no_audio_input(self):
        """Test missing audio input."""
        data = {}

        with pytest.raises(ValueError, match="must contain either"):
            validate_audio_input(data)

    def test_empty_base64_input(self):
        """Test empty base64 input."""
        data = {"audio_base64": ""}

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_audio_input(data)

    def test_invalid_base64_type(self):
        """Test base64 input with wrong type."""
        data = {"audio_base64": 123}

        with pytest.raises(TypeError, match="must be a string"):
            validate_audio_input(data)

    def test_invalid_url_type(self):
        """Test URL input with wrong type."""
        data = {"audio_url": 123}

        with pytest.raises(TypeError, match="must be a string"):
            validate_audio_input(data)

    def test_invalid_url_format(self):
        """Test URL with invalid format."""
        data = {"audio_url": "not-a-url"}

        with pytest.raises(ValueError, match="must be a valid URL"):
            validate_audio_input(data)

    def test_unsupported_audio_format(self):
        """Test unsupported audio format specifier."""
        data = {
            "audio_base64": "base64data",
            "audio_format": ".xyz"
        }

        with pytest.raises(ValueError, match="Unsupported audio format"):
            validate_audio_input(data)

    def test_supported_audio_format(self, sample_audio_base64):
        """Test supported audio format specifier."""
        data = {
            "audio_base64": sample_audio_base64,
            "audio_format": "wav"
        }
        input_type, _ = validate_audio_input(data)

        assert input_type == "base64"


# =============================================================================
# Timestamp Flag Validation Tests
# =============================================================================

class TestValidateTimestampFlag:
    """Tests for validate_timestamp_flag function."""

    def test_missing_timestamp_flag(self):
        """Test missing timestamp flag (should default to False)."""
        data = {}
        result = validate_timestamp_flag(data)

        assert result is False

    def test_timestamp_true(self):
        """Test timestamp flag set to True."""
        data = {"timestamp": True}
        result = validate_timestamp_flag(data)

        assert result is True

    def test_timestamp_false(self):
        """Test timestamp flag set to False."""
        data = {"timestamp": False}
        result = validate_timestamp_flag(data)

        assert result is False

    def test_timestamp_integer_1(self):
        """Test timestamp flag as integer 1."""
        data = {"timestamp": 1}
        result = validate_timestamp_flag(data)

        assert result is True

    def test_timestamp_integer_0(self):
        """Test timestamp flag as integer 0."""
        data = {"timestamp": 0}
        result = validate_timestamp_flag(data)

        assert result is False

    def test_timestamp_invalid_integer(self):
        """Test timestamp flag with invalid integer."""
        data = {"timestamp": 2}

        with pytest.raises(TypeError, match="must be 0 or 1"):
            validate_timestamp_flag(data)

    def test_timestamp_string_true(self):
        """Test timestamp flag as string 'true'."""
        data = {"timestamp": "true"}
        result = validate_timestamp_flag(data)

        assert result is True

    def test_timestamp_string_false(self):
        """Test timestamp flag as string 'false'."""
        data = {"timestamp": "false"}
        result = validate_timestamp_flag(data)

        assert result is False

    def test_timestamp_string_1(self):
        """Test timestamp flag as string '1'."""
        data = {"timestamp": "1"}
        result = validate_timestamp_flag(data)

        assert result is True

    def test_timestamp_string_0(self):
        """Test timestamp flag as string '0'."""
        data = {"timestamp": "0"}
        result = validate_timestamp_flag(data)

        assert result is False

    def test_timestamp_empty_string(self):
        """Test timestamp flag as empty string."""
        data = {"timestamp": ""}
        result = validate_timestamp_flag(data)

        assert result is False

    def test_timestamp_invalid_string(self):
        """Test timestamp flag with invalid string."""
        data = {"timestamp": "invalid"}

        with pytest.raises(TypeError, match="must be 'true', 'false'"):
            validate_timestamp_flag(data)

    def test_timestamp_invalid_type(self):
        """Test timestamp flag with invalid type."""
        data = {"timestamp": []}

        with pytest.raises(TypeError, match="must be a boolean"):
            validate_timestamp_flag(data)


# =============================================================================
# Handler Request Tests
# =============================================================================

class TestHandlerRequest:
    """Tests for handler request processing."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_decode_base64_audio(self):
        """Test base64 audio decoding."""
        from utils.validation import decode_base64_audio

        # Create a minimal WAV file
        wav_data = (
            b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00"
            b"\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        audio_base64 = base64.b64encode(wav_data).decode('utf-8')

        file_path, original_format = decode_base64_audio(audio_base64)

        try:
            assert os.path.exists(file_path)
            assert original_format is None  # No format in raw base64

            # Verify it's a valid WAV file
            with open(file_path, 'rb') as f:
                header = f.read(4)
                assert header == b"RIFF"
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_decode_base64_audio_with_data_uri(self):
        """Test base64 audio decoding with data URI."""
        from utils.validation import decode_base64_audio

        # Create a minimal WAV file
        wav_data = (
            b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00"
            b"\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        audio_base64 = base64.b64encode(wav_data).decode('utf-8')
        data_uri = f"data:audio/wav;base64,{audio_base64}"

        file_path, original_format = decode_base64_audio(data_uri)

        try:
            assert os.path.exists(file_path)
            assert original_format == ".wav"
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_decode_base64_audio_invalid(self):
        """Test base64 audio decoding with invalid data."""
        from utils.validation import decode_base64_audio

        with pytest.raises(ValueError, match="Invalid base64"):
            decode_base64_audio("not-valid-base64!!!")


# =============================================================================
# Mock Model Tests
# =============================================================================

class TestMockTranscription:
    """Tests using a mock model (no GPU required)."""

    @patch('handler.load_model')
    @patch('handler.convert_audio_for_parakeet')
    @patch('handler.decode_base64_audio')
    def test_handler_base64_success(
        self,
        mock_decode,
        mock_convert,
        mock_load_model,
        sample_audio_base64
    ):
        """Test handler with base64 input (mocked)."""
        # Skip importing handler if dependencies not available
        try:
            import handler
        except ImportError:
            pytest.skip("Handler dependencies not available")

        # Setup mocks
        mock_decode.return_value = ("/tmp/audio.wav", None)
        mock_convert.return_value = "/tmp/audio_converted.wav"

        mock_model = Mock()
        mock_result = Mock()
        mock_result.text = "Test transcription"
        mock_result.timestamp = None
        mock_model.transcribe.return_value = [mock_result]
        mock_load_model.return_value = mock_model

        # Create temp files for mocks
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp1:
            temp1 = tmp1.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
            temp2 = tmp2.name

        try:
            mock_decode.return_value = (temp1, None)
            mock_convert.return_value = temp2

            # Mock get_audio_duration
            with patch('handler.get_audio_duration') as mock_duration:
                mock_duration.return_value = 1.0

                # Call handler
                event = {
                    "audio_base64": sample_audio_base64,
                    "timestamp": False
                }

                result = handler.handler(event)

                # Verify result
                assert result.get("success") is True
                assert "text" in result
                assert result["text"] == "Test transcription"

        finally:
            # Cleanup
            for path in [temp1, temp2]:
                if os.path.exists(path):
                    os.unlink(path)

    @patch('handler.load_model')
    @patch('handler.convert_audio_for_parakeet')
    @patch('handler.decode_base64_audio')
    def test_handler_with_timestamps(
        self,
        mock_decode,
        mock_convert,
        mock_load_model,
        sample_audio_base64
    ):
        """Test handler with timestamps enabled (mocked)."""
        try:
            import handler
        except ImportError:
            pytest.skip("Handler dependencies not available")

        # Setup mocks
        mock_model = Mock()
        mock_result = Mock()
        mock_result.text = "Test transcription"
        mock_result.timestamp = {
            "word": [{"start": 0.0, "end": 0.5, "text": "Test"}],
            "segment": [{"start": 0.0, "end": 0.5, "segment": "Test"}],
        }
        mock_model.transcribe.return_value = [mock_result]
        mock_load_model.return_value = mock_model

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp1:
            temp1 = tmp1.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
            temp2 = tmp2.name

        try:
            mock_decode.return_value = (temp1, None)
            mock_convert.return_value = temp2

            with patch('handler.get_audio_duration') as mock_duration:
                mock_duration.return_value = 1.0

                event = {
                    "audio_base64": sample_audio_base64,
                    "timestamp": True
                }

                result = handler.handler(event)

                # Verify result
                assert result.get("success") is True
                assert "text" in result
                assert "timestamps" in result
                assert "word" in result["timestamps"]
                assert "segment" in result["timestamps"]

        finally:
            for path in [temp1, temp2]:
                if os.path.exists(path):
                    os.unlink(path)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in the handler."""

    def test_invalid_json_input(self):
        """Test handler with invalid JSON string."""
        try:
            import handler
        except ImportError:
            pytest.skip("Handler dependencies not available")

        result = handler.handler("not valid json")

        assert result.get("success") is False
        assert "error" in result

    def test_missing_audio_input(self):
        """Test handler with missing audio input."""
        try:
            import handler
        except ImportError:
            pytest.skip("Handler dependencies not available")

        result = handler.handler({})

        assert result.get("success") is False
        assert "error" in result


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
