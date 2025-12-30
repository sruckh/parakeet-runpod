#!/usr/bin/env python3
"""
Unit tests for audio conversion utilities.

These tests validate audio file handling, format conversion, and
metadata extraction functions.
"""

import os
import struct
import sys
import tempfile
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio import (
    is_valid_audio_file,
    is_parakeet_compatible,
    get_file_extension_from_format,
    sanitize_filename,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_wav_file():
    """Create a minimal valid WAV file for testing."""
    # Create a minimal WAV file (44 bytes header + 1000 samples of silence)
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.wav', delete=False) as f:
        # RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + 2000))  # file size - 8
        f.write(b'WAVE')

        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # chunk size
        f.write(struct.pack('<H', 1))   # audio format (PCM)
        f.write(struct.pack('<H', 1))   # channels (mono)
        f.write(struct.pack('<I', 16000))  # sample rate (16kHz)
        f.write(struct.pack('<I', 32000))  # byte rate
        f.write(struct.pack('<H', 2))   # block align
        f.write(struct.pack('<H', 16))  # bits per sample

        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', 2000))  # data size
        f.write(b'\x00\x00' * 1000)  # 1000 samples of silence

        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def incompatible_wav_file():
    """Create a WAV file that's not Parakeet-compatible (wrong sample rate)."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.wav', delete=False) as f:
        # RIFF header (44100 Hz instead of 16000 Hz)
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + 2000))
        f.write(b'WAVE')

        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<I', 44100))  # Wrong sample rate
        f.write(struct.pack('<I', 88200))
        f.write(struct.pack('<H', 2))
        f.write(struct.pack('<H', 16))

        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', 2000))
        f.write(b'\x00\x00' * 1000)

        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mp3_file():
    """Create a fake MP3 file for testing (just the extension)."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.mp3', delete=False) as f:
        f.write(b'ID3' + b'\x00' * 100)  # MP3 header
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


# =============================================================================
# File Validation Tests
# =============================================================================

class TestIsValidAudioFile:
    """Tests for is_valid_audio_file function."""

    def test_valid_wav_file(self, sample_wav_file):
        """Test validation of a valid WAV file."""
        assert is_valid_audio_file(sample_wav_file) is True

    def test_valid_mp3_file(self, mp3_file):
        """Test validation of an MP3 file."""
        assert is_valid_audio_file(mp3_file) is True

    def test_nonexistent_file(self):
        """Test validation of a nonexistent file."""
        assert is_valid_audio_file("/nonexistent/file.wav") is False

    def test_directory(self, tmpdir):
        """Test validation of a directory."""
        assert is_valid_audio_file(str(tmpdir)) is False

    def test_unsupported_extension(self):
        """Test validation of file with unsupported extension."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write(b'data')
            temp_path = f.name

        try:
            assert is_valid_audio_file(temp_path) is False
        finally:
            os.unlink(temp_path)

    def test_supported_extensions_case_insensitive(self):
        """Test that extensions are checked case-insensitively."""
        with tempfile.NamedTemporaryFile(suffix='.WAV', delete=False) as f:
            f.write(b'data')
            temp_path = f.name

        try:
            assert is_valid_audio_file(temp_path) is True
        finally:
            os.unlink(temp_path)

    def test_no_extension(self):
        """Test file without extension."""
        with tempfile.NamedTemporaryFile(suffix='', delete=False) as f:
            f.write(b'data')
            temp_path = f.name

        try:
            assert is_valid_audio_file(temp_path) is False
        finally:
            os.unlink(temp_path)


# =============================================================================
# Parakeet Compatibility Tests
# =============================================================================

class TestIsParakeetCompatible:
    """Tests for is_parakeet_compatible function."""

    @patch('utils.audio.get_audio_info')
    def test_compatible_wav_16khz_mono(self, mock_get_info):
        """Test WAV file with correct format (16kHz, mono)."""
        # Create a temp file that passes basic validation
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b'RIFF' + b'\x00' * 100)
            temp_path = f.name

        try:
            # Mock audio info to return compatible values
            mock_get_info.return_value = {
                'duration': 1.0,
                'sample_rate': 16000,
                'channels': 1,
                'codec': 'pcm_s16le',
                'format': 'wav'
            }

            compatible, reason = is_parakeet_compatible(temp_path)

            assert compatible is True
            assert reason == "OK"
        finally:
            os.unlink(temp_path)

    @patch('utils.audio.get_audio_info')
    def test_incompatible_sample_rate(self, mock_get_info):
        """Test WAV file with wrong sample rate."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b'RIFF' + b'\x00' * 100)
            temp_path = f.name

        try:
            mock_get_info.return_value = {
                'duration': 1.0,
                'sample_rate': 44100,  # Wrong sample rate
                'channels': 1,
                'codec': 'pcm_s16le',
                'format': 'wav'
            }

            compatible, reason = is_parakeet_compatible(temp_path)

            assert compatible is False
            assert "16000Hz" in reason
            assert "44100Hz" in reason
        finally:
            os.unlink(temp_path)

    @patch('utils.audio.get_audio_info')
    def test_incompatible_channels(self, mock_get_info):
        """Test WAV file with wrong channel count."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b'RIFF' + b'\x00' * 100)
            temp_path = f.name

        try:
            mock_get_info.return_value = {
                'duration': 1.0,
                'sample_rate': 16000,
                'channels': 2,  # Wrong (should be 1)
                'codec': 'pcm_s16le',
                'format': 'wav'
            }

            compatible, reason = is_parakeet_compatible(temp_path)

            assert compatible is False
            assert "channel" in reason.lower()
        finally:
            os.unlink(temp_path)

    @patch('utils.audio.get_audio_info')
    def test_incompatible_format(self, mock_get_info):
        """Test file with wrong format."""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'data')
            temp_path = f.name

        try:
            mock_get_info.return_value = {
                'duration': 1.0,
                'sample_rate': 16000,
                'channels': 1,
                'codec': 'mp3',
                'format': 'mp3'
            }

            compatible, reason = is_parakeet_compatible(temp_path)

            assert compatible is False
            assert "format" in reason.lower()
        finally:
            os.unlink(temp_path)

    def test_invalid_file(self):
        """Test compatibility check on invalid file."""
        compatible, reason = is_parakeet_compatible("/nonexistent/file.wav")

        assert compatible is False
        assert "Invalid" in reason or "not found" in reason


# =============================================================================
# Filename Sanitization Tests
# =============================================================================

class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_basic_filename(self):
        """Test basic filename sanitization."""
        assert sanitize_filename("audio.wav") == "audio.wav"

    def test_spaces_preserved(self):
        """Test that spaces are preserved."""
        assert sanitize_filename("my audio file.wav") == "my audio file.wav"

    def test_special_chars_replaced(self):
        """Test that special characters are replaced with underscore."""
        assert sanitize_filename("file@name#test$.wav") == "file_name_test_.wav"

    def test_leading_dots_removed(self):
        """Test that leading dots are removed."""
        assert sanitize_filename(".hidden.wav") == "hidden.wav"

    def test_leading_dashes_removed(self):
        """Test that leading dashes are removed."""
        assert sanitize_filename("-test.wav") == "test.wav"

    def test_long_filename_truncated(self):
        """Test that long filenames are truncated."""
        long_name = "a" * 300 + ".wav"
        result = sanitize_filename(long_name)
        assert len(result) <= 255

    def test_empty_filename(self):
        """Test that empty filename becomes 'audio'."""
        assert sanitize_filename("") == "audio"

    def test_only_special_chars(self):
        """Test filename with only special characters."""
        assert sanitize_filename("@@@") == "audio"

    def test_unicode_preserved(self):
        """Test that unicode characters are preserved."""
        assert "test" in sanitize_filename("файл.wav")


# =============================================================================
# Format Extension Tests
# =============================================================================

class TestGetFileExtensionFromFormat:
    """Tests for get_file_extension_from_format function."""

    def test_format_with_dot(self):
        """Test format string that already has a dot."""
        assert get_file_extension_from_format(".wav") == ".wav"

    def test_format_without_dot(self):
        """Test format string without a dot."""
        assert get_file_extension_from_format("wav") == ".wav"

    def test_uppercase_format(self):
        """Test uppercase format string."""
        assert get_file_extension_from_format("WAV") == ".wav"

    def test_mixed_case_format(self):
        """Test mixed case format string."""
        assert get_file_extension_from_format("Mp3") == ".mp3"

    def test_empty_format(self):
        """Test empty format string (should default to .wav)."""
        assert get_file_extension_from_format("") == ".wav"

    def test_format_with_dot_uppercase(self):
        """Test format with dot and uppercase."""
        assert get_file_extension_from_format(".FLAC") == ".flac"


# =============================================================================
# TempAudioFile Context Manager Tests
# =============================================================================

class TestTempAudioFile:
    """Tests for TempAudioFile context manager."""

    def test_creates_and_deletes_file(self):
        """Test that temp file is created and deleted."""
        from utils.audio import TempAudioFile

        with TempAudioFile(suffix=".wav") as path:
            assert os.path.exists(path)
            assert path.endswith(".wav")

            # Write some data
            with open(path, 'wb') as f:
                f.write(b"test data")

        # File should be deleted after context
        assert not os.path.exists(path)

    def test_default_suffix(self):
        """Test default suffix is .wav."""
        from utils.audio import TempAudioFile

        with TempAudioFile() as path:
            assert path.endswith(".wav")

    def test_custom_suffix(self):
        """Test custom suffix."""
        from utils.audio import TempAudioFile

        with TempAudioFile(suffix=".flac") as path:
            assert path.endswith(".flac")

    def test_exception_handling(self):
        """Test that file is deleted even if exception occurs."""
        from utils.audio import TempAudioFile

        path = None

        try:
            with TempAudioFile(suffix=".wav") as temp_path:
                path = temp_path
                raise ValueError("Test exception")
        except ValueError:
            pass

        # File should still be deleted
        assert not os.path.exists(path)


# =============================================================================
# Integration Tests (require ffmpeg)
# =============================================================================

class TestAudioConversion:
    """Tests that require ffmpeg to be installed."""

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Unix-specific test"
    )
    def test_convert_sample_wav(self, sample_wav_file):
        """Test converting a WAV file with ffmpeg."""
        try:
            from utils.audio import convert_audio_for_parakeet
        except ImportError:
            pytest.skip("ffmpeg-python not installed")

        # Mock ffmpeg to avoid actual conversion
        with patch('utils.audio.ffmpeg') as mock_ffmpeg:
            mock_input = Mock()
            mock_output = Mock()
            mock_ffmpeg.input.return_value = mock_input
            mock_input.output.return_value = mock_output
            mock_ffmpeg.run.return_value = None

            # Create expected output file
            with tempfile.NamedTemporaryFile(suffix='_converted.wav', delete=False) as tmp:
                output_path = tmp.name

            try:
                # Mock the run to create the output file
                def mock_run(*args, **kwargs):
                    with open(output_path, 'wb') as f:
                        f.write(b'converted')

                mock_ffmpeg.run.side_effect = mock_run

                result = convert_audio_for_parakeet(sample_wav_file)
                assert result == output_path

            finally:
                if os.path.exists(output_path):
                    os.unlink(output_path)

    def test_convert_nonexistent_file(self):
        """Test converting a nonexistent file."""
        try:
            from utils.audio import convert_audio_for_parakeet
        except ImportError:
            pytest.skip("ffmpeg-python not installed")

        with pytest.raises(FileNotFoundError):
            convert_audio_for_parakeet("/nonexistent/file.wav")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
