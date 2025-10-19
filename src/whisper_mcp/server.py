"""Insanely Fast Whisper MCP Server

Provides blazingly fast audio transcription using OpenAI's Whisper models
with optimizations from Hugging Face Transformers and Flash Attention 2.
"""

import tempfile
import json
import os
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field
from mcp.server.fastmcp import Context, FastMCP
import subprocess
import urllib.request

class ConfigSchema(BaseModel):
    """Configuration schema for Whisper transcription sessions."""

    model_name: str = Field(
        default="openai/whisper-large-v3",
        description="Whisper model to use (e.g., openai/whisper-large-v3, distil-whisper/large-v2)"
    )
    device_id: str = Field(
        default="0",
        description="Device ID for GPU (0, 1, etc.) or 'mps' for Mac"
    )
    batch_size: int = Field(
        default=24,
        description="Number of parallel batches (reduce if OOM occurs)"
    )
    flash: bool = Field(
        default=False,
        description="Use Flash Attention 2 for faster processing"
    )
    task: Literal["transcribe", "translate"] = Field(
        default="transcribe",
        description="Task to perform: transcribe or translate to English"
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'en', 'es', 'fr'). None for auto-detect"
    )
    timestamp: Literal["chunk", "word"] = Field(
        default="chunk",
        description="Timestamp granularity: chunk or word level"
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="Hugging Face token for speaker diarization"
    )
    num_speakers: Optional[int] = Field(
        default=None,
        description="Exact number of speakers (if known)"
    )
    min_speakers: Optional[int] = Field(
        default=None,
        description="Minimum number of speakers"
    )
    max_speakers: Optional[int] = Field(
        default=None,
        description="Maximum number of speakers"
    )


def create_server():
    """Create and configure the Insanely Fast Whisper MCP server."""

    server = FastMCP("Insanely Fast Whisper")

    # Default configuration (will be overridden by session config in Smithery)
    default_config = ConfigSchema()

    def _run_whisper(
        file_path: str,
        config: ConfigSchema,
        transcript_path: Optional[str] = None
    ) -> dict:
        """Internal function to run whisper transcription."""

        # Prepare command
        cmd = [
            "insanely-fast-whisper",
            "--file-name", file_path,
            "--model-name", config.model_name,
            "--device-id", config.device_id,
            "--batch-size", str(config.batch_size),
            "--task", config.task,
            "--timestamp", config.timestamp,
        ]

        # Add optional parameters
        if transcript_path:
            cmd.extend(["--transcript-path", transcript_path])

        if config.flash:
            cmd.extend(["--flash", "True"])

        if config.language:
            cmd.extend(["--language", config.language])

        if config.hf_token:
            cmd.extend(["--hf-token", config.hf_token])

        if config.num_speakers:
            cmd.extend(["--num-speakers", str(config.num_speakers)])
        elif config.min_speakers or config.max_speakers:
            if config.min_speakers:
                cmd.extend(["--min-speakers", str(config.min_speakers)])
            if config.max_speakers:
                cmd.extend(["--max-speakers", str(config.max_speakers)])

        # Run transcription
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Transcription failed: {result.stderr}")

        # Read and return transcript
        output_file = transcript_path or "output.json"
        with open(output_file, 'r') as f:
            transcript = json.load(f)

        # Clean up temporary output file if not specified
        if not transcript_path and os.path.exists("output.json"):
            os.remove("output.json")

        return transcript

    @server.tool()
    def transcribe_file(
        file_path: str,
        output_path: Optional[str] = None
    ) -> dict:
        """Transcribe an audio file using Whisper.

        Args:
            file_path: Path to the audio file (local path)
            output_path: Optional path to save the transcript JSON

        Returns:
            Dictionary containing the transcription with text and timestamps
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        return _run_whisper(file_path, default_config, output_path)

    @server.tool()
    def transcribe_url(
        url: str,
        output_path: Optional[str] = None
    ) -> dict:
        """Transcribe an audio file from a URL using Whisper.

        Args:
            url: URL of the audio file to transcribe
            output_path: Optional path to save the transcript JSON

        Returns:
            Dictionary containing the transcription with text and timestamps
        """
        # Download file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Download the file
            urllib.request.urlretrieve(url, tmp_path)

            # Transcribe
            result = _run_whisper(tmp_path, default_config, output_path)

            return result
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @server.tool()
    def transcribe_with_diarization(
        file_path: str,
        hf_token: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> dict:
        """Transcribe audio with speaker diarization.

        Args:
            file_path: Path to the audio file
            hf_token: Hugging Face token for Pyannote.audio
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            output_path: Optional path to save the transcript JSON

        Returns:
            Dictionary containing the transcription with speaker labels
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Create a config with diarization settings
        diarization_config = ConfigSchema(
            model_name=default_config.model_name,
            device_id=default_config.device_id,
            batch_size=default_config.batch_size,
            flash=default_config.flash,
            task=default_config.task,
            language=default_config.language,
            timestamp=default_config.timestamp,
            hf_token=hf_token,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        return _run_whisper(file_path, diarization_config, output_path)

    @server.tool()
    def get_supported_models() -> dict:
        """Get list of recommended Whisper models and their characteristics.

        Returns:
            Dictionary of model names and their descriptions
        """
        return {
            "models": [
                {
                    "name": "openai/whisper-large-v3",
                    "description": "Latest and most accurate Whisper model",
                    "speed": "Fast with optimizations",
                    "size": "Large (~3GB)",
                    "recommended": True
                },
                {
                    "name": "openai/whisper-large-v2",
                    "description": "Previous version, very accurate",
                    "speed": "Fast with optimizations",
                    "size": "Large (~3GB)"
                },
                {
                    "name": "distil-whisper/large-v2",
                    "description": "Distilled version - faster, slightly less accurate",
                    "speed": "2x faster than large-v2",
                    "size": "Large (~3GB)",
                    "recommended": True
                },
                {
                    "name": "openai/whisper-medium",
                    "description": "Good balance of speed and accuracy",
                    "speed": "Very fast",
                    "size": "Medium (~1.5GB)"
                },
                {
                    "name": "openai/whisper-small",
                    "description": "Fast but less accurate",
                    "speed": "Extremely fast",
                    "size": "Small (~500MB)"
                },
                {
                    "name": "openai/whisper-base",
                    "description": "Fastest but least accurate",
                    "speed": "Lightning fast",
                    "size": "Base (~150MB)"
                }
            ],
            "note": "All models support Flash Attention 2 for additional speed improvements"
        }

    @server.resource("config://current")
    def get_current_config() -> str:
        """Get the current default configuration."""
        return json.dumps(default_config.model_dump(), indent=2)

    @server.prompt()
    def transcribe_audio_prompt() -> str:
        """Prompt for transcribing audio files."""
        return """I can help you transcribe audio files using Whisper. Here's what I can do:

1. **transcribe_file**: Transcribe a local audio file
2. **transcribe_url**: Transcribe an audio file from a URL
3. **transcribe_with_diarization**: Transcribe with speaker identification (requires HF token)

Example usage:
- "Transcribe the audio file at /path/to/audio.mp3"
- "Transcribe https://example.com/audio.wav with speaker diarization"
- "What models are available?"

The transcription uses your configured model (default: whisper-large-v3) and can use Flash Attention 2 for even faster processing."""

    return server
