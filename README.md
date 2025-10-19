# Insanely Fast Whisper MCP Server

Blazingly fast audio transcription MCP server using OpenAI's Whisper with optimizations from Hugging Face Transformers and Flash Attention 2.

⚡️ **Transcribe 150 minutes (2.5 hours) of audio in less than 98 seconds!**

## Features

- 🚀 **Ultra-fast transcription** using optimized Whisper models
- 🎯 **Multiple transcription tools** for files, URLs, and diarization
- 🔧 **Configurable models** from tiny to large-v3, including distilled versions
- 💬 **Speaker diarization** with Pyannote.audio integration
- ⚡ **Flash Attention 2** support for even faster processing
- 🌍 **Multi-language** support with automatic language detection
- 📝 **Word or chunk-level timestamps** for precise timing
- 🎨 **Translation** mode to translate audio to English

## Installation

### Prerequisites

- Python >=3.10
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- NVIDIA GPU (recommended) or Apple Silicon Mac
- CUDA toolkit (for NVIDIA GPUs) or MPS support (for Mac)

### Install from Source

```bash
# Clone the repository
git clone <your-repo-url>
cd insanely-fast-whisper-mcp

# Install dependencies
uv sync

# Install insanely-fast-whisper CLI (required)
uv pip install insanely-fast-whisper
```

### Optional: Install Flash Attention 2

For maximum speed, install Flash Attention 2:

```bash
uv pip install flash-attn --no-build-isolation
```

## Usage

### Local Development

```bash
# Start the server in development mode
uv run dev
```

### Test in Smithery Playground

```bash
# Port-forward to Smithery Playground via ngrok
uv run playground
```

### Configuration

The server accepts session-specific configuration to customize transcription behavior:

```json
{
  "model_name": "openai/whisper-large-v3",
  "device_id": "0",
  "batch_size": 24,
  "flash": false,
  "task": "transcribe",
  "language": null,
  "timestamp": "chunk",
  "hf_token": null,
  "num_speakers": null,
  "min_speakers": null,
  "max_speakers": null
}
```

#### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model_name` | string | `openai/whisper-large-v3` | Whisper model to use |
| `device_id` | string | `"0"` | GPU device ID or "mps" for Mac |
| `batch_size` | integer | `24` | Number of parallel batches |
| `flash` | boolean | `false` | Use Flash Attention 2 |
| `task` | string | `"transcribe"` | "transcribe" or "translate" |
| `language` | string | `null` | Language code (e.g., "en", "es") or null for auto-detect |
| `timestamp` | string | `"chunk"` | "chunk" or "word" level timestamps |
| `hf_token` | string | `null` | Hugging Face token for diarization |
| `num_speakers` | integer | `null` | Exact number of speakers |
| `min_speakers` | integer | `null` | Minimum number of speakers |
| `max_speakers` | integer | `null` | Maximum number of speakers |

### Available Tools

#### 1. transcribe_file

Transcribe a local audio file.

```
Args:
  file_path (string): Path to the audio file
  output_path (string, optional): Path to save transcript JSON

Returns:
  Dictionary with transcription text and timestamps
```

#### 2. transcribe_url

Transcribe an audio file from a URL.

```
Args:
  url (string): URL of the audio file
  output_path (string, optional): Path to save transcript JSON

Returns:
  Dictionary with transcription text and timestamps
```

#### 3. transcribe_with_diarization

Transcribe with speaker identification (requires Hugging Face token).

```
Args:
  file_path (string): Path to the audio file
  num_speakers (integer, optional): Exact number of speakers
  min_speakers (integer, optional): Minimum number of speakers
  max_speakers (integer, optional): Maximum number of speakers
  output_path (string, optional): Path to save transcript JSON

Returns:
  Dictionary with transcription and speaker labels
```

#### 4. get_supported_models

Get list of available Whisper models and their characteristics.

```
Returns:
  Dictionary with model information
```

### Example Usage

**Transcribe a local file:**
```
Please transcribe the audio file at /path/to/recording.mp3
```

**Transcribe from URL:**
```
Transcribe this podcast episode: https://example.com/podcast.mp3
```

**Transcribe with speaker diarization:**
```
Transcribe /path/to/meeting.wav and identify the 3 speakers
```

**Use a faster model:**
```
Use distil-whisper/large-v2 to transcribe /path/to/audio.mp3
```

**Translate to English:**
```
Translate this Spanish audio to English: /path/to/spanish.mp3
```

## Supported Models

| Model | Speed | Accuracy | Size | Recommended |
|-------|-------|----------|------|-------------|
| `openai/whisper-large-v3` | Fast* | Highest | ~3GB | ✅ Yes |
| `distil-whisper/large-v2` | 2x Faster* | Very High | ~3GB | ✅ Yes |
| `openai/whisper-large-v2` | Fast* | Very High | ~3GB | |
| `openai/whisper-medium` | Very Fast | High | ~1.5GB | |
| `openai/whisper-small` | Extremely Fast | Medium | ~500MB | |
| `openai/whisper-base` | Lightning Fast | Low | ~150MB | |

\* With Flash Attention 2 and optimizations

## Deployment

### Deploy to Smithery

1. Create a GitHub repository
2. Push your code
3. Go to [Smithery](https://smithery.ai/new)
4. Click "Deploy" and select your repository

### Docker Deployment

```bash
# Build Docker image
docker build -t insanely-fast-whisper-mcp .

# Run container
docker run -p 8000:8000 insanely-fast-whisper-mcp
```

## Performance

Benchmarks on NVIDIA A100 - 80GB:

| Configuration | Time (150 min audio) | Speed-up |
|--------------|---------------------|----------|
| large-v3 (fp32, default) | ~31 min | 1x |
| large-v3 (fp16 + batching) | ~5 min | 6x |
| large-v3 (fp16 + Flash Attention 2) | ~1.6 min | **19x** |
| distil-large-v2 (fp16 + Flash Attention 2) | ~1.3 min | **24x** |

## FAQ

### How do I enable Flash Attention 2?

Set `"flash": true` in your session config. Make sure flash-attn is installed:

```bash
uv pip install flash-attn --no-build-isolation
```

### How do I use speaker diarization?

1. Get a Hugging Face token from [hf.co/settings/tokens](https://hf.co/settings/tokens)
2. Accept the Pyannote.audio terms at [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
3. Add your token to the session config: `"hf_token": "hf_xxx"`
4. Use the `transcribe_with_diarization` tool

### What audio formats are supported?

Any format supported by FFmpeg: MP3, WAV, M4A, FLAC, OGG, AAC, WMA, etc.

### How do I reduce memory usage?

Lower the `batch_size` in your config (default: 24). Try 16, 12, 8, or 4 if you encounter OOM errors.

### Can I run this on Mac?

Yes! Set `"device_id": "mps"` in your config. Note: MPS is less optimized than CUDA, so use smaller batch sizes (4-8).

## Troubleshooting

### Out of Memory (OOM) errors

Reduce `batch_size` in your config:
- NVIDIA GPUs: Try 16, 12, or 8
- Mac (MPS): Try 4
- Smaller GPUs: Try 4 or lower

### Flash Attention 2 installation fails

Make sure you have CUDA toolkit installed and use:
```bash
uv pip install flash-attn --no-build-isolation
```

### "Torch not compiled with CUDA enabled" on Windows

Manually install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Credits

- [Insanely Fast Whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) - Original CLI tool
- [OpenAI Whisper](https://github.com/openai/whisper) - Base model
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Optimizations
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) - Speed improvements
- [Smithery.ai](https://smithery.ai) - MCP deployment platform

## License

Apache 2.0

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/yourusername/insanely-fast-whisper-mcp/issues)
- Smithery Discord: [Join here](https://discord.gg/Afd38S5p9A)
