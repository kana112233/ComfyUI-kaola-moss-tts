# ComfyUI-kaola-moss-ttsd

A ComfyUI custom node for [MOSS-TTSD](https://github.com/OpenMOSS/MOSS-TTSD), a high-quality text-to-spoken dialogue generation model.

## Features
- **Text-to-Speech Generation**: Generate high-fidelity speech from text.
- **Multi-Speaker Support**: Supports standard `[S1]`, `[S2]` tags for dialogue generation.
- **Voice Cloning**: Clone voices using short reference audio clips.

## Installation

1.  Clone this repository into your `ComfyUI/custom_nodes` directory:
    ```bash
    cd custom_nodes
    git clone https://github.com/Startgame/ComfyUI-kaola-moss-ttsd.git
    ```

2.  Install dependencies:
    ```bash
    cd ComfyUI-kaola-moss-ttsd
    pip install -r requirements.txt
    ```

    *Note: For GPU acceleration, ensure you have a CUDA-compatible PyTorch installed.*

## Usage

1.  From the node menu, select `Kaola` -> `MOSS-TTSD` -> `MOSS-TTSD Generation`.
2.  Input your dialogue text. Use tags like `[S1]` and `[S2]` to switch speakers.
    - Example: `[S1] Hello there! [S2] Hi! How are you?`
3.  (Optional) Connect `reference_audio_s1` and/or `reference_audio_s2` to provide voice reference clips for specific speakers.
4.  Connect the `AUDIO` output to a `PreviewAudio` or `SaveAudio` node.

## Generation Modes

| Mode | Reference Audio | Reference Text | Description |
|------|:-:|:-:|-------------|
| `generation` | ❌ | ❌ | Pure generation with random voices |
| `voice_clone` | ✅ | ❌ | Clone voice timbre from reference audio |
| `continuation` | ✅ | ✅ | Continue speaking after reference audio |
| `voice_clone_and_continuation` | ✅ | ✅ | Clone + continue (best for single speaker) |

### `generation`
No reference audio needed. The model generates speech with random voices. Good for quick testing.

### `voice_clone`
**Recommended for multi-speaker dialogue.** Provide reference audio for each speaker. The model encodes each speaker's audio separately and uses `[S1]`/`[S2]` tags in the text to switch voices.

- **Inputs**: `reference_audio_s1` (required), `reference_audio_s2` (optional)
- **Text format**: `[S1] Hello! [S2] Hi there!`

### `continuation`
The model treats the reference audio as "already spoken content" and continues from where it left off. Requires both reference audio AND reference text (the transcript of the reference audio).

- **Inputs**: `reference_audio_s1` + `reference_text_s1` (required)
- **Best for**: Single-speaker scenarios where you want natural continuation

### `voice_clone_and_continuation`
Combines voice cloning (reference in user message) with continuation (prompt audio in assistant message). Provides the strongest voice identity for single-speaker use cases.

- **Inputs**: `reference_audio_s1` + `reference_text_s1` (required)
- **Best for**: Single-speaker with precise voice matching
- **Note**: For multi-speaker dialogue, use `voice_clone` instead — continuation mode concatenates all reference audio, so the model tends to use the last speaker's voice

## Models
The node will automatically download the [MOSS-TTSD-v1.0](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0) model from Hugging Face on the first run. You can configure a local path or different model ID in the node settings.

## License
Apache 2.0 (Inherited from MOSS-TTSD)
