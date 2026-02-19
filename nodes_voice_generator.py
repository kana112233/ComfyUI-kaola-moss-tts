import os
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

from .utils import (
    get_model_path,
    auto_download,
    _patch_tqdm_for_comfyui,
    normalize_text
)

# Try to import folder_paths for INPUT_TYPES
try:
    import folder_paths
except ImportError:
    folder_paths = None


class MossVoiceGeneratorLoadModel:
    @classmethod
    def INPUT_TYPES(s):
        # Default model ID
        model_options = ["OpenMOSS-Team/MOSS-VoiceGenerator"]
        # Add local models if available
        if folder_paths:
            local = folder_paths.get_filename_list("moss_ttsd")
            if local:
                model_options.extend(local)
        
        return {
            "required": {
                "model_path": (model_options,),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                "precision": (["fp32", "bf16"], {"default": "fp32"}),
            },
            "optional": {
                "moss_codec": ("MOSS_AUDIO_CODEC",),
            }
        }

    RETURN_TYPES = ("MOSS_VOICE_GENERATOR_MODEL",)
    RETURN_NAMES = ("moss_vg_model",)
    RETURN_TYPES = ("MOSS_VOICE_GENERATOR_MODEL",)
    RETURN_NAMES = ("moss_vg_model",)
    FUNCTION = "load_model"
    CATEGORY = "Kaola/MOSS-TTSD"
    DESCRIPTION = "Loads the MOSS-VoiceGenerator model. Connect 'moss_codec' to avoid redundant downloads/loading."

    def load_model(self, model_path, device, precision, moss_codec=None):
        # Auto-download if needed (mirroring behavior of MOSS-TTSD)
        if model_path == "OpenMOSS-Team/MOSS-VoiceGenerator":
            model_path = auto_download(model_path, "MOSS-VoiceGenerator")
        
        # Resolve local path
        model_path = get_model_path(model_path)
        print(f"[MOSS-VoiceGenerator] Loading model from: {model_path}")

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # Precision selection
        dtype = torch.float32
        if precision == "bf16":
            dtype = torch.bfloat16

        # Prepare kwargs
        load_kwargs = {"trust_remote_code": True}
        if os.path.exists(model_path):
            load_kwargs["local_files_only"] = True
            print(f"[MOSS-VoiceGenerator] Local path detect, enabling local_files_only=True")

        # Resolve Audio Tokenizer Path
        audio_tokenizer_path = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
        
        # Priority 1: Use connected codec node path
        if moss_codec and "path" in moss_codec:
            audio_tokenizer_path = moss_codec["path"]
            print(f"[MOSS-VoiceGenerator] Using codec path from connected node: {audio_tokenizer_path}")
        else:
            # Priority 2: Check standard ComfyUI path
            possible_local_tokenizer = get_model_path("MOSS-Audio-Tokenizer")
            if os.path.exists(possible_local_tokenizer):
                audio_tokenizer_path = possible_local_tokenizer
                print(f"[MOSS-VoiceGenerator] Found local audio tokenizer at: {audio_tokenizer_path}")
            else:
                # Priority 3: Fallback check peer directory
                if os.path.exists(model_path):
                    parent_dir = os.path.dirname(model_path)
                    peer_tokenizer = os.path.join(parent_dir, "MOSS-Audio-Tokenizer")
                    if os.path.exists(peer_tokenizer):
                        audio_tokenizer_path = peer_tokenizer
                        print(f"[MOSS-VoiceGenerator] Found peer audio tokenizer at: {audio_tokenizer_path}")
        
        # Load Processor
        print(f"[MOSS-VoiceGenerator] Loading processor from {model_path}...")
        print(f"[MOSS-VoiceGenerator] Using codec_path: {audio_tokenizer_path}")
        try:
            # Processor doesn't support local_files_only in its custom init
            processor_kwargs = {"trust_remote_code": True}
            
            processor = AutoProcessor.from_pretrained(
                model_path, 
                normalize_inputs=True,
                codec_path=audio_tokenizer_path,
                **processor_kwargs
            )
            print(f"[MOSS-VoiceGenerator] Processor loaded successfully.")
        except Exception as e:
            print(f"[MOSS-VoiceGenerator] Failed to load processor: {e}")
            raise e
        
        # Determine attention implementation
        attn_implementation = "eager"
        if device == "cuda":
            # Simple check for flash attention availability
            try:
                import flash_attn
                # Also check if metadata exists, as transformers relies on it
                from importlib.metadata import version, PackageNotFoundError
                try:
                    version("flash_attn")
                except PackageNotFoundError:
                    print("[MOSS-VoiceGenerator] flash_attn package metadata not found. Falling back to sdpa.")
                    raise ImportError("flash_attn metadata missing")

                if dtype in [torch.float16, torch.bfloat16]:
                    major, _ = torch.cuda.get_device_capability()
                    if major >= 8:
                        attn_implementation = "flash_attention_2"
                    else:
                        attn_implementation = "sdpa"
                else:
                    attn_implementation = "sdpa"
            except ImportError:
                attn_implementation = "sdpa"
        
        print(f"[MOSS-VoiceGenerator] Using attention implementation: {attn_implementation}")

        # Load Model
        print(f"[MOSS-VoiceGenerator] Loading model to {device} with {dtype}...")
        try:
            model = AutoModel.from_pretrained(
                model_path,
                attn_implementation=attn_implementation,
                torch_dtype=dtype,
                **load_kwargs
            ).to(device)
            model.eval()
            print(f"[MOSS-VoiceGenerator] Model loaded successfully.")
        except Exception as e:
            print(f"[MOSS-VoiceGenerator] Failed to load model: {e}")
            raise e
        
        # Ensure processor's audio tokenizer is on the correct device
        # Note: AutoProcessor usually loads the audio tokenizer.
        # We move it to device to match MOSS-TTSD behavior, although MOSS-TTSD nodes.py sometimes keeps it on CPU?
        # In MOSS-TTSD node we saw: processor.audio_tokenizer = processor.audio_tokenizer.cpu()
        # But VoiceGenerator demo says: processor.audio_tokenizer = processor.audio_tokenizer.to(device)
        # Let's follow the demo for VoiceGenerator.
        if hasattr(processor, "audio_tokenizer"):
            processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        moss_vg_model = {
            "model": model,
            "processor": processor,
            "device": device,
            "dtype": dtype
        }
        return (moss_vg_model,)


class MossVoiceGeneratorGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "moss_vg_model": ("MOSS_VOICE_GENERATOR_MODEL", {"tooltip": "Loaded MOSS-VoiceGenerator model."}),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test.", "tooltip": "The content to be spoken."}),
                "instruction": ("STRING", {"multiline": True, "default": "A clear, neutral voice for reading.", "tooltip": "Describe the desired voice characteristics (gender, age, tone, emotion)."}), 
                "audio_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Higher = more variation/drama."}),
                "audio_top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling probability."}),
                "audio_top_k": ("INT", {"default": 50, "min": 1, "max": 200, "tooltip": "Top-K sampling."}),
                "audio_repetition_penalty": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.1, "tooltip": "Penalty for repeating audio tokens."}),
                "max_new_tokens": ("INT", {"default": 2000, "min": 100, "max": 10000, "tooltip": "Max audio length in tokens."}),
                "text_normalize": ("BOOLEAN", {"default": True, "tooltip": "Normalize text input."}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Kaola/MOSS-TTSD"
    DESCRIPTION = """Generates a voice sample based on a text description.
    
    Parameters:
    - instruction: Describe the voice (e.g., 'A young female, soft voice').
    - text: The content to be spoken.
    - audio_temperature: 0.1-2.0. Higher = more variation.
    - audio_top_p/top_k: Sampling parameters.
    """

    def generate(self, moss_vg_model, text, instruction, audio_temperature, audio_top_p, audio_top_k, audio_repetition_penalty, max_new_tokens, text_normalize):
        model = moss_vg_model["model"]
        processor = moss_vg_model["processor"]
        device = moss_vg_model["device"]

        if text_normalize:
            text = normalize_text(text)
            # instructions usually don't need MOSS-TTSD specific normalization (like [S1] tags), but simple clean up is fine
            # normalize_text is designed for dialogue mostly. 
            # For VoiceGenerator, the text determines content, instruction determines timbre.
        
        # Build conversation
        # VoiceGenerator input format: https://huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator
        # conversations = [ [processor.build_user_message(text=text, instruction=instruction)] ]
        
        conversations = [[processor.build_user_message(text=text, instruction=instruction)]]
        
        # Preprocess
        batch = processor(conversations, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        print(f"[MOSS-VoiceGenerator] input_ids shape: {input_ids.shape}")

        with torch.no_grad(), _patch_tqdm_for_comfyui(model):
            try:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    audio_temperature=audio_temperature,
                    audio_top_p=audio_top_p,
                    audio_top_k=audio_top_k,
                    audio_repetition_penalty=audio_repetition_penalty,
                )
            except RuntimeError as e:
                if "device-side assert" in str(e):
                    raise RuntimeError(f"CUDA device-side assert triggered. This usually means FP16 instability. Please switch 'precision' to 'fp32' in the Loader node. Original error: {e}")
                raise e

        print(f"[MOSS-VoiceGenerator] model.generate() returned outputs")

        # Decode
        generated_audio_list = []
        messages = processor.decode(outputs)
        
        for i, message in enumerate(messages):
            if message is None:
                continue
            
            # audio_codes_list contains waveform tensors (decoded by processor internally using audio_tokenizer)
            if hasattr(message, "audio_codes_list") and message.audio_codes_list:
                for wav in message.audio_codes_list:
                    if isinstance(wav, torch.Tensor):
                        # Move to CPU float32 for ComfyUI
                        wav = wav.detach().cpu().float()
                        # Output format: ComfyUI expects [batch, channels, samples] or [batch, samples]? 
                        # MOSS-TTSD node adds unsqueeze(0).
                        # Waveform from processor is usually [time] or [channels, time].
                        # Official demo says: torchaudio.save(path, audio.unsqueeze(0), sr) ... implying audio is 1D tensor [T]
                        if wav.dim() == 1:
                            wav = wav.unsqueeze(0).unsqueeze(0) # [1, 1, T]
                        elif wav.dim() == 2:
                            wav = wav.unsqueeze(0) # [1, C, T]
                        
                        generated_audio_list.append(wav)
        
        if not generated_audio_list:
            # Fallback empty
            return (torch.zeros((1, 1, 1), dtype=torch.float32),)

        # Return first audio (batch size 1 assumption)
        fake_output = {"waveform": generated_audio_list[0], "sample_rate": int(processor.model_config.sampling_rate)}
        return (fake_output,)


class MossSoundEffectLoadModel:
    @classmethod
    def INPUT_TYPES(s):
        # Default model ID
        model_options = ["OpenMOSS-Team/MOSS-SoundEffect"]
        # Add local models if available
        if folder_paths:
            local = folder_paths.get_filename_list("moss_ttsd")
            if local:
                model_options.extend(local)
        
        return {
            "required": {
                "model_path": (model_options,),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                "precision": (["fp32", "bf16"], {"default": "fp32"}),
            },
            "optional": {
                "moss_codec": ("MOSS_AUDIO_CODEC",),
            }
        }

    RETURN_TYPES = ("MOSS_SOUND_EFFECT_MODEL",)
    RETURN_NAMES = ("moss_se_model",)
    FUNCTION = "load_model"
    CATEGORY = "Kaola/MOSS-TTSD"
    DESCRIPTION = "Loads the MOSS-SoundEffect model."

    def load_model(self, model_path, device, precision, moss_codec=None):
        # Auto-download
        if model_path == "OpenMOSS-Team/MOSS-SoundEffect":
            model_path = auto_download(model_path, "MOSS-SoundEffect")
        
        # Resolve local path
        model_path = get_model_path(model_path)
        print(f"[MOSS-SoundEffect] Loading model from: {model_path}")

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # Precision selection
        dtype = torch.float32
        if precision == "bf16":
            dtype = torch.bfloat16

        # Prepare kwargs
        load_kwargs = {"trust_remote_code": True}
        if os.path.exists(model_path):
            load_kwargs["local_files_only"] = True
            print(f"[MOSS-SoundEffect] Local path detect, enabling local_files_only=True")

        # Audio Tokenizer Logic (Shared)
        audio_tokenizer_path = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
        if moss_codec and "path" in moss_codec:
            audio_tokenizer_path = moss_codec["path"]
            print(f"[MOSS-SoundEffect] Using codec path from connected node: {audio_tokenizer_path}")
        else:
            possible_local = get_model_path("MOSS-Audio-Tokenizer")
            if os.path.exists(possible_local):
                audio_tokenizer_path = possible_local
            elif os.path.exists(model_path):
                 parent_dir = os.path.dirname(model_path)
                 peer = os.path.join(parent_dir, "MOSS-Audio-Tokenizer")
                 if os.path.exists(peer):
                     audio_tokenizer_path = peer

        # Load Processor
        print(f"[MOSS-SoundEffect] Loading processor from {model_path}...")
        try:
            processor_kwargs = {"trust_remote_code": True}
            processor = AutoProcessor.from_pretrained(
                model_path,
                codec_path=audio_tokenizer_path,
                **processor_kwargs
            )
        except Exception as e:
            print(f"[MOSS-SoundEffect] Failed to load processor: {e}")
            raise e
        
        # Attention implementation
        attn_implementation = "sdpa" # Default safe
        if device == "cuda":
            try:
                import flash_attn
                from importlib.metadata import version, PackageNotFoundError
                try:
                    version("flash_attn")
                    if dtype in [torch.float16, torch.bfloat16] and torch.cuda.get_device_capability()[0] >= 8:
                        attn_implementation = "flash_attention_2"
                except (ImportError, PackageNotFoundError):
                    pass
            except ImportError:
                pass
        
        print(f"[MOSS-SoundEffect] Using attention implementation: {attn_implementation}")

        # Load Model
        print(f"[MOSS-SoundEffect] Loading model to {device} w/ {dtype}...")
        try:
            model = AutoModel.from_pretrained(
                model_path,
                attn_implementation=attn_implementation,
                torch_dtype=dtype,
                **load_kwargs
            ).to(device)
            model.eval()
        except Exception as e:
            print(f"[MOSS-SoundEffect] Failed to load model: {e}")
            raise e
        
        if hasattr(processor, "audio_tokenizer"):
            processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        moss_se_model = {
            "model": model,
            "processor": processor,
            "device": device,
            "dtype": dtype
        }
        return (moss_se_model,)


class MossSoundEffectGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "moss_se_model": ("MOSS_SOUND_EFFECT_MODEL", {"tooltip": "Loaded MOSS-SoundEffect model."}),
                "text": ("STRING", {"multiline": True, "default": "birds chirping in the forest", "tooltip": "Description of the sound effect."}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 30.0, "step": 0.5, "tooltip": "Duration in seconds (approx 12.5 tokens/s)."}),
                "audio_temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "0.1-2.0. Higher = more variation."}),
                "audio_top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling probability."}),
                "audio_top_k": ("INT", {"default": 50, "min": 1, "max": 200, "tooltip": "Top-K sampling."}),
                "audio_repetition_penalty": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.1, "tooltip": "Penalty for repeating audio tokens."}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 100, "max": 4096, "tooltip": "Safety limit for max tokens."}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Kaola/MOSS-TTSD"
    DESCRIPTION = "Generates sound effects from text descriptions. Use 'duration_seconds' to control length."
    
    def generate(self, moss_se_model, text, duration_seconds, audio_temperature, audio_top_p, audio_top_k, audio_repetition_penalty, max_new_tokens):
        model = moss_se_model["model"]
        processor = moss_se_model["processor"]
        device = moss_se_model["device"]

        # Calculate tokens from duration (approx 12.5 tokens per second)
        tokens = int(duration_seconds * 12.5)
        print(f"[MOSS-SoundEffect] Target duration: {duration_seconds}s -> {tokens} tokens")

        # Build conversation for SoundEffect
        try:
             user_msg = processor.build_user_message(text=text, tokens=tokens)
        except TypeError:
             # Fallback
             print("[MOSS-SoundEffect] Warning: processor.build_user_message doesn't support 'tokens'. Duration control may fail.")
             user_msg = processor.build_user_message(text=text)

        conversations = [[user_msg]]
        
        # Preprocess
        batch = processor(conversations, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        print(f"[MOSS-SoundEffect] input_ids shape: {input_ids.shape}")

        with torch.no_grad(), _patch_tqdm_for_comfyui(model):
            try:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=min(max_new_tokens, tokens + 50),
                    audio_temperature=audio_temperature,
                    audio_top_p=audio_top_p,
                    audio_top_k=audio_top_k,
                    audio_repetition_penalty=audio_repetition_penalty,
                )
            except RuntimeError as e:
                if "device-side assert" in str(e):
                    raise RuntimeError(f"CUDA device-side assert triggered. Please ensure 'precision' is 'fp32'. Original error: {e}")
                raise e

        # Decode outputs
        generated_audio_list = []
        messages = processor.decode(outputs)
        
        for i, message in enumerate(messages):
            if message is None:
                continue
            
            if hasattr(message, "audio_codes_list") and message.audio_codes_list:
                for wav in message.audio_codes_list:
                    if isinstance(wav, torch.Tensor):
                        wav = wav.detach().cpu().float()
                        if wav.dim() == 1:
                            wav = wav.unsqueeze(0).unsqueeze(0) # [1, 1, T]
                        elif wav.dim() == 2:
                            wav = wav.unsqueeze(0) # [1, C, T]
                        generated_audio_list.append(wav)
        
        if not generated_audio_list:
             return ({"waveform": torch.zeros(1, 1, 24000), "sample_rate": 24000},)

        final_audio = torch.cat(generated_audio_list, dim=-1)
        
        # Ensure shape [batch, channels, samples]
        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0).unsqueeze(0)
        elif final_audio.dim() == 2:
             final_audio = final_audio.unsqueeze(0)

        return ({"waveform": final_audio, "sample_rate": 24000},)

# Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MossVoiceGeneratorLoadModel": MossVoiceGeneratorLoadModel,
    "MossVoiceGeneratorGenerate": MossVoiceGeneratorGenerate,
    "MossSoundEffectLoadModel": MossSoundEffectLoadModel,
    "MossSoundEffectGenerate": MossSoundEffectGenerate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossVoiceGeneratorLoadModel": "Load MOSS Voice Generator Model",
    "MossVoiceGeneratorGenerate": "Generates MOSS Voice from Text",
    "MossSoundEffectLoadModel": "Load MOSS Sound Effect Model",
    "MossSoundEffectGenerate": "Generates MOSS Sound Effect"
}
