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


class MossTTSLoadModel:
    @classmethod
    def INPUT_TYPES(s):
        # Default model IDs
        model_options = [
            "OpenMOSS-Team/MOSS-TTS",                   # 8B Model
            "OpenMOSS-Team/MOSS-TTS-Local-Transformer"  # 1.7B Local Model
        ]
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

    RETURN_TYPES = ("MOSS_TTS_MODEL",)
    RETURN_NAMES = ("moss_tts_model",)
    FUNCTION = "load_model"
    CATEGORY = "Kaola/MOSS-TTSD"
    DESCRIPTION = "Loads MOSS-TTS Foundation models (8B or 1.7B). Use for narration and zero-shot cloning."

    def load_model(self, model_path, device, precision, moss_codec=None):
        # Auto-download mappings
        if model_path == "OpenMOSS-Team/MOSS-TTS":
            model_path = auto_download(model_path, "MOSS-TTS")
        elif model_path == "OpenMOSS-Team/MOSS-TTS-Local-Transformer":
            model_path = auto_download(model_path, "MOSS-TTS-Local-Transformer")
        
        # Resolve local path
        model_path = get_model_path(model_path)
        print(f"[MOSS-TTS] Loading model from: {model_path}")

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
            print(f"[MOSS-TTS] Local path detect, enabling local_files_only=True")

        # Audio Tokenizer Logic
        audio_tokenizer_path = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
        if moss_codec and "path" in moss_codec:
            audio_tokenizer_path = moss_codec["path"]
            print(f"[MOSS-TTS] Using codec path from connected node: {audio_tokenizer_path}")
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
        print(f"[MOSS-TTS] Loading processor from {model_path}...")
        try:
            processor_kwargs = {"trust_remote_code": True}
            # Foundation models (MOSS-TTS) usually support normalize_inputs=True if they use Wav2Vec2Processor
            # But let's check safety. If it's same architecture as TTSD (MossTTSDelay), it should support it.
            # Local Transformer (1.7B) might be different.
            # Safe bet: assume similar to TTSD first. If fails, user reports.
            # Actually, VoiceGenerator supported it, SoundEffect didn't.
            # Foundation 8B is typically `MossTTSDelay` (like TTSD), so it should support it.
            # 1.7B Local is `MossTTSLocal`.
            # Let's try with it, catch TypeError if needed?
            # Or just omit it to be safe like SoundEffect?
            # TTSD uses it. 8B is parent of TTSD. So 8B likely needs it or supports it.
            
            processor = AutoProcessor.from_pretrained(
                model_path, 
                normalize_inputs=True, 
                codec_path=audio_tokenizer_path,
                **processor_kwargs
            )
        except TypeError:
            print("[MOSS-TTS] normalize_inputs not supported, retrying without it.")
            processor = AutoProcessor.from_pretrained(
                model_path,
                codec_path=audio_tokenizer_path,
                **processor_kwargs
            )
        except Exception as e:
            print(f"[MOSS-TTS] Failed to load processor: {e}")
            raise e
        
        # Determine attention implementation
        attn_implementation = "sdpa"
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
        
        print(f"[MOSS-TTS] Using attention implementation: {attn_implementation}")

        # Load Model
        print(f"[MOSS-TTS] Loading model to {device} w/ {dtype}...")
        try:
            model = AutoModel.from_pretrained(
                model_path,
                attn_implementation=attn_implementation,
                torch_dtype=dtype,
                **load_kwargs
            ).to(device)
            model.eval()
        except Exception as e:
            print(f"[MOSS-TTS] Failed to load model: {e}")
            raise e
        
        if hasattr(processor, "audio_tokenizer"):
            processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        moss_tts_model = {
            "model": model,
            "processor": processor,
            "device": device,
            "dtype": dtype
        }
        return (moss_tts_model,)


class MossTTSGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "moss_tts_model": ("MOSS_TTS_MODEL", {"tooltip": "Loaded MOSS-TTS Foundation model."}),
                "text": ("STRING", {"multiline": True, "default": "The quick brown fox jumps over the lazy dog.", "tooltip": "Text to narrate."}),
                "audio_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Higher = more variation."}),
                "audio_top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling probability."}),
                "audio_top_k": ("INT", {"default": 50, "min": 1, "max": 200, "tooltip": "Top-K sampling."}),
                "audio_repetition_penalty": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.1, "tooltip": "Penalty for repeating audio tokens."}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 100, "max": 10000, "tooltip": "Max audio tokens."}),
                "text_normalize": ("BOOLEAN", {"default": True, "tooltip": "Normalize text input."}),
            },
            "optional": {
                 "reference_audio": ("AUDIO", {"tooltip": "Reference audio for zero-shot cloning."}),
                 "instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional instruction (if supported by model)."}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Kaola/MOSS-TTSD"
    DESCRIPTION = "Generates speech using MOSS-TTS (Foundation). Supports zero-shot cloning if reference audio is provided."
    
    def generate(self, moss_tts_model, text, audio_temperature, audio_top_p, audio_top_k, audio_repetition_penalty, max_new_tokens, text_normalize, reference_audio=None, instruction=""):
        model = moss_tts_model["model"]
        processor = moss_tts_model["processor"]
        device = moss_tts_model["device"]

        if text_normalize:
            text = normalize_text(text)
        
        # Encode reference audio if provided
        reference_audio_codes = None
        if reference_audio is not None:
             # ComfyUI audio: {"waveform": [batch, channels, samples], "sample_rate": sr}
            waveform = reference_audio["waveform"]
            sr = reference_audio["sample_rate"]
            
            # Use only the first batch item
            if waveform.dim() == 3:
                waveform = waveform[0] # [channels, samples]
            
            # Convert to [channels, samples] expected by torchaudio / processor
            # Input to encode_audios_from_wav expect list of tensors [channels, samples] or [samples]
            # MOSS processor expects [samples] (1D) or [channels, samples] (2D)?
            # Examining TTSD node: wav1.mean(dim=0) -> [1, samples] -> [samples] ?
            # TTSD snippet: wav1 = torch.from_numpy(audio).transpose...
            # Processor.encode_audios_from_wav handles resampling? Yes usually.
            
            # Ensure tensor is on CPU for processing usually
            waveform = waveform.cpu()
            
            # Basic mixdown to mono if needed?
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Processor expects a list of wavs
            # NOTE: MOSS processor `encode_audios_from_wav` handles sample rate conversion if we pass it `sampling_rate` arg?
            # Or we must resample manually. TTSD node implementation resamples manually.
            target_sr = int(processor.model_config.sampling_rate)
            if sr != target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            
            # Encode
            # encode_audios_from_wav return list of codes.
            # Argument `audios` should be list of tensors.
            reference_audio_codes = processor.encode_audios_from_wav([waveform], sampling_rate=target_sr)[0]

        # Build conversation / prompt
        # MOSS-TTS Foundation prompt structure (similar to TTSD but simplified?)
        # Base MOSS-TTS also uses `build_user_message`.
        user_msg_kwargs = {"text": text}
        if reference_audio_codes is not None:
            user_msg_kwargs["reference"] = reference_audio_codes
        if instruction:
             user_msg_kwargs["instruction"] = instruction
        
        user_msg = processor.build_user_message(**user_msg_kwargs)
        conversations = [[user_msg]]
        
        # Preprocess
        batch = processor(conversations, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        print(f"[MOSS-TTS] input_ids shape: {input_ids.shape}")

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

# Mappings (to be imported by __init__)
NODE_CLASS_MAPPINGS = {
    "MossTTSLoadModel": MossTTSLoadModel,
    "MossTTSGenerate": MossTTSGenerate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossTTSLoadModel": "Load MOSS-TTS Foundation Model",
    "MossTTSGenerate": "Generates MOSS-TTS Speech"
}
