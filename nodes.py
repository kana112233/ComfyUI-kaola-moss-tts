import os
import sys
import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path

from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
import transformers

# Monkey-patch for PreTrainedConfig
# Still needed for some environments/older transformers mix
try:
    import transformers.configuration_utils
    if not hasattr(transformers.configuration_utils, "PreTrainedConfig"):
        if hasattr(transformers, "PreTrainedConfig"):
            transformers.configuration_utils.PreTrainedConfig = transformers.PreTrainedConfig
            print("Monkey-patched transformers.configuration_utils.PreTrainedConfig")
except ImportError:
    pass

# Try to import folder_paths
try:
    import folder_paths
    folder_paths.add_model_folder_path("moss_ttsd", os.path.join(folder_paths.models_dir, "moss_ttsd"))
except ImportError:
    folder_paths = None

def get_model_path(model_path):
    # Helper to resolve local paths
    if folder_paths:
        if model_path in folder_paths.get_filename_list("moss_ttsd"):
             return folder_paths.get_full_path("moss_ttsd", model_path)
    
    if model_path == "OpenMOSS-Team/MOSS-TTSD-v1.0" or model_path == "OpenMOSS-Team/MOSS-Audio-Tokenizer":
        try:
            if folder_paths:
                base_path = os.path.join(folder_paths.models_dir, "moss_ttsd")
            else:
                base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "moss_ttsd")
            
            # Map HF ID to local folder name
            folder_name = model_path.split("/")[-1]
            local_path = os.path.join(base_path, folder_name)

            if os.path.exists(local_path):
                print(f"Using local path for {model_path}: {local_path}")
                return local_path
        except Exception:
            pass
            
    return model_path

def auto_download(repo_id, local_dir_name):
    # Helper to auto-download if needed
    try:
        from huggingface_hub import snapshot_download
        if folder_paths:
            base_path = os.path.join(folder_paths.models_dir, "moss_ttsd")
        else:
            base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "moss_ttsd")
        
        local_path = os.path.join(base_path, local_dir_name)
        if not os.path.exists(local_path):
            print(f"Downloading {repo_id} to {local_path}...")
            snapshot_download(repo_id=repo_id, local_dir=local_path)
        return local_path
    except ImportError:
        print("huggingface_hub not installed, skipping auto-download check.")
        return repo_id
    except Exception as e:
        print(f"Download failed: {e}")
        return repo_id

class MossAudioCodecLoadModel:
    @classmethod
    def INPUT_TYPES(s):
        codec_options = ["OpenMOSS-Team/MOSS-Audio-Tokenizer"]
        if folder_paths:
            local = folder_paths.get_filename_list("moss_ttsd")
            if local: codec_options.extend(local)
        
        return {
            "required": {
                "codec_path": (codec_options,),
            }
        }

    RETURN_TYPES = ("MOSS_AUDIO_CODEC",)
    RETURN_NAMES = ("moss_codec",)
    FUNCTION = "load_codec"
    CATEGORY = "Kaola/MOSS-TTSD"

    def load_codec(self, codec_path):
        if codec_path == "OpenMOSS-Team/MOSS-Audio-Tokenizer":
            codec_path = auto_download(codec_path, "MOSS-Audio-Tokenizer")
        else:
            codec_path = get_model_path(codec_path)
            
        print(f"Loading MOSS-Audio-Tokenizer from {codec_path}...")
        
        # Load directly as AutoModel
        try:
             audio_tokenizer = AutoModel.from_pretrained(codec_path, trust_remote_code=True)
        except Exception as e:
             raise RuntimeError(f"Failed to load audio tokenizer: {e}")
        
        # Default to CPU to save VRAM
        audio_tokenizer = audio_tokenizer.cpu().eval()
        
        return ({"audio_tokenizer": audio_tokenizer, "path": codec_path},)


class MossTTSDLoadModel:
    @classmethod
    def INPUT_TYPES(s):
        model_options = ["OpenMOSS-Team/MOSS-TTSD-v1.0"]
        if folder_paths:
            local = folder_paths.get_filename_list("moss_ttsd")
            if local: model_options.extend(local)
        
        return {
            "required": {
                "model_path": (model_options,),
                "quantization": (["none", "8bit", "4bit"], {"default": "none"}),
            }
        }

    RETURN_TYPES = ("MOSS_TTSD_MODEL",)
    RETURN_NAMES = ("moss_model",)
    FUNCTION = "load_model"
    CATEGORY = "Kaola/MOSS-TTSD"

    def load_model(self, model_path, quantization):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        if quantization != "none" and device == "cpu":
            print("Quantization not supported on CPU. Forcing 'none'.")
            quantization = "none"

        if model_path == "OpenMOSS-Team/MOSS-TTSD-v1.0":
            model_path = auto_download(model_path, "MOSS-TTSD-v1.0")
        else:
            model_path = get_model_path(model_path)
        
        print(f"Loading MOSS-TTSD model from {model_path}...")

        # 1. Load Text Tokenizer
        # We use use_fast=False to avoid "ModelWrapper" errors with older tokenizers/remote code mismatch
        print("Loading Text Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        # 2. Construct Processor Manually
        # This bypasses AutoProcessor.from_pretrained which passes use_fast=False to AudioTokenizer (causing crash)
        # The actual class is MossTTSDelayProcessor (not MossTTSProcessor)
        print("Constructing Processor...")
        try:
            # Load model config (needed by processor for audio token IDs, sampling rate, etc.)
            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

            # Resolve the processor class from the remote code file
            processor_class = get_class_from_dynamic_module("processing_moss_tts.MossTTSDelayProcessor", model_path)
            
            # Instantiate with text tokenizer and config (audio_tokenizer added later via Codec node)
            processor = processor_class(tokenizer=tokenizer, audio_tokenizer=None, model_config=model_config)
            
        except Exception as e:
            print(f"Failed to manually construct processor: {e}")
            raise RuntimeError(
                f"Could not manually load MossTTSDelayProcessor from {model_path}. "
                f"Ensure 'processing_moss_tts.py' exists. Error: {e}"
            )

        # 3. Load Model with Quantization
        # Check accelerate
        try:
            import accelerate
            use_accelerate = True
        except ImportError:
            use_accelerate = False
            if quantization != "none":
                print("Warning: Quantization requires 'accelerate' and 'bitsandbytes'.")

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if use_accelerate:
            model_kwargs["device_map"] = "auto"
            model_kwargs["low_cpu_mem_usage"] = True
        
        if quantization == "8bit":
            model_kwargs["load_in_8bit"] = True
            print("Loading model in 8-bit precision...")
        elif quantization == "4bit":
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            print("Loading model in 4-bit precision...")

        attn_implementation = "flash_attention_2" if device == "cuda" else "sdpa"
        model = None
        try:
            model = AutoModel.from_pretrained(
                model_path,
                attn_implementation=attn_implementation,
                **model_kwargs
            )
        except Exception as e:
            print(f"Failed to load with flash_attention_2: {e}")
            try:
                model = AutoModel.from_pretrained(
                    model_path,
                    attn_implementation="sdpa",
                    **model_kwargs
                )
            except Exception as e2:
                print(f"Second loading attempt failed: {e2}")
                # Fallback to defaults
                model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype)

        if not use_accelerate and quantization == "none":
            model = model.to(device)
        
        model = model.eval()

        moss_model = {
            "model": model,
            "processor": processor,
            "device": device,
            "dtype": dtype
        }
        return (moss_model,)


class MossTTSDGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "moss_model": ("MOSS_TTSD_MODEL",),
                "moss_codec": ("MOSS_AUDIO_CODEC",),
                "text": ("STRING", {"multiline": True, "default": "[S1] Hello world."}),
                "audio_temperature": ("FLOAT", {"default": 1.1, "min": 0.1, "max": 2.0, "step": 0.1}),
                "audio_top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05}),
                "audio_repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.1}),
                "text_temperature": ("FLOAT", {"default": 1.1, "min": 0.1, "max": 2.0, "step": 0.1}),
                "max_new_tokens": ("INT", {"default": 2000, "min": 100, "max": 10000}),
            },
            "optional": {
                "reference_audio_s1": ("AUDIO",),
                "reference_text_s1": ("STRING", {"multiline": True, "default": ""}),
                "reference_audio_s2": ("AUDIO",),
                "reference_text_s2": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Kaola/MOSS-TTSD"

    def preprocess_audio(self, audio_data, target_sr):
        waveform = audio_data["waveform"]
        sr = audio_data["sample_rate"]
        if waveform.dim() == 3: waveform = waveform.squeeze(0)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform

    def generate(self, moss_model, moss_codec, text, audio_temperature, audio_top_p, audio_repetition_penalty, text_temperature, max_new_tokens,
                 reference_audio_s1=None, reference_text_s1=None, 
                 reference_audio_s2=None, reference_text_s2=None):
        
        model = moss_model["model"]
        processor = moss_model["processor"]
        device = moss_model["device"]
        
        # Attach audio tokenizer from codec node
        if moss_codec:
             processor.audio_tokenizer = moss_codec["audio_tokenizer"]
             # Check device of audio tokenizer? keep on CPU.
             processor.audio_tokenizer = processor.audio_tokenizer.cpu()

        target_sr = int(processor.model_config.sampling_rate)
        
        audio_inputs = []
        if reference_audio_s1: audio_inputs.append(self.preprocess_audio(reference_audio_s1, target_sr))
        if reference_audio_s2: audio_inputs.append(self.preprocess_audio(reference_audio_s2, target_sr))
            
        if audio_inputs:
            ref_wavs = audio_inputs # Already list of tensors
            
            # Ensure audio tokenizer is on CPU
            processor.audio_tokenizer = processor.audio_tokenizer.cpu()
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                reference_audio_codes = processor.encode_audios_from_wav(ref_wavs, sampling_rate=target_sr)

            full_prompt = ""
            if reference_text_s1: full_prompt += f"{reference_text_s1} "
            if reference_text_s2: full_prompt += f"{reference_text_s2} "
            full_prompt += text

            concat_wav = torch.cat(ref_wavs, dim=-1)

            with torch.no_grad():
                prompt_audio_list = processor.encode_audios_from_wav([concat_wav], sampling_rate=target_sr)
            
            prompt_audio = prompt_audio_list[0]

            conversations = [
                [
                    processor.build_user_message(
                        text=full_prompt,
                        reference=reference_audio_codes,
                    ),
                    processor.build_assistant_message(
                        audio_codes_list=[prompt_audio]
                    ),
                ],
            ]
        else:
            # generation mode: conversation must have ODD length, last message = user
            conversations = [
                [
                    processor.build_user_message(text=text),
                ]
            ]

        # Inference
        batch = processor(conversations, mode="continuation" if audio_inputs else "generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                audio_temperature=audio_temperature,
                audio_top_p=audio_top_p,
                audio_repetition_penalty=audio_repetition_penalty,
                text_temperature=text_temperature,
            )

        generated_audio = []
        for message in processor.decode(outputs):
            for audio in message.audio_codes_list:
                generated_audio.append(audio.detach().cpu())

        if not generated_audio:
            # ComfyUI AUDIO expects shape [batch, channels, samples]
            return ({"waveform": torch.zeros(1, 1, target_sr), "sample_rate": target_sr},)

        final_audio = torch.cat(generated_audio, dim=-1)
        # Ensure shape is [batch, channels, samples] for ComfyUI
        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0).unsqueeze(0)  # [samples] -> [1, 1, samples]
        elif final_audio.dim() == 2:
            final_audio = final_audio.unsqueeze(0)  # [channels, samples] -> [1, channels, samples]

        return ({"waveform": final_audio, "sample_rate": target_sr},)

NODE_CLASS_MAPPINGS = {
    "MossAudioCodecLoadModel": MossAudioCodecLoadModel,
    "MossTTSDLoadModel": MossTTSDLoadModel,
    "MossTTSDGenerate": MossTTSDGenerate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossAudioCodecLoadModel": "Load MOSS-Audio Codec",
    "MossTTSDLoadModel": "Load MOSS-TTSD Model",
    "MossTTSDGenerate": "Run MOSS-TTSD Generation"
}
