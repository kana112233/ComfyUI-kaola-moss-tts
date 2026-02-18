import os
import sys
import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path

# Add MOSS-TTSD to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
moss_path = os.path.join(current_dir, "MOSS-TTSD")
if moss_path not in sys.path:
    sys.path.append(moss_path)

from transformers import AutoModel, AutoProcessor
import transformers

# Monkey-patch for Qwen3 support (aliased to Qwen2)
# MOSS-TTSD references `transformers.models.qwen3` which doesn't exist in standard transformers.
try:
    from transformers.models import qwen2
    import sys
    
    if "transformers.models.qwen3" not in sys.modules:
        sys.modules["transformers.models.qwen3"] = qwen2
        transformers.models.qwen3 = qwen2
        
        # Alias classes
        if not hasattr(qwen2, "Qwen3Config"):
            setattr(qwen2, "Qwen3Config", qwen2.Qwen2Config)
        if not hasattr(qwen2, "Qwen3Model"):
            setattr(qwen2, "Qwen3Model", qwen2.Qwen2Model)
        if not hasattr(qwen2, "Qwen3ForCausalLM"):
            setattr(qwen2, "Qwen3ForCausalLM", qwen2.Qwen2ForCausalLM)
            
        print("Monkey-patched transformers.models.qwen3 -> qwen2")
except ImportError:
    print("Failed to patch Qwen3 support: transformers.models.qwen2 not found.")

# Monkey-patch for PreTrainedConfig in configuration_utils
# Some versions or environments might fail to expose it directly in configuration_utils due to circular imports or structure changes.
try:
    import transformers.configuration_utils
    if not hasattr(transformers.configuration_utils, "PreTrainedConfig"):
        if hasattr(transformers, "PreTrainedConfig"):
            transformers.configuration_utils.PreTrainedConfig = transformers.PreTrainedConfig
            print("Monkey-patched transformers.configuration_utils.PreTrainedConfig")
except ImportError:
    pass

# Try to import folder_paths from comfy
try:
    import folder_paths
    # Add a new folder type for moss_ttsd models
    folder_paths.add_model_folder_path("moss_ttsd", os.path.join(folder_paths.models_dir, "moss_ttsd"))
except ImportError:
    folder_paths = None

class MossTTSDLoadModel:
    @classmethod
    def INPUT_TYPES(s):
        # Default options
        model_options = ["OpenMOSS-Team/MOSS-TTSD-v1.0"]
        codec_options = ["OpenMOSS-Team/MOSS-Audio-Tokenizer"]
        
        if folder_paths:
            # Get list of local models if available
            local_models = folder_paths.get_filename_list("moss_ttsd")
            if local_models:
                 model_options.extend(local_models)
        
        return {
            "required": {
                "model_path": (model_options,),
                "codec_path": (codec_options,),
                "quantization": (["none", "8bit", "4bit"], {"default": "none"}),
            }
        }

    RETURN_TYPES = ("MOSS_TTSD_MODEL",)
    RETURN_NAMES = ("moss_model",)
    FUNCTION = "load"
    CATEGORY = "Kaola/MOSS-TTSD"

    def load(self, model_path, codec_path, quantization):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        if quantization != "none" and device == "cpu":
            print("Quantization not supported on CPU. Forcing 'none'.")
            quantization = "none"

        # Resolve paths if they are local filenames
        if folder_paths:
            if model_path in folder_paths.get_filename_list("moss_ttsd"):
                 model_path = folder_paths.get_full_path("moss_ttsd", model_path)
        
        # Use local model path if default HF Hub ID is specified
        if model_path == "OpenMOSS-Team/MOSS-TTSD-v1.0":
            try:
                import os
                # Use the existing v1.0 directory
                if folder_paths:
                    base_path = os.path.join(folder_paths.models_dir, "moss_ttsd")
                else:
                    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "moss_ttsd")

                local_model_path = os.path.join(base_path, "MOSS-TTSD-v1.0")

                if os.path.exists(local_model_path):
                    print(f"Using local MOSS-TTSD path: {local_model_path}")
                    model_path = local_model_path
            except Exception as e:
                print(f"Failed to redirect to local model: {e}")

        # Auto-download Tokenizer/Codec if default
        if codec_path == "OpenMOSS-Team/MOSS-Audio-Tokenizer":
            try:
                from huggingface_hub import snapshot_download
                import os
                if folder_paths:
                    base_path = os.path.join(folder_paths.models_dir, "moss_ttsd")
                else:
                    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "moss_ttsd")
                
                local_codec_path = os.path.join(base_path, "MOSS-Audio-Tokenizer")
                
                if not os.path.exists(local_codec_path):
                    print(f"Downloading MOSS-Audio-Tokenizer to {local_codec_path}...")
                    snapshot_download(repo_id=codec_path, local_dir=local_codec_path)
                
                print(f"Using local MOSS-Audio-Tokenizer: {local_codec_path}")
                codec_path = local_codec_path
            except Exception as e:
                print(f"Failed to auto-download/redirect codec: {e}")

        print(f"Loading MOSS-TTSD model from {model_path}...")
        # Use use_fast=False to avoid "data did not match any variant of untagged enum ModelWrapper"
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            codec_path=codec_path,
            use_fast=False,
        )
        
        # Audio tokenizer to CPU first to save VRAM and avoid OOM
        processor.audio_tokenizer = processor.audio_tokenizer.cpu().eval()

        # Check if accelerate is available for memory optimization
        try:
            import accelerate
            use_accelerate = True
        except ImportError:
            use_accelerate = False
            if quantization != "none":
                print("Warning: Quantization requires 'accelerate' and 'bitsandbytes'. Install them to use 8bit/4bit.")
            print("Accelerate not found. Install 'accelerate' for faster loading and less RAM usage.")

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if use_accelerate:
            model_kwargs["device_map"] = "auto"
            model_kwargs["low_cpu_mem_usage"] = True
        
        # Application of Quantization
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
            print(f"Failed to load with flash_attention_2/quantization settings, checking issues: {e}")
            
            # Fallback: try without quantization kwargs if they caused the issue
            if quantization != "none":
                 print("Fallback: Trying standard loading without quantization...")
                 if "load_in_8bit" in model_kwargs: del model_kwargs["load_in_8bit"]
                 if "load_in_4bit" in model_kwargs: del model_kwargs["load_in_4bit"]
                 if "bnb_4bit_compute_dtype" in model_kwargs: del model_kwargs["bnb_4bit_compute_dtype"]
            
            try:
                model = AutoModel.from_pretrained(
                    model_path,
                    attn_implementation="sdpa",
                    **model_kwargs
                )
            except Exception as e2:
                print(f"Second loading attempt failed: {e2}")
                # Last ditch attempt
                model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=dtype
                )

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
                "text": ("STRING", {"multiline": True, "default": "[S1] Hello world."}),
                "temperature": ("FLOAT", {"default": 1.1, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.1}),
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
        # ComfyUI audio format: {"waveform": tensor(ch, samples), "sample_rate": int}
        waveform = audio_data["waveform"]
        sr = audio_data["sample_rate"]
        
        # Ensure correct shape (channels, samples)
        if waveform.dim() == 3: # (batch, channels, samples)
             waveform = waveform.squeeze(0)
        
        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            
        return waveform

    def generate(self, moss_model, text, temperature, top_p, repetition_penalty, max_new_tokens,
                 reference_audio_s1=None, reference_text_s1=None, 
                 reference_audio_s2=None, reference_text_s2=None):
        
        model = moss_model["model"]
        processor = moss_model["processor"]
        device = moss_model["device"]
        
        target_sr = int(processor.model_config.sampling_rate)
        
        audio_inputs = []
        if reference_audio_s1:
            wav1 = self.preprocess_audio(reference_audio_s1, target_sr)
            audio_inputs.append(wav1)
        if reference_audio_s2:
            wav2 = self.preprocess_audio(reference_audio_s2, target_sr)
            audio_inputs.append(wav2)
            
        # Encode reference audios
        if audio_inputs:
            ref_wavs = []
            if reference_audio_s1: ref_wavs.append(self.preprocess_audio(reference_audio_s1, target_sr))
            if reference_audio_s2: ref_wavs.append(self.preprocess_audio(reference_audio_s2, target_sr))

            # Ensure audio tokenizer is on CPU to avoid VRAM OOM
            processor.audio_tokenizer = processor.audio_tokenizer.cpu()
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                reference_audio_codes = processor.encode_audios_from_wav(ref_wavs, sampling_rate=target_sr)

            # Construct full text prompt
            full_prompt = ""
            if reference_text_s1: full_prompt += f"{reference_text_s1} "
            if reference_text_s2: full_prompt += f"{reference_text_s2} "
            full_prompt += text

            # Concatenate wavs for the assistant prompt audio
            concat_wav = torch.cat(ref_wavs, dim=-1)

            # Encode prompt audio on CPU
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
            conversations = [
                [
                    processor.build_user_message(text=text),
                    processor.build_assistant_message(audio_codes_list=[]),
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
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        # Decode
        generated_audio = []
        for message in processor.decode(outputs):
            for audio in message.audio_codes_list:
                generated_audio.append(audio.detach().cpu())

        if not generated_audio:
            return ({"waveform": torch.zeros(1, target_sr), "sample_rate": target_sr},)

        # Concatenate generated segments
        final_audio = torch.cat(generated_audio, dim=-1)
        # Ensure (channels, samples)
        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0)

        return ({"waveform": final_audio, "sample_rate": target_sr},)

NODE_CLASS_MAPPINGS = {
    "MossTTSDLoadModel": MossTTSDLoadModel,
    "MossTTSDGenerate": MossTTSDGenerate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossTTSDLoadModel": "Load MOSS-TTSD Model",
    "MossTTSDGenerate": "Run MOSS-TTSD Generation"
}
