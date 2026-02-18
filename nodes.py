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

# Try to import folder_paths from comfy
try:
    import folder_paths
    # Add a new folder type for moss_ttsd models
    folder_paths.add_model_folder_path("moss_ttsd", os.path.join(folder_paths.models_dir, "moss_ttsd"))
except ImportError:
    folder_paths = None

class MossTTSDNode:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

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
            
            # Also check LLM folder? Or just keep it strictly moss_ttsd for now.
            # We can also look in 'LLM' or 'transformers' folders if they exist
            # but let's stick to specific folder or HF ID.
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "[S1] Hello world."}),
                "model_path": (model_options,),
                "codec_path": (codec_options,),
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

    def load_model(self, model_path, codec_path):
        # Resolve paths if they are local filenames
        if folder_paths:
            if model_path in folder_paths.get_filename_list("moss_ttsd"):
                 model_path = folder_paths.get_full_path("moss_ttsd", model_path)
            
            # Since codec path logic is similar, we could add a folder for it too,
            # currently just treating it as raw string or HF Hub ID unless we add specific folder logic.
        
        # Workaround for transformers bug with "v1.0" in repo name causing import errors
        if model_path == "OpenMOSS-Team/MOSS-TTSD-v1.0":
            try:
                from huggingface_hub import snapshot_download
                import os
                
                # Define sanitized local path
                # Use folder_paths if available, otherwise relative to current file
                if folder_paths:
                    base_path = os.path.join(folder_paths.models_dir, "moss_ttsd")
                else:
                    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "moss_ttsd")
                
                sanitized_name = "MOSS-TTSD-v1_0" # Replace dot with underscore
                local_model_path = os.path.join(base_path, sanitized_name)
                
                if not os.path.exists(local_model_path):
                    print(f"Downloading MOSS-TTSD to {local_model_path} to avoid import errors...")
                    snapshot_download(repo_id=model_path, local_dir=local_model_path)
                
                print(f"Redirecting 'OpenMOSS-Team/MOSS-TTSD-v1.0' to local sanitized path: {local_model_path}")
                model_path = local_model_path
                
            except Exception as e:
                print(f"Failed to auto-download/redirect model: {e}")
                print("Falling back to default HF loading (might fail with ModuleNotFoundError).")

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

        if self.model is None or self.processor is None:
            print(f"Loading MOSS-TTSD model from {model_path}...")
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                codec_path=codec_path,
            )
            self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(self.device).eval()
            
            attn_implementation = "flash_attention_2" if self.device == "cuda" else "sdpa"
            try:
                self.model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                    torch_dtype=self.dtype,
                ).to(self.device).eval()
            except Exception as e:
                print(f"Failed to load with flash_attention_2, falling back to sdpa: {e}")
                self.model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                    torch_dtype=self.dtype,
                ).to(self.device).eval()

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

    def generate(self, text, model_path, codec_path, temperature, top_p, repetition_penalty, max_new_tokens,
                 reference_audio_s1=None, reference_text_s1=None, 
                 reference_audio_s2=None, reference_text_s2=None):
        
        self.load_model(model_path, codec_path)
        target_sr = int(self.processor.model_config.sampling_rate)
        
        audio_inputs = []
        if reference_audio_s1:
            wav1 = self.preprocess_audio(reference_audio_s1, target_sr)
            audio_inputs.append(wav1)
        if reference_audio_s2:
            wav2 = self.preprocess_audio(reference_audio_s2, target_sr)
            audio_inputs.append(wav2)
            
        # Encode reference audios
        if audio_inputs:
            # Need to handle merging multiple references if provided
            # For simplicity, following the example structure
            # But the example concatenates them.
            # If we align with prompts S1/S2...
            
            # Re-reading example logic:
            # It encodes audios from wav for reference_audio_codes (used in user message)
            # And also encodes for prompt_audio (used in assistant message)
            
            # Let's simplify: if we have ref audio, use it.
            # The example combines wav1 and wav2 into one "reference_audio_codes" for the prompt??
            # Actually:
            # reference_audio_codes = processor.encode_audios_from_wav([wav1, wav2], sampling_rate=target_sr)
            # This returns a list of codes.
            
            ref_wavs = []
            if reference_audio_s1: ref_wavs.append(self.preprocess_audio(reference_audio_s1, target_sr))
            if reference_audio_s2: ref_wavs.append(self.preprocess_audio(reference_audio_s2, target_sr))
            
            # Handle empty ref (no cloning?)
            if not ref_wavs:
                 # What if no reference? Use pre-trained voices?
                 # MOSS-TTSD seems to require reference for continuation/cloning.
                 # If no ref provided, maybe fail or try generation mode without continuation?
                 # The example uses "continuation" mode.
                 pass

            reference_audio_codes = self.processor.encode_audios_from_wav(ref_wavs, sampling_rate=target_sr)
            
            # For prompt_audio (assistant message start), example concatenates prompts
            # concat_prompt_wav = torch.cat([wav1, wav2], dim=-1)
            # prompt_audio = processor.encode_audios_from_wav([concat_prompt_wav], sampling_rate=target_sr)[0]
            
            # Construct full text prompt
            # prompt_text_speaker1 + prompt_text_speaker2 + text_to_generate
            full_prompt = ""
            if reference_text_s1: full_prompt += f"{reference_text_s1} "
            if reference_text_s2: full_prompt += f"{reference_text_s2} "
            full_prompt += text
            
            # Concatenate wavs for the assistant prompt audio
            max_len = max([w.shape[-1] for w in ref_wavs])
            # We need to concat them along time dimension?
            # Example: torch.cat([wav1, wav2], dim=-1)
            # Ensure same number of channels?
            # Comfy audio usually stereo or mono. MOSS might expect mono?
            # sf.read returns (samples, channels) usually, but ALWAYS_2D=True
            # preprocess_audio returns (channels, samples).
            # We should probably mix down to mono if needed?
            
            # Let's just concat for now.
            concat_wav = torch.cat(ref_wavs, dim=-1)
            prompt_audio_list = self.processor.encode_audios_from_wav([concat_wav], sampling_rate=target_sr)
            prompt_audio = prompt_audio_list[0]

            conversations = [
                [
                    self.processor.build_user_message(
                        text=full_prompt,
                        reference=reference_audio_codes,
                    ),
                    self.processor.build_assistant_message(
                        audio_codes_list=[prompt_audio]
                    ),
                ],
            ]
        else:
            # No reference mode?
            # "generation" mode might not need references?
            # For now assume reference is provided or fallback to simple generation
            # If no reference, maybe just text?
            conversations = [
                [
                    self.processor.build_user_message(text=text),
                    self.processor.build_assistant_message(audio_codes_list=[]),
                ]
            ]

        # Inference
        batch = self.processor(conversations, mode="continuation" if audio_inputs else "generation")
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        # Decode
        generated_audio = []
        for message in self.processor.decode(outputs):
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
    "MossTTSDNode": MossTTSDNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossTTSDNode": "MOSS-TTSD Generation"
}
