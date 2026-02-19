import os
import re
import sys
import torch
import torchaudio
import soundfile as sf
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from contextlib import contextmanager
from tqdm import tqdm as _original_tqdm

# Try to import ComfyUI's folder_paths and utils
try:
    import folder_paths
    folder_paths.add_model_folder_path("moss_ttsd", os.path.join(folder_paths.models_dir, "moss_ttsd"))
except ImportError:
    folder_paths = None

try:
    import comfy.utils
    HAS_COMFY_PBAR = True
except ImportError:
    HAS_COMFY_PBAR = False


# ── Text utilities ───────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Normalize text for MOSS-TTSD: clean punctuation, merge speaker tags."""
    text = re.sub(r"\[(\d+)\]", r"[S\1]", text)
    remove_chars = "【】《》（）『』「」\" '\"-_\u201c\u201d～~\u2018\u2019"
    segments = re.split(r"(?=\[S\d+\])", text.replace("\n", " "))
    processed_parts = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        m = re.match(r"^(\[S\d+\])\s*(.*)", seg)
        tag, content = m.groups() if m else ("", seg)
        content = re.sub(f"[{re.escape(remove_chars)}]", "", content)
        content = re.sub(r"哈{2,}", "[笑]", content)
        content = re.sub(r"\b(ha(\s*ha)+)\b", "[laugh]", content, flags=re.IGNORECASE)
        content = content.replace("——", "，").replace("……", "，").replace("...", "，")
        content = content.replace("⸺", "，").replace("―", "，").replace("—", "，").replace("…", "，")
        internal_punct_map = str.maketrans({"；": "，", ";": ",", "：": "，", ":": ",", "、": "，"})
        content = content.translate(internal_punct_map).strip()
        content = re.sub(r"([，。？！,.?!])[，。？！,.?!]+", r"\1", content)
        if len(content) > 1:
            last_ch = "。" if content[-1] == "，" else ("." if content[-1] == "," else content[-1])
            body = content[:-1].replace("。", "，")
            content = body + last_ch
        processed_parts.append({"tag": tag, "content": content})
    if not processed_parts:
        return ""
    merged_lines = []
    current_tag = processed_parts[0]["tag"]
    current_content = [processed_parts[0]["content"]]
    for part in processed_parts[1:]:
        if part["tag"] == current_tag and current_tag:
            current_content.append(part["content"])
        else:
            merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())
            current_tag = part["tag"]
            current_content = [part["content"]]
    merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())
    return "".join(merged_lines).replace("\u2018", "'").replace("\u2019", "'")


def _merge_consecutive_speaker_tags(text: str) -> str:
    """Merge consecutive identical speaker tags, e.g. [S1]...[S1]... -> [S1]..."""
    segments = re.split(r"(?=\[S\d+\])", text)
    if not segments:
        return text
    merged_parts: List[str] = []
    current_tag = None
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        m = re.match(r"^(\[S\d+\])\s*(.*)", seg, re.DOTALL)
        if m:
            tag, content = m.groups()
            if tag == current_tag:
                merged_parts.append(content)
            else:
                current_tag = tag
                merged_parts.append(f"{tag}{content}")
        else:
            merged_parts.append(seg)
    return "".join(merged_parts)


def _build_prefixed_text(text: str, ref_texts: List[str]) -> str:
    """Prepend reference texts (with speaker tags) before target text, then merge tags."""
    parts: List[str] = []
    for i, rt in enumerate(ref_texts):
        tag = f"[S{i+1}]"
        if not rt.lstrip().startswith(tag):
            rt = f"{tag}{rt}"
        parts.append(rt)
    return _merge_consecutive_speaker_tags("".join(parts) + text)


# ── Progress Bar ─────────────────────────────────────────────────────────────

class ComfyUITqdm(_original_tqdm):
    """A tqdm wrapper that also updates ComfyUI's ProgressBar."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if HAS_COMFY_PBAR and self.total:
            self._comfy_pbar = comfy.utils.ProgressBar(self.total)
        else:
            self._comfy_pbar = None

    def update(self, n=1):
        super().update(n)
        if self._comfy_pbar:
            self._comfy_pbar.update_absolute(self.n, self.total)


@contextmanager
def _patch_tqdm_for_comfyui(model):
    """Temporarily replace tqdm in the model's module with ComfyUITqdm."""
    model_module = sys.modules.get(type(model).__module__)
    if model_module and hasattr(model_module, 'tqdm'):
        original = model_module.tqdm
        model_module.tqdm = ComfyUITqdm
        try:
            yield
        finally:
            model_module.tqdm = original
    else:
        yield


# ── Model Path & Download Utilities ──────────────────────────────────────────

def get_model_path(model_path):
    """Resolve local paths for models using ComfyUI's folder_paths if available."""
    if folder_paths:
        if model_path in folder_paths.get_filename_list("moss_ttsd"):
             return folder_paths.get_full_path("moss_ttsd", model_path)
    
    # Handle known HF IDs or general paths
    if model_path.startswith("OpenMOSS-Team/") or "/" in model_path:
        # Check if we have a local copy in models/moss_ttsd even if it's referenced by HF ID
        try:
            if folder_paths:
                base_path = os.path.join(folder_paths.models_dir, "moss_ttsd")
            else:
                base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "moss_ttsd")
            
            # Simple check: if valid local path doesn't exist, assume it might be a folder name derived from HF ID
            if not os.path.exists(model_path):
                folder_name = model_path.split("/")[-1]
                local_path = os.path.join(base_path, folder_name)

                if os.path.exists(local_path):
                    print(f"Using local path for {model_path}: {local_path}")
                    return local_path
        except Exception:
            pass
            
    return model_path


def auto_download(repo_id, local_dir_name):
    """Helper to auto-download models from HuggingFace if needed."""
    try:
        from huggingface_hub import snapshot_download
        if folder_paths:
            base_path = os.path.join(folder_paths.models_dir, "moss_ttsd")
        else:
            base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "moss_ttsd")
        
        local_path = os.path.join(base_path, local_dir_name)
        if not os.path.exists(local_path):
            print(f"[MOSS-TTSD] Local model not found at {local_path}.")
            print(f"[MOSS-TTSD] Attempting to download {repo_id} from HuggingFace...")
            print(f"[MOSS-TTSD] This may take a while depending on your internet connection.")
            try:
                snapshot_download(repo_id=repo_id, local_dir=local_path)
                print(f"[MOSS-TTSD] Download completed successfully.")
            except Exception as e:
                print(f"[MOSS-TTSD] Download failed: {e}")
                print(f"[MOSS-TTSD] Please download manually or check your connection.")
                raise e
        else:
             print(f"[MOSS-TTSD] Found local model at {local_path}")
             
        return local_path
    except ImportError:
        print("[MOSS-TTSD] huggingface_hub not installed, skipping auto-download check.")
        return repo_id
    except Exception as e:
        print(f"[MOSS-TTSD] Generic auto_download error: {e}")
        return repo_id
