
import unittest
import sys
import os
import torch
from unittest.mock import MagicMock

# Dynamically load the class to bypass relative import issues
# The node uses `from .utils import ...` which fails in standalone script
def load_node_class():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, "nodes_voice_generator.py")
    
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    
    # 1. Remove the problematic relative import
    # "from .utils import ..." -> "pass # import removed"
    lines = code.split("\n")
    new_lines = []
    
    # We strip 'from .utils' block or just make it robust
    # Easier: just exec in a context where relative import is ignored or caught? No exec is simpler with modified source
    skip_import_block = False
    for line in lines:
        if "from .utils import" in line:
            new_lines.append("pass # stripped relative import")
            skip_import_block = True
        elif skip_import_block and line.strip().endswith(")"): # end of multiline import
            skip_import_block = False
        elif skip_import_block:
            continue # inside multiline import
        else:
            new_lines.append(line)
            
    modified_code = "\n".join(new_lines)
    
    # 2. Setup mock environment
    module_scope = {
        "os": os,
        "torch": torch,
        "torchaudio": MagicMock(),
        "AutoModel": MagicMock(),
        "AutoProcessor": MagicMock(),
        "folder_paths": None, # or mock
        # Mock utils functions used in code
        "get_model_path": MagicMock(),
        "auto_download": MagicMock(),
        "_patch_tqdm_for_comfyui": MagicMock(),
        "normalize_text": MagicMock(),
        "__file__": file_path
    }
    
    # Execute the modified code
    exec(modified_code, module_scope)
    
    return module_scope["MossSoundEffectGenerate"]

MossSoundEffectGenerate = load_node_class()

class TestMossSoundEffectGenerate(unittest.TestCase):
    def setUp(self):
        self.node = MossSoundEffectGenerate()
        self.device = "cpu"
        self.mock_model = MagicMock()
        self.mock_processor = MagicMock()
        self.mock_processor.model_config.sampling_rate = 24000
        
        self.moss_se_model = {
            "model": self.mock_model,
            "processor": self.mock_processor,
            "device": self.device,
            "dtype": torch.float32
        }

    def test_token_calculation(self):
        """Test if duration is correctly converted to tokens (12.5 tokens/s)."""
        duration = 10.0
        expected_tokens = int(10.0 * 12.5) # 125
        
        # Mock processor behavior
        mock_batch = {
            "input_ids": torch.zeros((1, 10)),
            "attention_mask": torch.zeros((1, 10))
        }
        self.mock_processor.return_value = mock_batch
        self.mock_processor.build_user_message.return_value = "mock_msg"
        
        # Mock generate return
        self.mock_model.generate.return_value = "mock_outputs"
        
        # Mock decode
        mock_wav = torch.randn(1, 24000)
        mock_message = MagicMock()
        mock_message.audio_codes_list = [mock_wav]
        self.mock_processor.decode.return_value = [mock_message]

        # Run generate
        self.node.generate(
            moss_se_model=self.moss_se_model,
            text="test",
            duration_seconds=duration,
            audio_temperature=1.0,
            audio_top_p=0.8,
            audio_top_k=50,
            audio_repetition_penalty=1.0,
            max_new_tokens=1000
        )
        
        # Verify tokens passed to build_user_message
        self.mock_processor.build_user_message.assert_called_with(text="test", tokens=expected_tokens)

    def test_output_shape_correction(self):
        """Verify output tensor is reshaped to [1, 1, Samples]."""
        mock_wav_1d = torch.randn(24000)
        self._run_mock_generate_with_wav(mock_wav_1d)
        
        mock_wav_2d = torch.randn(1, 24000)
        self._run_mock_generate_with_wav(mock_wav_2d)

    def _run_mock_generate_with_wav(self, mock_wav):
        mock_message = MagicMock()
        mock_message.audio_codes_list = [mock_wav]
        self.mock_processor.decode.return_value = [mock_message]
        self.mock_processor.return_value = {"input_ids": torch.zeros(1,1), "attention_mask": torch.zeros(1,1)}
        
        output = self.node.generate(
            moss_se_model=self.moss_se_model,
            text="test",
            duration_seconds=1.0,
            audio_temperature=1.0,
            audio_top_p=0.8,
            audio_top_k=50,
            audio_repetition_penalty=1.0,
            max_new_tokens=1000
        )
        
        result_waveform = output[0]["waveform"]
        self.assertEqual(result_waveform.dim(), 3, f"Output should be 3D, got {result_waveform.dim()}D shape {result_waveform.shape}")
        self.assertEqual(result_waveform.shape[0], 1)
        self.assertEqual(result_waveform.shape[1], 1)

if __name__ == '__main__':
    unittest.main()
