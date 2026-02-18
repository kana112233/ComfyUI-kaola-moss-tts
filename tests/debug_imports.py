import sys
import os
import torch

# Add parent directory to sys.path to import nodes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes import MossTTSDLoadModel, MossTTSDGenerate

def test_generation():
    print("Testing MOSS-TTSD Node logic...")
    
    # 1. Initialize Loader
    loader = MossTTSDLoadModel()
    
    # Mock folder_paths if not available (for standalone testing)
    # The node handles folder_paths=None internally, so we can just pass strings.
    
    model_path = "OpenMOSS-Team/MOSS-TTSD-v1.0"
    codec_path = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
    quantization = "none" # or "8bit", "4bit" if you have bitsandbytes

    print(f"Loading model: {model_path}, quantization: {quantization}")
    
    try:
        # returns (moss_model,) tuple
        result = loader.load(model_path, codec_path, quantization)
        moss_model = result[0]
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Initialize Generator
    generator = MossTTSDGenerate()
    
    text = "Hello, this is a test of the MOSS TTSD system."
    print(f"Generating text: {text}")
    
    try:
        # generate returns ({"waveform": ..., "sample_rate": ...},)
        output = generator.generate(
            moss_model=moss_model,
            text=text,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.0,
            max_new_tokens=100  # Short generation for test
        )
        
        audio_data = output[0]
        waveform = audio_data["waveform"]
        sample_rate = audio_data["sample_rate"]
        
        print(f"Generation successful!")
        print(f"Output shape: {waveform.shape}, Sample rate: {sample_rate}")
        
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generation()
