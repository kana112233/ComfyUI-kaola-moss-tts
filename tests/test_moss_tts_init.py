
import sys
import os
import importlib

# Add the parent directory of the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # ComfyUI-kaola-moss-ttsd
custom_nodes_dir = os.path.dirname(project_root) # parent of project

sys.path.insert(0, custom_nodes_dir)

package_name = os.path.basename(project_root)

print(f"Attempting to import package: {package_name}")

try:
    module = importlib.import_module(package_name)
    print(f"Successfully imported {package_name}")
    
    mappings = getattr(module, "NODE_CLASS_MAPPINGS", {})
    
    expected_nodes = [
        "MossTTSLoadModel",
        "MossTTSGenerate"
    ]
    
    all_found = True
    for node in expected_nodes:
        if node in mappings:
            print(f"SUCCESS: Found {node}")
        else:
            print(f"FAILURE: Missing {node}")
            all_found = False
            
    if all_found:
        print("All MOSS-TTS Foundation nodes verified.")
    else:
        sys.exit(1)

except ImportError as e:
    print(f"FAILURE: Initial import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FAILURE: Runtime error during import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
