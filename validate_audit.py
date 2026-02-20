import os
import sys
import ast

# Configuration
MERS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mers")
SRC_DIR = os.path.join(MERS_ROOT, "src")
SERVER_FILE = os.path.join(MERS_ROOT, "server.py")

REQUIRED_FILES = [
    os.path.join(SRC_DIR, "visual_engine.py"),
    os.path.join(SRC_DIR, "audio_engine.py"),
    os.path.join(SRC_DIR, "fusion_engine.py"),
    SERVER_FILE
]

FORBIDDEN_IMPORTS = {
    "dlib": "Legacy dependency (Visual)",
    "pyaudio": "Legacy dependency (Audio)",
    "tensorflow": "Legacy dependency (Heavy Model)",
    "keras": "Legacy dependency (Heavy Model)"
}

def check_file_exists(path):
    if not os.path.exists(path):
        print(f"[FAIL] Missing file: {path}")
        return False
    return True

def check_imports(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    
    found_forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTS:
                    found_forbidden.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module in FORBIDDEN_IMPORTS:
                found_forbidden.append(node.module)
    
    if found_forbidden:
        print(f"[FAIL] Forbidden imports in {os.path.basename(file_path)}: {found_forbidden}")
        for imp in found_forbidden:
            print(f"  - {imp}: {FORBIDDEN_IMPORTS[imp]}")
        return False
    return True

def check_class_structure(file_path, class_name, required_methods):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    
    class_found = False
    methods_found = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            class_found = True
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods_found.add(item.name)
    
    if not class_found:
        print(f"[FAIL] Class '{class_name}' not found in {os.path.basename(file_path)}")
        return False
    
    missing_methods = [m for m in required_methods if m not in methods_found]
    if missing_methods:
        print(f"[FAIL] Missing methods in '{class_name}': {missing_methods}")
        return False
        
    print(f"[PASS] Structure validated for {class_name}")
    return True

def main():
    print("=== MERS Compliance Validation ===")
    all_pass = True
    
    # 1. File Existence
    print("\n--- Checking File Existence ---")
    for f in REQUIRED_FILES:
        if not check_file_exists(f):
            all_pass = False
        else:
            print(f"[PASS] Found {os.path.basename(f)}")

    # 2. Forbidden Imports
    print("\n--- Checking Forbidden Imports ---")
    for f in REQUIRED_FILES:
        if os.path.exists(f):
            if not check_imports(f):
                all_pass = False
            else:
                print(f"[PASS] No forbidden imports in {os.path.basename(f)}")

    # 3. Structure Validation
    print("\n--- Checking Code Structure ---")
    
    # VisualEngine
    if not check_class_structure(
        os.path.join(SRC_DIR, "visual_engine.py"), 
        "VisualEngine", 
        ["process_frame", "_detect_emotion_rules"]
    ):
        all_pass = False

    # AudioEngine
    if not check_class_structure(
        os.path.join(SRC_DIR, "audio_engine.py"), 
        "AudioEngine", 
        ["process_audio_data"]
    ):
        all_pass = False

    if all_pass:
        print("\n=== VALIDATION SUCCESSFUL: MERS Codebase Compliant ===")
        sys.exit(0)
    else:
        print("\n=== VALIDATION FAILED: Issues Found ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
