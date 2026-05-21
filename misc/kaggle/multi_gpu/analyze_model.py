"""
Model Output Analysis — Dispatcher
====================================

Reads `task` from config.yaml and runs the appropriate analyzer:
  - "sort"       → analyze_model_sorting.py
  - "arithmetic" → analyze_model_arithmetic.py

Usage:
    python misc/kaggle/multi_gpu/analyze_model.py
"""
import os
import sys
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(BASE_DIR, 'config.yaml')

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

task = CFG['training'].get('task', 'sort')

if __name__ == '__main__':
    if task == 'arithmetic':
        print(f"  Task: arithmetic → running analyze_model_arithmetic.py")
        from analyze_model_arithmetic import main
        main()
    elif task == 'sort':
        print(f"  Task: sort → running analyze_model_sorting.py")
        from analyze_model_sorting import main
        main()
    else:
        print(f"  ERROR: Unknown task '{task}' in config.yaml. Expected 'sort' or 'arithmetic'.")
        sys.exit(1)
