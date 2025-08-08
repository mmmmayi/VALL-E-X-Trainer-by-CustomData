#!/usr/bin/env python3
"""
Test script to verify imports work correctly
"""

import os
import sys

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")
print(f"Python path: {sys.path[:3]}")  # Show first 3 entries

try:
    from utils.g2p.symbols import symbols
    print("✅ Successfully imported symbols")
except ImportError as e:
    print(f"❌ Failed to import symbols: {e}")

try:
    from utils.g2p import PhonemeBpeTokenizer
    print("✅ Successfully imported PhonemeBpeTokenizer")
except ImportError as e:
    print(f"❌ Failed to import PhonemeBpeTokenizer: {e}")

try:
    from utils.prompt_making import make_prompt, make_transcript
    print("✅ Successfully imported prompt_making")
except ImportError as e:
    print(f"❌ Failed to import prompt_making: {e}")

try:
    from data.collation import get_text_token_collater
    print("✅ Successfully imported collation")
except ImportError as e:
    print(f"❌ Failed to import collation: {e}")

try:
    from data.dataset import create_dataloader
    print("✅ Successfully imported dataset")
except ImportError as e:
    print(f"❌ Failed to import dataset: {e}")

try:
    from data.tokenizer import AudioTokenizer, tokenize_audio
    print("✅ Successfully imported tokenizer")
except ImportError as e:
    print(f"❌ Failed to import tokenizer: {e}")

print("\n🎉 All imports tested!")

