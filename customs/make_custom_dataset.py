import glob
import torch
import numpy as np
import os
import torchaudio
import soundfile as sf
from utils.g2p.symbols import symbols
from utils.g2p import PhonemeBpeTokenizer
from utils.prompt_making import make_prompt, make_transcript
from data.collation import get_text_token_collater
from data.dataset import create_dataloader
from tqdm import tqdm
# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)

tokenizer_path = "./utils/g2p/bpe_69.json"
tokenizer = PhonemeBpeTokenizer(tokenizer_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_prompts(name, audio_prompt_path, transcript=None):
    text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
    text_collater = get_text_token_collater()
    codec = AudioTokenizer(device)
    wav_pr, sr = torchaudio.load(audio_prompt_path)
    # check length
    if wav_pr.size(-1) / sr > 15:
        raise ValueError(f"Prompt too long, expect length below 15 seconds, got {wav_pr / sr} seconds.")
    if wav_pr.size(0) == 2:
        wav_pr = wav_pr.mean(0, keepdim=True)
    text_pr, lang_pr = make_transcript(name, wav_pr, sr, transcript)

    # tokenize audio
    encoded_frames = tokenize_audio(codec, (wav_pr, sr))
    audio_tokens = encoded_frames[0][0].transpose(2, 1).cpu().numpy()

    # tokenize text
    phonemes, langs = text_tokenizer.tokenize(text=f"{text_pr}".strip())
    text_tokens, enroll_x_lens = text_collater(
        [
            phonemes
        ]
    )

    return audio_tokens, text_tokens, langs, text_pr
    
def find_audio_files(data_dirs, supported_formats=['.wav', '.flac']):
    """
    Find audio files from multiple directories
    
    Args:
        data_dirs: string (single path) or list (multiple paths)
        supported_formats: list of supported audio formats
    
    Returns:
        list of all found audio file paths
    """
    audio_paths = []
    
    # Process input: ensure data_dirs is in list format
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    
    # Iterate through all directories
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"Warning: directory does not exist - {data_dir}")
            continue
            
        print(f"Searching directory: {data_dir}")
        
        # Search for each supported format
        for format_ext in supported_formats:
            # Search current directory
            pattern = os.path.join(data_dir, f"*{format_ext}")
            files = glob.glob(pattern)
            
            # Recursively search subdirectories
            recursive_pattern = os.path.join(data_dir, f"**/*{format_ext}")
            recursive_files = glob.glob(recursive_pattern, recursive=True)
            
            # Merge results (remove duplicates)
            all_files = list(set(files + recursive_files))
            audio_paths.extend(all_files)
            
            print(f"  Found {len(all_files)} {format_ext} files")
    
    # Remove duplicates and sort
    audio_paths = sorted(list(set(audio_paths)))
    print(f"\nTotal found {len(audio_paths)} audio files")
    
    return audio_paths

def create_dataset(data_dir, lang):


    dataloader = create_dataloader(data_dir=data_dir,lang=lang)
    return dataloader

