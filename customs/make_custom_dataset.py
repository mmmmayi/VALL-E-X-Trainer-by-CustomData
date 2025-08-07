import h5py
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

def create_dataset(data_dirs, dataloader_process_only, output_dir, supported_formats=['.wav', '.flac']):
    if dataloader_process_only:
        
        # Ensure output directory is absolute path
        output_dir = os.path.abspath(output_dir)
        
        # Create output directory (if it doesn't exist)
        
        os.makedirs(output_dir, exist_ok=True)
        
        h5_output_path = os.path.join(output_dir, "audio_sum.hdf5")
        ann_output_path = os.path.join(output_dir, "audio_ann_sum.txt")
        
        # Find all audio files
        audio_paths = find_audio_files(data_dirs, supported_formats)
        
        if not audio_paths:
            print(f"Error: No supported audio files found in specified directories {supported_formats}")
            return None


        # Clear annotation file (if it exists)
        if os.path.exists(ann_output_path):
            os.remove(ann_output_path)
        
        # Create or open HDF5 file
        with h5py.File(h5_output_path, 'w') as h5_file:
            processed_count = 0
            failed_count = 0
            
            # Process each audio file
            for i, audio_path in enumerate(tqdm(audio_paths, desc="Processing audio files")):
                try:
                    # Generate unique stem (include path info to avoid duplicates)
                    rel_path = os.path.relpath(audio_path)
                    stem = rel_path.replace('/', '_').replace('\\', '_').replace('.', '_')
                    # Ensure stem is a valid HDF5 group name
                    stem = ''.join(c for c in stem if c.isalnum() or c in ('_', '-'))
                    
                    # Process audio file
                    audio_tokens, text_tokens, langs, text = make_prompts(name=stem, audio_prompt_path=audio_path)
                    
                    text_tokens = text_tokens.squeeze(0)
                    
                    # Create HDF5 group
                    grp = h5_file.create_group(stem)
                    grp.create_dataset('audio', data=audio_tokens)
                    
                    # Write to annotation file
                    with open(ann_output_path, 'a', encoding='utf-8') as ann_file:
                        audio, sample_rate = sf.read(audio_path)
                        duration = len(audio) / sample_rate
                        ann_file.write(f'{stem}|{duration}|{langs[0]}|{text}\n')

                    
                except Exception as e:
                    failed_count += 1
                    print(f"  Processing failed: {e}")
                    continue
    else:
        # dataloader mode requires single directory, use the first one
        if isinstance(data_dirs, list):
            data_dir = data_dirs[0]
            if len(data_dirs) > 1:
                print(f"Warning: dataloader mode only supports single directory, will use the first directory: {data_dir}")
        else:
            data_dir = data_dirs

        dataloader = create_dataloader(data_dir=data_dir)
        return dataloader

def main():

    import argparse
    
    parser = argparse.ArgumentParser(description='Create VALL-E X custom dataset')
    parser.add_argument('--data_dirs', nargs='+', required=True,
                       help='Input data directory paths (can be multiple)')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory path')
    parser.add_argument('--formats', nargs='+', default=['.wav', '.flac'],
                       help='Supported audio formats (default: .wav .flac)')
    parser.add_argument('--dataloader_only', action='store_true',
                       help='Only create dataloader, do not process data files')
    
    args = parser.parse_args()
    

    
    try:
        # Call create_dataset function
        result = create_dataset(
            data_dirs=args.data_dirs,
            dataloader_process_only=not args.dataloader_only,
            output_dir=args.output_dir,
            supported_formats=args.formats
        )
        
        if result is not None:
            print("Dataset creation completed!")
            if args.dataloader_only:
                print("Dataloader created")
            else:
                print(f"Data files saved to: {args.output_dir}")
        else:
            print("Dataset creation failed")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())