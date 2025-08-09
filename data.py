import argparse
import copy
import logging
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from pathlib import Path

from customs.make_custom_dataset import create_dataset

def main():

    import argparse

    parser = argparse.ArgumentParser(description='Create VALL-E X custom dataset')
    parser.add_argument('--data_dirs', nargs='+', required=True,
                       help='Input data directory paths (can be multiple)')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory path')
    parser.add_argument('--formats', nargs='+', default=['.wav', '.flac'],
                       help='Supported audio formats (default: .wav .flac)')
    parser.add_argument('--dataloader_only', required=True,
                       help='Only create dataloader, do not process data files')

    args = parser.parse_args()



    try:
        # Call create_dataset function
        result = create_dataset(
            data_dirs=args.data_dirs,
            dataloader_process_only=args.dataloader_only,
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
