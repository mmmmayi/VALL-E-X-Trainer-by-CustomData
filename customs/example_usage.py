#!/usr/bin/env python3
"""
make_custom_dataset.py 使用示例

修改后的功能：
1. 支持多个目录输入
2. 支持 .wav 和 .flac 格式
3. 递归搜索子目录
4. 智能处理文件名冲突
5. 详细的处理进度显示
"""

from make_custom_dataset import create_dataset, find_audio_files

def example_single_directory():
    """单目录使用示例"""
    print("=== 单目录示例 ===")
    
    # 原来的用法仍然支持
    data_dir = "/path/to/audio/folder"
    
    create_dataset(
        data_dirs=data_dir,              # 可以是字符串
        dataloader_process_only=True,
        output_dir=None,                 # 输出到 data_dir
        supported_formats=['.wav', '.flac']
    )

def example_multiple_directories():
    """多目录使用示例"""
    print("=== 多目录示例 ===")
    
    # 新功能：支持多个目录
    data_dirs = [
        "/path/to/english/audio",
        "/path/to/chinese/audio", 
        "/path/to/japanese/audio",
        "/path/to/mixed/audio"
    ]
    
    create_dataset(
        data_dirs=data_dirs,             # 列表形式
        dataloader_process_only=True,
        output_dir="/path/to/output",    # 指定输出目录
        supported_formats=['.wav', '.flac', '.mp3']  # 支持更多格式
    )

def example_find_files_only():
    """只查找文件不处理"""
    print("=== 查找文件示例 ===")
    
    # 只查找文件，不进行处理
    audio_files = find_audio_files(
        data_dirs=[
            "/path/to/folder1",
            "/path/to/folder2", 
            "/path/to/folder3"
        ],
        supported_formats=['.wav', '.flac']
    )
    
    print(f"找到 {len(audio_files)} 个音频文件:")
    for file in audio_files[:10]:  # 显示前10个
        print(f"  {file}")
    
    if len(audio_files) > 10:
        print(f"  ... 还有 {len(audio_files) - 10} 个文件")

def example_custom_formats():
    """自定义音频格式示例"""
    print("=== 自定义格式示例 ===")
    
    # 支持更多音频格式
    create_dataset(
        data_dirs="/path/to/audio",
        dataloader_process_only=True,
        supported_formats=['.wav', '.flac', '.mp3', '.m4a', '.ogg']
    )

def example_recursive_search():
    """递归搜索示例"""
    print("=== 递归搜索示例 ===")
    
    # 目录结构：
    # /audio_root/
    # ├── english/
    # │   ├── speaker1/
    # │   │   ├── file1.wav
    # │   │   └── file2.flac
    # │   └── speaker2/
    # │       └── file3.wav
    # ├── chinese/
    # │   └── file4.flac
    # └── mixed/
    #     ├── subfolder/
    #     │   └── file5.wav
    #     └── file6.flac
    
    # 会自动递归搜索所有子目录
    audio_files = find_audio_files(
        data_dirs="/audio_root",
        supported_formats=['.wav', '.flac']
    )
    
    # 预期结果：找到所有 6 个文件

def example_error_handling():
    """错误处理示例"""
    print("=== 错误处理示例 ===")
    
    # 包含不存在的目录
    data_dirs = [
        "/path/to/existing/folder",
        "/path/to/nonexistent/folder",  # 不存在
        "/another/valid/path"
    ]
    
    # 程序会自动跳过不存在的目录并继续处理
    create_dataset(
        data_dirs=data_dirs,
        dataloader_process_only=True,
        supported_formats=['.wav', '.flac']
    )

def example_output_control():
    """输出控制示例"""
    print("=== 输出控制示例 ===")
    
    # 场景1：单目录，输出到同目录
    create_dataset(
        data_dirs="/path/to/audio",
        dataloader_process_only=True,
        output_dir=None  # 输出到 /path/to/audio/
    )
    
    # 场景2：多目录，指定输出目录
    create_dataset(
        data_dirs=["/path/to/audio1", "/path/to/audio2"],
        dataloader_process_only=True,
        output_dir="/path/to/combined_output"  # 自定义输出位置
    )
    
    # 场景3：多目录，输出到第一个目录
    create_dataset(
        data_dirs=["/path/to/audio1", "/path/to/audio2"],
        dataloader_process_only=True,
        output_dir=None  # 输出到 /path/to/audio1/
    )

def example_real_usage():
    """实际使用示例"""
    print("=== 实际使用示例 ===")
    
    # 假设你有以下目录结构：
    # 
    # /datasets/
    # ├── librispeech/
    # │   ├── train-clean-100/
    # │   │   └── **/*.flac
    # │   └── train-clean-360/
    # │       └── **/*.flac
    # ├── ljspeech/
    # │   └── wavs/
    # │       └── *.wav
    # └── vctk/
    #     └── wav48/
    #         └── **/*.wav
    
    datasets_to_process = [
        "/datasets/librispeech/train-clean-100",
        "/datasets/librispeech/train-clean-360", 
        "/datasets/ljspeech/wavs",
        "/datasets/vctk/wav48"
    ]
    
    # 处理所有数据集
    create_dataset(
        data_dirs=datasets_to_process,
        dataloader_process_only=True,
        output_dir="/datasets/combined_training_data",
        supported_formats=['.wav', '.flac']
    )
    
    # 预期输出：
    # /datasets/combined_training_data/
    # ├── audio_sum.hdf5      # 所有音频的特征
    # └── audio_ann_sum.txt   # 标注文件

if __name__ == "__main__":
    # 请根据实际情况修改路径后运行
    print("请修改路径后运行相应的示例函数")
    
    # 取消注释以运行示例：
    # example_single_directory()
    # example_multiple_directories()
    # example_find_files_only()
    # example_custom_formats()
    # example_recursive_search()
    # example_error_handling()
    # example_output_control()
    # example_real_usage()
