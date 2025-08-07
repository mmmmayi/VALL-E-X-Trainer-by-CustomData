#!/usr/bin/env python3
"""
修改后的 make_custom_dataset.py 使用示例

🎯 重要变更：
- output_dir 参数现在是必需的！
- 所有输入目录的数据都会输出到同一个指定目录
- 支持多个不同路径的输入，统一输出管理
"""

from make_custom_dataset import create_dataset, find_audio_files

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 必须指定输出目录
    result = create_dataset(
        data_dirs="/path/to/audio/folder",
        dataloader_process_only=True,
        output_dir="/path/to/output/dataset",  # 必需参数！
        supported_formats=['.wav', '.flac']
    )
    
    print(f"处理结果: {result}")

def example_multiple_sources_single_output():
    """多个数据源，统一输出目录"""
    print("=== 多数据源统一输出示例 ===")
    
    # 多个不同来源的音频数据
    input_directories = [
        "/datasets/librispeech/train-clean-100",    # 英文数据集
        "/datasets/aishell/wav",                    # 中文数据集  
        "/datasets/jsut/wav",                       # 日文数据集
        "/recordings/custom_speech",                # 自定义录音
        "/downloads/podcast_audio",                 # 播客音频
    ]
    
    # 所有数据统一输出到一个目录
    unified_output = "/training/multilingual_dataset"
    
    print(f"将 {len(input_directories)} 个数据源合并到: {unified_output}")
    
    result = create_dataset(
        data_dirs=input_directories,
        dataloader_process_only=True,
        output_dir=unified_output,  # 统一输出位置
        supported_formats=['.wav', '.flac', '.mp3']
    )
    
    print(f"\n合并结果:")
    print(f"  📁 输出目录: {unified_output}")
    print(f"  📊 成功处理: {result['processed_count']} 文件")
    print(f"  ❌ 处理失败: {result['failed_count']} 文件")
    print(f"  📄 HDF5文件: {result['h5_path']}")
    print(f"  📝 标注文件: {result['ann_path']}")
    
    return result

def example_research_project():
    """研究项目使用示例"""
    print("=== 研究项目示例 ===")
    
    # 假设你在做多语言语音研究，需要整合各种数据源
    research_data_sources = [
        # 公开数据集
        "/datasets/commonvoice/en",
        "/datasets/commonvoice/zh", 
        "/datasets/commonvoice/ja",
        
        # 商业数据集
        "/datasets/librispeech/train-clean-360",
        "/datasets/vctk/wav48",
        
        # 实验室录制数据
        "/lab_recordings/emotions",
        "/lab_recordings/accents",
        
        # 合作伙伴数据
        "/partner_data/speech_corpus",
    ]
    
    # 项目专用的统一数据集目录
    project_dataset = "/research/multilingual_emotion_speech/dataset"
    
    print("🔬 研究项目数据整合:")
    print(f"   输入数据源: {len(research_data_sources)} 个")
    print(f"   输出位置: {project_dataset}")
    
    result = create_dataset(
        data_dirs=research_data_sources,
        dataloader_process_only=True,
        output_dir=project_dataset,
        supported_formats=['.wav', '.flac']  # 只使用高质量格式
    )
    
    # 项目数据统计
    print(f"\n📈 项目数据集统计:")
    print(f"   总音频文件: {result['total_files']}")
    print(f"   有效处理: {result['processed_count']}")
    print(f"   处理成功率: {result['processed_count']/result['total_files']*100:.1f}%")
    
    return result

def example_production_pipeline():
    """生产环境流水线示例"""
    print("=== 生产环境流水线示例 ===")
    
    # 生产环境中的多个数据源
    production_sources = [
        "/data/user_uploads/validated",      # 用户上传的验证音频
        "/data/tts_generated/high_quality",  # TTS生成的高质量音频
        "/data/studio_recordings/clean",     # 录音室录制的干净音频
        "/data/call_center/processed",       # 呼叫中心处理后的音频
    ]
    
    # 生产环境的训练数据目录
    production_dataset = "/ml_models/training_data/speech_v2.1"
    
    print("🏭 生产环境数据集构建:")
    for i, source in enumerate(production_sources, 1):
        print(f"   {i}. {source}")
    print(f"   ➡️  统一输出: {production_dataset}")
    
    try:
        result = create_dataset(
            data_dirs=production_sources,
            dataloader_process_only=True,
            output_dir=production_dataset,
            supported_formats=['.wav', '.flac']
        )
        
        print(f"\n✅ 生产数据集构建成功!")
        print(f"   版本: v2.1") 
        print(f"   文件数: {result['processed_count']}")
        print(f"   数据位置: {production_dataset}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ 生产数据集构建失败: {e}")
        return None

def example_error_handling():
    """错误处理示例"""
    print("=== 错误处理示例 ===")
    
    # 包含一些可能有问题的路径
    mixed_quality_sources = [
        "/datasets/good_data",           # 存在且有音频
        "/datasets/nonexistent_path",   # 不存在的路径
        "/datasets/empty_folder",       # 存在但没有音频文件
        "/datasets/corrupted_audio",    # 有损坏的音频文件
    ]
    
    try:
        result = create_dataset(
            data_dirs=mixed_quality_sources,
            dataloader_process_only=True,
            output_dir="/output/robust_dataset",
            supported_formats=['.wav', '.flac']
        )
        
        print(f"\n处理完成，即使有错误:")
        print(f"  成功: {result['processed_count']}")
        print(f"  失败: {result['failed_count']}")
        print(f"  总成功率: {result['processed_count']/result['total_files']*100:.1f}%")
        
    except ValueError as e:
        print(f"参数错误: {e}")
    except PermissionError as e:
        print(f"权限错误: {e}")
    except Exception as e:
        print(f"其他错误: {e}")

def example_simple_api():
    """简化API使用示例"""
    print("=== 简化API示例 ===")
    
    # 最简单的使用方式
    result = create_dataset(
        data_dirs="/path/to/audio",                    # 输入目录
        dataloader_process_only=True,                  # 处理模式  
        output_dir="/path/to/processed"                # 输出目录（必需！）
    )
    
    # 函数现在返回详细的处理结果
    return result

def example_validation():
    """参数验证示例"""
    print("=== 参数验证示例 ===")
    
    # 这些调用会失败，因为缺少必需参数
    try:
        # ❌ 缺少 output_dir - 会抛出 ValueError
        create_dataset(
            data_dirs="/path/to/audio",
            dataloader_process_only=True
            # output_dir=None  # 这样会失败！
        )
    except ValueError as e:
        print(f"❌ 预期的参数错误: {e}")
    
    try:
        # ❌ output_dir 为空字符串 - 会抛出 ValueError  
        create_dataset(
            data_dirs="/path/to/audio",
            dataloader_process_only=True,
            output_dir=""  # 空字符串也会失败！
        )
    except ValueError as e:
        print(f"❌ 预期的参数错误: {e}")
    
    # ✅ 正确的调用方式
    try:
        result = create_dataset(
            data_dirs="/path/to/audio",
            dataloader_process_only=True,
            output_dir="/valid/output/path"  # 必须提供有效路径
        )
        print("✅ 参数验证通过")
    except Exception as e:
        print(f"路径相关错误（正常，因为路径可能不存在）: {e}")

if __name__ == "__main__":
    print("🎯 make_custom_dataset.py 新版本使用指南\n")
    
    print("重要变更:")
    print("1. ✅ output_dir 参数现在是必需的")
    print("2. ✅ 支持多个输入目录，统一输出到指定位置") 
    print("3. ✅ 函数返回详细的处理结果信息")
    print("4. ✅ 更好的错误处理和验证")
    print("5. ✅ 详细的处理进度显示\n")
    
    print("请修改实际路径后运行以下示例:")
    print("- example_basic_usage()")
    print("- example_multiple_sources_single_output()")
    print("- example_research_project()")
    print("- example_production_pipeline()")
    print("- example_error_handling()")
    print("- example_validation()")
    
    # 运行验证示例（不需要实际文件）
    example_validation()
