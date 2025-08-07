#!/usr/bin/env python3
"""
示例：如何使用修改后的fairseq数据加载器

使用方法：
1. 准备fairseq格式的数据文件：
   - data_bin_at_0/train.bin, train.idx (音频token数据)
   - data_bin_st/train.bin, train.idx (语音token数据)
   - dict/dict.at.txt (音频token词典)
   - dict/dict.st.txt (语音token词典)
   - bpe_69.json (分词器)

2. 调用数据加载器
"""

from dataset import create_dataloader, create_multilingual_dataloader

def example_single_language():
    """单语言数据加载示例"""
    print("=== 单语言数据加载示例 ===")
    
    # 中文数据
    zh_dataloader = create_dataloader(
        data_dir="/path/to/zh_data",           # 包含data_bin_at_0, data_bin_st的目录
        dict_path="/path/to/dict",            # 包含dict.at.txt, dict.st.txt的目录  
        tokenizer_path="/path/to/bpe_69.json", # 分词器文件
        lang="zh",                            # 语言代码
        n_gpus=1,
        rank=0,
        num_workers=4,
        max_duration=120
    )
    
    # 测试数据加载
    for batch_idx, batch in enumerate(zh_dataloader):
        print(f"批次 {batch_idx}:")
        print(f"  音频特征形状: {batch['audio_features'].shape}")
        print(f"  文本token形状: {batch['text_tokens'].shape}")
        print(f"  语言: {batch['languages']}")
        print(f"  AT tokens数量: {len(batch['at_tokens'])}")
        print(f"  ST tokens数量: {len(batch['st_tokens'])}")
        
        if batch_idx >= 2:  # 只看前3个批次
            break

def example_multilingual():
    """多语言数据加载示例"""
    print("\n=== 多语言数据加载示例 ===")
    
    # 多语言数据
    multilingual_dataloader = create_multilingual_dataloader(
        data_dirs=[
            "/path/to/zh_data",  # 中文数据目录
            "/path/to/en_data",  # 英文数据目录
        ],
        dict_path="/path/to/dict",            # 共享词典目录
        tokenizer_path="/path/to/bpe_69.json", # 共享分词器
        langs=["zh", "en"],                   # 语言列表
        n_gpus=1,
        rank=0,
        num_workers=4,
        max_duration=120
    )
    
    # 测试多语言数据加载
    for batch_idx, batch in enumerate(multilingual_dataloader):
        print(f"批次 {batch_idx}:")
        print(f"  音频特征形状: {batch['audio_features'].shape}")
        print(f"  文本token形状: {batch['text_tokens'].shape}")
        print(f"  语言分布: {batch['languages']}")
        
        if batch_idx >= 2:  # 只看前3个批次
            break

def example_training_integration():
    """训练集成示例"""
    print("\n=== 训练集成示例 ===")
    
    # 创建数据加载器
    dataloader = create_dataloader(
        data_dir="/path/to/data",
        dict_path="/path/to/dict", 
        tokenizer_path="/path/to/bpe_69.json",
        lang="zh",
        max_duration=80  # 根据GPU内存调整
    )
    
    # 模拟训练循环
    for epoch in range(2):
        print(f"Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(dataloader):
            # 获取训练数据
            text_tokens = batch['text_tokens']          # [B, T_text]
            text_tokens_lens = batch['text_tokens_lens'] # [B]
            audio_features = batch['audio_features']     # [B, T_audio, 8]  
            audio_features_lens = batch['audio_features_lens'] # [B]
            languages = batch['languages']              # [B]
            
            # 这里可以传递给模型进行训练
            # loss = model(
            #     x=text_tokens,
            #     x_lens=text_tokens_lens, 
            #     y=audio_features,
            #     y_lens=audio_features_lens,
            #     languages=languages  # 新增语言信息！
            # )
            
            print(f"  批次 {batch_idx}: 文本长度={text_tokens_lens.max().item()}, "
                  f"音频长度={audio_features_lens.max().item()}")
            
            if batch_idx >= 3:  # 每个epoch只看前几个批次
                break

if __name__ == "__main__":
    # 注意：运行前需要准备实际的数据文件
    print("请修改数据路径后运行示例")
    
    # example_single_language()
    # example_multilingual() 
    # example_training_integration()
