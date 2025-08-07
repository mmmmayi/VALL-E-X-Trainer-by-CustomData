# Fairseq数据加载器迁移说明

## 概述

将原本基于HDF5的数据读取方式改为使用Fairseq的索引数据集（IndexedDataset），参考了`SLAM-LLM/src/slam_llm/datasets/vallex_dataset.py`的实现方式。

## 主要修改

### 1. 依赖变更

**移除**：
- `h5py` - HDF5文件读取
- 基于文本注释文件的数据索引

**新增**：
- `fairseq.data.Dictionary` - Fairseq词典
- `fairseq.data.data_utils` - 数据工具
- `fairseq.data.indexed_dataset` - 索引数据集

### 2. AudioDataset类重构

#### 原来的实现：
```python
class AudioDataset:
    def __init__(self, h5_path, ann_path, tokenizer_path):
        # 从HDF5文件和注释文件读取数据
        self._archive = h5py.File(h5_path, "r")
```

#### 新的实现：
```python  
class AudioDataset:
    def __init__(self, data_path, dict_path, tokenizer_path, lang="zh"):
        # 使用Fairseq索引数据集
        self.at_dataset = data_utils.load_indexed_dataset(at_data_path, self.at_dict, None)
        self.st_dataset = data_utils.load_indexed_dataset(st_data_path, self.st_dict, None)
```

### 3. 数据结构变更

#### 输入数据格式：
- **原来**：HDF5文件 + 文本注释文件（格式：`文件名|时长|语言|文本`）
- **现在**：Fairseq二进制数据文件
  - `data_bin_at_0/train.{bin,idx}` - 音频token数据
  - `data_bin_st/train.{bin,idx}` - 语音token数据  
  - `dict.at.txt` - 音频token词典
  - `dict.st.txt` - 语音token词典

#### 数据加载流程：
```python
# 原来
audio_tokens = archive[h5_path]['audio'][()]
text = self.texts[idx]

# 现在  
at_tokens = self.at_dataset[idx]  # 音频tokens
st_tokens = self.st_dataset[idx]  # 语音tokens
```

### 4. 新增功能

#### 多语言支持：
```python
def create_multilingual_dataloader(data_dirs, dict_path, tokenizer_path, langs=["zh", "en"]):
    # 支持同时加载多种语言的数据
```

#### 语言信息传递：
数据批次中现在包含语言ID，可以在训练时使用：
```python
batch = {
    'languages': torch.LongTensor([0, 1, 0, 1]),  # 0=zh, 1=en
    'at_tokens': [...],  # 原始AT tokens
    'st_tokens': [...],  # 原始ST tokens
    # ... 其他字段
}
```

## 使用方法

### 单语言数据加载：
```python
from data.dataset import create_dataloader

dataloader = create_dataloader(
    data_dir="/path/to/zh_data",           # 包含data_bin_at_0, data_bin_st
    dict_path="/path/to/dict",            # 包含dict.at.txt, dict.st.txt  
    tokenizer_path="/path/to/bpe_69.json",
    lang="zh",
    max_duration=120
)
```

### 多语言数据加载：
```python
from data.dataset import create_multilingual_dataloader

dataloader = create_multilingual_dataloader(
    data_dirs=["/path/to/zh_data", "/path/to/en_data"],
    dict_path="/path/to/dict",
    tokenizer_path="/path/to/bpe_69.json", 
    langs=["zh", "en"],
    max_duration=120
)
```

### 训练集成：
```python
for batch in dataloader:
    # 现在可以获取语言信息
    languages = batch['languages']  # [B] 语言ID张量
    
    # 传递给模型（需要修改模型forward方法支持语言参数）
    loss = model(
        x=batch['text_tokens'],
        x_lens=batch['text_tokens_lens'],
        y=batch['audio_features'], 
        y_lens=batch['audio_features_lens'],
        languages=languages  # 新增！
    )
```

## 数据准备

### 1. 准备Fairseq格式数据
需要使用Fairseq工具将原始数据转换为二进制格式：

```bash
# 示例：准备音频token数据
fairseq-preprocess \
    --only-source \
    --srcdict dict.at.txt \
    --source-lang at \
    --dataset-impl mmap \
    --destdir data_bin_at_0 \
    --source train.at

# 示例：准备语音token数据  
fairseq-preprocess \
    --only-source \
    --srcdict dict.st.txt \
    --source-lang st \
    --dataset-impl mmap \
    --destdir data_bin_st \
    --source train.st
```

### 2. 目录结构
```
data/
├── dict/
│   ├── dict.at.txt      # 音频token词典
│   └── dict.st.txt      # 语音token词典
├── data_bin_at_0/
│   ├── train.bin        # 音频token二进制数据
│   └── train.idx        # 音频token索引
├── data_bin_st/
│   ├── train.bin        # 语音token二进制数据  
│   └── train.idx        # 语音token索引
└── bpe_69.json          # 分词器
```

## 优势

1. **更高效的数据加载**：Fairseq的索引数据集比HDF5更快
2. **内存映射**：支持大数据集的内存映射加载
3. **标准化**：与SLAM-LLM项目的数据格式保持一致
4. **多语言支持**：天然支持多语言训练
5. **可扩展性**：易于添加新的语言和数据源

## 向后兼容

虽然API发生了变化，但核心功能保持不变：
- 动态批处理（DynamicBatchSampler）
- 数据并行（DistributedSampler）  
- 自定义collate函数

只需要更新数据路径和调用方式即可。

## 注意事项

1. 需要确保Fairseq数据文件存在且格式正确
2. 词典文件必须与数据文件匹配
3. 多语言训练时，需要修改模型的forward方法以支持语言参数
4. 建议先在小数据集上测试新的数据加载器
