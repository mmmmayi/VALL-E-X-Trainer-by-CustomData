# 多路径音频文件支持更新

## 概述

对 `make_custom_dataset.py` 进行了重大改进，现在支持：
- ✅ **多个目录输入**：可以同时处理多个不同的音频文件夹
- ✅ **多种音频格式**：支持 `.wav` 和 `.flac`，可扩展支持更多格式
- ✅ **递归搜索**：自动搜索子目录中的音频文件
- ✅ **智能文件名处理**：避免不同目录间的文件名冲突
- ✅ **详细进度显示**：实时显示处理进度和结果统计

## 主要变更

### 1. 新增 `find_audio_files` 函数

```python
def find_audio_files(data_dirs, supported_formats=['.wav', '.flac']):
    """
    从多个目录中查找音频文件
    
    Args:
        data_dirs: 字符串（单个路径）或列表（多个路径）
        supported_formats: 支持的音频格式列表
    
    Returns:
        所有找到的音频文件路径列表
    """
```

**功能特点**：
- 支持单个目录（字符串）或多个目录（列表）
- 递归搜索所有子目录
- 自动去重和排序
- 详细的搜索进度显示

### 2. 增强 `create_dataset` 函数

```python
def create_dataset(data_dirs, dataloader_process_only, output_dir=None, supported_formats=['.wav', '.flac']):
```

**新增参数**：
- `data_dirs`: 支持字符串或列表输入
- `output_dir`: 可选的输出目录
- `supported_formats`: 支持的音频格式列表

## 使用方式

### 原有用法（向后兼容）

```python
# 单目录，只处理 .wav 文件
create_dataset(
    data_dir="/path/to/audio",  # 原参数名仍支持
    dataloader_process_only=True
)
```

### 新用法示例

#### 1. 单目录 + 多格式

```python
create_dataset(
    data_dirs="/path/to/audio",
    dataloader_process_only=True,
    supported_formats=['.wav', '.flac']
)
```

#### 2. 多目录处理

```python
create_dataset(
    data_dirs=[
        "/path/to/english_audio",
        "/path/to/chinese_audio", 
        "/path/to/japanese_audio"
    ],
    dataloader_process_only=True,
    output_dir="/path/to/combined_output"
)
```

#### 3. 只查找文件不处理

```python
audio_files = find_audio_files(
    data_dirs=["/path1", "/path2", "/path3"],
    supported_formats=['.wav', '.flac', '.mp3']
)
print(f"找到 {len(audio_files)} 个音频文件")
```

## 技术改进

### 1. 智能文件名处理

```python
# 原来：可能出现重名冲突
stem = os.path.splitext(os.path.basename(audio_path))[0]

# 现在：包含路径信息，避免冲突
rel_path = os.path.relpath(audio_path)
stem = rel_path.replace('/', '_').replace('\\', '_').replace('.', '_')
stem = ''.join(c for c in stem if c.isalnum() or c in ('_', '-'))
```

**例子**：
```
输入文件：
- /data/english/speaker1/hello.wav
- /data/chinese/speaker2/hello.wav

生成的stem：
- data_english_speaker1_hello_wav
- data_chinese_speaker2_hello_wav
```

### 2. 递归搜索

```python
# 当前目录搜索
pattern = os.path.join(data_dir, f"*{format_ext}")
files = glob.glob(pattern)

# 递归搜索子目录
recursive_pattern = os.path.join(data_dir, f"**/*{format_ext}")
recursive_files = glob.glob(recursive_pattern, recursive=True)
```

### 3. 错误处理和进度显示

```python
# 详细的处理状态
processed_count = 0
failed_count = 0

for i, audio_path in enumerate(audio_paths):
    try:
        print(f"\n处理文件 {i+1}/{len(audio_paths)}: {audio_path}")
        # ... 处理逻辑 ...
        processed_count += 1
        print(f"  ✓ 成功处理 (时长: {duration:.2f}s, 语言: {langs[0]})")
    except Exception as e:
        failed_count += 1
        print(f"  ✗ 处理失败: {e}")
        continue

print(f"\n=== 处理完成 ===")
print(f"成功处理: {processed_count} 个文件")
print(f"处理失败: {failed_count} 个文件")
```

### 4. 灵活的输出控制

```python
# 自动确定输出目录
if output_dir is None:
    if isinstance(data_dirs, str):
        output_dir = data_dirs                    # 单目录：输出到该目录
    else:
        output_dir = data_dirs[0]                 # 多目录：输出到第一个目录

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)
```

## 实际使用场景

### 场景1：处理多个数据集

```python
# 合并多个公开数据集
datasets = [
    "/datasets/librispeech/train-clean-100",
    "/datasets/librispeech/train-clean-360",
    "/datasets/ljspeech/wavs", 
    "/datasets/vctk/wav48"
]

create_dataset(
    data_dirs=datasets,
    dataloader_process_only=True,
    output_dir="/datasets/combined_training_data",
    supported_formats=['.wav', '.flac']
)
```

### 场景2：处理分散的音频文件

```python
# 处理分散在不同位置的音频文件
audio_sources = [
    "/recordings/interviews",
    "/recordings/podcasts", 
    "/recordings/lectures",
    "/user_uploads/audio"
]

create_dataset(
    data_dirs=audio_sources,
    dataloader_process_only=True,
    output_dir="/processed_audio_dataset"
)
```

### 场景3：递归处理复杂目录结构

```python
# 自动处理深层嵌套的目录结构
# /audio_collection/
# ├── english/
# │   ├── news/
# │   │   ├── 2023/
# │   │   │   └── *.wav
# │   │   └── 2024/
# │   │       └── *.flac
# │   └── books/
# │       └── audiobooks/
# │           └── *.wav
# └── chinese/
#     └── podcasts/
#         └── *.flac

create_dataset(
    data_dirs="/audio_collection",
    dataloader_process_only=True,
    supported_formats=['.wav', '.flac']
)
# 自动找到所有子目录中的音频文件
```

## 性能优化

1. **去重机制**：使用 `set()` 去除重复文件
2. **批量处理**：一次性获取所有文件路径，然后批量处理
3. **内存优化**：逐个处理文件，避免同时加载所有文件到内存
4. **错误隔离**：单个文件失败不影响其他文件的处理

## 兼容性

- ✅ **向后兼容**：原有的单目录用法完全不受影响
- ✅ **参数兼容**：新参数都有默认值，可选使用
- ✅ **输出兼容**：生成的HDF5和标注文件格式不变

## 扩展性

添加新的音频格式支持：

```python
# 添加更多格式
supported_formats = ['.wav', '.flac', '.mp3', '.m4a', '.ogg', '.wma']

create_dataset(
    data_dirs=your_dirs,
    dataloader_process_only=True,
    supported_formats=supported_formats
)
```

## 注意事项

1. **文件名唯一性**：使用路径信息生成唯一的stem，避免重名冲突
2. **目录存在性**：自动检查目录是否存在，跳过无效路径
3. **HDF5组名限制**：确保生成的组名符合HDF5规范
4. **内存管理**：大量文件时注意内存使用，必要时分批处理

通过这些改进，`make_custom_dataset.py` 现在可以更灵活、更高效地处理各种音频数据集场景。
