import torch
from customs.make_custom_dataset import create_dataset
from models.macros import NUM_TEXT_TOKENS, NUM_AUDIO_TOKENS
from tqdm import tqdm
 
# 改成你的数据与语言
DATA_DIR = "/scratch/users/astar/ares/ma_yi/output/vallex"
LANG = ["zh"]  # 或 ["en"]，与训练一致
 
dl = create_dataset(DATA_DIR, LANG)
total_items = len(dl.dataset)  # 数据集样本总数

text_min, text_max = float("inf"), float("-inf")
audio_min, audio_max = float("inf"), float("-inf")
# 每个量化分片（列）最大/最小
per_q_max = None
per_q_min = None

num_batches = 0
num_items = 0

pbar = tqdm(total=total_items, unit="item", desc="Scanning dataset", smoothing=0.1)
with torch.no_grad():
    for batch in dl:
        num_batches += 1
        B = batch["text_tokens"].shape[0]
        num_items += B
 
        # 文本：仅统计有效长度
        tt = batch["text_tokens"]
        tt_lens = batch["text_tokens_lens"]
        for i in range(B):
            L = int(tt_lens[i].item())
            if L <= 0:
                continue
            v = tt[i, :L]
            text_min = min(text_min, int(v.min().item()))
            text_max = max(text_max, int(v.max().item()))
 
        # 音频：按有效帧长统计；-1 是 pad，需要切掉
        af = batch["audio_features"]
        af_lens = batch["audio_features_lens"]
        for i in range(B):
            L = int(af_lens[i].item())
            if L <= 0:
                continue
            v = af[i, :L, :]  # [L, 8]
            cur_min = int(v.min().item())
            cur_max = int(v.max().item())
            audio_min = min(audio_min, cur_min)
            audio_max = max(audio_max, cur_max)
            if per_q_max is None:
                Q = v.shape[-1]
                per_q_max = [-10**9]*Q
                per_q_min = [10**9]*Q
            for q in range(v.shape[-1]):
                per_q_min[q] = min(per_q_min[q], int(v[:, q].min().item()))
                per_q_max[q] = max(per_q_max[q], int(v[:, q].max().item()))

        # 更新进度条
        pbar.update(B)

pbar.close()
print(f"Scanned batches={num_batches}, items={num_items}")
print(f"text_tokens: min={text_min}, max={text_max}, limit(NUM_TEXT_TOKENS)={NUM_TEXT_TOKENS}")
print(f"audio_features: min={audio_min}, max={audio_max}, limit(NUM_AUDIO_TOKENS)={NUM_AUDIO_TOKENS}")
if per_q_max is not None:
    print("per-quantizer max:", per_q_max)
    print("per-quantizer min:", per_q_min)
 
# 越界检查提示
if text_max >= NUM_TEXT_TOKENS or text_min < 0:
    print("[WARN] text token 越界：请将 models/macros.py 的 NUM_TEXT_TOKENS 调整为 >= 实际词表大小，或修正数据。")
if audio_max >= NUM_AUDIO_TOKENS or audio_min < 0:
    print("[WARN] audio token 越界：请将 NUM_AUDIO_TOKENS 与你的 codec 码本一致，或修正数据。")