# Copyright      2023                           (authors: Feiteng Li)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
modified from lhoste.dataset.speech_synthesis.py
"""
import os
import torch
import math
#from tokenizers import Tokenizer
from typing import Union, List
import numpy as np
from tqdm import tqdm
#from utils.g2p import PhonemeBpeTokenizer
from data.collation import get_text_token_collater
from fairseq.data import Dictionary
from fairseq.data import (
    data_utils,
    indexed_dataset,
)
from fairseq import utils
text_collater = get_text_token_collater()

_pad        = '_'
_punctuation = ',.!?-~…'
_letters = 'NQabdefghijklmnopstuvwxyzɑæʃʑçɯɪɔɛɹðəɫɥɸʊɾʒθβŋɦ⁼ʰ`^#*=ˈˌ→↓↑ '
symbols = [_pad] + list(_punctuation) + list(_letters)


def seq2phone(tokens: Union[List, np.ndarray]):
    """
    Convert tokenized phoneme ID sequence back to phoneme string
    :param tokens: phoneme tokens
    :return: recovered phoneme sequence
    """
    phones = "".join([symbols[i] for i in tokens])
    return phones

class DynamicBatchSampler(torch.utils.data.Sampler):
    def __init__(self, set_lang, sampler, num_tokens_fn, num_buckets=100, min_size=0, max_size=1000,
                 max_tokens=None, max_sentences=None, drop_last=False):
        """

        :param sampler:
        :param num_tokens_fn: 根据idx返回样本的长度的函数
        :param num_buckets: 利用桶原理将相似长度的样本放在一个batchsize中，桶的数量
        :param min_size: 最小长度的样本， 小于这个值的样本会被过滤掉。 依据这个值来创建样桶
        :param max_size: 最大长度的样本
        :param max_sentences: batch_size, 但是这里可以通过max_sentences 和 max_tokens 共同控制最终的大小
        """
        super(DynamicBatchSampler, self).__init__(sampler)
        self.sampler = sampler
        self.num_tokens_fn = num_tokens_fn
        self.set_lang = set_lang
        self.num_buckets = num_buckets

        self.min_size = min_size
        self.max_size = max_size

        assert max_size <= max_tokens, "max_size should be smaller than max tokens"
        assert max_tokens is not None or max_sentences is not None, \
            "max_tokens and max_sentences should not be null at the same time, please specify one parameter at least"
        self.max_tokens = max_tokens if max_tokens is not None else float('Inf')
        self.max_sentences = max_sentences if max_sentences is not None else float('Inf')
        self.drop_last = drop_last

    def set_epoch(self, epoch):
        self.set_lang()
        self.sampler.set_epoch(epoch)
    def is_batch_full(self, num_tokens, batch):
        if len(batch) == 0:
            return False
        if len(batch) == self.max_sentences:
            return True
        if num_tokens > self.max_tokens:
            return True
        return False

    def __iter__(self):
        
        buckets = [[] for _ in range(self.num_buckets)]
        sample_len = [0] * self.num_buckets

        for idx in self.sampler:
            idx_length = self.num_tokens_fn(idx)
            if not (self.min_size <= idx_length <= self.max_size):
                print("sentence at index {} of size {} exceeds max_tokens, the sentence is ignored".format(idx, idx_length))
                continue

            index_buckets = math.floor((idx_length - self.min_size) / (self.max_size - self.min_size + 1)
                                       * self.num_buckets)
            sample_len[index_buckets] = max(sample_len[index_buckets], idx_length)

            num_tokens = (len(buckets[index_buckets]) + 1) * sample_len[index_buckets]
            if self.is_batch_full(num_tokens, buckets[index_buckets]):
                # yield this batch
                yield buckets[index_buckets]
                buckets[index_buckets] = []
                sample_len[index_buckets] = 0

            buckets[index_buckets].append(idx)

        # process left-over
        leftover_batch = []
        leftover_sample_len = 0
        leftover = [idx for bucket in buckets for idx in bucket]
        for idx in leftover:
            idx_length = self.num_tokens_fn(idx)
            leftover_sample_len = max(leftover_sample_len, idx_length)
            num_tokens = (len(leftover_batch) + 1) * leftover_sample_len
            if self.is_batch_full(num_tokens, leftover_batch):
                yield leftover_batch
                leftover_batch = []
                leftover_sample_len = 0
            leftover_batch.append(idx)

        if len(leftover_batch) > 0 and not self.drop_last:
            yield leftover_batch

    def __len__(self):
        # we do not know the exactly batch size, so do not call len(dataloader)
        pass


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, lang=['zh','en'],num_quantizers=8):
        super().__init__()
        self.data_dir = data_dir
        self.num_quantizers = num_quantizers
        self.at_dict = Dictionary.load(os.path.join(self.data_dir, "checkpoint/slam_vallex/pretrained_model", "dict.at.txt"))
        self.st_dict = Dictionary.load(os.path.join(self.data_dir, "checkpoint/slam_vallex/pretrained_model","dict.st.txt"))
       
        self.at_dict.tts_flag = self.at_dict.add_symbol("<TTS>")
        self.st_dict.asr_flag = self.st_dict.add_symbol("<ASR>")
        self.at_dict.mt_flag = self.st_dict.add_symbol("<MT>")
        assert len(lang)>0, "lang must be in ['zh','en']"
        self.len=0
        self.lang_id_dict = {}
        if 'zh' in lang:
            self.zh_at_dataset, self.zh_st_dataset, self.zh_dur, self.zh_datanum = self.load_data("data_wenet")
            self.len=self.zh_datanum
            self.lang_id_dict["zh"] = 0
            self.lang = "zh"
        if 'en' in lang:
            self.en_at_dataset, self.en_st_dataset, self.en_dur, self.en_datanum = self.load_data("data_libri_12s")
            self.len = max(self.len,self.en_datanum)
            self.lang_id_dict["en"] = 1
            self.lang = "en"

    def load_data(self, prefix):
        at_dataset=[]
        for i in range(self.num_quantizers):
            temp_data_path = os.path.join(
            self.data_dir, prefix,"data_bin_at_{}".format(i),'train'
            )
            at_dataset.append(data_utils.load_indexed_dataset(temp_data_path, self.at_dict, None))

        temp_data_path = os.path.join(
            self.data_dir, prefix,"data_bin_st",'train'
        )
        assert indexed_dataset.dataset_exists(temp_data_path, impl=None), temp_data_path
        st_dataset = data_utils.load_indexed_dataset(temp_data_path, self.st_dict, None)
        
        temp_data_path = os.path.join(
            self.data_dir, prefix,"dur",'train'
        )
        assert indexed_dataset.dataset_exists(temp_data_path, impl=None), temp_data_path
        dur_dataset = data_utils.load_indexed_dataset(temp_data_path, None, None)

        assert len(at_dataset) == len(st_dataset) == len(dur_dataset), "at_dataset and st_dataset have different lengths"
        return at_dataset, st_dataset, dur_dataset,len(st_dataset)

    def __len__(self):
        return self.len

    def get_dur(self, idx):
        if self.lang == "zh":
            return float(self.zh_dur[idx].item())
        if self.lang == "en":
            return float(self.en_dur[idx].item())

    def set_lang(self):
        """随机选择一种语言并赋值给 self.lang。"""
        if not self.lang_id_dict:
            return
        keys = list(self.lang_id_dict.keys())
        self.lang = str(np.random.choice(keys))

    def get_at(self,idx,data):
        """
        从 8 个量化子数据集中取出相同 idx 的序列，直接在列维拼接成 [L, 8] 的张量；
        :param idx: 样本索引
        :param data: 长度为 8 的数据集列表，每个元素支持 __getitem__(idx)
        :return: torch.LongTensor，形状 [L, 8]
        """
        sequences = []
        for ds in data:
            seq = ds[idx]
            if isinstance(seq, np.ndarray):
                seq = torch.from_numpy(seq)
            elif not torch.is_tensor(seq):
                seq = torch.tensor(seq)
            seq = seq.long().view(-1)
            sequences.append(seq)

        assert len(sequences)>0 , "sequences is empty"
        return torch.stack(sequences, dim=1)

    def __getitem__(self, index):
        if self.lang == "zh":
            idx = index%self.zh_datanum
            at = self.get_at(idx,self.zh_at_dataset)
            st = self.zh_st_dataset[idx]

        if self.lang == "en":  
            idx = index%self.en_datanum
            at = self.get_at(idx,self.en_at_dataset)
            st = self.en_st_dataset[idx]

        cptpho_tokens, enroll_x_lens = text_collater([st])
        cptpho_tokens = cptpho_tokens.squeeze(0)
        text_token_lens = enroll_x_lens[0]
        return {
                "audio_features": at,
                "audio_features_lens": at.shape[0],
                "text_tokens": np.array(cptpho_tokens),
                "text_tokens_lens": text_token_lens,
                "language": self.lang_id_dict[self.lang]
        }

def collate(batch):

    audio_features_lens_s = [b['audio_features_lens'] for b in batch]
    # create an empty tensor with maximum audio feature length
    audio_features_s = torch.zeros([len(batch), max(audio_features_lens_s), 8], dtype=torch.int64) - 1 # audio pad with -1

    text_tokens_lens_s = [b['text_tokens_lens'] for b in batch]
    # create an empty tensor with maximum text tokens length
    text_tokens_s = torch.zeros([len(batch), max(text_tokens_lens_s)], dtype=torch.int64) + 3 # [PAD] token id 3

    language_s = [b['language'] for b in batch]

    for i, b in enumerate(batch):
        audio_features = b['audio_features']
        audio_features_lens = b['audio_features_lens']
        audio_features_s[i, :audio_features_lens, :] = torch.LongTensor(audio_features)

        text_tokens = b['text_tokens']
        text_tokens_lens = b['text_tokens_lens']
        text_tokens_s[i, :text_tokens_lens] = torch.LongTensor(text_tokens)

    batch = {
        'audio_features': audio_features_s,
        'audio_features_lens': torch.LongTensor(np.array(audio_features_lens_s)),
        'text_tokens': text_tokens_s,
        'text_tokens_lens': torch.LongTensor(np.array(text_tokens_lens_s)),
        'languages': torch.LongTensor(np.array(language_s)),
    }
    return batch

def create_dataloader(data_dir, lang, n_gpus=1, rank=0, num_workers=0, num_buckets=10, max_duration=120):
    train_dataset = AudioDataset(data_dir=data_dir,lang=lang)
    ran_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
    dynamic_sampler = DynamicBatchSampler(train_dataset.set_lang, ran_sampler, train_dataset.get_dur, num_buckets=num_buckets, max_size=20,
                                          max_tokens=max_duration)


    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=num_workers, collate_fn=collate,
                                               batch_sampler=dynamic_sampler)

    return train_loader
