# -*- encoding: utf-8 -*-
'''
@File    :   Dataset.py
@Time    :   2022/08/29 19:05:54
@Author  :   lujun
@Version :   1.0
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   SPN4RE 模型的Dataset
'''
import os
import torch
import json
from tqdm import tqdm
from utils.utils import find_head_idx
from torch.utils.data import DataLoader, Dataset
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


class Spn4REDataset(Dataset):
    def __init__(self, args, is_training) -> None:
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(
            args.pretrain_path, cache_dir="./bertbaseuncased")
        self.is_training = is_training
        with open(os.path.join(args.data_dir, "rel2id.json"), 'r') as f:
            relation = json.load(f)
        self.rel2id = relation[1]
        self.rels_set = list(self.rel2id.values())
        self.relation_size = len(self.rel2id)
        if is_training:
            filename = os.path.join(args.data_dir, "train_triples.json")
        else:
            filename = os.path.join(args.data_dir, "dev_triples.json")

        with open(filename, 'r') as f:
            lines = json.load(f)
        self.datas = self.preprocess(lines)

    def preprocess(self, lines):
        samples = []
        for i in tqdm(range(len(lines)), desc="prepare data"):
            token_sent = [self.tokenizer.cls_token] + self.tokenizer.tokenize(
                self.remove_accents(lines[i]["text"])) + [self.tokenizer.sep_token]
            triples = lines[i]["triple_list"]
            target = {"relation": [], "head_start_index": [], "head_end_index": [
            ], "tail_start_index": [], "tail_end_index": []}
            for triple in triples:
                head_entity = self.remove_accents(triple[0])
                tail_entity = self.remove_accents(triple[2])
                head_token = self.tokenizer.tokenize(head_entity)
                tail_token = self.tokenizer.tokenize(tail_entity)
                relation_id = self.rel2id[triple[1]]

                head_start_index = find_head_idx(token_sent, head_token, 0)
                head_end_index = head_start_index + len(head_token) - 1
                assert head_end_index >= head_start_index
                tail_start_index = find_head_idx(
                    token_sent, tail_token, head_end_index+1)
                if tail_start_index == -1:
                    tail_start_index = find_head_idx(token_sent, tail_token, 0)
                tail_end_index = tail_start_index + len(tail_token) - 1
                assert tail_end_index >= tail_start_index
                target["relation"].append(relation_id)
                target["head_start_index"].append(head_start_index)
                target["head_end_index"].append(head_end_index)
                target["tail_start_index"].append(tail_start_index)
                target["tail_end_index"].append(tail_end_index)
            sent_id = self.tokenizer.convert_tokens_to_ids(token_sent)
            samples.append([i, sent_id, target])
        return samples

    def __len__(self,):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]

    def remove_accents(self, text: str) -> str:
        accents_translation_table = str.maketrans(
            "áéíóúýàèìòùỳâêîôûŷäëïöüÿñÁÉÍÓÚÝÀÈÌÒÙỲÂÊÎÔÛŶÄËÏÖÜŸ",
            "aeiouyaeiouyaeiouyaeiouynAEIOUYAEIOUYAEIOUYAEIOUY"
        )
        return text.translate(accents_translation_table)


def collate_fn(batch_list):
    batch_size = len(batch_list)
    sent_idx = [ele[0] for ele in batch_list]
    sent_ids = [ele[1] for ele in batch_list]
    targets = [ele[2] for ele in batch_list]
    sent_lens = list(map(len, sent_ids))
    max_sent_len = max(sent_lens)
    input_ids = torch.zeros(
        (batch_size, max_sent_len), requires_grad=False).long()
    attention_mask = torch.zeros(
        (batch_size, max_sent_len), requires_grad=False, dtype=torch.float32)
    for idx, (seq, seqlen) in enumerate(zip(sent_ids, sent_lens)):
        input_ids[idx, :seqlen] = torch.LongTensor(seq)
        attention_mask[idx, :seqlen] = torch.FloatTensor([1] * seqlen)

    targets = [{k: torch.tensor(v, dtype=torch.long)
                for k, v in t.items()} for t in targets]
    info = {"seq_len": sent_lens, "sent_idx": sent_idx}
    return input_ids, attention_mask, targets, info
