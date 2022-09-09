# -*- encoding: utf-8 -*-
'''
@File    :   Dataset.py
@Time    :   2022/08/29 19:04:04
@Author  :   lujun
@Version :   1.0
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   TDeer 模型的Dataset
'''

import os
import torch
import json
import copy
import numpy as np
from utils.utils import find_head_idx
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


class TDEERDataset(Dataset):
    def __init__(self, args, is_training=False):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(args.pretrain_path)
        self.is_training = is_training
        self.neg_samples = args.neg_samples
        self.batch_size = args.batch_size
        with open(os.path.join(args.data_dir, "rel2id.json"), 'r') as f:
            relation = json.load(f)
        self.rel2id = relation[1]
        self.rels_set = list(self.rel2id.values())
        self.relation_size = len(self.rel2id)
        self.max_sample_triples = args.max_sample_triples
        self.datas = []
        if is_training:
            filenames = os.path.join(args.data_dir, "train_triples.json")
        else:
            filenames = os.path.join(args.data_dir, "dev_triples.json")
        with open(filenames, 'r') as f:
            lines = json.load(f)
        if self.is_training:
            self.preprocess(lines)
        else:
            self.proprecss_val(lines)

    def proprecss_val(self, datas):
        for data in datas:
            neg_datas = []
            text = data['text']
            text_tokened = self.tokenizer(text, return_offsets_mapping=True)
            input_ids = text_tokened['input_ids']
            attention_masks = text_tokened['attention_mask']
            # 将[cls]和[sep]mask掉
            text_masks = self.get_text_mask(attention_masks)
            token_type_ids = text_tokened['token_type_ids']
            offset_mapping = text_tokened['offset_mapping']
            triples_index_set = set()   # (sub head, sub tail, obj head, obj tail, rel)
            triples_set = set()
            for triple in data['triple_list']:
                subj, rel, obj = triple
                triples_set.add((subj, rel, obj))
                subj, rel, obj = triple
                rel_idx = self.rel2id[rel]
                subj_tokened = self.tokenizer.encode(subj)
                obj_tokened = self.tokenizer.encode(obj)
                subj_head_idx = find_head_idx(input_ids, subj_tokened[1:-1], 0)
                subj_tail_idx = subj_head_idx + len(subj_tokened[1:-1]) - 1
                obj_head_idx = find_head_idx(
                    input_ids, obj_tokened[1:-1], subj_tail_idx+1)
                if obj_head_idx == -1:
                    obj_head_idx = find_head_idx(
                        input_ids, obj_tokened[1:-1], 0)
                obj_tail_idx = obj_head_idx + len(obj_tokened[1:-1]) - 1
                if subj_head_idx == -1 or obj_head_idx == -1:
                    continue
                # 0表示subject，1表示object
                triples_index_set.add(
                    (subj_head_idx, subj_tail_idx,
                     obj_head_idx, obj_tail_idx, rel_idx)
                )

            # postive samples
            self.datas.append({
                "text": text,
                'token_ids': input_ids,
                "attention_masks": attention_masks,
                "offset_mapping": offset_mapping,
                "text_masks": text_masks,
                'segment_ids': token_type_ids,
                "length": len(input_ids),
                "triples_set": triples_set,
                "triples_index_set": triples_index_set
            })

    def get_text_mask(self, attention_masks):
        new_atten_mask = copy.deepcopy(attention_masks)
        new_atten_mask[0] = 0
        new_atten_mask[-1] = 0
        return new_atten_mask

    def preprocess(self, datas):
        for data in datas:
            pos_datas = []
            neg_datas = []
            text = data['text']
            text_tokened = self.tokenizer(text, max_length=512, truncation=True, return_offsets_mapping=True)
            input_ids = text_tokened['input_ids']
            attention_masks = text_tokened['attention_mask']
            # 将[cls]和[sep]mask掉
            text_masks = self.get_text_mask(attention_masks)
            token_type_ids = text_tokened['token_type_ids']
            offset_mapping = text_tokened['offset_mapping']
            text_length = len(input_ids)
            entity_set = set()  # (head idx, tail idx)
            triples_set = set()   # (sub head, sub tail, obj head, obj tail, rel)
            subj_rel_set = set()   # (sub head, sub tail, rel)
            subj_set = set()   # (sub head, sub tail)
            rel_set = set()
            trans_map = defaultdict(list)   # {(sub_head, rel): [tail_heads]}
            triples_sets = set()
            for triple in data['triple_list']:
                subj, rel, obj = triple
                triples_sets.add((subj, rel, obj))
                rel_idx = self.rel2id[rel]
                subj_tokened = self.tokenizer.encode(subj)
                obj_tokened = self.tokenizer.encode(obj)
                subj_head_idx = find_head_idx(input_ids, subj_tokened[1:-1], 0)
                subj_tail_idx = subj_head_idx + len(subj_tokened[1:-1]) - 1
                obj_head_idx = find_head_idx(
                    input_ids, obj_tokened[1:-1], subj_tail_idx+1)
                if obj_head_idx == -1:
                    obj_head_idx = find_head_idx(
                        input_ids, obj_tokened[1:-1], 0)
                obj_tail_idx = obj_head_idx + len(obj_tokened[1:-1]) - 1
                if subj_head_idx == -1 or obj_head_idx == -1:
                    continue
                # 0表示subject，1表示object
                entity_set.add((subj_head_idx, subj_tail_idx, 0))
                entity_set.add((obj_head_idx, obj_tail_idx, 1))
                subj_rel_set.add((subj_head_idx, subj_tail_idx, rel_idx))
                subj_set.add((subj_head_idx, subj_tail_idx))
                triples_set.add(
                    (subj_head_idx, subj_tail_idx,
                     obj_head_idx, obj_tail_idx, rel_idx)
                )
                rel_set.add(rel_idx)
                trans_map[(subj_head_idx, subj_tail_idx, rel_idx)
                          ].append(obj_head_idx)

            if not rel_set:
                continue

            # 当前句子中的所有subjects和objects
            entity_heads = np.zeros((text_length, 2))
            entity_tails = np.zeros((text_length, 2))
            for (head, tail, _type) in entity_set:
                entity_heads[head][_type] = 1
                entity_tails[tail][_type] = 1

            # 当前句子的所有关系
            rels = np.zeros(self.relation_size)
            for idx in rel_set:
                rels[idx] = 1

            if self.max_sample_triples is not None:
                triples_list = list(triples_set)
                np.random.shuffle(triples_list)
                triples_list = triples_list[:self.max_sample_triples]
            else:
                triples_list = list(triples_set)

            neg_history = set()
            for subj_head_idx, subj_tail_idx, obj_head_idx, obj_tail_idx, rel_idx in triples_list:
                current_neg_datas = []
                # 一个subject作为输入样例
                sample_obj_heads = np.zeros(text_length)
                for idx in trans_map[(subj_head_idx, subj_tail_idx, rel_idx)]:
                    sample_obj_heads[idx] = 1.0
                # postive samples
                pos_datas.append({
                    "text": text,
                    'token_ids': input_ids,
                    "attention_masks": attention_masks,
                    "text_masks": text_masks,
                    "offset_mapping": offset_mapping,
                    'segment_ids': token_type_ids,
                    'entity_heads': entity_heads,  # 所有实体的头部
                    'entity_tails': entity_tails,  # 所有实体的尾部
                    'rels': rels,  # 所有关系
                    'sample_subj_head': subj_head_idx,  # 单个subject的头部
                    'sample_subj_tail': subj_tail_idx,  # 单个subject的尾部
                    'sample_rel': rel_idx,  # 单个subject对应的关系
                    'sample_obj_heads': sample_obj_heads,  # 单个subject和rel对应的object
                    "length": len(input_ids),
                    "triples_sets": triples_sets
                })

                # 只针对训练数据集进行负采样
                if not self.is_training:
                    continue
                # 将subject和object对调创建负例，对应的object全为0
                # 1. inverse (tail as subj)
                neg_subj_head_idx = obj_head_idx
                neg_sub_tail_idx = obj_tail_idx
                neg_pair = (neg_subj_head_idx, neg_sub_tail_idx, rel_idx)
                if neg_pair not in subj_rel_set and neg_pair not in neg_history:
                    current_neg_datas.append({
                        "text": text,
                        'token_ids': input_ids,
                        "attention_masks": attention_masks,
                        "offset_mapping": offset_mapping,
                        "text_masks": text_masks,
                        'segment_ids': token_type_ids,
                        'entity_heads': entity_heads,
                        'entity_tails': entity_tails,
                        'rels': rels,
                        'sample_subj_head': neg_subj_head_idx,
                        'sample_subj_tail': neg_sub_tail_idx,
                        'sample_rel': rel_idx,
                        # set 0 for negative samples
                        'sample_obj_heads': np.zeros(text_length),
                        "length": len(input_ids),
                        "triples_sets": triples_sets
                    })
                    neg_history.add(neg_pair)

                # 随机选择其他关系作为负例，对应的object全为0
                # 2. (pos sub, neg_rel)
                for neg_rel_idx in rel_set - {rel_idx}:
                    neg_pair = (subj_head_idx, subj_tail_idx, neg_rel_idx)
                    if neg_pair not in subj_rel_set and neg_pair not in neg_history:
                        current_neg_datas.append({
                            "text": text,
                            'token_ids': input_ids,
                            "attention_masks": attention_masks,
                            "text_masks": text_masks,
                            "offset_mapping": offset_mapping,
                            'segment_ids': token_type_ids,
                            'entity_heads': entity_heads,
                            'entity_tails': entity_tails,
                            'rels': rels,
                            'sample_subj_head': subj_head_idx,
                            'sample_subj_tail': subj_tail_idx,
                            'sample_rel': neg_rel_idx,
                            # set 0 for negative samples
                            'sample_obj_heads': np.zeros(text_length),
                            "length": len(input_ids),
                            "triples_sets": triples_sets
                        })
                        neg_history.add(neg_pair)

                # 随机选择一个subject作为负例，对应的object全为0
                # 3. (neg sub, pos rel)
                for (neg_subj_head_idx, neg_sub_tail_idx) in subj_set - {(subj_head_idx, subj_tail_idx)}:
                    neg_pair = (neg_subj_head_idx, neg_sub_tail_idx, rel_idx)
                    if neg_pair not in subj_rel_set and neg_pair not in neg_history:
                        current_neg_datas.append({
                            "text": text,
                            'token_ids': input_ids,
                            'segment_ids': token_type_ids,
                            "attention_masks": attention_masks,
                            "text_masks": text_masks,
                            "offset_mapping": offset_mapping,
                            'entity_heads': entity_heads,
                            'entity_tails': entity_tails,
                            'rels': rels,
                            'sample_subj_head': neg_subj_head_idx,
                            'sample_subj_tail': neg_sub_tail_idx,
                            'sample_rel': rel_idx,
                            # set 0 for negative samples
                            'sample_obj_heads': np.zeros(text_length),
                            "length": len(input_ids),
                            "triples_sets": triples_sets
                        })
                        neg_history.add(neg_pair)

                np.random.shuffle(current_neg_datas)
                if self.neg_samples is not None:
                    current_neg_datas = current_neg_datas[:self.neg_samples]
                neg_datas += current_neg_datas
            current_datas = pos_datas + neg_datas
            self.datas.extend(current_datas)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]


def collate_fn(batches):
    """_summary_
    Args:
        batches (_type_): _description_
    """
    max_len = max([batch['length'] for batch in batches])
    batch_size = len(batches)

    batch_tokens = torch.zeros((batch_size, max_len), dtype=torch.int32)
    batch_attention_masks = torch.zeros(
        (batch_size, max_len), dtype=torch.float32)
    batch_segments = torch.zeros((batch_size, max_len), dtype=torch.int32)
    batch_entity_heads = torch.zeros(
        (batch_size, max_len, 2), dtype=torch.float32)
    batch_entity_tails = torch.zeros(
        (batch_size, max_len, 2), dtype=torch.float32)
    batch_rels = []
    batch_sample_subj_head, batch_sample_subj_tail = [], []
    batch_sample_rel = []
    batch_sample_obj_heads = torch.zeros(
        (batch_size, max_len), dtype=torch.float32)
    batch_texts = []
    batch_text_masks = torch.zeros((batch_size, max_len), dtype=torch.float32)
    batch_offsets = []
    batch_triples_sets = []
    for i, obj in enumerate(batches):
        length = obj['length']
        batch_texts.append(obj['text'])
        batch_offsets.append(obj['offset_mapping'])
        batch_tokens[i, :length] = torch.tensor(obj['token_ids'])
        batch_attention_masks[i, :length] = torch.tensor(
            obj['attention_masks'])
        batch_text_masks[i, :length] = torch.tensor(obj['text_masks'])
        batch_segments[i, :length] = torch.tensor(obj['segment_ids'])
        batch_entity_heads[i, :length, :] = torch.tensor(obj['entity_heads'])
        batch_entity_tails[i, :length, :] = torch.tensor(obj['entity_tails'])
        batch_rels.append(obj['rels'])
        batch_sample_subj_head.append([obj['sample_subj_head']])
        batch_sample_subj_tail.append([obj['sample_subj_tail']])

        batch_sample_rel.append([obj['sample_rel']])
        batch_sample_obj_heads[i, :length] = torch.tensor(
            obj['sample_obj_heads'])
        batch_triples_sets.append(obj['triples_sets'])
    batch_rels = torch.from_numpy(np.array(batch_rels, dtype=np.int64))
    batch_sample_subj_head = torch.from_numpy(
        np.array(batch_sample_subj_head, dtype=np.int64))
    batch_sample_subj_tail = torch.from_numpy(
        np.array(batch_sample_subj_tail, dtype=np.int64))
    batch_sample_rel = torch.from_numpy(
        np.array(batch_sample_rel, dtype=np.int64))

    return [batch_texts, batch_offsets, batch_tokens, batch_attention_masks, batch_segments, batch_entity_heads, batch_entity_tails, batch_rels, batch_sample_subj_head, batch_sample_subj_tail, batch_sample_rel, batch_sample_obj_heads, batch_triples_sets, batch_text_masks]


def collate_fn_val(batches):
    """_summary_
    Args:
        batches (_type_): _description_
    """
    max_len = max([batch['length'] for batch in batches])
    batch_size = len(batches)

    batch_tokens = torch.zeros((batch_size, max_len), dtype=torch.int32)
    batch_attention_masks = torch.zeros(
        (batch_size, max_len), dtype=torch.float32)
    batch_segments = torch.zeros((batch_size, max_len), dtype=torch.int32)
    batch_text_masks = torch.zeros((batch_size, max_len), dtype=torch.float32)

    batch_triple_set = []
    batch_triples_index_set = []
    batch_texts = []
    batch_offsets = []
    for i, obj in enumerate(batches):
        length = obj['length']
        batch_texts.append(obj['text'])
        batch_offsets.append(obj['offset_mapping'])
        batch_tokens[i, :length] = torch.tensor(obj['token_ids'])
        batch_attention_masks[i, :length] = torch.tensor(
            obj['attention_masks'])
        batch_text_masks[i, :length] = torch.tensor(obj['text_masks'])
        batch_segments[i, :length] = torch.tensor(obj['segment_ids'])
        batch_triple_set.append(obj['triples_set'])
        batch_triples_index_set.append(obj['triples_index_set'])

    return [batch_texts, batch_offsets, batch_tokens, batch_attention_masks, batch_segments, batch_triple_set, batch_triples_index_set, batch_text_masks]
