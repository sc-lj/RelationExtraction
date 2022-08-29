# -*- encoding: utf-8 -*-
'''
@File    :   Dataset.py
@Time    :   2022/08/29 19:13:39
@Author  :   lujun
@Version :   1.0
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   None
'''

import torch
import json
import random
import numpy as np
from tqdm import tqdm
from OneRel.utils import TAG2ID
from utils.utils import find_head_idx
from torch.utils.data import Dataset
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


class OneRelDataset(Dataset):
    def __init__(self, filename, args, is_training):
        self.tokenizer = BertTokenizerFast.from_pretrained(
            args.pretrain_path, cache_dir="./bertbaseuncased")
        with open(args.relation, 'r') as f:
            relation = json.load(f)
        self.rel2id = relation[1]
        self.rels_set = list(self.rel2id.values())
        self.relation_size = len(self.rel2id)
        self.args = args
        with open(filename, 'r') as f:
            lines = json.load(f)

        if is_training:
            self.datas = self.preprocess_train(lines)
        else:
            self.datas = self.preprocess_val(lines)

    def preprocess_val(self, lines):
        datas = []
        for line in tqdm(lines):
            root_text = line['text']
            tokens = self.tokenizer.tokenize(root_text)
            tokens = [self.tokenizer.cls_token]+ tokens + [self.tokenizer.sep_token]
            if len(tokens) > 512:
                tokens = tokens[: 512]
            text_len = len(tokens)

            token_output = self.tokenizer(root_text,return_offsets_mapping=True)
            token_ids = token_output['input_ids']
            masks = token_output['attention_mask']
            offset_mapping = token_output['offset_mapping']
            if len(token_ids) > text_len:
                token_ids = token_ids[:text_len]
                masks = masks[:text_len]
            token_ids = np.array(token_ids)
            masks = np.array(masks)
            loss_masks = masks
            triple_matrix = np.zeros((self.relation_size, text_len, text_len))
            datas.append([token_ids, masks, loss_masks, text_len,offset_mapping,
                         triple_matrix, self.lower(line['triple_list']), tokens, root_text.lower()])
        return datas

    def preprocess_train(self, lines):
        datas = []
        for line in tqdm(lines):
            root_text = line['text']
            tokens = self.tokenizer.tokenize(root_text)
            tokens = [self.tokenizer.cls_token]+ tokens + [self.tokenizer.sep_token]
            if len(tokens) > 512:
                tokens = tokens[: 512]
            text_len = len(tokens)
            s2ro_map = {}
            for triple in line['triple_list']:
                triple = (self.tokenizer.tokenize(triple[0]), triple[1], self.tokenizer.tokenize(triple[2]))
                sub_head_idx = find_head_idx(tokens, triple[0],0)
                obj_head_idx = find_head_idx(tokens, triple[2],sub_head_idx + len(triple[0]))
                if obj_head_idx == -1:
                    obj_head_idx = find_head_idx(tokens, triple[2],0)
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    if sub not in s2ro_map:
                        s2ro_map[sub] = []
                    s2ro_map[sub].append(
                        (obj_head_idx, obj_head_idx + len(triple[2]) - 1, self.rel2id[triple[1]]))

            if s2ro_map:
                token_output = self.tokenizer(root_text,return_offsets_mapping=True)
                token_ids = token_output['input_ids']
                masks = token_output['attention_mask']
                offset_mapping = token_output['offset_mapping']
                if len(token_ids) > text_len:
                    token_ids = token_ids[:text_len]
                    masks = masks[:text_len]
                mask_length = len(masks)
                token_ids = np.array(token_ids)
                masks = np.array(masks)
                loss_masks = np.ones((mask_length, mask_length))
                triple_matrix = np.zeros(
                    (self.relation_size, text_len, text_len))
                for s in s2ro_map:
                    sub_head = s[0]
                    sub_tail = s[1]
                    for ro in s2ro_map.get((sub_head, sub_tail), []):
                        obj_head, obj_tail, relation = ro
                        # 赋值顺序不能变，先赋值3，在赋值2，最后赋值1，
                        # 当obj_tail和obj_head一致时，即object是单个字，该位置赋值为HB-TB，
                        # 当sub_tail和sub_head一致时，即subject是单个字，该位置赋值为HB-TE，
                        # 当object和subject都相同时，该位置赋值为HB-TB
                        triple_matrix[relation][sub_tail][obj_tail] = TAG2ID['HE-TE']
                        triple_matrix[relation][sub_head][obj_tail] = TAG2ID['HB-TE']
                        triple_matrix[relation][sub_head][obj_head] = TAG2ID['HB-TB'] 

                datas.append([token_ids, masks, loss_masks, text_len,offset_mapping,
                             triple_matrix, self.lower(line['triple_list']), tokens, root_text.lower()])
        return datas

    def lower(self,triples):
        lower_triples = []
        for line in triples:
            line = [l.lower() for l in line]
            lower_triples.append(line)
        return lower_triples

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]


def collate_fn(batch, rel_num):
    batch = list(filter(lambda x: x is not None, batch))
    batch.sort(key=lambda x: x[3], reverse=True)
    token_ids, masks, loss_masks, text_len,offset_mappings, triple_matrix, triples, tokens, texts = zip(
        *batch)
    cur_batch_len = len(batch)
    max_text_len = max(text_len)
    batch_token_ids = torch.LongTensor(cur_batch_len, max_text_len).zero_()
    batch_masks = torch.LongTensor(cur_batch_len, max_text_len).zero_()
    batch_loss_masks = torch.LongTensor(
        cur_batch_len, 1, max_text_len, max_text_len).zero_()
    # if use WebNLG_star, modify tag_size 24 to 171
    batch_triple_matrix = torch.LongTensor(
        cur_batch_len, rel_num, max_text_len, max_text_len).zero_()

    for i in range(cur_batch_len):
        batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
        batch_masks[i, :text_len[i]].copy_(torch.from_numpy(masks[i]))
        batch_loss_masks[i, 0, :text_len[i], :text_len[i]].copy_(
            torch.from_numpy(loss_masks[i]))
        batch_triple_matrix[i, :, :text_len[i], :text_len[i]].copy_(
            torch.from_numpy(triple_matrix[i]))

    return {'token_ids': batch_token_ids,
            'mask': batch_masks,
            'loss_mask': batch_loss_masks,
            'triple_matrix': batch_triple_matrix,
            'triples': triples,
            'tokens': tokens,
            "texts": texts,
            "offset_map":offset_mappings}


