# -*- encoding: utf-8 -*-
'''
@File    :   Dataset.py
@Time    :   2022/08/29 19:09:38
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
from utils.utils import find_head_idx
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from PRGC.utils import Label2IdxSub,Label2IdxObj
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

class InputExample(object):
    """a single set of samples of data
    """

    def __init__(self, text, en_pair_list, re_list, rel2ens):
        self.text = text
        self.en_pair_list = en_pair_list
        self.re_list = re_list
        self.rel2ens = rel2ens


class PRGCDataset(Dataset):
    def __init__(self,args,filename,is_training):
        super().__init__()
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained(args.pretrain_path,cache_dir = "./bertbaseuncased")
        self.is_training = is_training
        self.batch_size = args.batch_size
        with open(args.relation,'r') as f:
            relation = json.load(f)
        self.rel2id = relation[1]
        self.rels_set = list(self.rel2id.values())
        self.relation_size = len(self.rel2id)
        self.max_sample_triples = args.max_sample_triples
        with open(filename,'r') as f:
            lines = json.load(f)
        self.datas = self.preprocess(lines)
    
    def preprocess(self,lines):
        examples = []
        for sample in lines:
            text = sample['text']
            rel2ens = defaultdict(list)
            en_pair_list = []
            re_list = []

            for triple in sample['triple_list']:
                en_pair_list.append([triple[0], triple[-1]])
                re_list.append(self.rel2id[triple[1]])
                rel2ens[self.rel2id[triple[1]]].append((triple[0], triple[-1]))
            example = InputExample(text=text, en_pair_list=en_pair_list, re_list=re_list, rel2ens=rel2ens)
            examples.append(example)
        max_text_len = self.args.max_seq_len
        # multi-process
        # with Pool(10) as p:
        #     convert_func = functools.partial(self.convert, max_text_len=max_text_len, tokenizer=self.tokenizer, rel2idx=self.rel2id,
        #                                     ensure_rel=self.args.ensure_rel,num_negs=self.args.num_negs)
        #     features = p.map(func=convert_func, iterable=examples)
        # # return list(chain(*features))
        features = []
        for example in tqdm(examples,desc="convert example"):
            feature = self.convert(example,max_text_len=max_text_len, tokenizer=self.tokenizer, rel2idx=self.rel2id,
                                            ensure_rel=self.args.ensure_rel,num_negs=self.args.num_negs)
            features.extend(feature)
        return features
    

    def convert(self,example: InputExample, max_text_len: int, tokenizer, rel2idx, ensure_rel,num_negs):
        """转换函数 for CarFaultRelation data
        Args:
            example (_type_): 一个样本示例
            max_text_len (_type_): 样本的最大长度
            tokenizer (_type_): _description_
            rel2idx (dict): 关系的索引
            ex_params (_type_): 额外的参数
        Returns:
            _type_: _description_
        """
        text_tokens = tokenizer.tokenize(example.text)
        # cut off
        if len(text_tokens) > max_text_len:
            text_tokens = text_tokens[:max_text_len]

        # token to id
        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        attention_mask = [1] * len(input_ids)
        # zero-padding up to the sequence length
        if len(input_ids) < max_text_len:
            pad_len = max_text_len - len(input_ids)
            # token_pad_id=0
            input_ids += [0] * pad_len
            attention_mask += [0] * pad_len

        # train data
        if self.is_training:
            # construct tags of correspondence and relation
            # subject和object相关性 target
            corres_tag = np.zeros((max_text_len, max_text_len))
            rel_tag = len(rel2idx) * [0]
            for en_pair, rel in zip(example.en_pair_list, example.re_list):
                # get sub and obj head
                sub_head, obj_head, _, _ = self._get_so_head(
                    en_pair, tokenizer, text_tokens)
                # construct relation tag
                rel_tag[rel] = 1
                # 只将head 的index标记为1
                if sub_head != -1 and obj_head != -1:
                    corres_tag[sub_head][obj_head] = 1

            sub_feats = []
            # positive samples，标记subject和object的序列
            for rel, en_ll in example.rel2ens.items():
                # init
                tags_sub = max_text_len * [Label2IdxSub['O']]
                tags_obj = max_text_len * [Label2IdxSub['O']]
                for en in en_ll:
                    # get sub and obj head
                    sub_head, obj_head, sub, obj = self._get_so_head(
                        en, tokenizer, text_tokens)
                    if sub_head != -1 and obj_head != -1:
                        if sub_head + len(sub) <= max_text_len:
                            tags_sub[sub_head] = Label2IdxSub['B-H']
                            tags_sub[sub_head + 1:sub_head +
                                    len(sub)] = (len(sub) - 1) * [Label2IdxSub['I-H']]
                        if obj_head + len(obj) <= max_text_len:
                            tags_obj[obj_head] = Label2IdxObj['B-T']
                            tags_obj[obj_head + 1:obj_head +
                                    len(obj)] = (len(obj) - 1) * [Label2IdxObj['I-T']]
                # 相同关系下的所有subject和object对
                seq_tag = [tags_sub, tags_obj]

                # sanity check
                assert len(input_ids) == len(tags_sub) == len(tags_obj) == len(
                    attention_mask) == max_text_len, f'length is not equal!!'
                sub_feats.append(InputFeatures(
                    input_tokens=text_tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    corres_tag=corres_tag,
                    seq_tag=seq_tag,
                    relation=rel,
                    rel_tag=rel_tag
                ))
            # relation judgement ablation
            if not ensure_rel:
                # negative samples, 采样一些负样本的关系数据集
                neg_rels = set(rel2idx.values()).difference(set(example.re_list))
                neg_rels = random.sample(neg_rels, k=num_negs)
                for neg_rel in neg_rels:
                    # init，针对关系的负样本，只对subject和object的序列全部置为O，其他的沿用正样本的数据
                    seq_tag = max_text_len * [Label2IdxSub['O']]
                    # sanity check
                    assert len(input_ids) == len(seq_tag) == len(
                        attention_mask) == max_text_len, f'length is not equal!!'
                    seq_tag = [seq_tag, seq_tag]
                    sub_feats.append(InputFeatures(
                        input_tokens=text_tokens,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        corres_tag=corres_tag,
                        seq_tag=seq_tag,
                        relation=neg_rel,
                        rel_tag=rel_tag
                    ))
        # val and test data
        else:
            triples = []
            for rel, en in zip(example.re_list, example.en_pair_list):
                # get sub and obj head
                sub_head, obj_head, sub, obj = self._get_so_head(
                    en, tokenizer, text_tokens)
                if sub_head != -1 and obj_head != -1:
                    h_chunk = ('H', sub_head, sub_head + len(sub))
                    t_chunk = ('T', obj_head, obj_head + len(obj))
                    triples.append((h_chunk, t_chunk, rel))
            sub_feats = [
                InputFeatures(
                    input_tokens=text_tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    triples=triples
                )
            ]

        # get sub-feats
        return sub_feats

    def _get_so_head(self,en_pair, tokenizer, text_tokens):
        sub = tokenizer.tokenize(en_pair[0])
        obj = tokenizer.tokenize(en_pair[1])
        subj_head_idx = find_head_idx(text_tokens, sub,0)
        subj_tail_idx = subj_head_idx + len(sub) - 1
        obj_head_idx = find_head_idx(text_tokens, obj,subj_tail_idx+1)
        if obj_head_idx == -1:
            obj_head_idx = find_head_idx(text_tokens, obj,0)
        return subj_head_idx, obj_head_idx, sub, obj

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        return data


class InputFeatures(object):
    """
    Desc:
        a single set of features of data
    """
    def __init__(self,input_tokens,input_ids,attention_mask,seq_tag=None,corres_tag=None,relation=None,triples=None,rel_tag=None):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.seq_tag = seq_tag
        self.corres_tag = corres_tag
        self.relation = relation
        self.triples = triples
        self.rel_tag = rel_tag


def collate_fn_train(features):
    """将InputFeatures转换为Tensor
    Args:
        features (List[InputFeatures])
    Returns:
        tensors (List[Tensors])
    """
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    seq_tags = torch.tensor([f.seq_tag for f in features], dtype=torch.long)
    poten_relations = torch.tensor([f.relation for f in features], dtype=torch.long)
    corres_tags = torch.tensor([f.corres_tag for f in features], dtype=torch.long)
    rel_tags = torch.tensor([f.rel_tag for f in features], dtype=torch.long)
    tensors = [input_ids, attention_mask, seq_tags, poten_relations, corres_tags, rel_tags]
    return tensors


def collate_fn_test(features):
    """将InputFeatures转换为Tensor
    Args:
        features (List[InputFeatures])
    Returns:
        tensors (List[Tensors])
    """
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    triples = [f.triples for f in features]
    input_tokens = [f.input_tokens for f in features]
    tensors = [input_ids, attention_mask, triples, input_tokens]
    return tensors
    
 
 
