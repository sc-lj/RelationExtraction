#! -*- coding:utf-8 -*-

import json
import time
from typing import Dict, List, Set

import numpy as np
from tqdm import tqdm
# from langml.tensor_typing import Models


def rematch(offsets: List) -> List:
    mapping = []
    for offset in offsets:
        if offset[0] == 0 and offset[1] == 0:
            mapping.append([])
        else:
            mapping.append([i for i in range(offset[0], offset[1])])
    return mapping


def partial_match(pred_set, gold_set):
    pred = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in pred_set}
    gold = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
    return pred, gold


def remove_space(data_set):
    data_set = {(i[0].replace(' ', ''), i[1], i[2].replace(' ', '')) for i in data_set}
    return data_set



def statistics_text_length(filename,tokenizer):
    with open(filename,'r') as f:
        lines = json.load(f)
    max_length = 0
    for line in lines:
        max_length = max(max_length,len(tokenizer.tokenize(line['text'])))
    
    return max_length

    