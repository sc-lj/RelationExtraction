#! -*- coding:utf-8 -*-

import json

def statistics_text_length(filename,tokenizer):
    with open(filename,'r') as f:
        lines = json.load(f)
    max_length = 0
    for line in lines:
        max_length = max(max_length,len(tokenizer.tokenize(line['text'])))
    
    return max_length

def rematch(offsets):
    mapping = []
    for offset in offsets:
        if offset[0] == 0 and offset[1] == 0:
            mapping.append([])
        else:
            mapping.append([i for i in range(offset[0], offset[1])])
    return mapping

