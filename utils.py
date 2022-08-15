#! -*- coding:utf-8 -*-

import json

def statistics_text_length(filename,tokenizer):
    with open(filename,'r') as f:
        lines = json.load(f)
    max_length = 0
    for line in lines:
        max_length = max(max_length,len(tokenizer.tokenize(line['text'])))
    
    return max_length

    