#! -*- coding:utf-8 -*-

import json


def statistics_text_length(filename, tokenizer):
    with open(filename, 'r') as f:
        lines = json.load(f)
    max_length = 0
    for line in lines:
        max_length = max(max_length, len(tokenizer.tokenize(line['text'])))

    return max_length


def rematch(offsets):
    mapping = []
    for offset in offsets:
        if offset[0] == 0 and offset[1] == 0:
            mapping.append([])
        else:
            mapping.append([i for i in range(offset[0], offset[1])])
    return mapping


def find_head_idx(source, target, sub_index):
    target_len = len(target)
    for i in range(sub_index, len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def update_arguments(args, config):
    """将config中的参数更新到args中
    Args:
        args ([type]): [description]
        config ([type]): [description]
    """
    for key, value in config.items():
        # 对于args中设置的值为最终值,即使config里面有冲突的值,仍以args中的参数值为准
        if key in args:
            print(f"该参数{key}的原值为{value},新值为{args.__dict__[key]}")
            continue
        args.__setattr__(key, value)
    return args
