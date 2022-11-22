#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import defaultdict
import os
from typing import List, Dict
from UIE.utils import SPAN_START, TYPE_START, TYPE_END, null_span, TEXT_START, get_label_name_tree

debug = True

def match_sublist(the_list, to_match):
    """
    :param the_list: [1, 2, 3, 4, 5, 6, 1, 2, 4, 5]
    :param to_match: [1, 2]
    :return: [(0, 1), (6, 7)]
    """
    len_to_match = len(to_match)
    matched_list = list()
    for index in range(len(the_list) - len_to_match + 1):
        if to_match == the_list[index:index + len_to_match]:
            matched_list += [(index, index + len_to_match - 1)]
    return matched_list


def find_bracket_position(generated_text, _type_start, _type_end):
    """找到生成的文本中，标签开始type_start和标签结束type_end的index，以便判断后续的操作类型
    Args:
        generated_text (_type_): 生成的文本ids
        _type_start (_type_): type_start的id
        _type_end (_type_): type_end的id
    Returns:
        _type_: _description_
    """
    bracket_position = {_type_start: list(), _type_end: list()}
    for index, char in enumerate(generated_text):
        if char in bracket_position:
            bracket_position[char] += [index]
    return bracket_position


def build_sentence_tree(sentence):
    tree = defaultdict(set)
    for prev_token, next_token in zip(sentence[:-1], sentence[1:]):
        tree[prev_token].add(next_token)
    return tree


def generated_search_prefix_tree(generated, prefix_tree, tokenizer):
    tree = prefix_tree
    # Leaf is KEY_VALUE_SPLIT
    for token in generated:
        if token not in tree:
            return [tokenizer.eos_token]
        tree = tree[token]
    return list(tree)


def generated_search_src_sequence(generated, src_sequence, end_sequence_search_tokens=None):
    if len(generated) == 0:
        # All src tokens are valid before generation
        return src_sequence

    matched_tuples = match_sublist(the_list=src_sequence, to_match=generated)

    valid_token = list()
    for _, end in matched_tuples:
        next_index = end + 1
        if next_index < len(src_sequence):
            valid_token += [src_sequence[next_index]]

    if end_sequence_search_tokens:
        valid_token += end_sequence_search_tokens

    return valid_token


class SpotAsocConstraintDecoder():
    def __init__(self, tokenizer, type_schema,source_prefix, *args, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.source_prefix = source_prefix
        self.source_prefix_tokenized = tokenizer.encode(source_prefix,
                                                        add_special_tokens=False) if source_prefix else []
        self.tree_end = self.tokenizer.convert_tokens_to_ids([SPAN_START])[0]
        self.type_tree = get_label_name_tree(type_schema.type_list, self.tokenizer, end_symbol=self.tree_end)
        self.role_tree = get_label_name_tree(type_schema.role_list, self.tokenizer, end_symbol=self.tree_end)
        self.type_start = self.tokenizer.convert_tokens_to_ids([TYPE_START])[0]  # 标签的开始标志
        self.type_end = self.tokenizer.convert_tokens_to_ids([TYPE_END])[0]  # 标签的结束标志
        self.span_start = self.tokenizer.convert_tokens_to_ids([SPAN_START])[0]  # 文本span的开始标志
        self.null_span = self.tokenizer.convert_tokens_to_ids([null_span])[0]
        self.text_start = self.tokenizer.convert_tokens_to_ids([TEXT_START])[0]

    def constraint_decoding(self, src_sentence, tgt_generated):
        if self.source_prefix_tokenized:
            # Remove Source Prefix for Generation
            src_sentence = src_sentence[len(self.source_prefix_tokenized):]

        valid_token_ids = self.get_state_valid_tokens(src_sentence.tolist(), tgt_generated.tolist())
        return valid_token_ids

    def check_state(self, tgt_generated):
        if tgt_generated[-1] == self.tokenizer.pad_token_id:
            return 'start', -1

        # special_token_set = {EVENT_TYPE_LEFT, EVENT_TYPE_RIGHT}
        special_token_set = {self.type_start, self.type_end, self.span_start}
        special_index_token = list(filter(lambda x: x[1] in special_token_set, list(enumerate(tgt_generated))))
        # 生成的最后一个特殊字符
        last_special_index, last_special_token = special_index_token[-1]

        if len(special_index_token) == 1:
            if last_special_token != self.type_start:
                return 'error', 0

        bracket_position = find_bracket_position(tgt_generated, _type_start=self.type_start, _type_end=self.type_end)
        # type_start的数量，和type_end的数量
        start_number, end_number = len(bracket_position[self.type_start]), len(bracket_position[self.type_end])

        # 两者相等，结束生成任务
        if start_number == end_number:
            return 'end_generate', -1
        # type_start的数量比type_end的数量多一个，开始新的生成
        if start_number == end_number + 1:
            state = 'start_first_generation'
        # type_start的数量比type_end的数量多两个，开始触发相关词生成
        elif start_number == end_number + 2:
            state = 'generate_trigger'
            # 如果生成的最后一个特殊字符是span_start,那么开始生成触发词文本
            if last_special_token == self.span_start:
                state = 'generate_trigger_text'
        # type_start的数量比type_end的数量多三个，开始新的生成角色文本
        elif start_number == end_number + 3:
            state = 'generate_role'
            # 如果生成的最后一个特殊字符是span_start,那么开始生成角色文本
            if last_special_token == self.span_start:
                state = 'generate_role_text'
        else:
            state = 'error'
        return state, last_special_index

    def search_prefix_tree_and_sequence(self, generated: List[str], prefix_tree: Dict, src_sentence: List[str],
                                        end_sequence_search_tokens: List[str] = None):
        """
        Generate Type Name + Text Span
        :param generated:
        :param prefix_tree:
        :param src_sentence:
        :param end_sequence_search_tokens:
        :return:
        """
        tree = prefix_tree
        for index, token in enumerate(generated):
            tree = tree[token]
            is_tree_end = len(tree) == 1 and self.tree_end in tree

            if is_tree_end:
                valid_token = generated_search_src_sequence(
                    generated=generated[index + 1:],
                    src_sequence=src_sentence,
                    end_sequence_search_tokens=end_sequence_search_tokens,
                )
                return valid_token

            if self.tree_end in tree:
                try:
                    valid_token = generated_search_src_sequence(
                        generated=generated[index + 1:],
                        src_sequence=src_sentence,
                        end_sequence_search_tokens=end_sequence_search_tokens,
                    )
                    return valid_token
                except IndexError:
                    # Still search tree
                    continue

        valid_token = list(tree.keys())
        return valid_token

    def get_state_valid_tokens(self, src_sentence, tgt_generated):
        """

        :param src_sentence:
        :param tgt_generated:
        :return:
            List[str], valid token list
        """
        if self.tokenizer.eos_token_id in src_sentence:
            src_sentence = src_sentence[:src_sentence.index(self.tokenizer.eos_token_id)]

        if self.text_start in src_sentence:
            src_sentence = src_sentence[src_sentence.index(self.text_start) + 1:]

        state, index = self.check_state(tgt_generated)

        print("State: %s" % state) if debug else None

        if state == 'error':
            print("Decode Error:")
            print("Src:", self.tokenizer.convert_ids_to_tokens(src_sentence))
            print("Tgt:", self.tokenizer.convert_ids_to_tokens(tgt_generated))
            valid_tokens = [self.tokenizer.eos_token_id]

        elif state == 'start':
            valid_tokens = [self.type_start]

        elif state == 'start_first_generation':
            valid_tokens = [self.type_start, self.type_end]

        elif state == 'generate_trigger':

            if tgt_generated[-1] == self.type_start:
                # Start Event Label
                return list(self.type_tree.keys())

            elif tgt_generated[-1] == self.type_end:
                # EVENT_TYPE_LEFT: Start a new role
                # EVENT_TYPE_RIGHT: End this event
                return [self.type_start, self.type_end]
            else:
                valid_tokens = self.search_prefix_tree(
                    generated=tgt_generated[index + 1:],
                    prefix_tree=self.type_tree,
                    end_search_tokens=[self.span_start]
                )

        elif state in {'generate_trigger_text'}:
            generated = tgt_generated[index + 1:]

            if len(generated) > 0 and generated[-1] == self.null_span:
                return [self.type_end, self.type_start]

            valid_tokens = generated_search_src_sequence(
                generated=generated,
                src_sequence=src_sentence + [self.null_span],
                end_sequence_search_tokens=[self.type_end, self.type_start],
            )

        elif state in {'generate_role_text'}:
            generated = tgt_generated[index + 1:]

            if len(generated) > 0 and generated[-1] == self.null_span:
                return [self.type_end]

            valid_tokens = generated_search_src_sequence(
                generated=generated,
                src_sequence=src_sentence + [self.null_span],
                end_sequence_search_tokens=[self.type_end],
            )

        elif state == 'generate_role':

            if tgt_generated[-1] == self.type_start:
                # Start Role Label
                return list(self.role_tree.keys())

            generated = tgt_generated[index + 1:]
            valid_tokens = self.search_prefix_tree(
                generated=generated,
                prefix_tree=self.role_tree,
                end_search_tokens=[self.span_start]
            )

        elif state == 'end_generate':
            valid_tokens = [self.tokenizer.eos_token_id]

        else:
            raise NotImplementedError('State `%s` for %s is not implemented.' % (state, self.__class__))

        print("Valid: %s" % self.tokenizer.convert_ids_to_tokens(valid_tokens)) if debug else None
        return valid_tokens

    def search_prefix_tree(self, generated: List[str], prefix_tree: Dict,
                           end_search_tokens: List[str] = None):
        """
        Generate Type Name + Text Span
        :param generated:
        :param prefix_tree:
        :param src_sentence:
        :param end_search_tokens:
        :return:
        """
        tree = prefix_tree
        for index, token in enumerate(generated):
            tree = tree[token]
            is_tree_end = len(tree) == 1 and self.tree_end in tree

            if is_tree_end:
                return end_search_tokens

        valid_token = list(tree.keys())
        if self.tree_end in valid_token:
            valid_token.remove(self.tree_end)
            valid_token += end_search_tokens
        return valid_token


class SpotConstraintDecoder(SpotAsocConstraintDecoder):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

    def check_state(self, tgt_generated):
        if tgt_generated[-1] == self.tokenizer.pad_token_id:
            return 'start', -1

        special_token_set = {self.type_start, self.type_end, self.span_start}
        special_index_token = list(filter(lambda x: x[1] in special_token_set, list(enumerate(tgt_generated))))

        last_special_index, last_special_token = special_index_token[-1]

        if len(special_index_token) == 1:
            if last_special_token != self.type_start:
                return 'error', 0

        bracket_position = find_bracket_position(tgt_generated, _type_start=self.type_start, _type_end=self.type_end)
        start_number, end_number = len(bracket_position[self.type_start]), len(bracket_position[self.type_end])

        if start_number == end_number:
            return 'end_generate', -1
        if start_number == end_number + 1:
            state = 'start_first_generation'
        elif start_number == end_number + 2:
            state = 'generate_span'
            if last_special_token == self.span_start:
                state = 'generate_span_text'
        else:
            state = 'error'
        return state, last_special_index

    def get_state_valid_tokens(self, src_sentence, tgt_generated):
        """

        :param src_sentence:
        :param tgt_generated:
        :return:
            List[str], valid token list
        """
        if self.tokenizer.eos_token_id in src_sentence:
            src_sentence = src_sentence[:src_sentence.index(self.tokenizer.eos_token_id)]

        if self.text_start in src_sentence:
            src_sentence = src_sentence[src_sentence.index(self.text_start) + 1:]

        state, index = self.check_state(tgt_generated)

        print("State: %s" % state) if debug else None

        if state == 'error':
            print("Decode Error:")
            print("Src:", self.tokenizer.convert_ids_to_tokens(src_sentence))
            print("Tgt:", self.tokenizer.convert_ids_to_tokens(tgt_generated))
            valid_tokens = [self.tokenizer.eos_token_id]

        elif state == 'start':
            valid_tokens = [self.type_start]

        elif state == 'start_first_generation':
            valid_tokens = [self.type_start, self.type_end]

        elif state == 'generate_span':

            if tgt_generated[-1] == self.type_start:
                # Start Event Label
                return list(self.type_tree.keys())

            elif tgt_generated[-1] == self.type_end:
                raise RuntimeError('Invalid %s in %s' % (self.type_end, tgt_generated))

            else:
                valid_tokens = self.search_prefix_tree(
                    generated=tgt_generated[index + 1:],
                    prefix_tree=self.type_tree,
                    end_search_tokens=[self.span_start]
                )

        elif state == 'generate_span_text':
            generated = tgt_generated[index + 1:]
            valid_tokens = generated_search_src_sequence(
                generated=generated,
                src_sequence=src_sentence + [self.null_span],
                end_sequence_search_tokens=[self.type_end],
            )

        elif state == 'end_generate':
            valid_tokens = [self.tokenizer.eos_token_id]

        else:
            raise NotImplementedError('State `%s` for %s is not implemented.' % (state, self.__class__))

        print("Valid: %s" % valid_tokens) if debug else None
        return valid_tokens


def get_constraint_decoder(tokenizer, type_schema, source_prefix=None):
    if len(type_schema.role_list) == 0:
        task_map = SpotConstraintDecoder
    else:
        task_map = SpotAsocConstraintDecoder
    return task_map(tokenizer=tokenizer, type_schema=type_schema, source_prefix=source_prefix)


