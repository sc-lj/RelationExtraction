
import os
import json
import torch
import itertools
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer


class PLMarkerDataset(Dataset):
    def __init__(self, tokenizer, args, is_training):
        self.is_training = is_training
        self.max_seq_length = args.max_seq_length
        self.max_pair_length = args.max_pair_length
        self.max_entity_length = self.max_pair_length*2
        self.use_typemarker = args.use_typemarker
        self.model_type = args.m_type
        self.no_sym = args.no_sym
        self.args = args

        # 实体id
        self.type2index = json.load(open(os.path.join(args.data_dir, 'ner2id.json')))
        self.num_ner_labels = len(self.type2index)

        # 关系id
        relations = json.load(open(os.path.join(args.data_dir, 'rel2id.json')))
        relation_list = relations['relation']
        self.rel2index = {label: i for i, label in enumerate(relation_list)}
        relation_number = len(relation_list)
        # 使用对某些关系采用双向识别，即处于关系下的triple对是无向的。
        if args.no_sym:  # 不对特定关系采用双向识别
            self.sym_labels = relations['no_sym']
            # 关系类别进行double，去除了本身是可以双向的关系，其他的关系进行double
            self.num_labels = relation_number*2 - 1
        else:
            self.sym_labels = relations['no_sym'] + relations['sym']
            self.num_labels = relation_number*2 - 3

        if is_training:
            filename = os.path.join(args.data_dir, "train.json")
        else:
            filename = os.path.join(args.data_dir, "dev.json")

        with open(filename, 'r') as f:
            lines = f.readlines()

        self.tokenizer = tokenizer
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token

        # {(样本id, 句子id):[(实体start, 实体end, 实体label)]}
        self.global_predicted_ners = {}
        self.initialize(lines)

    def initialize(self, lines):
        max_num_subwords = self.max_seq_length - 4  # for two marker

        def tokenize_word(text):
            if (isinstance(self.tokenizer, RobertaTokenizer) and (text[0] != "'") and (len(text) != 1 or not self.is_punctuation(text))):
                return self.tokenizer.tokenize(text, add_prefix_space=True)
            return self.tokenizer.tokenize(text)

        self.ner_tot_recall = 0
        self.tot_recall = 0
        self.data = []
        # ((样本id, 句子id), (实体start, 实体end), 实体label))
        self.ner_golden_labels = set([])
        # ((样本id, 句子id), (sub_start,sub_end), (obj_start,obj_end), 关系label))
        self.golden_labels = set([])
        # ((样本id, 句子id), (sub_start,sub_end,sub_token), (obj_start,obj_end,obj_token), 关系label))
        self.golden_labels_withner = set([])
        maxR = 0
        maxL = 0
        for l_idx, line in tqdm(enumerate(lines)):
            data = json.loads(line)
            sentences = data['sentences']
            if 'predicted_ner' in data:       # e2e predict
                ners = data['predicted_ner']
            else:
                ners = data['ner']

            std_ners = data['ner']
            relations = data['relations']
            # 统计所有关系的数量
            for sentence_relation in relations:
                for x in sentence_relation:
                    # 判断标签是否在定义的对称关系中
                    if x[4] in self.sym_labels[1:]:
                        self.tot_recall += 2
                    else:
                        self.tot_recall += 1

            sentence_boundaries = [0]
            words = []
            L = 0
            # 统计每个句子的前后边界
            for i in range(len(sentences)):
                L += len(sentences[i])
                sentence_boundaries.append(L)
                words += sentences[i]

            tokens = [tokenize_word(w) for w in words]
            subwords = [w for li in tokens for w in li]
            maxL = max(maxL, len(subwords))  # 子token的最大长度
            # 子token的边界
            token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))
            # 每句话的子token长度集合
            subword_sentence_boundaries = [sum(len(li) for li in tokens[:p]) for p in sentence_boundaries]
            for n in range(len(subword_sentence_boundaries) - 1):
                sentence_ners = ners[n]  # 第n个句子标准或者预测的ner
                sentence_relations = relations[n]
                std_ner = std_ners[n]  # 第n个句子的ner
                std_entity_labels = {}
                self.ner_tot_recall += len(std_ner)
                for start, end, label in std_ner:
                    std_entity_labels[(start, end)] = label
                    self.ner_golden_labels.add(((l_idx, n), (start, end), label))
                self.global_predicted_ners[(l_idx, n)] = list(sentence_ners)

                # 取当前句子及其后面一句，组合成一个pair句子输入
                # 相邻两句话起始位置
                doc_sent_start, doc_sent_end = subword_sentence_boundaries[n: n + 2]
                left_length = doc_sent_start
                right_length = len(subwords) - doc_sent_end  # 当前pair句子右边距离文本end的长度
                sentence_length = doc_sent_end - doc_sent_start  # 两句话的长度

                # 如果句子长度小于最大长度，计算左右实际要补齐多少
                if sentence_length < max_num_subwords:
                    # 左右各要补齐多少长度，才能达到规定的最大句子长度
                    half_context_length = int((max_num_subwords - sentence_length) / 2)
                    if left_length < right_length:
                        left_context_length = min(left_length, half_context_length)
                        right_context_length = min(right_length, max_num_subwords - left_context_length - sentence_length)
                    else:
                        right_context_length = min(right_length, half_context_length)
                        left_context_length = min(left_length, max_num_subwords - right_context_length - sentence_length)

                # pair对子句起始位置需要向左偏移的长度
                doc_offset = doc_sent_start - left_context_length
                target_tokens = subwords[doc_offset: doc_sent_end + right_context_length]
                target_tokens = [self.cls_token] + target_tokens[: self.max_seq_length - 4] + [self.sep_token]
                assert(len(target_tokens) <= self.max_seq_length - 2)

                # 获取所有sub、obj的所有的relation
                # {(sub_start,sub_end,obj_start,obj_end):relation}
                pos2label = {}
                for x in sentence_relations:
                    # sub_start,sub_end,obj_start,obj_end
                    pos2label[(x[0], x[1], x[2], x[3])] = self.rel2index[x[4]]
                    self.golden_labels.add(((l_idx, n), (x[0], x[1]), (x[2], x[3]), x[4]))
                    self.golden_labels_withner.add(((l_idx, n), (x[0], x[1], std_entity_labels[(x[0], x[1])]),
                                                   (x[2], x[3], std_entity_labels[(x[2], x[3])]), x[4]))
                    if x[4] in self.sym_labels[1:]:  # 对于关系在定义的对称关系中的，sub和obj进行对调
                        self.golden_labels.add(((l_idx, n),  (x[2], x[3]), (x[0], x[1]), x[4]))
                        self.golden_labels_withner.add(((l_idx, n), (x[2], x[3], std_entity_labels[(
                            x[2], x[3])]), (x[0], x[1], std_entity_labels[(x[0], x[1])]), x[4]))

                for x in sentence_relations:
                    w = (x[2], x[3], x[0], x[1])
                    if w not in pos2label:
                        # 对于关系在定义的对称关系中的，sub和obj进行对调，其关系是仍然为原来的关系
                        if x[4] in self.sym_labels[1:]:
                            pos2label[w] = self.rel2index[x[4]]  # bug
                        else:
                            # 对非双向的关系的index进行double，即其关系为反方向
                            pos2label[w] = self.rel2index[x[4]] + len(self.rel2index) - len(self.sym_labels)

                # 当前第n个子句的所有ner
                entities = list(sentence_ners)
                if self.is_training:
                    entities.append((10000, 10000, 'NIL'))  # only for NER

                # subject 实体
                for sub in entities:
                    cur_ins = []  # 对于每个subject实体，在当前pair子句中遍历出其所有的可能的object
                    if sub[0] < 10000:
                        # 该实体在当前的pair子句中的起始index
                        sub_s = token2subword[sub[0]] - doc_offset + 1
                        sub_e = token2subword[sub[1]+1] - doc_offset
                        # 实体类型
                        sub_label = self.type2index[sub[2]]

                        if self.use_typemarker:
                            # 是否使用实体标签作为特殊符号
                            l_m = '[unused%d]' % (2 + sub_label)
                            r_m = '[unused%d]' % (2 + sub_label + len(self.type2index))
                        else:
                            l_m = '[unused0]'
                            r_m = '[unused1]'
                        # 插入特殊符号
                        sub_tokens = target_tokens[:sub_s] + [l_m] + target_tokens[sub_s:sub_e+1] + [r_m] + target_tokens[sub_e+1:]
                        sub_e += 2
                    else:
                        sub_s = len(target_tokens)
                        sub_e = len(target_tokens)+1
                        sub_tokens = target_tokens + ['[unused0]',  '[unused1]']
                        sub_label = -1

                    if sub_e >= self.max_seq_length-1:
                        continue
                    # assert(sub_e < self.max_seq_length)
                    # object实体
                    for start, end, obj_label in sentence_ners:
                        if self.model_type.endswith('nersub'):
                            # 剔除与subject完全一致的object
                            if start == sub[0] and end == sub[1]:
                                continue

                        doc_entity_start = token2subword[start]
                        doc_entity_end = token2subword[end+1]
                        # object 左偏移
                        left = doc_entity_start - doc_offset + 1
                        right = doc_entity_end - doc_offset

                        # 如果object在subject右边，其index 根据情况+2或者+1，这是因为文本中嵌入了两个特殊符号
                        obj = (start, end)
                        if obj[0] >= sub[0]:
                            left += 1
                            if obj[0] > sub[1]:
                                left += 1

                        if obj[1] >= sub[0]:
                            right += 1
                            if obj[1] > sub[1]:
                                right += 1
                        # 默认关系为NIL
                        label = pos2label.get((sub[0], sub[1], obj[0], obj[1]), 0)

                        if right >= self.max_seq_length-1:
                            continue
                        # ((实体stat，实体end,实体类型))
                        cur_ins.append(((left, right, self.type2index[obj_label]), label, obj))

                    maxR = max(maxR, len(cur_ins))
                    dL = self.max_pair_length
                    if self.args.shuffle:
                        np.random.shuffle(cur_ins)

                    for i in range(0, len(cur_ins), dL):
                        examples = cur_ins[i: i + dL]
                        item = {
                            'index': (l_idx, n),
                            'sentence': sub_tokens,
                            'examples': examples,
                            # (sub[0], sub[1], sub_label),
                            'sub': (sub, (sub_s, sub_e), sub_label),
                        }
                        self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        sub, sub_position, sub_label = entry['sub']
        input_ids = self.tokenizer.convert_tokens_to_ids(entry['sentence'])

        L = len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
        attention_mask = torch.zeros((self.max_entity_length+self.max_seq_length,
                                     self.max_entity_length+self.max_seq_length), dtype=torch.int64)
        attention_mask[:L, :L] = 1

        # object 的start
        input_ids = input_ids + [3] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (self.max_pair_length - len(entry['examples']))
        # object 的end
        input_ids = input_ids + [4] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * \
            (self.max_pair_length - len(entry['examples']))  # for debug

        labels = []
        ner_labels = []
        mention_pos = []
        mention_2 = []
        position_ids = list(range(self.max_seq_length)) + [0] * self.max_entity_length
        num_pair = self.max_pair_length

        for x_idx, obj in enumerate(entry['examples']):
            m2 = obj[0]
            label = obj[1]
            # mention 的index
            mention_pos.append((m2[0], m2[1]))
            # 该mention的初始的index
            mention_2.append(obj[2])

            w1 = x_idx  # 第x_idx个pair的start
            w2 = w1 + num_pair  # 第x_idx个pair的end

            w1 += self.max_seq_length
            w2 += self.max_seq_length

            position_ids[w1] = m2[0]
            position_ids[w2] = m2[1]

            for xx in [w1, w2]:
                for yy in [w1, w2]:
                    # 关注该pair内部
                    attention_mask[xx, yy] = 1
                # 关注该pair的start与所有的input_ids
                attention_mask[xx, :L] = 1

            labels.append(label)
            ner_labels.append(m2[2])

            if self.use_typemarker:
                l_m = '[unused%d]' % (2 + m2[2] + len(self.type2index)*2)
                r_m = '[unused%d]' % (2 + m2[2] + len(self.type2index)*3)
                l_m = self.tokenizer._convert_token_to_id(l_m)
                r_m = self.tokenizer._convert_token_to_id(r_m)
                input_ids[w1] = l_m
                input_ids[w2] = r_m

        pair_L = len(entry['examples'])
        if self.args.att_left:
            attention_mask[self.max_seq_length: self.max_seq_length + pair_L, self.max_seq_length: self.max_seq_length+pair_L] = 1
        if self.args.att_right:
            attention_mask[self.max_seq_length+num_pair: self.max_seq_length+num_pair +
                           pair_L, self.max_seq_length+num_pair: self.max_seq_length+num_pair+pair_L] = 1

        mention_pos += [(0, 0)] * (num_pair - len(mention_pos))
        labels += [-1] * (num_pair - len(labels))
        ner_labels += [-1] * (num_pair - len(ner_labels))

        item = [torch.tensor(input_ids),
                attention_mask,
                torch.tensor(position_ids),
                torch.tensor(sub_position),
                torch.tensor(mention_pos),
                torch.tensor(labels, dtype=torch.int64),
                torch.tensor(ner_labels, dtype=torch.int64),
                torch.tensor(sub_label, dtype=torch.int64)
                ]

        if not self.is_training:
            item.append(entry['index'])
            item.append(sub)
            item.append(mention_2)

        return item


def collate_fn(batch):
    fields = [x for x in zip(*batch)]

    num_metadata_fields = 3
    # don't stack metadata fields
    stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]
    # add them as lists not torch tensors
    stacked_fields.extend(fields[-num_metadata_fields:])

    return stacked_fields
