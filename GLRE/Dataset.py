# -*- encoding: utf-8 -*-
'''
@File    :   Dataset.py
@Time    :   2022/08/29 19:28:52
@Author  :   lujun
@Version :   1.0
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   GLRE 模型的Dataset
'''
import os
import json
import six
import torch
import numpy as np
import itertools
import random
from tqdm import tqdm
import scipy.sparse as sp
from collections import OrderedDict
from collections import namedtuple
from torch.utils.data import Dataset
from GLRE.utils import find_cross, get_distance, sparse_mxs_to_torch_sparse_tensor
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


class GLREDataset(Dataset):
    def __init__(self, args, is_training=True) -> None:
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(
            args.pretrain_path, cache_dir="./bertbaseuncased")
        self.is_training = is_training
        # 忽略相应关系的标签
        self.label2ignore = -1
        self.ign_label = "NA"

        # 实体id
        self.type2index = json.load(
            open(os.path.join(args.base_file, 'ner2id.json')))
        self.index2type = {v: k for k, v in self.type2index.items()}
        self.n_type, self.type2count = len(self.type2index.keys()), {}

        # 关系id
        self.rel2index = json.load(
            open(os.path.join(args.base_file, 'rel2id.json')))
        self.index2rel = {v: k for k, v in self.rel2index.items()}
        self.n_rel, self.rel2count = len(self.rel2index.keys()), {}

        self.word2index = json.load(
            open(os.path.join(self.base_file, "word2id.json")))
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.n_words, self.word2count = len(
            self.word2index.keys()), {'<UNK>': 1}

        self.singletons = []

        self.documents, self.entities, self.pairs = [], OrderedDict(), OrderedDict()

        # 将距离分为9组
        self.dis2idx_dir = np.zeros((800), dtype='int64')  # distance feature
        self.dis2idx_dir[1] = 1
        self.dis2idx_dir[2:] = 2
        self.dis2idx_dir[4:] = 3
        self.dis2idx_dir[8:] = 4
        self.dis2idx_dir[16:] = 5
        self.dis2idx_dir[32:] = 6
        self.dis2idx_dir[64:] = 7
        self.dis2idx_dir[128:] = 8
        self.dis2idx_dir[256:] = 9
        self.dis_size = 20
        self.PairInfo = namedtuple('PairInfo', 'type direction cross')

        if is_training:
            filename = os.path.join(args.data_dir, "train_annotated.json")
        else:
            filename = os.path.join(args.data_dir, "dev.json")

        with open(filename, 'r') as f:
            lines = json.load(f)
        lengths, sents, entities_cor_id = self.preprocess(lines)
        self.lowercase = True

    def preprocess(self, lines):
        """DocRED数据预处理
        Args:
            lines (_type_): _description_
        """
        lengths = []
        sents = []
        doc_id = -1
        document_meta = []
        entities_cor_id = {}
        for line in tqdm(lines):
            text_meta = {}
            documents = {}
            line = json.loads(line)
            doc_id += 1
            towrite_meta = str(doc_id) + "\t"  # pmid 0
            text_meta['pmid'] = doc_id

            Ls = [0]
            L = 0
            # 统计文档中每句话的长度，以及文档的总长度
            for x in line['sents']:
                L += len(x)
                Ls.append(L)
            # 将每句话中，如果某个字符带有空格，用特殊符号代替
            for x_index, x in enumerate(line['sents']):
                for ix_index, ix in enumerate(x):
                    if " " in ix:
                        assert ix == " " or ix == "  ", print(ix)
                        line['sents'][x_index][ix_index] = "_"
            # 拼接文档中句子
            sentence = [" ".join(x) for x in line['sents']]
            towrite_meta += "||".join(sentence)  # txt 1

            text_meta['txt'] = sentence
            self.documents.append(
                {"pmid": doc_id, "text": [t.split(' ') for t in sentence]})

            # 统计每个文档中最大句子长度
            lengths += [max([len(s) for s in [t.split(' ')
                            for t in sentence]])]
            # 句子数量
            sents += [len(sentence)]

            document_list = []
            for x in line['sents']:
                document_list.append(" ".join(x))
            p = " ".join(sentence)
            document = "\n".join(document_list)
            assert "   " not in document
            assert "||" not in p and "\t" not in p

            # 修正文档中标注的实体的基本信息
            vertexSet = line['vertexSet']
            for j in range(len(vertexSet)):
                for k in range(len(vertexSet[j])):
                    vertexSet[j][k]['name'] = str(vertexSet[j][k]['name']).replace('4.\nStranmillis Road',
                                                                                   'Stranmillis Road')
                    vertexSet[j][k]['name'] = str(
                        vertexSet[j][k]['name']).replace("\n", "")

            # 将文档中的实体在句子中位置信息修正为在文档中位置信息
            # point position added with sent start position
            for j in range(len(vertexSet)):
                for k in range(len(vertexSet[j])):
                    vertexSet[j][k]['sent_id'] = int(
                        vertexSet[j][k]['sent_id'])

                    sent_id = vertexSet[j][k]['sent_id']
                    assert sent_id < len(Ls)-1
                    sent_id = min(len(Ls)-1, sent_id)
                    dl = Ls[sent_id]
                    pos1 = vertexSet[j][k]['pos'][0]
                    pos2 = vertexSet[j][k]['pos'][1]
                    # 在文档中位置信息
                    vertexSet[j][k]['pos'] = (pos1 + dl, pos2 + dl)
                    # 在当前句子中位置信息
                    vertexSet[j][k]['s_pos'] = (pos1, pos2)

            # 组合成训练的标签
            labels = line.get('labels', [])
            train_triple = set([])
            towrite = ""
            for label in labels:
                train_triple.add((label['h'], label['t']))
            # 将数据集中其他实体进行两两匹配，组合成关系为NA的triple组
            na_triple = []
            for j in range(len(vertexSet)):
                for k in range(len(vertexSet)):
                    if (j != k):
                        if (j, k) not in train_triple:
                            na_triple.append((j, k))
                            labels.append({'h': j, 'r': 'NA', 't': k})

            sen_len = len(sentence)
            word_len = sum([len(t.split(' ')) for t in sentence])

            if doc_id not in self.entities:
                self.entities[doc_id] = OrderedDict()

            if doc_id not in self.pairs:
                self.pairs[doc_id] = OrderedDict()

            label_metas = []
            entities_dist = []
            for label in labels:
                l_meta = {}
                rel = label['r']  # 'type'
                dir = "L2R"  # no use 'dir'
                # 有关系的实体对保存在vertexSet中的实际信息
                head = vertexSet[label['h']]
                tail = vertexSet[label['t']]
                # head和tail实体是否在同一个句子中
                cross = find_cross(head, tail)
                l_meta["rel"] = str(rel)
                l_meta['direction'] = dir
                l_meta["cross"] = str(cross)  # head,tail 是否出现在同一个句子中
                l_meta["head"] = [head[0]['pos'][0],
                                  head[0]['pos'][1]]  # head实体的在文档中index
                l_meta["tail"] = [tail[0]['pos'][0],
                                  tail[0]['pos'][1]]  # tail实体的在文档中index

                # rel:0,dir:1,cross:2,head_pos:3,tail_pos:4
                towrite = towrite + "\t" + str(rel) + "\t" + str(dir) + "\t" + str(cross) + "\t" + str(
                    head[0]['pos'][0]) + "-" + str(head[0]['pos'][1]) + "\t" + str(tail[0]['pos'][0]) + "-" + str(
                    tail[0]['pos'][1])

                head_ent_info = {}
                # 某个实体可能出现多个句子中
                head_ent_info['id'] = label['h']  # 出现在vertexSet中的位置
                head_ent_info["name"] = [g['name'] for g in head]  # 实体name
                head_ent_info["type"] = [str(g['type'])
                                         for g in head]  # 出现在不同句子中，该name的实体类型
                head_ent_info["mstart"] = [
                    str(g['pos'][0]) for g in head]  # 出现在不同句子中，开始的位置
                head_ent_info["mend"] = [str(g['pos'][1])
                                         for g in head]  # 出现在不同句子中，结束的位置
                head_ent_info["sentNo"] = [
                    str(g['sent_id']) for g in head]  # 出现在不同句子中的id

                for x in head_ent_info["mstart"]:
                    assert int(x) <= word_len - \
                        1, print(label_metas, '\t', word_len)
                for x in head_ent_info["mend"]:
                    assert int(x) <= word_len - \
                        1, print(label_metas, '\t', word_len)
                for x in head_ent_info["sentNo"]:
                    assert int(x) <= sen_len - \
                        1, print(label_metas, '\t', word_len)

                head_ent_info["mstart"] = [
                    str(min(int(x), word_len - 1)) for x in head_ent_info["mstart"]]
                head_ent_info["mend"] = [
                    str(min(int(x), word_len)) for x in head_ent_info["mend"]]
                head_ent_info["sentNo"] = [
                    str(min(int(x), sen_len - 1)) for x in head_ent_info["sentNo"]]

                l_meta["head_ent_info"] = head_ent_info

                # h_label:5,name:6,type:7,h_h_pos:8,h_t_pos:9,h_sent:10
                towrite += "\t" + str(label['h']) + "\t" + '||'.join([g['name'] for g in head]) + "\t" + ":".join([str(g['type']) for g in head]) \
                    + "\t" + ":".join([str(g['pos'][0]) for g in head]) + "\t" + ":".join(
                    [str(g['pos'][1]) for g in head]) + "\t" \
                    + ":".join([str(g['sent_id']) for g in head])

                tail_ent_info = {}
                # 某个实体可能出现多个句子中
                tail_ent_info['id'] = label['t']  # 出现在vertexSet中的位置
                tail_ent_info["name"] = [g['name'] for g in tail]
                tail_ent_info["type"] = [str(g['type'])
                                         for g in tail]  # 出现在不同句子中，该name的实体类型
                tail_ent_info["mstart"] = [
                    str(g['pos'][0]) for g in tail]  # 出现在不同句子中，开始的位置
                tail_ent_info["mend"] = [str(g['pos'][1])
                                         for g in tail]  # 出现在不同句子中，结束的位置
                tail_ent_info["sentNo"] = [
                    str(g['sent_id']) for g in tail]  # 出现在不同句子中的id

                for x in tail_ent_info["mstart"]:
                    assert int(x) <= word_len, print(
                        label_metas, '\t', word_len)
                for x in tail_ent_info["mend"]:
                    assert int(x) <= word_len, print(
                        label_metas, '\t', word_len)
                for x in tail_ent_info["sentNo"]:
                    assert int(x) <= sen_len - \
                        1, print(label_metas, '\t', word_len)

                tail_ent_info["mstart"] = [
                    str(min(int(x), word_len - 1)) for x in tail_ent_info["mstart"]]
                tail_ent_info["mend"] = [
                    str(min(int(x), word_len)) for x in tail_ent_info["mend"]]
                tail_ent_info["sentNo"] = [
                    str(min(int(x), sen_len - 1)) for x in tail_ent_info["sentNo"]]

                l_meta["tail_ent_info"] = tail_ent_info
                label_metas.append(l_meta)

                # t_label:11,name:12,type:13,t_h_pos:14,t_t_pos:15,t_sent:16
                towrite += "\t" + str(label['t']) + "\t" + '||'.join([g['name'] for g in tail]) + "\t" + ":".join([str(g['type']) for g in tail]) \
                    + "\t" + ":".join([str(g['pos'][0]) for g in tail]) + "\t" + ":".join(
                    [str(g['pos'][1]) for g in tail]) + "\t" \
                    + ":".join([str(g['sent_id']) for g in tail])

                # entities
                if head_ent_info['id'] not in self.entities[doc_id]:
                    self.entities[doc_id][head_ent_info['index']
                                          ] = head_ent_info
                    entities_dist.append((head_ent_info['index'], min(
                        [int(a) for a in head_ent_info["mstart"]])))

                if tail_ent_info['id'] not in self.entities[doc_id]:
                    self.entities[doc_id][tail_ent_info['index']
                                          ] = tail_ent_info
                    entities_dist.append((tail_ent_info['index'], min(
                        [int(a) for a in tail_ent_info["mstart"]])))

                entity_pair_dis = get_distance(
                    head_ent_info["sentNo"], tail_ent_info["sentNo"])
                if (head_ent_info['id'], tail_ent_info['id']) not in self.pairs[doc_id]:
                    self.pairs[doc_id][(head_ent_info['id'], tail_ent_info['id'])] = [
                        self.PairInfo(rel, dir, entity_pair_dis)]

                else:
                    self.pairs[doc_id][(head_ent_info['id'], tail_ent_info['id'])].append(
                        self.PairInfo(rel, dir, entity_pair_dis))

            entities_dist.sort(key=lambda x: x[1], reverse=False)
            entities_cor_id[doc_id] = {}
            for coref_id, key in enumerate(entities_dist):
                entities_cor_id[doc_id][key[0]] = coref_id + 1

            text_meta['label'] = label_metas

            document_meta.append(text_meta)

        for did, p in self.pairs.items():
            for k, vs in p.items():
                for v in vs:
                    self.add_relation(v.type)

        self.find_max_length(lengths)
        # map types and positions and relation types
        for d in self.documents:
            self.add_document(d['text'])

        return lengths, sents, entities_cor_id

    def add_document(self, document):
        for sentence in document:
            self.add_sentence(sentence)

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if self.lowercase:
            word = word.lower()
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            if word not in self.word2count:
                self.word2count[word] = 0
            self.word2count[word] += 1

    def find_max_length(self, length):
        """ Maximum distance between words """
        for l in length:
            if l-1 > self.max_distance:
                self.max_distance = l-1

    def add_relation(self, rel):
        assert rel in self.rel2index
        if rel not in self.rel2index:
            self.rel2index[rel] = self.n_rel
            self.rel2count[rel] = 1
            self.index2rel[self.n_rel] = rel
            self.n_rel += 1
        else:
            if rel not in self.rel2count:
                self.rel2count[rel] = 0
            self.rel2count[rel] += 1

    def find_singletons(self, min_w_freq=1):
        """
        Find items with frequency <= 2 and based on probability
        """
        self.singletons = frozenset([elem for elem, val in self.word2count.items()
                                     if (val <= min_w_freq) and elem != 'UNK'])

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        document = self.documents[index]
        pmid = document['pmid']
        sentences = document['text']
        prune_recall = {"0-max": 0, "0-1": 0,
                        "0-3": 0, "1-3": 0, "1-max": 0, "3-max": 0}
        # TEXT
        doc = []
        sens_len = []
        words = []
        start_idx = 0
        for i, sentence in enumerate(sentences):
            words += sentence
            start_idx += len(sentence)
            sent = []
            if self.is_training:
                for w, word in enumerate(sentence):
                    if self.lowercase:
                        word = word.lower()
                    if word not in self.word2index:
                        # UNK words = singletons for train
                        sent += [self.word2index['UNK']]
                    elif (word in self.singletons) and (random.uniform(0, 1) < float(self.unk_w_prob)):
                        sent += [self.word2index['UNK']]
                    else:
                        sent += [self.word2index[word]]

            else:
                for w, word in enumerate(sentence):
                    if self.lowercase:
                        word = word.lower()
                    if word in self.word2index:
                        sent += [self.word2index[word]]
                    else:
                        sent += [self.word2index['UNK']]
            assert len(sentence) == len(sent), '{}, {}'.format(
                len(sentence), len(sent))
            doc += [sent]
            sens_len.append(len(sent))

        bert_token, bert_mask, bert_starts = self.bert.subword_tokenize_to_ids(
            words)

        # NER
        # ner = [0] * sum(sens_len)
        # for id_, (e, i) in enumerate(self.entities[pmid].items()):
        #     for sent_id, m1, m2, itype in zip(i.sentNo, i.mstart, i.mend, i.type):
        #         for j in range(int(m1), int(m2)):
        #             ner[j] = self.type2index[itype]

        # ENTITIES [id, type, start, end] + NODES [id, type, start, end, node_type_id]
        nodes = []
        ent = []
        ent_sen_mask = np.zeros(
            (len(self.entities[pmid].items()), len(sens_len)), dtype=np.float32)
        for id_, (e, i) in enumerate(self.entities[pmid].items()):
            nodes += [[id_, self.type2index[i['type'][0]], min([int(ms) for ms in i['mstart']]),
                       min([int(me) for me in i['mend']]), int(i['sentNo'][0]), 0]]

            for sen_id in i.sentNo:
                ent_sen_mask[id_][int(sen_id)] = 1.0
        entity_size = len(nodes)

        nodes_mention = []
        for id_, (e, i) in enumerate(self.entities[pmid].items()):
            for sent_id, m1, m2 in zip(i['sentNo'], i['mstart'], i['mend']):
                ent += [[id_, self.type2index[i['type'][0]],
                         int(m1), int(m2), int(sent_id), 1]]
                nodes_mention += [[id_, self.type2index[i['type']
                                                        [0]], int(m1), int(m2), int(sent_id), 1]]

        ent.sort(key=lambda x: x[0], reverse=False)
        nodes_mention.sort(key=lambda x: x[0], reverse=False)
        nodes += nodes_mention

        for s, sentence in enumerate(sentences):
            nodes += [[s, s, s, s, s, 2]]

        nodes = np.array(nodes)

        max_node_cnt = max(max_node_cnt, nodes.shape[0])
        ent = np.array(ent)

        # RELATIONS
        ents_keys = list(self.entities[pmid].keys())  # in order
        trel = -1 * np.ones((len(ents_keys), len(ents_keys)))
        relation_multi_label = np.zeros(
            (len(ents_keys), len(ents_keys), self.n_rel))
        rel_info = np.empty((len(ents_keys), len(ents_keys)), dtype='object_')
        for id_, (r, ii) in enumerate(self.pairs[pmid].items()):
            rt = np.random.randint(len(ii))
            trel[ents_keys.index(r[0]), ents_keys.index(
                r[1])] = self.rel2index[ii[0].type]
            relation_set = set()
            for i in ii:
                assert relation_multi_label[ents_keys.index(
                    r[0]), ents_keys.index(r[1]), self.rel2index[i.type]] != 1.0
                relation_multi_label[ents_keys.index(r[0]), ents_keys.index(
                    r[1]), self.rel2index[i.type]] = 1.0
                assert self.ign_label == "NA" or self.ign_label == "1:NR:2"
                if i.type != self.ign_label:
                    assert relation_multi_label[ents_keys.index(r[0]), ents_keys.index(
                        r[1]), self.rel2index[self.ign_label]] != 1.0
                relation_set.add(self.rel2index[i.type])

                if i.type != self.ign_label:
                    dis_cross = int(i.cross)
                    if dis_cross == 0:
                        prune_recall['0-1'] += 1
                        prune_recall['0-3'] += 1
                        prune_recall['0-max'] += 1
                    elif dis_cross < 3:
                        prune_recall['0-3'] += 1
                        prune_recall['1-3'] += 1
                        prune_recall['1-max'] += 1
                        prune_recall['0-max'] += 1
                    else:
                        prune_recall['0-max'] += 1
                        prune_recall['3-max'] += 1
                        prune_recall['1-max'] += 1

            rel_info[ents_keys.index(r[0]), ents_keys.index(r[1])] = OrderedDict(
                [('pmid', pmid),
                 ('sentA', self.entities[pmid][r[0]]['sentNo']),
                 ('sentB',
                  self.entities[pmid][r[1]]['sentNo']),
                 ('doc', self.documents[pmid]),
                 ('entA', self.entities[pmid][r[0]]),
                 ('entB', self.entities[pmid][r[1]]),
                 ('rel', relation_set),
                 ('dir', ii[rt].direction),
                 ('cross', ii[rt].cross)])

            assert nodes[ents_keys.index(r[0])][2] == min(
                [int(ms) for ms in self.entities[pmid][r[0]]['mstart']])

            #######################
            # DISTANCES
            #######################
            xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(
                nodes.shape[0]), indexing='ij')

            r_id, c_id = nodes[xv, 5], nodes[yv, 5]  # node type
            r_Eid, c_Eid = nodes[xv, 0], nodes[yv, 0]
            r_Sid, c_Sid = nodes[xv, 4], nodes[yv, 4]
            r_Ms, c_Ms = nodes[xv, 2], nodes[yv, 2]
            r_Me, c_Me = nodes[xv, 3]-1, nodes[yv, 3]-1

            # dist feature
            dist_dir_h_t = np.full((r_id.shape[0], r_id.shape[0]), 0)

            # MM: mention-mention
            a_start = np.where(np.logical_or(r_id == 1, r_id == 3) & np.logical_or(
                c_id == 1, c_id == 3), r_Ms, -1)
            b_start = np.where(np.logical_or(r_id == 1, r_id == 3) & np.logical_or(
                c_id == 1, c_id == 3), c_Ms, -1)

            dis = a_start - b_start
            dis_index = np.where(
                dis < 0, -self.dis2idx_dir[-dis], self.dis2idx_dir[dis])
            condition = (np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3)
                         & (a_start != -1) & (b_start != -1))
            dist_dir_h_t = np.where(condition, dis_index, dist_dir_h_t)

            # EE: entity-entity
            a_start = np.where((r_id == 0) & (c_id == 0), r_Ms, -1)
            b_start = np.where((r_id == 0) & (c_id == 0), c_Ms, -1)
            dis = a_start - b_start

            dis_index = np.where(
                dis < 0, -self.dis2idx_dir[-dis], self.dis2idx_dir[dis])
            condition = ((r_id == 0) & (c_id == 0) & (
                a_start != -1) & (b_start != -1))
            dist_dir_h_t = np.where(condition, dis_index, dist_dir_h_t)

            #######################
            # GRAPH CONNECTIONS
            #######################
            adjacency = np.full((r_id.shape[0], r_id.shape[0]), 0, 'i')
            rgcn_adjacency = np.full((5, r_id.shape[0], r_id.shape[0]), 0.0)

            # mention-mention
            adjacency = np.where(np.logical_or(r_id == 1, r_id == 3) & np.logical_or(
                c_id == 1, c_id == 3) & (r_Sid == c_Sid), 1, adjacency)  # in same sentence
            rgcn_adjacency[0] = np.where(
                np.logical_or(r_id == 1, r_id == 3) & np.logical_or(
                    c_id == 1, c_id == 3) & (r_Sid == c_Sid), 1,
                rgcn_adjacency[0])

            # entity-mention
            adjacency = np.where((r_id == 0) & (c_id == 1) & (
                r_Eid == c_Eid), 1, adjacency)  # belongs to entity
            adjacency = np.where((r_id == 1) & (c_id == 0)
                                 & (r_Eid == c_Eid), 1, adjacency)
            rgcn_adjacency[1] = np.where((r_id == 0) & (c_id == 1) & (
                r_Eid == c_Eid), 1, rgcn_adjacency[1])  # belongs to entity
            rgcn_adjacency[1] = np.where((r_id == 1) & (
                c_id == 0) & (r_Eid == c_Eid), 1, rgcn_adjacency[1])

            # sentence-sentence (direct + indirect)
            adjacency = np.where((r_id == 2) & (c_id == 2), 1, adjacency)
            rgcn_adjacency[2] = np.where(
                (r_id == 2) & (c_id == 2), 1, rgcn_adjacency[2])

            # mention-sentence
            adjacency = np.where(np.logical_or(r_id == 1, r_id == 3) & (
                c_id == 2) & (r_Sid == c_Sid), 1, adjacency)  # belongs to sentence
            adjacency = np.where((r_id == 2) & np.logical_or(
                c_id == 1, c_id == 3) & (r_Sid == c_Sid), 1, adjacency)
            rgcn_adjacency[3] = np.where(np.logical_or(r_id == 1, r_id == 3) & (
                c_id == 2) & (r_Sid == c_Sid), 1, rgcn_adjacency[3])  # belongs to sentence
            rgcn_adjacency[3] = np.where((r_id == 2) & np.logical_or(
                c_id == 1, c_id == 3) & (r_Sid == c_Sid), 1, rgcn_adjacency[3])

            # entity-sentence
            for x, y in zip(xv.ravel(), yv.ravel()):
                if nodes[x, 5] == 0 and nodes[y, 5] == 2:  # this is an entity-sentence edge
                    z = np.where((r_Eid == nodes[x, 0]) & (r_id == 1) & (
                        c_id == 2) & (c_Sid == nodes[y, 4]))

                    # at least one M in S
                    temp_ = np.where((r_id == 1) & (c_id == 2)
                                     & (r_Sid == c_Sid), 1, adjacency)
                    temp_ = np.where((r_id == 2) & (c_id == 1)
                                     & (r_Sid == c_Sid), 1, temp_)
                    adjacency[x, y] = 1 if (temp_[z] == 1).any() else 0
                    adjacency[y, x] = 1 if (temp_[z] == 1).any() else 0
                    rgcn_adjacency[4][x, y] = 1 if (temp_[z] == 1).any() else 0
                    rgcn_adjacency[4][y, x] = 1 if (temp_[z] == 1).any() else 0

            rgcn_adjacency = sparse_mxs_to_torch_sparse_tensor(
                [sp.coo_matrix(rgcn_adjacency[i]) for i in range(5)])

            dist_dir_h_t = dist_dir_h_t[0: entity_size, 0:entity_size]
            data = {'ents': ent, 'rels': trel, 'multi_rels': relation_multi_label,
                    'dist_dir': dist_dir_h_t, 'text': doc, 'info': rel_info,
                    'adjacency': adjacency, 'rgcn_adjacency': rgcn_adjacency,
                    'section': np.array([len(self.entities[pmid].items()), ent.shape[0], len(doc), sum([len(s) for s in doc])]),
                    'word_sec': np.array([len(s) for s in doc]),
                    'bert_token': bert_token, 'bert_mask': bert_mask, 'bert_starts': bert_starts}

        return data, prune_recall


def collate_fn(batch, NA_NUM, NA_id, istrain=False):
    new_batch = {'entities': [], 'bert_token': [],
                 'bert_mask': [], 'bert_starts': [], 'pos_idx': []}
    ent_count, sent_count, word_count = 0, 0, 0
    full_text = []

    for i, b in enumerate(batch):
        current_text = list(itertools.chain.from_iterable(b['text']))
        full_text += current_text
        new_batch['bert_token'] += [b['bert_token']]
        new_batch['bert_mask'] += [b['bert_mask']]
        new_batch['bert_starts'] += [b['bert_starts']]

        temp = []
        for e in b['ents']:
            temp += [[e[0] + ent_count, e[1], e[2] + word_count, e[3] + word_count,
                      e[4] + sent_count, e[4], e[5]]]  # id  type start end sent_id

        new_batch['entities'] += [np.array(temp)]
        word_count += sum([len(s) for s in b['text']])
        ent_count = max([t[0] for t in temp]) + 1
        sent_count += len(b['text'])

    new_batch['entities'] = np.concatenate(
        new_batch['entities'], axis=0)  # 56, 6
    new_batch['entities'] = torch.as_tensor(new_batch['entities']).long()
    new_batch['bert_token'] = torch.as_tensor(
        np.concatenate(new_batch['bert_token'])).long()
    new_batch['bert_mask'] = torch.as_tensor(
        np.concatenate(new_batch['bert_mask'])).long()
    new_batch['bert_starts'] = torch.as_tensor(
        np.concatenate(new_batch['bert_starts'])).long()

    batch_ = [{k: v for k, v in b.items() if (
        k != 'info' and k != 'text' and k != 'rgcn_adjacency')} for b in batch]
    converted_batch = concat_examples(batch_, padding=-1)

    converted_batch['adjacency'][converted_batch['adjacency'] == -1] = 0
    converted_batch['dist_dir'][converted_batch['dist_dir'] == -1] = 0

    new_batch['adjacency'] = converted_batch['adjacency'].float()  # 2,71,71
    new_batch['distances_dir'] = converted_batch['dist_dir'].long()  # 2,71,71
    new_batch['relations'] = converted_batch['rels'].float()
    new_batch['multi_relations'] = converted_batch['multi_rels'].float().clone()
    if istrain and NA_NUM < 1.0:
        index = new_batch['multi_relations'][:, :, :, NA_id].nonzero()
        if index.size(0) != 0:
            value = (torch.rand(len(index)) < NA_NUM).float()
            if (value == 0).all():
                value[0] = 1.0
            new_batch['multi_relations'][index[:, 0],
                                         index[:, 1], index[:, 2], NA_id] = value

    new_batch['section'] = converted_batch['section'].long()  # 2, 4
    new_batch['word_sec'] = converted_batch['word_sec'][converted_batch['word_sec']
                                                        != -1].long()  # 21
    new_batch['rgcn_adjacency'] = convert_3dsparse_to_4dsparse(
        [b['rgcn_adjacency'] for b in batch])

    return new_batch


def concat_examples(batch, padding=-1):
    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]

    if isinstance(first_elem, tuple):
        result = []
        if not isinstance(padding, tuple):
            padding = [padding] * len(first_elem)

        for i in six.moves.range(len(first_elem)):
            result.append(torch.as_tensor(_concat_arrays(
                [example[i] for example in batch], padding[i])))

        return tuple(result)

    elif isinstance(first_elem, dict):
        result = {}
        if not isinstance(padding, dict):
            padding = {key: padding for key in first_elem}
        padding['multi_rels'] = 0
        padding['entA_mapping'] = 0
        padding['entB_mapping'] = 0
        padding['dep_adj'] = 0

        for key in first_elem:
            # flag = True
            # if key == "ent_sen_mask":
            #     flag = False
            result[key] = torch.as_tensor(_concat_arrays(
                [example[key] for example in batch], padding[key]))

        return result

    else:
        return torch.as_tensor(_concat_arrays(batch, padding))


def _concat_arrays(arrays, padding):
    # Convert `arrays` to numpy.ndarray if `arrays` consists of the built-in
    # types such as int, float or list.
    if not isinstance(arrays[0], type(torch.get_default_dtype())):
        arrays = np.asarray(arrays)

    if padding is not None:
        arr_concat = _concat_arrays_with_padding(arrays, padding)
    else:
        arr_concat = np.concatenate([array[None] for array in arrays])

    return arr_concat


def _concat_arrays_with_padding(arrays, padding):
    shape = np.array(arrays[0].shape, dtype=int)
    for array in arrays[1:]:
        if np.any(shape != array.shape):
            np.maximum(shape, array.shape, shape)
    shape = tuple(np.insert(shape, 0, len(arrays)))

    result = np.full(shape, padding, dtype=arrays[0].dtype)
    for i in six.moves.range(len(arrays)):
        src = arrays[i]
        slices = tuple(slice(dim) for dim in src.shape)
        result[(i,) + slices] = src

    return result


def convert_3dsparse_to_4dsparse(sparse_mxs):
    """
    :param sparse_mxs: [3d_sparse_tensor]
    :return:
    """
    max_shape = 0
    for mx in sparse_mxs:
        max_shape = max(max_shape, mx.shape[1])
    b_index = []
    indexs = []
    values = []
    for index, sparse_mx in enumerate(sparse_mxs):
        indices_ = sparse_mx._indices()
        values_ = sparse_mx._values()
        b_index.extend([index] * values_.shape[0])
        indexs.append(indices_)
        values.append(values_)
    indexs = torch.cat(indexs, dim=-1)
    b_index = torch.as_tensor(b_index)
    b_index = b_index.unsqueeze(0)
    indices = torch.cat([b_index, indexs], dim=0)
    values = torch.cat(values, dim=-1)
    shape = torch.Size(
        [len(sparse_mxs), sparse_mxs[0].shape[0], max_shape, max_shape])
    return torch.sparse.FloatTensor(indices, values, shape)
