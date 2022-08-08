import torch
from tqdm import tqdm
import os
import json
import re
import copy



class HandshakingTaggingScheme(object):
    def __init__(self, args):
        super().__init__()
        with open(args.relation, 'r') as f:
            relation = json.load(f)
        rel2id = relation[1]
        self.rel2id = rel2id
        self.id2rel = {ind: rel for rel, ind in rel2id.items()}

        self.separator = "\u2E80"
        self.link_types = {"SH2OH",  # subject head to object head
                           "OH2SH",  # object head to subject head
                           "ST2OT",  # subject tail to object tail
                           "OT2ST",  # object tail to subject tail
                           }
        # 将关系类型和组合类型进行组合,{RELNAME\u2E80SH2OH,RELNAME\u2E80OH2SH}
        self.tags = {self.separator.join(
            [rel, lt]) for rel in self.rel2id.keys() for lt in self.link_types}
        with open(args.ent2id_path, 'r') as f:
            entity_type2id = json.load(f)
        self.ent2id = entity_type2id
        self.id2ent = {ind: ent for ent, ind in self.ent2id.items()}
        # 将实体类型和实体表示组合
        # EH2ET: entity head to entity tail {ENTNAME\u2E80EH2ET}
        self.tags |= {self.separator.join(
            [ent, "EH2ET"]) for ent in self.ent2id.keys()}

        self.tags = sorted(self.tags)

        self.tag2id = {t: idx for idx, t in enumerate(self.tags)}
        self.id2tag = {idx: t for t, idx in self.tag2id.items()}
        self.matrix_size = args.max_seq_len

        # map,上三角组合
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_idx2matrix_idx = [(ind, end_ind) for ind in range(
            self.matrix_size) for end_ind in list(range(self.matrix_size))[ind:]]
        # [[0,1,2,3,4],[0,5,6,7,8],[0,0,9,10,11]]
        self.matrix_idx2shaking_idx = [
            [0 for i in range(self.matrix_size)] for j in range(self.matrix_size)]
        for shaking_ind, matrix_ind in enumerate(self.shaking_idx2matrix_idx):
            self.matrix_idx2shaking_idx[matrix_ind[0]
                                        ][matrix_ind[1]] = shaking_ind

    def get_tag_size(self):
        return len(self.tag2id)

    def get_spots(self, sample):
        """将一个样本中所有的sub,obj的head,tail的索引与tag id进行组合

        Args:
            sample (_type_): _description_
        Returns:
            matrix_spots: [(tok_pos1, tok_pos2, tag_id), ]
        """
        matrix_spots = []
        spot_memory_set = set()

        def add_spot(spot):
            memory = "{},{},{}".format(*spot)
            if memory not in spot_memory_set:
                matrix_spots.append(spot)
                spot_memory_set.add(memory)
        # if entity_list exist, need to distinguish entity types
        # if self.ent2id is not None and "entity_list" in sample:
        for ent in sample["entity_list"]:
            # "entity_1_index,entity_2_index,entity_type\u2E80EH2ET"
            add_spot((ent["tok_span"][0], ent["tok_span"][1] - 1,
                     self.tag2id[self.separator.join([ent["type"], "EH2ET"])]))

        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            rel = rel["predicate"]
            # if self.ent2id is None: # set all entities to default type
            #     add_spot((subj_tok_span[0], subj_tok_span[1] - 1, self.tag2id[self.separator.join(["DEFAULT", "EH2ET"])]))
            #     add_spot((obj_tok_span[0], obj_tok_span[1] - 1, self.tag2id[self.separator.join(["DEFAULT", "EH2ET"])]))
            if subj_tok_span[0] <= obj_tok_span[0]:
                # "sub_head_index,obj_head_index,rel_type\u2E80SH2OH"
                add_spot((subj_tok_span[0], obj_tok_span[0],
                         self.tag2id[self.separator.join([rel, "SH2OH"])]))
            else:
                # "obj_head_index,sub_head_index,rel_type\u2E80OH2SH"
                add_spot((obj_tok_span[0], subj_tok_span[0],
                         self.tag2id[self.separator.join([rel, "OH2SH"])]))

            if subj_tok_span[1] <= obj_tok_span[1]:
                # "sub_tail_index,obj_tail_index,rel_type\u2E80ST2OT"
                add_spot((subj_tok_span[1] - 1, obj_tok_span[1] - 1,
                         self.tag2id[self.separator.join([rel, "ST2OT"])]))
            else:
                # "obj_tail_index,sub_tail_index,rel_type\u2E80OT2ST"
                add_spot((obj_tok_span[1] - 1, subj_tok_span[1] - 1,
                         self.tag2id[self.separator.join([rel, "OT2ST"])]))
        return matrix_spots

    def spots2shaking_tag(self, spots):
        ''' 舍弃了
        convert spots to matrix tag
        spots: [(start_ind, end_ind, tag_id), ]
        return: 
            shaking_tag: (shaking_seq_len, tag_size)
        '''
        # 握手序列的长度
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_tag = torch.zeros(shaking_seq_len, len(self.tag2id)).long()
        for sp in spots:
            shaking_idx = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
            shaking_tag[shaking_idx][sp[2]] = 1
        return shaking_tag

    def spots2shaking_tag4batch(self, batch_spots):
        ''' 对一个batch中的数据进行转换
        batch_spots: a batch of spots, [spots1, spots2, ...]
            spots: [(start_ind, end_ind, tag_id), ]
        return: 
            batch_shaking_tag: (batch_size, shaking_seq_len, tag_size)
        '''
        # 握手序列的长度
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_tag = torch.zeros(
            len(batch_spots), shaking_seq_len, len(self.tag2id)).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_idx = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
                batch_shaking_tag[batch_id][shaking_idx][sp[2]] = 1
        return batch_shaking_tag

    def get_spots_fr_shaking_tag(self, shaking_tag):
        ''' 将shaking tag转换为spots
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, tag_id)
        spots: [(start_ind, end_ind, tag_id), ]
        '''
        spots = []
        nonzero_points = torch.nonzero(shaking_tag, as_tuple=False)
        for index,point in enumerate(nonzero_points):
            shaking_idx, tag_idx = point[0].item(), point[1].item()
            pos1, pos2 = self.shaking_idx2matrix_idx[shaking_idx]
            spot = (pos1, pos2, tag_idx)
            spots.append(spot)
        return spots

    def decode_rel(self,
                   text,
                   shaking_tag,
                   tok2char_span,
                   tok_offset=0, char_offset=0):
        '''
        # 解码关系
        shaking_tag: (shaking_seq_len, tag_id_num)
        '''
        rel_list = []
        matrix_spots = self.get_spots_fr_shaking_tag(shaking_tag)

        # entity
        head_ind2entities = {}
        ent_list = []
        for sp in matrix_spots:
            # 组合的tag
            tag = self.id2tag[sp[2]]
            # 实体(关系)类型及其连接类型
            ent_type, link_type = tag.split(self.separator)
            # for an entity, the start position can not be larger than the end pos.
            if link_type != "EH2ET" or sp[0] > sp[1]:
                continue
            # 获取token spans
            char_span_list = tok2char_span[sp[0]:sp[1] + 1]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            ent_text = text[char_sp[0]:char_sp[1]]
            entity = {
                "type": ent_type,
                "text": ent_text,
                "tok_span": [sp[0], sp[1] + 1],
                "char_span": char_sp,
            }
            # take ent_head_pos as the key to entity list
            head_key = str(sp[0])
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            # 实体头部index为key
            head_ind2entities[head_key].append(entity)
            ent_list.append(entity)

        # tail link，sub和obj的尾部index
        tail_link_memory_set = set()
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            rel, link_type = tag.split(self.separator)
            if link_type == "ST2OT":
                tail_link_memory = self.separator.join(
                    [rel, str(sp[0]), str(sp[1])])
                tail_link_memory_set.add(tail_link_memory)
            elif link_type == "OT2ST":
                tail_link_memory = self.separator.join(
                    [rel, str(sp[1]), str(sp[0])])
                tail_link_memory_set.add(tail_link_memory)

        # head link，sub和obj的头部index
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            rel, link_type = tag.split(self.separator)

            if link_type == "SH2OH":
                subj_head_key, obj_head_key = str(sp[0]), str(sp[1])
            elif link_type == "OH2SH":
                subj_head_key, obj_head_key = str(sp[1]), str(sp[0])
            else:
                continue

            if subj_head_key not in head_ind2entities or obj_head_key not in head_ind2entities:
                # no entity start with subj_head_key and obj_head_key
                continue

            # all entities start with this subject head
            subj_list = head_ind2entities[subj_head_key]
            # all entities start with this object head
            obj_list = head_ind2entities[obj_head_key]

            # go over all subj-obj pair to check whether the tail link exists
            for subj in subj_list:
                for obj in obj_list:
                    tail_link_memory = self.separator.join(
                        [rel, str(subj["tok_span"][1] - 1), str(obj["tok_span"][1] - 1)])
                    if tail_link_memory not in tail_link_memory_set:
                        # no such relation
                        continue
                    rel_list.append({
                        "subject": subj["text"],
                        "object": obj["text"],
                        "subj_tok_span": [subj["tok_span"][0] + tok_offset, subj["tok_span"][1] + tok_offset],
                        "obj_tok_span": [obj["tok_span"][0] + tok_offset, obj["tok_span"][1] + tok_offset],
                        "subj_char_span": [subj["char_span"][0] + char_offset, subj["char_span"][1] + char_offset],
                        "obj_char_span": [obj["char_span"][0] + char_offset, obj["char_span"][1] + char_offset],
                        "predicate": rel,
                    })
            # recover the positons in the original text
            for ent in ent_list:
                ent["char_span"] = [ent["char_span"][0] +
                                    char_offset, ent["char_span"][1] + char_offset]
                ent["tok_span"] = [ent["tok_span"][0] +
                                   tok_offset, ent["tok_span"][1] + tok_offset]

        return rel_list, ent_list

    def trans2ee(self, rel_list, ent_list):
        sepatator = "_"  # \u2E80
        trigger_set, arg_iden_set, arg_class_set = set(), set(), set()
        trigger_offset2vote = {}
        trigger_offset2trigger_text = {}
        trigger_offset2trigger_char_span = {}
        # get candidate trigger types from relation
        for rel in rel_list:
            trigger_offset = rel["obj_tok_span"]
            trigger_offset_str = "{},{}".format(
                trigger_offset[0], trigger_offset[1])
            trigger_offset2trigger_text[trigger_offset_str] = rel["object"]
            trigger_offset2trigger_char_span[trigger_offset_str] = rel["obj_char_span"]
            _, event_type = rel["predicate"].split(sepatator)

            if trigger_offset_str not in trigger_offset2vote:
                trigger_offset2vote[trigger_offset_str] = {}
            trigger_offset2vote[trigger_offset_str][event_type] = trigger_offset2vote[trigger_offset_str].get(
                event_type, 0) + 1

        # get candidate trigger types from entity types
        for ent in ent_list:
            t1, t2 = ent["type"].split(sepatator)
            assert t1 == "Trigger" or t1 == "Argument"
            if t1 == "Trigger":  # trigger
                event_type = t2
                trigger_span = ent["tok_span"]
                trigger_offset_str = "{},{}".format(
                    trigger_span[0], trigger_span[1])
                trigger_offset2trigger_text[trigger_offset_str] = ent["text"]
                trigger_offset2trigger_char_span[trigger_offset_str] = ent["char_span"]
                if trigger_offset_str not in trigger_offset2vote:
                    trigger_offset2vote[trigger_offset_str] = {}
                trigger_offset2vote[trigger_offset_str][event_type] = trigger_offset2vote[trigger_offset_str].get(
                    event_type, 0) + 1.1  # if even, entity type makes the call

        # voting
        tirigger_offset2event = {}
        for trigger_offet_str, event_type2score in trigger_offset2vote.items():
            event_type = sorted(event_type2score.items(),
                                key=lambda x: x[1], reverse=True)[0][0]
            # final event type
            tirigger_offset2event[trigger_offet_str] = event_type

        # generate event list
        trigger_offset2arguments = {}
        for rel in rel_list:
            trigger_offset = rel["obj_tok_span"]
            argument_role, event_type = rel["predicate"].split(sepatator)
            trigger_offset_str = "{},{}".format(
                trigger_offset[0], trigger_offset[1])
            # filter false relations
            if tirigger_offset2event[trigger_offset_str] != event_type:
                #                 set_trace()
                continue

            # append arguments
            if trigger_offset_str not in trigger_offset2arguments:
                trigger_offset2arguments[trigger_offset_str] = []
            trigger_offset2arguments[trigger_offset_str].append({
                "text": rel["subject"],
                "type": argument_role,
                "char_span": rel["subj_char_span"],
                "tok_span": rel["subj_tok_span"],
            })
        event_list = []
        for trigger_offset_str, event_type in tirigger_offset2event.items():
            arguments = trigger_offset2arguments[trigger_offset_str] if trigger_offset_str in trigger_offset2arguments else [
            ]
            event = {
                "trigger": trigger_offset2trigger_text[trigger_offset_str],
                "trigger_char_span": trigger_offset2trigger_char_span[trigger_offset_str],
                "trigger_tok_span": trigger_offset_str.split(","),
                "trigger_type": event_type,
                "argument_list": arguments,
            }
            event_list.append(event)
        return event_list


class DataMaker4Bert():
    def __init__(self, tokenizer, shaking_tagger: HandshakingTaggingScheme):
        self.tokenizer = tokenizer
        self.shaking_tagger = shaking_tagger

    def get_indexed_data(self, data, max_seq_len, data_type="train"):
        """对所有数据转换

        Args:
            data (_type_): _description_
            max_seq_len (_type_): _description_
            data_type (str, optional): _description_. Defaults to "train".

        Returns:
            _type_: _description_
        """
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc="Generate indexed train or valid data"):
            text = sample["text"]
            # codes for bert input
            codes = self.tokenizer.encode_plus(text,
                                               return_offsets_mapping=True,
                                               add_special_tokens=False,
                                               max_length=max_seq_len,
                                               truncation=True,
                                               pad_to_max_length=True)

            # tagging
            matrix_spots = None
            if data_type != "test":
                # 将样本转变为tplinker的tag id
                matrix_spots = self.shaking_tagger.get_spots(sample)

            # get codes
            input_ids = torch.tensor(codes["input_ids"]).long()
            attention_mask = torch.tensor(codes["attention_mask"]).long()
            token_type_ids = torch.tensor(codes["token_type_ids"]).long()
            tok2char_span = codes["offset_mapping"]

            sample_tp = (sample,
                         input_ids,
                         attention_mask,
                         token_type_ids,
                         tok2char_span,
                         matrix_spots,
                         )
            indexed_samples.append(sample_tp)
        return indexed_samples

    def generate_batch(self, batch_data, data_type="train"):
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        tok2char_span_list = []
        matrix_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            attention_mask_list.append(tp[2])
            token_type_ids_list.append(tp[3])
            tok2char_span_list.append(tp[4])
            if data_type != "test":
                matrix_spots_list.append(tp[5])

        # @specific: indexed by bert tokenizer
        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)

        batch_shaking_tag = None
        if data_type != "test":
            batch_shaking_tag = self.shaking_tagger.spots2shaking_tag4batch(
                matrix_spots_list)

        return sample_list, \
            batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, \
            batch_shaking_tag


class DataMaker4BiLSTM():
    def __init__(self, text2indices, get_tok2char_span_map, shaking_tagger: HandshakingTaggingScheme):
        self.text2indices = text2indices
        self.shaking_tagger = shaking_tagger
        self.get_tok2char_span_map = get_tok2char_span_map

    def get_indexed_data(self, data, max_seq_len, data_type="train"):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc="Generate indexed train or valid data"):
            text = sample["text"]
            # tagging
            matrix_spots = None
            if data_type != "test":
                matrix_spots = self.shaking_tagger.get_spots(sample)
            tok2char_span = self.get_tok2char_span_map(text)
            tok2char_span.extend(
                [(-1, -1)] * (max_seq_len - len(tok2char_span)))
            input_ids = self.text2indices(text, max_seq_len)

            sample_tp = (sample,
                         input_ids,
                         tok2char_span,
                         matrix_spots,
                         )
            indexed_samples.append(sample_tp)
        return indexed_samples

    def generate_batch(self, batch_data, data_type="train"):
        sample_list = []
        input_ids_list = []
        tok2char_span_list = []
        matrix_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            tok2char_span_list.append(tp[2])
            if data_type != "test":
                matrix_spots_list.append(tp[3])

        batch_input_ids = torch.stack(input_ids_list, dim=0)

        batch_shaking_tag = None
        if data_type != "test":
            batch_shaking_tag = self.shaking_tagger.spots2shaking_tag4batch(
                matrix_spots_list)

        return sample_list, \
            batch_input_ids, tok2char_span_list, \
            batch_shaking_tag


class MetricsCalculator():
    def __init__(self, shaking_tagger:HandshakingTaggingScheme):
        self.shaking_tagger = shaking_tagger
        self.last_weights = None  # for exponential moving averaging

    def GHM(self, gradient, bins=10, beta=0.9):
        '''
        gradient_norm: gradient_norms of all examples in this batch; (batch_size, shaking_seq_len)
        '''
        avg = torch.mean(gradient)
        std = torch.std(gradient) + 1e-12
        # normalization and pass through sigmoid to 0 ~ 1.
        gradient_norm = torch.sigmoid((gradient - avg) / std)

        min_, max_ = torch.min(gradient_norm), torch.max(gradient_norm)
        gradient_norm = (gradient_norm - min_) / (max_ - min_)
        # ensure elements in gradient_norm != 1.
        gradient_norm = torch.clamp(gradient_norm, 0, 0.9999999)

        example_sum = torch.flatten(gradient_norm).size()[0]  # N

        # calculate weights
        current_weights = torch.zeros(bins).to(gradient.device)
        hits_vec = torch.zeros(bins).to(gradient.device)
        count_hits = 0  # coungradient_normof hits
        for i in range(bins):
            bar = float((i + 1) / bins)
            hits = torch.sum((gradient_norm <= bar)) - count_hits
            count_hits += hits
            hits_vec[i] = hits.item()
            current_weights[i] = example_sum / bins / \
                (hits.item() + example_sum / bins)
        # EMA: exponential moving averaging
        # print()
        # print("hits_vec: {}".format(hits_vec))
        # print("current_weights: {}".format(current_weights))
        if self.last_weights is None:
            self.last_weights = torch.ones(bins).to(
                gradient.device)  # init by ones
        current_weights = self.last_weights * \
            beta + (1 - beta) * current_weights
        self.last_weights = current_weights
        # print("ema current_weights: {}".format(current_weights))

        # weights4examples: pick weights for all examples
        weight_pk_idx = (gradient_norm / (1 / bins)).long()[:, :, None]
        weights_rp = current_weights[None, None, :].repeat(
            gradient_norm.size()[0], gradient_norm.size()[1], 1)
        weights4examples = torch.gather(
            weights_rp, -1, weight_pk_idx).squeeze(-1)
        weights4examples /= torch.sum(weights4examples)
        return weights4examples * gradient  # return weighted gradients

    # loss func
    def _multilabel_categorical_crossentropy(self, y_pred, y_true, ghm=True):
        """
        y_pred: (batch_size, shaking_seq_len, type_size)
        y_true: (batch_size, shaking_seq_len, type_size)
        y_true and y_pred have the same shape，elements in y_true are either 0 or 1，
             1 tags positive classes，0 tags negtive classes(means tok-pair does not have this type of link).
        """
        y_pred = (1 - 2 * y_true) * \
            y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred oudtuts of pos classes
        # mask the pred oudtuts of neg classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])  # st - st
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        if ghm:
            return (self.GHM(neg_loss + pos_loss, bins=1000)).sum()
        else:
            return (neg_loss + pos_loss).mean()

    def loss_func(self, y_pred, y_true, ghm):
        return self._multilabel_categorical_crossentropy(y_pred, y_true, ghm=ghm)

    def get_sample_accuracy(self, pred, truth):
        '''
        计算该batch的pred与truth全等的样本比例
        '''
        # (batch_size, ..., seq_len, tag_size) -> (batch_size, ..., seq_len)
        # pred = torch.argmax(pred, dim = -1)
        # (batch_size, ..., seq_len) -> (batch_size, seq_len)
        pred = pred.view(pred.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred).float(), dim=1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(
            correct_tag_num) * truth.size()[-1]).float()
        sample_acc = torch.mean(sample_acc_)

        return sample_acc

    def get_mark_sets_event(self, event_list):
        trigger_iden_set, trigger_class_set, arg_iden_set, arg_class_set = set(), set(), set(), set()
        for event in event_list:
            event_type = event["trigger_type"]
            trigger_offset = event["trigger_tok_span"]
            trigger_iden_set.add("{}\u2E80{}".format(
                trigger_offset[0], trigger_offset[1]))
            trigger_class_set.add("{}\u2E80{}\u2E80{}".format(
                event_type, trigger_offset[0], trigger_offset[1]))
            for arg in event["argument_list"]:
                argument_offset = arg["tok_span"]
                argument_role = arg["type"]
                arg_iden_set.add("{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(
                    event_type, trigger_offset[0], trigger_offset[1], argument_offset[0], argument_offset[1]))
                arg_class_set.add("{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(
                    event_type, trigger_offset[0], trigger_offset[1], argument_offset[0], argument_offset[1], argument_role))

        return trigger_iden_set, trigger_class_set, arg_iden_set, arg_class_set

    # def get_mark_sets_rel(self, pred_rel_list, gold_rel_list, pred_ent_list, gold_ent_list, pattern = "only_head_text", gold_event_list = None):
    #     if pattern == "only_head_index":
    #         gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in gold_rel_list])
    #         pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in pred_rel_list])
    #         gold_ent_set = set(["{}\u2E80{}".format(ent["tok_span"][0], ent["type"]) for ent in gold_ent_list])
    #         pred_ent_set = set(["{}\u2E80{}".format(ent["tok_span"][0], ent["type"]) for ent in pred_ent_list])

    #     elif pattern == "whole_span":
    #         gold_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in gold_rel_list])
    #         pred_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in pred_rel_list])
    #         gold_ent_set = set(["{}\u2E80{}\u2E80{}".format(ent["tok_span"][0], ent["tok_span"][1], ent["type"]) for ent in gold_ent_list])
    #         pred_ent_set = set(["{}\u2E80{}\u2E80{}".format(ent["tok_span"][0], ent["tok_span"][1], ent["type"]) for ent in pred_ent_list])

    #     elif pattern == "whole_text":
    #         gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in gold_rel_list])
    #         pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in pred_rel_list])
    #         gold_ent_set = set(["{}\u2E80{}".format(ent["text"], ent["type"]) for ent in gold_ent_list])
    #         pred_ent_set = set(["{}\u2E80{}".format(ent["text"], ent["type"]) for ent in pred_ent_list])

    #     elif pattern == "only_head_text":
    #         gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in gold_rel_list])
    #         pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in pred_rel_list])
    #         gold_ent_set = set(["{}\u2E80{}".format(ent["text"].split(" ")[0], ent["type"]) for ent in gold_ent_list])
    #         pred_ent_set = set(["{}\u2E80{}".format(ent["text"].split(" ")[0], ent["type"]) for ent in pred_ent_list])

    #     return pred_rel_set, gold_rel_set, pred_ent_set, gold_ent_set

    def get_mark_sets_rel(self, rel_list, ent_list, pattern="only_head_text"):
        if pattern == "only_head_index":
            rel_set = set(["{}\u2E80{}\u2E80{}".format(
                rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in rel_list])
            ent_set = set(["{}\u2E80{}".format(
                ent["tok_span"][0], ent["type"]) for ent in ent_list])

        elif pattern == "whole_span":
            rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"]
                          [1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in rel_list])
            ent_set = set(["{}\u2E80{}\u2E80{}".format(
                ent["tok_span"][0], ent["tok_span"][1], ent["type"]) for ent in ent_list])

        elif pattern == "whole_text":
            rel_set = set(["{}\u2E80{}\u2E80{}".format(
                rel["subject"], rel["predicate"], rel["object"]) for rel in rel_list])
            ent_set = set(["{}\u2E80{}".format(ent["text"], ent["type"])
                          for ent in ent_list])

        elif pattern == "only_head_text":
            rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(
                " ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in rel_list])
            ent_set = set(["{}\u2E80{}".format(ent["text"].split(
                " ")[0], ent["type"]) for ent in ent_list])

        return rel_set, ent_set

    def _cal_cpg(self, pred_set, gold_set, cpg):
        '''
        cpg is a list: [correct_num, pred_num, gold_num]
        '''
        for mark_str in pred_set:
            if mark_str in gold_set:
                cpg[0] += 1
        cpg[1] += len(pred_set)
        cpg[2] += len(gold_set)

    def cal_rel_cpg(self, pred_rel_list, pred_ent_list, gold_rel_list, gold_ent_list, ere_cpg_dict, pattern):
        '''
        ere_cpg_dict = {
                "rel_cpg": [0, 0, 0],
                "ent_cpg": [0, 0, 0],
                }
        pattern: metric pattern
        '''
        gold_rel_set, gold_ent_set = self.get_mark_sets_rel(
            gold_rel_list, gold_ent_list, pattern)
        pred_rel_set, pred_ent_set = self.get_mark_sets_rel(
            pred_rel_list, pred_ent_list, pattern)

        self._cal_cpg(pred_rel_set, gold_rel_set, ere_cpg_dict["rel_cpg"])
        self._cal_cpg(pred_ent_set, gold_ent_set, ere_cpg_dict["ent_cpg"])

    def cal_event_cpg(self, pred_event_list, gold_event_list, ee_cpg_dict):
        '''
        ee_cpg_dict = {
            "trigger_iden_cpg": [0, 0, 0],
            "trigger_class_cpg": [0, 0, 0],
            "arg_iden_cpg": [0, 0, 0],
            "arg_class_cpg": [0, 0, 0],
        }
        '''
        pred_trigger_iden_set, pred_trigger_class_set, pred_arg_iden_set, pred_arg_class_set = self.get_mark_sets_event(
            pred_event_list)

        gold_trigger_iden_set, gold_trigger_class_set, gold_arg_iden_set, gold_arg_class_set = self.get_mark_sets_event(
            gold_event_list)

        self._cal_cpg(pred_trigger_iden_set, gold_trigger_iden_set,
                      ee_cpg_dict["trigger_iden_cpg"])
        self._cal_cpg(pred_trigger_class_set, gold_trigger_class_set,
                      ee_cpg_dict["trigger_class_cpg"])
        self._cal_cpg(pred_arg_iden_set, gold_arg_iden_set,
                      ee_cpg_dict["arg_iden_cpg"])
        self._cal_cpg(pred_arg_class_set, gold_arg_class_set,
                      ee_cpg_dict["arg_class_cpg"])

    def get_cpg(self, sample_list,
                tok2char_span_list,
                batch_pred_shaking_tag,
                pattern="only_head_text"):
        '''
        return correct number, predict number, gold number (cpg)
        '''

        ee_cpg_dict = {
            "trigger_iden_cpg": [0, 0, 0],
            "trigger_class_cpg": [0, 0, 0],
            "arg_iden_cpg": [0, 0, 0],
            "arg_class_cpg": [0, 0, 0],
        }
        ere_cpg_dict = {
            "rel_cpg": [0, 0, 0],
            "ent_cpg": [0, 0, 0],
        }

        # go through all sentences
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            text = sample["text"]
            tok2char_span = tok2char_span_list[ind]
            pred_shaking_tag = batch_pred_shaking_tag[ind]

            pred_rel_list, pred_ent_list = self.shaking_tagger.decode_rel(text,
                                                                          pred_shaking_tag,
                                                                          tok2char_span)  # decoding
            gold_rel_list = sample["relation_list"]
            gold_ent_list = sample["entity_list"]

            if pattern == "event_extraction":
                pred_event_list = self.shaking_tagger.trans2ee(
                    pred_rel_list, pred_ent_list)  # transform to event list
                gold_event_list = sample["event_list"]
                self.cal_event_cpg(
                    pred_event_list, gold_event_list, ee_cpg_dict)
            else:
                self.cal_rel_cpg(pred_rel_list, pred_ent_list,
                                 gold_rel_list, gold_ent_list, ere_cpg_dict, pattern)

        if pattern == "event_extraction":
            return ee_cpg_dict
        else:
            return ere_cpg_dict

    def get_prf_scores(self, correct_num, pred_num, gold_num):
        minimini = 1e-12
        precision = correct_num / (pred_num + minimini)
        recall = correct_num / (gold_num + minimini)
        f1 = 2 * precision * recall / (precision + recall + minimini)
        return precision, recall, f1


class TplinkerDataProcess():
    def __init__(self,args,filename,get_tok2char_span_map,is_training) -> None:
        self._get_tok2char_span_map = get_tok2char_span_map
        self.args = args
        with open(filename, 'r') as f:
            lines = json.load(f)
        if is_training:
            self.data_type = "train"
        else:
            self.data_type = "val"
        self.preprocess(lines)
        
    def preprocess(self, data):
        """预处理文本
        Args:
            lines (_type_): _description_
        """
        error_statistics = {}
        data = self.transform_data(
            data, dataset_type=self.data_type, add_id=True)
        data = self.clean_data_wo_span(
            data, separate=self.args.separate_char_by_white)
        if self.args.add_char_span:
            data, miss_sample_list = self.add_char_span(
                data, self.args.ignore_subword)
            error_statistics["miss_samples"] = len(miss_sample_list)
        rel_set = set()
        ent_set = set()

        for sample in tqdm(data, desc="building relation type set and entity type set"):
            if "entity_list" not in sample:  # if "entity_list" not in sample, generate entity list with default type
                ent_list = []
                for rel in sample["relation_list"]:
                    ent_list.append({
                        "text": rel["subject"],
                        "type": "DEFAULT",
                        "char_span": rel["subj_char_span"],
                    })
                    ent_list.append({
                        "text": rel["object"],
                        "type": "DEFAULT",
                        "char_span": rel["obj_char_span"],
                    })
                sample["entity_list"] = ent_list

            for ent in sample["entity_list"]:
                ent_set.add(ent["type"])

            for rel in sample["relation_list"]:
                rel_set.add(rel["predicate"])

        data = self.add_tok_span(data)
        if self.args.check_tok_span:
            span_error_memory = self.check_tok_span(data)
            error_statistics["tok_span_error"] = len(span_error_memory)

        rel_set = sorted(rel_set)
        rel2id = {rel: ind for ind, rel in enumerate(rel_set)}

        ent_set = sorted(ent_set)
        ent2id = {ent: ind for ind, ent in enumerate(ent_set)}

        data_path = os.path.join(
            self.args.data_out_dir, "{}.json".format(self.data_type))
        json.dump(data, open(data_path, "w", encoding="utf-8"),
                  ensure_ascii=False)

        rel2id_path = os.path.join(self.args.data_out_dir, "rel2id.json")
        if not os.path.exists(rel2id_path):
            json.dump(rel2id, open(rel2id_path, "w",
                      encoding="utf-8"), ensure_ascii=False)

        if not os.path.exists(self.args.ent2id_path):
            json.dump(ent2id, open(self.args.ent2id_path, "w",
                      encoding="utf-8"), ensure_ascii=False)
    
    def add_char_span(self, dataset, ignore_subword_match=True):
        miss_sample_list = []
        for sample in tqdm(dataset, desc="adding char level spans"):
            entities = [rel["subject"] for rel in sample["relation_list"]]
            entities.extend([rel["object"] for rel in sample["relation_list"]])
            if "entity_list" in sample:
                entities.extend([ent["text"] for ent in sample["entity_list"]])
            ent2char_spans = self._get_ent2char_spans(
                sample["text"], entities, ignore_subword_match=ignore_subword_match)

            new_relation_list = []
            for rel in sample["relation_list"]:
                subj_char_spans = ent2char_spans[rel["subject"]]
                obj_char_spans = ent2char_spans[rel["object"]]
                for subj_sp in subj_char_spans:
                    for obj_sp in obj_char_spans:
                        new_relation_list.append({
                            "subject": rel["subject"],
                            "object": rel["object"],
                            "subj_char_span": subj_sp,
                            "obj_char_span": obj_sp,
                            "predicate": rel["predicate"],
                        })

            if len(sample["relation_list"]) > len(new_relation_list):
                miss_sample_list.append(sample)
            sample["relation_list"] = new_relation_list

            if "entity_list" in sample:
                new_ent_list = []
                for ent in sample["entity_list"]:
                    for char_sp in ent2char_spans[ent["text"]]:
                        new_ent_list.append({
                            "text": ent["text"],
                            "type": ent["type"],
                            "char_span": char_sp,
                        })
                sample["entity_list"] = new_ent_list
        return dataset, miss_sample_list

    # check token level span
    def check_tok_span(self, data):
        def extr_ent(text, tok_span, tok2char_span):
            char_span_list = tok2char_span[tok_span[0]:tok_span[1]]
            char_span = (char_span_list[0][0], char_span_list[-1][1])
            decoded_ent = text[char_span[0]:char_span[1]]
            return decoded_ent

        span_error_memory = set()
        for sample in tqdm(data, desc="check tok spans"):
            text = sample["text"]
            tok2char_span = self._get_tok2char_span_map(text)
            for ent in sample["entity_list"]:
                tok_span = ent["tok_span"]
                if extr_ent(text, tok_span, tok2char_span) != ent["text"]:
                    span_error_memory.add(
                        "extr ent: {}---gold ent: {}".format(extr_ent(text, tok_span, tok2char_span), ent["text"]))

            for rel in sample["relation_list"]:
                subj_tok_span, obj_tok_span = rel["subj_tok_span"], rel["obj_tok_span"]
                if extr_ent(text, subj_tok_span, tok2char_span) != rel["subject"]:
                    span_error_memory.add(
                        "extr: {}---gold: {}".format(extr_ent(text, subj_tok_span, tok2char_span), rel["subject"]))
                if extr_ent(text, obj_tok_span, tok2char_span) != rel["object"]:
                    span_error_memory.add(
                        "extr: {}---gold: {}".format(extr_ent(text, obj_tok_span, tok2char_span), rel["object"]))

        return span_error_memory


    def _get_ent2char_spans(self, text, entities, ignore_subword_match=True):
        '''
        获得实体的span 索引
        if ignore_subword_match is true, find entities with whitespace around, e.g. "entity" -> " entity "
        '''
        entities = sorted(entities, key=lambda x: len(x), reverse=True)
        text_cp = " {} ".format(text) if ignore_subword_match else text
        ent2char_spans = {}
        for ent in entities:
            spans = []
            target_ent = " {} ".format(ent) if ignore_subword_match else ent
            for m in re.finditer(re.escape(target_ent), text_cp):
                # avoid matching a inner number of a number
                if not ignore_subword_match and re.match("\d+", target_ent):
                    if (m.span()[0] - 1 >= 0 and re.match("\d", text_cp[m.span()[0] - 1])) or (m.span()[1] < len(text_cp) and re.match("\d", text_cp[m.span()[1]])):
                        continue
                span = [m.span()[0], m.span()[1] -
                        2] if ignore_subword_match else m.span()
                spans.append(span)
            # if len(spans) == 0:
            #     set_trace()
            ent2char_spans[ent] = spans
        return ent2char_spans


    def transform_data(self, data, dataset_type, add_id=True):
        '''
        转换数据格式，并清洗数据
        '''
        normal_sample_list = []
        for ind, sample in tqdm(enumerate(data), desc="Transforming data format"):
            text = sample["text"]
            rel_list = sample["triple_list"]
            subj_key, pred_key, obj_key = 0, 1, 2

            normal_sample = {
                "text": text,
            }
            if add_id:
                normal_sample["id"] = "{}_{}".format(dataset_type, ind)
            normal_rel_list = []
            for rel in rel_list:
                normal_rel = {
                    "subject": rel[subj_key],
                    "predicate": rel[pred_key],
                    "object": rel[obj_key],
                }
                normal_rel_list.append(normal_rel)
            normal_sample["relation_list"] = normal_rel_list
            normal_sample_list.append(normal_sample)

        return self._clean_sp_char(normal_sample_list)

    def _clean_sp_char(self, dataset):
        """清洗文本

        Args:
            dataset (_type_): _description_
        """
        def clean_text(text):
            text = re.sub("�", "", text)
            # text = re.sub("([A-Za-z]+)", r" \1 ", text)
            # text = re.sub("(\d+)", r" \1 ", text)
            # text = re.sub("\s+", " ", text).strip()
            return text
        for sample in tqdm(dataset, desc="Clean"):
            sample["text"] = clean_text(sample["text"])
            for rel in sample["relation_list"]:
                rel["subject"] = clean_text(rel["subject"])
                rel["object"] = clean_text(rel["object"])
        return dataset

    def clean_data_wo_span(self, ori_data, separate=False, data_type="train"):
        '''
        删除重复空格，并对非 A-Za-z0-9 等特殊字符，用空格分开
        '''
        def clean_text(text):
            text = re.sub("\s+", " ", text).strip()
            if separate:
                text = re.sub("([^A-Za-z0-9])", r" \1 ", text)
                text = re.sub("\s+", " ", text).strip()
            return text

        for sample in tqdm(ori_data, desc="clean data"):
            sample["text"] = clean_text(sample["text"])
            if data_type == "test":
                continue
            for rel in sample["relation_list"]:
                rel["subject"] = clean_text(rel["subject"])
                rel["object"] = clean_text(rel["object"])
        return ori_data

    def clean_data_w_span(self, ori_data):
        '''
        stripe whitespaces and change spans
        add a stake to bad samples(char span error) and remove them from the clean data
        '''
        bad_samples, clean_data = [], []

        def strip_white(entity, entity_char_span):
            p = 0
            while entity[p] == " ":
                entity_char_span[0] += 1
                p += 1

            p = len(entity) - 1
            while entity[p] == " ":
                entity_char_span[1] -= 1
                p -= 1
            return entity.strip(), entity_char_span

        for sample in tqdm(ori_data, desc="clean data w char spans"):
            text = sample["text"]

            bad = False
            for rel in sample["relation_list"]:
                # rm whitespaces
                rel["subject"], rel["subj_char_span"] = strip_white(
                    rel["subject"], rel["subj_char_span"])
                rel["object"], rel["obj_char_span"] = strip_white(
                    rel["object"], rel["obj_char_span"])

                subj_char_span = rel["subj_char_span"]
                obj_char_span = rel["obj_char_span"]
                if rel["subject"] not in text or rel["subject"] != text[subj_char_span[0]:subj_char_span[1]] or \
                        rel["object"] not in text or rel["object"] != text[obj_char_span[0]:obj_char_span[1]]:
                    rel["stake"] = 0
                    bad = True

            if bad:
                bad_samples.append(copy.deepcopy(sample))

            new_rel_list = [
                rel for rel in sample["relation_list"] if "stake" not in rel]
            if len(new_rel_list) > 0:
                sample["relation_list"] = new_rel_list
                clean_data.append(sample)
        return clean_data, bad_samples



    def add_tok_span(self, dataset):
        '''
        dataset must has char level span
        '''
        def char_span2tok_span(char_span, char2tok_span):
            tok_span_list = char2tok_span[char_span[0]:char_span[1]]
            tok_span = [tok_span_list[0][0], tok_span_list[-1][1]]
            return tok_span

        for sample in tqdm(dataset, desc="adding token level spans"):
            text = sample["text"]
            char2tok_span = self._get_char2tok_span(sample["text"])
            for rel in sample["relation_list"]:
                subj_char_span = rel["subj_char_span"]
                obj_char_span = rel["obj_char_span"]
                rel["subj_tok_span"] = char_span2tok_span(
                    subj_char_span, char2tok_span)
                rel["obj_tok_span"] = char_span2tok_span(
                    obj_char_span, char2tok_span)
            for ent in sample["entity_list"]:
                char_span = ent["char_span"]
                ent["tok_span"] = char_span2tok_span(char_span, char2tok_span)
            if "event_list" in sample:
                for event in sample["event_list"]:
                    event["trigger_tok_span"] = char_span2tok_span(
                        event["trigger_char_span"], char2tok_span)
                    for arg in event["argument_list"]:
                        arg["tok_span"] = char_span2tok_span(
                            arg["char_span"], char2tok_span)
        return dataset

    def get_tok2char_span_map_(self, text):
        tokens = text.split(" ")
        tok2char_span = []
        char_num = 0
        for tok in tokens:
            tok2char_span.append((char_num, char_num + len(tok)))
            char_num += len(tok) + 1  # +1: whitespace
        return tok2char_span


    def _get_char2tok_span(self, text):
        '''
        map character index to token level span
        '''
        tok2char_span = self._get_tok2char_span_map(text)
        char_num = None
        for tok_ind in range(len(tok2char_span) - 1, -1, -1):
            if tok2char_span[tok_ind][1] != 0:
                char_num = tok2char_span[tok_ind][1]
                break
        char2tok_span = [[-1, -1]
                         for _ in range(char_num)]  # [-1, -1] is whitespace
        for tok_ind, char_sp in enumerate(tok2char_span):
            for char_ind in range(char_sp[0], char_sp[1]):
                tok_sp = char2tok_span[char_ind]
                # 因为char to tok 也可能出现1对多的情况，比如韩文。所以char_span的pos1以第一个tok_ind为准，pos2以最后一个tok_ind为准
                if tok_sp[0] == -1:
                    tok_sp[0] = tok_ind
                tok_sp[1] = tok_ind + 1
        return char2tok_span

