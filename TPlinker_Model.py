import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel,BertModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention
from torch.utils.data import DataLoader,Dataset
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
import numpy as np
from collections import defaultdict
import json
from transformers import DataCollatorWithPadding
import argparse
import pytorch_lightning as pl
import copy
from tqdm import tqdm
import math
from torch.nn import Parameter

class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim = 0, center = True, scale = True, epsilon = None, conditional = False,
                 hidden_units = None, hidden_activation = 'linear', hidden_initializer = 'xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        # self.hidden_activation = activations.get(hidden_activation) keras中activation为linear时，返回原tensor,unchanged
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features = self.cond_dim, out_features = self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features = self.cond_dim, out_features = input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features = self.cond_dim, out_features = input_dim, bias=False)

        self.initialize_weights()


    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            # 下面这两个为什么都初始化为0呢? 
            # 为了防止扰乱原来的预训练权重，两个变换矩阵可以全零初始化（单层神经网络可以用全零初始化，连续的多层神经网络才不应当用全零初始化），这样在初始状态，模型依然保持跟原来的预训练模型一致。
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)


    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            # for _ in range(K.ndim(inputs) - K.ndim(cond)): # K.ndim: 以整数形式返回张量中的轴数。
            # TODO: 这两个为什么有轴数差呢？ 为什么在 dim=1 上增加维度??
            # 为了保持维度一致，cond可以是（batch_size, cond_dim）
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)
            
            # cond在加入beta和gamma之前做一次线性变换，以保证与input维度一致
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs**2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) **2
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs
    
class HandshakingKernel(nn.Module):
    def __init__(self, hidden_size, shaking_type, inner_enc_type):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == "cln":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
        elif shaking_type == "cln_plus":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
            self.inner_context_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
            
        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = Parameter(torch.rand(hidden_size))
        elif inner_enc_type == "lstm":
            self.inner_context_lstm = nn.LSTM(hidden_size, 
                           hidden_size, 
                           num_layers = 1, 
                           bidirectional = False, 
                           batch_first = True)
     
    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type = "lstm"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = torch.mean(seqence, dim = -2)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim = -2)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * torch.mean(seqence, dim = -2) + (1 - self.lamtha) * torch.max(seqence, dim = -2)[0]
            return pooling
        if "pooling" in inner_enc_type:
            inner_context = torch.stack([pool(seq_hiddens[:, :i+1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])], dim = 1)
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)
            
        return inner_context
    
    def forward(self, seq_hiddens):
        '''
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        '''
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            visible_hiddens = seq_hiddens[:, ind:, :] # ind: only look back
            repeat_hiddens = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1)  
            
            if self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim = -1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim = -1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cln":
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
            elif self.shaking_type == "cln_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
                shaking_hiddens = self.inner_context_cln(shaking_hiddens, inner_context)

            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim = 1)
        return long_shaking_hiddens


class HandshakingTaggingScheme(object):
    def __init__(self, rel2id, max_seq_len, entity_type2id):
        super().__init__()
        self.rel2id = rel2id
        self.id2rel = {ind:rel for rel, ind in rel2id.items()}
 
        self.separator = "\u2E80"
        self.link_types = {"SH2OH", # subject head to object head
                     "OH2SH", # object head to subject head
                     "ST2OT", # subject tail to object tail
                     "OT2ST", # object tail to subject tail
                     }
        self.tags = {self.separator.join([rel, lt]) for rel in self.rel2id.keys() for lt in self.link_types}
        
        self.ent2id = entity_type2id
        self.id2ent = {ind:ent for ent, ind in self.ent2id.items()}
        self.tags |= {self.separator.join([ent, "EH2ET"]) for ent in self.ent2id.keys()} # EH2ET: entity head to entity tail

        self.tags = sorted(self.tags)
        
        self.tag2id = {t:idx for idx, t in enumerate(self.tags)}
        self.id2tag = {idx:t for t, idx in self.tag2id.items()}
        self.matrix_size = max_seq_len
        
        # map
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_idx2matrix_idx = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in list(range(self.matrix_size))[ind:]]

        self.matrix_idx2shaking_idx = [[0 for i in range(self.matrix_size)] for j in range(self.matrix_size)]
        for shaking_ind, matrix_ind in enumerate(self.shaking_idx2matrix_idx):
            self.matrix_idx2shaking_idx[matrix_ind[0]][matrix_ind[1]] = shaking_ind
    
    def get_tag_size(self):
        return len(self.tag2id)
    
    def get_spots(self, sample):
        '''
        matrix_spots: [(tok_pos1, tok_pos2, tag_id), ]
        '''
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
            add_spot((ent["tok_span"][0], ent["tok_span"][1] - 1, self.tag2id[self.separator.join([ent["type"], "EH2ET"])]))
            
        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            rel = rel["predicate"]
            # if self.ent2id is None: # set all entities to default type
            #     add_spot((subj_tok_span[0], subj_tok_span[1] - 1, self.tag2id[self.separator.join(["DEFAULT", "EH2ET"])]))
            #     add_spot((obj_tok_span[0], obj_tok_span[1] - 1, self.tag2id[self.separator.join(["DEFAULT", "EH2ET"])]))
            if subj_tok_span[0] <= obj_tok_span[0]:
                add_spot((subj_tok_span[0], obj_tok_span[0], self.tag2id[self.separator.join([rel, "SH2OH"])]))
            else:
                add_spot((obj_tok_span[0], subj_tok_span[0], self.tag2id[self.separator.join([rel, "OH2SH"])]))
            if subj_tok_span[1] <= obj_tok_span[1]:
                add_spot((subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "ST2OT"])]))
            else:
                add_spot((obj_tok_span[1] - 1, subj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "OT2ST"])]))
        return matrix_spots

    def spots2shaking_tag(self, spots):
        '''
        convert spots to matrix tag
        spots: [(start_ind, end_ind, tag_id), ]
        return: 
            shaking_tag: (shaking_seq_len, tag_size)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_tag = torch.zeros(shaking_seq_len, len(self.tag2id)).long()
        for sp in spots:
            shaking_idx = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
            shaking_tag[shaking_idx][sp[2]] = 1
        return shaking_tag

    def spots2shaking_tag4batch(self, batch_spots):
        '''
        batch_spots: a batch of spots, [spots1, spots2, ...]
            spots: [(start_ind, end_ind, tag_id), ]
        return: 
            batch_shaking_tag: (batch_size, shaking_seq_len, tag_size)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_tag = torch.zeros(len(batch_spots), shaking_seq_len, len(self.tag2id)).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_idx = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
                batch_shaking_tag[batch_id][shaking_idx][sp[2]] = 1
        return batch_shaking_tag
        
    def get_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, tag_id)
        spots: [(start_ind, end_ind, tag_id), ]
        '''
        spots = []
        nonzero_points = torch.nonzero(shaking_tag, as_tuple = False)
        for point in nonzero_points:
            shaking_idx, tag_idx = point[0].item(), point[1].item()
            pos1, pos2 = self.shaking_idx2matrix_idx[shaking_idx]
            spot = (pos1, pos2, tag_idx)
            spots.append(spot)
        return spots

    def decode_rel(self,
              text, 
              shaking_tag,
              tok2char_span, 
              tok_offset = 0, char_offset = 0):
        '''
        shaking_tag: (shaking_seq_len, tag_id_num)
        '''
        rel_list = []
        matrix_spots = self.get_spots_fr_shaking_tag(shaking_tag)
        
        # entity
        head_ind2entities = {}
        ent_list = []
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            ent_type, link_type = tag.split(self.separator) 
            if link_type != "EH2ET" or sp[0] > sp[1]: # for an entity, the start position can not be larger than the end pos.
                continue
            
            char_span_list = tok2char_span[sp[0]:sp[1] + 1]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            ent_text = text[char_sp[0]:char_sp[1]] 
            entity = {
                "type": ent_type,
                "text": ent_text,
                "tok_span": [sp[0], sp[1] + 1],
                "char_span": char_sp,
            }
            head_key = str(sp[0]) # take ent_head_pos as the key to entity list
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            head_ind2entities[head_key].append(entity)
            ent_list.append(entity)
            
        # tail link
        tail_link_memory_set = set()
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            rel, link_type = tag.split(self.separator) 
            if link_type == "ST2OT":
                tail_link_memory = self.separator.join([rel, str(sp[0]), str(sp[1])])
                tail_link_memory_set.add(tail_link_memory)
            elif link_type == "OT2ST":
                tail_link_memory = self.separator.join([rel, str(sp[1]), str(sp[0])])
                tail_link_memory_set.add(tail_link_memory)

        # head link
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
            
            subj_list = head_ind2entities[subj_head_key] # all entities start with this subject head
            obj_list = head_ind2entities[obj_head_key] # all entities start with this object head
            
            # go over all subj-obj pair to check whether the tail link exists
            for subj in subj_list:
                for obj in obj_list:
                    tail_link_memory = self.separator.join([rel, str(subj["tok_span"][1] - 1), str(obj["tok_span"][1] - 1)])
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
                ent["char_span"] = [ent["char_span"][0] + char_offset, ent["char_span"][1] + char_offset]
                ent["tok_span"] = [ent["tok_span"][0] + tok_offset, ent["tok_span"][1] + tok_offset]
                
        return rel_list, ent_list
    
    def trans2ee(self, rel_list, ent_list):
        sepatator = "_" # \u2E80
        trigger_set, arg_iden_set, arg_class_set = set(), set(), set()
        trigger_offset2vote = {}
        trigger_offset2trigger_text = {}
        trigger_offset2trigger_char_span = {}
        # get candidate trigger types from relation
        for rel in rel_list:
            trigger_offset = rel["obj_tok_span"]
            trigger_offset_str = "{},{}".format(trigger_offset[0], trigger_offset[1])
            trigger_offset2trigger_text[trigger_offset_str] = rel["object"]
            trigger_offset2trigger_char_span[trigger_offset_str] = rel["obj_char_span"]
            _, event_type = rel["predicate"].split(sepatator)

            if trigger_offset_str not in trigger_offset2vote:
                trigger_offset2vote[trigger_offset_str] = {}
            trigger_offset2vote[trigger_offset_str][event_type] = trigger_offset2vote[trigger_offset_str].get(event_type, 0) + 1
            
        # get candidate trigger types from entity types
        for ent in ent_list:
            t1, t2 = ent["type"].split(sepatator)
            assert t1 == "Trigger" or t1 == "Argument"
            if t1 == "Trigger": # trigger
                event_type = t2
                trigger_span = ent["tok_span"]
                trigger_offset_str = "{},{}".format(trigger_span[0], trigger_span[1])
                trigger_offset2trigger_text[trigger_offset_str] = ent["text"]
                trigger_offset2trigger_char_span[trigger_offset_str] = ent["char_span"]
                if trigger_offset_str not in trigger_offset2vote:
                    trigger_offset2vote[trigger_offset_str] = {}
                trigger_offset2vote[trigger_offset_str][event_type] = trigger_offset2vote[trigger_offset_str].get(event_type, 0) + 1.1 # if even, entity type makes the call

        # voting
        tirigger_offset2event = {}
        for trigger_offet_str, event_type2score in trigger_offset2vote.items():
            event_type = sorted(event_type2score.items(), key = lambda x: x[1], reverse = True)[0][0]
            tirigger_offset2event[trigger_offet_str] = event_type # final event type

        # generate event list
        trigger_offset2arguments = {}
        for rel in rel_list:
            trigger_offset = rel["obj_tok_span"]
            argument_role, event_type = rel["predicate"].split(sepatator)
            trigger_offset_str = "{},{}".format(trigger_offset[0], trigger_offset[1])
            if tirigger_offset2event[trigger_offset_str] != event_type: # filter false relations
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
            arguments = trigger_offset2arguments[trigger_offset_str] if trigger_offset_str in trigger_offset2arguments else []
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
    def __init__(self, tokenizer, shaking_tagger):
        self.tokenizer = tokenizer
        self.shaking_tagger = shaking_tagger
    
    def get_indexed_data(self, data, max_seq_len, data_type = "train"):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc = "Generate indexed train or valid data"):
            text = sample["text"]
            # codes for bert input
            codes = self.tokenizer.encode_plus(text, 
                                    return_offsets_mapping = True, 
                                    add_special_tokens = False,
                                    max_length = max_seq_len, 
                                    truncation = True,
                                    pad_to_max_length = True)


            # tagging
            matrix_spots = None
            if data_type != "test":
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

    def generate_batch(self, batch_data, data_type = "train"):
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
        batch_input_ids = torch.stack(input_ids_list, dim = 0)
        batch_attention_mask = torch.stack(attention_mask_list, dim = 0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim = 0)
        
        batch_shaking_tag = None
        if data_type != "test":
            batch_shaking_tag = self.shaking_tagger.spots2shaking_tag4batch(matrix_spots_list)

        return sample_list, \
              batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, \
                batch_shaking_tag

class DataMaker4BiLSTM():
    def __init__(self, text2indices, get_tok2char_span_map, shaking_tagger):
        self.text2indices = text2indices
        self.shaking_tagger = shaking_tagger
        self.get_tok2char_span_map = get_tok2char_span_map
        
    def get_indexed_data(self, data, max_seq_len, data_type = "train"):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc = "Generate indexed train or valid data"):
            text = sample["text"]
            # tagging
            matrix_spots = None
            if data_type != "test":
                matrix_spots = self.shaking_tagger.get_spots(sample)
            tok2char_span = self.get_tok2char_span_map(text)
            tok2char_span.extend([(-1, -1)] * (max_seq_len - len(tok2char_span)))
            input_ids = self.text2indices(text, max_seq_len)

            sample_tp = (sample, 
                     input_ids,
                     tok2char_span,
                     matrix_spots,
                    )
            indexed_samples.append(sample_tp)       
        return indexed_samples
  
    def generate_batch(self, batch_data, data_type = "train"):
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

        batch_input_ids = torch.stack(input_ids_list, dim = 0)
        
        batch_shaking_tag = None
        if data_type != "test":
            batch_shaking_tag = self.shaking_tagger.spots2shaking_tag4batch(matrix_spots_list)
        
        return sample_list, \
                batch_input_ids, tok2char_span_list, \
                batch_shaking_tag

class TPLinkerPlusBert(nn.Module):
    def __init__(self, bert_path,
                 tag_size, 
                 shaking_type, 
                 inner_enc_type,
                 tok_pair_sample_rate = 1):
        super().__init__()
        self.encoder = BertModel.from_pretrained(bert_path)
        self.tok_pair_sample_rate = tok_pair_sample_rate
        
        shaking_hidden_size = self.encoder.config.hidden_size
           
        self.fc = nn.Linear(shaking_hidden_size, tag_size)
            
        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(shaking_hidden_size, shaking_type, inner_enc_type)
        
    def forward(self, input_ids, 
                attention_mask, 
                token_type_ids
               ):
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]
        
        seq_len = last_hidden_state.size()[1]
        # shaking_hiddens: (batch_size, shaking_seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(last_hidden_state)
        
        sampled_tok_pair_indices = None
        if self.training:
            # randomly sample segments of token pairs
            shaking_seq_len = shaking_hiddens.size()[1]
            segment_len = int(shaking_seq_len * self.tok_pair_sample_rate)
            seg_num = math.ceil(shaking_seq_len // segment_len)
            start_ind = torch.randint(seg_num, []) * segment_len
            end_ind = min(start_ind + segment_len, shaking_seq_len)
            # sampled_tok_pair_indices: (batch_size, ~segment_len) ~end_ind - start_ind <= segment_len
            sampled_tok_pair_indices = torch.arange(start_ind, end_ind)[None, :].repeat(shaking_hiddens.size()[0], 1)
#             sampled_tok_pair_indices = torch.randint(shaking_seq_len, (shaking_hiddens.size()[0], segment_len))
            sampled_tok_pair_indices = sampled_tok_pair_indices.to(shaking_hiddens.device)

            # sampled_tok_pair_indices will tell model what token pairs should be fed into fcs
            # shaking_hiddens: (batch_size, ~segment_len, hidden_size)
            shaking_hiddens = shaking_hiddens.gather(1, sampled_tok_pair_indices[:,:,None].repeat(1, 1, shaking_hiddens.size()[-1]))
 
        # outputs: (batch_size, segment_len, tag_size) or (batch_size, shaking_seq_len, tag_size)
        outputs = self.fc(shaking_hiddens)

        return outputs, sampled_tok_pair_indices

class TPLinkerPlusBiLSTM(nn.Module):
    def __init__(self, init_word_embedding_matrix, 
                 emb_dropout_rate, 
                 enc_hidden_size, 
                 dec_hidden_size, 
                 rnn_dropout_rate,
                 tag_size,
                 shaking_type,
                 inner_enc_type,
                 tok_pair_sample_rate = 1
                ):
        super().__init__()
        self.word_embeds = nn.Embedding.from_pretrained(init_word_embedding_matrix, freeze = False)
        self.emb_dropout = nn.Dropout(emb_dropout_rate)
        self.enc_lstm = nn.LSTM(init_word_embedding_matrix.size()[-1], 
                        enc_hidden_size // 2, 
                        num_layers = 1, 
                        bidirectional = True, 
                        batch_first = True)
        self.dec_lstm = nn.LSTM(enc_hidden_size, 
                        dec_hidden_size // 2, 
                        num_layers = 1, 
                        bidirectional = True, 
                        batch_first = True)
        self.rnn_dropout = nn.Dropout(rnn_dropout_rate)
        self.tok_pair_sample_rate = tok_pair_sample_rate
        
        shaking_hidden_size = dec_hidden_size
            
        self.fc = nn.Linear(shaking_hidden_size, tag_size)
            
        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(shaking_hidden_size, shaking_type, inner_enc_type)
        
    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # embedding: (batch_size, seq_len, emb_dim)
        embedding = self.word_embeds(input_ids)
        embedding = self.emb_dropout(embedding)
        # lstm_outputs: (batch_size, seq_len, enc_hidden_size)
        lstm_outputs, _ = self.enc_lstm(embedding)
        lstm_outputs = self.rnn_dropout(lstm_outputs)
        # lstm_outputs: (batch_size, seq_len, dec_hidden_size)
        lstm_outputs, _ = self.dec_lstm(lstm_outputs)
        lstm_outputs = self.rnn_dropout(lstm_outputs)
        
        seq_len = lstm_outputs.size()[1]
        # shaking_hiddens: (batch_size, shaking_seq_len, dec_hidden_size)
        shaking_hiddens = self.handshaking_kernel(lstm_outputs)
           
        sampled_tok_pair_indices = None
        if self.training:
            # randomly sample segments of token pairs
            shaking_seq_len = shaking_hiddens.size()[1]
            segment_len = int(shaking_seq_len * self.tok_pair_sample_rate)
            seg_num = math.ceil(shaking_seq_len // segment_len)
            start_ind = torch.randint(seg_num, []) * segment_len
            end_ind = min(start_ind + segment_len, shaking_seq_len)
            # sampled_tok_pair_indices: (batch_size, ~segment_len) ~end_ind - start_ind <= segment_len
            sampled_tok_pair_indices = torch.arange(start_ind, end_ind)[None, :].repeat(shaking_hiddens.size()[0], 1)
#             sampled_tok_pair_indices = torch.randint(shaking_hiddens, (shaking_hiddens.size()[0], segment_len))
            sampled_tok_pair_indices = sampled_tok_pair_indices.to(shaking_hiddens.device)

            # sampled_tok_pair_indices will tell model what token pairs should be fed into fcs
            # shaking_hiddens: (batch_size, ~segment_len, hidden_size)
            shaking_hiddens = shaking_hiddens.gather(1, sampled_tok_pair_indices[:,:,None].repeat(1, 1, shaking_hiddens.size()[-1]))
 
        # outputs: (batch_size, segment_len, tag_size) or (batch_size, shaking_hiddens, tag_size)
        outputs = self.fc(shaking_hiddens)
        return outputs, sampled_tok_pair_indices
   


class TDEERPytochLighting(pl.LightningModule):
    def __init__(self,args) -> None:
        super().__init__()
        self.model = TPLinkerPlusBert(args.bert_path,args.tag_size, args.shaking_type, args.inner_enc_type,args.tok_pair_sample_rate)


