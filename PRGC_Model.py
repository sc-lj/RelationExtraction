from collections import Counter
import pytorch_lightning as pl
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import json
from transformers import BertPreTrainedModel, BertModel,BertTokenizerFast
import random
import numpy as np
from collections import defaultdict
import functools
from multiprocessing import Pool
from itertools import chain



Label2IdxSub = {"B-H": 1, "I-H": 2, "O": 0}
Label2IdxObj = {"B-T": 1, "I-T": 2, "O": 0}


class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        features_output = self.hidden2tag(features_tmp)
        return features_output


class SequenceLabelForSO(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(SequenceLabelForSO, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag_sub = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.hidden2tag_obj = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        """
        Args:
            input_features: (bs, seq_len, h)
        """
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        sub_output = self.hidden2tag_sub(features_tmp)
        obj_output = self.hidden2tag_obj(features_tmp)
        return sub_output, obj_output


class PRGC(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        self.seq_tag_size = params.seq_tag_size
        self.rel_num = params.rel_num

        # pretrain model
        self.bert = BertModel(config)
        # sequence tagging
        self.sequence_tagging_sub = MultiNonLinearClassifier(config.hidden_size * 2, self.seq_tag_size, params.drop_prob)
        self.sequence_tagging_obj = MultiNonLinearClassifier(config.hidden_size * 2, self.seq_tag_size, params.drop_prob)
        self.sequence_tagging_sum = SequenceLabelForSO(config.hidden_size, self.seq_tag_size, params.drop_prob)
        # global correspondence
        self.global_corres = MultiNonLinearClassifier(config.hidden_size * 2, 1, params.drop_prob)
        # relation judgement
        self.rel_judgement = MultiNonLinearClassifier(config.hidden_size, params.rel_num, params.drop_prob)
        self.rel_embedding = nn.Embedding(params.rel_num, config.hidden_size)

        self.init_weights()

    @staticmethod
    def masked_avgpool(sent, mask):
        mask_ = mask.masked_fill(mask == 0, -1e9).float()
        score = torch.softmax(mask_, -1)
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            seq_tags=None,
            potential_rels=None,
            corres_tags=None,
            rel_tags=None,
            ex_params=None
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            rel_tags: (bs, rel_num)
            potential_rels: (bs,), only in train stage.
            seq_tags: (bs, 2, seq_len)
            corres_tags: (bs, seq_len, seq_len)
            ex_params: experiment parameters
        """
        # get params for experiments
        corres_threshold, rel_threshold = ex_params.get('corres_threshold', 0.5), ex_params.get('rel_threshold', 0.1)
        # ablation study
        ensure_corres, ensure_rel = ex_params['ensure_corres'], ex_params['ensure_rel']
        # pre-train model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        bs, seq_len, h = sequence_output.size()

        if ensure_rel:
            # (bs, h)
            h_k_avg = self.masked_avgpool(sequence_output, attention_mask)
            # (bs, rel_num)
            rel_pred = self.rel_judgement(h_k_avg)

        corres_mask = None
        # before fuse relation representation
        if ensure_corres:
            # for every position $i$ in sequence, should concate $j$ to predict.
            sub_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (bs, s, s, h)
            obj_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (bs, s, s, h)
            # batch x seq_len x seq_len x 2*hidden
            corres_pred = torch.cat([sub_extend, obj_extend], 3)
            # (bs, seq_len, seq_len)
            corres_pred = self.global_corres(corres_pred).squeeze(-1)
            mask_tmp1 = attention_mask.unsqueeze(-1)
            mask_tmp2 = attention_mask.unsqueeze(1)
            corres_mask = mask_tmp1 * mask_tmp2

        # relation predict and data construction in inference stage
        xi, pred_rels = None, None
        if ensure_rel and seq_tags is None:
            # (bs, rel_num)
            rel_pred_onehot = torch.where(torch.sigmoid(rel_pred) > rel_threshold,
                                          torch.ones(rel_pred.size(), device=rel_pred.device),
                                          torch.zeros(rel_pred.size(), device=rel_pred.device))

            # if potential relation is null
            for idx, sample in enumerate(rel_pred_onehot):
                if 1 not in sample:
                    # (rel_num,)
                    max_index = torch.argmax(rel_pred[idx])
                    sample[max_index] = 1
                    rel_pred_onehot[idx] = sample

            # 2*(sum(x_i),)
            bs_idxs, pred_rels = torch.nonzero(rel_pred_onehot, as_tuple=True)
            # get x_i
            xi_dict = Counter(bs_idxs.tolist())
            xi = [xi_dict[idx] for idx in range(bs)]

            pos_seq_output = []
            pos_potential_rel = []
            pos_attention_mask = []
            for bs_idx, rel_idx in zip(bs_idxs, pred_rels):
                # (seq_len, h)
                pos_seq_output.append(sequence_output[bs_idx])
                pos_attention_mask.append(attention_mask[bs_idx])
                pos_potential_rel.append(rel_idx)
            # (sum(x_i), seq_len, h)
            sequence_output = torch.stack(pos_seq_output, dim=0)
            # (sum(x_i), seq_len)
            attention_mask = torch.stack(pos_attention_mask, dim=0)
            # (sum(x_i),)
            potential_rels = torch.stack(pos_potential_rel, dim=0)
        # ablation of relation judgement
        elif not ensure_rel and seq_tags is None:
            # construct test data
            sequence_output = sequence_output.repeat((1, self.rel_num, 1)).view(bs * self.rel_num, seq_len, h)
            attention_mask = attention_mask.repeat((1, self.rel_num)).view(bs * self.rel_num, seq_len)
            potential_rels = torch.arange(0, self.rel_num, device=input_ids.device).repeat(bs)

        # (bs/sum(x_i), h)
        rel_emb = self.rel_embedding(potential_rels)

        # relation embedding vector fusion
        rel_emb = rel_emb.unsqueeze(1).expand(-1, seq_len, h)
        if ex_params['emb_fusion'] == 'concat':
            # (bs/sum(x_i), seq_len, 2*h)
            decode_input = torch.cat([sequence_output, rel_emb], dim=-1)
            # (bs/sum(x_i), seq_len, tag_size)
            output_sub = self.sequence_tagging_sub(decode_input)
            output_obj = self.sequence_tagging_obj(decode_input)
        elif ex_params['emb_fusion'] == 'sum':
            # (bs/sum(x_i), seq_len, h)
            decode_input = sequence_output + rel_emb
            # (bs/sum(x_i), seq_len, tag_size)
            output_sub, output_obj = self.sequence_tagging_sum(decode_input)
        return output_sub, output_obj,corres_mask

class PRGCPytochLighting(pl.LightningModule):
    def __init__(self,args) -> None:
        super().__init__()
        self.model = PRGC(args.pretrain_path,args)
    
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
    
    def training_step(self, batches,batch_idx):
        input_ids, attention_mask, seq_tags, relations, corres_tags, rel_tags = batches
        # compute model output and loss
        output_sub, output_obj,corres_mask= self.model(input_ids, attention_mask=attention_mask, seq_tags=seq_tags,
                                                   potential_rels=relations, corres_tags=corres_tags, rel_tags=rel_tags,
                                                   ex_params=ex_params)
        # calculate loss
        attention_mask = attention_mask.view(-1)
        # sequence label loss
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_seq_sub = (loss_func(output_sub.view(-1, self.seq_tag_size),
                                    seq_tags[:, 0, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
        loss_seq_obj = (loss_func(output_obj.view(-1, self.seq_tag_size),
                                    seq_tags[:, 1, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
        loss_seq = (loss_seq_sub + loss_seq_obj) / 2
        # init
        loss_matrix, loss_rel = torch.tensor(0), torch.tensor(0)
        if ensure_corres:
            corres_pred = corres_pred.view(bs, -1)
            corres_mask = corres_mask.view(bs, -1)
            corres_tags = corres_tags.view(bs, -1)
            loss_func = nn.BCEWithLogitsLoss(reduction='none')
            loss_matrix = (loss_func(corres_pred,
                                        corres_tags.float()) * corres_mask).sum() / corres_mask.sum()

        if ensure_rel:
            loss_func = nn.BCEWithLogitsLoss(reduction='mean')
            loss_rel = loss_func(rel_pred, rel_tags.float())

        loss = loss_seq + loss_matrix + loss_rel
        
    def validation_step(self,batches,batch_idx):
        input_ids, attention_mask, seq_tags, relations, corres_tags, rel_tags = batches
        # compute model output and loss
        output_sub, output_obj,corres_mask = self.model(input_ids, attention_mask=attention_mask, seq_tags=seq_tags,
                                                   potential_rels=relations, corres_tags=corres_tags, rel_tags=rel_tags,
                                                   ex_params=ex_params)
        # (sum(x_i), seq_len)
        pred_seq_sub = torch.argmax(torch.softmax(output_sub, dim=-1), dim=-1)
        pred_seq_obj = torch.argmax(torch.softmax(output_obj, dim=-1), dim=-1)
        # (sum(x_i), 2, seq_len)
        pred_seqs = torch.cat([pred_seq_sub.unsqueeze(1), pred_seq_obj.unsqueeze(1)], dim=1)
        if ensure_corres:
            corres_pred = torch.sigmoid(corres_pred) * corres_mask
            # (bs, seq_len, seq_len)
            pred_corres_onehot = torch.where(corres_pred > corres_threshold,
                                                torch.ones(corres_pred.size(), device=corres_pred.device),
                                                torch.zeros(corres_pred.size(), device=corres_pred.device))
            return pred_seqs, pred_corres_onehot, xi, pred_rels
        return pred_seqs, xi, pred_rels
        
    def configure_optimizers(self):
        """[配置优化参数]
        """
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.8,'lr':2e-5},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.0,'lr':2e-5},
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and 'bert' not in n], 'weight_decay': 0.8,'lr':2e-4},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and 'bert' not in n], 'weight_decay': 0.0,'lr':2e-4}
                ]
    
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        # StepLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        milestones = list(range(2,50,2))
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,gamma=0.85)
        # StepLR = WarmupLR(optimizer,25000)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict



class PRGCDataset(Dataset):
    def __init__(self,args,filename,is_training):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(args.pretrain_path,cache_dir = "./bertbaseuncased")
        self.is_training = is_training
        self.batch_size = args.batch_size
        with open(args.relation,'r') as f:
            relation = json.load(f)
        self.rel2id = relation[1]
        self.rels_set = list(self.rel2id.values())
        self.relation_size = len(self.rel2id)
        self.max_sample_triples = args.max_sample_triples
        self.datas = []
        with open(filename,'r') as f:
            lines = json.load(f)
        
    
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
        max_text_len = self.args.max_seq_length
        # multi-process
        with Pool(10) as p:
            convert_func = functools.partial(self.convert, max_text_len=max_text_len, tokenizer=self.tokenizer, rel2idx=self.rel2id,
                                            ensure_rel=self.args.ensure_rel,num_negs=self.args.num_negs)
            features = p.map(func=convert_func, iterable=examples)
        return list(chain(*features))
    
    
    def convert(self,example, max_text_len, tokenizer, rel2idx,ensure_rel,num_negs):
        """convert function
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
            corres_tag = np.zeros((max_text_len, max_text_len))
            rel_tag = len(rel2idx) * [0]
            for en_pair, rel in zip(example.en_pair_list, example.re_list):
                # get sub and obj head
                sub_head, obj_head, _, _ = self._get_so_head(en_pair, tokenizer, text_tokens)
                # construct relation tag
                rel_tag[rel] = 1
                if sub_head != -1 and obj_head != -1:
                    corres_tag[sub_head][obj_head] = 1

            sub_feats = []
            # positive samples
            for rel, en_ll in example.rel2ens.items():
                # init
                tags_sub = max_text_len * [Label2IdxSub['O']]
                tags_obj = max_text_len * [Label2IdxSub['O']]
                for en in en_ll:
                    # get sub and obj head
                    sub_head, obj_head, sub, obj = self._get_so_head(en, tokenizer, text_tokens)
                    if sub_head != -1 and obj_head != -1:
                        if sub_head + len(sub) <= max_text_len:
                            tags_sub[sub_head] = Label2IdxSub['B-H']
                            tags_sub[sub_head + 1:sub_head + len(sub)] = (len(sub) - 1) * [Label2IdxSub['I-H']]
                        if obj_head + len(obj) <= max_text_len:
                            tags_obj[obj_head] = Label2IdxObj['B-T']
                            tags_obj[obj_head + 1:obj_head + len(obj)] = (len(obj) - 1) * [Label2IdxObj['I-T']]
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
                # negative samples
                neg_rels = set(rel2idx.values()).difference(set(example.re_list))
                neg_rels = random.sample(neg_rels, k=num_negs)
                for neg_rel in neg_rels:
                    # init
                    seq_tag = max_text_len * [Label2IdxSub['O']]
                    # sanity check
                    assert len(input_ids) == len(seq_tag) == len(attention_mask) == max_text_len, f'length is not equal!!'
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
                sub_head, obj_head, sub, obj = self._get_so_head(en, tokenizer, text_tokens)
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
        sub_head = find_head_idx(source=text_tokens, target=sub)
        if sub == obj:
            obj_head = find_head_idx(source=text_tokens[sub_head + len(sub):], target=obj)
            if obj_head != -1:
                obj_head += sub_head + len(sub)
            else:
                obj_head = sub_head
        else:
            obj_head = find_head_idx(source=text_tokens, target=obj)
        return sub_head, obj_head, sub, obj


class InputExample(object):
    """a single set of samples of data
    """

    def __init__(self, text, en_pair_list, re_list, rel2ens):
        self.text = text
        self.en_pair_list = en_pair_list
        self.re_list = re_list
        self.rel2ens = rel2ens


class InputFeatures(object):
    """
    Desc:
        a single set of features of data
    """

    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 seq_tag=None,
                 corres_tag=None,
                 relation=None,
                 triples=None,
                 rel_tag=None
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.seq_tag = seq_tag
        self.corres_tag = corres_tag
        self.relation = relation
        self.triples = triples
        self.rel_tag = rel_tag



def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1



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
    
    