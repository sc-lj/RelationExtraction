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
from PRGC_metric import tag_mapping_corres
from tqdm import tqdm
from utils import find_head_idx


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
        self.rel_num = params.relation_number
        self.corres_threshold, self.rel_threshold = 0.5,0.5
        self.emb_fusion = params.emb_fusion
        # pretrain model
        self.bert = BertModel(config)
        # sequence tagging
        self.sequence_tagging_sub = MultiNonLinearClassifier(config.hidden_size * 2, self.seq_tag_size, params.drop_prob)
        self.sequence_tagging_obj = MultiNonLinearClassifier(config.hidden_size * 2, self.seq_tag_size, params.drop_prob)
        self.sequence_tagging_sum = SequenceLabelForSO(config.hidden_size, self.seq_tag_size, params.drop_prob)
        # global correspondence
        self.global_corres = MultiNonLinearClassifier(config.hidden_size * 2, 1, params.drop_prob)
        # relation judgement
        self.rel_judgement = MultiNonLinearClassifier(config.hidden_size, self.rel_num, params.drop_prob)
        self.rel_embedding = nn.Embedding(self.rel_num, config.hidden_size)

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
            potential_rels=None
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
        # pre-train model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        bs, seq_len, h = sequence_output.size()

        # 预测关系
        # (bs, h)
        h_k_avg = self.masked_avgpool(sequence_output, attention_mask)
        # (bs, rel_num)
        rel_pred = self.rel_judgement(h_k_avg)

        corres_mask = None
        # before fuse relation representation
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
        if potential_rels is None:
            # 使用预测的关系获取关系的embedding信息，也可以使用gold关系获取关系的embedding信息
            # (bs, rel_num)
            rel_pred_onehot = torch.where(torch.sigmoid(rel_pred) > self.rel_threshold,
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
            # get x_i 统计每个样本的关系数量
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
        else:
            pos_attention_mask = attention_mask

        # 获取关系的embedding信息
        # (bs/sum(x_i), h)
        rel_emb = self.rel_embedding(potential_rels)

        # relation embedding vector fusion
        rel_emb = rel_emb.unsqueeze(1).expand(-1, seq_len, h)
        if self.emb_fusion == 'concat':
            # 将关系的信息与句子的token信息融合在一起
            # (bs/sum(x_i), seq_len, 2*h)
            decode_input = torch.cat([sequence_output, rel_emb], dim=-1)
            # 分别抽取关系中subject和object字段
            # (bs/sum(x_i), seq_len, tag_size)
            output_sub = self.sequence_tagging_sub(decode_input)
            output_obj = self.sequence_tagging_obj(decode_input)
        elif self.emb_fusion == 'sum':
            # (bs/sum(x_i), seq_len, h)
            decode_input = sequence_output + rel_emb
            # 采用同一个头抽取关系中subject和object字段
            # (bs/sum(x_i), seq_len, tag_size)
            output_sub, output_obj = self.sequence_tagging_sum(decode_input)
        return output_sub, output_obj,corres_mask,rel_pred,pos_attention_mask,corres_pred,xi,pred_rels


class PRGCPytochLighting(pl.LightningModule):
    def __init__(self,args) -> None:
        super().__init__()
        self.seq_tag_size = len(Label2IdxSub)
        args.seq_tag_size = self.seq_tag_size 
        self.model = PRGC.from_pretrained(args.pretrain_path,args)
        self.corres_threshold = 0.5
        self.save_hyperparameters(args)
        
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
    def training_step(self, batches,batch_idx):
        input_ids, attention_mask, seq_tags, relation, corres_tags, rel_tags = batches
        bs = input_ids.shape[0]
        # compute model output and loss
        output_sub, output_obj,corres_mask,rel_pred,pos_attention_mask,corres_pred,_,_ = self.model(input_ids, attention_mask=attention_mask,potential_rels=relation)
        # calculate loss
        pos_attention_mask = pos_attention_mask.view(-1)
        # sequence label loss
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_seq_sub = (loss_func(output_sub.view(-1, self.seq_tag_size),
                                    seq_tags[:, 0, :].reshape(-1)) * pos_attention_mask).sum() / pos_attention_mask.sum()
        loss_seq_obj = (loss_func(output_obj.view(-1, self.seq_tag_size),
                                    seq_tags[:, 1, :].reshape(-1)) * pos_attention_mask).sum() / pos_attention_mask.sum()
        loss_seq = (loss_seq_sub + loss_seq_obj) / 2
        # init
        corres_pred = corres_pred.view(bs, -1)
        corres_mask = corres_mask.view(bs, -1)
        corres_tags = corres_tags.view(bs, -1)
        loss_func = nn.BCEWithLogitsLoss(reduction='none')
        loss_matrix = (loss_func(corres_pred,
                                    corres_tags.float()) * corres_mask).sum() / corres_mask.sum()

        loss_func = nn.BCEWithLogitsLoss(reduction='mean')
        loss_rel = loss_func(rel_pred, rel_tags.float())

        loss = loss_seq + loss_matrix + loss_rel
        return loss
    
    def validation_step(self,batches,batch_idx):
        input_ids, attention_mask, triples, input_tokens = batches
        # compute model output and loss
        output_sub, output_obj,corres_mask,_,_,corres_pred,xi,pred_rels = self.model(input_ids, attention_mask=attention_mask)
        # (sum(x_i), seq_len)
        pred_seq_sub = torch.argmax(torch.softmax(output_sub, dim=-1), dim=-1)
        pred_seq_obj = torch.argmax(torch.softmax(output_obj, dim=-1), dim=-1)
        # (sum(x_i), 2, seq_len)
        pred_seqs = torch.cat([pred_seq_sub.unsqueeze(1), pred_seq_obj.unsqueeze(1)], dim=1)
        corres_pred = torch.sigmoid(corres_pred) * corres_mask
        # (bs, seq_len, seq_len)
        pred_corres = torch.where(corres_pred > self.corres_threshold,
                                            torch.ones(corres_pred.size(), device=corres_pred.device),
                                            torch.zeros(corres_pred.size(), device=corres_pred.device))
        # (bs,)
        xi = np.array(xi)
        # (sum(s_i),)
        pred_rels = pred_rels.detach().cpu().numpy()
        pred_corres = pred_corres.detach().cpu().numpy()
        pred_seqs = pred_seqs.detach().cpu().numpy()
        # decode by per batch
        xi_index = np.cumsum(xi).tolist()
        # (bs+1,)
        xi_index.insert(0, 0)
        return pred_seqs, pred_corres, xi_index, pred_rels,triples,input_tokens
    
    def validation_epoch_end(self, outputs):
        predictions = []
        ground_truths = []
        correct_num, predict_num, gold_num = 0, 0, 0
        for pred_seqs, pred_corres, xi_index, pred_rels,triples,input_tokens in outputs:
            bs = len(xi_index)-1
            for idx in range(bs):
                pre_triple = tag_mapping_corres(predict_tags=pred_seqs[xi_index[idx]:xi_index[idx + 1]],
                                                    pre_corres=pred_corres[idx],
                                                    pre_rels=pred_rels[xi_index[idx]:xi_index[idx + 1]],
                                                    label2idx_sub=Label2IdxSub,
                                                    label2idx_obj=Label2IdxObj)
                gold_triples = self.span2str(triples[idx], input_tokens[idx])
                pre_triples = self.span2str(pre_triple, input_tokens[idx])
                ground_truths.append(list(set(gold_triples)))
                predictions.append(list(set(pre_triples)))
                # counter
                correct_num += len(set(pre_triples) & set(gold_triples))
                predict_num += len(set(pre_triples))
                gold_num += len(set(gold_triples))
        metrics = self.get_metrics(correct_num, predict_num, gold_num)
        self.log("f1",metrics["f1"], prog_bar=True)
        self.log("acc",metrics["precision"], prog_bar=True)
        self.log("recall",metrics["recall"], prog_bar=True)
        self.log("gold_num",metrics["gold_num"], prog_bar=True)
        self.log("predict_num",metrics["predict_num"], prog_bar=True)
        self.log("correct_num",metrics["correct_num"], prog_bar=True)

    def get_metrics(self,correct_num, predict_num, gold_num):
        p = correct_num / predict_num if predict_num > 0 else 0
        r = correct_num / gold_num if gold_num > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return {
            'correct_num': correct_num,
            'predict_num': predict_num,
            'gold_num': gold_num,
            'precision': p,
            'recall': r,
            'f1': f1
        }

    def span2str(self,triples, tokens):
        def _concat(token_list):
            result = ''
            for idx, t in enumerate(token_list):
                if idx == 0:
                    result = t
                elif t.startswith('##'):
                    result += t.lstrip('##')
                else:
                    result += ' ' + t
            return result

        output = []
        for triple in triples:
            rel = triple[-1]
            sub_tokens = tokens[triple[0][1]:triple[0][-1]]
            obj_tokens = tokens[triple[1][1]:triple[1][-1]]
            sub = _concat(sub_tokens)
            obj = _concat(obj_tokens)
            output.append((sub, obj, rel))
        return output


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
    
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        # StepLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        milestones = list(range(2, 50, 2))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.85)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", verbose = True, patience = 6)
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=self.args.decay_steps, gamma=self.args.decay_rate)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.num_step * self.args.rewarm_epoch_num, self.args.T_mult)
        # StepLR = WarmupLR(optimizer,25000)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optim_dict



class InputExample(object):
    """a single set of samples of data
    """

    def __init__(self, text, en_pair_list, re_list, rel2ens):
        self.text = text
        self.en_pair_list = en_pair_list
        self.re_list = re_list
        self.rel2ens = rel2ens


class PRGCDataset(Dataset):
    def __init__(self,args,filename,is_training):
        super().__init__()
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained(args.pretrain_path,cache_dir = "./bertbaseuncased")
        self.is_training = is_training
        self.batch_size = args.batch_size
        with open(args.relation,'r') as f:
            relation = json.load(f)
        self.rel2id = relation[1]
        self.rels_set = list(self.rel2id.values())
        self.relation_size = len(self.rel2id)
        self.max_sample_triples = args.max_sample_triples
        with open(filename,'r') as f:
            lines = json.load(f)
        self.datas = self.preprocess(lines)
    
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
        max_text_len = self.args.max_seq_len
        # multi-process
        # with Pool(10) as p:
        #     convert_func = functools.partial(self.convert, max_text_len=max_text_len, tokenizer=self.tokenizer, rel2idx=self.rel2id,
        #                                     ensure_rel=self.args.ensure_rel,num_negs=self.args.num_negs)
        #     features = p.map(func=convert_func, iterable=examples)
        # # return list(chain(*features))
        features = []
        for example in tqdm(examples,desc="convert example"):
            feature = self.convert(example,max_text_len=max_text_len, tokenizer=self.tokenizer, rel2idx=self.rel2id,
                                            ensure_rel=self.args.ensure_rel,num_negs=self.args.num_negs)
            features.extend(feature)
        return features
    

    def convert(self,example: InputExample, max_text_len: int, tokenizer, rel2idx, ensure_rel,num_negs):
        """转换函数 for CarFaultRelation data
        Args:
            example (_type_): 一个样本示例
            max_text_len (_type_): 样本的最大长度
            tokenizer (_type_): _description_
            rel2idx (dict): 关系的索引
            ex_params (_type_): 额外的参数
        Returns:
            _type_: _description_
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
            # subject和object相关性 target
            corres_tag = np.zeros((max_text_len, max_text_len))
            rel_tag = len(rel2idx) * [0]
            for en_pair, rel in zip(example.en_pair_list, example.re_list):
                # get sub and obj head
                sub_head, obj_head, _, _ = self._get_so_head(
                    en_pair, tokenizer, text_tokens)
                # construct relation tag
                rel_tag[rel] = 1
                # 只将head 的index标记为1
                if sub_head != -1 and obj_head != -1:
                    corres_tag[sub_head][obj_head] = 1

            sub_feats = []
            # positive samples，标记subject和object的序列
            for rel, en_ll in example.rel2ens.items():
                # init
                tags_sub = max_text_len * [Label2IdxSub['O']]
                tags_obj = max_text_len * [Label2IdxSub['O']]
                for en in en_ll:
                    # get sub and obj head
                    sub_head, obj_head, sub, obj = self._get_so_head(
                        en, tokenizer, text_tokens)
                    if sub_head != -1 and obj_head != -1:
                        if sub_head + len(sub) <= max_text_len:
                            tags_sub[sub_head] = Label2IdxSub['B-H']
                            tags_sub[sub_head + 1:sub_head +
                                    len(sub)] = (len(sub) - 1) * [Label2IdxSub['I-H']]
                        if obj_head + len(obj) <= max_text_len:
                            tags_obj[obj_head] = Label2IdxObj['B-T']
                            tags_obj[obj_head + 1:obj_head +
                                    len(obj)] = (len(obj) - 1) * [Label2IdxObj['I-T']]
                # 相同关系下的所有subject和object对
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
                # negative samples, 采样一些负样本的关系数据集
                neg_rels = set(rel2idx.values()).difference(set(example.re_list))
                neg_rels = random.sample(neg_rels, k=num_negs)
                for neg_rel in neg_rels:
                    # init，针对关系的负样本，只对subject和object的序列全部置为O，其他的沿用正样本的数据
                    seq_tag = max_text_len * [Label2IdxSub['O']]
                    # sanity check
                    assert len(input_ids) == len(seq_tag) == len(
                        attention_mask) == max_text_len, f'length is not equal!!'
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
                sub_head, obj_head, sub, obj = self._get_so_head(
                    en, tokenizer, text_tokens)
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
        subj_head_idx = find_head_idx(text_tokens, sub,0)
        subj_tail_idx = subj_head_idx + len(sub) - 1
        obj_head_idx = find_head_idx(text_tokens, obj,subj_tail_idx+1)
        if obj_head_idx == -1:
            obj_head_idx = find_head_idx(text_tokens, obj,0)
        return subj_head_idx, obj_head_idx, sub, obj

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        return data


class InputFeatures(object):
    """
    Desc:
        a single set of features of data
    """
    def __init__(self,input_tokens,input_ids,attention_mask,seq_tag=None,corres_tag=None,relation=None,triples=None,rel_tag=None):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.seq_tag = seq_tag
        self.corres_tag = corres_tag
        self.relation = relation
        self.triples = triples
        self.rel_tag = rel_tag


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
    
 
 
    