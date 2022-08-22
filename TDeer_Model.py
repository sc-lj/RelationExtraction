import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention
from torch.utils.data import DataLoader, Dataset
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
import numpy as np
from collections import defaultdict
import json
from transformers import DataCollatorWithPadding
import argparse
import pytorch_lightning as pl
import copy
from EMA import FGM
from loss_func import WarmupLR, TwoStepLR, MLFocalLoss, BCEFocalLoss
import math
import os
from utils import rematch,find_head_idx

class Linear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = Linear(config.hidden_size, config.hidden_size)
        self.key = Linear(config.hidden_size, config.hidden_size)
        self.value = Linear(config.hidden_size, config.hidden_size)
        self.attention_activation = nn.ReLU()
        self.attention_epsilon = 1e10

    def forward(self, input_ids, mask):
        q = self.query(input_ids)
        k = self.key(input_ids)
        v = self.value(input_ids)

        q = self.attention_activation(q)
        k = self.attention_activation(k)
        v = self.attention_activation(v)

        e = torch.matmul(q, k.transpose(2, 1))
        e -= self.attention_epsilon * (1.0 - mask)
        a = torch.softmax(e, -1)
        v_o = torch.matmul(a, v)
        v_o += input_ids
        return v_o


class ConditionalLayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape,
                 cond_shape,
                 eps=1e-12):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(normalized_shape))
        self.weight_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.bias_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        """
        此处初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
        """
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

        nn.init.zeros_(self.weight_dense.weight)
        nn.init.zeros_(self.bias_dense.weight)

    def forward(self, inputs, cond=None):
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'
        # cond = torch.unsqueeze(cond, 1)  # (b, 1, h*2)

        weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
        bias = self.bias_dense(cond) + self.bias  # (b, 1, h)

        mean = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
        outputs = inputs - mean  # (b, s, h)

        variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)  # (b, s, 1)

        outputs = outputs / std  # (b, s, h)

        outputs = outputs*weight + bias

        return outputs


class SpatialDropout(nn.Module):
    """
    对字级别的向量进行丢弃
    """

    def __init__(self, drop_prob):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob

    @staticmethod
    def _make_noise(input):
        return input.new().resize_(input.size(0), 1, input.size(2))

    def forward(self, inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output


class TDEER(nn.Module):
    def __init__(self, pretrain_path, relation_number):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            pretrain_path, cache_dir="./bertbaseuncased")
        config = BertConfig.from_pretrained(
            pretrain_path, cache_dir="./bertbaseuncased")

        hidden_size = config.hidden_size
        relation_size = relation_number
        self.relation_embedding = nn.Embedding(relation_size, hidden_size)
        self.entity_heads_out = nn.Linear(
            hidden_size, 2)  # 预测subjects,objects的头部位置
        self.entity_tails_out = nn.Linear(
            hidden_size, 2)  # 预测subjects,objects的尾部位置
        self.rels_out = nn.Linear(hidden_size, relation_size)  # 关系预测
        self.relu = nn.ReLU6()
        self.rel_feature = nn.Linear(hidden_size, hidden_size)
        # self.attention = BertSelfAttention(config)
        self.attention = Attention(config)
        self.obj_head = nn.Linear(hidden_size, 1)
        self.words_dropout = SpatialDropout(0.1)
        self.conditionlayernormal = ConditionalLayerNorm(
            hidden_size, hidden_size)
        self.hidden_size = hidden_size

    def relModel(self, pooler_output):
        # predict relations
        # [batch,relation]
        pred_rels = self.rels_out(pooler_output)
        return pred_rels

    def entityModel(self, last_hidden_size):
        # predict entity
        # last_hidden_size = self.words_dropout(last_hidden_size)
        # [batch,seq_len,2]
        pred_entity_heads = self.entity_heads_out(last_hidden_size)
        # [batch,seq_len,2]
        pred_entity_tails = self.entity_tails_out(last_hidden_size)
        return pred_entity_heads, pred_entity_tails

    def objModel(self, relation, last_hidden_size, sub_head, sub_tail, attention_mask):
        """_summary_
        Args:
            relation (_type_): [batch_size,1] or [batch_size, rel_num]
            last_hidden_size (_type_): [batch_size,seq_len,hidden_size]
            sub_head (_type_): [batch_size,1] or [batch_size, rel_num]
            sub_tail (_type_): [batch_size,1] or [batch_size, rel_num]
        Returns:
            _type_: _description_
        """
        # last_hidden_size = self.words_dropout(last_hidden_size)
        # attention_mask = self.expand_attention_masks(attention_mask)
        attention_mask = attention_mask.unsqueeze(1)
        # [batch_size,1,hidden_size]
        rel_feature = self.relation_embedding(relation)
        # [batch_size,1,hidden_size]
        rel_feature = self.relu(self.rel_feature(rel_feature))
        # [batch_size,1,1]
        sub_head = sub_head.unsqueeze(-1)
        # [batch_size,1,hidden_size]
        sub_head = sub_head.repeat(1, 1, self.hidden_size)
        # [batch_size,1,hidden_size]
        sub_head_feature = last_hidden_size.gather(1, sub_head)
        # [batch_size,1,1]
        sub_tail = sub_tail.unsqueeze(-1)
        # [batch_size,1,hidden_size]
        sub_tail = sub_tail.repeat(1, 1, self.hidden_size)
        # [batch_size,1,hidden_size]
        sub_tail_feature = last_hidden_size.gather(1, sub_tail)
        sub_feature = (sub_head_feature+sub_tail_feature)/2
        if relation.shape[1] != 1:
            # [rel_num,1,hidden_size]
            rel_feature = rel_feature.transpose(1, 0)
            # [rel_num,1,hidden_size]
            sub_feature = sub_feature.transpose(1, 0)
        last_hidden_size = self.conditionlayernormal(
            last_hidden_size, rel_feature+sub_feature)  # [batch_size,seq_len,hidden_size
        # [batch_size,seq_len,hidden_size]
        # obj_feature = last_hidden_size+rel_feature+sub_feature
        obj_feature = last_hidden_size
        # value,*_ = self.attention(obj_feature,attention_mask)
        value = self.attention(obj_feature, attention_mask)
        # [batch_size,seq_len,1]
        pred_obj_head = self.obj_head(value)
        # [batch_size,seq_len]
        pred_obj_head = pred_obj_head.squeeze(-1)
        return pred_obj_head

    def textEncode(self, input_ids, attention_masks, token_type_ids):
        bert_output = self.bert(input_ids, attention_masks, token_type_ids)
        return bert_output

    def forward(self, input_ids, attention_masks, token_type_ids, relation=None, sub_head=None, sub_tail=None):
        """_summary_

        Args:
            input_ids (_type_): [batch_size,seq_len]
            attention_masks (_type_): [batch_size,seq_len]
            token_type_ids (_type_): [batch_size,seq_len]
            relation (_type_, optional): [batch_size,1]. Defaults to None. subject 对应的关系(可以是正样本,也可也是负样本关系)
            sub_head (_type_, optional): [batch_size,1]. Defaults to None. subject 的head. 主要是为了预测object.如果是负样本关系,则预测不出object.
            sub_tail (_type_, optional): [batch_size,1]. Defaults to None. subject 的tail. 主要是为了预测object.如果是负样本关系,则预测不出object.
        Returns:
            _type_: _description_
        """
        # 文本编码
        bert_output = self.textEncode(
            input_ids, attention_masks, token_type_ids)
        last_hidden_size = bert_output[0]
        last_hidden_size = self.words_dropout(last_hidden_size)
        pooler_output = bert_output[1]
        pred_rels = self.relModel(pooler_output)
        pred_entity_heads, pred_entity_tails = self.entityModel(
            last_hidden_size)
        pred_obj_head = self.objModel(
            relation, last_hidden_size, sub_head, sub_tail, attention_masks)
        return pred_rels, pred_entity_heads, pred_entity_tails, pred_obj_head

    def expand_attention_masks(self, attention_mask):
        batch_size, seq_length = attention_mask.shape
        causal_mask = attention_mask.unsqueeze(2).repeat(
            1, 1, seq_length) * attention_mask[:, None, :]
        # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(attention_mask.dtype)
        extended_attention_mask = causal_mask[:, None, :, :]
        extended_attention_mask = (1e-10)*(1-extended_attention_mask)
        return extended_attention_mask


class TDEERPytochLighting(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = TDEER(args.pretrain_path, args.relation_number)
        # 只针对对抗训练时，关闭自动优化
        # self.automatic_optimization = False
        self.fgm = FGM(self.model)
        with open(args.relation, 'r') as f:
            relation = json.load(f)
        self.id2rel = relation[0]
        self.rel_loss = nn.MultiLabelSoftMarginLoss()
        self.entity_head_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.entity_tail_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.obj_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.focal_loss = MLFocalLoss()
        self.b_focal_loss = BCEFocalLoss(alpha=0.25, gamma=2)
        self.threshold = 0.5
        self.args = args
        self.epoch = 0
        self.save_hyperparameters(args)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def training_step_adv(self, batch, step_idx):
        opt = self.optimizers()
        opt.zero_grad()
        loss = self(batch)
        self.manual_backward(loss)
        self.fgm.attack()
        loss_adv = self(batch)
        self.fgm.restore()
        self.manual_backward(loss_adv)
        opt.step()

    def training_step(self, batch, step_idx):
        batch_texts, batch_offsets, batch_tokens, batch_attention_masks, batch_segments, batch_entity_heads, batch_entity_tails, batch_rels, \
            batch_sample_subj_head, batch_sample_subj_tail, batch_sample_rel, batch_sample_obj_heads, batch_triple_sets, batch_text_masks = batch
        output = self.model(batch_tokens, batch_attention_masks, batch_segments,
                            relation=batch_sample_rel, sub_head=batch_sample_subj_head, sub_tail=batch_sample_subj_tail)
        pred_rels, pred_entity_heads, pred_entity_tails, pred_obj_head = output

        rel_loss = self.rel_loss(pred_rels, batch_rels) + \
            self.focal_loss(pred_rels, batch_rels)

        batch_text_mask = batch_text_masks.reshape(-1, 1)

        pred_entity_heads = pred_entity_heads.reshape(-1, 2)
        batch_entity_heads = batch_entity_heads.reshape(-1, 2)
        entity_head_loss = self.entity_head_loss(
            pred_entity_heads, batch_entity_heads)
        entity_head_loss = (
            entity_head_loss*batch_text_mask).sum()/batch_text_mask.sum()

        pred_entity_tails = pred_entity_tails.reshape(-1, 2)
        batch_entity_tails = batch_entity_tails.reshape(-1, 2)
        entity_tail_loss = self.entity_tail_loss(
            pred_entity_tails, batch_entity_tails)
        entity_tail_loss = (
            entity_tail_loss*batch_text_mask).sum()/batch_text_mask.sum()

        pred_obj_head = pred_obj_head.reshape(-1, 1)
        batch_sample_obj_heads = batch_sample_obj_heads.reshape(-1, 1)
        obj_loss = self.obj_loss(pred_obj_head, batch_sample_obj_heads) + \
            self.b_focal_loss(pred_obj_head, batch_sample_obj_heads)
        obj_loss = (obj_loss*batch_text_mask).sum()/batch_text_mask.sum()

        return rel_loss+entity_head_loss+entity_tail_loss+5*obj_loss

    def validation_step(self, batch, step_idx):
        batch_texts, batch_offsets, batch_tokens, batch_attention_masks, batch_segments, batch_triple_sets, batch_triples_index_set, batch_text_masks = batch
        bert_output = self.model.textEncode(
            batch_tokens, batch_attention_masks, batch_segments,)
        last_hidden_size = bert_output[0]
        pooler_output = bert_output[1]
        entity_heads_logits, entity_tails_logits = self.model.entityModel(
            last_hidden_size)
        entity_heads_logits = torch.sigmoid(entity_heads_logits)
        entity_tails_logits = torch.sigmoid(entity_tails_logits)
        relations_logits = self.model.relModel(pooler_output)
        relations_logits = torch.sigmoid(relations_logits)
        batch_size = entity_heads_logits.shape[0]
        entity_heads_logits = entity_heads_logits.cpu().numpy()
        entity_tails_logits = entity_tails_logits.cpu().numpy()
        relations_logits = relations_logits.cpu().numpy()
        batch_text_masks = batch_text_masks.cpu().numpy()

        pred_triple_sets = []
        for index in range(batch_size):
            mapping = rematch(batch_offsets[index])
            text = batch_texts[index]
            text_attention_mask = batch_text_masks[index].reshape(-1, 1)
            entity_heads_logit = entity_heads_logits[index]*text_attention_mask
            entity_tails_logit = entity_tails_logits[index]*text_attention_mask

            entity_heads, entity_tails = np.where(
                entity_heads_logit > self.threshold), np.where(entity_tails_logit > self.threshold)
            subjects = []
            entity_map = {}
            for head, head_type in zip(*entity_heads):
                for tail, tail_type in zip(*entity_tails):
                    if head <= tail and head_type == tail_type:
                        if head >= len(mapping) or tail >= len(mapping):
                            break
                        entity = self.decode_entity(text, mapping, head, tail)
                        if head_type == 0:
                            subjects.append((entity, head, tail))
                        else:
                            entity_map[head] = entity
                        break

            triple_set = set()
            if len(subjects):
                # translating decoding
                relations = np.where(relations_logits[index] > self.threshold)[
                    0].tolist()
                if relations:
                    batch_sub_heads = []
                    batch_sub_tails = []
                    batch_rels = []
                    batch_sub_entities = []
                    batch_rel_types = []
                    for (sub, sub_head, sub_tail) in subjects:
                        for rel in relations:
                            batch_sub_heads.append([sub_head])
                            batch_sub_tails.append([sub_tail])
                            batch_rels.append([rel])
                            batch_sub_entities.append(sub)
                            batch_rel_types.append(self.id2rel[str(rel)])
                    batch_sub_heads = torch.tensor(
                        batch_sub_heads, dtype=torch.long, device=last_hidden_size.device)
                    batch_sub_tails = torch.tensor(
                        batch_sub_tails, dtype=torch.long, device=last_hidden_size.device)
                    batch_rels = torch.tensor(
                        batch_rels, dtype=torch.long, device=last_hidden_size.device)
                    hidden = last_hidden_size[index].unsqueeze(0)
                    attention_mask = batch_attention_masks[index].unsqueeze(0)
                    batch_sub_heads = batch_sub_heads.transpose(1, 0)
                    batch_sub_tails = batch_sub_tails.transpose(1, 0)
                    batch_rels = batch_rels.transpose(1, 0)
                    obj_head_logits = self.model.objModel(
                        batch_rels, hidden, batch_sub_heads, batch_sub_tails, attention_mask)
                    obj_head_logits = torch.sigmoid(obj_head_logits)
                    obj_head_logits = obj_head_logits.cpu().numpy()
                    text_attention_mask = text_attention_mask.reshape(1, -1)
                    for sub, rel, obj_head_logit in zip(batch_sub_entities, batch_rel_types, obj_head_logits):
                        obj_head_logit = obj_head_logit*text_attention_mask
                        for h in np.where(obj_head_logit > self.threshold)[1].tolist():
                            if h in entity_map:
                                obj = entity_map[h]
                                triple_set.add((sub, rel, obj))
            pred_triple_sets.append(triple_set)
        return batch_texts, pred_triple_sets, batch_triple_sets

    def validation_epoch_end(self, outputs):
        preds, targets = [], []
        texts = []
        for text, pred, target in outputs:
            preds.extend(pred)
            targets.extend(target)
            texts.extend(text)

        correct = 0
        predict = 0
        total = 0
        orders = ['subject', 'relation', 'object']

        os.makedirs(os.path.join(self.args.output_path,
                    self.args.model_type), exist_ok=True)
        writer = open(os.path.join(self.args.output_path, self.args.model_type,
                      'val_output_{}.json'.format(self.epoch)), 'w')
        for text, pred, target in zip(*(texts, preds, targets)):
            pred = set([tuple(l) for l in pred])
            target = set([tuple(l) for l in target])
            correct += len(set(pred) & (target))
            predict += len(set(pred))
            total += len(set(target))
            new = [dict(zip(orders, triple)) for triple in pred - target]
            lack = [dict(zip(orders, triple)) for triple in target - pred]
            if len(new) or len(lack):
                result = json.dumps({
                                    'text': text,
                                    'golds': [
                                        dict(zip(orders, triple)) for triple in target
                                    ],
                                    'preds': [
                                        dict(zip(orders, triple)) for triple in pred
                                    ],
                                    'new': new,
                                    'lack': lack
                                    }, ensure_ascii=False)
                writer.write(result + '\n')
        writer.close()

        self.epoch += 1
        real_acc = round(correct/predict, 5) if predict != 0 else 0
        real_recall = round(correct/total, 5)
        real_f1 = round(2*(real_recall*real_acc)/(real_recall +
                        real_acc), 5) if (real_recall+real_acc) != 0 else 0
        self.log("tot", total, prog_bar=True)
        self.log("cor", correct, prog_bar=True)
        self.log("pred", predict, prog_bar=True)
        self.log("recall", real_recall, prog_bar=True)
        self.log("acc", real_acc, prog_bar=True)
        self.log("f1", real_f1, prog_bar=True)

        only_sub_rel_cor = 0
        only_sub_rel_pred = 0
        only_sub_rel_tot = 0
        for pred, target in zip(*(preds, targets)):
            pred = [list(l) for l in pred]
            pred = [(l[0], l[1]) for l in pred if len(l)]
            target = [(l[0], l[1]) for l in target]
            only_sub_rel_cor += len(set(pred).intersection(set(target)))
            only_sub_rel_pred += len(set(pred))
            only_sub_rel_tot += len(set(target))

        real_acc = round(only_sub_rel_cor/only_sub_rel_pred,
                         5) if only_sub_rel_pred != 0 else 0
        real_recall = round(only_sub_rel_cor/only_sub_rel_tot, 5)
        real_f1 = round(2*(real_recall*real_acc)/(real_recall +
                        real_acc), 5) if (real_recall+real_acc) != 0 else 0
        self.log("sr_tot", only_sub_rel_tot, prog_bar=True)
        self.log("sr_cor", only_sub_rel_cor, prog_bar=True)
        self.log("sr_pred", only_sub_rel_pred, prog_bar=True)
        self.log("sr_rec", real_recall, prog_bar=True)
        self.log("sr_acc", real_acc, prog_bar=True)
        self.log("sr_f1", real_f1, prog_bar=True)

    def decode_entity(self, text: str, mapping, start: int, end: int):
        s = mapping[start]
        e = mapping[end]
        s = 0 if not s else s[0]
        e = len(text) - 1 if not e else e[-1]
        entity = text[s: e + 1]
        return entity

    def configure_optimizers(self):
        """[配置优化参数]
        """
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.8, 'lr':2e-5},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.0, 'lr':2e-5},
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay) and 'bert' not in n], 'weight_decay': 0.8, 'lr':2e-4},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay) and 'bert' not in n], 'weight_decay': 0.0, 'lr':2e-4}
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


def partial_match(pred_set, gold_set):
    pred = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(
        ' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in pred_set}
    gold = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(
        ' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
    return pred, gold


def remove_space(data_set):
    data_set = {(i[0].replace(' ', ''), i[1], i[2].replace(' ', ''))
                for i in data_set}
    return data_set


class TDEERDataset(Dataset):
    def __init__(self, train_file, args, is_training=False):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(
            args.pretrain_path, cache_dir="./bertbaseuncased")
        self.is_training = is_training
        self.neg_samples = args.neg_samples
        self.batch_size = args.batch_size
        with open(args.relation, 'r') as f:
            relation = json.load(f)
        self.rel2id = relation[1]
        self.rels_set = list(self.rel2id.values())
        self.relation_size = len(self.rel2id)
        self.max_sample_triples = args.max_sample_triples
        self.datas = []
        with open(train_file, 'r') as f:
            lines = json.load(f)
        if self.is_training:
            self.preprocess(lines)
        else:
            self.proprecss_val(lines)

    def proprecss_val(self, datas):
        for data in datas:
            neg_datas = []
            text = data['text']
            text_tokened = self.tokenizer(text, return_offsets_mapping=True)
            input_ids = text_tokened['input_ids']
            attention_masks = text_tokened['attention_mask']
            text_masks = self.get_text_mask(attention_masks)
            token_type_ids = text_tokened['token_type_ids']
            offset_mapping = text_tokened['offset_mapping']
            triples_index_set = set()   # (sub head, sub tail, obj head, obj tail, rel)
            triples_set = set()
            for triple in data['triple_list']:
                subj, rel, obj = triple
                triples_set.add((subj, rel, obj))
                subj, rel, obj = triple
                rel_idx = self.rel2id[rel]
                subj_tokened = self.tokenizer.encode(subj)
                obj_tokened = self.tokenizer.encode(obj)
                subj_head_idx = find_head_idx(input_ids, subj_tokened[1:-1],0)
                subj_tail_idx = subj_head_idx + len(subj_tokened[1:-1]) - 1
                obj_head_idx = find_head_idx(input_ids, obj_tokened[1:-1],subj_tail_idx+1)
                if obj_head_idx == -1:
                    obj_head_idx = find_head_idx(input_ids, obj_tokened[1:-1],0)
                obj_tail_idx = obj_head_idx + len(obj_tokened[1:-1]) - 1
                if subj_head_idx == -1 or obj_head_idx == -1:
                    continue
                # 0表示subject，1表示object
                triples_index_set.add(
                    (subj_head_idx, subj_tail_idx,
                     obj_head_idx, obj_tail_idx, rel_idx)
                )

            # postive samples
            self.datas.append({
                "text": text,
                'token_ids': input_ids,
                "attention_masks": attention_masks,
                "offset_mapping": offset_mapping,
                "text_masks": text_masks,
                'segment_ids': token_type_ids,
                "length": len(input_ids),
                "triples_set": triples_set,
                "triples_index_set": triples_index_set
            })

    def get_text_mask(self, attention_masks):
        new_atten_mask = copy.deepcopy(attention_masks)
        new_atten_mask[0] = 0
        new_atten_mask[-1] = 0
        return new_atten_mask

    def preprocess(self, datas):
        for data in datas:
            pos_datas = []
            neg_datas = []
            text = data['text']
            text_tokened = self.tokenizer(text, return_offsets_mapping=True)
            input_ids = text_tokened['input_ids']
            attention_masks = text_tokened['attention_mask']
            text_masks = self.get_text_mask(attention_masks)
            token_type_ids = text_tokened['token_type_ids']
            offset_mapping = text_tokened['offset_mapping']
            text_length = len(input_ids)
            entity_set = set()  # (head idx, tail idx)
            triples_set = set()   # (sub head, sub tail, obj head, obj tail, rel)
            subj_rel_set = set()   # (sub head, sub tail, rel)
            subj_set = set()   # (sub head, sub tail)
            rel_set = set()
            trans_map = defaultdict(list)   # {(sub_head, rel): [tail_heads]}
            triples_sets = set()
            for triple in data['triple_list']:
                subj, rel, obj = triple
                triples_sets.add((subj, rel, obj))
                rel_idx = self.rel2id[rel]
                subj_tokened = self.tokenizer.encode(subj)
                obj_tokened = self.tokenizer.encode(obj)
                subj_head_idx = find_head_idx(input_ids, subj_tokened[1:-1],0)
                subj_tail_idx = subj_head_idx + len(subj_tokened[1:-1]) - 1
                obj_head_idx = find_head_idx(input_ids, obj_tokened[1:-1],subj_tail_idx+1)
                if obj_head_idx == -1:
                    obj_head_idx = find_head_idx(input_ids, obj_tokened[1:-1],0)
                obj_tail_idx = obj_head_idx + len(obj_tokened[1:-1]) - 1
                if subj_head_idx == -1 or obj_head_idx == -1:
                    continue
                # 0表示subject，1表示object
                entity_set.add((subj_head_idx, subj_tail_idx, 0))
                entity_set.add((obj_head_idx, obj_tail_idx, 1))
                subj_rel_set.add((subj_head_idx, subj_tail_idx, rel_idx))
                subj_set.add((subj_head_idx, subj_tail_idx))
                triples_set.add(
                    (subj_head_idx, subj_tail_idx,
                     obj_head_idx, obj_tail_idx, rel_idx)
                )
                rel_set.add(rel_idx)
                trans_map[(subj_head_idx, subj_tail_idx, rel_idx)
                          ].append(obj_head_idx)

            if not rel_set:
                continue

            # 当前句子中的所有subjects和objects
            entity_heads = np.zeros((text_length, 2))
            entity_tails = np.zeros((text_length, 2))
            for (head, tail, _type) in entity_set:
                entity_heads[head][_type] = 1
                entity_tails[tail][_type] = 1

            # 当前句子的所有关系
            rels = np.zeros(self.relation_size)
            for idx in rel_set:
                rels[idx] = 1

            if self.max_sample_triples is not None:
                triples_list = list(triples_set)
                np.random.shuffle(triples_list)
                triples_list = triples_list[:self.max_sample_triples]
            else:
                triples_list = list(triples_set)

            neg_history = set()
            for subj_head_idx, subj_tail_idx, obj_head_idx, obj_tail_idx, rel_idx in triples_list:
                current_neg_datas = []
                # 一个subject作为输入样例
                sample_obj_heads = np.zeros(text_length)
                for idx in trans_map[(subj_head_idx, subj_tail_idx, rel_idx)]:
                    sample_obj_heads[idx] = 1.0
                # postive samples
                pos_datas.append({
                    "text": text,
                    'token_ids': input_ids,
                    "attention_masks": attention_masks,
                    "text_masks": text_masks,
                    "offset_mapping": offset_mapping,
                    'segment_ids': token_type_ids,
                    'entity_heads': entity_heads,  # 所有实体的头部
                    'entity_tails': entity_tails,  # 所有实体的尾部
                    'rels': rels,  # 所有关系
                    'sample_subj_head': subj_head_idx,  # 单个subject的头部
                    'sample_subj_tail': subj_tail_idx,  # 单个subject的尾部
                    'sample_rel': rel_idx,  # 单个subject对应的关系
                    'sample_obj_heads': sample_obj_heads,  # 单个subject和rel对应的object
                    "length": len(input_ids),
                    "triples_sets": triples_sets
                })

                # 只针对训练数据集进行负采样
                if not self.is_training:
                    continue
                # 将subject和object对调创建负例，对应的object全为0
                # 1. inverse (tail as subj)
                neg_subj_head_idx = obj_head_idx
                neg_sub_tail_idx = obj_tail_idx
                neg_pair = (neg_subj_head_idx, neg_sub_tail_idx, rel_idx)
                if neg_pair not in subj_rel_set and neg_pair not in neg_history:
                    current_neg_datas.append({
                        "text": text,
                        'token_ids': input_ids,
                        "attention_masks": attention_masks,
                        "offset_mapping": offset_mapping,
                        "text_masks": text_masks,
                        'segment_ids': token_type_ids,
                        'entity_heads': entity_heads,
                        'entity_tails': entity_tails,
                        'rels': rels,
                        'sample_subj_head': neg_subj_head_idx,
                        'sample_subj_tail': neg_sub_tail_idx,
                        'sample_rel': rel_idx,
                        # set 0 for negative samples
                        'sample_obj_heads': np.zeros(text_length),
                        "length": len(input_ids),
                        "triples_sets": triples_sets
                    })
                    neg_history.add(neg_pair)

                # 随机选择其他关系作为负例，对应的object全为0
                # 2. (pos sub, neg_rel)
                for neg_rel_idx in rel_set - {rel_idx}:
                    neg_pair = (subj_head_idx, subj_tail_idx, neg_rel_idx)
                    if neg_pair not in subj_rel_set and neg_pair not in neg_history:
                        current_neg_datas.append({
                            "text": text,
                            'token_ids': input_ids,
                            "attention_masks": attention_masks,
                            "text_masks": text_masks,
                            "offset_mapping": offset_mapping,
                            'segment_ids': token_type_ids,
                            'entity_heads': entity_heads,
                            'entity_tails': entity_tails,
                            'rels': rels,
                            'sample_subj_head': subj_head_idx,
                            'sample_subj_tail': subj_tail_idx,
                            'sample_rel': neg_rel_idx,
                            # set 0 for negative samples
                            'sample_obj_heads': np.zeros(text_length),
                            "length": len(input_ids),
                            "triples_sets": triples_sets
                        })
                        neg_history.add(neg_pair)

                # 随机选择一个subject作为负例，对应的object全为0
                # 3. (neg sub, pos rel)
                for (neg_subj_head_idx, neg_sub_tail_idx) in subj_set - {(subj_head_idx, subj_tail_idx)}:
                    neg_pair = (neg_subj_head_idx, neg_sub_tail_idx, rel_idx)
                    if neg_pair not in subj_rel_set and neg_pair not in neg_history:
                        current_neg_datas.append({
                            "text": text,
                            'token_ids': input_ids,
                            'segment_ids': token_type_ids,
                            "attention_masks": attention_masks,
                            "text_masks": text_masks,
                            "offset_mapping": offset_mapping,
                            'entity_heads': entity_heads,
                            'entity_tails': entity_tails,
                            'rels': rels,
                            'sample_subj_head': neg_subj_head_idx,
                            'sample_subj_tail': neg_sub_tail_idx,
                            'sample_rel': rel_idx,
                            # set 0 for negative samples
                            'sample_obj_heads': np.zeros(text_length),
                            "length": len(input_ids),
                            "triples_sets": triples_sets
                        })
                        neg_history.add(neg_pair)

                np.random.shuffle(current_neg_datas)
                if self.neg_samples is not None:
                    current_neg_datas = current_neg_datas[:self.neg_samples]
                neg_datas += current_neg_datas
            current_datas = pos_datas + neg_datas
            self.datas.extend(current_datas)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]


def collate_fn(batches):
    """_summary_
    Args:
        batches (_type_): _description_
    """
    max_len = max([batch['length'] for batch in batches])
    batch_size = len(batches)

    batch_tokens = torch.zeros((batch_size, max_len), dtype=torch.int32)
    batch_attention_masks = torch.zeros(
        (batch_size, max_len), dtype=torch.float32)
    batch_segments = torch.zeros((batch_size, max_len), dtype=torch.int32)
    batch_entity_heads = torch.zeros(
        (batch_size, max_len, 2), dtype=torch.float32)
    batch_entity_tails = torch.zeros(
        (batch_size, max_len, 2), dtype=torch.float32)
    batch_rels = []
    batch_sample_subj_head, batch_sample_subj_tail = [], []
    batch_sample_rel = []
    batch_sample_obj_heads = torch.zeros(
        (batch_size, max_len), dtype=torch.float32)
    batch_texts = []
    batch_text_masks = torch.zeros((batch_size, max_len), dtype=torch.float32)
    batch_offsets = []
    batch_triples_sets = []
    for i, obj in enumerate(batches):
        length = obj['length']
        batch_texts.append(obj['text'])
        batch_offsets.append(obj['offset_mapping'])
        batch_tokens[i, :length] = torch.tensor(obj['token_ids'])
        batch_attention_masks[i, :length] = torch.tensor(
            obj['attention_masks'])
        batch_text_masks[i, :length] = torch.tensor(obj['text_masks'])
        batch_segments[i, :length] = torch.tensor(obj['segment_ids'])
        batch_entity_heads[i, :length, :] = torch.tensor(obj['entity_heads'])
        batch_entity_tails[i, :length, :] = torch.tensor(obj['entity_tails'])
        batch_rels.append(obj['rels'])
        batch_sample_subj_head.append([obj['sample_subj_head']])
        batch_sample_subj_tail.append([obj['sample_subj_tail']])

        batch_sample_rel.append([obj['sample_rel']])
        batch_sample_obj_heads[i, :length] = torch.tensor(
            obj['sample_obj_heads'])
        batch_triples_sets.append(obj['triples_sets'])
    batch_rels = torch.from_numpy(np.array(batch_rels, dtype=np.int64))
    batch_sample_subj_head = torch.from_numpy(
        np.array(batch_sample_subj_head, dtype=np.int64))
    batch_sample_subj_tail = torch.from_numpy(
        np.array(batch_sample_subj_tail, dtype=np.int64))
    batch_sample_rel = torch.from_numpy(
        np.array(batch_sample_rel, dtype=np.int64))

    return [batch_texts, batch_offsets, batch_tokens, batch_attention_masks, batch_segments, batch_entity_heads, batch_entity_tails, batch_rels, batch_sample_subj_head, batch_sample_subj_tail, batch_sample_rel, batch_sample_obj_heads, batch_triples_sets, batch_text_masks]


def collate_fn_val(batches):
    """_summary_
    Args:
        batches (_type_): _description_
    """
    max_len = max([batch['length'] for batch in batches])
    batch_size = len(batches)

    batch_tokens = torch.zeros((batch_size, max_len), dtype=torch.int32)
    batch_attention_masks = torch.zeros(
        (batch_size, max_len), dtype=torch.float32)
    batch_segments = torch.zeros((batch_size, max_len), dtype=torch.int32)
    batch_text_masks = torch.zeros((batch_size, max_len), dtype=torch.float32)

    batch_triple_set = []
    batch_triples_index_set = []
    batch_texts = []
    batch_offsets = []
    for i, obj in enumerate(batches):
        length = obj['length']
        batch_texts.append(obj['text'])
        batch_offsets.append(obj['offset_mapping'])
        batch_tokens[i, :length] = torch.tensor(obj['token_ids'])
        batch_attention_masks[i, :length] = torch.tensor(
            obj['attention_masks'])
        batch_text_masks[i, :length] = torch.tensor(obj['text_masks'])
        batch_segments[i, :length] = torch.tensor(obj['segment_ids'])
        batch_triple_set.append(obj['triples_set'])
        batch_triples_index_set.append(obj['triples_index_set'])

    return [batch_texts, batch_offsets, batch_tokens, batch_attention_masks, batch_segments, batch_triple_set, batch_triples_index_set, batch_text_masks]
