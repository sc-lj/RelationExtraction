# -*- encoding: utf-8 -*-
'''
File    :   SPN4RE_Model.py
Time    :   2022/08/13 10:14:59
Author  :   lujun
Version :   1.0
Contact :   779365135@qq.com
License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
Desc    :   for SPN4RE model
'''
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention, BertSelfAttention, BertModel
import torch.nn.functional as F
from SPN4RE.utils import generate_triple, formulate_gold
import pytorch_lightning as pl


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask):
        """关系的解码器
        Args:
            hidden_states (_type_): 关系隐层,初始为关系的映射,后面都是上一层decoder输出, [batch_size,num_generated_triples,hidden_size]
            encoder_hidden_states (_type_): 输入token的表征,[batch_size,seq_len,hidden_size]
            encoder_attention_mask (_type_): attention mask,[batch_size,seq_len]
        Raises:
            ValueError: _description_
        Returns:
            _type_: _description_
        """
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (
            1.0 - encoder_extended_attention_mask) * -10000.0
        # attention_output 作为self attention中query，encoder_hidden_states作为self attention中key，value
        # [batch_size,num_generated_triples,hidden_size]
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,  encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        # add cross attentions if we output attention weights
        outputs = outputs + cross_attention_outputs[1:]
        # [batch_size,num_generated_triples,intermediate_size]
        intermediate_output = self.intermediate(attention_output)
        # [batch_size,num_generated_triples,intermediate_size] -> # [batch_size,num_generated_triples,hidden_size]
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class SetDecoder(nn.Module):
    def __init__(self, config, num_generated_triples, num_layers, num_classes, return_intermediate=False):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_generated_triples = num_generated_triples
        self.layers = nn.ModuleList([DecoderLayer(config)
                                    for _ in range(num_layers)])
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 对模型产生的每一个triple，采用query的方式(即生成query的embedding)，并与文本token的表征的进行交互
        self.query_embed = nn.Embedding(
            num_generated_triples, config.hidden_size)
        # 预测关系，加上了一个“无关系”的标签
        self.decoder2class = nn.Linear(config.hidden_size, num_classes + 1)
        self.decoder2span = nn.Linear(config.hidden_size, 4)

        # 预测subject，object的start，end 4个部分的的分类头
        self.head_start_metric_1 = nn.Linear(
            config.hidden_size, config.hidden_size)
        self.head_end_metric_1 = nn.Linear(
            config.hidden_size, config.hidden_size)
        self.tail_start_metric_1 = nn.Linear(
            config.hidden_size, config.hidden_size)
        self.tail_end_metric_1 = nn.Linear(
            config.hidden_size, config.hidden_size)
        self.head_start_metric_2 = nn.Linear(
            config.hidden_size, config.hidden_size)
        self.head_end_metric_2 = nn.Linear(
            config.hidden_size, config.hidden_size)
        self.tail_start_metric_2 = nn.Linear(
            config.hidden_size, config.hidden_size)
        self.tail_end_metric_2 = nn.Linear(
            config.hidden_size, config.hidden_size)
        self.head_start_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.head_end_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_start_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_end_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)

        torch.nn.init.orthogonal_(self.head_start_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_start_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)

    def forward(self, encoder_hidden_states, encoder_attention_mask):
        bsz = encoder_hidden_states.size()[0]
        # [batch_size,num_generated_triples,hidden_size]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        # [batch_size,num_generated_triples,hidden_size]
        hidden_states = self.dropout(self.LayerNorm(hidden_states))
        all_hidden_states = ()
        # query embedding 与文本token的交互
        for i, layer_module in enumerate(self.layers):
            if self.return_intermediate:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]
        # 对query 的hidden 进行分类
        class_logits = self.decoder2class(hidden_states)

        # [batch_size,num_generated_triples,1,hidden_size] + [batch_size,1,seq_len,hidden_size] = [batch_size,num_generated_triples,seq_len,hidden_size]
        # [batch_size,num_generated_triples,seq_len,hidden_size]->[batch_size,num_generated_triples,seq_len]
        head_start_logits = self.head_start_metric_3(torch.tanh(
            self.head_start_metric_1(hidden_states).unsqueeze(2) + self.head_start_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()

        # [batch_size,num_generated_triples,1,hidden_size] + [batch_size,1,seq_len,hidden_size] = [batch_size,num_generated_triples,seq_len,hidden_size]
        # [batch_size,num_generated_triples,seq_len,hidden_size]->[batch_size,num_generated_triples,seq_len]
        head_end_logits = self.head_end_metric_3(torch.tanh(
            self.head_end_metric_1(hidden_states).unsqueeze(2) + self.head_end_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()

        # [batch_size,num_generated_triples,1,hidden_size] + [batch_size,1,seq_len,hidden_size] = [batch_size,num_generated_triples,seq_len,hidden_size]
        # [batch_size,num_generated_triples,seq_len,hidden_size]->[batch_size,num_generated_triples,seq_len]
        tail_start_logits = self.tail_start_metric_3(torch.tanh(
            self.tail_start_metric_1(hidden_states).unsqueeze(2) + self.tail_start_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()

        # [batch_size,num_generated_triples,1,hidden_size] + [batch_size,1,seq_len,hidden_size] = [batch_size,num_generated_triples,seq_len,hidden_size]
        # [batch_size,num_generated_triples,seq_len,hidden_size]->[batch_size,num_generated_triples,seq_len]
        tail_end_logits = self.tail_end_metric_3(torch.tanh(
            self.tail_end_metric_1(hidden_states).unsqueeze(2) + self.tail_end_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()

        return class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, loss_weight, matcher):
        super().__init__()
        self.cost_relation = loss_weight["relation"]
        self.cost_head = loss_weight["head_entity"]
        self.cost_tail = loss_weight["tail_entity"]
        self.matcher = matcher

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ 将预测和target进行匹配
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_rel_logits": Tensor of dim [batch_size, num_generated_triples, num_classes] with the classification logits
                 "{head, tail}_{start, end}_logits": Tensor of dim [batch_size, num_generated_triples, seq_len] with the predicted index logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_generated_triples, num_gold_triples)
        """
        # 预测的关系
        bsz, num_generated_triples = outputs["pred_rel_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        # [bsz * num_generated_triples, num_classes]
        pred_rel = outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)
        # [len(sum(x,[])),1] 将关系展平,[[2,3],[2]]-> [2,3,2]
        gold_rel = torch.cat([v["relation"] for v in targets])
        # after masking the pad token
        # [bsz * num_generated_triples, seq_len]
        pred_head_start = outputs["head_start_logits"].flatten(
            0, 1).softmax(-1)
        # [bsz * num_generated_triples, seq_len]
        pred_head_end = outputs["head_end_logits"].flatten(0, 1).softmax(-1)
        # [bsz * num_generated_triples, seq_len]
        pred_tail_start = outputs["tail_start_logits"].flatten(
            0, 1).softmax(-1)
        # [bsz * num_generated_triples, seq_len]
        pred_tail_end = outputs["tail_end_logits"].flatten(0, 1).softmax(-1)

        # [len(sum(x,[])),1] 将关系展平,[[2,3],[2]]-> [2,3,2]
        gold_head_start = torch.cat([v["head_start_index"] for v in targets])
        # [len(sum(x,[])),1] 将关系展平,[[2,3],[2]]-> [2,3,2]
        gold_head_end = torch.cat([v["head_end_index"] for v in targets])
        # [len(sum(x,[])),1] 将关系展平,[[2,3],[2]]-> [2,3,2]
        gold_tail_start = torch.cat([v["tail_start_index"] for v in targets])
        # [len(sum(x,[])),1] 将关系展平,[[2,3],[2]]-> [2,3,2]
        gold_tail_end = torch.cat([v["tail_end_index"] for v in targets])
        if self.matcher == "avg":
            # 从预测的关系或者token概率中，选取在gold关系或者token中的概率
            # [bsz*num_generated_triples,len(x_i)] x_i 是一个batch 展品后的target 长度
            cost = - self.cost_relation * pred_rel[:, gold_rel] - self.cost_head * 1/2 * (
                pred_head_start[:, gold_head_start] + pred_head_end[:, gold_head_end]) - self.cost_tail * 1/2 * (pred_tail_start[:, gold_tail_start] + pred_tail_end[:, gold_tail_end])
        elif self.matcher == "min":
            cost = torch.cat([pred_head_start[:, gold_head_start].unsqueeze(1), pred_rel[:, gold_rel].unsqueeze(1), pred_head_end[:, gold_head_end].unsqueeze(
                1), pred_tail_start[:, gold_tail_start].unsqueeze(1), pred_tail_end[:, gold_tail_end].unsqueeze(1)], dim=1)
            cost = - torch.min(cost, dim=1)[0]
        else:
            raise ValueError("Wrong matcher")
        # [bsz,num_generated_triples,len(x_i)] x_i 是一个batch 展品后的target 长度
        cost = cost.view(bsz, num_generated_triples, -1).cpu()
        # 每个样本中target的数量
        num_gold_triples = [len(v["relation"]) for v in targets]
        # cost.split(num_gold_triples, -1) 按照num_gold_triples对cost按照-1维度进行切分。
        # linear_sum_assignment:假设有4个工人（workers）a,b,c,d，有三项任务（job）p,q,r，每个工人干每一项活的成本都不同，那么便可构造一个代价矩阵（cost matrix）,
        # 使用linear_sum_assignment 得到最佳分配下的行列索引值，
        # cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2], [1, 1, 1]])
        # r, c = linear_sum_assignment(cost) => (array([1, 2, 3]), array([1, 2, 0]))
        # "最小成本："cost[r, c].sum()
        # 获取每个batch下，gold 标签在num_generated_triples中的索引
        # 获取每个样本下，num_generated_triples 成本(损失最小的)某一个或某几个num_generated_triple，以及某一个或某几个targe order
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(
            cost.split(num_gold_triples, -1))]
        # 成本最小的组成的，num_generated_triples的index 和 target order
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetCriterion(nn.Module):
    """ 计算损失函数
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, subject position and object position)
    """

    def __init__(self, num_classes, loss_weight, na_coef, losses, matcher):
        """ Create the criterion.
        Parameters:
            num_classes: number of relation categories
            matcher: module able to compute a matching between targets and proposals
            loss_weight: dict containing as key the names of the losses and as values their relative weight.
            na_coef: list containg the relative classification weight applied to the NA category and positional classification weight applied to the [SEP]
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.matcher = HungarianMatcher(loss_weight, matcher)
        self.losses = losses
        rel_weight = torch.ones(self.num_classes + 1)
        rel_weight[-1] = na_coef
        self.register_buffer('rel_weight', rel_weight)

    def forward(self, outputs, targets):
        """ 计算损失函数
        Parameters:
             outputs: dict of tensors,
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        # target 与 num_generated_triples 组合成本最小时，num_generated_triples的index和target order的index
        indices = self.matcher(outputs, targets)
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            # 判断是否含有关系
            if loss == "entity" and self.empty_targets(targets):
                pass
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices))
        losses = sum(losses[k] * self.loss_weight[k]
                     for k in losses.keys() if k in self.loss_weight)
        return losses

    def relation_loss(self, outputs, targets, indices):
        """关系分类损失函数。Classification loss (NLL)
        targets dicts must contain the key "relation" containing a tensor of dim [bsz]
        """
        # [bsz, num_generated_triples, num_rel+1]
        src_logits = outputs['pred_rel_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["relation"][i] for t, (_, i) in zip(targets, indices)])
        # [bsz, num_generated_triples] 并全用 self.num_classes初始化填充
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss = F.cross_entropy(src_logits.flatten(
            0, 1), target_classes.flatten(0, 1), weight=self.rel_weight)
        losses = {'relation': loss}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty triples
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_rel_logits = outputs['pred_rel_logits']
        device = pred_rel_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_rel_logits.argmax(-1) !=
                     pred_rel_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        # 对indices展开，[[1,3,4],[2,4]] -> [0,0,0,1,1]
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        # [[1,3,4],[2,4]] -> [1,3,4,2,4]
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        # 对indices展开，[[1,3,4],[2,4]] -> [0,0,0,1,1]
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        # 对indices展开，[[1,3,4],[2,4]] -> [0,0,0,1,1]
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices,  **kwargs):
        loss_map = {
            'relation': self.relation_loss,
            'cardinality': self.loss_cardinality,
            'entity': self.entity_loss
        }
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def entity_loss(self, outputs, targets, indices):
        """Compute the losses related to the position of head entity or tail entity
        """
        # ([sum(xi)],[sum(xi)]),分别是(batch idxs, num_generated_triples indexs)
        idx = self._get_src_permutation_idx(indices)
        # [bsz,num_generated_triples,seq_len]-> [sum(xi),seq_len]
        selected_pred_head_start = outputs["head_start_logits"][idx]
        selected_pred_head_end = outputs["head_end_logits"][idx]
        selected_pred_tail_start = outputs["tail_start_logits"][idx]
        selected_pred_tail_end = outputs["tail_end_logits"][idx]

        target_head_start = torch.cat(
            [t["head_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_head_end = torch.cat(
            [t["head_end_index"][i] for t, (_, i) in zip(targets, indices)])
        target_tail_start = torch.cat(
            [t["tail_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_tail_end = torch.cat(
            [t["tail_end_index"][i] for t, (_, i) in zip(targets, indices)])

        head_start_loss = F.cross_entropy(
            selected_pred_head_start, target_head_start)
        head_end_loss = F.cross_entropy(
            selected_pred_head_end, target_head_end)
        tail_start_loss = F.cross_entropy(
            selected_pred_tail_start, target_tail_start)
        tail_end_loss = F.cross_entropy(
            selected_pred_tail_end, target_tail_end)
        losses = {'head_entity': 1/2*(head_start_loss + head_end_loss),
                  "tail_entity": 1/2*(tail_start_loss + tail_end_loss)}
        # print(losses)
        return losses

    @staticmethod
    def empty_targets(targets):
        flag = True
        for target in targets:
            if len(target["relation"]) != 0:
                flag = False
                break
        return flag


class SeqEncoder(nn.Module):
    def __init__(self, args):
        super(SeqEncoder, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.pretrain_path)
        # if args.fix_bert_embeddings:
        #     self.bert.embeddings.word_embeddings.weight.requires_grad = False
        #     self.bert.embeddings.position_embeddings.weight.requires_grad = False
        #     self.bert.embeddings.token_type_embeddings.weight.requires_grad = False
        self.config = self.bert.config

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=None)
        last_hidden_state, pooler_output = outputs[0], outputs[1]
        return last_hidden_state, pooler_output


class SetPred4RE(nn.Module):
    def __init__(self, args, num_classes):
        super(SetPred4RE, self).__init__()
        self.args = args
        # 文本编码
        self.encoder = SeqEncoder(args)
        config = self.encoder.config
        self.num_classes = num_classes
        # 关系解码器
        self.decoder = SetDecoder(config, args.num_generated_triples,
                                  args.num_decoder_layers, num_classes, return_intermediate=False)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooler_output = self.encoder(
            input_ids, attention_mask)
        class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = self.decoder(
            encoder_hidden_states=last_hidden_state, encoder_attention_mask=attention_mask)
        # head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = span_logits.split(1, dim=-1)
        # [batch_size,num_generated_triples,seq_len]
        head_start_logits = head_start_logits.squeeze(-1).masked_fill(
            (1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        head_end_logits = head_end_logits.squeeze(-1).masked_fill(
            (1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        tail_start_logits = tail_start_logits.squeeze(-1).masked_fill(
            (1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        # [bsz, num_generated_triples, seq_len]
        tail_end_logits = tail_end_logits.squeeze(-1).masked_fill(
            (1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        outputs = {'pred_rel_logits': class_logits, 'head_start_logits': head_start_logits,
                   'head_end_logits': head_end_logits, 'tail_start_logits': tail_start_logits, 'tail_end_logits': tail_end_logits}
        return outputs


class Spn4REPytochLighting(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.num_classes = args.relation_number
        self.model = SetPred4RE(args, self.num_classes)
        self.criterion = SetCriterion(self.num_classes,  loss_weight=self.get_loss_weight(
            args.loss_weight), na_coef=args.na_rel_coef, losses=["entity", "relation"], matcher=args.matcher)

    def get_loss_weight(self, loss_weight):
        return {"relation": loss_weight[0], "head_entity": loss_weight[1], "tail_entity": loss_weight[2]}

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def training_step(self, batches, batch_idx):
        input_ids, attention_mask, targets, _ = batches
        outputs = self.model(input_ids, attention_mask)
        loss = self.criterion(outputs, targets)
        return loss

    def validation_step(self, batches, batch_idx):
        input_ids, attention_mask, targets, info = batches
        gold = formulate_gold(targets, info)
        outputs = self.model(input_ids, attention_mask)
        prediction = generate_triple(outputs, info, self.args, self.num_classes)
        return gold, prediction

    def validation_epoch_end(self, outputs):
        golds = {}
        preds = {}
        for gold, pred in outputs:
            golds.update(gold)
            preds.update(pred)
        result = self.metric(preds, golds)
        self.log("f1", result["f1"], prog_bar=True)
        self.log("acc", result["prec"], prog_bar=True)
        self.log("recall", result["recall"], prog_bar=True)

    def metric(self, pred, gold):
        assert pred.keys() == gold.keys()
        gold_num = 0  # gold数量
        rel_num = 0  # 一个样本中所有关系预测正确的数量
        ent_num = 0  # 一个样本中所有实体预测正确的数量
        right_num = 0  # 一个样本中所有都正确的数量
        pred_num = 0  # 预测数量
        for sent_idx in pred:
            gold_num += len(gold[sent_idx])
            pred_correct_num = 0
            # 一个样本中所有关系，subject，object的组合
            prediction = list(set([(ele.pred_rel, ele.head_start_index, ele.head_end_index,
                              ele.tail_start_index, ele.tail_end_index) for ele in pred[sent_idx]]))
            pred_num += len(prediction)
            for ele in prediction:
                if ele in gold[sent_idx]:
                    right_num += 1
                    pred_correct_num += 1
                if ele[0] in [e[0] for e in gold[sent_idx]]:
                    rel_num += 1
                if ele[1:] in [e[1:] for e in gold[sent_idx]]:
                    ent_num += 1

        if pred_num == 0:
            precision = -1
            r_p = -1
            e_p = -1
        else:
            precision = (right_num + 0.0) / pred_num
            e_p = (ent_num + 0.0) / pred_num
            r_p = (rel_num + 0.0) / pred_num

        if gold_num == 0:
            recall = -1
            r_r = -1
            e_r = -1
        else:
            recall = (right_num + 0.0) / gold_num
            e_r = ent_num / gold_num
            r_r = rel_num / gold_num

        if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
            f_measure = -1
        else:
            f_measure = 2 * precision * recall / (precision + recall)

        return {"prec": precision, "recall": recall, "f1": f_measure}

    def configure_optimizers(self):
        """[配置优化参数]
        """
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.8, 'lr':2e-5},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.0, 'lr':2e-5},
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                        and 'bert' not in n], 'weight_decay': 0.8, 'lr':2e-4},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and 'bert' not in n], 'weight_decay': 0.0, 'lr':2e-4}
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
