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
from SPN4RE_utils import generate_triple


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
        self.query_embed = nn.Embedding(
            num_generated_triples, config.hidden_size)
        self.decoder2class = nn.Linear(config.hidden_size, num_classes + 1)
        self.decoder2span = nn.Linear(config.hidden_size, 4)

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
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        hidden_states = self.dropout(self.LayerNorm(hidden_states))
        all_hidden_states = ()
        for i, layer_module in enumerate(self.layers):
            if self.return_intermediate:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

        class_logits = self.decoder2class(hidden_states)

        head_start_logits = self.head_start_metric_3(torch.tanh(
            self.head_start_metric_1(hidden_states).unsqueeze(2) + self.head_start_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()
        head_end_logits = self.head_end_metric_3(torch.tanh(
            self.head_end_metric_1(hidden_states).unsqueeze(2) + self.head_end_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()

        tail_start_logits = self.tail_start_metric_3(torch.tanh(
            self.tail_start_metric_1(hidden_states).unsqueeze(2) + self.tail_start_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()
        tail_end_logits = self.tail_end_metric_3(torch.tanh(
            self.tail_end_metric_1(hidden_states).unsqueeze(2) + self.tail_end_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()

        return class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask
    ):
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
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,  encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        # add cross attentions if we output attention weights
        outputs = outputs + cross_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


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
        """ Performs the matching

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
        bsz, num_generated_triples = outputs["pred_rel_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        # [bsz * num_generated_triples, num_classes]
        pred_rel = outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)
        gold_rel = torch.cat([v["relation"] for v in targets])
        # after masking the pad token
        pred_head_start = outputs["head_start_logits"].flatten(
            0, 1).softmax(-1)  # [bsz * num_generated_triples, seq_len]
        pred_head_end = outputs["head_end_logits"].flatten(0, 1).softmax(-1)
        pred_tail_start = outputs["tail_start_logits"].flatten(
            0, 1).softmax(-1)
        pred_tail_end = outputs["tail_end_logits"].flatten(0, 1).softmax(-1)

        gold_head_start = torch.cat([v["head_start_index"] for v in targets])
        gold_head_end = torch.cat([v["head_end_index"] for v in targets])
        gold_tail_start = torch.cat([v["tail_start_index"] for v in targets])
        gold_tail_end = torch.cat([v["tail_end_index"] for v in targets])
        if self.matcher == "avg":
            cost = - self.cost_relation * pred_rel[:, gold_rel] - self.cost_head * 1/2 * (
                pred_head_start[:, gold_head_start] + pred_head_end[:, gold_head_end]) - self.cost_tail * 1/2 * (pred_tail_start[:, gold_tail_start] + pred_tail_end[:, gold_tail_end])
        elif self.matcher == "min":
            cost = torch.cat([pred_head_start[:, gold_head_start].unsqueeze(1), pred_rel[:, gold_rel].unsqueeze(1), pred_head_end[:, gold_head_end].unsqueeze(
                1), pred_tail_start[:, gold_tail_start].unsqueeze(1), pred_tail_end[:, gold_tail_end].unsqueeze(1)], dim=1)
            cost = - torch.min(cost, dim=1)[0]
        else:
            raise ValueError("Wrong matcher")
        cost = cost.view(bsz, num_generated_triples, -1).cpu()
        num_gold_triples = [len(v["relation"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(
            cost.split(num_gold_triples, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetCriterion(nn.Module):
    """ This class computes the loss for Set_RE.
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
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == "entity" and self.empty_targets(targets):
                pass
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices))
        losses = sum(losses[k] * self.loss_weight[k]
                     for k in losses.keys() if k in self.loss_weight)
        return losses

    def relation_loss(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "relation" containing a tensor of dim [bsz]
        """
        src_logits = outputs['pred_rel_logits']  # [bsz, num_generated_triples, num_rel+1]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["relation"][i] for t, (_, i) in zip(targets, indices)])
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
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
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
        idx = self._get_src_permutation_idx(indices)
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
        self.bert = BertModel.from_pretrained(args.bert_directory)
        if args.fix_bert_embeddings:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False
        self.config = self.bert.config

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooler_output = self.bert(
            input_ids, attention_mask=attention_mask)
        return last_hidden_state, pooler_output


class SetPred4RE(nn.Module):

    def __init__(self, args, num_classes):
        super(SetPred4RE, self).__init__()
        self.args = args
        self.encoder = SeqEncoder(args)
        config = self.encoder.config
        self.num_classes = num_classes
        self.decoder = SetDecoder(config, args.num_generated_triples,
                                  args.num_decoder_layers, num_classes, return_intermediate=False)
        self.criterion = SetCriterion(num_classes,  loss_weight=self.get_loss_weight(
            args), na_coef=args.na_rel_coef, losses=["entity", "relation"], matcher=args.matcher)

    def forward(self, input_ids, attention_mask, targets=None):
        last_hidden_state, pooler_output = self.encoder(
            input_ids, attention_mask)
        class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = self.decoder(
            encoder_hidden_states=last_hidden_state, encoder_attention_mask=attention_mask)
        # head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = span_logits.split(1, dim=-1)
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
        if targets is not None:
            loss = self.criterion(outputs, targets)
            return loss, outputs
        else:
            return outputs

    def gen_triples(self, input_ids, attention_mask, info):
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            # print(outputs)
            pred_triple = generate_triple(
                outputs, info, self.args, self.num_classes)
            # print(pred_triple)
        return pred_triple

    def batchify(self, batch_list):
        batch_size = len(batch_list)
        sent_idx = [ele[0] for ele in batch_list]
        sent_ids = [ele[1] for ele in batch_list]
        targets = [ele[2] for ele in batch_list]
        sent_lens = list(map(len, sent_ids))
        max_sent_len = max(sent_lens)
        input_ids = torch.zeros(
            (batch_size, max_sent_len), requires_grad=False).long()
        attention_mask = torch.zeros(
            (batch_size, max_sent_len), requires_grad=False, dtype=torch.float32)
        for idx, (seq, seqlen) in enumerate(zip(sent_ids, sent_lens)):
            input_ids[idx, :seqlen] = torch.LongTensor(seq)
            attention_mask[idx, :seqlen] = torch.FloatTensor([1] * seqlen)
        if self.args.use_gpu:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            targets = [{k: torch.tensor(v, dtype=torch.long, requires_grad=False).cuda(
            ) for k, v in t.items()} for t in targets]
        else:
            targets = [{k: torch.tensor(
                v, dtype=torch.long, requires_grad=False) for k, v in t.items()} for t in targets]
        info = {"seq_len": sent_lens, "sent_idx": sent_idx}
        return input_ids, attention_mask, targets, info

    @staticmethod
    def get_loss_weight(args):
        return {"relation": args.rel_loss_weight, "head_entity": args.head_ent_loss_weight, "tail_entity": args.tail_ent_loss_weight}
