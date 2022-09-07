import os
import json
import torch
import numpy as np
import torch.nn as nn
from collections import Counter
import pytorch_lightning as pl
from utils.loss_func import GlobalCrossEntropy
from transformers import BertPreTrainedModel, BertModel
from PRGC.utils import tag_mapping_corres,Label2IdxSub,Label2IdxObj


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


class ConditionalLayerNorm(nn.Module):
    def __init__(self,normalized_shape,cond_shape,eps=1e-12):
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


class biaffine(nn.Module):
    def __init__(self, in_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.U = torch.nn.Parameter(torch.randn(in_size + int(bias_x), 1, in_size + int(bias_y)))

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping


class BiaffineTagger(nn.Module):
    def __init__(self, hidden_size):
        super(BiaffineTagger, self).__init__()
        self.start_layer = nn.Linear(hidden_size, 128)
        self.end_layer = nn.Linear(hidden_size, 128)
        self.biaffne_layer = biaffine(128)

    def forward(self, hidden):
        start_logits = self.start_layer(hidden)
        end_logits = self.end_layer(hidden)
        span_logits = self.biaffne_layer(start_logits, end_logits)
        span_logits = span_logits.squeeze(-1).contiguous()
        return span_logits


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
        self.cond_layer = ConditionalLayerNorm(config.hidden_size,config.hidden_size)
        self.layernormal = nn.LayerNorm(config.hidden_size)
        # 双仿射注意力机制
        self.biaffine_tagger = BiaffineTagger(config.hidden_size)

        # relation judgement
        self.rel_judgement = MultiNonLinearClassifier(config.hidden_size, self.rel_num, params.drop_prob)
        self.rel_embedding = nn.Embedding(self.rel_num, config.hidden_size)
        self.params = params
        self.init_weights()

    @staticmethod
    def masked_avgpool(sent, mask):
        mask_ = mask.masked_fill(mask == 0, -1e9).float()
        score = torch.softmax(mask_, -1)
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)

    def global_corres_v0(self,sequence_output,attention_mask,seq_len):
        """原论文中的global Correspondence 计算方式
        Args:
            sequence_output ([type]): [description]
            attention_mask ([type]): [description]
            seq_len ([type]): [description]
        Returns:
            [type]: [description]
        """
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
        return corres_pred,corres_mask

    def global_corres_biaffine(self,sequence_output,attention_mask):
        """使用BiaffineTagger计算global Correspondence
        Args:
            sequence_output ([type]): [description]
            attention_mask ([type]): [description]
        """
        hidden = self.layernormal(sequence_output)
        pred_corres = self.biaffine_tagger(hidden)
        mask_tmp1 = attention_mask.unsqueeze(-1)
        mask_tmp2 = attention_mask.unsqueeze(1)
        corres_mask = mask_tmp1 * mask_tmp2
        return pred_corres,corres_mask

    def forward(self, input_ids=None, attention_mask=None, potential_rels=None):
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
        # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(input_ids,attention_mask=attention_mask,output_hidden_states=True)
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        bs, seq_len, h = sequence_output.size()

        # 预测关系 (bs, h)
        if self.params.avgpool:
            h_k_avg = self.masked_avgpool(sequence_output, attention_mask)
            pooled_output = h_k_avg
        rel_pred = self.rel_judgement(pooled_output) # [bs, rel_num]

        if self.params.biaffine:
            # 使用不同版本计算corres_pred
            corres_pred,corres_mask = self.global_corres_biaffine(sequence_output,attention_mask)
        else:
            # 原文献中的方法
            corres_pred,corres_mask = self.global_corres_v0(sequence_output,attention_mask,seq_len)

        
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
        self.save_hyperparameters(args)
        if args.is_glo:
            self.corres_threshold = 0.
            self.corres_global_loss_func = GlobalCrossEntropy()
        else:
            self.corres_threshold = 0.5
            self.corres_loss_func = nn.BCEWithLogitsLoss(reduction='none')
        self.rel_loss_func = nn.BCEWithLogitsLoss(reduction='mean')
        self.ent_loss_func = nn.CrossEntropyLoss(reduction='none')
        self.args = args
        self.epoch = 0

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
        loss_seq_sub = (self.ent_loss_func(output_sub.view(-1, self.seq_tag_size),
                                    seq_tags[:, 0, :].reshape(-1)) * pos_attention_mask).sum() / pos_attention_mask.sum()
        loss_seq_obj = (self.ent_loss_func(output_obj.view(-1, self.seq_tag_size),
                                    seq_tags[:, 1, :].reshape(-1)) * pos_attention_mask).sum() / pos_attention_mask.sum()
        loss_seq = (loss_seq_sub + loss_seq_obj) / 2
        # init
        corres_pred = corres_pred.view(bs, -1)
        corres_mask = corres_mask.view(bs, -1)
        corres_tags = corres_tags.view(bs, -1)
        
        if self.args.is_glo:
            loss_matrix = self.corres_global_loss_func(corres_tags.float(),corres_pred)
        else:
            loss_matrix = (self.corres_loss_func(corres_pred,corres_tags.float()) * corres_mask).sum() / corres_mask.sum()

        loss_rel = self.rel_loss_func(rel_pred, rel_tags.float())

        loss = loss_seq + loss_matrix + loss_rel
        return loss
    
    def validation_step(self,batches,batch_idx):
        texts, input_ids, attention_mask, triples, input_tokens = batches
        # compute model output and loss
        output_sub, output_obj,corres_mask,_,_,corres_pred,xi,pred_rels = self.model(input_ids, attention_mask=attention_mask)
        # (sum(x_i), seq_len)
        pred_seq_sub = torch.argmax(torch.softmax(output_sub, dim=-1), dim=-1)
        pred_seq_obj = torch.argmax(torch.softmax(output_obj, dim=-1), dim=-1)
        # (sum(x_i), 2, seq_len)
        pred_seqs = torch.cat([pred_seq_sub.unsqueeze(1), pred_seq_obj.unsqueeze(1)], dim=1)
        if self.args.is_glo:
            corres_pred = corres_pred * corres_mask
        else:
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
        return texts, pred_seqs, pred_corres, xi_index, pred_rels,triples,input_tokens
    
    def validation_epoch_end(self, outputs):
        os.makedirs(os.path.join(self.args.output_path,
                    self.args.model_type), exist_ok=True)
        writer = open(os.path.join(self.args.output_path, self.args.model_type,
                      'val_output_{}.json'.format(self.epoch)), 'w')
        predictions = []
        ground_truths = []
        correct_num, predict_num, gold_num = 0, 0, 0
        error_result = []
        orders = ['subject', 'relation' , 'object']
        for texts,pred_seqs, pred_corres, xi_index, pred_rels,triples,input_tokens in outputs:
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
                new = [dict(zip(orders, triple)) for triple in set(pre_triples) - set(gold_triples)]
                lack = [dict(zip(orders, triple)) for triple in set(gold_triples) - set(pre_triples)]
                if len(new) or len(lack):
                    result = {'text': texts[idx],
                            'golds': [dict(zip(orders, triple)) for triple in gold_triples],
                            'preds': [dict(zip(orders, triple)) for triple in pre_triples],
                            'new': new,
                            'lack': lack
                            }
                    error_result.append(result)
        writer.write(json.dumps(error_result,ensure_ascii=False) + '\n')
        writer.close()

        metrics = self.get_metrics(correct_num, predict_num, gold_num)
        self.log("f1",metrics["f1"], prog_bar=True)
        self.log("acc",metrics["precision"], prog_bar=True)
        self.log("recall",metrics["recall"], prog_bar=True)
        self.log("gold_num",metrics["gold_num"], prog_bar=True)
        self.log("predict_num",metrics["predict_num"], prog_bar=True)
        self.log("correct_num",metrics["correct_num"], prog_bar=True)
        self.epoch += 1

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
            output.append((sub, int(rel), obj))
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




    