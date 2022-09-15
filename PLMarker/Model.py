import os
import json
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from collections import defaultdict
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel


class BertForACEBothOneDropoutSubNoNer(BertPreTrainedModel):
    """ 不进行ner的预测 """

    def __init__(self, config, args):
        super().__init__(config)
        self.max_seq_length = args.max_seq_length
        self.num_labels = args.num_labels
        self.num_ner_labels = args.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.ner_classifier = nn.Linear(config.hidden_size*2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.alpha = torch.tensor([args.alpha] + [1.0] * (self.num_labels-1), dtype=torch.float32)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, mentions=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, sub_positions=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len-seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len+ent_len]
        e2_hidden_states = hidden_states[:, seq_len+ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        # ner_prediction_scores = self.ner_classifier(self.dropout(feature_vector))

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        m1_scores = self.re_classifier_m1(m1_states)  # bsz, num_label
        m2_scores = self.re_classifier_m2(feature_vector)  # bsz, ent_len, num_label
        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        ner_prediction_scores = None
        return re_prediction_scores, ner_prediction_scores


class BertForACEBothOneDropoutSub(BertPreTrainedModel):
    """ 进行ner的预测 """

    def __init__(self, config, args):
        super().__init__(config)
        self.max_seq_length = args.max_seq_length
        self.num_labels = args.num_labels
        self.num_ner_labels = args.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # ner预测
        self.ner_classifier = nn.Linear(config.hidden_size*2, self.num_ner_labels)
        self.re_classifier_m1 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size*2, self.num_labels)

        self.alpha = torch.tensor([args.alpha] + [1.0] * (self.num_labels-1), dtype=torch.float32)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, sub_positions=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len-seq_len) // 2

        # [batch_size,ent_len,hidden_size]
        e1_hidden_states = hidden_states[:, seq_len:seq_len+ent_len]
        e2_hidden_states = hidden_states[:, seq_len+ent_len:]
        # [batch_size,ent_len,2*hidden_size]
        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        ner_prediction_scores = self.ner_classifier(feature_vector)

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        m1_scores = self.re_classifier_m1(m1_states)  # bsz, num_label
        m2_scores = self.re_classifier_m2(feature_vector)  # bsz, ent_len, num_label
        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores
        return re_prediction_scores, ner_prediction_scores


class PLMakerPytochLighting(pl.LightningModule):
    def __init__(self, args, tokenizer) -> None:
        super().__init__()
        self.golden_labels = args.golden_labels
        self.golden_labels_withner = args.golden_labels_withner
        self.ner_golden_labels = args.ner_golden_labels
        self.global_predicted_ners = args.global_predicted_ners
        self.tot_recall = args.tot_recall
        # 删除args中相关的属性，不用保存在参数列表中
        args.__delattr__("golden_labels")
        args.__delattr__("golden_labels_withner")
        args.__delattr__("global_predicted_ners")
        args.__delattr__("ner_golden_labels")
        args.__delattr__("tot_recall")

        self.args = args
        self.ner2id = json.load(open(os.path.join(args.data_dir, 'ner2id.json')))
        self.id2ner = {v: k for k, v in self.ner2id.items()}
        self.num_ner_labels = args.num_ner_labels
        self.num_labels = args.num_labels

        if args.m_type == "bertsub":
            self.model = BertForACEBothOneDropoutSub.from_pretrained(args.pretrain_path, args)
        elif args.m_type == "bertnonersub":
            self.model = BertForACEBothOneDropoutSubNoNer.from_pretrained(args.pretrain_path, args)
        else:
            raise ValueError(f"错误的{args.m_type},请填写bertnonersub,bertsub")
        self.loss_fct_re = CrossEntropyLoss(ignore_index=-1)
        self.loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
        relations = json.load(open(os.path.join(args.data_dir, 'rel2id.json')))
        self.relation_list = relations['relation']
        self.rel2index = {label: i for i, label in enumerate(self.relation_list)}
        self.num_label = len(self.relation_list)
        if args.lminit:
            word_embeddings = self.model.bert.embeddings.word_embeddings.weight.data
            subs, sube = 1, 2
            objs, obje = 3, 4
            subject_id = tokenizer.encode('subject', add_special_tokens=False)
            assert(len(subject_id) == 1)
            subject_id = subject_id[0]
            object_id = tokenizer.encode('object', add_special_tokens=False)
            assert(len(object_id) == 1)
            object_id = object_id[0]

            mask_id = tokenizer.encode('[MASK]', add_special_tokens=False)
            assert(len(mask_id) == 1)
            mask_id = mask_id[0]
            word_embeddings[subs].copy_(word_embeddings[mask_id])
            word_embeddings[sube].copy_(word_embeddings[subject_id])

            word_embeddings[objs].copy_(word_embeddings[mask_id])
            word_embeddings[obje].copy_(word_embeddings[object_id])
            # self.model.bert.embeddings.word_embeddings.weight.data = word_embeddings

        # 使用对某些关系采用双向识别，即处于关系下的triple对是无向的。
        if args.no_sym:  # 不对特定关系采用双向识别
            self.sym_labels = relations['no_sym']
        else:
            self.sym_labels = relations['no_sym'] + relations['sym']

        self.save_hyperparameters(args)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'position_ids':   batch[2],
                  'sub_positions': batch[3]
                  }

        labels = batch[5]
        ner_labels = batch[6]

        re_prediction_scores, ner_prediction_scores = self.model(**inputs)
        re_loss = self.loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
        if ner_prediction_scores is not None:
            ner_loss = self.loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))
        else:
            ner_loss = 0
        loss = re_loss + ner_loss
        return loss

    def validation_step(self, batch, batch_ids):
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'position_ids':   batch[2],
                  'sub_positions': batch[3]
                  }
        scores = defaultdict(dict)
        # ner_pred = not args.m_type.endswith('noner')
        example_subs = set([])

        subs = batch[-2]
        batch_m2s = batch[-1]
        indexs = batch[-3]
        ner_labels = batch[6]

        re_prediction_scores, ner_prediction_scores = self.model(**inputs)

        if self.args.eval_logsoftmax:  # perform a bit better
            logits = F.log_softmax(re_prediction_scores, dim=-1)

        elif self.args.eval_softmax:
            logits = F.softmax(re_prediction_scores, dim=-1)

        if self.args.use_ner_results or self.args.m_type.endswith('nonersub'):
            ner_preds = ner_labels
        else:
            ner_preds = torch.argmax(ner_prediction_scores, dim=-1)
        logits = logits.cpu().numpy()
        ner_preds = ner_preds.cpu().numpy()
        for i in range(len(indexs)):
            index = indexs[i]
            sub = subs[i]
            m2s = batch_m2s[i]
            example_subs.add(((index[0], index[1]), (sub[0], sub[1])))
            for j in range(len(m2s)):
                obj = m2s[j]
                ner_label = self.id2ner[ner_preds[i, j]]
                scores[(index[0], index[1])][((sub[0], sub[1]), (obj[0], obj[1]))] = (logits[i, j].tolist(), ner_label)

        return scores

    def validation_epoch_end(self, outputs):
        scores = {}
        for output in outputs:
            scores.update(output)
        cor = 0
        tot_pred = 0
        cor_with_ner = 0
        ner_cor = 0
        ner_tot_pred = 0
        ner_ori_cor = 0
        tot_output_results = defaultdict(list)
        for example_index, pair_dict in sorted(scores.items(), key=lambda x: x[0]):
            visited = set([])
            sentence_results = []
            for k1, (v1, v2_ner_label) in pair_dict.items():
                if k1 in visited:
                    continue
                visited.add(k1)
                if v2_ner_label == 'NIL':
                    continue
                v1 = list(v1)
                m1 = k1[0]
                m2 = k1[1]
                if m1 == m2:
                    continue
                k2 = (m2, m1)
                v2s = pair_dict.get(k2, None)
                if v2s is not None:
                    visited.add(k2)
                    v2, v1_ner_label = v2s
                    v2 = v2[: len(self.sym_labels)] + v2[self.num_label:] + v2[len(self.sym_labels): self.num_label]
                    for j in range(len(v2)):
                        v1[j] += v2[j]
                # else:
                #     assert (False)
                if v1_ner_label == 'NIL':
                    continue

                pred_label = np.argmax(v1)
                if pred_label > 0:
                    if pred_label >= self.num_label:
                        pred_label = pred_label - self.num_label + len(self.sym_labels)
                        m1, m2 = m2, m1
                        v1_ner_label, v2_ner_label = v2_ner_label, v1_ner_label
                    pred_score = v1[pred_label]
                    sentence_results.append((pred_score, m1, m2, pred_label, v1_ner_label, v2_ner_label))

            sentence_results.sort(key=lambda x: -x[0])
            no_overlap = []

            def is_overlap(m1, m2):
                if m2[0] <= m1[0] and m1[0] <= m2[1]:
                    return True
                if m1[0] <= m2[0] and m2[0] <= m1[1]:
                    return True
                return False

            output_preds = []
            for item in sentence_results:
                m1 = item[1]
                m2 = item[2]
                overlap = False
                for x in no_overlap:
                    _m1 = x[1]
                    _m2 = x[2]
                    # same relation type & overlap subject & overlap object --> delete
                    if item[3] == x[3] and (is_overlap(m1, _m1) and is_overlap(m2, _m2)):
                        overlap = True
                        break
                pred_label = self.relation_list[item[3]]
                if not overlap:
                    no_overlap.append(item)
            pos2ner = {}
            for item in no_overlap:
                m1 = item[1]
                m2 = item[2]
                pred_label = self.relation_list[item[3]]
                tot_pred += 1
                if pred_label in self.sym_labels:
                    tot_pred += 1  # duplicate
                    if (example_index, m1, m2, pred_label) in self.golden_labels or (example_index, m2, m1, pred_label) in self.golden_labels:
                        cor += 2
                else:
                    if (example_index, m1, m2, pred_label) in self.golden_labels:
                        cor += 1

                if m1 not in pos2ner:
                    pos2ner[m1] = item[4]
                if m2 not in pos2ner:
                    pos2ner[m2] = item[5]

                output_preds.append((m1, m2, pred_label))
                if pred_label in self.sym_labels:
                    if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in self.golden_labels_withner  \
                            or (example_index,  (m2[0], m2[1], pos2ner[m2]), (m1[0], m1[1], pos2ner[m1]), pred_label) in self.golden_labels_withner:
                        cor_with_ner += 2
                else:
                    if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in self.golden_labels_withner:
                        cor_with_ner += 1

            tot_output_results[example_index[0]].append((example_index[1],  output_preds))

            # refine NER results
            ner_results = list(self.global_predicted_ners[example_index])
            for i in range(len(ner_results)):
                start, end, label = ner_results[i]
                if (example_index, (start, end), label) in self.ner_golden_labels:
                    ner_ori_cor += 1
                if (start, end) in pos2ner:
                    label = pos2ner[(start, end)]
                if (example_index, (start, end), label) in self.ner_golden_labels:
                    ner_cor += 1
                ner_tot_pred += 1

        ner_p = ner_cor / ner_tot_pred if ner_tot_pred > 0 else 0
        ner_r = ner_cor / len(self.ner_golden_labels)
        ner_f1 = 2 * (ner_p * ner_r) / (ner_p + ner_r) if ner_cor > 0 else 0.0

        p = round(cor / tot_pred, 5) if tot_pred > 0 else 0
        recall = round(cor / self.tot_recall, 5)
        f1 = round(2 * (p * recall) / (p + recall), 5) if cor > 0 else 0.0
        assert(self.tot_recall == len(self.golden_labels))

        p_with_ner = cor_with_ner / tot_pred if tot_pred > 0 else 0
        r_with_ner = cor_with_ner / self.tot_recall
        assert(self.tot_recall == len(self.golden_labels_withner))
        f1_with_ner = 2 * (p_with_ner * r_with_ner) / (p_with_ner + r_with_ner) if cor_with_ner > 0 else 0.0

        results = {'f1':  f1,  'f1_with_ner': f1_with_ner, 'ner_f1': ner_f1}

        self.log("tot", float(self.tot_recall), prog_bar=True)
        self.log("cor", float(cor), prog_bar=True)
        self.log("pred", float(tot_pred), prog_bar=True)
        self.log("recall", float(recall), prog_bar=True)
        self.log("acc", float(p), prog_bar=True)
        self.log("f1", float(f1), prog_bar=True)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        if self.args.warmup_steps == -1:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*self.args.t_total), num_training_steps=self.args.t_total)
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.t_total)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optim_dict
