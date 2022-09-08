import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel


class BertForACEBothOneDropoutSubNoNer(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.ner_classifier = nn.Linear(config.hidden_size*2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(
            config.hidden_size*2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(
            config.hidden_size*2, self.num_labels)

        self.alpha = torch.tensor(
            [config.alpha] + [1.0] * (self.num_labels-1), dtype=torch.float32)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, mentions=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, sub_positions=None, labels=None, ner_labels=None):
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
        # Add hidden states and attention if they are here
        outputs = (re_prediction_scores, ) + outputs[2:]

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(
                ignore_index=-1,  weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(
                re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = 0
            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)
        return outputs


class BertForACEBothOneDropoutSub(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(
            config.hidden_size*2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(
            config.hidden_size*2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(
            config.hidden_size*2, self.num_labels)

        self.alpha = torch.tensor(
            [config.alpha] + [1.0] * (self.num_labels-1), dtype=torch.float32)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, mentions=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, sub_positions=None, labels=None, ner_labels=None,):

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
        # Add hidden states and attention if they are here
        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[2:]

        if labels is not None:

            re_loss = loss_fct_re(
                re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(
                ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)
        return outputs


class PLMakerPytochLighting(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.model = BertForACEBothOneDropoutSubNoNer.from_pretrained(args.pretrain_path)
        self.loss_fct_re = CrossEntropyLoss(ignore_index=-1,  weight=self.alpha)
        self.loss_fct_ner = CrossEntropyLoss(ignore_index=-1)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'position_ids':   batch[2],
                  }

        labels = batch[5]
        ner_labels = batch[6]

        inputs['sub_positions'] = batch[3]
        if self.args.model_type.find('span') != -1:
            inputs['mention_pos'] = batch[4]
        if self.args.model_type.endswith('bertonedropoutnersub'):
            inputs['sub_ner_labels'] = batch[7]

        re_prediction_scores, ner_prediction_scores = self.model(**inputs)
        re_loss = self.loss_fct_re(
            re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
        if ner_prediction_scores is not None:
            ner_loss = self.loss_fct_ner(
                ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))
        else:
            ner_loss = 0
        loss = re_loss + ner_loss
        return loss

    def validation_step(self, batch, batch_ids):
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'position_ids':   batch[2],
                  }

        labels = batch[5]
        ner_labels = batch[6]

        inputs['sub_positions'] = batch[3]
        if self.args.model_type.find('span') != -1:
            inputs['mention_pos'] = batch[4]
        if self.args.model_type.endswith('bertonedropoutnersub'):
            inputs['sub_ner_labels'] = batch[7]

        re_prediction_scores, ner_prediction_scores = self.model(**inputs)

        return

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        if self.args.warmup_steps == -1:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(0.1*self.args.t_total), num_training_steps=self.args.t_total
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.t_total
            )
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optim_dict
