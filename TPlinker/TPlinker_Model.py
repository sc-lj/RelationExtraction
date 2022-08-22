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
from tqdm import tqdm
import math
from torch.nn import Parameter
import re
import os
from TPlinker.TPlinker_utils import HandshakingTaggingScheme, MetricsCalculator, DataMaker4Bert


class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
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
                self.hidden_dense = nn.Linear(
                    in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(
                    in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(
                    in_features=self.cond_dim, out_features=input_dim, bias=False)

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
            std = (variance + self.epsilon) ** 2
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
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
        elif shaking_type == "cln_plus":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
            self.inner_context_cln = LayerNorm(
                hidden_size, hidden_size, conditional=True)

        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = Parameter(torch.rand(hidden_size))
        elif inner_enc_type == "lstm":
            self.inner_context_lstm = nn.LSTM(hidden_size,
                                              hidden_size,
                                              num_layers=1,
                                              bidirectional=False,
                                              batch_first=True)

    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type="lstm"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = torch.mean(seqence, dim=-2)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim=-2)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * \
                    torch.mean(seqence, dim=-2) + (1 - self.lamtha) * \
                    torch.max(seqence, dim=-2)[0]
            return pooling
        if "pooling" in inner_enc_type:
            inner_context = torch.stack(
                [pool(seq_hiddens[:, :i+1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])], dim=1)
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)

        return inner_context

    def forward(self, seq_hiddens):
        ''' Handshaking Kernel 的机制，当前字符与后续字符的所有组合的表征。
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: (batch_size, (1 + seq_len) * \
                               seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        '''
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            visible_hiddens = seq_hiddens[:, ind:, :]  # ind: only look back
            # 将hidden_each_step shape转换为与visible_hiddens一致
            repeat_hiddens = hidden_each_step[:, None, :].repeat(
                1, seq_len - ind, 1)
            # 当前ind 位置的token的表征与该位置后面的所有token的表征的交互方式
            if self.shaking_type == "cat":  # 直接拼接
                shaking_hiddens = torch.cat(
                    [repeat_hiddens, visible_hiddens], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_plus":
                # 对该位置后的所有token的表征进行二次提取
                inner_context = self.enc_inner_hiddens(
                    visible_hiddens, self.inner_enc_type)
                shaking_hiddens = torch.cat(
                    [repeat_hiddens, visible_hiddens, inner_context], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cln":
                # 使用layernormal进行特征提取
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
            elif self.shaking_type == "cln_plus":
                # 二次交互后，再加layer normal
                inner_context = self.enc_inner_hiddens(
                    visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
                shaking_hiddens = self.inner_context_cln(
                    shaking_hiddens, inner_context)

            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim=1)
        return long_shaking_hiddens


class TPLinkerPlusBert(nn.Module):
    def __init__(self, bert_path,
                 tag_size,
                 shaking_type,
                 inner_enc_type,
                 tok_pair_sample_rate=1):
        super().__init__()
        self.encoder = BertModel.from_pretrained(bert_path)
        self.tok_pair_sample_rate = tok_pair_sample_rate

        shaking_hidden_size = self.encoder.config.hidden_size

        self.fc = nn.Linear(shaking_hidden_size, tag_size)

        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(
            shaking_hidden_size, shaking_type, inner_enc_type)

    def forward(self, input_ids,
                attention_mask,
                token_type_ids
                ):
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(
            input_ids, attention_mask, token_type_ids)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        seq_len = last_hidden_state.size()[1]
        # shaking_hiddens: (batch_size, shaking_seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(last_hidden_state)

        sampled_tok_pair_indices = None
        if self.training:
            # randomly sample segments of token pairs
            # 对Handshaking Kernel 的序列分段，并随机采样
            shaking_seq_len = shaking_hiddens.size()[1]
            # 每段的长度
            segment_len = int(shaking_seq_len * self.tok_pair_sample_rate)
            # 可以分多少段
            seg_num = math.ceil(shaking_seq_len // segment_len)
            # 随机取一段，并获取该段的start 索引
            start_ind = torch.randint(seg_num, []) * segment_len
            end_ind = min(start_ind + segment_len, shaking_seq_len)
            # sampled_tok_pair_indices: (batch_size, ~segment_len) ~end_ind - start_ind <= segment_len
            # 该段在原序列的索引
            sampled_tok_pair_indices = torch.arange(start_ind, end_ind)[
                None, :].repeat(shaking_hiddens.size()[0], 1)
            # sampled_tok_pair_indices = torch.randint(shaking_seq_len, (shaking_hiddens.size()[0], segment_len))
            sampled_tok_pair_indices = sampled_tok_pair_indices.to(
                shaking_hiddens.device)

            # sampled_tok_pair_indices will tell model what token pairs should be fed into fcs
            # shaking_hiddens: (batch_size, ~segment_len, hidden_size)
            # 获取该段的handshaking表征
            shaking_hiddens = shaking_hiddens.gather(
                1, sampled_tok_pair_indices[:, :, None].repeat(1, 1, shaking_hiddens.size()[-1]))

        # outputs: (batch_size, segment_len, tag_size) or (batch_size, shaking_seq_len, tag_size)
        outputs = self.fc(shaking_hiddens)

        return outputs, sampled_tok_pair_indices


class TPlinkerPytochLighting(pl.LightningModule):
    def __init__(self, args, handshaking_tagger) -> None:
        super().__init__()
        self.args = args
        self.metrics = MetricsCalculator(handshaking_tagger)
        self.tag_size = handshaking_tagger.get_tag_size()  # handshake 构建的标签序列长度
        self.model = TPLinkerPlusBert(args.pretrain_path, self.tag_size,
                                      args.shaking_type, args.inner_enc_type, args.tok_pair_sample_rate)
        self.step = -1
        self.num_step = args.num_step
        self.save_hyperparameters(args)

    def training_step(self, batch, idx):
        sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, batch_shaking_tag = batch
        pred_small_shaking_outputs, sampled_tok_pair_indices = self.model(
            batch_input_ids, batch_attention_mask, batch_token_type_ids)
        # 采样段对应的标签
        batch_small_shaking_tag = batch_shaking_tag.gather(
            1, sampled_tok_pair_indices[:, :, None].repeat(1, 1, self.tag_size))
        loss = self.loss_func(pred_small_shaking_outputs,
                              batch_small_shaking_tag)
        return loss

    def loss_func(self, y_pred, y_true):
        return self.metrics.loss_func(y_pred, y_true, ghm=self.args.ghm)

    def validation_step(self, batch, step_idx):
        sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, batch_shaking_tag = batch
        pred_shaking_outputs, _ = self.model(batch_input_ids,
                                             batch_attention_mask,
                                             batch_token_type_ids)

        pred_shaking_tag = (pred_shaking_outputs > 0.).long()
        sample_acc = self.metrics.get_sample_accuracy(pred_shaking_tag,
                                                      batch_shaking_tag)
        if self.step <= 0:
            self.step += 1
            return sample_acc.item(), {}, 1
        cpg_dict = self.metrics.get_cpg(sample_list,
                                        tok2char_span_list,
                                        pred_shaking_tag,
                                        self.args.match_pattern)
        return sample_acc.item(), cpg_dict, len(sample_list)

    def validation_epoch_end(self, outputs):
        total_sample_acc = 0.
        total_cpg_dict = {}
        number = 0
        for sample_acc, cpg_dict, num in outputs:
            total_sample_acc += sample_acc
            # init total_cpg_dict
            for k in cpg_dict.keys():
                if k not in total_cpg_dict:
                    total_cpg_dict[k] = [0, 0, 0]

            for k, cpg in cpg_dict.items():
                for idx, n in enumerate(cpg):
                    total_cpg_dict[k][idx] += cpg[idx]
            number += num
        if len(total_cpg_dict) == 0:
            print("total_cpg_dict:", total_cpg_dict)
            return
        avg_sample_acc = total_sample_acc / number
        rel_prf = self.metrics.get_prf_scores(
            total_cpg_dict["rel_cpg"][0], total_cpg_dict["rel_cpg"][1], total_cpg_dict["rel_cpg"][2])
        ent_prf = self.metrics.get_prf_scores(
            total_cpg_dict["ent_cpg"][0], total_cpg_dict["ent_cpg"][1], total_cpg_dict["ent_cpg"][2])
        final_score = rel_prf[2]
        log_dict = {
            "val_shaking_tag_acc": avg_sample_acc,
            "prec": rel_prf[0],
            "recall": rel_prf[1],
            "f1": rel_prf[2],
            "val_ent_prec": ent_prf[0],
            "val_ent_recall": ent_prf[1],
            "val_ent_f1": ent_prf[2]}
        self.log("f1", log_dict["f1"], prog_bar=True)
        self.log("acc", log_dict["prec"], prog_bar=True)
        self.log("recall", log_dict["recall"], prog_bar=True)
        self.log("ent_f1", log_dict["val_ent_f1"], prog_bar=True)
        self.log("ent_prec", log_dict["val_ent_prec"], prog_bar=True)
        self.log("ent_rec", log_dict["val_ent_recall"], prog_bar=True)
        self.log("tag_acc", log_dict["val_shaking_tag_acc"], prog_bar=True)

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
        StepLR = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.85)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", verbose = True, patience = 6)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.args.decay_steps, gamma=self.args.decay_rate)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.num_step * self.args.rewarm_epoch_num, self.args.T_mult)
        # StepLR = WarmupLR(optimizer,25000)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optim_dict


class TPlinkerDataset(Dataset):
    def __init__(self, args, data_maker: DataMaker4Bert, tokenizer, is_training=False):
        super().__init__()
        self.is_training = is_training
        self.datas = []
        self._tokenize = tokenizer.tokenize
        self.get_tok2char_span_map = lambda text: tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]

        self.args = args

        if is_training:
            self.data_type = "train"
        else:
            self.data_type = "val"

        data_path = os.path.join(
            self.args.data_out_dir, "{}.json".format(self.data_type))

        with open(data_path, 'r') as f:
            data = json.load(f)

        # data = self.split_into_short_samples(
        #     data, max_seq_len=self.args.max_seq_len, data_type=self.data_type)

        self.datas = data_maker.get_indexed_data(data, args.max_seq_len)

    def split_into_short_samples(self, sample_list, max_seq_len, sliding_len=20, encoder="BERT", data_type="train"):
        """当max_seq_len小于实际的文本长度时，对实体和样本进行截断。
        Args:
            sample_list (_type_): _description_
            max_seq_len (_type_): _description_
            sliding_len (int, optional): _description_. Defaults to 20.
            encoder (str, optional): _description_. Defaults to "BERT".
            data_type (str, optional): _description_. Defaults to "train".
        Returns:
            _type_: _description_
        """
        new_sample_list = []
        for sample in tqdm(sample_list, desc="Splitting into subtexts"):
            text_id = sample["id"]
            text = sample["text"]
            tokens = self._tokenize(text)
            tok2char_span = self.get_tok2char_span_map(text)

            # sliding at token level
            split_sample_list = []
            for start_ind in range(0, len(tokens), sliding_len):
                if encoder == "BERT":  # if use bert, do not split a word into two samples
                    while "##" in tokens[start_ind]:
                        start_ind -= 1
                end_ind = start_ind + max_seq_len

                char_span_list = tok2char_span[start_ind:end_ind]
                char_level_span = [char_span_list[0][0], char_span_list[-1][1]]
                sub_text = text[char_level_span[0]:char_level_span[1]]

                new_sample = {
                    "id": text_id,
                    "text": sub_text,
                    "tok_offset": start_ind,
                    "char_offset": char_level_span[0],
                }
                if data_type == "test":  # test set
                    if len(sub_text) > 0:
                        split_sample_list.append(new_sample)
                else:
                    # train or valid dataset, only save spo and entities in the subtext
                    # spo
                    sub_rel_list = []
                    for rel in sample["relation_list"]:
                        subj_tok_span = rel["subj_tok_span"]
                        obj_tok_span = rel["obj_tok_span"]
                        # if subject and object are both in this subtext, add this spo to new sample
                        if subj_tok_span[0] >= start_ind and subj_tok_span[1] <= end_ind \
                                and obj_tok_span[0] >= start_ind and obj_tok_span[1] <= end_ind:
                            new_rel = copy.deepcopy(rel)
                            # start_ind: tok level offset
                            new_rel["subj_tok_span"] = [
                                subj_tok_span[0] - start_ind, subj_tok_span[1] - start_ind]
                            new_rel["obj_tok_span"] = [
                                obj_tok_span[0] - start_ind, obj_tok_span[1] - start_ind]
                            # char level offset
                            new_rel["subj_char_span"][0] -= char_level_span[0]
                            new_rel["subj_char_span"][1] -= char_level_span[0]
                            new_rel["obj_char_span"][0] -= char_level_span[0]
                            new_rel["obj_char_span"][1] -= char_level_span[0]
                            sub_rel_list.append(new_rel)

                    # entity
                    sub_ent_list = []
                    for ent in sample["entity_list"]:
                        tok_span = ent["tok_span"]
                        # if entity in this subtext, add the entity to new sample
                        if tok_span[0] >= start_ind and tok_span[1] <= end_ind:
                            new_ent = copy.deepcopy(ent)
                            new_ent["tok_span"] = [tok_span[0] -
                                                   start_ind, tok_span[1] - start_ind]

                            new_ent["char_span"][0] -= char_level_span[0]
                            new_ent["char_span"][1] -= char_level_span[0]

                            sub_ent_list.append(new_ent)

                    new_sample["entity_list"] = sub_ent_list  # maybe empty
                    new_sample["relation_list"] = sub_rel_list  # maybe empty
                    split_sample_list.append(new_sample)

                # all segments covered, no need to continue
                if end_ind > len(tokens):
                    break

            new_sample_list.extend(split_sample_list)
        return new_sample_list

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]
