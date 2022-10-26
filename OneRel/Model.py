import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel
import torch
import pytorch_lightning as pl
import numpy as np
import json
import os
from utils.utils import rematch
from utils.loss_func import MultiCEFocalLoss
from OneRel.utils import TAG2ID


class OneRelModel(nn.Module):
    def __init__(self, config):
        super(OneRelModel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(
            config.pretrain_path, cache_dir='./bertbaseuncased')
        self.bert_dim = self.bert.config.hidden_size
        self.relation_matrix = nn.Linear(
            self.bert_dim * 3, self.config.relation_number * self.config.tag_size)
        self.projection_matrix = nn.Linear(
            self.bert_dim * 2, self.bert_dim * 3)

        self.dropout = nn.Dropout(self.config.dropout_prob)
        self.dropout_2 = nn.Dropout(self.config.entity_pair_dropout)
        self.activation = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.bert(
            input_ids, attention_mask=attention_mask)[0]
        encoded_text = self.dropout(encoded_text)
        # encoded_text: [batch_size, seq_len, bert_dim(768)] 1,2,3
        batch_size, seq_len, bert_dim = encoded_text.size()
        # head: [batch_size, seq_len * seq_len, bert_dim(768)] 1,1,1, 2,2,2, 3,3,3
        head_representation = encoded_text.unsqueeze(2).expand(
            batch_size, seq_len, seq_len, bert_dim).reshape(batch_size, seq_len*seq_len, bert_dim)
        # tail: [batch_size, seq_len * seq_len, bert_dim(768)] 1,2,3, 1,2,3, 1,2,3
        tail_representation = encoded_text.repeat(1, seq_len, 1)
        # [batch_size, seq_len * seq_len, bert_dim(768)*2]
        entity_pairs = torch.cat(
            [head_representation, tail_representation], dim=-1)

        # [batch_size, seq_len * seq_len, bert_dim(768)*3]
        entity_pairs = self.projection_matrix(entity_pairs)
        entity_pairs = self.dropout_2(entity_pairs)
        entity_pairs = self.activation(entity_pairs)

        # [batch_size, seq_len * seq_len, rel_num * tag_size] -> [batch_size, seq_len, seq_len, rel_num, tag_size]
        output = self.relation_matrix(entity_pairs).reshape(
            batch_size, seq_len, seq_len, self.config.relation_number, self.config.tag_size)
        return output


class OneRelPytochLighting(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.model = OneRelModel(args)
        self.save_hyperparameters(args)
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.focal_loss = MultiCEFocalLoss(self.args.tag_size)
        with open(os.path.join(args.data_dir, "rel2id.json"), 'r') as f:
            relation = json.load(f)
        self.id2rel = relation[0]
        self.epoch = 0

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def training_step(self, batches, batch_idx):
        batch_token_ids = batches['token_ids']
        batch_attention_masks = batches['mask']
        batch_loss_masks = batches['loss_mask']
        triple_matrix = batches['triple_matrix']
        # [batch_size, seq_len, seq_len, rel_num, tag_size]
        outputs = self.model(batch_token_ids, batch_attention_masks)
        # [batch_size, tag_size, rel_num, seq_len, seq_len]
        outputs = outputs.permute(0, 4, 3, 1, 2)
        loss = self.loss(outputs, triple_matrix)
        loss = torch.sum(loss * batch_loss_masks) / torch.sum(batch_loss_masks)
        focal_loss = self.focal_loss(outputs, triple_matrix)
        focal_loss = torch.sum(
            focal_loss * batch_loss_masks) / torch.sum(batch_loss_masks)
        return loss+focal_loss

    def validation_step(self, batches, batch_idx):
        batch_token_ids = batches['token_ids']
        batch_attention_masks = batches['mask']
        batch_loss_masks = batches['loss_mask']
        triple_matrix = batches['triple_matrix']
        triples = batches['triples']
        tokens = batches['tokens']
        texts = batches['texts']
        offset_maps = batches['offset_map']
        # [batch_size, seq_len, seq_len, rel_num, tag_size]
        pred_triple_matrix = self.model(batch_token_ids, batch_attention_masks)
        # [batch_size, seq_len, seq_len, rel_num]->[batch_size, rel_num, seq_len, seq_len]
        pred_triple_matrix = pred_triple_matrix.argmax(
            dim=-1).permute(0, 3, 1, 2)
        pred_triples = self.parse_prediction(
            pred_triple_matrix, batch_loss_masks, texts, offset_maps, triples)
        return pred_triples, triples, texts

    def parse_prediction(self, pred_triple_matrix, batch_loss_masks, texts, offset_maps, triples):
        batch_size, rel_numbers, seq_lens, seq_lens = pred_triple_matrix.shape
        batch_triple_list = []
        for batch in range(batch_size):
            mapping = rematch(offset_maps[batch])
            triple = triples[batch]
            triple_matrix = pred_triple_matrix[batch].cpu().numpy()
            masks = batch_loss_masks[batch].cpu().numpy()
            triple_matrix = triple_matrix*masks
            text = texts[batch]
            triple_list = []
            for r_index in range(rel_numbers):
                rel_triple_matrix = triple_matrix[r_index]
                heads, tails = np.where(rel_triple_matrix > 0)
                pair_numbers = len(heads)
                rel_triple = rel_triple_matrix[(heads, tails)]
                # if pair_numbers>0:
                #     print(r_index,heads, tails,rel_triple)
                rel = self.id2rel[str(int(r_index))]
                for i in range(pair_numbers):
                    h_start_index = heads[i]
                    t_start_index = tails[i]
                    # 如果当前第一个标签为HB-TB,即subject begin，object begin
                    if rel_triple_matrix[h_start_index][t_start_index] == TAG2ID['HB-TB']:
                        # 如果下一个标签为HB-TE,即subject begin，object end
                        find_hb_te = False
                        if i+1 < pair_numbers:
                            t_end_index = tails[i+1]
                            if rel_triple_matrix[h_start_index][t_end_index] == TAG2ID['HB-TE']:
                                # 那么就向下找
                                find_hb_te = True
                                find_he_te = False
                                for h_end_index in range(h_start_index, seq_lens):
                                    # 向下找到了结尾位置,即subject end，object end
                                    if rel_triple_matrix[h_end_index][t_end_index] == TAG2ID['HE-TE']:
                                        sub = self.decode_entity(
                                            text, mapping, h_start_index, h_end_index)
                                        obj = self.decode_entity(
                                            text, mapping, t_start_index, t_end_index)

                                        if len(sub) > 0 and len(obj) > 0:
                                            triple_list.append((sub, rel, obj))
                                            find_he_te = True
                                        # break
                                if not find_he_te:
                                    # subject是单个词
                                    h_end_index = h_start_index
                                    sub = self.decode_entity(
                                        text, mapping, h_start_index, h_end_index)
                                    obj = self.decode_entity(
                                        text, mapping, t_start_index, t_end_index)
                                    if len(sub) > 0 and len(obj) > 0:
                                        triple_list.append((sub, rel, obj))
                        # object 是单个词
                        if not find_hb_te:
                            t_end_index = t_start_index
                            find_he_te = False
                            for h_end_index in range(h_start_index, seq_lens):
                                # 向下找到了结尾位置,即subject end，object end
                                if rel_triple_matrix[h_end_index][t_end_index] == TAG2ID['HE-TE']:
                                    sub = self.decode_entity(
                                        text, mapping, h_start_index, h_end_index)
                                    obj = self.decode_entity(
                                        text, mapping, t_start_index, t_end_index)
                                    if len(sub) > 0 and len(obj) > 0:
                                        triple_list.append((sub, rel, obj))
                                        find_he_te = True
                                    # break
                            if not find_he_te:
                                # subject是单个词，且object 是单个词
                                h_end_index = h_start_index
                                sub = self.decode_entity(
                                    text, mapping, h_start_index, h_end_index)
                                obj = self.decode_entity(
                                    text, mapping, t_start_index, t_end_index)
                                if len(sub) > 0 and len(obj) > 0:
                                    triple_list.append((sub, rel, obj))
            batch_triple_list.append(triple_list)
        return batch_triple_list

    def decode_entity(self, text: str, mapping, start: int, end: int):
        s = mapping[start]
        e = mapping[end]
        s = 0 if not s else s[0]
        e = len(text) - 1 if not e else e[-1]
        entity = text[s: e + 1]
        return entity

    def validation_epoch_end(self, outputs):
        preds, targets, texts = [], [], []
        for pred, target, text in outputs:
            preds.extend(pred)
            targets.extend(target)
            texts.extend(text)

        correct = 0
        predict = 0
        total = 0
        orders = ['subject', 'relation', 'object']

        log_dir = [log.log_dir for log in self.loggers if hasattr(log,"log_dir")][0]
        os.makedirs(os.path.join(log_dir,"output"), exist_ok=True)
        writer = open(os.path.join(log_dir, "output",'val_output_{}.json'.format(self.epoch)), 'w')
        
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
