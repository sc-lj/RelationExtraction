import torch
import math
import os
import torch.nn as nn
from UIE.utils import *
import pytorch_lightning as pl
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from UIE.scorer import *
from transformers.trainer_pt_utils import LabelSmoother
from UIE.constraint_decoder import get_constraint_decoder


class UIEPytochLighting(pl.LightningModule):
    def __init__(self, args, tokenizer, t5_model) -> None:
        super().__init__()
        self.args = args
        self.model = t5_model
        if args.record_schema and os.path.exists(args.record_schema):
            record_schema = RecordSchema.read_from_file(args.record_schema)
        else:
            record_schema = None

        if args.source_prefix is not None:
            if args.source_prefix == 'schema':
                prefix = PrefixGenerator.get_schema_prefix(schema=record_schema)
            elif args.source_prefix.startswith('meta'):
                prefix = ""
            else:
                prefix = args.source_prefix
        else:
            prefix = ""
        self.tokenizer = tokenizer
        self.args = args

        self.to_remove_token_list = list()
        if tokenizer.bos_token:
            self.to_remove_token_list += [tokenizer.bos_token]
        if tokenizer.eos_token:
            self.to_remove_token_list += [tokenizer.eos_token]
        if tokenizer.pad_token:
            self.to_remove_token_list += [tokenizer.pad_token]
            # Label smoothing by sum token loss, different from different Label smootheing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None
        if args.record_schema and os.path.exists(args.record_schema):
            self.record_schema = RecordSchema.read_from_file(args.record_schema)
        else:
            self.record_schema = None

        self.decoding_type_schema = self.record_schema
        if args.constraint_decoding:
            self.constraint_decoder = get_constraint_decoder(tokenizer=self.tokenizer,
                                                             type_schema=self.decoding_type_schema,
                                                             source_prefix=prefix)
        else:
            self.constraint_decoder = None
        self.save_hyperparameters(args)
        self.epoch = 0

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, inputs, batch_idx):
        labels = inputs.pop("labels")
        outputs = self.model(**inputs)
        loss = self.label_smoother(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        def prefix_allowed_tokens_fn(batch_id, sent):
            src_sentence = batch['input_ids'][batch_id]
            return self.constraint_decoder.constraint_decoding(src_sentence=src_sentence,
                                                               tgt_generated=sent)
        gen_kwargs = {
            "max_length": self.args.max_target_length,
            "num_beams": self.args.num_beams,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn if self.constraint_decoder else None,
        }

        generated_tokens = self.model.generate(batch["input_ids"],attention_mask=batch["attention_mask"],**gen_kwargs)

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        outputs = self.model(**batch)

        if self.label_smoother is not None:
            loss = self.label_smoother(outputs, batch["labels"]).mean().detach()
        else:
            loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()

        labels = batch["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return loss.cpu().numpy(), generated_tokens.cpu().numpy(), labels.cpu().numpy()

    def validation_epoch_end(self, outputs):
        preds = []
        labels = []
        for los, pred_token, label in outputs:
            preds.append(pred_token)
            labels.append(label)

        labels = np.vstack(labels)
        preds = np.vstack(preds)
        metrics = self.compute_metrics([preds, labels])

        log_dir = [log.log_dir for log in self.loggers if hasattr(log, "log_dir")][0]
        os.makedirs(os.path.join(log_dir, "output"), exist_ok=True)
        output_eval_file = os.path.join(log_dir, "output", 'val_results_{}.txt'.format(self.epoch))

        with open(output_eval_file, "w") as writer:
            for key, value in sorted(metrics.items()):
                writer.write(f"{key} = {value}\n")

        eval_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        eval_preds = [self.postprocess_text(pred) for pred in eval_preds]

        output_preds_file = os.path.join(log_dir, "output", 'val_preds_{}.txt'.format(self.epoch))
        with open(output_preds_file, "w") as writer:
            writer.write("\n".join(eval_preds))

        self.epoch += 1

        self.log("f1", metrics["overall-F1"], prog_bar=True)
        self.log("sp_f1", metrics["spot-F1"], prog_bar=True)
        self.log("sp_P", metrics["spot-P"], prog_bar=True)
        self.log("sp_R", metrics["spot-R"], prog_bar=True)
        self.log("as_f1", metrics["asoc-F1"], prog_bar=True)
        self.log("as_P", metrics["asoc-P"], prog_bar=True)
        self.log("as_R", metrics["asoc-R"], prog_bar=True)
        self.log("re_f1", metrics["record-F1"], prog_bar=True)
        self.log("re_P", metrics["record-P"], prog_bar=True)
        self.log("re_R", metrics["record-R"], prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        def prefix_allowed_tokens_fn(batch_id, sent):
            # print(self.tokenizer.convert_ids_to_tokens(inputs['labels'][batch_id]))
            src_sentence = batch['input_ids'][batch_id]
            return self.constraint_decoder.constraint_decoding(src_sentence=src_sentence,
                                                               tgt_generated=sent)
        gen_kwargs = {
            "max_length": self.args.max_target_length,
            "num_beams": self.args.num_beams,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn if self.constraint_decoder else None,
        }

        generated_tokens = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **gen_kwargs,
        )

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        outputs = self.model(**batch)

        if self.label_smoother is not None:
            loss = self.label_smoother(outputs, batch["labels"]).mean().detach()
        else:
            loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()

        labels = batch["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return loss, generated_tokens, labels

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if self.args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        decoded_preds = [self.postprocess_text(x) for x in decoded_preds]
        decoded_labels = [self.postprocess_text(x) for x in decoded_labels]

        result = eval_pred(pred_lns=decoded_preds, tgt_lns=decoded_labels, label_constraint=self.record_schema)

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def postprocess_text(self, x_str):
        # Clean `bos` `eos` `pad` for cleaned text
        for to_remove_token in self.to_remove_token_list:
            x_str = x_str.replace(to_remove_token, '')

        return x_str.strip()

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones((tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device)
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

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


def eval_pred(pred_lns: List[str], tgt_lns: List[str], label_constraint: RecordSchema, text_list=None, raw_list=None):
    predict_parser = SpotAsocPredictParser(label_constraint=label_constraint)
    well_formed_list, counter = predict_parser.decode(tgt_lns, pred_lns, text_list, raw_list)

    spot_metric = Metric()
    asoc_metric = Metric()
    record_metric = RecordMetric()
    ordered_record_metric = OrderedRecordMetric()

    for instance in well_formed_list:
        spot_metric.count_instance(instance['gold_spot'], instance['pred_spot'])
        asoc_metric.count_instance(instance['gold_asoc'], instance['pred_asoc'])
        record_metric.count_instance(instance['gold_record'], instance['pred_record'])
        ordered_record_metric.count_instance(instance['gold_record'], instance['pred_record'])

    spot_result = spot_metric.compute_f1(prefix='spot-')
    asoc_result = asoc_metric.compute_f1(prefix='asoc-')
    record_result = record_metric.compute_f1(prefix='record-')
    ordered_record_result = ordered_record_metric.compute_f1(prefix='ordered-record-')

    overall_f1 = spot_result.get('spot-F1', 0.) + asoc_result.get('asoc-F1', 0.)
    # print(counter)
    result = {'overall-F1': overall_f1}
    result.update(spot_result)
    result.update(asoc_result)
    result.update(record_result)
    result.update(ordered_record_result)
    result.update(counter)
    return result
