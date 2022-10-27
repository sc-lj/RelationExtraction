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


class TPlinkerPytochLighting(pl.LightningModule):
    def __init__(self, args, tokenizer) -> None:
        super().__init__()
        self.args = args
        to_add_special_token = list()
        for special_token in [TYPE_START, TYPE_END, SPAN_START, SPOT_PROMPT, ASOC_PROMPT]:
            if special_token not in tokenizer.get_vocab():
                to_add_special_token += [special_token]
        tokenizer.add_special_tokens(
            {"additional_special_tokens": to_add_special_token})
        self.model = T5ForConditionalGeneration.from_pretrained(args.pretrain_path)
        self.model.resize_token_embeddings(len(tokenizer))
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

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if self.args .ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        decoded_preds = [self.postprocess_text(x) for x in decoded_preds]
        decoded_labels = [self.postprocess_text(x) for x in decoded_labels]

        result = get_extract_metrics(
            pred_lns=decoded_preds,
            tgt_lns=decoded_labels,
            label_constraint=record_schema,
            decoding_format=self.args .decoding_format,
        )

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def postprocess_text(self, x_str):
        # Clean `bos` `eos` `pad` for cleaned text
        for to_remove_token in self.to_remove_token_list:
            x_str = x_str.replace(to_remove_token, '')

        return x_str.strip()


def get_extract_metrics(pred_lns: List[str], tgt_lns: List[str], label_constraint: RecordSchema, decoding_format='tree'):
    predict_parser = get_predict_parser(decoding_schema=decoding_format, label_constraint=label_constraint)
    return eval_pred(
        predict_parser=predict_parser,
        gold_list=tgt_lns,
        pred_list=pred_lns
    )


decoding_format_dict = {
    'spotasoc': SpotAsocPredictParser,
}


def get_predict_parser(decoding_schema, label_constraint):
    return decoding_format_dict[decoding_schema](label_constraint=label_constraint)


def eval_pred(predict_parser, gold_list, pred_list, text_list=None, raw_list=None):
    well_formed_list, counter = predict_parser.decode(
        gold_list, pred_list, text_list, raw_list
    )

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
