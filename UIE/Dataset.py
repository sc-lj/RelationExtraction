import os
import json
import copy
from tqdm import tqdm
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention
from torch.utils.data import DataLoader, Dataset
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast
from UIE.utils import RecordSchema, PrefixGenerator, TYPE_START, TYPE_END, TEXT_START, SPAN_START, SPOT_PROMPT, ASOC_PROMPT, null_span
from UIE.utils import SpotAsocNoiser, DynamicSSIGenerator, DataCollatorForMetaSeq2Seq
from transformers import DataCollatorForSeq2Seq


class TPlinkerDataset(Dataset):
    def __init__(self, args, is_training=False):
        super().__init__()
        self.is_training = is_training
        self.tokenizer = T5TokenizerFast.from_pretrained(args.pretain_path)
        to_remove_token_list = list()
        if self.tokenizer.bos_token:
            to_remove_token_list += [self.tokenizer.bos_token]
        if self.tokenizer.eos_token:
            to_remove_token_list += [self.tokenizer.eos_token]
        if self.tokenizer.pad_token:
            to_remove_token_list += [self.tokenizer.pad_token]
        if is_training:
            to_add_special_token = list()
            for special_token in [TYPE_START, TYPE_END, TEXT_START, SPAN_START, SPOT_PROMPT, ASOC_PROMPT]:
                if special_token not in self.tokenizer.get_vocab():
                    to_add_special_token += [special_token]

            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": self.tokenizer.special_tokens_map_extended['additional_special_tokens'] + to_add_special_token}
            )
        self.tokenizer_len = len(self.tokenizer)

        if args.record_schema and os.path.exists(args.record_schema):
            record_schema = RecordSchema.read_from_file(args.record_schema)
        else:
            record_schema = None

        if args.source_prefix is not None:
            if args.source_prefix == 'schema':
                self.prefix = PrefixGenerator.get_schema_prefix(schema=record_schema)
            elif args.source_prefix.startswith('meta'):
                self.prefix = ""
            else:
                self.prefix = args.source_prefix
        else:
            self.prefix = ""
        self.max_target_length = args.max_target_length
        self.padding = args.padding
        self.label_pad_token_id = -100 if self.args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id

    def __getitem__(self, index):
        example = examples[index]
        inputs = example["text"]
        targets = example["record"]
        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs, max_length=self.args.max_source_length, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        model_inputs['sample_prompt'] = [False] * len(model_inputs['input_ids'])
        if self.args.source_prefix is not None and self.args.source_prefix.startswith('meta'):
            model_inputs['spots'] = example['spot']
            model_inputs['asocs'] = example['asoc']
            model_inputs['spot_asoc'] = example['spot_asoc']
            # sample_prompt=True for Finetune and Pretrain
            model_inputs['sample_prompt'] = [True] * len(model_inputs['input_ids'])
        if not self.is_training:
            # 对验证集
            model_inputs['sample_prompt'] = [False] * len(model_inputs['input_ids'])
        return model_inputs

    def collate_fn(self,):
        if self.args.source_prefix.startswith('meta'):
            if self.args.spot_noise > 0 or self.args.asoc_noise > 0:
                if self.args.decoding_format == 'spotasoc':
                    spot_asoc_nosier = SpotAsocNoiser(
                        spot_noise_ratio=self.args.spot_noise,
                        asoc_noise_ratio=self.args.asoc_noise,
                        null_span=null_span,
                    )
                else:
                    raise NotImplementedError(
                        f"decoding_format {self.args.decoding_format} is not implemented."
                    )
            else:
                spot_asoc_nosier = None

            data_collator = DataCollatorForMetaSeq2Seq(
                self.tokenizer,
                model=model,
                label_pad_token_id=self.label_pad_token_id,
                pad_to_multiple_of=8 if self.args.fp16 else None,
                max_length=self.args.max_source_length,
                max_prefix_length=self.args.max_prefix_length,
                max_target_length=self.args.max_target_length,
                negative_sampler=DynamicSSIGenerator(
                    tokenizer=self.tokenizer,
                    schema=record_schema,
                    positive_rate=self.args.meta_positive_rate,
                    negative=self.args.meta_negative,
                    ordered_prompt=self.args.ordered_prompt,
                ),
                spot_asoc_nosier=spot_asoc_nosier,
                decoding_format=self.args.decoding_format,
            )
        else:
            data_collator = DataCollatorForSeq2Seq(
                self.tokenizer,
                model=model,
                label_pad_token_id=self.label_pad_token_id,
                pad_to_multiple_of=8 if self.args.fp16 else None,
            )
