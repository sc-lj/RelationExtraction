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


class UIEDataset(Dataset):
    def __init__(self, args, tokenizer, is_training=False):
        super().__init__()
        self.is_training = is_training
        self.tokenizer = tokenizer

        if args.record_schema and os.path.exists(args.record_schema):
            self.record_schema = RecordSchema.read_from_file(args.record_schema)
        else:
            self.record_schema = None

        if args.source_prefix is not None:
            if args.source_prefix == 'schema':
                self.prefix = PrefixGenerator.get_schema_prefix(schema=self.record_schema)
            elif args.source_prefix.startswith('meta'):
                self.prefix = ""
            else:
                self.prefix = args.source_prefix
        else:
            self.prefix = ""
        self.args = args
        self.max_target_length = args.max_target_length
        if is_training:
            filenames = os.path.join(args.post_data_path, "train.json")
        else:
            filenames = os.path.join(args.post_data_path, "val.json")
        with open(filenames, 'r') as f:
            self.examples = f.readlines()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        example = json.loads(example)
        inputs = self.prefix + example["text"]
        targets = example["record"]
        model_inputs = self.tokenizer(inputs, max_length=self.args.max_source_length, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        model_inputs['sample_prompt'] = [False] * len(model_inputs['input_ids'])
        if self.args.source_prefix is not None and self.args.source_prefix.startswith('meta'):
            model_inputs['spots'] = example['spot']  # 抽取的实体类型
            model_inputs['asocs'] = example['asoc']  # 该样本中抽取的实体之间的存在的关系
            model_inputs['spot_asoc'] = example['spot_asoc']  # 该样本中，实体和关系之间配对组合
            # sample_prompt=True for Finetune and Pretrain
            model_inputs['sample_prompt'] = [True] * len(model_inputs['input_ids'])
        if not self.is_training:
            # 对验证集
            model_inputs['sample_prompt'] = [False] * len(model_inputs['input_ids'])
        return model_inputs


class CollateFn():
    def __init__(self, args, tokenizer, model) -> None:
        args = args
        model = model
        label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        if args.record_schema and os.path.exists(args.record_schema):
            record_schema = RecordSchema.read_from_file(args.record_schema)
        else:
            record_schema = None

        if args.source_prefix.startswith('meta'):
            if args.spot_noise > 0 or args.asoc_noise > 0:
                spot_asoc_nosier = SpotAsocNoiser(
                    spot_noise_ratio=args.spot_noise,
                    asoc_noise_ratio=args.asoc_noise,
                    null_span=null_span,
                )
            else:
                spot_asoc_nosier = None

            self.data_collator = DataCollatorForMetaSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if args.float16 else None,
                max_length=args.max_source_length,
                max_prefix_length=args.max_prefix_length,
                max_target_length=args.max_target_length,
                negative_sampler=DynamicSSIGenerator(
                    tokenizer=tokenizer,
                    schema=record_schema,
                    positive_rate=args.meta_positive_rate,
                    negative=args.meta_negative,
                    ordered_prompt=args.ordered_prompt,
                ),
                spot_asoc_nosier=spot_asoc_nosier,
            )
        else:
            self.data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if args.float16 else None,
            )

    def __call__(self, features):
        return self.data_collator(features)


def add_special_token_tokenizer(pretrain_path):
    """为tokenizer中添加特殊符号
    Args:
        pretrain_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    tokenizer = T5TokenizerFast.from_pretrained(pretrain_path)
    to_remove_token_list = list()
    if tokenizer.bos_token:
        to_remove_token_list += [tokenizer.bos_token]
    if tokenizer.eos_token:
        to_remove_token_list += [tokenizer.eos_token]
    if tokenizer.pad_token:
        to_remove_token_list += [tokenizer.pad_token]
    to_add_special_token = list()
    for special_token in [TYPE_START, TYPE_END, TEXT_START, SPAN_START, SPOT_PROMPT, ASOC_PROMPT]:
        if special_token not in tokenizer.get_vocab():
            to_add_special_token += [special_token]
    tokenizer.add_tokens(tokenizer.additional_special_tokens + to_add_special_token, special_tokens=True)

    return tokenizer
