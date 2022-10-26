import torch
import math
import os
import torch.nn as nn
from UIE.utils import *
import pytorch_lightning as pl
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers import DataCollatorForSeq2Seq


class TPlinkerPytochLighting(pl.LightningModule):
    def __init__(self, args,tokenizer) -> None:
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


