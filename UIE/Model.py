import torch
import math
import torch.nn as nn
import pytorch_lightning as pl
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers import DataCollatorForSeq2Seq




class TPlinkerPytochLighting(pl.LightningModule):
    def __init__(self, args,tokenizer) -> None:
        super().__init__()
        self.args = args
        self.model = T5ForConditionalGeneration.from_pretrained(args.pretrain_path)
        self.model.resize_token_embeddings(len(tokenizer))

