import os
import json
import copy
from tqdm import tqdm
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention
from torch.utils.data import DataLoader, Dataset
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast

class TPlinkerDataset(Dataset):
    def __init__(self, args,tokenizer, is_training=False):
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

