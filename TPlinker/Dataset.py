import os   
import json
import copy
from tqdm import tqdm
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention
from torch.utils.data import DataLoader, Dataset
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from TPlinker.utils import HandshakingTaggingScheme, MetricsCalculator, DataMaker4Bert

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

