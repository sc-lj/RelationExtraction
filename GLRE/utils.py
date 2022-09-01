import torch
from torch import nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.autograd import Variable


def sparse_mxs_to_torch_sparse_tensor(sparse_mxs):
    """
    Convert a list of scipy sparse matrix to a torch sparse tensor.
    """
    max_shape = 0
    for mx in sparse_mxs:
        max_shape = max(max_shape, mx.shape[0])
    b_index = []
    row_index = []
    col_index = []
    value = []
    for index, sparse_mx in enumerate(sparse_mxs):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        b_index.extend([index] * len(sparse_mx.row))
        row_index.extend(sparse_mx.row)
        col_index.extend(sparse_mx.col)
        value.extend(sparse_mx.data)
    indices = torch.from_numpy(
        np.vstack((b_index, row_index, col_index)).astype(np.int64))
    values = torch.FloatTensor(value)
    shape = torch.Size([len(sparse_mxs), max_shape, max_shape])
    return torch.sparse.FloatTensor(indices, values, shape)


def get_distance(e1_sentNo, e2_sentNo):
    distance = 10000
    for e1 in e1_sentNo:
        for e2 in e2_sentNo:
            distance = min(distance, abs(int(e2) - int(e1)))
    return distance


def find_cross(head, tail):
    """检查两个是否在同一个句子中
    Args:
        head (_type_): _description_
        tail (_type_): _description_

    Returns:
        _type_: _description_
    """
    non_cross = False
    for m1 in head:
        for m2 in tail:
            if m1['sent_id'] == m2['sent_id']:
                non_cross = True
    if non_cross:
        return 'NON-CROSS'
    else:
        return 'CROSS'


def split_n_pad(nodes, section, pad=0, return_mask=False):
    """split tensor and pad
    Args:
        nodes ([type]): [description]
        section ([type]): [description]
        pad (int, optional): [description]. Defaults to 0.
        return_mask (bool, optional): [description]. Defaults to False.
    Returns:
        [type]: [description]
    """
    assert nodes.shape[0] == sum(section.tolist()), print(
        nodes.shape[0], sum(section.tolist()))
    # 单独切分每个句子的word的表征，sent_num(句子数量)个[word_num(每个句子word数量),hidden_size]
    nodes = torch.split(nodes, section.tolist())
    # [sent_num,max_word_num,hidden_size]
    nodes = pad_sequence(nodes, batch_first=True, padding_value=pad)
    if not return_mask:
        return nodes
    else:
        max_v = max(section.tolist())
        temp_ = torch.arange(max_v).unsqueeze(
            0).repeat(nodes.size(0), 1).to(nodes)
        mask = (temp_ < section.unsqueeze(1))
        # mask = torch.zeros(nodes.size(0), max_v).to(nodes)
        # for index, sec in enumerate(section.tolist()):
        #    mask[index, :sec] = 1
        # assert (mask1==mask).all(), print(mask1)
        return nodes, mask


def rm_pad(input, lens, max_v=None):
    """
    :param input: [sent_num,max_word_num,hidden_size]
    :param lens: sent_num
    :return:
    """
    if max_v is None:
        max_v = lens.max()
    # [sent_num,max_word_num]
    temp_ = torch.arange(max_v).unsqueeze(
        0).repeat(lens.size(0), 1).to(input.device)
    remove_pad = (temp_ < lens.unsqueeze(1))
    return input[remove_pad]


def rm_pad_between(input, s_lens, e_lens, max_v):
    """
    :param input: batch_size * len * dim
    :param lens: batch_size
    :return:
    """
    temp_ = torch.arange(max_v).unsqueeze(0).repeat(
        s_lens.size(0), 1).to(input.device)
    # print(temp_ < e_lens.unsqueeze(1))
    remove_pad = (temp_ < e_lens.unsqueeze(1)) & (temp_ >= s_lens.unsqueeze(1))
    return input[remove_pad]


def pool(h, mask, type='max'):
    """AI is creating summary for pool

    Args:
        h ([type]): [description] shape:[mention_num,hidden_size]
        mask ([type]): shape:[mention_num,1]
        type (str, optional): [description]. Defaults to 'max'.

    Returns:
        [type]: [description]
    """
    if type == 'max':
        h = h.masked_fill(mask, -1e12)
        # [hidden_size]
        return torch.max(h, -2)[0]
    elif type == 'avg' or type == "mean":
        h = h.masked_fill(mask, 0)
        return h.sum(-2) / (mask.size(-2) - mask.float().sum(-2))
    elif type == "logsumexp":
        h = h.masked_fill(mask, -1e12)
        return torch.logsumexp(h, -2)
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(-2)


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class EmbedLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout, ignore=None, freeze=False, pretrained=None, mapping=None):
        """
        Args:
            num_embeddings: (tensor) number of unique items
            embedding_dim: (int) dimensionality of vectors
            dropout: (float) dropout rate
            trainable: (bool) train or not
            pretrained: (dict) pretrained embeddings
            mapping: (dict) mapping of items to unique ids
        """
        super(EmbedLayer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.freeze = freeze
        self.ignore = ignore

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=ignore)
        self.embedding.weight.requires_grad = not freeze

        if pretrained:
            self.load_pretrained(pretrained, mapping)

        self.drop = nn.Dropout(dropout)

    def load_pretrained(self, pretrained, mapping):
        """
        Args:
            weights: (dict) keys are words, values are vectors
            mapping: (dict) keys are words, values are unique ids
            trainable: (bool)

        Returns: updates the embedding matrix with pre-trained embeddings
        """
        # if self.freeze:
        pret_embeds = torch.zeros((self.num_embeddings, self.embedding_dim))
        # else:
        # pret_embeds = nn.init.normal_(torch.empty((self.num_embeddings, self.embedding_dim)))
        for word in mapping.keys():
            if word in pretrained:
                pret_embeds[mapping[word], :] = torch.from_numpy(
                    pretrained[word])
            elif word.lower() in pretrained:
                pret_embeds[mapping[word], :] = torch.from_numpy(
                    pretrained[word.lower()])
        self.embedding = self.embedding.from_pretrained(
            pret_embeds, freeze=self.freeze)  # , padding_idx=self.ignore

    def forward(self, xs):
        """
        Args:
            xs: (tensor) batchsize x word_ids

        Returns: (tensor) batchsize x word_ids x dimensionality
        """
        embeds = self.embedding(xs)
        if self.drop.p > 0:
            embeds = self.drop(embeds)

        return embeds


class Encoder(nn.Module):
    def __init__(self, input_size, rnn_size, num_layers, bidirectional, dropout):
        """
        Wrapper for LSTM encoder
        Args:
            input_size (int): the size of the input features
            rnn_size (int):
            num_layers (int):
            bidirectional (bool):
            dropout (float):
        Returns: outputs, last_outputs
        - **outputs** of shape `(batch, seq_len, hidden_size)`:
          tensor containing the output features `(h_t)`
          from the last layer of the LSTM, for each t.
        - **last_outputs** of shape `(batch, hidden_size)`:
          tensor containing the last output features
          from the last layer of the LSTM, for each t=seq_len.
        """
        super(Encoder, self).__init__()

        self.enc = nn.LSTM(input_size=input_size,
                           hidden_size=rnn_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)

        # the dropout "layer" for the output of the RNN
        self.drop = nn.Dropout(dropout)

        # define output feature size
        self.feature_size = rnn_size
        self.rnn_size = rnn_size

        if bidirectional:
            self.feature_size *= 2

        self.num_layers = num_layers
        self.bidirectional = bidirectional

    @staticmethod
    def sort(lengths):
        # indices that result in sorted sequence
        sorted_len, sorted_idx = lengths.sort()
        _, original_idx = sorted_idx.sort(0, descending=True)
        reverse_idx = torch.linspace(lengths.size(
            0) - 1, 0, lengths.size(0)).long()  # for big-to-small

        return sorted_idx, original_idx, reverse_idx

    def forward(self, embeds, lengths, hidden=None):
        """
        This is the heart of the model. This function, defines how the data
        passes through the network.
        Args:
            embs (tensor): word embeddings
            lengths (list): the lengths of each sentence
        Returns: the logits for each class
        """
        # sort sequence
        sorted_idx, original_idx, reverse_idx = self.sort(lengths)
        # pad - sort - pack
        embeds = nn.utils.rnn.pad_sequence(
            embeds, batch_first=True, padding_value=0)
        embeds = embeds[sorted_idx][reverse_idx]  # big-to-small
        embeds = self.drop(embeds)  # apply dropout for input
        packed = pack_padded_sequence(embeds, list(
            lengths[sorted_idx][reverse_idx].data), batch_first=True)

        self.enc.flatten_parameters()
        out_packed, (h, c) = self.enc(packed, hidden)
        if self.bidirectional:
            h = h.reshape(self.num_layers, 2, -1, self.rnn_size)[-1, :, :, :]
        else:
            h = h.reshape(self.num_layers, 1, -1, self.rnn_size)[-1, :, :, :]

        # unpack
        outputs, _ = pad_packed_sequence(out_packed, batch_first=True)

        # apply dropout to the outputs of the RNN
        outputs = self.drop(outputs)

        # unsort the list
        outputs = outputs[reverse_idx][original_idx][reverse_idx]
        h = h.permute(1, 0, 2)[reverse_idx][original_idx][reverse_idx]
        return outputs, h.view(-1, self.feature_size)


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, num_units, nlayers, bidir, dropout):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.LSTM(input_size_, output_size_,
                             1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)

        self.init_hidden = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.init_c = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])

        self.dropout = LockedDropout(dropout)
        self.nlayers = nlayers

        self.reset_parameters()

    @staticmethod
    def sort(lengths):
        # indices that result in sorted sequence
        sorted_len, sorted_idx = lengths.sort()
        _, original_idx = sorted_idx.sort(0, descending=True)
        reverse_idx = torch.linspace(lengths.size(
            0) - 1, 0, lengths.size(0)).long()  # for big-to-small

        return sorted_idx, original_idx, reverse_idx

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths):
        # sort sequence
        sorted_idx, original_idx, reverse_idx = self.sort(input_lengths)

        # pad - sort - pack
        input = nn.utils.rnn.pad_sequence(
            input, batch_first=True, padding_value=0)
        input = input[sorted_idx][reverse_idx]  # big-to-small

        bsz, slen = input.size(0), input.size(1)
        output = input
        lens = list(input_lengths[sorted_idx][reverse_idx].data)
        outputs = []
        hiddens = []

        for i in range(self.nlayers):
            hidden, c = self.get_init(bsz, i)

            output = self.dropout(output)

            output = pack_padded_sequence(output, lens, batch_first=True)
            self.rnns[i].flatten_parameters()
            output, (hidden, cn) = self.rnns[i](output, (hidden, c))

            output, _ = pad_packed_sequence(output, batch_first=True)
            if output.size(1) < slen:  # used for parallel
                padding = Variable(output.data.new(1, 1, 1).zero_())
                output = torch.cat([output, padding.expand(output.size(
                    0), slen - output.size(1), output.size(2))], dim=1)

            hiddens.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            outputs.append(output)

        # assert torch.equal(outputs[-1][reverse_idx][original_idx][reverse_idx], inpu)
        return outputs[-1][reverse_idx][original_idx][reverse_idx], hiddens[-1][reverse_idx][original_idx][reverse_idx]


class Classifier(nn.Module):
    def __init__(self, in_size, out_size, dropout):
        """
        Args:
            in_size: input tensor dimensionality
            out_size: outpout tensor dimensionality
            dropout: dropout rate
        """
        super(Classifier, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.lin = nn.Linear(in_features=in_size,
                             out_features=out_size,
                             bias=True)

    def forward(self, xs):
        """
        Args:
            xs: (tensor) batchsize x * x features

        Returns: (tensor) batchsize x * x class_size
        """
        if self.drop.p > 0:
            xs = self.drop(xs)

        xs = self.lin(xs)
        return xs
