# -*- encoding: utf-8 -*-
'''
@File    :   GLRE_Model.py
@Time    :   2022/08/26 15:43:52
@Author  :   lujun
@Version :   1.0
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   文档级关系抽取算法
'''


import torch
import torch.nn as nn
from Attention import *
from GLRE.utils import *
# from GLRE.config import *
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


class RGCN_Layer(nn.Module):
    """ A Relation GCN module operated on documents graphs. """

    def __init__(self, in_dim, mem_dim, num_layers, gcn_in_drop, gcn_out_drop, relation_cnt=5):
        """
        Args:
            in_dim (_type_): GCN layer 输入的维度
            mem_dim (_type_): GCN layer 中间层以及输出的维度
            num_layers (_type_): GCN layer 的层数
            relation_cnt (int, optional): 边的关系数量. Defaults to 5.
        """
        super().__init__()
        self.layers = num_layers
        self.device = torch.device("cuda")
        self.mem_dim = mem_dim
        self.relation_cnt = relation_cnt
        self.in_dim = in_dim

        self.in_drop = nn.Dropout(gcn_in_drop)
        self.gcn_drop = nn.Dropout(gcn_out_drop)

        # gcn layer
        self.W_0 = nn.ModuleList()
        self.W_r = nn.ModuleList()
        # 为每个关系构建相应的多个权重矩阵
        for i in range(relation_cnt):
            self.W_r.append(nn.ModuleList())
        # 每层的GCN矩阵
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W_0.append(nn.Linear(input_dim, self.mem_dim).to(self.device))
            for W in self.W_r:
                W.append(nn.Linear(input_dim, self.mem_dim).to(self.device))

    def forward(self, nodes, adj, section):
        """
        Args:
            nodes (_type_): [batch_size,node_size,node_emb] 节点表征矩阵
            adj (_type_):  邻接矩阵rgcn模型的邻接矩阵,shape:[batch_size,5,node_size,node_size],5是5种边
            section (_type_): (Tensor <B, 3>) #entities/#mentions/#sentences per batch
        Returns:
            _type_: _description_
        """
        gcn_inputs = self.in_drop(nodes)
        maskss = []
        denomss = []
        for batch in range(adj.shape[0]):
            masks = []
            denoms = []  # 所有关系的度矩阵
            for i in range(self.relation_cnt):
                # 求某个关系的度矩阵
                denom = torch.sparse.sum(adj[batch, i], dim=1).to_dense()
                # 防止点与点之间是单向连接
                t_g = denom + torch.sparse.sum(adj[batch, i], dim=0).to_dense()
                mask = t_g.eq(0)  # 度矩阵的mask矩阵
                denoms.append(denom.unsqueeze(1))
                masks.append(mask)
            denoms = torch.sum(torch.stack(denoms), 0)  # 聚合所有关系的度矩阵
            denoms = denoms + 1  # +1 表示边的自连接 [node_size,1]
            masks = sum(masks)  # [node_size]
            maskss.append(masks)
            denomss.append(denoms)
        denomss = torch.stack(denomss)  # [batch_size,node_size,1]

        # sparse rgcn layer
        for l in range(self.layers):
            gAxWs = []
            for j in range(self.relation_cnt):
                gAxW = []
                # 邻接矩阵和权重的相乘
                bxW = self.W_r[j][l](gcn_inputs)
                for batch in range(adj.shape[0]):
                    # 每个文档
                    xW = bxW[batch]
                    AxW = torch.sparse.mm(adj[batch][j], xW)  # 255, 25
                    # 除以相应关系的度
                    # AxW = AxW/ denomss[batch][j]  # 255, 25
                    gAxW.append(AxW)
                gAxW = torch.stack(gAxW)  # [batch_size,node_size,node_emb]
                gAxWs.append(gAxW)
            # # [batch_size,5,node_size,node_emb]
            gAxWs = torch.stack(gAxWs, dim=1)
            # self loop
            gAxWs = F.relu(
                (torch.sum(gAxWs, 1) + self.W_0[l](gcn_inputs)) / denomss)
            gcn_inputs = self.gcn_drop(gAxWs) if l < self.layers - 1 else gAxWs
        return gcn_inputs, maskss


class Local_rep_layer(nn.Module):
    def __init__(self, query, rgcn_hidden_dim, attn_head_num, attn_drop):
        super(Local_rep_layer, self).__init__()
        self.query = query
        input_dim = rgcn_hidden_dim
        self.device = torch.device("cuda")

        # 分别对
        self.multiheadattention = MultiHeadAttention(
            input_dim, num_heads=attn_head_num, dropout=attn_drop)
        self.multiheadattention1 = MultiHeadAttention(input_dim, num_heads=attn_head_num,
                                                      dropout=attn_drop)

    def forward(self, info, section, nodes, global_nodes):
        """
        Args:
            info (_type_): 所有mention信息, shape:[mention_num,7],一个batch中mention_num为mention的数量;<entity_id, entity_type, start_wid, end_wid, sentence_id, origin_sen_id, node_type>
            section (_type_): 保存每个文档中的实体数量,mention数量,句子数量,words数量, shape:[batch_size,4]
            nodes (_type_): (entities, mentions, sentences) <batch_size * node_size>
            global_nodes (_type_): RGCN提取的全局node节点特征 [batch_size,node_size,hidden_size]
        Returns:
            _type_: _description_
        """
        entities, mentions, sentences = nodes  # entity_size * dim
        # [batch_size ,entity_size ,hidden_size]
        entities = split_n_pad(entities, section[:, 0])
        if self.query == 'global':  # 是否使用RGCN提取的全局node节点特征
            entities = global_nodes
        # 单个batch中最大的entity_size
        entity_size = section[:, 0].max()
        # [batch_size,mention_size,hidden_size]
        mentions = split_n_pad(mentions, section[:, 1])
        # [sent_num,hidden_size]
        mention_sen_rep = F.embedding(
            info[:, 4], sentences)
        # [batch_size,max_mention_num,hidden_size]
        mention_sen_rep = split_n_pad(mention_sen_rep, section[:, 1])
        eid_ranges = torch.arange(0, max(info[:, 0]) + 1).to(self.device)
        # [batch_size,entity_size]
        eid_ranges = split_n_pad(eid_ranges, section[:, 0], pad=-2)
        # np.meshgrid(a,b)是按a作为x轴信息，b为y轴信息
        # torch.meshgrid(x,y)是按输入x作为y轴信息，将x每个维度复制y次，生成第一个矩阵;输入y作为x轴信息，将y每个维度复制x次，作为第二个矩阵
        r_idx, c_idx = torch.meshgrid(torch.arange(entity_size).to(self.device),
                                      torch.arange(entity_size).to(self.device))
        # 只取实体节点的表征 [batch_size,entity_size,entity_size,hidden_size]
        query_1 = entities[:, r_idx]  # 实体对的tail实体
        query_2 = entities[:, c_idx]  # 实体对的head实体
        # [batch_size,max_mention_num,7]
        info = split_n_pad(info, section[:, 1], pad=-1)
        # torch.broadcast_tensors 将x按照y的方向进行扩充，y按照x的方向进行扩充,最终生成的两个矩阵都有相同的维度
        m_ids, e_ids = torch.broadcast_tensors(
            info[:, :, 0].unsqueeze(1), eid_ranges.unsqueeze(-1))
        # [batch_size , entity_size , mention_size]
        index_m = torch.ne(m_ids, e_ids).to(self.device)
        """
        [[[0, 1, 2],
         [3, 4, 5]]]
        ==>
        [[[0, 1, 2],
         [3, 4, 5],
         [0, 1, 2],
         [3, 4, 5]]]
        """
        # [batch_size,entity_size*entity_size,mention_size]，构建多头注意力机制的mask矩阵
        index_m_h = index_m.unsqueeze(2).repeat(1, 1, entity_size, 1).reshape(
            index_m.shape[0], entity_size*entity_size, -1).to(self.device)
        """
        [[[0, 1, 2],
         [3, 4, 5]]]
        ==>
        [[[0, 1, 2],
         [3, 4, 5],
         [0, 1, 2],
         [3, 4, 5]]]
        """
        # [batch_size,entity_size*entity_size,mention_size]，构建多头注意力机制的mask矩阵
        index_m_t = index_m.unsqueeze(1).repeat(1, entity_size, 1, 1).reshape(
            index_m.shape[0], entity_size*entity_size, -1).to(self.device)
        # 分别输入初始句子节点表征、初始mention节点表征、实体全局表征有关
        # 分别对实体对的头部和尾部使用多头注意力机制
        # key (mention_sen_rep): [图卷积之前的初始句子节点表征],shape:[batch_size,max_mention_num,hidden_size]
        # value (mentions): [初始mention节点表征],shape:[batch_size,mention_size,hidden_size]
        # query (query_*): [实体全局表征有关] shape:[batch_size,entity_size,entity_size,hidden_size]
        # attn_mask (index_m_*): [description]. shape:[batch_size,entity_size*entity_size,mention_size].
        entitys_pair_rep_h, h_score = self.multiheadattention(
            mention_sen_rep, mentions, query_2, index_m_h)
        entitys_pair_rep_t, t_score = self.multiheadattention1(
            mention_sen_rep, mentions, query_1, index_m_t)
        return entitys_pair_rep_h, entitys_pair_rep_t


class GLREModule(nn.Module):
    def __init__(self, args,) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(args.pretrain_path)
        hidden_size = self.bert.config.hidden_size

        # 对节点类型进行embedding
        self.type_embed = EmbedLayer(num_embeddings=3,
                                     embedding_dim=args.type_dim,
                                     dropout=0.0)

        # global node rep
        rgcn_input_dim = args.lstm_dim + args.type_dim

        self.rgcn_layer = RGCN_Layer(
            rgcn_input_dim, args.rgcn_hidden_dim, args.rgcn_num_layers, args.gcn_in_drop, args.gcn_out_drop, relation_cnt=5)
        self.encoder = EncoderLSTM(input_size=hidden_size,
                                   num_units=args.lstm_dim,
                                   nlayers=args.bilstm_layers,
                                   bidir=True,
                                   dropout=args.drop_i)
        self.more_lstm = args.more_lstm
        if self.more_lstm:
            pretrain_hidden_size = args.lstm_dim*2
        else:
            pretrain_hidden_size = hidden_size

        self.pretrain_lm_linear_re = nn.Linear(
            pretrain_hidden_size, args.lstm_dim)
        self.finaldist = args.finaldist

        if self.finaldist:
            self.dist_embed_dir = EmbedLayer(num_embeddings=20, embedding_dim=args.dist_dim,
                                             dropout=0.0,
                                             ignore=10,
                                             freeze=False,
                                             pretrained=None,
                                             mapping=None)

        if args.rgcn_num_layers == 0:
            input_dim = rgcn_input_dim * 2
        else:
            input_dim = args.rgcn_hidden_dim * 2

        self.local_rep_layer = Local_rep_layer(
            args.query, args.rgcn_hidden_dim, args.att_head_num, args.att_drop)
        input_dim += args.lstm_dim * 2

        if self.finaldist:
            input_dim += args.dist_dim * 2
        self.context_att = args.context_att
        # if self.context_att:
        #     self.self_att = SelfAttention(input_dim, 1.0)
        #     input_dim = input_dim * 2

        self.mlp_layer = args.mlp_layers
        if self.mlp_layer > -1:
            hidden_dim = args.mlp_dim
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(self.mlp_layer - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            self.out_mlp = nn.Sequential(*layers)
            input_dim = hidden_dim

        self.classifier = Classifier(in_size=input_dim,
                                     out_size=args.rel_size,
                                     dropout=args.drop_o)

        self.query = args.query
        assert self.query == 'init' or self.query == 'global'

        self.dataset = args.dataset

    def encoding_layer(self, word_vec, seq_lens):
        """
        Encoder Layer -> Encode sequences using BiLSTM.
        @:param word_sec [list]
        @:param seq_lens [list]
        """
        ys, _ = self.encoder(torch.split(
            word_vec, seq_lens.tolist(), dim=0), seq_lens)  # 20, 460, 128
        return ys

    def graph_layer(self, nodes, info, section):
        """
        Graph Layer -> Construct a document-level graph
        The graph edges hold representations for the connections between the nodes.
        Args:
            nodes ([type]): node_type_num 个 各个节点表征信息
            info ([type]): 所有mention信息, shape:[mention_num,7]
            section ([type]): 每个batch的各个节点数量信息([batch_size,3]) #entities/#mentions/#sentences per batch
        Returns:
            (Tensor) graph, (Tensor) tensor_mapping, (Tensors) indices, (Tensor) node information
        """
        # all nodes in order: entities - mentions - sentences，将所有节点信息表征拼接起来
        nodes = torch.cat(nodes, dim=0)  # e + m + s (all)
        # info/node: node type | semantic type | sentence ID
        # 获取了节点类型，节点实体类型，句子id
        nodes_info = self.node_info(section, info)
        # 将节点的表征和节点类型的embedding进行拼接,shape: [all_node_num,hidden_size+rel_embedding_size],all_node_num=section.sum()
        # 类似将每个节点类型表征加到节点表征上去
        nodes = torch.cat((nodes, self.type_embed(nodes_info[:, 0])), dim=1)
        # re-order nodes per document (batch)
        nodes = self.rearrange_nodes(nodes, section)
        # torch.Size([4, 76, 210]) batch_size * node_size * node_emb
        # 按照batch切分后，进行padding，[batch_size,max_node_number,hidden_size+rel_embedding_size]
        nodes = split_n_pad(nodes, section.sum(dim=1))
        # re-order nodes per document (batch)
        nodes_info = self.rearrange_nodes(nodes_info, section)

        # [batch_size,max_node_number,node_type_size]
        nodes_info = split_n_pad(nodes_info, section.sum(dim=1), pad=-1)

        return nodes, nodes_info

    def node_layer(self, encoded_seq, info, word_sec):
        """获取各种节点表征
        Args:
            encoded_seq (_type_): 一个batch中所有句子的表征,shape:[sent_num,max_word_num,hidden_size]
            info (_type_): 所有mention信息, shape:[mention_num,7]
            word_sec (_type_): 每个句子的word数量, shape:[batch_size*sent_num]

        Returns:
            _type_: _description_
        """
        # SENTENCE NODES，句子节点的表征
        # sentence nodes (avg of sentence words)
        sentences = torch.mean(encoded_seq, dim=1)
        # MENTION & ENTITY NODES,删除了句子中word的padding表征
        encoded_seq_token = rm_pad(encoded_seq, word_sec)
        # mention节点表征,[mention_num,hidden_size]
        mentions = self.merge_tokens(info, encoded_seq_token)
        # 实体节点表征
        entities = self.merge_mentions(info, mentions)  # entity nodes
        return (entities, mentions, sentences)

    @staticmethod
    def merge_tokens(info, enc_seq, type="mean"):
        """基于mention的start-end ids信息,合并token表征,进行平均,然后表示为mention节点表征
        Args:
            info (_type_): [mention_num,7]
            enc_seq (_type_): [sent_num*word_num,hidden_size],sent_num 为一个batch中的句子数量,word_num为句子中word数量
            type (str, optional): _description_. Defaults to "mean".

        Returns:
            _type_: _description_
        """
        mentions = []
        for i in range(info.shape[0]):
            if type == "max":
                mention = torch.max(
                    enc_seq[info[i, 2]: info[i, 3], :], dim=-2)[0]
            else:  # mean
                mention = torch.mean(
                    enc_seq[info[i, 2]: info[i, 3], :], dim=-2)
            mentions.append(mention)
        mentions = torch.stack(mentions)
        return mentions

    @staticmethod
    def merge_mentions(info, mentions, type="mean"):
        """合并含义相同的mention,最终表示为entity 表征
        Merge mentions into entities;
        Find which rows (mentions) have the same entity id and average them
        Args:
            info ([type]): 所有mention信息, shape:[mention_num,7]
            mentions ([type]): [description],shape:[mention_num,hidden_size]
            type (str, optional): [description]. Defaults to "mean".
        Returns:
            [type]: [description]
        """
        # torch.broadcast_tensors 将x按照y的方向进行扩充，y按照x的方向进行扩充
        # [mention_unique_num,mention_num],mention_unique_num是mention去重后数量
        m_ids, e_ids = torch.broadcast_tensors(info[:, 0].unsqueeze(0),
                                               torch.arange(0, max(info[:, 0]) + 1).unsqueeze(-1).to(info.device))
        # 具有相同的id的index为False，即在相同位置，不同的mention id设置为True，相同的设置为False，为后续对相同的mention id进行聚合
        index_f = torch.ne(m_ids, e_ids).bool().to(info.device)
        entities = []
        # 遍历所有实体
        for i in range(index_f.shape[0]):
            # 对相同的实体的mention进行聚合操作，[hidden_size]
            entity = pool(mentions, index_f[i, :].unsqueeze(-1), type=type)
            entities.append(entity)
        entities = torch.stack(entities)
        return entities

    def node_info(self, section, info):
        """AI is creating summary for node_info
        Args:
            section ([type]): 每个batch的各个节点数量信息([batch_size,3]) #entities/#mentions/#sentences per batch
            info ([type]): 所有mention信息, shape:[mention_num,7]
        """
        device = section.device
        # 节点类型
        # 重复张量的元素,对torch.arange(3)中每个元素按照section.sum(dim=0)中值大小重复相应的次数,shape*[section.sum()]
        typ = torch.repeat_interleave(torch.arange(3).to(
            device), section.sum(dim=0))  # node types (0,1,2)
        # 计算非负数组中每个值的频率。然后进行累加,即只取相同mention的第一个mention；[unique_mention_num]
        rows_ = torch.bincount(info[:, 0]).cumsum(dim=0)
        rows_ = torch.cat([torch.tensor([0]).to(device),
                          rows_[:-1]]).to(device)  #
        # batch中 句子的类型为-1
        stypes = torch.neg(torch.ones(section[:, 2].sum())).to(
            device).long()  # semantic type sentences = -1
        # info[:, 1] 为实体的类型
        all_types = torch.cat((info[:, 1][rows_], info[:, 1], stypes), dim=0)
        # section.sum(dim=0)[2]一个batch内总的句子数量
        sents_ = torch.arange(section.sum(dim=0)[2]).to(device)
        # info[:, 4] 每个mention的在该batch中展开后的句子id
        sent_id = torch.cat(
            (info[:, 4][rows_], info[:, 4], sents_), dim=0)  # sent_id
        # 拼接节点类型，实体类型，句子id
        return torch.cat((typ.unsqueeze(-1), all_types.unsqueeze(-1), sent_id.unsqueeze(-1)), dim=1)

    @staticmethod
    def rearrange_nodes(nodes, section):
        """将全局的entity-mention-senttence的排序，重新在每个文档内部按照entity-mention-senttence顺序进行排序，不断按照文档进行叠加。
        Re-arrange nodes so that they are in 'Entity - Mention - Sentence' order for each document (batch)
        Args:
            nodes ([type]): [description]
            section ([type]): 每个batch的各个节点数量信息([batch_size,3]) #entities/#mentions/#sentences per batch
        Returns:
            [type]: [description]
        """
        device = nodes.device
        tmp1 = section.t().contiguous().view(-1).long().to(device)
        # section.numel() = batch_size*3，用于temp2中的索引
        tmp3 = torch.arange(section.numel()).view(section.size(1),
                                                  section.size(0)).t().contiguous().view(-1).long().to(device)
        # section.sum() 所有节点数量总和，然后按照tmp1不等量切分，主要分为三部分，前面1/3,中间1/3,后面1/3分别是不同节点
        tmp2 = torch.arange(section.sum()).to(device).split(tmp1.tolist())
        # padding 后，按照tmp3的索引进行重排
        tmp2 = pad_sequence(tmp2, batch_first=True,
                            padding_value=-1)[tmp3].view(-1)  # 进行padding
        # 移除-1
        tmp2 = tmp2[(tmp2 != -1).nonzero().squeeze()]  # remove -1 (padded)
        nodes = torch.index_select(nodes, 0, tmp2)
        return nodes

    @staticmethod
    def select_pairs(nodes_info, idx, dataset='docred'):
        """Select (entity node) pairs for classification based on input parameter restrictions (i.e. their entity type).
        Args:
            nodes_info ([type]): [description], shape:[batch_size,max_node_number,node_type_size]
            idx ([tuple]): [[max_node_number,max_node_number]]
            dataset (str, optional): [description]. Defaults to 'docred'.

        Returns:
            [type]: [description]
        """
        # [batch_size,max_node_number,max_node_number]
        sel = torch.zeros(nodes_info.size(0), nodes_info.size(
            1), nodes_info.size(1)).to(nodes_info.device)
        # [batch_size,max_node_number,max_node_number]
        a_ = nodes_info[..., 0][:, idx[0]]
        # [batch_size,max_node_number,max_node_number]
        b_ = nodes_info[..., 0][:, idx[1]]
        # 针对不同数据
        if dataset == 'cdr':
            c_ = nodes_info[..., 1][:, idx[0]]
            d_ = nodes_info[..., 1][:, idx[1]]
            condition1 = torch.eq(a_, 0) & torch.eq(b_, 0) & torch.ne(
                idx[0], idx[1])  # needs to be an entity node (id=0)
            condition2 = torch.eq(c_, 1) & torch.eq(
                d_, 2)  # h=medicine, t=disease
            sel = torch.where(condition1 & condition2,
                              torch.ones_like(sel), sel)
        else:
            condition1 = torch.eq(a_, 0) & torch.eq(
                b_, 0) & torch.ne(idx[0], idx[1])
            sel = torch.where(condition1, torch.ones_like(sel), sel)
        return sel.nonzero().unbind(dim=1), sel.nonzero()[:, 0]

    def forward(self, input_ids, attention_mask, token_starts, section, word_sec, entities, rgcn_adjacency, distances_dir, multi_relations):
        """_summary_

        Args:
            input_ids (_type_): 输入的input ids, shape:[batch_size,seq_len]
            attention_mask (_type_): 输入的attention mask, shape:[batch_size,seq_len]
            token_starts (_type_): 输入的每个token的起始位置在input ids中的位置,毕竟有些token会被分为两个input id, shape:[batch_size,seq_len]
            section (_type_): 保存每个文档中的实体数量,mention数量,句子数量,words数量, shape:[batch_size,4]
            word_sec (_type_): 每个句子的word数量, shape:[batch_size*sent_num],sent_num为每个文本的句子数量
            entities (_type_): 所有mention信息, shape:[mention_num,7],一个batch中mention_num为mention的数量
            rgcn_adjacency (_type_): rgcn模型的邻接矩阵,shape:[batch_size,5,node_number,node_number],5是5种边
            distances_dir (_type_): _description_ ,shape:[batch_size,entity_size,entity_size]
            multi_relations (_type_): _description_ ,shape:[batch_size,entity_size,entity_size,rel_size]

        Returns:
            _type_: _description_
        """
        # [batch_size,seq_len,hidden_size]
        context_output = self.bert(input_ids, attention_mask=attention_mask)[0]
        # 这是取每个word的start边界？为啥？
        context_output = [layer[starts.nonzero().squeeze(1)] for layer, starts in
                          zip(context_output, token_starts)]
        context_output_pad = []
        for output, word_len in zip(context_output, section[:, 3]):
            if output.size(0) < word_len:
                padding = Variable(output.data.new(1, 1).zero_())
                output = torch.cat([output, padding.expand(
                    word_len - output.size(0), output.size(1))], dim=0)
            context_output_pad.append(output)
        # 所有word的start边界组合成的context,[batch_size*word_len,hidden_size]
        context_output = torch.cat(context_output_pad, dim=0)

        if self.more_lstm:
            # 使用lstm对模型文本进行二次编码
            context_output = self.encoding_layer(context_output, section[:, 3])
            context_output = rm_pad(context_output, section[:, 3])
        encoded_seq = self.pretrain_lm_linear_re(context_output)
        # [sent_num,max_word_num,hidden_size]
        encoded_seq = split_n_pad(encoded_seq, word_sec)

        # Graph
        # Global Representation Layer,(entities, mentions, sentences)
        nodes = self.node_layer(encoded_seq, entities, word_sec)
        init_nodes = nodes
        # 构建graph的node信息,
        # [batch_size,max_node_number,hidden_size+rel_embedding_size]和[batch_size,max_node_number,node_type_size]
        nodes, nodes_info = self.graph_layer(nodes, entities, section[:, 0:3])
        # RGCN 模型模块 [batch_size,node_size,hidden_size]
        nodes, _ = self.rgcn_layer(nodes, rgcn_adjacency, section[:, 0:3])
        entity_size = section[:, 0].max()
        device = entity_size.device
        # np.meshgrid(a,b)是按a作为x轴信息，b为y轴信息
        # torch.meshgrid(x,y)是按输入x作为y轴信息，输入y作为x轴信息.
        r_idx, c_idx = torch.meshgrid(torch.arange(entity_size).to(device),
                                      torch.arange(entity_size).to(device))
        # [batch_size,entity_size,entity_size,hidden_size]
        relation_rep_h = nodes[:, r_idx]
        relation_rep_t = nodes[:, c_idx]

        # Local Representation Layer
        entitys_pair_rep_h, entitys_pair_rep_t = self.local_rep_layer(
            entities, section, init_nodes, nodes)
        # 将局部表征与全局表征concatenate
        relation_rep_h = torch.cat(
            (relation_rep_h, entitys_pair_rep_h), dim=-1)
        relation_rep_t = torch.cat(
            (relation_rep_t, entitys_pair_rep_t), dim=-1)

        if self.finaldist:
            # 相对距离的embedding
            dis_h_2_t = distances_dir + 10
            dis_t_2_h = -distances_dir + 10
            dist_dir_h_t_vec = self.dist_embed_dir(dis_h_2_t)
            dist_dir_t_h_vec = self.dist_embed_dir(dis_t_2_h)
            # 将相对距离embedding纳入到语义特征中
            relation_rep_h = torch.cat(
                (relation_rep_h, dist_dir_h_t_vec), dim=-1)
            relation_rep_t = torch.cat(
                (relation_rep_t, dist_dir_t_h_vec), dim=-1)
        # 拼接实体对的语义特征,[batch_size,entity_size,entity_size,-1]
        graph_select = torch.cat((relation_rep_h, relation_rep_t), dim=-1)

        # if self.context_att:
        #     # 这里应该使用了target中的关系标签？为啥？
        #     relation_mask = torch.sum(torch.ne(multi_relations, 0), -1).gt(0)
        #     graph_select = self.self_att(
        #         graph_select, graph_select, relation_mask)

        # Classification
        # np.meshgrid(a,b)是按a作为x轴信息，b为y轴信息
        # torch.meshgrid(x,y)是按输入x作为y轴信息，输入y作为x轴信息.
        r_idx, c_idx = torch.meshgrid(torch.arange(nodes_info.size(1)).to(device),
                                      torch.arange(nodes_info.size(1)).to(device))

        select, _ = self.select_pairs(nodes_info, (r_idx, c_idx), self.dataset)
        graph_select = graph_select[select]
        if self.mlp_layer > -1:
            graph_select = self.out_mlp(graph_select)
        graph = self.classifier(graph_select)

        return graph, select


class GLREModuelPytochLighting(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = GLREModule(args)
        self.rel_size = args.rel_size
        self.ignore_label = args.label2ignore
        self.index2rel = args.index2rel
        self.args = args
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def count_predictions(self, y, t):
        """Count number of TP, FP, FN, TN for each relation class
        Args:
            y ([type]): [prediction]
            t ([type]): [ground truth]
        Returns:
            [type]: [description]
        """
        label_num = torch.as_tensor([self.rel_size]).long().to(self.device)
        ignore_label = torch.as_tensor(
            [self.ignore_label]).long().to(self.device)

        # where the ground truth needs to be ignored
        mask_t = torch.eq(t, ignore_label).view(-1)
        # where the predicted needs to be ignored
        mask_p = torch.eq(y, ignore_label).view(-1)

        true = torch.where(mask_t, label_num,
                           t.view(-1).long().to(self.device))  # ground truth
        pred = torch.where(mask_p, label_num,
                           y.view(-1).long().to(self.device))  # output of NN

        tp_mask = torch.where(torch.eq(pred, true), true,
                              label_num)  # True Positive
        fp_mask = torch.where(torch.ne(pred, true), pred,
                              label_num)  # False Positive
        fn_mask = torch.where(torch.ne(pred, true), true,
                              label_num)  # False Negative

        tp = torch.bincount(
            tp_mask, minlength=self.rel_size + 1)[:self.rel_size]
        fp = torch.bincount(
            fp_mask, minlength=self.rel_size + 1)[:self.rel_size]
        fn = torch.bincount(
            fn_mask, minlength=self.rel_size + 1)[:self.rel_size]
        tn = torch.sum(mask_t & mask_p)
        return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'ttotal': t.shape[0]}

    def estimate_loss(self, pred_pairs, truth, multi_truth):
        """Softmax cross entropy loss.
        Args:
            pred_pairs ([type]): (pair_numbers, rel_size)
            truth ([type]): [description]
            multi_truth ([type]): (pair_numbers, rel_size)
        Returns:
            [type]: (Tensor) loss, (Tensors) TP/FP/FN
        """
        # [pair_numbers]
        multi_mask = torch.sum(torch.ne(multi_truth, 0), -1).gt(0)

        pred_pairs = pred_pairs[multi_mask]
        multi_truth = multi_truth[multi_mask]
        truth = truth[multi_mask]
        # label smoothing
        # multi_truth -= self.smoothing * ( multi_truth  - 1. / multi_truth.shape[-1])
        loss = torch.sum(self.loss(pred_pairs, multi_truth)) / (
            torch.sum(multi_mask) * self.rel_size)

        return loss, pred_pairs, multi_truth, multi_mask, truth

    def training_step(self, batches, batch_idx):
        bert_token, bert_mask, bert_starts = batches['bert_token'], batches['bert_mask'], batches['bert_starts']
        section, word_sec, entities = batches['section'], batches['word_sec'], batches['entities']
        rgcn_adj, dist_dir, multi_rel = batches['rgcn_adjacency'], batches['distances_dir'], batches['multi_relations']
        relations = batches['relations']
        graph, select = self.model(bert_token, bert_mask, bert_starts, section, word_sec, entities, rgcn_adj, dist_dir, multi_rel)
        loss, pred_pairs, multi_truth, mask, truth = self.estimate_loss(graph, relations[select], multi_rel[select])

        return loss

    def validation_step(self, batches, batch_idx):
        bert_token, bert_mask, bert_starts = batches['bert_token'], batches['bert_mask'], batches['bert_starts']
        section, word_sec, entities = batches['section'], batches['word_sec'], batches['entities']
        rgcn_adj, dist_dir, multi_rel = batches['rgcn_adjacency'], batches['distances_dir'], batches['multi_relations']
        relations = batches['relations']
        graph, select = self.model(bert_token, bert_mask, bert_starts, section, word_sec, entities, rgcn_adj, dist_dir, multi_rel)
        loss, pred_pairs, multi_truths, mask, truth = self.estimate_loss(graph, relations[select], multi_rel[select])

        pred_pairs = torch.sigmoid(pred_pairs)
        predictions = pred_pairs.data.argmax(dim=1)
        stats = self.count_predictions(predictions, truth)

        output = {}
        test_info = []
        output['loss'] = [loss.item()]
        output['tp'] = [stats['tp'].to('cpu').data.numpy()]
        output['fp'] = [stats['fp'].to('cpu').data.numpy()]
        output['fn'] = [stats['fn'].to('cpu').data.numpy()]
        output['tn'] = [stats['tn'].to('cpu').data.numpy()]
        output['preds'] = [predictions.to('cpu').data.numpy()]

        test_infos = batches['info'][select[0].to('cpu').data.numpy(),
                                     select[1].to('cpu').data.numpy(),
                                     select[2].to('cpu').data.numpy()][mask.to('cpu').data.numpy()]
        test_info += [test_infos]

        pred_pairs = pred_pairs.data.cpu().numpy()
        multi_truths = multi_truths.data.cpu().numpy()
        output['true'] = multi_truths.sum() - multi_truths[:,
                                                           self.ignore_label].sum()
        return output

    def validation_epoch_end(self, outputs) -> None:
        output = {'tp': [], 'fp': [], 'fn': [],
                  'tn': [], 'loss': [], 'preds': [], 'true': 0}
        for out in outputs:
            output['loss'] += out['loss']
            output['tp'] += out['tp']
            output['fp'] += out['fp']
            output['fn'] += out['fn']
            output['tn'] += out['tn']
            output['preds'] += out['preds']
            output['true'] += out['true']
        scores = self.prf1(output['tp'], output['fp'],
                           output['fn'], output['tn'])
        self.log("tot", float(scores['total']), prog_bar=True)
        self.log("cor", float(scores['tp']), prog_bar=True)
        self.log("pred", float(scores['pred']), prog_bar=True)
        self.log("f1", float(scores['f1']), prog_bar=True)
        self.log("recall", float(scores['recall']), prog_bar=True)
        self.log("prec", float(scores['precsion']), prog_bar=True)
        self.log("acc", float(scores['acc']), prog_bar=True)

    def fbeta_score(self, precision, recall, beta=1.0):
        beta_square = beta * beta
        if (precision != 0.0) and (recall != 0.0):
            res = ((1 + beta_square) * precision * recall / (beta_square * precision + recall))
        else:
            res = 0.0
        return res

    def prf1(self, tp_, fp_, fn_, tn_):
        tp_ = np.sum(tp_, axis=0)
        fp_ = np.sum(fp_, axis=0)
        fn_ = np.sum(fn_, axis=0)
        tn_ = np.sum(tn_, axis=0)

        atp = np.sum(tp_)
        afp = np.sum(fp_)
        afn = np.sum(fn_)
        atn = np.sum(tn_)

        micro_p = (1.0 * atp) / (atp + afp) if (atp + afp != 0) else 0.0
        micro_r = (1.0 * atp) / (atp + afn) if (atp + afn != 0) else 0.0
        micro_f = self.fbeta_score(micro_p, micro_r)

        acc = (atp + atn) / (atp + atn + afp +
                             afn) if (atp + atn + afp + afn) else 0.0
        acc_NA = atn / (atn + afp) if (atn + afp) else 0.0
        acc_not_NA = atp / (atp + afn) if (atp + afn) else 0.0
        return {'acc': acc, 'NA_acc': acc_NA, 'not_NA_acc': acc_not_NA,
                'precsion': micro_p, 'recall': micro_r, 'f1': micro_f,
                'tp': atp, 'true': atp + afn, 'pred': atp + afp, 'total': (atp + atn + afp + afn)}

    def configure_optimizers(self):
        """[配置优化参数]
        """
        paramsbert = []
        paramsbert0reg = []
        paramsothers = []
        paramsothers0reg = []
        for p_name, p_value in self.named_parameters():
            if not p_value.requires_grad:
                continue
            if 'bert' in p_name or 'pretrain_lm' in p_name or 'word_embed' in p_name:
                if '.bias' in p_name:
                    paramsbert0reg += [p_value]
                else:
                    paramsbert += [p_value]
            else:
                if '.bias' in p_name:
                    paramsothers0reg += [p_value]
                else:
                    paramsothers += [p_value]
        groups = [dict(params=paramsbert, lr=self.args.bert_lr),
                  dict(params=paramsothers),
                  dict(params=paramsbert0reg, lr=self.args.bert_lr, weight_decay=0.0),
                  dict(params=paramsothers0reg, weight_decay=0.0)]
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        optimizer = torch.optim.Adam(groups, lr=self.args.lr, weight_decay=float(self.args.reg), amsgrad=True)

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
