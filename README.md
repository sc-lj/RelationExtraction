# TDEER 🦌

Official Paper For [TDEER: An Efficient Translating Decoding Schema for Joint Extraction of Entities and Relations](https://aclanthology.org/2021.emnlp-main.635/)
Official code For [github](https://github.com/4AI/TDEER)

## Overview

TDEER 是一种用于联合提取实体和关系的有效模型。 与预测主客体关系的常见解码方法不同，我们采用提出的翻译解码模式：subject + relation -> objects 来解码三元组。 通过提出的翻译解码模式，TDEER 可以有效地处理重叠三元组问题。

![overview](docs/TDEER-Overview.png)

参数量：113 M


# PRGC 🦌

Official Paper For [PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction](https://arxiv.org/abs/2106.09895)
Official code For [github](https://github.com/hy-struggle/PRGC)

## Overview
采用3阶段联合学习的方式抽取关系，即先预测句子中存在哪些潜在的关系，分别对预测出在关系embedding矩阵中获取相关的关系表征加在token表征中，用两个分类头分别预测该关系下有哪些subject和object(采用BIO的标注方式)，最后学习subject和object对齐的矩阵(所有关系的subject和object对都在这个矩阵中表示出来)。

![overview](docs/PRGC-Overview.png)

参数量：113 M


# TPLinker 🦌

Official Paper For [TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking](https://arxiv.org/abs/2010.13415) 
Official code For [github](https://github.com/131250208/TPlinker-joint-extraction)

## Overview
采用一阶段联合学习的方式抽取关系。对于N个关系，采用2N+1个头进行预测，即其中一个通过下标的方式预测实体的头尾，2N个头分别是N个预测subject-to-object头部(SH-to-OH)和subject-to-object尾部(ST-to-OT)是否存在关系。最后通过一定规则解码，得出所有关系对。

![overview](docs/TPLinker-Overview.png)

参数量：116 M


# SPN4RE 🦌

Official Paper For [SPN4RE:Joint Entity and Relation Extraction with Set Prediction Networks](https://arxiv.org/abs/2011.01675) 
Official code For [github](https://github.com/DianboWork/SPN4RE)

## Overview
采用集合预测方式，预先定义最大的预测triple数量m，然后对这m个将要预测的triple采用query形式，生成embedding，与token表征进行cross attention交互，预测句子中的关系类型，以及subject和object的start和end索引。最后采用Bipartite Matching Loss，即使用匈牙利算法先得到ground truth triple和预测的triple之间最佳匹配(cost最小)，然后基于最佳匹配计算损失。
个人认为，本文的核心创新点就是利用将triples当作一个集合去预测，并使用Bipartite Matching Loss作为模型的训练的对象。

![overview](docs/SPN4RE-Overview.png)

参数量：142 M


# OneRel 🦌

该模型没有源码，这是根据论文复现出来的。但是其中还有很多疑惑之处。比如，如何让模型实现row(行)一定表示subject(head)，让columns(列)表示object(tail)呢？
Official Paper For [OneRel: Joint Entity and Relation Extraction with One Module in One Step](https://arxiv.org/abs/2203.05412) 
NO Official code For [github](https://github.com/ssnvxia/OneRel)

## Overview
采用一步学习一个M矩阵(K x 4 x L x L)，L是文本长度，K是关系数量，4表示4种标记，HB-TB，HB-TE，HE-TE，-这4种关系，其借鉴了知识图谱嵌入HOLE的思想，设计其pair对的学习对象。

![overview](docs/OneRel-Overview.png)

参数量：113 M


# GLRE 🦌

Official Paper For [Global-to-Local Neural Networks for Document-Level Relation Extraction](https://arxiv.org/abs/2203.05412) 
Official code For [github](https://github.com/nju-websoft/GLRE)

## Overview
文档级别的关系抽取。本文通过编码层编码文档信息；全局表征层将文档中的句子、mention、实体等构建层一个异构图，并使用R-GCN来提取实体的全局表征；局部表征层利用多头注意力机制将实体全局表征作为Query、句子节点表征作为Key，mention节点表征作为Value，最终提取实体的局部表征；最后的分类层，聚合了所有的实体对以提取文档的主题信息表征，并与实体对的target关系表征进行结合，预测该实体对的关系。
根据其源码，其对整个文档是作为一个句子输入到bert的tokenizer中，并没有对每个句子单独解析出其token。其实现有点伪文档关系抽取的味道？
ps: ```这是一篇文档级关系抽取算法，说是文档级，但是却把一篇文档不做任何处理，进行拼接输入到预训练模型中，如果超过预训练模型的长度限制，就进行截断。这与句子级关系抽取有何区别？就是多了几个句子而已？```

![overview](docs/GLRE-Overview.png)

参数量：113 M




# 项目说明
本项目主要是在NYT句子集关系抽取数据集和DocRED篇章级关系抽取数据进行相关模型的实验。相应的数据下载后解压到data文件夹中。

