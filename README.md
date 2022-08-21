# TDEER 🦌

Official Code For [TDEER: An Efficient Translating Decoding Schema for Joint Extraction of Entities and Relations](https://aclanthology.org/2021.emnlp-main.635/)

## Overview

TDEER 是一种用于联合提取实体和关系的有效模型。 与预测主客体关系的常见解码方法不同，我们采用提出的翻译解码模式：subject + relation -> objects 来解码三元组。 通过提出的翻译解码模式，TDEER 可以有效地处理重叠三元组问题。

![overview](docs/TDEER-Overview.png)

参数量：113 M


# PRGC 🦌

Official Code For [PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction](https://arxiv.org/abs/2106.09895) 

![overview](docs/PRGC-Overview.png)

参数量：113 M


# TPLinker 🦌

Official Code For [TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking](https://arxiv.org/abs/2010.13415) 


![overview](docs/TPLinker-Overview.png)

参数量：116 M


# SPN4RE 🦌

Official Code For [SPAN4RE:Joint Entity and Relation Extraction with Set Prediction Networks](https://arxiv.org/abs/2011.01675) 


![overview](docs/SPN4RE-Overview.png)

参数量：142 M


# OneRel 🦌
该模型没有源码，这是根据论文复现出来的。但是其中还有很多疑惑之处。比如，如何让模型实现row(行)一定表示subject(head)，让columns(列)表示object(tail)呢？
Official Code For [OneRel: Joint Entity and Relation Extraction with One Module in One Step](https://arxiv.org/abs/2203.05412) 


![overview](docs/OneRel-Overview.png)

参数量：113 M

