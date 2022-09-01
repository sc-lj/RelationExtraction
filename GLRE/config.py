# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2022/08/30 21:23:28
@Author  :   lujun
@Version :   1.0
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   GLRE 模型的相关参数
'''

QUERY = 'global'  # or 'init'
MORE_LSTM = True

# encoder
LSTM_DIM = 256
OUT_DIM = 256
TYPE_DIM = 20
DIST_DIM = 20
FINALDIST = True
BILSTM_LAYERS = 1
RGCN_HIDDEN_DIM = 256
RGCN_NUM_LAYERS = 2
GCN_IN_DROP = 0.2
GCN_OUT_DROP = 0.2

# network
DROP_I = 0.5
DROP_O = 0.3  # 0.3
ATTN_HEAD_NUM = 2
ATTN_DROP = 0.0
UNK_W_PROB = 0.5
MIN_W_FREQ = 1
NA_NUM = 0.5  # 0.1==5:1,
MLP_LAYERS = 1
MLP_DIM = 512
# 将标签信息融入到训练中的attention中了？
CONTEXT_ATT = False

DATASET = 'docred'   # or cdr
