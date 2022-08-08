# -*- encoding: utf-8 -*-
'''
@File    :   loss_func.py
@Time    :   2021/07/27 19:30:49
@Author  :   lujun
@Version :   1.0
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   损失函数
'''

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing import Union
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

def bce2d(pred, target, reduction='mean'):
    """[动态为每个batch的正例和负例添加权重]

    Args:
        pred ([type]): [[B,1]]
        target ([type]): [[B,1]]
        reduction (str, optional): [description]. Defaults to 'mean'.

    Returns:
        [type]: [description]
    """
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()
    # num_pos = torch.sum(pos)
    # num_neg = torch.sum(neg)
    # num_total = num_pos + num_neg
    # alpha_pos = num_neg / num_total
    # alpha_neg = num_pos / num_total
    # weights = alpha_pos * pos + 1.5*alpha_neg * neg
    weights = pos*17 + neg*1.2
    return F.binary_cross_entropy_with_logits(pred, target, weights, reduction = reduction)

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=2, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class BCEFocalLoss(nn.Module):
    """[二分类的focal loss 损失函数]
    """
    def __init__(self, alpha=0.25, gamma=5, size_average=False):
        """[focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)]
        Args:
            alpha (float, optional): [阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于数量较多的负类,]. Defaults to 0.25.
            gamma (int, optional): [伽马γ,难易样本调节参数. retainnet中设置为2]. Defaults to 2.
            num_classes (int, optional): [类别数量]. Defaults to 3.
            size_average (bool, optional): [损失计算方式,默认取均值]. Defaults to True.
        """

        super(BCEFocalLoss,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.epsilon = 1.e-6 #半精度浮点数，表示的最小数字

    def forward(self, preds, target):
        """[focal_loss损失计算]
        Args:
            preds ([type]): [预测类别. size: [B]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数]
            target ([type]): [实际类别. size:[B] ]

        Returns:
            [type]: [description]
        """
        
        pt = torch.sigmoid(preds) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)    
        pt = torch.clamp(pt,self.epsilon,1-self.epsilon)    
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)   
        # if self.size_average:        
        #     loss = loss.mean()
        # else:            
        #     loss = loss.sum() 
        return loss

class MLFocalLoss(nn.Module):
    """[多标签的focal loss 损失函数]
    """
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        """[focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)]
        Args:
            alpha (float, optional): [阿尔法α,控制类别不平衡的,alpha越大,recall会越高]
            gamma (int, optional): [伽马γ,难易样本调节参数. retainnet中设置为2]. Defaults to 2.
            num_classes (int, optional): [类别数量]. Defaults to 3.
            size_average (bool, optional): [损失计算方式,默认取均值]. Defaults to True.
        """

        super(MLFocalLoss,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.epsilon = 1.e-9 #半精度浮点数，表示的最小数字

    def forward(self, preds, target):
        """[focal_loss损失计算]
        Args:
            preds ([type]): [预测类别. size: [B]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数]
            target ([type]): [实际类别. size:[B] ]

        Returns:
            [type]: [description]
        """
        
        pt = torch.sigmoid(preds) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)  
        zeros = torch.zeros_like(pt, dtype=pt.dtype)
        pos_p_sub = torch.where(target > zeros, target - pt, zeros)
        neg_p_sub = torch.where(target > zeros, zeros, pt)

        per_entry_cross_ent = - self.alpha * (pos_p_sub ** self.gamma) * torch.log(pt + self.epsilon) - (1-self.alpha) * (neg_p_sub ** self.gamma) * torch.log(1.0 - pt + self.epsilon)  
        # per_entry_cross_ent = -torch.log(torch.clamp(pt,self.epsilon, 1.0)) - neg_p_sub * torch.log(torch.clamp(1.0 - pt, self.epsilon, 1.0))  
        # per_entry_cross_ent = -(pos_p_sub ** self.gamma)*torch.log(pt) - (neg_p_sub**self.gamma) * torch.log(1.0 - pt)  


        if self.size_average:        
            loss = per_entry_cross_ent.mean()
        else:            
            loss = per_entry_cross_ent.sum()
        if torch.isinf(loss):
            print(loss)   
        return loss


class MultiCrossentropy(nn.Module):

    def __init__(self, reduction='mean'):
        super(MultiCrossentropy, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """[]

        Args:
            pred ([type]): [description]
            target ([type]): [description]

        Returns:
            [type]: [description]
        """
        pred = (1. - 2 * target) * pred
        pred_pos = pred * target - 1e12 * (1 - target)
        pred_neg = pred * (1. - target) - 1e12 * target
        zero_pad = torch.zeros_like(pred)[..., :1]
        pred_pos = torch.cat([pred_pos, zero_pad], dim=-1)
        pred_neg = torch.cat([pred_neg, zero_pad], dim=-1)
        loss = torch.logsumexp(pred_pos, dim=-1) + torch.logsumexp(pred_neg, dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class MultiCEFocalLoss(nn.Module):
    """[多类别 focal loss函数]
    """
    def __init__(self, num_classes, alpha=0.25, gamma=2, size_average=True):
        """[focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)]

        Args:
            alpha (float, optional): [阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于数量较多的负类, retainnet中设置为0.25]. Defaults to 0.25.
            gamma (int, optional): [伽马γ,难易样本调节参数. retainnet中设置为2]. Defaults to 2.
            num_classes (int, optional): [类别数量]. Defaults to 3.
            size_average (bool, optional): [损失计算方式,默认取均值]. Defaults to True.
        """

        super(MultiCEFocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma
        self.elipson = 1e-20

    def forward(self, preds, labels):
        """[focal_loss损失计算]
        Args:
            preds ([type]): [预测类别. size:[B,C,N] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数]
            labels ([type]): [实际类别. size:[B,N] or [B] ]

        Returns:
            [type]: [description]
        """
        # assert preds.dim()==2 and labels.dim()==1  
        # preds = preds.transpose(-1, -2)     
        # preds = preds.view(-1,preds.size(-1))        
        alpha = self.alpha.to(preds.device)        
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)        
        preds_logsoft = torch.log((preds_softmax-self.elipson).abs())
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )        
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))        
        alpha = alpha.gather(0,labels.view(-1))        
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(alpha, loss.t())        
        if self.size_average:        
            loss = loss.mean()        
        else:            
            loss = loss.sum()        
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 1e-20
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert(logits.size(0) == labels.size(0))
        assert(logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length],device=logits.device).scatter_(1, new_label, 1)

        # calculate log
        # log_p = F.log_softmax(logits,dim=1)
        # pt = label_onehot * log_p
        p_softmax = F.softmax(logits,dim=1)
        pt = (label_onehot * p_softmax).sum(1)
        log_p = torch.log((pt- self.elipson).abs())
        
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

class MultiCEBiTemperedLogisticLoss(_Loss):
    """[多类别的带噪声数据的神经网络双温逻辑损失]
    """
    def __init__(self, reduction='mean', t1=1, t2=1, label_smoothing=0.0, num_iters=5):
        super().__init__(reduction=reduction)
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing
        self.num_iters = num_iters

    @classmethod
    def log_t(cls, u, t):
        """Compute log_t for `u`."""

        if t == 1.0:
            return torch.log(u)
        else:
            return (u ** (1.0 - t) - 1.0) / (1.0 - t)

    @classmethod
    def exp_t(cls, u, t):
        """Compute exp_t for `u`."""

        if t == 1.0:
            return torch.exp(u)
        else:
            return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

    @classmethod
    def compute_normalization_fixed_point(cls, inputs, t, num_iters=5):
        """Returns the normalization value for each example (t > 1.0).
        Args:
        inputs: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (> 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """

        # mu = torch.max(inputs, dim=-1).values.view(-1, 1)
        mu = torch.max(inputs, dim=-1).values.unsqueeze(-1)
        normalized_activations_step_0 = inputs - mu

        normalized_activations = normalized_activations_step_0
        i = 0
        while i < num_iters:
            i += 1
            # logt_partition = torch.sum(cls.exp_t(normalized_activations, t), dim=-1).view(-1, 1)
            logt_partition = torch.sum(cls.exp_t(normalized_activations, t), dim=-1).unsqueeze(-1)
            normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

        # logt_partition = torch.sum(cls.exp_t(normalized_activations, t), dim=-1).view(-1, 1)
        logt_partition = torch.sum(cls.exp_t(normalized_activations, t), dim=-1).unsqueeze(-1)

        return -cls.log_t(1.0 / logt_partition, t) + mu

    @classmethod
    def compute_normalization(cls, inputs, t, num_iters=5):
        """Returns the normalization value for each example.
        Args:
        inputs: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """

        if t < 1.0:
            return None  # not implemented as these values do not occur in the authors experiments...
        else:
            return cls.compute_normalization_fixed_point(inputs, t, num_iters)

    @classmethod
    def tempered_softmax(cls, inputs, t, num_iters=5):
        """Tempered softmax function.
        Args:
        inputs: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature tensor > 0.0.
        num_iters: Number of iterations to run the method.
        Returns:
        A probabilities tensor.
        """
        if t == 1.0:
            normalization_constants = torch.log(torch.sum(torch.exp(inputs), dim=-1,keepdim=True))
        else:
            normalization_constants = cls.compute_normalization(inputs, t, num_iters)

        return cls.exp_t(inputs - normalization_constants, t)

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, inputs, smoothing=0.0):
        assert 0 <= smoothing < 1
        n_classes = inputs.size(-1)
        with torch.no_grad():
            # targets = torch.empty_like(inputs,device=targets.device).fill_(smoothing / (n_classes - 1)).scatter_(-1, targets.data.unsqueeze(-1), 1. - smoothing)
            targets = torch.where(target==1,1. - smoothing,smoothing / (n_classes - 1))

        return targets

    def forward(self, inputs, targets):
        """Bi-Tempered Logistic Loss with custom gradient.
        Args:
        inputs: A multi-dimensional tensor with last dimension `num_classes`,[B,N,C] or [B,C] 
        targets: A tensor with shape and dtype as inputs, [B,N,C] or [B,C]
        t1: Temperature 1 (< 1.0 for boundedness).
        t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
        label_smoothing: Label smoothing parameter between [0, 1).
        num_iters: Number of iterations to run the method.
        Returns:
        A loss tensor.
        """
        if self.label_smoothing > 0.0:
            targets = MultiCEBiTemperedLogisticLoss._smooth_one_hot(targets, inputs, self.label_smoothing)
        # else:
        #     targets = targets.unsqueeze(-1)
        probabilities = self.tempered_softmax(inputs, self.t2, self.num_iters)

        temp1 = (self.log_t(targets + 1e-10, self.t1) - self.log_t(probabilities, self.t1)) * targets
        temp2 = (1 / (2 - self.t1)) * (torch.pow(targets, 2 - self.t1) - torch.pow(probabilities, 2 - self.t1))
        loss = temp1 - temp2

        loss = loss.sum(dim=-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss

class GlobalCrossEntropy(nn.Module):
    def __init__(self, interval=0.):
        super(GlobalCrossEntropy, self).__init__()
        self.interval = interval

    def multilabel_categorical_crossentropy(self, y_true, y_pred, threshold):
        """多标签分类的交叉熵
        说明：
            1. y_true和y_pred的shape一致，y_true的元素非0即1，
            1表示对应的类为目标类，0表示对应的类为非目标类；
            2. 请保证y_pred的值域是全体实数，换言之一般情况下
            y_pred不用加激活函数，尤其是不能加sigmoid或者
            softmax；
            3. 预测阶段则输出y_pred大于0的类；
            4. 详情请看：https://kexue.fm/archives/7359 。
        """
        y_pred = (1 - 2 * y_true) * y_pred  # 将标签为1的pred取反
        y_pred_neg = y_pred - y_true * 1e12  # 将标签为1的pred取无限小
        y_pred_pos = y_pred - (1 - y_true) * 1e12  # 将标签为0的pred 取无限小
        thresh = torch.zeros_like(y_pred[..., :1])
        thresh[:] = threshold
        y_pred_neg += self.interval  # 给负例加上一些间隔
        y_pred_pos -= self.interval
        y_pred_neg = torch.cat([y_pred_neg, thresh], axis=-1)
        y_pred_pos = torch.cat([y_pred_pos, -thresh], axis=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return 1.1*neg_loss + pos_loss

    def forward(self, y_pred, y_true, threshold=0):
        """给GlobalPointer设计的交叉熵
        """
        if y_pred.dim() > 3:
            batch_size, rel_num = y_pred.shape[0], y_pred.shape[1]
            bh = batch_size*rel_num
        else:
            bh = y_pred.shape[0]
        y_true = torch.reshape(y_true, (bh, -1))
        y_pred = torch.reshape(y_pred, (bh, -1))
        loss = self.multilabel_categorical_crossentropy(
            y_true, y_pred, threshold)
        return torch.mean(loss)


class MutualInforLoss(nn.Module):
    """通过互信息思想来缓解类别不平衡问题
        https://kexue.fm/archives/7615
    Args:
        nn (_type_): _description_
    """
    def __init__(self,prior,tau=1.0) -> None:
        super().__init__()
        # 自己定义好prior,shape为[num_classes]
        self.prior=torch.from_numpy(np.log(prior+1e-8))
        self.tau = tau

    def forward(self,inputs,targets):
        """多标签损失函数
            不能先用sigmoid函数
        Args:
            inputs (_type_): [batch_size,num_classes]
            targets (_type_): _description_
        """
        batch_size = inputs.shape[0]
        prior = self.prior.repeat(batch_size,1)
        inputs = inputs+self.tau*prior
        return F.multilabel_soft_margin_loss(inputs, targets, weight=None, reduction= 'mean')


class WarmupLR(_LRScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr
            * self.warmup_steps ** 0.5
            * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.7)
            for lr in self.base_lrs
        ]

    def set_step(self, step: int):
        self.last_epoch = step

class TwoStepLR(_LRScheduler):

    def __init__(self, optimizer,lr,steps,  last_epoch=-1, verbose=False):
        self.learning_rate = lr
        self.steps = steps
        self.min_learning_rate: float = 5e-6
        self.passed = -1
        super(TwoStepLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.passed < self.steps:
            self.passed += 1
            return [(self.passed + 1.) / self.steps * group['lr'] for group in self.optimizer.param_groups]
        elif self.steps <= self.passed < self.steps * 2:
            self.passed += 1
            return [(2 - (self.passed + 1.) / self.steps) * (group['lr'] - self.min_learning_rate)+self.min_learning_rate for group in self.optimizer.param_groups]
        else:
            return [group['lr']*-0.1 for group in self.optimizer.param_groups]



if __name__ == "__main__":
    focal_loss = MLFocalLoss()
    preds = torch.randn((100,10))
    target = torch.empty((100,10),dtype=torch.long).random_(2)
    print(focal_loss(preds,target))


