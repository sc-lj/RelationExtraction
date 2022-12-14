import torch
import collections


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list.
    Args:
        logits (_type_):list[seq_len]
        n_best_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 按照logit的值进行排序，从大到小，选择前n_best_size个概率值的index
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def generate_span(start_logits, end_logits, info, args):
    """生成subject 和object 的head和tail的候选索引
    Args:
        start_logits (_type_): [bsz,num_generated_triples,seq_len]
        end_logits (_type_): [bsz,num_generated_triples,seq_len]
        info (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    seq_lens = info["seq_len"]  # including [CLS] and [SEP]
    sent_idxes = info["sent_idx"]
    sent_tokens = info["tokens"]
    _Prediction = collections.namedtuple(
        "Prediction", ["start_index", "end_index", "start_prob", "end_prob", "sub_token"]
    )
    output = {}
    start_probs = start_logits.softmax(-1)
    end_probs = end_logits.softmax(-1)
    start_probs = start_probs.cpu().tolist()
    end_probs = end_probs.cpu().tolist()
    for (start_prob, end_prob, seq_len, sent_idx, sent_token) in zip(start_probs, end_probs, seq_lens, sent_idxes, sent_tokens):
        output[sent_idx] = {}
        for triple_id in range(args.num_generated_triples):
            predictions = []
            start_indexes = _get_best_indexes(
                start_prob[triple_id], args.n_best_size)
            end_indexes = _get_best_indexes(
                end_prob[triple_id], args.n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the sentence. We throw out all
                    # invalid predictions.
                    if start_index >= (seq_len-1):  # [SEP]
                        continue
                    if end_index >= (seq_len-1):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > args.max_length:
                        continue
                    predictions.append(
                        _Prediction(
                            start_index=start_index,
                            end_index=end_index,
                            start_prob=start_prob[triple_id][start_index],
                            end_prob=end_prob[triple_id][end_index],
                            sub_token=_concat(sent_token[start_index:end_index+1])
                        )
                    )
            output[sent_idx][triple_id] = predictions
    return output


def generate_relation(pred_rel_logits, info, args):
    """获取每个num_generated_triples 概率最大的关系
    Args:
        pred_rel_logits (_type_): [bsz, num_generated_triples, num_classes]
        info (_type_): _description_
        args (_type_): _description_
    Returns:
        _type_: _description_
    """
    rel_probs, pred_rels = torch.max(pred_rel_logits.softmax(-1), dim=2)
    rel_probs = rel_probs.cpu().tolist()
    pred_rels = pred_rels.cpu().tolist()
    sent_idxes = info["sent_idx"]
    output = {}
    _Prediction = collections.namedtuple(
        "Prediction", ["pred_rel", "rel_prob"]
    )
    for (rel_prob, pred_rel, sent_idx) in zip(rel_probs, pred_rels, sent_idxes):
        output[sent_idx] = {}
        for triple_id in range(args.num_generated_triples):
            output[sent_idx][triple_id] = _Prediction(
                pred_rel=pred_rel[triple_id],
                rel_prob=rel_prob[triple_id])
    return output


def generate_triple(output, info, args, num_classes):
    _Pred_Triple = collections.namedtuple(
        "Pred_Triple", ["pred_rel", "rel_prob", "head_start_index", "head_end_index", "head_start_prob",
                        "head_end_prob", "tail_start_index", "tail_end_index", "tail_start_prob", "tail_end_prob",
                        "subject", "object"]
    )
    pred_head_ent_dict = generate_span(
        output["head_start_logits"], output["head_end_logits"], info, args)
    pred_tail_ent_dict = generate_span(
        output["tail_start_logits"], output["tail_end_logits"], info, args)
    pred_rel_dict = generate_relation(output['pred_rel_logits'], info, args)
    triples = {}
    for sent_idx in pred_rel_dict:
        triples[sent_idx] = []
        for triple_id in range(args.num_generated_triples):
            # 预测的关系
            pred_rel = pred_rel_dict[sent_idx][triple_id]
            # 预测的subject
            pred_head = pred_head_ent_dict[sent_idx][triple_id]
            # 预测的object
            pred_tail = pred_tail_ent_dict[sent_idx][triple_id]
            triple = generate_strategy(
                pred_rel, pred_head, pred_tail, num_classes, _Pred_Triple)
            if triple:
                triples[sent_idx].append(triple)
    return triples


def generate_strategy(pred_rel, pred_head, pred_tail, num_classes, _Pred_Triple):
    # 预测的关系不是无关的关系
    if pred_rel.pred_rel != num_classes:
        # subject、object都不为空
        if pred_head and pred_tail:
            for ele in pred_head:
                # subject 的开头不是token中的第一个，即[cls]
                if ele.start_index != 0:
                    break
            head = ele
            subject = head.sub_token
            for ele in pred_tail:
                # object 的开头不是token中的第一个，即[cls]
                if ele.start_index != 0:
                    break
            tail = ele
            object = head.sub_token
            return _Pred_Triple(pred_rel=pred_rel.pred_rel, rel_prob=pred_rel.rel_prob, head_start_index=head.start_index,
                                head_end_index=head.end_index, head_start_prob=head.start_prob, head_end_prob=head.end_prob,
                                tail_start_index=tail.start_index, tail_end_index=tail.end_index, tail_start_prob=tail.start_prob,
                                tail_end_prob=tail.end_prob, subject=subject, object=object)
        else:
            return
    else:
        return


def formulate_gold(target, info):
    sent_idxes = info["sent_idx"]
    sent_tokens = info['tokens']
    gold = {}
    for i in range(len(sent_idxes)):
        gold[sent_idxes[i]] = []
        sent_token = sent_tokens[i]
        for j in range(len(target[i]["relation"])):
            rel = target[i]["relation"][j].item()
            head_start_index = target[i]["head_start_index"][j].item()
            head_end_index = target[i]["head_end_index"][j].item()
            tail_start_index = target[i]["tail_start_index"][j].item()
            tail_end_index = target[i]["tail_end_index"][j].item()
            subject_token = _concat(sent_token[head_start_index:head_end_index+1])
            object_token = _concat(sent_token[tail_start_index:tail_end_index+1])
            gold[sent_idxes[i]].append((rel, head_start_index, head_end_index, tail_start_index, tail_end_index, subject_token, object_token))
    return gold


def _concat(token_list):
    result = ''
    for idx, t in enumerate(token_list):
        if idx == 0:
            result = t
        elif t.startswith('##'):
            result += t.lstrip('##')
        else:
            result += ' ' + t
    return result
