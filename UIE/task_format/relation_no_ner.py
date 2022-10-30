# -*- encoding: utf-8 -*-
'''
File    :   relation_no_ner.py
Time    :   2022/10/30 11:34:21
Author  :   lujun
Version :   1.0
Contact :   779365135@qq.com
License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
Desc    :   只针对关系抽取，不进行实体类别识别
'''


import json
from typing import List
from utils.utils import find_head_idx
from UIE.task_format.utils import tokens_to_str, change_ptb_token_back, Entity, Label, Relation, Sentence, Span
from UIE.task_format.task_format import TaskFormat


class Rel(TaskFormat):
    """
    {
        "text": "Massachusetts ASTON MAGNA Great Barrington ; also at Bard College , Annandale-on-Hudson , N.Y. , July 1-Aug .",
        "triple_list": [[
                "Annandale-on-Hudson",
                "/location/location/contains",
                "College"
            ]]
    }
    """

    def __init__(self, sentence_json, language='en'):
        super().__init__(
            language=language
        )
        self.tokens = sentence_json['text'].split(" ")
        for index in range(len(self.tokens)):
            self.tokens[index] = change_ptb_token_back(self.tokens[index])
        if self.tokens is None:
            print('[sentence without tokens]:', sentence_json)
            exit(1)
        self.triple_list = sentence_json['triple_list']

    def generate_instance(self):
        subject_entities = dict()
        object_entities = dict()
        relations = list()

        for index, triple in enumerate(self.triple_list):
            subj, rel, obj = triple

            if subj not in subject_entities:
                sub_token = subj.split(" ")
                sub_s = find_head_idx(self.tokens, sub_token, 0)
                if sub_s == -1:
                    print(f"sub token {subj} 未在文本‘{self.tokens}’中找到")
                    continue
                sub_e = sub_s+len(sub_token)
                subject_entities[subj] = Entity(
                    span=Span(
                        tokens=sub_token,
                        indexes=list(range(sub_s, sub_e)),
                        text=tokens_to_str(sub_token, language=self.language),
                    ),
                    label=Label('subjects')
                )

            if obj not in object_entities:
                obj_token = obj.split(" ")
                obj_s = find_head_idx(self.tokens, obj_token, 0)
                if obj_s == -1:
                    print(f"sub token {obj} 未在文本‘{self.tokens}’中找到")
                    continue
                obj_e = obj_s+len(sub_token)

                object_entities[obj] = Entity(
                    span=Span(
                        tokens=obj_token,
                        indexes=list(range(obj_s, obj_e)),
                        text=tokens_to_str(obj_token, language=self.language),
                    ),
                    label=Label('objects')
                )

            relations += [Relation(
                arg1=subject_entities[subj],
                arg2=object_entities[obj],
                label=Label(rel),
            )]

        return Sentence(
            tokens=self.tokens,
            entities=list(subject_entities.values()) + list(object_entities.values()),
            relations=relations,
        )

    @staticmethod
    def load_from_file(filename, language='en') -> List[Sentence]:
        sentence_list = list()
        raw_instance_list = json.load(open(filename))
        print(f"{filename}: {len(raw_instance_list)}")
        for instance in raw_instance_list:
            instance = Rel(sentence_json=instance, language=language).generate_instance()
            sentence_list += [instance]
        return sentence_list
