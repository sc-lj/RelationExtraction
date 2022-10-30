#!/usr/bin/env python
# -*- coding:utf-8 -*-


import json
from typing import List
from UIE.task_format.utils import tokens_to_str, change_ptb_token_back, Entity, Label, Relation, Sentence, Span
from UIE.task_format.task_format import TaskFormat


class JointER(TaskFormat):
    """
    {"tokens": ["In", "Queens", ",", "North", "Shore", "Towers", ",", "near", "the", "Nassau", "border", ",", "supplanted", "a", "golf", "course", ",", "and", "housing", "replaced", "a", "gravel", "quarry", "in", "Douglaston", "."], 
    "triple_list": [["Douglaston", "/location/neighborhood/neighborhood_of", "Queens"], 
                ["Queens", "/location/location/contains", "Douglaston"]], 
    "spo_details": [[24, 25, "LOCATION", "/location/neighborhood/neighborhood_of", 1, 2, "LOCATION"], 
                    [1, 2, "LOCATION", "/location/location/contains", 24, 25, "LOCATION"]], 
    }
    """

    def __init__(self, sentence_json, language='en'):
        super().__init__(
            language=language
        )
        self.tokens = sentence_json['tokens']
        for index in range(len(self.tokens)):
            self.tokens[index] = change_ptb_token_back(self.tokens[index])
        if self.tokens is None:
            print('[sentence without tokens]:', sentence_json)
            exit(1)
        self.spo_list = sentence_json['triple_list']
        self.spo_details = sentence_json['spo_details']

    def generate_instance(self):
        entities = dict()
        relations = dict()
        entity_map = dict()

        for spo_index, spo in enumerate(self.spo_details):
            s_s, s_e, s_t = spo[0], spo[1], spo[2]
            tokens = self.tokens[s_s: s_e]
            indexes = list(range(s_s, s_e))
            if (s_s, s_e, s_t) not in entity_map:
                entities[(s_s, s_e, s_t)] = Entity(
                    span=Span(
                        tokens=tokens,
                        indexes=indexes,
                        text=tokens_to_str(tokens, language=self.language),
                    ),
                    label=Label(s_t)
                )

            o_s, o_e, o_t = spo[4], spo[5], spo[6]
            tokens = self.tokens[o_s: o_e]
            indexes = list(range(o_s, o_e))
            if (o_s, o_e, o_t) not in entity_map:
                entities[(o_s, o_e, o_t)] = Entity(
                    span=Span(
                        tokens=tokens,
                        indexes=indexes,
                        text=tokens_to_str(tokens, language=self.language),
                    ),
                    label=Label(o_t)
                )

            relations[spo_index] = Relation(
                arg1=entities[(s_s, s_e, s_t)],
                arg2=entities[(o_s, o_e, o_t)],
                label=Label(spo[3]),
            )

        return Sentence(
            tokens=self.tokens,
            entities=entities.values(),
            relations=relations.values(),
        )

    @staticmethod
    def load_from_file(filename, language='en') -> List[Sentence]:
        sentence_list = list()
        raw_instance_list = json.load(open(filename))
        print(f"{filename}: {len(raw_instance_list)}")
        for instance in raw_instance_list:
            instance = JointER(sentence_json=instance, language=language).generate_instance()
            sentence_list += [instance]
        return sentence_list
