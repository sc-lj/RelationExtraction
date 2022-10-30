# -*- encoding: utf-8 -*-
'''
File    :   gen_data_format.py
Time    :   2022/10/29 23:08:09
Author  :   lujun
Version :   1.0
Contact :   779365135@qq.com
License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
Desc    :   将原始数据转换为训练数据格式
'''


from typing import List, Dict, Union, Tuple
from collections import defaultdict, Counter
from UIE.utils import RecordSchema, BaseStructureMarker
from UIE.task_format.utils import tokens_to_str, Entity, Event, Label, Relation, Span, Sentence
import abc
import os
import json
from tqdm import tqdm
from UIE.task_format.utils import label_format
import yaml
import os
from typing import Dict
import UIE.task_format as task_format


class GenerationFormat:
    __metaclass__ = abc.ABCMeta

    def __init__(self, structure_maker, label_mapper: Dict = None, language: str = 'en') -> None:
        self.structure_maker = structure_maker
        self.language = language
        self.label_mapper = {} if label_mapper is None else label_mapper

        # 用于从数据中统计 Schema
        self.record_role_map = defaultdict(set)

    def get_label_str(self, label: Label):
        return self.label_mapper.get(label.__repr__(), label.__repr__())

    @abc.abstractmethod
    def annotate_entities(self, tokens: List[str], entities: List[Entity]): pass

    @abc.abstractmethod
    def annotate_given_entities(self, tokens: List[str], entities: Union[List[Entity], Entity]): pass

    @abc.abstractmethod
    def annotate_events(self, tokens: List[str], events: List[Event]): pass

    @abc.abstractmethod
    def annotate_event_given_predicate(self, tokens: List[str], event: Event): pass

    @abc.abstractmethod
    def annotate_relation_extraction(self, tokens: List[str], relations: List[Relation]): pass

    def output_schema(self, filename: str):
        """自动导出 Schema 文件
        每个 Schema 文件包含三行
            - 第一行为 Record 的类别名称列表
            - 第二行为 Role 的类别名称列表
            - 第三行为 Record-Role 映射关系字典
        Args:
            filename (str): [description]
        """
        record_list = list(self.record_role_map.keys())
        role_set = set()
        for record in self.record_role_map:
            role_set.update(self.record_role_map[record])
            self.record_role_map[record] = list(self.record_role_map[record])
        role_list = list(role_set)

        record_schema = RecordSchema(type_list=record_list,
                                     role_list=role_list,
                                     type_role_dict=self.record_role_map
                                     )
        record_schema.write_to_file(filename)

    def get_entity_schema(self, entities: List[Entity]):
        schema_role_map = set()
        for entity in entities:
            schema_role_map.add(self.get_label_str(entity.label))
        return RecordSchema(type_list=list(schema_role_map), role_list=list(), type_role_dict=dict())

    def get_relation_schema(self, relations: List[Relation]):
        record_role_map = defaultdict(set)
        role_set = set()

        for relation in relations:
            record_role_map[self.get_label_str(relation.label)].add(self.get_label_str(relation.arg1.label))
            record_role_map[self.get_label_str(relation.label)].add(self.get_label_str(relation.arg2.label))

        for record in record_role_map:
            role_set.update(record_role_map[record])
            record_role_map[record] = list(self.record_role_map[record])

        return RecordSchema(
            type_list=list(record_role_map.keys()),
            role_list=list(role_set),
            type_role_dict=record_role_map
        )

    def get_event_schema(self, events: List[Event]):
        record_role_map = defaultdict(set)
        role_set = set()

        for event in events:
            for role, _ in event.args:
                record_role_map[self.get_label_str(event.label)].add(self.get_label_str(role))

        for record in record_role_map:
            role_set.update(record_role_map[record])
            record_role_map[record] = list(self.record_role_map[record])

        return RecordSchema(
            type_list=list(record_role_map.keys()),
            role_list=list(role_set),
            type_role_dict=record_role_map
        )


def change_name_using_label_mapper(label_name, label_mapper):
    if label_mapper is None or len(label_mapper) == 0:
        return label_name
    if label_name not in label_mapper:
        print(f"{label_name} not found in mapper")
        global global_mislabel_log
        if label_name not in global_mislabel_log:
            global_mislabel_log.add(label_name)
    return label_mapper.get(label_name, label_name)


def convert_spot_asoc(spot_asoc_instance, structure_maker):
    spot_instance_str_rep_list = list()
    for spot in spot_asoc_instance:
        spot_str_rep = [
            spot['label'],
            structure_maker.target_span_start,
            spot['span'],
        ]
        for asoc_label, asoc_span in spot.get('asoc', list()):
            asoc_str_rep = [
                structure_maker.span_start,
                asoc_label,
                structure_maker.target_span_start,
                asoc_span,
                structure_maker.span_end,
            ]
            spot_str_rep += [' '.join(asoc_str_rep)]
        spot_instance_str_rep_list += [' '.join([
            structure_maker.record_start,
            ' '.join(spot_str_rep),
            structure_maker.record_end,
        ])]
    target_text = ' '.join([
        structure_maker.sent_start,
        ' '.join(spot_instance_str_rep_list),
        structure_maker.sent_end,
    ])
    return target_text


class Text2SpotAsoc(GenerationFormat):
    def __init__(self, structure_maker: BaseStructureMarker, label_mapper: Dict = None, language: str = 'en') -> None:
        super().__init__(structure_maker=structure_maker, label_mapper=label_mapper, language=language)

    def annotate_entities(self, tokens: List[str], entities: List[Entity]):
        """ Convert Entities

        Args:
            tokens (List[str]): ['Trump', 'visits', 'China', '.']
            entities (List[Entity]): [description]

        Returns:
            source (str): Trump visits China.
            target (str): { [ Person : Trump ] [ Geo-political : China ] }
        """
        return self.annonote_graph(tokens=tokens, entities=entities)[:2]

    def augment_source_span(self, tokens: List[str], span: Span):
        """[summary]
        Args:
            tokens (List[str]): ['Trump', 'visits', 'China', '.']
            span (Span): Trump
        Returns:
            [type]: ['(', 'Trump', ')', 'visits', 'China', '.']
        """
        return tokens[:span.indexes[0]] + [self.structure_maker.source_span_start] + tokens[span.indexes[0]:span.indexes[-1] + 1] + [self.structure_maker.source_span_end] + tokens[span.indexes[-1] + 1:]

    def annotate_given_entities(self, tokens: List[str], entities):
        """
        entityies is List
        :param tokens: ['Trump', 'visits', 'China', '.']
        :param entities: ['Trump', 'China']
        :return:
            source (str): ( Trump ) ( China ) : Trump visits China .
            target (str): { [ Person : Trump ] [ Geo-political : China ] }

        entityies is Entity
        :param tokens: ['Trump', 'visits', 'China', '.']
        :param entities: 'Trump'
        :return:
            source (str): < Trump > visits China .
            target (str): { [ Person : Trump ] }
        """
        if isinstance(entities, list):
            entitytokens = []
            for entity in entities:
                entitytokens += [self.structure_maker.span_start]
                entitytokens += entity.span.tokens
                entitytokens += [self.structure_maker.span_end]
            source_text = tokens_to_str(entitytokens + [self.structure_maker.sep_marker] + tokens, language=self.language)
            _, target_text = self.annonote_graph(tokens=tokens, entities=entities)[:2]

        elif isinstance(entities, Entity):
            marked_tokens = self.augment_source_span(tokens=tokens, span=entities.span)
            source_text = tokens_to_str(marked_tokens, language=self.language)
            _, target_text = self.annonote_graph(tokens=tokens, entities=[entities])[:2]

        return source_text, target_text

    def annotate_events(self, tokens: List[str], events: List[Event]):
        """
        :param tokens: ['Trump', 'visits', 'China', '.']
        :param events:

        :return:
            source (str): Trump visits China.
            target (str): { [ Visit : visits ( Person : Trump ) ( Location : China ) ] }
        """
        return self.annonote_graph(tokens=tokens, events=events)[:2]

    def annotate_event_given_predicate(self, tokens: List[str], event: Event):
        """Annotate Event Given Predicate

        Args:
            tokens (List[str]): ['Trump', 'visits', 'China', '.']
            event (Event): Given Predicate

        Returns:
            [type]: [description]
        """
        marked_tokens = self.augment_source_span(tokens=tokens, span=event.span)
        source_text = tokens_to_str(marked_tokens, language=self.language)
        _, target_text = self.annonote_graph(tokens=tokens, events=[event])[:2]
        return source_text, target_text

    def annotate_relation_extraction(self, tokens: List[str], relations: List[Relation]):
        """
        :param tokens: ['Trump', 'visits', 'China', '.']
        :param relations:

        :return:
            source (str): Trump visits China.
            target (str): { [ Person : Trump ( Visit : China ) ] }
        """
        return self.annonote_graph(tokens=tokens, relations=relations)[:2]

    def annotate_entities_and_relation_extraction(self, tokens: List[str], entities: List[Entity], relations: List[Relation]):
        """
        :param tokens: ['Trump', 'visits', 'China', '.']
        :param relations:

        :return:
            source (str): Trump visits China.
            target (str): { [ Person : Trump ( Visit : China ) ] [ Geo-political : China ] }
        """
        return self.annonote_graph(tokens=tokens, entities=entities, relations=relations)[:2]

    def annonote_graph(self, tokens: List[str], entities: List[Entity] = [], relations: List[Relation] = [], events: List[Event] = []):
        """Convert Entity Relation Event to Spot-Assocation Graph

        Args:
            tokens (List[str]): Token List
            entities (List[Entity], optional): Entity List. Defaults to [].
            relations (List[Relation], optional): Relation List. Defaults to [].
            events (List[Event], optional): Event List. Defaults to [].

        Returns:
            str: [description]
                {
                    [ Person : Trump ( Visit : China ) ]
                    [ Visit : visits ( Person : Trump ) ( Location : China ) ]
                    [ Geo-political : China ]
                }
            set: Set of Spot
            set: Set of Asoc
        """
        spot_dict = dict()
        asoc_dict = defaultdict(list)
        spot_str_rep_list = list()

        def add_spot(spot):
            spot_key = (tuple(spot.span.indexes), self.get_label_str(spot.label))
            spot_dict[spot_key] = spot

            if self.get_label_str(spot.label) not in self.record_role_map:
                self.record_role_map[self.get_label_str(spot.label)] = set()

        def add_asoc(spot, asoc: Label, tail):
            spot_key = (tuple(spot.span.indexes), self.get_label_str(spot.label))
            asoc_dict[spot_key] += [(tail.span.indexes, tail, self.get_label_str(asoc))]

            self.record_role_map[self.get_label_str(spot.label)].add(self.get_label_str(asoc))

        for entity in entities:
            add_spot(spot=entity)

        for relation in relations:
            add_spot(spot=relation.arg1)
            add_asoc(spot=relation.arg1, asoc=relation.label, tail=relation.arg2)

        for event in events:
            add_spot(spot=event)
            for arg_role, argument in event.args:
                add_asoc(spot=event, asoc=arg_role, tail=argument)

        spot_asoc_instance = list()
        for spot_key in sorted(spot_dict.keys()):
            offset, label = spot_key

            if spot_dict[spot_key].span.is_empty_span():
                continue

            spot_instance = {'span': spot_dict[spot_key].span.text,
                             'label': label,
                             'asoc': list(),
                             }
            for _, tail, asoc in sorted(asoc_dict.get(spot_key, [])):

                if tail.span.is_empty_span():
                    continue

                spot_instance['asoc'] += [(asoc, tail.span.text)]
            spot_asoc_instance += [spot_instance]

        target_text = convert_spot_asoc(
            spot_asoc_instance,
            structure_maker=self.structure_maker,
        )

        source_text = tokens_to_str(tokens, language=self.language)
        spot_labels = set([label for _, label in spot_dict.keys()])
        asoc_labels = set()
        for _, asoc_list in asoc_dict.items():
            for _, _, asoc in asoc_list:
                asoc_labels.add(asoc)
        return source_text, target_text, spot_labels, asoc_labels, spot_asoc_instance


class Dataset:
    def __init__(self, name: str, path: str, data_class: task_format.TaskFormat, split_dict: Dict, language: str, mapper: Dict, other: Dict = None) -> None:
        self.name = name
        self.path = path
        self.data_class = data_class
        self.split_dict = split_dict
        self.language = language
        self.mapper = mapper
        self.other = other

    def load_dataset(self):
        datasets = {}
        for split_name, filename in self.split_dict.items():
            # 读取训练，验证，测试数据
            datasets[split_name] = self.data_class.load_from_file(
                filename=os.path.join(self.path, filename),
                language=self.language,
                **self.other,
            )
        return datasets


def convert_graph(config_file):
    """数据格式转换的主函数

    Args:
        config_file (_type_): 相关任务的配置文件
    """
    dataset_config = yaml.load(open(config_file), Loader=yaml.FullLoader)
    dataset_config = dataset_config['data_params']
    if 'mapper' in dataset_config:
        mapper = dataset_config['mapper']
        for key in mapper:
            mapper[key] = label_format(mapper[key])
    else:
        print(f"{dataset_config['name']} without label mapper.")
        mapper = None
    output_folder = dataset_config['post_data_path']
    dataset = Dataset(
        name=dataset_config['name'],  # 数据集名字 Name of Dataset
        path=dataset_config['path'],  # 数据集路径 Path of Dataset
        data_class=getattr(task_format, dataset_config['data_class']),  # 数据集对应的 Task Format 名字 Raw data loader
        split_dict=dataset_config['split'],   # 数据集不同划分文件地址 Data Split Path
        language=dataset_config['language'],  # 数据集语言 Dataset Language
        mapper=mapper,
        other=dataset_config.get('other', {}),
    )
    datasets = dataset.load_dataset()
    label_mapper = dataset.mapper
    language = dataset.language

    convertor = Text2SpotAsoc(
        structure_maker=BaseStructureMarker(),
        language=language,
        label_mapper=label_mapper,
    )

    counter = Counter()

    os.makedirs(output_folder, exist_ok=True)

    schema_counter = {
        "entity": list(),
        "relation": list(),
        "event": list(),
    }
    for data_type, instance_list in datasets.items():
        with open(os.path.join(output_folder, f"{data_type}.json"), "w") as output:
            for instance in tqdm(instance_list):
                counter.update([f"{data_type} sent"])
                converted_graph = convertor.annonote_graph(
                    tokens=instance.tokens,
                    entities=instance.entities,
                    relations=instance.relations,
                    events=instance.events,
                )
                src, tgt, spot_labels, asoc_labels = converted_graph[:4]
                spot_asoc = converted_graph[4]

                schema_counter["entity"] += instance.entities
                schema_counter["relation"] += instance.relations
                schema_counter["event"] += instance.events

                output.write("%s\n" % json.dumps({
                    "text": src,
                    "tokens": instance.tokens,
                    "record": tgt,
                    "entity": [entity.to_offset(label_mapper) for entity in instance.entities],
                    "relation": [relation.to_offset(ent_label_mapper=label_mapper, rel_label_mapper=label_mapper) for relation in instance.relations],
                    "event": [event.to_offset(evt_label_mapper=label_mapper) for event in instance.events],
                    "spot": list(spot_labels),
                    "asoc": list(asoc_labels),
                    "spot_asoc": spot_asoc,
                },
                    ensure_ascii=False,
                )
                )
    # 保存该任务下所有提及的schema
    convertor.output_schema(os.path.join(output_folder, "record.schema"))
    # 获取实体相关的schema
    convertor.get_entity_schema(schema_counter["entity"]).write_to_file(os.path.join(output_folder, f"entity.schema"))
    # 获取relation任务的相关schema
    convertor.get_relation_schema(schema_counter["relation"]).write_to_file(
        os.path.join(output_folder, f"relation.schema")
    )
    convertor.get_event_schema(schema_counter["event"]).write_to_file(os.path.join(output_folder, f"event.schema"))


if __name__ == "__main__":
    config_file = ""

    output_folder = ""
    convert_graph(output_folder, config_file)
