import json
from collections import defaultdict
from typing import List


SPOT_PROMPT = '<spot>'
ASOC_PROMPT = '<asoc>'

# 标签的开始标志
TYPE_START = '<extra_id_0>'
# 标签的结束标志
TYPE_END = '<extra_id_1>'
# 输入文本的开始标志
TEXT_START = '<extra_id_2>'
# 文本span的开始标志
SPAN_START = '<extra_id_5>'
# 非文本span的标志
null_span = '<extra_id_6>'
null_label = '<extra_id_7>'



class BaseStructureMarker():
    def __init__(self) -> None:
        super().__init__()
        self.sent_start = '<extra_id_0>'
        self.sent_end = '<extra_id_1>'
        self.record_start = '<extra_id_0>'
        self.record_end = '<extra_id_1>'
        self.span_start = '<extra_id_0>'
        self.span_end = '<extra_id_1>'
        self.text_start = '<extra_id_2>'
        self.source_span_start = '<extra_id_3>'
        self.source_span_end = '<extra_id_4>'
        self.target_span_start = '<extra_id_5>'
        self.null_span = '<extra_id_6>'
        self.null_label = '<extra_id_7>'


def add_special_token(tokenizer):
    """为tokenizer中添加特殊符号
    Args:
        tokenizer ([type]): [description]
    Returns:
        [type]: [description]
    """
    to_add_special_token = list()
    for special_token in [TYPE_START, TYPE_END, TEXT_START, SPAN_START, SPOT_PROMPT, ASOC_PROMPT]:
        if special_token not in tokenizer.get_vocab():
            to_add_special_token += [special_token]

    tokenizer.add_special_tokens(
        {"additional_special_tokens": tokenizer.special_tokens_map_extended['additional_special_tokens'] + to_add_special_token}
    )
    return tokenizer


class RecordSchema:
    def __init__(self, type_list, role_list, type_role_dict):
        self.type_list = type_list
        self.role_list = role_list
        self.type_role_dict = type_role_dict

    def __repr__(self) -> str:
        return f"Type: {self.type_list}\n Role: {self.role_list}\n Map: {self.type_role_dict}"

    @staticmethod
    def get_empty_schema():
        return RecordSchema(type_list=list(), role_list=list(), type_role_dict=dict())

    @staticmethod
    def read_from_file(filename):
        lines = open(filename).readlines()
        type_list = json.loads(lines[0])
        role_list = json.loads(lines[1])
        type_role_dict = json.loads(lines[2])
        return RecordSchema(type_list, role_list, type_role_dict)

    def write_to_file(self, filename):
        with open(filename, 'w') as output:
            output.write(json.dumps(self.type_list) + '\n')
            output.write(json.dumps(self.role_list) + '\n')
            output.write(json.dumps(self.type_role_dict) + '\n')


def merge_schema(schema_list: List[RecordSchema]):
    type_set = set()
    role_set = set()
    type_role_dict = defaultdict(list)

    for schema in schema_list:

        for type_name in schema.type_list:
            type_set.add(type_name)

        for role_name in schema.role_list:
            role_set.add(role_name)

        for type_name in schema.type_role_dict:
            type_role_dict[type_name] += schema.type_role_dict[type_name]

    for type_name in type_role_dict:
        type_role_dict[type_name] = list(set(type_role_dict[type_name]))

    return RecordSchema(type_list=list(type_set),
                        role_list=list(role_set),
                        type_role_dict=type_role_dict,
                        )



class TaskConfig:
    def __init__(self, task_dict) -> None:
        self.dataset_name = task_dict.get('name', '')
        self.task_name = task_dict.get('task', '')
        self.data_path = task_dict.get('path', '')
        self.decoding_format = task_dict.get('decoding_format', '')
        self.weight = int(task_dict.get('weight', 0))
        self.sel2record = task_dict.get('sel2record', '')
        self.metrics = task_dict.get('metrics', [])
        self.eval_match_mode = task_dict.get('eval_match_mode', 'normal')
        self.schema = RecordSchema.read_from_file(f"{self.data_path}/{self.task_name}.schema")

    def __repr__(self) -> str:
        return f"dataset: {self.dataset_name}\n" \
               f"task   : {self.task_name}\n" \
               f"format : {self.decoding_format}\n" \
               f"path   : {self.data_path}\n" \
               f"schema : {self.schema}\n" \
               f"metrics: {self.metrics}\n" \
               f"eval_match_mode : {self.eval_match_mode}"

    @staticmethod
    def load_list_from_yaml(task_config):
        import yaml
        configs = yaml.load(open(task_config), Loader=yaml.FullLoader)
        task_configs = filter(lambda x: x.startswith('T'), configs)
        for task_config in task_configs:
            yield TaskConfig(configs[task_config])


class PrefixGenerator:
    def __init__(self, prefix_dict) -> None:
        self.type_list = prefix_dict.get('type', 'task dataset').split()
        self.position = prefix_dict.get('position', 'encoder')

    def __repr__(self) -> str:
        return f"Type.   : {self.type_list}\n" \
               f"Position: {self.position}\n"

    @staticmethod
    def load_from_yaml(dataset_config):
        import yaml
        configs = yaml.load(open(dataset_config), Loader=yaml.FullLoader)
        return PrefixGenerator(configs['Prefix'])

    @staticmethod
    def get_schema_prefix(schema: RecordSchema, add_split=True):
        prefix_list = list()
        for spot_label in sorted(schema.type_list):
            prefix_list += [SPOT_PROMPT, spot_label]
        for asoc_label in sorted(schema.role_list):
            prefix_list += [ASOC_PROMPT, asoc_label]
        prefix = ' '.join(prefix_list)
        if add_split:
            return prefix + f' {TEXT_START} '
        else:
            return prefix

    @staticmethod
    def get_dataset_name_prefix(dataset: TaskConfig, add_split=True):
        if add_split:
            return dataset.dataset_name + f' {TEXT_START}'
        else:
            return dataset.dataset_name

    @staticmethod
    def get_task_name_prefix(dataset: TaskConfig, add_split=True):
        if add_split:
            return dataset.task_name + f' {TEXT_START}'
        else:
            return dataset.task_name

    def get_prefix_by_dataset(self, dataset: TaskConfig):
        prefix_list = list()
        for prefix_type in self.type_list:
            if prefix_type == 'task':
                prefix = self.get_task_name_prefix(dataset, add_split=False)
            elif prefix_type == 'dataset':
                prefix = self.get_dataset_name_prefix(dataset, add_split=False)
            elif prefix_type == 'schema':
                prefix = self.get_schema_prefix(dataset.schema, add_split=False)
            elif prefix_type == 'meta':
                # Meta 使用 Schema 的 Prefix
                prefix = self.get_schema_prefix(dataset.schema, add_split=False)
            else:
                raise NotImplementedError(
                    "Prefix Type %s is not supported" % prefix_type
                )
            prefix_list += [prefix]
        return ' '.join(prefix_list) + f' {TEXT_START}'
