import os
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from utils.Callback import EMACallBack
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
from utils.utils import statistics_text_length, update_arguments
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


def parser_args():
    parser = argparse.ArgumentParser(description='各个模型公共参数')
    parser.add_argument('--model_type', default="plmarker",
                        type=str, help='定义模型类型', choices=['tdeer', "tplinker", "prgc", "spn4re", "one4rel", "glre", "plmarker"])
    parser.add_argument('--pretrain_path', type=str, default="./bertbaseuncased", help='定义预训练模型路径')
    parser.add_argument('--data_dir', type=str, default="data/scierc", help='定义数据集路径')
    parser.add_argument('--lr', default=2e-5, type=float, help='specify the learning rate')
    parser.add_argument('--epoch', default=20, type=int, help='specify the epoch size')
    parser.add_argument('--batch_size', default=8, type=int, help='specify the batch size')
    parser.add_argument('--output_path', default="event_extract", type=str, help='将每轮的验证结果保存的路径')
    parser.add_argument('--float16', default=False, type=bool, help='是否采用浮点16进行半精度计算')
    parser.add_argument('--grad_accumulations_steps', default=2, type=int, help='梯度累计步骤')

    # 不同学习率scheduler的参数
    parser.add_argument('--decay_rate', default=0.999, type=float, help='StepLR scheduler 相关参数')
    parser.add_argument('--decay_steps', default=100, type=int, help='StepLR scheduler 相关参数')
    parser.add_argument('--T_mult', default=1.0, type=float, help='CosineAnnealingWarmRestarts scheduler 相关参数')
    parser.add_argument('--rewarm_epoch_num', default=2, type=int, help='CosineAnnealingWarmRestarts scheduler 相关参数')

    args = parser.parse_args()
    # 根据超参数文件更新参数
    config_file = os.path.join("config", "{}.yaml".format(args.model_type))
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    args = update_arguments(args, config)

    return args


def main():
    args = parser_args()
    if args.model_type == "tdeer":
        from TDeer import TDEERPytochLighting, TDEERDataset, collate_fn, collate_fn_val
        train_dataset = TDEERDataset(args, is_training=True)
        train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_dataset = TDEERDataset(args, is_training=False)
        val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn_val, batch_size=args.batch_size, shuffle=False)

        relation_number = train_dataset.relation_size
        args.relation_number = relation_number
        args.steps = len(train_dataset)
        model = TDEERPytochLighting(args)

    elif args.model_type == "tplinker":
        from TPlinker import TPlinkerPytochLighting, TPlinkerDataset, HandshakingTaggingScheme, DataMaker4Bert, TplinkerDataProcess
        tokenizer = BertTokenizerFast.from_pretrained(args.pretrain_path, add_special_tokens=False, do_lower_case=True)
        max_length = statistics_text_length(args.train_file, tokenizer)
        print("最大文本长度为:", max_length)
        args.max_seq_len = max_length+2

        def get_tok2char_span_map(text): return tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        # 数据预处理
        data_path = os.path.join(args.data_out_dir, "train.json")
        if not os.path.exists(data_path):
            TplinkerDataProcess(args, args.train_file, get_tok2char_span_map, is_training=True)
        data_path = os.path.join(args.data_out_dir, "val.json")
        if not os.path.exists(data_path):
            TplinkerDataProcess(args, args.val_file, get_tok2char_span_map, is_training=False)

        handshaking_tagger = HandshakingTaggingScheme(args)
        data_maker = DataMaker4Bert(tokenizer, handshaking_tagger)

        train_dataset = TPlinkerDataset(args, data_maker, tokenizer, is_training=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=data_maker.generate_batch)

        val_dataset = TPlinkerDataset(args, data_maker, tokenizer, is_training=False)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_maker.generate_batch)
        args.num_step = len(train_dataloader)
        model = TPlinkerPytochLighting(args, handshaking_tagger)

    elif args.model_type == "prgc":
        from PRGC import PRGCPytochLighting, PRGCDataset, collate_fn_test, collate_fn_train
        tokenizer = BertTokenizerFast.from_pretrained(args.pretrain_path)
        filename = os.path.join(args.data_dir, "train_triples.json")
        max_length = statistics_text_length(filename, tokenizer)
        print("最大文本长度为:", max_length)
        args.max_seq_len = max_length

        train_dataset = PRGCDataset(args, is_training=True)
        train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_train, batch_size=args.batch_size,
                                      shuffle=True, num_workers=8, pin_memory=True)

        val_dataset = PRGCDataset(args, is_training=False)
        val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn_test, batch_size=args.batch_size, shuffle=False)

        relation_number = train_dataset.relation_size
        args.relation_number = relation_number
        model = PRGCPytochLighting(args)

    elif args.model_type == "spn4re":
        from SPN4RE import Spn4REPytochLighting, Spn4REDataset, collate_fn
        tokenizer = BertTokenizerFast.from_pretrained(args.pretrain_path)
        filename = os.path.join(args.data_dir, "train_triples.json")
        max_length = statistics_text_length(filename, tokenizer)
        print("最大文本长度为:", max_length)
        max_length = min(max_length, 512)
        args.max_length = max_length
        train_dataset = Spn4REDataset(args, is_training=True)
        train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

        val_dataset = Spn4REDataset(args, is_training=False)
        val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False)

        relation_number = train_dataset.relation_size
        args.relation_number = relation_number
        args.steps = len(train_dataset)
        model = Spn4REPytochLighting(args)

    elif args.model_type == "one4rel":
        from OneRel import OneRelPytochLighting, OneRelDataset, collate_fn, TAG2ID
        train_dataset = OneRelDataset(args, is_training=True)
        relation_number = train_dataset.relation_size
        def new_collate_fn(x): return collate_fn(x, relation_number)
        train_dataloader = DataLoader(train_dataset, collate_fn=new_collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=8)

        val_dataset = OneRelDataset(args, is_training=False)
        val_dataloader = DataLoader(val_dataset, collate_fn=new_collate_fn, batch_size=args.batch_size, shuffle=False)

        args.relation_number = relation_number
        args.tag_size = len(TAG2ID)
        args.steps = len(train_dataset)
        model = OneRelPytochLighting(args)

    elif args.model_type == "glre":
        from GLRE import GLREModuelPytochLighting, GLREDataset, collate_fn
        train_dataset = GLREDataset(args, is_training=True)
        relation_number = train_dataset.n_rel
        label2ignore = train_dataset.label2ignore

        def train_collate_fn(x): return collate_fn(x, label2ignore, args.NA_NUM, istrain=True)
        train_dataloader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=args.batch_size, shuffle=True)

        val_dataset = GLREDataset(args, is_training=False)

        def val_collate_fn(x): return collate_fn(x, label2ignore, args.NA_NUM, istrain=False)
        val_dataloader = DataLoader(val_dataset, collate_fn=val_collate_fn, batch_size=args.batch_size, shuffle=False)

        args.label2ignore = label2ignore
        args.rel_size = relation_number
        args.steps = len(train_dataset)
        args.index2rel = train_dataset.index2rel
        model = GLREModuelPytochLighting(args)

    elif args.model_type == "plmarker":
        from PLMarker import PLMakerPytochLighting, PLMarkerDataset, collate_fn
        tokenizer = BertTokenizerFast.from_pretrained(args.pretrain_path)

        train_dataset = PLMarkerDataset(tokenizer, args, is_training=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        args.num_labels = train_dataset.num_labels
        args.num_ner_labels = train_dataset.num_ner_labels

        val_dataset = PLMarkerDataset(tokenizer, args, is_training=False)
        val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False)

        # 使用验证的标签集与计算相关的指标
        args.golden_labels = val_dataset.golden_labels
        args.golden_labels_withner = val_dataset.golden_labels_withner
        args.ner_golden_labels = val_dataset.ner_golden_labels
        args.global_predicted_ners = val_dataset.global_predicted_ners
        args.tot_recall = val_dataset.tot_recall

        args.t_total = len(train_dataloader) // args.grad_accumulations_steps * args.epoch
        args.steps = len(train_dataset)
        model = PLMakerPytochLighting(args, tokenizer)
    else:
        raise ValueError(f"目前不支持 该model type:{args.model_type}")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=8,
        verbose=True,
        monitor='f1',  # 监控val_acc指标
        mode='max',
        save_last=True,
        # dirpath = args.save_model_path.format(args.model_type),
        every_n_epochs=1,
        # filename = "{epoch:02d}{f1:.3f}{acc:.3f}{recall:.3f}",
        filename="{epoch:02d}{f1:.3f}{acc:.3f}{recall:.3f}{sr_rec:.3f}{sr_acc:.3f}",
    )

    early_stopping_callback = EarlyStopping(monitor="f1",
                                            patience=8,
                                            mode="max",
                                            )

    ema_callback = EMACallBack()
    swa_callback = StochasticWeightAveraging()

    trainer = pl.Trainer(max_epochs=args.epoch,
                         gpus=[1],
                         # accelerator = 'dp',
                         # plugins=DDPPlugin(find_unused_parameters=True),
                         check_val_every_n_epoch=1,  # 每多少epoch执行一次validation
                         callbacks=[checkpoint_callback,
                                    early_stopping_callback],
                         accumulate_grad_batches=args.grad_accumulations_steps,  # 累计梯度计算
                         precision=16 if args.float16 else 32,  # 半精度训练
                         gradient_clip_val=3,  # 梯度剪裁,梯度范数阈值
                         progress_bar_refresh_rate=5,  # 进度条默认每几个step更新一次
                         # O0：纯FP32训练,
                         # O1：混合精度训练，根据黑白名单自动决定使用FP16（GEMM, 卷积）还是FP32（Softmax）进行计算。
                         # O2：“几乎FP16”混合精度训练，不存在黑白名单，除了Batch norm，几乎都是用FP16计算
                         # O3：纯FP16训练，很不稳定，但是可以作为speed的baseline；
                         amp_level="O1",  # 混合精度训练
                         move_metrics_to_cpu=True,
                         amp_backend="apex",
                         # resume_from_checkpoint =""
                         )
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
