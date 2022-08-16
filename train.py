import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from pytorch_lightning.plugins import DDPPlugin
from EMA import EMACallBack
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import json
import os
from utils import statistics_text_length

def parser_args():
    parser = argparse.ArgumentParser(description='TDEER cli')
    parser.add_argument('--model_type', default="span4re",
                        type=str, help='specify max sample triples', choices=['tdeer', "tplinker","prgc","span4re"])
    parser.add_argument('--pretrain_path', type=str,
                        default="./bertbaseuncased", help='specify the model name')
    parser.add_argument('--relation', type=str,
                        default="./data/data/NYT/rel2id.json", help='specify the relation path')
    parser.add_argument('--train_file', type=str,
                        default="./data/data/NYT/train_triples.json", help='specify the train path')
    parser.add_argument('--val_file', type=str,
                        default="./data/data/NYT/dev_triples.json", help='specify the dev path')
    parser.add_argument('--test_path', type=str,
                        default="./data/data/NYT/test_triples.json", help='specify the test path')
    parser.add_argument('--bert_dir', type=str,
                        help='specify the pre-trained bert model')
    parser.add_argument('--learning_rate', default=5e-5,
                        type=float, help='specify the learning rate')
    parser.add_argument('--epoch', default=100, type=int,
                        help='specify the epoch size')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='specify the batch size')
    parser.add_argument('--output_path', default="event_extract", type=str, help='将每轮的验证结果保存的路径')

    # For TDEER Model
    parser.add_argument('--max_sample_triples', default=None,
                        type=int, help='specify max sample triples')
    parser.add_argument('--neg_samples', default=2, type=int,
                        help='specify negative sample num')

    # for TPlinker Model
    parser.add_argument('--shaking_type', default="cln_plus",choices=['cat',"cat_plus","cln","cln_plus"],
                        type=str, help='specify max sample triples')
    parser.add_argument('--tok_pair_sample_rate', default=1,)
    parser.add_argument('--inner_enc_type', default="lstm", type=str,choices=['mix_pooling',"max_pooling","mean_pooling","lstm"],
                        help='valid only if cat_plus or cln_plus is set. It is the way how to encode inner tokens between each token pairs. If you only want to reproduce the results, just leave it alone.')
    parser.add_argument('--match_pattern', default="whole_text",choices=["whole_text"],
                        type=str, help='specify max sample triples')
    parser.add_argument('--max_seq_len', default=512, type=int,
                        help='specify negative sample num')
    parser.add_argument('--sliding_len', default=20, type=int,
                        help='specify negative sample num')
    parser.add_argument('--loss_weight_recover_steps', default=6000, type=int,
                        help='the paramter of encoder lstm ')

    parser.add_argument('--encoder', default="BERT",
                        type=str, help='specify max sample triples')
    parser.add_argument('--enc_hidden_size', default=100, type=int,
                        help='the paramter of encoder lstm ')
    parser.add_argument('--dec_hidden_size', default=20, type=int,
                        help='the paramter of encoder lstm ')
    parser.add_argument('--emb_dropout', default=0.1,
                        type=float, help='the paramter of encoder lstm')
    parser.add_argument('--rnn_dropout', default=0.1, type=float,
                        help='the paramter of encoder lstm ')
    parser.add_argument('--word_embedding_dim', default=300, type=int,
                        help='the paramter of encoder lstm ')
    parser.add_argument('--data_out_dir', default="./data/data/NYT", type=str,
                        help='处理后的数据保存的路径')
    # for tplinker preprocess args
    parser.add_argument('--separate_char_by_white', default=False, type=bool,
                        help='e.g. "$%sdkn839," -> "$% sdkn839," , will make original char spans invalid')
    parser.add_argument('--add_char_span', default=True, type=bool,
                        help='set add_char_span to false if it already exists')
    parser.add_argument('--ignore_subword', default=True, type=bool,
                        help=' when adding character level spans, match words with whitespace around: " word ", to avoid subword match, set false for chinese')
    parser.add_argument('--check_tok_span', default=True, type=bool,
                        help="check whether there is any error with token spans, if there is, print the unmatch info")
    parser.add_argument(
        '--ent2id_path', default="./data/data/NYT/ent2id.json", type=str, help="预处理的实体标签的保存路径")
    parser.add_argument('--ghm', default=False,type=bool,help="是否使用GHM算法进行损失平滑")
    parser.add_argument('--decay_rate', default=0.999,type=float,help="StepLR 参数")
    parser.add_argument('--decay_steps', default=100,type=int,help="StepLR 参数")
    parser.add_argument('--T_mult', default=1,type=float,help="CosineAnnealingWarmRestarts 参数")
    parser.add_argument('--rewarm_epoch_num', default=2,type=int,help="CosineAnnealingWarmRestarts 参数")

    # for PRGC model
    parser.add_argument('--corres_threshold', type=float,
                        default=0.5, help="threshold of global correspondence")
    parser.add_argument('--rel_threshold', type=float,
                        default=0.5, help="threshold of relation judgement")
    parser.add_argument('--emb_fusion', type=str,
                        default="concat", help="way to embedding")
    parser.add_argument('--ensure_rel', default=True, help="是否需要对关系进行负采样")
    parser.add_argument('--num_negs', type=int, default=4,
                    help="当对关系进行负采样时,负采样的个数")
    parser.add_argument('--drop_prob', type=float, default=0.2,help="对各个预测模块采用的drop out率")
    
    # for span4re model argument
    parser.add_argument('--loss_weight', type=list,
                        default=[1,2,2], help="关系,subject和object的损失权重")
    parser.add_argument('--na_rel_coef', type=float,
                        default=1, help="无关关系的权重系数")
    parser.add_argument('--matcher',type=str, default="avg", choices=['avg', 'min'])
    parser.add_argument('--num_decoder_layers',type=int, default=3, help="decoder 的层数")
    parser.add_argument('--num_generated_triples', type=int, default=10,
                    help="解码时，最大生成triple的数量")
    parser.add_argument('--n_best_size', type=int, default=5,help="解码时,选择前多少个triple作为最终的triple")
    
    args = parser.parse_args()
    return args


def main():
    args = parser_args()
    if args.model_type == "tdeer":
        from TDeer_Model import TDEERDataset, collate_fn, collate_fn_val, TDEERPytochLighting
        train_dataset = TDEERDataset(args.train_file, args, is_training=True)
        train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn,
                                      batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_dataset = TDEERDataset(args.val_file, args, is_training=False)
        val_dataloader = DataLoader(
            val_dataset, collate_fn=collate_fn_val, batch_size=args.batch_size, shuffle=False)
        
        relation_number = train_dataset.relation_size
        args.relation_number = relation_number
        args.steps = len(train_dataset)
        model = TDEERPytochLighting(args)

    elif args.model_type == "tplinker":
        from TPlinker_Model import TPlinkerDataset, TPlinkerPytochLighting
        from tplinker_utils import HandshakingTaggingScheme, DataMaker4Bert, TplinkerDataProcess

        tokenizer = BertTokenizerFast.from_pretrained(
            args.pretrain_path, cache_dir="./bertbaseuncased", add_special_tokens=False, do_lower_case=True)
        max_length = statistics_text_length(args.train_file,tokenizer)
        print("最大文本长度为:",max_length)
        args.max_seq_len = max_length+2
        get_tok2char_span_map = lambda text: tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        # 数据预处理
        data_path = os.path.join(args.data_out_dir, "train.json")
        if not os.path.exists(data_path):
            TplinkerDataProcess(args, args.train_file,
                                get_tok2char_span_map, is_training=True)
        data_path = os.path.join(args.data_out_dir, "val.json")
        if not os.path.exists(data_path):
            TplinkerDataProcess(args, args.val_file,
                                get_tok2char_span_map, is_training=False)

        handshaking_tagger = HandshakingTaggingScheme(args)
        data_maker = DataMaker4Bert(tokenizer, handshaking_tagger)

        train_dataset = TPlinkerDataset(
            args, data_maker, tokenizer, is_training=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6,
                                      collate_fn=data_maker.generate_batch)
        val_dataset = TPlinkerDataset(
            args, data_maker, tokenizer, is_training=False)
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, collate_fn=data_maker.generate_batch)
        args.num_step = len(train_dataloader)
        model = TPlinkerPytochLighting(args, handshaking_tagger)

    elif args.model_type == "prgc":
        from PRGC_Model import PRGCDataset,PRGCPytochLighting,collate_fn_test,collate_fn_train
        tokenizer = BertTokenizerFast.from_pretrained(args.pretrain_path,cache_dir = "./bertbaseuncased")
        max_length = statistics_text_length(args.train_file,tokenizer)
        print("最大文本长度为:",max_length)
        args.max_seq_len = max_length
        
        train_dataset = PRGCDataset( args,args.train_file, is_training=True)
        train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_train,
                                      batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_dataset = PRGCDataset( args,args.val_file, is_training=False)
        val_dataloader = DataLoader(
            val_dataset, collate_fn=collate_fn_test, batch_size=args.batch_size, shuffle=False)
        
        relation_number = train_dataset.relation_size
        args.relation_number = relation_number
        model = PRGCPytochLighting(args)
    
    elif args.model_type == "span4re":
        from SPN4RE_Model import Span4REDataset,Span4REPytochLighting,collate_fn
        tokenizer = BertTokenizerFast.from_pretrained(args.pretrain_path,cache_dir = "./bertbaseuncased")
        max_length = statistics_text_length(args.train_file,tokenizer)
        print("最大文本长度为:",max_length)
        args.max_span_length = max_length
        train_dataset = Span4REDataset(args.train_file, args, is_training=True)
        train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn,
                                      batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_dataset = Span4REDataset(args.val_file, args, is_training=False)
        val_dataloader = DataLoader(
            val_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False)
        
        relation_number = train_dataset.relation_size
        args.relation_number = relation_number
        args.steps = len(train_dataset)
        model = Span4REPytochLighting(args)
        
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

    trainer = pl.Trainer(max_epochs=20,
                         gpus=[1],
                         # accelerator = 'dp',
                         # plugins=DDPPlugin(find_unused_parameters=True),
                         check_val_every_n_epoch=1,  # 每多少epoch执行一次validation
                         callbacks=[checkpoint_callback,
                                    early_stopping_callback],
                         accumulate_grad_batches=3,  # 累计梯度计算
                         # precision=16, # 半精度训练
                         gradient_clip_val=3,  # 梯度剪裁,梯度范数阈值
                         progress_bar_refresh_rate=5,  # 进度条默认每几个step更新一次
                         amp_level="O1",  # 混合精度训练
                         move_metrics_to_cpu=True,
                         amp_backend="apex",
                         # resume_from_checkpoint ="lightning_logs/version_5/checkpoints/epoch=01f1=0.950acc=0.950recall=0.950.ckpt"
                         )
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
