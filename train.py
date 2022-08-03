import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint,StochasticWeightAveraging
from EMA import EMACallBack
import pytorch_lightning as pl
from torch.utils.data import DataLoader


def parser_args():
    parser = argparse.ArgumentParser(description='TDEER cli')
    parser.add_argument('--pretrain_path', type=str, default="bertbaseuncased", help='specify the model name')
    parser.add_argument('--relation', type=str, default="TDEER/data/data/NYT/rel2id.json", help='specify the relation path')
    parser.add_argument('--train_file', type=str,default="TDEER-Torch/data/data/NYT/train_triples.json", help='specify the train path')
    parser.add_argument('--val_file', type=str,default="TDEER-Torch/data/data/NYT/dev_triples.json", help='specify the dev path')
    parser.add_argument('--test_path', type=str,default="TDEER-Torch/data/data/NYT/test_triples.json", help='specify the test path')
    parser.add_argument('--bert_dir', type=str, help='specify the pre-trained bert model')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='specify the learning rate')
    parser.add_argument('--epoch', default=100, type=int, help='specify the epoch size')
    parser.add_argument('--batch_size', default=40, type=int, help='specify the batch size')
    parser.add_argument('--model_type', default="tdeer",type=str, help='specify max sample triples')
    parser.add_argument('--output_path', default="event_extract", type=str, help='将每轮的验证结果保存的路径')

    # For TDEER Model
    parser.add_argument('--max_sample_triples', default=None, type=int, help='specify max sample triples')
    parser.add_argument('--neg_samples', default=2, type=int, help='specify negative sample num')

    
    args = parser.parse_args()
    return args

args = parser_args()


checkpoint_callback = ModelCheckpoint(
    save_top_k= 8,
    verbose=True,
    monitor='f1', # 监控val_acc指标
    mode='max',
    save_last = True, 
    # dirpath = args.save_model_path.format(args.model_type),
    every_n_epochs = 1,
    # filename = "{epoch:02d}{f1:.3f}{acc:.3f}{recall:.3f}",
    filename = "{epoch:02d}{f1:.3f}{acc:.3f}{recall:.3f}{sr_rec:.3f}{sr_acc:.3f}",
)

early_stopping_callback = EarlyStopping(monitor="f1",
                            patience = 8,
                            mode = "max",
                            )

if args.model_type == "tdeer":
    from TDeer_Model import TDEERDataset,collate_fn,collate_fn_val,TDEERPytochLighting
    train_dataset = TDEERDataset(args.train_file,args,is_training=True)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True,num_workers=8,pin_memory=True)
    val_dataset = TDEERDataset(args.val_file,args,is_training=False)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn_val, batch_size=args.batch_size, shuffle=False)
    tokenizer_size = len(train_dataset.tokenizer)
    relation_number = train_dataset.relation_size
    args.relation_number = relation_number
    model = TDEERPytochLighting(args)

elif args.model_type == "tplinker":
    pass
elif args.model_type == "prgc":
    pass


ema_callback = EMACallBack()
swa_callback = StochasticWeightAveraging()

trainer = pl.Trainer(max_epochs=20, 
                    gpus=[0], 
                    # accelerator = 'dp',
                    # plugins=DDPPlugin(find_unused_parameters=True),
                    check_val_every_n_epoch=1, # 每多少epoch执行一次validation
                    callbacks = [checkpoint_callback,early_stopping_callback],
                    accumulate_grad_batches = 3,# 累计梯度计算
                    # precision=16, # 半精度训练
                    gradient_clip_val=3, #梯度剪裁,梯度范数阈值
                    progress_bar_refresh_rate = 5, # 进度条默认每几个step更新一次
                    amp_level = "O1",# 混合精度训练
                    move_metrics_to_cpu = True,
                    amp_backend = "apex",
                    # resume_from_checkpoint ="lightning_logs/version_5/checkpoints/epoch=01f1=0.950acc=0.950recall=0.950.ckpt"
                    )

trainer.fit(model,train_dataloader=train_dataloader,val_dataloaders=val_dataloader)


