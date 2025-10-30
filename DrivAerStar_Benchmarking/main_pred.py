import os
import sys
import time
import yaml
import wandb

import lightning as L
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger,CSVLogger

from data_module import *
from networks import *
from utils import get_args,class_builder,class_name
from pytorch_lightning.callbacks import ModelCheckpoint

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def set_logger(model_type,data_type=''):
    Timestamp = time.strftime('test_%m%d_%H_%M_%S')

    writer_path = os.path.join('logs', model_type, data_type ,Timestamp)

    if not os.path.exists(writer_path):
        os.makedirs(writer_path)

    return Timestamp, writer_path


def main():
    args = get_args()
    print('===args===')
    print(args)
    print('===args===')
    config = load_config(args.config_path)

    Timestamp, writer_path= set_logger(model_type = config['network']['type'],data_type = config['Data']['type'])
    logger = CSVLogger(writer_path)
    if args.use_wandb:
        logger = (logger,WandbLogger(project="MNIST"))
    
    # save config
    with open(os.path.join(writer_path, 'config.yaml'), 'w') as f_write:
        config['load_ckpt'] = args.load_ckpt_path
        yaml.safe_dump(config, f_write)

    # data_module
    data_module = class_builder(config['Data'])
    data_module.setup()

    args.model_type = config['network']['type']
    print(args)

    print('train dataset length', len(data_module.train_dataloader().dataset))
    print('val dataset length', len(data_module.val_dataloader().dataset))

    # Trainer

    early_stop_callback = L.pytorch.callbacks.EarlyStopping(
        monitor="val_loss", 
        mode="min", 
        patience=50)

    ckpt_callback = L.pytorch.callbacks.ModelCheckpoint(
        # dirpath=args.load_ckpt_path,
        monitor='val_loss',
        save_top_k=1,
        mode='min',
    ) 

    # seed_everything
    L.fabric.seed_everything(config['Seed'])
    trainer = L.Trainer(**config['Trainer']['args'], 
                        logger=logger,
                        callbacks=[early_stop_callback, ckpt_callback])

    # 模型
    if 'cross' in config['network']['type']:
        '''
        两组normalizer, 一组是针对点数据，一组针对矩阵数据
        '''
        model = class_builder(
            config['network'],
            normalizer_y=data_module.normalizer_y, 
            normalizer_x=data_module.normalizer_x,
            point_normalizer_x = data_module.point_normalizer_x,
            point_normalizer_y = data_module.point_normalizer_y,
            use_wandb=args.use_wandb)
    else:    
        if args.load_ckpt_path is None:
            model = class_builder(config['network'])
        else:
            model = class_name(
                config['network']).load_from_checkpoint(args.load_ckpt_path, config['network'] )

        
    summary = pl.utilities.model_summary.ModelSummary(model,max_depth=1)
    print(summary) 

    # test
    model.set_normalizer(
                normalizer_y=data_module.normalizer_y, 
                normalizer_x=data_module.normalizer_x,)
    model.set_test_args(writer_path)
    res = trainer.predict(model=model, 
                    ckpt_path = args.load_ckpt_path,
                dataloaders=data_module.test_dataloader())
    

    
    if wandb.run is not None:
        wandb.finish()
    if args.use_tb:
        fp.close()
        
if __name__ == '__main__':
    main()
