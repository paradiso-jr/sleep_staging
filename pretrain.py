import os
import yaml
import random
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

from tensorboardX import SummaryWriter

from utils.func import accuracy
from utils.augment import AugmentMethod
from utils.data import TorchDataset
from utils.loss import info_nce_loss
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from models.encoder import TimeSeriesBertEncoder
from torch.cuda.amp import GradScaler, autocast
def get_args():
    file_name = 'pretrain'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str, required=True)
    # parser.add_argument("--local-rank", type=int, default=-1)

    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        super().__init__()
        self.init_params(args)
       
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def init_params(self, args):
        # device
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.device = torch.device('cuda', self.local_rank)

        # config
        config = yaml.safe_load(open(args.config, 'r'))
        self.config_path = args.config

        # dset
        self.temporal_context_length = config['FINE_TUNING']['TCM']['temporal_context_length']
        self.window_size = config['FINE_TUNING']['TCM']['window_size']
        self.sfreq, self.rfreq = config['FINE_TUNING']['TCM']['sfreq'], config['FINE_TUNING']['TCM']['rfreq']
        self.ft_paths = config['PRETRAIN']['train']['ft_paths']
        
        # model
        self.model = self.get_pretrained_model(config=config).to(self.device)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        # train
        self.batch_size = config['PRETRAIN']['train']['batch_size']
        self.epochs = config['PRETRAIN']['train']['epochs']
        self.lr = float(config['PRETRAIN']['train']['lr'])
        self.log_every_n_steps = config['PRETRAIN']['train']['log_every_n_steps']
        self.n_labels = config['PRETRAIN']['train']['n_labels']
        self.seed = config['PRETRAIN']['train']['seed']
        self.accumulation_steps = config['PRETRAIN']['train']['accumulation_steps']
        self.temperature = float(config['PRETRAIN']['train']['temperature'])

        self.augment = AugmentMethod()
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.scaler = GradScaler(enabled=True)
        torch.cuda.empty_cache()

    
    def get_pretrained_model(self, config):
        # 1. Prepared Pretrained Model
        backbone = TimeSeriesBertEncoder(in_channel=config['PRETRAIN']['model']['in_channel'], 
                                        h_dim=config['PRETRAIN']['model']['h_dim'], 
                                        vocab_size=config['PRETRAIN']['model']['vocab_size'],
                                        beta=config['PRETRAIN']['model']['beta'],)
        backbone = nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
        if config['PRETRAIN']['model']['ckpt_path']:
            ckpt = torch.load(config['bert']['ckpt_path'])
            backbone.load_state_dict(ckpt['state_dict'])

        return backbone
    
    def train(self):
        ft_paths = os.listdir(self.ft_paths)
        ft_paths = [x for x in ft_paths if x.endswith('.npz')]
        ft_paths = [os.path.join(self.ft_paths, x) for x in ft_paths]
        dataset = TorchDataset(paths=ft_paths,
                               temporal_context_length=self.temporal_context_length,
                               window_size=self.window_size,
                               sfreq=self.sfreq, rfreq=self.rfreq)
                
        # 分布式数据集
        train_sampler = DistributedSampler(dataset)

        train_dataloader = DataLoader(dataset=dataset,
                                        sampler=train_sampler,
                                        batch_size=self.batch_size,
                                        shuffle=False)
        # tensorboard logger
        if self.local_rank == 0:
            self.writer = SummaryWriter(comment=f"_pretrain_{os.path.basename(self.config_path)}")

            # logging
            logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                                filename=os.path.join(self.writer.logdir, 'training.log'),    
                                level=logging.DEBUG,
                                datefmt='%Y-%m-%d %H:%M:%S',)

            logging.info(f"Start pretrain for {self.epochs} epochs.")
            logging.info(f"Training with # step: {len(train_dataloader)}.")
            logging.info(f"Training with batch_size: {self.batch_size}.")
            logging.info(f"Training with learning rate: {self.lr}")
        torch.distributed.barrier()
        n_iter = 0
        for epoch in range(self.epochs):
            torch.cuda.empty_cache()
            self.model.train()
            train_sampler.set_epoch(epoch)
            for batch in train_dataloader:
                x, _ = batch
                bs = x.shape[0]
                temporal_context_length = x.shape[1]
                x = x.reshape(bs*temporal_context_length, -1)

                x_aug = self.augment(x)

                x_concat = torch.concat([x, x_aug], dim=0)
                x_concat = x_concat.unsqueeze(1).float().to(self.device)
                with autocast(dtype=torch.float16):
                    if n_iter % self.accumulation_steps != 0:
                        with self.model.no_sync():
                            last_hidden_state, _, embedding_loss, _ = self.model(x_concat)
                            last_hidden_state = torch.mean(last_hidden_state[:,1:-1,:], dim=1)
                            logits, labels, nce_loss = info_nce_loss(last_hidden_state,
                                                                    temperature=self.temperature)
                            loss = nce_loss + embedding_loss
                            loss = loss  / self.accumulation_steps
                            self.scaler.scale(loss).backward()
                    else:
                        last_hidden_state, _, embedding_loss, _ = self.model(x_concat)
                        last_hidden_state = torch.mean(last_hidden_state[:,1:-1,:], dim=1)
                        logits, labels, nce_loss = info_nce_loss(last_hidden_state,
                                                                temperature=self.temperature)
                        loss = nce_loss + embedding_loss
                        loss = loss  / self.accumulation_steps
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                    
                # logging
                if n_iter % self.log_every_n_steps == 0:
                    # gather scalers
                    all_loss = [torch.zeros_like(loss) for _ in range(torch.distributed.get_world_size())]
                    torch.distributed.all_gather(all_loss, loss)
                    all_loss = torch.mean(torch.stack(all_loss))

                    all_logits = [torch.zeros_like(logits) for _ in range(torch.distributed.get_world_size())]
                    torch.distributed.all_gather(all_logits, logits)
                    all_logits = torch.cat(all_logits, dim=0)

                    all_labels = [torch.zeros_like(labels) for _ in range(torch.distributed.get_world_size())]
                    torch.distributed.all_gather(all_labels, labels)
                    all_labels = torch.cat(all_labels, dim=0)

                    if self.local_rank == 0:
                        top1, top5 = accuracy(all_logits, all_labels, topk=(1, 5))
                        self.writer.add_scalar('loss', all_loss, global_step=n_iter)
                        self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                        self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                        self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                        self.save_ckpt(self.model.state_dict())

                    torch.distributed.barrier()
                n_iter += 1
            self.scheduler.step()
    def save_ckpt(self, model_state):
        save_path = os.path.join(self.writer.logdir, 'pretrain_model.pth')
        torch.save({
            'backbone_name': 'BERT_pretrain',
            'model_state': model_state,
            'paths': {'train_paths': self.ft_paths}
        }, save_path)


if __name__ == '__main__':
    augments = get_args()

    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    torch.distributed.init_process_group(backend='nccl')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    trainer = Trainer(augments)
    trainer.train()
