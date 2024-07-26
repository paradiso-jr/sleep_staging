import os
import math
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

from mamba_ssm import Mamba
from utils.func import accuracy, label_weight, calculate_accuracy_per_label
from utils.data import TorchDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from models.encoder import TimeSeriesBertEncoder
from sklearn.metrics import accuracy_score, f1_score
from torch.cuda.amp import GradScaler, autocast


def get_args():
    file_name = 'fine_tuning'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str, required=True)
    parser.add_argument("--local-rank", type=int, default=-1)

    return parser.parse_args()

class TemporalContextModule(nn.Module):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__()
        self.backbone = self.freeze_backbone(backbone)
        self.backbone_final_length = backbone_final_length
        self.embed_dim = embed_dim
        self.embed_layer = nn.Sequential(
            nn.Linear(backbone_final_length, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def apply_backbone(self, x):
        bs = x.shape[0]
        temporal_context_length = x.shape[1]
        x = x.reshape(bs*temporal_context_length, 1, -1)
        x, _, _, _ = self.backbone(x)
        # do not mean across cls and sep token
        x = torch.mean(x[:,1:-1,:], dim=1)
        x = self.embed_layer(x)
        x = x.reshape(bs, temporal_context_length, -1)
        return x

    @staticmethod
    def freeze_backbone(backbone: nn.Module):
        for name, params in backbone.named_parameters():
            if "model.encoder.layer.11" in name or "embedding" in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
            if "quantizer" in name:
                params.requires_grad = False

        return backbone

class MAMBA_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone=backbone, backbone_final_length=backbone_final_length, embed_dim=embed_dim)
        self.mamba_heads = 8
        self.mamba_layer = 1
        self.mamba = nn.Sequential(*[
            Mamba(d_model=self.embed_dim,
                  d_state=16,
                  d_conv=4,
                  expand=2)
            for _ in range(self.mamba_layer)
        ])
        self.fc = nn.Linear(self.embed_dim, 5)

    def forward(self, x):
        x = self.apply_backbone(x)
        x = self.mamba(x)
        x = self.fc(x)
        return x


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
        self.ft_paths = config['FINE_TUNING']['train']['ft_paths']
        
        # model
        self.temporal_context_modules = config['FINE_TUNING']['TCM']['temporal_context_modules']
        self.model = self.get_pretrained_model(config=config).to(self.device)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        # train
        self.batch_size = config['FINE_TUNING']['train']['batch_size']
        self.warmup_epochs = config['FINE_TUNING']['train']['warmup_epochs']
        self.epochs = config['FINE_TUNING']['train']['epochs']
        self.lr = float(config['FINE_TUNING']['train']['lr'])
        self.log_every_n_steps = config['FINE_TUNING']['train']['log_every_n_steps']
        self.n_labels = config['FINE_TUNING']['train']['n_labels']
        self.seed = config['FINE_TUNING']['train']['seed']
        self.accumulation_steps = config['FINE_TUNING']['train']['accumulation_steps']

        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr)

        lambda_lr = self.get_lr_lambda(self.warmup_epochs, self.epochs, self.lr, eta_min=2e-5)
        self.scheduler = opt.lr_scheduler.LambdaLR(self.optimizer, lambda_lr)
        self.scaler = GradScaler(enabled=True)
        torch.cuda.empty_cache()

    def train(self):
        ft_paths = os.listdir(self.ft_paths)
        ft_paths = [x for x in ft_paths if x.endswith('.npz')]
        ft_paths = [os.path.join(self.ft_paths, x) for x in ft_paths]
        dataset = TorchDataset(paths=ft_paths,
                               temporal_context_length=self.temporal_context_length,
                               window_size=self.window_size,
                               sfreq=self.sfreq, rfreq=self.rfreq)
        
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, lengths=[0.96, 0.04])
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=label_weight(dataset.y).to(self.device))
        # 分布式数据集
        train_sampler = DistributedSampler(train_dataset)

        train_dataloader, eval_dataloader = DataLoader(dataset=train_dataset,
                                                       sampler=train_sampler,
                                                       batch_size=self.batch_size,
                                                       shuffle=False), \
                                            DataLoader(dataset=eval_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=False)
        # only write device 0
        # tensorboard logger
        if self.local_rank == 0:
            self.writer = SummaryWriter(comment=f"_finetune_{os.path.basename(self.config_path)}")

            # logging
            logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                                filename=os.path.join(self.writer.logdir, 'training.log'),    
                                level=logging.DEBUG,
                                datefmt='%Y-%m-%d %H:%M:%S',)

            logging.info(f"Start finetune for {self.epochs} epochs.")
            logging.info(f"Training with # step: {len(train_dataloader)}.")
            logging.info(f"Training with batch_size: {self.batch_size}.")
            logging.info(f"Training with learning rate: {self.lr}")
        torch.distributed.barrier()
        state_dict, best_mf1 = None, 0.0
        best_pred, best_real = [], []

        n_iter = 0
        for epoch in range(self.epochs):
            torch.cuda.empty_cache()
            self.model.train()
            epoch_train_loss = []
            
            train_sampler.set_epoch(epoch)
            for batch in train_dataloader:
                with autocast(dtype=torch.float16):
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    if n_iter % self.accumulation_steps != 0:
                        with self.model.no_sync():
                            out = self.model(x)
                            loss, pred, real = self.get_loss(out, y)
                            loss = loss  / self.accumulation_steps
                            self.scaler.scale(loss).backward()
                    else:
                        out = self.model(x)
                        loss, pred, real = self.get_loss(out, y)
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

                    all_pred = [torch.zeros_like(pred) for _ in range(torch.distributed.get_world_size())]
                    torch.distributed.all_gather(all_pred, pred)
                    all_pred = torch.cat(all_pred, dim=0)

                    all_real = [torch.zeros_like(real) for _ in range(torch.distributed.get_world_size())]
                    torch.distributed.all_gather(all_real, real)
                    all_real = torch.cat(all_real, dim=0)
                    if self.local_rank == 0:
                        epoch_train_loss.append(float(all_loss.detach().cpu().item()))

                        top1, top3 = accuracy(all_pred, all_real, topk=(1, 3))
                        self.writer.add_scalar('loss', all_loss, global_step=n_iter)
                        self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                        self.writer.add_scalar('acc/top3', top3[0], global_step=n_iter)
                        self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)

                        # acc per label
                        acc_per_label = calculate_accuracy_per_label(all_pred, all_real)
                        for i, accuarcy in enumerate(acc_per_label):
                            self.writer.add_scalar(f'acc_labels/class_{i}_accuary', accuarcy.item(), global_step=n_iter)

                    torch.distributed.barrier()
                n_iter += 1
            self.scheduler.step()
            
            if self.local_rank == 0:
                self.model.eval()
                epoch_test_loss = []
                epoch_real, epoch_pred = [], []
                with torch.no_grad():
                    for batch in eval_dataloader:
                        x, y = batch
                        x, y = x.to(self.device), y.to(self.device)

                        try:
                            out = self.model.module(x)
                        except IndexError:
                            continue
                        loss, pred, real = self.get_loss(out, y)
                        pred = torch.argmax(pred, dim=-1)
                        epoch_real.extend(list(real.detach().cpu().numpy()))
                        epoch_pred.extend(list(pred.detach().cpu().numpy()))
                        epoch_test_loss.append(float(loss.detach().cpu().item()))

                epoch_train_loss, epoch_test_loss = np.mean(epoch_train_loss), np.mean(epoch_test_loss)
                eval_acc, eval_mf1 = accuracy_score(y_true=epoch_real, y_pred=epoch_pred), \
                                    f1_score(y_true=epoch_real, y_pred=epoch_pred, average='macro')

                logging.info(f"[Epoch] : {epoch+1:03d}")
                logging.info(f"[Train Loss] => {epoch_train_loss:.4f}")
                logging.info(f"[Evaluation Loss] => {epoch_test_loss:.4f}")
                logging.info(f"[Evaluation Accuracy] => {eval_acc:.4f}")
                logging.info(f"[Evaluation Macro-F1] => {eval_mf1:.4f}")
                if best_mf1 < eval_mf1:
                    best_mf1 = eval_mf1
                    model_state = self.model.module.state_dict()
                    best_pred, best_real = epoch_pred, epoch_real
                    self.save_ckpt(model_state, best_pred, best_real)
            torch.distributed.barrier()

    def save_ckpt(self, model_state, pred, real):
        save_path = os.path.join(self.writer.logdir, 'fine_tuning_best_model.pth')
        torch.save({
            'backbone_name': 'BERT_FineTuning',
            'model_state': model_state,
            'result': {'real': real, 'pred': pred},
            'paths': {'train_paths': self.ft_paths}
        }, save_path)

    def get_pretrained_model(self, config):
        # 1. Prepared Pretrained Model
        backbone = TimeSeriesBertEncoder(in_channel=config['PRETRAIN']['model']['in_channel'], 
                                        h_dim=config['PRETRAIN']['model']['h_dim'], 
                                        vocab_size=config['PRETRAIN']['model']['vocab_size'],
                                        beta=config['PRETRAIN']['model']['beta'],)        


        if config['PRETRAIN']['model']['ckpt_path']:
            ckpt = torch.load(config['PRETRAIN']['model']['ckpt_path'], map_location=self.device)
            single_GPU_dict = dict()
            for k, v in ckpt['model_state'].items():
                single_GPU_dict[k[7:]] = v
            backbone.load_state_dict(single_GPU_dict)

        # 2. Temporal Context Module
        tcm = self.get_temporal_context_module()
        model = tcm(backbone=backbone,
                    backbone_final_length=config['PRETRAIN']['model']['embed_dim'],
                    embed_dim=config['FINE_TUNING']['TCM']['embed_dim'])

        if config['FINE_TUNING']['TCM']['ckpt_path']:
            ckpt = torch.load(config['FINE_TUNING']['TCM']['ckpt_path'], map_location=self.device)
            model.load_state_dict(ckpt['model_state'], )

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    def get_temporal_context_module(self):
        if self.temporal_context_modules == 'lstm':
            return LSTM_TCM
        if self.temporal_context_modules == 'mha':
            return MHA_TCM
        if self.temporal_context_modules == 'lstm_mha':
            return LSTM_MHA_TCM
        if self.temporal_context_modules == 'mamba':
            return MAMBA_TCM

    def get_loss(self, pred, real):
        if pred.dim() == 3:
            pred = pred.view(-1, pred.size(2))
            real = real.view(-1, 1)
        labels = F.one_hot(real, num_classes=self.n_labels).float().squeeze().to(self.device)
        loss = self.criterion(pred, labels)
        return loss, pred, real
    
    @staticmethod
    def get_lr_lambda(warmup_epochs, total_epochs, eta_max, eta_min=0,):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warm-up
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * progress))
        return lr_lambda

if __name__ == '__main__':
    augments = get_args()
#    os.environ['CUDA_VISIBLE_DEVICES'] = str(os.environ['LOCAL_RANK'])
    # 每个进程根据自己的local_rank设置应该使用的GPU
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    torch.distributed.init_process_group(backend='nccl')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    trainer = Trainer(augments)
    trainer.train()
