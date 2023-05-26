import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_constant_schedule_with_warmup
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from mapl import MAPL
from datasets import COCODataset


class LitMAPL(LightningModule):

    def __init__(self, clip_model_id: str, gpt_model_id: str,  *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = MAPL(clip_model_id=clip_model_id, gpt_model_id=gpt_model_id)
    
    def training_step(self, batch, batch_idx):
        output = self.model(pixel_values=batch['image'], target_ids=batch['caption'])
        loss = output.loss
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        output = self.model(pixel_values=batch['image'], target_ids=batch['caption'])
        loss = output.loss
        self.log('val_loss', loss, sync_dist=True)
    
    def get_param_groups(self):
        with_wd = []
        without_wd = []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if p.ndim < 2 or 'bn' in n or 'ln' in n or 'bias' in n:
                    without_wd.append(p)
                else:
                    with_wd.append(p)
        return [
            {'params': with_wd},
            {'params': without_wd, 'weight_decay': 0}
        ]
    
    def configure_optimizers(self):
        params = self.get_param_groups()
        optimizer = AdamW(
            params,
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
            weight_decay=self.hparams.weight_decay
        )
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]
    
    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for name, param in state_dict.items():
            if name.startswith('model.mapper'):
                new_name = name[len('model.mapper.'):]
                new_state_dict[new_name] = param
        checkpoint['state_dict'] = new_state_dict
        return checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=Path.home() / 'scratch/datasets')
    parser.add_argument('--save_dir', type=str, default=Path.home() / 'scratch/lightning_logs')
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_devices', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_per_device', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=float, default=1500)
    parser.add_argument('--clip_model_id', type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument('--gpt_model_id', type=str, default='EleutherAI/gpt-j-6B')
    return parser.parse_args()


def main():
    args = parse_args()

    torch.set_float32_matmul_precision('high')
    
    lit_model = LitMAPL(**vars(args))
    model = lit_model.model

    data_dir = Path(args.data_dir) / 'coco'
    train_dataset = COCODataset(split='train', data_dir=data_dir, image_transform=model.image_transform, text_transform=model.text_transform)
    val_dataset = COCODataset(split='val', data_dir=data_dir, image_transform=model.image_transform, text_transform=model.text_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_per_device, shuffle=True, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_per_device, shuffle=False, num_workers=args.num_workers, collate_fn=val_dataset.collate_fn)

    logger = WandbLogger(save_dir=args.save_dir, project='mapl')
    callbacks = [
        ModelCheckpoint(monitor='val_loss', mode='min', save_weights_only=True),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(monitor='val_loss', mode='min', patience=3)
    ]

    num_devices = torch.cuda.device_count() * args.num_nodes
    accumulate_grad_batches = max(1, args.batch_size // (args.batch_size_per_device * num_devices))

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        num_nodes=args.num_nodes,
        devices=args.num_devices,
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_false',
        precision='bf16',
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=-1
    )
    trainer.fit(lit_model, train_loader, val_loader)


if __name__ == '__main__':
    main()
