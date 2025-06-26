from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import lightning as pl
import torch.nn as nn


class ModelCompilation(pl.LightningModule):

    def __init__(self,
                 model: torch.nn.Module,
                 epochs: int,
                 mode: str,
                 metrics: dict,
                 loss_function,
                 optimizer: torch.optim,
                 learning_rate: float,
                 lr_schedule: str,
                 accelerator: str):
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.mode = mode
        self.metrics = nn.ModuleDict({
            stage: metrics.clone(prefix=stage) for stage in ["train_", "val_", "test_"]
        })
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.accelerator = accelerator

    def configure_optimizers(self):
        self.train_optimizer = self.optimizer(self.parameters(), lr=self.learning_rate, weight_decay=0.05,
                                              betas=(0.9, 0.98), eps=1.0e-9)
        return {
            'optimizer': self.train_optimizer,
            'lr_scheduler': self.get_lr_schedule(self.lr_schedule)
        }

    def forward(self, x_n, x_a):
        pred = self.model.forward(x_n, x_a)
        return pred

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, 'test')
        return loss

    def common_step(self, batch, batch_idx, stage):
        normal_images, anomaly_images, anomaly_masks = batch  # Извлекаем батчи с нормальными, аномальными изображениями и масками.
        pred_mask = self.forward(x_n=normal_images,
                                 x_a=anomaly_images)  # Подаём их на вход модели и получаем предсказание в виде маски
        loss = self.loss_function(pred_mask, anomaly_masks)  # Осуществляем расчёт ошибки
        pred_mask = pred_mask.sigmoid()  # Применение сигмоиды к выходу модели для получения маски с вероятностями

        on_step = False if (stage == 'test') or (
                stage == 'val') else True  # Логирование каждого шага только для обучения

        # Считаем метрики в зависимости от текущего stage
        stage_prefix = stage + "_"
        current_metrics = self.metrics[stage_prefix]
        current_metrics(pred_mask, anomaly_masks)

        # Логирование метрик
        self.log_dict(current_metrics, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)
        self.log(stage_prefix + 'loss', loss, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)

        # Логирование скорости обучения в процессе тренировки
        if (stage == 'train'):
            self.log('learing_rate', self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return loss

    def get_lr_schedule(self, lr_schedule):
        return {'scheduler': ReduceLROnPlateau(optimizer=self.train_optimizer, mode="min", factor=0.5, patience=7,
                                               min_lr=1e-6),
                'interval': 'epoch',  # Update the scheduler every epoch
                'frequency': 1,  # Apply the scheduler every epoch
                "monitor": "val_loss",
                }
