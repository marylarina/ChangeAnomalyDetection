import segmentation_models_pytorch as smp
import torch.nn as nn


class FocalDiceLoss(nn.Module):

    def __init__(self,
                 mode,
                 gamma=2.0,
                 alpha=None,
                 beta=5.0,
                 smooth=0.0,
                 ignore_index=-100):
        super(FocalDiceLoss, self).__init__()
        self.beta = beta
        self.dice_loss = smp.losses.DiceLoss(mode=mode, smooth=smooth, ignore_index=ignore_index)
        self.focal_loss = smp.losses.FocalLoss(mode=mode, gamma=gamma, alpha=alpha, ignore_index=ignore_index)

    def forward(self, y_pred, y_true):

        # Используем комбинацию двух функций ошибок: Focal Loss + Dice Loss.
        # Коэффицент beta необходим для выравнивания различий в масштабе значений ошибки между функциями
        return self.beta * self.focal_loss(y_pred.contiguous(), y_true) + self.dice_loss(y_pred, y_true)
