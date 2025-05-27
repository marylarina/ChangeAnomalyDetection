import torch.nn as nn
from einops import rearrange
import torch

class HardAlignmentModule(nn.Module):
    def __init__(self, in_channels):
        super(HardAlignmentModule, self).__init__()
        # 1x1 свёртка для уменьшения размерности канала
        self.conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)

    def forward(self, Fq_l, Fp_l):
        # Входные размерности
        b, c, h, w = Fq_l.shape

        # Сохраняем оригинальные признаки до свёртки
        Fp_l_original = Fp_l.clone()

        # Применяем свёртку для уменьшения размерности канала у признаков запроса и подсказки
        Fq_l = self.conv(Fq_l)  # Признаки запроса после свёртки
        Fp_l = self.conv(Fp_l)  # Признаки подсказки после свёртки

        # Переводим признаки в форму [батч, каналы, высота*ширина] для упрощения вычислений
        Fq_l_flat = rearrange(Fq_l, "b c h w -> b c (h w)")
        Fp_l_flat = rearrange(Fp_l, "b c h w -> b c (h w)")

        # Вычисляем косинусное сходство между признаками запроса и подсказки (сперва нормализуем их)
        Fq_l_flat = Fq_l_flat / torch.norm(Fq_l_flat, dim=1, keepdim=True)
        Fp_l_flat = Fp_l_flat / torch.norm(Fp_l_flat, dim=1, keepdim=True)

        # Вычисляем косинусное сходство между признаками запроса и подсказки
        similarity = torch.bmm(Fq_l_flat.transpose(1, 2), Fp_l_flat)  # Размерность карты попарных сходств: [b, h*w, h*w]

        # Находим наиболее похожий признак подсказки для каждого признака запроса
        _, idx = similarity.max(dim=-1)  # Индексы наиболее похожих признаков подсказки (h*w)

        # Преобразуем индексы обратно в 2D-координаты (высота, ширина)
        idx = idx.view(b, 1, h * w)  # Размерность: [b, h, w]
        idx = idx.expand(-1, c, -1)  # Мы расширяем индексы по каналам, чтобы применить их ко всем каналам изображения

        # Извлекаем наиболее похожие признаки из исходных признаков подсказки
        Fp_l_original_flat = rearrange(Fp_l_original, "b c h w -> b c (h w)")
        aligned_Fp_l = Fp_l_original_flat.gather(2, idx)  # Извлекаем признаки из исходных данных

        # Преобразуем выровненные признаки обратно в пространственные размеры
        aligned_Fp_l = rearrange(aligned_Fp_l, "b c (h w) -> b c h w", h=h, w=w)

        return aligned_Fp_l


class SoftAlignmentModule(nn.Module):
    def __init__(self, in_channels):
        super(SoftAlignmentModule, self).__init__()
        # 1x1 свёртка для уменьшения размерности канала в 2 раза
        self.conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)

    def forward(self, Fq_l, Fp_l):
        # Входные размерности
        b, c, h, w = Fq_l.shape

        # Сохраняем оригинальные признаки до свёртки
        Fp_l_original = Fp_l.clone()

        # Применяем свёртку для уменьшения размерности канала у признаков запроса и подсказки
        Fq_l = self.conv(Fq_l)  # Признаки запроса после свёртки
        Fp_l = self.conv(Fp_l)  # Признаки подсказки после свёртки

        # Переводим признаки в форму [батч, каналы, высота*ширина] для упрощения вычислений
        Fq_l_flat = rearrange(Fq_l, "b c h w -> b c (h w)")
        Fp_l_flat = rearrange(Fp_l, "b c h w -> b c (h w)")

        # Вычисляем скалярное произведение между признаками запроса и подсказки
        similarity = torch.bmm(Fq_l_flat.transpose(1, 2), Fp_l_flat)  # Размерность: [b, h*w, h*w]

        # Применяем softmax чтобы получить вероятность для каждого пикселя подсказки в отношении к каждому пикселю запроса.
        attention_weights = torch.softmax(similarity, dim=-1)  # Размерность: [b, h*w, h*w]

        # Взвешиваем признаки подсказки с использованием attention_weights
        Fp_l_original_flat = rearrange(Fp_l_original, "b c h w -> b c (h w)")
        aligned_Fp_l = torch.bmm(attention_weights, Fp_l_original_flat.transpose(1, 2))  # Размерность: [b, h*w, c]

        # Преобразуем обратно в пространственные размеры
        aligned_Fp_l = rearrange(aligned_Fp_l, "b (h w) c -> b c h w", h=h, w=w)

        return aligned_Fp_l


import numpy as np
from sklearn.metrics import roc_curve



def find_optimal_threshold(y_true, y_pred):
    """
    Метод для нахождения оптимального порога, который максимизирует разницу между
    True Positive Rate (TPR) и False Positive Rate (FPR) с использованием ROC-кривой.

    Метод использует ROC-кривую (Receiver Operating Characteristic curve),
    которая позволяет оценить, как хорошо модель различает аномалии от нормальных пикселей при разных значениях порога.

    ROC-кривая — это график, который строится на основе двух величин:
    - **TPR (True Positive Rate)** — доля правильно предсказанных аномальных пикселей.
    - **FPR (False Positive Rate)** — доля пикселей, которые были ошибочно предсказаны как аномальные, хотя на самом деле они нормальные.

    Чем больше разница между TPR и FPR, тем лучше модель, потому что она правильно классифицирует аномалии и не ошибается с нормальными пикселями.

    Мы вычисляем разницу между TPR и FPR для разных порогов. Порог, при котором эта разница максимальна, и будет оптимальным.
    Он обеспечит наилучшее разделение между аномальными и нормальными пикселями.

    Параметры:
    - y_true (array-like): Истинные метки (0 для нормальных пикселей, 1 для аномальных).
    - y_pred (array-like): Предсказанные вероятности, указывающие вероятность того, что пиксель является аномалией.

    Возвращает:
    - optimal_threshold (float): Оптимальный порог для классификации пикселей как аномальных.
    """
    # Вычисляем ROC-кривую
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Находим оптимальный порог (максимальное значение TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold