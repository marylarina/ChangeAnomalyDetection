import torch
import timm
from einops import rearrange
from torch import nn, einsum
from timm.models.layers import trunc_normal_
import torch.nn.functional as F

from HardAndSoftAlign import SoftAlignmentModule, HardAlignmentModule


class DefectVisionFormer(nn.Module):

    def __init__(self,
                 encoder_name,
                 align_type,
                 fusion_type,
                 decoder_name,
                 num_classes):
        super(DefectVisionFormer, self).__init__()

        # Инициализация энкодера (сверточной нейросети для извлечения признаков)
        self.encoder = self.get_encoder(encoder_name=encoder_name)
        # Получение количества выходных каналов на каждом уровне энкодера.
        encoder_channels = self.get_encoder_channels()

        # Определяем число голов в механизме внимания (в зависимости от глубины энкодера), по-умолчанию 1 везде
        num_heads = [1, 1, 1, 1, 1] if (len(encoder_channels) == 5) else [1, 1, 1, 1]

        # Инициализация модуля выравнивания признаков AlignFeatureModule
        self.feature_alignment_module = AlignFeatureModule(align_type=align_type, fusion_type=fusion_type,
                                                               encoder_channels=encoder_channels, num_heads=num_heads)

        # Определение каналов для декодера (на основе энкодера и метода объединения признаков)
        decoder_channels = self.get_decoder_channels(encoder_channels=encoder_channels, fusion_type=fusion_type)
        # Инициализация декодера для сегментации дефектов.
        self.decoder = self.get_decoder(decoder_name=decoder_name, decoder_channels=decoder_channels,
                                        num_classes=num_classes)

    def forward(self, x_n, x_a):
        features_n = self.encoder(x_n)
        features_a = self.encoder(x_a)
        features_n_a = self.feature_alignment_module(features_n, features_a)
        output_mask = self.decoder(features_n_a)
        final_mask = nn.functional.interpolate(output_mask, size=x_n.shape[-2:], mode="bilinear", align_corners=False)
        return final_mask

    def get_encoder(self, encoder_name):
        if encoder_name == 'efficientnet_b4':
            encoder = timm.create_model(model_name="tf_efficientnet_b4.ns_jft_in1k", pretrained=True,
                                        features_only=True)
        elif encoder_name == 'efficientnet_v2_s':
            encoder = timm.create_model(model_name="tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True,
                                        features_only=True)
        elif encoder_name == 'convnext_v2':
            encoder = timm.create_model(model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384", pretrained=True,
                                        features_only=True)
        else:
            raise ValueError(f"Неизвестное имя энкодера: {encoder_name}")

        # Замораживаем все слои, чтобы веса энкодераа не обучались
        for param in encoder.parameters():
            param.requires_grad = False

        # Перевод в режим оценки (выключает обучение BatchNorm и Dropout)
        encoder.eval()

        return encoder

    def get_encoder_channels(self):
        encoder_info = self.encoder.feature_info
        encoder_channels = [info['num_chs'] for info in encoder_info]
        return encoder_channels

    def get_decoder_channels(self, encoder_channels, fusion_type):
        decoder_channels = encoder_channels[:]
        if fusion_type == 'concat':
            # Увеличиваем количество каналов для последних трёх высокоуровневых карт признаков, если используется конкатенация
            decoder_channels[-3:] = [ch * 2 for ch in decoder_channels[-3:]]
        else:
            # При любом другом методе слияния признаков fusion_type размерность признаков остаётся неизменной
            pass
        return decoder_channels

    def get_decoder(self, decoder_name, decoder_channels, num_classes):
        if decoder_name == 'u_net':
            return UNetDecoder(in_channels=decoder_channels, out_channels=num_classes)
        else:
            raise ValueError(f"Неизвестное имя декодера: {decoder_name}")


class AlignFeatureModule(nn.Module):
    def __init__(self,
                 align_type,
                 fusion_type,
                 encoder_channels,
                 num_heads):
        super(AlignFeatureModule, self).__init__()
        self.align_type = align_type
        self.fusion_type = fusion_type
        self.encoder_channels = encoder_channels
        self.num_heads = num_heads
        # Определяем, какие уровни признаков не буду обрабатываться и сохраняться неизменно (например, низкоуровневые признаки)
        self.keep_stage_features = set(range(len(encoder_channels) - 3))

        # Создаём список модулей MultiheadAttention для каждого уровня признаков. Будем использовать их как CrossAttention.
        self.cross_attentions = nn.ModuleList([MultiheadAttention(embed_dim=channels,
                                                                  num_heads=self.num_heads[i],
                                                                  attn_dropout=0,
                                                                  proj_dropout=0,
                                                                  bias=True,
                                                                  qk_reduction_ratio=0.5
                                                                  # Снижение размерности query и key в 2 раза для уменьшения вычислительных затрат
                                                                  )
                                               for i, channels in enumerate(encoder_channels)])
        if align_type == 'soft' or align_type == 'hard':
            self.soft_alignments = nn.ModuleList([SoftAlignmentModule(in_channels=channels)
                                                  for i, channels in enumerate(encoder_channels)])

            self.hard_alignments = nn.ModuleList([HardAlignmentModule(in_channels=channels)
                                                  for i, channels in enumerate(encoder_channels)])

    def forward(self, features_n, features_a):
        # Cross-attention: использование аномальных картинок как query и нормальных как key/value
        final_features = []
        if self.align_type == 'multihead':
            for i, cross_attention in enumerate(self.cross_attentions):
                if i in self.keep_stage_features:
                    # Оставляем низкоуровневые признаки аномального изображения без изменений
                    final_features.append(features_a[i])
                else:
                    # Для признаков среднего и высокого уровня осуществляем обработку и выравнивание признаков, используя cross_attention.
                    _, _, h, w = features_a[i].shape

                    # Перестраиваем тензоры в формат (B, H*W, C) для работы с механизмом внимания
                    feature_a = rearrange(features_a[i], "b c h w -> b (h w) c")
                    feature_n = rearrange(features_n[i], "b c h w -> b (h w) c")

                    # Применяем cross-attention для улучшения нормальных признаков feature_n (аномальные признаки как query, нормальные как key/value)
                    feature_n = cross_attention(query=feature_a, key=feature_n, value=feature_n)
                    # Применяем cross-attention для улучшения аномальных признаков feature_a (новые нормальные признаки как query, аномальные как key/value)
                    feature_a = cross_attention(query=feature_n, key=feature_a, value=feature_a)

                    # Обратно перестраиваем тензоры в формат (B, C, H, W)
                    feature_n = rearrange(feature_n, "b (h w) c -> b c h w", h=h, w=w)
                    feature_a = rearrange(feature_a, "b (h w) c -> b c h w", h=h, w=w)

                    # Объединяем обработанные признаки согласно выбранному методу слияния признаков fusion_type
                    final_features.append(self.feature_fusion(features_a=feature_a, features_n=feature_n))
            return final_features
        if self.align_type == 'soft':
            for i, soft_alignment in enumerate(self.soft_alignments):
                if i in self.keep_stage_features:
                    # Оставляем низкоуровневые признаки аномального изображения без изменений
                    final_features.append(features_a[i])
                else:
                    # Применяем soft_alignment для улучшения нормальных признаков feature_n (аномальные признаки как query, нормальные как key/value)
                    feature_n = soft_alignment(features_a[i], features_n[i])
                    # Применяем soft_alignment для улучшения аномальных признаков feature_a (новые нормальные признаки как query, аномальные как key/value)
                    feature_a = soft_alignment(features_n[i], features_a[i])

                    # Объединяем обработанные признаки согласно выбранному методу слияния признаков fusion_type
                    final_features.append(self.feature_fusion(features_a=feature_a, features_n=feature_n))
            return final_features
        elif self.align_type == 'hard':
            for i, hard_alignment in enumerate(self.hard_alignments):
                if i in self.keep_stage_features:
                    # Оставляем низкоуровневые признаки аномального изображения без изменений
                    final_features.append(features_a[i])
                else:
                    # Применяем hard_alignment для улучшения нормальных признаков feature_n (аномальные признаки как query, нормальные как key/value)
                    feature_n = hard_alignment(features_a[i], features_n[i])
                    # Применяем hard_alignment для улучшения аномальных признаков feature_a (новые нормальные признаки как query, аномальные как key/value)
                    feature_a = hard_alignment(features_n[i], features_a[i])

                    # Объединяем обработанные признаки согласно выбранному методу слияния признаков fusion_type
                    final_features.append(self.feature_fusion(features_a=feature_a, features_n=feature_n))
            return final_features
        else:
            raise ValueError(f"Неизвестный тип выравнивания признаков: {self.align_type}")

    def feature_fusion(self, features_a, features_n):
        if (self.fusion_type == 'concat'):
            return torch.cat([features_a, features_n], dim=1)  # Конкатенация признаков по канальной размерности
        elif (self.fusion_type == 'add'):
            return features_a + features_n  # Сложение признаков
        elif (self.fusion_type == 'abs_diff'):
            return torch.abs(features_a - features_n)  # Абсолютная разница признаков
        else:
            raise ValueError(f"Неизвестный тип объединения признаков: {self.fusion_type}")


class AlignFeatureModule_SoftHard(nn.Module):
    def __init__(self,
                 align_type,
                 fusion_type,
                 encoder_channels,
                 num_heads):
        super(AlignFeatureModule_SoftHard, self).__init__()
        self.align_type = align_type
        self.fusion_type = fusion_type
        self.encoder_channels = encoder_channels
        self.num_heads = num_heads
        # Определяем, какие уровни признаков не буду обрабатываться и сохраняться неизменно (например, низкоуровневые признаки)
        self.keep_stage_features = set(range(len(encoder_channels) - 3))

        self.soft_alignments = nn.ModuleList([SoftAlignmentModule(in_channels=channels)
                                              for i, channels in enumerate(encoder_channels)])

        self.hard_alignments = nn.ModuleList([HardAlignmentModule(in_channels=channels)
                                              for i, channels in enumerate(encoder_channels)])

    def forward(self, features_n, features_a):
        final_features = []
        if self.align_type == 'soft':
            for i, soft_alignment in enumerate(self.soft_alignments):
                if i in self.keep_stage_features:
                    # Оставляем низкоуровневые признаки аномального изображения без изменений
                    final_features.append(features_a[i])
                else:
                    # Применяем soft_alignment для улучшения нормальных признаков feature_n (аномальные признаки как query, нормальные как key/value)
                    feature_n = soft_alignment(features_a[i], features_n[i])
                    # Применяем soft_alignment для улучшения аномальных признаков feature_a (новые нормальные признаки как query, аномальные как key/value)
                    feature_a = soft_alignment(features_n[i], features_a[i])

                    # Объединяем обработанные признаки согласно выбранному методу слияния признаков fusion_type
                    final_features.append(self.feature_fusion(features_a=feature_a, features_n=feature_n))
            return final_features
        elif self.align_type == 'hard':
            for i, hard_alignment in enumerate(self.hard_alignments):
                if i in self.keep_stage_features:
                    # Оставляем низкоуровневые признаки аномального изображения без изменений
                    final_features.append(features_a[i])
                else:
                    # Применяем hard_alignment для улучшения нормальных признаков feature_n (аномальные признаки как query, нормальные как key/value)
                    feature_n = hard_alignment(features_a[i], features_n[i])
                    # Применяем hard_alignment для улучшения аномальных признаков feature_a (новые нормальные признаки как query, аномальные как key/value)
                    feature_a = hard_alignment(features_n[i], features_a[i])

                    # Объединяем обработанные признаки согласно выбранному методу слияния признаков fusion_type
                    final_features.append(self.feature_fusion(features_a=feature_a, features_n=feature_n))
            return final_features
        else:
            raise ValueError(f"Неизвестный тип выравнивания признаков: {self.align_type}")

    def feature_fusion(self, features_a, features_n):
        if (self.fusion_type == 'concat'):
            return torch.cat([features_a, features_n], dim=1)  # Конкатенация признаков по канальной размерности
        elif (self.fusion_type == 'add'):
            return features_a + features_n  # Сложение признаков
        elif (self.fusion_type == 'abs_diff'):
            return torch.abs(features_a - features_n)  # Абсолютная разница признаков
        else:
            raise ValueError(f"Неизвестный тип объединения признаков: {self.fusion_type}")


class UNetDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(UNetDecoder, self).__init__()
        # Переворачиваем список входных каналов для декодера, так как входные данные для декодера идут в обратном порядке
        in_channels = in_channels[::-1]

        # Создаем список Upsample слоев для увеличения размера изображения в два раза
        # Каждый элемент списка - это слой, увеличивающий размер входа с использованием билинейной интерполяции
        self.ups = nn.ModuleList(
            [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) for i in range(len(in_channels) - 1)])

        # Создаем список слоев свертки, каждый из которых содержит:
        # 1. Свертку с фильтрами 3x3 и нормализацией
        # 2. Блоки с ReLU активацией
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=in_channels[i + 1] + in_channels[i], out_channels=in_channels[i + 1], kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[i + 1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels[i + 1], out_channels=in_channels[i + 1], kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(in_channels[i + 1]),
            nn.ReLU()
        ) for i in range(len(in_channels) - 1)])

        # Последний слой для классификации пикселей (количество выходных каналов равно числу классов)
        self.pixel_classifier = nn.Conv2d(in_channels[-1], out_channels, kernel_size=1)

    def forward(self, features_of_each_stage):
        # Переворачиваем список входных признаков для обработки в обратном порядке
        features_of_each_stage = features_of_each_stage[::-1]

        # Начинаем с первого элемента
        x = features_of_each_stage[0]

        # Применяем Upsample и свертки к каждому элементу списка признаков, начиная с второго
        for i in range(len(features_of_each_stage) - 1):
            x = self.ups[i](x)  # Увеличиваем размерность
            x = torch.cat([x, features_of_each_stage[i + 1]],
                          dim=1)  # Конкатенируем с соответствующей фичей из энкодера (skip-connection)
            x = self.convs[i](x)  # Применяем сверточный блок

        # В конце, применяем пиксельный классификатор для получения финальной выходной маски
        return self.pixel_classifier(x)


class MultiheadAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_dropout=0.,
                 proj_dropout=0.,
                 bias=False,
                 kdim=None,
                 vdim=None,
                 qk_reduction_ratio=0.5):
        super(MultiheadAttention, self).__init__()

        # Сохраняем количество голов и вычисляем размерность каждой головы
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads

        # Масштабирующий коэффициент, используется для нормализации скалярного произведения в внимании
        self.scale = float(head_dim) ** -0.5  # масштабирующий (нормализующий) коэффициент 1 / sqrt(head_dim)

        # Устанавливаем размерности для ключей и значений, если они не заданы
        kdim = kdim or embed_dim
        vdim = vdim or embed_dim

        # Линейные слои для query (Q), key (K) и value (V)
        self.q = nn.Linear(embed_dim, int(embed_dim * qk_reduction_ratio), bias=bias)
        self.k = nn.Linear(embed_dim, int(kdim * qk_reduction_ratio), bias=bias)
        self.v = nn.Linear(embed_dim, vdim, bias=bias)

        # Dropout для внимания (если задан)
        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout > 0. else nn.Identity()

        # Финальный линейный слой и Dropout для этого слоя (если задан)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout) if proj_dropout > 0. else nn.Identity()

        # Инициализация весов
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, query, key, value):
        # Применение линейных слоёв для получения query (Q), key (K) и value (V)
        q, k, v = self.q(query), self.k(key), self.v(value)

        # Разделение q, k, v на головы, изменение формы тензоров на (batch_size, num_heads, seq_len, head_dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        # Скалярное произведение query и key и получение карты внимания
        attn = einsum('b h i d, b h j d -> b h i j', q,
                      k) * self.scale  # Масштабированное скалярное произведение для внимания
        attn = F.softmax(attn, dim=-1)  # Применение softmax для нормализации карты внимания
        attn = self.attn_drop(attn)  # Применение дроп-аут к вниманию для регуляризации

        # Взвешиваем value, используя вычисленную карту внимания attn
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Применяем финальный линейный слой и дроп-аут
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
