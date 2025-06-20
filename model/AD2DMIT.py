import torch
import torch.nn as nn
from torchvision.models import resnet34
import numpy as np
import matplotlib.pyplot as plt
from .layers import Conv2D,CBAM,Attention
import torch.nn.functional as F

class AD2D_MIL(nn.Module):
    """
    Accurate Screening of COVID-19 Using Attention-Based Deep 3D Multiple Instance Learning
    """

    def __init__(self, in_channel, hidden, category, num_layer, image_size, patches=64, visualize=False):
        """Constructor for AD2D_MIL"""
        super(AD2D_MIL, self).__init__()
        feature_extractor = []
        temp = int(np.log2(patches)) // 2
        num_layer = 9 - temp
        factor = 2 ** (temp + 2)
        # print('temp{},patch{},layer{},factor{}'.format(temp, patches, num_layer, factor))
        for i in range(num_layer):
            feature_extractor.append(Conv2D(in_channel, factor * (2 ** i), ac=True))
            feature_extractor.append(nn.MaxPool2d(2, 2))
            in_channel = factor * (2 ** i)
            # print(in_channel)
        feature_extractor.append(Conv2D(in_channel, hidden, ac=True))
        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.attention = Attention(hidden, hidden * 2, (image_size // (2 ** num_layer)) ** 2)
        # print('patch', (image_size // (2 ** num_layer)) ** 2)

        self.visualize = visualize

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(hidden * 2, hidden * 2 * 2),
            nn.BatchNorm1d(hidden * 2 * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2 * 2, category),
            # nn.Sigmoid()
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        # # 可视化原始输入图像（第一个batch的第一张图）
        # if self.visualize:
        #     self._visualize_input(x[0].cpu().detach(), title="Input Image")

        # 特征提取
        features = self.feature_extractor(x)

        # 可视化特征图
        # if self.visualize :
        #     self._visualize_feature_maps(features[0].cpu().detach(), title="Feature Maps")

        # 注意力机制
        out = features.permute((0, 2, 3, 1))
        out = out.contiguous()
        out = out.view(-1, out.size(-1))
        weight = self.attention(out).unsqueeze(-1)

        # # Visualize features and attention weights
        # if self.visualize:
        #     self._visualize_features(features, weight.squeeze(-1),
        #                              "Feature Maps and Attention Weights")

        # 可视化注意力权重
        # if self.visualize :
        #     self._visualize_attention_weights(weight[0].cpu().detach(), title="Attention Weights")

        # 可视化注意力权重叠加图
        # if self.visualize:
        #     self._visualize_attention_overlay(weight[0], x[0], title="Attention Map Overlay")

        # 分类
        out = out.view(x.size(0), -1, out.size(-1))
        out = weight * out
        out = torch.sum(out, 1)
        out = self.classifier(out).squeeze(-1)
        return out

    def _visualize_input(self, img, title="Input Image"):
        """可视化输入图像"""
        img = img.numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    def _visualize_feature_maps(self, features, title="Feature Maps", n_maps=16):
        """可视化特征图"""
        plt.figure(figsize=(12, 12))
        for i in range(min(n_maps, features.size(0))):
            plt.subplot(4, 4, i + 1)
            feat_map = features[i].numpy()
            feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
            plt.imshow(feat_map, cmap='gray')
            plt.title(f"Channel {i}")
            plt.axis('off')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def _visualize_attention_weights(self, weights, title="Attention Weights"):
        """可视化注意力权重"""
        weights = weights.view(-1).numpy()
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(weights)), weights)
        plt.title(title)
        plt.xlabel("Patch Index")
        plt.ylabel("Attention Weight")
        plt.show()

    def _visualize_attention_overlay(self, attention_weights, input_img, title="Attention Overlay"):
        """
        将 attention 权重（[1, 1024, 1]）恢复为 32x32，再插值为 2048x2048，
        叠加到原图上可视化
        """
        import matplotlib.pyplot as plt
        import torch.nn.functional as F

        # 1. 注意力权重 reshape 成 32x32
        attention_map = attention_weights.view(1, 1, 32, 32)  # [1,1,32,32]

        # 2. 上采样到 2048x2048
        upsampled_attn = F.interpolate(attention_map, size=(2048, 2048), mode='bilinear', align_corners=False)
        upsampled_attn = upsampled_attn.squeeze().cpu().detach().numpy()  # [2048,2048]
        upsampled_attn = (upsampled_attn - upsampled_attn.min()) / (upsampled_attn.max() - upsampled_attn.min())

        # 3. 获取原始输入图像（反归一化后用于显示）
        img = input_img.squeeze().cpu().detach().numpy()
        img = (img - img.min()) / (img.max() - img.min())

        # 4. 显示叠加图
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray')
        plt.imshow(upsampled_attn, cmap='jet', alpha=0.5)  # 热图叠加
        plt.title(title)
        plt.axis('off')
        plt.colorbar(label='Attention')
        plt.tight_layout()
        plt.show()

    def _visualize_features(self, features, weights, title="Feature Maps"):
        """
        Visualize feature maps and attention weights
        Args:
            features: extracted features [B, C, H, W]
            weights: attention weights [B, N] where N=H*W
            title: figure title
        """
        # Get the first sample in batch
        features = features[0].detach().cpu()
        weights = weights[0].detach().cpu()

        # Reshape weights to match feature spatial dimensions
        h, w = features.shape[1], features.shape[2]
        weight_map = weights.view(h, w).numpy()

        # Get channel-averaged feature map
        avg_features = torch.mean(features, dim=0).numpy()

        plt.figure(figsize=(15, 5))

        # 1. Plot channel-averaged features
        plt.subplot(1, 3, 1)
        plt.imshow(avg_features, cmap='jet')
        plt.title('Averaged Feature Map')
        plt.colorbar()
        plt.axis('off')

        # 2. Plot attention weights
        plt.subplot(1, 3, 2)
        plt.imshow(weight_map, cmap='hot')
        plt.title('Attention Weights')
        plt.colorbar()
        plt.axis('off')

        # 3. Plot weighted features
        plt.subplot(1, 3, 3)
        weighted_features = avg_features * weight_map
        plt.imshow(weighted_features, cmap='jet')
        plt.title('Weighted Features')
        plt.colorbar()
        plt.axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def forward_feat(self, x):
        """
        在不丢弃任何中间特征的情况下做一次前向推理，
        返回 (logits, feats)，其中 feats 是用于可视化的 feature map，
        包括卷积特征和加权后的特征图。
        """
        # 特征提取
        features = self.feature_extractor(x)
        # 注意力机制
        out = features.permute((0, 2, 3, 1))
        out = out.contiguous()
        out = out.view(-1, out.size(-1))
        weight = self.attention(out).unsqueeze(-1)

        # 分类
        out = out.view(x.size(0), -1, out.size(-1))
        out = weight * out
        out = torch.sum(out, 1)
        out = self.classifier(out).squeeze(-1)

        # 7. 返回分类结果和特征图
        return out, features

class AD2D_MIL_Log_Continue(nn.Module):
    """
    Log10之后进行回归预测
    """

    def __init__(self, in_channel, hidden, image_size, patches=64, visualize=False):
        """Constructor for AD2D_MIL"""
        super(AD2D_MIL_Log_Continue, self).__init__()
        feature_extractor = []
        temp = int(np.log2(patches)) // 2
        num_layer = 9 - temp
        factor = 2 ** (temp + 2)
        # print('temp{},patch{},layer{},factor{}'.format(temp, patches, num_layer, factor))
        for i in range(num_layer):
            feature_extractor.append(Conv2D(in_channel, factor * (2 ** i), ac=True))
            feature_extractor.append(nn.MaxPool2d(2, 2))
            in_channel = factor * (2 ** i)
            # print(in_channel)
        feature_extractor.append(Conv2D(in_channel, hidden, ac=True))
        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.attention = Attention(hidden, hidden * 2, (image_size // (2 ** num_layer)) ** 2)
        # print('patch', (image_size // (2 ** num_layer)) ** 2)

        self.visualize = visualize

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(hidden * 2, hidden * 2 * 2),
            nn.BatchNorm1d(hidden * 2 * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2 * 2, 1),
            # nn.Sigmoid()
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        # # 可视化原始输入图像（第一个batch的第一张图）
        # if self.visualize:
        #     self._visualize_input(x[0].cpu().detach(), title="Input Image")

        # 特征提取
        features = self.feature_extractor(x)

        # 可视化特征图
        # if self.visualize :
        #     self._visualize_feature_maps(features[0].cpu().detach(), title="Feature Maps")

        # 注意力机制
        out = features.permute((0, 2, 3, 1))
        out = out.contiguous()
        out = out.view(-1, out.size(-1))
        weight = self.attention(out).unsqueeze(-1)

        # # Visualize features and attention weights
        # if self.visualize:
        #     self._visualize_features(features, weight.squeeze(-1),
        #                              "Feature Maps and Attention Weights")

        # 可视化注意力权重
        # if self.visualize :
        #     self._visualize_attention_weights(weight[0].cpu().detach(), title="Attention Weights")

        # 可视化注意力权重叠加图
        # if self.visualize:
        #     self._visualize_attention_overlay(weight[0], x[0], title="Attention Map Overlay")

        # 分类
        out = out.view(x.size(0), -1, out.size(-1))
        out = weight * out
        out = torch.sum(out, 1)
        out = self.classifier(out).squeeze(-1)
        return out

    def _visualize_input(self, img, title="Input Image"):
        """可视化输入图像"""
        img = img.numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    def _visualize_feature_maps(self, features, title="Feature Maps", n_maps=16):
        """可视化特征图"""
        plt.figure(figsize=(12, 12))
        for i in range(min(n_maps, features.size(0))):
            plt.subplot(4, 4, i + 1)
            feat_map = features[i].numpy()
            feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
            plt.imshow(feat_map, cmap='gray')
            plt.title(f"Channel {i}")
            plt.axis('off')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def _visualize_attention_weights(self, weights, title="Attention Weights"):
        """可视化注意力权重"""
        weights = weights.view(-1).numpy()
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(weights)), weights)
        plt.title(title)
        plt.xlabel("Patch Index")
        plt.ylabel("Attention Weight")
        plt.show()

    def _visualize_attention_overlay(self, attention_weights, input_img, title="Attention Overlay"):
        """
        将 attention 权重（[1, 1024, 1]）恢复为 32x32，再插值为 2048x2048，
        叠加到原图上可视化
        """
        import matplotlib.pyplot as plt
        import torch.nn.functional as F

        # 1. 注意力权重 reshape 成 32x32
        attention_map = attention_weights.view(1, 1, 32, 32)  # [1,1,32,32]

        # 2. 上采样到 2048x2048
        upsampled_attn = F.interpolate(attention_map, size=(2048, 2048), mode='bilinear', align_corners=False)
        upsampled_attn = upsampled_attn.squeeze().cpu().detach().numpy()  # [2048,2048]
        upsampled_attn = (upsampled_attn - upsampled_attn.min()) / (upsampled_attn.max() - upsampled_attn.min())

        # 3. 获取原始输入图像（反归一化后用于显示）
        img = input_img.squeeze().cpu().detach().numpy()
        img = (img - img.min()) / (img.max() - img.min())

        # 4. 显示叠加图
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray')
        plt.imshow(upsampled_attn, cmap='jet', alpha=0.5)  # 热图叠加
        plt.title(title)
        plt.axis('off')
        plt.colorbar(label='Attention')
        plt.tight_layout()
        plt.show()

    def _visualize_features(self, features, weights, title="Feature Maps"):
        """
        Visualize feature maps and attention weights
        Args:
            features: extracted features [B, C, H, W]
            weights: attention weights [B, N] where N=H*W
            title: figure title
        """
        # Get the first sample in batch
        features = features[0].detach().cpu()
        weights = weights[0].detach().cpu()

        # Reshape weights to match feature spatial dimensions
        h, w = features.shape[1], features.shape[2]
        weight_map = weights.view(h, w).numpy()

        # Get channel-averaged feature map
        avg_features = torch.mean(features, dim=0).numpy()

        plt.figure(figsize=(15, 5))

        # 1. Plot channel-averaged features
        plt.subplot(1, 3, 1)
        plt.imshow(avg_features, cmap='jet')
        plt.title('Averaged Feature Map')
        plt.colorbar()
        plt.axis('off')

        # 2. Plot attention weights
        plt.subplot(1, 3, 2)
        plt.imshow(weight_map, cmap='hot')
        plt.title('Attention Weights')
        plt.colorbar()
        plt.axis('off')

        # 3. Plot weighted features
        plt.subplot(1, 3, 3)
        weighted_features = avg_features * weight_map
        plt.imshow(weighted_features, cmap='jet')
        plt.title('Weighted Features')
        plt.colorbar()
        plt.axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def forward_feat(self, x):
        """
        在不丢弃任何中间特征的情况下做一次前向推理，
        返回 (logits, feats)，其中 feats 是用于可视化的 feature map，
        包括卷积特征和加权后的特征图。
        """
        # 特征提取
        features = self.feature_extractor(x)
        # 注意力机制
        out = features.permute((0, 2, 3, 1))
        out = out.contiguous()
        out = out.view(-1, out.size(-1))
        weight = self.attention(out).unsqueeze(-1)

        # 分类
        out = out.view(x.size(0), -1, out.size(-1))
        out = weight * out
        out = torch.sum(out, 1)
        out = self.classifier(out).squeeze(-1)

        # 7. 返回分类结果和特征图
        return out, features

class AD2D_MIL_bin_Q(nn.Module):
    """
    分箱，最后一位为分位数
    """

    def __init__(self, in_channel, hidden, category, image_size, patches=64, visualize=False):
        """Constructor for AD2D_MIL"""
        super(AD2D_MIL_bin_Q, self).__init__()
        feature_extractor = []
        temp = int(np.log2(patches)) // 2
        num_layer = 9 - temp
        factor = 2 ** (temp + 2)
        # print('temp{},patch{},layer{},factor{}'.format(temp, patches, num_layer, factor))
        for i in range(num_layer):
            feature_extractor.append(Conv2D(in_channel, factor * (2 ** i), ac=True))
            feature_extractor.append(nn.MaxPool2d(2, 2))
            in_channel = factor * (2 ** i)
            # print(in_channel)
        feature_extractor.append(Conv2D(in_channel, hidden, ac=True))
        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.attention = Attention(hidden, hidden * 2, (image_size // (2 ** num_layer)) ** 2)
        # print('patch', (image_size // (2 ** num_layer)) ** 2)

        self.visualize = visualize

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(hidden * 2, hidden * 2 * 2),
            nn.BatchNorm1d(hidden * 2 * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2 * 2, category),
            # nn.Sigmoid()
            # nn.Softmax(dim=1)
        )
        # 分位数回归头：输出一个 [0,1] 之间的值
        self.quantile_head = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden * 2 * 2),
            nn.BatchNorm1d(hidden * 2 * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # # 可视化原始输入图像（第一个batch的第一张图）
        # if self.visualize:
        #     self._visualize_input(x[0].cpu().detach(), title="Input Image")

        # 特征提取
        features = self.feature_extractor(x)

        # 可视化特征图
        # if self.visualize :
        #     self._visualize_feature_maps(features[0].cpu().detach(), title="Feature Maps")

        # 注意力机制
        out = features.permute((0, 2, 3, 1))
        out = out.contiguous()
        out = out.view(-1, out.size(-1))
        weight = self.attention(out).unsqueeze(-1)

        # # Visualize features and attention weights
        # if self.visualize:
        #     self._visualize_features(features, weight.squeeze(-1),
        #                              "Feature Maps and Attention Weights")

        # 可视化注意力权重
        # if self.visualize :
        #     self._visualize_attention_weights(weight[0].cpu().detach(), title="Attention Weights")

        # 可视化注意力权重叠加图
        # if self.visualize:
        #     self._visualize_attention_overlay(weight[0], x[0], title="Attention Map Overlay")

        # 分类
        out = out.view(x.size(0), -1, out.size(-1))
        out = weight * out
        out = torch.sum(out, 1)
        logits  = self.classifier(out).squeeze(-1)
        pct = self.quantile_head(out).view(-1)  # B
        return logits,pct

    def _visualize_input(self, img, title="Input Image"):
        """可视化输入图像"""
        img = img.numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    def _visualize_feature_maps(self, features, title="Feature Maps", n_maps=16):
        """可视化特征图"""
        plt.figure(figsize=(12, 12))
        for i in range(min(n_maps, features.size(0))):
            plt.subplot(4, 4, i + 1)
            feat_map = features[i].numpy()
            feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
            plt.imshow(feat_map, cmap='gray')
            plt.title(f"Channel {i}")
            plt.axis('off')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def _visualize_attention_weights(self, weights, title="Attention Weights"):
        """可视化注意力权重"""
        weights = weights.view(-1).numpy()
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(weights)), weights)
        plt.title(title)
        plt.xlabel("Patch Index")
        plt.ylabel("Attention Weight")
        plt.show()

    def _visualize_attention_overlay(self, attention_weights, input_img, title="Attention Overlay"):
        """
        将 attention 权重（[1, 1024, 1]）恢复为 32x32，再插值为 2048x2048，
        叠加到原图上可视化
        """
        import matplotlib.pyplot as plt
        import torch.nn.functional as F

        # 1. 注意力权重 reshape 成 32x32
        attention_map = attention_weights.view(1, 1, 32, 32)  # [1,1,32,32]

        # 2. 上采样到 2048x2048
        upsampled_attn = F.interpolate(attention_map, size=(2048, 2048), mode='bilinear', align_corners=False)
        upsampled_attn = upsampled_attn.squeeze().cpu().detach().numpy()  # [2048,2048]
        upsampled_attn = (upsampled_attn - upsampled_attn.min()) / (upsampled_attn.max() - upsampled_attn.min())

        # 3. 获取原始输入图像（反归一化后用于显示）
        img = input_img.squeeze().cpu().detach().numpy()
        img = (img - img.min()) / (img.max() - img.min())

        # 4. 显示叠加图
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray')
        plt.imshow(upsampled_attn, cmap='jet', alpha=0.5)  # 热图叠加
        plt.title(title)
        plt.axis('off')
        plt.colorbar(label='Attention')
        plt.tight_layout()
        plt.show()

    def _visualize_features(self, features, weights, title="Feature Maps"):
        """
        Visualize feature maps and attention weights
        Args:
            features: extracted features [B, C, H, W]
            weights: attention weights [B, N] where N=H*W
            title: figure title
        """
        # Get the first sample in batch
        features = features[0].detach().cpu()
        weights = weights[0].detach().cpu()

        # Reshape weights to match feature spatial dimensions
        h, w = features.shape[1], features.shape[2]
        weight_map = weights.view(h, w).numpy()

        # Get channel-averaged feature map
        avg_features = torch.mean(features, dim=0).numpy()

        plt.figure(figsize=(15, 5))

        # 1. Plot channel-averaged features
        plt.subplot(1, 3, 1)
        plt.imshow(avg_features, cmap='jet')
        plt.title('Averaged Feature Map')
        plt.colorbar()
        plt.axis('off')

        # 2. Plot attention weights
        plt.subplot(1, 3, 2)
        plt.imshow(weight_map, cmap='hot')
        plt.title('Attention Weights')
        plt.colorbar()
        plt.axis('off')

        # 3. Plot weighted features
        plt.subplot(1, 3, 3)
        weighted_features = avg_features * weight_map
        plt.imshow(weighted_features, cmap='jet')
        plt.title('Weighted Features')
        plt.colorbar()
        plt.axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def forward_feat(self, x):
        """
        在不丢弃任何中间特征的情况下做一次前向推理，
        返回 (logits, feats)，其中 feats 是用于可视化的 feature map，
        包括卷积特征和加权后的特征图。
        """
        # 特征提取
        features = self.feature_extractor(x)
        # 注意力机制
        out = features.permute((0, 2, 3, 1))
        out = out.contiguous()
        out = out.view(-1, out.size(-1))
        weight = self.attention(out).unsqueeze(-1)

        # 分类
        out = out.view(x.size(0), -1, out.size(-1))
        out = weight * out
        out = torch.sum(out, 1)
        out = self.classifier(out).squeeze(-1)

        # 7. 返回分类结果和特征图
        return out, features


class AD2D_MIL_bin_MutiQ(nn.Module):
    """
    分箱，最后一位为分位数,多个回归头
    """

    def __init__(self, in_channel, hidden, category, image_size,label_ranges, patches=64, visualize=False, prob_threshold=0.25):
        """Constructor for AD2D_MIL"""
        super(AD2D_MIL_bin_MutiQ, self).__init__()
        feature_extractor = []
        temp = int(np.log2(patches)) // 2
        num_layer = 9 - temp
        factor = 2 ** (temp + 2)
        for i in range(num_layer):
            feature_extractor.append(Conv2D(in_channel, factor * (2 ** i), ac=True))
            feature_extractor.append(nn.MaxPool2d(2, 2))
            in_channel = factor * (2 ** i)
            # print(in_channel)
        feature_extractor.append(Conv2D(in_channel, hidden, ac=True))
        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.attention = Attention(hidden, hidden * 2, (image_size // (2 ** num_layer)) ** 2)
        self.label_ranges = label_ranges
        self.prob_threshold = prob_threshold
        self.category = category

        self.visualize = visualize

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden * 2 * 2),
            nn.BatchNorm1d(hidden * 2 * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2 * 2, category),
        )
        # 分位数回归头：输出一个 [0,1] 之间的值
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                nn.BatchNorm1d(hidden * 2),
                nn.ReLU(),
                nn.Linear(hidden * 2, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid()
            ) for _ in range(category)
        ])

    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)  # (B, C, H, W)

        # 注意力机制
        out = features.permute((0, 2, 3, 1))  # (B, H, W, C)
        out = out.contiguous().view(-1, out.size(-1))  # (B*H*W, C)
        weight = self.attention(out).unsqueeze(-1)  # (B*H*W, 1)

        # 分类特征聚合
        out = out.view(x.size(0), -1, out.size(-1))  # (B, N, C)
        weight = weight.view(x.size(0), -1, 1)  # (B, N, 1)
        out = weight * out  # (B, N, C)
        aggregated = torch.sum(out, dim=1)  # (B, C)

        # 分类概率
        logits = self.classifier(aggregated)  # (B, category)
        bin_probs = F.softmax(logits, dim=1)  # (B, category)
        valid_mask = bin_probs > self.prob_threshold  # (B, category)

        B, C = aggregated.shape
        K = self.category

        # 批量计算每个类别的分位数回归预测 (B, K)
        pcts = torch.cat([head(aggregated) for head in self.reg_heads], dim=1)  # (B, K)

        # log10(ATP) = lower + (upper - lower) * pct
        device = x.device
        lower_bounds = torch.tensor([self.label_ranges[i][0] for i in range(K)], device=device)  # (K,)
        upper_bounds = torch.tensor([self.label_ranges[i][1] for i in range(K)], device=device)  # (K,)
        range_widths = upper_bounds - lower_bounds  # (K,)
        log10_atp_preds = lower_bounds + pcts * range_widths  # (B, K)
        atp_preds = torch.pow(10.0, log10_atp_preds)  # (B, K)

        # 按概率加权生成最终ATP预测
        final_preds = torch.zeros(B, device=device)
        for i in range(B):
            probs = bin_probs[i]  # (K,)
            valid_bins = torch.where(valid_mask[i])[0]
            if len(valid_bins) == 0:
                valid_bins = torch.topk(probs, 3).indices
            weights = probs[valid_bins]
            weights = weights / weights.sum()
            pred_vals = atp_preds[i][valid_bins]
            final_preds[i] = torch.sum(pred_vals * weights)

        # 得到最大概率的类的预测atp
        max_prob_bins = bin_probs.argmax(dim=1)  # (B,)
        pcts_max_bin = pcts[torch.arange(B), max_prob_bins]  # (B,)
        log10_atp_max = lower_bounds[max_prob_bins] + range_widths[max_prob_bins] * pcts_max_bin  # (B,)
        atp_max_prob_pred = torch.pow(10.0, log10_atp_max).unsqueeze(1)  # (B, 1)

        return final_preds.unsqueeze(1), logits, pcts,atp_max_prob_pred  # (B, 1), (B, K), (B, K)


