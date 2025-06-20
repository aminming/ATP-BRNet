import cv2
import torch
import torch.nn as nn
from layers import Conv2D, ResBlock, Attention,CBAM
from torchvision.models import resnet34
import numpy as np
import matplotlib.pyplot as plt
from hrnet import HighResolutionNet

class AD2D_MIL(nn.Module):
    """
    Accurate Screening of COVID-19 Using Attention-Based Deep 3D Multiple Instance Learning
    """

    def __init__(self, in_channel, category, num_layer, image_size, hidden= 32, patches=64, visualize=False):
        """Constructor for AD2D_MIL"""
        super(AD2D_MIL, self).__init__()

        self.feature_extractor = HighResolutionNet()
        self.attention = Attention(hidden, hidden * 2, (image_size // 4) ** 2)
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
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 可视化原始输入图像（第一个batch的第一张图）
        if self.visualize:
            self._visualize_input(x[0].cpu().detach(), title="Input Image")

        # 特征提取
        features = self.feature_extractor(x)

        # 可视化特征图
        if self.visualize :
            self._visualize_feature_maps(features[0].cpu().detach(), title="Feature Maps")

        # 注意力机制
        out = features.permute((0, 2, 3, 1))
        out = out.contiguous()
        out = out.view(-1, out.size(-1))
        weight = self.attention(out).unsqueeze(-1)

        # 可视化注意力权重
        # if self.visualize :
        #     self._visualize_attention_weights(weight[0].cpu().detach(), title="Attention Weights")

        # 可视化注意力权重叠加图
        if self.visualize:
            self._visualize_attention_overlay(weight[0], x[0], title="Attention Map Overlay")

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
        attention_map = attention_weights.view(1, 1, 128, 128)  # [1,1,32,32]

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

class AD2D_MIL_HRNet_CBAM(nn.Module):
    """
    多实例学习 + HRNet 特征提取 + CBAM 注意力，用于类器官 ATP 6 类分类
    输入: 单通道明场图像 [B,1,H,W]
    输出: 6 分类 logits [B,6]
    """
    def __init__(self, in_channel=1, hidden=32, num_classes=6,image_size=512, visualize=False):
        super(AD2D_MIL_HRNet_CBAM, self).__init__()
        # 1. HRNet 骨干，返回各分支特征列表
        #    修改 HighResolutionNet 支持 input_channels 参数
        self.backbone = HighResolutionNet()
        # 2. CBAM 注意力模块，作用于高分辨率分支
        self.cbam = CBAM(hidden)
        # 可视化开关
        self.visualize = visualize
        # 3. 分类头：全局池化 + MLP
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),                        # [B, hidden]
        #     nn.Linear((image_size // 4) ** 2, hidden * hidden),       # [B, hidden*2]
        #     nn.BatchNorm1d(hidden * hidden),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden * hidden, hidden * 4),
        #     nn.BatchNorm1d(hidden * 4),
        #     nn.ReLU(),
        #     nn.Linear(hidden * 4, num_classes)
        # )
        # URNET3_improve_logbin_6class_16_0.8810.pth
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(hidden * 2, hidden * 2 * 2),
            nn.BatchNorm1d(hidden * 2 * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2 * 2, num_classes),
            # nn.Sigmoid()
            # nn.Softmax(dim=1)
        )
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),  # [B, hidden]
        #     nn.Linear((image_size // 4) ** 2, hidden * hidden),  # [B, hidden*2]
        #     nn.LayerNorm(hidden * hidden),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden * hidden, hidden * 4),
        #     nn.LayerNorm(hidden * 4),
        #     nn.ReLU(),
        #     nn.Linear(hidden * 4, num_classes)
        # )

    def forward(self, x):
        # 可视化原始输入图像（第一个batch的第一张图）
        # if self.visualize:
        #     self._visualize_input(x[0].cpu().detach(), title="Input Image")
        # x: [B, 1, H, W]
        # 1. 骨干特征提取高分辨率分支特征
        features = self.backbone(x)

        # 2. CBAM 通道 + 空间注意力
        x_att = self.cbam(features)  # [B, hidden, H', W']

        # Visualization
        if self.visualize:
            self.visualize_features2(x, features, x_att, "Feature Maps with CBAM Attention")

        # 3. 全局池化 + 分类
        pooled = self.global_pool(x_att)  # [B, hidden, 1, 1]
        x_att = pooled.view(pooled.size(0), -1)  # [B, hidden * H' * W'] [B, 128*128]
        logits = self.classifier(x_att)  # [B, num_classes]
        return logits

    def forward_feat(self, x):
        # x: [B, 1, H, W]
        # 1. 骨干特征提取高分辨率分支特征
        features = self.backbone(x)  # 提取的特征图列表

        # 2. CBAM 通道 + 空间注意力
        x_att = self.cbam(features)  # [B, hidden, H', W']

        # Visualization
        if self.visualize:
            self.visualize_features(x, features, x_att, "Feature Maps with CBAM Attention")

        # 3. 全局池化 + 分类
        pooled = self.global_pool(x_att)  # [B, hidden, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [B, hidden * H' * W']
        logits = self.classifier(pooled)  # [B, num_classes]

        # 返回分类 logits 和特征图 (features, x_att)
        return logits, features  # 这里返回特征图 x_att (CBAM后的特征图)

    def visualize_features(self, x, features, x_att, title="Feature Maps"):
        """
        Visualize input, intermediate features, and attention-weighted features
        Args:
            x: input tensor [B,1,H,W]
            features: backbone features [B,C,H,W]
            x_att: attention-weighted features [B,C,H,W]
            title: figure title
        """
        # Convert tensors to numpy arrays
        x_np = x.detach().cpu().numpy()
        features_np = features.detach().cpu().numpy()
        x_att_np = x_att.detach().cpu().numpy()

        # Get batch size and select first sample in batch
        b, c, h, w = features_np.shape
        idx = 0  # visualize first sample in batch

        plt.figure(figsize=(18, 6))

        # 1. Plot input image
        plt.subplot(1, 3, 1)
        plt.imshow(x_np[idx, 0], cmap='gray')
        plt.title('Input Image')
        plt.axis('off')

        # 2. Plot channel-averaged backbone features
        plt.subplot(1, 3, 2)
        avg_features = np.mean(features_np[idx], axis=0)
        plt.imshow(avg_features, cmap='jet')
        plt.title('Backbone Features')
        plt.colorbar()
        plt.axis('off')

        # 3. Plot channel-averaged attention-weighted features
        plt.subplot(1, 3, 3)
        avg_att = np.mean(x_att_np[idx], axis=0)
        plt.imshow(avg_att, cmap='jet')
        plt.title('Attention-weighted Features')
        plt.colorbar()
        plt.axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def visualize_features2(self, x, features, x_att, title="Feature Maps"):
        """
        Visualize input, intermediate features, and attention-weighted features
        Args:
            x: input tensor [B,1,H,W] (e.g., 512×512)
            features: backbone features [B,C,H,W] (e.g., 128×128)
            x_att: attention-weighted features [B,C,H,W] (e.g., 128×128)
            title: figure title
        """
        # Convert tensors to numpy arrays
        x_np = x.detach().cpu().numpy()
        features_np = features.detach().cpu().numpy()
        x_att_np = x_att.detach().cpu().numpy()

        # Get batch size and select first sample in batch
        b, c, h, w = features_np.shape
        idx = 0  # visualize first sample in batch

        plt.figure(figsize=(18, 6))

        # Prepare the input image (grayscale)
        input_img = x_np[idx, 0]

        # Prepare feature maps (normalized to [0,1] and upsampled to match input size)
        avg_features = np.mean(features_np[idx], axis=0)
        avg_features = (avg_features - avg_features.min()) / (avg_features.max() - avg_features.min())
        avg_features = cv2.resize(avg_features, (input_img.shape[1], input_img.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)

        avg_att = np.mean(x_att_np[idx], axis=0)
        avg_att = (avg_att - avg_att.min()) / (avg_att.max() - avg_att.min())
        avg_att = cv2.resize(avg_att, (input_img.shape[1], input_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        # 1. Plot input image
        plt.subplot(1, 3, 1)
        plt.imshow(input_img, cmap='gray')
        plt.title('Input Image')
        plt.axis('off')

        # 2. Plot backbone features as overlay
        plt.subplot(1, 3, 2)
        plt.imshow(input_img, cmap='gray')
        plt.imshow(avg_features, cmap='jet', alpha=0.5)  # alpha controls transparency
        plt.title('Backbone Features Overlay')
        plt.axis('off')

        # 3. Plot attention-weighted features as overlay
        plt.subplot(1, 3, 3)
        plt.imshow(input_img, cmap='gray')
        plt.imshow(avg_att, cmap='jet', alpha=0.5)
        plt.title('Attention-weighted Features Overlay')
        plt.axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


class AD2D_MIL_LogATP(nn.Module):
    """
    基于注意力的深度 2D 多实例学习模型，用于连续ATP值的预测，
    最终输出为预测 ATP 值取对数后的结果。
    """

    def __init__(self, in_channel, hidden, image_size, patches=64, visualize=False):
        """
        参数:
          in_channel: 输入图像通道数
          hidden: 特征提取最后一层输出维度
          image_size: 原始图像尺寸（宽或高，假设正方形）
          patches: 分块数目（用于动态计算层数）
          visualize: 是否启用中间特征和权重可视化
        """
        super(AD2D_MIL_LogATP, self).__init__()
        # 本任务为回归预测，目标是一个连续值：log(ATP)
        category = 1
        feature_extractor = []
        temp = int(np.log2(patches)) // 2
        num_layer = 9 - temp
        factor = 2 ** (temp + 2)
        # 构造特征提取器
        for i in range(num_layer):
            feature_extractor.append(Conv2D(in_channel, factor * (2 ** i), ac=True))
            feature_extractor.append(nn.MaxPool2d(2, 2))
            in_channel = factor * (2 ** i)
        # 最后一个卷积层，将通道数转换为 hidden
        feature_extractor.append(Conv2D(in_channel, hidden, ac=True))
        self.feature_extractor = nn.Sequential(*feature_extractor)

        # 注意力层：将特征图重排后注意力机制计算权重
        self.attention = Attention(hidden, hidden * 2, (image_size // (2 ** num_layer)) ** 2)
        self.visualize = visualize

        # 分类器（回归头），输出单个 ATP 预测值
        # 为确保 ATP 为正，后续采用 Softplus 激活
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden * 2 * 2),
            nn.BatchNorm1d(hidden * 2 * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2 * 2, category)
        )
        # 强制输出为正
        self.softplus = nn.Softplus()

    def forward(self, x):
        # 可视化输入（仅对第一个 batch 第一个样本进行可视化）
        if self.visualize:
            self._visualize_input(x[0].cpu().detach(), title="Input Image")

        # 特征提取
        features = self.feature_extractor(x)
        if self.visualize:
            self._visualize_feature_maps(features[0].cpu().detach(), title="Feature Maps")

        # Attention模块：将特征图排列成适合注意力网络的二维形式
        out = features.permute(0, 2, 3, 1).contiguous()  # shape: [batch, H, W, C]
        out = out.view(-1, out.size(-1))  # shape: [batch * patches, C]
        weight = self.attention(out).unsqueeze(-1)  # shape: [batch * patches, 1]
        if self.visualize:
            self._visualize_attention_weights(weight[0].cpu().detach(), title="Attention Weights")

        # 重塑特征图为 [batch, num_patches, C] 并加权求和
        out = out.view(x.size(0), -1, out.size(-1))
        out = weight * out
        out = torch.sum(out, dim=1)

        # 分类器输出 ATP 预测值（此时为原始 ATP 预测值，但我们希望是正数）
        atp = self.softplus(self.classifier(out)).squeeze(-1)

        # 为了得到预测连续值的对数，计算 log(ATP + eps) ，eps 防止数值不稳定
        eps = 1e-6
        log_atp = torch.log(atp + eps)
        return log_atp

    def _visualize_input(self, img, title="Input Image"):
        """将单张输入图像进行可视化"""
        # 这里假设 img 为 [C, H, W]，转为 [H, W, C]
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        plt.figure(figsize=(8, 8))
        plt.imshow(img_np)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def _visualize_feature_maps(self, features, title="Feature Maps", n_maps=16):
        """可视化前 n 张特征图"""
        plt.figure(figsize=(12, 12))
        for i in range(min(n_maps, features.size(0))):
            plt.subplot(4, 4, i + 1)
            feat_map = features[i].cpu().numpy()
            feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
            plt.imshow(feat_map, cmap='viridis')
            plt.title(f"Channel {i}")
            plt.axis('off')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def _visualize_attention_weights(self, weights, title="Attention Weights"):
        """将注意力权重可视化"""
        weights = weights.view(-1).cpu().numpy()
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(weights)), weights)
        plt.title(title)
        plt.xlabel("Patch Index")
        plt.ylabel("Attention Weight")
        plt.show()