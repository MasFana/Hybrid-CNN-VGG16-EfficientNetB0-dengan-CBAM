import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torch_models
from torchvision import transforms as torch_transforms


class MCDropoutPT(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)


class ChannelAttentionPT(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttentionPT(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return x * self.sigmoid(out)


class CBAM_PT(nn.Module):
    def __init__(self, planes, ratio=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttentionPT(planes, ratio)
        self.sa = SpatialAttentionPT(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class VGGEffAttnNetPT(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.3):
        super().__init__()
        vgg = torch_models.vgg16(weights=None)
        self.vgg_features = vgg.features
        self.vgg_avgpool = vgg.avgpool
        self.vgg_fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(),
            MCDropoutPT(dropout_rate),
        )

        eff = torch_models.efficientnet_b0(weights=None)
        self.eff_features = eff.features
        self.eff_cbam = CBAM_PT(planes=1280)
        self.eff_gap = nn.AdaptiveAvgPool2d(1)
        self.eff_fc = MCDropoutPT(dropout_rate)

        self.fusion_fc = nn.Sequential(
            nn.Linear(256 + 1280, 256),
            nn.ReLU(),
            MCDropoutPT(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        v_x = self.vgg_features(x)
        v_x = self.vgg_avgpool(v_x)
        v_x = torch.flatten(v_x, 1)
        v_out = self.vgg_fc(v_x)

        e_x = self.eff_features(x)
        e_att = self.eff_cbam(e_x)
        e_refined = e_x + e_att
        e_refined = self.eff_gap(e_refined)
        e_refined = torch.flatten(e_refined, 1)
        e_out = self.eff_fc(e_refined)

        concat = torch.cat((v_out, e_out), dim=1)
        return self.fusion_fc(concat)


def load_torch_model(path):
    if not os.path.exists(path):
        return None, f"File '{path}' tidak ditemukan."
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VGGEffAttnNetPT(num_classes=6)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, "Success"
    except Exception as e:
        return None, str(e)


def preprocess_torch(image: Image.Image):
    transform = torch_transforms.Compose(
        [
            torch_transforms.Resize((224, 224)),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    return transform(image).unsqueeze(0)
