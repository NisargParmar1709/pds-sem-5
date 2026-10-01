# models.py
import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bart = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bart.parameters():
            param.requires_grad = False
        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.projection(pooled_output)


class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # keep pretrained for now (warning is ok)
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_fts = self.backbone.fc.in_features
        # Project the video features to 128 to match text/audio dims
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # expected input: [batch, frames, channels, H, W]
        x = x.transpose(1, 2)  # -> [batch, channels, frames, H, W]
        return self.backbone(x)  # returns [batch, 128]


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            # input channels = 64 (n_mels)
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # second conv: output 128 channels -> BatchNorm must be 128
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),     # <<-- FIXED (was 64)
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # output shape [batch, 128, 1]
        )

        # if you don't want to freeze audio convs, remove the loop below
        # for param in self.conv_layer.parameters():
        #     param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128, 128),   # project to 128 to match other modalities
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # expected x shape: [batch, 1, 64, time] (how you create sample in main)
        # remove that intermediate 1 channel dim:
        if x.dim() == 4:
            x = x.squeeze(1)  # -> [batch, 64, time]
        features = self.conv_layer(x)   # -> [batch, 128, 1]
        features = features.squeeze(-1)  # -> [batch, 128]
        return self.projection(features)


class MultimodelSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # fusion layer: expects 128 * 3 = 384 input
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )

    def forward(self, text_inputs, video_frames, audio_features):
        text_features = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask']
        )  # [batch, 128]

        video_features = self.video_encoder(video_frames)  # [batch, 128]
        audio_features = self.audio_encoder(audio_features)  # [batch, 128]

        combined_features = torch.cat([
            text_features,
            video_features,
            audio_features
        ], dim=1)  # [batch, 384]

        fused_features = self.fusion_layer(combined_features)
        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)

        return {
            'emotions': emotion_output,
            'sentiments': sentiment_output
        }

