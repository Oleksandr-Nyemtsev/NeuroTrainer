import torch
import torch.nn as nn


class SleepCNN(nn.Module):

    def __init__(
        self,
        input_channels=2,
        num_classes=5,
    ):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=input_channels,
                out_channels=16,
                kernel_size=7,
                padding=3,
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2,
            ),

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2,
            ),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(
            64,
            num_classes,
        )


    def forward(self, x):

        x = self.features(x)

        x = self.global_pool(x)

        x = x.squeeze(-1)

        x = self.classifier(x)

        return x