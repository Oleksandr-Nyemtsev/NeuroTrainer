import torch
import torch.nn as nn
import torch.nn.functional as F


class CovariancePooling(nn.Module):
    """
    Convert learned feature maps into covariance matrices.

    Input:
        x: (batch, channels, time)

    Output:
        cov: (batch, channels, channels)
    """

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # center over time
        x = x - x.mean(dim=-1, keepdim=True)

        n = x.shape[-1]

        cov = torch.matmul(
            x,
            x.transpose(1, 2),
        ) / max(n - 1, 1)

        # SPD regularization
        eye = torch.eye(
            cov.shape[-1],
            device=cov.device,
            dtype=cov.dtype,
        )

        cov = cov + self.eps * eye.unsqueeze(0)

        return cov


class LogEig(nn.Module):
    """
    Matrix logarithm for SPD matrices using eigendecomposition.
    """

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, cov):
        eigvals, eigvecs = torch.linalg.eigh(cov)

        eigvals = torch.clamp(
            eigvals,
            min=self.eps,
        )

        log_eigvals = torch.log(eigvals)

        log_cov = (
            eigvecs
            @ torch.diag_embed(log_eigvals)
            @ eigvecs.transpose(1, 2)
        )

        return log_cov


class SymmetricVectorize(nn.Module):
    """
    Vectorize upper triangle of symmetric matrix.
    """

    def __init__(self, channels):
        super().__init__()

        indices = torch.triu_indices(
            channels,
            channels,
        )

        self.register_buffer(
            "row_idx",
            indices[0],
        )

        self.register_buffer(
            "col_idx",
            indices[1],
        )

    def forward(self, x):
        return x[
            :,
            self.row_idx,
            self.col_idx,
        ]


class HybridCRGSleep(nn.Module):
    """
    CNN + covariance/SPD + log-domain classifier.

    Input:
        (batch, 2, 3000)

    Output:
        logits for:
        Wake / N1 / N2 / N3 / REM
    """

    def __init__(self, num_classes=5):
        super().__init__()

        # --------------------------------------------------
        # TEMPORAL FEATURE EXTRACTION
        # --------------------------------------------------

        self.temporal = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=16,
                kernel_size=31,
                padding=15,
            ),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.MaxPool1d(4),

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=15,
                padding=7,
            ),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(4),

            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(32),
            nn.ELU(),
        )

        # --------------------------------------------------
        # RIEMANNIAN-LIKE BLOCK
        # --------------------------------------------------

        self.covariance = CovariancePooling(
            eps=1e-4
        )

        self.logeig = LogEig(
            eps=1e-4
        )

        self.vectorize = SymmetricVectorize(
            channels=32
        )

        # 32 x 32 symmetric matrix
        # independent values:
        # 32 * 33 / 2 = 528

        self.classifier = nn.Sequential(
            nn.Linear(528, 128),
            nn.ELU(),
            nn.Dropout(0.3),

            nn.Linear(128, 32),
            nn.ELU(),
            nn.Dropout(0.2),

            nn.Linear(
                32,
                num_classes,
            ),
        )

    def forward(self, x):
        x = self.temporal(x)

        cov = self.covariance(x)

        log_cov = self.logeig(cov)

        features = self.vectorize(log_cov)

        logits = self.classifier(features)

        return logits