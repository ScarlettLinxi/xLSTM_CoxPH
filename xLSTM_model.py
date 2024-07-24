import torch
import torch.nn as nn
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

# Define xLSTM configuration
cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="cuda",
            num_heads=4,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
    context_length=5,  # Number of time steps
    num_blocks=7,
    embedding_dim=52,  # Feature dimension
    slstm_at=[1],
)

class xLSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(xLSTMModel, self).__init__()
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y = self.xlstm_stack(x)
        risk_score = self.fc(y[:, -1, :])  # Use the output of the last time step
        return risk_score