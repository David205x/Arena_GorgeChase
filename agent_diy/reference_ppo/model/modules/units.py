import torch
from torch import nn
from .vector import (
    get_fc_layer,
    ResidualBlock,
)


class UnitEncoder(nn.Module):
    def __init__(self, input_dim: int = 64, token_dim: int = 64):
        super().__init__()
        self.fc_layer = get_fc_layer(input_dim, token_dim)
        self.pre_resblock = ResidualBlock(token_dim)
        self.post_resblock = ResidualBlock(token_dim)

    def forward(self, inputs, valids):
        valids = valids.to(torch.bool)
        mask = ~valids
        tokens = self.fc_layer(inputs)
        tokens = self.pre_resblock(tokens)
        tokens = self.post_resblock(tokens)
        tokens = tokens.masked_fill(mask.unsqueeze(-1), 0.0)
        return tokens


# 本来我是想参照AlphaStar那样用Transformer来处理Organ序列的, 所以下面写了一些Transformer相关的东西
def get_transformer(token_dim: int = 128, num_layers: int = 1):
    encoder_block = nn.TransformerEncoderLayer(
        d_model=token_dim,
        dim_feedforward=token_dim,
        nhead=2,
        dropout=0.0,
        activation='gelu',
        batch_first=True,
    )
    transformer = nn.TransformerEncoder(encoder_block, num_layers=num_layers)
    return transformer


class UnitTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        token_dim: int = 128,
        global_token_dim: int = 256,
        num_layers: int = 1,
    ):
        super().__init__()
        self.fc_layer = get_fc_layer(input_dim, token_dim)
        self.pre_resblock = ResidualBlock(token_dim)
        # self.post_resblock = ResidualBlock(hidden_dim)
        self.transformer = get_transformer(token_dim, num_layers)

        self.conv1d = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.SiLU(),
            get_fc_layer(token_dim, token_dim * 2, orthogonal_init=False),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(token_dim * 2),
            nn.SiLU(),
            get_fc_layer(token_dim * 2, global_token_dim, orthogonal_init=False),
            nn.LayerNorm(global_token_dim),
        )

    def forward(self, inputs, valids):
        valids = valids.to(torch.bool)
        mask = ~valids
        tokens = self.fc_layer(inputs)
        tokens = self.pre_resblock(tokens)
        tokens = self.transformer(tokens, src_key_padding_mask=mask)
        tokens = tokens.masked_fill(mask.unsqueeze(-1), 0.0)

        encoded_tokens = self.conv1d(tokens)
        encoded_tokens = encoded_tokens.masked_fill(mask.unsqueeze(-1), 0.0)
        global_token = encoded_tokens.sum(-2) / valids.sum(-1, keepdim=True)
        global_token = self.decoder(global_token)
        return tokens, global_token


class UnitEncoderWithGlobalToken(nn.Module):
    def __init__(
        self, input_dim: int = 64, hidden_dim: int = 64, output_dim: int = 256
    ):
        super().__init__()
        self.fc_layer = get_fc_layer(input_dim, hidden_dim)
        self.pre_resblock = ResidualBlock(hidden_dim)
        self.post_resblock = ResidualBlock(hidden_dim)

        self.conv1d = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            get_fc_layer(hidden_dim, hidden_dim * 2, orthogonal_init=False),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            get_fc_layer(hidden_dim * 2, output_dim, orthogonal_init=False),
            nn.LayerNorm(output_dim),
        )

    def forward(self, inputs, valids):
        valids = valids.to(torch.bool)
        mask = ~valids
        tokens = self.fc_layer(inputs)
        tokens = self.pre_resblock(tokens)
        tokens = self.post_resblock(tokens)
        tokens = tokens.masked_fill(mask.unsqueeze(-1), 0.0)

        encoded_tokens = self.conv1d(tokens)
        encoded_tokens = encoded_tokens.masked_fill(mask.unsqueeze(-1), 0.0)
        global_token = encoded_tokens.sum(-2) / valids.sum(-1, keepdim=True)
        global_token = self.decoder(global_token)
        return tokens, global_token
