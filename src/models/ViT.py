import torch
import torch.nn as nn

# TODO: This is gen AI code. I need to make sure it is implemented properly.
# TODO: Get the code from a reputable source

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3501):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ViT1D_multi_task(nn.Module):
    def __init__(self, input_dim=3501, patch_size=16, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.25):
        super(ViT1D_multi_task, self).__init__()
        
        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
        num_patches = input_dim // patch_size
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        
        self.pos_drop = nn.Dropout(p=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-task specific layers
        self.crysystem_head = nn.Linear(d_model, 7)
        self.blt_head = nn.Linear(d_model, 6)
        self.spg_head = nn.Linear(d_model, 230)
        self.composition_head = nn.Linear(d_model, 118)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.transpose(1, 2)  # (B, num_patches, d_model)
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = self.transformer(x)
        
        # Use the class token for classification
        x = x[:, 0]
        
        # Multi-task specific layers
        crysystem_out = self.crysystem_head(self.dropout(x))
        blt_out = self.blt_head(self.dropout(x))
        spg_out = self.spg_head(self.dropout(x))
        composition_out = self.composition_head(self.dropout(x))
        
        return {
            'spg': spg_out,
            'crysystem': crysystem_out,
            'blt': blt_out,
            'composition': composition_out
        }

# Example usage:
# model = VisionTransformer1D(in_channels=1, num_classes=230, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1, patch_size=16)
# x = torch.randn(32, 1, 3501)  # Batch size of 32, 1 channel, 3501 data points
# output = model(x)
# print(output.shape)  # Should be (32, 230)