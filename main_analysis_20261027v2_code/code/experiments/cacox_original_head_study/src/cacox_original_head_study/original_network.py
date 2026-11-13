"""Exact class/function extracts from immutable reference model.py and pycox loss.py.
Architecture and original CoxPH tie approximation are intentionally preserved.
Training/control changes live in training.py, not in these classes.
"""
import math
import torch
from torch import nn, Tensor

class FeatureTokenizer(nn.Module):
    """
    将数值特征转换为嵌入向量 (Feature Tokenization)
    对于输入 x (batch, num_features)，输出 (batch, num_features, embed_dim)
    """
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        
        # 为每个特征学习一个权重向量和偏置
        self.weights = nn.Parameter(torch.randn(1, num_features, embed_dim))
        self.biases = nn.Parameter(torch.zeros(1, num_features, embed_dim))
        
        # 初始化
        nn.init.xavier_uniform_(self.weights)
        
    def forward(self, x):
        # x: (batch, num_features)
        # 扩展 x 为 (batch, num_features, 1)
        x_expanded = x.unsqueeze(-1)
        
        # 广播乘法和加法
        out = x_expanded * self.weights + self.biases
        return out

class CrossNet(nn.Module):
    """
    Deep Cross Network (DCN) module for explicit feature interactions.
    Formula: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
    """
    def __init__(self, in_features, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.kernels = nn.ParameterList([
            nn.Parameter(torch.randn(in_features, 1)) for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(in_features, 1)) for _ in range(num_layers)
        ])
        
        for p in self.kernels:
            nn.init.xavier_normal_(p)
            
    def forward(self, x):
        # x: (batch, in_features)
        x_0 = x.unsqueeze(2) # (batch, in_features, 1)
        x_l = x_0
        
        for i in range(self.num_layers):
            # W_l * x_l + b_l
            # tensordot: (batch, in_features, 1) * (in_features, 1) -> (batch, 1, 1)
            # But we want explicit interaction.
            # x_l: (batch, in_features, 1)
            # kernels[i]: (in_features, 1)
            
            # x_l^T * w: (batch, 1, in_features) * (in_features, 1) -> (batch, 1, 1) scalar per sample
            xl_w = torch.matmul(x_l.transpose(1, 2), self.kernels[i]) 
            
            # x_0 * scalar + b + x_l
            dot_ = torch.matmul(x_0, xl_w) # (batch, in_features, 1)
            x_l = dot_ + self.biases[i] + x_l
            
        return x_l.squeeze(2)

class GatingMechanism(nn.Module):
    """
    Feature Gating Mechanism to learn feature importance.
    x_gated = x * sigmoid(W * x + b)
    """
    def __init__(self, in_features):
        super().__init__()
        self.gate = nn.Linear(in_features, in_features)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        mask = self.sigmoid(self.gate(x))
        return x * mask

class CrossAttentionCox(nn.Module):
    """
    Hybrid architecture combining Transformer and Deep Cross Network (DCN)
    with Feature Gating.
    """
    def __init__(self, in_features, num_blocks=2, hidden_dim=32, out_features=1, 
                 dropout=0.1, batch_norm=False, output_bias=True, num_heads=4,
                 num_cross_layers=2):
        super().__init__()
        
        # 1. Feature Gating
        self.gating = GatingMechanism(in_features)
        
        # 2. Branch 1: Transformer (AttentionNetwork without head)
        # Reuse FeatureTokenizer and TransformerEncoder from AttentionNetwork logic
        self.tokenizer = FeatureTokenizer(in_features, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2, 
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.trans_flatten_dim = in_features * hidden_dim
        self.trans_norm = nn.LayerNorm(self.trans_flatten_dim)
        
        # 3. Branch 2: Cross Network
        self.cross_net = CrossNet(in_features, num_layers=num_cross_layers)
        
        # 4. Fusion & Output Head
        fusion_dim = self.trans_flatten_dim + in_features
        
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features, bias=output_bias)
        )
        
    def forward(self, x):
        # 1. Gating
        x_g = self.gating(x)
        
        # 2. Transformer Branch
        x_emb = self.tokenizer(x_g)
        x_trans = self.transformer(x_emb)
        x_trans_flat = x_trans.reshape(x_trans.size(0), -1)
        x_trans_out = self.trans_norm(x_trans_flat)
        
        # 3. Cross Net Branch
        x_cross_out = self.cross_net(x_g)
        
        # 4. Fusion
        x_combined = torch.cat([x_trans_out, x_cross_out], dim=1)
        
        # 5. Output
        out = self.head(x_combined)
        # out = torch.clamp(out, -20.0, 20.0)
        return out

def cox_ph_loss_sorted(log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    tt_test = - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())
    return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())

def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, eps)

def build_network(in_features, hidden_dim=96, num_blocks=2, dropout=.15,
                  num_cross_layers=2, num_heads=4):
    if hidden_dim % num_heads:
        raise ValueError("hidden_dim must be divisible by num_heads; no silent resize")
    net = CrossAttentionCox(in_features=in_features, hidden_dim=hidden_dim,
        num_blocks=num_blocks, dropout=dropout, num_cross_layers=num_cross_layers,
        num_heads=num_heads, output_bias=True)
    # Exact kaiming branch of source ModelConfig._apply_weight_initialization.
    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.)
    return net
