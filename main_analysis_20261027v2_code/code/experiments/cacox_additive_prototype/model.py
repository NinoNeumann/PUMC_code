"""Grouped neural additive ensemble; original encoded columns are unchanged."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class GroupedAdditiveEnsemble(nn.Module):
    """Eight independently parameterized additive risk functions, vectorized.

    One subnetwork per ORIGINAL variable, including its full dummy block. Each
    subnetwork has two hidden ReLU layers. There are no cross-variable hidden
    connections, spline-risk skips, data preprocessing, or learned data splits.
    Padding is internal only and masked; every original input column appears
    once. Outputs are (batch, members, 1), compatible with the existing loss API.
    """
    def __init__(self, features, groups, width, dropout=.15, members=8):
        super().__init__()
        if sorted(c for group in groups for c in group)!=list(range(features)):
            raise ValueError('Groups must cover all original columns exactly once')
        if any(not group for group in groups) or width not in [8,16] or members!=8:
            raise ValueError('Registered grouped-additive architecture required')
        self.features,self.groups,self.width,self.members=features,[list(g) for g in groups],width,members
        G,Q=len(groups),max(map(len,groups));indices=torch.zeros(G,Q,dtype=torch.long);mask=torch.zeros(G,Q)
        for j,group in enumerate(groups):indices[j,:len(group)]=torch.tensor(group);mask[j,:len(group)]=1.
        self.register_buffer('indices',indices);self.register_buffer('padding_mask',mask)
        self.w1=nn.Parameter(torch.zeros(G,members,Q,width));self.b1=nn.Parameter(torch.empty(G,members,width))
        self.w2=nn.Parameter(torch.empty(G,members,width,width));self.b2=nn.Parameter(torch.empty(G,members,width))
        self.head=nn.Parameter(torch.empty(G,members,width));self.dropout=float(dropout)
        # Initialization uses actual group widths, never padded feature counts.
        with torch.no_grad():
            for j,group in enumerate(groups):
                self.w1[j,:,:len(group),:].uniform_(-1/math.sqrt(len(group)),1/math.sqrt(len(group)))
                self.b1[j].uniform_(-1/math.sqrt(len(group)),1/math.sqrt(len(group)))
            for tensor in [self.w2,self.b2,self.head]:tensor.uniform_(-1/math.sqrt(width),1/math.sqrt(width))

    def forward(self,x_num,x_cat=None):
        if x_cat is not None or x_num.ndim!=2 or x_num.shape[1]!=self.features:
            raise ValueError('Original frozen encoded matrix required')
        x=x_num[:,self.indices]*self.padding_mask
        h=F.relu(torch.einsum('bgp,gkph->bgkh',x,self.w1)+self.b1)
        h=F.dropout(h,self.dropout,self.training)
        h=F.relu(torch.einsum('bgkh,gkhl->bgkl',h,self.w2)+self.b2)
        h=F.dropout(h,self.dropout,self.training)
        return (h*self.head).sum(-1).sum(1).unsqueeze(-1)

    def active_parameter_count(self):
        padding=int((1-self.padding_mask).sum())*self.members*self.width
        return sum(p.numel() for p in self.parameters())-padding


def groups_from_metadata(metadata):
    columns=metadata['columns'];variables=metadata['variables']
    groups=[[j for j,name in enumerate(columns) if name==v or name.startswith(v+'=')] for v in variables]
    if sorted(j for group in groups for j in group)!=list(range(len(columns))):
        raise ValueError('Original variable grouping does not cover encoded columns')
    if any(not group for group in groups):raise ValueError('Cannot remove an original variable')
    return groups
