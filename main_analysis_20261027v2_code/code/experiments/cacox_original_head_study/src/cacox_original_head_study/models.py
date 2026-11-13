"""Exact original CACox; change only its final prediction layer for the new head."""
import hashlib
import math
import torch
from torch import nn
from .original_network import build_network
from .runtime import seed_all


def make_model(n_raw, spec, common, seed):
    seed_all(seed)
    net=build_network(n_raw,common["hidden"],common["num_blocks"],common["dropout"],
                      common["num_cross_layers"],common["num_heads"])
    if spec["kind"]=="discrete":
        # Build the original model first: identical shared weights and RNG stream.
        state=torch.random.get_rng_state()
        layer=nn.Linear(common["hidden"],spec["bins"],bias=True)
        nn.init.kaiming_uniform_(layer.weight,a=math.sqrt(5))
        nn.init.zeros_(layer.bias)
        net.head[-1]=layer
        torch.random.set_rng_state(state)
    elif spec["kind"]!="cox":
        raise ValueError("Unsupported head")
    return net.to(common["device"])


def feature_digest(net):
    digest=hashlib.sha256()
    for name,value in net.state_dict().items():
        if name.startswith("head.6."):continue
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
