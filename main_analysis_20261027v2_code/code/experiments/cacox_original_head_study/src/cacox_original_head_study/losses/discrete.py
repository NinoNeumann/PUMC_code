"""Masked right-censored discrete survival likelihood; mean per patient."""
import torch


def logistic_hazard_nll(logits, labels):
    mask, target = labels
    if logits.shape != mask.shape or target.shape != mask.shape:
        raise ValueError("Head and label dimensions differ")
    return ((torch.nn.functional.softplus(logits) - logits * target) * mask).sum(1).mean()
