"""PR51 experimental pairwise decision objective, NOT a trained world model.

Cost means actual terminal makespan given a single intervention under
identical public prefix & keyed exogenous noise, with original CP-SAT
continuation. This objective is only eligible when a differentiable
action-conditioned predicted transition is connected to cost predictions.

Research lineage: VAML / VaGraM. This pairwise surrogate is a proposed
adaptation, NOT an exact reimplementation or a calibrated guarantee.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

def pairwise_decision_loss(predicted_cost, factual_cost,valid=None):
    if predicted_cost.ndim!=1 or factual_cost.ndim!=1 or predicted_cost.shape!=factual_cost.shape:
        raise ValueError("candidate score vectors must be matched 1d tensors")
    n=predicted_cost.shape[0]
    if n<2:
        return predicted_cost.sum()*0.
    valid=(torch.ones_like(factual_cost,dtype=torch.bool) if valid is None else valid.bool())
    if valid.shape!=factual_cost.shape:
        raise ValueError("per-candidate completed/censored mask mismatch")
    # Use only observed decisive true pair labels. Ties are not fabricated into
    # arbitrary winners; later model prediction calibration is separate.
    factual_delta=factual_cost[:,None]-factual_cost[None,:]
    predicted_delta=predicted_cost[:,None]-predicted_cost[None,:]
    ids=torch.arange(n,device=predicted_cost.device)
    mask=(ids[:,None]<ids[None,:]) & valid[:,None] & valid[None,:] & (factual_delta!=0)
    if not bool(mask.any()):
        return predicted_cost.sum()*0.
    return F.softplus(
        -torch.sign(factual_delta[mask]).to(predicted_cost.dtype)*predicted_delta[mask]
    ).mean()


def combined_world_model_loss(prediction_loss, predicted_cost, factual_cost,weight,valid=None):
    if not 0<=weight<=10:
        raise ValueError("decision weight must be validated, never chosen on test")
    dec=pairwise_decision_loss(predicted_cost,factual_cost,valid)
    return prediction_loss+weight*dec


def gradient_path_present(shared_transition_parameters):
    """Call only after backward on a *decision-only* loss."""
    found=False
    for param in shared_transition_parameters:
        grad=param.grad
        found=found or (grad is not None and bool(torch.isfinite(grad).all())
                        and bool(grad.abs().sum().item()>0))
    return found
