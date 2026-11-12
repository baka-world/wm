"""PR54 genuine learned action-conditioned ONE-STEP relational world model.

History-GRU with same parameters for both controlled arms, followed by
the EXISTING PyTorch-Geometric RGCNConv graph message passing. The only
difference in the snapshot control is masking all historical frames
strictly before current t. Both predict the same public t+1 entity states
and anonymous newly visible/completed task stage counts.

There is no Q head, policy network, future job identifier, hidden simulator
schedule or claim of an executable H20 autoregressive simulator.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv,global_mean_pool

from experiments.world_model_integration.decision_learning_pr51.pr51_train import (
    RELATIONS
)


class PublicTemporalRelationalWorld(nn.Module):
    def __init__(self,hidden=48,history_enabled=True):
        super().__init__()
        self.history_enabled=bool(history_enabled)
        self.hidden=hidden
        self.embedding=nn.Linear(16,hidden)
        self.history_gru=nn.GRU(16,hidden,batch_first=True)
        self.fusion=nn.Linear(2*hidden,hidden)
        self.relation_step1=RGCNConv(
            hidden,hidden,num_relations=2*len(RELATIONS),num_bases=4)
        self.relation_step2=RGCNConv(
            hidden,hidden,num_relations=2*len(RELATIONS),num_bases=4)
        self.node_next_head=nn.Sequential(
            nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,6))
        self.new_task_event_head=nn.Sequential(
            nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,2))

    def forward(self,batch):
        h_input=batch.history*batch.history_mask.unsqueeze(-1)
        if not self.history_enabled:
            h_input=torch.cat((torch.zeros_like(h_input[:,:-1]),h_input[:,-1:]),dim=1)
        history_encoded=self.history_gru(h_input)[1][-1]
        static=F.relu(self.embedding(batch.x))
        state=F.relu(self.fusion(torch.cat((history_encoded,static),dim=-1)))
        if batch.edge_index.numel():
            state=F.relu(self.relation_step1(state,batch.edge_index,batch.edge_type))
            state=F.relu(self.relation_step2(state,batch.edge_index,batch.edge_type))
        logits=self.node_next_head(state)
        probs=torch.sigmoid(logits)
        task_mask=batch.x[:,1,None]>0.5
        task_status=torch.softmax(logits[:,:4],dim=-1)
        node_next=torch.cat((torch.where(task_mask,task_status,probs[:,:4]),
                             probs[:,4:]),dim=-1)
        event_logits=self.new_task_event_head(global_mean_pool(state,batch.batch))
        event_probs=torch.sigmoid(event_logits)
        visible=event_probs[:,1]
        completed=visible*event_probs[:,0]
        return node_next,torch.stack((completed,visible),dim=-1)


def supervised_losses(node,event,batch):
    mask=batch.target_mask
    state_mse=((node-batch.node_target).square()*mask).sum()/mask.sum().clamp(min=1.)
    event_mse=F.mse_loss(event,batch.event_target)
    # The status target is the original public event lifecycle. Pseudolabels
    # for unseen new task IDs never become existing graph nodes.
    task=(batch.x[:,1]>0.5)&(mask[:,:4].sum(dim=1)==4)
    if torch.any(task):
        targets=batch.node_target[task,:4]
        target_ids=targets.argmax(dim=-1)
        status_ce=F.nll_loss(torch.log(node[task,:4].clamp(min=1e-6)),target_ids)
    else:
        status_ce=node.sum()*0.
    return {
        "node_masked_mse":state_mse,
        "new_stage_event_mse":event_mse,
        "task_status_ce":status_ce,
        "total":state_mse+0.3*event_mse+0.1*status_ce
    }
