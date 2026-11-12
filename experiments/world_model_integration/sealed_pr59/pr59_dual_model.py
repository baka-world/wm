"""PR59 learnable PUBLIC MAP + TEMPORAL ENTITY relation world-model prototype.

This is an untrained GPU-HANDOFF architecture, not a fitted checkpoint.
Two official PyTorch Geometric SAGEConv map updates, an entity↔map
bipartite lookup and the verified PR54 temporal RGCN transition.
Unlike PR56's map-sidecar-only pilot, map features are consumed by the
actual next-state prediction and have a differentiable map-to-output path.

KEEP the same parameterization in map_enabled vs map_masked controls.
No standalone trainable Q/policy head, no future hidden device schedule.
The topological Data subclass explicitly corrects multi-graph batching
offsets for both map and entity index namespaces.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv,global_mean_pool

from experiments.world_model_integration.temporal_pr54.pr54_model import (
    PublicTemporalRelationalWorld,supervised_losses
)
from experiments.world_model_integration.temporal_pr54.pr54_train import (
    example_to_graph
)


class PublicTopologyTemporalData(Data):
    def __inc__(self,key,value,*args,**kwargs):
        if key=="map_edge_index":
            return int(self.map_x.shape[0])
        if key=="entity_to_map_index":
            # Exactly two separate index namespaces:
            #  0: original task/robot/station/resource entity
            #  1: this scenario's public map-cell graph
            return torch.tensor([[int(self.x.shape[0])],
                                 [int(self.map_x.shape[0])]],dtype=torch.long)
        return super().__inc__(key,value,*args,**kwargs)


def actual_public_record_to_graph(record):
    inp=record["input"]
    if ("public_map" not in inp
        or inp["feature_last_observed_tick"]!=inp["public_tick"]
        or record["target"]["next_public_tick"]!=inp["public_tick"]+1):
        raise AssertionError("map-conditioned world input/label time boundary broken")
    entity=example_to_graph(record)
    topo=inp["public_map"]
    if topo["schema"]!="pr56-public-grid-cells-v1":
        raise AssertionError("unregistered public-geometry model input")
    map_x=torch.tensor(topo["map_node_features"],dtype=torch.float32)
    map_edge=torch.tensor(topo["directed_map_edges"],dtype=torch.long).T.contiguous()
    links=torch.tensor(topo["public_entity_to_map_cell_index"],
                       dtype=torch.long).T.contiguous()
    if (map_x.ndim!=2 or map_x.shape[1]!=3 or map_x.shape[0]<3
        or map_edge.shape[0]!=2 or links.shape[0]!=2):
        raise AssertionError("empty or incompatible PyG public geometry")
    if (map_edge.numel()==0 or map_edge.max()>=len(map_x)
        or map_edge.min()<0 or links.numel()==0
        or links[0].max()>=entity.x.shape[0]
        or links[1].max()>=map_x.shape[0]):
        raise AssertionError("mixed PyG entity/map IDs not locally valid")
    types=entity.x[links[0],:]
    if not torch.all((types[:,0]>0.5)|(types[:,2]>0.5)):
        raise AssertionError("future/unknown task nodes cannot attach to map cells")
    if len(set(topo["static_map_digest"]))<10:
        raise AssertionError("map source SHA not present")
    data=PublicTopologyTemporalData(
        **entity.to_dict(),
        map_x=map_x,
        map_edge_index=map_edge,
        entity_to_map_index=links,
        map_node_count=torch.tensor([len(map_x)],dtype=torch.long)
    )
    return data


class PublicMapTemporalRelationalWorld(nn.Module):
    """Explicit learned PUBLIC route topology to actual one-step state dynamics."""

    def __init__(self,hidden=48,history_enabled=True,map_enabled=True):
        super().__init__()
        self.hidden=hidden
        self.map_enabled=map_enabled
        self.entity_world=PublicTemporalRelationalWorld(
            hidden=hidden,history_enabled=history_enabled)
        self.map_conv1=SAGEConv(3,hidden)
        self.map_conv2=SAGEConv(hidden,hidden)
        self.map_fusion=nn.Linear(2*hidden,hidden)
        self.map_gate=nn.Parameter(torch.tensor(0.0))

    def forward(self,batch):
        core=self.entity_world
        h_input=batch.history*batch.history_mask.unsqueeze(-1)
        if not core.history_enabled:
            h_input=torch.cat((torch.zeros_like(h_input[:,:-1]),h_input[:,-1:]),dim=1)
        h=core.history_gru(h_input)[1][-1]
        node=F.relu(core.fusion(torch.cat(
            (h,F.relu(core.embedding(batch.x))),dim=-1)))
        if self.map_enabled:
            if batch.map_x.numel()==0 or batch.map_edge_index.numel()==0:
                raise AssertionError("missing actual public map edges")
            map_state=F.relu(self.map_conv1(batch.map_x,batch.map_edge_index))
            map_state=F.relu(self.map_conv2(map_state,batch.map_edge_index))
            map_groups=torch.repeat_interleave(
                torch.arange(batch.num_graphs,device=map_state.device),
                batch.map_node_count)
            if map_groups.numel()!=map_state.shape[0]:
                raise AssertionError("PyG map count batching lost cell nodes")
            globals_=global_mean_pool(map_state,map_groups)
            local=torch.zeros(
                node.shape[0],self.hidden,device=node.device,dtype=node.dtype)
            entity_indices,cell_indices=batch.entity_to_map_index
            if (entity_indices.min()<0 or entity_indices.max()>=node.shape[0]
                or cell_indices.min()<0 or cell_indices.max()>=map_state.shape[0]):
                raise AssertionError("mixed entity/map cell batch offset invalid")
            # Hard contamination guard: a link must point into the SAME
            # PyG sample, not into a different topology's row after batching.
            if not torch.equal(batch.batch[entity_indices],map_groups[cell_indices]):
                raise AssertionError("cross-scenario map/entity link leakage")
            local.index_add_(0,entity_indices,map_state[cell_indices])
            counts=torch.zeros((node.shape[0],1),device=node.device,
                               dtype=node.dtype)
            counts.index_add_(0,entity_indices,
                              torch.ones((len(entity_indices),1),
                                         device=node.device,dtype=node.dtype))
            local=local/counts.clamp(min=1)
            context=self.map_fusion(torch.cat((local,globals_[batch.batch]),dim=-1))
            node=node+torch.sigmoid(self.map_gate)*torch.tanh(context)
        # Same PR54 business relationship edges; map info enters state
        # transition BEFORE the node-next and new-task event prediction.
        if batch.edge_index.numel():
            node=F.relu(core.relation_step1(node,batch.edge_index,batch.edge_type))
            node=F.relu(core.relation_step2(node,batch.edge_index,batch.edge_type))
        logits=core.node_next_head(node)
        probs=torch.sigmoid(logits)
        is_task=batch.x[:,1,None]>0.5
        task_status=torch.softmax(logits[:,:4],dim=-1)
        node_next=torch.cat((
            torch.where(is_task,task_status,probs[:,:4]),probs[:,4:]),dim=-1)
        events=torch.sigmoid(core.new_task_event_head(
            global_mean_pool(node,batch.batch)))
        visible=events[:,1]
        completed=visible*events[:,0]
        return node_next,torch.stack((completed,visible),dim=-1)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


__all__=["PublicTopologyTemporalData","actual_public_record_to_graph",
         "PublicMapTemporalRelationalWorld","supervised_losses","parameter_count"]
