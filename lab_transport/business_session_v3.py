"""Frozen CPU v3 ablations. Only Train optimizes; Validation selects."""
import hashlib
import math
import statistics
import time
from pathlib import Path
import torch
from torch import nn
from business_transport.session_features_v3 import INPUT_SCHEMA_V3,QUERY_WIDTH_V3,encode_query_v3
from business_transport.session_events import QUERY_WIDTH,QUERY_LABEL_VERSION,encode_query
from business_transport.state_features import SCHEMA_VERSION
from lab_transport.business_session_learning import BusinessSessionEventModel,metrics,verify_dataset
from lab_transport.business_state_learning import encode_tokens,TOKEN_WIDTH
VERSION='business-sets-session-v3'
SCHEMA_V2='business-service-query-input-v2'

def encode_rows(rows,schema):
    if not rows:raise ValueError('empty rows')
    encoder=encode_query_v3 if schema==INPUT_SCHEMA_V3 else encode_query if schema==SCHEMA_V2 else None
    if encoder is None:raise ValueError('unsupported v3 input schema')
    items=[encode_tokens(row['features']) for row in rows]
    t=torch.zeros(len(rows),max(len(x) for x in items),TOKEN_WIDTH)
    m=torch.zeros(t.shape[:2])
    q=torch.tensor([encoder(row['features'],row['query']) for row in rows],dtype=torch.float32)
    for i,x in enumerate(items):t[i,:len(x)]=x;m[i,:len(x)]=1.
    return t,m,q

def probs(model,rows,schema):
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(*encode_rows(rows,schema))).tolist()

def stats(rows,p):
    s=metrics(rows,p,.5)
    y=[row['label'] for row in rows]
    s['log_loss']=-sum(a*math.log(max(1e-7,min(1-1e-7,v)))+
                         (1-a)*math.log(max(1e-7,min(1-1e-7,1-v)))
                         for a,v in zip(y,p))/len(y)
    z=sorted(p);n=len(z)
    s['distribution']={'min':z[0],'p10':z[int((n-1)*.1)],
                       'median':z[int((n-1)*.5)],'p90':z[int((n-1)*.9)],
                       'max':z[-1],'mean':statistics.mean(z)}
    return s

def grouped(rows,p,key):
    from experiments.world_model_integration.empirical_baselines import elapsed_group,_target
    def tag(r):
        if key=='trajectory':return r['trajectory_group']
        if key=='station':return r['query']['station_id']
        if key=='phase':return r['query']['phase']
        if key=='elapsed':return elapsed_group(_target(r)[0])
        raise ValueError('invalid grouping')
    out={}
    for r,prob in zip(rows,p):out.setdefault(tag(r),[]).append((r,prob))
    return {k:stats([x for x,y in v],[y for x,y in v]) for k,v in sorted(out.items())}

def fit_variant(dataset,name,protocol,checkpoint):
    verify_dataset(dataset)
    if name not in ('V3-A','V3-B','V3-C'):raise ValueError('wrong ablation')
    checkpoint=Path(checkpoint)
    if checkpoint.exists():raise FileExistsError('checkpoint cannot overwrite')
    spec=protocol['ablation'][name];cpu=protocol['cpu']
    schema=SCHEMA_V2 if spec['input']=='v2' else INPUT_SCHEMA_V3
    width=QUERY_WIDTH if schema==SCHEMA_V2 else QUERY_WIDTH_V3
    torch.set_num_threads(1);torch.manual_seed(cpu['seed'])
    model=BusinessSessionEventModel(query_width=width).cpu()
    optim=torch.optim.Adam(model.parameters(),lr=cpu['lr'])
    training=dataset['train'];validation=dataset['validation']
    x,m,q=encode_rows(training,schema);vx,vm,vq=encode_rows(validation,schema)
    y=torch.tensor([r['label'] for r in training],dtype=torch.float32)
    vy=torch.tensor([r['label'] for r in validation],dtype=torch.float32)
    batch=len(training) if spec['batch_size']=='full' else spec['batch_size']
    start=time.monotonic();history=[];best=None;bad=0;updates=0
    for epoch in range(1,spec['epochs_max']+1):
        perm=torch.randperm(len(y),generator=torch.Generator().manual_seed(cpu['shuffle_seed_base']+epoch))
        losses=[];steps=0;model.train()
        for ix in perm.split(batch):
            if time.monotonic()-start>=cpu['seconds_max_per_variant']:
                raise TimeoutError('frozen CPU variant limit: '+name)
            optim.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(x[ix],m[ix],q[ix]),y[ix])
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward();optim.step();updates+=1;steps+=1
            losses.append((float(loss),len(ix)))
        model.eval()
        with torch.inference_mode():
            logits=model(vx,vm,vq)
            p=torch.sigmoid(logits).tolist()
            val_loss=float(nn.functional.binary_cross_entropy_with_logits(logits,vy))
        s=stats(validation,p)
        elapsed=grouped(validation,p,'elapsed')
        history.append({'epoch':epoch,'epoch_optimizer_steps':steps,
          'cumulative_optimizer_steps':updates,
          'train_loss':sum(v*n for v,n in losses)/len(y),'validation_loss':val_loss,
          'validation_brier':s['brier'],'validation_distribution':s['distribution'],
          'elapsed_groups':{k:{'count':v['count'],'positive':v['positive'],
              'brier':v['brier']} for k,v in elapsed.items()}})
        if best is None or s['brier']<best['brier']-1e-8:
            best={'epoch':epoch,'brier':s['brier'],'updates':updates,
                  'weights':{k:v.detach().clone() for k,v in model.state_dict().items()}}
            bad=0
        else:bad+=1
        if bad>=spec['patience']:break
    if name=='V3-B' and updates!=10:
        raise ValueError('feature-only ablation must use precisely 10 updates')
    model.load_state_dict(best['weights'])
    final_p=probs(model,validation,schema)
    metadata={'model_version':VERSION,'input_schema':schema,
      'label_version':QUERY_LABEL_VERSION,'observation_schema':SCHEMA_VERSION,
      'variant':name,'protocol':protocol['protocol'],'online_verified':False,
      'optimization_steps_total':updates,'optimization_steps_best':best['updates'],
      'epoch_best':best['epoch'],'epochs_completed':len(history),
      'elapsed_wall_seconds':time.monotonic()-start,
      'validation':stats(validation,final_p),'validation_elapsed':grouped(validation,final_p,'elapsed'),
      'history':history,'train_only_optimization':True}
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'metadata':metadata,'model_state':model.state_dict()},checkpoint)
    metadata['sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return metadata

def load_v3(path,*,experimental=False):
    saved=torch.load(path,map_location='cpu',weights_only=True);m=saved['metadata']
    if m.get('model_version')!=VERSION or m.get('input_schema') not in (SCHEMA_V2,INPUT_SCHEMA_V3):
        raise ValueError('v3 checkpoint version/schema mismatch')
    if m.get('observation_schema')!=SCHEMA_VERSION or m.get('label_version')!=QUERY_LABEL_VERSION:
        raise ValueError('v3 checkpoint label/observation mismatch')
    if m.get('online_verified') is not True and not experimental:
        raise ValueError('v3 model not verified for production use')
    width=QUERY_WIDTH if m['input_schema']==SCHEMA_V2 else QUERY_WIDTH_V3
    model=BusinessSessionEventModel(query_width=width)
    model.load_state_dict(saved['model_state'])
    return model.eval(),m
