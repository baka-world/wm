"""Aggregate PR60 A100 single-arm/seed runs into compact evidence.

Reads only local trusted checkpoints/reports produced by pr59_gpu_train.py.
No BLIND/TEST data is read or generated.
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, statistics
from pathlib import Path
import torch

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs-root',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--gpu-monitor',type=Path)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    runs=[]
    for selpath in sorted(args.runs_root.glob('pr60-map_*-*-20ep-e32c402/selection.json')):
        sel=json.loads(selpath.read_text())
        for arm,info in sel['controlled_arms'].items():
            for result in sel['all_training_seed_checkpoints']:
                if result['arm']!=arm: continue
                run=selpath.parent
                seed=result['seed']
                report_path=run/arm/str(seed)/'train_report.json'
                report=json.loads(report_path.read_text())
                best_epoch=min(report['per_epoch_validation_only'],
                               key=lambda x:x['selection_score'])
                ckpt_path=run/arm/str(seed)/'best.pt'
                ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
                per=ckpt['frozen_validation_metrics']['per_real_independent_scenario']
                runs.append({'run':str(run),'arm':arm,'seed':seed,
                    'selection':sel,'result':result,'report':report,
                    'best_epoch':best_epoch,'per_scenario':per,
                    'checkpoint':str(ckpt_path),'checkpoint_sha256':sha256(ckpt_path),
                    'duration_wall_seconds':sel.get('duration_wall_seconds')})
    if not runs:
        raise SystemExit('no run selection.json found')
    # per-epoch
    with (args.output/'per_epoch.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['arm','seed','epoch','training_loss','macro_node_mse','macro_device_brier','macro_new_task_mae','selection_score'])
        for r in runs:
            for e in r['report']['per_epoch_validation_only']:
                v=e['validation']
                w.writerow([r['arm'],r['seed'],e['epoch'],e['training_loss_batch_avg'],v['macro_node_mse'],v['macro_device_brier'],v['macro_new_task_mae'],e['selection_score']])
    # best per run
    with (args.output/'arm_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['arm','seed','params','best_epoch','selection_score','macro_node_mse','macro_device_brier','macro_new_task_mae','peak_memory_allocated_bytes','wall_seconds'])
        for r in runs:
            v=r['best_epoch']['validation']
            w.writerow([r['arm'],r['seed'],r['result']['model_parameters'],r['best_epoch']['epoch'],r['best_epoch']['selection_score'],
                        v['macro_node_mse'],v['macro_device_brier'],v['macro_new_task_mae'],
                        r['report']['peak_memory_allocated_bytes'],r['duration_wall_seconds']])
    # per scenario (best checkpoint)
    with (args.output/'per_scenario.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['arm','seed','id','family','regime','censored','n_real_steps','node_mse','device_brier','next_new_public_stage_mae'])
        for r in runs:
            for x in r['per_scenario']:
                w.writerow([r['arm'],r['seed'],x['id'],x['family'],x['regime'],x['true_time_right_censored'],x['n_real_steps'],x['node_mse'],x['device_brier'],x['next_new_public_stage_mae']])
    # arm summary
    arms=sorted({r['arm'] for r in runs})
    with (args.output/'arm_summary.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['arm','seeds','params','mean_node_mse','std_node_mse','mean_device_brier','std_device_brier','mean_new_task_mae','std_new_task_mae','mean_selection_score','total_wall_seconds'])
        for arm in arms:
            rs=[r for r in runs if r['arm']==arm]
            def vals(k): return [r['best_epoch']['validation'][k] for r in rs]
            w.writerow([arm,len(rs),rs[0]['result']['model_parameters'],
                        statistics.mean(vals('macro_node_mse')),statistics.pstdev(vals('macro_node_mse')),
                        statistics.mean(vals('macro_device_brier')),statistics.pstdev(vals('macro_device_brier')),
                        statistics.mean(vals('macro_new_task_mae')),statistics.pstdev(vals('macro_new_task_mae')),
                        statistics.mean(r['best_epoch']['selection_score'] for r in rs),
                        sum(r['duration_wall_seconds'] or 0 for r in rs)])
    # seed-level paired compare if both arms
    summary={'schema':'pr60-a100-gpu-aggregate-v1','runs':len(runs),'arms':{}}
    for arm in arms:
        rs=[r for r in runs if r['arm']==arm]
        summary['arms'][arm]={'seeds':[r['seed'] for r in rs],'params':[r['result']['model_parameters'] for r in rs],
            'node_mse':[r['best_epoch']['validation']['macro_node_mse'] for r in rs],
            'device_brier':[r['best_epoch']['validation']['macro_device_brier'] for r in rs],
            'new_task_mae':[r['best_epoch']['validation']['macro_new_task_mae'] for r in rs],
            'checkpoints':{str(r['seed']):{'path':r['checkpoint'],'sha256':r['checkpoint_sha256']} for r in rs}}
    if 'map_on' in summary['arms'] and 'map_off' in summary['arms']:
        on={r['seed']:r for r in runs if r['arm']=='map_on'}
        off={r['seed']:r for r in runs if r['arm']=='map_off'}
        seeds=sorted(set(on)&set(off))
        summary['paired']={}
        for metric in ('macro_node_mse','macro_device_brier','macro_new_task_mae','selection_score'):
            vals=[]
            for s in seeds:
                a=on[s]['best_epoch']['validation'][metric] if metric!='selection_score' else on[s]['best_epoch']['selection_score']
                b=off[s]['best_epoch']['validation'][metric] if metric!='selection_score' else off[s]['best_epoch']['selection_score']
                vals.append({'seed':s,'map_on':a,'map_off':b,'delta_on_minus_off':a-b})
            summary['paired'][metric]=vals
        summary['paired_seed_wins']={m:sum(1 for x in v if x['delta_on_minus_off']<0) for m,v in summary['paired'].items()}
    summary['blind_test']={'files_read':0,'episodes_generated':0}
    summary['limitation']='Teacher-forced true t->t+1 validation on 12 independent scenes; no H20/autoregressive/online claim.'
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # checksums
    files=[p for p in sorted(args.output.iterdir()) if p.is_file() and p.name!='SHA256SUMS']
    with (args.output/'SHA256SUMS').open('w') as f:
        for p in files: f.write(f'{sha256(p)}  {p.name}\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
